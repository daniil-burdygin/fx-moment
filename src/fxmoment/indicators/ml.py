"""Обучаемый индикатор: классический бустинг предсказывает «день — локальный минимум в окне ±h»,
порог — под асимметричную цену ошибки; пуш — только при факте уровня (ADR-0005)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from fxmoment.config import BUY_NOW
from fxmoment.indicators.base import Indicator, rearm_events, rolling_pct_rank
from fxmoment.indicators.features import build_features
from fxmoment.labels import benefit_fwd_bps, local_min_label


class LearnedMinimum(Indicator):
    name = "ml_localmin"
    speed = "medium"
    scenario = BUY_NOW
    direction = "down"
    trainable = True

    def __init__(
        self,
        h: int = 10,
        tol_bps: float = 10.0,
        gate_window: int = 120,
        gate_pct: float = 0.20,
        fp_cost: float = 3.0,
        min_pos_rate: float = 0.25,
        rearm: int = 5,
        seed: int = 0,
    ) -> None:
        super().__init__(
            h=h,
            tol_bps=tol_bps,
            gate_window=gate_window,
            gate_pct=gate_pct,
            fp_cost=fp_cost,
            min_pos_rate=min_pos_rate,
            rearm=rearm,
            seed=seed,
        )
        self.h = h
        self.tol_bps = tol_bps
        self.gate_window = gate_window
        self.gate_pct = gate_pct
        self.fp_cost = fp_cost
        self.min_pos_rate = min_pos_rate
        self.rearm = rearm
        self.seed = seed
        self.model_: HistGradientBoostingClassifier | None = None
        self.threshold_: float = 1.0
        self.fitted_: bool = False
        self.feature_names_: list[str] = []

    def fact_fields(self) -> tuple[str, ...]:
        return ("proba", "pct_rank", "window")

    def warmup(self) -> int:
        return 270  # rank250 плюс vol_rank250 поверх vol20 (features.py)

    def _new_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            min_samples_leaf=30,
            early_stopping=False,
            random_state=self.seed,
        )

    def fit(
        self,
        rate: pd.Series,
        context: pd.DataFrame | None = None,
        train_start: str | pd.Timestamp | None = None,
    ) -> LearnedMinimum:
        """Обучение на первых 80 % строк, порог — на последних 20 % среди дней, прошедших гейт уровня.
        Рабочая модель — та же, на чьих вероятностях выбран порог: переобучение на 100 % меняло
        распределение вероятностей, и порог переставал что-либо значить (аудит 03.09).
        `train_start` — с какой даты брать строки в обучение; история раньше служит только разогревом
        признаков."""
        x = build_features(rate, context)
        y = local_min_label(rate, self.h, self.tol_bps)
        ok = x.notna().all(axis=1) & y.notna()
        if train_start is not None:
            ok &= x.index >= pd.Timestamp(train_start)
        x, y = x[ok], y[ok].astype(int)
        self.fitted_ = False
        if len(x) < 200 or y.nunique() < 2:
            self.model_ = None
            self.threshold_ = 1.0
            return self
        w = np.where(y.to_numpy() == 1, 1.0, self.fp_cost)
        cut = int(len(x) * 0.8)
        model = self._new_model().fit(x.iloc[:cut], y.iloc[:cut], sample_weight=w[:cut])
        p_val = model.predict_proba(x.iloc[cut:])[:, 1]
        val_idx = x.index[cut:]
        b_val = benefit_fwd_bps(rate, self.h).reindex(val_idx).to_numpy()
        gate_val = (x[f"rank{self.gate_window}"].reindex(val_idx) <= self.gate_pct).to_numpy()
        self.threshold_ = _choose_threshold(p_val[gate_val], b_val[gate_val], self.fp_cost, self.min_pos_rate)
        self.model_ = model
        self.feature_names_ = list(x.columns)
        self.fitted_ = True
        return self

    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        x = build_features(rate, context)
        proba = pd.Series(np.nan, index=rate.index)
        if self.model_ is not None:
            ok = x.notna().all(axis=1)
            if ok.any():
                proba[ok] = self.model_.predict_proba(x.loc[ok, self.feature_names_])[:, 1]
        key = f"_rank_{self.gate_window}"
        if context is not None and key in context.columns:
            rank = context[key].reindex(rate.index)
        else:
            rank = rolling_pct_rank(rate, self.gate_window)
        cond = (proba >= self.threshold_) & (rank <= self.gate_pct)
        out = pd.DataFrame(index=rate.index)
        out["signal"] = rearm_events(cond, self.rearm)
        out["strength"] = proba.fillna(0.0).clip(0, 1)
        out["proba"] = proba
        out["pct_rank"] = rank
        out["window"] = float(self.gate_window)
        return out


def _choose_threshold(
    p: np.ndarray, benefit: np.ndarray, fp_cost: float, min_pos_rate: float = 0.25
) -> float:
    """Порог по асимметричной выгоде на валидации среди дней, прошедших гейт уровня: ошибочный пуш
    (выгода < 0) штрафуется в fp_cost раз. Среди порогов, оставляющих не меньше min_pos_rate
    гейтовых дней, берётся максимум средней асимметричной выгоды; нет допустимых — медиана."""
    if len(p) == 0:
        return 1.0
    asym = np.where(benefit > 0, benefit, fp_cost * benefit)
    best_thr, best_val = float(np.quantile(p, 0.5)), -np.inf
    for q in np.linspace(0.0, 0.9, 46):
        thr = float(np.quantile(p, q))
        pred = p >= thr
        ok = pred & ~np.isnan(asym)
        if ok.sum() < 10 or pred.mean() < min_pos_rate:
            continue
        val = float(asym[ok].mean())
        if val > best_val:
            best_thr, best_val = thr, val
    return best_thr
