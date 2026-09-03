"""Обучаемый индикатор: классический бустинг предсказывает «день — локальный минимум в окне ±h»,
порог — под асимметричную цену ошибки; пуш — только при факте уровня (ADR-0005)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from fxmoment.config import BUY_NOW, CORRIDORS
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
        self.pos_rate_val_: float = float("nan")  # доля гейтовых дней валидации выше порога
        self.feature_names_: list[str] = []
        self.train_rows_: int = 0  # строк в обучении модели (без валидации)
        self.train_corridors_: int = 0  # коридоров в обучении: 1 у своего, больше у объединённого

    def fact_fields(self) -> tuple[str, ...]:
        return ("proba", "pct_rank", "window")

    def fit_info(self) -> dict[str, Any]:
        """Что дописать в параметры калибровки после обучения; у своего коридора ничего — иначе
        сдвинулись бы байты `calibration.csv` и `signals.csv` прогона по умолчанию."""
        return {}

    def _features(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        return build_features(rate, context)

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
        x = self._features(rate, context)
        y = local_min_label(rate, self.h, self.tol_bps)
        ok = x.notna().all(axis=1) & y.notna()
        if train_start is not None:
            ok &= x.index >= pd.Timestamp(train_start)
        x, y = x[ok], y[ok].astype(int)
        self.fitted_ = False
        self.pos_rate_val_ = float("nan")
        self.train_rows_ = 0
        self.train_corridors_ = 0
        if len(x) < 200 or y.nunique() < 2:
            self.model_ = None
            self.threshold_ = 1.0
            return self
        w = np.where(y.to_numpy() == 1, 1.0, self.fp_cost)
        cut = int(len(x) * 0.8)
        model = self._new_model().fit(x.iloc[:cut], y.iloc[:cut], sample_weight=w[:cut])
        self.train_rows_ = cut
        self.train_corridors_ = 1
        p_val = model.predict_proba(x.iloc[cut:])[:, 1]
        val_idx = x.index[cut:]
        b_val = benefit_fwd_bps(rate, self.h).reindex(val_idx).to_numpy()
        gate_val = (x[f"rank{self.gate_window}"].reindex(val_idx) <= self.gate_pct).to_numpy()
        self.threshold_ = _choose_threshold(p_val[gate_val], b_val[gate_val], self.fp_cost, self.min_pos_rate)
        gated = p_val[gate_val]
        # доля 1,0 значит: порог сел на минимум вероятностей, ML-фильтр не отличим от гейта уровня
        self.pos_rate_val_ = float((gated >= self.threshold_).mean()) if len(gated) else float("nan")
        self.model_ = model
        self.feature_names_ = list(x.columns)
        self.fitted_ = True
        return self

    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        x = self._features(rate, context)
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


class LearnedMinimumPooled(LearnedMinimum):
    """Тот же бустинг, обученный на строках всех коридоров прогона до даты среза с признаком коридора
    (💬 03.09 вечер, пункт 5). Порог — по-прежнему на коридор, среди гейтовых дней его валидации.
    Ряды других коридоров берутся из контекста: движок кладёт их туда, когда в прогоне есть
    объединённый индикатор (`context_columns`); без них обучение сводится к своему коридору, и это
    видно по `_train_corridors` в `calibration.csv`. Имя в матрице то же, `ml_localmin`: вариант
    заменяет обучаемый слот стека, а не добавляет второй, поэтому `compare-runs` сравнивает его с
    базовым напрямую; отличие видно по `pooled=True` в метке и по провенансу отчёта."""

    pooled = True

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.params["pooled"] = True

    def fit_info(self) -> dict[str, Any]:
        return {"_train_corridors": self.train_corridors_, "_train_rows": self.train_rows_}

    def _features(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        f = build_features(rate, context)
        own = _corridor_of(rate)
        for c in CORRIDORS:  # набор фиксирован конфигом: имена признаков не зависят от состава прогона
            f[f"corr_{c}"] = 1.0 if c == own else 0.0
        return f

    def fit(
        self,
        rate: pd.Series,
        context: pd.DataFrame | None = None,
        train_start: str | pd.Timestamp | None = None,
    ) -> LearnedMinimumPooled:
        """Обучение на объединённых строках коридоров до даты среза; дата среза и дни валидации — те
        же, что у `LearnedMinimum` на своём коридоре (последние 20 % его строк), так что отличается
        только модель. Признаки чужих коридоров считаются на контексте без служебного кэша
        `_rank_*` (он посчитан по своему ряду)."""
        own = _corridor_of(rate)
        others = [c for c in CORRIDORS if c != own and context is not None and c in context.columns]
        plain = _currency_context(context)
        xs: dict[str, pd.DataFrame] = {}
        ys: dict[str, pd.Series] = {}
        for c in (own, *others):
            r = rate if c == own else context[c].dropna()  # type: ignore[index]
            x = self._features(r, context if c == own else plain)
            y = local_min_label(r, self.h, self.tol_bps)
            ok = x.notna().all(axis=1) & y.notna()
            if train_start is not None:
                ok &= x.index >= pd.Timestamp(train_start)
            xs[c], ys[c] = x[ok], y[ok].astype(int)
        self.fitted_ = False
        self.pos_rate_val_ = float("nan")
        self.train_rows_ = 0
        self.train_corridors_ = 0
        self.model_ = None
        self.threshold_ = 1.0
        x_own, y_own = xs[own], ys[own]
        if len(x_own) < 200 or y_own.nunique() < 2:
            return self
        cut = int(len(x_own) * 0.8)
        cut_date = x_own.index[cut]  # первая дата валидации своего коридора
        x_tr = pd.concat([xs[c][xs[c].index < cut_date] for c in xs])
        y_tr = pd.concat([ys[c][ys[c].index < cut_date] for c in ys])
        if y_tr.nunique() < 2:
            return self
        w = np.where(y_tr.to_numpy() == 1, 1.0, self.fp_cost)
        model = self._new_model().fit(x_tr, y_tr, sample_weight=w)
        x_val = x_own.iloc[cut:]
        p_val = model.predict_proba(x_val)[:, 1]
        b_val = benefit_fwd_bps(rate, self.h).reindex(x_val.index).to_numpy()
        gate_val = (x_val[f"rank{self.gate_window}"] <= self.gate_pct).to_numpy()
        self.threshold_ = _choose_threshold(p_val[gate_val], b_val[gate_val], self.fp_cost, self.min_pos_rate)
        gated = p_val[gate_val]
        self.pos_rate_val_ = float((gated >= self.threshold_).mean()) if len(gated) else float("nan")
        self.model_ = model
        self.feature_names_ = list(x_own.columns)
        self.train_rows_ = int(len(x_tr))
        self.train_corridors_ = 1 + sum(1 for c in others if bool((xs[c].index < cut_date).any()))
        self.fitted_ = True
        return self


def _corridor_of(rate: pd.Series) -> str:
    """Коридор — имя ряда: движок берёт столбец панели, и имя столбца приходит вместе с ним."""
    return "" if rate.name is None else str(rate.name)


def _currency_context(context: pd.DataFrame | None) -> pd.DataFrame | None:
    """Контекст без служебных столбцов `_rank_*`, `_dsm_*`: они посчитаны по своему ряду."""
    if context is None:
        return None
    return context[[c for c in context.columns if not str(c).startswith("_")]]


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
