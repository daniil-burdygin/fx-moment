"""Движок: калибровка на обучении, события на тесте, метрики, функция среза.

Инвариант: индикаторы каузальны, поэтому compute() на ряде до test_end и на ряде до T ≤ test_end
дают одинаковые строки на T. На этом стоит signals_as_of и тест на заглядывание вперёд."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fxmoment import metrics
from fxmoment.backtest.walkforward import Split, make_splits, split_for_date
from fxmoment.config import (
    ANALYSIS_START,
    CALIBRATION_FREQ_RANGE,
    CALIBRATION_H,
    CONTEXT,
    CORRIDORS,
    HORIZONS,
    PRIMARY_TOL_BPS,
    TOLERANCES_BPS,
)
from fxmoment.indicators import ALL_INDICATORS, Indicator
from fxmoment.indicators.features import enrich_context

SIGNAL_COLUMNS = [
    "date",
    "corridor",
    "indicator",
    "indicator_label",
    "scenario",
    "direction",
    "strength",
    "speed",
    "split",
    "rate",
    "params",
    "facts",
]


@dataclass
class BacktestResult:
    signals: pd.DataFrame  # события по всем окнам (только signal == True)
    matrix: pd.DataFrame  # индикатор × коридор × окно × горизонт × метрики
    calibration: pd.DataFrame  # выбранные параметры по окнам и коридорам
    splits: list[Split] = field(default_factory=list)

    def summary(self, h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS) -> pd.DataFrame:
        """Сводка по окнам: медианы lift и выгоды, доля окон с lift ≥ 1,3 и < 1."""
        m = self.matrix[(self.matrix["h"] == h) & (self.matrix["tol_bps"] == tol_bps)]
        g = m.groupby(["indicator", "corridor"])
        out = pd.DataFrame(
            {
                "windows": g.size(),
                "events": g["n_events"].sum(),
                "lift_min_median": g["lift"].median(),
                "lift_mean_median": g["lift_mean"].median(),
                "share_lift_mean_ge_1_3": g["lift_mean"].apply(lambda s: float((s >= 1.3).mean())),
                "share_lift_mean_lt_1": g["lift_mean"].apply(lambda s: float((s < 1.0).mean())),
                "hit_mean_pooled": g.apply(
                    lambda d: _pooled(d, "hit_mean", "n_scored"), include_groups=False
                ),
                "base_mean_pooled": g["base_mean"].mean(),
                "benefit_excess_median_bps": g["benefit_excess_bps"].median(),
                "benefit_fwd_median_bps": g["benefit_fwd_bps"].median(),
                "benefit_sym_median_bps": g["benefit_sym_bps"].median(),
                "freq_per_week_median": g["freq_per_week"].median(),
                "empty_month_share_mean": g["empty_month_share"].mean(),
            }
        )
        return out.reset_index()


def _pooled(d: pd.DataFrame, col: str, weight: str) -> float:
    w = d[weight].fillna(0).to_numpy()
    v = d[col].to_numpy()
    ok = ~np.isnan(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan


def _calibrate(
    cls: type[Indicator],
    rate_train: pd.Series,
    ctx_train: pd.DataFrame,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> tuple[dict[str, Any], list[dict]]:
    """Сетка на обучении (ADR-0004): среди точек с n ≥ 30 и частотой индикатора в
    CALIBRATION_FREQ_RANGE — максимум hit rate «по среднему» (= lift_mean) на h при рабочем допуске,
    при равенстве — больше выгода сверх случайного дня. Нет допустимых точек — лучшая по выгоде
    сверх случайного дня среди ближайших к диапазону по частоте.
    Возвращает параметры (с флагами `_feasible`, `_n_feasible`) и журнал сетки."""
    lo, hi = CALIBRATION_FREQ_RANGE
    window = (rate_train.index[0], rate_train.index[-1])
    log = []
    for params in cls.grid():
        ind = cls(**params)
        ev = ind.compute(rate_train, ctx_train)["signal"]
        m = metrics.evaluate_events(rate_train, ev, ind.scenario, h, window, tol_bps, with_ci=False)
        feasible = m["n_scored"] >= 30 and lo <= m["freq_per_week"] <= hi
        log.append({**params, **m, "feasible": feasible})
    df = pd.DataFrame(log)
    feas = df[df["feasible"]]
    if len(feas):
        best = feas.sort_values(["hit_mean", "benefit_excess_bps"], ascending=[False, False]).iloc[0]
    else:
        df = df.assign(_dist=(df["freq_per_week"].clip(lower=lo, upper=hi) - df["freq_per_week"]).abs())
        near = df[df["_dist"] <= df["_dist"].min() + 1e-9]
        best = near.sort_values(["benefit_excess_bps", "n_scored"], ascending=[False, False]).iloc[0]
    keys = list(cls.grid()[0].keys())
    chosen = {k: _native(best[k]) for k in keys}
    chosen["_feasible"] = bool(best["feasible"])
    chosen["_n_feasible"] = int(df["feasible"].sum())
    return chosen, log


def _native(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _fit_indicator(
    cls: type[Indicator], rate_train: pd.Series, ctx_train: pd.DataFrame
) -> tuple[Indicator, dict[str, Any], list[dict]]:
    if cls.trainable:
        ind = cls()
        ind.fit(rate_train, ctx_train)  # type: ignore[attr-defined]
        return ind, {"threshold": round(float(ind.threshold_), 4)}, []  # type: ignore[attr-defined]
    params, log = _calibrate(cls, rate_train, ctx_train)
    ctor = {k: v for k, v in params.items() if not k.startswith("_")}
    return cls(**ctor), params, log


def _event_rows(
    out: pd.DataFrame, ind: Indicator, corridor: str, split: Split, rate: pd.Series, params: dict
) -> list[dict]:
    ev = out[out["signal"].astype(bool)]
    rows = []
    for t, row in ev.iterrows():
        facts = {f: _native(row[f]) for f in ind.fact_fields() if f in row.index}
        rows.append(
            {
                "date": t,
                "corridor": corridor,
                "indicator": ind.name,
                "indicator_label": ind.label(),
                "scenario": ind.scenario,
                "direction": ind.direction,
                "strength": float(row["strength"]),
                "speed": ind.speed,
                "split": split.id,
                "rate": float(rate.loc[t]),
                "params": json.dumps(params, ensure_ascii=False),
                "facts": json.dumps(facts, ensure_ascii=False),
            }
        )
    return rows


def run_backtest(
    panel: pd.DataFrame,
    corridors: tuple[str, ...] = CORRIDORS,
    indicators: tuple[type[Indicator], ...] = ALL_INDICATORS,
    horizons: tuple[int, ...] = HORIZONS,
    analysis_start: str = ANALYSIS_START,
    splits: list[Split] | None = None,
    tolerances: tuple[float, ...] = TOLERANCES_BPS,
) -> BacktestResult:
    panel = panel.loc[pd.Timestamp(analysis_start) :]
    splits = splits or make_splits(panel.index)
    ctx_all = panel[[c for c in CONTEXT if c in panel.columns]]
    sig_rows: list[dict] = []
    mat_rows: list[dict] = []
    cal_rows: list[dict] = []
    for corridor in corridors:
        rate = panel[corridor].dropna()
        ctx = enrich_context(rate, ctx_all)
        for split in splits:
            rate_train = rate.loc[: split.train_end]
            ctx_train = ctx.loc[: split.train_end]
            rate_upto = rate.loc[: split.test_end]
            ctx_upto = ctx.loc[: split.test_end]
            for cls in indicators:
                ind, params, log = _fit_indicator(cls, rate_train, ctx_train)
                cal_rows.append(
                    {
                        "corridor": corridor,
                        "indicator": cls.name,
                        "split": split.id,
                        "window": split.label(),
                        "params": json.dumps(params, ensure_ascii=False),
                    }
                )
                out = ind.compute(rate_upto, ctx_upto).loc[split.test_start : split.test_end]
                sig_rows.extend(_event_rows(out, ind, corridor, split, rate, params))
                for h in horizons:
                    for tol in tolerances:
                        m = metrics.evaluate_events(
                            rate, out["signal"], ind.scenario, h, (split.test_start, split.test_end), tol
                        )
                        mat_rows.append(
                            {
                                "indicator": cls.name,
                                "corridor": corridor,
                                "split": split.id,
                                "window": split.label(),
                                "scenario": ind.scenario,
                                "speed": ind.speed,
                                **m,
                            }
                        )
    signals = pd.DataFrame(sig_rows, columns=SIGNAL_COLUMNS)
    if len(signals):
        signals = signals.sort_values(["corridor", "date", "indicator"]).reset_index(drop=True)
    return BacktestResult(signals, pd.DataFrame(mat_rows), pd.DataFrame(cal_rows), splits)


def signals_as_of(
    panel: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    corridors: tuple[str, ...] = CORRIDORS,
    indicators: tuple[type[Indicator], ...] = ALL_INDICATORS,
    analysis_start: str = ANALYSIS_START,
    splits: list[Split] | None = None,
    lookback: int = 0,
) -> pd.DataFrame:
    """Состояние всех индикаторов на дату среза — по данным с pub_date ≤ cutoff и параметрам,
    откалиброванным на окне, которое действует в эту дату. `lookback` — сколько предыдущих дней
    публикации вернуть вместе с датой среза."""
    cutoff = pd.Timestamp(cutoff)
    full = panel.loc[pd.Timestamp(analysis_start) :]
    splits = splits or make_splits(full.index)
    split = split_for_date(splits, cutoff)
    avail = full.loc[:cutoff]
    ctx_all = avail[[c for c in CONTEXT if c in avail.columns]]
    rows: list[dict] = []
    for corridor in corridors:
        rate = avail[corridor].dropna()
        ctx = enrich_context(rate, ctx_all)
        rate_train = rate.loc[: split.train_end]
        ctx_train = ctx.loc[: split.train_end]
        for cls in indicators:
            ind, params, _ = _fit_indicator(cls, rate_train, ctx_train)
            out = ind.compute(rate, ctx)
            tail = out.iloc[-(lookback + 1) :]
            for t, row in tail.iterrows():
                facts = {f: _native(row[f]) for f in ind.fact_fields() if f in row.index}
                rows.append(
                    {
                        "date": t,
                        "corridor": corridor,
                        "indicator": ind.name,
                        "indicator_label": ind.label(),
                        "scenario": ind.scenario,
                        "direction": ind.direction,
                        "signal": bool(row["signal"]),
                        "strength": float(row["strength"]),
                        "speed": ind.speed,
                        "split": split.id,
                        "rate": float(rate.loc[t]),
                        "params": json.dumps(params, ensure_ascii=False),
                        "facts": json.dumps(facts, ensure_ascii=False),
                    }
                )
    return pd.DataFrame(rows)
