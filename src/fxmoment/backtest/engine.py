"""Движок: калибровка на обучении, события на тесте, метрики, функция среза.

Инвариант: индикаторы каузальны, поэтому compute() на ряде до test_end и на ряде до T ≤ test_end
дают одинаковые строки на T. На этом стоит signals_as_of и тест на заглядывание вперёд.

Индикаторы считаются по всей доступной истории (сырой ряд ЦБ с 2015): разогрев окон и сезонный
профиль берутся из неё. Обучение, калибровка и оценка начинаются с analysis_start (ADR-0004)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fxmoment import labels, metrics
from fxmoment.backtest.walkforward import Split, make_splits, split_for_date
from fxmoment.config import (
    ANALYSIS_START,
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

# Исходы, усечённые концом тестового окна: для ранжирования индикаторов в следующем окне (ADR-0006).
TRUNC_COLUMNS = metrics.TRUNC_COLUMNS


@dataclass
class BacktestResult:
    signals: pd.DataFrame  # события по всем окнам (только signal == True)
    matrix: pd.DataFrame  # индикатор × коридор × окно × горизонт × метрики
    calibration: pd.DataFrame  # выбранные параметры по окнам и коридорам
    splits: list[Split] = field(default_factory=list)

    def summary(self, h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS) -> pd.DataFrame:
        """Сводка по окнам: медианы lift и выгоды, доли окон с lift ≥ 1,3 и < 1 среди окон с событиями,
        число молчащих окон, pooled hit и база (обе взвешены по событиям окна)."""
        m = self.matrix[(self.matrix["h"] == h) & (self.matrix["tol_bps"] == tol_bps)]
        if m.empty:
            return pd.DataFrame()
        g = m.groupby(["indicator", "corridor"])
        out = pd.DataFrame(
            {
                "windows": g.size(),
                "silent_windows": g["n_scored"].apply(lambda s: int((s.fillna(0) == 0).sum())),
                "events": g["n_events"].sum(),
                "lift_min_median": g["lift"].median(),
                "lift_mean_median": g["lift_mean"].median(),
                "share_lift_mean_ge_1_3": g["lift_mean"].apply(lambda s: _share(s, 1.3, ge=True)),
                "share_lift_mean_lt_1": g["lift_mean"].apply(lambda s: _share(s, 1.0, ge=False)),
                "hit_mean_pooled": g.apply(
                    lambda d: _pooled(d, "hit_mean", "n_scored"), include_groups=False
                ),
                "base_mean_pooled": g.apply(
                    lambda d: _pooled(d, "base_mean", "n_scored"), include_groups=False
                ),
                "benefit_excess_median_bps": g["benefit_excess_bps"].median(),
                "benefit_fwd_median_bps": g["benefit_fwd_bps"].median(),
                "benefit_sym_median_bps": g["benefit_sym_bps"].median(),
                "freq_per_week_median": g["freq_per_week"].median(),
                "empty_month_share_mean": g["empty_month_share"].mean(),
            }
        )
        out["lift_mean_pooled"] = out["hit_mean_pooled"] / out["base_mean_pooled"]
        return out.reset_index()


def _share(s: pd.Series, thr: float, ge: bool) -> float:
    v = s.dropna()
    if not len(v):
        return np.nan
    return float((v >= thr).mean()) if ge else float((v < thr).mean())


def _pooled(d: pd.DataFrame, col: str, weight: str) -> float:
    w = d[weight].fillna(0).to_numpy(dtype=float)
    v = d[col].to_numpy(dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan


def _calibrate(
    cls: type[Indicator],
    rate_train: pd.Series,
    ctx_train: pd.DataFrame,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
    eval_start: str | pd.Timestamp | None = None,
) -> tuple[dict[str, Any], list[dict]]:
    """Сетка на обучении (ADR-0004). Допустимая точка: n ≥ min_n и частота индикатора в [lo, hi]
    (границы — `cls.calibration_bounds()`, по умолчанию CALIBRATION_FREQ_RANGE и
    MIN_CALIBRATION_EVENTS; медленный индикатор задаёт свои), обе меряются после разогрева
    индикатора и не раньше eval_start.
    Целевая функция — **медиана lift «по среднему» по календарным годам обучения** (год входит, если
    в нём ≥ 5 событий): параметры должны работать в большинстве лет, а не в одном удачном; при
    равенстве — больше выгода сверх случайного дня. Максимум сырого hit rate по всему окну
    (первая версия) выбирал точку с одним удачным годом и проваливался вне выборки.
    Нет допустимых точек — среди ближайших к диапазону по частоте лучшая по той же целевой функции
    (медиана годовых lift), затем по выгоде; частота здесь только отсечка, иначе запасная ветка
    выбирала самую частую точку, а не самую точную (аудит 03.09). Возвращает параметры (с флагами
    `_feasible`, `_n_feasible`, `_score`) и журнал сетки."""
    lo, hi, min_n = cls.calibration_bounds()
    hit_all = labels.hit_for_scenario(rate_train, cls.scenario, h, tol_bps, mode="mean")
    bf_all = labels.benefit_fwd_bps(rate_train, h)
    years_all = pd.Series(rate_train.index.year, index=rate_train.index)
    log = []
    for params in cls.grid():
        ind = cls(**params)
        ev = ind.compute(rate_train, ctx_train)["signal"]
        first = rate_train.index[min(ind.warmup(rate_train.index), len(rate_train) - 1)]
        start = max(first, pd.Timestamp(eval_start)) if eval_start is not None else first
        days = rate_train.loc[start:].index
        ev_days = ev[ev.astype(bool)].index.intersection(days)
        hits = hit_all.reindex(ev_days).dropna()
        base = hit_all.reindex(days).dropna()
        n_scored = int(len(hits))
        freq = metrics.frequency_per_week(len(ev_days), start, rate_train.index[-1])
        excess = (
            float(bf_all.reindex(ev_days).dropna().mean() - bf_all.reindex(days).dropna().mean())
            if n_scored
            else np.nan
        )
        yearly = []
        for y, hy in hits.groupby(years_all.reindex(hits.index)):
            by = base[years_all.reindex(base.index) == y]
            if len(hy) >= 5 and len(by) and by.mean() > 0:
                yearly.append(float(hy.mean() / by.mean()))
        score = float(np.median(yearly)) if yearly else np.nan
        feasible = n_scored >= min_n and lo <= freq <= hi and bool(yearly)
        log.append(
            {
                **params,
                "n_events": int(len(ev_days)),
                "n_scored": n_scored,
                "hit_mean": float(hits.mean()) if n_scored else np.nan,
                "base_mean": float(base.mean()) if len(base) else np.nan,
                "freq_per_week": freq,
                "benefit_excess_bps": excess,
                "score": score,
                "years": len(yearly),
                "feasible": feasible,
            }
        )
    df = pd.DataFrame(log)
    feas = df[df["feasible"]]
    if len(feas):
        best = feas.sort_values(["score", "benefit_excess_bps"], ascending=[False, False]).iloc[0]
    else:
        df = df.assign(_dist=(df["freq_per_week"].clip(lower=lo, upper=hi) - df["freq_per_week"]).abs())
        near = df[df["_dist"] <= df["_dist"].min() + 1e-9]
        best = near.sort_values(
            ["score", "benefit_excess_bps", "n_scored"], ascending=[False, False, False], na_position="last"
        ).iloc[0]
    keys = list(cls.grid()[0].keys())
    chosen = {k: _native(best[k]) for k in keys}
    chosen["_feasible"] = bool(best["feasible"])
    chosen["_n_feasible"] = int(df["feasible"].sum())
    chosen["_score"] = round(float(best["score"]), 4) if not np.isnan(best["score"]) else None
    return chosen, log


def _native(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def fit_indicator(
    cls: type[Indicator],
    rate_train: pd.Series,
    ctx_train: pd.DataFrame,
    eval_start: str | pd.Timestamp | None = None,
    fixed_params: bool = False,
) -> tuple[Indicator, dict[str, Any], list[dict]]:
    """Индикатор с параметрами, выбранными только по данным до конца обучения.
    fixed_params — правила берут априорные параметры по умолчанию (заданы до первого прогона)
    без сетки: контрольный прогон «стоит ли калибровка своих денег»; ML обучается как обычно."""
    if fixed_params and not cls.trainable:
        ind = cls()
        return ind, {**ind.params, "_feasible": None, "_n_feasible": 0, "_fixed": True}, []
    if cls.trainable:
        ind = cls()
        ind.fit(rate_train, ctx_train, train_start=eval_start)  # type: ignore[attr-defined]
        params = {
            **ind.params,  # h, tol_bps, gate_window, gate_pct, fp_cost, min_pos_rate, rearm, seed
            "threshold": round(float(ind.threshold_), 4),  # type: ignore[attr-defined]
            "_fitted": bool(ind.fitted_),  # type: ignore[attr-defined]
            # доля гейтовых дней валидации выше порога; 1,0 — порог сел на минимум, фильтр пуст
            "_pos_rate_val": round(float(ind.pos_rate_val_), 3)  # type: ignore[attr-defined]
            if ind.fitted_  # type: ignore[attr-defined]
            else None,
        }
        return ind, params, []
    params, log = _calibrate(cls, rate_train, ctx_train, eval_start=eval_start)
    ctor = {k: v for k, v in params.items() if not k.startswith("_")}
    return cls(**ctor), params, log


_fit_indicator = fit_indicator


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
    fixed_params: bool = False,
) -> BacktestResult:
    ana = panel.loc[pd.Timestamp(analysis_start) :]
    splits = splits or make_splits(ana.index)
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
            win = (split.test_start, split.test_end)
            for cls in indicators:
                ind, params, log = fit_indicator(
                    cls, rate_train, ctx_train, eval_start=analysis_start, fixed_params=fixed_params
                )
                cal_rows.append(
                    {
                        "corridor": corridor,
                        "indicator": cls.name,
                        "split": split.id,
                        "window": split.label(),
                        "params": json.dumps(params, ensure_ascii=False),
                        "feasible": params.get("_feasible"),
                        "n_feasible": params.get("_n_feasible"),
                        "fitted": params.get("_fitted"),
                    }
                )
                out = ind.compute(rate_upto, ctx_upto).loc[split.test_start : split.test_end]
                sig_rows.extend(_event_rows(out, ind, corridor, split, rate, params))
                for h in horizons:
                    for tol in tolerances:
                        m = metrics.evaluate_events(rate, out["signal"], ind.scenario, h, win, tol)
                        row = {
                            "indicator": cls.name,
                            "corridor": corridor,
                            "split": split.id,
                            "window": split.label(),
                            "scenario": ind.scenario,
                            "speed": ind.speed,
                            **m,
                        }
                        if h == CALIBRATION_H and tol == PRIMARY_TOL_BPS:
                            mt = metrics.evaluate_events(
                                rate_upto, out["signal"], ind.scenario, h, win, tol, with_ci=False
                            )
                            row.update({c: mt[s] for c, s in metrics.TRUNC_SOURCE.items()})
                        mat_rows.append(row)
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
    fixed_params: bool = False,
) -> pd.DataFrame:
    """Состояние всех индикаторов на дату среза — по данным с pub_date ≤ cutoff и параметрам,
    откалиброванным на окне, которое действует в эту дату; после последнего тестового окна — на
    живом окне (обучение до его начала минус зазор, `split_for_date`). `lookback` — сколько
    предыдущих дней публикации вернуть вместе с датой среза."""
    cutoff = pd.Timestamp(cutoff)
    ana = panel.loc[pd.Timestamp(analysis_start) :]
    splits = splits or make_splits(ana.index)
    split = split_for_date(splits, cutoff, ana.index)  # после последнего окна — живое окно
    avail = panel.loc[:cutoff]
    ctx_all = avail[[c for c in CONTEXT if c in avail.columns]]
    rows: list[dict] = []
    for corridor in corridors:
        rate = avail[corridor].dropna()
        ctx = enrich_context(rate, ctx_all)
        rate_train = rate.loc[: split.train_end]
        ctx_train = ctx.loc[: split.train_end]
        for cls in indicators:
            ind, params, _ = fit_indicator(
                cls, rate_train, ctx_train, eval_start=analysis_start, fixed_params=fixed_params
            )
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
