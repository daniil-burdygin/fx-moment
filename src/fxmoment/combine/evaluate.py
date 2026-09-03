"""Оценка итогового потока: политика применяется к событиям каждого окна, отправленные пуши
меряются теми же метриками, что и индикаторы. Ранг надёжности — по hit rate индикатора
на предыдущих окнах (walk-forward: ранг для окна k берётся из окон < k)."""

from __future__ import annotations

import pandas as pd

from fxmoment import metrics
from fxmoment.backtest.engine import BacktestResult
from fxmoment.combine.policy import PolicyParams, apply_policy
from fxmoment.config import CALIBRATION_H, HORIZONS, PRIMARY_TOL_BPS, TOLERANCES_BPS


def rank_from_history(
    matrix: pd.DataFrame, corridor: str, before_split: int, h: int = CALIBRATION_H
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Ранг индикаторов по hit rate «по среднему» на окнах < before_split и список отключённых:
    индикатор глушится, если на прошлых окнах его pooled lift_mean < 1 при ≥ 20 событиях.
    Без истории — порядок по скорости, ничего не отключено."""
    m = matrix[
        (matrix["corridor"] == corridor)
        & (matrix["split"] < before_split)
        & (matrix["h"] == h)
        & (matrix["tol_bps"] == PRIMARY_TOL_BPS)
    ]
    if m.empty:
        order = ["level", "dip_vs_trend", "ml_localmin", "seasonality", "reversal", "momentum"]
        return {name: i for i, name in enumerate(order)}, ()
    g = m.groupby("indicator")
    pooled_hit = g.apply(
        lambda d: (d["hit_mean"] * d["n_scored"]).sum() / max(d["n_scored"].sum(), 1), include_groups=False
    )
    pooled_base = g["base_mean"].mean()
    n = g["n_scored"].sum()
    lift = pooled_hit / pooled_base
    muted = tuple(str(i) for i in lift.index if n[i] >= 20 and lift[i] < 1.0)
    rank = {name: i for i, name in enumerate(pooled_hit.sort_values(ascending=False).index)}
    return rank, muted


def evaluate_stream(
    result: BacktestResult,
    panel: pd.DataFrame,
    params: PolicyParams | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    tolerances: tuple[float, ...] = TOLERANCES_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (события с решениями политики, матрица метрик потока по коридорам и окнам)."""
    params = params or PolicyParams()
    decided: list[pd.DataFrame] = []
    rows: list[dict] = []
    for corridor in result.signals["corridor"].unique():
        rate = panel[corridor].dropna()
        for split in result.splits:
            ev = result.signals[
                (result.signals["corridor"] == corridor) & (result.signals["split"] == split.id)
            ]
            if ev.empty:
                continue
            rank, muted = rank_from_history(result.matrix, corridor, split.id)
            calendar = rate.loc[split.test_start : split.test_end].index
            dec = apply_policy(
                ev, rank, calendar, PolicyParams(**{**params.__dict__, "muted": params.muted + muted})
            )
            decided.append(dec)
            sent = dec[dec["decision"] == "sent"]
            for scenario in sent["push_scenario"].unique():
                sc = sent[sent["push_scenario"] == scenario]
                events = pd.Series(False, index=rate.index)
                events.loc[sc["date"]] = True
                for h in horizons:
                    for tol in tolerances:
                        m = metrics.evaluate_events(
                            rate, events, scenario, h, (split.test_start, split.test_end), tol
                        )
                        rows.append(
                            {
                                "corridor": corridor,
                                "split": split.id,
                                "window": split.label(),
                                "scenario": scenario,
                                "sent_total": int(len(sent)),
                                **m,
                            }
                        )
    decided_df = pd.concat(decided, ignore_index=True) if decided else result.signals.iloc[0:0]
    return decided_df, pd.DataFrame(rows)


def stream_summary(
    stream_matrix: pd.DataFrame, h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    m = stream_matrix[(stream_matrix["h"] == h) & (stream_matrix["tol_bps"] == tol_bps)]
    g = m.groupby(["corridor", "scenario"])
    return pd.DataFrame(
        {
            "windows": g.size(),
            "pushes": g["n_events"].sum(),
            "hit_pooled": g.apply(
                lambda d: (d["hit_rate"] * d["n_scored"]).sum() / max(d["n_scored"].sum(), 1),
                include_groups=False,
            ),
            "base_pooled": g["base_rate"].mean(),
            "lift_median": g["lift"].median(),
            "benefit_fwd_median_bps": g["benefit_fwd_bps"].median(),
            "freq_per_week_median": g["freq_per_week"].median(),
            "empty_month_share_mean": g["empty_month_share"].mean(),
            "longest_gap_days_max": g["longest_gap_days"].max(),
        }
    ).reset_index()
