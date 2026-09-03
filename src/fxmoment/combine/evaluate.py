"""Оценка итогового потока: политика применяется к событиям каждого окна, отправленные пуши
меряются теми же метриками, что и индикаторы. Ранг надёжности — walk-forward: для окна k по
исходам окон < k, усечённым концом каждого окна (ADR-0006)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment import metrics
from fxmoment.backtest.engine import BacktestResult
from fxmoment.backtest.walkforward import Split
from fxmoment.combine.policy import PolicyParams, apply_policy, storm_flag
from fxmoment.config import CALIBRATION_H, FREQUENCY_BAND, HORIZONS, PRIMARY_TOL_BPS, TOLERANCES_BPS

DEFAULT_ORDER = ("level", "dip_vs_trend", "ml_localmin", "seasonality", "reversal", "momentum")
TRUNC = ("hit_mean_trunc", "base_mean_trunc", "n_scored_trunc", "benefit_excess_trunc")


def rank_from_history(
    matrix: pd.DataFrame, corridor: str, before_split: int, h: int = CALIBRATION_H, min_events: int = 20
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Ранг индикаторов по pooled lift «по среднему» на окнах < before_split и список отключённых.

    Исходы берутся усечённые концом окна (`*_trunc`): полные исходы последних h дней окна k−1
    досчитываются внутри окна k, и без усечения ранг подсматривал бы в будущее (аудит 03.09).
    Индикатор глушится, если при ≥ min_events событиях pooled lift < 1 ИЛИ средняя выгода сверх
    случайного дня ≤ 0: по условиям кейса сигнал без выгоды в бп не засчитывается, даже если
    угадывает направление. Без истории (или без усечённых столбцов) — порядок по умолчанию."""
    default = {name: i for i, name in enumerate(DEFAULT_ORDER)}
    m = matrix[
        (matrix["corridor"] == corridor)
        & (matrix["split"] < before_split)
        & (matrix["h"] == h)
        & (matrix["tol_bps"] == PRIMARY_TOL_BPS)
    ]
    if m.empty or not set(TRUNC) <= set(m.columns):
        return default, ()
    m = m.dropna(subset=["hit_mean_trunc"])
    if m.empty:
        return default, ()
    g = m.groupby("indicator")
    n = g["n_scored_trunc"].sum()
    hit = g.apply(
        lambda d: (d["hit_mean_trunc"] * d["n_scored_trunc"]).sum() / max(d["n_scored_trunc"].sum(), 1),
        include_groups=False,
    )
    base = g.apply(
        lambda d: (d["base_mean_trunc"] * d["n_scored_trunc"]).sum() / max(d["n_scored_trunc"].sum(), 1),
        include_groups=False,
    )
    lift = (hit / base.replace(0, np.nan)).astype(float)
    excess = g.apply(
        lambda d: (d["benefit_excess_trunc"] * d["n_scored_trunc"]).sum() / max(d["n_scored_trunc"].sum(), 1),
        include_groups=False,
    )
    muted = tuple(str(i) for i in lift.index if n[i] >= min_events and (lift[i] < 1.0 or not excess[i] > 0))
    rank = {str(name): i for i, name in enumerate(lift.fillna(-1).sort_values(ascending=False).index)}
    for name in DEFAULT_ORDER:  # индикаторы без истории — после ранжированных
        rank.setdefault(name, len(rank))
    return rank, muted


def _shape_row(
    rate: pd.Series,
    sent_dates: pd.Series | pd.DatetimeIndex,
    split: Split,
    storm: pd.Series,
    dec: pd.DataFrame,
) -> dict:
    days = rate.loc[split.test_start : split.test_end].index
    ev_idx = pd.DatetimeIndex(sent_dates)
    cl = metrics.clumpiness(ev_idx, days)
    end = min(split.test_end, rate.index[-1])
    return {
        "split": split.id,
        "window": split.label(),
        "pushes": int(len(ev_idx)),
        "storm_days": int(storm.reindex(days).fillna(False).sum()),
        "storm_blocked": int((dec["decision"] == "storm").sum()) if len(dec) else 0,
        "freq_per_week": metrics.frequency_per_week(len(ev_idx), split.test_start, end),
        "clump_share_series": cl.share_series,
        "clump_cv_gaps": cl.cv_gaps,
        "longest_gap_days": cl.longest_gap_days,
        "empty_month_share": cl.empty_month_share,
    }


def evaluate_stream(
    result: BacktestResult,
    panel: pd.DataFrame,
    params: PolicyParams | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    tolerances: tuple[float, ...] = TOLERANCES_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает (события с решениями политики, матрица метрик потока по коридорам, окнам и
    сценариям, форма потока по коридорам и окнам — частота и кучность всех пушей вместе).
    История отправок переносится через границы окон: охлаждение и конфликт действуют как в живой
    системе."""
    params = params or PolicyParams()
    decided: list[pd.DataFrame] = []
    rows: list[dict] = []
    shape: list[dict] = []
    for corridor in result.signals["corridor"].unique():
        rate = panel[corridor].dropna()
        storm = storm_flag(rate, params.storm_vol_window, params.storm_rank_window, params.storm_rank)
        history: list[tuple[pd.Timestamp, str]] = []
        for split in result.splits:
            ev = result.signals[
                (result.signals["corridor"] == corridor) & (result.signals["split"] == split.id)
            ]
            rank, muted = rank_from_history(result.matrix, corridor, split.id)
            dec = ev.iloc[0:0]
            if ev.empty:
                sent = ev.iloc[0:0]
            else:
                p = PolicyParams(**{**params.__dict__, "muted": params.muted + muted})
                dec = apply_policy(ev, rank, rate.index, p, prior_sent={corridor: history}, storm=storm)
                decided.append(dec)
                sent = dec[dec["decision"] == "sent"]
                history.extend(zip(sent["date"], sent["push_scenario"], strict=True))
            shape.append(
                {
                    "corridor": corridor,
                    **_shape_row(rate, sent["date"] if len(sent) else [], split, storm, dec),
                }
            )
            for scenario in sent["push_scenario"].unique() if len(sent) else []:
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
    return decided_df, pd.DataFrame(rows), pd.DataFrame(shape)


def _pooled(d: pd.DataFrame, col: str) -> float:
    w = d["n_scored"].fillna(0).to_numpy(dtype=float)
    v = d[col].to_numpy(dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan


def stream_summary(
    stream_matrix: pd.DataFrame, h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    """Точность потока по коридорам и сценариям: pooled hit и база «по среднему» (взвешены по
    событиям окна), медианы lift и выгоды по окнам."""
    if stream_matrix.empty:
        return pd.DataFrame()
    m = stream_matrix[(stream_matrix["h"] == h) & (stream_matrix["tol_bps"] == tol_bps)]
    g = m.groupby(["corridor", "scenario"])
    out = pd.DataFrame(
        {
            "windows": g.size(),
            "pushes": g["n_events"].sum(),
            "hit_mean_pooled": g.apply(lambda d: _pooled(d, "hit_mean"), include_groups=False),
            "base_mean_pooled": g.apply(lambda d: _pooled(d, "base_mean"), include_groups=False),
            "lift_mean_median": g["lift_mean"].median(),
            "lift_min_median": g["lift"].median(),
            "benefit_excess_median_bps": g["benefit_excess_bps"].median(),
            "benefit_fwd_median_bps": g["benefit_fwd_bps"].median(),
            "freq_per_week_median": g["freq_per_week"].median(),
        }
    )
    out["lift_mean_pooled"] = out["hit_mean_pooled"] / out["base_mean_pooled"]
    return out.reset_index()


def stream_shape_summary(shape: pd.DataFrame) -> pd.DataFrame:
    """Форма потока по коридорам: все отправленные пуши вместе, без разбивки по сценариям —
    полоса 1–2 в неделю относится к коридору (ADR-0006 п. 1)."""
    if shape.empty:
        return pd.DataFrame()
    lo, hi = FREQUENCY_BAND
    g = shape.groupby("corridor")
    return pd.DataFrame(
        {
            "windows": g.size(),
            "pushes": g["pushes"].sum(),
            "freq_per_week_median": g["freq_per_week"].median(),
            "freq_per_week_min": g["freq_per_week"].min(),
            "freq_per_week_max": g["freq_per_week"].max(),
            "share_windows_in_band": g["freq_per_week"].apply(
                lambda s: float(((s >= lo) & (s <= hi)).mean())
            ),
            "empty_month_share_mean": g["empty_month_share"].mean(),
            "longest_gap_days_max": g["longest_gap_days"].max(),
            "clump_share_series_mean": g["clump_share_series"].mean(),
            "storm_days": g["storm_days"].sum(),
            "storm_blocked": g["storm_blocked"].sum(),
        }
    ).reset_index()
