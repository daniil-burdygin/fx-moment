"""Метрики сигнала (ADR-0003): hit rate, база случайного дня, lift, выгода, частота, кучность,
цена ожидания. Все считаются по событиям (True в булевом ряду) внутри оценочного окна."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxmoment import labels


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def block_bootstrap_ci(
    values: pd.Series, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Бутстреп по календарным месяцам: события внутри месяца пересекаются по горизонту."""
    v = values.dropna()
    if len(v) < 2:
        return (np.nan, np.nan)
    blocks = [g.to_numpy() for _, g in v.groupby(v.index.to_period("M"))]
    if len(blocks) < 2:
        return bootstrap_ci(v.to_numpy(), n_boot, alpha, seed)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(blocks), size=len(blocks))
        means[b] = np.concatenate([blocks[i] for i in pick]).mean()
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def frequency_per_week(event_index: pd.DatetimeIndex, window: pd.DatetimeIndex) -> float:
    weeks = max((window[-1] - window[0]).days, 1) / 7
    return len(event_index) / weeks


@dataclass(frozen=True)
class Clumpiness:
    share_series: float  # доля событий с промежутком ≤ 3 дней публикации от предыдущего
    cv_gaps: float  # коэффициент вариации промежутков (дни публикации)
    longest_gap_days: float  # самый длинный промежуток без событий, календарных дней
    empty_month_share: float  # доля календарных месяцев окна без событий


def clumpiness(event_index: pd.DatetimeIndex, window: pd.DatetimeIndex, series_gap: int = 3) -> Clumpiness:
    months = pd.period_range(window[0], window[-1], freq="M")
    if len(event_index) == 0:
        return Clumpiness(np.nan, np.nan, float((window[-1] - window[0]).days), 1.0)
    pos = window.get_indexer(event_index)
    gaps = np.diff(pos)
    share = float((gaps <= series_gap).mean()) if len(gaps) else 0.0
    cv = float(gaps.std() / gaps.mean()) if len(gaps) and gaps.mean() > 0 else np.nan
    edges = [window[0], *event_index, window[-1]]
    longest = max((b - a).days for a, b in zip(edges[:-1], edges[1:], strict=True))
    busy = set(event_index.to_period("M"))
    empty = sum(1 for m in months if m not in busy) / len(months)
    return Clumpiness(share, cv, float(longest), float(empty))


def evaluate_events(
    rate: pd.Series,
    events: pd.Series,
    scenario: str,
    h: int,
    window: tuple[pd.Timestamp, pd.Timestamp],
    tol_bps: float = 0.0,
    wait_k: int = 5,
    with_ci: bool = True,
) -> dict:
    """Метрики одного индикатора на одном коридоре в одном оценочном окне для горизонта h."""
    start, end = window
    days = rate.loc[start:end].index
    hit_all = labels.hit_for_scenario(rate, scenario, h, tol_bps)
    base_days = hit_all.loc[days].dropna()
    base = float(base_days.mean()) if len(base_days) else np.nan
    ev_idx = events.loc[start:end]
    ev_idx = ev_idx[ev_idx.fillna(False).astype(bool)].index
    hits = hit_all.loc[ev_idx].dropna()
    n = int(len(hits))
    hit = float(hits.mean()) if n else np.nan
    lift = hit / base if n and base and base > 0 else np.nan
    # прочтение «по среднему»: клиент против своего типичного дня в горизонте
    hit_mean_all = labels.hit_for_scenario(rate, scenario, h, tol_bps, mode="mean")
    base_mean = float(hit_mean_all.loc[days].dropna().mean()) if len(days) else np.nan
    hits_mean = hit_mean_all.loc[ev_idx].dropna()
    hit_mean = float(hits_mean.mean()) if len(hits_mean) else np.nan
    lift_mean = hit_mean / base_mean if len(hits_mean) and base_mean and base_mean > 0 else np.nan
    bf_all = labels.benefit_fwd_bps(rate, h)
    bf_base = float(bf_all.loc[days].dropna().mean()) if len(days) else np.nan  # выгода случайного дня
    bf = bf_all.loc[ev_idx].dropna()
    bs = labels.benefit_sym_bps(rate, h).loc[ev_idx].dropna()
    wk = labels.wait_benefit_bps(rate, wait_k).loc[ev_idx].dropna()
    ci_lo, ci_hi = bootstrap_ci(bf.to_numpy()) if (with_ci and n >= 5) else (np.nan, np.nan)
    bci_lo, bci_hi = block_bootstrap_ci(bf) if (with_ci and n >= 5) else (np.nan, np.nan)
    lift_lo, lift_hi = (np.nan, np.nan)
    if with_ci and n >= 5 and base and base > 0:
        lo, hi = bootstrap_ci(hits.to_numpy())
        lift_lo, lift_hi = lo / base, hi / base
    cl = clumpiness(ev_idx, days)
    return {
        "h": h,
        "tol_bps": tol_bps,
        "n_events": int(len(ev_idx)),
        "n_scored": n,
        "hit_rate": hit,
        "base_rate": base,
        "lift": lift,
        "lift_ci_lo": lift_lo,
        "lift_ci_hi": lift_hi,
        "hit_mean": hit_mean,
        "base_mean": base_mean,
        "lift_mean": lift_mean,
        "benefit_fwd_bps": float(bf.mean()) if len(bf) else np.nan,
        "benefit_random_day_bps": bf_base,
        "benefit_excess_bps": (float(bf.mean()) - bf_base) if len(bf) and not np.isnan(bf_base) else np.nan,
        "benefit_excess_ci_lo": (ci_lo - bf_base)
        if not np.isnan(ci_lo) and not np.isnan(bf_base)
        else np.nan,
        "benefit_excess_ci_hi": (ci_hi - bf_base)
        if not np.isnan(ci_hi) and not np.isnan(bf_base)
        else np.nan,
        "benefit_fwd_ci_lo": ci_lo,
        "benefit_fwd_ci_hi": ci_hi,
        "benefit_fwd_block_ci_lo": bci_lo,
        "benefit_fwd_block_ci_hi": bci_hi,
        "benefit_sym_bps": float(bs.mean()) if len(bs) else np.nan,
        f"vs_wait{wait_k}_bps": float(wk.mean()) if len(wk) else np.nan,
        "freq_per_week": frequency_per_week(ev_idx, days),
        "clump_share_series": cl.share_series,
        "clump_cv_gaps": cl.cv_gaps,
        "longest_gap_days": cl.longest_gap_days,
        "empty_month_share": cl.empty_month_share,
    }


def price_of_waiting(
    rate: pd.Series, fast_events: pd.Series, slow_events: pd.Series, k: int = 10
) -> pd.DataFrame:
    """Для каждого быстрого сигнала — первое медленное подтверждение в пределах k дней публикации
    и изменение курса действия между ними, в бп (> 0 — ожидание стоило денег)."""
    idx = rate.index
    fast_idx = fast_events[fast_events.fillna(False).astype(bool)].index
    slow_pos = np.flatnonzero(slow_events.reindex(idx).fillna(False).astype(bool).to_numpy())
    rows = []
    for t in fast_idx:
        p = idx.get_loc(t)
        later = slow_pos[(slow_pos > p) & (slow_pos <= p + k)]
        if len(later):
            q = int(later[0])
            rows.append((t, idx[q], q - p, (rate.iloc[q] / rate.iloc[p] - 1) * 1e4, True))
        else:
            rows.append((t, pd.NaT, np.nan, np.nan, False))
    return pd.DataFrame(rows, columns=["fast_date", "slow_date", "days_waited", "delta_bps", "confirmed"])
