"""Разворот как факт, а не прогноз (💬 03.09 вечер, пункт 6).

Сценарий «окно закрывается» до сих пор мерился как прогноз — «через h дней курс не ниже» — и на
четырёх коридорах из пяти это монетка. Но клиенту пуш о развороте нужен не как прогноз: он говорит
тому, кто получил `BUY_NOW` и не перевёл, сколько уже стоило ожидание. Это факт о прошлом, и он
меряется сожалением: изменение курса действия с даты последнего `BUY_NOW` за k дней публикации до
разворота, бп; > 0 — курс вырос, ожидание стоило денег.

Рядом — прогнозные прочтения на горизонтах клиента, чтобы монетку не прятать: через h дней и до конца
календарного месяца («сегодня не хуже оставшихся дней месяца»), оба против базы случайного дня
своего окна. Текст пуша не меняется: он и так называет только прошлое (инвариант 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment import labels, metrics
from fxmoment.backtest.walkforward import Split
from fxmoment.config import BUY_NOW, CALIBRATION_H, WINDOW_CLOSING

PAIRINGS: tuple[str, ...] = ("events", "stream")  # по событиям индикаторов / по пушам итогового потока
REGRET_LOOKBACK = 20  # дней публикации назад, где ищется последний BUY_NOW (≈ месяц клиента)
REGRET_THRESHOLD_BPS = 25.0  # рабочий допуск: сожаление меньше него клиент не заметит
MIN_CI_EVENTS = 5
COLUMNS = [
    "corridor",
    "pairing",
    "n_reversal",
    "n_paired",
    "paired_share",
    "days_since_buy_median",
    "regret_median_bps",
    "regret_mean_bps",
    "regret_ci_lo",
    "regret_ci_hi",
    "share_regret_positive",
    "share_regret_gt_tol",
    "hit_fwd",
    "base_fwd",
    "lift_fwd",
    "hit_rest_of_month",
    "base_rest_of_month",
    "lift_rest_of_month",
]


def rest_of_month_hit(rate: pd.Series) -> pd.Series:
    """1,0 — курс дня не выше среднего оставшихся дней публикации того же календарного месяца
    («сегодня не хуже, чем ждать до конца месяца»); NaN в последний день публикации месяца.
    Смотрит в будущее — только для оценки, никогда внутри индикатора."""
    month = rate.index.to_period("M")
    rev = rate[::-1]
    rev_month = month[::-1]
    tail_sum = rev.groupby(rev_month).cumsum()[::-1] - rate
    tail_cnt = pd.Series(1.0, index=rate.index)[::-1].groupby(rev_month).cumsum()[::-1] - 1.0
    rest_mean = tail_sum / tail_cnt.replace(0.0, np.nan)
    out = (rest_mean >= rate).astype(float)
    out[rest_mean.isna()] = np.nan
    return out


def pair_reversals(
    rate: pd.Series, reversal_dates: pd.DatetimeIndex, buy_dates: pd.DatetimeIndex, k: int = REGRET_LOOKBACK
) -> pd.DataFrame:
    """Для каждого разворота — последний `BUY_NOW` в пределах k дней публикации до него и изменение
    курса действия между ними в бп."""
    idx = rate.index
    buy_pos = np.unique(idx.get_indexer(buy_dates))
    buy_pos = buy_pos[buy_pos >= 0]
    rows = []
    for t1 in reversal_dates:
        if t1 not in idx:
            continue
        p1 = int(idx.get_loc(t1))
        prior = buy_pos[(buy_pos < p1) & (buy_pos >= p1 - k)]
        if len(prior):
            p0 = int(prior[-1])
            rows.append((t1, idx[p0], p1 - p0, float(rate.iloc[p1] / rate.iloc[p0] - 1) * 1e4, True))
        else:
            rows.append((t1, pd.NaT, np.nan, np.nan, False))
    return pd.DataFrame(rows, columns=["reversal_date", "buy_date", "days_since_buy", "regret_bps", "paired"])


def _scenario_events(
    signals: pd.DataFrame, decided: pd.DataFrame | None, pairing: str
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if pairing == "events":
        return signals[signals["scenario"] == WINDOW_CLOSING], signals[signals["scenario"] == BUY_NOW]
    if decided is None or decided.empty:
        return None
    sent = decided[decided["decision"] == "sent"]
    return sent[sent["push_scenario"] == WINDOW_CLOSING], sent[sent["push_scenario"] == BUY_NOW]


def _base_by_split(hit: pd.Series, splits: list[Split]) -> dict[int, float]:
    return {sp.id: float(hit.loc[sp.test_start : sp.test_end].dropna().mean()) for sp in splits}


def _sums(
    dates: pd.DatetimeIndex, hit: pd.Series, sod: pd.Series, base: dict[int, float]
) -> tuple[float, float, int]:
    """(сумма попаданий, сумма баз окон событий, число размеченных) — складывается по коридорам."""
    hv = hit.reindex(dates)
    ok = hv.notna().to_numpy()
    d = dates[ok]
    return float(hv[ok].sum()), float(pd.Series(d).map(sod).map(base).sum()), int(ok.sum())


def _row(corridor: str, pairing: str, pairs: pd.DataFrame, fwd: tuple, rest: tuple) -> dict:
    reg = pairs.loc[pairs["paired"], "regret_bps"].to_numpy(dtype=float)
    lo, hi = metrics.bootstrap_ci(reg) if len(reg) >= MIN_CI_EVENTS else (np.nan, np.nan)
    hs, bs, n = fwd
    rs, rbs, rn = rest
    return {
        "corridor": corridor,
        "pairing": pairing,
        "n_reversal": int(len(pairs)),
        "n_paired": int(len(reg)),
        "paired_share": float(len(reg) / len(pairs)) if len(pairs) else np.nan,
        "days_since_buy_median": float(pairs.loc[pairs["paired"], "days_since_buy"].median())
        if len(reg)
        else np.nan,
        "regret_median_bps": float(np.median(reg)) if len(reg) else np.nan,
        "regret_mean_bps": float(reg.mean()) if len(reg) else np.nan,
        "regret_ci_lo": lo,
        "regret_ci_hi": hi,
        "share_regret_positive": float((reg > 0).mean()) if len(reg) else np.nan,
        "share_regret_gt_tol": float((reg > REGRET_THRESHOLD_BPS).mean()) if len(reg) else np.nan,
        "hit_fwd": hs / n if n else np.nan,
        "base_fwd": bs / n if n else np.nan,
        "lift_fwd": hs / bs if bs > 0 else np.nan,
        "hit_rest_of_month": rs / rn if rn else np.nan,
        "base_rest_of_month": rbs / rn if rn else np.nan,
        "lift_rest_of_month": rs / rbs if rbs > 0 else np.nan,
    }


def reversal_regret_table(
    signals: pd.DataFrame,
    decided: pd.DataFrame | None,
    panel: pd.DataFrame,
    splits: list[Split],
    k: int = REGRET_LOOKBACK,
    h: int = CALIBRATION_H,
) -> pd.DataFrame:
    """По коридорам и строкой `all`, для двух пар источников: события индикаторов (`events`) и
    отправленные пуши потока (`stream`). `regret_*` — сожаление по парам «последний BUY_NOW →
    разворот», `hit_fwd` — прогнозное прочтение через h дней (f_h ≥ a_T, как в матрице),
    `hit_rest_of_month` — до конца месяца; базы — по всем дням публикации окна события."""
    from fxmoment.analysis import _split_of_day

    rows: list[dict] = []
    for pairing in PAIRINGS:
        pick = _scenario_events(signals, decided, pairing)
        if pick is None or pick[0].empty:
            continue
        wc, buy = pick
        all_pairs: list[pd.DataFrame] = []
        fwd_tot, rest_tot = np.zeros(3), np.zeros(3)
        for corridor in sorted(wc["corridor"].unique()):
            if corridor not in panel.columns:
                continue
            rate = panel[corridor].dropna()
            sod = _split_of_day(rate, splits)
            hit_h = labels.hit_window_closing(rate, h)
            hit_rest = rest_of_month_hit(rate)
            t1 = pd.DatetimeIndex(pd.to_datetime(wc.loc[wc["corridor"] == corridor, "date"])).unique()
            t1 = t1[t1.isin(rate.index)].sort_values()
            t0 = pd.DatetimeIndex(pd.to_datetime(buy.loc[buy["corridor"] == corridor, "date"]))
            pairs = pair_reversals(rate, t1, t0, k)
            fwd = _sums(t1, hit_h, sod, _base_by_split(hit_h, splits))
            rest = _sums(t1, hit_rest, sod, _base_by_split(hit_rest, splits))
            rows.append(_row(str(corridor), pairing, pairs, fwd, rest))
            all_pairs.append(pairs)
            fwd_tot += np.asarray(fwd)
            rest_tot += np.asarray(rest)
        if all_pairs:
            pooled = pd.concat(all_pairs, ignore_index=True)
            rows.append(_row("all", pairing, pooled, tuple(fwd_tot), tuple(rest_tot)))
    return pd.DataFrame(rows, columns=COLUMNS)
