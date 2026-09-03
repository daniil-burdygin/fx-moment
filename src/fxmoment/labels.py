"""Разметка попаданий и выгоды (ADR-0003). Все функции смотрят в будущее — только для оценки
и обучения с зазором, никогда внутри индикатора.

Курс действия a_T = rate[T] (последний опубликованный на T). f_i = rate.shift(-i) — курс действия
i-го следующего дня публикации. Ниже курс — лучше для отправителя.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def future_values(rate: pd.Series, h: int) -> pd.DataFrame:
    return pd.concat({i: rate.shift(-i) for i in range(1, h + 1)}, axis=1)


def past_values(rate: pd.Series, h: int) -> pd.DataFrame:
    return pd.concat({i: rate.shift(i) for i in range(1, h + 1)}, axis=1)


def future_min(rate: pd.Series, h: int) -> pd.Series:
    return future_values(rate, h).min(axis=1, skipna=False)


def future_mean(rate: pd.Series, h: int) -> pd.Series:
    return future_values(rate, h).mean(axis=1, skipna=False)


def future_at(rate: pd.Series, h: int) -> pd.Series:
    return rate.shift(-h)


def _as_float_flag(cond: pd.Series, valid: pd.Series) -> pd.Series:
    out = cond.astype(float)
    out[~valid] = np.nan
    return out


def hit_buy_now(rate: pd.Series, h: int, tol_bps: float = 0.0, mode: str = "min") -> pd.Series:
    """1.0 — «сейчас выгодно» подтвердилось. Три прочтения «курс остался не хуже h дней» (ADR-0003):
    `min` — ни один из h дней не был ниже a_T·(1 − tol) (строгое, «сожаление оракула»);
    `mean` — средний курс h дней не ниже a_T·(1 − tol) (клиент против своего типичного дня);
    `end` — курс через h дней не ниже a_T·(1 − tol)."""
    if mode == "mean":
        ref = future_mean(rate, h)
    elif mode == "end":
        ref = future_at(rate, h)
    else:
        ref = future_min(rate, h)
    return _as_float_flag(ref >= rate * (1 - tol_bps / 1e4), ref.notna())


def hit_window_closing(rate: pd.Series, h: int, tol_bps: float = 0.0) -> pd.Series:
    """1.0 — «окно закрывается» подтвердилось: через h дней курс не ниже a_T·(1 + tol)."""
    fa = future_at(rate, h)
    return _as_float_flag(fa >= rate * (1 + tol_bps / 1e4), fa.notna())


def hit_for_scenario(
    rate: pd.Series, scenario: str, h: int, tol_bps: float = 0.0, mode: str = "min"
) -> pd.Series:
    if scenario == "WINDOW_CLOSING":
        return hit_window_closing(rate, h, tol_bps)
    return hit_buy_now(rate, h, tol_bps, mode)


def benefit_fwd_bps(rate: pd.Series, h: int) -> pd.Series:
    """Выгода вперёд: среднее будущих h курсов к a_T, в бп. > 0 — клиент выиграл, действуя сейчас."""
    return (future_mean(rate, h) / rate - 1) * 1e4


def benefit_sym_bps(rate: pd.Series, h: int) -> pd.Series:
    """Выгода по определению кейса: среднее прошлых и будущих h курсов к a_T, в бп."""
    both = pd.concat([past_values(rate, h), future_values(rate, h)], axis=1)
    return (both.mean(axis=1, skipna=False) / rate - 1) * 1e4


def wait_benefit_bps(rate: pd.Series, k: int) -> pd.Series:
    """Действовать сейчас против «подождать k дней»: f_k к a_T, в бп. > 0 — сейчас было лучше."""
    return (future_at(rate, k) / rate - 1) * 1e4


def local_min_label(rate: pd.Series, h: int, tol_bps: float = 10.0) -> pd.Series:
    """Разметка для обучаемого индикатора: день — локальный минимум в окне ±h с допуском."""
    both = pd.concat([past_values(rate, h), future_values(rate, h)], axis=1)
    wmin = both.min(axis=1, skipna=False)
    return _as_float_flag(rate <= wmin * (1 + tol_bps / 1e4), wmin.notna())
