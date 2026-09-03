"""Уровень с вычтенным дрейфом локальной ноги (💬 03.09 вечер, пункт 2).

Курс RUB→X = USD/RUB × (X/USD)⁻¹: общий фактор — доллар к рублю, локальная нога `local = rate / usd`
— стоимость местной валюты в долларах. У UZS локальная нога устойчиво дешевеет, и «один из низких за
полгода» у такого ряда — почти любой день: новый минимум ставит не выбор момента, а сам дрейф.
Индикатор считает процентиль уровня среди значений окна, приведённых к сегодняшнему дрейфу:
rate_s · exp(d_t · (t − s)), где d_t — средний дневной лог-шаг локальной ноги за скользящие
`drift_window` дней публикации (только прошлое). Сам дрейф — отдельный факт для текста («курс сума к
доллару за год снизился на …»). Стоит рядом с `level`, не вместо: вариант `backtest --with-level-drift`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from fxmoment.config import BUY_NOW
from fxmoment.indicators.base import Indicator, rearm_events

STEPS_PER_YEAR = 250  # дней публикации в году — для факта «за год»


def local_drift(rate: pd.Series, context: pd.DataFrame | None, drift_window: int) -> pd.Series:
    """Средний дневной лог-шаг локальной ноги `rate / usd` за скользящие `drift_window` дней
    публикации; без USD в контексте — самого курса (факт `local_leg` = 0). Определён с позиции
    `drift_window`: нужны разность и полное окно."""
    leg = rate / context["USD"].reindex(rate.index) if _has_usd(context) else rate
    return np.log(leg).diff().rolling(drift_window).mean()


def drift_adjusted_rank(rate: pd.Series, drift: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Процентиль сегодняшнего курса среди значений окна, приведённых к сегодняшнему дрейфу, и
    позиция минимума приведённого окна (дней публикации назад). Значение дня s внутри окна —
    rate_s · exp(d_t · (t − s)): при отрицательном дрейфе прошлое «спускается» к сегодняшнему
    уровню, и минимум, поставленный одним дрейфом, перестаёт быть низким. Нулевой дрейф даёт
    ровно `rolling_pct_rank` и `rolling_days_since_min`."""
    values = rate.to_numpy(dtype=float)
    n = len(values)
    rank = np.full(n, np.nan)
    dsm = np.full(n, np.nan)
    if n >= window:
        win = sliding_window_view(values, window)  # строка t: дни t − window + 1 … t
        k = np.arange(window - 1, -1, -1, dtype=float)  # t − s по столбцам
        d = drift.reindex(rate.index).to_numpy(dtype=float)[window - 1 :]
        adjusted = win * np.exp(d[:, None] * k[None, :])
        today = adjusted[:, -1][:, None]
        ok = ~np.isnan(d)
        rank[window - 1 :] = np.where(ok, np.mean(adjusted <= today, axis=1), np.nan)
        dsm[window - 1 :] = np.where(ok, window - 1 - np.argmin(adjusted, axis=1), np.nan)
    return pd.Series(rank, index=rate.index), pd.Series(dsm, index=rate.index)


def _has_usd(context: pd.DataFrame | None) -> bool:
    return context is not None and "USD" in context.columns


class LevelDrift(Indicator):
    name = "level_drift"
    speed = "medium"
    STEP_PARAMS = ("window", "drift_window", "stall_days", "rearm")
    scenario = BUY_NOW
    direction = "down"

    def __init__(
        self,
        window: int = 120,
        pct: float = 0.10,
        drift_window: int = 250,
        stall_days: int = 0,
        rearm: int = 5,
    ) -> None:
        super().__init__(
            window=window, pct=pct, drift_window=drift_window, stall_days=stall_days, rearm=rearm
        )
        self.window = window
        self.pct = pct
        self.drift_window = drift_window
        self.stall_days = stall_days
        self.rearm = rearm

    @classmethod
    def grid(cls) -> list[dict]:
        # та же сетка, что у `level`; окно дрейфа одно (год), чтобы сравнение с `level` было
        # сравнением приведения, а не ещё одной степени свободы
        return [
            {"window": w, "pct": p, "drift_window": 250, "stall_days": s, "rearm": r}
            for w in (60, 120, 250)
            for p in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
            for s in (0, 3)
            for r in (3, 5)
        ]

    def fact_fields(self) -> tuple[str, ...]:
        return ("pct_rank", "window", "days_since_min", "drift_pct_year", "local_leg")

    def warmup(self, index: pd.DatetimeIndex | None = None) -> int:
        # дрейф определён с позиции drift_window, окно — с window − 1
        return max(self.window - 1, self.drift_window)

    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        drift = local_drift(rate, context, self.drift_window)
        rank, dsm = drift_adjusted_rank(rate, drift, self.window)
        cond = rank <= self.pct
        if self.stall_days > 0:
            cond &= dsm >= self.stall_days
        out = pd.DataFrame(index=rate.index)
        out["signal"] = rearm_events(cond, self.rearm)
        out["strength"] = np.clip(1 - rank / self.pct, 0, 1).astype(float)
        out["pct_rank"] = rank
        out["window"] = float(self.window)
        out["days_since_min"] = dsm
        out["drift_pct_year"] = (np.exp(drift * STEPS_PER_YEAR) - 1) * 100
        out["local_leg"] = 1.0 if _has_usd(context) else 0.0
        return out
