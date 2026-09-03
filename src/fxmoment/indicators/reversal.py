"""Разворот вверх: курс был у минимума окна и оттолкнулся на rise_bps — «окно закрывается»."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment.config import WINDOW_CLOSING
from fxmoment.indicators.base import Indicator, rearm_events, rolling_days_since_min


class Reversal(Indicator):
    name = "reversal"
    speed = "slow"
    STEP_PARAMS = ("window", "max_days_since_min", "rearm")
    scenario = WINDOW_CLOSING
    direction = "up"

    def __init__(
        self, window: int = 120, rise_bps: float = 50, max_days_since_min: int = 10, rearm: int = 5
    ) -> None:
        super().__init__(window=window, rise_bps=rise_bps, max_days_since_min=max_days_since_min, rearm=rearm)
        self.window = window
        self.rise_bps = rise_bps
        self.max_days_since_min = max_days_since_min
        self.rearm = rearm

    @classmethod
    def grid(cls) -> list[dict]:
        return [
            {"window": w, "rise_bps": r, "max_days_since_min": d, "rearm": a}
            for w in (60, 120)
            for r in (20, 30, 50, 80)
            for d in (5, 10, 20)
            for a in (3, 5)
        ]

    def fact_fields(self) -> tuple[str, ...]:
        return ("rise_pct", "min_rate", "days_since_min", "window")

    def warmup(self, index: pd.DatetimeIndex | None = None) -> int:
        return self.window - 1  # rolling(window).min() впервые определён на позиции window − 1

    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        wmin = rate.rolling(self.window).min()
        dsm = None
        if context is not None and f"_dsm_{self.window}" in context.columns:
            dsm = context[f"_dsm_{self.window}"].reindex(rate.index)
        if dsm is None:
            dsm = rolling_days_since_min(rate, self.window)
        rise_bps = (rate / wmin - 1) * 1e4
        cond = (rise_bps >= self.rise_bps) & (dsm >= 1) & (dsm <= self.max_days_since_min)
        out = pd.DataFrame(index=rate.index)
        out["signal"] = rearm_events(cond, self.rearm)
        out["strength"] = np.clip(rise_bps / (2 * self.rise_bps), 0, 1).astype(float)
        out["rise_pct"] = rise_bps / 100
        out["min_rate"] = wmin
        out["days_since_min"] = dsm
        out["window"] = float(self.window)
        return out
