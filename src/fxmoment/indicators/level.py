"""Уровень: курс в нижних процентилях скользящего окна; вариант с остановкой падения."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment.config import BUY_NOW
from fxmoment.indicators.base import (
    Indicator,
    rearm_events,
    rolling_days_since_min,
    rolling_pct_rank,
)


class Level(Indicator):
    name = "level"
    speed = "medium"
    scenario = BUY_NOW
    direction = "down"

    def __init__(self, window: int = 120, pct: float = 0.10, stall_days: int = 0, rearm: int = 5) -> None:
        super().__init__(window=window, pct=pct, stall_days=stall_days, rearm=rearm)
        self.window = window
        self.pct = pct
        self.stall_days = stall_days
        self.rearm = rearm

    @classmethod
    def grid(cls) -> list[dict]:
        return [
            {"window": w, "pct": p, "stall_days": s, "rearm": r}
            for w in (60, 120, 250)
            for p in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
            for s in (0, 3)
            for r in (3, 5)
        ]

    def fact_fields(self) -> tuple[str, ...]:
        return ("pct_rank", "window", "days_since_min")

    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        rank = _cached(context, f"_rank_{self.window}")
        if rank is None:
            rank = rolling_pct_rank(rate, self.window)
        dsm = _cached(context, f"_dsm_{self.window}")
        if dsm is None:
            dsm = rolling_days_since_min(rate, self.window)
        rank = rank.reindex(rate.index)
        dsm = dsm.reindex(rate.index)
        cond = rank <= self.pct
        if self.stall_days > 0:
            cond &= dsm >= self.stall_days
        out = pd.DataFrame(index=rate.index)
        out["signal"] = rearm_events(cond, self.rearm)
        out["strength"] = np.clip(1 - rank / self.pct, 0, 1).astype(float)
        out["pct_rank"] = rank
        out["window"] = float(self.window)
        out["days_since_min"] = dsm
        return out


def _cached(context: pd.DataFrame | None, key: str) -> pd.Series | None:
    if context is not None and key in context.columns:
        return context[key]
    return None
