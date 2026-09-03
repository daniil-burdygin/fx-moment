"""Моментум: курс валюты получателя в рублях снижается n дней подряд. Быстрый индикатор."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment.config import BUY_NOW
from fxmoment.indicators.base import Indicator, down_streak, rearm_events


class Momentum(Indicator):
    name = "momentum"
    speed = "fast"
    scenario = BUY_NOW
    direction = "down"

    def __init__(self, n: int = 4, rearm: int = 3) -> None:
        super().__init__(n=n, rearm=rearm)
        self.n = n
        self.rearm = rearm

    @classmethod
    def grid(cls) -> list[dict]:
        return [{"n": n, "rearm": 3} for n in (3, 4, 5, 6)]

    def fact_fields(self) -> tuple[str, ...]:
        return ("streak", "drop_pct")

    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        streak = down_streak(rate)
        r = rate.to_numpy(dtype=float)
        s = streak.to_numpy(dtype=int)
        pos = np.arange(len(r)) - s
        drop = np.full(len(r), np.nan)
        ok = s > 0
        drop[ok] = (r[ok] / r[pos[ok]] - 1) * 100
        cond = streak >= self.n
        out = pd.DataFrame(index=rate.index)
        out["signal"] = rearm_events(cond, self.rearm)
        out["strength"] = np.clip(streak / (2 * self.n), 0, 1).astype(float)
        out["streak"] = streak.astype(float)
        out["drop_pct"] = drop
        return out
