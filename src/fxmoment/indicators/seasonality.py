"""Сезонность: в конце месяца m−1 — если в прошлые годы курс в месяце m обычно был выше,
чем в m−1 (перевести заранее). Считается только по завершённым прошлым годам."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment.config import BUY_NOW
from fxmoment.indicators.base import Indicator, rearm_events


class Seasonality(Indicator):
    name = "seasonality"
    speed = "slow"
    scenario = BUY_NOW
    direction = "down"

    def __init__(
        self, min_share: float = 0.7, min_years: int = 4, from_day: int = 24, rearm: int = 20
    ) -> None:
        super().__init__(min_share=min_share, min_years=min_years, from_day=from_day, rearm=rearm)
        self.min_share = min_share
        self.min_years = min_years
        self.from_day = from_day
        self.rearm = rearm

    @classmethod
    def grid(cls) -> list[dict]:
        return [{"min_share": s, "min_years": 4, "from_day": 24, "rearm": 20} for s in (0.6, 0.7, 0.8)]

    def fact_fields(self) -> tuple[str, ...]:
        return ("k_years", "n_years", "target_month")

    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        monthly = rate.groupby([rate.index.year, rate.index.month]).mean()
        monthly.index = pd.MultiIndex.from_tuples(monthly.index, names=["year", "month"])
        rose: dict[tuple[int, int], bool] = {}
        for (y, m), val in monthly.items():
            prev = (y, m - 1) if m > 1 else (y - 1, 12)
            if prev in monthly.index:
                rose[(y, m)] = bool(val > monthly.loc[prev])
        idx = rate.index
        k_arr = np.full(len(idx), np.nan)
        n_arr = np.full(len(idx), np.nan)
        tm_arr = np.full(len(idx), np.nan)
        cond = np.zeros(len(idx), dtype=bool)
        cache: dict[tuple[int, int], tuple[int, int]] = {}
        for i, d in enumerate(idx):
            if d.day < self.from_day:
                continue
            target_year, target_month = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
            key = (target_year, target_month)
            if key not in cache:
                past = [rose[(y, m)] for (y, m) in rose if m == target_month and y < target_year]
                cache[key] = (int(sum(past)), len(past))
            k, n = cache[key]
            k_arr[i], n_arr[i], tm_arr[i] = k, n, target_month
            cond[i] = n >= self.min_years and k / n >= self.min_share
        out = pd.DataFrame(index=idx)
        out["signal"] = rearm_events(pd.Series(cond, index=idx), self.rearm)
        share = np.where(n_arr > 0, k_arr / np.where(n_arr > 0, n_arr, 1), 0.0)
        out["strength"] = np.clip(share, 0, 1).astype(float)
        out["k_years"] = k_arr
        out["n_years"] = n_arr
        out["target_month"] = tm_arr
        return out
