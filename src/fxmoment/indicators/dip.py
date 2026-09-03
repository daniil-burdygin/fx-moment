"""Провал относительно тренда: отклонение курса от простой скользящей средней за `span` дней
публикации в нижних процентилях своего окна. Собственный индикатор команды: на трендовом ряду
«нижние процентили уровня» ловят только редкие развороты, а провал относительно тренда распределён
по времени равномернее. Средняя простая, а не экспоненциальная, чтобы факт в тексте пуша
(«ниже среднего за 8 недель») был буквально проверяем (аудит 03.09)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment.config import BUY_NOW
from fxmoment.indicators.base import Indicator, rearm_events


class Dip(Indicator):
    name = "dip_vs_trend"
    speed = "medium"
    scenario = BUY_NOW
    direction = "down"

    def __init__(self, span: int = 40, window: int = 120, pct: float = 0.10, rearm: int = 5) -> None:
        super().__init__(span=span, window=window, pct=pct, rearm=rearm)
        self.span = span
        self.window = window
        self.pct = pct
        self.rearm = rearm

    @classmethod
    def grid(cls) -> list[dict]:
        return [
            {"span": s, "window": w, "pct": p, "rearm": r}
            for s in (20, 40, 80)
            for w in (120, 250)
            for p in (0.05, 0.10, 0.15, 0.20)
            for r in (3, 5)
        ]

    def fact_fields(self) -> tuple[str, ...]:
        return ("dev_pct", "pct_rank", "window", "span")

    def warmup(self) -> int:
        return self.span + self.window

    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame:
        trend = rate.rolling(self.span).mean()
        dev = rate / trend - 1
        rank = dev.rolling(self.window).apply(lambda w: float(np.mean(w <= w[-1])), raw=True)
        cond = (rank <= self.pct) & (dev < 0)
        out = pd.DataFrame(index=rate.index)
        out["signal"] = rearm_events(cond, self.rearm)
        out["strength"] = np.clip(1 - rank / self.pct, 0, 1).astype(float)
        out["dev_pct"] = dev * 100
        out["pct_rank"] = rank
        out["window"] = float(self.window)
        out["span"] = float(self.span)
        return out
