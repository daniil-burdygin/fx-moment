"""Базовый класс индикатора и общие каузальные примитивы."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from fxmoment.config import BUY_NOW


class Indicator(ABC):
    """Индикатор — чистая функция: строка выхода на дату T зависит только от входа с индексом ≤ T.

    Выход compute(): DataFrame с индексом ряда и колонками
    `signal` (bool — событие для политики потока), `strength` (0…1) и фактами для текста пуша.
    """

    name: str = ""
    speed: str = "medium"  # fast / medium / slow (ADR-0005)
    scenario: str = BUY_NOW
    direction: str = "down"  # направление курса, при котором срабатывает: down / up
    trainable: bool = False

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = params

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        """Сетка параметров для калибровки walk-forward."""
        return [{}]

    def fact_fields(self) -> tuple[str, ...]:
        return ()

    def warmup(self) -> int:
        """Сколько первых дней ряда индикатор не определён (окна, серии). Калибровка меряет частоту
        и базу только после разогрева, иначе длинные окна выглядят реже коротких (аудит 03.09)."""
        return 0

    @abstractmethod
    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame: ...

    def label(self) -> str:
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({inner})"


def rearm_events(cond: pd.Series, rearm: int) -> pd.Series:
    """Событие — первый день, когда условие истинно, и затем не чаще, чем раз в `rearm` дней
    (между событиями не меньше `rearm` дней публикации; rearm = 0 — каждый день условия).

    Последовательный проход слева направо: состояние зависит только от прошлого."""
    c = cond.fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(c), dtype=bool)
    last = -(10**9)
    for i, v in enumerate(c):
        if v and i - last >= max(rearm, 1):
            out[i] = True
            last = i
    return pd.Series(out, index=cond.index)


def rolling_pct_rank(rate: pd.Series, window: int) -> pd.Series:
    """Доля значений скользящего окна (включая текущее), не превышающих текущее. Низко = дёшево."""
    return rate.rolling(window).apply(lambda w: float(np.mean(w <= w[-1])), raw=True)


def rolling_days_since_min(rate: pd.Series, window: int) -> pd.Series:
    """Сколько дней публикации назад был минимум скользящего окна (0 — сегодня)."""
    return rate.rolling(window).apply(lambda w: float(len(w) - 1 - int(np.argmin(w))), raw=True)


def down_streak(rate: pd.Series) -> pd.Series:
    """Длина серии снижений, заканчивающейся сегодня."""
    down = rate.diff() < 0
    return down.astype(int).groupby((~down).cumsum()).cumsum()


def up_streak(rate: pd.Series) -> pd.Series:
    up = rate.diff() > 0
    return up.astype(int).groupby((~up).cumsum()).cumsum()


def empty_output(index: pd.Index, facts: tuple[str, ...] = ()) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["signal"] = False
    out["strength"] = 0.0
    for f in facts:
        out[f] = np.nan
    return out
