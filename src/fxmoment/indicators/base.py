"""Базовый класс индикатора и общие каузальные примитивы."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from fxmoment.config import BUY_NOW, CALIBRATION_FREQ_RANGE, MIN_CALIBRATION_EVENTS


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

    # Ключи сетки, измеряемые в ШАГАХ РЯДА (дни публикации на дневной оси, бары на внутридневной).
    # Только они умножаются на масштаб профиля; проценты, пороги в бп и календарные числа — нет.
    STEP_PARAMS: tuple[str, ...] = ()

    @classmethod
    def grid(cls) -> list[dict[str, Any]]:
        """Сетка параметров для калибровки walk-forward, в шагах дневного ряда."""
        return [{}]

    @classmethod
    def scaled_grid(cls, scale: int = 1) -> list[dict[str, Any]]:
        """Та же сетка в шагах ряда другой частоты (ADR-0010): окно «120 дней» на часовом ряду —
        это 120 × баров в дне, а процент и порог в базисных пунктах остаются собой.

        Ноль не превращается в единицу: `stall_days = 0` значит «условие выключено», а не «один шаг».
        Точки, совпавшие после округления, схлопываются — иначе калибровка считала бы одно и то же
        по нескольку раз."""
        if scale == 1:
            return cls.grid()
        seen: list[dict[str, Any]] = []
        for point in cls.grid():
            scaled = {k: (_scale_step(v, scale) if k in cls.STEP_PARAMS else v) for k, v in point.items()}
            if scaled not in seen:
                seen.append(scaled)
        return seen

    @classmethod
    def scaled_defaults(cls, scale: int = 1) -> dict[str, Any]:
        """Параметры по умолчанию в шагах ряда другой частоты (для контрольного прогона без сетки)."""
        base = cls().params
        return {k: (_scale_step(v, scale) if k in cls.STEP_PARAMS else v) for k, v in base.items()}

    def fact_fields(self) -> tuple[str, ...]:
        return ()

    def warmup(self, index: pd.DatetimeIndex | None = None) -> int:
        """Позиция первого дня, где выход индикатора определён (до неё сигнала нет). Калибровка меряет
        частоту и базу только после разогрева, иначе длинные окна выглядят реже коротких (аудит 03.09).
        `index` — календарь ряда: нужен индикаторам, чей разогрев зависит от дат (сезонность)."""
        return 0

    @classmethod
    def calibration_bounds(cls) -> tuple[float, float, int]:
        """(частота от, частота до, минимум событий) для допустимой точки сетки при калибровке.
        Медленный индикатор переопределяет: месячный сигнал физически не даёт 0,3 в неделю."""
        return (CALIBRATION_FREQ_RANGE[0], CALIBRATION_FREQ_RANGE[1], MIN_CALIBRATION_EVENTS)

    @abstractmethod
    def compute(self, rate: pd.Series, context: pd.DataFrame | None = None) -> pd.DataFrame: ...

    def label(self) -> str:
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({inner})"


def _scale_step(value: Any, scale: int) -> Any:
    """Число шагов ряда в другом масштабе. Ноль остаётся нулём — это выключенное условие."""
    if not isinstance(value, int) or isinstance(value, bool) or value == 0:
        return value
    return max(1, int(round(value * scale)))


def rearm_events(cond: pd.Series, rearm: int) -> pd.Series:
    """Событие — первый день, когда условие истинно, и затем не чаще, чем раз в `rearm` дней
    (между событиями не меньше `rearm` дней публикации; rearm ≤ 1 — каждый день условия:
    промежуток между соседними днями и так равен одному дню, поэтому точка rearm = 1 в сетках
    не используется).

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
