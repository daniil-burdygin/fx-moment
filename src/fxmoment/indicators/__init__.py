"""Индикаторы. Каждый — каузальная функция от ряда до даты T (CLAUDE.md, инвариант 1)."""

from fxmoment.indicators.base import Indicator, rearm_events
from fxmoment.indicators.dip import Dip
from fxmoment.indicators.level import Level
from fxmoment.indicators.level_drift import LevelDrift
from fxmoment.indicators.ml import LearnedMinimum, LearnedMinimumPooled
from fxmoment.indicators.momentum import Momentum
from fxmoment.indicators.reversal import Reversal
from fxmoment.indicators.seasonality import Seasonality

BASE_INDICATORS: tuple[type[Indicator], ...] = (Momentum, Level, Reversal, Seasonality, Dip)
ALL_INDICATORS: tuple[type[Indicator], ...] = (*BASE_INDICATORS, LearnedMinimum)
# вариантные индикаторы (💬 03.09 вечер, пункты 2 и 5): в прогон по умолчанию не входят,
# включаются флагами `backtest --with-level-drift` и `backtest --ml pooled`
VARIANT_INDICATORS: tuple[type[Indicator], ...] = (LevelDrift, LearnedMinimumPooled)

__all__ = [
    "ALL_INDICATORS",
    "BASE_INDICATORS",
    "Dip",
    "Indicator",
    "LearnedMinimum",
    "LearnedMinimumPooled",
    "Level",
    "LevelDrift",
    "Momentum",
    "Reversal",
    "Seasonality",
    "VARIANT_INDICATORS",
    "rearm_events",
]
