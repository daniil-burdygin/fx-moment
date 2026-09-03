"""Индикаторы. Каждый — каузальная функция от ряда до даты T (CLAUDE.md, инвариант 1)."""

from fxmoment.indicators.base import Indicator, rearm_events
from fxmoment.indicators.dip import Dip
from fxmoment.indicators.level import Level
from fxmoment.indicators.ml import LearnedMinimum
from fxmoment.indicators.momentum import Momentum
from fxmoment.indicators.reversal import Reversal
from fxmoment.indicators.seasonality import Seasonality

BASE_INDICATORS: tuple[type[Indicator], ...] = (Momentum, Level, Reversal, Seasonality, Dip)
ALL_INDICATORS: tuple[type[Indicator], ...] = (*BASE_INDICATORS, LearnedMinimum)

__all__ = [
    "ALL_INDICATORS",
    "BASE_INDICATORS",
    "Dip",
    "Indicator",
    "LearnedMinimum",
    "Level",
    "Momentum",
    "Reversal",
    "Seasonality",
    "rearm_events",
]
