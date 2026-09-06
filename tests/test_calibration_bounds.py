"""Полоса допустимости калибровки, заданная снаружи (режим точности, Ш7): она заменяет
класс-специфичные `Indicator.calibration_bounds()` у всех правил разом, и разбор аргументов CLI
делает это один раз на границе."""

import pandas as pd
import pytest

from fxmoment.backtest.engine import fit_indicator
from fxmoment.cli import calibration_bounds_arg
from fxmoment.config import CALIBRATION_FREQ_RANGE, MIN_CALIBRATION_EVENTS, SLOW_MIN_CALIBRATION_EVENTS
from fxmoment.indicators import BASE_INDICATORS, Momentum, Seasonality
from fxmoment.indicators.features import enrich_context

BAND = (0.1, 0.3, 5)


def _freq_of(log: pd.DataFrame, params: dict) -> float:
    """Частота выбранной точки: строка журнала сетки с её параметрами (служебные `_*` — не параметры)."""
    grid = {k: v for k, v in params.items() if not k.startswith("_")}
    row = log[(log[list(grid)] == pd.Series(grid)).all(axis=1)]
    return float(row["freq_per_week"].iloc[0])


def _log(cls, panel, bounds):
    rate = panel["TJS"].dropna()
    ctx = enrich_context(rate, panel[["USD"]])
    _, params, log = fit_indicator(cls, rate, ctx, eval_start=rate.index[300], bounds=bounds)
    return params, pd.DataFrame(log)


@pytest.mark.parametrize("cls", BASE_INDICATORS, ids=[c.name for c in BASE_INDICATORS])
def test_bounds_replace_class_specific_bounds(cls, panel):
    """Допустимая точка при заданной полосе лежит в полосе и набирает её минимум событий —
    у каждого правила, включая сезонность с собственными границами (0,05–0,3 и 12 событий)."""
    lo, hi, min_n = BAND
    _, log = _log(cls, panel, BAND)
    feas = log[log["feasible"]]
    assert feas["freq_per_week"].between(lo, hi).all(), f"{cls.name}: точка вне полосы"
    assert (feas["n_scored"] >= min_n).all(), f"{cls.name}: точка ниже минимума событий"


def test_class_bounds_still_act_without_argument(panel):
    """Без полосы всё как было: у моментума — общая полоса, у сезонности — своя."""
    _, fast = _log(Momentum, panel, None)
    assert fast[fast["feasible"]]["freq_per_week"].between(*CALIBRATION_FREQ_RANGE).all()
    _, slow = _log(Seasonality, panel, None)
    sf = slow[slow["feasible"]]
    assert sf["freq_per_week"].between(*Seasonality.calibration_bounds()[:2]).all()
    assert (sf["n_scored"] >= SLOW_MIN_CALIBRATION_EVENTS).all()


def test_narrow_band_moves_the_chosen_point(panel):
    """Полоса — не косметика: у частого правила выбранная точка при 0,1–0,3 в неделю другая и реже."""
    narrow, nlog = _log(Momentum, panel, BAND)
    wide, wlog = _log(Momentum, panel, None)
    assert narrow != wide
    assert _freq_of(nlog, narrow) < _freq_of(wlog, wide)


def test_bounds_argument_parsed_once_at_the_boundary():
    assert calibration_bounds_arg(None, 0) is None
    assert calibration_bounds_arg([0.1, 0.3], 0) == (0.1, 0.3, SLOW_MIN_CALIBRATION_EVENTS)
    assert calibration_bounds_arg([0.3, 2.5], 0) == (0.3, 2.5, MIN_CALIBRATION_EVENTS)
    assert calibration_bounds_arg([0.1, 0.3], 7) == (0.1, 0.3, 7)
    for bad, events in (([0.3, 0.1], 0), ([0.0, 0.3], 0), ([0.2, 0.2], 0), ([0.1, 0.3], -1), (None, 12)):
        with pytest.raises(ValueError):
            calibration_bounds_arg(bad, events)
