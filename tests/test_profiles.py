"""Профиль ряда и масштабирование сеток (ADR-0010).

Главное, что проверяется: дневной путь после параметризации ведёт себя ровно как раньше. Профиль
задумывался как добавление второй оси, а не как правка первой, и молчаливый сдвиг дневных
результатов был бы худшим исходом этой работы — отчёты сдачи посчитаны на ней.
"""

import pandas as pd
import pytest

from fxmoment.backtest.engine import fit_indicator
from fxmoment.config import (
    CALIBRATION_FREQ_RANGE,
    CALIBRATION_H,
    HORIZONS,
    MIN_TEST_DAYS,
    PURGE_DAYS,
)
from fxmoment.indicators import Dip, Level, Momentum, Reversal, Seasonality
from fxmoment.indicators.base import _scale_step
from fxmoment.indicators.features import CACHED_WINDOWS, enrich_context
from fxmoment.profiles import DAILY, INTRADAY, first_test_for


def test_daily_profile_repeats_config_constants():
    assert DAILY.step_scale == 1
    assert DAILY.horizons == HORIZONS
    assert DAILY.calibration_h == CALIBRATION_H
    assert DAILY.purge == PURGE_DAYS
    assert DAILY.min_test_steps == MIN_TEST_DAYS


def test_scale_one_leaves_grid_identical():
    for cls in (Momentum, Level, Reversal, Dip, Seasonality):
        assert cls.scaled_grid(1) == cls.grid()
        assert cls.scaled_defaults(1) == cls().params


def test_step_params_scale_and_others_do_not():
    point = next(p for p in Level.scaled_grid(9) if p["pct"] == 0.05)
    assert point["window"] in (540, 1080, 2250)  # 60 / 120 / 250 дней в барах
    assert point["pct"] == 0.05  # процент — не шаг ряда
    # ноль остаётся нулём: stall_days = 0 значит «условие выключено», а не «один бар»
    assert 0 in {p["stall_days"] for p in Level.scaled_grid(9)}
    assert 27 in {p["stall_days"] for p in Level.scaled_grid(9)}
    # порог в базисных пунктах не масштабируется
    assert {p["rise_bps"] for p in Reversal.scaled_grid(9)} == {p["rise_bps"] for p in Reversal.grid()}
    # `n` моментума — определение индикатора, а не окно: серия из 36 падающих баров не бывает
    assert {p["n"] for p in Momentum.scaled_grid(9)} == {p["n"] for p in Momentum.grid()}
    # календарное число месяца у сезонности тоже остаётся собой
    assert {p["from_day"] for p in Seasonality.scaled_grid(9)} == {20, 24}


def test_scaled_grid_drops_duplicates_after_rounding():
    grid = Level.scaled_grid(9)
    assert len(grid) == len({tuple(sorted(p.items())) for p in grid})


def test_context_cache_matches_scaled_windows():
    """Кэш `_rank_w` обязан совпасть с окнами масштабированной сетки: иначе индикатор молча
    пересчитает их сам и прогон замедлится в разы."""
    idx = pd.date_range("2024-01-01", periods=300, freq="h")
    rate = pd.Series(range(1, 301), index=idx, dtype=float)
    ctx = enrich_context(rate, None, scale=9)
    for w0 in CACHED_WINDOWS:
        assert f"_rank_{_scale_step(w0, 9)}" in ctx.columns
    windows = {p["window"] for p in Level.scaled_grid(9)}
    assert windows <= {_scale_step(w, 9) for w in CACHED_WINDOWS}


def test_intraday_profile_is_a_month_in_bars():
    assert INTRADAY.calibration_h == 20 * INTRADAY.step_scale
    assert INTRADAY.purge == INTRADAY.calibration_h  # зазор равен максимальному горизонту
    assert INTRADAY.horizons[0] == 1  # час — самый короткий горизонт, ради него всё и затевалось
    assert Seasonality not in INTRADAY.indicators
    assert INTRADAY.policy.cooldown_days == 3 * INTRADAY.step_scale


def test_frequency_band_is_not_in_profile():
    """Полоса частоты про канал, а не про данные, и профилем не переопределяется."""
    assert not hasattr(INTRADAY, "calibration_freq_range")
    assert CALIBRATION_FREQ_RANGE == (0.3, 2.5)


def test_first_test_waits_for_enough_training(panel):
    idx = panel.index
    start = first_test_for(idx, DAILY)
    assert start >= pd.Timestamp(DAILY.first_test)
    short = idx[:100]
    with pytest.raises(ValueError, match="короче обучения"):
        first_test_for(short, DAILY)


def test_fit_indicator_defaults_do_not_scale(panel):
    """Вызов без профиля обязан дать ровно то же, что и раньше: дневная сетка, дневной горизонт."""
    rate = panel["TJS"].dropna()
    ctx = enrich_context(rate, panel[["USD"]])
    a, params_a, _ = fit_indicator(Level, rate, ctx, eval_start=rate.index[250])
    b, params_b, _ = fit_indicator(
        Level, rate, ctx, eval_start=rate.index[250], calibration_h=CALIBRATION_H, grid_scale=1
    )
    assert params_a == params_b
    assert a.params == b.params
