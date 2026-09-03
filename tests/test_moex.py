"""Ось и сверка внутридневного ряда Мосбиржи (ADR-0010). Сети здесь нет: проверяются чистые
функции разбора и агрегации, вход — таблица той же формы, что отдаёт коннектор."""

import numpy as np
import pandas as pd
import pytest

from fxmoment.data.compare import compare_by_hour, compare_levels, daily_close, liquidity
from fxmoment.data.moex import MOEX_FACEVALUE, MOEX_SECURITIES, to_bar_panel


def make_bars() -> pd.DataFrame:
    """Два дня торгов: CNY торгуется 10–12 часов, UZS — только в 10 (разные торговые часы)."""
    rows = []
    for day in ("2024-03-04", "2024-03-05"):
        for hour, close in ((10, 12.0), (11, 12.1), (12, 12.2)):
            begin = pd.Timestamp(f"{day} {hour:02d}:00:00")
            rows.append(
                {
                    "currency": "CNY",
                    "begin": begin,
                    "known_at": begin + pd.Timedelta(hours=1),
                    "end": begin + pd.Timedelta(minutes=53),  # ISS отдаёт время последней сделки
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "unit_rate": close,
                }
            )
        begin = pd.Timestamp(f"{day} 10:00:00")
        rows.append(
            {
                "currency": "UZS",
                "begin": begin,
                "known_at": begin + pd.Timedelta(hours=1),
                "end": begin + pd.Timedelta(minutes=5),
                "open": 70.0,
                "high": 70.0,
                "low": 70.0,
                "close": 70.0,
                "unit_rate": 0.007,
            }
        )
    return pd.DataFrame(rows)


def test_securities_and_nominals_agree():
    assert set(MOEX_SECURITIES) == set(MOEX_FACEVALUE)
    assert all(v > 0 for v in MOEX_FACEVALUE.values())


def test_bar_panel_axis_is_known_at_and_keeps_gaps():
    panel = to_bar_panel(make_bars())
    # ось — момент, когда закрытие стало известно: бар 10:00 сидит на 11:00
    assert panel.index[0] == pd.Timestamp("2024-03-04 11:00:00")
    assert panel.index.name == "known_at"
    # пропуски не заполняются: у UZS нет баров 11 и 12 часов, там NaN, а не перенос последнего
    assert pd.isna(panel.loc["2024-03-04 12:00:00", "UZS"])
    assert panel["UZS"].dropna().shape[0] == 2
    assert panel["CNY"].dropna().shape[0] == 6


def test_daily_close_is_causal_by_hour():
    panel = to_bar_panel(make_bars())
    bars = panel["CNY"]
    # до 11 часов известен только бар 10:00
    early = daily_close(bars, until_hour=11)
    assert early.loc["2024-03-04"] == 12.1
    late = daily_close(bars, until_hour=18)
    assert late.loc["2024-03-04"] == 12.2
    # срез ряда по времени не меняет уже посчитанные дни
    cut = bars.loc[: pd.Timestamp("2024-03-04 23:00:00")]
    assert daily_close(cut, until_hour=11).loc["2024-03-04"] == early.loc["2024-03-04"]


def test_liquidity_counts_bars_per_day():
    liq = liquidity(make_bars()).set_index("currency")
    assert liq.loc["CNY", "bars"] == 6
    assert liq.loc["CNY", "trading_days"] == 2
    assert liq.loc["CNY", "bars_per_day_median"] == 3.0
    assert liq.loc["UZS", "bars_per_day_median"] == 1.0
    assert liq.loc["UZS", "share_days_single_bar"] == 1.0


def test_compare_reports_thin_pairs_instead_of_inventing_numbers():
    bar_panel = to_bar_panel(make_bars())
    cbr = pd.DataFrame(
        {"CNY": [12.05, 12.15], "UZS": [0.0071, 0.0072]},
        index=pd.to_datetime(["2024-03-04", "2024-03-05"]),
    )
    levels = compare_levels(cbr, bar_panel).set_index("currency")
    # два дня — меньше порога в 30: метрики не выдумываются, а помечаются
    assert levels.loc["CNY", "note"] == "мало общих дней"
    assert compare_by_hour(cbr, bar_panel).empty


def test_evening_bar_stays_in_its_own_trading_day():
    """Ось `known_at` = начало бара + час, поэтому бар 23:00 садится на 00:00 следующего дня.
    День берётся у НАЧАЛА бара: иначе вечерняя сессия отрезалась бы от своего дня (аудит 03.09)."""
    idx = pd.to_datetime(["2024-03-04 11:00:00", "2024-03-04 20:00:00", "2024-03-05 00:00:00"])
    bars = pd.Series([12.0, 12.5, 12.9], index=idx)  # последний бар начался в 23:00 четвёртого
    close = daily_close(bars)
    assert list(close.index.date.astype(str)) == ["2024-03-04"]
    assert close.iloc[0] == 12.9  # вечерний бар остался в своём дне и стал его закрытием


def _long_bars(days: int = 80) -> pd.DataFrame:
    """Ряд подлиннее порога в 30 дней. Дневной шаг РАЗНЫЙ: у ряда с постоянным шагом дисперсия
    изменений нулевая и корреляция вырождается, то есть тест ничего бы не проверял."""
    rng = np.random.default_rng(11)
    steps = np.exp(np.cumsum(rng.normal(0.0004, 0.004, days)))
    rows = []
    day = pd.Timestamp("2024-01-01")
    for i in range(days):
        base = 12.0 * steps[i]
        for hour in range(10, 19):
            begin = day + pd.Timedelta(days=i, hours=hour)
            close = base * (1 + 0.0002 * (hour - 10))
            rows.append(
                {
                    "currency": "CNY",
                    "begin": begin,
                    "known_at": begin + pd.Timedelta(hours=1),
                    "end": begin + pd.Timedelta(minutes=53),
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "unit_rate": close,
                }
            )
    return pd.DataFrame(rows)


def test_compare_levels_measures_the_gap_between_sources():
    """Числовое ядро сверки: расхождение и связь изменений считаются, а не только помечаются."""
    bars = _long_bars()
    panel = to_bar_panel(bars)
    close = daily_close(panel["CNY"])
    cbr = pd.DataFrame({"CNY": close.to_numpy() * 1.001}, index=close.index)  # фиксинг ровно на 10 бп ниже
    levels = compare_levels(cbr, panel).set_index("currency")
    assert levels.loc["CNY", "note"] == ""
    assert levels.loc["CNY", "n_days"] == len(close)
    assert levels.loc["CNY", "median_diff_bps"] == pytest.approx(-9.99, abs=0.1)
    assert levels.loc["CNY", "corr_changes"] == pytest.approx(1.0, abs=1e-6)


def test_compare_by_hour_uses_one_step_for_both_series():
    """Изменения биржи и фиксинга берутся за ОДИН шаг — соседние дни публикации. Раньше изменение
    фиксинга считалось на плотном календаре, а биржи — на прорежённом соединении (аудит 03.09)."""
    bars = _long_bars()
    panel = to_bar_panel(bars)
    close = daily_close(panel["CNY"])
    cbr_full = pd.Series(close.to_numpy() * 1.001, index=close.index)
    holes = cbr_full.drop(cbr_full.index[[10, 11, 30]])  # у фиксинга выходные внутри периода
    by_hour = compare_by_hour(pd.DataFrame({"CNY": holes}), panel)
    assert not by_hour.empty
    # число сопоставленных шагов меньше числа общих дней ровно на разрывы
    row = by_hour[by_hour["hour"] == 18].iloc[0]
    assert row["n_paired_changes"] < row["n_days"]
    assert row["r2_change"] == pytest.approx(1.0, abs=1e-6)
    # флаг окна расчёта, а не публикации; и прямые торги — только у пары, торгуемой на бирже
    assert bool(by_hour[by_hour["hour"] == 14].iloc[0]["inside_cbr_window"])
    assert not bool(by_hour[by_hour["hour"] == 15].iloc[0]["inside_cbr_window"])
    assert bool(by_hour.iloc[0]["cbr_uses_exchange"])


def test_liquidity_on_empty_snapshot_returns_empty_table():
    empty = pd.DataFrame(columns=["currency", "begin"])
    assert liquidity(empty).empty
    assert "bars_per_day_median" in liquidity(empty).columns
