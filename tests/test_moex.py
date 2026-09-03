"""Ось и сверка внутридневного ряда Мосбиржи (ADR-0010). Сети здесь нет: проверяются чистые
функции разбора и агрегации, вход — таблица той же формы, что отдаёт коннектор."""

import pandas as pd

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
