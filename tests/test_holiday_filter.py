"""Предпраздничный фильтр политики: окно, каузальность, флаг CLI (замер Ш1)."""

import subprocess
import sys

import pandas as pd

from fxmoment.backtest import make_splits, run_backtest
from fxmoment.combine import PolicyParams, apply_policy, evaluate_stream, holiday_flag
from fxmoment.indicators import Level

# Навруз в Таджикистане — блок 21–24 марта, дата фиксированная и одна и та же во все годы таблицы.
NAVRUZ_2024 = pd.Timestamp("2024-03-21")


def _events(dates: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame(
        [(d, "TJS", "level", "BUY_NOW", 0.8) for d in dates],
        columns=["date", "corridor", "indicator", "scenario", "strength"],
    )


def test_holiday_flag_marks_k_publication_days_before_holiday():
    idx = pd.bdate_range("2024-02-01", "2024-04-30")
    flag = holiday_flag(idx, "TJS", 5)
    march = list(flag.index[flag & (flag.index >= "2024-03-01") & (flag.index <= "2024-03-31")])
    assert [d.strftime("%Y-%m-%d") for d in march] == [
        # пять дней публикации перед 8 марта (День Матери)
        "2024-03-01",
        "2024-03-04",
        "2024-03-05",
        "2024-03-06",
        "2024-03-07",
        # и перед Наврузом 21 марта; выходные 16–17 марта в k не входят
        "2024-03-14",
        "2024-03-15",
        "2024-03-18",
        "2024-03-19",
        "2024-03-20",
    ]
    assert not bool(flag.loc[NAVRUZ_2024])  # на самом празднике события нет
    assert not bool(flag.loc[pd.Timestamp("2024-03-08")])  # и на 8 марта тоже
    assert not bool(flag.loc[pd.Timestamp("2024-03-13")])  # шестой день до Навруза — уже снаружи
    # окно узкое: большинство дней публикации снаружи
    assert 0 < float(flag.mean()) < 0.5
    # флаг зависит только от оси публикации и таблицы праздников — коридор без страны пуст
    assert not holiday_flag(idx, "USD", 5).any()
    assert not holiday_flag(idx, "TJS", 0).any()


def test_holiday_filter_mutes_push_inside_window_only():
    idx = pd.bdate_range("2024-02-01", "2024-04-30")
    flag = holiday_flag(idx, "TJS", 5)
    inside, outside = pd.Timestamp("2024-03-19"), pd.Timestamp("2024-03-12")
    assert bool(flag.loc[inside]) and not bool(flag.loc[outside])
    out = apply_policy(_events([outside, inside]), {"level": 0}, idx, PolicyParams(), holiday=flag)
    d = dict(zip(out["date"], out["decision"], strict=True))
    assert d[outside] == "sent" and d[inside] == "holiday"
    # без флага тот же день уходит в поток: снимает именно фильтр, а не охлаждение
    free = apply_policy(_events([inside]), {"level": 0}, idx, PolicyParams())
    assert free["decision"].iloc[0] == "sent"
    # снятый фильтром пуш охлаждения не начинает — следующий день публикации свободен
    pair = apply_policy(
        _events([inside, pd.Timestamp("2024-03-20"), pd.Timestamp("2024-03-21")]),
        {"level": 0},
        idx,
        PolicyParams(),
        holiday=flag,
    )
    assert list(pair["decision"]) == ["holiday", "holiday", "sent"]


def test_holiday_filter_reads_no_future_beyond_the_holiday_calendar():
    """Фильтр знает только даты праздников и ось публикации: значения курса в него не входят,
    праздник за пределами таблицы окна не создаёт, а флаг дня не зависит от событий после него."""
    idx = pd.bdate_range("2024-02-01", "2024-04-30")
    flag = holiday_flag(idx, "TJS", 5)
    # окно строго ДО праздника: ни один флаг не стоит на празднике или после него
    starts = pd.DatetimeIndex(["2024-01-01", "2024-03-08", NAVRUZ_2024, "2024-05-09"])
    for start in starts:
        after = flag.index[flag & (flag.index >= start) & (flag.index < start + pd.Timedelta(days=4))]
        assert not len(after)
    # календарь праздников кончается 2026 годом: дальше правило молчит, а не достраивает праздники
    beyond = pd.bdate_range("2030-01-01", "2030-06-30")
    assert not holiday_flag(beyond, "TJS", 5).any()
    # флаг не читает ряд: он функция оси и таблицы, поэтому повторный вызов на том же индексе тот же
    assert flag.equals(holiday_flag(idx, "TJS", 5))


def test_holiday_filter_thins_stream_in_backtest(panel):
    """Прогон с фильтром: пушей не больше, чем без него, снятые помечены `holiday` и посчитаны
    в форме потока."""
    splits = make_splits(panel.loc["2018-01-01":].index, first_test="2020-01-01", test_months=6)
    result = run_backtest(
        panel, corridors=("TJS",), indicators=(Level,), splits=splits, horizons=(20,), tolerances=(25.0,)
    )
    base_dec, _, base_shape = evaluate_stream(
        result, panel, params=PolicyParams(), horizons=(20,), tolerances=(25.0,)
    )
    dec, _, shape = evaluate_stream(
        result, panel, params=PolicyParams(holiday_filter=5), horizons=(20,), tolerances=(25.0,)
    )
    sent_base = int((base_dec["decision"] == "sent").sum())
    sent = int((dec["decision"] == "sent").sum())
    blocked = int((dec["decision"] == "holiday").sum())
    assert blocked > 0 and sent < sent_base
    assert int(shape["holiday_blocked"].sum()) == blocked
    assert int(base_shape["holiday_blocked"].sum()) == 0
    # каждый снятый пуш лежит в предпраздничном окне (обратное неверно: снятый пуш не начинает
    # охлаждения, и часть прежних `cooldown` в других днях становится `sent`)
    flag = holiday_flag(panel["TJS"].dropna().index, "TJS", 5)
    assert flag.reindex(dec[dec["decision"] == "holiday"]["date"]).all()
    assert not flag.reindex(dec[dec["decision"] == "sent"]["date"]).any()


def test_backtest_help_shows_holiday_filter():
    out = subprocess.run(
        [sys.executable, "-m", "fxmoment.cli", "backtest", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "--holiday-filter" in out and "страны-получателя" in out
