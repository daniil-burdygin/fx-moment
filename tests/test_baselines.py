import pandas as pd

from fxmoment import baselines
from fxmoment.backtest import make_splits, run_backtest
from fxmoment.indicators import Level


def test_calendar_events_first_all_and_causal():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    first = baselines.calendar_events(idx, 20, 25, "first")
    every = baselines.calendar_events(idx, 20, 25, "all")
    assert first.sum() == 3  # по одному событию в месяц
    days = first.index[first].day
    assert (days >= 20).all() and (days <= 25).all()
    assert every.sum() > first.sum() and (every & first).sum() == 3
    # причинность: срез календаря не меняет прошлые события
    cut = idx[:40]
    assert first.loc[cut].equals(baselines.calendar_events(cut, 20, 25, "first"))
    # 25-е выпало на выходной (май 2024) → первый день публикации после него, не пропуск месяца
    idx_may = pd.bdate_range("2024-05-01", "2024-05-31")
    d25 = baselines.calendar_events(idx_may, 25, 31, "first")
    assert d25.sum() == 1 and int(d25.index[d25].day[0]) == 27


def test_calendar_matrix_summary_and_comparison_on_synthetic(panel):
    splits = make_splits(panel.loc["2018-01-01":].index, first_test="2020-01-01", test_months=6)
    cal = baselines.calendar_matrix(
        panel, splits, corridors=("TJS", "KZT"), rules=(("day20-25", 20, 25, ("first",)),)
    )
    assert set(cal["corridor"]) == {"TJS", "KZT"} and len(cal) == 2 * len(splits)
    assert (cal["n_events"] > 0).all()
    summ = baselines.calendar_summary(cal)
    assert set(summ["corridor"]) == {"TJS", "KZT", "all"}
    assert list(summ.columns) == baselines.SUMMARY_COLUMNS
    result = run_backtest(panel, corridors=("TJS", "KZT"), indicators=(Level,), splits=splits, horizons=(20,))
    cmp_ = baselines.calendar_vs_stack(cal, result.matrix)
    assert list(cmp_["stack"]) == ["level"] and list(cmp_.columns) == baselines.COMPARE_COLUMNS
    assert cmp_["blocks"].iloc[0] > cmp_["blocks_by_window"].iloc[0]
    # правило против самого себя — нулевая разница и вердикт «разницы нет»
    same = baselines.calendar_vs_stack(cal, cal.assign(indicator="level"))
    assert abs(float(same["diff_lift"].iloc[0])) < 1e-12
    assert same["verdict_lift"].iloc[0] == "разницы нет"


def _holidays(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["country", "date", "name", "movable"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_holiday_events_count_publication_days_not_calendar_days():
    idx = pd.bdate_range("2021-01-01", "2021-01-31")  # только рабочие дни: суббот и воскресений нет
    starts = pd.DatetimeIndex(["2021-01-13"])  # среда
    ev = baselines.holiday_events(idx, starts, 3)
    assert list(ev.index[ev].strftime("%Y-%m-%d")) == ["2021-01-08", "2021-01-11", "2021-01-12"]
    # k отсчитывается по оси публикации: три дня перекрыли выходные 9–10 января
    one = baselines.holiday_events(idx, starts, 1)
    assert list(one.index[one].strftime("%Y-%m-%d")) == ["2021-01-12"]
    # событие не встаёт на сам праздник и не выходит за начало ряда
    assert not bool(ev.loc[pd.Timestamp("2021-01-13")])
    early = baselines.holiday_events(idx, pd.DatetimeIndex(["2021-01-04"]), 5)
    assert early.sum() == 1 and early.index[early][0] == pd.Timestamp("2021-01-01")


def test_holiday_events_never_fall_on_days_without_publication():
    # в оси публикации нет выходных и есть дыра 18–22 января (нерабочие дни ЦБ)
    idx = pd.bdate_range("2021-01-01", "2021-01-31").drop(pd.bdate_range("2021-01-18", "2021-01-22"))
    starts = pd.DatetimeIndex(["2021-01-24", "2021-01-16"])
    ev = baselines.holiday_events(idx, starts, 3)
    assert set(ev.index) == set(idx)  # ряд событий живёт только на оси публикации
    fired = pd.DatetimeIndex(ev.index[ev])
    assert fired.isin(idx).all() and not fired.isin(pd.bdate_range("2021-01-18", "2021-01-22")).any()
    assert fired.weekday.max() <= 4
    # оба праздника выпали на нерабочие дни ЦБ: три последних дня публикации перед ними — 13–15
    # января, а не 21–23 календарных. Дыра 18–22 в отсчёт k не входит.
    assert list(fired.strftime("%Y-%m-%d")) == ["2021-01-13", "2021-01-14", "2021-01-15"]


def test_holiday_starts_glues_blocks_and_filters_scope():
    hol = _holidays(
        [
            ("TJ", "2021-03-21", "Международный праздник Навруз", "no"),
            ("TJ", "2021-03-22", "Международный праздник Навруз", "no"),
            ("TJ", "2021-03-23", "Международный праздник Навруз", "no"),
            ("TJ", "2021-05-09", "День Победы", "no"),
            ("TJ", "2021-05-13", "Иди Рамазон", "yes"),
            ("UZ", "2021-01-01", "Новый год", "no"),
        ]
    )
    assert list(baselines.holiday_starts(hol, "TJ", "all").strftime("%Y-%m-%d")) == [
        "2021-03-21",
        "2021-05-09",
        "2021-05-13",
    ]
    assert list(baselines.holiday_starts(hol, "TJ", "major").strftime("%Y-%m-%d")) == [
        "2021-03-21",
        "2021-05-13",
    ]
    assert list(baselines.holiday_starts(hol, "TJ", "fixed").strftime("%Y-%m-%d")) == [
        "2021-03-21",
        "2021-05-09",
    ]
    assert list(baselines.holiday_starts(hol, "UZ", "all").strftime("%Y-%m-%d")) == ["2021-01-01"]


def test_holiday_calendar_snapshot_covers_all_corridors_and_test_windows():
    hol = baselines.load_holidays()
    assert set(baselines.CORRIDOR_COUNTRY.values()) <= set(hol["country"])
    for country in baselines.CORRIDOR_COUNTRY.values():
        for scope in baselines.HOLIDAY_SCOPES:
            starts = baselines.holiday_starts(hol, country, scope)
            assert len(starts) > 0
            assert starts.min() <= pd.Timestamp("2018-12-31")
            assert starts.max() >= pd.Timestamp("2026-01-01")


def test_holiday_matrix_and_comparison_on_synthetic(panel):
    splits = make_splits(panel.loc["2018-01-01":].index, first_test="2020-01-01", test_months=6)
    hol = baselines.holiday_matrix(panel, splits, corridors=("TJS", "KZT"), ks=(3,), scopes=("major",))
    assert set(hol["corridor"]) == {"TJS", "KZT"} and len(hol) == 2 * len(splits)
    assert set(hol["indicator"]) == {"holiday:k3:major"} and (hol["n_events"] > 0).all()
    summ = baselines.calendar_summary(hol)
    assert set(summ["corridor"]) == {"TJS", "KZT", "all"}
    cal = baselines.calendar_matrix(
        panel, splits, corridors=("TJS", "KZT"), rules=(("day25", 25, 31, ("first",)),)
    )
    cmp_ = baselines.holiday_vs_baselines(hol, cal, pd.DataFrame(), bases=("calendar:day25:first",))
    assert list(cmp_.columns) == baselines.HOLIDAY_COMPARE_COLUMNS
    assert list(cmp_["rule"]) == ["holiday:k3:major"] and list(cmp_["baseline"]) == ["calendar:day25:first"]
    same = baselines.holiday_vs_baselines(
        hol, hol.assign(indicator="calendar:day25:first"), pd.DataFrame(), bases=("calendar:day25:first",)
    )
    assert abs(float(same["diff_lift"].iloc[0])) < 1e-12 and same["verdict_lift"].iloc[0] == "разницы нет"
