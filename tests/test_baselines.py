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
