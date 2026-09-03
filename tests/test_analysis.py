import numpy as np
import pandas as pd
import pytest

from fxmoment import analysis
from fxmoment.backtest import make_splits, run_backtest
from fxmoment.backtest.engine import BacktestResult
from fxmoment.combine import evaluate_stream
from fxmoment.indicators import Level, Momentum, Seasonality


def _small_result(panel: pd.DataFrame):
    splits = make_splits(panel.loc["2018-01-01":].index, first_test="2020-01-01", test_months=6)
    return run_backtest(panel, corridors=("TJS", "KZT"), indicators=(Momentum, Level), splits=splits), splits


def test_price_of_waiting_and_transfer_on_synthetic(panel):
    result, splits = _small_result(panel)
    pw = analysis.price_of_waiting_table(result, panel, k=10)
    assert set(pw["corridor"]) <= {"TJS", "KZT"}
    assert ((pw["confirmed_share"] >= 0) & (pw["confirmed_share"] <= 1)).all()
    assert pw["verdict"].isin(analysis.VERDICTS).all()
    assert pw["difference_within_ci"].dtype == bool
    tr = analysis.transfer_table(result, panel, source="KZT", indicators=(Momentum, Level))
    assert set(tr["corridor"]) == {"TJS", "KZT"}
    # на коридоре-источнике перенесённые параметры совпадают со своими → те же события
    own = result.matrix[(result.matrix["corridor"] == "KZT") & (result.matrix["h"] == 20)]
    own = own[own["tol_bps"] == 25.0].set_index(["indicator", "split"])["n_events"]
    src = tr[(tr["corridor"] == "KZT") & (tr["h"] == 20)].set_index(["indicator", "split"])["n_events"]
    assert (own.sort_index() == src.sort_index()).all()
    cmp_ = analysis.transfer_compare(result, tr)
    assert "lift_drop" in cmp_.columns and len(cmp_) > 0
    assert analysis.transfer_compare(result, tr.iloc[0:0]).empty


def test_transfer_on_source_matches_own_run_for_calendar_indicator(panel):
    """Сезонность считает годы по переданной истории: перенос обязан считать её по той же истории, что
    бэктест (вся панель), иначе на коридоре-источнике «свои» и «перенесённые» события расходятся."""
    splits = make_splits(panel.loc["2018-01-01":].index, first_test="2020-01-01", test_months=12)
    result = run_backtest(panel, corridors=("TJS",), indicators=(Seasonality,), splits=splits, horizons=(20,))
    tr = analysis.transfer_table(result, panel, source="TJS", indicators=(Seasonality,))
    own = result.matrix[result.matrix["tol_bps"] == 25.0].set_index("split")["n_events"].sort_index()
    src = tr.set_index("split")["n_events"].sort_index()
    assert (own == src).all()


def test_frontier_points_are_out_of_sample_and_flagged(panel):
    result, splits = _small_result(panel)
    grid = [{"window": w, "pct": p, "stall_days": 0, "rearm": 3} for w in (60, 120) for p in (0.1, 0.3)]
    fr = analysis.frontier_table(panel, "TJS", splits, grid=grid)
    assert len(fr) == len(grid)
    assert fr["freq_per_week"].between(0, 5).all()
    assert fr["on_frontier"].dtype == bool
    # сводка без шокового режима: окна 2022 не должны попасть (в синтетике их может не быть вовсе)
    ns = analysis.summary_without_shock(result)
    assert set(ns.columns) >= {"indicator", "corridor", "lift_mean_median"}
    decided, sm, shape = evaluate_stream(result, panel)
    if len(sm):
        st = analysis.stream_summary_without_shock(sm, splits)
        assert "freq_per_week_scenario_median" in st.columns


def test_band_table_omits_operating_point_when_it_is_not_defined(panel):
    """Столбцов рабочей точки нет вовсе, когда её не считали, — вместо столбца из NaN.

    Пустой столбец в отчёте читается как «калибровка точки не дала», хотя вопрос там
    просто не задан: без шоковых окон рабочей точки нет, поток их не исключает.
    """
    _, splits = _small_result(panel)
    grid = [{"window": 60, "pct": p, "stall_days": 0, "rearm": 3} for p in (0.1, 0.3)]
    tables = {"TJS": analysis.frontier_table(panel, "TJS", splits, grid=grid)}
    with_point = analysis._band_table(tables, {"TJS": {"level (калибровка walk-forward)": (0.4, 1.2)}})
    without = analysis._band_table(tables, {})
    assert list(with_point["calibrated_lift_mean"]) == [1.2]
    assert "calibrated_lift_mean" not in without.columns
    assert "calibrated_freq" not in without.columns
    assert set(without.columns) < set(with_point.columns)


def test_monthly_baseline_falls_back_on_short_months(panel):
    """Месяц короче MIN_MONTH_DAYS отдаётся базе окна и считается в fallback_events: месячная
    оценка по трём дням шумнее того, что она измеряет."""
    from fxmoment import analysis
    from fxmoment.backtest.walkforward import Split, make_splits

    splits = make_splits(panel.index)
    rate = panel["TJS"].dropna()
    days = analysis._test_days(rate, splits)
    # событие в каждый первый день месяца — месяцы длинные, отката быть не должно
    dates = pd.Series(days[days.day <= 2])
    signals = pd.DataFrame(
        {"corridor": "TJS", "indicator": "level", "date": dates, "split": 0, "scenario": "BUY_NOW"}
    )
    result = BacktestResult(signals, pd.DataFrame(), pd.DataFrame(), splits)
    table = analysis.monthly_baseline_table(result, panel)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["fallback_events"] == 0
    assert row["events"] > 10
    # обе базы — доли, значит в (0, 1)
    assert 0 < row["base_window"] < 1 and 0 < row["base_month"] < 1

    # короткое окно даёт месяц из двух дней публикации — он обязан уйти в откат на базу окна
    short = [Split(0, splits[0].train_end, days[0], days[0] + pd.Timedelta(days=1))]
    result_short = BacktestResult(
        signals.assign(date=pd.Series([days[0]])), pd.DataFrame(), pd.DataFrame(), short
    )
    table_short = analysis.monthly_baseline_table(result_short, panel)
    assert table_short.iloc[0]["fallback_events"] == 1


def test_day_of_month_detrending_removes_linear_drift():
    """На чистом линейном тренде без сезонности снятое отклонение обязано быть нулевым, а
    неснятое — нет: иначе поправка ничего не поправляет."""
    from fxmoment import analysis
    from fxmoment.backtest.walkforward import Split

    idx = pd.bdate_range("2024-01-01", "2024-12-31")
    rate = pd.Series(100 * (1 + 0.001) ** np.arange(len(idx)), index=idx)
    panel = pd.DataFrame({"TJS": rate})
    splits = [Split(0, idx[0], idx[0], idx[-1])]
    table = analysis.day_of_month_table(panel, splits, corridors=("TJS",))
    assert table["dev_detrended_bps"].abs().max() < 1.0
    assert table["dev_from_month_mean_bps"].abs().max() > 50.0


def test_tax_window_requires_detrended_effect_too():
    """Вердикт «механизм подтверждается» не ставится, если эффект живёт только до снятия тренда."""
    from fxmoment import analysis

    table = pd.DataFrame(
        {
            "corridor": ["TJS", "TJS"],
            "day_of_month": [24, 5],
            "n_days": [10, 10],
            "n_days_detrended": [10, 10],
            "dev_from_month_mean_bps": [-50.0, 50.0],
            "dev_detrended_bps": [5.0, -5.0],  # после снятия тренда знак обратный
            "in_tax_window": [True, False],
        }
    )
    out = analysis.tax_window_summary(table)
    assert out.loc[0, "difference_bps"] < 0
    assert not bool(out.loc[0, "supports_hypothesis"])


def test_weighted_drops_weight_together_with_missing_value():
    """Строка без остатка выпадает вместе со своим весом: иначе числитель и знаменатель считались
    бы по разным наборам чисел месяца и среднее уезжало бы к нулю (аудит 03.09)."""
    from fxmoment import analysis

    g = pd.DataFrame(
        {
            "dev_detrended_bps": [-30.0, float("nan")],
            "n_days_detrended": [10, 0],
            "n_days": [10, 40],
        }
    )
    got = analysis._weighted(g, "dev_detrended_bps", "n_days_detrended")
    assert got == pytest.approx(-30.0)


def test_monthly_baseline_labels_each_indicator_by_its_own_scenario(panel):
    """Разворот объявлен WINDOW_CLOSING, и попадание у него своё (ADR-0003). Раньше вся таблица
    размечалась одной функцией BUY_NOW, и строка разворота в ADR-0011 была посчитана чужой
    меткой (аудит 03.09). На ОДНИХ И ТЕХ ЖЕ датах два сценария обязаны разойтись."""
    from fxmoment import analysis
    from fxmoment.backtest.walkforward import make_splits

    splits = make_splits(panel.index)
    rate = panel["TJS"].dropna()
    days = analysis._test_days(rate, splits)
    dates = pd.Series(days[days.day <= 3])
    signals = pd.concat(
        [
            pd.DataFrame({"corridor": "TJS", "indicator": "level", "date": dates, "split": 0,
                          "scenario": "BUY_NOW"}),
            pd.DataFrame({"corridor": "TJS", "indicator": "reversal", "date": dates, "split": 0,
                          "scenario": "WINDOW_CLOSING"}),
        ],
        ignore_index=True,
    )
    result = BacktestResult(signals, pd.DataFrame(), pd.DataFrame(), splits)
    table = analysis.monthly_baseline_table(result, panel).set_index("indicator")
    assert table.loc["level", "events"] == table.loc["reversal", "events"]
    assert table.loc["level", "hit_mean_pooled"] != table.loc["reversal", "hit_mean_pooled"]
    assert table.loc["level", "base_window"] != table.loc["reversal", "base_window"]


def test_monthly_baseline_separates_the_two_bases(panel):
    """То, ради чего написана вся таблица: две базы обязаны различаться, когда события кучкуются
    в месяцах определённого вида. Совпадение баз означало бы, что месячная считается по окну."""
    from fxmoment import analysis
    from fxmoment.backtest.walkforward import make_splits

    splits = make_splits(panel.index)
    rate = panel["TJS"].dropna()
    days = analysis._test_days(rate, splits)
    month_mean = rate.reindex(days).groupby(days.to_period("M")).transform("mean")
    cheap = days[rate.reindex(days).to_numpy() < month_mean.to_numpy()]  # дни ниже среднего за месяц
    signals = pd.DataFrame(
        {"corridor": "TJS", "indicator": "level", "date": pd.Series(cheap), "split": 0,
         "scenario": "BUY_NOW"}
    )
    result = BacktestResult(signals, pd.DataFrame(), pd.DataFrame(), splits)
    row = analysis.monthly_baseline_table(result, panel).iloc[0]
    assert row["base_window"] != row["base_month"]
    assert row["lift_window"] != row["lift_month"]
    assert row["excess_diff_bps"] == pytest.approx(
        row["excess_month_bps"] - row["excess_window_bps"], abs=1e-9
    )


def test_calibration_vs_fixed_is_paired_and_reads_the_interval():
    """Сравнение калибровки с априорными точками парное по блокам «коридор × окно» и само читает
    интервал. Раньше эти интервалы считались вручную и жили только в тексте — числа отчёта не было,
    а утверждение было (аудит 03.09)."""
    from fxmoment import analysis

    def matrix(hit):
        rows = []
        for corridor in ("KZT", "TJS"):
            for split in range(6):
                rows.append(
                    {
                        "corridor": corridor,
                        "indicator": "level",
                        "split": split,
                        "h": 20,
                        "tol_bps": 25.0,
                        "hit_mean": hit,
                        "base_mean": 0.5,
                        "n_scored": 40,
                    }
                )
        return pd.DataFrame(rows)

    same = analysis.calibration_vs_fixed(matrix(0.6), matrix(0.6)).set_index("indicator")
    assert same.loc["level", "better"] == "разницы нет"
    assert same.loc["level", "diff_fixed_minus_calibrated"] == pytest.approx(0.0)
    assert same.loc["level", "blocks"] == 12  # два коридора × шесть окон, не 24 строки

    better = analysis.calibration_vs_fixed(matrix(0.5), matrix(0.7)).set_index("indicator")
    assert better.loc["level", "better"] == "априорные"
    assert better.loc["level", "diff_ci_lo"] > 0
    worse = analysis.calibration_vs_fixed(matrix(0.7), matrix(0.5)).set_index("indicator")
    assert worse.loc["level", "better"] == "калибровка"
    assert worse.loc["level", "diff_ci_hi"] < 0


def test_paired_comparison_by_window_uses_fewer_blocks_and_wider_interval():
    """Блок по окну — 11 наблюдений вместо 55: коридоры скоррелированы с USD/RUB, и блок по паре
    «коридор × окно» считает пять почти одинаковых коридоров пятью наблюдениями. На двух одинаковых
    коридорах интервал по окнам обязан быть шире, а точечная оценка — той же (03.09 вечер)."""
    from fxmoment import analysis

    def matrix(shift_per_split: float):
        rows = []
        for corridor in ("KZT", "TJS"):
            for split in range(8):
                rows.append(
                    {
                        "corridor": corridor,
                        "indicator": "level",
                        "split": split,
                        "h": 20,
                        "tol_bps": 25.0,
                        "hit_mean": 0.45 + 0.01 * split + shift_per_split * split,
                        "base_mean": 0.5,
                        "n_scored": 40,
                        "benefit_excess_bps": 10.0 * split + 300 * shift_per_split * split,
                    }
                )
        return pd.DataFrame(rows)

    both = analysis.paired_pooled_both(matrix(0.0), matrix(0.02)).set_index("indicator")
    row = both.loc["level"]
    assert row["blocks"] == 16 and row["blocks_by_window"] == 8
    assert row["diff_lift"] > 0
    width_pair = row["diff_lift_ci_hi"] - row["diff_lift_ci_lo"]
    width_window = row["diff_lift_ci_hi_by_window"] - row["diff_lift_ci_lo_by_window"]
    assert width_window > width_pair > 0
    assert row["diff_benefit_ci_hi_by_window"] - row["diff_benefit_ci_lo_by_window"] > (
        row["diff_benefit_ci_hi"] - row["diff_benefit_ci_lo"]
    )
    cmp_ = analysis.calibration_vs_fixed(matrix(0.0), matrix(0.02)).set_index("indicator")
    assert {"better", "better_by_window", "diff_ci_lo_by_window"} <= set(cmp_.columns)
    with pytest.raises(ValueError):
        analysis.paired_pooled_comparison(matrix(0.0), matrix(0.0), block="corridor")
