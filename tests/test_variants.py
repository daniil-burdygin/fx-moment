import pandas as pd
import pytest

from fxmoment import metrics, variants
from fxmoment.analysis import first_test_from_labels
from fxmoment.backtest import run_backtest
from fxmoment.combine import history_status, rank_from_history
from fxmoment.indicators import Level
from fxmoment.metrics import MONTH_TRUNC_COLUMNS

KW = dict(
    corridors=("TJS",), indicators=(Level,), analysis_start="2018-01-01", horizons=(20,), tolerances=(25.0,)
)


def test_first_test_variant_reproduces_common_windows_byte_for_byte(panel):
    """Тест с более раннего окна — расширяющееся обучение от того же начала: общие окна обязаны дать
    те же строки матрицы, иначе параметризация что-то сдвинула (пункт 8, 03.09 вечер)."""
    late = run_backtest(panel, first_test="2020-07-01", **KW)
    early = run_backtest(panel, first_test="2020-01-01", **KW)
    assert len(early.splits) == len(late.splits) + 1
    assert set(MONTH_TRUNC_COLUMNS) <= set(late.matrix.columns)
    chk = variants.overlap_check(late.matrix, early.matrix)
    assert (chk["rows_differ"] == 0).all() and chk["rows"].iloc[0] == len(late.matrix)
    extra = variants.extra_windows_summary(early.matrix, set(late.matrix["window"]))
    assert set(extra["window"]) == {"all", early.splits[0].label()}
    # порча одной строки видна и по столбцу, и по числу строк
    spoiled = early.matrix.copy()
    spoiled.loc[spoiled.index[-1], "hit_mean"] += 0.1
    bad = variants.overlap_check(late.matrix, spoiled).set_index("column")
    assert bad.loc["hit_mean", "rows_differ"] == 1 and bad.loc["hit_mean", "max_abs_diff"] == pytest.approx(
        0.1
    )
    assert bad.loc["n_events", "rows_differ"] == 0


def test_monthly_base_in_evaluate_events(panel):
    rate = panel["TJS"].dropna()
    idx = rate.index
    ev = pd.Series(False, index=idx)
    ev.loc[idx[(idx >= "2020-01-01") & (idx <= "2020-06-30") & (idx.day <= 2)]] = True
    win = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))
    w = metrics.evaluate_events(rate, ev, "BUY_NOW", 20, win, 25.0, with_ci=False)
    m = metrics.evaluate_events(rate, ev, "BUY_NOW", 20, win, 25.0, with_ci=False, base="month")
    assert w["hit_mean"] == m["hit_mean"] and w["n_scored"] == m["n_scored"]
    assert w["base_mean"] != m["base_mean"] and w["benefit_excess_bps"] != m["benefit_excess_bps"]
    assert w["hit_rate"] == m["hit_rate"] and w["base_rate"] == m["base_rate"]  # строгое прочтение — по окну
    with pytest.raises(ValueError):
        metrics.evaluate_events(rate, ev, "BUY_NOW", 20, win, 25.0, base="year")
    # месяц короче пяти размеченных дней отдаёт базу окна
    days = idx[(idx >= "2020-01-01") & (idx <= "2020-01-03")].append(
        idx[(idx >= "2020-02-01") & (idx <= "2020-02-29")]
    )
    series = pd.Series(1.0, index=idx)
    series.loc[days[:3]] = 5.0
    base = metrics.monthly_base(series, days, days[:1].append(days[-1:]))
    assert base[0] == pytest.approx(float(series.loc[days].mean()))  # январь: три дня → окно
    assert base[1] == pytest.approx(1.0)  # февраль: полный месяц → свой


def test_rank_from_history_on_month_base_reads_month_columns():
    row = {
        "corridor": "TJS",
        "indicator": "level",
        "split": 0,
        "h": 20,
        "tol_bps": 25.0,
        "hit_mean_trunc": 0.5,
        "base_mean_trunc": 0.4,
        "n_scored_trunc": 40,
        "benefit_excess_trunc": 10.0,
        "base_mean_month_trunc": 0.6,
        "benefit_excess_month_trunc": -5.0,
    }
    m = pd.DataFrame([row])
    _rank_w, muted_w = rank_from_history(m, "TJS", 1)
    _rank_m, muted_m = rank_from_history(m, "TJS", 1, rank_base="month")
    assert muted_w == () and muted_m == ("level",)  # на месячной базе lift < 1 и выгода ≤ 0
    assert "base_mean_month_trunc" in history_status(
        m.drop(columns=["base_mean_month_trunc"]), rank_base="month"
    )
    assert history_status(m, rank_base="year") is not None
    with pytest.raises(ValueError):
        rank_from_history(m, "TJS", 1, rank_base="year", strict=True)


def test_first_test_from_labels():
    assert first_test_from_labels(["2019-01…2019-06", "2019-07…2019-12"]) == "2019-01-01"
    assert first_test_from_labels([]) is None
    assert first_test_from_labels(["bad"]) is None


def test_compare_runs_writes_all_tables(panel, tmp_path):
    from fxmoment.combine import PolicyParams
    from fxmoment.report import write_report

    late = run_backtest(panel, first_test="2020-07-01", **KW)
    early = run_backtest(panel, first_test="2020-01-01", **KW)
    write_report(late, panel, tmp_path / "latest")
    write_report(early, panel, tmp_path / "from2020", policy=PolicyParams(rank_base="month"))
    out = variants.compare_runs(tmp_path / "latest", tmp_path / "from2020", panel)
    names = (
        "matrix_overlap.csv",
        "matrix_compare.csv",
        "stream_compare.csv",
        "stream_raw.csv",
        "stream_shape.csv",
        "decisions.csv",
        "extra_windows.csv",
        "extra_windows_stream.csv",
        "extra_windows_shape.csv",
        "README.md",
    )
    for name in names:
        assert (out / name).exists()
    overlap = pd.read_csv(out / "matrix_overlap.csv")
    assert (overlap["rows_differ"] == 0).all()
    mcmp = pd.read_csv(out / "matrix_compare.csv").set_index("indicator")
    assert mcmp.loc["level", "diff_lift"] == pytest.approx(0.0)
    raw = pd.read_csv(out / "stream_raw.csv").set_index("scenario")
    assert raw.loc["all", "events_latest"] > 0 and abs(raw.loc["all", "diff_hit"]) <= 1
    extra = pd.read_csv(out / "extra_windows_shape.csv")
    assert extra.loc[extra["corridor"] == "all", "windows"].item() == 1
    prov = pd.read_json(out / "provenance.json", typ="series")
    assert prov["variant"]["rank_base"] == "month" and prov["latest"]["rank_base"] == "window"
