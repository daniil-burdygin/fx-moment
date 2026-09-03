import pandas as pd

from fxmoment import analysis
from fxmoment.backtest import make_splits, run_backtest
from fxmoment.combine import evaluate_stream
from fxmoment.indicators import Level, Momentum


def _small_result(panel: pd.DataFrame):
    splits = make_splits(panel.loc["2018-01-01":].index, first_test="2020-01-01", test_months=6)
    return run_backtest(panel, corridors=("TJS", "KZT"), indicators=(Momentum, Level), splits=splits), splits


def test_price_of_waiting_and_transfer_on_synthetic(panel):
    result, splits = _small_result(panel)
    pw = analysis.price_of_waiting_table(result, panel, k=10)
    assert set(pw["corridor"]) <= {"TJS", "KZT"}
    assert ((pw["confirmed_share"] >= 0) & (pw["confirmed_share"] <= 1)).all()
    assert pw["verdict"].isin(["ждать подтверждения", "слать сразу"]).all()
    tr = analysis.transfer_table(result, panel, source="KZT", indicators=(Momentum, Level))
    assert set(tr["corridor"]) == {"TJS", "KZT"}
    # на коридоре-источнике перенесённые параметры совпадают со своими → те же события
    own = result.matrix[(result.matrix["corridor"] == "KZT") & (result.matrix["h"] == 20)]
    own = own[own["tol_bps"] == 25.0].set_index(["indicator", "split"])["n_events"]
    src = tr[(tr["corridor"] == "KZT") & (tr["h"] == 20)].set_index(["indicator", "split"])["n_events"]
    assert (own.sort_index() == src.sort_index()).all()
    cmp_ = analysis.transfer_compare(result, tr)
    assert "lift_drop" in cmp_.columns and len(cmp_) > 0


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
        assert "freq_per_week_median" in st.columns
