import numpy as np
import pandas as pd
import pytest

from fxmoment import execution
from fxmoment.backtest import make_splits, run_backtest
from fxmoment.indicators import Level


def _level_signals(panel):
    splits = make_splits(panel.loc["2018-01-01":].index, first_test="2020-01-01", test_months=6)
    return run_backtest(panel, corridors=("TJS",), indicators=(Level,), splits=splits, horizons=(20,)).signals


def test_zero_spread_reproduces_fixing_and_positive_spread_costs_it(panel):
    signals = _level_signals(panel)
    zero = {"CNY": pd.Series(np.zeros(300))}
    table = execution.execution_survival_table(signals, None, panel, zero)
    assert list(table.columns) == execution.COLUMNS
    row = table[(table["corridor"] == "TJS") & (table["source"] == "level")].iloc[0]
    assert row["spread_source"] == "CNY" and row["n"] == len(signals)
    assert row["benefit_fwd_exec_bps"] == pytest.approx(row["benefit_fwd_bps"])
    assert row["hit_mean_exec"] == pytest.approx(row["hit_mean"])
    # факт уровня истинен в день события по построению, на T+1 — доля в [0, 1]
    assert row["own_fact_T"] == pytest.approx(1.0)
    assert 0.0 <= row["own_survival"] <= 1.0
    assert 0.0 <= row["gate_fact_T1"] <= 1.0
    # постоянный спред снимается целиком: он не меняет выбор дня
    flat = {"CNY": pd.Series(np.full(300, 50.0))}
    frow = execution.execution_survival_table(signals, None, panel, flat)
    frow = frow[(frow["corridor"] == "TJS") & (frow["source"] == "level")].iloc[0]
    assert frow["spread_mean_bps"] == pytest.approx(50.0) and frow["spread_p90_dev_bps"] == pytest.approx(0.0)
    assert frow["benefit_fwd_exec_bps"] == pytest.approx(row["benefit_fwd_bps"])
    # разброс спреда остаётся: в девятом дециле отклонения выгода падает примерно на него
    noisy = {"CNY": pd.Series(np.tile([-100.0, 0.0, 100.0], 100))}
    costly = execution.execution_survival_table(signals, None, panel, noisy)
    crow = costly[(costly["corridor"] == "TJS") & (costly["source"] == "level")].iloc[0]
    assert crow["spread_mean_bps"] == pytest.approx(0.0)
    p90 = crow["spread_p90_dev_bps"]
    assert 90.0 <= p90 <= 100.0
    assert crow["benefit_fwd_at_p90_spread_bps"] == pytest.approx(row["benefit_fwd_bps"] - p90, abs=2.0)
    assert (
        abs(crow["benefit_fwd_exec_bps"] - row["benefit_fwd_bps"]) < 1.0
    )  # симметричный шум в среднем нейтрален
    # строка all дублирует единственный коридор
    all_row = costly[(costly["corridor"] == "all") & (costly["source"] == "level")].iloc[0]
    assert all_row["n"] == crow["n"]


def test_without_spreads_execution_columns_are_empty_not_invented(panel):
    signals = _level_signals(panel)
    table = execution.execution_survival_table(signals, None, panel, {})
    row = table.iloc[0]
    assert row["spread_source"] == "" and np.isnan(row["benefit_fwd_exec_bps"])
    assert not np.isnan(row["benefit_fwd_bps"])


def test_stream_pushes_are_a_separate_source(panel):
    signals = _level_signals(panel)
    decided = signals.assign(decision="sent", push_scenario="BUY_NOW").iloc[:5]
    table = execution.execution_survival_table(signals, decided, panel, {})
    stream = table[(table["source"] == execution.STREAM_LABEL) & (table["corridor"] == "TJS")]
    assert len(stream) == 1 and stream.iloc[0]["n"] == 5
