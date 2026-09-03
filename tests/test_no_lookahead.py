"""Функция среза обязана давать на дату T то же, что полный прогон, — для правил и для ML."""

import json

import pandas as pd

from fxmoment.backtest import make_splits, run_backtest, signals_as_of
from fxmoment.indicators import ALL_INDICATORS


def test_signals_as_of_equals_full_run(panel):
    corridors = ("TJS", "KZT")
    splits = make_splits(panel.index, first_test="2019-07-01", test_months=6, purge_days=20)
    full = run_backtest(
        panel,
        corridors=corridors,
        indicators=ALL_INDICATORS,
        analysis_start="2016-01-04",
        splits=splits,
        horizons=(5,),
    )
    # несколько дат среза в разных окнах, включая границы окон
    probes = [splits[1].test_start, splits[2].test_end, panel.index[-40], panel.index[-1]]
    for t in probes:
        t = panel.index[panel.index.searchsorted(pd.Timestamp(t))]
        state = signals_as_of(
            panel,
            t,
            corridors=corridors,
            indicators=ALL_INDICATORS,
            analysis_start="2016-01-04",
            splits=splits,
        )
        fired = state[state["signal"]]
        expect = full.signals[full.signals["date"] == t]
        got = {(r.corridor, r.indicator, round(r.strength, 9), r.facts) for r in fired.itertuples()}
        want = {(r.corridor, r.indicator, round(r.strength, 9), r.facts) for r in expect.itertuples()}
        assert got == want, f"расхождение на {t.date()}: {got ^ want}"
        # параметры на дату среза — те же, что в прогоне
        for r in state.itertuples():
            cal = full.calibration[
                (full.calibration["corridor"] == r.corridor)
                & (full.calibration["indicator"] == r.indicator)
                & (full.calibration["split"] == r.split)
            ]
            assert json.loads(cal["params"].iloc[0]) == json.loads(r.params)
