"""Заглядывание вперёд — дисквалифицирующее условие кейса. Три шва: функция среза против полного
прогона (каузальность compute), обучение и калибровка против данных после train_end, ранжирование
индикаторов против исходов текущего окна (tests/test_policy.py)."""

import json

import numpy as np
import pandas as pd

from fxmoment.backtest import make_splits, run_backtest, signals_as_of
from fxmoment.indicators import ALL_INDICATORS

CORRIDORS = ("TJS", "KZT")


def test_signals_as_of_equals_full_run(panel):
    splits = make_splits(panel.index, first_test="2019-07-01", test_months=6, purge_days=20)
    full = run_backtest(
        panel,
        corridors=CORRIDORS,
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
            corridors=CORRIDORS,
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


def test_fit_and_calibration_ignore_data_after_train_end(panel):
    """Порча всего ряда после train_end не меняет выбранные параметры и порог ML."""
    splits = make_splits(panel.index, first_test="2020-01-01", test_months=6, purge_days=20)
    split = splits[1]
    rng = np.random.default_rng(3)
    spoiled = panel.copy()
    after = spoiled.index > split.train_end
    noise = np.exp(np.cumsum(rng.normal(0, 0.03, after.sum())))
    for col in spoiled.columns:
        spoiled.loc[after, col] = spoiled.loc[after, col].to_numpy() * noise
    clean_run = run_backtest(
        panel, corridors=CORRIDORS, indicators=ALL_INDICATORS, analysis_start="2016-01-04", splits=[split]
    )
    spoiled_run = run_backtest(
        spoiled, corridors=CORRIDORS, indicators=ALL_INDICATORS, analysis_start="2016-01-04", splits=[split]
    )
    a = clean_run.calibration.set_index(["corridor", "indicator"])["params"].sort_index()
    b = spoiled_run.calibration.set_index(["corridor", "indicator"])["params"].sort_index()
    assert a.to_dict() == b.to_dict()
    # а тестовые метрики при этом различаются — порча действительно попала в тест
    assert not clean_run.matrix["hit_mean"].equals(spoiled_run.matrix["hit_mean"])
