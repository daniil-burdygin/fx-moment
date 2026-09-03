"""Заглядывание вперёд — дисквалифицирующее условие кейса. Три шва: функция среза против полного
прогона (каузальность compute), обучение и калибровка против данных после train_end, ранжирование
индикаторов против исходов текущего окна (tests/test_policy.py)."""

import json

import numpy as np
import pandas as pd

from fxmoment import metrics
from fxmoment.backtest import make_splits, run_backtest, signals_as_of, split_for_date
from fxmoment.indicators import ALL_INDICATORS, Level
from fxmoment.metrics import TRUNC_COLUMNS

CORRIDORS = ("TJS", "KZT")


def test_signals_as_of_equals_full_run(panel):
    splits = make_splits(panel.index, first_test="2019-07-01", test_months=6, purge_days=20)
    full = run_backtest(
        panel,
        corridors=CORRIDORS,
        indicators=ALL_INDICATORS,
        analysis_start="2017-03-01",
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
            analysis_start="2017-03-01",
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
        panel, corridors=CORRIDORS, indicators=ALL_INDICATORS, analysis_start="2017-03-01", splits=[split]
    )
    spoiled_run = run_backtest(
        spoiled, corridors=CORRIDORS, indicators=ALL_INDICATORS, analysis_start="2017-03-01", splits=[split]
    )
    a = clean_run.calibration.set_index(["corridor", "indicator"])["params"].sort_index()
    b = spoiled_run.calibration.set_index(["corridor", "indicator"])["params"].sort_index()
    assert a.to_dict() == b.to_dict()
    # а тестовые метрики при этом различаются — порча действительно попала в тест
    assert not clean_run.matrix["hit_mean"].equals(spoiled_run.matrix["hit_mean"])


def test_truncated_outcomes_ignore_data_after_window_end(panel):
    """Столбцы *_trunc окна k не меняются от порчи данных после его test_end; полные исходы — меняются."""
    splits = make_splits(panel.index, first_test="2020-01-01", test_months=6, purge_days=20)[:2]
    rng = np.random.default_rng(5)
    spoiled = panel.copy()
    after = spoiled.index > splits[-1].test_end
    for col in spoiled.columns:
        spoiled.loc[after, col] = spoiled.loc[after, col].to_numpy() * np.exp(
            np.cumsum(rng.normal(0, 0.05, after.sum()))
        )
    kw = dict(  # усечённые столбцы пишутся только для h = 20 и допуска 25 бп
        corridors=("TJS",),
        indicators=(Level,),
        analysis_start="2017-03-01",
        splits=splits,
        horizons=(20,),
        tolerances=(25.0,),
    )
    a = run_backtest(panel, **kw).matrix
    b = run_backtest(spoiled, **kw).matrix
    for c in TRUNC_COLUMNS:
        assert np.allclose(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float), equal_nan=True), c
    assert (a["n_scored_trunc"] <= a["n_scored"]).all()
    # механизм усечения: событие в последний день окна есть в n_events, но исхода на ряде до test_end нет
    sp = splits[0]
    rate = panel["TJS"]
    days = rate.loc[sp.test_start : sp.test_end].index
    events = pd.Series(False, index=rate.index)
    events.loc[[days[0], days[-1]]] = True
    win = (sp.test_start, sp.test_end)
    full = metrics.evaluate_events(rate, events, "BUY_NOW", 20, win, 25.0, with_ci=False)
    trunc = metrics.evaluate_events(rate.loc[: sp.test_end], events, "BUY_NOW", 20, win, 25.0, with_ci=False)
    assert full["n_events"] == trunc["n_events"] == 2
    assert full["n_scored"] == 2 and trunc["n_scored"] == 1


def test_live_split_after_last_window(panel):
    """Дата после последнего тестового окна живёт на живом окне: обучение до его начала минус зазор."""
    idx = panel.index
    splits = make_splits(idx, first_test="2019-07-01", test_months=6, purge_days=20, min_test_days=100)
    date = idx[-1]
    assert date > splits[-1].test_end
    live = split_for_date(splits, date, idx, purge_days=20)
    assert live.id == len(splits) and live.test_start == splits[-1].test_end + pd.Timedelta(days=1)
    assert idx.get_loc(live.train_end) == idx.searchsorted(live.test_start) - 21
    assert split_for_date(splits, date).id == splits[-1].id  # без индекса — параметры последнего окна
    state = signals_as_of(
        panel, date, corridors=("TJS",), indicators=(Level,), analysis_start="2017-03-01", splits=splits
    )
    assert (state["split"] == live.id).all()
