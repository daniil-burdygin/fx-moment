"""Вариантные индикаторы (💬 03.09 вечер, пункты 2 и 5): уровень с вычтенным дрейфом локальной ноги и
обучаемый, общий на коридоры. Те же швы, что у остальных: каузальность, разогрев, обучение только до
train_end, функция среза = полный прогон, и сравнение вариантов машинкой `compare-runs`."""

import json

import numpy as np
import pandas as pd

from fxmoment import variants
from fxmoment.backtest import make_splits, run_backtest, signals_as_of
from fxmoment.backtest.engine import context_columns
from fxmoment.config import CORRIDORS
from fxmoment.indicators import LearnedMinimum, LearnedMinimumPooled, Level, LevelDrift
from fxmoment.indicators.base import rolling_days_since_min, rolling_pct_rank
from fxmoment.indicators.features import enrich_context
from fxmoment.indicators.level_drift import drift_adjusted_rank
from fxmoment.texts.library import check_message, render

TWO = ("TJS", "KZT")


def _drifting_panel(
    n: int = 800, step: float = -0.0005, noise_sd: float = 0.0, seed: int = 0
) -> pd.DataFrame:
    """Локальная нога дешевеет по экспоненте при неподвижном долларе; `noise_sd` добавляет к ней
    случайное блуждание."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    walk = np.cumsum(rng.normal(0, noise_sd, n)) if noise_sd else np.zeros(n)
    return pd.DataFrame({"USD": 60.0, "UZS": 10.0 * np.exp(step * np.arange(n) + walk)}, index=idx)


def test_level_drift_removes_steady_drift():
    """Ряд с дрейфом −0,05 % в день и шумом: `level` видит новый минимум почти каждый день (процентиль
    у нуля), `level_drift` после приведения к сегодняшнему дрейфу — процентиль около половины и
    сигналы в разы реже. На ряде без шума все приведённые значения равны сегодняшнему, и процентиль
    там определяют ошибки округления — поэтому ранг меряется на шуме, а дрейф — без него."""
    panel = _drifting_panel(noise_sd=0.002)
    rate, ctx = panel["UZS"], panel[["USD"]]
    plain = Level(window=120, pct=0.10, rearm=0).compute(rate, ctx)
    adj = LevelDrift(window=120, pct=0.10, drift_window=250, rearm=0).compute(rate, ctx)
    tail = slice(300, None)
    assert plain["pct_rank"].iloc[tail].mean() < 0.2
    assert 0.35 < adj["pct_rank"].iloc[tail].mean() < 0.65
    assert adj["signal"].iloc[tail].mean() < 0.3 < plain["signal"].iloc[tail].mean()
    assert adj["pct_rank"].iloc[:250].isna().all()  # до разогрева дрейфа процентиля нет
    assert (adj["local_leg"] == 1.0).all()
    pure = _drifting_panel()
    exact = LevelDrift(window=120, pct=0.10, rearm=0).compute(pure["UZS"], pure[["USD"]])
    year = (np.exp(-0.0005 * 250) - 1) * 100
    assert np.allclose(exact["drift_pct_year"].dropna(), year)
    # без USD дрейф считается по самому курсу, и это видно в факте
    alone = LevelDrift(window=120, pct=0.10, rearm=0).compute(rate, None)
    assert (alone["local_leg"] == 0.0).all()
    assert np.allclose(alone["drift_pct_year"].dropna(), adj["drift_pct_year"].dropna())  # доллар неподвижен


def test_drift_adjusted_rank_with_zero_drift_is_plain_rank(panel):
    rate = panel["TJS"]
    rank, dsm = drift_adjusted_rank(rate, pd.Series(0.0, index=rate.index), 60)
    pd.testing.assert_series_equal(rank, rolling_pct_rank(rate, 60), check_names=False)
    pd.testing.assert_series_equal(dsm, rolling_days_since_min(rate, 60), check_names=False)


def test_level_drift_causal_and_warmup(panel):
    rate = panel["TJS"]
    ctx = enrich_context(rate, panel[["USD"]])
    for params in LevelDrift.grid()[::7]:
        ind = LevelDrift(**params)
        out = ind.compute(rate, ctx)
        w = ind.warmup(rate.index)
        assert not out["signal"].iloc[:w].any() and not np.isnan(out["pct_rank"].iloc[w]), params
    ind = LevelDrift()
    base = ind.compute(rate, ctx)
    cut = 1200
    perturbed = rate.copy()
    perturbed.iloc[cut:] *= 1.5
    alt = ind.compute(perturbed, enrich_context(perturbed, panel[["USD"]]))
    pd.testing.assert_frame_equal(base.iloc[:cut], alt.iloc[:cut])


def test_level_drift_text_passes_checker():
    facts = {"pct_rank": 0.08, "window": 120, "days_since_min": 2, "drift_pct_year": -3.4, "local_leg": 1.0}
    title, body = render("UZS", "BUY_NOW", "level_drift", 0.0071, facts)
    assert "снизился на 3,4 %" in body and check_message(title, body) == []
    _title, up = render("UZS", "BUY_NOW", "level_drift", 0.0071, {**facts, "drift_pct_year": 1.2})
    assert "вырос на 1,2 %" in up


def test_context_columns_adds_corridors_only_for_pooled(panel):
    assert context_columns(panel, ("USD",), TWO, (Level, LearnedMinimum)) == ["USD"]
    assert context_columns(panel, ("USD",), TWO, (Level, LearnedMinimumPooled)) == ["USD", "TJS", "KZT"]
    assert context_columns(panel, ("USD", "EUR"), (), (LearnedMinimumPooled,)) == ["USD"]


def test_pooled_fit_uses_all_corridors_and_own_threshold(panel):
    rate = panel["TJS"]
    ctx = enrich_context(rate, panel[["USD", "TJS", "KZT", "UZS"]]).iloc[:1000]
    pooled = LearnedMinimumPooled().fit(rate.iloc[:1000], ctx, train_start="2016-06-01")
    single = LearnedMinimum().fit(rate.iloc[:1000], ctx, train_start="2016-06-01")
    assert pooled.fitted_ and single.fitted_
    assert pooled.train_corridors_ == 3 and single.train_corridors_ == 1
    assert pooled.train_rows_ > 2.5 * single.train_rows_  # три коридора вместо одного
    assert {f"corr_{c}" for c in CORRIDORS} <= set(pooled.feature_names_)
    assert "corr_TJS" not in single.feature_names_
    assert pooled.fit_info() == {"_train_corridors": 3, "_train_rows": pooled.train_rows_}
    assert single.fit_info() == {} and pooled.params["pooled"] is True and "pooled" not in single.params
    out = pooled.compute(rate, ctx)
    fired = out[out["signal"]]
    assert (fired["pct_rank"] <= pooled.gate_pct).all()
    assert out["proba"].notna().sum() > 0
    # без чужих рядов в контексте объединённый учится на своём, и это не молчит
    alone = LearnedMinimumPooled().fit(rate.iloc[:1000], enrich_context(rate, panel[["USD"]]).iloc[:1000])
    assert alone.train_corridors_ == 1 and alone.fit_info()["_train_corridors"] == 1


def test_pooled_signals_as_of_equals_full_run(panel):
    """Функция среза с объединённым обучаемым даёт то же, что полный прогон: чужие ряды в контексте
    тоже режутся датой среза."""
    splits = make_splits(panel.index, first_test="2019-07-01", test_months=6, purge_days=20)
    inds = (Level, LearnedMinimumPooled)
    kw = dict(corridors=TWO, indicators=inds, analysis_start="2017-03-01", splits=splits)
    full = run_backtest(panel, horizons=(5,), **kw)
    cal = full.calibration[full.calibration["indicator"] == "ml_localmin"]
    params = [json.loads(p) for p in cal["params"]]
    assert all(p["pooled"] is True and p["_train_corridors"] == 2 for p in params)
    for t in (splits[1].test_start, panel.index[-1]):
        t = panel.index[panel.index.searchsorted(pd.Timestamp(t))]
        state = signals_as_of(panel, t, **kw)
        fired = state[state["signal"]]
        expect = full.signals[full.signals["date"] == t]
        got = {(r.corridor, r.indicator, round(r.strength, 9), r.facts) for r in fired.itertuples()}
        want = {(r.corridor, r.indicator, round(r.strength, 9), r.facts) for r in expect.itertuples()}
        assert got == want, f"расхождение на {t.date()}: {got ^ want}"
        for r in state.itertuples():
            c = full.calibration[
                (full.calibration["corridor"] == r.corridor)
                & (full.calibration["indicator"] == r.indicator)
                & (full.calibration["split"] == r.split)
            ]
            assert json.loads(c["params"].iloc[0]) == json.loads(r.params)


def test_pooled_fit_ignores_data_after_train_end(panel):
    """Порча всех рядов панели после train_end — своего и чужих — не меняет порог и параметры."""
    split = make_splits(panel.index, first_test="2020-01-01", test_months=6, purge_days=20)[1]
    rng = np.random.default_rng(11)
    spoiled = panel.copy()
    after = spoiled.index > split.train_end
    noise = np.exp(np.cumsum(rng.normal(0, 0.03, after.sum())))
    for col in spoiled.columns:
        spoiled.loc[after, col] = spoiled.loc[after, col].to_numpy() * noise
    kw = dict(corridors=TWO, indicators=(LearnedMinimumPooled,), analysis_start="2017-03-01", splits=[split])
    a = run_backtest(panel, **kw).calibration.set_index("corridor")["params"].sort_index()
    b = run_backtest(spoiled, **kw).calibration.set_index("corridor")["params"].sort_index()
    assert a.to_dict() == b.to_dict()


def test_compare_runs_pairs_and_extra_indicators(panel, tmp_path):
    from fxmoment.report import write_report

    kw = dict(corridors=("TJS",), analysis_start="2018-01-01", horizons=(20,), tolerances=(25.0,))
    late = run_backtest(panel, indicators=(Level,), first_test="2020-07-01", **kw)
    drift = run_backtest(panel, indicators=(Level, LevelDrift), first_test="2020-07-01", **kw)
    write_report(late, panel, tmp_path / "latest")
    write_report(
        drift, panel, tmp_path / "level-drift", notes={"ml": "local", "extra_indicators": ["level_drift"]}
    )
    out = variants.compare_runs(tmp_path / "latest", tmp_path / "level-drift", pairs={"level_drift": "level"})
    overlap = pd.read_csv(out / "matrix_overlap.csv")
    assert (overlap["rows_differ"] == 0).all() and overlap["rows"].iloc[0] == len(late.matrix)
    pairs = pd.read_csv(out / "pairs_compare.csv")
    assert set(pairs["pair"]) == {"level_drift → level"} and set(pairs["corridor"]) == {"all", "TJS"}
    assert {"diff_lift", "diff_lift_ci_lo_by_window", "verdict_benefit"} <= set(pairs.columns)
    extra = pd.read_csv(out / "extra_indicators.csv")
    assert set(extra["indicator"]) == {"level_drift"} and set(extra["corridor"]) == {"TJS", "all"}
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    assert prov["variant"]["extra_indicators"] == ["level_drift"] and prov["latest"]["ml"] is None
    assert prov["variant_only_indicators"] == ["level_drift"] and prov["pairs"] == {"level_drift": "level"}
    assert "level_drift → level" in (out / "README.md").read_text(encoding="utf-8")
    # пара без одной из сторон молча пропускается, а не падает
    assert variants.pairs_compare(late.matrix, drift.matrix, {"nope": "level"}).empty
