"""Наборы признаков обучаемого за флагом (`backtest --ml-features eurusd,timesfm`). Три шва: без
флага прогон не меняется, включённый набор меняет только обучаемый, и признак каузален — прогноз на
дату T построен по ряду до T, а порча прогнозных столбцов после train_end не двигает калибровку."""

import numpy as np
import pandas as pd
import pytest

from fxmoment.backtest import make_splits, run_backtest, signals_as_of
from fxmoment.backtest.engine import context_columns
from fxmoment.config import CONTEXT
from fxmoment.data.forecast import (
    HORIZON,
    attach_forecast,
    build_snapshot,
    column_name,
    feature_names,
    features_from_quantiles,
    is_forecast_column,
)
from fxmoment.indicators import BASE_INDICATORS, LearnedMinimum, LearnedMinimumPooled
from fxmoment.indicators.features import build_features, enrich_context
from fxmoment.indicators.ml import _currency_context

TWO = ("TJS", "KZT")


def with_eur(panel: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    """EUR/RUB как доллар со своим блужданием: EUR/USD тогда не функция ни одного коридора."""
    rng = np.random.default_rng(seed)
    out = panel.copy()
    out["EUR"] = panel["USD"] * np.exp(np.cumsum(rng.normal(0, 0.004, len(panel))))
    return out


def synthetic_snapshot(index: pd.DatetimeIndex, currencies: tuple[str, ...], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in currencies:
        for t in index:
            feats = {f: float(rng.normal(0, 50)) for f in feature_names()}
            rows.append({"currency": c, "pub_date": t, **feats})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ выключено по умолчанию


def test_without_flag_extra_columns_do_not_reach_features(panel):
    """Столбцы EUR и снимка есть в панели, флага нет — признаков от них нет, и контекст прежний."""
    rich = attach_forecast(with_eur(panel), synthetic_snapshot(panel.index, ("TJS", "USD")))
    assert context_columns(rich, CONTEXT, TWO, (LearnedMinimum,)) == ["USD", "EUR"]
    ctx = enrich_context(rich["TJS"], rich[context_columns(rich, CONTEXT, TWO, (LearnedMinimum,))])
    assert not [c for c in ctx.columns if c.startswith(("_eurusd", "_fc_", "_usd_fc_"))]
    x = build_features(rich["TJS"], ctx)
    assert not [c for c in x.columns if c.startswith(("eurusd_", "fc_", "usd_fc_"))]


def test_run_without_flag_is_byte_identical_to_run_without_the_data(panel):
    """Матрица прогона без флагов не зависит от того, несёт ли панель EUR и снимок прогнозов."""
    rich = attach_forecast(with_eur(panel), synthetic_snapshot(panel.index, ("TJS", "KZT", "USD")))
    kw = dict(corridors=TWO, indicators=(*BASE_INDICATORS, LearnedMinimum), analysis_start="2017-03-01")
    kw["splits"] = make_splits(panel.index, first_test="2020-01-01", test_months=6, purge_days=20)[1:]
    plain = run_backtest(panel, horizons=(20,), **kw)
    same = run_backtest(rich, horizons=(20,), **kw)
    for name in ("matrix", "signals", "calibration"):
        pd.testing.assert_frame_equal(getattr(plain, name), getattr(same, name))


# ------------------------------------------------------------------ EUR/USD


def test_eurusd_features_appear_only_with_the_flag_and_are_causal(panel):
    rich = with_eur(panel)
    rate, cols = rich["TJS"], rich[["USD", "EUR"]]
    ctx = enrich_context(rate, cols, ml_features=("eurusd",))
    pd.testing.assert_series_equal(ctx["_eurusd"], rich["EUR"] / rich["USD"], check_names=False)
    x = build_features(rate, ctx)
    assert {"eurusd_ret5", "eurusd_ret20", "eurusd_rank250"} <= set(x.columns)
    # порча EUR после среза не меняет ни одного признака до среза
    cut = 900
    spoiled = rich.copy()
    spoiled.iloc[cut:, spoiled.columns.get_loc("EUR")] *= 1.4
    y = build_features(rate, enrich_context(rate, spoiled[["USD", "EUR"]], ml_features=("eurusd",)))
    pd.testing.assert_frame_equal(x.iloc[:cut], y.iloc[:cut])
    # без EUR в контексте набор молчит, а не падает
    alone = build_features(rate, enrich_context(rate, rich[["USD"]], ml_features=("eurusd",)))
    assert not [c for c in alone.columns if c.startswith("eurusd_")]


def test_flag_changes_only_the_learnable_indicator(panel):
    """Пять базовых индикаторов обязаны совпасть построчно: разница изолирована в обучаемом."""
    rich = with_eur(panel)
    kw = dict(corridors=TWO, indicators=(*BASE_INDICATORS, LearnedMinimum), analysis_start="2017-03-01")
    kw["splits"] = make_splits(panel.index, first_test="2020-01-01", test_months=6, purge_days=20)[1:]
    off = run_backtest(rich, horizons=(20,), **kw).matrix
    on = run_backtest(rich, horizons=(20,), ml_features=("eurusd",), **kw).matrix
    base = (off["indicator"] != "ml_localmin").to_numpy()
    pd.testing.assert_frame_equal(
        off[base].reset_index(drop=True), on[base].reset_index(drop=True)
    )
    ml_off = off[~base].reset_index(drop=True)
    ml_on = on[~base].reset_index(drop=True)
    assert not ml_off["n_events"].equals(ml_on["n_events"])


def test_unknown_feature_set_is_refused():
    with pytest.raises(ValueError):
        enrich_context(pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-02"])), None, ml_features=("нет",))


# ------------------------------------------------------------------ снимок TimesFM


def test_features_from_quantiles_are_bps_to_rate():
    a = 10.0
    q = np.tile(np.linspace(9.9, 10.1, 9), (HORIZON, 1))
    q[:, 4] = 10.0 * (1 + 1e-4 * np.arange(1, HORIZON + 1))  # медиана растёт на 1 бп в шаг
    f = features_from_quantiles(a, q)
    assert f["med_h1_bps"] == pytest.approx(1.0)
    assert f["med_h20_bps"] == pytest.approx(20.0)
    assert f["mean5_bps"] == pytest.approx(3.0)
    assert f["mean20_bps"] == pytest.approx(10.5)
    assert f["min20_bps"] == pytest.approx(1.0)
    assert f["q10_h20_bps"] == pytest.approx(-100.0)
    assert f["q90_h20_bps"] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        features_from_quantiles(a, q[:5])


def test_attach_keeps_each_forecast_on_its_own_date_without_filling(panel):
    snap = synthetic_snapshot(panel.index[:3], ("TJS",))
    out = attach_forecast(panel, snap)
    col = column_name("TJS", "mean20_bps")
    assert list(out.columns[: len(panel.columns)]) == list(panel.columns)
    assert out[col].notna().sum() == 3  # пропуски не заполняются
    want = snap.loc[snap["pub_date"] == panel.index[1], "mean20_bps"].iloc[0]
    assert out.loc[panel.index[1], col] == want
    fc_cols = {c for c in out.columns if is_forecast_column(c)}
    got = context_columns(out, CONTEXT, TWO, (LearnedMinimum,), ("timesfm",))
    assert set(got) == {"USD", *fc_cols}
    assert context_columns(out, CONTEXT, TWO, (LearnedMinimum,)) == ["USD"]  # без флага контекст прежний


def test_enrich_context_maps_own_corridor_and_usd_only(panel):
    snap = synthetic_snapshot(panel.index, ("TJS", "KZT", "USD"))
    p = attach_forecast(panel, snap)
    cols = context_columns(p, CONTEXT, TWO, (LearnedMinimum,), ("timesfm",))
    ctx = enrich_context(p["TJS"], p[cols], ml_features=("timesfm",))
    pd.testing.assert_series_equal(
        ctx["_fc_mean20_bps"], p[column_name("TJS", "mean20_bps")], check_names=False
    )
    pd.testing.assert_series_equal(
        ctx["_usd_fc_mean20_bps"], p[column_name("USD", "mean20_bps")], check_names=False
    )
    assert not ctx["_fc_mean20_bps"].equals(p[column_name("KZT", "mean20_bps")])
    assert "_usd_fc_min20_bps" not in ctx.columns  # от доллара один признак
    x = build_features(p["TJS"], ctx)
    want = {"fc_mean5", "fc_mean20", "fc_min20", "fc_q10_h20", "fc_q90_h20", "usd_fc_mean20"}
    assert want <= set(x.columns)


def test_pooled_gives_each_corridor_its_own_forecast(panel):
    """У объединённого обучаемого строки чужого коридора идут с ЕГО прогнозом, а не со своим и не
    без признака: иначе пятая часть обучения молча ехала бы на пропусках."""
    p = attach_forecast(with_eur(panel), synthetic_snapshot(panel.index, ("TJS", "KZT", "USD")))
    cols = context_columns(p, CONTEXT, TWO, (LearnedMinimumPooled,), ("timesfm",))
    ctx = enrich_context(p["TJS"], p[cols], ml_features=("eurusd", "timesfm"))
    other = _currency_context(ctx, "KZT")
    pd.testing.assert_series_equal(
        other["_fc_mean20_bps"], p[column_name("KZT", "mean20_bps")], check_names=False
    )
    pd.testing.assert_series_equal(other["_eurusd"], ctx["_eurusd"])  # общий фактор переносится как есть
    assert "_usd_fc_mean20_bps" in other.columns
    assert not [c for c in other.columns if c.startswith(("_rank_", "_dsm_"))]
    x_own = build_features(p["TJS"], ctx)
    x_other = build_features(p["KZT"].dropna(), other)
    assert set(x_own.columns) == set(x_other.columns)  # один набор столбцов на все коридоры


def test_ml_learns_on_forecast_features_and_calibration_ignores_them_after_train_end(panel):
    snap = synthetic_snapshot(panel.index, ("TJS", "USD"))
    p = attach_forecast(panel, snap)
    split = make_splits(p.index, first_test="2020-01-01", test_months=6, purge_days=20)[1]
    rate = p["TJS"]
    cols = context_columns(p, CONTEXT, ("TJS",), (LearnedMinimum,), ("timesfm",))
    ctx = enrich_context(rate, p[cols], ml_features=("timesfm",))
    ind = LearnedMinimum().fit(rate.loc[: split.train_end], ctx.loc[: split.train_end], "2017-03-01")
    assert ind.fitted_ and "fc_mean20" in ind.feature_names_ and "usd_fc_mean20" in ind.feature_names_
    spoiled = p.copy()
    after = spoiled.index > split.train_end
    for col in [c for c in spoiled.columns if is_forecast_column(c)]:
        spoiled.loc[after, col] = 999.0
    kw = {
        "corridors": ("TJS",),
        "indicators": (LearnedMinimum,),
        "analysis_start": "2017-03-01",
        "splits": [split],
        "ml_features": ("timesfm",),
    }
    clean = run_backtest(p, **kw)
    dirty = run_backtest(spoiled, **kw)
    assert clean.calibration["params"].tolist() == dirty.calibration["params"].tolist()


def test_signals_as_of_equals_full_run_with_ml_features(panel):
    """Срез = полный прогон при включённых наборах: EUR и прогнозные столбцы режутся датой среза."""
    p = attach_forecast(with_eur(panel), synthetic_snapshot(panel.index, ("TJS", "KZT", "USD")))
    splits = make_splits(p.index, first_test="2020-01-01", test_months=6, purge_days=20)
    kw = dict(
        corridors=TWO,
        indicators=(LearnedMinimum,),
        analysis_start="2017-03-01",
        splits=splits,
        ml_features=("eurusd", "timesfm"),
    )
    full = run_backtest(p, horizons=(5,), **kw)
    for t in (splits[1].test_start, p.index[-1]):
        t = p.index[p.index.searchsorted(pd.Timestamp(t))]
        state = signals_as_of(p, t, **kw)
        fired = state[state["signal"]]
        expect = full.signals[full.signals["date"] == t]
        got = {(r.corridor, r.indicator, round(r.strength, 9), r.facts) for r in fired.itertuples()}
        want = {(r.corridor, r.indicator, round(r.strength, 9), r.facts) for r in expect.itertuples()}
        assert got == want, f"расхождение на {t.date()}: {got ^ want}"


class _RecordingModel:
    """Заглушка модели: запоминает длины контекстов каждого вызова, отвечает ровной медианой."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def predict_batch(self, contexts, horizon, return_quantiles=True, **_):
        self.calls.append([len(c) for c in contexts])
        for c in contexts:
            q = np.tile(np.linspace(0.99, 1.01, 9) * float(c[-1]), (horizon, 1)).astype(np.float32)
            yield type("Out", (), {"quantiles": q, "forecast": q[:, 4]})()


def test_snapshot_batches_hold_one_context_length_each(panel):
    model = _RecordingModel()
    start = panel.index[100]
    snap = build_snapshot(panel, model, ("TJS", "USD"), start=str(start.date()), cap=300, batch=64)
    assert all(len(set(lengths)) == 1 for lengths in model.calls)
    assert max(max(c) for c in model.calls) == 300
    assert len(snap) == 2 * (len(panel) - 100)
    assert snap.groupby("currency")["pub_date"].is_monotonic_increasing.all()
    assert snap["mean20_bps"].abs().max() < 0.01  # ровная медиана → прогноз равен курсу (float32)
