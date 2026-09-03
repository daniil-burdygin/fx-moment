"""Снимок прогнозов TimesFM (замер): признаки в бп, стыковка по дате без заполнения, разводка по
коридорам, каузальность обучения с прогнозными столбцами. Модель здесь не грузится."""

import numpy as np
import pandas as pd
import pytest

from fxmoment.backtest import make_splits, run_backtest
from fxmoment.backtest.engine import context_columns
from fxmoment.config import CONTEXT
from fxmoment.data.forecast import (
    HORIZON,
    attach_forecast,
    column_name,
    feature_names,
    features_from_quantiles,
    is_forecast_column,
)
from fxmoment.indicators import LearnedMinimum
from fxmoment.indicators.features import build_features, enrich_context


def synthetic_snapshot(index: pd.DatetimeIndex, currencies: tuple[str, ...], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in currencies:
        for t in index:
            feats = {f: float(rng.normal(0, 50)) for f in feature_names()}
            rows.append({"currency": c, "pub_date": t, **feats})
    return pd.DataFrame(rows)


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
    assert set(context_columns(out, CONTEXT)) == {"USD", *fc_cols}
    assert context_columns(panel, CONTEXT) == ["USD"]  # без снимка контекст прежний


def test_enrich_context_maps_own_corridor_and_usd_only(panel):
    snap = synthetic_snapshot(panel.index, ("TJS", "KZT", "USD"))
    p = attach_forecast(panel, snap)
    ctx = enrich_context(p["TJS"], p[context_columns(p, CONTEXT)])
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
    x0 = build_features(p["TJS"], enrich_context(p["TJS"], p[["USD"]]))
    assert not [c for c in x0.columns if c.startswith(("fc_", "usd_fc_"))]


def test_ml_learns_on_forecast_features_and_calibration_ignores_them_after_train_end(panel):
    snap = synthetic_snapshot(panel.index, ("TJS", "USD"))
    p = attach_forecast(panel, snap)
    split = make_splits(p.index, first_test="2020-01-01", test_months=6, purge_days=20)[1]
    rate = p["TJS"]
    ctx = enrich_context(rate, p[context_columns(p, CONTEXT)])
    ind = LearnedMinimum().fit(rate.loc[: split.train_end], ctx.loc[: split.train_end], "2017-03-01")
    assert ind.fitted_ and "fc_mean20" in ind.feature_names_ and "usd_fc_mean20" in ind.feature_names_
    spoiled = p.copy()
    after = spoiled.index > split.train_end
    for col in [c for c in spoiled.columns if is_forecast_column(c)]:
        spoiled.loc[after, col] = 999.0
    kw = {"corridors": ("TJS",), "indicators": (LearnedMinimum,), "analysis_start": "2017-03-01"}
    kw["splits"] = [split]
    clean = run_backtest(p, **kw)
    dirty = run_backtest(spoiled, **kw)
    assert clean.calibration["params"].tolist() == dirty.calibration["params"].tolist()
