import numpy as np
import pandas as pd
import pytest

from fxmoment.indicators import ALL_INDICATORS, LearnedMinimum, Level, Momentum, Reversal
from fxmoment.indicators.features import enrich_context


def test_momentum_fires_after_n_down_days():
    r = pd.Series([10, 9.9, 9.8, 9.7, 9.6, 9.7], index=pd.bdate_range("2026-01-01", periods=6), dtype=float)
    out = Momentum(n=3, rearm=0).compute(r)
    assert out["signal"].tolist() == [False, False, False, True, True, False]
    assert out.loc[out.index[3], "streak"] == 3


def test_level_bottom_percentile():
    r = pd.Series(np.r_[np.linspace(10, 11, 20), [9.0]], index=pd.bdate_range("2026-01-01", periods=21))
    out = Level(window=10, pct=0.10, rearm=0).compute(r)
    assert bool(out["signal"].iloc[-1]) and out["pct_rank"].iloc[-1] == 0.1


def test_reversal_on_v_shape():
    vals = np.r_[np.linspace(10, 9, 15), np.linspace(9.02, 9.3, 10)]
    r = pd.Series(vals, index=pd.bdate_range("2026-01-01", periods=len(vals)))
    out = Reversal(window=12, rise_bps=100, max_days_since_min=10, rearm=0).compute(r)
    assert out["signal"].any()
    first = out.index[out["signal"].to_numpy().argmax()]
    assert out.loc[first, "rise_pct"] >= 1.0


@pytest.mark.parametrize("cls", ALL_INDICATORS)
def test_indicator_is_causal(cls, panel):
    """Изменение будущего не меняет выход на прошлых датах — общий тест на каузальность."""
    rate = panel["TJS"]
    ctx = enrich_context(rate, panel[["USD"]])
    ind = cls()
    if cls.trainable:
        ind.fit(rate.iloc[:900], ctx.iloc[:900])
    base = ind.compute(rate, ctx)
    cut = 1200
    perturbed = rate.copy()
    perturbed.iloc[cut:] = perturbed.iloc[cut:] * 1.5
    ctx_p = enrich_context(perturbed, panel[["USD"]])
    alt = ind.compute(perturbed, ctx_p)
    pd.testing.assert_frame_equal(base.iloc[:cut], alt.iloc[:cut])


def test_ml_gate_requires_low_level(panel):
    rate = panel["TJS"]
    ctx = enrich_context(rate, panel[["USD"]])
    ind = LearnedMinimum().fit(rate.iloc[:1000], ctx.iloc[:1000])
    out = ind.compute(rate, ctx)
    fired = out[out["signal"]]
    assert (fired["pct_rank"] <= ind.gate_pct).all()
