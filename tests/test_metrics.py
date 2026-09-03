import numpy as np
import pandas as pd

from fxmoment import metrics


def test_lift_is_one_when_events_are_all_days():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(1)
    rate = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))), index=idx)
    events = pd.Series(True, index=idx)
    m = metrics.evaluate_events(rate, events, "BUY_NOW", 5, (idx[0], idx[-1]))
    assert abs(m["lift"] - 1.0) < 1e-9
    assert m["n_events"] == 300


def test_clumpiness_detects_series_and_empty_months():
    idx = pd.bdate_range("2024-01-01", periods=130)
    ev = pd.DatetimeIndex([idx[0], idx[1], idx[2], idx[100]])
    c = metrics.clumpiness(ev, idx)
    assert c.share_series == 2 / 3
    assert c.empty_month_share > 0


def test_price_of_waiting_matches_first_confirmation():
    idx = pd.bdate_range("2024-01-01", periods=30)
    rate = pd.Series(np.linspace(10, 11, 30), index=idx)
    fast = pd.Series(False, index=idx)
    slow = pd.Series(False, index=idx)
    fast.iloc[5] = True
    slow.iloc[8] = True
    slow.iloc[9] = True
    df = metrics.price_of_waiting(rate, fast, slow, k=10)
    assert df.loc[0, "days_waited"] == 3 and df.loc[0, "confirmed"]
    assert df.loc[0, "delta_bps"] > 0
