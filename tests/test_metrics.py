import numpy as np
import pandas as pd

from fxmoment import labels, metrics
from fxmoment.backtest import BacktestResult
from fxmoment.config import WINDOW_CLOSING


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


def test_frequency_per_week_uses_calendar_length():
    # 2024-01-01 … 2024-06-30 = 182 дня = 26 недель
    assert (
        abs(metrics.frequency_per_week(26, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")) - 1.0)
        < 1e-12
    )
    assert np.isnan(metrics.frequency_per_week(3, pd.Timestamp("2024-02-01"), pd.Timestamp("2024-01-01")))


def test_window_closing_ignores_tolerance():
    r = pd.Series(
        [10, 10.2, 9.8, 10.5, 10.4, 10.6, 10.1], index=pd.bdate_range("2026-01-01", periods=7), dtype=float
    )
    strict = labels.hit_window_closing(r, 2, 0.0)
    for tol in (10.0, 25.0, 50.0):
        pd.testing.assert_series_equal(labels.hit_for_scenario(r, WINDOW_CLOSING, 2, tol), strict)


def test_summary_pools_hit_and_base_by_events():
    rows = []
    for split, hit, base, n in (
        (0, 0.8, 0.4, 10),
        (1, 0.0, 0.5, 2),
        (2, np.nan, 0.45, 0),
        (3, np.nan, 0.45, 0),
    ):
        rows.append(
            {
                "indicator": "level",
                "corridor": "TJS",
                "split": split,
                "h": 20,
                "tol_bps": 25.0,
                "n_events": n,
                "n_scored": n,
                "lift": np.nan,
                "lift_mean": hit / base if n else np.nan,
                "hit_mean": hit,
                "base_mean": base,
                "benefit_excess_bps": np.nan,
                "benefit_fwd_bps": np.nan,
                "benefit_sym_bps": np.nan,
                "freq_per_week": 0.3,
                "empty_month_share": 0.5,
            }
        )
    s = BacktestResult(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame()).summary()
    row = s.iloc[0]
    assert row["windows"] == 4 and row["silent_windows"] == 2
    assert abs(row["hit_mean_pooled"] - (0.8 * 10 + 0.0 * 2) / 12) < 1e-12
    assert abs(row["base_mean_pooled"] - (0.4 * 10 + 0.5 * 2) / 12) < 1e-12
    assert abs(row["lift_mean_pooled"] - 1.6) < 1e-12
    assert abs(row["share_lift_mean_lt_1"] - 0.5) < 1e-12  # из двух окон с событиями
    assert BacktestResult(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame()).summary(h=5).empty
