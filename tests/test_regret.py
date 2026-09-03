import numpy as np
import pandas as pd
import pytest

from fxmoment import regret
from fxmoment.backtest.walkforward import Split
from fxmoment.config import BUY_NOW, WINDOW_CLOSING


def test_rest_of_month_hit_reads_the_calendar():
    idx = pd.bdate_range("2024-01-01", "2024-02-29")
    rising = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    hit = regret.rest_of_month_hit(rising)
    assert np.isnan(hit.loc["2024-01-31"]) and np.isnan(hit.iloc[-1])  # последний день месяца
    assert hit.drop(["2024-01-31", "2024-02-29"]).eq(1.0).all()  # растущий ряд: сегодня всегда не хуже
    falling = regret.rest_of_month_hit(rising[::-1].set_axis(idx))
    assert falling.drop(["2024-01-31", "2024-02-29"]).eq(0.0).all()


def test_reversal_regret_pairs_last_buy_now_within_k():
    idx = pd.bdate_range("2024-01-01", "2024-06-28")
    rate = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    panel = pd.DataFrame({"TJS": rate})
    splits = [Split(0, idx[0], idx[0], idx[-1])]
    t0, t0b, t1, far = idx[8], idx[10], idx[15], idx[60]
    signals = pd.DataFrame(
        {
            "date": [t0, t0b, t1, far],
            "corridor": "TJS",
            "indicator": ["level", "level", "reversal", "reversal"],
            "scenario": [BUY_NOW, BUY_NOW, WINDOW_CLOSING, WINDOW_CLOSING],
            "split": 0,
        }
    )
    table = regret.reversal_regret_table(signals, None, panel, splits, k=20, h=5)
    assert list(table.columns) == regret.COLUMNS
    assert set(table["pairing"]) == {"events"}  # потока нет — строк stream нет
    row = table[table["corridor"] == "TJS"].iloc[0]
    assert row["n_reversal"] == 2 and row["n_paired"] == 1  # второй разворот дальше k дней
    assert row["days_since_buy_median"] == 5  # последний BUY_NOW, а не первый
    assert row["regret_median_bps"] == pytest.approx((rate[t1] / rate[t0b] - 1) * 1e4)
    assert row["share_regret_positive"] == 1.0
    # растущий ряд: разворот «подтверждается» всегда, но и база равна единице → lift 1
    assert row["hit_fwd"] == pytest.approx(1.0) and row["lift_fwd"] == pytest.approx(1.0)
    assert row["lift_rest_of_month"] == pytest.approx(1.0)
    all_row = table[table["corridor"] == "all"].iloc[0]
    assert all_row["n_reversal"] == 2
