import numpy as np
import pandas as pd

from fxmoment import labels


def series(values):
    return pd.Series(values, index=pd.bdate_range("2026-01-01", periods=len(values)), dtype=float)


def test_hit_buy_now_requires_no_lower_future_rate():
    r = series([10, 10.1, 10.2, 9.9, 10.5, 10.6, 10.7])
    hit = labels.hit_buy_now(r, h=2)
    assert hit.iloc[0] == 1.0  # будущие 10.1, 10.2 не ниже 10
    assert hit.iloc[1] == 0.0  # через два дня 9.9 < 10.1
    assert np.isnan(hit.iloc[-1])  # горизонт не наступил


def test_hit_window_closing_requires_rise_at_h():
    r = series([10, 10.2, 9.8, 10.5])
    hit = labels.hit_window_closing(r, h=1)
    assert hit.iloc[0] == 1.0 and hit.iloc[1] == 0.0 and hit.iloc[2] == 1.0


def test_benefit_forward_sign():
    r = series([10, 11, 12, 13])
    b = labels.benefit_fwd_bps(r, h=2)
    assert b.iloc[0] > 0  # курс потом выше — действовать сейчас было выгодно
    w = labels.wait_benefit_bps(r, k=1)
    assert abs(w.iloc[0] - 1000.0) < 1e-6


def test_local_min_label_with_tolerance():
    r = series([10, 9.5, 9.501, 10, 11])
    lab = labels.local_min_label(r, h=1, tol_bps=10)
    assert lab.iloc[1] == 1.0 and lab.iloc[2] == 1.0 and lab.iloc[3] == 0.0
