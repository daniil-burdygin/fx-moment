import numpy as np
import pandas as pd
import pytest

from fxmoment import evening
from fxmoment.config import BUY_NOW
from fxmoment.indicators.base import rolling_pct_rank


def make_forts(index: pd.DatetimeIndex, asset: str = "USD", drift: float = 0.0) -> pd.DataFrame:
    """Часовые бары фронт-контракта: по одному на каждый час 10:00–23:00 каждого дня панели.

    Цена растёт на `drift` за час, поэтому вечерний бар всегда выше опорного на известную
    величину — вечерний сдвиг проверяем точным числом, а не диапазоном."""
    rows = []
    for day_i, day in enumerate(index):
        for hour in range(10, 24):
            begin = day + pd.Timedelta(hours=hour)
            price = 100.0 + day_i + drift * hour
            rows.append(
                {
                    "asset": asset,
                    "secid": "SiZ0",
                    "expiry": day + pd.Timedelta(days=400),
                    "begin": begin,
                    "known_at": begin + pd.Timedelta(hours=1),
                    "end": begin + pd.Timedelta(minutes=59),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "unit_rate": price / 1000,
                }
            )
    return pd.DataFrame(rows)


def test_evening_bar_is_the_last_one_closing_by_21_msk(panel):
    bars = make_forts(panel.index[:5], drift=1.0)
    moves = evening.daily_moves(bars, "USD", anchor_hour=15)
    assert len(moves) == 5
    # опора — бар 15:00, вечерний — 20:00 (закрывается в 21:00), бары 21:00–23:00 не берутся
    assert (moves["evening_hour"] == evening.EVENING_LAST_HOUR).all()
    day0 = moves.iloc[0]
    assert day0["evening"] - day0["anchor"] == pytest.approx(5.0)


def test_gate_sees_only_bars_of_the_push_day(panel):
    """Ни один бар после даты T на её вечерний сдвиг не влияет — иначе это взгляд в будущее."""
    days = panel.index[:400]
    bars = make_forts(days, drift=0.5)
    base = evening.daily_moves(bars, "USD", anchor_hour=15)
    future = bars.copy()
    later = future["begin"] > days[200] + pd.Timedelta(hours=23)
    future.loc[later, ["open", "high", "low", "close"]] *= 5.0
    perturbed = evening.daily_moves(future, "USD", anchor_hour=15)
    upto_t = base.index <= days[200]
    pd.testing.assert_series_equal(base["delta"][upto_t], perturbed["delta"][upto_t])


def test_table_is_unchanged_by_bars_after_the_last_push(panel):
    bars = make_forts(panel.index, drift=0.3)
    decided = pd.DataFrame(
        {
            "date": panel.index[300:340],
            "corridor": "TJS",
            "decision": "sent",
            "push_scenario": BUY_NOW,
        }
    )
    table = evening.evening_gate_table(decided, panel, bars, anchor_hours=(15,))
    assert list(table.columns) == evening.COLUMNS
    cut = bars[bars["begin"] <= panel.index[340]]
    trimmed = evening.evening_gate_table(decided, panel, cut, anchor_hours=(15,))
    both = ["corridor", "group", "n", "gate_survival_T1", "hit_mean", "hit_mean_evening"]
    pd.testing.assert_frame_equal(table[both], trimmed[both])


def test_evening_rank_matches_rolling_rank_when_evening_equals_fixing(panel):
    rate = panel["TJS"].dropna()
    reference = rolling_pct_rank(rate, evening.GATE_WINDOW)
    for pos in (200, 500, 900):
        got = evening.evening_rank(rate, pos, float(rate.iloc[pos]))
        assert got == pytest.approx(float(reference.iloc[pos]))


def test_flat_evening_keeps_every_fact_and_a_rise_breaks_some(panel):
    days = panel.index
    decided = pd.DataFrame(
        {
            "date": days[300:600],
            "corridor": "TJS",
            "decision": "sent",
            "push_scenario": BUY_NOW,
        }
    )
    flat = evening.evening_gate_table(decided, panel, make_forts(days), anchor_hours=(15,))
    row = flat[(flat["corridor"] == "TJS") & (flat["group"] == evening.GROUP_FACT)].iloc[0]
    assert row["evening_fact_share"] == pytest.approx(1.0)
    rise = evening.evening_gate_table(decided, panel, make_forts(days, drift=2.0), anchor_hours=(15,))
    rrow = rise[(rise["corridor"] == "TJS") & (rise["group"] == evening.GROUP_FACT)].iloc[0]
    assert rrow["evening_fact_share"] < 1.0


def test_estimate_quality_is_perfect_when_the_move_is_the_fixing_change(panel):
    rate = panel["TJS"].dropna()
    moves = pd.DataFrame({"delta": (rate.shift(-1) / rate - 1)}, index=rate.index).dropna()
    q = evening.estimate_quality(rate, moves, since=str(rate.index[100].date()))
    assert q["est_corr"] == pytest.approx(1.0)
    assert q["est_mae_bps"] == pytest.approx(0.0, abs=1e-9)
    assert q["est_sign_share"] == pytest.approx(1.0)
    noise = moves.copy()
    noise["delta"] = np.zeros(len(noise))
    empty = evening.estimate_quality(rate, noise, since=str(rate.index[100].date()))
    assert np.isnan(empty["est_corr"])  # постоянная оценка не коррелирует ни с чем
