"""Окно клиента (client_window.py): окна строятся в оси публикации, стратегии выбирают день внутри
окна, оракул не хуже любой стратегии, пуш совпадает с оракулом там, где пуш пришёл в лучший день,
а порча ряда после окна ничего в окне не меняет."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment import client_window as cw
from fxmoment.backtest.walkforward import Split
from fxmoment.config import BUY_NOW


def _panel(n: int = 900, seed: int = 3) -> pd.DataFrame:
    """Курс внутри каждого месяца падает к концу: лучший день окна — последний, привычка — худший."""
    idx = pd.bdate_range("2018-01-01", periods=n, name="pub_date")
    rng = np.random.default_rng(seed)
    base = 10 + 0.01 * np.arange(n)  # медленный рост, чтобы окна не сравнивались с прошлым годом
    intra = -0.05 * (idx.day.to_numpy() - 1)  # −5 бп за каждый день месяца: к концу дешевле
    rate = base + intra + rng.normal(0, 0.002, n)
    return pd.DataFrame({"TJS": rate, "KZT": rate * 0.02 + rng.normal(0, 1e-5, n)}, index=idx)


def _splits(idx: pd.DatetimeIndex) -> list[Split]:
    return [Split(0, idx[300], idx[320], idx[-1])]


def test_windows_axis_and_strategies_inside_window() -> None:
    panel = _panel()
    idx = panel.index
    persona = cw.Persona(10, wait_days=5)
    wins = cw.client_windows(idx, persona, idx[320], idx[-1])
    assert wins and all(last - pos == 5 for pos, last in wins)
    assert all(
        idx[pos].day >= 10 and (idx[pos - 1].day < 10 or idx[pos - 1].month != idx[pos].month)
        for pos, _ in wins
    )
    table = cw.window_table(panel, _splits(idx), None, corridors=("TJS",), persona_list=[persona])
    assert (
        (table["n_days"] == 6).all()
        and (table["day_habit"] == 0).all()
        and (table["day_deadline"] == 5).all()
    )
    for name in cw.STRATEGIES:
        if name == "random":
            continue
        assert table[f"day_{name}"].between(0, 5).all()
        assert (table["rate_oracle"] <= table[f"rate_{name}"] + 1e-12).all()
    # курс падает к концу месяца → оракул почти всегда срок, привычка — худший день
    assert (table["day_oracle"] == 5).mean() > 0.9
    assert (table["rate_deadline"] < table["rate_habit"]).mean() > 0.9
    # без решений политики пуша нет — стратегия пуша сводится к сроку
    assert (~table["has_push"]).all() and table["rate_push"].equals(table["rate_deadline"])


def test_push_at_best_day_matches_oracle_and_summary_shape() -> None:
    panel = _panel()
    idx = panel.index
    persona = cw.Persona(5, wait_days=10)
    wins = cw.client_windows(idx, persona, idx[320], idx[-1])
    # пуш ровно в лучший день каждого второго окна
    rows = []
    for n, (pos, last) in enumerate(wins):
        if n % 2 == 0:
            r = panel["TJS"].to_numpy()[pos : last + 1]
            rows.append(
                {
                    "date": idx[pos + int(np.argmin(r))],
                    "corridor": "TJS",
                    "decision": "sent",
                    "push_scenario": BUY_NOW,
                }
            )
    rows.append({"date": idx[wins[1][0]], "corridor": "TJS", "decision": "muted", "push_scenario": None})
    decisions = pd.DataFrame(rows)
    table = cw.window_table(panel, _splits(idx), decisions, corridors=("TJS",), persona_list=[persona])
    with_push = table[table["has_push"]]
    assert len(with_push) == (len(wins) + 1) // 2
    assert np.allclose(with_push["rate_push"], with_push["rate_oracle"])
    assert table[~table["has_push"]]["rate_push"].equals(table[~table["has_push"]]["rate_deadline"])
    summary = cw.summarize(table, reps=200)
    assert set(summary["strategy"]) == set(cw.STRATEGIES)
    assert set(summary["corridor"]) == {"TJS", "all"} and set(summary["persona"]) == {persona.label, "all"}
    assert set(summary["family"]) == {"wait"}
    oracle = summary[(summary["strategy"] == "oracle") & (summary["corridor"] == "TJS")].iloc[0]
    push = summary[(summary["strategy"] == "push") & (summary["corridor"] == "TJS")].iloc[0]
    habit = summary[(summary["strategy"] == "habit") & (summary["corridor"] == "TJS")].iloc[0]
    assert abs(oracle["oracle_share"] - 1.0) < 1e-9 and oracle["regret_share"] == 0.0
    assert habit["benefit_vs_habit_bps"] == 0.0 and habit["share_shifted"] == 0.0
    assert 0.4 < push["share_shifted"] < 0.6 and push["benefit_vs_habit_bps"] > 0
    assert push["benefit_vs_habit_ci_lo"] <= push["benefit_vs_habit_bps"] <= push["benefit_vs_habit_ci_hi"]
    assert abs(push["rub_per_transfer"] - push["benefit_vs_habit_bps"] / 1e4 * cw.TRANSFER_RUB) < 1e-9


def test_future_beyond_window_does_not_change_window_rows() -> None:
    panel = _panel()
    idx = panel.index
    persona = cw.Persona(15, wait_days=5)
    splits = _splits(idx)
    a = cw.window_table(panel, splits, None, corridors=("TJS", "KZT"), persona_list=[persona])
    cut = a["end"].iloc[len(a) // 2]
    spoiled = panel.copy()
    spoiled.loc[spoiled.index > cut, :] *= 0.5  # обвал после окна
    b = cw.window_table(spoiled, splits, None, corridors=("TJS", "KZT"), persona_list=[persona])
    cols = [c for c in a.columns if c.startswith(("rate_", "day_", "has_"))]
    pd.testing.assert_frame_equal(
        a[a["end"] <= cut][cols].reset_index(drop=True), b[b["end"] <= cut][cols].reset_index(drop=True)
    )


def test_screen_flag_is_causal_and_bounded() -> None:
    rate = _panel()["TJS"]
    flag = cw.screen_flag(rate)
    assert flag.dtype == bool and not flag.iloc[: cw.SCREEN_WINDOW - 1].any()
    head = cw.screen_flag(rate.iloc[:500])
    assert head.equals(flag.iloc[:500])


def test_accelerate_family_habit_is_window_end_and_push_only_brings_forward() -> None:
    panel = _panel()
    idx = panel.index
    persona = cw.Persona(5, habit_day=20)
    assert persona.family == "accelerate" and persona.label == "a05_h20"
    wins = cw.client_windows(idx, persona, idx[320], idx[-1])
    assert wins and all(
        idx[pos].day >= 5 and idx[last].day >= 20 and idx[last].month == idx[pos].month for pos, last in wins
    )
    # пуш в лучший день каждого окна
    rows = []
    for pos, last in wins:
        r = panel["TJS"].to_numpy()[pos : last + 1]
        rows.append(
            {
                "date": idx[pos + int(np.argmin(r))],
                "corridor": "TJS",
                "decision": "sent",
                "push_scenario": BUY_NOW,
            }
        )
    table = cw.window_table(
        panel, _splits(idx), pd.DataFrame(rows), corridors=("TJS",), persona_list=[persona]
    )
    assert (table["family"] == "accelerate").all()
    assert (table["day_habit"] == table["n_days"] - 1).all()  # привычка — последний день окна
    assert np.allclose(table["rate_push"], table["rate_oracle"])
    # без пуша — привычный день, а не ожидание: стратегия пуша равна привычке
    plain = cw.window_table(panel, _splits(idx), None, corridors=("TJS",), persona_list=[persona])
    assert plain["rate_push"].equals(plain["rate_habit"]) and (~plain["has_push"]).all()
    summary = cw.summarize(plain, reps=100)
    push = summary[(summary["strategy"] == "push") & (summary["corridor"] == "TJS")].iloc[0]
    assert (
        push["share_shifted"] == 0.0 and push["benefit_vs_habit_bps"] == 0.0 and push["regret_share"] == 0.0
    )
    import pytest

    with pytest.raises(ValueError):
        cw.Persona(20, habit_day=10)
    with pytest.raises(ValueError):
        cw.Persona(5)
