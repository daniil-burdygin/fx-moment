import math

from fxmoment.pilot import clients_needed, sample_size_per_arm, weeks_to_power


def test_sample_size_matches_the_formula_in_the_plan():
    """n ≈ 15,7 · p(1 − p) / Δ² при мощности 80 % и α = 5 % — та же формула, что в pilot.md."""
    n = sample_size_per_arm(0.10, 0.02)
    assert n == math.ceil(2 * (1.96 + 0.842) ** 2 * 0.09 / 0.0004)
    assert 3500 < n < 3600


def test_segment_size_binds_and_no_duration_saves_a_small_one():
    """Сигнал коридорный: он уходит всем клиентам сразу, поэтому выборка задана размером сегмента,
    а не накапливается со временем. Маленький сегмент не спасает никакая длительность."""
    need = clients_needed(0.10, 0.02, delivery_rate=0.9)
    assert need > 2 * sample_size_per_arm(0.10, 0.02)  # недоставленные пуши поднимают порог
    assert weeks_to_power(need, 0.5, 0.9, 0.10, 0.02) < math.inf
    assert weeks_to_power(need - 1, 0.5, 0.9, 0.10, 0.02) == math.inf
    # длительность порога не сдвигает: год ожидания не добавит ни одного клиента
    assert weeks_to_power(need - 1, 0.5, 0.9, 0.10, 0.02, exposures=52) == math.inf


def test_weeks_are_waiting_for_signals_plus_the_outcome_window():
    # два сигнала по 0,5 в неделю — четыре недели, плюс неделя окна исхода
    assert weeks_to_power(10_000, 0.5, 0.9, 0.10, 0.02, exposures=2, outcome_days=7) == 5.0
    assert weeks_to_power(10_000, 0.5, 0.9, 0.10, 0.02, exposures=1) == 3.0
    # чаще сигнал — быстрее замер, но окно исхода не сжимается
    assert weeks_to_power(10_000, 2.0, 0.9, 0.10, 0.02, exposures=1) == 1.5
    assert weeks_to_power(10_000, 0.0, 0.9, 0.10, 0.02) == math.inf
