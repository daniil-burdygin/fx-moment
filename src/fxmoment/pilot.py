"""Калькулятор срока пилота (docs/product/pilot.md). Параметры подставляет банк.

Главное допущение, из которого следует всё остальное: **сигнал коридорный, а не клиентский**.
В день срабатывания он уходит всем клиентам коридора сразу, а холдаут закреплён по клиенту на весь
пилот. Значит выборка НЕ накапливается со временем: сколько клиентов в сегменте, столько и будет
после первого же сигнала. Время нужно на другое — дождаться сигналов и закрыть окно исхода.

Первая редакция считала иначе: делила нужное число клиентов на поток «клиенты × сигналы в неделю»,
то есть считала каждую пару «клиент × сигнал» независимым наблюдением и накапливала их. При
коридорном сигнале это завышает мощность в разы и прячет главный риск пилота — сегмент, который
мал по числу клиентов, не спасёт никакая длительность.
"""

from __future__ import annotations

import math

Z_ALPHA = {0.05: 1.96, 0.01: 2.576}
Z_POWER = {0.8: 0.842, 0.9: 1.282}


def sample_size_per_arm(p: float, delta: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Клиентов на группу для детекции прироста конверсии delta при базовой p (двусторонний тест)."""
    z = Z_ALPHA[alpha] + Z_POWER[power]
    return math.ceil(2 * (z**2) * p * (1 - p) / (delta**2))


def clients_needed(
    p: float,
    delta: float,
    delivery_rate: float,
    power: float = 0.8,
    alpha: float = 0.05,
    holdout_share: float = 0.5,
) -> int:
    """Сколько клиентов должно быть в коридоре, чтобы мощность вообще набралась.

    Группа сигнала теряет недоставленные пуши, холдаут не теряет ничего, поэтому связывает
    та из групп, которая меньше. Ниже этого порога пилот не наберёт мощность НИКОГДА — это
    ограничение сегмента, а не срока."""
    need = sample_size_per_arm(p, delta, power, alpha)
    treated = (1 - holdout_share) * delivery_rate
    control = holdout_share
    smaller = min(treated, control)
    return math.ceil(need / smaller) if smaller > 0 else 0


def weeks_to_power(
    n_clients: int,
    signals_per_week: float,
    delivery_rate: float,
    p: float,
    delta: float,
    power: float = 0.8,
    alpha: float = 0.05,
    holdout_share: float = 0.5,
    exposures: int = 1,
    outcome_days: int = 7,
) -> float:
    """Недель до готового замера: ожидание `exposures` сигналов плюс окно исхода.

    `inf`, если клиентов в коридоре не хватает: столько наблюдений не наберётся ни за какой срок.
    Сколько нужно — `clients_needed`."""
    if n_clients < clients_needed(p, delta, delivery_rate, power, alpha, holdout_share):
        return math.inf
    if signals_per_week <= 0:
        return math.inf
    return exposures / signals_per_week + outcome_days / 7
