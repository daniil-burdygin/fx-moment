"""Калькулятор срока пилота (docs/product/pilot.md). Параметры подставляет банк."""

from __future__ import annotations

import math


def sample_size_per_arm(p: float, delta: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Клиентов на группу для детекции прироста конверсии delta при базовой p (двусторонний тест)."""
    z = {0.05: 1.96, 0.01: 2.576}[alpha] + {0.8: 0.842, 0.9: 1.282}[power]
    return math.ceil(2 * (z**2) * p * (1 - p) / (delta**2))


def weeks_to_power(
    n_clients: int,
    signals_per_week: float,
    delivery_rate: float,
    p: float,
    delta: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Недель до набора выборки при половине клиентов в холдауте."""
    need = 2 * sample_size_per_arm(p, delta, power, alpha)
    events_per_week = n_clients * signals_per_week * delivery_rate
    return need / events_per_week if events_per_week > 0 else math.inf
