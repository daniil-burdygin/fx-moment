"""Профиль ряда: всё, что у движка привязано к шагу времени, в одном месте (ADR-0010).

Дневной профиль повторяет константы `config.py` буква в букву — это и есть проверка того, что
параметризация ничего не сдвинула. Внутридневной пересчитывает те же величины в барах.

Что НЕ входит в профиль и почему:

- **Полоса частоты 0,3–2,5 сигнала в неделю.** Она про канал, а не про данные: клиент получает
  пуши в календарных неделях независимо от того, дневной ряд под ними или часовой.
- **Допуск попадания 25 бп.** Он про цену исполнения, а не про шаг времени.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fxmoment.combine.policy import PolicyParams
from fxmoment.config import (
    ANALYSIS_START,
    CALIBRATION_H,
    CONTEXT,
    CORRIDORS,
    FIRST_TEST,
    HORIZONS,
    MIN_TEST_DAYS,
    PURGE_DAYS,
    TEST_MONTHS,
)
from fxmoment.indicators import ALL_INDICATORS, Dip, Indicator, Level, Momentum, Reversal

# Медиана баров в торговом дне по парам с реальным внутридневным рынком (CNY 11, KZT 9, AMD 9;
# замер 03.09.2026, `reports/intraday/moex_liquidity.csv`). Априорная оценка ADR-0010 была 8 —
# заменена измеренной, иначе «месяц» в барах разошёлся бы с месяцем в днях на восьмую часть.
BARS_PER_DAY = 9


@dataclass(frozen=True)
class Profile:
    """Шаг ряда и всё, что от него зависит. `step_scale` — сколько шагов этого ряда в одном дне
    публикации: он же множитель сеток индикаторов (`Indicator.scaled_grid`)."""

    name: str
    step_scale: int
    horizons: tuple[int, ...]
    calibration_h: int
    purge: int  # шагов ряда между концом обучения и началом теста
    min_test_steps: int  # окно короче не создаётся
    test_months: int
    first_test: str
    analysis_start: str
    corridors: tuple[str, ...]
    context: tuple[str, ...]
    indicators: tuple[type[Indicator], ...]
    policy: PolicyParams
    # шагов обучения, которые обязаны предшествовать первому тесту: разогрев самого длинного окна
    # сетки плюс зазор. Нужен там, где ряд начинается позже общего `first_test` (AMD с 2022-06).
    min_train_steps: int
    # «серия» в кучности: шагов ряда между соседними пушами, ближе которых поток считается серией
    series_gap: int = 3
    note: str = ""
    tolerances: tuple[float, ...] = field(default_factory=tuple)


DAILY = Profile(
    name="daily",
    step_scale=1,
    horizons=HORIZONS,
    calibration_h=CALIBRATION_H,
    purge=PURGE_DAYS,
    min_test_steps=MIN_TEST_DAYS,
    test_months=TEST_MONTHS,
    first_test=FIRST_TEST,
    analysis_start=ANALYSIS_START,
    corridors=CORRIDORS,
    context=CONTEXT,
    indicators=ALL_INDICATORS,
    policy=PolicyParams(),
    min_train_steps=250 + PURGE_DAYS,
    series_gap=3,
    note="дневной фиксинг ЦБ — база отчёта и защиты",
)

INTRADAY = Profile(
    name="intraday",
    step_scale=BARS_PER_DAY,
    # 1 бар — час, 4 — полдня, 9 — день, 45 — неделя, 180 — месяц. Первые два отвечают на вопрос
    # ADR-0010 (а): есть ли внутри дня момент, которого на дневном ряду не видно.
    horizons=(1, 4, 9, 45, 180),
    calibration_h=180,  # месяц: горизонт решения клиента тот же, ось другая
    purge=180,  # зазор равен максимальному горизонту, как на дневной оси
    min_test_steps=60 * BARS_PER_DAY,
    test_months=TEST_MONTHS,
    first_test=FIRST_TEST,
    analysis_start="2018-01-01",
    # только пары с внутридневным рынком (замер 03.09: у UZS два бара в день, у KGS четыре,
    # у TJS 17 баров за два года). CNY — не коридор, а эталон ликвидной пары.
    corridors=("KZT", "AMD", "CNY"),
    context=("CNY",),
    # сезонность выключена (ADR-0010 п. 2); ML выключен отдельно и по двум причинам: окна его
    # признаков заданы в днях внутри build_features, а его собственные шаговые параметры
    # (h = 10 у разметки локального минимума, gate_window = 120, rearm = 5) не объявлены
    # в STEP_PARAMS. Прогон на них был бы молчаливой методической ошибкой, а не переносом оси
    indicators=(Momentum, Level, Reversal, Dip),
    policy=PolicyParams(
        cooldown_days=3 * BARS_PER_DAY,
        max_per_window=2,
        window_days=5 * BARS_PER_DAY,
        conflict_window=10 * BARS_PER_DAY,
        storm_vol_window=20 * BARS_PER_DAY,
        storm_rank_window=250 * BARS_PER_DAY,
        storm_rank=0.95,
    ),
    min_train_steps=250 * BARS_PER_DAY + 180,
    series_gap=3 * BARS_PER_DAY,
    note="часовые свечи Мосбиржи — второй источник, не замена (ADR-0010)",
)

PROFILES: dict[str, Profile] = {p.name: p for p in (DAILY, INTRADAY)}


def first_test_for(index: pd.DatetimeIndex, profile: Profile) -> pd.Timestamp:
    """Начало первого тестового окна для ряда, который может начинаться позже общего `first_test`.

    Берётся первое начало месяца, которому предшествует не меньше `min_train_steps` шагов ряда:
    иначе окно теста опиралось бы на обучение короче самого длинного окна сетки, и калибровка
    выбирала бы параметры по разогреву. Возвращает не раньше `profile.first_test`."""
    want = pd.Timestamp(profile.first_test)
    if len(index) <= profile.min_train_steps:
        raise ValueError(f"ряд короче обучения: {len(index)} шагов при нужных {profile.min_train_steps}")
    earliest = index[profile.min_train_steps]
    start = max(want, earliest.normalize().replace(day=1) + pd.DateOffset(months=1))
    return pd.Timestamp(start)
