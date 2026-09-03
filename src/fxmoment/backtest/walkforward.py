"""Разбиение на окна walk-forward: расширяющееся обучение, тест по 6 месяцев, зазор 20 дней."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fxmoment.config import FIRST_TEST, PURGE_DAYS, TEST_MONTHS


@dataclass(frozen=True)
class Split:
    id: int
    train_end: pd.Timestamp  # последняя дата публикации в обучении
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def label(self) -> str:
        return f"{self.test_start:%Y-%m}…{self.test_end:%Y-%m}"


def make_splits(
    index: pd.DatetimeIndex,
    first_test: str = FIRST_TEST,
    test_months: int = TEST_MONTHS,
    purge_days: int = PURGE_DAYS,
) -> list[Split]:
    """Тестовые окна подряд от first_test до конца индекса (последнее может быть короче)."""
    splits: list[Split] = []
    test_start = pd.Timestamp(first_test)
    sid = 0
    while test_start <= index[-1]:
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), index[-1])
        pos = index.searchsorted(test_start)  # первая дата публикации ≥ test_start
        train_pos = pos - purge_days - 1
        if train_pos < 0:
            raise ValueError("недостаточно истории до первого тестового окна")
        splits.append(Split(sid, index[train_pos], test_start, test_end))
        sid += 1
        test_start = test_start + pd.DateOffset(months=test_months)
    return splits


def split_for_date(splits: list[Split], date: pd.Timestamp) -> Split:
    """Окно, чьи параметры действуют на дату (после последнего окна — последнее)."""
    date = pd.Timestamp(date)
    if date < splits[0].test_start:
        raise ValueError(f"дата {date.date()} раньше первого тестового окна {splits[0].test_start.date()}")
    for s in splits:
        if s.test_start <= date <= s.test_end:
            return s
    return splits[-1]
