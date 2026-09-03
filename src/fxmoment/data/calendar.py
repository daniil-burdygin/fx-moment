"""Ось времени проекта — дата публикации ЦБ (ADR-0002).

pub_date = eff_date − 1 календарный день. Ряд состоит из дней публикации (рабочих дней ЦБ),
поэтому серий нулевых изменений из-за выходных нет по построению.
"""

from __future__ import annotations

import pandas as pd


def to_publication_panel(long_df: pd.DataFrame) -> pd.DataFrame:
    """Широкая таблица: индекс pub_date, столбцы — валюты, значения — курс за 1 единицу."""
    df = long_df.copy()
    df["eff_date"] = pd.to_datetime(df["eff_date"])
    df["pub_date"] = df["eff_date"] - pd.Timedelta(days=1)
    panel = df.pivot(index="pub_date", columns="currency", values="unit_rate").sort_index()
    panel.columns.name = None
    panel.index.name = "pub_date"
    # У разных валют один календарь ЦБ; редкие расхождения закрываем переносом последнего курса.
    return panel.ffill()


def as_of(panel: pd.DataFrame, cutoff: str | pd.Timestamp) -> pd.DataFrame:
    """Данные, доступные на дату среза: записи с pub_date ≤ cutoff."""
    return panel.loc[: pd.Timestamp(cutoff)]
