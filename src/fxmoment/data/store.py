"""Снимки сырых данных с датой выгрузки: воспроизводимость без сети.

Два источника, две пары «csv + meta»: дневной фиксинг ЦБ (`cbr_daily`) и часовые свечи
Мосбиржи (`moex_hourly`, ADR-0010). Метаданные пишутся всегда: без даты выгрузки отчёт
невоспроизводим."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fxmoment.data.calendar import to_publication_panel


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


RAW_DIR = repo_root() / "data" / "raw"
RAW_CSV = RAW_DIR / "cbr_daily.csv"
RAW_META = RAW_DIR / "cbr_daily.meta.json"
MOEX_CSV = RAW_DIR / "moex_hourly.csv"
MOEX_META = RAW_DIR / "moex_hourly.meta.json"


def save_raw(long_df: pd.DataFrame, source: str, start: str, end: str) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(RAW_CSV, index=False, date_format="%Y-%m-%d")
    meta = {
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "requested_start": start,
        "requested_end": end,
        "currencies": sorted(long_df["currency"].unique().tolist()),
        "rows": int(len(long_df)),
        "first_eff_date": str(long_df["eff_date"].min().date()),
        "last_eff_date": str(long_df["eff_date"].max().date()),
    }
    RAW_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"нет снимка {RAW_CSV}: выполните `fxmoment fetch`")
    return pd.read_csv(RAW_CSV, parse_dates=["eff_date"])


def load_meta() -> dict:
    return json.loads(RAW_META.read_text(encoding="utf-8")) if RAW_META.exists() else {}


def load_panel() -> pd.DataFrame:
    return to_publication_panel(load_raw())


def save_moex_raw(long_df: pd.DataFrame, source: str, start: str, end: str, interval: int) -> None:
    """Снимок часовых свечей Мосбиржи (ADR-0010) рядом со снимком ЦБ."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(MOEX_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
    per_currency = {
        str(cur): {
            "bars": int(len(g)),
            "first_bar": str(g["begin"].min()),
            "last_bar": str(g["begin"].max()),
        }
        for cur, g in long_df.groupby("currency")
    }
    meta = {
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "interval_minutes": interval,
        "requested_start": start,
        "requested_end": end,
        "rows": int(len(long_df)),
        "currencies": per_currency,
    }
    MOEX_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_moex_raw() -> pd.DataFrame:
    if not MOEX_CSV.exists():
        raise FileNotFoundError(f"нет снимка {MOEX_CSV}: выполните `fxmoment fetch-moex`")
    return pd.read_csv(MOEX_CSV, parse_dates=["begin", "known_at", "end"])


def load_moex_meta() -> dict:
    return json.loads(MOEX_META.read_text(encoding="utf-8")) if MOEX_META.exists() else {}


def load_bar_panel() -> pd.DataFrame:
    from fxmoment.data.moex import to_bar_panel

    return to_bar_panel(load_moex_raw())
