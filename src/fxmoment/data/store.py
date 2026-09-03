"""Снимок сырых данных ЦБ с датой выгрузки: воспроизводимость без сети."""

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
