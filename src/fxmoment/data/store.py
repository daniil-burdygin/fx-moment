"""Снимки сырых данных с датой выгрузки: воспроизводимость без сети.

Три источника, три пары «csv + meta»: дневной фиксинг ЦБ (`cbr_daily`), часовые свечи валютного
рынка Мосбиржи (`moex_hourly`, ADR-0010) и часовые свечи фронт-контрактов срочного рынка
(`moex_futures_hourly`) — единственный открытый ряд после 18:00 МСК. Метаданные пишутся всегда:
без даты выгрузки отчёт невоспроизводим."""

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
FORTS_CSV = RAW_DIR / "moex_futures_hourly.csv"
FORTS_META = RAW_DIR / "moex_futures_hourly.meta.json"


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
    """Снимок часовых свечей Мосбиржи (ADR-0010) рядом со снимком ЦБ.

    Путь снимка один, поэтому выгрузка с другим интервалом или по части валют затёрла бы полный
    часовой снимок молча. Метаданные пишут и код интервала, и бумагу каждой валюты — чтобы
    несовпадение было видно тому, кто снимок читает."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(MOEX_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
    from fxmoment.data.moex import ISS_INTERVAL_LENGTH, MOEX_FACEVALUE, MOEX_SECURITIES

    per_currency = {
        str(cur): {
            "security": MOEX_SECURITIES.get(str(cur)),
            "facevalue": MOEX_FACEVALUE.get(str(cur)),
            "bars": int(len(g)),
            "first_bar": str(g["begin"].min()),
            "last_bar": str(g["begin"].max()),
        }
        for cur, g in long_df.groupby("currency")
    }
    meta = {
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "interval_code": interval,  # код ISS, не минуты: 60 — час, 24 — день
        "interval_length": str(ISS_INTERVAL_LENGTH[interval]),
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


def save_forts_raw(long_df: pd.DataFrame, source: str, start: str, end: str, interval: int) -> None:
    """Снимок часовых свечей фронт-контрактов срочного рынка (Si, CR).

    В метаданных — не только границы ряда, но и перечень склеенных контрактов с их экспирациями:
    склейка фронта проверяема только вместе с ним."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(FORTS_CSV, index=False, date_format="%Y-%m-%d %H:%M:%S")
    from fxmoment.data.forts import FORTS_ASSETS
    from fxmoment.data.moex import ISS_INTERVAL_LENGTH

    per_asset = {
        str(asset): {
            "prefix": FORTS_ASSETS.get(str(asset), ("", 0))[0],
            "facevalue": FORTS_ASSETS.get(str(asset), ("", 0))[1],
            "bars": int(len(g)),
            "first_bar": str(g["begin"].min()),
            "last_bar": str(g["begin"].max()),
            "contracts": {
                str(sec): str(pd.Timestamp(exp).date())
                for sec, exp in g.groupby("secid")["expiry"].first().items()
            },
        }
        for asset, g in long_df.groupby("asset")
    }
    meta = {
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "refetch": "fxmoment fetch-forts --start <ISO> (ISS, открытые данные, ключа не требует)",
        "interval_code": interval,  # код ISS, не минуты: 60 — час, 24 — день
        "interval_length": str(ISS_INTERVAL_LENGTH[interval]),
        "requested_start": start,
        "requested_end": end,
        "rows": int(len(long_df)),
        "assets": per_asset,
    }
    FORTS_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_forts_raw() -> pd.DataFrame:
    if not FORTS_CSV.exists():
        raise FileNotFoundError(f"нет снимка {FORTS_CSV}: выполните `fxmoment fetch-forts`")
    return pd.read_csv(FORTS_CSV, parse_dates=["expiry", "begin", "known_at", "end"])


def load_forts_meta() -> dict:
    return json.loads(FORTS_META.read_text(encoding="utf-8")) if FORTS_META.exists() else {}


def load_forts_bar_panel() -> pd.DataFrame:
    from fxmoment.data.forts import to_bar_panel

    return to_bar_panel(load_forts_raw())
