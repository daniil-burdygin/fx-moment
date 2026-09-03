"""Выгрузка официальных курсов ЦБ РФ (XML_dynamic.asp), открытые данные.

Запись ЦБ содержит дату ДЕЙСТВИЯ курса, номинал и курс за номинал. Нормализация на 1 единицу
и переход к дате публикации — в calendar.py (ADR-0002).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

import pandas as pd
import requests

from fxmoment.config import ALL_CURRENCIES, CBR_IDS

DYNAMIC_URL = "https://www.cbr.ru/scripts/XML_dynamic.asp"
VALFULL_URL = "https://www.cbr.ru/scripts/XML_valFull.asp"
RAW_COLUMNS = ["currency", "eff_date", "nominal", "value", "unit_rate"]


def _cbr_date(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _num(text: str | None) -> float:
    if text is None:
        raise ValueError("пустое числовое поле в ответе ЦБ")
    return float(text.replace(",", ".").replace("\xa0", "").replace(" ", ""))


def resolve_ids(session: requests.Session | None = None) -> dict[str, str]:
    """ISO-код → ID ЦБ по справочнику XML_valFull (для сверки статической таблицы)."""
    s = session or requests.Session()
    r = s.get(VALFULL_URL, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out: dict[str, str] = {}
    for item in root.iter("Item"):
        iso = (item.findtext("ISO_Char_Code") or "").strip()
        if iso and iso not in out:
            out[iso] = item.get("ID", "")
    return out


def fetch_dynamic(
    cbr_id: str, start: date, end: date, session: requests.Session | None = None
) -> pd.DataFrame:
    """Курсы одной валюты за период: eff_date, nominal, value, unit_rate (за 1 единицу)."""
    s = session or requests.Session()
    params = {"date_req1": _cbr_date(start), "date_req2": _cbr_date(end), "VAL_NM_RQ": cbr_id}
    r = s.get(DYNAMIC_URL, params=params, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    rows = []
    for rec in root.iter("Record"):
        eff = pd.to_datetime(rec.get("Date"), format="%d.%m.%Y")
        nominal = int(_num(rec.findtext("Nominal")))
        value = _num(rec.findtext("Value"))
        vunit_text = rec.findtext("VunitRate")
        unit = _num(vunit_text) if vunit_text else value / nominal
        rows.append((eff, nominal, value, unit))
    df = pd.DataFrame(rows, columns=["eff_date", "nominal", "value", "unit_rate"])
    if not df.empty and (abs(df["value"] / df["nominal"] - df["unit_rate"]) > 1e-9 * df["unit_rate"]).any():
        raise ValueError(f"{cbr_id}: VunitRate расходится с value/nominal")
    return df.sort_values("eff_date").reset_index(drop=True)


def fetch_all(
    start: date,
    end: date,
    currencies: tuple[str, ...] = ALL_CURRENCIES,
    session: requests.Session | None = None,
    verify_ids: bool = True,
) -> pd.DataFrame:
    """Длинная таблица по всем валютам: currency, eff_date, nominal, value, unit_rate."""
    s = session or requests.Session()
    if verify_ids:
        live = resolve_ids(s)
        for iso in currencies:
            if live.get(iso) and live[iso] != CBR_IDS[iso]:
                raise ValueError(f"код ЦБ для {iso} изменился: {CBR_IDS[iso]} → {live[iso]}")
    parts = []
    for iso in currencies:
        df = fetch_dynamic(CBR_IDS[iso], start, end, s)
        df.insert(0, "currency", iso)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out[RAW_COLUMNS]
