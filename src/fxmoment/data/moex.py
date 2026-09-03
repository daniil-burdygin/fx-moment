"""Выгрузка часовых свечей валютных пар Мосбиржи (ISS, открытые данные) — ADR-0010.

Второй источник рядом с фиксингом ЦБ: биржевой курс живёт внутри дня, а фиксинг — одна точка.
Свеча котируется в рублях за номинал бумаги (`FACEVALUE`), поэтому здесь же нормализация на
1 единицу валюты — тот же `unit_rate`, что у ЦБ, иначе ряды несравнимы.

Анонимному доступу ISS отдаёт OHLC без объёмов (`value`, `volume` = null) — торговую активность
по этому источнику мерить нельзя, и она нигде не используется.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

ISS_ROOT = "https://iss.moex.com/iss"
BOARD_URL = f"{ISS_ROOT}/engines/currency/markets/selt/boards/CETS/securities"
DESCRIPTION_URL = f"{ISS_ROOT}/securities/{{sec}}.json"
PAGE_SIZE = 500  # ISS отдаёт свечи страницами; смещение — параметр start

# `interval` в ISS — КОД из перечисления, а не число минут: 60 — час, но 24 — день, 7 — неделя,
# 31 — месяц, 4 — квартал (замер `candleborders` 03.09.2026). Длительность свечи берётся отсюда,
# иначе для дневного кода ось `known_at` уехала бы на 24 минуты вперёд от начала свечи, то есть
# внутрь неё самой, — и «известно к моменту T» перестало бы быть правдой.
ISS_INTERVAL_LENGTH: dict[int, pd.Timedelta | pd.DateOffset] = {
    1: pd.Timedelta(minutes=1),
    10: pd.Timedelta(minutes=10),
    60: pd.Timedelta(hours=1),
    24: pd.Timedelta(days=1),
    7: pd.Timedelta(weeks=1),
    31: pd.DateOffset(months=1),
    4: pd.DateOffset(months=3),
}
RAW_COLUMNS = ["currency", "begin", "known_at", "end", "open", "high", "low", "close", "unit_rate"]

# Валюта → бумага режима CETS с расчётами «завтра» (TOM). USD и EUR на бирже не торгуются
# с 12.06.2024, поэтому внутридневного ряда по ним нет; CNY — контекст рублёвой стороны.
MOEX_SECURITIES: dict[str, str] = {
    "CNY": "CNYRUB_TOM",
    "KZT": "KZTRUB_TOM",
    "AMD": "AMDRUB_TOM",
    "UZS": "UZSRUB_TOM",
    "KGS": "KGSRUB_TOM",
    "TJS": "TJSRUB_TOM",  # торги прекращены 30.10.2024, ряд исторический
}

# Рублей за столько единиц валюты котируется бумага (поле FACEVALUE справочника ISS).
# Сверяется с живым справочником при выгрузке — как коды ЦБ в cbr.py.
MOEX_FACEVALUE: dict[str, int] = {
    "CNY": 1,
    "KZT": 100,
    "AMD": 100,
    "UZS": 10000,
    "KGS": 100,
    "TJS": 10,
}

# Первая часовая свеча по данным candleborders (замер 03.09.2026, ADR-0010).
FIRST_HOURLY_CANDLE: dict[str, str] = {
    "CNY": "2013-04-15",
    "KZT": "2018-12-18",
    "AMD": "2022-06-27",
    "UZS": "2022-09-12",
    "KGS": "2022-11-03",
    "TJS": "2022-11-07",
}


def resolve_facevalue(currency: str, session: requests.Session | None = None) -> int:
    """Номинал котировки бумаги из живого справочника ISS (для сверки статической таблицы)."""
    s = session or requests.Session()
    r = s.get(
        DESCRIPTION_URL.format(sec=MOEX_SECURITIES[currency]),
        params={"iss.only": "description"},
        timeout=30,
    )
    r.raise_for_status()
    block = r.json()["description"]
    name_at = block["columns"].index("name")
    value_at = block["columns"].index("value")
    rows = {row[name_at]: row[value_at] for row in block["data"]}
    if "FACEVALUE" not in rows:
        raise ValueError(f"{currency}: в справочнике ISS нет FACEVALUE")
    return int(float(rows["FACEVALUE"]))


def candle_borders(
    currency: str, interval: int = 60, session: requests.Session | None = None
) -> tuple[str, str]:
    """Границы доступной истории свечей выбранного интервала: (первая, последняя)."""
    s = session or requests.Session()
    r = s.get(
        f"{BOARD_URL}/{MOEX_SECURITIES[currency]}/candleborders.json",
        params={"iss.only": "borders"},
        timeout=30,
    )
    r.raise_for_status()
    block = r.json()["borders"]
    rows = [dict(zip(block["columns"], row, strict=True)) for row in block["data"]]
    hit = [row for row in rows if row["interval"] == interval]
    if not hit:
        raise ValueError(f"{currency}: интервал {interval} недоступен")
    return (str(hit[0]["begin"]), str(hit[0]["end"]))


def fetch_candles(
    currency: str,
    start: date,
    end: date,
    interval: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Свечи одной пары за период, постранично по `start`: begin, known_at, end, OHLC.

    `end` в ответе ISS — время ПОСЛЕДНЕЙ СДЕЛКИ внутри свечи (например 10:53:31), а не граница
    интервала, поэтому осью ряда он служить не может: шаг был бы неровным. `known_at` = begin +
    длительность интервала — момент, когда закрытие свечи заведомо известно; это и есть ось
    (см. to_bar_panel). Длительность берётся из `ISS_INTERVAL_LENGTH`, потому что `interval` —
    код, а не минуты.

    Страница короче `PAGE_SIZE` означает конец выборки; смещение растёт на число полученных строк,
    поэтому пропусков и дублей на границе страниц нет."""
    if interval not in ISS_INTERVAL_LENGTH:
        known = ", ".join(str(k) for k in sorted(ISS_INTERVAL_LENGTH))
        raise ValueError(f"неизвестный код интервала ISS {interval}; известные: {known}")
    s = session or requests.Session()
    url = f"{BOARD_URL}/{MOEX_SECURITIES[currency]}/candles.json"
    params = {
        "interval": interval,
        "from": start.isoformat(),
        "till": end.isoformat(),
        "iss.only": "candles",
    }
    rows: list[list] = []
    columns: list[str] = []
    offset = 0
    while True:
        r = s.get(url, params={**params, "start": offset}, timeout=60)
        r.raise_for_status()
        block = r.json()["candles"]
        columns = block["columns"]
        page = block["data"]
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += len(page)
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return pd.DataFrame(columns=["begin", "known_at", "end", "open", "high", "low", "close"])
    df = df[["begin", "end", "open", "high", "low", "close"]].copy()
    df["begin"] = pd.to_datetime(df["begin"])
    df["end"] = pd.to_datetime(df["end"])
    df.insert(1, "known_at", df["begin"] + ISS_INTERVAL_LENGTH[interval])
    df = df.drop_duplicates(subset="begin").sort_values("begin").reset_index(drop=True)
    return df


def fetch_all(
    start: date,
    end: date,
    currencies: tuple[str, ...] = tuple(MOEX_SECURITIES),
    interval: int = 60,
    session: requests.Session | None = None,
    verify_facevalue: bool = True,
) -> pd.DataFrame:
    """Длинная таблица по всем парам: currency, begin, end, OHLC и `unit_rate` = close / номинал."""
    s = session or requests.Session()
    parts = []
    for iso in currencies:
        if verify_facevalue:
            live = resolve_facevalue(iso, s)
            if live != MOEX_FACEVALUE[iso]:
                raise ValueError(f"номинал котировки {iso} изменился: {MOEX_FACEVALUE[iso]} → {live}")
        df = fetch_candles(iso, start, end, interval, s)
        if df.empty:
            continue
        df.insert(0, "currency", iso)
        df["unit_rate"] = df["close"] / MOEX_FACEVALUE[iso]
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.concat(parts, ignore_index=True)[RAW_COLUMNS]


def to_bar_panel(long_df: pd.DataFrame) -> pd.DataFrame:
    """Широкая таблица баров: индекс `known_at`, столбцы — валюты, значения `unit_rate`.

    Ось — момент, когда закрытие свечи стало известно (begin + интервал), а не `begin` и не `end`
    из ответа ISS: тот же смысл, что `pub_date` у ЦБ, — «данные, доступные на T». Срез по этой оси
    механически проверяем.

    Пропуски НЕ заполняются: у пар разные торговые часы (UZS и KGS торгуются до 15:00, CNY — до 19:00),
    и `ffill` создал бы серии нулевых изменений — ровно то, чего дневная ось избегает по построению.
    Каждый коридор считается на своей оси через `panel[corridor].dropna()`."""
    df = long_df.copy()
    df["known_at"] = pd.to_datetime(df["known_at"])
    panel = df.pivot(index="known_at", columns="currency", values="unit_rate").sort_index()
    panel.columns.name = None
    panel.index.name = "known_at"
    return panel
