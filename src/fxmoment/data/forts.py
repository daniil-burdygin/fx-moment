"""Часовые свечи квартальных фьючерсов Мосбиржи (ISS, рынок `forts`) — вечерняя сессия.

Зачем отдельно от `moex.py`: валютный рынок CETS закрывается раньше вечера (в снимке
`moex_hourly.csv` баров после 18:00 нет), а решение о пуше принимается вечером T, когда
фиксинг ЦБ уже опубликован, а рынок ещё идёт — до 23:50 на срочном. Фьючерс на доллар (Si)
и на юань (CR) — единственный открытый ряд, покрывающий это окно.

Контракт квартальный: март, июнь, сентябрь, декабрь; SECID = префикс актива + буква месяца
(H, M, U, Z) + последняя цифра года. Цифра повторяется раз в десять лет, поэтому год берётся
не из кода, а из `LSTTRADE` справочника ISS и сверяется с ожидаемым — иначе ряд 2021 года
молча склеился бы с рядом 2011-го.

Ряд склеивается по фронт-контракту: на дату d действует ближайший контракт, чья последняя
дата торгов строго позже d (в день экспирации ликвидность уже в следующем). Свечи каждого
контракта тянутся только за его окно фронта, поэтому склейка не зависит от того, сколько
контракт торговался до того, как стал ближним.

Абсолютный уровень фьючерса нам не нужен: используется только относительное изменение внутри
одного дня, а оно берётся по одному и тому же контракту, поэтому скачок на ролле в расчёт
не попадает. `unit_rate` пишется в снимок для читаемости — рублей за 1 единицу валюты.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from fxmoment.data.moex import ISS_INTERVAL_LENGTH, ISS_ROOT, PAGE_SIZE

FORTS_URL = f"{ISS_ROOT}/engines/futures/markets/forts/securities"
DESCRIPTION_URL = f"{ISS_ROOT}/securities/{{sec}}.json"

# Буква месяца исполнения → месяц. Квартальные серии Мосбиржи: третий четверг месяца.
MONTH_CODES: dict[str, int] = {"H": 3, "M": 6, "U": 9, "Z": 12}

# Валюта → (префикс SECID, рублей за столько единиц валюты котируется контракт).
# Si — 1000 USD в контракте, цена в рублях за лот; CR — 1000 CNY, цена в рублях за 1 юань.
FORTS_ASSETS: dict[str, tuple[str, int]] = {"USD": ("Si", 1000), "CNY": ("CR", 1)}

RAW_COLUMNS = [
    "asset",
    "secid",
    "expiry",
    "begin",
    "known_at",
    "end",
    "open",
    "high",
    "low",
    "close",
    "unit_rate",
]


def contract_codes(asset: str, start: date, end: date) -> list[tuple[str, int, int]]:
    """Коды квартальных контрактов, чья экспирация попадает в [start, end] с запасом квартала:
    (SECID, год, месяц). Год ожидаемый — по нему сверяется `LSTTRADE`."""
    prefix, _ = FORTS_ASSETS[asset]
    out: list[tuple[str, int, int]] = []
    for year in range(start.year, end.year + 2):
        for letter, month in sorted(MONTH_CODES.items(), key=lambda kv: kv[1]):
            out.append((f"{prefix}{letter}{year % 10}", year, month))
    return sorted(out, key=lambda t: (t[1], t[2]))


def contract_expiry(secid: str, expected_year: int, session: requests.Session | None = None) -> date | None:
    """Дата последних торгов контракта из справочника ISS; None — контракта не существует.

    Год сверяется с ожидаемым: SECID повторяется каждые десять лет, и ISS на запрос `SiH1`
    отдаёт один контракт, а какой именно — видно только по `LSTTRADE`."""
    s = session or requests.Session()
    r = s.get(DESCRIPTION_URL.format(sec=secid), params={"iss.only": "description"}, timeout=30)
    r.raise_for_status()
    block = r.json()["description"]
    rows = {row[0]: row[2] for row in block["data"]}
    raw = rows.get("LSTTRADE")
    if not raw:
        return None
    last = date.fromisoformat(str(raw))
    return last if last.year == expected_year else None


def front_windows(
    asset: str, start: date, end: date, session: requests.Session | None = None
) -> list[tuple[str, date, date, date]]:
    """Окна фронт-контракта, покрывающие [start, end]: (SECID, экспирация, с, по).

    Контракт — фронт с дня, следующего за экспирацией предыдущего, по свой день экспирации
    включительно; на саму дату экспирации фронтом считается уже следующий (см. `front_series`).
    Несуществующие коды (актив ещё не торговался) пропускаются: у CNY-фьючерса история
    начинается позже, чем у долларового, и выдумывать её нечем."""
    s = session or requests.Session()
    known: list[tuple[str, date]] = []
    for secid, year, _ in contract_codes(asset, start, end):
        expiry = contract_expiry(secid, year, s)
        if expiry is not None:
            known.append((secid, expiry))
    known.sort(key=lambda t: t[1])
    windows: list[tuple[str, date, date, date]] = []
    previous: date | None = None
    for secid, expiry in known:
        window_start = start if previous is None else previous
        if expiry < start:
            previous = expiry
            continue
        windows.append((secid, expiry, window_start, min(expiry, end)))
        previous = expiry
        if expiry >= end:
            break
    return windows


def fetch_candles(
    secid: str,
    start: date,
    end: date,
    interval: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Свечи одного контракта за период, постранично по `start` (ISS отдаёт по 500).

    `known_at` = begin + длительность интервала — момент, когда закрытие свечи заведомо
    известно; та же ось, что у валютных свечей в `moex.py`."""
    if interval not in ISS_INTERVAL_LENGTH:
        raise ValueError(f"неизвестный код интервала ISS {interval}")
    s = session or requests.Session()
    url = f"{FORTS_URL}/{secid}/candles.json"
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
    return df.drop_duplicates(subset="begin").sort_values("begin").reset_index(drop=True)


def fetch_front_series(
    asset: str,
    start: date,
    end: date,
    interval: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Склеенный ряд фронт-контракта одного актива за период: длинная таблица RAW_COLUMNS."""
    s = session or requests.Session()
    _, facevalue = FORTS_ASSETS[asset]
    parts: list[pd.DataFrame] = []
    for secid, expiry, window_start, window_end in front_windows(asset, start, end, s):
        df = fetch_candles(secid, window_start, window_end, interval, s)
        if df.empty:
            continue
        # Окно фронта — [экспирация предыдущего, своя экспирация): в день своей экспирации
        # контракт уже не фронт, и этот день закрывает следующий — так между окнами нет ни
        # разрыва, ни нахлёста.
        day = df["begin"].dt.date
        df = df[(day >= window_start) & (day < expiry)]
        if df.empty:
            continue
        df.insert(0, "asset", asset)
        df.insert(1, "secid", secid)
        df.insert(2, "expiry", pd.Timestamp(expiry))
        df["unit_rate"] = df["close"] / facevalue
        parts.append(df)
    if not parts:
        return pd.DataFrame(columns=RAW_COLUMNS)
    out = pd.concat(parts, ignore_index=True)[RAW_COLUMNS]
    return out.drop_duplicates(subset=["asset", "begin"], keep="first").reset_index(drop=True)


def fetch_all(
    start: date,
    end: date,
    assets: tuple[str, ...] = tuple(FORTS_ASSETS),
    interval: int = 60,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    s = session or requests.Session()
    parts = [fetch_front_series(a, start, end, interval, s) for a in assets]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def to_bar_panel(long_df: pd.DataFrame) -> pd.DataFrame:
    """Широкая таблица: индекс `known_at`, столбцы — активы, значения `unit_rate`.

    Пропуски не заполняются — как и в валютной панели: у активов разные торговые часы,
    а `ffill` создал бы серии нулевых изменений."""
    df = long_df.copy()
    df["known_at"] = pd.to_datetime(df["known_at"])
    panel = df.pivot(index="known_at", columns="asset", values="unit_rate").sort_index()
    panel.columns.name = None
    panel.index.name = "known_at"
    return panel
