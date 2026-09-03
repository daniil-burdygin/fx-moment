"""Константы проекта. Определения — в CLAUDE.md и docs/decisions/."""

from __future__ import annotations

CORRIDORS: tuple[str, ...] = ("TJS", "UZS", "KGS", "AMD", "KZT")
CONTEXT: tuple[str, ...] = ("USD", "EUR", "CNY")
ALL_CURRENCIES: tuple[str, ...] = CORRIDORS + CONTEXT

# Коды валют ЦБ (параметр VAL_NM_RQ). При выгрузке сверяются с XML_valFull.
CBR_IDS: dict[str, str] = {
    "USD": "R01235",
    "EUR": "R01239",
    "CNY": "R01375",
    "KZT": "R01335",
    "TJS": "R01670",
    "UZS": "R01717",
    "KGS": "R01370",
    "AMD": "R01060",
}

HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)
FREQUENCY_BAND: tuple[float, float] = (1.0, 2.0)  # сигналов на коридор в неделю, итоговый поток
CALIBRATION_FREQ_RANGE: tuple[float, float] = (0.3, 2.5)  # допустимо для отдельного индикатора
MIN_CALIBRATION_EVENTS = 30  # событий на обучении, чтобы точка сетки считалась допустимой
# Медленный индикатор (сезонность: не чаще раза в месяц) переопределяет обе границы —
# Indicator.calibration_bounds(); месячный сигнал физически не даёт 0,3 в неделю (аудит 03.09).
TOLERANCES_BPS: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0)  # допуски в определении попадания
PRIMARY_TOL_BPS = 25.0  # рабочий допуск (ADR-0003)

RAW_START = "2015-01-01"
MOEX_RAW_START = "2018-01-01"  # снимок Мосбиржи (ADR-0010): год разогрева до первого окна теста
ANALYSIS_START = "2018-01-01"
FIRST_TEST = "2021-01-01"
TEST_MONTHS = 6
PURGE_DAYS = 20  # зазор между обучением и тестом, дней публикации (= максимальный горизонт)
MIN_TEST_DAYS = 60  # окно короче не создаётся: хвост в 46 дней весил в медианах как полугодие (аудит 03.09)
CALIBRATION_H = 20  # горизонт калибровки: месячное решение клиента (ADR-0004)

BUY_NOW = "BUY_NOW"
WINDOW_CLOSING = "WINDOW_CLOSING"
WATCH = "WATCH"

# Режим шоковой волатильности (ADR-0004): отчёт даётся с ним и без него
SHOCK_REGIME: tuple[str, str] = ("2022-02-24", "2022-07-31")

# Единицы показа в текстах: рублей за столько единиц валюты
DISPLAY_UNIT: dict[str, int] = {
    "TJS": 1,
    "KGS": 100,
    "KZT": 100,
    "AMD": 100,
    "UZS": 10000,
    "USD": 1,
    "EUR": 1,
    "CNY": 1,
}
CURRENCY_GENITIVE: dict[str, str] = {
    "TJS": "сомони",
    "UZS": "сума",
    "KGS": "сома",
    "AMD": "драма",
    "KZT": "тенге",
}
UNIT_LABEL: dict[str, str] = {
    "TJS": "1 сомони",
    "KGS": "100 сомов",
    "KZT": "100 тенге",
    "AMD": "100 драмов",
    "UZS": "10 000 сумов",
}
COUNTRY: dict[str, str] = {
    "TJS": "Таджикистан",
    "UZS": "Узбекистан",
    "KGS": "Кыргызстан",
    "AMD": "Армения",
    "KZT": "Казахстан",
}
MONTH_NAMES_PREP: dict[int, str] = {
    1: "январе",
    2: "феврале",
    3: "марте",
    4: "апреле",
    5: "мае",
    6: "июне",
    7: "июле",
    8: "августе",
    9: "сентябре",
    10: "октябре",
    11: "ноябре",
    12: "декабре",
}
