"""Библиотека текстов пушей и чекер запрещённых формулировок (ADR-0007, docs/product/texts.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fxmoment.config import (
    BUY_NOW,
    CURRENCY_GENITIVE,
    DISPLAY_UNIT,
    MONTH_NAMES_PREP,
    UNIT_LABEL,
    WINDOW_CLOSING,
)


@dataclass(frozen=True)
class Template:
    scenario: str
    indicator: str
    title: str
    body: str
    button: str = "Перевести"


TEMPLATES: dict[tuple[str, str], Template] = {
    (BUY_NOW, "level"): Template(
        BUY_NOW,
        "level",
        "Курс {cur}: один из низких за {months} мес.",
        "Сейчас {rate} ₽ за {unit} — ниже, чем в {pct} % дней за последние {months} мес.",
    ),
    (BUY_NOW, "level_stall"): Template(
        BUY_NOW,
        "level_stall",
        "Курс {cur} остановился на низком уровне",
        "{rate} ₽ за {unit} — ниже, чем в {pct} % дней за {months} мес. "
        "Снижение остановилось {days} дн. назад.",
    ),
    (BUY_NOW, "momentum"): Template(
        BUY_NOW,
        "momentum",
        "Курс {cur} снижается {n}-й день подряд",
        "Сейчас {rate} ₽ за {unit}, за {n} дн. — минус {drop} %.",
    ),
    (WINDOW_CLOSING, "reversal"): Template(
        WINDOW_CLOSING,
        "reversal",
        "Курс {cur} пошёл вверх",
        "+{rise} % от минимума за {months} мес.: было {min_rate} ₽, сейчас {rate} ₽ за {unit}.",
    ),
    (BUY_NOW, "seasonality"): Template(
        BUY_NOW,
        "seasonality",
        "В {month} курс {cur} обычно выше",
        "В {k} из {n} последних лет курс {cur} в {month} был выше, чем месяцем раньше. "
        "Сейчас {rate} ₽ за {unit}.",
    ),
    (BUY_NOW, "dip_vs_trend"): Template(
        BUY_NOW,
        "dip_vs_trend",
        "Курс {cur} ниже своего среднего за {span_weeks} нед.",
        "Сейчас {rate} ₽ за {unit} — на {dev} % ниже среднего за {span_weeks} нед.; "
        "такой провал был реже, чем в {pct} % дней за {months} мес.",
    ),
    (BUY_NOW, "ml_localmin"): Template(
        BUY_NOW,
        "ml_localmin",
        "Курс {cur} на низком уровне",
        "Сейчас {rate} ₽ за {unit} — ниже, чем в {pct} % дней за последние {months} мес.",
    ),
}

# Запрещённые обороты: шаблон → почему. Проверяются без учёта регистра.
FORBIDDEN: tuple[tuple[str, str], ...] = (
    (r"\b(скоро|вот-вот)\b", "утверждение о будущем"),
    (
        r"\b(вырастет|упадёт|подорожает|подешевеет|укрепится|ослабнет|будет расти|будет падать)\b",
        "прогноз курса",
    ),
    (r"\bуспе(й|йте|ть|ете)\b", "призыв с намёком на будущее движение"),
    (r"\bне упусти", "призыв с намёком на будущее"),
    (r"\bпока не (поздно|подорожал)", "намёк на будущее подорожание"),
    (r"\bсейчас или никогда\b", "давление, намёк на будущее"),
    (r"\bгарантир", "гарантия"),
    (r"\b(самый|наи)?лучш(ий|ая|ее) курс", "непроверяемое превосходство"),
    (r"\b(заработа|доходност|инвестиц|прибыл)", "инвестиционная лексика"),
    (r"\b(рекоменду|совету|стоит перевести|лучше перевести)", "совет вместо факта"),
    (r"\b(идеальн|как никогда|лучше не будет)", "оценка будущего без измерения"),
    (r"\b(теряете|потеряете|каждый день ожидания)", "утверждение об убытке в будущем"),
    (r"\bвыгодн(о|ый|ая|ее)\b(?!.*\d+ ?%)", "«выгодно» без измерения (нужны процент и окно)"),
)


def check_text(text: str) -> list[tuple[str, str]]:
    """Список (фрагмент, причина) для каждого нарушения; пустой список — чисто."""
    hits = []
    low = text.lower()
    for pattern, reason in FORBIDDEN:
        m = re.search(pattern, low)
        if m:
            hits.append((m.group(0), reason))
    return hits


def _fmt_rate(value: float, corridor: str) -> str:
    unit = DISPLAY_UNIT.get(corridor, 1)
    v = value * unit
    digits = 2 if v >= 1 else 4
    return f"{v:,.{digits}f}".replace(",", " ").replace(".", ",")


def render(corridor: str, scenario: str, indicator: str, rate: float, facts: dict) -> tuple[str, str]:
    """Заголовок и текст пуша из факта индикатора. Числа — из факта, никаких оценок будущего."""
    key = (scenario, indicator)
    if indicator == "level" and facts.get("days_since_min", 0) and facts.get("days_since_min", 0) >= 3:
        key = (BUY_NOW, "level_stall")
    t = TEMPLATES[key]
    cur = CURRENCY_GENITIVE.get(corridor, corridor)
    window = float(facts.get("window", 120) or 120)
    months = max(1, round(window / 21))
    values = {
        "cur": cur,
        "unit": UNIT_LABEL.get(corridor, corridor),
        "rate": _fmt_rate(rate, corridor),
        "months": months,
        "pct": round((1 - float(facts.get("pct_rank", 0.1) or 0.1)) * 100),
        "days": int(facts.get("days_since_min", 0) or 0),
        "n": int(facts.get("streak", 0) or 0),
        "drop": f"{abs(float(facts.get('drop_pct', 0) or 0)):.1f}".replace(".", ","),
        "rise": f"{float(facts.get('rise_pct', 0) or 0):.1f}".replace(".", ","),
        "min_rate": _fmt_rate(float(facts.get("min_rate", rate) or rate), corridor),
        "k": int(facts.get("k_years", 0) or 0),
        "n_years": int(facts.get("n_years", 0) or 0),
        "month": MONTH_NAMES_PREP.get(int(facts.get("target_month", 1) or 1), ""),
        "dev": f"{abs(float(facts.get('dev_pct', 0) or 0)):.1f}".replace(".", ","),
        "span_weeks": max(1, round(float(facts.get("span", 40) or 40) / 5)),
    }
    if indicator == "seasonality":
        values["n"] = values["n_years"]
    return t.title.format(**values), t.body.format(**values)


def library_texts() -> list[tuple[str, str, str]]:
    """Все шаблоны, отрендеренные с образцовыми значениями, для прогона чекера."""
    sample = {
        "pct_rank": 0.12,
        "window": 120,
        "days_since_min": 4,
        "streak": 4,
        "drop_pct": -1.3,
        "rise_pct": 0.8,
        "min_rate": 9.05,
        "k_years": 7,
        "n_years": 9,
        "target_month": 12,
        "dev_pct": -1.4,
        "span": 40,
    }
    out = []
    for (scenario, indicator), _t in TEMPLATES.items():
        ind = "level" if indicator == "level_stall" else indicator
        facts = dict(sample)
        if indicator == "level":
            facts["days_since_min"] = 0
        title, body = render("TJS", scenario, ind, 9.12, facts)
        out.append((f"{scenario}/{indicator}", title, body))
    return out
