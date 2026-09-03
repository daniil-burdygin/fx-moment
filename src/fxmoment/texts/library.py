"""Библиотека текстов пушей и чекер запрещённых формулировок (ADR-0007, docs/product/texts.md)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fxmoment.config import (
    BUY_NOW,
    CURRENCY_GENITIVE,
    DISPLAY_UNIT,
    MONTH_NAMES_ACC,
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
        "В {month} курс {cur} в среднем был выше в {k} из {n} последних лет",
        "Сейчас {rate} ₽ за {unit}. В {k} из {n} последних лет средний курс {cur} за {month_acc} был выше "
        "среднего за предыдущий месяц.",
    ),
    (BUY_NOW, "dip_vs_trend"): Template(
        BUY_NOW,
        "dip_vs_trend",
        "Курс {cur} ниже своего среднего за {span} рабочих дней",
        "Сейчас {rate} ₽ за {unit} — на {dev} % ниже среднего за последние {span} рабочих дней, "
        "включая сегодняшний; такой провал был реже, чем в {pct} % дней за {months} мес.",
    ),
    (BUY_NOW, "ml_localmin"): Template(
        BUY_NOW,
        "ml_localmin",
        "Курс {cur} на низком уровне",
        "Сейчас {rate} ₽ за {unit} — ниже, чем в {pct} % дней за последние {months} мес.",
    ),
}

# Запрещённые обороты: шаблон → почему. Текст приводится к нижнему регистру, «ё» → «е», любой
# пробельный символ (в том числе неразрывный) → пробел, поэтому шаблоны пишутся через «е» и с
# обычным пробелом. Основы даются БЕЗ закрывающей границы слова, чтобы ловить любую словоформу
# («идеальн» → идеальный, идеальная, идеальнее); закрывающий \b стоит только у полных словоформ.
# Регрессия 03.09: закрывающий \b после основы «идеальн» пропускал «Идеальный момент».
FORBIDDEN: tuple[tuple[str, str], ...] = (
    (r"\b(скоро|вот-вот|завтра|на днях|в ближайш)", "утверждение о будущем"),
    (
        r"\b(выраст(ет|ут)|упад(ет|ут)|подорожа(ет|ют)|подешеве(ет|ют)|укреп(ится|ятся)|"
        r"ослабн(ет|ут)|сниз(ится|ятся)|подним(ется|утся)|верн(ется|утся)|продолж(ит|ат)|"
        r"начн(ет|ут)|стан(ет|ут)|буд(ет|ут)|обвал(ится|ятся)|взлет(ит|ят)|пойд(ет|ут)|"
        r"сохран(ится|ятся)|отскоч(ит|ат)|отскок)\b",
        "прогноз курса, будущее время",
    ),
    (r"\b(ожида|прогноз|вероятно|скорее всего|наверняка)", "прогноз"),
    (r"\bуспе(й|йте|ть|ете|ешь|ем)\b", "призыв с намёком на будущее движение"),
    (r"\bне (упус|пропус|тян|откладыва|жди)", "призыв с намёком на будущее"),
    (r"\b(поспеш|торопит|спешит|пора\b|самое время|лови(те)? момент)", "давление, совет вместо факта"),
    (r"\bпоследн(ий|яя|ее) (шанс|возможность)", "давление, намёк на будущее"),
    (r"\bпока не (поздно|подорожал)", "намёк на будущее подорожание"),
    (
        r"\b(сейчас или никогда|только (сегодня|сейчас)|(дальше|потом) (только )?дорож)",
        "давление, намёк на будущее",
    ),
    (r"\bгарантир", "гарантия"),
    (
        r"\b(самый |наи)?лучш(ий|ая|ее|его|ему|им|ей|ую) (курс|момент|время|день)",
        "непроверяемое превосходство",
    ),
    (r"\b(заработ|доходност|инвестиц|инвестир|прибыл|сэконом)", "инвестиционная лексика"),
    (r"\b(рекоменду|совету|стоит перев|лучше перев|пора перев)", "совет вместо факта"),
    (
        r"\b(идеальн|оптимальн|как никогда|лучше не (будет|найти)|ниже (уже )?не будет|"
        r"максимальн(ая|ую|ой) выгод)",
        "оценка будущего без измерения",
    ),
    (r"\bдн(о|а|е|у)\b", "оценка будущего без измерения («дно»)"),
    (r"\b(теряете|потеряете|каждый день ожидания|упущенн)", "утверждение об убытке в будущем"),
)
# «Выгодно» в любой форме допустимо только рядом с измерением: процент И окно (мес., нед., дн., лет).
_ADVANTAGE = re.compile(r"\bвыгод(н\w*|а|ы|е|у|ой|ою)\b")
_MEASURE = re.compile(r"\d+ ?%")
_WINDOW = re.compile(r"\b(мес\.?|месяц\w*|нед\.?|недел\w*|дн\.?|дней|дня|лет|год\w*)")
_SPACES = re.compile(r"[\s  ]+")


def _norm(text: str) -> str:
    return _SPACES.sub(" ", text.lower().replace("ё", "е"))


def check_text(text: str) -> list[tuple[str, str]]:
    """Список (фрагмент, причина) для каждого нарушения; пустой список — чисто. Все вхождения каждого
    правила, а не первое. «Выгодно» допустимо только при проценте И окне измерения где угодно в тексте."""
    hits = []
    low = _norm(text)
    for pattern, reason in FORBIDDEN:
        for m in re.finditer(pattern, low):
            hits.append((m.group(0), reason))
    m = _ADVANTAGE.search(low)
    if m and not (_MEASURE.search(low) and _WINDOW.search(low)):
        hits.append((m.group(0), "«выгодно» без измерения (нужны процент и окно)"))
    return hits


def check_message(title: str, body: str) -> list[tuple[str, str]]:
    """Чекер для пуша целиком: измерение в теле оправдывает «выгодно» в заголовке и наоборот.
    Склейка пробелом: оборот, разорванный границей заголовок/тело, тоже ловится."""
    return check_text(f"{title} {body}")


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
        "month_acc": MONTH_NAMES_ACC.get(int(facts.get("target_month", 1) or 1), ""),
        "dev": f"{abs(float(facts.get('dev_pct', 0) or 0)):.1f}".replace(".", ","),
        # простая средняя за span дней публикации (dip.py); в неделях не пересчитываем — день
        # публикации не равен 1/5 недели (аудит 03.09), а «рабочих дней» проверяемо буквально
        "span": int(float(facts.get("span", 40) or 40)),
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
