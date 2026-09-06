"""Прозрачные правила без индикаторов — база, с которой сравнивается стек (💬 03.09 вечер, пункт 4).

Календарное правило «переводи 20–25-го» не имеет параметров, калибровки и модели. Если стек едва
обгоняет его, это вывод для жюри и факт для экрана перевода. Правило оценивается той же метрикой и
на тех же окнах walk-forward, что индикаторы (`metrics.evaluate_events`), а разница со стеком —
парным блочным бутстрепом (`analysis.paired_pooled_both`): по парам «коридор × окно» и по окнам."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fxmoment import metrics
from fxmoment.backtest.walkforward import Split
from fxmoment.config import BUY_NOW, CALIBRATION_H, CORRIDORS, PRIMARY_TOL_BPS

# first — первый день публикации окна в каждом месяце (один пуш в месяц); all — каждый день окна:
# клиент переводит в любой его день, это не пуш, а правило на экране (частота 4–6 в месяц).
RULE_MODES: tuple[str, ...] = ("first", "all")
# (имя, первое число месяца, последнее, режимы). Окна заданы до замера и под найденный провал
# 15–25 числа не подгоняются: 20–25 — формулировка правила, 20–28 — налоговое окно
# (`analysis.TAX_WINDOW`), «25-е» — одно число: первый день публикации с 25-го. У окна 20–28 режима
# first нет: первый день публикации с 20-го тот же, что у 20–25, строки совпали бы.
CALENDAR_RULES: tuple[tuple[str, int, int, tuple[str, ...]], ...] = (
    ("day20-25", 20, 25, RULE_MODES),
    ("day20-28", 20, 28, ("all",)),
    ("day25", 25, 31, RULE_MODES),
)
STACK_INDICATORS: tuple[str, ...] = ("level", "dip_vs_trend", "ml_localmin", "seasonality")
STREAM_LABEL = "stream BUY_NOW"

SUMMARY_COLUMNS = [
    "indicator",
    "corridor",
    "windows",
    "active_windows",
    "events",
    "lift_mean_median",
    "lift_mean_pooled",
    "benefit_excess_median_bps",
    "benefit_excess_pooled_bps",
    "freq_per_week_median",
    "empty_month_share_median",
]
COMPARE_COLUMNS = [
    "rule",
    "stack",
    "blocks",
    "events_rule",
    "events_stack",
    "lift_rule",
    "lift_stack",
    "diff_lift",
    "diff_lift_ci_lo",
    "diff_lift_ci_hi",
    "verdict_lift",
    "benefit_rule",
    "benefit_stack",
    "diff_benefit",
    "diff_benefit_ci_lo",
    "diff_benefit_ci_hi",
    "verdict_benefit",
    "blocks_by_window",
    "diff_lift_ci_lo_by_window",
    "diff_lift_ci_hi_by_window",
    "verdict_lift_by_window",
    "diff_benefit_ci_lo_by_window",
    "diff_benefit_ci_hi_by_window",
    "verdict_benefit_by_window",
]


def rule_label(name: str, mode: str) -> str:
    return f"calendar:{name}:{mode}"


def calendar_events(index: pd.DatetimeIndex, lo: int, hi: int, mode: str = "first") -> pd.Series:
    """Булев ряд событий календарного правила на днях публикации. Причинно: первый день окна в
    месяце виден по уже прошедшим дням того же месяца, будущие дни не нужны."""
    if mode not in RULE_MODES:
        raise ValueError(f"неизвестный режим правила {mode!r}; допустимы {', '.join(RULE_MODES)}")
    day = np.asarray(index.day)
    inside = pd.Series((day >= lo) & (day <= hi), index=index)
    if mode == "all":
        return inside
    order = inside.astype(int).groupby(index.to_period("M")).cumsum()
    return inside & (order == 1)


def calendar_matrix(
    panel: pd.DataFrame,
    splits: list[Split],
    corridors: tuple[str, ...] = CORRIDORS,
    rules: tuple[tuple[str, int, int, tuple[str, ...]], ...] = CALENDAR_RULES,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Строки как в `matrix.csv`: правило × коридор × окно с метриками `evaluate_events` (без
    интервалов). `indicator` — метка правила: таблицу читает та же машинка, что матрицу."""
    rows: list[dict] = []
    for corridor in corridors:
        if corridor not in panel.columns:
            continue
        rate = panel[corridor].dropna()
        for name, lo, hi, modes in rules:
            for mode in modes:
                events = calendar_events(rate.index, lo, hi, mode)
                for sp in splits:
                    m = metrics.evaluate_events(
                        rate, events, BUY_NOW, h, (sp.test_start, sp.test_end), tol_bps, with_ci=False
                    )
                    rows.append(
                        {
                            "indicator": rule_label(name, mode),
                            "rule": name,
                            "mode": mode,
                            "corridor": corridor,
                            "split": sp.id,
                            "window": sp.label(),
                            "scenario": BUY_NOW,
                            **m,
                        }
                    )
    return pd.DataFrame(rows)


def _pooled(g: pd.DataFrame, column: str, weight: str = "n_scored") -> float:
    w = g[weight].fillna(0.0).to_numpy(dtype=float)
    v = g[column].to_numpy(dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else float("nan")


def calendar_summary(cal: pd.DataFrame) -> pd.DataFrame:
    """Сводка правила в терминах матрицы: по коридорам и строкой `all`; медианы — по активным окнам,
    pooled — взвешенно по событиям окна."""
    if cal.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict] = []
    for by in (["indicator", "corridor"], ["indicator"]):
        for key, g in cal.groupby(by):
            act = g[g["n_scored"].fillna(0) > 0]
            den = float((act["base_mean"] * act["n_scored"]).sum()) if len(act) else 0.0
            row = dict(zip(by, key, strict=True))
            row.setdefault("corridor", "all")
            row.update(
                {
                    "windows": int(len(g)),
                    "active_windows": int(len(act)),
                    "events": int(g["n_events"].sum()),
                    "lift_mean_median": float(act["lift_mean"].median()) if len(act) else np.nan,
                    "lift_mean_pooled": float((act["hit_mean"] * act["n_scored"]).sum() / den)
                    if den
                    else np.nan,
                    "benefit_excess_median_bps": float(act["benefit_excess_bps"].median())
                    if len(act)
                    else np.nan,
                    "benefit_excess_pooled_bps": _pooled(act, "benefit_excess_bps") if len(act) else np.nan,
                    "freq_per_week_median": float(g["freq_per_week"].median()),
                    "empty_month_share_median": float(g["empty_month_share"].median()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def calendar_vs_stack(
    cal: pd.DataFrame,
    matrix: pd.DataFrame,
    stream_matrix: pd.DataFrame | None = None,
    modes: tuple[str, ...] = ("first",),
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Стек против правила парным блочным бутстрепом: `diff_*` = стек − правило, интервалы по парам
    «коридор × окно» и по окнам (`*_by_window`). Стек — каждый индикатор `BUY_NOW` из матрицы и
    `BUY_NOW` итогового потока; правило — режимы `modes` (по умолчанию один пуш в месяц: та же
    частотная категория, что у стека)."""
    from fxmoment.analysis import paired_pooled_both, read_interval

    known = set(matrix["indicator"]) if len(matrix) else set()
    stack: dict[str, pd.DataFrame] = {
        name: matrix[matrix["indicator"] == name] for name in STACK_INDICATORS if name in known
    }
    if stream_matrix is not None and len(stream_matrix):
        stack[STREAM_LABEL] = stream_matrix[stream_matrix["scenario"] == BUY_NOW]
    rows: list[dict] = []
    for label, rule_rows in cal.groupby("indicator"):
        if str(rule_rows["mode"].iloc[0]) not in modes:
            continue
        for name, other in stack.items():
            key = f"{name} − {label}"
            cmp = paired_pooled_both(
                rule_rows.assign(indicator=key), other.assign(indicator=key), h=h, tol_bps=tol_bps
            )
            hit = cmp[cmp["indicator"] == key] if len(cmp) else cmp
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append(
                {
                    "rule": label,
                    "stack": name,
                    "blocks": int(r["blocks"]),
                    "events_rule": int(r["events_a"]),
                    "events_stack": int(r["events_b"]),
                    "lift_rule": r["lift_a"],
                    "lift_stack": r["lift_b"],
                    "diff_lift": r["diff_lift"],
                    "diff_lift_ci_lo": r["diff_lift_ci_lo"],
                    "diff_lift_ci_hi": r["diff_lift_ci_hi"],
                    "benefit_rule": r["benefit_a"],
                    "benefit_stack": r["benefit_b"],
                    "diff_benefit": r["diff_benefit"],
                    "diff_benefit_ci_lo": r["diff_benefit_ci_lo"],
                    "diff_benefit_ci_hi": r["diff_benefit_ci_hi"],
                    "blocks_by_window": int(r["blocks_by_window"]),
                    "diff_lift_ci_lo_by_window": r["diff_lift_ci_lo_by_window"],
                    "diff_lift_ci_hi_by_window": r["diff_lift_ci_hi_by_window"],
                    "diff_benefit_ci_lo_by_window": r["diff_benefit_ci_lo_by_window"],
                    "diff_benefit_ci_hi_by_window": r["diff_benefit_ci_hi_by_window"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=COMPARE_COLUMNS)
    for metric in ("lift", "benefit"):
        for suffix in ("", "_by_window"):
            out[f"verdict_{metric}{suffix}"] = read_interval(
                out[f"diff_{metric}_ci_lo{suffix}"], out[f"diff_{metric}_ci_hi{suffix}"], "стек", "правило"
            )
    return out[COMPARE_COLUMNS]


# --------------------------------------------------- праздники страны-получателя (Ш1)

# Коридор → страна получателя в `data/holidays.csv`. Российские строки календаря правилом не
# используются: нерабочие дни РФ уже вырезаны из оси публикации ЦБ, отдельным событием им быть негде.
CORRIDOR_COUNTRY: dict[str, str] = {"TJS": "TJ", "UZS": "UZ", "KGS": "KG", "AMD": "AM", "KZT": "KZ"}
# «Крупные» праздники — те, к которым деньги посылают домой: Навруз во всех написаниях, оба айта,
# Новый год и дни независимости. Список задан по постановке кейса до замера и по числам не подбирался.
MAJOR_PATTERNS: tuple[str, ...] = (
    "навруз",
    "нооруз",
    "наурыз",
    "ҳайит",
    " айт",
    "рамазон",
    "курбон",
    "новый год",
    "новогодние каникулы",
    "независимост",
)
# `all` — все нерабочие праздничные дни страны; `major` — только крупные; `fixed` — только праздники
# с фиксированной датой (`movable=no`). Последний отвечает на возражение о причинности: дата айта
# объявляется за 1–3 недели, фиксированная известна на годы вперёд.
HOLIDAY_SCOPES: tuple[str, ...] = ("all", "major", "fixed")
HOLIDAY_K: tuple[int, ...] = (1, 3, 5)  # дней публикации до праздника


def holidays_path() -> Path:
    from fxmoment.data.store import repo_root

    return repo_root() / "data" / "holidays.csv"


def load_holidays(path: Path | None = None) -> pd.DataFrame:
    """Календарь нерабочих праздничных дней шести стран (`data/holidays.md` — правило отбора и
    источники). Строка — один нерабочий день одной страны."""
    p = path or holidays_path()
    if not p.exists():
        raise FileNotFoundError(f"нет календаря праздников {p}")
    return pd.read_csv(p, parse_dates=["date"])


def holiday_starts(holidays: pd.DataFrame, country: str, scope: str = "all") -> pd.DatetimeIndex:
    """Начала праздничных блоков страны: подряд идущие нерабочие дни — один праздник, и правило
    целится в его первый день, а не в каждый. Отбор `scope` идёт до склейки: у «крупных» блок свой."""
    if scope not in HOLIDAY_SCOPES:
        raise ValueError(f"неизвестный отбор {scope!r}; допустимы {', '.join(HOLIDAY_SCOPES)}")
    d = holidays[holidays["country"] == country]
    if scope == "major":
        low = d["name"].astype(str).str.lower()
        keep = np.zeros(len(d), dtype=bool)
        for pat in MAJOR_PATTERNS:
            keep |= low.str.contains(pat, regex=False).to_numpy()
        d = d[keep]
    elif scope == "fixed":
        d = d[d["movable"].astype(str).str.lower() == "no"]
    dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(d["date"]))))
    if not len(dates):
        return dates
    gap = np.r_[True, np.asarray((dates[1:] - dates[:-1]).days) > 1]
    return dates[gap]


def holiday_events(index: pd.DatetimeIndex, starts: pd.DatetimeIndex, k: int) -> pd.Series:
    """Булев ряд событий на днях публикации: k последних дней публикации перед началом праздничного
    блока. Отсчёт идёт по оси публикации, а не по календарю, поэтому выходные и нерабочие дни ЦБ в k
    не входят и события на них не встают. Событие на самом празднике не ставится: дни берутся строго
    до его первого дня.

    Причинность. Правило смотрит вперёд по календарю, а не по курсу: дата праздника известна заранее
    (фиксированная — на годы, плавающая — за 1–3 недели, `data/holidays.md`), значение курса в неё не
    входит. Это допустимо и это единственное место в проекте, где событие опирается на знание о
    будущей дате; отбор `fixed` оставляет только даты, известные вне всяких оговорок."""
    if k < 1:
        raise ValueError(f"k = {k}: дней публикации до праздника должно быть не меньше одного")
    idx = pd.DatetimeIndex(index)
    flag = np.zeros(len(idx), dtype=bool)
    for p in np.asarray(idx.searchsorted(starts, side="left")):
        if p > 0:
            flag[max(0, int(p) - k) : int(p)] = True
    return pd.Series(flag, index=idx)


def holiday_rule_label(k: int, scope: str) -> str:
    return f"holiday:k{k}:{scope}"


def holiday_matrix(
    panel: pd.DataFrame,
    splits: list[Split],
    corridors: tuple[str, ...] = CORRIDORS,
    holidays: pd.DataFrame | None = None,
    ks: tuple[int, ...] = HOLIDAY_K,
    scopes: tuple[str, ...] = HOLIDAY_SCOPES,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Строки как в `matrix.csv`: правило × коридор × окно с метриками `evaluate_events`. Праздники
    берутся у страны получателя коридора (`CORRIDOR_COUNTRY`)."""
    hol = load_holidays() if holidays is None else holidays
    rows: list[dict] = []
    for corridor in corridors:
        if corridor not in panel.columns or corridor not in CORRIDOR_COUNTRY:
            continue
        country = CORRIDOR_COUNTRY[corridor]
        rate = panel[corridor].dropna()
        for scope in scopes:
            starts = holiday_starts(hol, country, scope)
            for k in ks:
                events = holiday_events(rate.index, starts, k)
                for sp in splits:
                    m = metrics.evaluate_events(
                        rate, events, BUY_NOW, h, (sp.test_start, sp.test_end), tol_bps, with_ci=False
                    )
                    rows.append(
                        {
                            "indicator": holiday_rule_label(k, scope),
                            "k": k,
                            "scope": scope,
                            "country": country,
                            "corridor": corridor,
                            "split": sp.id,
                            "window": sp.label(),
                            "scenario": BUY_NOW,
                            **m,
                        }
                    )
    return pd.DataFrame(rows)


HOLIDAY_BASELINES: tuple[str, ...] = ("calendar:day25:first", "calendar:day25:all", "seasonality")
HOLIDAY_COMPARE_COLUMNS = [
    "rule",
    "baseline",
    "blocks",
    "events_rule",
    "events_base",
    "lift_rule",
    "lift_base",
    "diff_lift",
    "diff_lift_ci_lo",
    "diff_lift_ci_hi",
    "verdict_lift",
    "benefit_rule",
    "benefit_base",
    "diff_benefit",
    "diff_benefit_ci_lo",
    "diff_benefit_ci_hi",
    "verdict_benefit",
    "blocks_by_window",
    "diff_lift_ci_lo_by_window",
    "diff_lift_ci_hi_by_window",
    "verdict_lift_by_window",
    "diff_benefit_ci_lo_by_window",
    "diff_benefit_ci_hi_by_window",
    "verdict_benefit_by_window",
]


def holiday_vs_baselines(
    hol: pd.DataFrame,
    calendar: pd.DataFrame,
    matrix: pd.DataFrame,
    bases: tuple[str, ...] = HOLIDAY_BASELINES,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Правило праздников против прозрачных баз тем же парным блочным бутстрепом, что стек против
    календаря: `diff_*` = праздники − база, интервалы по парам «коридор × окно» и по окнам. Базы —
    календарь «первый день с 25-го» в обоих режимах и индикатор сезонности: сезонность тоже ловит
    внутримесячную форму, и без неё непонятно, добавляет ли праздник что-то к ней."""
    from fxmoment.analysis import paired_pooled_both, read_interval

    pool: dict[str, pd.DataFrame] = {}
    for name in bases:
        src = calendar if name.startswith("calendar:") else matrix
        if len(src) and name in set(src["indicator"]):
            pool[name] = src[src["indicator"] == name]
    rows: list[dict] = []
    for label, rule_rows in hol.groupby("indicator"):
        for name, base in pool.items():
            key = f"{label} − {name}"
            cmp = paired_pooled_both(
                base.assign(indicator=key), rule_rows.assign(indicator=key), h=h, tol_bps=tol_bps
            )
            hit = cmp[cmp["indicator"] == key] if len(cmp) else cmp
            if hit.empty:
                continue
            r = hit.iloc[0]
            rows.append(
                {
                    "rule": label,
                    "baseline": name,
                    "blocks": int(r["blocks"]),
                    "events_rule": int(r["events_b"]),
                    "events_base": int(r["events_a"]),
                    "lift_rule": r["lift_b"],
                    "lift_base": r["lift_a"],
                    "diff_lift": r["diff_lift"],
                    "diff_lift_ci_lo": r["diff_lift_ci_lo"],
                    "diff_lift_ci_hi": r["diff_lift_ci_hi"],
                    "benefit_rule": r["benefit_b"],
                    "benefit_base": r["benefit_a"],
                    "diff_benefit": r["diff_benefit"],
                    "diff_benefit_ci_lo": r["diff_benefit_ci_lo"],
                    "diff_benefit_ci_hi": r["diff_benefit_ci_hi"],
                    "blocks_by_window": int(r["blocks_by_window"]),
                    "diff_lift_ci_lo_by_window": r["diff_lift_ci_lo_by_window"],
                    "diff_lift_ci_hi_by_window": r["diff_lift_ci_hi_by_window"],
                    "diff_benefit_ci_lo_by_window": r["diff_benefit_ci_lo_by_window"],
                    "diff_benefit_ci_hi_by_window": r["diff_benefit_ci_hi_by_window"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=HOLIDAY_COMPARE_COLUMNS)
    for metric in ("lift", "benefit"):
        for suffix in ("", "_by_window"):
            out[f"verdict_{metric}{suffix}"] = read_interval(
                out[f"diff_{metric}_ci_lo{suffix}"], out[f"diff_{metric}_ci_hi{suffix}"], "праздники", "база"
            )
    return out[HOLIDAY_COMPARE_COLUMNS]
