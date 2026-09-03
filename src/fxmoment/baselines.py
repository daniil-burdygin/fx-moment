"""Прозрачные правила без индикаторов — база, с которой сравнивается стек (💬 03.09 вечер, пункт 4).

Календарное правило «переводи 20–25-го» не имеет параметров, калибровки и модели. Если стек едва
обгоняет его, это вывод для жюри и факт для экрана перевода. Правило оценивается той же метрикой и
на тех же окнах walk-forward, что индикаторы (`metrics.evaluate_events`), а разница со стеком —
парным блочным бутстрепом (`analysis.paired_pooled_both`): по парам «коридор × окно» и по окнам."""

from __future__ import annotations

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
