"""Машина времени: страница «сигналы как на дату T» из готового отчёта (строка Ш9 плана).

Страница ничего не считает сама. Ряды, пуши, решения политики и тексты берутся из `reports/<прогон>/`
и панели ЦБ, вердикт попадания — теми же функциями `labels`, что в бэктесте. Скрипт на странице
только прячет всё после даты T и по кнопке раскрывает следующие h дней публикации; журнал попаданий
на дату T считает проверенными лишь пуши, чей горизонт закрылся не позже T."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fxmoment import labels
from fxmoment.backtest.walkforward import Split
from fxmoment.config import ANALYSIS_START, BUY_NOW, CALIBRATION_H, CORRIDORS, PRIMARY_TOL_BPS, WINDOW_CLOSING
from fxmoment.texts import render

TEMPLATE = Path(__file__).with_name("templates") / "timemachine.html"
SERIES: tuple[str, ...] = CORRIDORS + ("USD",)
PLACEHOLDER = "__PAYLOAD__"


def _num(value: Any) -> float | None:
    """Число для JSON: None вместо NaN, шесть значащих цифр — курс сума 0,00736 не теряет знаков."""
    if value is None or pd.isna(value):
        return None
    return float(f"{float(value):.6g}")


def push_verdicts(rate: pd.Series, h: int, tol_bps: float) -> pd.DataFrame:
    """Вердикты на каждую дату публикации ряда теми же функциями, что размечают матрицу: попадание
    `BUY_NOW` по среднему и по минимуму с допуском, попадание `WINDOW_CLOSING` без допуска (tol_up = 0),
    выгода вперёд и средний курс следующих h дней. NaN — горизонт ещё не закрыт."""
    return pd.DataFrame(
        {
            "hit_mean": labels.hit_buy_now(rate, h, tol_bps, mode="mean"),
            "hit_min": labels.hit_buy_now(rate, h, tol_bps, mode="min"),
            "hit_wc": labels.hit_window_closing(rate, h, 0.0),
            "benefit_fwd_bps": labels.benefit_fwd_bps(rate, h),
            "future_mean": labels.future_mean(rate, h),
        }
    )


def _flag(value: Any) -> bool | None:
    return None if value is None or pd.isna(value) else bool(value)


def _window_bases(matrix: pd.DataFrame, splits: list[Split], h: int, tol_bps: float) -> list[dict[str, Any]]:
    """База случайного дня по окнам и коридорам: у `BUY_NOW` — рабочий допуск, у `WINDOW_CLOSING` —
    строки с допуском 0 (ADR-0003: допуск там не применяется)."""
    if matrix.empty:
        return [
            {
                "id": s.id,
                "label": s.label(),
                "start": f"{s.test_start:%Y-%m-%d}",
                "end": f"{s.test_end:%Y-%m-%d}",
                "base": {},
            }
            for s in splits
        ]
    m = matrix[matrix["h"] == h]
    bn = (
        m[(m["tol_bps"] == tol_bps) & (m["scenario"] == BUY_NOW)]
        .groupby(["corridor", "split"])["base_mean"]
        .first()
    )
    wc = (
        m[(m["tol_bps"] == 0) & (m["scenario"] == WINDOW_CLOSING)]
        .groupby(["corridor", "split"])["base_mean"]
        .first()
    )
    out = []
    for s in splits:
        base = {
            c: {BUY_NOW: _num(bn.get((c, s.id))), WINDOW_CLOSING: _num(wc.get((c, s.id)))} for c in CORRIDORS
        }
        out.append(
            {
                "id": s.id,
                "label": s.label(),
                "start": f"{s.test_start:%Y-%m-%d}",
                "end": f"{s.test_end:%Y-%m-%d}",
                "base": base,
            }
        )
    return out


def timemachine_payload(
    panel: pd.DataFrame,
    decisions: pd.DataFrame,
    matrix: pd.DataFrame,
    splits: list[Split],
    provenance: dict[str, Any],
    *,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
    start: str = ANALYSIS_START,
    run: str = "latest",
) -> dict[str, Any]:
    """Данные страницы: ось дат публикации от `start`, ряды коридоров и доллара, отправленные пуши с
    текстами и вердиктами, удержанные политикой события с причиной, базы окон, провенанс прогона."""
    sub = panel.loc[pd.Timestamp(start) :]
    dates = [f"{d:%Y-%m-%d}" for d in sub.index]
    pos = {d: i for i, d in enumerate(sub.index)}
    series = {c: [_num(v) for v in sub[c]] for c in SERIES if c in sub.columns}
    verdicts = {c: push_verdicts(panel[c].dropna(), h, tol_bps) for c in CORRIDORS if c in panel.columns}

    pushes: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for row in decisions.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if date not in pos or row.corridor not in verdicts:
            continue
        i = pos[date]
        if row.decision != "sent":
            held.append(
                {"i": i, "c": row.corridor, "ind": row.indicator, "sc": row.scenario, "why": row.decision}
            )
            continue
        facts = json.loads(row.facts) if isinstance(row.facts, str) and row.facts else {}
        scenario = row.push_scenario if isinstance(row.push_scenario, str) else row.scenario
        title, body = render(row.corridor, scenario, row.indicator, float(row.rate), facts)
        v = verdicts[row.corridor]
        vr = v.loc[date] if date in v.index else None
        hit = None
        if vr is not None:
            hit = _flag(vr["hit_wc"] if scenario == WINDOW_CLOSING else vr["hit_mean"])
        pushes.append(
            {
                "i": i,
                "c": row.corridor,
                "ind": row.indicator,
                "sc": scenario,
                "split": int(row.split),
                "rate": _num(row.rate),
                "strength": _num(row.strength),
                "speed": row.speed,
                "title": title,
                "body": body,
                "facts": facts,
                "hit": hit,
                "hit_min": None if vr is None else _flag(vr["hit_min"]),
                "benefit": None if vr is None else _num(vr["benefit_fwd_bps"]),
                "fmean": None if vr is None else _num(vr["future_mean"]),
            }
        )
    pushes.sort(key=lambda p: (p["c"], p["i"]))
    held.sort(key=lambda e: (e["c"], e["i"]))
    idx = [p["i"] for p in pushes]
    meta = {
        "run": run,
        "code": provenance.get("code"),
        "last_eff_date": provenance.get("last_eff_date"),
        "built_at_utc": provenance.get("built_at_utc"),
        "h": h,
        "tol_bps": tol_bps,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "first_push": dates[min(idx)] if idx else None,
        "last_push": dates[max(idx)] if idx else None,
        "n_pushes": len(pushes),
        "n_held": len(held),
    }
    return {
        "meta": meta,
        "dates": dates,
        "series": series,
        "windows": _window_bases(matrix, splits, h, tol_bps),
        "pushes": pushes,
        "held": held,
    }


def render_timemachine(payload: dict[str, Any]) -> str:
    """HTML страницы: шаблон с данными внутри, без внешних ресурсов — открывается офлайн."""
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"в шаблоне {TEMPLATE} нет метки {PLACEHOLDER}")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, data)


def load_inputs(
    panel: pd.DataFrame, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[Split], dict[str, Any]]:
    """Решения политики, матрица, окна и провенанс прогона. Окна восстанавливаются по панели тем же
    `load_result`, что и в анализе, — другой снимок данных отвергается там же."""
    from fxmoment.analysis import backtest_provenance, load_result

    dec_path = out_dir / "stream_decisions.csv"
    if not dec_path.exists():
        raise FileNotFoundError(f"нет {dec_path} — сначала `fxmoment backtest`")
    decisions = pd.read_csv(dec_path, parse_dates=["date"])
    result = load_result(panel, out_dir)
    return decisions, result.matrix, result.splits, backtest_provenance(out_dir)


def write_timemachine(panel: pd.DataFrame, out_dir: Path, path: Path, run: str = "latest") -> dict[str, Any]:
    """Собирает страницу из прогона `out_dir` в `path`; возвращает `meta` данных."""
    decisions, matrix, splits, provenance = load_inputs(panel, out_dir)
    payload = timemachine_payload(panel, decisions, matrix, splits, provenance, run=run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_timemachine(payload), encoding="utf-8")
    return payload["meta"]
