"""Замер TimesFM 3 (`docs/decisions/timesfm-experiment.md`): три вопроса к одному снимку прогнозов.

1. Навык прогноза на ряде ЦБ: ошибка медианы против «курс не изменится», доля угаданных
   направлений против доли большинства, покрытие полосы 10–90 %; по периодам до и после среза
   данных предобучения модели (ноябрь 2023 по карточке модели).
2. Прогноз как правило «модель ждёт курс выше на ≥ порог бп» — той же метрикой, что индикаторы
   матрицы, по тем же окнам walk-forward, с гейтом уровня и без. Контроль, не индикатор продукта.
3. Признаки прогноза в обучаемом индикаторе: парное сравнение `reports/latest` и `reports/timesfm`
   той же машинкой, что калибровка против априорных (`paired_pooled_comparison`)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fxmoment import labels, metrics
from fxmoment.analysis import paired_pooled_comparison
from fxmoment.backtest import make_splits
from fxmoment.config import ANALYSIS_START, BUY_NOW, CALIBRATION_H, CORRIDORS, PRIMARY_TOL_BPS
from fxmoment.data.forecast import STEPS, load_forecast_meta, load_snapshot
from fxmoment.data.store import repo_root
from fxmoment.indicators.base import rearm_events, rolling_pct_rank
from fxmoment.report import git_hash

PRETRAIN_CUTOFF = "2023-11-30"  # срез данных предобучения по карточке модели (Wikipedia pageviews)
RULE_THRESHOLDS_BPS: tuple[float, ...] = (0.0, 25.0, 50.0)
RULE_REARM = 5  # как у обучаемого индикатора
GATE_WINDOW, GATE_PCT = 120, 0.20  # гейт уровня, ADR-0005 п. 2
MIN_ROWS = 20
SNAPSHOT_KEYS = (
    "model",
    "revision",
    "generated_at_utc",
    "rows",
    "device",
    "context_cap",
    "horizon",
    "self_check_max_diff_bps",
)
SKILL_COLUMNS = [
    "currency",
    "period",
    "n",
    "mae_ratio",
    "dir_accuracy",
    "dir_majority",
    "share_forecast_up",
    "coverage_10_90",
]


def skill_table(panel: pd.DataFrame, snapshot: pd.DataFrame, start: str = ANALYSIS_START) -> pd.DataFrame:
    """Навык медианы прогноза на шаге s против наивного «курс не изменится» (f_s = a_T).

    `mae_ratio` < 1 — модель точнее наивного; `dir_accuracy` против `dir_majority` — доля
    угаданных знаков против доли большинства (то, что даёт правило «всегда вниз/вверх»);
    `coverage_10_90` — доля исходов внутри полосы квантилей 0,1–0,9, у калиброванной модели 0,8."""
    rows: list[dict] = []
    for ccy, g in snapshot.groupby("currency"):
        rate = panel[str(ccy)].dropna()
        g = g.set_index("pub_date").sort_index()
        for s in STEPS:
            real = (labels.future_at(rate, s) / rate - 1) * 1e4
            df = pd.DataFrame(
                {
                    "real": real.reindex(g.index),
                    "med": g[f"med_h{s}_bps"],
                    "q10": g[f"q10_h{s}_bps"],
                    "q90": g[f"q90_h{s}_bps"],
                }
            ).dropna()
            df = df.loc[pd.Timestamp(start) :]
            cut = pd.Timestamp(PRETRAIN_CUTOFF)
            after = df.loc[cut + pd.Timedelta(days=1) :]
            periods = (("все", df), ("до среза", df.loc[:cut]), ("после среза", after))
            for period, sub in periods:
                if len(sub) < MIN_ROWS:
                    continue
                nz = sub[(sub["real"] != 0) & (sub["med"] != 0)]
                dir_acc = float((np.sign(nz["med"]) == np.sign(nz["real"])).mean()) if len(nz) else np.nan
                inside = (sub["real"] >= sub["q10"]) & (sub["real"] <= sub["q90"])
                mae_model = float((sub["med"] - sub["real"]).abs().mean())
                mae_naive = float(sub["real"].abs().mean())
                rows.append(
                    {
                        "currency": ccy,
                        "step": s,
                        "period": period,
                        "n": int(len(sub)),
                        "mae_model_bps": mae_model,
                        "mae_naive_bps": mae_naive,
                        "mae_ratio": mae_model / mae_naive if mae_naive else np.nan,
                        "dir_accuracy": dir_acc,
                        "dir_majority": float(max((sub["real"] > 0).mean(), (sub["real"] < 0).mean())),
                        "share_forecast_up": float((sub["med"] > 0).mean()),
                        "share_realized_up": float((sub["real"] > 0).mean()),
                        "coverage_10_90": float(inside.mean()),
                        "mean_forecast_bps": float(sub["med"].mean()),
                        "mean_realized_bps": float(sub["real"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _pooled(g: pd.DataFrame, num: str, weight: str = "n_scored") -> float:
    w = g[weight].fillna(0.0)
    return float((g[num].fillna(0.0) * w).sum() / w.sum()) if w.sum() else np.nan


def rule_table(
    panel: pd.DataFrame,
    snapshot: pd.DataFrame,
    thresholds: tuple[float, ...] = RULE_THRESHOLDS_BPS,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Прогноз как правило `BUY_NOW`: «средний прогноз на 20 шагов выше курса действия на ≥ порог»,
    охлаждение как у обучаемого, с гейтом уровня и без. Строки по окнам walk-forward и сводка
    в терминах матрицы: медианы по активным окнам, pooled по событиям, строка `all` по коридорам."""
    ana = panel.loc[pd.Timestamp(ANALYSIS_START) :]
    splits = make_splits(ana.index)
    snap = snapshot.set_index(["currency", "pub_date"])["mean20_bps"].sort_index()
    rows: list[dict] = []
    for ccy in CORRIDORS:
        if ccy not in snap.index.get_level_values(0):
            continue
        rate = panel[ccy].dropna()
        fc = snap.loc[ccy].reindex(rate.index)
        gate = rolling_pct_rank(rate, GATE_WINDOW) <= GATE_PCT
        for thr in thresholds:
            for gated in (False, True):
                cond = fc >= thr
                if gated:
                    cond = cond & gate
                signal = rearm_events(cond, RULE_REARM)
                for sp in splits:
                    m = metrics.evaluate_events(
                        rate, signal, BUY_NOW, h, (sp.test_start, sp.test_end), tol_bps, with_ci=False
                    )
                    key = {"threshold_bps": thr, "gated": gated, "corridor": ccy, "split": sp.id}
                    rows.append({**key, "window": sp.label(), **m})
    detail = pd.DataFrame(rows)
    summ: list[dict] = []
    keys = ["threshold_bps", "gated"]
    for by in (keys + ["corridor"], keys):
        for key, g in detail.groupby(by):
            act = g[g["n_scored"] > 0]
            den = float((act["base_mean"] * act["n_scored"]).sum()) if len(act) else 0.0
            pooled = float((act["hit_mean"] * act["n_scored"]).sum() / den) if den else np.nan
            row = dict(zip(by, key, strict=True))
            row.update(
                {
                    "corridor": row.get("corridor", "all"),
                    "windows": int(len(g)),
                    "active_windows": int(len(act)),
                    "events": int(g["n_events"].sum()),
                    "lift_mean_median": float(act["lift_mean"].median()) if len(act) else np.nan,
                    "lift_pooled": pooled,
                    "benefit_excess_median": float(act["benefit_excess_bps"].median())
                    if len(act)
                    else np.nan,
                    "benefit_excess_pooled": _pooled(act, "benefit_excess_bps") if len(act) else np.nan,
                    "freq_per_week_median": float(g["freq_per_week"].median()),
                    "empty_month_share_median": float(g["empty_month_share"].median()),
                }
            )
            summ.append(row)
    cols = [
        "threshold_bps",
        "gated",
        "corridor",
        "windows",
        "active_windows",
        "events",
        "lift_mean_median",
        "lift_pooled",
        "benefit_excess_median",
        "benefit_excess_pooled",
        "freq_per_week_median",
        "empty_month_share_median",
    ]
    return detail, pd.DataFrame(summ)[cols]


def stream_pushes(out_dir: Path) -> int | None:
    """Пушей итогового потока по форме потока отчёта (`pushes` по коридорам)."""
    path = out_dir / "stream_shape_summary.csv"
    if not path.exists() or path.stat().st_size <= 1:
        return None
    d = pd.read_csv(path)
    return int(d["pushes"].sum()) if "pushes" in d.columns else None


def _md(df: pd.DataFrame) -> str:
    if not len(df):
        return "нет данных"
    d = df.round(3)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = d.itertuples(index=False)
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def write_forecast_report(
    panel: pd.DataFrame, out_dir: Path | None = None, window_from: str = "2024-01"
) -> Path:
    out = out_dir or (repo_root() / "reports" / "timesfm" / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    snap = load_snapshot()
    meta = load_forecast_meta()
    skill = skill_table(panel, snap)
    skill.to_csv(out / "forecast_skill.csv", index=False)
    detail, rule = rule_table(panel, snap)
    detail.to_csv(out / "forecast_rule_windows.csv", index=False)
    rule.to_csv(out / "forecast_rule.csv", index=False)
    latest, tfm = repo_root() / "reports" / "latest", repo_root() / "reports" / "timesfm"
    cmp_all = cmp_late = pd.DataFrame()
    if (latest / "matrix.csv").exists() and (tfm / "matrix.csv").exists():
        a, b = pd.read_csv(latest / "matrix.csv"), pd.read_csv(tfm / "matrix.csv")
        cmp_all = paired_pooled_comparison(a, b)
        cmp_late = paired_pooled_comparison(a, b, window_from=window_from)
        cmp_all.to_csv(out / "ml_compare.csv", index=False)
        cmp_late.to_csv(out / "ml_compare_late.csv", index=False)
    provenance = {
        "code": git_hash(),
        "built_at_utc": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "forecast_snapshot": {k: meta.get(k) for k in SNAPSHOT_KEYS},
        "pushes_latest": stream_pushes(latest),
        "pushes_timesfm": stream_pushes(tfm),
        "window_from": window_from,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=1))
    skill_h20 = skill[skill["step"] == 20][SKILL_COLUMNS]
    rule_all = rule[rule["corridor"] == "all"]
    lines = [
        "# Замер TimesFM 3 — навык прогноза, прогноз как правило, признаки в обучаемом",
        "",
        f"Снимок прогнозов: {meta.get('model')}@{str(meta.get('revision', ''))[:8]}, "
        f"{meta.get('rows')} строк, "
        f"контекст до {meta.get('context_cap')} дней публикации, горизонт {meta.get('horizon')}, устройство "
        f"{meta.get('device')}, самопроверка {meta.get('self_check_max_diff_bps')} бп. Код: `{git_hash()}`. "
        f"Сформировано {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.",
        "",
        "Лицензия весов 3.0 некоммерческая: это замер, в продукт модель не идёт. Веса предобучены на данных "
        f"до {PRETRAIN_CUTOFF}; окна с {window_from} — единственные, где модель заведомо не видела периода.",
        "",
        "## 1. Навык на шаге 20 (h = 20), все валюты",
        "",
        "`mae_ratio` — ошибка медианы к ошибке «курс не изменится» (< 1 — точнее наивного); `dir_accuracy` — "
        "доля угаданных знаков, рядом `dir_majority` — доля большинства; `coverage_10_90` — покрытие полосы "
        "0,1–0,9 "
        "(у калиброванной модели 0,8). Полная таблица по шагам 1/5/10/20 — `forecast_skill.csv`.",
        "",
        _md(skill_h20),
        "",
        "## 2. Прогноз как правило BUY_NOW, h = 20, допуск 25 бп, суммарно по коридорам",
        "",
        "Правило: «средний прогноз на 20 шагов выше курса действия на ≥ порог бп», охлаждение 5 дней; "
        "`gated` — "
        "с гейтом уровня (нижние 20 % окна 120). По коридорам — `forecast_rule.csv`, по окнам — "
        "`forecast_rule_windows.csv`.",
        "",
        _md(rule_all),
        "",
        "## 3. Обучаемый индикатор: признаки прогноза (b) против без них (a)",
        "",
        "Парный блочный бутстреп по парам «коридор × окно», h = 20, допуск 25 бп. Остальные индикаторы "
        "обязаны "
        "показать нулевую разницу — признаки идут только в обучаемый.",
        "",
        "Все окна:",
        "",
        _md(cmp_all),
        "",
        f"Окна с {window_from}:",
        "",
        _md(cmp_late),
        "",
        f"Пушей итогового потока: {provenance['pushes_latest']} без признаков, "
        f"{provenance['pushes_timesfm']} с ними.",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return out
