"""Отчёт бэктеста: таблицы в reports/, графики коридоров с отметками, фронтир частота—точность."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from fxmoment.backtest.engine import BacktestResult  # noqa: E402
from fxmoment.data.store import load_meta, repo_root  # noqa: E402

MARKERS = {
    "momentum": ("v", "tab:red"),
    "level": ("o", "tab:blue"),
    "reversal": ("^", "tab:green"),
    "seasonality": ("s", "tab:orange"),
    "ml_localmin": ("D", "tab:purple"),
    "dip_vs_trend": ("P", "tab:cyan"),
}


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def write_report(result: BacktestResult, panel: pd.DataFrame, out_dir: Path | None = None) -> Path:
    out = out_dir or (repo_root() / "reports" / "latest")
    out.mkdir(parents=True, exist_ok=True)
    result.signals.to_csv(out / "signals.csv", index=False)
    result.matrix.to_csv(out / "matrix.csv", index=False)
    result.calibration.to_csv(out / "calibration.csv", index=False)
    summary = result.summary(h=20)
    summary.to_csv(out / "summary_h20_tol25.csv", index=False)
    result.summary(h=5).to_csv(out / "summary_h5_tol25.csv", index=False)
    result.summary(h=20, tol_bps=0.0).to_csv(out / "summary_h20_tol0.csv", index=False)
    from fxmoment.combine import evaluate_stream, stream_summary

    decided, stream_matrix = evaluate_stream(result, panel)
    decided.to_csv(out / "stream_decisions.csv", index=False)
    stream_matrix.to_csv(out / "stream_matrix.csv", index=False)
    stream = stream_summary(stream_matrix) if len(stream_matrix) else pd.DataFrame()
    stream.to_csv(out / "stream_summary_h20_tol25.csv", index=False)
    meta = load_meta()
    lines = [
        "# Бэктест — сводка (h = 20, допуск 25 бп, медианы по окнам walk-forward)",
        "",
        f"Снимок данных: {meta.get('fetched_at_utc', '?')} "
        f"(последняя дата действия {meta.get('last_eff_date', '?')}). "
        f"Код: `{_git_hash()}`. Сформировано {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.",
        "",
        "Окна: " + ", ".join(s.label() for s in result.splits),
        "",
        _md_table(summary.round(3)),
        "",
        "## Итоговый поток после политики (ADR-0006), h = 20, допуск 25 бп",
        "",
        _md_table(stream.round(3)) if len(stream) else "поток пуст",
        "",
        "Полная матрица — `matrix.csv` (все горизонты, допуски и окна). События — `signals.csv`, "
        "решения политики — `stream_decisions.csv`, метрики потока — `stream_matrix.csv`.",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    for corridor in result.signals["corridor"].unique():
        plot_corridor(panel[corridor], result.signals, corridor, out / f"chart_{corridor}.png")
    return out


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def plot_corridor(
    rate: pd.Series, signals: pd.DataFrame, corridor: str, path: Path, start: str | None = None
) -> None:
    r = rate.dropna()
    if start:
        r = r.loc[start:]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(r.index, r.values, color="black", linewidth=0.8, label=f"ЦБ РФ RUB→{corridor} (за 1 ед.)")
    ev = signals[(signals["corridor"] == corridor) & (signals["date"] >= r.index[0])]
    for ind, (marker, color) in MARKERS.items():
        e = ev[ev["indicator"] == ind]
        if len(e):
            ax.scatter(
                e["date"], e["rate"], marker=marker, color=color, s=28, label=f"{ind} ({len(e)})", zorder=3
            )
    ax.set_title(f"{corridor}: курс и срабатывания индикаторов на тестовых окнах")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
