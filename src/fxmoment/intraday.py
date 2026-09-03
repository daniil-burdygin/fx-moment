"""Внутридневной прогон на часовых свечах Мосбиржи (ADR-0010, проверка 3а).

Вопрос один: есть ли внутри дня момент, которого нет на дневном ряду. Методика та же — walk-forward,
калибровка только на прошлом, усечённые исходы, база случайного шага внутри окна, — меняется ось:
шаг ряда не день публикации, а часовой бар (`fxmoment.profiles.INTRADAY`).

Прогон идёт по коридорам отдельно, а не одной панелью: у пар разные торговые часы и разные даты
начала торгов, и общий календарь пришлось бы заполнять переносом последнего курса — теми самыми
сериями нулевых изменений, которых дневная ось избегает по построению.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fxmoment.backtest.engine import BacktestResult, run_backtest
from fxmoment.backtest.walkforward import make_splits
from fxmoment.combine import evaluate_stream, stream_shape_summary, stream_summary
from fxmoment.config import PRIMARY_TOL_BPS, TOLERANCES_BPS
from fxmoment.profiles import DAILY, INTRADAY, Profile, first_test_for


def run_profile(panel: pd.DataFrame, profile: Profile = INTRADAY) -> dict[str, BacktestResult]:
    """Бэктест по каждому коридору профиля на его собственной оси. Коридор без достаточной истории
    пропускается с явной записью в `skipped`, а не молча: пропуск, о котором не сказано, читается
    как «проверили и ничего не нашли»."""
    out: dict[str, BacktestResult] = {}
    for corridor in profile.corridors:
        if corridor not in panel.columns:
            continue
        rate = panel[corridor].dropna()
        try:
            first_test = first_test_for(rate.index, profile)
        except ValueError:
            continue
        splits = make_splits(
            rate.index,
            first_test=str(first_test.date()),
            test_months=profile.test_months,
            purge_days=profile.purge,
            min_test_days=profile.min_test_steps,
        )
        out[corridor] = run_backtest(
            panel,
            corridors=(corridor,),
            indicators=profile.indicators,
            horizons=profile.horizons,
            analysis_start=profile.analysis_start,
            splits=splits,
            calibration_h=profile.calibration_h,
            grid_scale=profile.step_scale,
            context=profile.context,
        )
    return out


def skipped_corridors(panel: pd.DataFrame, profile: Profile = INTRADAY) -> list[tuple[str, str]]:
    """Коридоры профиля, для которых прогон невозможен, и причина."""
    rows = []
    for corridor in profile.corridors:
        if corridor not in panel.columns:
            rows.append((corridor, "нет ряда в снимке"))
            continue
        n = int(panel[corridor].notna().sum())
        if n <= profile.min_train_steps:
            rows.append((corridor, f"{n} баров при нужных {profile.min_train_steps} на обучение"))
    return rows


def daily_vs_intraday(
    daily_matrix: pd.DataFrame,
    intraday_matrix: pd.DataFrame,
    daily_h: int = DAILY.calibration_h,
    intraday_h: int = INTRADAY.calibration_h,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Сравнение осей на общем периоде: месяц в днях против месяца в барах.

    Период обрезается с ОБЕИХ сторон по пересечению окон: иначе дневной ряд получал бы премию за
    2021–2022 годы, которых у биржевого ряда нет, а внутридневной — за хвост, который дневное
    правило `MIN_TEST_DAYS` отбрасывает. Сетки окон двух осей сдвинуты (внутридневная начинается
    от `first_test_for` своего ряда), поэтому граничные окна совпадают не день в день; обе
    границы названы в столбцах `first_window` и `last_window`.

    Сравнение честно по периоду и по горизонту, но НЕ по числу наблюдений: на часовой оси шагов
    в девять раз больше, поэтому события и база считаются по разным множествам."""
    rows = []
    for corridor, intra in intraday_matrix.groupby("corridor"):
        intra = intra[(intra["h"] == intraday_h) & (intra["tol_bps"] == tol_bps)]
        if intra.empty:
            continue
        day = daily_matrix[
            (daily_matrix["corridor"] == corridor)
            & (daily_matrix["h"] == daily_h)
            & (daily_matrix["tol_bps"] == tol_bps)
        ]
        # метка окна «2023-06…2023-11» одной ширины, поэтому лексикографический порядок = хронологический
        if len(day):
            lo = max(str(intra["window"].min()), str(day["window"].min()))
            hi = min(str(intra["window"].max()), str(day["window"].max()))
            day = day[(day["window"] >= lo) & (day["window"] <= hi)]
            intra = intra[(intra["window"] >= lo) & (intra["window"] <= hi)]
        else:
            lo, hi = str(intra["window"].min()), str(intra["window"].max())
        if intra.empty:
            continue
        for indicator in sorted(set(intra["indicator"]) | set(day["indicator"])):
            i = intra[intra["indicator"] == indicator]
            d = day[day["indicator"] == indicator]
            rows.append(
                {
                    "corridor": corridor,
                    "indicator": indicator,
                    "first_window": lo,
                    "last_window": hi,
                    "daily_events": int(d["n_events"].sum()) if len(d) else 0,
                    "daily_lift_mean_pooled": _pooled_lift(d),
                    "daily_excess_median_bps": (
                        float(d["benefit_excess_bps"].median()) if len(d) else float("nan")
                    ),
                    "intraday_events": int(i["n_events"].sum()),
                    "intraday_lift_mean_pooled": _pooled_lift(i),
                    "intraday_excess_median_bps": float(i["benefit_excess_bps"].median()),
                    "note": _compare_note(corridor, indicator, len(d), len(i)),
                }
            )
    return pd.DataFrame(rows)


def _compare_note(corridor: str, indicator: str, n_daily: int, n_intraday: int) -> str:
    """Почему половина строки пуста. Пустая клетка без объяснения читается как «ноль событий»,
    а не как «здесь вообще не считали»."""
    if not n_daily and corridor not in DAILY.corridors:
        return f"{corridor} — контекст рублёвой стороны, дневного прогона по нему нет"
    if not n_intraday:
        return "на часовой оси индикатор выключен профилем (ADR-0010)"
    if not n_daily:
        return "на дневной оси индикатор в этих окнах не считался"
    return ""


def _pooled_lift(m: pd.DataFrame) -> float:
    """Pooled lift «по среднему»: hit и база взвешены числом оценённых событий окна.

    Молчащие окна дают NaN в hit_mean при ненулевом весе, и без их отсева весь столбец
    обращался бы в NaN — тот же отсев, что в `engine._pooled` (найдено на первом прогоне)."""
    if m.empty:
        return float("nan")
    w = m["n_scored"].fillna(0).to_numpy(dtype=float)
    hit_v = m["hit_mean"].to_numpy(dtype=float)
    base_v = m["base_mean"].to_numpy(dtype=float)
    ok = ~np.isnan(hit_v) & ~np.isnan(base_v) & (w > 0)
    if not ok.any():
        return float("nan")
    hit = (hit_v[ok] * w[ok]).sum() / w[ok].sum()
    base = (base_v[ok] * w[ok]).sum() / w[ok].sum()
    return float(hit / base) if base else float("nan")


def write_intraday_report(
    results: dict[str, BacktestResult],
    panel: pd.DataFrame,
    out_dir: Path,
    profile: Profile = INTRADAY,
    daily_matrix: pd.DataFrame | None = None,
    daily_dir: Path | None = None,
) -> Path:
    from fxmoment.analysis import backtest_provenance
    from fxmoment.data.store import load_moex_meta
    from fxmoment.report import _md_table, git_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    def merge(attr: str) -> pd.DataFrame:
        parts = [getattr(r, attr) for r in results.values()]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    matrix = merge("matrix")
    signals = merge("signals")
    calibration = merge("calibration")
    matrix.to_csv(out_dir / "matrix_bars.csv", index=False)
    signals.to_csv(out_dir / "signals_bars.csv", index=False)
    calibration.to_csv(out_dir / "calibration_bars.csv", index=False)

    summaries = []
    for corridor, r in results.items():
        s = r.summary(h=profile.calibration_h, tol_bps=PRIMARY_TOL_BPS)
        if len(s):
            summaries.append(s.assign(corridor=corridor))
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    # имя файла — из профиля: горизонт в нём не константа, а параметр оси
    summary_name = f"summary_bars_h{profile.calibration_h}_tol{int(PRIMARY_TOL_BPS)}.csv"
    summary.to_csv(out_dir / summary_name, index=False)

    short = []
    for r in results.values():
        for h in (h for h in profile.horizons if h < profile.calibration_h):
            s = r.summary(h=h, tol_bps=PRIMARY_TOL_BPS)
            if len(s):
                short.append(s.assign(h=h))
    short_df = pd.concat(short, ignore_index=True) if short else pd.DataFrame()
    short_df.to_csv(out_dir / "summary_bars_short_horizons.csv", index=False)

    streams, shapes = [], []
    for r in results.values():
        _, sm, sh = evaluate_stream(
            r,
            panel,
            params=profile.policy,
            horizons=profile.horizons,
            tolerances=TOLERANCES_BPS,
            calibration_h=profile.calibration_h,
            series_gap=profile.series_gap,
        )
        if len(sm):
            streams.append(sm)
        if len(sh):
            shapes.append(sh)
    stream_matrix = pd.concat(streams, ignore_index=True) if streams else pd.DataFrame()
    shape = pd.concat(shapes, ignore_index=True) if shapes else pd.DataFrame()
    stream = stream_summary(stream_matrix, h=profile.calibration_h) if len(stream_matrix) else pd.DataFrame()
    shape_sum = stream_shape_summary(shape) if len(shape) else pd.DataFrame()
    stream.to_csv(out_dir / "stream_bars_summary.csv", index=False)
    shape_sum.to_csv(out_dir / "stream_bars_shape.csv", index=False)

    has_daily = daily_matrix is not None and len(matrix) > 0
    compare = daily_vs_intraday(daily_matrix, matrix) if has_daily else pd.DataFrame()
    compare.to_csv(out_dir / "daily_vs_intraday.csv", index=False)

    meta = load_moex_meta()
    # чужие числа — чужой провенанс: сравнение осей считается по матрице ДНЕВНОГО отчёта, и без
    # его штампа свежий хеш кода выдавался бы за происхождение этих строк (аудит 03.09)
    daily_prov = backtest_provenance(daily_dir) if (has_daily and daily_dir is not None) else None
    (out_dir / "provenance_bars.json").write_text(
        json.dumps(
            {
                "code": git_hash(),
                "fetched_at_utc": meta.get("fetched_at_utc"),
                "interval_length": meta.get("interval_length"),
                "profile": profile.name,
                "bars_per_day": profile.step_scale,
                "daily_report": daily_prov or None,
                "built_at_utc": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
                "windows": {c: [s.label() for s in r.splits] for c, r in results.items()},
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    skipped = skipped_corridors(panel, profile)
    lines = [
        "# Внутридневной прогон на часовых свечах Мосбиржи (ADR-0010)",
        "",
        f"Снимок Мосбиржи: {meta.get('fetched_at_utc', '?')}, {meta.get('rows', '?')} баров. "
        f"Код: `{git_hash()}`. Сформировано {datetime.now(UTC):%Y-%m-%d %H:%M} UTC.",
        "",
        f"Ось — часовой бар, {profile.step_scale} баров в торговом дне (медиана по парам с рынком). "
        f"Горизонты в барах: {', '.join(map(str, profile.horizons))}; калибровочный — "
        f"{profile.calibration_h} (месяц), зазор {profile.purge}. Окна сеток индикаторов "
        f"умножены на {profile.step_scale}: «120 дней» — это {120 * profile.step_scale} баров.",
        "",
        "Сезонность и обучаемый индикатор выключены. Сезонность — по ADR-0010; ML — потому что окна "
        "его признаков заданы в днях внутри `build_features` и профилем не масштабируются, так что "
        "прогон на них был бы не переносом на другую ось, а молчаливой методической ошибкой.",
        "",
    ]
    if skipped:
        lines += [
            "**Коридоры без прогона.** " + "; ".join(f"{c} — {why}" for c, why in skipped) + ".",
            "",
        ]
    lines += [
        f"## Точность по индикаторам, горизонт {profile.calibration_h} баров (месяц), допуск 25 бп",
        "",
        _md_table(summary.round(3)) if len(summary) else "событий нет",
        "",
        "## Короткие горизонты: есть ли момент внутри дня",
        "",
        "1 бар — час, 4 — полдня, 9 — торговый день. Если внутридневного момента нет, lift на этих "
        "горизонтах не отличается от единицы — тот же ответ, что дал дневной ряд на 1–5 днях.",
        "",
        _md_table(short_df.round(3)) if len(short_df) else "событий нет",
        "",
        "## Итоговый поток на часовой оси",
        "",
        "Политика та же, что на дневной оси, и переведена в бары профилем: охлаждение "
        f"{profile.policy.cooldown_days} баров, не больше {profile.policy.max_per_window} пушей за "
        f"{profile.policy.window_days}, шторм по волатильности за {profile.policy.storm_vol_window}. "
        "Ранг индикаторов и отключение слабых считаются на горизонте профиля "
        f"({profile.calibration_h} баров), а не на дневном.",
        "",
        _md_table(shape_sum.round(3)) if len(shape_sum) else "поток пуст",
        "",
        _md_table(stream.round(3)) if len(stream) else "",
        "",
        "## Дневная ось против часовой на общем периоде",
        "",
        f"Месяц в днях (h = {DAILY.calibration_h}) против месяца в барах (h = {profile.calibration_h}) "
        "на пересечении окон обеих осей: период режется с двух сторон, границы — в столбцах "
        "`first_window` и `last_window`. Сетки окон сдвинуты (внутридневная начинается от своего "
        "ряда), поэтому границы совпадают не день в день. Сравнение честно по периоду и горизонту, "
        "но не по числу наблюдений: на часовой оси шагов в девять раз больше.",
        "",
        _md_table(compare.round(3)) if len(compare) else "нет общих окон",
    ]
    (out_dir / "README_bars.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir
