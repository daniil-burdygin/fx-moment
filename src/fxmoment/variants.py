"""Вариантный прогон против `reports/latest` (💬 03.09 вечер: пункты 1 и 8, дальше 2 и 5).

Вариант — тот же бэктест с одним изменённым условием (начало окон теста, база ранга, набор
индикаторов), записанный в свой каталог `reports/<вариант>/`. Сравнение идёт по общим окнам: строки
матрицы обязаны совпасть там, где вариант условие не трогает (проверка, что параметризация ничего не
сдвинула), поток сравнивается парным блочным бутстрепом по тем же окнам, а окна только у варианта
сводятся отдельно, по каждому окну — чтобы режим (2020) был виден, а не растворялся в среднем."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fxmoment.analysis import paired_pooled_both, read_interval
from fxmoment.config import CALIBRATION_H, PRIMARY_TOL_BPS
from fxmoment.report import _md_table, git_hash

MATRIX_KEYS: tuple[str, ...] = ("indicator", "corridor", "window", "h", "tol_bps")
RUN_LOCAL: tuple[str, ...] = ("split",)  # номер окна свой у каждого прогона, сравнивать нечего
TOL = 1e-9
DECISIONS: tuple[str, ...] = ("sent", "muted", "thinned", "cooldown", "storm")
# что из провенанса прогона переносится в провенанс сравнения
RUN_KEYS: tuple[str, ...] = ("code", "built_at_utc", "first_test", "rank_base", "ml", "extra_indicators")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()


def _provenance(run: Path) -> dict:
    path = run / "provenance.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _pooled(g: pd.DataFrame, column: str, weight: str = "n_scored") -> float:
    w = g[weight].fillna(0.0).to_numpy(dtype=float)
    v = g[column].to_numpy(dtype=float)
    ok = ~np.isnan(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else float("nan")


def overlap_check(
    latest: pd.DataFrame, variant: pd.DataFrame, keys: tuple[str, ...] = MATRIX_KEYS
) -> pd.DataFrame:
    """По общим ключам: для каждого столбца, который есть в обоих прогонах, — сколько строк
    расходятся больше `TOL` и наибольшая |разница|. NaN в обоих — совпадение, NaN в одном — нет."""
    common = [c for c in latest.columns if c in variant.columns and c not in keys and c not in RUN_LOCAL]
    merged = latest.merge(variant, on=list(keys), suffixes=("_a", "_b"))
    rows: list[dict] = []
    for c in common:
        a, b = merged[f"{c}_a"], merged[f"{c}_b"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            both_nan = a.isna() & b.isna()
            one_nan = a.isna() ^ b.isna()
            diff = (a - b).abs().where(~both_nan, 0.0)
            differ = (diff > TOL) | one_nan
            top = float(diff[~one_nan].max()) if (~one_nan).any() else np.nan
        else:
            differ = a.fillna("").astype(str) != b.fillna("").astype(str)
            top = np.nan
        rows.append(
            {"column": c, "rows": int(len(merged)), "rows_differ": int(differ.sum()), "max_abs_diff": top}
        )
    return pd.DataFrame(rows)


def align_splits(latest: pd.DataFrame, variant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Общие окна по метке `window`; `split` варианта переписывается на id того же окна у latest —
    блоки парного бутстрепа собираются по `split`, а нумерация окон у варианта своя."""
    ids = latest.drop_duplicates("window").set_index("window")["split"]
    common = sorted(set(latest["window"]) & set(variant["window"]))
    a = latest[latest["window"].isin(common)]
    b = variant[variant["window"].isin(common)].copy()
    b["split"] = b["window"].map(ids)
    return a, b


def _rename_ab(cmp: pd.DataFrame, first: str) -> pd.DataFrame:
    out = cmp.rename(
        columns={
            "indicator": first,
            "events_a": "events_latest",
            "events_b": "events_variant",
            "lift_a": "lift_latest",
            "lift_b": "lift_variant",
            "benefit_a": "benefit_latest",
            "benefit_b": "benefit_variant",
        }
    )
    for metric in ("lift", "benefit"):
        for suffix in ("", "_by_window"):
            out[f"verdict_{metric}{suffix}"] = read_interval(
                out[f"diff_{metric}_ci_lo{suffix}"], out[f"diff_{metric}_ci_hi{suffix}"], "вариант", "latest"
            )
    return out


def matrix_compare(
    latest: pd.DataFrame, variant: pd.DataFrame, h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    """Индикаторы на общих окнах парным бутстрепом (вариант − latest); у варианта, который
    индикаторы не трогает, здесь нули."""
    a, b = align_splits(latest, variant)
    cmp = paired_pooled_both(a, b, h=h, tol_bps=tol_bps)
    return _rename_ab(cmp, "indicator") if len(cmp) else cmp


def pairs_compare(
    latest: pd.DataFrame,
    variant: pd.DataFrame,
    pairs: dict[str, str],
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Индикатор варианта против ДРУГОГО индикатора latest на общих окнах (`level_drift` против
    `level`): строки варианта переименовываются в индикатор latest и идут в тот же парный бутстреп —
    по всем коридорам (`corridor = all`) и по каждому отдельно. Пара, у которой одной стороны нет,
    пропускается."""
    if not pairs or latest.empty or variant.empty:
        return pd.DataFrame()
    a, b = align_splits(latest, variant)
    rows: list[pd.DataFrame] = []
    for v_name, l_name in pairs.items():
        la = a[a["indicator"] == l_name]
        vb = b[b["indicator"] == v_name].assign(indicator=l_name)
        if la.empty or vb.empty:
            continue
        for corridor in ("all", *sorted(set(la["corridor"]) & set(vb["corridor"]))):
            xa = la if corridor == "all" else la[la["corridor"] == corridor]
            xb = vb if corridor == "all" else vb[vb["corridor"] == corridor]
            cmp = paired_pooled_both(xa, xb, h=h, tol_bps=tol_bps)
            cmp = cmp[cmp["indicator"] == l_name] if len(cmp) else cmp
            if cmp.empty:
                continue
            out = _rename_ab(cmp.assign(indicator=f"{v_name} → {l_name}"), "pair")
            out.insert(1, "corridor", corridor)
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def extra_indicators_summary(
    variant: pd.DataFrame,
    latest_indicators: set[str],
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Индикаторы только у варианта (`level_drift`): по коридорам и по всем — окна, события, pooled
    lift «по среднему» и выгода сверх случайного дня (взвешены событиями окна), медиана частоты,
    доля пустых месяцев."""
    if variant.empty:
        return pd.DataFrame()
    m = variant[
        (variant["h"] == h) & (variant["tol_bps"] == tol_bps) & ~variant["indicator"].isin(latest_indicators)
    ]
    if m.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for ind, mi in m.groupby("indicator"):
        for corridor, g in [*mi.groupby("corridor"), ("all", mi)]:
            hit, base = _pooled(g, "hit_mean"), _pooled(g, "base_mean")
            rows.append(
                {
                    "indicator": ind,
                    "corridor": corridor,
                    "windows": int(g["window"].nunique()),
                    "events": int(g["n_events"].fillna(0).sum()),
                    "lift_mean_pooled": hit / base if base > 0 else np.nan,
                    "benefit_excess_pooled_bps": _pooled(g, "benefit_excess_bps"),
                    "freq_per_week_median": float(g["freq_per_week"].median()),
                    "empty_month_share_mean": float(g["empty_month_share"].mean()),
                }
            )
    return pd.DataFrame(rows)


def stream_compare(
    latest: pd.DataFrame, variant: pd.DataFrame, h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    """Итоговый поток на общих окнах по сценариям парным бутстрепом (вариант − latest)."""
    if latest.empty or variant.empty:
        return pd.DataFrame()
    a, b = align_splits(latest, variant)
    cmp = paired_pooled_both(
        a.assign(indicator=a["scenario"]), b.assign(indicator=b["scenario"]), h=h, tol_bps=tol_bps
    )
    return _rename_ab(cmp, "scenario") if len(cmp) else cmp


def shape_compare(latest: pd.DataFrame, variant: pd.DataFrame) -> pd.DataFrame:
    """Форма потока по коридорам на общих окнах: пуши, частота, пустые месяцы, снятые штормом."""
    if latest.empty or variant.empty:
        return pd.DataFrame()
    a, b = align_splits(latest, variant)

    def agg(df: pd.DataFrame, tag: str) -> pd.DataFrame:
        g = df.groupby("corridor")
        return pd.DataFrame(
            {
                f"pushes_{tag}": g["pushes"].sum(),
                f"freq_per_week_median_{tag}": g["freq_per_week"].median(),
                f"empty_month_share_mean_{tag}": g["empty_month_share"].mean(),
                f"storm_blocked_{tag}": g["storm_blocked"].sum(),
            }
        )

    out = agg(a, "latest").join(agg(b, "variant"), how="outer").reset_index()
    out["pushes_diff"] = out["pushes_variant"] - out["pushes_latest"]
    return out


def decisions_compare(
    latest: pd.DataFrame, variant: pd.DataFrame, latest_windows: list[str], variant_windows: list[str]
) -> pd.DataFrame:
    """Решения политики по индикаторам на общих окнах: сколько событий отправлено, заглушено,
    прорежено, охлаждено, снято штормом — в каждом прогоне."""
    if latest.empty or variant.empty:
        return pd.DataFrame()
    common = set(latest_windows) & set(variant_windows)

    def counts(df: pd.DataFrame, windows: list[str], tag: str) -> pd.DataFrame:
        labels = df["split"].map(dict(enumerate(windows)))
        d = df[labels.isin(common)]
        c = d.groupby(["indicator", "decision"]).size().unstack(fill_value=0)
        return c.reindex(columns=list(DECISIONS), fill_value=0).add_suffix(f"_{tag}")

    out = counts(latest, latest_windows, "latest").join(
        counts(variant, variant_windows, "variant"), how="outer"
    )
    return out.fillna(0).astype(int).reset_index()


def extra_windows_summary(
    variant: pd.DataFrame, latest_windows: set[str], h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    """Окна только у варианта: pooled lift и выгода по индикаторам — по коридорам, по всем и по
    каждому окну (все коридоры вместе)."""
    m = variant[
        (variant["h"] == h) & (variant["tol_bps"] == tol_bps) & ~variant["window"].isin(latest_windows)
    ]
    if m.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for by in (["indicator", "corridor"], ["indicator"], ["indicator", "window"]):
        for key, g in m.groupby(by):
            act = g[g["n_scored"].fillna(0) > 0]
            den = float((act["base_mean"] * act["n_scored"]).sum()) if len(act) else 0.0
            row = dict(zip(by, key, strict=True))
            row.setdefault("corridor", "all")
            row.setdefault("window", "all")
            row.update(
                {
                    "windows": int(g["window"].nunique()),
                    "events": int(g["n_events"].sum()),
                    "lift_mean_pooled": float((act["hit_mean"] * act["n_scored"]).sum() / den)
                    if den
                    else np.nan,
                    "benefit_excess_pooled_bps": _pooled(act, "benefit_excess_bps") if len(act) else np.nan,
                    "freq_per_week_median": float(g["freq_per_week"].median()),
                }
            )
            rows.append(row)
    cols = [
        "indicator",
        "corridor",
        "window",
        "windows",
        "events",
        "lift_mean_pooled",
        "benefit_excess_pooled_bps",
        "freq_per_week_median",
    ]
    return pd.DataFrame(rows)[cols]


def stream_raw(
    latest: pd.DataFrame, variant: pd.DataFrame, h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    """Без базы: доля попаданий «по среднему» и выгода вперёд пушей на общих окнах. Ответ на
    возражение «ранг по месяцу судят оконной базой»: ряд и окна у прогонов одни, поэтому сырые
    исходы пушей сравнимы вовсе без базы."""
    if latest.empty or variant.empty:
        return pd.DataFrame()
    a, b = align_splits(latest, variant)
    scenarios = [*sorted(set(a["scenario"]) | set(b["scenario"])), "all"]
    rows: list[dict] = []
    for scen in scenarios:
        row: dict = {"scenario": scen}
        for tag, df in (("latest", a), ("variant", b)):
            g = df[(df["h"] == h) & (df["tol_bps"] == tol_bps)]
            if scen != "all":
                g = g[g["scenario"] == scen]
            act = g[g["n_scored"].fillna(0) > 0]
            row[f"events_{tag}"] = int(act["n_scored"].sum())
            row[f"hit_{tag}"] = _pooled(act, "hit_mean") if len(act) else np.nan
            row[f"benefit_fwd_{tag}"] = _pooled(act, "benefit_fwd_bps") if len(act) else np.nan
        row["diff_hit"] = row["hit_variant"] - row["hit_latest"]
        row["diff_benefit_fwd"] = row["benefit_fwd_variant"] - row["benefit_fwd_latest"]
        rows.append(row)
    return pd.DataFrame(rows)


def extra_windows_stream(
    variant: pd.DataFrame, latest_windows: set[str], h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    """Итоговый поток на окнах только у варианта: те же сводки, что по индикаторам, по сценариям."""
    if variant.empty:
        return pd.DataFrame()
    out = extra_windows_summary(variant.assign(indicator=variant["scenario"]), latest_windows, h, tol_bps)
    return out.rename(columns={"indicator": "scenario"})


def extra_windows_shape(variant: pd.DataFrame, latest_windows: set[str]) -> pd.DataFrame:
    """Форма потока на окнах только у варианта: пуши, частота, пустые месяцы, шторм — по коридорам."""
    if variant.empty:
        return pd.DataFrame()
    s = variant[~variant["window"].isin(latest_windows)]
    if s.empty:
        return pd.DataFrame()
    g = s.groupby("corridor")
    out = pd.DataFrame(
        {
            "windows": g["window"].nunique(),
            "pushes": g["pushes"].sum(),
            "freq_per_week_median": g["freq_per_week"].median(),
            "empty_month_share_mean": g["empty_month_share"].mean(),
            "storm_blocked": g["storm_blocked"].sum(),
        }
    ).reset_index()
    total = {
        "corridor": "all",
        "windows": int(s["window"].nunique()),
        "pushes": int(s["pushes"].sum()),
        "freq_per_week_median": float(s["freq_per_week"].median()),
        "empty_month_share_mean": float(s["empty_month_share"].mean()),
        "storm_blocked": int(s["storm_blocked"].sum()),
    }
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def _md(df: pd.DataFrame) -> str:
    return _md_table(df.round(3)) if len(df) else "нет данных"


def compare_runs(
    latest_dir: Path,
    variant_dir: Path,
    panel: pd.DataFrame | None = None,
    pairs: dict[str, str] | None = None,
) -> Path:
    """Пишет `reports/<вариант>/vs_latest/`: проверку общих окон, парные сравнения матрицы и потока,
    форму потока, решения политики, сводку окон и индикаторов только у варианта и пары
    «индикатор варианта против индикатора latest» (`pairs`). `panel` не нужна: всё берётся из CSV
    обоих прогонов; параметр оставлен для симметрии с `analyze`."""
    _ = panel
    pairs = pairs or {}
    out = variant_dir / "vs_latest"
    out.mkdir(parents=True, exist_ok=True)
    pa, pb = _provenance(latest_dir), _provenance(variant_dir)
    la, lb = list(pa.get("windows") or []), list(pb.get("windows") or [])
    ma, mb = _read(latest_dir / "matrix.csv"), _read(variant_dir / "matrix.csv")
    overlap = overlap_check(ma, mb)
    overlap.to_csv(out / "matrix_overlap.csv", index=False)
    mcmp = matrix_compare(ma, mb)
    mcmp.to_csv(out / "matrix_compare.csv", index=False)
    sa, sb = _read(latest_dir / "stream_matrix.csv"), _read(variant_dir / "stream_matrix.csv")
    scmp = stream_compare(sa, sb)
    scmp.to_csv(out / "stream_compare.csv", index=False)
    raw = stream_raw(sa, sb)
    raw.to_csv(out / "stream_raw.csv", index=False)
    sha, shb = _read(latest_dir / "stream_shape.csv"), _read(variant_dir / "stream_shape.csv")
    shape = shape_compare(sha, shb)
    shape.to_csv(out / "stream_shape.csv", index=False)
    latest_windows = set(ma["window"]) if len(ma) else set()
    extra_stream = extra_windows_stream(sb, latest_windows)
    extra_stream.to_csv(out / "extra_windows_stream.csv", index=False)
    extra_shape = extra_windows_shape(shb, latest_windows)
    extra_shape.to_csv(out / "extra_windows_shape.csv", index=False)
    dec = decisions_compare(
        _read(latest_dir / "stream_decisions.csv"), _read(variant_dir / "stream_decisions.csv"), la, lb
    )
    dec.to_csv(out / "decisions.csv", index=False)
    extra = extra_windows_summary(mb, latest_windows)
    extra.to_csv(out / "extra_windows.csv", index=False)
    latest_indicators = set(ma["indicator"]) if len(ma) else set()
    extra_ind = extra_indicators_summary(mb, latest_indicators)
    extra_ind.to_csv(out / "extra_indicators.csv", index=False)
    pcmp = pairs_compare(ma, mb, pairs)
    pcmp.to_csv(out / "pairs_compare.csv", index=False)
    bad = overlap[overlap["rows_differ"] > 0]
    common_n = int(overlap["rows"].iloc[0]) if len(overlap) else 0
    provenance = {
        "code": git_hash(),
        "built_at_utc": f"{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}",
        "latest": {k: pa.get(k) for k in RUN_KEYS},
        "variant": {k: pb.get(k) for k in RUN_KEYS},
        "common_windows": sorted(set(la) & set(lb)),
        "variant_only_windows": [w for w in lb if w not in set(la)],
        "variant_only_indicators": sorted(set(mb["indicator"]) - latest_indicators) if len(mb) else [],
        "pairs": pairs,
        "matrix_rows_compared": common_n,
        "matrix_columns_differ": bad["column"].tolist(),
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    show = [
        c
        for c in (
            "pair",
            "corridor",
            "scenario",
            "indicator",
            "blocks",
            "events_latest",
            "events_variant",
            "lift_latest",
            "lift_variant",
            "diff_lift",
            "diff_lift_ci_lo",
            "diff_lift_ci_hi",
            "verdict_lift",
            "verdict_lift_by_window",
            "benefit_latest",
            "benefit_variant",
            "diff_benefit",
            "diff_benefit_ci_lo",
            "diff_benefit_ci_hi",
            "verdict_benefit",
            "verdict_benefit_by_window",
        )
    ]
    lines = [
        f"# Вариант `{variant_dir.name}` против `latest`",
        "",
        f"latest: код `{pa.get('code', '?')}`, первое окно {pa.get('first_test', '?')}, база ранга "
        f"{pa.get('rank_base', 'window')}. Вариант: код `{pb.get('code', '?')}`, первое окно "
        f"{pb.get('first_test', '?')}, база ранга {pb.get('rank_base', 'window')}, обучаемый "
        f"{pb.get('ml') or 'local'}, добавлены индикаторы: "
        f"{', '.join(provenance['variant_only_indicators']) or 'нет'}. Сравнение: код "
        f"`{git_hash()}`, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Общих окон "
        f"{len(provenance['common_windows'])}, только у варианта {len(provenance['variant_only_windows'])}.",
        "",
        "## Общие окна: совпадают ли строки матрицы",
        "",
        "Вариант, который не трогает индикаторы (другое начало окон, другая база ранга), обязан дать на "
        "общих окнах те же строки матрицы побайтово: обучение расширяющееся от `ANALYSIS_START`, и "
        "параметризация ничего не сдвигает. Расхождение здесь — дефект, а не результат.",
        "",
        (
            f"Все {len(overlap)} общих столбцов совпали на {common_n} строках."
            if len(overlap) and bad.empty
            else _md(bad)
        ),
        "",
        "## Индикаторы на общих окнах (вариант − latest, парный бутстреп)",
        "",
        _md(mcmp[[c for c in show if c in mcmp.columns]]) if len(mcmp) else "нет данных",
        "",
        "## Пары индикаторов на общих окнах (индикатор варианта − индикатор latest)",
        "",
        "Индикатор варианта против другого индикатора latest (`compare-runs --pair level_drift=level`): "
        "строки варианта идут в тот же парный бутстреп под именем индикатора latest. По всем коридорам "
        "(`all`) и по каждому отдельно.",
        "",
        _md(pcmp[[c for c in show if c in pcmp.columns]]) if len(pcmp) else "пар не задано",
        "",
        "## Индикаторы только у варианта",
        "",
        _md(extra_ind),
        "",
        "## Итоговый поток на общих окнах по сценариям (вариант − latest)",
        "",
        _md(scmp[[c for c in show if c in scmp.columns]]) if len(scmp) else "нет данных",
        "",
        "## Поток на общих окнах без базы",
        "",
        "Доля попаданий «по среднему» и выгода вперёд самих пушей, без вычитания случайного дня: ряд и "
        "окна у прогонов одни, поэтому сырые исходы сравнимы и не зависят от выбора базы.",
        "",
        _md(raw),
        "",
        "## Форма потока на общих окнах",
        "",
        _md(shape),
        "",
        "## Решения политики по индикаторам на общих окнах",
        "",
        _md(dec),
        "",
        "## Окна только у варианта",
        "",
        "По коридорам, по всем коридорам (`corridor = all`) и по каждому окну отдельно (`window`), "
        "чтобы режим отдельного полугодия был виден.",
        "",
        _md(extra),
        "",
        "### Итоговый поток на окнах только у варианта",
        "",
        _md(extra_stream),
        "",
        _md(extra_shape),
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return out
