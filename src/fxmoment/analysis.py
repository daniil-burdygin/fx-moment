"""Анализы поверх сохранённого бэктеста (пункты 5–7 постановки, ADR-0004 п. 3, ADR-0008):
цена ожидания «быстрый → медленный», перенос параметров между коридорами, сводки без шокового
режима 2022, фронтир «частота — точность» для индикатора уровня.

Вход — `reports/latest/` (signals.csv, matrix.csv, calibration.csv, stream_matrix.csv) и панель
курсов; бэктест заново не гоняется. Все оценки — на тестовых окнах walk-forward."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxmoment import labels, metrics
from fxmoment.backtest.engine import BacktestResult
from fxmoment.backtest.walkforward import Split, make_splits
from fxmoment.config import (
    ANALYSIS_START,
    BUY_NOW,
    CALIBRATION_H,
    CONTEXT,
    CORRIDORS,
    FREQUENCY_BAND,
    PRIMARY_TOL_BPS,
    SHOCK_REGIME,
)
from fxmoment.data.store import repo_root
from fxmoment.indicators import BASE_INDICATORS, Indicator, Level
from fxmoment.indicators.base import rolling_days_since_min, rolling_pct_rank
from fxmoment.indicators.features import enrich_context

# Сетка фронтира шире калибровочной `Level.grid()`: нужны и частые, и редкие точки
FRONTIER_GRID: list[dict[str, Any]] = [
    {"window": w, "pct": p, "stall_days": s, "rearm": r}
    for w in (20, 40, 60, 120, 250)
    for p in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    for s in (0, 3)
    for r in (1, 3, 5)
]


def load_result(panel: pd.DataFrame, out_dir: Path | None = None) -> BacktestResult:
    """Собирает BacktestResult из CSV отчёта. Окна восстанавливаются по панели, поэтому снимок
    данных должен быть тем же, на котором делался бэктест."""
    out = out_dir or (repo_root() / "reports" / "latest")
    if not (out / "matrix.csv").exists():
        raise FileNotFoundError(f"нет {out / 'matrix.csv'} — сначала `fxmoment backtest`")
    signals = pd.read_csv(out / "signals.csv", parse_dates=["date"])
    matrix = pd.read_csv(out / "matrix.csv")
    calibration = pd.read_csv(out / "calibration.csv")
    splits = make_splits(panel.loc[pd.Timestamp(ANALYSIS_START) :].index)
    return BacktestResult(signals, matrix, calibration, splits)


def backtest_provenance(out_dir: Path | None = None) -> dict[str, Any]:
    """Код и снимок, которыми собран бэктест (`provenance.json` из write_report); анализ штампует их
    рядом со своими, чтобы свежий хеш кода не выдавался за происхождение чужих чисел (аудит 03.09)."""
    out = out_dir or (repo_root() / "reports" / "latest")
    path = out / "provenance.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ---------------------------------------------------------------- вспомогательное


def _event_series(dates: pd.Series | pd.DatetimeIndex, index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(False, index=index)
    d = pd.DatetimeIndex(dates)
    s.loc[d[d.isin(index)]] = True
    return s


def _test_days(rate: pd.Series, splits: list[Split]) -> pd.DatetimeIndex:
    idx = rate.index
    mask = np.zeros(len(idx), dtype=bool)
    for s in splits:
        mask |= (idx >= s.test_start) & (idx <= s.test_end)
    return idx[mask]


def _test_weeks(rate: pd.Series, splits: list[Split]) -> float:
    """Недели тестовых окон по календарным границам — как metrics.frequency_per_week."""
    total = 0.0
    for sp in splits:
        end = min(sp.test_end, rate.index[-1])
        total += max((end - sp.test_start).days + 1, 1) / 7
    return total


def _split_of_day(rate: pd.Series, splits: list[Split]) -> pd.Series:
    """Тестовый день → id окна. База случайного дня всегда берётся из окна события: события
    кучкуются в периодах падения, и общий пул смешал бы выбор периода с выбором дня."""
    parts = [pd.Series(sp.id, index=rate.loc[sp.test_start : sp.test_end].index) for sp in splits]
    return pd.concat(parts) if parts else pd.Series(dtype=int)


def _test_months(splits: list[Split]) -> list[pd.Period]:
    months: list[pd.Period] = []
    for s in splits:
        months.extend(pd.period_range(s.test_start, s.test_end, freq="M"))
    return months


def shock_split_ids(splits: list[Split], regime: tuple[str, str] = SHOCK_REGIME) -> set[int]:
    """Окна, пересекающиеся с шоковым режимом (ADR-0004)."""
    a, b = pd.Timestamp(regime[0]), pd.Timestamp(regime[1])
    return {s.id for s in splits if s.test_start <= b and s.test_end >= a}


def _ci(values: np.ndarray) -> tuple[float, float]:
    return metrics.bootstrap_ci(np.asarray(values, dtype=float))


def _excess(dates: pd.DatetimeIndex, bf: pd.Series, sod: pd.Series, base_bf: dict[int, float]) -> np.ndarray:
    """Выгода вперёд события минус выгода случайного дня его окна, бп."""
    d = pd.DatetimeIndex(dates)
    return bf.reindex(d).to_numpy() - pd.Series(d).map(sod).map(base_bf).to_numpy()


def _lift(dates: pd.DatetimeIndex, hit_all: pd.Series, sod: pd.Series, base_hit: dict[int, float]) -> float:
    """Hit rate событий к базе их окон (взвешенно по событиям)."""
    d = pd.DatetimeIndex(dates)
    hv = hit_all.reindex(d)
    ok = hv.notna().to_numpy()
    if not ok.any():
        return np.nan
    base = float(pd.Series(d[ok]).map(sod).map(base_hit).to_numpy().mean())
    return float(hv[ok].mean() / base) if base > 0 else np.nan


# ---------------------------------------------------------------- цена ожидания (пункт 6)


def price_of_waiting_table(
    result: BacktestResult,
    panel: pd.DataFrame,
    fast: str = "momentum",
    slow: str = "level",
    k: int = 10,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
    exclude_shock: bool = False,
) -> pd.DataFrame:
    """По коридорам: сколько быстрых сигналов подтверждается медленным в пределах k дней публикации,
    как меняется курс за время ожидания (бп, > 0 — ожидание стоило денег), и интегральная метрика —
    выгода на пуш сверх случайного дня своего окна для стратегий «слать сразу» и «ждать подтверждения»."""
    rows = []
    splits = result.splits
    if exclude_shock:
        ids = shock_split_ids(splits)
        splits = [sp for sp in splits if sp.id not in ids]
    split_ids = [sp.id for sp in splits]
    for corridor in [c for c in CORRIDORS if c in result.signals["corridor"].unique()]:
        rate = panel[corridor].dropna()
        sig = result.signals[
            (result.signals["corridor"] == corridor) & (result.signals["split"].isin(split_ids))
        ]
        fast_ev = _event_series(sig.loc[sig["indicator"] == fast, "date"], rate.index)
        slow_ev = _event_series(sig.loc[sig["indicator"] == slow, "date"], rate.index)
        if not fast_ev.any() or not slow_ev.any():
            continue
        pw = metrics.price_of_waiting(rate, fast_ev, slow_ev, k)
        sod = _split_of_day(rate, splits)
        bf = labels.benefit_fwd_bps(rate, h)
        hit_all = labels.hit_for_scenario(rate, BUY_NOW, h, tol_bps, mode="mean")
        base_bf = {sp.id: float(bf.loc[sp.test_start : sp.test_end].dropna().mean()) for sp in splits}
        base_hit = {sp.id: float(hit_all.loc[sp.test_start : sp.test_end].dropna().mean()) for sp in splits}

        fast_days = pd.DatetimeIndex(pw["fast_date"])
        slow_days = pd.DatetimeIndex(slow_ev[slow_ev].index)
        conf = pw[pw["confirmed"]]
        unconf = pw[~pw["confirmed"]]
        confirmed_days = pd.DatetimeIndex(conf["slow_date"])
        # безусловная цена ожидания: без подтверждения — изменение курса за те же k дней
        pos = rate.index.get_indexer(fast_days)
        later = pos + k
        full_h = later < len(rate)  # горизонт k дней целиком в ряду; иначе цена не считается (не клип)
        delta_k = np.full(len(pos), np.nan)
        delta_k[full_h] = (rate.to_numpy()[later[full_h]] / rate.to_numpy()[pos[full_h]] - 1) * 1e4
        delta_all = np.where(pw["confirmed"].to_numpy(), pw["delta_bps"].to_numpy(dtype=float), delta_k)
        sent_days = confirmed_days.unique()  # два быстрых сигнала с одним подтверждением — один пуш
        # «ждать»: клиент действует в день подтверждения; без подтверждения пуша нет (вклад 0)
        wait_value = np.zeros(len(pw))
        wait_value[pw["confirmed"].to_numpy()] = _excess(confirmed_days, bf, sod, base_bf)
        send_now = _excess(fast_days, bf, sod, base_bf)
        weeks = _test_weeks(rate, splits)
        delta = conf["delta_bps"].to_numpy(dtype=float)
        lo, hi = _ci(delta) if len(delta) >= 5 else (np.nan, np.nan)
        alo, ahi = _ci(delta_all) if len(delta_all) >= 5 else (np.nan, np.nan)
        wlo, whi = _ci(wait_value) if len(wait_value) >= 5 else (np.nan, np.nan)
        flo, fhi = _ci(send_now) if len(send_now) >= 5 else (np.nan, np.nan)
        rows.append(
            {
                "corridor": corridor,
                "fast": fast,
                "slow": slow,
                "k_days": k,
                "n_fast": int(len(pw)),
                "n_fast_full_horizon": int(full_h.sum()),
                "n_slow": int(slow_ev.sum()),
                "n_sent_wait": int(len(sent_days)),
                "confirmed_share": float(pw["confirmed"].mean()),
                "days_waited_median": float(conf["days_waited"].median()) if len(conf) else np.nan,
                "price_of_waiting_bps": float(delta.mean()) if len(delta) else np.nan,
                "price_of_waiting_ci_lo": lo,
                "price_of_waiting_ci_hi": hi,
                "price_of_waiting_median_bps": float(np.median(delta)) if len(delta) else np.nan,
                "price_of_waiting_all_bps": float(np.nanmean(delta_all)) if len(delta_all) else np.nan,
                "price_of_waiting_all_ci_lo": alo,
                "price_of_waiting_all_ci_hi": ahi,
                "fast_hit_mean": float(hit_all.reindex(fast_days).dropna().mean()),
                "slow_hit_mean": float(hit_all.reindex(slow_days).dropna().mean()),
                "fast_lift_mean": _lift(fast_days, hit_all, sod, base_hit),
                "slow_lift_mean": _lift(slow_days, hit_all, sod, base_hit),
                "fast_freq_per_week": len(fast_days) / weeks if weeks else np.nan,
                "slow_freq_per_week": len(slow_days) / weeks if weeks else np.nan,
                "send_now_excess_bps": float(np.nanmean(send_now)),
                "send_now_excess_ci_lo": flo,
                "send_now_excess_ci_hi": fhi,
                "unconfirmed_excess_bps": float(
                    np.nanmean(_excess(pd.DatetimeIndex(unconf["fast_date"]), bf, sod, base_bf))
                )
                if len(unconf)
                else np.nan,
                "confirmed_fast_excess_bps": float(
                    np.nanmean(_excess(pd.DatetimeIndex(conf["fast_date"]), bf, sod, base_bf))
                )
                if len(conf)
                else np.nan,
                "wait_excess_per_fast_bps": float(np.nanmean(wait_value)),
                "wait_excess_per_sent_bps": float(np.nanmean(_excess(sent_days, bf, sod, base_bf)))
                if len(sent_days)
                else np.nan,
                "wait_excess_ci_lo": wlo,
                "wait_excess_ci_hi": whi,
                "slow_excess_bps": float(np.nanmean(_excess(slow_days, bf, sod, base_bf))),
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        w, s = df["wait_excess_per_fast_bps"], df["send_now_excess_bps"]
        best = pd.concat([w, s], axis=1).max(axis=1)
        df["verdict"] = np.select(
            [w.isna() | s.isna(), best <= 0, w > s, s > w],
            ["данных не хватает", "обе не лучше случайного дня", "ждать подтверждения", "слать сразу"],
            default="стратегии равны",
        )
        # интервалы стратегий пересекаются — вердикт по точечным оценкам не доказан
        df["difference_within_ci"] = (df["wait_excess_ci_lo"] <= df["send_now_excess_ci_hi"]) & (
            df["send_now_excess_ci_lo"] <= df["wait_excess_ci_hi"]
        )
    return df


VERDICTS = (
    "данных не хватает",
    "обе не лучше случайного дня",
    "ждать подтверждения",
    "слать сразу",
    "стратегии равны",
)


# ---------------------------------------------------------------- перенос параметров (ADR-0004 п. 3)


def transfer_table(
    result: BacktestResult,
    panel: pd.DataFrame,
    source: str = "KZT",
    indicators: tuple[type[Indicator], ...] | None = None,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Параметры, откалиброванные на `source` в каждом окне (из calibration.csv), применяются ко всем
    коридорам; метрики на тестовых окнах. Обучаемый индикатор не переносится — модель не сохраняется."""
    inds = tuple(c for c in (indicators or BASE_INDICATORS) if not c.trainable)
    full = panel  # индикаторы считаются по всей истории (разогрев с 2015), как в run_backtest
    ctx_all = full[[c for c in CONTEXT if c in full.columns]]
    cal = result.calibration[result.calibration["corridor"] == source]
    ran = set(result.matrix["corridor"].unique())
    series = {c: full[c].dropna() for c in CORRIDORS if c in full.columns and c in ran}
    ctxs = {c: enrich_context(r, ctx_all) for c, r in series.items()}
    rows = []
    for split in result.splits:
        for cls in inds:
            rec = cal[(cal["indicator"] == cls.name) & (cal["split"] == split.id)]
            if rec.empty:
                continue
            params = json.loads(rec.iloc[0]["params"])
            ctor = {k: v for k, v in params.items() if not k.startswith("_")}
            for corridor, rate in series.items():
                ind = cls(**ctor)
                out = ind.compute(rate.loc[: split.test_end], ctxs[corridor].loc[: split.test_end])
                m = metrics.evaluate_events(
                    rate,
                    out["signal"],
                    ind.scenario,
                    h,
                    (split.test_start, split.test_end),
                    tol_bps,
                    with_ci=False,
                )
                rows.append(
                    {
                        "indicator": cls.name,
                        "corridor": corridor,
                        "source": source,
                        "transferred": corridor != source,
                        "split": split.id,
                        "window": split.label(),
                        "params": json.dumps(ctor, ensure_ascii=False),
                        **m,
                    }
                )
    return pd.DataFrame(rows)


def transfer_compare(
    result: BacktestResult,
    transferred: pd.DataFrame,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Своя калибровка против перенесённой: медианы lift «по среднему», выгоды сверх случайного дня
    и частоты по окнам. `lift_drop` > 0,2 — порог из concept.md."""
    if transferred.empty:
        return pd.DataFrame()
    nat = result.matrix[(result.matrix["h"] == h) & (result.matrix["tol_bps"] == tol_bps)]
    tr = transferred[(transferred["h"] == h) & (transferred["tol_bps"] == tol_bps)]

    def agg(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        g = df.groupby(["indicator", "corridor"])
        return pd.DataFrame(
            {
                f"{prefix}_events": g["n_events"].sum(),
                f"{prefix}_lift_mean_median": g["lift_mean"].median(),
                f"{prefix}_excess_median_bps": g["benefit_excess_bps"].median(),
                f"{prefix}_freq_per_week_median": g["freq_per_week"].median(),
            }
        )

    out = agg(nat, "own").join(agg(tr, "transferred"), how="inner")
    out["lift_drop"] = out["own_lift_mean_median"] - out["transferred_lift_mean_median"]
    out["source"] = transferred["source"].iloc[0] if len(transferred) else ""
    return out.reset_index()


# ---------------------------------------------------------------- без шокового режима


def summary_without_shock(
    result: BacktestResult, h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    ids = shock_split_ids(result.splits)
    sub = BacktestResult(
        result.signals, result.matrix[~result.matrix["split"].isin(ids)], result.calibration, result.splits
    )
    return sub.summary(h=h, tol_bps=tol_bps)


def stream_summary_without_shock(
    stream_matrix: pd.DataFrame, splits: list[Split], h: int = CALIBRATION_H, tol_bps: float = PRIMARY_TOL_BPS
) -> pd.DataFrame:
    from fxmoment.combine.evaluate import stream_summary

    ids = shock_split_ids(splits)
    return stream_summary(stream_matrix[~stream_matrix["split"].isin(ids)], h=h, tol_bps=tol_bps)


# ---------------------------------------------------------------- база: полугодие против месяца


MIN_MONTH_DAYS = 5  # дней публикации в календарном месяце, иначе месячная база слишком шумная


def _base_by_split(
    series: pd.Series, days: pd.DatetimeIndex, sod: pd.Series, splits: list[Split]
) -> dict[int, float]:
    """Среднее ряда по тестовым дням каждого окна — нынешняя база случайного дня."""
    return {s.id: float(series.reindex(days[sod.reindex(days) == s.id]).dropna().mean()) for s in splits}


def monthly_baseline_table(
    result: BacktestResult,
    panel: pd.DataFrame,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """База случайного дня, стратифицированная по КАЛЕНДАРНОМУ МЕСЯЦУ, против нынешней — по окну.

    Зачем. Нынешняя база сравнивает сигнал со средним днём полугодового окна, а клиент, переводящий
    раз в месяц, выбирает день внутри месяца. Если внутри полугодия есть месяцы систематически лучше
    прочих, часть нынешнего lift — заслуга попадания в удачный месяц, а не в удачный день.

    Месяц с числом дней публикации меньше `MIN_MONTH_DAYS` отдаётся базе окна: месячная оценка по
    трём дням шумнее того, что она измеряет. Сколько раз так вышло — в `fallback_events`.

    Интервал разницы выгод — бутстреп по событиям; события внутри месяца перекрываются горизонтом,
    поэтому интервал оптимистичен, и это тот же оптимизм, что у остальных интервалов отчёта."""
    rows: list[dict] = []
    for corridor, grp in result.signals.groupby("corridor"):
        rate = panel[corridor].dropna()
        days = _test_days(rate, result.splits)
        if not len(days):
            continue
        sod = _split_of_day(rate, result.splits)
        bf_all = labels.benefit_fwd_bps(rate, h)
        base_bf_split = _base_by_split(bf_all, days, sod, result.splits)
        month_of = pd.Series(days.to_period("M"), index=days)
        long_months = {
            m: idx.index for m, idx in month_of.groupby(month_of) if len(idx.index) >= MIN_MONTH_DAYS
        }
        base_bf_month = {m: float(bf_all.reindex(d).dropna().mean()) for m, d in long_months.items()}
        # разметка попадания зависит от сценария: разворот — WINDOW_CLOSING, у него своё условие и
        # свой нулевой допуск (ADR-0003). Считаем по сценарию и кэшируем в пределах коридора.
        by_scenario: dict[str, tuple[pd.Series, dict[int, float], dict[pd.Period, float]]] = {}
        for indicator, ev in grp.groupby("indicator"):
            scenario = str(ev["scenario"].iloc[0])
            if scenario not in by_scenario:
                ha = labels.hit_for_scenario(rate, scenario, h, tol_bps, mode="mean")
                by_scenario[scenario] = (
                    ha,
                    _base_by_split(ha, days, sod, result.splits),
                    {m: float(ha.reindex(d).dropna().mean()) for m, d in long_months.items()},
                )
            hit_all, base_hit_split, base_hit_month = by_scenario[scenario]
            dates = pd.DatetimeIndex(pd.to_datetime(ev["date"])).intersection(days)
            hv = hit_all.reindex(dates)
            ok = hv.notna()
            dates_ok = dates[ok.to_numpy()]
            if not len(dates_ok):
                continue
            split_ids = pd.Series(dates_ok).map(sod)
            months = pd.Series(dates_ok).map(month_of)
            base_split = split_ids.map(base_hit_split).to_numpy(dtype=float)
            base_month = months.map(base_hit_month).to_numpy(dtype=float)
            fallback = np.isnan(base_month)
            base_month = np.where(fallback, base_split, base_month)
            bf_ev = bf_all.reindex(dates_ok).to_numpy(dtype=float)
            bf_split = split_ids.map(base_bf_split).to_numpy(dtype=float)
            bf_month = np.where(fallback, bf_split, months.map(base_bf_month).to_numpy(dtype=float))
            hit = float(hv[ok].mean())
            mean_base_split = float(np.nanmean(base_split))
            mean_base_month = float(np.nanmean(base_month))
            excess_split = bf_ev - bf_split
            excess_month = bf_ev - bf_month
            diff = excess_month - excess_split
            lo, hi = _ci(diff) if len(diff) >= 5 else (np.nan, np.nan)
            rows.append(
                {
                    "indicator": indicator,
                    "corridor": corridor,
                    "events": int(len(dates_ok)),
                    "fallback_events": int(fallback.sum()),
                    "hit_mean_pooled": hit,
                    "base_window": mean_base_split,
                    "base_month": mean_base_month,
                    "lift_window": hit / mean_base_split if mean_base_split > 0 else np.nan,
                    "lift_month": hit / mean_base_month if mean_base_month > 0 else np.nan,
                    "excess_window_bps": float(np.nanmean(excess_split)),
                    "excess_month_bps": float(np.nanmean(excess_month)),
                    "excess_diff_bps": float(np.nanmean(diff)),
                    "excess_diff_ci_lo": lo,
                    "excess_diff_ci_hi": hi,
                }
            )
    return pd.DataFrame(rows).sort_values(["indicator", "corridor"]).reset_index(drop=True)


# ---------------------------------------------------------------- сезонность и налоговый период


TAX_WINDOW = (20, 28)  # числа месяца, на которые приходится продажа валюты экспортёрами под налоги


def day_of_month_table(
    panel: pd.DataFrame,
    splits: list[Split],
    corridors: tuple[str, ...] = CORRIDORS,
) -> pd.DataFrame:
    """Отклонение курса от среднего за свой календарный месяц по числам месяца, бп.

    Гипотеза о механизме сезонности: экспортёры продают валюту под налоги к 28-му, рубль к концу
    месяца крепче, и курс валюты получателя в рублях в эти дни ниже — то есть отправителю дешевле.
    Отрицательное отклонение в окне 20–28 подтверждает механизм.

    Рядом — та же величина после снятия ЛИНЕЙНОГО тренда внутри месяца (`dev_detrended_bps`).
    Рубль за период ослаб, и у растущего ряда начало месяца механически ниже среднего за месяц, а
    конец выше — независимо от всякой сезонности. Линейная составляющая это и снимает; то, что
    остаётся, линейным дрейфом объяснить нельзя.

    Это ДИАГНОСТИКА, а не индикатор: среднее за месяц включает будущие дни этого месяца, и в
    сигнальный контур такая величина попасть не может. Причинная проверка той же гипотезы —
    индикатор сезонности, который смотрит только на прошлые годы."""
    rows = []
    for corridor in corridors:
        if corridor not in panel.columns:
            continue
        rate = panel[corridor].dropna()
        days = _test_days(rate, splits)
        r = rate.reindex(days).dropna()
        if not len(r):
            continue
        month = r.index.to_period("M")
        dev_bps = (r / r.groupby(month).transform("mean") - 1) * 1e4
        detrended = _detrend_within_month(r, month)
        for day in sorted(set(r.index.day)):
            mask = r.index.day == day
            v = dev_bps[mask].to_numpy(dtype=float)
            # у коротких месяцев остатка нет (прямую по двум точкам снимать нечего), и без отсева
            # один такой месяц обратил бы в NaN весь столбец этого числа
            vd = detrended[mask].dropna().to_numpy(dtype=float)
            lo, hi = _ci(v) if len(v) >= 5 else (np.nan, np.nan)
            dlo, dhi = _ci(vd) if len(vd) >= 5 else (np.nan, np.nan)
            rows.append(
                {
                    "corridor": corridor,
                    "day_of_month": int(day),
                    "n_days": int(len(v)),
                    "n_days_detrended": int(len(vd)),
                    "dev_from_month_mean_bps": float(v.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "dev_detrended_bps": float(vd.mean()) if len(vd) else np.nan,
                    "detrended_ci_lo": dlo,
                    "detrended_ci_hi": dhi,
                    "in_tax_window": TAX_WINDOW[0] <= int(day) <= TAX_WINDOW[1],
                }
            )
    return pd.DataFrame(rows)


def _detrend_within_month(r: pd.Series, month: pd.PeriodIndex) -> pd.Series:
    """Остаток после снятия линейного тренда внутри каждого месяца, бп от среднего за месяц.

    Месяц короче трёх дней публикации остаётся как есть: прямую по двум точкам провести можно,
    но остаток от неё тождественно нулевой и в среднее не должен идти."""
    out = pd.Series(np.nan, index=r.index)
    for _, idx in pd.Series(month, index=r.index).groupby(month):
        d = idx.index
        y = r.reindex(d).to_numpy(dtype=float)
        if len(d) < 3 or not np.isfinite(y).all():
            continue
        x = np.arange(len(d), dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        out.loc[d] = resid / y.mean() * 1e4
    return out


def _weighted(
    g: pd.DataFrame, column: str = "dev_from_month_mean_bps", weight: str = "n_days"
) -> float:
    """Среднее отклонение, взвешенное числом дней: у чисел 29–31 наблюдений меньше.

    Строки без значения выпадают вместе со своим весом: иначе числитель считался бы по одному
    набору чисел месяца, а знаменатель — по другому, и среднее уезжало бы к нулю."""
    ok = g[g[column].notna()]
    n = float(ok[weight].sum())
    return float((ok[column] * ok[weight]).sum() / n) if n else float("nan")


def tax_window_summary(day_table: pd.DataFrame) -> pd.DataFrame:
    """Сводка по гипотезе: отклонение внутри налогового окна против остальных дней месяца."""
    rows = []
    for corridor, g in day_table.groupby("corridor"):
        inside = g[g["in_tax_window"]]
        outside = g[~g["in_tax_window"]]
        n_in = int(inside["n_days"].sum())
        n_out = int(outside["n_days"].sum())
        mean_in = _weighted(inside) if n_in else np.nan
        mean_out = _weighted(outside) if n_out else np.nan
        det_in = _weighted(inside, "dev_detrended_bps", "n_days_detrended") if n_in else np.nan
        det_out = _weighted(outside, "dev_detrended_bps", "n_days_detrended") if n_out else np.nan
        rows.append(
            {
                "corridor": corridor,
                "days_in_window": n_in,
                "days_outside": n_out,
                "dev_in_window_bps": mean_in,
                "dev_outside_bps": mean_out,
                "difference_bps": mean_in - mean_out,
                "detrended_in_window_bps": det_in,
                "detrended_outside_bps": det_out,
                "detrended_difference_bps": det_in - det_out,
                # механизм подтверждается, если в налоговом окне курс НИЖЕ (отправителю дешевле),
                # и подтверждается ЧЕСТНО, только если это выживает снятие тренда
                "supports_hypothesis": bool(mean_in < mean_out and det_in < det_out),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- калибровка против априорных


BLOCK_KEYS: dict[str, list[str]] = {"corridor_split": ["corridor", "split"], "split": ["split"]}
BY_WINDOW_SUFFIX = "_by_window"


def paired_pooled_comparison(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
    n_boot: int = 4000,
    seed: int = 7,
    window_from: str | None = None,
    block: str = "corridor_split",
) -> pd.DataFrame:
    """Два прогона на одних окнах: pooled lift «по среднему» и выгода сверх случайного дня
    (взвешенная событиями) по каждому индикатору и суммарно, разница b − a и её интервал.

    Сравнение парное и блочное: на каждой итерации бутстрепа оба прогона пересобираются по ОДНИМ И
    ТЕМ ЖЕ блокам. Иначе интервал мерил бы ещё и разницу в том, какие полугодия попали в выборку, а
    не разницу прогонов. `window_from` оставляет окна не раньше метки (`2024-01`) — например, после
    среза предобучения внешней модели.

    `block` — что считается наблюдением: `corridor_split` — пара «коридор × окно» (55 блоков на пяти
    коридорах), `split` — окно целиком, суммы по коридорам (11 блоков). Коридоры скоррелированы с
    USD/RUB на 0,83–0,97, и блок по паре считает пять коридоров пятью независимыми наблюдениями —
    интервал выходит уже, чем он есть. Блок по окну честнее к общему фактору, но по 11 блокам
    процентильный интервал груб. Точечные оценки от блока не зависят: суммы одни и те же."""
    if block not in BLOCK_KEYS:
        raise ValueError(f"неизвестный блок {block!r}; допустимы {', '.join(BLOCK_KEYS)}")
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    rng_benefit = np.random.default_rng(seed + 1)

    def blocks(m: pd.DataFrame, indicator: str | None) -> pd.DataFrame:
        s = m[(m["h"] == h) & (m["tol_bps"] == tol_bps)]
        if window_from:
            s = s[s["window"] >= window_from]
        if indicator:
            s = s[s["indicator"] == indicator]
        s = s.dropna(subset=["hit_mean", "base_mean", "n_scored"])
        benefit = s["benefit_excess_bps"].fillna(0.0) if "benefit_excess_bps" in s.columns else 0.0
        n = s["n_scored"]
        s = s.assign(num=s["hit_mean"] * n, den=s["base_mean"] * n, bnum=benefit * n)
        return s.groupby(BLOCK_KEYS[block])[["num", "den", "n_scored", "bnum"]].sum()

    def ratio(x: np.ndarray, num: int, den: int) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x[..., den] > 0, x[..., num] / x[..., den], np.nan)

    names = sorted(set(matrix_a["indicator"]) | set(matrix_b["indicator"]))
    for indicator in [*names, None]:
        ga, gb = blocks(matrix_a, indicator), blocks(matrix_b, indicator)
        keys = sorted(set(ga.index) | set(gb.index))
        if not keys:
            continue
        a = ga.reindex(keys).fillna(0.0).to_numpy(dtype=float)
        b = gb.reindex(keys).fillna(0.0).to_numpy(dtype=float)
        lift_a, lift_b = float(ratio(a.sum(axis=0), 0, 1)), float(ratio(b.sum(axis=0), 0, 1))
        ben_a, ben_b = float(ratio(a.sum(axis=0), 3, 2)), float(ratio(b.sum(axis=0), 3, 2))
        idx = rng.integers(0, len(keys), (n_boot, len(keys)))
        sa, sb = a[idx].sum(axis=1), b[idx].sum(axis=1)
        ok = (sa[:, 1] > 0) & (sb[:, 1] > 0)
        diff = sb[ok, 0] / sb[ok, 1] - sa[ok, 0] / sa[ok, 1]
        lo, hi = (np.percentile(diff, [2.5, 97.5]) if ok.sum() > 1 else (np.nan, np.nan))
        idx_b = rng_benefit.integers(0, len(keys), (n_boot, len(keys)))
        sa, sb = a[idx_b].sum(axis=1), b[idx_b].sum(axis=1)
        okb = (sa[:, 2] > 0) & (sb[:, 2] > 0)
        bdiff = sb[okb, 3] / sb[okb, 2] - sa[okb, 3] / sa[okb, 2]
        blo, bhi = (np.percentile(bdiff, [2.5, 97.5]) if okb.sum() > 1 else (np.nan, np.nan))
        rows.append(
            {
                "indicator": indicator or "all",
                "block": block,
                "blocks": len(keys),
                "events_a": int(a[:, 2].sum()),
                "events_b": int(b[:, 2].sum()),
                "lift_a": lift_a,
                "lift_b": lift_b,
                "diff_lift": lift_b - lift_a,
                "diff_lift_ci_lo": float(lo),
                "diff_lift_ci_hi": float(hi),
                "benefit_a": ben_a,
                "benefit_b": ben_b,
                "diff_benefit": ben_b - ben_a,
                "diff_benefit_ci_lo": float(blo),
                "diff_benefit_ci_hi": float(bhi),
            }
        )
    return pd.DataFrame(rows)


CI_COLUMNS: tuple[str, ...] = (
    "blocks",
    "diff_lift_ci_lo",
    "diff_lift_ci_hi",
    "diff_benefit_ci_lo",
    "diff_benefit_ci_hi",
)


def paired_pooled_both(matrix_a: pd.DataFrame, matrix_b: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Оба интервала рядом: по парам «коридор × окно» (столбцы `paired_pooled_comparison`) и по окнам
    (те же столбцы с суффиксом `_by_window`). Точечные оценки у обоих блоков одни, поэтому
    дублируются только интервалы и число блоков."""
    pair = paired_pooled_comparison(matrix_a, matrix_b, block="corridor_split", **kwargs)
    if not len(pair):
        return pair
    win = paired_pooled_comparison(matrix_a, matrix_b, block="split", **kwargs)
    win = win.set_index("indicator")[list(CI_COLUMNS)].add_suffix(BY_WINDOW_SUFFIX)
    return pair.drop(columns=["block"]).join(win, on="indicator")


def read_interval(
    lo: pd.Series, hi: pd.Series, above: str, below: str, none: str = "разницы нет"
) -> np.ndarray:
    """Вердикт по интервалу разницы b − a: целиком выше нуля — `above`, целиком ниже — `below`."""
    return np.where(lo > 0, above, np.where(hi < 0, below, none))


def calibration_vs_fixed(
    matrix: pd.DataFrame,
    fixed_matrix: pd.DataFrame,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
    n_boot: int = 4000,
    seed: int = 7,
) -> pd.DataFrame:
    """Стоит ли калибровка по сетке своих денег: pooled lift калиброванных правил против априорных —
    `paired_pooled_both` в прежних именах столбцов, интервал по парам «коридор × окно» и рядом
    по окнам (`*_by_window`, 11 блоков).

    Раньше эти интервалы считались вручную и жили только в тексте `docs/STATUS.md` — то есть числа
    отчёта не было, а утверждение было (аудит 03.09). Теперь оно есть здесь и в `README.md`.

    Строка `all` — суммарно по всем правилам. У обучаемого индикатора разницы нет по построению:
    `--fixed-params` трогает только правила, модель обучается одинаково."""
    cmp = paired_pooled_both(matrix, fixed_matrix, h=h, tol_bps=tol_bps, n_boot=n_boot, seed=seed)
    if not len(cmp):
        return cmp
    out = pd.DataFrame(
        {
            "indicator": cmp["indicator"],
            "blocks": cmp["blocks"],
            "events_calibrated": cmp["events_a"],
            "events_fixed": cmp["events_b"],
            "lift_calibrated": cmp["lift_a"],
            "lift_fixed": cmp["lift_b"],
            "diff_fixed_minus_calibrated": cmp["diff_lift"],
            "diff_ci_lo": cmp["diff_lift_ci_lo"],
            "diff_ci_hi": cmp["diff_lift_ci_hi"],
        }
    )
    out["better"] = read_interval(out["diff_ci_lo"], out["diff_ci_hi"], "априорные", "калибровка")
    out["blocks_by_window"] = cmp["blocks" + BY_WINDOW_SUFFIX]
    out["diff_ci_lo_by_window"] = cmp["diff_lift_ci_lo" + BY_WINDOW_SUFFIX]
    out["diff_ci_hi_by_window"] = cmp["diff_lift_ci_hi" + BY_WINDOW_SUFFIX]
    out["better_by_window"] = read_interval(
        out["diff_ci_lo_by_window"], out["diff_ci_hi_by_window"], "априорные", "калибровка"
    )
    return out


# ---------------------------------------------------------------- достижимость уровней вероятности


def confidence_table(matrix: pd.DataFrame, indicator: str = "level") -> pd.DataFrame:
    """Hit rate «не хуже в течение h дней» (строгое прочтение, min) и «по среднему» по горизонтам и
    допускам, взвешенно по событиям окон; рядом — база. Показывает, при каких h и допуске достижимы
    уровни 90/95/99 % из постановки."""
    m = matrix[matrix["indicator"] == indicator]

    def pooled(d: pd.DataFrame, col: str) -> float:
        w = d["n_scored"].fillna(0).to_numpy()
        v = d[col].to_numpy(dtype=float)
        ok = ~np.isnan(v) & (w > 0)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() > 0 else np.nan

    g = m.groupby(["h", "tol_bps"])
    out = pd.DataFrame(
        {
            "events": g["n_scored"].sum(),
            "hit_min": g.apply(lambda d: pooled(d, "hit_rate"), include_groups=False),
            "base_min": g.apply(lambda d: pooled(d, "base_rate"), include_groups=False),
            "hit_mean": g.apply(lambda d: pooled(d, "hit_mean"), include_groups=False),
            "base_mean": g.apply(lambda d: pooled(d, "base_mean"), include_groups=False),
        }
    ).reset_index()
    out["indicator"] = indicator
    return out


def tolerance_for_confidence(
    result: BacktestResult,
    panel: pd.DataFrame,
    indicator: str = "level",
    h: int = CALIBRATION_H,
    levels: tuple[float, ...] = (0.9, 0.95, 0.99),
) -> pd.DataFrame:
    """Какой допуск (бп) нужен, чтобы утверждение «курс не хуже в течение h дней» выполнялось с заданной
    вероятностью: квантили худшего хода курса против клиента после события индикатора и после
    случайного тестового дня."""
    rows = []
    for corridor in [c for c in CORRIDORS if c in result.signals["corridor"].unique()]:
        rate = panel[corridor].dropna()
        adverse = (1 - labels.future_min(rate, h) / rate) * 1e4  # бп, > 0 — курс уходил ниже a_T
        days = _test_days(rate, result.splits)
        sig = result.signals[
            (result.signals["corridor"] == corridor) & (result.signals["indicator"] == indicator)
        ]
        ev = adverse.reindex(pd.DatetimeIndex(sig["date"])).dropna()
        rnd = adverse.reindex(days).dropna()
        row = {"corridor": corridor, "indicator": indicator, "h": h, "n_events": int(len(ev))}
        for lv in levels:
            row[f"tol_bps_for_{int(lv * 100)}_events"] = float(ev.quantile(lv)) if len(ev) else np.nan
            row[f"tol_bps_for_{int(lv * 100)}_random_day"] = float(rnd.quantile(lv)) if len(rnd) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- фронтир «частота — точность» (ADR-0008)


def frontier_table(
    panel: pd.DataFrame,
    corridor: str,
    splits: list[Split],
    grid: list[dict[str, Any]] | None = None,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
    exclude_shock: bool = False,
) -> pd.DataFrame:
    """Каждая точка сетки уровня — фиксированные параметры, оценённые на всех тестовых окнах
    (без выбора, поэтому без смещения отбора). База случайного дня — окна событий, взвешенно по их
    числу. Кривая нужна, чтобы увидеть, что стоит частота: рабочая точка калибровки отмечается
    отдельно."""
    grid = grid or FRONTIER_GRID
    if exclude_shock:
        ids = shock_split_ids(splits)
        splits = [s for s in splits if s.id not in ids]
    full = panel  # разогрев по всей истории, как у рабочей точки калибровки (иначе точки несопоставимы)
    rate = full[corridor].dropna()
    ctx = enrich_context(rate, full[[c for c in CONTEXT if c in full.columns]])
    for w in sorted({g["window"] for g in grid}):
        if f"_rank_{w}" not in ctx.columns:
            ctx[f"_rank_{w}"] = rolling_pct_rank(rate, w)
            ctx[f"_dsm_{w}"] = rolling_days_since_min(rate, w)
    days = _test_days(rate, splits)
    sod = _split_of_day(rate, splits)
    weeks = _test_weeks(rate, splits)
    months = _test_months(splits)
    hit_mean_all = labels.hit_for_scenario(rate, BUY_NOW, h, tol_bps, mode="mean")
    hit_min_all = labels.hit_for_scenario(rate, BUY_NOW, h, tol_bps, mode="min")
    bf = labels.benefit_fwd_bps(rate, h)
    base_mean = {sp.id: float(hit_mean_all.loc[sp.test_start : sp.test_end].dropna().mean()) for sp in splits}
    base_min = {sp.id: float(hit_min_all.loc[sp.test_start : sp.test_end].dropna().mean()) for sp in splits}
    base_bf = {sp.id: float(bf.loc[sp.test_start : sp.test_end].dropna().mean()) for sp in splits}
    rows = []
    for params in grid:
        ev = Level(**params).compute(rate, ctx)["signal"]
        ev_days = ev[ev.astype(bool)].index.intersection(days)
        n = int(len(ev_days))
        hm = hit_mean_all.reindex(ev_days).dropna()
        hn = hit_min_all.reindex(ev_days).dropna()
        sp_m = pd.Series(hm.index).map(sod)
        sp_n = pd.Series(hn.index).map(sod)
        bm = float(sp_m.map(base_mean).mean()) if len(hm) else np.nan
        bn = float(sp_n.map(base_min).mean()) if len(hn) else np.nan
        ex = bf.reindex(ev_days) - pd.Series(ev_days).map(sod).map(base_bf).to_numpy()
        ex = ex.dropna()
        busy = set(ev_days.to_period("M"))
        rows.append(
            {
                **params,
                "n_events": n,
                "freq_per_week": n / weeks if weeks else np.nan,
                "hit_mean": float(hm.mean()) if len(hm) else np.nan,
                "base_mean": bm,
                "lift_mean": float(hm.mean()) / bm if len(hm) and bm and bm > 0 else np.nan,
                "hit_min": float(hn.mean()) if len(hn) else np.nan,
                "lift_min": float(hn.mean()) / bn if len(hn) and bn and bn > 0 else np.nan,
                "benefit_excess_bps": float(ex.mean()) if len(ex) else np.nan,
                "empty_month_share": sum(1 for m in months if m not in busy) / len(months)
                if months
                else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["corridor"] = corridor
    df["on_frontier"] = _pareto_flags(df, "freq_per_week", "lift_mean")
    return df


MIN_EVENTS_POINT = 30  # точка сетки с меньшим числом событий на фронтир и в границы не идёт


def _pareto_flags(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Верхняя огибающая: точка на фронтире, если нет точки с не меньшей частотой и большим lift."""
    flags = pd.Series(False, index=df.index)
    best = -np.inf
    for i in df.sort_values([x, y], ascending=[False, False]).index:
        yv = df.at[i, y]
        if not np.isnan(yv) and yv > best and df.at[i, "n_events"] >= MIN_EVENTS_POINT:
            flags.at[i] = True
            best = yv
    return flags


def _pooled_point(rows: pd.DataFrame, weeks: float) -> tuple[float, float] | None:
    """(частота, lift «по среднему») по окнам той же агрегацией, что и фронтир: hit и база
    взвешены по числу событий окна."""
    w = rows["n_scored"].fillna(0).to_numpy(dtype=float)
    hit = rows["hit_mean"].to_numpy(dtype=float)
    base = rows["base_mean"].to_numpy(dtype=float)
    ok = (w > 0) & ~np.isnan(hit) & ~np.isnan(base)
    if not ok.any() or weeks <= 0:
        return None
    lift = (hit[ok] * w[ok]).sum() / (base[ok] * w[ok]).sum()
    return float(rows["n_events"].sum() / weeks), float(lift)


def operating_points(
    result: BacktestResult,
    panel: pd.DataFrame,
    stream_matrix: pd.DataFrame | None,
    corridor: str,
    stream_shape: pd.DataFrame | None = None,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> dict[str, tuple]:
    """Рабочие точки для отметки на фронтире: калиброванный уровень и итоговый поток, агрегированные
    так же, как точки сетки. У точки потока точность — по пушам BUY_NOW (у WINDOW_CLOSING другая
    метка попадания), а частота — по всем пушам коридора из `stream_shape`, потому что полоса 1–2 в
    неделю относится к коридору целиком (ADR-0006 п. 1); без формы потока — частота BUY_NOW."""
    pts: dict[str, tuple] = {}
    weeks = _test_weeks(panel[corridor].dropna(), result.splits)
    m = result.matrix
    lv = m[
        (m["indicator"] == "level") & (m["corridor"] == corridor) & (m["h"] == h) & (m["tol_bps"] == tol_bps)
    ]
    pt = _pooled_point(lv, weeks) if len(lv) else None
    if pt:
        pts["level (калибровка walk-forward)"] = pt
    if stream_matrix is not None and len(stream_matrix):
        sm = stream_matrix
        st = sm[
            (sm["corridor"] == corridor)
            & (sm["scenario"] == BUY_NOW)
            & (sm["h"] == h)
            & (sm["tol_bps"] == tol_bps)
        ]
        pt = _pooled_point(st, weeks) if len(st) else None
        if pt:
            label = "итоговый поток BUY_NOW"
            if stream_shape is not None and len(stream_shape) and weeks:
                total = float(stream_shape.loc[stream_shape["corridor"] == corridor, "pushes"].sum())
                pt = (total / weeks, pt[1])
                label = "итоговый поток (частота — все пуши, точность — BUY_NOW)"
            pts[label] = pt
    return pts


def plot_frontier(
    tables: dict[str, pd.DataFrame], points: dict[str, dict[str, tuple]], path: Path, subtitle: str = ""
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corridors = list(tables)
    fig, axes = plt.subplots(1, len(corridors), figsize=(4.2 * len(corridors), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    lo, hi = FREQUENCY_BAND
    for ax, corridor in zip(axes, corridors, strict=True):
        df = tables[corridor]
        ok = df["n_events"] >= MIN_EVENTS_POINT
        ax.axvspan(lo, hi, color="tab:green", alpha=0.08, label="полоса 1–2 в неделю")
        ax.axhline(1.0, color="grey", linewidth=0.8)
        ax.axhline(1.3, color="grey", linewidth=0.8, linestyle="--")
        ax.scatter(
            df.loc[ok, "freq_per_week"],
            df.loc[ok, "lift_mean"],
            s=10,
            color="lightgrey",
            label="точки сетки (n ≥ 30)",
        )
        fr = df[df["on_frontier"]].sort_values("freq_per_week")
        ax.plot(fr["freq_per_week"], fr["lift_mean"], color="tab:blue", linewidth=1.5, label="фронтир")
        markers = {
            "level (калибровка walk-forward)": ("o", "tab:blue"),
            "итоговый поток BUY_NOW": ("*", "tab:red"),
        }
        for name, (x, y) in points.get(corridor, {}).items():
            m, c = markers.get(name, ("x", "black"))
            ax.scatter([x], [y], marker=m, s=90, color=c, zorder=5, edgecolor="black", label=name)
        ax.set_title(corridor)
        ax.set_xlabel("сигналов в неделю")
        ax.set_xlim(0, max(2.6, float(df["freq_per_week"].max()) * 1.02))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("lift «по среднему», h = 20, допуск 25 бп")
    axes[-1].legend(loc="upper right", fontsize=7)
    title = "Что стоит частота: индикатор уровня на тестовых окнах 2021–2026 (фиксированные параметры сетки)"
    fig.suptitle(title + (f" — {subtitle}" if subtitle else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- запись


def write_analysis(
    result: BacktestResult,
    panel: pd.DataFrame,
    out_dir: Path | None = None,
    source: str = "KZT",
    k: int = 10,
) -> Path:
    out = (out_dir or (repo_root() / "reports" / "latest")) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pw = price_of_waiting_table(result, panel, k=k)
    pw.to_csv(out / "price_of_waiting.csv", index=False)
    pw_ns = price_of_waiting_table(result, panel, k=k, exclude_shock=True)
    pw_ns.to_csv(out / "price_of_waiting_no_shock.csv", index=False)
    tr = transfer_table(result, panel, source=source)
    tr.to_csv(out / "transfer_matrix.csv", index=False)
    cmp_ = transfer_compare(result, tr)
    cmp_.to_csv(out / "transfer_compare.csv", index=False)
    no_shock = summary_without_shock(result)
    no_shock.to_csv(out / "summary_no_shock_h20_tol25.csv", index=False)
    conf = pd.concat(
        [confidence_table(result.matrix, ind) for ind in result.matrix["indicator"].unique()],
        ignore_index=True,
    )
    conf.to_csv(out / "confidence_by_h_tol.csv", index=False)
    tol_needed = tolerance_for_confidence(result, panel)
    tol_needed.to_csv(out / "tolerance_for_confidence.csv", index=False)
    monthly = monthly_baseline_table(result, panel)
    monthly.to_csv(out / "monthly_baseline.csv", index=False)
    dom = day_of_month_table(panel, result.splits)
    dom.to_csv(out / "day_of_month.csv", index=False)
    tax = tax_window_summary(dom) if len(dom) else pd.DataFrame()
    tax.to_csv(out / "tax_window.csv", index=False)
    # контрольный прогон — отдельная команда; без него сравнение пропускается, а не выдумывается
    fixed_path = repo_root() / "reports" / "fixed" / "matrix.csv"
    fixed_cmp = pd.DataFrame()
    if fixed_path.exists():
        fixed_cmp = calibration_vs_fixed(result.matrix, pd.read_csv(fixed_path))
    if len(fixed_cmp):
        fixed_cmp.to_csv(out / "calibration_vs_fixed.csv", index=False)
    if len(dom):
        plot_day_of_month(dom, out / "day_of_month.png")
    stream_path = out.parent / "stream_matrix.csv"
    sm = pd.read_csv(stream_path) if stream_path.exists() else None
    shape_path = out.parent / "stream_shape.csv"
    shape = pd.read_csv(shape_path) if shape_path.exists() and shape_path.stat().st_size > 1 else None
    stream_ns = pd.DataFrame()
    if sm is not None and len(sm):
        stream_ns = stream_summary_without_shock(sm, result.splits)
        stream_ns.to_csv(out / "stream_summary_no_shock_h20_tol25.csv", index=False)
    extra = _final_analyses(result, panel, out, sm)
    tables: dict[str, pd.DataFrame] = {}
    points: dict[str, dict[str, tuple]] = {}
    tables_ns: dict[str, pd.DataFrame] = {}
    for corridor in [c for c in CORRIDORS if c in result.signals["corridor"].unique()]:
        tables[corridor] = frontier_table(panel, corridor, result.splits)
        tables[corridor].to_csv(out / f"frontier_{corridor}.csv", index=False)
        tables_ns[corridor] = frontier_table(panel, corridor, result.splits, exclude_shock=True)
        tables_ns[corridor].to_csv(out / f"frontier_{corridor}_no_shock.csv", index=False)
        points[corridor] = operating_points(result, panel, sm, corridor, shape)
    if tables:
        plot_frontier(tables, points, out / "frontier.png")
        plot_frontier(tables_ns, {}, out / "frontier_no_shock.png", subtitle="без окон шокового режима 2022")
    (out / "README.md").write_text(
        _analysis_readme(
            pw,
            pw_ns,
            cmp_,
            no_shock,
            stream_ns,
            conf,
            tol_needed,
            tables,
            tables_ns,
            points,
            monthly,
            tax,
            fixed_cmp,
            extra,
        ),
        encoding="utf-8",
    )
    return out


def _final_analyses(
    result: BacktestResult, panel: pd.DataFrame, out: Path, stream_matrix: pd.DataFrame | None
) -> dict[str, pd.DataFrame]:
    """Три таблицы, добавленные 03.09 вечером (💬 пункты 4, 3, 6): календарное правило как база стека,
    выживаемость пуша до исполнения, разворот как сожаление. Импорт локальный: модули сами читают
    помощников отсюда."""
    from fxmoment.baselines import calendar_matrix, calendar_summary, calendar_vs_stack
    from fxmoment.execution import execution_survival_table, load_spreads
    from fxmoment.regret import reversal_regret_table

    ran = tuple(c for c in CORRIDORS if c in result.signals["corridor"].unique())
    cal = calendar_matrix(panel, result.splits, corridors=ran)
    cal.to_csv(out / "calendar_rule_windows.csv", index=False)
    cal_sum = calendar_summary(cal)
    cal_sum.to_csv(out / "calendar_rule.csv", index=False)
    cal_cmp = calendar_vs_stack(cal, result.matrix, stream_matrix)
    cal_cmp.to_csv(out / "calendar_vs_stack.csv", index=False)
    dec_path = out.parent / "stream_decisions.csv"
    decided = (
        pd.read_csv(dec_path, parse_dates=["date"])
        if dec_path.exists() and dec_path.stat().st_size > 1
        else None
    )
    survival = execution_survival_table(result.signals, decided, panel, load_spreads(panel))
    survival.to_csv(out / "execution_survival.csv", index=False)
    regret = reversal_regret_table(result.signals, decided, panel, result.splits)
    regret.to_csv(out / "reversal_regret.csv", index=False)
    return {"calendar": cal_sum, "calendar_vs_stack": cal_cmp, "survival": survival, "regret": regret}


def _py(v: Any) -> Any:
    """numpy-скаляр → python-скаляр для JSON."""
    return v.item() if hasattr(v, "item") else v


def _md(df: pd.DataFrame) -> str:
    from fxmoment.report import _md_table

    return _md_table(df.round(3)) if len(df) else "нет данных"


def _band_table(tables: dict[str, pd.DataFrame], points: dict[str, dict[str, tuple]]) -> pd.DataFrame:
    band_rows = []
    lo, hi = FREQUENCY_BAND
    for corridor, df in tables.items():
        enough = df["n_events"] >= MIN_EVENTS_POINT
        inband = df[(df["freq_per_week"] >= lo) & (df["freq_per_week"] <= hi) & enough]
        best = inband.sort_values("lift_mean", ascending=False).head(1)
        rare = df[(df["freq_per_week"] < 0.5) & enough].sort_values("lift_mean", ascending=False).head(1)
        band_rows.append(
            {
                "corridor": corridor,
                "bound_in_band_freq": float(best["freq_per_week"].iloc[0]) if len(best) else np.nan,
                "bound_in_band_lift_mean": float(best["lift_mean"].iloc[0]) if len(best) else np.nan,
                "bound_in_band_excess_bps": float(best["benefit_excess_bps"].iloc[0])
                if len(best)
                else np.nan,
                "bound_rare_freq": float(rare["freq_per_week"].iloc[0]) if len(rare) else np.nan,
                "bound_rare_lift_mean": float(rare["lift_mean"].iloc[0]) if len(rare) else np.nan,
                "bound_rare_excess_bps": float(rare["benefit_excess_bps"].iloc[0]) if len(rare) else np.nan,
            }
        )
        # рабочая точка известна только для полного набора окон: она считается по потоку, который
        # окон шокового режима не исключает. Без неё столбца нет вовсе — пустой столбец в отчёте
        # читался бы как «калибровка не дала точки», а не как «здесь этот вопрос не задан».
        if points:
            pt = points.get(corridor, {}).get("level (калибровка walk-forward)", (np.nan, np.nan))
            band_rows[-1]["calibrated_freq"] = pt[0]
            band_rows[-1]["calibrated_lift_mean"] = pt[1]
    return pd.DataFrame(band_rows)


def plot_day_of_month(day_table: pd.DataFrame, path: Path) -> None:
    """Форма месяца: отклонение курса по числам, до и после снятия линейного тренда."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_raw, ax_det) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, col, title in (
        (ax_raw, "dev_from_month_mean_bps", "Отклонение от среднего за месяц"),
        (ax_det, "dev_detrended_bps", "То же после снятия линейного тренда"),
    ):
        for corridor, g in day_table.groupby("corridor"):
            g = g.sort_values("day_of_month")
            ax.plot(g["day_of_month"], g[col], marker=".", linewidth=1, alpha=0.6, label=str(corridor))
        mean = day_table.groupby("day_of_month")[col].mean()
        ax.plot(mean.index, mean.to_numpy(), color="black", linewidth=2.2, label="среднее")
        ax.axhline(0, color="grey", linewidth=1)
        ax.axvspan(TAX_WINDOW[0], TAX_WINDOW[1], color="tab:orange", alpha=0.12)
        ax.set_title(title)
        ax.set_xlabel("число месяца (оранжевое — окно налогов, 20–28)")
        ax.set_ylabel("бп; ниже нуля — отправителю дешевле")
        ax.grid(alpha=0.3)
    ax_raw.legend(fontsize=8, ncol=2)
    fig.suptitle("Форма месяца: когда внутри месяца курс ниже своего среднего")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _analysis_readme(
    pw: pd.DataFrame,
    pw_ns: pd.DataFrame,
    cmp_: pd.DataFrame,
    no_shock: pd.DataFrame,
    stream_ns: pd.DataFrame,
    conf: pd.DataFrame,
    tol_needed: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    tables_ns: dict[str, pd.DataFrame],
    points: dict[str, dict[str, tuple]],
    monthly: pd.DataFrame,
    tax: pd.DataFrame,
    fixed_cmp: pd.DataFrame | None = None,
    extra: dict[str, pd.DataFrame] | None = None,
) -> str:
    from fxmoment.execution import STREAM_LABEL

    extra = extra or {}
    band = _band_table(tables, points)
    band_ns = _band_table(tables_ns, {})
    cal = extra.get("calendar", pd.DataFrame())
    cal_first = cal[cal["indicator"].str.endswith(":first")] if len(cal) else cal
    cal_all = cal[cal["indicator"].str.endswith(":all") & (cal["corridor"] == "all")] if len(cal) else cal
    cal_cmp = extra.get("calendar_vs_stack", pd.DataFrame())
    cmp_show = [
        "rule",
        "stack",
        "events_rule",
        "events_stack",
        "lift_rule",
        "lift_stack",
        "diff_lift",
        "diff_lift_ci_lo",
        "diff_lift_ci_hi",
        "verdict_lift",
        "diff_lift_ci_lo_by_window",
        "diff_lift_ci_hi_by_window",
        "verdict_lift_by_window",
        "benefit_rule",
        "benefit_stack",
        "diff_benefit",
        "verdict_benefit",
        "verdict_benefit_by_window",
    ]
    surv = extra.get("survival", pd.DataFrame())
    surv_stream = surv[surv["source"] == STREAM_LABEL] if len(surv) else surv
    surv_show = [
        "corridor",
        "spread_source",
        "n",
        "gate_fact_T",
        "gate_fact_T1",
        "gate_survival",
        "own_fact_n",
        "own_fact_T1",
        "own_survival",
        "n_scored",
        "benefit_fwd_bps",
        "benefit_fwd_exec_bps",
        "benefit_fwd_at_p90_spread_bps",
        "hit_mean",
        "hit_mean_exec",
        "share_positive",
        "share_positive_exec",
        "spread_mean_bps",
        "spread_p90_dev_bps",
    ]
    regret = extra.get("regret", pd.DataFrame())
    pw_cols = [
        "corridor",
        "n_fast",
        "confirmed_share",
        "days_waited_median",
        "price_of_waiting_bps",
        "price_of_waiting_ci_lo",
        "price_of_waiting_ci_hi",
        "price_of_waiting_all_bps",
        "price_of_waiting_all_ci_lo",
        "price_of_waiting_all_ci_hi",
        "fast_hit_mean",
        "slow_hit_mean",
        "fast_lift_mean",
        "slow_lift_mean",
        "send_now_excess_bps",
        "send_now_excess_ci_lo",
        "send_now_excess_ci_hi",
        "wait_excess_per_fast_bps",
        "wait_excess_ci_lo",
        "wait_excess_ci_hi",
        "wait_excess_per_sent_bps",
        "slow_excess_bps",
        "verdict",
        "difference_within_ci",
    ]
    cmp_cols = [
        "indicator",
        "corridor",
        "own_lift_mean_median",
        "transferred_lift_mean_median",
        "lift_drop",
        "own_excess_median_bps",
        "transferred_excess_median_bps",
        "own_freq_per_week_median",
        "transferred_freq_per_week_median",
    ]
    from fxmoment.report import stamp

    prov = backtest_provenance()
    lines = [
        "# Анализы поверх бэктеста (h = 20, допуск 25 бп, тестовые окна walk-forward)",
        "",
        stamp(),
        f"Бэктест, по которому считались анализы: код `{prov.get('code', '?')}`, снимок "
        f"{prov.get('fetched_at_utc', '?')}, собран {prov.get('built_at_utc', '?')}."
        if prov
        else "Бэктест без `provenance.json` (собран старым кодом) — его хеш неизвестен.",
        "",
        "## Цена ожидания: быстрый `momentum` → подтверждение медленным `level` (пункт 6 постановки)",
        "",
        "`price_of_waiting_bps` — изменение курса между днём быстрого сигнала и днём подтверждения "
        "(> 0 — курс вырос, ожидание стоило денег; < 0 — курс продолжил падать, ожидание принесло); "
        "величина условна на подтверждение, которое само наступает при падении курса, поэтому рядом "
        "`price_of_waiting_all_bps` — по всем быстрым сигналам, для неподтверждённых изменение за те же "
        "k дней. "
        "`send_now_excess_bps` — выгода сверх случайного дня, если слать по быстрому сразу; "
        "`wait_excess_per_fast_bps` — если ждать подтверждения, на один быстрый сигнал (без подтверждения "
        "пуша нет, вклад 0 — непотраченный слот здесь не премируется); `wait_excess_per_sent_bps` — та же "
        "стратегия на один отправленный пуш (уникальные дни подтверждения, `n_sent_wait`). Вердикт — по "
        "точечным оценкам выгоды на быстрый сигнал; `difference_within_ci` = True — интервалы стратегий "
        "перекрываются, и вердикт читать как наклон, а не как доказательство. «Обе не лучше случайного "
        "дня» — обе выгоды ≤ 0. "
        "Быстрые сигналы, у которых k дней не поместились в ряд, в цену не входят (`n_fast_full_horizon`).",
        "",
        _md(pw[[c for c in pw_cols if c in pw.columns]]) if len(pw) else "нет данных",
        "",
        "То же без окон шокового режима 2022:",
        "",
        _md(pw_ns[[c for c in pw_cols if c in pw_ns.columns]]) if len(pw_ns) else "нет данных",
        "",
        "## Перенос параметров между коридорами (ADR-0004 п. 3)",
        "",
        "Параметры каждого окна, откалиброванные на коридоре-источнике, применены ко всем коридорам. "
        "`lift_drop` = своя калибровка − перенесённая; порог устойчивости из концепции — падение < 0,2.",
        "",
        _md(cmp_[[c for c in cmp_cols if c in cmp_.columns]]) if len(cmp_) else "нет данных",
        "",
        "## Без шокового режима 2022 (окна, пересекающиеся с 24.02–31.07.2022, исключены)",
        "",
        _md(
            no_shock[
                [
                    c
                    for c in (
                        "indicator",
                        "corridor",
                        "windows",
                        "events",
                        "lift_mean_median",
                        "share_lift_mean_ge_1_3",
                        "share_lift_mean_lt_1",
                        "benefit_excess_median_bps",
                        "freq_per_week_median",
                    )
                    if c in no_shock.columns
                ]
            ]
        )
        if len(no_shock)
        else "нет данных",
        "",
        "Итоговый поток без шокового режима:",
        "",
        _md(stream_ns) if len(stream_ns) else "нет данных",
        "",
        "## Достижимость уровней 90/95/99 % (индикатор уровня, все окна, взвешенно по событиям)",
        "",
        "`hit_min` — строгое прочтение «курс не хуже в течение h дней» с допуском, `hit_mean` — «не хуже "
        "среднего за h дней». Уровень считается достижимым, если hit при этих h и допуске не ниже него.",
        "",
        _md(conf[conf["indicator"] == "level"].drop(columns=["indicator"])) if len(conf) else "нет данных",
        "",
        "Какой допуск нужен, чтобы «не хуже в течение 20 дней» выполнялось с вероятностью 90/95/99 % "
        "(квантили худшего хода курса после события уровня и после случайного дня, бп):",
        "",
        _md(tol_needed) if len(tol_needed) else "нет данных",
        "",
        "## Фронтир «частота — точность» индикатора уровня (ADR-0008)",
        "",
        "Каждая точка — фиксированные параметры сетки на всех тестовых окнах, без отбора внутри точки; "
        "разогрев индикатора — по всей истории, как у рабочей точки. "
        "`bound_in_band_*` — точка с максимальным lift среди точек в полосе 1–2 в неделю с ≥ 30 событиями, "
        "`bound_rare_*` — то же среди редких (< 0,5); `*_excess_bps` — выгода этой точки, а не максимум "
        "выгоды. Это **апостериорная верхняя граница** по lift на тех же окнах, не рабочая точка и не "
        "рекомендация параметров. Рабочая точка walk-forward — `calibrated_*`; её положение под огибающей — "
        "цена честной калибровки. График — `frontier.png`, точки — `frontier_<коридор>.csv`.",
        "",
        _md(band) if len(band) else "нет данных",
        "",
        "То же без окон шокового режима 2022 (`frontier_no_shock.png`); рабочей точки здесь нет — "
        "она считается по итоговому потоку, а он шоковые окна не исключает:",
        "",
        _md(band_ns) if len(band_ns) else "нет данных",
        "",
        "## База: полугодовое окно против календарного месяца",
        "",
        "Нынешняя база сравнивает сигнал со средним днём полугодового окна. Клиент, переводящий раз в "
        "месяц, выбирает день внутри месяца, и его альтернатива — типичный день ЭТОГО месяца. Две базы "
        "отвечают на разные вопросы, и обе стоят в отчёте: месячная — «попали ли мы в удачный день», "
        "оконная — «выиграл ли клиент за полугодие». `excess_diff_*` — насколько выгода растёт при "
        "переходе на месячную базу, с бутстреп-интервалом по событиям.",
        "",
        _md(monthly) if len(monthly) else "нет данных",
        "",
        "## Сезонность и налоговый период",
        "",
        "Гипотеза о механизме: экспортёры продают валюту под налоги к 28-му числу, рубль к концу месяца "
        "крепче, и курс валюты получателя в рублях в эти дни ниже — отправителю дешевле. "
        "`dev_in_window_bps` — отклонение от среднего за свой месяц внутри окна 20–28 числа, "
        "`detrended_*` — то же после снятия линейного тренда внутри месяца: у растущего ряда начало "
        "месяца механически ниже среднего, а конец выше, и без этой поправки часть эффекта была бы "
        "дрейфом рубля. Числа по дням — `day_of_month.csv`, график — `day_of_month.png`.",
        "",
        _md(tax) if len(tax) else "нет данных",
        "",
        "Это диагностика, а не индикатор: среднее за месяц включает будущие дни этого месяца. "
        "Причинная проверка той же гипотезы — индикатор сезонности, он смотрит только на прошлые годы. "
        "`dev_outside_bps` самостоятельной информации не несёт: отклонения от среднего за свой месяц "
        "в сумме по всем числам дают ноль, поэтому оно равно `dev_in_window_bps`, взятому с обратным "
        "знаком и умноженному на отношение числа дней. Столбец оставлен для читаемости строки, а не "
        "как второй замер.",
        "",
        "## Калибровка против априорных параметров",
        "",
        "Стоит ли сетка своих денег. `lift_calibrated` — pooled lift «по среднему» правил, "
        "откалиброванных walk-forward (`reports/latest`); `lift_fixed` — тех же правил на априорных "
        "точках без сетки (`reports/fixed`, команда `backtest --fixed-params`). Интервал разницы — "
        "парный блочный бутстреп: на каждой итерации оба прогона пересобираются по одним и тем же "
        "блокам, иначе интервал мерил бы ещё и разницу в том, какие полугодия попали в выборку. Блок — "
        "пара «коридор × окно» (`diff_ci_*`, 55 блоков) и рядом окно целиком (`*_by_window`, 11 блоков): "
        "коридоры скоррелированы с USD/RUB на 0,83–0,97, и первый интервал считает пять почти одинаковых "
        "коридоров пятью наблюдениями, то есть уже, чем он есть; второй честнее к общему фактору, но по 11 "
        "блокам груб. Столбцы `better*` читают интервал: «разницы нет» значит, что он содержит ноль.",
        "",
        "У обучаемого индикатора разницы нет по построению: `--fixed-params` трогает только правила, "
        "модель обучается одинаково. Контроль смещён в пользу априорных точек — они заданы до сетки, "
        "но не до данных.",
        "",
        (
            _md(fixed_cmp.round(3))
            if fixed_cmp is not None and len(fixed_cmp)
            else "контрольного прогона нет — сделайте `fxmoment backtest --fixed-params`"
        ),
        "",
        "## Календарное правило как прозрачная база",
        "",
        "Правило без индикаторов, параметров и модели. `day20-25` — первый день публикации с 20-го по "
        "25-е число каждого месяца; `day20-28` — то же для налогового окна; `day25` — первый день "
        "публикации с 25-го. Режим `first` — один пуш в месяц (у окна 20–28 его нет: первый день с 20-го тот "
        "же, что у 20–25); `all` — каждый день окна: клиент переводит в любой его день, это правило для "
        "экрана, а не пуш. Оценка той же метрикой и на тех же окнах, что у "
        "индикаторов; медианы — по активным окнам, pooled — взвешенно по событиям; строка `all` — по всем "
        "коридорам. По окнам — `calendar_rule_windows.csv`.",
        "",
        _md(cal_first) if len(cal_first) else "нет данных",
        "",
        "Режим `all`, по всем коридорам:",
        "",
        _md(cal_all) if len(cal_all) else "нет данных",
        "",
        "Стек против правила (режим `first`): парный блочный бутстреп, `diff_*` = стек − правило, интервалы "
        "по парам «коридор × окно» и по окнам (`*_by_window`). Вердикт «разницы нет» — интервал содержит "
        "ноль, и стек от календаря не отличим. `stream BUY_NOW` — пуши итогового потока после политики. "
        "Все столбцы — `calendar_vs_stack.csv`.",
        "",
        _md(cal_cmp[[c for c in cmp_show if c in cal_cmp.columns]]) if len(cal_cmp) else "нет данных",
        "",
        "## Выживаемость сигнала до исполнения",
        "",
        "Пуш уходит вечером T по курсу действия, клиент переводит на T+1 по курсу приложения. "
        "`gate_fact_T` и `gate_fact_T1` — доля пушей, у которых гейт уровня ADR-0005 (`rank120 ≤ 0,2`) "
        "истинен на T и на T+1; `gate_survival` — доля переживших среди тех, у кого он был истинен на T. "
        "`own_*` — то же для собственного процентильного факта индикатора, на который ссылается текст "
        "(уровень, провал, гейт обучаемого; у сезонности и моментума такого факта нет, `own_fact_n` — "
        "сколько пушей его имеют). `*_exec` — попадание «по среднему», выгода вперёд и доля пушей с "
        "положительной выгодой при исполнении по курсу a_T·(1 + ε), где ε — отклонение сегодняшнего "
        "расхождения приложения с фиксингом от его среднего, со знаком. Среднее (`spread_mean_bps`) снято: "
        "постоянный спред одинаково удорожает день пуша и любой другой день клиента и на выбор дня не "
        "влияет. Распределение взято из сверки биржи с фиксингом за период оценки (`spread_source`: CNY — "
        "пара, где оба источника мерят один курс; KZT и AMD — их собственные пары, верхняя граница "
        "разброса), ожидание — по всему распределению; `benefit_fwd_at_p90_spread_bps` — при ε в девятом "
        "дециле (`spread_p90_dev_bps`). Здесь пуши `BUY_NOW` итогового потока; по индикаторам — "
        "`execution_survival.csv`.",
        "",
        _md(surv_stream[[c for c in surv_show if c in surv_stream.columns]])
        if len(surv_stream)
        else "нет данных",
        "",
        "## Разворот как факт: сожаление с даты последнего BUY_NOW",
        "",
        "«Окно закрывается» меряется не как прогноз, а как факт для того, кто получил `BUY_NOW` и не "
        "перевёл: `regret_*` — изменение курса действия с даты последнего `BUY_NOW` за 20 дней публикации "
        "до разворота, бп (> 0 — ожидание стоило денег); `share_regret_gt_tol` — доля пар с сожалением "
        "больше рабочего допуска 25 бп; `paired_share` — доля разворотов, перед которыми такой `BUY_NOW` "
        "был. `pairing` — по событиям индикаторов (`events`) или по отправленным пушам потока (`stream`). "
        "Прогнозные прочтения рядом, чтобы монетку не прятать: `*_fwd` — через 20 дней курс не ниже (как "
        "в матрице), `*_rest_of_month` — сегодня не хуже среднего оставшихся дней месяца; базы — по всем "
        "дням окна события.",
        "",
        _md(regret) if len(regret) else "нет данных",
    ]
    return "\n".join(lines)
