"""Выживаемость сигнала до исполнения (💬 03.09 вечер, пункт 3).

Пуш уходит вечером T по курсу действия a_T, а клиент переводит на T+1 по курсу приложения, который
живёт в реальном времени. Две меры на каждый пуш `BUY_NOW`.

(а) Факт, на который ссылается пуш, ещё истинен по курсу, опубликованному на T+1: гейт уровня
    ADR-0005 (`rank120 ≤ 0,2`) — один для всех, и собственный процентильный факт индикатора
    (уровень: `pct_rank ≤ pct` в своём окне; провал: то же для отклонения от тренда и оно ниже нуля;
    обучаемый: его гейт). У сезонности и моментума процентильного факта нет, у них эта мера не
    определена.
(б) Попадание и выгода вперёд при курсе исполнения a_T · (1 + ε), где ε — отклонение сегодняшнего
    расхождения курса приложения с фиксингом от его среднего. Постоянная часть расхождения снята:
    она одинаково удорожает и день пуша, и любой другой день клиента, то есть на выбор дня не
    влияет; на выбор влияет только то, что сегодня спред не такой, как обычно. Распределение
    берётся из сверки биржи с фиксингом ЦБ (`data/compare.py`, `reports/intraday/`) за период
    оценки: по умолчанию пара CNY, где оба источника мерят один курс; для коридоров с собственным
    внутридневным рынком (KZT, AMD) рядом их пара — верхняя граница разброса, там биржевая пара и
    кросс-курс ЦБ связаны слабо. Знак сохраняется: ε > 0 — приложение сегодня дороже обычного,
    клиенту хуже. Ожидание берётся по всему эмпирическому распределению, а не по выборке из него —
    то же самое без случайности.

Это число под механику «момент изменился» (`docs/product/star-task.md`): доля пушей, чей факт не
дожил до утра, и выгода, которая остаётся после спреда."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fxmoment import labels
from fxmoment.config import BUY_NOW, CALIBRATION_H, FIRST_TEST, PRIMARY_TOL_BPS
from fxmoment.data.compare import daily_close, diff_bps
from fxmoment.indicators import Dip, Level
from fxmoment.indicators.base import rolling_pct_rank

GATE_WINDOW, GATE_PCT = 120, 0.20  # гейт уровня, ADR-0005 п. 2
SPREAD_REFERENCE = "CNY"  # пара, где биржа и фиксинг мерят один и тот же курс
OWN_MARKET: tuple[str, ...] = ("KZT", "AMD")  # коридоры с внутридневным рынком (сверка 03.09)
MIN_SPREAD_DAYS = 250
STREAM_LABEL = "stream BUY_NOW"
COLUMNS = [
    "corridor",
    "source",
    "spread_source",
    "n",
    "n_next",
    "gate_fact_T",
    "gate_fact_T1",
    "gate_survival",
    "own_fact_n",
    "own_fact_T",
    "own_fact_T1",
    "own_survival",
    "n_scored",
    "benefit_fwd_bps",
    "hit_mean",
    "share_positive",
    "benefit_fwd_exec_bps",
    "hit_mean_exec",
    "share_positive_exec",
    "benefit_fwd_at_p90_spread_bps",
    "spread_days",
    "spread_mean_bps",
    "spread_p90_dev_bps",
]


def spread_samples(
    panel: pd.DataFrame, bar_panel: pd.DataFrame, currency: str, since: str = FIRST_TEST
) -> pd.Series:
    """Расхождение биржевого закрытия дня с фиксингом того же дня, бп со знаком, по дням с `since`
    (период оценки: до него распределение другого рынка)."""
    if currency not in bar_panel.columns or currency not in panel.columns:
        return pd.Series(dtype=float)
    cbr = panel[currency].dropna()
    cbr.index = cbr.index.normalize()
    d = diff_bps(daily_close(bar_panel[currency]), cbr)
    return d.loc[pd.Timestamp(since) :] if len(d) else d


def load_spreads(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Распределения ε по валютам из снимка Мосбиржи; без снимка — пусто, и часть (б) пропускается
    с пустыми столбцами, а не выдумывается."""
    from fxmoment.data.store import MOEX_CSV, load_bar_panel

    if not MOEX_CSV.exists():
        return {}
    bars = load_bar_panel()
    out: dict[str, pd.Series] = {}
    for cur in (SPREAD_REFERENCE, *OWN_MARKET):
        s = spread_samples(panel, bars, cur)
        if len(s) >= MIN_SPREAD_DAYS:
            out[cur] = s
    return out


def own_fact_series(rate: pd.Series, indicator: str, params: dict) -> pd.Series | None:
    """Собственный процентильный факт индикатора как булев ряд на всём ряде; None — факта нет."""
    ctor = {k: v for k, v in params.items() if not k.startswith("_")}
    if indicator == "level" and "pct" in ctor:
        return Level(**ctor).compute(rate)["pct_rank"] <= float(ctor["pct"])
    if indicator == "dip_vs_trend" and "pct" in ctor:
        out = Dip(**ctor).compute(rate)
        return (out["pct_rank"] <= float(ctor["pct"])) & (out["dev_pct"] < 0)
    if indicator == "ml_localmin" and "gate_window" in ctor and "gate_pct" in ctor:
        return rolling_pct_rank(rate, int(ctor["gate_window"])) <= float(ctor["gate_pct"])
    return None


def _pushes(signals: pd.DataFrame, decided: pd.DataFrame | None) -> pd.DataFrame:
    """События `BUY_NOW` по источникам: каждый индикатор из `signals` и отправленные пуши потока."""
    cols = ["date", "corridor", "indicator", "params"]
    ev = signals.loc[signals["scenario"] == BUY_NOW, cols].assign(source=lambda d: d["indicator"])
    parts = [ev]
    if decided is not None and len(decided):
        sent = decided[(decided["decision"] == "sent") & (decided["push_scenario"] == BUY_NOW)]
        parts.append(sent[cols].assign(source=STREAM_LABEL))
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _params(raw: object) -> dict:
    return json.loads(raw) if isinstance(raw, str) and raw else {}


def _per_push(pushes: pd.DataFrame, panel: pd.DataFrame, h: int) -> pd.DataFrame:
    """По каждому пушу: гейт и собственный факт на T и T+1 (NaN — не определён или T+1 вне ряда),
    отношение среднего будущих h курсов к a_T (NaN — горизонт не поместился)."""
    frames: list[pd.DataFrame] = []
    for corridor, grp in pushes.groupby("corridor", sort=True):
        if corridor not in panel.columns:
            continue
        rate = panel[corridor].dropna()
        idx = rate.index
        gate = (rolling_pct_rank(rate, GATE_WINDOW) <= GATE_PCT).to_numpy(dtype=float)
        x_all = (labels.future_mean(rate, h) / rate).to_numpy(dtype=float)
        pos = idx.get_indexer(pd.DatetimeIndex(grp["date"]))
        ok = pos >= 0
        nxt = np.where(ok & (pos + 1 < len(idx)), pos + 1, -1)
        cache: dict[tuple[str, str], np.ndarray | None] = {}
        own_t = np.full(len(grp), np.nan)
        own_t1 = np.full(len(grp), np.nan)
        for i, (ind, raw) in enumerate(zip(grp["indicator"], grp["params"], strict=True)):
            key = (str(ind), str(raw))
            if key not in cache:
                s = own_fact_series(rate, str(ind), _params(raw))
                cache[key] = None if s is None else s.fillna(False).to_numpy(dtype=bool)
            arr = cache[key]
            if arr is None or not ok[i]:
                continue
            own_t[i] = float(arr[pos[i]])
            if nxt[i] >= 0:
                own_t1[i] = float(arr[nxt[i]])
        safe_pos, safe_nxt = np.maximum(pos, 0), np.maximum(nxt, 0)
        frames.append(
            pd.DataFrame(
                {
                    "corridor": corridor,
                    "source": grp["source"].to_numpy(),
                    "date": grp["date"].to_numpy(),
                    "gate_T": np.where(ok, gate[safe_pos], np.nan),
                    "gate_T1": np.where(nxt >= 0, gate[safe_nxt], np.nan),
                    "own_T": own_t,
                    "own_T1": own_t1,
                    "x": np.where(ok, x_all[safe_pos], np.nan),
                }
            )
        )
    columns = ["corridor", "source", "date", "gate_T", "gate_T1", "own_T", "own_T1", "x"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _share(values: pd.Series) -> float:
    return float(values.mean()) if values.notna().any() else np.nan


def _survival(pp: pd.DataFrame) -> dict:
    g_t, g_t1 = pp["gate_T"], pp["gate_T1"]
    both = g_t.notna() & g_t1.notna()
    held = both & (g_t == 1.0)
    o_t, o_t1 = pp["own_T"], pp["own_T1"]
    oboth = o_t.notna() & o_t1.notna()
    oheld = oboth & (o_t == 1.0)
    return {
        "n": int(len(pp)),
        "n_next": int(both.sum()),
        "gate_fact_T": _share(g_t),
        "gate_fact_T1": _share(g_t1),
        "gate_survival": _share(g_t1[held]) if held.any() else np.nan,
        "own_fact_n": int(oboth.sum()),
        "own_fact_T": _share(o_t[oboth]) if oboth.any() else np.nan,
        "own_fact_T1": _share(o_t1[oboth]) if oboth.any() else np.nan,
        "own_survival": _share(o_t1[oheld]) if oheld.any() else np.nan,
    }


def _at_execution(x: np.ndarray, samples: np.ndarray | None, tol_bps: float) -> dict:
    """Попадание «по среднему», выгода вперёд и доля пушей с положительной выгодой — по фиксингу и
    в ожидании по распределению ε (расхождение минус его среднее); отдельно точка ε = p90 —
    девятый дециль отклонения со знаком."""
    x = x[~np.isnan(x)]
    thr = 1 - tol_bps / 1e4
    out: dict = {
        "n_scored": int(len(x)),
        "benefit_fwd_bps": float(((x - 1) * 1e4).mean()) if len(x) else np.nan,
        "hit_mean": float((x >= thr).mean()) if len(x) else np.nan,
        "share_positive": float((x > 1).mean()) if len(x) else np.nan,
    }
    exec_keys = (
        "benefit_fwd_exec_bps",
        "hit_mean_exec",
        "share_positive_exec",
        "benefit_fwd_at_p90_spread_bps",
        "spread_days",
        "spread_mean_bps",
        "spread_p90_dev_bps",
    )
    if samples is None or not len(samples) or not len(x):
        out.update(dict.fromkeys(exec_keys, np.nan))
        return out
    raw = np.asarray(samples, dtype=float)
    center = float(raw.mean())
    eps = (raw - center) / 1e4
    ratio = x[:, None] / (1 + eps[None, :])
    p90 = float(np.quantile(eps, 0.9))
    out.update(
        {
            "benefit_fwd_exec_bps": float(((ratio - 1) * 1e4).mean()),
            "hit_mean_exec": float((ratio >= thr).mean()),
            "share_positive_exec": float((ratio > 1).mean()),
            "benefit_fwd_at_p90_spread_bps": float(((x / (1 + p90) - 1) * 1e4).mean()),
            "spread_days": int(len(eps)),
            "spread_mean_bps": center,
            "spread_p90_dev_bps": p90 * 1e4,
        }
    )
    return out


def _spread_options(corridor: str, spreads: dict[str, pd.Series]) -> list[tuple[str, np.ndarray | None]]:
    opts: list[tuple[str, np.ndarray | None]] = []
    if SPREAD_REFERENCE in spreads:
        opts.append((SPREAD_REFERENCE, spreads[SPREAD_REFERENCE].to_numpy(dtype=float)))
    if corridor in spreads and corridor != SPREAD_REFERENCE:
        opts.append((corridor, spreads[corridor].to_numpy(dtype=float)))
    return opts or [("", None)]


def execution_survival_table(
    signals: pd.DataFrame,
    decided: pd.DataFrame | None,
    panel: pd.DataFrame,
    spreads: dict[str, pd.Series] | None = None,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """По коридорам (и строкой `all`) для каждого источника `BUY_NOW` — индикаторы и итоговый поток:
    доля пушей, у которых факт держится на T+1, и попадание с выгодой при исполнении со спредом.
    `spread_source` — чьё распределение ε: пара CNY или собственная пара коридора."""
    spreads = spreads or {}
    pushes = _pushes(signals, decided)
    if pushes.empty:
        return pd.DataFrame(columns=COLUMNS)
    pp = _per_push(pushes, panel, h)
    if pp.empty:
        return pd.DataFrame(columns=COLUMNS)
    groups: list[tuple[str, str, pd.DataFrame]] = [
        (str(c), str(s), g) for (c, s), g in pp.groupby(["corridor", "source"], sort=True)
    ]
    groups += [("all", str(s), g) for s, g in pp.groupby("source", sort=True)]
    rows: list[dict] = []
    for corridor, source, g in groups:
        base = {"corridor": corridor, "source": source, **_survival(g)}
        for name, samples in _spread_options(corridor, spreads):
            rows.append(
                {
                    **base,
                    "spread_source": name,
                    **_at_execution(g["x"].to_numpy(dtype=float), samples, tol_bps),
                }
            )
    return pd.DataFrame(rows)[COLUMNS]
