"""Режим бюджетного правила как признак дня (строка Ш14 плана).

Минфин России в третий рабочий день месяца объявляет, покупает он валюту и золото на внутреннем
рынке или продаёт, называет период и ежедневный объём в млрд ₽; операции зеркалирует Банк России.
Объявление известно заранее и с датой, поэтому режим дня T — каузальная функция от прошлого: в него
входят только строки с `announce_date ≤ T`. Снимок и его границы — `data/minfin_fx_operations.csv`
и `data/minfin_fx_operations.md`.

Здесь не индикатор, а разрез: попадание и выгода индикатора уровня и итогового потока `BUY_NOW`,
посчитанные отдельно в днях покупки, продажи и паузы. Признак обучаемому этот модуль не добавляет.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fxmoment import labels, metrics
from fxmoment.config import BUY_NOW, CALIBRATION_H, CORRIDORS, PRIMARY_TOL_BPS

REGIMES: tuple[str, ...] = ("buy", "sell", "none")
LEVEL_INDICATOR = "level"
STREAM_LABEL = "stream BUY_NOW"
SNAPSHOT = "data/minfin_fx_operations.csv"

COLUMNS = [
    "source",
    "corridor",
    "regime",
    "days",
    "events",
    "hit_mean",
    "base_mean",
    "lift_mean",
    "lift_ci_lo",
    "lift_ci_hi",
    "benefit_fwd_bps",
    "benefit_random_day_bps",
    "benefit_excess_bps",
    "benefit_excess_ci_lo",
    "benefit_excess_ci_hi",
    "verdict_benefit",
]


def load_announcements(path: Path | None = None) -> pd.DataFrame:
    """Снимок объявлений: одна строка — одно объявление, отсортировано по дате публикации."""
    from fxmoment.data.store import repo_root

    src = path or (repo_root() / SNAPSHOT)
    df = pd.read_csv(src, parse_dates=["announce_date", "from_date", "to_date"])
    bad = set(df["direction"]) - set(REGIMES)
    if bad:
        raise ValueError(f"неизвестное направление в снимке: {', '.join(sorted(bad))}")
    return df.sort_values("announce_date", kind="stable").reset_index(drop=True)


def regime_series(
    index: pd.DatetimeIndex, ann: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None
) -> pd.Series:
    """Режим каждого дня публикации: `buy`, `sell` или `none`.

    Правило (оно же — граница каузальности): день T берёт объявление с самой поздней
    `announce_date ≤ T`, чей период операций накрывает T; если такого нет — самое позднее объявление
    с `announce_date ≤ T` вообще. Второй случай — это паузы (период в релизе не назван) и дни между
    концом одного периода и началом следующего. До первого объявления режим `none`: снимок начинается
    2018-06, а бюджетное правило до этого работало — отсюда и `none`, «режим неизвестен», а не «покупки».
    `as_of` обрезает снимок так же, как это сделал бы прогон на дату среза."""
    ann = load_announcements() if ann is None else ann
    if as_of is not None:
        ann = ann[ann["announce_date"] <= pd.Timestamp(as_of)]
    out = pd.Series("none", index=index, dtype=object)
    # объявления идут по возрастанию даты публикации, поэтому позднее переписывает раннее
    for row in ann.sort_values("announce_date", kind="stable").itertuples():
        seen = index >= row.announce_date
        if pd.isna(row.from_date):
            out[seen] = row.direction
            continue
        # период назван: он и есть режим; хвост после периода режима не задаёт
        inside = seen & (index >= row.from_date)
        if not pd.isna(row.to_date):
            inside &= index <= row.to_date
        out[inside] = row.direction
        # дни между объявлением и началом периода наследуют прошлый режим — их не трогаем
    return out


def _events_by_source(
    signals: pd.DataFrame, decided: pd.DataFrame | None, corridor: str
) -> dict[str, pd.DatetimeIndex]:
    """Даты событий по источникам: индикатор уровня и отправленные пуши `BUY_NOW` итогового потока."""
    out: dict[str, pd.DatetimeIndex] = {}
    s = signals[(signals["corridor"] == corridor) & (signals["scenario"] == BUY_NOW)]
    lvl = s[s["indicator"] == LEVEL_INDICATOR]
    out[LEVEL_INDICATOR] = pd.DatetimeIndex(pd.to_datetime(lvl["date"]).unique()).sort_values()
    if decided is not None and len(decided):
        d = decided[
            (decided["corridor"] == corridor)
            & (decided["decision"] == "sent")
            & (decided["push_scenario"] == BUY_NOW)
        ]
        out[STREAM_LABEL] = pd.DatetimeIndex(pd.to_datetime(d["date"]).unique()).sort_values()
    return out


def _block_ci(values: pd.Series, seed: int = 0) -> tuple[float, float]:
    if len(values.dropna()) < 5:
        return (np.nan, np.nan)
    return metrics.block_bootstrap_ci(values, seed=seed)


def regime_table(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    decided: pd.DataFrame | None,
    window: tuple[pd.Timestamp, pd.Timestamp],
    corridors: tuple[str, ...] = CORRIDORS,
    ann: pd.DataFrame | None = None,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
) -> pd.DataFrame:
    """Попадание и выгода в разрезе режима дня T.

    База сравнения — случайный день **того же режима**, а не всего окна: иначе таблица мерила бы не
    качество сигнала внутри режима, а то, насколько сам режим отличается от среднего дня. Интервал —
    блочный бутстреп по календарным месяцам на разнице «событие − база режима» (события внутри месяца
    пересекаются по горизонту h)."""
    ann = load_announcements() if ann is None else ann
    start, end = window
    rows: list[dict] = []
    pooled: dict[tuple[str, str], list[pd.Series]] = {}
    for corridor in corridors:
        if corridor not in panel.columns:
            continue
        rate = panel[corridor].dropna()
        idx = rate.loc[start:end].index
        if not len(idx):
            continue
        reg = regime_series(idx, ann)
        hit_all = labels.hit_for_scenario(rate, BUY_NOW, h, tol_bps, mode="mean")
        bf_all = labels.benefit_fwd_bps(rate, h)
        for source, ev in _events_by_source(signals, decided, corridor).items():
            ev = ev[(ev >= start) & (ev <= end)].intersection(idx)
            for regime in REGIMES:
                days = idx[(reg == regime).to_numpy()]
                hit_days = hit_all.loc[days].dropna()
                bf_days = bf_all.loc[days].dropna()
                base_mean = float(hit_days.mean()) if len(hit_days) else np.nan
                bf_base = float(bf_days.mean()) if len(bf_days) else np.nan
                ev_r = ev.intersection(days)
                hits = hit_all.loc[ev_r].dropna()
                bfs = bf_all.loc[ev_r].dropna()
                hit = float(hits.mean()) if len(hits) else np.nan
                lift = hit / base_mean if len(hits) and base_mean and base_mean > 0 else np.nan
                excess = bfs - bf_base if len(bfs) else pd.Series(dtype=float)
                lo, hi = _block_ci(excess)
                if len(hits) and base_mean and base_mean > 0:
                    hlo, hhi = _block_ci(hits, seed=1)
                    lift_lo, lift_hi = hlo / base_mean, hhi / base_mean
                else:
                    lift_lo = lift_hi = np.nan
                rows.append(
                    {
                        "source": source,
                        "corridor": corridor,
                        "regime": regime,
                        "days": int(len(days)),
                        "events": int(len(ev_r)),
                        "hit_mean": hit,
                        "base_mean": base_mean,
                        "lift_mean": lift,
                        "lift_ci_lo": lift_lo,
                        "lift_ci_hi": lift_hi,
                        "benefit_fwd_bps": float(bfs.mean()) if len(bfs) else np.nan,
                        "benefit_random_day_bps": bf_base,
                        "benefit_excess_bps": float(excess.mean()) if len(excess) else np.nan,
                        "benefit_excess_ci_lo": lo,
                        "benefit_excess_ci_hi": hi,
                    }
                )
                pooled.setdefault((source, regime), []).append(excess)
                pooled.setdefault((source, regime + "|hit"), []).append(hits - base_mean)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=COLUMNS)
    out = pd.concat([out, _pooled_rows(out, pooled), _diff_rows(pooled)], ignore_index=True)
    from fxmoment.analysis import read_interval

    out["verdict_benefit"] = read_interval(
        out["benefit_excess_ci_lo"], out["benefit_excess_ci_hi"], "выгода есть", "выгоды нет"
    )
    return out[COLUMNS]


def _pooled_rows(rows: pd.DataFrame, pooled: dict[tuple[str, str], list[pd.Series]]) -> pd.DataFrame:
    """Строки `corridor = all`: события всех коридоров в одном блоке-месяце, база — своя у каждого
    коридора (она уже вычтена), поэтому складывать разницы можно."""
    out: list[dict] = []
    for source in rows["source"].unique():
        for regime in REGIMES:
            g = rows[(rows["source"] == source) & (rows["regime"] == regime)]
            excess = pd.concat(pooled.get((source, regime), []) or [pd.Series(dtype=float)])
            hits = pd.concat(pooled.get((source, regime + "|hit"), []) or [pd.Series(dtype=float)])
            lo, hi = _block_ci(excess.sort_index())
            hlo, hhi = _block_ci(hits.sort_index(), seed=1)
            n = g["events"].sum()
            base = float((g["base_mean"] * g["events"]).sum() / n) if n else np.nan
            hit = float(hits.mean() + base) if len(hits) and not np.isnan(base) else np.nan
            out.append(
                {
                    "source": source,
                    "corridor": "all",
                    "regime": regime,
                    "days": int(g["days"].sum()),
                    "events": int(g["events"].sum()),
                    "hit_mean": hit,
                    "base_mean": base,
                    "lift_mean": hit / base if hit == hit and base and base > 0 else np.nan,
                    "lift_ci_lo": (hlo + base) / base if hlo == hlo and base else np.nan,
                    "lift_ci_hi": (hhi + base) / base if hhi == hhi and base else np.nan,
                    "benefit_fwd_bps": np.nan,
                    "benefit_random_day_bps": np.nan,
                    "benefit_excess_bps": float(excess.mean()) if len(excess) else np.nan,
                    "benefit_excess_ci_lo": lo,
                    "benefit_excess_ci_hi": hi,
                }
            )
    return pd.DataFrame(out)


def _diff_between(
    a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 2
) -> tuple[float, float, float]:
    """Разница средних b − a с интервалом: блоки-месяцы каждой выборки пересобираются независимо.
    Парности здесь нет и быть не может — месяц целиком принадлежит одному режиму."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    blocks = [
        [g.to_numpy() for _, g in v.groupby(v.index.to_period("M"))] for v in (a, b)
    ]
    draws = np.empty(n_boot)
    for i in range(n_boot):
        means = []
        for bl in blocks:
            pick = rng.integers(0, len(bl), size=len(bl))
            means.append(np.concatenate([bl[j] for j in pick]).mean())
        draws[i] = means[1] - means[0]
    return (float(b.mean() - a.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def _diff_rows(pooled: dict[tuple[str, str], list[pd.Series]]) -> pd.DataFrame:
    """Режим против режима по всем коридорам: разница выгоды сверх случайного дня и разница
    попадания сверх базы своего режима. Ноль внутри интервала — режим точность пушей не двигает."""
    out: list[dict] = []
    sources = sorted({s for s, _ in pooled})
    for source in sources:
        for regime in ("sell", "none"):
            pair = {}
            for key, other in (("buy", regime), ("buy|hit", regime + "|hit")):
                a = pooled.get((source, key), [])
                b = pooled.get((source, other), [])
                if not a or not b:
                    continue
                pair[key] = _diff_between(
                    pd.concat(a).sort_index(), pd.concat(b).sort_index(), seed=3 if "hit" in key else 2
                )
            if "buy" not in pair:
                continue
            d, lo, hi = pair["buy"]
            hd, hlo, hhi = pair.get("buy|hit", (np.nan, np.nan, np.nan))
            out.append(
                {
                    "source": source,
                    "corridor": "all",
                    "regime": f"{regime}−buy",
                    "days": 0,
                    "events": 0,
                    "hit_mean": hd,
                    "base_mean": np.nan,
                    "lift_mean": np.nan,
                    "lift_ci_lo": hlo,
                    "lift_ci_hi": hhi,
                    "benefit_fwd_bps": np.nan,
                    "benefit_random_day_bps": np.nan,
                    "benefit_excess_bps": d,
                    "benefit_excess_ci_lo": lo,
                    "benefit_excess_ci_hi": hi,
                }
            )
    return pd.DataFrame(out)
