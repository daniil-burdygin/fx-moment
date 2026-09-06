"""Вечерний гейт: сверка факта пуша с рынком после фиксинга (Ш13).

Пуш считается вечером T по курсу действия a_T. Фиксинг ЦБ снят с рынка до 15:30, а валютный
рынок к вечеру уходит дальше — и утром T+1, когда клиент переводит, факт уровня держится
у 93 % пушей (`analysis/execution_survival.csv`). Вопрос строки: если перед отправкой, около
17:00–21:00 МСК, пересчитать факт на вечернем рынке, сколько пушей это снимет и станут ли
оставшиеся точнее.

Оценка «куда ушёл рынок после фиксинга» берётся с срочного рынка Мосбиржи: валютный рынок к
этому часу закрыт, фьючерс идёт до 23:50. Относительное изменение фронт-контракта от опорного
бара до вечернего применяется к a_T — получается вечерний курс коридора. Коридоры СНГ считаются
одним множителем Si: их валюты управляются против доллара, и внутридневное движение рубля к ним
идёт долларовой ногой. Насколько это допущение верно, меряется отдельно (столбцы `est_*`) —
и контролем служит CNY, где фьючерс и фиксинг мерят один и тот же курс.

Причинность. Вечерний бар берётся не позже 20:00 МСК: его закрытие известно к 21:00, то есть до
конца окна отправки. Ни одного бара следующего дня в расчёт не входит — это проверяет тест."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxmoment import labels
from fxmoment.config import BUY_NOW, CALIBRATION_H, FIRST_TEST, PRIMARY_TOL_BPS
from fxmoment.execution import GATE_PCT, GATE_WINDOW

# Опорный бар — тот, что накрывает срез фиксинга 15:30 (бар 15:00 идёт с 15:00 до 16:00).
# Запасной 14:00 — цена ровно на 15:00; обе опоры известны на T, разница между ними и есть мера
# того, насколько результат держится за выбор получаса.
ANCHOR_HOURS: tuple[int, ...] = (15, 14)
EVENING_FIRST_HOUR = 18  # первый вечерний бар
EVENING_LAST_HOUR = 20  # последний допустимый: закрывается в 21:00, к концу окна отправки
CONTROL_ASSET = "CNY"  # фьючерс и фиксинг мерят один курс — контроль допущения
CORRIDOR_ASSET = "USD"  # множитель Si для коридоров СНГ

GROUP_ALL = "все пуши"
GROUP_FACT = "факт на T"
GROUP_PASS = "прошли гейт"
GROUP_DROP = "сняты гейтом"

COLUMNS = [
    "corridor",
    "asset",
    "anchor_hour",
    "group",
    "n",
    "evening_fact_share",
    "gate_survival_T1",
    "n_scored",
    "hit_mean",
    "benefit_fwd_bps",
    "share_positive",
    "n_scored_evening",
    "hit_mean_evening",
    "benefit_fwd_evening_bps",
    "est_n",
    "est_corr",
    "est_mae_bps",
    "est_sign_share",
    "move_mean_abs_bps",
]


def daily_moves(
    forts_long: pd.DataFrame,
    asset: str,
    anchor_hour: int,
    evening_first: int = EVENING_FIRST_HOUR,
    evening_last: int = EVENING_LAST_HOUR,
) -> pd.DataFrame:
    """По каждой торговой дате: опорный курс, вечерний курс и их относительное изменение.

    Оба бара — одной даты и одного контракта (склейка фронта в `data/forts.py` даёт на дату один
    контракт), поэтому скачок цены на ролле в изменение не попадает. Дата без опорного или без
    вечернего бара в таблицу не входит: заполнять её нечем."""
    df = forts_long[forts_long["asset"] == asset].copy()
    if df.empty:
        return pd.DataFrame(columns=["anchor", "evening", "evening_hour", "delta"])
    df["begin"] = pd.to_datetime(df["begin"])
    df["day"] = df["begin"].dt.normalize()
    df["hour"] = df["begin"].dt.hour
    anchor = (
        df[df["hour"] == anchor_hour].drop_duplicates("day", keep="last").set_index("day")["close"]
    )
    evening_bars = df[(df["hour"] >= evening_first) & (df["hour"] <= evening_last)]
    last = evening_bars.sort_values("begin").drop_duplicates("day", keep="last").set_index("day")
    out = pd.DataFrame({"anchor": anchor, "evening": last["close"], "evening_hour": last["hour"]})
    out = out.dropna(subset=["anchor", "evening"])
    out["delta"] = out["evening"] / out["anchor"] - 1
    return out.sort_index()


def evening_rank(rate: pd.Series, pos: int, evening_rate: float, window: int = GATE_WINDOW) -> float:
    """Процентиль вечернего курса в том же скользящем окне, что и `rolling_pct_rank`: последнее
    значение окна заменено вечерним курсом, остальные — опубликованные до T."""
    if pos + 1 < window:
        return np.nan
    tail = rate.to_numpy()[pos + 1 - window : pos]
    return float(np.mean(np.append(tail, evening_rate) <= evening_rate))


def estimate_quality(rate: pd.Series, moves: pd.DataFrame, since: str = FIRST_TEST) -> dict:
    """Насколько вечерний сдвиг фьючерса предсказывает фактическое изменение фиксинга T → T+1.

    Считается на всех днях публикации периода оценки, не только на днях пушей: это свойство
    допущения, а не выборки. `est_mae_bps` — средняя абсолютная ошибка предсказания в базисных
    пунктах, `est_sign_share` — доля дней, где знак совпал (нулевые фактические изменения,
    которых на оси публикации нет по построению, в знак не входят)."""
    idx = rate.index
    realized = (rate.shift(-1) / rate - 1).loc[pd.Timestamp(since) :]
    delta = moves["delta"].reindex(idx).loc[pd.Timestamp(since) :]
    both = pd.concat([realized.rename("real"), delta.rename("est")], axis=1).dropna()
    if len(both) < 2:
        return {"est_n": int(len(both)), "est_corr": np.nan, "est_mae_bps": np.nan, "est_sign_share": np.nan}
    signs = np.sign(both["real"]) * np.sign(both["est"])
    return {
        "est_n": int(len(both)),
        "est_corr": float(both["real"].corr(both["est"])),
        "est_mae_bps": float((both["real"] - both["est"]).abs().mean() * 1e4),
        "est_sign_share": float((signs > 0).mean()),
    }


def _per_push(
    pushes: pd.DataFrame, panel: pd.DataFrame, moves: pd.DataFrame, h: int
) -> pd.DataFrame:
    """По каждому пушу: факт уровня на T, вечером T и утром T+1, и отношение среднего будущих h
    курсов к a_T. NaN — величина не определена (нет вечернего бара, окно не набралось,
    горизонт не поместился)."""
    frames: list[pd.DataFrame] = []
    for corridor, grp in pushes.groupby("corridor", sort=True):
        if corridor not in panel.columns:
            continue
        rate = panel[corridor].dropna()
        idx = rate.index
        from fxmoment.indicators.base import rolling_pct_rank

        gate = (rolling_pct_rank(rate, GATE_WINDOW) <= GATE_PCT).to_numpy(dtype=float)
        x_all = (labels.future_mean(rate, h) / rate).to_numpy(dtype=float)
        values = rate.to_numpy(dtype=float)
        dates = pd.DatetimeIndex(grp["date"])
        pos = idx.get_indexer(dates)
        delta = moves["delta"].reindex(dates).to_numpy(dtype=float)
        rows = {
            "corridor": corridor,
            "date": dates,
            "gate_T": np.full(len(grp), np.nan),
            "gate_T1": np.full(len(grp), np.nan),
            "gate_evening": np.full(len(grp), np.nan),
            "delta": delta,
            "x": np.full(len(grp), np.nan),
        }
        for i, p in enumerate(pos):
            if p < 0:
                continue
            rows["gate_T"][i] = gate[p]
            rows["x"][i] = x_all[p]
            if p + 1 < len(idx):
                rows["gate_T1"][i] = gate[p + 1]
            if not np.isnan(delta[i]):
                rank = evening_rank(rate, p, values[p] * (1 + delta[i]))
                rows["gate_evening"][i] = np.nan if np.isnan(rank) else float(rank <= GATE_PCT)
        frame = pd.DataFrame(rows)
        # Цена, доступная клиенту вечером, а не фиксинг: тот же будущий курс, делённый на
        # вечерний уровень. Ею меряется, остаётся ли выгода снятого пуша выгодой.
        frame["x_evening"] = frame["x"] / (1 + frame["delta"])
        frames.append(frame)
    columns = ["corridor", "date", "gate_T", "gate_T1", "gate_evening", "delta", "x", "x_evening"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _scored(g: pd.DataFrame, tol_bps: float) -> dict:
    """Попадание и выгода вперёд по двум ценам: по фиксингу a_T (как во всём отчёте) и по
    вечернему курсу — тому, что клиент реально увидит, когда пуш до него дойдёт."""
    thr = 1 - tol_bps / 1e4
    x = g["x"].to_numpy(dtype=float)
    x = x[~np.isnan(x)]
    xe = g["x_evening"].to_numpy(dtype=float)
    xe = xe[~np.isnan(xe)]
    return {
        "n_scored": int(len(x)),
        "hit_mean": float((x >= thr).mean()) if len(x) else np.nan,
        "benefit_fwd_bps": float(((x - 1) * 1e4).mean()) if len(x) else np.nan,
        "share_positive": float((x > 1).mean()) if len(x) else np.nan,
        "n_scored_evening": int(len(xe)),
        "hit_mean_evening": float((xe >= thr).mean()) if len(xe) else np.nan,
        "benefit_fwd_evening_bps": float(((xe - 1) * 1e4).mean()) if len(xe) else np.nan,
    }


def _survival(g: pd.DataFrame) -> float:
    held = g[g["gate_T"] == 1.0]["gate_T1"].dropna()
    return float(held.mean()) if len(held) else np.nan


def _mean_defined(quality: dict[str, dict], key: str) -> float:
    """Среднее по коридорам без предупреждения на пустом срезе: нечего усреднять — NaN."""
    vals = [q[key] for q in quality.values() if not np.isnan(q[key])]
    return float(np.mean(vals)) if vals else np.nan


def _groups(pp: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    fact = pp[pp["gate_T"] == 1.0]
    return [
        (GROUP_ALL, pp),
        (GROUP_FACT, fact),
        (GROUP_PASS, fact[fact["gate_evening"] == 1.0]),
        (GROUP_DROP, fact[fact["gate_evening"] == 0.0]),
    ]


def evening_gate_table(
    decided: pd.DataFrame,
    panel: pd.DataFrame,
    forts_long: pd.DataFrame,
    h: int = CALIBRATION_H,
    tol_bps: float = PRIMARY_TOL_BPS,
    anchor_hours: tuple[int, ...] = ANCHOR_HOURS,
) -> pd.DataFrame:
    """Таблица вечернего гейта по отправленным пушам `BUY_NOW` итогового потока.

    Строки — коридор (и `all`) × опорный час × группа: все пуши, пуши с истинным фактом на T,
    из них прошедшие вечерний гейт и снятые им. Отдельно контрольные строки CNY: там гейт
    считается не на пушах, а на качестве самой оценки — фьючерс и фиксинг мерят один курс."""
    sent = decided[(decided["decision"] == "sent") & (decided["push_scenario"] == BUY_NOW)]
    pushes = sent[["date", "corridor"]].copy()
    pushes["date"] = pd.to_datetime(pushes["date"])
    rows: list[dict] = []
    for anchor in anchor_hours:
        corridor_moves = daily_moves(forts_long, CORRIDOR_ASSET, anchor)
        pp = _per_push(pushes, panel, corridor_moves, h)
        if pp.empty:
            continue
        quality = {
            corridor: estimate_quality(panel[corridor].dropna(), corridor_moves)
            for corridor in sorted(pp["corridor"].unique())
            if corridor in panel.columns
        }
        move_abs = float(corridor_moves["delta"].abs().mean() * 1e4) if len(corridor_moves) else np.nan
        blocks: list[tuple[str, pd.DataFrame]] = [(str(c), g) for c, g in pp.groupby("corridor", sort=True)]
        blocks.append(("all", pp))
        empty_est = dict.fromkeys(("est_n", "est_corr", "est_mae_bps", "est_sign_share"), np.nan)
        pooled_est = {
            "est_n": int(sum(q["est_n"] for q in quality.values())),
            **{k: _mean_defined(quality, k) for k in ("est_corr", "est_mae_bps", "est_sign_share")},
        }
        for corridor, block in blocks:
            est = pooled_est if corridor == "all" else quality.get(corridor, empty_est)
            fact = block["gate_T"] == 1.0
            known = int((fact & block["gate_evening"].notna()).sum())
            share = float((fact & (block["gate_evening"] == 1.0)).sum() / known) if known else np.nan
            for name, g in _groups(block):
                rows.append(
                    {
                        "corridor": corridor,
                        "asset": CORRIDOR_ASSET,
                        "anchor_hour": anchor,
                        "group": name,
                        "n": int(len(g)),
                        "evening_fact_share": share if name == GROUP_FACT else np.nan,
                        "gate_survival_T1": _survival(g),
                        **_scored(g, tol_bps),
                        **est,
                        "move_mean_abs_bps": move_abs,
                    }
                )
        control = daily_moves(forts_long, CONTROL_ASSET, anchor)
        if len(control) and CONTROL_ASSET in panel.columns:
            rows.append(
                {
                    "corridor": CONTROL_ASSET,
                    "asset": CONTROL_ASSET,
                    "anchor_hour": anchor,
                    "group": "контроль оценки",
                    "n": int(len(control)),
                    "evening_fact_share": np.nan,
                    "gate_survival_T1": np.nan,
                    "n_scored": 0,
                    "hit_mean": np.nan,
                    "benefit_fwd_bps": np.nan,
                    "share_positive": np.nan,
                    "n_scored_evening": 0,
                    "hit_mean_evening": np.nan,
                    "benefit_fwd_evening_bps": np.nan,
                    **estimate_quality(panel[CONTROL_ASSET].dropna(), control),
                    "move_mean_abs_bps": float(control["delta"].abs().mean() * 1e4),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)
