"""Политика потока (ADR-0006): приоритет по надёжности, конфликт сценариев, охлаждение,
прореживание. Вход — события индикаторов, выход — те же события с решением.

Правило причинно: в день T решение принимается по событиям дня T и по уже отправленным пушам;
«самый сильный из серии» выбрать нельзя — серию видно только задним числом."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxmoment.config import BUY_NOW, WINDOW_CLOSING


def storm_flag(
    rate: pd.Series, vol_window: int = 20, rank_window: int = 250, rank: float = 0.95
) -> pd.Series:
    """Причинный флаг шторма: реализованная волатильность дневных лог-доходностей за vol_window дней
    в верхних (1 − rank) своего скользящего года. Лебедя не предскажешь — но день, когда он уже
    прилетел, виден по волатильности, и в такой день поток молчит (ADR-0006)."""
    from fxmoment.indicators.base import rolling_pct_rank

    if rank > 1:
        return pd.Series(False, index=rate.index)
    vol = np.log(rate).diff().rolling(vol_window).std()
    pct = rolling_pct_rank(vol, rank_window)
    return (pct > rank).fillna(False)


@dataclass(frozen=True)
class PolicyParams:
    cooldown_days: int = 3  # дней публикации тишины после пуша
    max_per_window: int = 2  # не больше пушей за скользящее окно
    window_days: int = 5  # длина скользящего окна, дней публикации
    conflict_window: int = 10  # BUY_NOW в последние N дней → WINDOW_CLOSING имеет приоритет
    muted: tuple[str, ...] = ()  # индикаторы, исключённые по бэктесту (только WATCH)
    storm_vol_window: int = 20  # реализованная волатильность за N дней публикации
    storm_rank_window: int = 250  # её ранг в скользящем году
    storm_rank: float = 0.95  # ранг выше → шторм, пуши BUY_NOW не уходят (> 1 отключает правило)


def apply_policy(
    events: pd.DataFrame,
    rank: dict[str, int],
    calendar: pd.DatetimeIndex,
    params: PolicyParams | None = None,
    prior_sent: dict[str, list[tuple[pd.Timestamp, str]]] | None = None,
    storm: pd.Series | None = None,
) -> pd.DataFrame:
    """Решение по каждому событию: sent / muted / thinned / cooldown / storm.

    storm — булев ряд по календарю (см. storm_flag): в день шторма пуш BUY_NOW не уходит —
    «переводите сейчас» в день обвала и есть самая дорогая ошибка; WINDOW_CLOSING в шторм тоже
    молчит, потому что отскок в шторм ничего не подтверждает.

    rank: индикатор → ранг надёжности (меньше = надёжнее). calendar — дни публикации коридора
    (лучше весь ряд: тогда позиции глобальны и история `prior_sent` переносится через границы
    окон). prior_sent: коридор → уже отправленные (дата, сценарий) до этих событий."""
    params = params or PolicyParams()
    out = events.copy()
    out["decision"] = ""
    out["push_scenario"] = ""
    pos = pd.Series(range(len(calendar)), index=calendar)
    for corridor, grp in out.groupby("corridor", sort=False):
        sent: list[tuple[int, str]] = [
            (int(pos.loc[d]), sc) for d, sc in (prior_sent or {}).get(str(corridor), []) if d in pos.index
        ]
        for day, day_events in grp.groupby("date", sort=True):
            p = int(pos.loc[day])
            live = day_events[~day_events["indicator"].isin(params.muted)]
            for i in day_events.index.difference(live.index):
                out.loc[i, "decision"] = "muted"
            if live.empty:
                continue
            # приоритет: ранг индикатора, затем сила
            live = live.assign(_rank=live["indicator"].map(rank).fillna(99)).sort_values(
                ["_rank", "strength"], ascending=[True, False]
            )
            chosen = live.index[0]
            scen = live.loc[chosen, "scenario"]
            recent_buy = any(sc == BUY_NOW and p - q <= params.conflict_window for q, sc in sent)
            if (live["scenario"] == WINDOW_CLOSING).any() and (live["scenario"] == BUY_NOW).any():
                scen = WINDOW_CLOSING if recent_buy else BUY_NOW
                chosen = live[live["scenario"] == scen].index[0]
            for i in live.index.difference([chosen]):
                out.loc[i, "decision"] = "thinned"
            if storm is not None and bool(storm.get(day, False)):
                out.loc[chosen, "decision"] = "storm"
                continue
            if sent and p - sent[-1][0] <= params.cooldown_days:
                out.loc[chosen, "decision"] = "cooldown"
                continue
            if sum(1 for q, _ in sent if p - q < params.window_days) >= params.max_per_window:
                out.loc[chosen, "decision"] = "cooldown"
                continue
            out.loc[chosen, "decision"] = "sent"
            out.loc[chosen, "push_scenario"] = scen
            sent.append((p, scen))
    return out
