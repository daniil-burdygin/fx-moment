"""Политика потока (ADR-0006): приоритет по надёжности, конфликт сценариев, охлаждение,
прореживание. Вход — события индикаторов, выход — те же события с решением."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fxmoment.config import BUY_NOW, WINDOW_CLOSING


@dataclass(frozen=True)
class PolicyParams:
    cooldown_days: int = 3  # дней публикации тишины после пуша
    max_per_window: int = 2  # не больше пушей за скользящее окно
    window_days: int = 5  # длина скользящего окна, дней публикации
    conflict_window: int = 10  # BUY_NOW в последние N дней → WINDOW_CLOSING имеет приоритет
    muted: tuple[str, ...] = ()  # индикаторы, исключённые по бэктесту (только WATCH)


def apply_policy(
    events: pd.DataFrame,
    rank: dict[str, int],
    calendar: pd.DatetimeIndex,
    params: PolicyParams | None = None,
) -> pd.DataFrame:
    """Решение по каждому событию: sent / muted / conflict / cooldown / thinned.

    rank: индикатор → ранг надёжности (меньше = надёжнее). calendar — дни публикации
    для отсчёта охлаждения."""
    params = params or PolicyParams()
    out = events.copy()
    out["decision"] = ""
    out["push_scenario"] = ""
    pos = pd.Series(range(len(calendar)), index=calendar)
    for _corridor, grp in out.groupby("corridor", sort=False):
        sent: list[tuple[int, str]] = []  # (позиция дня, сценарий)
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
