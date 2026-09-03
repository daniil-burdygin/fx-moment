"""Окно клиента: бэктест решения клиента, а не индикатора (строка Ш5 плана, 💬 04.09 00:30).

Два семейства портретов. «Ожидание»: деньги пришли в день s, семье они нужны не позже чем через k
дней публикации, привычка — перевести сразу, без пуша клиент ждёт до срока. «Ускорение»: деньги есть
с дня a, привычный день перевода — h-е число, пуш может только приблизить перевод, без пуша клиент
переводит в привычный день, и цены ожидания у него нет. В окне клиент выбирает один день. Все
стратегии сравниваются на одном и том же окне:
привычка (первый день окна), ожидание до срока, календарь «с 25-го», факт на экране (курс ниже, чем в
80 % дней за полгода), первый пуш потока внутри окна, случайный день, оракул. Метрики — в базисных
пунктах против привычки и против случайного дня, в рублях на перевод и в доле выигрыша оракула; доля
окон, где перевод сдвинулся, — измеримая часть формулы потенциала кейсодателя «доля сдвинутых ×
чек × маржа». Интервалы — блочный бутстреп по месяцам: блок держит все коридоры месяца вместе,
потому что коридоры почти один фактор (корреляция с USD/RUB 0,83–0,97)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxmoment.backtest.walkforward import Split
from fxmoment.config import BUY_NOW, CORRIDORS, PRIMARY_TOL_BPS

SALARY_DAYS: tuple[int, ...] = (5, 10, 15, 20, 25)
WAIT_DAYS: tuple[int, ...] = (5, 10, 15)  # дней публикации от прихода денег до срока
AVAIL_DAYS: tuple[int, ...] = (5, 10)  # «ускорение»: с какого числа деньги есть
HABIT_DAYS: tuple[int, ...] = (15, 20, 25)  # «ускорение»: привычное число перевода
FAMILIES: tuple[str, ...] = ("wait", "accelerate")
STRATEGIES: tuple[str, ...] = (
    "habit",
    "deadline",
    "random",
    "calendar25",
    "screen",
    "push",
    "push_or_calendar",
    "oracle",
)
STRATEGY_LABELS: dict[str, str] = {
    "habit": "привычка: первый день окна",
    "deadline": "ждать до срока",
    "random": "случайный день окна",
    "calendar25": "календарь: первый день с 25-го, иначе срок",
    "screen": "экран: первый день с фактом уровня, иначе срок",
    "push": "первый пуш потока, иначе срок",
    "push_or_calendar": "первый пуш, иначе календарь, иначе срок",
    "oracle": "оракул: лучший день окна",
}
TRANSFER_RUB = 30_000.0
SCREEN_WINDOW = 120  # факт на экране: «ниже, чем в 80 % дней за полгода» — тот же гейт, что в execution.py
SCREEN_PCT = 0.2
CALENDAR_DAY = 25
BOOTSTRAP_REPS = 2000

WINDOW_COLUMNS = [
    "corridor",
    "persona",
    "salary_day",
    "wait_days",
    "month",
    "start",
    "end",
    "n_days",
]


@dataclass(frozen=True)
class Persona:
    """`wait`: окно от первого дня публикации не раньше `salary_day` на `wait_days` дней публикации,
    привычка — первый день. `accelerate`: окно от `salary_day` (деньги есть) до первого дня публикации
    не раньше `habit_day` того же месяца, привычка — последний день окна."""

    salary_day: int
    wait_days: int | None = None
    habit_day: int | None = None

    def __post_init__(self) -> None:
        if (self.wait_days is None) == (self.habit_day is None):
            raise ValueError("портрет задаётся либо wait_days, либо habit_day")
        if self.habit_day is not None and self.habit_day <= self.salary_day:
            raise ValueError("привычный день перевода должен быть позже дня прихода денег")

    @property
    def family(self) -> str:
        return "wait" if self.wait_days is not None else "accelerate"

    @property
    def label(self) -> str:
        if self.wait_days is not None:
            return f"s{self.salary_day:02d}_k{self.wait_days:02d}"
        return f"a{self.salary_day:02d}_h{self.habit_day:02d}"


def personas(
    salary_days: tuple[int, ...] = SALARY_DAYS,
    wait_days: tuple[int, ...] = WAIT_DAYS,
    avail_days: tuple[int, ...] = AVAIL_DAYS,
    habit_days: tuple[int, ...] = HABIT_DAYS,
) -> list[Persona]:
    waiting = [Persona(s, wait_days=k) for s in salary_days for k in wait_days]
    accel = [Persona(a, habit_day=h) for a in avail_days for h in habit_days]
    return waiting + accel


def screen_flag(rate: pd.Series, window: int = SCREEN_WINDOW, pct: float = SCREEN_PCT) -> pd.Series:
    """Факт на экране перевода: сегодняшний курс не выше, чем в доле `1 − pct` дней последних `window`
    дней публикации (включая сегодня). Только прошлое, параметры заданы до замера."""
    rank = rate.rolling(window, min_periods=window).apply(lambda x: float((x <= x[-1]).mean()), raw=True)
    return (rank <= pct).fillna(False)


def client_windows(
    index: pd.DatetimeIndex, persona: Persona, start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[int, int]]:
    """Окна клиента как пары позиций в оси публикации: первый день публикации не раньше s-го числа
    каждого месяца (в том же месяце) и он же плюс k дней публикации. Окно за краем ряда не строится."""
    out: list[tuple[int, int]] = []
    months = pd.period_range(start, end, freq="M")
    for m in months:
        anchor = pd.Timestamp(year=m.year, month=m.month, day=min(persona.salary_day, m.days_in_month))
        pos = int(index.searchsorted(anchor))
        if pos >= len(index) or index[pos].month != m.month or index[pos].year != m.year:
            continue
        if persona.wait_days is not None:
            last = pos + persona.wait_days
        else:
            habit = pd.Timestamp(year=m.year, month=m.month, day=min(int(persona.habit_day), m.days_in_month))
            last = int(index.searchsorted(habit))
            if last >= len(index) or index[last].month != m.month or index[last].year != m.year:
                continue
        if last >= len(index) or last <= pos or index[pos] < start or index[last] > end:
            continue
        out.append((pos, last))
    return out


def _first_true(mask: np.ndarray, default: int) -> int:
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else default


def window_table(
    panel: pd.DataFrame,
    splits: list[Split],
    decisions: pd.DataFrame | None,
    corridors: tuple[str, ...] = CORRIDORS,
    persona_list: list[Persona] | None = None,
) -> pd.DataFrame:
    """Строка на окно «коридор × портрет × месяц»: курс каждой стратегии и выбранный ею день окна
    (позиция от 0 до k = срок; привычка — 0 у семейства «ожидание» и k у «ускорения»; у случайного
    дня курс — среднее по окну, позиции нет). Без пуша обе семьи переводят в день k: срок либо привычка.
    Период — тестовые окна walk-forward: только там есть пуши потока."""
    persona_list = persona_list or personas()
    start, end = splits[0].test_start, splits[-1].test_end
    pushes: set[tuple[str, pd.Timestamp]] = set()
    if decisions is not None and len(decisions):
        sent = decisions[(decisions["decision"] == "sent") & (decisions["push_scenario"] == BUY_NOW)]
        pushes = {(str(c), pd.Timestamp(d)) for c, d in zip(sent["corridor"], sent["date"], strict=True)}
    rows: list[dict] = []
    for corridor in corridors:
        if corridor not in panel.columns:
            continue
        rate = panel[corridor].dropna()
        index = rate.index
        values = rate.to_numpy(dtype=float)
        screen = screen_flag(rate).to_numpy(dtype=bool)
        pushed = np.array([(corridor, d) in pushes for d in index], dtype=bool)
        days = index.day.to_numpy()
        for persona in persona_list:
            for pos, last in client_windows(index, persona, start, end):
                r = values[pos : last + 1]
                k = last - pos
                cal = _first_true(days[pos : last + 1] >= CALENDAR_DAY, k)
                push = _first_true(pushed[pos : last + 1], -1)
                habit_idx = 0 if persona.family == "wait" else k
                pick = {
                    "habit": habit_idx,
                    "deadline": k,
                    "calendar25": cal,
                    "screen": _first_true(screen[pos : last + 1], k),
                    "push": push if push >= 0 else k,
                    "push_or_calendar": push if push >= 0 else cal,
                    "oracle": int(np.argmin(r)),
                }
                row = {
                    "corridor": corridor,
                    "family": persona.family,
                    "persona": persona.label,
                    "salary_day": persona.salary_day,
                    "wait_days": persona.wait_days,
                    "habit_day": persona.habit_day,
                    "month": f"{index[pos]:%Y-%m}",
                    "start": index[pos],
                    "end": index[last],
                    "n_days": k + 1,
                    "has_push": bool(push >= 0),
                    "has_calendar": bool(np.any(days[pos : last + 1] >= CALENDAR_DAY)),
                    "has_screen": bool(np.any(screen[pos : last + 1])),
                    "rate_random": float(r.mean()),
                }
                for name, j in pick.items():
                    row[f"rate_{name}"] = float(r[j])
                    row[f"day_{name}"] = j
                rows.append(row)
    return pd.DataFrame(rows)


def _bps(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator - 1.0) * 1e4


def _block_ci(
    values: pd.Series, blocks: pd.Series, reps: int = BOOTSTRAP_REPS, seed: int = 0
) -> tuple[float, float]:
    """Интервал среднего блочным бутстрепом: блоки (месяцы) берутся с возвращением целиком."""
    g = pd.DataFrame({"v": values.to_numpy(dtype=float), "b": blocks.to_numpy()}).groupby("b")["v"]
    sums = g.sum().to_numpy(dtype=float)
    counts = g.count().to_numpy(dtype=float)
    n = len(sums)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _shifted(w: pd.DataFrame, name: str) -> pd.Series:
    """Окна, где стратегия увела перевод с дня привычки ПО СВОЕМУ СИГНАЛУ: у пуша — пуш пришёл и день
    другой; у календаря и экрана — их день был в окне и он другой. Ожидание до срока без сигнала
    сдвигом коммуникации не считается, хотя курс в сводке за него платит."""
    if name == "random":
        return pd.Series(False, index=w.index)
    moved = w[f"day_{name}"] != w["day_habit"]
    if name == "push":
        return moved & w["has_push"]
    if name == "push_or_calendar":
        return moved & (w["has_push"] | w["has_calendar"])
    if name == "calendar25":
        return moved & w["has_calendar"]
    if name == "screen":
        return moved & w["has_screen"]
    return moved


def summarize(
    windows: pd.DataFrame,
    strategies: tuple[str, ...] = STRATEGIES,
    tol_bps: float = PRIMARY_TOL_BPS,
    transfer_rub: float = TRANSFER_RUB,
    reps: int = BOOTSTRAP_REPS,
) -> pd.DataFrame:
    """Сводка «коридор × семейство × портрет × стратегия» плюс строки `all` по коридорам и по портретам
    внутри семейства (семейства не смешиваются: у них разная привычка). `benefit_vs_habit_bps` —
    насколько курс стратегии ниже курса привычки; `oracle_share` — доля выигрыша оракула над случайным
    днём, взятая стратегией (по суммам); `share_shifted` — доля окон, где стратегия увела перевод с дня
    привычки по своему сигналу (`_shifted`); `benefit_when_shifted_bps` — выгода против привычки в этих
    окнах; `regret_share` — доля окон, где стратегия хуже привычки больше чем на допуск;
    `rub_per_transfer` — выгода против привычки на перевод `transfer_rub`."""
    if windows.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    corridor_keys = [(c, windows[windows["corridor"] == c]) for c in sorted(windows["corridor"].unique())]
    corridor_keys.append(("all", windows))
    for corridor, wc in corridor_keys:
        for family in FAMILIES:
            wf = wc[wc["family"] == family]
            if wf.empty:
                continue
            persona_keys = [(p, wf[wf["persona"] == p]) for p in sorted(wf["persona"].unique())]
            persona_keys.append(("all", wf))
            for persona, w in persona_keys:
                gain = float(_bps(w["rate_random"], w["rate_oracle"]).sum())
                for name in strategies:
                    r = w[f"rate_{name}"]
                    vs_habit = _bps(w["rate_habit"], r)
                    vs_random = _bps(w["rate_random"], r)
                    is_random = name == "random"
                    shifted = _shifted(w, name)
                    lo_h, hi_h = _block_ci(vs_habit, w["month"], reps=reps)
                    lo_r, hi_r = _block_ci(vs_random, w["month"], reps=reps)
                    if shifted.any():
                        lo_s, hi_s = _block_ci(vs_habit[shifted], w.loc[shifted, "month"], reps=reps)
                    else:
                        lo_s, hi_s = np.nan, np.nan
                    rows.append(
                        {
                            "corridor": corridor,
                            "family": family,
                            "persona": persona,
                            "strategy": name,
                            "label": STRATEGY_LABELS[name],
                            "windows": int(len(w)),
                            "months": int(w["month"].nunique()),
                            "benefit_vs_habit_bps": float(vs_habit.mean()),
                            "benefit_vs_habit_ci_lo": lo_h,
                            "benefit_vs_habit_ci_hi": hi_h,
                            "benefit_vs_random_bps": float(vs_random.mean()),
                            "benefit_vs_random_ci_lo": lo_r,
                            "benefit_vs_random_ci_hi": hi_r,
                            "oracle_share": float(vs_random.sum() / gain) if gain > 0 else np.nan,
                            "share_shifted": np.nan if is_random else float(shifted.mean()),
                            "benefit_when_shifted_bps": float(vs_habit[shifted].mean())
                            if shifted.any()
                            else np.nan,
                            "benefit_when_shifted_ci_lo": lo_s,
                            "benefit_when_shifted_ci_hi": hi_s,
                            "regret_share": float((_bps(r, w["rate_habit"]) > tol_bps).mean()),
                            "not_worse_share": float((_bps(r, w["rate_habit"]) <= tol_bps).mean()),
                            "mean_day": np.nan if is_random else float(w[f"day_{name}"].mean()),
                            "rub_per_transfer": float(vs_habit.mean() / 1e4 * transfer_rub),
                        }
                    )
    return pd.DataFrame(rows)


def client_window_tables(
    panel: pd.DataFrame,
    splits: list[Split],
    decisions: pd.DataFrame | None,
    corridors: tuple[str, ...] = CORRIDORS,
    reps: int = BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Окна и сводка одним вызовом — для `analyze` и команды `client-window`."""
    windows = window_table(panel, splits, decisions, corridors=corridors)
    return windows, summarize(windows, reps=reps)
