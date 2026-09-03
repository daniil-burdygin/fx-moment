"""Сверка двух источников курса: биржевое закрытие Мосбиржи против фиксинга ЦБ (ADR-0010, п. 3б).

Вопрос продукта: можно ли слать сигнал днём по бирже, не дожидаясь публикации ЦБ. Он распадается
на два измеримых.

1. **Насколько ряды вообще один и тот же курс.** Расхождение уровня в базисных пунктах и связь
   дневных изменений. Большое расхождение означает, что дневной бэктест на ряде ЦБ не переносится
   на биржевое исполнение без поправки.
2. **К какому часу биржа определяет сегодняшний фиксинг.** ЦБ публикует курс дня T примерно в
   15:30 МСК; бары с началом раньше 15:00 заведомо ему предшествуют. Если расхождение
   «биржа в час H против фиксинга дня T» выходит на полку уже к 11–12 часам, то к этому часу
   фиксинг фактически известен, и пуш можно слать до публикации.

Плотность торгов меряется здесь же: коридор, у которого нет внутридневного рынка, не может дать
внутридневного сигнала ни при какой методике, и это отдельный отрицательный результат.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HOURS = tuple(range(10, 19))  # часы, по которым строится «к какому часу биржа знает фиксинг»
CBR_PUBLICATION_HOUR = 15  # ЦБ публикует курс дня примерно в 15:30 МСК


def daily_close(bars: pd.Series, until_hour: int | None = None) -> pd.Series:
    """Последний известный на день бар: индекс — календарный день, значение — курс закрытия.

    `until_hour` — брать только бары, НАЧАВШИЕСЯ до этого часа включительно (например 12 — бары
    10:00, 11:00, 12:00). Ось ряда — `known_at` = начало бара + час, поэтому отбор идёт по
    `known_at.hour - 1`; бар 12:00 известен в 13:00."""
    s = bars.dropna()
    if until_hour is not None:
        s = s[(s.index.hour - 1) <= until_hour]
    if s.empty:
        return pd.Series(dtype=float)
    return s.groupby(s.index.normalize()).last()


def liquidity(long_df: pd.DataFrame) -> pd.DataFrame:
    """Плотность внутридневного рынка по парам: сколько баров в день реально торгуется."""
    df = long_df.copy()
    df["day"] = pd.to_datetime(df["begin"]).dt.normalize()
    rows = []
    for cur, g in df.groupby("currency"):
        per_day = g.groupby("day").size()
        span_days = (g["day"].max() - g["day"].min()).days
        rows.append(
            {
                "currency": str(cur),
                "bars": int(len(g)),
                "trading_days": int(len(per_day)),
                "first_bar": str(g["begin"].min()),
                "last_bar": str(g["begin"].max()),
                "span_years": round(span_days / 365.25, 1) if span_days > 0 else 0.0,
                # доля календарных рабочих дней периода, в которые были сделки
                "share_days_traded": round(
                    len(per_day) / max(len(pd.bdate_range(g["day"].min(), g["day"].max())), 1), 3
                ),
                "bars_per_day_median": float(per_day.median()),
                "share_days_single_bar": round(float((per_day == 1).mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("bars_per_day_median", ascending=False).reset_index(drop=True)


def _diff_bps(moex: pd.Series, cbr: pd.Series) -> pd.Series:
    joined = pd.concat({"moex": moex, "cbr": cbr}, axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    return (joined["moex"] / joined["cbr"] - 1) * 1e4


def compare_levels(cbr_panel: pd.DataFrame, bar_panel: pd.DataFrame) -> pd.DataFrame:
    """Биржевое закрытие дня против фиксинга, опубликованного в тот же день.

    Оба ряда — рублей за 1 единицу валюты. `corr_changes` считается по дневным изменениям, а не
    по уровням: уровни скоррелированы трендом рубля и дали бы 0,99 при любой связи."""
    rows = []
    for cur in bar_panel.columns:
        if cur not in cbr_panel.columns:
            continue
        moex = daily_close(bar_panel[cur])
        cbr = cbr_panel[cur].dropna()
        cbr.index = cbr.index.normalize()
        d = _diff_bps(moex, cbr)
        if len(d) < 30:
            rows.append({"currency": cur, "n_days": int(len(d)), "note": "мало общих дней"})
            continue
        both = pd.concat({"moex": moex, "cbr": cbr}, axis=1).dropna()
        ch = both.pct_change().dropna()
        rows.append(
            {
                "currency": cur,
                "n_days": int(len(d)),
                "first_day": str(both.index.min().date()),
                "last_day": str(both.index.max().date()),
                "median_diff_bps": float(d.median()),
                "median_abs_diff_bps": float(d.abs().median()),
                "p90_abs_diff_bps": float(d.abs().quantile(0.9)),
                "corr_levels": float(both["moex"].corr(both["cbr"])),
                "corr_changes": float(ch["moex"].corr(ch["cbr"])) if len(ch) > 2 else np.nan,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def compare_by_hour(
    cbr_panel: pd.DataFrame, bar_panel: pd.DataFrame, hours: tuple[int, ...] = HOURS
) -> pd.DataFrame:
    """К какому часу биржа определяет фиксинг ТОГО ЖЕ дня.

    Для каждого часа H: расхождение «последний бар, начавшийся не позже H» против фиксинга дня и
    доля дисперсии дневного изменения фиксинга, объяснённая изменением биржи к этому часу
    (`r2_change` = квадрат корреляции изменений). Часы до 15 предшествуют публикации ЦБ —
    у них связь причинная, а не совпадение по времени."""
    rows = []
    for cur in bar_panel.columns:
        if cur not in cbr_panel.columns:
            continue
        cbr = cbr_panel[cur].dropna()
        cbr.index = cbr.index.normalize()
        cbr_ch = cbr.pct_change()
        for h in hours:
            moex = daily_close(bar_panel[cur], until_hour=h)
            d = _diff_bps(moex, cbr)
            if len(d) < 30:
                continue
            both = pd.concat({"moex": moex, "cbr_ch": cbr_ch}, axis=1).dropna()
            moex_ch = both["moex"].pct_change()
            pair = pd.concat({"m": moex_ch, "c": both["cbr_ch"]}, axis=1).dropna()
            r = float(pair["m"].corr(pair["c"])) if len(pair) > 2 else np.nan
            rows.append(
                {
                    "currency": cur,
                    "hour": h,
                    "before_cbr_publication": h < CBR_PUBLICATION_HOUR,
                    "n_days": int(len(d)),
                    "median_abs_diff_bps": float(d.abs().median()),
                    "p90_abs_diff_bps": float(d.abs().quantile(0.9)),
                    "r2_change": float(r**2) if not np.isnan(r) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_by_hour(by_hour: pd.DataFrame, path: Path) -> None:
    """Расхождение с фиксингом по часам: где кривая выходит на полку, там фиксинг уже определён."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if by_hour.empty:
        return
    fig, (ax_diff, ax_r2) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for cur, g in by_hour.groupby("currency"):
        g = g.sort_values("hour")
        ax_diff.plot(g["hour"], g["median_abs_diff_bps"], marker="o", label=str(cur))
        ax_r2.plot(g["hour"], g["r2_change"], marker="o", label=str(cur))
    for ax, title, ylabel in (
        (ax_diff, "Расхождение с фиксингом дня", "медиана |разницы|, бп"),
        (ax_r2, "Дневное изменение фиксинга,\nобъяснённое движением биржи", "R² изменений"),
    ):
        ax.axvline(CBR_PUBLICATION_HOUR, color="grey", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("час торгов (МСК), бары не позже него")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    ax_diff.set_yscale("log")
    ax_diff.legend(fontsize=8)
    fig.suptitle("К какому часу биржа определяет фиксинг ЦБ (пунктир — публикация ЦБ, ≈ 15:30)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_comparison(
    cbr_panel: pd.DataFrame,
    bar_panel: pd.DataFrame,
    out_dir: Path,
    long_df: pd.DataFrame | None = None,
) -> Path:
    from fxmoment.data.store import load_moex_meta, load_moex_raw
    from fxmoment.report import _md_table, git_hash

    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_moex_raw() if long_df is None else long_df
    liq = liquidity(raw)
    levels = compare_levels(cbr_panel, bar_panel)
    by_hour = compare_by_hour(cbr_panel, bar_panel)
    liq.to_csv(out_dir / "moex_liquidity.csv", index=False)
    levels.to_csv(out_dir / "cbr_vs_moex.csv", index=False)
    by_hour.to_csv(out_dir / "cbr_vs_moex_by_hour.csv", index=False)
    plot_by_hour(by_hour, out_dir / "cbr_vs_moex_by_hour.png")

    meta = load_moex_meta()
    lines = [
        "# Биржа против фиксинга ЦБ — сверка источников (ADR-0010)",
        "",
        f"Снимок Мосбиржи: {meta.get('fetched_at_utc', '?')}, "
        f"часовые свечи режима CETS, {meta.get('rows', '?')} баров. Код: `{git_hash()}`.",
        "",
        "## Плотность внутридневного рынка",
        "",
        "Бар в ISS появляется только при сделках, поэтому число баров в день — прямая мера того,",
        "есть ли внутри дня чему двигаться. `share_days_traded` — доля рабочих дней периода со сделками.",
        "",
        _md_table(liq.round(3)),
        "",
        "## Биржевое закрытие дня против фиксинга, опубликованного в тот же день",
        "",
        "Оба ряда — рублей за 1 единицу. `corr_changes` — по дневным изменениям (уровни",
        "скоррелированы общим трендом рубля и связь переоценивают).",
        "",
        _md_table(levels.round(3)),
        "",
        "## К какому часу биржа определяет сегодняшний фиксинг",
        "",
        f"ЦБ публикует курс дня около {CBR_PUBLICATION_HOUR}:30 МСК; строки с "
        "`before_cbr_publication` = True относятся к барам, заведомо ему предшествующим.",
        "`r2_change` — доля дисперсии дневного изменения фиксинга, объяснённая движением биржи к часу H.",
        "График — `cbr_vs_moex_by_hour.png`.",
        "",
        _md_table(by_hour.round(3)),
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir
