"""CLI: fetch / backtest / signals --as-of / check-texts."""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

from fxmoment.config import ALL_CURRENCIES, ANALYSIS_START, RAW_START


def cmd_fetch(args: argparse.Namespace) -> int:
    from fxmoment.data.cbr import DYNAMIC_URL, fetch_all
    from fxmoment.data.store import RAW_CSV, save_raw

    start = date.fromisoformat(args.start)
    end = date.today()
    df = fetch_all(start, end, ALL_CURRENCIES)
    save_raw(df, DYNAMIC_URL, args.start, end.isoformat())
    print(f"{len(df)} строк, {df['currency'].nunique()} валют → {RAW_CSV}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from fxmoment.backtest import run_backtest
    from fxmoment.data.store import load_panel
    from fxmoment.indicators import ALL_INDICATORS, BASE_INDICATORS
    from fxmoment.report import write_report

    panel = load_panel()
    inds = BASE_INDICATORS if args.no_ml else ALL_INDICATORS
    corridors = tuple(args.corridors.split(",")) if args.corridors else None
    kwargs = {"corridors": corridors} if corridors else {}
    result = run_backtest(panel, indicators=inds, analysis_start=args.start, **kwargs)
    out = write_report(result, panel)
    pd.set_option("display.width", 250)
    for h in (20, 5):
        print(f"\n=== h = {h}, допуск {args.tol:g} бп ===")
        print(result.summary(h=h, tol_bps=args.tol).round(3).to_string(index=False))
    stream_path = out / "stream_summary_h20_tol25.csv"
    if stream_path.exists():
        print("\n=== итоговый поток после политики, h = 20, допуск 25 бп ===")
        print(pd.read_csv(stream_path).round(3).to_string(index=False))
    print(f"\nотчёт → {out}")
    return 0


def cmd_signals(args: argparse.Namespace) -> int:
    from fxmoment.backtest import signals_as_of
    from fxmoment.data.store import load_panel
    from fxmoment.indicators import ALL_INDICATORS, BASE_INDICATORS
    from fxmoment.texts import render

    panel = load_panel()
    inds = BASE_INDICATORS if args.no_ml else ALL_INDICATORS
    df = signals_as_of(panel, args.as_of, indicators=inds, lookback=args.lookback)
    df = df[df["signal"]] if args.only_signals else df
    pd.set_option("display.width", 200)
    cols = ["date", "corridor", "indicator", "scenario", "signal", "strength", "speed", "rate", "facts"]
    print(df[cols].to_string(index=False))
    if args.texts:
        import json

        for _, row in df[df["signal"]].iterrows():
            title, body = render(
                row["corridor"], row["scenario"], row["indicator"], row["rate"], json.loads(row["facts"])
            )
            print(f"\n[{row['corridor']} {row['date']:%Y-%m-%d} {row['indicator']}] {title}\n  {body}")
    return 0


def cmd_check_texts(args: argparse.Namespace) -> int:
    from fxmoment.texts import check_text, library_texts

    bad = 0
    for name, title, body in library_texts():
        hits = check_text(title) + check_text(body)
        status = "ok" if not hits else "НАРУШЕНИЕ"
        print(f"{status:10} {name}: {title} — {body}")
        for frag, reason in hits:
            print(f"           ↳ «{frag}»: {reason}")
        bad += bool(hits)
    if args.text:
        hits = check_text(args.text)
        print("\nтекст:", "чисто" if not hits else "; ".join(f"«{f}»: {r}" for f, r in hits))
        bad += bool(hits)
    return 2 if bad else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fxmoment", description="Сигнальный слой «выгодный момент для перевода»")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="выгрузить курсы ЦБ в data/raw/")
    f.add_argument("--start", default=RAW_START)
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest", help="walk-forward бэктест → reports/latest/")
    b.add_argument("--start", default=ANALYSIS_START, help="начало окна анализа")
    b.add_argument("--corridors", default="", help="через запятую, по умолчанию все пять")
    b.add_argument("--no-ml", action="store_true", help="только базовые индикаторы")
    b.add_argument("--tol", type=float, default=25.0, help="допуск попадания для сводки, бп")
    b.set_defaults(func=cmd_backtest)

    s = sub.add_parser("signals", help="состояние индикаторов на дату среза")
    s.add_argument("--as-of", required=True, help="дата среза YYYY-MM-DD (дата публикации)")
    s.add_argument("--lookback", type=int, default=0, help="сколько предыдущих дней показать")
    s.add_argument("--only-signals", action="store_true")
    s.add_argument("--texts", action="store_true", help="напечатать тексты пушей для сработавших")
    s.add_argument("--no-ml", action="store_true")
    s.set_defaults(func=cmd_signals)

    c = sub.add_parser("check-texts", help="прогнать библиотеку текстов и (опц.) свой текст через чекер")
    c.add_argument("--text", default="")
    c.set_defaults(func=cmd_check_texts)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
