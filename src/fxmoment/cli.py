"""CLI: fetch / backtest / analyze / signals --as-of [--decide] / check-texts."""

from __future__ import annotations

import argparse
import json
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
    result = run_backtest(
        panel, indicators=inds, analysis_start=args.start, fixed_params=args.fixed_params, **kwargs
    )
    from fxmoment.data.store import repo_root

    out_dir = repo_root() / "reports" / ("fixed" if args.fixed_params else "latest")
    out = write_report(result, panel, out_dir)
    pd.set_option("display.width", 250)
    for h in (20, 5):
        print(f"\n=== h = {h}, допуск {args.tol:g} бп ===")
        print(result.summary(h=h, tol_bps=args.tol).round(3).to_string(index=False))
    for name, title in (
        ("stream_shape_summary.csv", "форма итогового потока по коридорам"),
        ("stream_summary_h20_tol25.csv", "точность итогового потока по сценариям, h = 20, допуск 25 бп"),
    ):
        path = out / name
        if path.exists() and path.stat().st_size > 1:
            print(f"\n=== {title} ===")
            print(pd.read_csv(path).round(3).to_string(index=False))
    print(f"\nотчёт → {out}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from fxmoment import analysis
    from fxmoment.data.store import load_panel

    panel = load_panel()
    result = analysis.load_result(panel)
    out = analysis.write_analysis(result, panel, source=args.source, k=args.k)
    pd.set_option("display.width", 250)
    for name, title in (
        ("price_of_waiting.csv", "цена ожидания"),
        ("transfer_compare.csv", f"перенос параметров с {args.source}"),
    ):
        df = pd.read_csv(out / name)
        cols = [c for c in df.columns if not c.endswith("_ci_lo") and not c.endswith("_ci_hi")]
        print(f"\n=== {title} ===")
        print(df[cols].round(3).to_string(index=False))
    print(f"\nанализы → {out}")
    return 0


def cmd_signals(args: argparse.Namespace) -> int:
    from fxmoment.backtest import signals_as_of
    from fxmoment.data.store import load_panel
    from fxmoment.indicators import ALL_INDICATORS, BASE_INDICATORS
    from fxmoment.texts import render

    panel = load_panel()
    inds = BASE_INDICATORS if args.no_ml else ALL_INDICATORS
    lookback = max(args.lookback, 15) if args.decide else args.lookback
    df = signals_as_of(panel, args.as_of, indicators=inds, lookback=lookback)
    shown = df[df["signal"]] if (args.only_signals or args.decide) else df
    pd.set_option("display.width", 200)
    cols = ["date", "corridor", "indicator", "scenario", "signal", "strength", "speed", "rate", "facts"]
    print(shown[cols].to_string(index=False))
    if args.texts:
        for _, row in df[df["signal"]].iterrows():
            title, body = render(
                row["corridor"], row["scenario"], row["indicator"], row["rate"], json.loads(row["facts"])
            )
            print(f"\n[{row['corridor']} {row['date']:%Y-%m-%d} {row['indicator']}] {title}\n  {body}")
    if args.decide:
        _print_decisions(panel, df, pd.Timestamp(args.as_of))
    return 0


def _print_decisions(panel: pd.DataFrame, state: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """Решение политики на дату среза: ранг и отключения — из матрицы последнего бэктеста по окнам
    до текущего, охлаждение — по событиям последних 15 дней публикации (история отправок раньше
    этого окна не моделируется)."""
    from fxmoment.backtest import make_splits, split_for_date
    from fxmoment.combine import PolicyParams, apply_policy, rank_from_history
    from fxmoment.data.store import repo_root
    from fxmoment.texts import render

    matrix_path = repo_root() / "reports" / "latest" / "matrix.csv"
    matrix = pd.read_csv(matrix_path) if matrix_path.exists() else pd.DataFrame()
    splits = make_splits(panel.loc[ANALYSIS_START:].index)
    split = split_for_date(splits, cutoff)
    fired = state[state["signal"]]
    print(f"\n=== решение политики на {cutoff:%Y-%m-%d} (ранг по окнам < {split.id}) ===")
    for corridor in state["corridor"].unique():
        rank, muted = rank_from_history(matrix, corridor, split.id) if len(matrix) else ({}, ())
        ev = fired[fired["corridor"] == corridor]
        calendar = panel[corridor].dropna().loc[:cutoff].index
        if ev.empty:
            print(f"{corridor}: событий нет — молчим")
            continue
        dec = apply_policy(ev, rank, calendar, PolicyParams(muted=muted))
        today = dec[dec["date"] == cutoff]
        sent = today[today["decision"] == "sent"]
        if len(sent):
            row = sent.iloc[0]
            title, body = render(
                corridor, row["push_scenario"], row["indicator"], row["rate"], json.loads(row["facts"])
            )
            print(f"{corridor}: ПУШ по {row['indicator']} — {title} / {body}")
        elif len(today):
            reasons = ", ".join(f"{r.indicator}: {r.decision}" for r in today.itertuples())
            print(f"{corridor}: не отправляем ({reasons})")
        else:
            print(f"{corridor}: на дату среза событий нет — молчим")
        if muted:
            print(f"   отключены по прошлым окнам: {', '.join(muted)}")


def cmd_check_texts(args: argparse.Namespace) -> int:
    from fxmoment.texts import check_message, check_text, library_texts

    bad = 0
    for name, title, body in library_texts():
        hits = check_message(title, body)
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
    b.add_argument(
        "--fixed-params",
        action="store_true",
        help="правила с априорными параметрами без калибровки → reports/fixed/ (контроль калибровки)",
    )
    b.set_defaults(func=cmd_backtest)

    a = sub.add_parser("analyze", help="анализы поверх reports/latest → reports/latest/analysis/")
    a.add_argument("--source", default="KZT", help="коридор-источник для переноса параметров")
    a.add_argument("--k", type=int, default=10, help="дней публикации ожидания подтверждения")
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("signals", help="состояние индикаторов на дату среза")
    s.add_argument("--as-of", required=True, help="дата среза YYYY-MM-DD (дата публикации)")
    s.add_argument("--lookback", type=int, default=0, help="сколько предыдущих дней показать")
    s.add_argument("--only-signals", action="store_true")
    s.add_argument("--texts", action="store_true", help="напечатать тексты пушей для сработавших")
    s.add_argument("--decide", action="store_true", help="применить политику потока и показать решение")
    s.add_argument("--no-ml", action="store_true")
    s.set_defaults(func=cmd_signals)

    c = sub.add_parser("check-texts", help="прогнать библиотеку текстов и (опц.) свой текст через чекер")
    c.add_argument("--text", default="")
    c.set_defaults(func=cmd_check_texts)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
