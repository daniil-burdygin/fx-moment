"""Walk-forward бэктест, калибровка и функция среза (ADR-0004)."""

from fxmoment.backtest.engine import BacktestResult, run_backtest, signals_as_of
from fxmoment.backtest.walkforward import Split, make_splits, split_for_date

__all__ = ["BacktestResult", "run_backtest", "signals_as_of", "Split", "make_splits", "split_for_date"]
