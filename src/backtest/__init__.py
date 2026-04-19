"""Backtest framework for MCMC derivatives strategy.

Modules:
    data_fetcher     -- Fetch and cache historical OHLC bars from DNSE REST API
    bar_replay       -- Walk-forward backtest engine (no look-ahead bias)
    metrics          -- Performance metrics: Sharpe, Sortino, MDD, win rate, CAGR
    statistical_tests -- Monte Carlo permutation test for strategy significance
"""

from src.backtest.data_fetcher import DataFetcher, OhlcBar
from src.backtest.bar_replay import BarReplay, BacktestConfig, TradeRecord
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.backtest.statistical_tests import monte_carlo_permutation, PermResult

__all__ = [
    "DataFetcher", "OhlcBar",
    "BarReplay", "BacktestConfig", "TradeRecord",
    "BacktestMetrics", "compute_metrics",
    "monte_carlo_permutation", "PermResult",
]
