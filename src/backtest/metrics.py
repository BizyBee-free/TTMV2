"""Performance metrics for backtest results.

Computes standard quantitative finance metrics from a trade log
and equity curve produced by BarReplay.

Metrics computed:
    - total_return_pct    : cumulative return over the period
    - cagr_pct            : compound annual growth rate
    - sharpe_ratio        : annualized Sharpe (risk-free = 0)
    - sortino_ratio       : annualized Sortino (downside deviation)
    - max_drawdown_pct    : peak-to-trough drawdown on equity curve
    - calmar_ratio        : CAGR / abs(max_drawdown)
    - win_rate_pct        : percentage of profitable trades
    - avg_win             : average P&L of winning trades
    - avg_loss            : average P&L of losing trades (negative)
    - profit_factor       : gross profit / abs(gross loss)
    - n_trades            : total trades
    - n_wins / n_losses   : breakdown
    - expectancy          : win_rate * avg_win + loss_rate * avg_loss
    - avg_mcmc_ms         : average MCMC computation time

Usage::

    from src.backtest.metrics import compute_metrics
    metrics = compute_metrics(trade_log, equity_curve, n_bars=90, bar_type="D")
    print(metrics.sharpe_ratio, metrics.max_drawdown_pct)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.backtest.bar_replay import TradeRecord

# Annualisation factors
_TRADING_DAYS_PER_YEAR = 252
_TRADING_HOURS_PER_YEAR = int(5.75 * _TRADING_DAYS_PER_YEAR)  # ~1449
_TRADING_30M_BARS_PER_YEAR = int(2 * _TRADING_HOURS_PER_YEAR)  # ~2898
_TRADING_15M_BARS_PER_YEAR = int(4 * _TRADING_HOURS_PER_YEAR)  # ~5796
_TRADING_5M_BARS_PER_YEAR = int(12 * _TRADING_HOURS_PER_YEAR)  # ~17388
_TRADING_4H_BARS_PER_YEAR = int(_TRADING_HOURS_PER_YEAR / 4)   # ~362


def _bars_per_year_from_bar_type(bar_type: str) -> int:
    """Map bar_type string to annualisation factor.

    Supported aliases:
      - Daily:  "D", "1D"
      - 4-hour: "4H", "240"
      - Hourly: "60", "1H", "H"
      - 30-min: "30", "30M"
      - 15-min: "15", "15M"
      - 5-min:  "5", "5M"
    """
    bt = (bar_type or "D").upper()
    if bt in {"D", "1D"}:
        return _TRADING_DAYS_PER_YEAR
    if bt in {"4H", "240"}:
        return _TRADING_4H_BARS_PER_YEAR
    if bt in {"60", "1H", "H"}:
        return _TRADING_HOURS_PER_YEAR
    if bt in {"30", "30M"}:
        return _TRADING_30M_BARS_PER_YEAR
    if bt in {"15", "15M"}:
        return _TRADING_15M_BARS_PER_YEAR
    if bt in {"5", "5M"}:
        return _TRADING_5M_BARS_PER_YEAR
    return _TRADING_DAYS_PER_YEAR


@dataclass
class BacktestMetrics:
    """Complete set of performance metrics for one backtest run."""
    # Basic
    n_bars: int = 0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_hold: int = 0          # bars with HOLD signal

    # Returns
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0

    # Trade quality
    win_rate_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Performance timing
    avg_mcmc_ms: float = 0.0
    total_mcmc_ms: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}

    def summary_table(self) -> str:
        """Return a Markdown table string for TEST_REPORT.md."""
        rows = [
            ("Bars", self.n_bars),
            ("Trades", self.n_trades),
            ("Win rate", f"{self.win_rate_pct:.1f}%"),
            ("Total P&L", f"{self.total_pnl:.2f}"),
            ("Total return", f"{self.total_return_pct:.2f}%"),
            ("CAGR", f"{self.cagr_pct:.2f}%"),
            ("Sharpe ratio", f"{self.sharpe_ratio:.3f}"),
            ("Sortino ratio", f"{self.sortino_ratio:.3f}"),
            ("Max drawdown", f"{self.max_drawdown_pct:.2f}%"),
            ("Calmar ratio", f"{self.calmar_ratio:.3f}"),
            ("Profit factor", f"{self.profit_factor:.3f}"),
            ("Expectancy", f"{self.expectancy:.4f}"),
            ("Avg win", f"{self.avg_win:.4f}"),
            ("Avg loss", f"{self.avg_loss:.4f}"),
            ("Avg MCMC ms", f"{self.avg_mcmc_ms:.1f}"),
        ]
        lines = ["| Metric | Value |", "|--------|-------|"]
        for name, val in rows:
            lines.append(f"| {name} | {val} |")
        return "\n".join(lines)


def compute_metrics(
    trade_log: List[TradeRecord],
    equity_curve: np.ndarray,
    n_bars: int,
    bar_type: str = "D",
    nav_start: float = 100.0,
) -> BacktestMetrics:
    """Compute all performance metrics from trade log + equity curve.

    Args:
        trade_log:    List of TradeRecord from BarReplay.run().
        equity_curve: Cumulative P&L array (same length as bars).
        n_bars:       Total bars in the period (for CAGR annualisation).
        bar_type:     Timeframe code used for annualisation.
                      Supported: "D"/"1D", "4H"/"240", "60"/"1H",
                      "30"/"30M", "15"/"15M", "5"/"5M".
        nav_start:    Notional starting NAV (for % return calculation).
                      Defaults to 100 (treat P&L as points/contracts).

    Returns:
        BacktestMetrics with all fields populated.
    """
    m = BacktestMetrics()
    m.n_bars = n_bars
    m.n_trades = len(trade_log)

    if m.n_trades == 0:
        return m

    # ------------------------------------------------------------------ #
    # Basic trade stats
    # ------------------------------------------------------------------ #
    pnls = np.array([t.net_pnl for t in trade_log])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    m.n_wins = int(len(wins))
    m.n_losses = int(len(losses))
    m.total_pnl = float(np.sum(pnls))
    m.total_return_pct = (m.total_pnl / nav_start) * 100.0 if nav_start > 0 else 0.0
    m.win_rate_pct = (m.n_wins / m.n_trades) * 100.0
    m.avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    m.avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0

    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_rate = m.n_wins / m.n_trades
    loss_rate = m.n_losses / m.n_trades
    m.expectancy = win_rate * m.avg_win + loss_rate * m.avg_loss

    # ------------------------------------------------------------------ #
    # CAGR
    # ------------------------------------------------------------------ #
    bars_per_year = _bars_per_year_from_bar_type(bar_type)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1.0

    final_nav = nav_start + m.total_pnl
    if years > 0 and nav_start > 0 and final_nav > 0:
        m.cagr_pct = ((final_nav / nav_start) ** (1.0 / years) - 1.0) * 100.0
    else:
        m.cagr_pct = 0.0

    # ------------------------------------------------------------------ #
    # Sharpe ratio (per-trade returns, annualized)
    # ------------------------------------------------------------------ #
    if len(pnls) > 1:
        # Convert P&L to return fractions relative to nav_start
        ret_series = pnls / nav_start
        mean_ret = float(np.mean(ret_series))
        std_ret = float(np.std(ret_series, ddof=1))
        if std_ret > 1e-10:
            # Annualise: trades_per_year ≈ n_trades / years
            trades_per_year = m.n_trades / years if years > 0 else m.n_trades
            m.sharpe_ratio = (mean_ret / std_ret) * math.sqrt(trades_per_year)
        else:
            m.sharpe_ratio = 0.0

        # Sortino: only downside deviation
        down_ret = ret_series[ret_series < 0]
        if len(down_ret) > 0:
            downside_std = float(np.std(down_ret, ddof=1))
            if downside_std > 0:
                m.sortino_ratio = (mean_ret / downside_std) * math.sqrt(trades_per_year)
            else:
                m.sortino_ratio = float("inf") if mean_ret > 0 else 0.0
        else:
            m.sortino_ratio = float("inf") if mean_ret > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Max drawdown on equity curve
    # ------------------------------------------------------------------ #
    if len(equity_curve) > 0:
        nav_curve = nav_start + equity_curve
        peak = nav_curve[0]
        max_dd = 0.0
        max_dd_abs = 0.0
        for nav in nav_curve:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
                max_dd_abs = peak - nav
        m.max_drawdown_pct = max_dd * 100.0
        m.max_drawdown_abs = max_dd_abs

    # ------------------------------------------------------------------ #
    # Calmar
    # ------------------------------------------------------------------ #
    if m.max_drawdown_pct > 0:
        m.calmar_ratio = m.cagr_pct / m.max_drawdown_pct
    else:
        m.calmar_ratio = float("inf") if m.cagr_pct > 0 else 0.0

    # ------------------------------------------------------------------ #
    # MCMC timing
    # ------------------------------------------------------------------ #
    mcmc_times = [t.mcmc_elapsed_ms for t in trade_log if t.mcmc_elapsed_ms > 0]
    if mcmc_times:
        m.avg_mcmc_ms = float(np.mean(mcmc_times))
        m.total_mcmc_ms = float(np.sum(mcmc_times))

    return m


def compare_metrics(is_metrics: BacktestMetrics, oos_metrics: BacktestMetrics) -> str:
    """Generate a Markdown comparison table for in-sample vs out-of-sample."""
    rows = [
        ("Sharpe ratio",   f"{is_metrics.sharpe_ratio:.3f}",   f"{oos_metrics.sharpe_ratio:.3f}"),
        ("Sortino ratio",  f"{is_metrics.sortino_ratio:.3f}",  f"{oos_metrics.sortino_ratio:.3f}"),
        ("Max drawdown",   f"{is_metrics.max_drawdown_pct:.2f}%", f"{oos_metrics.max_drawdown_pct:.2f}%"),
        ("Win rate",       f"{is_metrics.win_rate_pct:.1f}%",  f"{oos_metrics.win_rate_pct:.1f}%"),
        ("Total return",   f"{is_metrics.total_return_pct:.2f}%", f"{oos_metrics.total_return_pct:.2f}%"),
        ("CAGR",          f"{is_metrics.cagr_pct:.2f}%",      f"{oos_metrics.cagr_pct:.2f}%"),
        ("N trades",       str(is_metrics.n_trades),            str(oos_metrics.n_trades)),
        ("Profit factor",  f"{is_metrics.profit_factor:.3f}",  f"{oos_metrics.profit_factor:.3f}"),
    ]
    lines = ["| Metric | In-Sample | Out-of-Sample |", "|--------|-----------|---------------|"]
    for name, is_val, oos_val in rows:
        lines.append(f"| {name} | {is_val} | {oos_val} |")

    # Add degradation ratio for Sharpe
    if is_metrics.sharpe_ratio != 0:
        ratio = oos_metrics.sharpe_ratio / is_metrics.sharpe_ratio
        lines.append(f"\n**OOS/IS Sharpe ratio**: {ratio:.2f} (target >= 0.50)")

    return "\n".join(lines)
