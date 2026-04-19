"""
Execution realism for TTM V2 backtests: latency (shifted fills) and slippage.

Does not alter signal generation or scoring — only how simulated fills are priced.

Assumptions:
  - ``OhlcBar.unix_ts`` is the bar period **open** (seconds); bar end = unix_ts + bar_seconds.
  - Signal is known at bar close; ``signal_timestamp`` = unix_ts(signal_bar) + bar_seconds.
  - Entry executes on the first bar whose open time >= signal_timestamp + latency_ms/1000.
  - Fill price before slippage: **open** of that execution bar (next tradeable price in bar data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

from src.backtest.bar_replay import TradeRecord
from src.backtest.data_fetcher import OhlcBar
from src.backtest.metrics import BacktestMetrics, compute_metrics

SlippageMode = Literal["none", "base", "worst_case"]


def resolution_to_bar_seconds(resolution: str) -> int:
    """Map DNSE-style resolution to bar length in seconds."""
    r = (resolution or "15").strip().upper()
    if r in ("1D", "D"):
        return 86400
    if r in ("1H", "60", "H"):
        return 3600
    if r in ("30", "30M"):
        return 1800
    if r in ("15", "15M"):
        return 900
    if r in ("5", "5M"):
        return 300
    if r in ("1", "1M"):
        return 60
    try:
        return max(60, int(r) * 60) if int(r) < 24 else int(r)
    except ValueError:
        return 900


@dataclass(frozen=True)
class ExecutionRealismConfig:
    """Latency + slippage grid parameters (V2 execution layer only)."""

    latency_ms: int = 0
    slippage_mode: SlippageMode = "none"
    bar_seconds: int = 900
    worst_case_slippage_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if int(self.latency_ms) < 0:
            raise ValueError("latency_ms must be >= 0")
        if self.slippage_mode not in ("none", "base", "worst_case"):
            raise ValueError("slippage_mode must be none|base|worst_case")


def signal_timestamp_unix(bar: OhlcBar, bar_seconds: int) -> int:
    """Wall-clock time (seconds) when the bar's close is known."""
    return int(bar.unix_ts) + int(bar_seconds)


def find_execution_bar_index(
    bars: Sequence[OhlcBar],
    signal_bar_index: int,
    latency_ms: int,
    bar_seconds: int,
) -> Optional[int]:
    """
    First bar index j such that bar open time >= signal_close_time + latency.

    Returns None if no such bar exists (not enough history).
    """
    if not bars or signal_bar_index < 0 or signal_bar_index >= len(bars):
        return None
    sig_close = signal_timestamp_unix(bars[signal_bar_index], bar_seconds)
    due = float(sig_close) + float(latency_ms) / 1000.0
    n = len(bars)
    for j in range(max(0, signal_bar_index), n):
        if float(bars[j].unix_ts) + 1e-9 >= due:
            return int(j)
    return None


def entry_due_unix(signal_bar: OhlcBar, latency_ms: int, bar_seconds: int) -> float:
    """Earliest time a fill may occur: signal bar close + latency."""
    return float(signal_timestamp_unix(signal_bar, bar_seconds)) + float(latency_ms) / 1000.0


def execution_timestamp_unix(bars: Sequence[OhlcBar], exec_bar_index: int) -> int:
    """Time of the execution bar open (first tradeable instant in our discrete model)."""
    return int(bars[exec_bar_index].unix_ts)


def _bid_ask_spread_proxy(bar: OhlcBar) -> float:
    """Proxy when Level-2 is unavailable: cap range-based spread vs mid."""
    h, lo = float(bar.high), float(bar.low)
    mid = max((h + lo) * 0.5, 1e-9)
    rng = max(h - lo, 1e-9)
    return float(min(rng / 8.0, 0.002 * mid))


def _volatility_price(
    bars: Sequence[OhlcBar],
    bar_index: int,
    lookback: int = 20,
) -> float:
    """Recent close-to-close volatility in price units (std of simple returns)."""
    i0 = max(1, int(bar_index) - int(lookback) + 1)
    rets: List[float] = []
    for k in range(i0, int(bar_index) + 1):
        c0 = float(bars[k - 1].close)
        c1 = float(bars[k].close)
        if c0 > 1e-12:
            rets.append((c1 - c0) / c0)
    if len(rets) < 2:
        c = float(bars[bar_index].close)
        h, lo = float(bars[bar_index].high), float(bars[bar_index].low)
        return float(0.01 * c * min(1.0, (h - lo) / max(c, 1e-9)))
    return float(np.std(np.asarray(rets, dtype=np.float64), ddof=1) * float(bars[bar_index].close))


def slippage_abs_at_bar(
    bars: Sequence[OhlcBar],
    bar_index: int,
    mode: SlippageMode,
    *,
    worst_case_multiplier: float = 2.0,
) -> float:
    """
    slippage = 0.5 * bid_ask_spread + 0.1 * volatility  (price units).

    Modes:
      - none: 0
      - base: formula as above
      - worst_case: multiplier * base
    """
    if mode == "none":
        return 0.0
    b = bars[int(bar_index)]
    spread = _bid_ask_spread_proxy(b)
    vol = _volatility_price(bars, int(bar_index))
    slip = 0.5 * spread + 0.1 * vol
    if mode == "worst_case":
        slip *= float(worst_case_multiplier)
    return float(max(0.0, slip))


def adjust_fill_price(
    side: str,
    *,
    is_entry: bool,
    base_price: float,
    slippage_abs: float,
) -> float:
    """
    Adverse selection: long buy / short sell pay more spread on entry;
    long sell / short buy on exit.
    """
    su = str(side).upper()
    if su == "LONG":
        if is_entry:
            return float(base_price) + float(slippage_abs)
        return float(base_price) - float(slippage_abs)
    if su == "SHORT":
        if is_entry:
            return float(base_price) - float(slippage_abs)
        return float(base_price) + float(slippage_abs)
    return float(base_price)


def v2_metrics_from_trade_jsonl_rows(
    trade_rows: List[Dict[str, Any]],
    *,
    n_bars: int,
    bar_type: str,
    nav_start: float = 100.0,
) -> BacktestMetrics:
    """
    Build :class:`BacktestMetrics` from CLOSED rows for model v2 (``compute_metrics``).

    Expects each row to have ``net_pnl`` or ``pnl`` and optional ``exit_bar_index`` for equity curve.
    """
    pnls: List[float] = []
    exit_ix: List[int] = []
    for row in trade_rows:
        if str(row.get("model") or "") != "v2":
            continue
        if row.get("event_type") != "trade" or row.get("event") != "CLOSED":
            continue
        p = row.get("net_pnl", row.get("pnl"))
        if p is None:
            continue
        pnls.append(float(p))
        eb = row.get("exit_bar_index", row.get("bar_index"))
        exit_ix.append(int(eb) if eb is not None else 0)

    if not pnls:
        m = BacktestMetrics()
        m.n_bars = n_bars
        return m

    eq = np.zeros(max(n_bars, max(exit_ix) + 1 if exit_ix else 0), dtype=np.float64)
    for p, ix in zip(pnls, exit_ix):
        if 0 <= ix < len(eq):
            eq[ix] += float(p)
    equity_curve = np.cumsum(eq)

    tlog: List[TradeRecord] = []
    for p in pnls:
        tlog.append(
            TradeRecord(
                symbol="TTM",
                entry_bar=0,
                exit_bar=0,
                entry_date="",
                exit_date="",
                side="BUY",
                entry_price=0.0,
                exit_price=0.0,
                pnl=float(p),
                commission=0.0,
                net_pnl=float(p),
                p_up=0.0,
                p_down=0.0,
                confidence=0.0,
                mcmc_elapsed_ms=0.0,
            )
        )
    return compute_metrics(tlog, equity_curve, n_bars=n_bars, bar_type=bar_type, nav_start=nav_start)


def format_execution_audit(
    *,
    signal_timestamp: str,
    execution_timestamp: str,
    signal_price: float,
    execution_price: float,
    latency_ms: int,
    slippage_mode: str,
    slippage_abs_entry: float,
    slippage_abs_exit: float,
) -> Dict[str, Any]:
    return {
        "signal_timestamp": signal_timestamp,
        "execution_timestamp": execution_timestamp,
        "signal_price": signal_price,
        "execution_price": execution_price,
        "latency_ms": int(latency_ms),
        "slippage_mode": str(slippage_mode),
        "slippage_abs_entry": float(slippage_abs_entry),
        "slippage_abs_exit": float(slippage_abs_exit),
        "exit_logic_note": "Exit triggers use bar close (signal path); fills apply slippage only.",
    }
