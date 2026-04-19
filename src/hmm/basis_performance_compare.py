"""Side-by-side HMM walk-forward metrics: with vs without basis (4 vs 3 features).

Used by tests and ``scripts/compare_hmm_basis.py`` to report Sharpe, P&L, MDD,
timeframe, and calendar span of the bar window.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

from src.backtest.data_fetcher import OhlcBar
from src.backtest.hmm_replay import HMMBacktestConfig, HMMBarReplay
from src.backtest.metrics import compute_metrics
from src.hmm.feature_engineer import HMMConfig


@dataclass(frozen=True)
class HMMPerfSnapshot:
    """One HMM backtest run summary for comparison tables."""
    variant: str
    timeframe: str
    period_start: str
    period_end: str
    n_bars: int
    sharpe_ratio: float
    total_pnl: float
    max_drawdown_pct: float
    total_return_pct: float
    n_trades: int


def run_hmm_perf_snapshot(
    bars: List[OhlcBar],
    *,
    variant: str,
    timeframe: str,
    bar_type: str,
    hmm_config: HMMConfig,
    index_closes: Optional[List[float]],
    open_interest: Optional[List[float]] = None,
    warmup_bars: int = 30,
    confidence_threshold: float = 0.65,
    refit_every: int = 1,
    symbol: str = "TEST",
    commission_pct: float = 0.0,
    sliding_window_bars: Optional[int] = None,
) -> HMMPerfSnapshot:
    """Run ``HMMBarReplay`` + ``compute_metrics`` and return a compact snapshot."""
    use_basis = hmm_config.use_basis
    if use_basis:
        if index_closes is None or len(index_closes) != len(bars):
            raise ValueError(
                "index_closes with len(bars) required when hmm_config.use_basis is True"
            )
    else:
        index_closes = None

    if hmm_config.use_open_interest:
        if open_interest is None or len(open_interest) != len(bars):
            raise ValueError(
                "open_interest with len(bars) required when hmm_config.use_open_interest is True"
            )
    else:
        open_interest = None

    cfg = HMMBacktestConfig(
        symbol=symbol,
        hmm_config=hmm_config,
        warmup_bars=warmup_bars,
        confidence_threshold=confidence_threshold,
        position_size=1,
        commission_pct=commission_pct,
        refit_every=refit_every,
        sliding_window_bars=sliding_window_bars,
        index_closes=index_closes,
        open_interest=open_interest,
    )
    replay = HMMBarReplay()
    trades, equity = replay.run(bars, cfg)
    m = compute_metrics(trades, equity, len(bars), bar_type=bar_type)

    start = bars[0].date_str if bars else ""
    end = bars[-1].date_str if bars else ""
    return HMMPerfSnapshot(
        variant=variant,
        timeframe=timeframe,
        period_start=start,
        period_end=end,
        n_bars=len(bars),
        sharpe_ratio=m.sharpe_ratio,
        total_pnl=m.total_pnl,
        max_drawdown_pct=m.max_drawdown_pct,
        total_return_pct=m.total_return_pct,
        n_trades=m.n_trades,
    )


def compare_basis_vs_no_basis(
    bars: List[OhlcBar],
    index_closes: List[float],
    *,
    base_hmm_config: HMMConfig,
    timeframe: str,
    bar_type: str,
    symbol: str = "TEST",
    warmup_bars: int = 30,
    confidence_threshold: float = 0.65,
    refit_every: int = 1,
    commission_pct: float = 0.0,
    sliding_window_bars: Optional[int] = None,
) -> Tuple[HMMPerfSnapshot, HMMPerfSnapshot]:
    """Same OHLC path; first snapshot without basis, second with basis (4 features).

    ``index_closes`` must align with ``bars`` (one index close per bar).
    """
    if len(index_closes) != len(bars):
        raise ValueError("index_closes must have same length as bars")

    cfg_no = replace(base_hmm_config, use_basis=False)
    cfg_yes = replace(base_hmm_config, use_basis=True)

    snap_no = run_hmm_perf_snapshot(
        bars,
        variant="HMM (no basis)",
        timeframe=timeframe,
        bar_type=bar_type,
        hmm_config=cfg_no,
        index_closes=None,
        symbol=symbol,
        warmup_bars=warmup_bars,
        confidence_threshold=confidence_threshold,
        refit_every=refit_every,
        commission_pct=commission_pct,
        sliding_window_bars=sliding_window_bars,
    )
    snap_yes = run_hmm_perf_snapshot(
        bars,
        variant="HMM (with basis)",
        timeframe=timeframe,
        bar_type=bar_type,
        hmm_config=cfg_yes,
        index_closes=index_closes,
        symbol=symbol,
        warmup_bars=warmup_bars,
        confidence_threshold=confidence_threshold,
        refit_every=refit_every,
        commission_pct=commission_pct,
        sliding_window_bars=sliding_window_bars,
    )
    return snap_no, snap_yes


def snapshots_to_markdown(a: HMMPerfSnapshot, b: HMMPerfSnapshot) -> str:
    """Two-row Markdown table: Sharpe, P&L, MDD, timeframe, period (start->end), bars, trades."""
    lines = [
        "| Variant | Timeframe | Period (start->end) | Bars | Trades | Sharpe | Total P&L | MDD % | Return % |",
        "|---------|-----------|---------------------|------|--------|--------|-----------|-------|----------|",
        _row(a),
        _row(b),
    ]
    return "\n".join(lines)


def _row(s: HMMPerfSnapshot) -> str:
    period = f"{s.period_start}->{s.period_end}"
    return (
        f"| {s.variant} | {s.timeframe} | {period} | {s.n_bars} | {s.n_trades} | "
        f"{s.sharpe_ratio:.4f} | {s.total_pnl:.4f} | {s.max_drawdown_pct:.4f} | "
        f"{s.total_return_pct:.4f} |"
    )
