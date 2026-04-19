"""Walk-forward backtest engine for the HMM regime strategy.

Mirrors the interface of BarReplay (same inputs/outputs) so that
compute_metrics() and monte_carlo_permutation() work unchanged.

Execution model (T0 — same session):
    - At bar i, use features from bars[0:i] (no look-ahead)
    - HMM is re-fitted on the expanding window each bar
    - Signal: BULL → Long, BEAR → Short, FLAT → skip
    - Trade: enter at open[i], exit at close[i]
    - P&L = (close[i] - open[i]) * direction * position_size - commission

Walk-forward warm-up:
    - The first `warmup_bars` bars build the initial feature history.
    - The HMM is NOT fitted until we have at least max(warmup_bars, MIN_FIT_BARS)
      feature rows, to ensure stable Baum-Welch convergence.

Performance note:
    Re-fitting GaussianHMM on every bar is expensive (O(n * k^2) per bar).
    Set `refit_every` > 1 to refit less frequently (e.g. every 5 bars)
    while updating the feature window every bar.

Usage::

    from src.backtest.hmm_replay import HMMBarReplay, HMMBacktestConfig
    from src.backtest.data_fetcher import OhlcBar
    from src.hmm.feature_engineer import HMMConfig

    hmm_cfg = HMMConfig(k_states=3)
    config  = HMMBacktestConfig(
        symbol="VN30F1M",
        hmm_config=hmm_cfg,
        confidence_threshold=0.65,
        warmup_bars=30,
    )
    replay = HMMBarReplay()
    trades, equity = replay.run(bars, config)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from src.backtest.bar_replay import TradeRecord   # reuse same dataclass
from src.backtest.data_fetcher import OhlcBar
from src.hmm.feature_engineer import HMMConfig, HMMFeatureEngineer
from src.hmm.hmm_signal import FLAT, LONG, SHORT, HMMSignalGenerator
from src.hmm.regime_model import HMMRegimeModel
from src.logger import get_logger

if TYPE_CHECKING:
    from src.hmm.hmm_live_trace import HMMTraceSnapshot

logger = get_logger("hmm_replay")

# Minimum feature rows before attempting first fit
_MIN_FIT_BARS = 20


@dataclass
class HMMBacktestConfig:
    """Configuration for an HMM walk-forward backtest run."""
    symbol: str = "VN30F1M"
    hmm_config: HMMConfig = field(default_factory=HMMConfig)
    warmup_bars: int = 30
    confidence_threshold: float = 0.65
    position_size: int = 1
    commission_pct: float = 0.0       # fraction of trade value
    refit_every: int = 1              # refit HMM every N bars (1 = every bar)
    seed: Optional[int] = 42
    sliding_window_bars: Optional[int] = None  # if set, fit only on last W bars (not expanding)
    # Per-bar index close (same order as bars) when hmm_config.use_basis is True
    index_closes: Optional[List[float]] = None
    # Per-bar OI level (same order as bars) when hmm_config.use_open_interest is True
    open_interest: Optional[List[float]] = None


def _ic_slice(
    index_closes: Optional[List[float]],
    start: int,
    end: int,
) -> Optional[List[float]]:
    if index_closes is None:
        return None
    return index_closes[start:end]


def _oi_slice(
    open_interest: Optional[List[float]],
    start: int,
    end: int,
) -> Optional[List[float]]:
    if open_interest is None:
        return None
    return open_interest[start:end]


class HMMBarReplay:
    """Walk-forward backtest engine using HMM regime detection.

    Produces the same (List[TradeRecord], equity_curve) output as BarReplay
    so that compute_metrics() can be called directly.
    """

    def run(
        self,
        bars: List[OhlcBar],
        config: HMMBacktestConfig,
    ) -> Tuple[List[TradeRecord], np.ndarray]:
        """Run the walk-forward HMM simulation.

        Args:
            bars:   Historical bars, oldest-first.
            config: HMMBacktestConfig.

        Returns:
            (trade_log, equity_curve) where equity_curve is a 1-D array
            of cumulative P&L (length == len(bars)).
        """
        n = len(bars)
        if n < 2:
            logger.warning("HMMBarReplay: insufficient bars (<2)")
            return [], np.zeros(n)

        engineer  = HMMFeatureEngineer(config.hmm_config)
        model     = HMMRegimeModel(config.hmm_config)
        signal_gen: Optional[HMMSignalGenerator] = None

        trades:        List[TradeRecord] = []
        equity         = np.zeros(n)
        cumulative_pnl = 0.0
        min_fit        = max(config.warmup_bars, _MIN_FIT_BARS)

        for i in range(n):
            equity[i] = cumulative_pnl

            # Need at least min_fit bars in history before first trade
            if i < min_fit:
                continue

            # Re-fit HMM on expanding or sliding window bars[0:i]
            if (i - min_fit) % config.refit_every == 0:
                if config.sliding_window_bars:
                    start = max(0, i - config.sliding_window_bars)
                    train_slice = bars[start:i]
                else:
                    start = 0
                    train_slice = bars[:i]
                ic_train = _ic_slice(config.index_closes, start, i)
                oi_train = _oi_slice(config.open_interest, start, i)
                X_train = engineer.compute(
                    train_slice, index_closes=ic_train, open_interest=oi_train
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = model.fit(X_train)
                if fitted:
                    signal_gen = HMMSignalGenerator(model.state_labels)

            if not model.is_fitted or signal_gen is None:
                continue

            # Feature vector for predicting signal for bar[i]
            # Uses bars[0:i] — no look-ahead
            X_pred = engineer.compute(
                bars[:i],
                index_closes=_ic_slice(config.index_closes, 0, i),
                open_interest=_oi_slice(config.open_interest, 0, i),
            )
            if X_pred.shape[0] == 0:
                continue

            proba   = model.get_latest_state_proba(X_pred)
            direction = signal_gen.get_direction(proba, config.confidence_threshold)

            if direction == FLAT:
                continue

            # T0 execution: enter at open[i], exit at close[i]
            bar         = bars[i]
            entry_price = bar.open
            exit_price  = bar.close

            raw_pnl  = (exit_price - entry_price) * direction * config.position_size
            comm     = (entry_price + exit_price) * config.position_size * config.commission_pct
            net_pnl  = raw_pnl - comm
            cumulative_pnl += net_pnl
            equity[i] = cumulative_pnl

            side = "BUY" if direction == LONG else "SELL"
            dom_label = signal_gen.dominant_state_label(proba)
            dom_prob  = signal_gen.dominant_state_proba(proba)

            trades.append(TradeRecord(
                symbol      = config.symbol,
                entry_bar   = i,
                exit_bar    = i,
                entry_date  = bar.date_str,
                exit_date   = bar.date_str,
                side        = side,
                entry_price = entry_price,
                exit_price  = exit_price,
                pnl         = raw_pnl,
                commission  = comm,
                net_pnl     = net_pnl,
                p_up        = float(proba[0]) if direction == LONG else 0.0,
                p_down      = float(proba[-1]) if direction == SHORT else 0.0,
                confidence  = dom_prob,
                mcmc_elapsed_ms = 0.0,
            ))

        logger.info(
            "HMMBarReplay: finished",
            extra={"n_bars": n, "n_trades": len(trades), "final_pnl": cumulative_pnl},
        )
        return trades, equity

    def last_bar_signal(
        self,
        bars: List[OhlcBar],
        config: HMMBacktestConfig,
        return_trace: bool = False,
    ) -> Tuple[int, str, str, Optional["HMMTraceSnapshot"]]:
        """Compute HMM direction for the **last** bar only (same logic as ``run`` loop).

        Returns:
            (direction, dominant_label, detail, trace) where trace is set when
            ``return_trace=True`` (live debug: state probs, guards, feature tail).
        """
        from src.hmm.hmm_live_trace import (
            HMMTraceSnapshot,
            build_trace_from_ok,
            trace_early_exit,
        )

        n = len(bars)
        last_bar = bars[-1] if bars else None
        if n < 2:
            tr = (
                trace_early_exit(
                    n_bars=n,
                    detail="insufficient_bars",
                    hmm_cfg=config.hmm_config,
                    last_bar=last_bar,
                    reason="need_at_least_2_bars",
                    confidence_threshold=config.confidence_threshold,
                )
                if return_trace
                else None
            )
            return FLAT, "", "insufficient_bars", tr

        engineer = HMMFeatureEngineer(config.hmm_config)
        model = HMMRegimeModel(config.hmm_config)
        signal_gen: Optional[HMMSignalGenerator] = None

        min_fit = max(config.warmup_bars, _MIN_FIT_BARS)
        i = n - 1
        if i < min_fit:
            tr = (
                trace_early_exit(
                    n_bars=n,
                    detail="warmup",
                    hmm_cfg=config.hmm_config,
                    last_bar=last_bar,
                    reason=f"bar_index_{i}_lt_min_fit_{min_fit}",
                    confidence_threshold=config.confidence_threshold,
                )
                if return_trace
                else None
            )
            return FLAT, "", "warmup", tr

        for j in range(min_fit, i + 1):
            if (j - min_fit) % config.refit_every == 0:
                if config.sliding_window_bars:
                    start = max(0, j - config.sliding_window_bars)
                    train_slice = bars[start:j]
                else:
                    start = 0
                    train_slice = bars[:j]
                ic_train = _ic_slice(config.index_closes, start, j)
                oi_train = _oi_slice(config.open_interest, start, j)
                X_train = engineer.compute(
                    train_slice, index_closes=ic_train, open_interest=oi_train
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = model.fit(X_train)
                if fitted:
                    signal_gen = HMMSignalGenerator(model.state_labels)

        if not model.is_fitted or signal_gen is None:
            tr = (
                trace_early_exit(
                    n_bars=n,
                    detail="model_not_fitted",
                    hmm_cfg=config.hmm_config,
                    last_bar=last_bar,
                    reason="hmm_fit_failed_or_insufficient_train_rows",
                    confidence_threshold=config.confidence_threshold,
                )
                if return_trace
                else None
            )
            return FLAT, "", "model_not_fitted", tr

        X_pred = engineer.compute(
            bars[:i],
            index_closes=_ic_slice(config.index_closes, 0, i),
            open_interest=_oi_slice(config.open_interest, 0, i),
        )
        if X_pred.shape[0] == 0:
            tr = (
                trace_early_exit(
                    n_bars=n,
                    detail="no_features",
                    hmm_cfg=config.hmm_config,
                    last_bar=last_bar,
                    reason="empty_feature_matrix",
                    confidence_threshold=config.confidence_threshold,
                )
                if return_trace
                else None
            )
            return FLAT, "", "no_features", tr

        proba = model.get_latest_state_proba(X_pred)
        direction = signal_gen.get_direction(proba, config.confidence_threshold)
        label = signal_gen.dominant_state_label(proba)
        tr: Optional[HMMTraceSnapshot] = None
        if return_trace:
            tr = build_trace_from_ok(
                n=n,
                detail="ok",
                direction=direction,
                label=label,
                confidence_threshold=config.confidence_threshold,
                hmm_cfg=config.hmm_config,
                proba=proba,
                state_labels_map=model.state_labels,
                X_pred=X_pred,
                last_bar=last_bar,
                state_means=model.state_means,
            )
        return direction, label, "ok", tr
