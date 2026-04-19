"""Walk-forward backtest engine for the MCMC derivatives strategy.

Execution model (T0 -- same session open/close):
    - Signal is generated at the END of bar i-1 (after close is known)
    - Trade executes on bar i: enter at open[i], exit at close[i]
    - P&L = (close[i] - open[i]) * direction * position_size
    - No look-ahead: the model at bar i only sees data from bars 0..i-1

Walk-forward warm-up:
    - The first `warmup_bars` bars are fed into MarkovModel and MCMCEngine
      for initialization but generate no trades.
    - Trading begins when `MarkovModel.is_ready == True` (n_sessions >= MIN_SESSIONS=5)
      AND bar index >= warmup_bars.

Usage::

    from src.backtest.bar_replay import BarReplay, BacktestConfig
    from src.backtest.data_fetcher import OhlcBar

    config = BacktestConfig(symbol="VN30F2506", confidence_threshold=0.75)
    bars = [...]  # list[OhlcBar]
    replay = BarReplay()
    trade_log, equity_curve = replay.run(bars, config)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.backtest.data_fetcher import OhlcBar
from src.logger import get_logger
from src.mcmc.markov_model import MarkovModel
from src.mcmc.mcmc_engine import MCMCEngine, MCMCResult

logger = get_logger("bar_replay")

# Total session length in bars for dt calculation.
# For daily bars: dt=1.0 (one full session per bar)
# For hourly bars: fraction of a trading day per bar
_SESSION_HOURS = 5.75  # VN30F session: 09:00-14:45 ≈ 5.75h


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""
    symbol: str = "VN30F2506"
    bar_type: str = "D"               # "D" or "60" (hourly)
    warmup_bars: int = 50             # bars fed to model before trading starts
    confidence_threshold: float = 0.75
    position_size: int = 1            # contracts per trade
    mh_iterations: int = 200          # MH sampler iterations (use fewer for speed)
    mh_burnin: int = 100              # MH burn-in
    n_paths: int = 2000               # GBM paths (fewer for batch backtest speed)
    seed: Optional[int] = 42
    commission_pct: float = 0.0       # commission as fraction of trade value (0 = none)
    fast_mode: bool = False           # if True, skip MCMCEngine, use Markov-only signal


@dataclass
class TradeRecord:
    """A single completed trade."""
    symbol: str
    entry_bar: int      # index in bars array
    exit_bar: int
    entry_date: str
    exit_date: str
    side: str           # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    pnl: float
    commission: float
    net_pnl: float
    p_up: float
    p_down: float
    confidence: float
    mcmc_elapsed_ms: float = 0.0

    @property
    def return_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price * 100.0 * (
            1 if self.side == "BUY" else -1
        )


class BarReplay:
    """Walk-forward backtest engine.

    Replays historical bars through MarkovModel + MCMCEngine, generating
    and executing signals with strict no-look-ahead guarantees.
    """

    def run(
        self,
        bars: List[OhlcBar],
        config: BacktestConfig,
    ) -> Tuple[List[TradeRecord], np.ndarray]:
        """Run the walk-forward simulation.

        Args:
            bars:   Historical bars sorted oldest-first.
            config: Backtest configuration.

        Returns:
            (trade_log, equity_curve)
            equity_curve: 1-D numpy array of cumulative P&L per bar,
                          same length as bars (0 during warm-up).
        """
        n = len(bars)
        if n < 2:
            logger.warning("BarReplay: insufficient bars (<2), returning empty result")
            return [], np.zeros(n)

        markov = MarkovModel(window=max(config.warmup_bars, 10))
        engine = MCMCEngine(
            n_iterations=config.mh_iterations,
            burnin=config.mh_burnin,
            n_paths=config.n_paths,
            seed=config.seed,
        ) if not config.fast_mode else None

        trades: List[TradeRecord] = []
        equity = np.zeros(n)
        cumulative_pnl = 0.0

        # Keep rolling log-returns for MCMCEngine
        log_returns: List[float] = []

        for i in range(n):
            bar = bars[i]

            # --- Always update model with PREVIOUS bar's data first ---
            # This ensures at step i, the model knows about bars[0..i-1].
            # No look-ahead: we never feed bar[i] into the model before
            # generating a signal for bar[i].
            if i > 0:
                prev = bars[i - 1]
                markov.update(prev.open, prev.close)
                if prev.open > 0 and prev.close > 0:
                    log_ret = float(np.log(prev.close / prev.open))
                    log_returns.append(log_ret)

            equity[i] = cumulative_pnl

            # --- Skip during warm-up or when model is not ready ---
            if i < config.warmup_bars or not markov.is_ready:
                continue

            prev = bars[i - 1]

            # --- Signal generation (model trained on bars 0..i-1 only) ---
            try:
                p_up, p_down = markov.predict()
            except RuntimeError:
                continue

            # MCMC prediction (or fast Markov-only)
            mcmc_elapsed = 0.0
            if engine is not None and len(log_returns) >= 3:
                dt = self._session_dt(config.bar_type)
                result = engine.run(
                    log_returns=np.array(log_returns),
                    current_price=prev.close,
                    p_up_markov=p_up,
                    dt=dt,
                )
                p_up_final = result.p_up
                p_down_final = result.p_down
                confidence = max(p_up_final, p_down_final)
                mcmc_elapsed = result.elapsed_ms
            else:
                # Fallback to Markov-only signal
                p_up_final = p_up
                p_down_final = p_down
                confidence = max(p_up, p_down)

            # --- Determine signal ---
            signal = None
            if p_up_final > config.confidence_threshold:
                signal = "BUY"
            elif p_down_final > config.confidence_threshold:
                signal = "SELL"

            # --- Execute trade on bar i (T0: enter open, exit close) ---
            if signal is not None and bar.open > 0 and bar.close > 0:
                direction = 1 if signal == "BUY" else -1
                entry_price = bar.open
                exit_price = bar.close
                gross_pnl = (exit_price - entry_price) * direction * config.position_size
                commission = (entry_price + exit_price) * config.position_size * config.commission_pct
                net_pnl = gross_pnl - commission
                cumulative_pnl += net_pnl

                trade = TradeRecord(
                    symbol=config.symbol,
                    entry_bar=i,
                    exit_bar=i,
                    entry_date=bar.date_str,
                    exit_date=bar.date_str,
                    side=signal,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=gross_pnl,
                    commission=commission,
                    net_pnl=net_pnl,
                    p_up=round(p_up_final, 4),
                    p_down=round(p_down_final, 4),
                    confidence=round(confidence, 4),
                    mcmc_elapsed_ms=round(mcmc_elapsed, 1),
                )
                trades.append(trade)
                equity[i] = cumulative_pnl

        total_trades = len(trades)
        wins = sum(1 for t in trades if t.net_pnl > 0)
        logger.info(
            "BarReplay: complete",
            extra={
                "symbol": config.symbol,
                "n_bars": n,
                "n_trades": total_trades,
                "wins": wins,
                "total_pnl": round(cumulative_pnl, 2),
            },
        )
        return trades, equity

    def _session_dt(self, bar_type: str) -> float:
        """Return dt (fraction of full session) for GBM simulation.

        For daily bars, the entire session is consumed: dt=1.0.
        For hourly bars, each bar is 1/session_hours of the full session.
        """
        if bar_type == "D":
            return 1.0
        try:
            bar_hours = float(bar_type) / 60.0
            return min(1.0, bar_hours / _SESSION_HOURS)
        except (ValueError, TypeError):
            return 1.0
