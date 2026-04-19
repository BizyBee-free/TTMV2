"""MCMC-based T0 derivatives trading strategy.

Strategy summary:
    1. On each quote tick, check if a new OHLC candle has closed since
       the last MCMC computation.
    2. If yes, update the Markov model with the closed candle and run the
       full MCMC engine (MH sampling + 10k GBM paths).
    3. If P(UP) > MCMC_CONFIDENCE_THRESHOLD and we are flat -> BUY signal.
       If P(DOWN) > MCMC_CONFIDENCE_THRESHOLD and we are flat -> SELL signal.
       If we hold a position and the reverse probability exceeds threshold -> EXIT.
    4. T0 safety net: if time >= MCMC_FORCE_CLOSE_MINUTE, force-close any
       open position using a market-priced limit order and stop generating
       new signals.
    5. Cooldown: after each round-trip (open+close), wait MCMC_COOLDOWN_TICKS
       quote ticks before allowing the next signal.

Integration:
    This class extends StrategyBase and is wired by the paper_test.py runner:

        strategy = MCMCDerivativesStrategy("mcmc_t0", symbols=[symbol])
        strategy.set_risk_manager(risk_mgr)
        strategy.start(order_mgr, tracker, data_buffer)
        market_data_mgr.on_quote(strategy.handle_quote)
        market_data_mgr.on_ohlc(strategy.handle_ohlc)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import List, Optional

import numpy as np

from src.config import MarketType, OrderSide, OrderType, Settings, get_settings
from src.logger import get_logger
from src.mcmc.markov_model import MarkovModel, MarkovState
from src.strategies.exit_rules import exit_config_from_settings, should_exit
from src.mcmc.mcmc_engine import MCMCEngine, MCMCResult
from src.order_manager import ManagedOrder, OrderManager
from src.position_tracker import PositionTracker
from src.risk_manager import RiskManager
from src.strategy_base import Signal, SignalEvent, StrategyBase

logger = get_logger("mcmc_derivatives")

# Total VN30F session length in minutes: 09:00-14:30 = 330 min
_SESSION_TOTAL_MINUTES = 330.0
_SESSION_START_HOUR = 9
_SESSION_START_MINUTE = 0


class MCMCDerivativesStrategy(StrategyBase):
    """T0 MCMC strategy for VN30 derivatives (or any configured symbol).

    Args:
        name:           Strategy name for logging.
        symbols:        List with exactly one derivatives symbol.
        settings:       Application settings.
        markov_window:  Override for MCMC_ROLLING_WINDOW.
        mh_iterations:  Override for MCMC_MH_ITERATIONS.
        mh_burnin:      Override for MCMC_MH_BURNIN.
        n_paths:        Override for MCMC_NUM_PATHS.
        confidence:     Override for MCMC_CONFIDENCE_THRESHOLD.
        seed:           Optional random seed for MCMCEngine reproducibility.
    """

    def __init__(
        self,
        name: str = "mcmc_t0",
        symbols: Optional[List[str]] = None,
        settings: Optional[Settings] = None,
        markov_window: Optional[int] = None,
        mh_iterations: Optional[int] = None,
        mh_burnin: Optional[int] = None,
        n_paths: Optional[int] = None,
        confidence: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        cfg = settings or get_settings()
        sym = symbols or [cfg.MCMC_DERIVATIVE_SYMBOL]
        super().__init__(name=name, symbols=sym, settings=cfg)

        self._markov = MarkovModel(
            window=markov_window or cfg.MCMC_ROLLING_WINDOW
        )
        self._mcmc = MCMCEngine(
            n_iterations=mh_iterations or cfg.MCMC_MH_ITERATIONS,
            burnin=mh_burnin or cfg.MCMC_MH_BURNIN,
            n_paths=n_paths or cfg.MCMC_NUM_PATHS,
            seed=seed,
        )

        self._confidence = confidence or cfg.MCMC_CONFIDENCE_THRESHOLD
        self._force_close_minute = cfg.MCMC_FORCE_CLOSE_MINUTE
        self._position_size = cfg.MCMC_POSITION_SIZE
        self._cooldown_ticks = cfg.MCMC_COOLDOWN_TICKS
        self._exit_config = exit_config_from_settings(cfg)

        self._risk_mgr: Optional[RiskManager] = None
        self._bars_in_trade = 0
        self._position_was_flat = True

        # Internal state
        self._last_ohlc_time: Optional[float] = None
        self._last_mcmc_result: Optional[MCMCResult] = None
        self._last_ohlc_candles: list = []

        self._t0_closed = False          # force-close sent this session
        self._cooldown_remaining = 0     # ticks to wait
        self._round_trips = 0
        self._open_side: Optional[OrderSide] = None

        # Performance tracking
        self._mcmc_calls = 0
        self._mcmc_total_ms = 0.0
        self._signals_skipped_cooldown = 0
        self._signals_skipped_risk = 0

    # ------------------------------------------------------------------ #
    # Wiring helpers
    # ------------------------------------------------------------------ #

    def set_risk_manager(self, risk_mgr: RiskManager) -> None:
        """Wire the RiskManager before calling start()."""
        self._risk_mgr = risk_mgr
        risk_mgr.on_stoploss(self._on_stoploss_triggered)

    # ------------------------------------------------------------------ #
    # StrategyBase lifecycle
    # ------------------------------------------------------------------ #

    def on_start(self) -> None:
        logger.info(
            f"[{self.name}] Strategy starting",
            extra={
                "symbol": self.symbols[0],
                "confidence": self._confidence,
                "force_close_minute": self._force_close_minute,
                "cooldown_ticks": self._cooldown_ticks,
            },
        )
        # Pre-load OHLC history into Markov model
        if self._buffer is not None:
            asyncio.ensure_future(self._preload_history())

    def on_stop(self) -> None:
        logger.info(
            f"[{self.name}] Strategy stopping",
            extra={
                "round_trips": self._round_trips,
                "mcmc_calls": self._mcmc_calls,
                "avg_mcmc_ms": round(
                    self._mcmc_total_ms / max(1, self._mcmc_calls), 1
                ),
            },
        )

    # ------------------------------------------------------------------ #
    # Market data handlers
    # ------------------------------------------------------------------ #

    def on_tick(self, symbol: str, quote=None, trade=None) -> Optional[SignalEvent]:
        """Called on every quote tick. Entry point for all signal logic."""
        if symbol not in self.symbols:
            return None

        current_price = self._extract_price(quote, trade)
        if current_price <= 0:
            return None

        # 1. T0 force-close check (highest priority)
        if self._is_past_force_close_time():
            self._maybe_force_close(symbol, current_price)
            return None

        # 2. Risk manager tick check (stoploss monitoring)
        if self._risk_mgr is not None:
            self._risk_mgr.on_tick_risk_check(symbol, current_price)
            if self._risk_mgr.is_halted:
                return None

        # 3. Cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._signals_skipped_cooldown += 1
            return None

        pos = self._tracker.get_position(symbol) if self._tracker else None
        is_flat = pos is None or pos.is_flat

        # 4. Exit signals (Markov + exit config; does not require cached MCMC)
        if pos and not pos.is_flat and self._markov.is_ready:
            exit_sig = self._maybe_exit_signal(symbol, pos, current_price)
            if exit_sig is not None:
                return exit_sig

        if not is_flat:
            return None

        # 5. Entry signals (flat only; need latest MCMC)
        if self._last_mcmc_result is None:
            return None

        result = self._last_mcmc_result
        if result.p_up > self._confidence:
            return self._make_signal(
                symbol, Signal.BUY, current_price,
                result.p_up, "Entry long: P(UP) high",
            )
        if result.p_down > self._confidence:
            return self._make_signal(
                symbol, Signal.SELL, current_price,
                result.p_down, "Entry short: P(DOWN) high",
            )

        return None

    def handle_ohlc(self, ohlc) -> None:
        """Called when a new OHLC candle closes. Triggers MCMC recompute."""
        o_sym = (getattr(ohlc, "symbol", None) or "").strip().upper()
        if not o_sym or o_sym not in self.symbols:
            return

        new_time = getattr(ohlc, "time", None)

        # Only process genuinely new (closed) candles, not live updates
        if new_time is not None and new_time == self._last_ohlc_time:
            return

        self._last_ohlc_time = new_time
        self._last_ohlc_candles.append(ohlc)
        self._markov.update(float(ohlc.open), float(ohlc.close))

        if self._tracker is not None:
            p = self._tracker.get_position(o_sym)
            if p is not None and not p.is_flat:
                self._bars_in_trade += 1

        logger.debug(f"[{self.name}] New OHLC candle", extra={
            "symbol": o_sym, "open": ohlc.open, "close": ohlc.close,
            "time": new_time, "sessions": self._markov.n_sessions,
        })

        # Compute MCMC in the background (non-blocking for quote handler)
        asyncio.ensure_future(self._run_mcmc_async(ohlc.close))

    # ------------------------------------------------------------------ #
    # on_signal: place the actual order
    # ------------------------------------------------------------------ #

    def on_signal(self, event: SignalEvent) -> None:
        """Place order when a signal is confirmed by risk checks."""
        if self._risk_mgr is not None:
            from src.order_manager import OrderRequest
            req = OrderRequest(
                symbol=event.symbol,
                side=OrderSide.BUY if event.signal == Signal.BUY else OrderSide.SELL,
                order_type=OrderType.LO,
                price=event.price,
                quantity=self._position_size,
                market_type=MarketType.DERIVATIVE.value,
            )
            allowed, reason = self._risk_mgr.pre_trade_check(
                req,
                paper_mode=self._settings.PAPER_MODE,
            )
            if not allowed:
                self._signals_skipped_risk += 1
                logger.info(f"[{self.name}] Signal blocked by risk: {reason}")
                return

        self.place_order(
            symbol=event.symbol,
            side=OrderSide.BUY if event.signal == Signal.BUY else OrderSide.SELL,
            order_type=OrderType.LO,
            price=event.price,
            quantity=self._position_size,
            market_type=MarketType.DERIVATIVE.value,
        )
        self._open_side = OrderSide.BUY if event.signal == Signal.BUY else OrderSide.SELL

    def on_fill(self, order: ManagedOrder) -> None:
        """Track round-trips and apply cooldown after close."""
        if order.symbol not in self.symbols:
            return

        pos = self._tracker.get_position(order.symbol) if self._tracker else None
        if pos is None:
            return

        was_flat = self._position_was_flat
        self._position_was_flat = pos.is_flat

        if pos.is_flat:
            self._round_trips += 1
            self._cooldown_remaining = self._cooldown_ticks
            self._open_side = None
            self._bars_in_trade = 0
            logger.info(f"[{self.name}] Round-trip #{self._round_trips} complete, cooldown {self._cooldown_ticks} ticks")
        elif was_flat and not pos.is_flat:
            self._bars_in_trade = 0

    # ------------------------------------------------------------------ #
    # Internal MCMC pipeline
    # ------------------------------------------------------------------ #

    async def _run_mcmc_async(self, current_price: float) -> None:
        """Run MCMC computation asynchronously (won't block tick handler)."""
        if not self._markov.is_ready:
            return

        symbol = self.symbols[0]
        candles = self._last_ohlc_candles[-(self._settings.MCMC_ROLLING_WINDOW):]
        log_returns = self._markov.get_log_returns(candles)

        if len(log_returns) < 3:
            return

        # Markov prediction
        try:
            p_up, p_down = self._markov.predict()
        except RuntimeError:
            return

        # Time fraction remaining in session
        dt = self._session_dt_remaining()

        # Run MCMC (CPU-bound, runs in same thread -- fast enough at < 100ms)
        result = self._mcmc.run(
            log_returns=log_returns,
            current_price=current_price,
            p_up_markov=p_up,
            dt=dt,
        )

        self._last_mcmc_result = result
        self._mcmc_calls += 1
        self._mcmc_total_ms += result.elapsed_ms

        logger.info(
            f"[{self.name}] MCMC computed",
            extra={
                "symbol": symbol,
                "p_up": round(result.p_up, 4),
                "p_down": round(result.p_down, 4),
                "elapsed_ms": round(result.elapsed_ms, 1),
                "mu": round(result.mu_posterior_mean, 5),
                "sigma": round(result.sigma_posterior_mean, 5),
                "n_paths": result.n_paths,
            },
        )

    async def _preload_history(self) -> None:
        """Load existing OHLC history from DataBuffer into MarkovModel."""
        symbol = self.symbols[0]
        candles = await self._buffer.get_ohlc_history(symbol)
        loaded = self._markov.feed_ohlc_history(candles)
        self._last_ohlc_candles = list(candles)
        logger.info(f"[{self.name}] Pre-loaded {loaded} OHLC sessions from buffer")
        if loaded >= self._markov.MIN_SESSIONS:
            latest = candles[-1] if candles else None
            if latest:
                asyncio.ensure_future(self._run_mcmc_async(float(latest.close)))

    # ------------------------------------------------------------------ #
    # Config-driven exit (see src.strategies.exit_rules)
    # ------------------------------------------------------------------ #

    def _maybe_exit_signal(
        self,
        symbol: str,
        pos,
        current_price: float,
    ) -> Optional[SignalEvent]:
        """Evaluate should_exit; log each evaluation; return close signal if needed."""
        try:
            p_up, p_down = self._markov.predict()
        except RuntimeError:
            return None

        stats = self._markov.get_stats()
        A = stats.matrix
        window = self._settings.MCMC_ROLLING_WINDOW
        candles = self._last_ohlc_candles[-window:] if self._last_ohlc_candles else []
        mu = self._markov.empirical_state_mean_log_returns(candles)

        pnl_points = float(pos.unrealized_pnl(current_price))
        last = self._markov.last_state
        last_idx = int(last) if last is not None else 0

        posterior = np.array([p_down, p_up], dtype=np.float64)

        state = {
            "pnl_points": pnl_points,
            "posterior": posterior,
            "transition_matrix": A,
            "state_returns": mu,
            "bars_in_trade": self._bars_in_trade,
            "last_state_index": last_idx,
        }
        position = {"direction": "LONG" if pos.is_long else "SHORT"}

        decision = should_exit(position, state, self._exit_config)

        er = decision.get("expected_return")
        er_log = None
        if er is not None and isinstance(er, (float, int, np.floating)):
            if not (isinstance(er, float) and np.isnan(er)):
                er_log = float(er)

        exit_reason = decision["reason"] if decision.get("exit") else "hold"

        logger.info(
            f"[{self.name}] exit_decision",
            extra={
                "direction": position["direction"],
                "pnl": round(pnl_points, 6),
                "expected_return": er_log,
                "state_probabilities": decision.get("state_probabilities", []),
                "exit_reason": exit_reason,
            },
        )

        if not decision["exit"]:
            return None

        if pos.is_long:
            return self._make_signal(
                symbol,
                Signal.SELL,
                current_price,
                float(p_down),
                f"Exit: {decision['reason']}",
            )
        if pos.is_short:
            return self._make_signal(
                symbol,
                Signal.BUY,
                current_price,
                float(p_up),
                f"Exit: {decision['reason']}",
            )
        return None

    # ------------------------------------------------------------------ #
    # T0 force-close logic
    # ------------------------------------------------------------------ #

    def _maybe_force_close(self, symbol: str, current_price: float) -> None:
        """Send an immediate close order if a position is open."""
        if self._t0_closed:
            return
        if self._tracker is None:
            return

        pos = self._tracker.get_position(symbol)
        if pos.is_flat:
            return

        close_side = OrderSide.SELL if pos.is_long else OrderSide.BUY
        logger.warning(
            f"[{self.name}] T0 FORCE CLOSE",
            extra={
                "symbol": symbol, "qty": pos.quantity,
                "side": close_side.value, "price": current_price,
            },
        )
        self.place_order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.LO,
            price=current_price,
            quantity=abs(pos.quantity),
            market_type=MarketType.DERIVATIVE.value,
        )
        self._t0_closed = True

    def _is_past_force_close_time(self) -> bool:
        """Return True if current time is at or past force-close minute."""
        now = datetime.now()
        current_minute = now.hour * 60 + now.minute
        return current_minute >= self._force_close_minute

    # ------------------------------------------------------------------ #
    # Stoploss callback (wired via RiskManager.on_stoploss)
    # ------------------------------------------------------------------ #

    def _on_stoploss_triggered(self, symbol: str, current_price: float) -> None:
        """Force-close position when stoploss is hit."""
        if symbol not in self.symbols:
            return
        logger.warning(f"[{self.name}] Stoploss callback: closing {symbol} at {current_price}")
        self._maybe_force_close(symbol, current_price)
        self._t0_closed = True  # prevent double-close

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _make_signal(
        self,
        symbol: str,
        signal: Signal,
        price: float,
        confidence: float,
        reason: str,
    ) -> SignalEvent:
        return SignalEvent(
            signal=signal,
            symbol=symbol,
            price=price,
            quantity=self._position_size,
            confidence=round(confidence, 4),
            reason=reason,
        )

    def _extract_price(self, quote, trade) -> float:
        if quote is not None:
            mid = ((getattr(quote, "best_bid", 0) or 0)
                   + (getattr(quote, "best_ask", 0) or 0))
            if mid > 0:
                return mid / 2.0
            return float(getattr(quote, "last_price", 0) or 0)
        if trade is not None:
            return float(getattr(trade, "price", 0) or 0)
        return 0.0

    def _session_dt_remaining(self) -> float:
        """Fraction of session remaining (0.0 - 1.0)."""
        now = datetime.now()
        session_start_min = _SESSION_START_HOUR * 60 + _SESSION_START_MINUTE
        current_min = now.hour * 60 + now.minute
        elapsed = max(0, current_min - session_start_min)
        remaining = max(0.0, _SESSION_TOTAL_MINUTES - elapsed)
        return remaining / _SESSION_TOTAL_MINUTES

    @property
    def extended_stats(self) -> dict:
        base = self.stats
        base.update({
            "mcmc_calls": self._mcmc_calls,
            "avg_mcmc_ms": round(self._mcmc_total_ms / max(1, self._mcmc_calls), 1),
            "round_trips": self._round_trips,
            "markov_sessions": self._markov.n_sessions,
            "last_p_up": round(self._last_mcmc_result.p_up, 4) if self._last_mcmc_result else None,
            "last_p_down": round(self._last_mcmc_result.p_down, 4) if self._last_mcmc_result else None,
            "signals_skipped_cooldown": self._signals_skipped_cooldown,
            "signals_skipped_risk": self._signals_skipped_risk,
            "t0_closed": self._t0_closed,
        })
        return base
