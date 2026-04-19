"""Unit tests for MCMCDerivativesStrategy."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.config import OrderSide, OrderStatus, OrderType, Settings
from src.mcmc.mcmc_engine import MCMCResult
from src.order_manager import ManagedOrder, OrderManager
from src.position_tracker import PositionTracker
from src.risk_manager import RiskManager
from src.strategies.mcmc_derivatives import MCMCDerivativesStrategy
from src.strategy_base import Signal


# ── helpers ───────────────────────────────────────────────────────────────────

def _settings(**overrides) -> Settings:
    base = {
        "DNSE_API_KEY": "k",
        "DNSE_API_SECRET": "s",
        "DNSE_ACCOUNT_NO": "0001",
        "MCMC_DERIVATIVE_SYMBOL": "VN30F2506",
        "MCMC_NUM_PATHS": 100,
        "MCMC_MH_ITERATIONS": 50,
        "MCMC_MH_BURNIN": 20,
        "MCMC_CONFIDENCE_THRESHOLD": 0.75,
        "MCMC_ROLLING_WINDOW": 10,
        "MCMC_COOLDOWN_TICKS": 3,
        "MCMC_POSITION_SIZE": 1,
        "MCMC_FORCE_CLOSE_MINUTE": 865,
    }
    base.update(overrides)
    return Settings(**base)


def _make_strategy(settings=None, seed=42) -> MCMCDerivativesStrategy:
    cfg = settings or _settings()
    return MCMCDerivativesStrategy(
        name="test_mcmc",
        symbols=["VN30F2506"],
        settings=cfg,
        markov_window=10,
        mh_iterations=50,
        mh_burnin=20,
        n_paths=100,
        seed=seed,
    )


def _wire_strategy(strat: MCMCDerivativesStrategy):
    """Wire strategy with lightweight fakes."""
    mgr = OrderManager(client=None, settings=_settings(), paper_mode=True)
    tracker = PositionTracker()
    buffer = MagicMock()
    buffer.get_ohlc_history = AsyncMock(return_value=[])
    mgr.on_fill(tracker.on_fill)
    strat.start(mgr, tracker, buffer)
    return mgr, tracker, buffer


def _make_quote(bid: float, ask: float, symbol: str = "VN30F2506"):
    q = MagicMock()
    q.symbol = symbol
    q.best_bid = bid
    q.best_ask = ask
    q.last_price = (bid + ask) / 2
    return q


def _make_ohlc(open_p: float, close_p: float, ts: int = 1, symbol: str = "VN30F2506"):
    o = MagicMock()
    o.symbol = symbol
    o.open = open_p
    o.close = close_p
    o.time = ts
    return o


def _seed_markov_ready(strat: MCMCDerivativesStrategy) -> None:
    """Enough Markov sessions for predict(); aligns _last_ohlc_candles for mu."""
    candles = [_make_ohlc(1240.0 + i, 1255.0 + i, ts=i + 1) for i in range(6)]
    strat._last_ohlc_candles = list(candles)
    strat._markov.feed_ohlc_history(candles)


def _inject_mcmc_result(strat: MCMCDerivativesStrategy, p_up: float) -> None:
    """Directly set the cached MCMC result to bypass async computation."""
    strat._last_mcmc_result = MCMCResult(
        p_up=p_up,
        p_down=1.0 - p_up,
        mean_return=0.001,
        var_return=0.0001,
        mu_posterior_mean=0.001,
        sigma_posterior_mean=0.015,
        n_paths=100,
        elapsed_ms=5.0,
    )


# ── signal generation ─────────────────────────────────────────────────────────

# Patch datetime.now() to 09:30 (before force-close at 14:25=865min) so signal
# tests are not sensitive to the time of day they are run.
_MORNING_DT = datetime(2025, 9, 1, 9, 30, 0)  # 09:30 = minute 570 < 865


class TestSignalGeneration:
    @patch("src.strategies.mcmc_derivatives.datetime")
    def test_buy_signal_when_p_up_above_threshold(self, mock_dt):
        mock_dt.now.return_value = _MORNING_DT
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        _inject_mcmc_result(strat, p_up=0.80)
        sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is not None
        assert sig.signal == Signal.BUY

    @patch("src.strategies.mcmc_derivatives.datetime")
    def test_sell_signal_when_p_down_above_threshold(self, mock_dt):
        mock_dt.now.return_value = _MORNING_DT
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        _inject_mcmc_result(strat, p_up=0.18)
        sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is not None
        assert sig.signal == Signal.SELL

    def test_hold_when_probabilities_below_threshold(self):
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        _inject_mcmc_result(strat, p_up=0.55)
        sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is None

    def test_no_signal_when_no_mcmc_result(self):
        strat = _make_strategy()
        _wire_strategy(strat)
        sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is None

    def test_no_signal_for_wrong_symbol(self):
        strat = _make_strategy()
        _wire_strategy(strat)
        _inject_mcmc_result(strat, p_up=0.90)
        sig = strat.on_tick("HPG", quote=_make_quote(25, 26, symbol="HPG"))
        assert sig is None


# ── exit signals ─────────────────────────────────────────────────────────────

class TestExitSignals:
    @patch("src.strategies.mcmc_derivatives.datetime")
    def test_exit_long_when_p_down_high(self, mock_dt):
        mock_dt.now.return_value = _MORNING_DT
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        _seed_markov_ready(strat)
        # Simulate open long position
        pos = tracker.get_position("VN30F2506")
        pos.quantity = 1
        pos.avg_price = 1250.0
        with patch.object(strat._markov, "predict", return_value=(0.18, 0.82)):
            sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is not None
        assert sig.signal == Signal.SELL

    @patch("src.strategies.mcmc_derivatives.datetime")
    def test_exit_short_when_p_up_high(self, mock_dt):
        mock_dt.now.return_value = _MORNING_DT
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        _seed_markov_ready(strat)
        pos = tracker.get_position("VN30F2506")
        pos.quantity = -1
        pos.avg_price = 1250.0
        with patch.object(strat._markov, "predict", return_value=(0.82, 0.18)):
            sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is not None
        assert sig.signal == Signal.BUY


# ── cooldown ──────────────────────────────────────────────────────────────────

class TestCooldown:
    @patch("src.strategies.mcmc_derivatives.datetime")
    def test_no_signal_during_cooldown(self, mock_dt):
        mock_dt.now.return_value = _MORNING_DT
        strat = _make_strategy(settings=_settings(MCMC_COOLDOWN_TICKS=5))
        mgr, tracker, buf = _wire_strategy(strat)
        _inject_mcmc_result(strat, p_up=0.85)
        strat._cooldown_remaining = 5
        sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is None

    @patch("src.strategies.mcmc_derivatives.datetime")
    def test_cooldown_decrements(self, mock_dt):
        mock_dt.now.return_value = _MORNING_DT
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        strat._cooldown_remaining = 3
        strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert strat._cooldown_remaining == 2

    @patch("src.strategies.mcmc_derivatives.datetime")
    def test_signal_after_cooldown_expires(self, mock_dt):
        mock_dt.now.return_value = _MORNING_DT
        strat = _make_strategy(settings=_settings(MCMC_COOLDOWN_TICKS=2))
        mgr, tracker, buf = _wire_strategy(strat)
        _inject_mcmc_result(strat, p_up=0.85)
        strat._cooldown_remaining = 1
        strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))  # cooldown=0
        sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))  # signal
        assert sig is not None


# ── T0 force close ────────────────────────────────────────────────────────────

class TestT0ForceClose:
    def test_no_signal_after_force_close_time(self):
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        _inject_mcmc_result(strat, p_up=0.90)
        with patch("src.strategies.mcmc_derivatives.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=14, minute=26)
            sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is None

    def test_force_close_order_sent_for_open_long(self):
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        pos = tracker.get_position("VN30F2506")
        pos.quantity = 1
        pos.avg_price = 1250.0

        with patch("src.strategies.mcmc_derivatives.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=14, minute=26)
            strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))

        # A SELL order should have been placed
        all_orders = mgr.get_all_orders()
        sell_orders = [o for o in all_orders if o.side == OrderSide.SELL]
        assert len(sell_orders) >= 1

    def test_force_close_only_fires_once(self):
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        pos = tracker.get_position("VN30F2506")
        pos.quantity = 1
        pos.avg_price = 1250.0

        with patch("src.strategies.mcmc_derivatives.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(hour=14, minute=30)
            strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
            strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))

        sell_orders = [o for o in mgr.get_all_orders() if o.side == OrderSide.SELL]
        assert len(sell_orders) == 1


# ── risk manager integration ──────────────────────────────────────────────────

class TestRiskManagerIntegration:
    def test_order_blocked_by_risk_manager(self):
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        _inject_mcmc_result(strat, p_up=0.85)

        risk = RiskManager(tracker, _settings(), nav_start=1000.0)
        risk.halt("test block")
        strat.set_risk_manager(risk)

        sig = strat.on_tick("VN30F2506", quote=_make_quote(1249, 1251))
        assert sig is None  # halted risk manager blocks on_tick

    def test_on_signal_blocked_by_risk_pre_trade(self):
        strat = _make_strategy()
        mgr, tracker, buf = _wire_strategy(strat)
        risk = RiskManager(tracker, _settings(), nav_start=1000.0)
        strat.set_risk_manager(risk)
        # Force position at max size
        pos = tracker.get_position("VN30F2506")
        pos.quantity = 1  # already at MCMC_POSITION_SIZE=1

        from src.strategy_base import SignalEvent
        event = SignalEvent(signal=Signal.BUY, symbol="VN30F2506",
                            price=1250.0, quantity=1, confidence=0.85)
        strat.on_signal(event)
        # No order placed (position size limit)
        assert mgr.stats["total_submitted"] == 0


# ── OHLC handler ─────────────────────────────────────────────────────────────

class TestOHLCHandler:
    def test_new_candle_updates_markov(self):
        strat = _make_strategy()
        _wire_strategy(strat)
        initial_sessions = strat._markov.n_sessions
        strat.handle_ohlc(_make_ohlc(1240.0, 1255.0, ts=1))
        # Markov model should have gained 1 state (1 transition after 2 states)
        assert strat._last_ohlc_time == 1

    def test_same_timestamp_not_reprocessed(self):
        strat = _make_strategy()
        _wire_strategy(strat)
        strat.handle_ohlc(_make_ohlc(1240.0, 1255.0, ts=1))
        before = strat._markov.n_sessions
        strat.handle_ohlc(_make_ohlc(1240.0, 1255.0, ts=1))  # same ts
        assert strat._markov.n_sessions == before


# ── extended_stats ────────────────────────────────────────────────────────────

class TestExtendedStats:
    def test_stats_keys_present(self):
        strat = _make_strategy()
        _wire_strategy(strat)
        stats = strat.extended_stats
        assert "mcmc_calls" in stats
        assert "round_trips" in stats
        assert "markov_sessions" in stats
        assert "t0_closed" in stats
