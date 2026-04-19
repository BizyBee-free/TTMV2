"""Unit tests for RiskManager."""

import pytest
from unittest.mock import MagicMock, patch

from src.config import OrderSide, OrderType, Settings
from src.order_manager import OrderRequest
from src.position_tracker import DailyPnL, Position, PositionTracker
from src.risk_manager import RiskManager, RiskRejectReason


# ── helpers ───────────────────────────────────────────────────────────────────

def _settings(**overrides) -> Settings:
    base = {
        "DNSE_API_KEY": "k",
        "DNSE_API_SECRET": "s",
        "DNSE_ACCOUNT_NO": "0001",
        "MAX_DAILY_LOSS_PCT": 2.0,
        "STOPLOSS_DEFAULT_PCT": 3.0,
        "MCMC_POSITION_SIZE": 1,
    }
    base.update(overrides)
    return Settings(**base)


def _tracker_with_daily_pnl(net: float) -> PositionTracker:
    t = PositionTracker()
    t._daily_pnl.realized = net
    return t


def _buy_req(symbol="VN30F2506") -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.LO,
        price=1250.0,
        quantity=1,
        market_type="DERIVATIVE",
    )


def _sell_req(symbol="VN30F2506") -> OrderRequest:
    return OrderRequest(
        symbol=symbol,
        side=OrderSide.SELL,
        order_type=OrderType.LO,
        price=1250.0,
        quantity=1,
        market_type="DERIVATIVE",
    )


# ── daily loss halt ───────────────────────────────────────────────────────────

class TestDailyLossHalt:
    def test_allow_when_within_limit(self):
        t = _tracker_with_daily_pnl(-10.0)
        r = RiskManager(t, _settings(), nav_start=1000.0)
        ok, _ = r.pre_trade_check(_buy_req())
        assert ok is True

    def test_reject_when_loss_exceeds_limit(self):
        t = _tracker_with_daily_pnl(-25.0)
        r = RiskManager(t, _settings(MAX_DAILY_LOSS_PCT=2.0), nav_start=1000.0)
        ok, reason = r.pre_trade_check(_buy_req())
        assert ok is False
        assert "DAILY_LOSS" in reason

    def test_halt_set_after_breach(self):
        t = _tracker_with_daily_pnl(-25.0)
        r = RiskManager(t, _settings(), nav_start=1000.0)
        r.pre_trade_check(_buy_req())
        assert r.is_halted is True

    def test_subsequent_orders_rejected_when_halted(self):
        t = _tracker_with_daily_pnl(-25.0)
        r = RiskManager(t, _settings(), nav_start=1000.0)
        r.pre_trade_check(_buy_req())
        ok, reason = r.pre_trade_check(_sell_req())
        assert ok is False
        assert "HALTED" in reason

    def test_no_nav_start_no_halt_by_default(self):
        """Without nav_start, the % loss check cannot trigger (no denominator)."""
        t = _tracker_with_daily_pnl(-9999.0)
        r = RiskManager(t, _settings(), nav_start=0.0)
        ok, _ = r.pre_trade_check(_buy_req())
        assert ok is True

    def test_halt_callback_invoked(self):
        t = _tracker_with_daily_pnl(-25.0)
        r = RiskManager(t, _settings(), nav_start=1000.0)
        cb = MagicMock()
        r.on_halt(cb)
        r.pre_trade_check(_buy_req())
        cb.assert_called_once()

    def test_resume_lifts_halt(self):
        t = _tracker_with_daily_pnl(-25.0)
        r = RiskManager(t, _settings(), nav_start=1000.0)
        r.pre_trade_check(_buy_req())
        assert r.is_halted
        r.resume()
        assert not r.is_halted


# ── position size limit ────────────────────────────────────────────────────────

class TestPositionSizeLimit:
    def test_allow_when_flat(self):
        t = PositionTracker()
        r = RiskManager(t, _settings(MCMC_POSITION_SIZE=1))
        ok, _ = r.pre_trade_check(_buy_req())
        assert ok is True

    def test_reject_when_long_at_max(self):
        t = PositionTracker()
        pos = t.get_position("VN30F2506")
        pos.quantity = 1  # already at max
        r = RiskManager(t, _settings(MCMC_POSITION_SIZE=1))
        ok, reason = r.pre_trade_check(_buy_req())
        assert ok is False
        assert "POSITION_SIZE" in reason

    def test_reject_when_short_at_max(self):
        t = PositionTracker()
        pos = t.get_position("VN30F2506")
        pos.quantity = -1
        r = RiskManager(t, _settings(MCMC_POSITION_SIZE=1))
        ok, reason = r.pre_trade_check(_sell_req())
        assert ok is False
        assert "POSITION_SIZE" in reason

    def test_allow_sell_when_long_closing(self):
        """Selling to close a long is allowed regardless of max size."""
        t = PositionTracker()
        pos = t.get_position("VN30F2506")
        pos.quantity = 1  # long position
        r = RiskManager(t, _settings(MCMC_POSITION_SIZE=1))
        # Sell to close -- position is 1, max is 1, sell check: pos <= -max? 1 > -1 -> OK
        ok, _ = r.pre_trade_check(_sell_req())
        assert ok is True


# ── stoploss ──────────────────────────────────────────────────────────────────

class TestStoploss:
    def test_no_trigger_when_flat(self):
        t = PositionTracker()
        r = RiskManager(t, _settings(STOPLOSS_DEFAULT_PCT=3.0))
        triggered = r.on_tick_risk_check("VN30F2506", 1200.0)
        assert triggered is False

    def test_stoploss_triggers_when_loss_exceeds_pct(self):
        t = PositionTracker()
        pos = t.get_position("VN30F2506")
        pos.quantity = 1
        pos.avg_price = 1250.0
        r = RiskManager(t, _settings(STOPLOSS_DEFAULT_PCT=3.0))
        cb = MagicMock()
        r.on_stoploss(cb)
        # Price dropped 4% -> triggers stoploss (3% threshold)
        triggered = r.on_tick_risk_check("VN30F2506", 1200.0)
        assert triggered is True
        cb.assert_called_once_with("VN30F2506", 1200.0)

    def test_stoploss_not_triggered_when_within_limit(self):
        t = PositionTracker()
        pos = t.get_position("VN30F2506")
        pos.quantity = 1
        pos.avg_price = 1250.0
        r = RiskManager(t, _settings(STOPLOSS_DEFAULT_PCT=5.0))
        # Price dropped only 2%
        triggered = r.on_tick_risk_check("VN30F2506", 1225.0)
        assert triggered is False

    def test_short_stoploss_on_price_rise(self):
        t = PositionTracker()
        pos = t.get_position("VN30F2506")
        pos.quantity = -1
        pos.avg_price = 1200.0
        r = RiskManager(t, _settings(STOPLOSS_DEFAULT_PCT=3.0))
        cb = MagicMock()
        r.on_stoploss(cb)
        # Price rose 5% on short -> loss > 3%
        triggered = r.on_tick_risk_check("VN30F2506", 1262.0)
        assert triggered is True
        cb.assert_called_once()


# ── manual halt ───────────────────────────────────────────────────────────────

class TestManualHalt:
    def test_manual_halt(self):
        t = PositionTracker()
        r = RiskManager(t, _settings())
        r.halt("test halt")
        assert r.is_halted
        assert r.halt_reason == "test halt"

    def test_order_rejected_when_manually_halted(self):
        t = PositionTracker()
        r = RiskManager(t, _settings())
        r.halt("maintenance")
        ok, reason = r.pre_trade_check(_buy_req())
        assert ok is False
        assert "HALTED" in reason


# ── stats / snapshot ──────────────────────────────────────────────────────────

class TestRiskManagerStats:
    def test_stats_counts_checks_and_rejections(self):
        t = PositionTracker()
        r = RiskManager(t, _settings(MCMC_POSITION_SIZE=1))
        pos = t.get_position("VN30F2506")
        pos.quantity = 1  # force rejection
        r.pre_trade_check(_buy_req())  # rejected
        r.pre_trade_check(_sell_req())  # allowed (closing)
        snap = r.stats
        assert snap.total_checks == 2
        assert snap.total_rejected == 1
