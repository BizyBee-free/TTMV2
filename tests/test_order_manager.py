"""Tests for Phase 3: OrderManager, PositionTracker, StrategyBase."""

import json
import os
import sys
import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import OrderSide, OrderStatus, OrderType, Settings
from src.order_manager import ManagedOrder, OrderManager, OrderRequest, OrderRejectReason
from src.position_tracker import PositionTracker, Position, DailyPnL
from src.strategy_base import StrategyBase, Signal, SignalEvent
from src.data_buffer import DataBuffer


# ===========================================================================
#  Fixtures
# ===========================================================================

@pytest.fixture
def mock_settings():
    with patch.dict(os.environ, {
        "DNSE_API_KEY": "test-key",
        "DNSE_API_SECRET": "test-secret",
        "DNSE_ACCOUNT_NO": "0001000115",
        "PAPER_MODE": "true",
    }):
        yield Settings()


def make_request(**overrides) -> OrderRequest:
    defaults = dict(
        symbol="HPG", side=OrderSide.BUY, order_type=OrderType.LO,
        price=25.0, quantity=100, loan_package_id=0, market_type="STOCK",
    )
    defaults.update(overrides)
    return OrderRequest(**defaults)


# ===========================================================================
#  OrderManager tests
# ===========================================================================

class TestOrderManagerPaper:
    def test_submit_paper_order(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request())

        assert order.status == OrderStatus.NEW
        assert order.is_paper
        assert order.order_id.startswith("PAPER-")
        assert order.symbol == "HPG"
        assert order.quantity == 100
        assert mgr.stats["total_submitted"] == 1

    def test_paper_fill_full(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request())
        mgr.paper_fill(order.order_id, fill_price=25.1)

        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 100
        assert order.avg_fill_price == 25.1
        assert order.is_terminal
        assert mgr.stats["total_filled"] == 1

    def test_paper_fill_partial(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request(quantity=200))

        mgr.paper_fill(order.order_id, fill_price=25.0, fill_qty=80)
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_qty == 80
        assert order.remaining_qty == 120

        mgr.paper_fill(order.order_id, fill_price=25.2, fill_qty=120)
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 200
        expected_avg = (25.0 * 80 + 25.2 * 120) / 200
        assert order.avg_fill_price == pytest.approx(expected_avg, rel=1e-6)

    def test_paper_cancel(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request())

        assert mgr.cancel(order.order_id)
        assert order.status == OrderStatus.CANCELED
        assert order.is_terminal
        assert len(mgr.get_active_orders()) == 0

    def test_paper_modify(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request(price=25.0))

        assert mgr.modify(order.order_id, price=26.0)
        assert order.price == 26.0

    def test_reject_invalid_lot_size(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request(quantity=150))

        assert order.status == OrderStatus.REJECTED
        assert "INVALID_QUANTITY" in order.reject_reason
        assert mgr.stats["total_rejected"] == 1

    def test_reject_zero_price_limit(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request(price=0))

        assert order.status == OrderStatus.REJECTED
        assert "INVALID_PRICE" in order.reject_reason

    def test_odd_lot_allowed(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request(quantity=50))

        assert order.status == OrderStatus.NEW

    def test_derivative_any_qty(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request(quantity=7, market_type="DERIVATIVE"))

        assert order.status == OrderStatus.NEW

    def test_fill_callback(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        fills = []
        mgr.on_fill(lambda o: fills.append(o))

        order = mgr.submit(make_request())
        mgr.paper_fill(order.order_id, fill_price=25.0)

        assert len(fills) == 1
        assert fills[0].order_id == order.order_id

    def test_update_callback(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        updates = []
        mgr.on_update(lambda o: updates.append(o.status))

        order = mgr.submit(make_request())
        mgr.cancel(order.order_id)

        assert OrderStatus.CANCELED in updates

    def test_get_active_orders(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        o1 = mgr.submit(make_request(symbol="HPG"))
        o2 = mgr.submit(make_request(symbol="VNM"))
        mgr.paper_fill(o1.order_id, fill_price=25.0)

        active = mgr.get_active_orders()
        assert len(active) == 1
        assert active[0].symbol == "VNM"

        active_hpg = mgr.get_active_orders("HPG")
        assert len(active_hpg) == 0

    def test_get_orders_by_status(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        mgr.submit(make_request())
        mgr.submit(make_request(quantity=150))  # rejected

        new_orders = mgr.get_orders_by_status(OrderStatus.NEW)
        rejected = mgr.get_orders_by_status(OrderStatus.REJECTED)
        assert len(new_orders) == 1
        assert len(rejected) == 1

    def test_cannot_fill_terminal_order(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request())
        mgr.cancel(order.order_id)

        assert not mgr.paper_fill(order.order_id, fill_price=25.0)

    def test_cannot_cancel_terminal_order(self, mock_settings):
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        order = mgr.submit(make_request())
        mgr.paper_fill(order.order_id, fill_price=25.0)

        assert not mgr.cancel(order.order_id)


class TestOrderManagerLive:
    @patch("src.dnse_client.DNSEClient")
    def test_submit_live_success(self, MockSDK, mock_settings):
        instance = MockSDK.return_value
        instance.post_order.return_value = (200, json.dumps({"orderId": "12345"}))

        from src.dnse_client import BeeTradeClient
        client = BeeTradeClient(mock_settings)
        client.token_manager.set_token("fake-token")

        mgr = OrderManager(client=client, settings=mock_settings, paper_mode=False)
        order = mgr.submit(make_request())

        assert order.order_id == "12345"
        assert order.status == OrderStatus.PENDING_NEW
        assert not order.is_paper

    @patch("src.dnse_client.DNSEClient")
    def test_submit_live_api_error(self, MockSDK, mock_settings):
        instance = MockSDK.return_value
        instance.post_order.return_value = (400, json.dumps({"code": "OA-200", "message": "Bad request"}))

        from src.dnse_client import BeeTradeClient
        client = BeeTradeClient(mock_settings)
        client.token_manager.set_token("fake-token")

        mgr = OrderManager(client=client, settings=mock_settings, paper_mode=False)
        order = mgr.submit(make_request())

        assert order.status == OrderStatus.REJECTED
        assert "OA-200" in order.reject_reason


# ===========================================================================
#  PositionTracker tests
# ===========================================================================

class TestPositionTracker:
    def _make_filled_order(self, symbol="HPG", side=OrderSide.BUY,
                           qty=100, price=25.0) -> ManagedOrder:
        return ManagedOrder(
            order_id="test-001", symbol=symbol, side=side,
            order_type=OrderType.LO, price=price, quantity=qty,
            market_type="STOCK", status=OrderStatus.FILLED,
            filled_qty=qty, avg_fill_price=price, is_paper=True,
        )

    def test_buy_opens_long(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(qty=100, price=25.0))

        pos = tracker.get_position("HPG")
        assert pos.quantity == 100
        assert pos.avg_price == 25.0
        assert pos.is_long

    def test_sell_closes_long(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(side=OrderSide.BUY, qty=100, price=25.0))
        tracker.on_fill(self._make_filled_order(side=OrderSide.SELL, qty=100, price=26.0))

        pos = tracker.get_position("HPG")
        assert pos.is_flat
        assert pos.realized_pnl == pytest.approx(100.0)  # (26-25)*100

    def test_partial_sell(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(side=OrderSide.BUY, qty=200, price=25.0))
        tracker.on_fill(self._make_filled_order(side=OrderSide.SELL, qty=100, price=26.0))

        pos = tracker.get_position("HPG")
        assert pos.quantity == 100
        assert pos.avg_price == 25.0
        assert pos.realized_pnl == pytest.approx(100.0)

    def test_add_to_long(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(side=OrderSide.BUY, qty=100, price=25.0))
        tracker.on_fill(self._make_filled_order(side=OrderSide.BUY, qty=100, price=27.0))

        pos = tracker.get_position("HPG")
        assert pos.quantity == 200
        assert pos.avg_price == pytest.approx(26.0)

    def test_unrealized_pnl(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(side=OrderSide.BUY, qty=100, price=25.0))

        pos = tracker.get_position("HPG")
        assert pos.unrealized_pnl(26.0) == pytest.approx(100.0)
        assert pos.unrealized_pnl(24.0) == pytest.approx(-100.0)
        assert pos.unrealized_pnl_pct(26.0) == pytest.approx(4.0)

    def test_daily_pnl(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(side=OrderSide.BUY, qty=100, price=25.0))
        tracker.on_fill(self._make_filled_order(side=OrderSide.SELL, qty=100, price=26.0))

        daily = tracker.daily_pnl
        assert daily.realized == pytest.approx(100.0)
        assert daily.trade_count == 1
        assert daily.win_count == 1
        assert daily.win_rate == pytest.approx(100.0)

    def test_losing_trade(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(side=OrderSide.BUY, qty=100, price=25.0))
        tracker.on_fill(self._make_filled_order(side=OrderSide.SELL, qty=100, price=24.0))

        daily = tracker.daily_pnl
        assert daily.realized == pytest.approx(-100.0)
        assert daily.loss_count == 1

    def test_total_unrealized_pnl(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(symbol="HPG", side=OrderSide.BUY, qty=100, price=25.0))
        tracker.on_fill(self._make_filled_order(symbol="VNM", side=OrderSide.BUY, qty=50, price=80.0))

        total = tracker.total_unrealized_pnl({"HPG": 26.0, "VNM": 82.0})
        assert total == pytest.approx(200.0)  # (26-25)*100 + (82-80)*50

    def test_get_open_positions(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(symbol="HPG", side=OrderSide.BUY, qty=100, price=25.0))
        tracker.on_fill(self._make_filled_order(symbol="VNM", side=OrderSide.BUY, qty=50, price=80.0))
        tracker.on_fill(self._make_filled_order(symbol="VNM", side=OrderSide.SELL, qty=50, price=82.0))

        open_pos = tracker.get_open_positions()
        assert "HPG" in open_pos
        assert "VNM" not in open_pos

    def test_reset_daily(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(side=OrderSide.BUY, qty=100, price=25.0))
        tracker.on_fill(self._make_filled_order(side=OrderSide.SELL, qty=100, price=26.0))

        tracker.reset_daily()
        assert tracker.daily_pnl.realized == 0.0
        assert tracker.daily_pnl.trade_count == 0
        assert len(tracker.trade_log) == 0

    def test_market_value(self):
        tracker = PositionTracker()
        tracker.on_fill(self._make_filled_order(qty=100, price=25.0))

        pos = tracker.get_position("HPG")
        assert pos.market_value(26.0) == 2600.0
        assert pos.cost_basis == 2500.0


# ===========================================================================
#  StrategyBase tests
# ===========================================================================

class DummyStrategy(StrategyBase):
    """Minimal strategy for testing the base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ticks_received = []
        self.signals_received = []
        self.fills_received = []
        self.started = False
        self.stopped = False

    def on_tick(self, symbol, quote=None, trade=None):
        self.ticks_received.append({"symbol": symbol, "quote": quote, "trade": trade})
        if quote and hasattr(quote, "bid") and quote.bid:
            return SignalEvent(
                signal=Signal.BUY, symbol=symbol,
                price=quote.bid[0].price, quantity=100,
                confidence=0.8, reason="test signal",
            )
        return None

    def on_signal(self, event):
        self.signals_received.append(event)

    def on_fill(self, order):
        self.fills_received.append(order)

    def on_start(self):
        self.started = True

    def on_stop(self):
        self.stopped = True


class TestStrategyBase:
    def test_start_stop(self, mock_settings):
        strat = DummyStrategy(name="test", symbols=["HPG"], settings=mock_settings)
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        tracker = PositionTracker()
        buf = DataBuffer()

        strat.start(mgr, tracker, buf)
        assert strat.is_running
        assert strat.started

        strat.stop()
        assert not strat.is_running
        assert strat.stopped

    def test_handle_quote_generates_signal(self, mock_settings):
        from vendor.dnse.trading_websocket.models import Quote, PriceLevel
        strat = DummyStrategy(name="test", symbols=["HPG"], settings=mock_settings)
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        tracker = PositionTracker()
        buf = DataBuffer()
        strat.start(mgr, tracker, buf)

        quote = Quote(
            marketId=1, boardId=1, symbol="HPG", isin="",
            bid=[PriceLevel(price=25.0, quantity=1000)],
            offer=[PriceLevel(price=25.1, quantity=500)],
            totalBidQtty=1000, totalOfferQtty=500,
        )
        strat.handle_quote(quote)

        assert len(strat.ticks_received) == 1
        assert len(strat.signals_received) == 1
        assert strat.signals_received[0].signal == Signal.BUY

    def test_ignores_unsubscribed_symbols(self, mock_settings):
        from vendor.dnse.trading_websocket.models import Quote, PriceLevel
        strat = DummyStrategy(name="test", symbols=["HPG"], settings=mock_settings)
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        strat.start(mgr, PositionTracker(), DataBuffer())

        quote = Quote(
            marketId=1, boardId=1, symbol="VNM", isin="",
            bid=[PriceLevel(price=80.0, quantity=100)],
            offer=[], totalBidQtty=100, totalOfferQtty=0,
        )
        strat.handle_quote(quote)
        assert len(strat.ticks_received) == 0

    def test_place_order_helper(self, mock_settings):
        strat = DummyStrategy(name="test", symbols=["HPG"], settings=mock_settings)
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        strat.start(mgr, PositionTracker(), DataBuffer())

        order = strat.place_order("HPG", OrderSide.BUY, OrderType.LO, 25.0, 100)
        assert order.status == OrderStatus.NEW
        assert strat._order_count == 1

    def test_get_position_helper(self, mock_settings):
        strat = DummyStrategy(name="test", symbols=["HPG"], settings=mock_settings)
        tracker = PositionTracker()
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        strat.start(mgr, tracker, DataBuffer())

        pos = strat.get_position("HPG")
        assert pos.is_flat

    def test_stats(self, mock_settings):
        strat = DummyStrategy(name="test", symbols=["HPG"], settings=mock_settings)
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        strat.start(mgr, PositionTracker(), DataBuffer())

        stats = strat.stats
        assert stats["name"] == "test"
        assert stats["running"]
        assert stats["symbols"] == ["HPG"]


# ===========================================================================
#  Integration: OrderManager + PositionTracker + Strategy
# ===========================================================================

class TestIntegration:
    def test_full_trade_lifecycle(self, mock_settings):
        """Submit order -> fill -> position updated -> P&L correct."""
        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        tracker = PositionTracker()
        mgr.on_fill(tracker.on_fill)

        # Buy
        buy = mgr.submit(make_request(side=OrderSide.BUY, quantity=100, price=25.0))
        mgr.paper_fill(buy.order_id, fill_price=25.0)

        pos = tracker.get_position("HPG")
        assert pos.quantity == 100
        assert pos.avg_price == 25.0

        # Sell
        sell = mgr.submit(make_request(side=OrderSide.SELL, quantity=100, price=26.0))
        mgr.paper_fill(sell.order_id, fill_price=26.0)

        pos = tracker.get_position("HPG")
        assert pos.is_flat
        assert pos.realized_pnl == pytest.approx(100.0)
        assert tracker.daily_pnl.win_count == 1

    def test_strategy_driven_trade(self, mock_settings):
        """Strategy generates signal -> places order -> gets fill callback."""
        from vendor.dnse.trading_websocket.models import Quote, PriceLevel

        mgr = OrderManager(settings=mock_settings, paper_mode=True)
        tracker = PositionTracker()
        buf = DataBuffer()

        class AutoBuyStrategy(StrategyBase):
            def on_tick(self, symbol, quote=None, trade=None):
                if quote and not self.get_active_orders(symbol):
                    return SignalEvent(Signal.BUY, symbol, price=25.0, quantity=100)
                return None

            def on_signal(self, event):
                self.place_order(event.symbol, OrderSide.BUY, OrderType.LO,
                                 event.price, event.quantity)

        strat = AutoBuyStrategy(name="auto_buy", symbols=["HPG"], settings=mock_settings)
        strat.start(mgr, tracker, buf)
        mgr.on_fill(tracker.on_fill)

        quote = Quote(
            marketId=1, boardId=1, symbol="HPG", isin="",
            bid=[PriceLevel(price=25.0, quantity=1000)],
            offer=[PriceLevel(price=25.1, quantity=500)],
            totalBidQtty=1000, totalOfferQtty=500,
        )
        strat.handle_quote(quote)

        active = mgr.get_active_orders("HPG")
        assert len(active) == 1

        mgr.paper_fill(active[0].order_id, fill_price=25.0)

        pos = tracker.get_position("HPG")
        assert pos.quantity == 100
