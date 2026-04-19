"""Unit tests for PaperEngine (auto-fill paper orders vs live quotes)."""

import pytest
from unittest.mock import MagicMock, call

from src.config import OrderSide, OrderStatus, OrderType, Settings
from src.order_manager import ManagedOrder, OrderManager, OrderRequest
from src.paper_engine import FillRecord, PaperEngine


# ── helpers ───────────────────────────────────────────────────────────────────

def _settings() -> Settings:
    return Settings(
        DNSE_API_KEY="k",
        DNSE_API_SECRET="s",
        DNSE_ACCOUNT_NO="0001",
    )


def _order_mgr() -> OrderManager:
    return OrderManager(client=None, settings=_settings(), paper_mode=True)


def _quote(bid: float, ask: float, symbol: str = "VN30F2506"):
    q = MagicMock()
    q.symbol = symbol
    q.best_bid = bid
    q.best_ask = ask
    return q


def _place_order(mgr: OrderManager, side: OrderSide, price: float, qty: int = 1) -> ManagedOrder:
    req = OrderRequest(
        symbol="VN30F2506",
        side=side,
        order_type=OrderType.LO,
        price=price,
        quantity=qty,
        market_type="DERIVATIVE",
    )
    return mgr.submit(req)


# ── session lifecycle ─────────────────────────────────────────────────────────

class TestPaperEngineLifecycle:
    def test_not_running_by_default(self):
        e = PaperEngine(_order_mgr())
        assert not e.is_running

    def test_start_makes_running(self):
        e = PaperEngine(_order_mgr())
        e.start()
        assert e.is_running

    def test_stop_makes_not_running(self):
        e = PaperEngine(_order_mgr())
        e.start()
        e.stop()
        assert not e.is_running

    def test_reset_clears_fills(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        order = _place_order(mgr, OrderSide.BUY, 1250.0)
        e.on_quote(_quote(bid=1248.0, ask=1252.0))  # ask(1252) > price(1250) -> NOT filled
        assert len(e.fill_history) == 0
        e.reset()
        assert len(e.fill_history) == 0


# ── LO BUY fill logic ─────────────────────────────────────────────────────────

class TestLOBuyFill:
    def test_fills_when_ask_le_order_price(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        order = _place_order(mgr, OrderSide.BUY, price=1250.0)
        e.on_quote(_quote(bid=1248.0, ask=1249.0))  # ask(1249) <= price(1250) -> FILL
        assert order.status == OrderStatus.FILLED

    def test_no_fill_when_ask_above_order_price(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        order = _place_order(mgr, OrderSide.BUY, price=1248.0)
        e.on_quote(_quote(bid=1247.0, ask=1251.0))  # ask(1251) > price(1248) -> NO fill
        assert order.status == OrderStatus.NEW

    def test_fill_at_ask_price(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        order = _place_order(mgr, OrderSide.BUY, price=1252.0)
        e.on_quote(_quote(bid=1249.0, ask=1250.0))
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(1250.0)


# ── LO SELL fill logic ────────────────────────────────────────────────────────

class TestLOSellFill:
    def test_fills_when_bid_ge_order_price(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        order = _place_order(mgr, OrderSide.SELL, price=1248.0)
        e.on_quote(_quote(bid=1249.0, ask=1251.0))  # bid(1249) >= price(1248) -> FILL
        assert order.status == OrderStatus.FILLED

    def test_no_fill_when_bid_below_order_price(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        order = _place_order(mgr, OrderSide.SELL, price=1255.0)
        e.on_quote(_quote(bid=1252.0, ask=1253.0))  # bid(1252) < price(1255) -> NO fill
        assert order.status == OrderStatus.NEW

    def test_fill_at_bid_price(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        order = _place_order(mgr, OrderSide.SELL, price=1248.0)
        e.on_quote(_quote(bid=1250.0, ask=1252.0))
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(1250.0)


# ── slippage ──────────────────────────────────────────────────────────────────

class TestSlippage:
    def test_buy_slippage_increases_fill_price(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr, tick_size=0.5, slippage_ticks=2)
        e.start()
        order = _place_order(mgr, OrderSide.BUY, price=1255.0)
        e.on_quote(_quote(bid=1249.0, ask=1250.0))
        assert order.status == OrderStatus.FILLED
        # fill = ask(1250) + 2*0.5 = 1251
        assert order.avg_fill_price == pytest.approx(1251.0)

    def test_sell_slippage_decreases_fill_price(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr, tick_size=0.5, slippage_ticks=2)
        e.start()
        order = _place_order(mgr, OrderSide.SELL, price=1248.0)
        e.on_quote(_quote(bid=1250.0, ask=1252.0))
        assert order.status == OrderStatus.FILLED
        # fill = bid(1250) - 2*0.5 = 1249
        assert order.avg_fill_price == pytest.approx(1249.0)


# ── no fill when engine not running ──────────────────────────────────────────

class TestNotRunning:
    def test_no_fill_before_start(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        order = _place_order(mgr, OrderSide.BUY, price=1255.0)
        e.on_quote(_quote(bid=1249.0, ask=1250.0))
        assert order.status == OrderStatus.NEW  # not filled

    def test_no_fill_after_stop(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        e.stop()
        order = _place_order(mgr, OrderSide.BUY, price=1255.0)
        e.on_quote(_quote(bid=1249.0, ask=1250.0))
        assert order.status == OrderStatus.NEW


# ── session report ────────────────────────────────────────────────────────────

class TestSessionReport:
    def test_fill_count_in_report(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        _place_order(mgr, OrderSide.BUY, price=1255.0)
        _place_order(mgr, OrderSide.BUY, price=1255.0)
        e.on_quote(_quote(bid=1249.0, ask=1250.0))
        report = e.get_session_report()
        assert report.fills == 2
        assert report.total_volume == 2
        assert "VN30F2506" in report.symbols_traded

    def test_empty_session_report(self):
        e = PaperEngine(_order_mgr())
        e.start()
        report = e.get_session_report()
        assert report.fills == 0
        assert report.total_volume == 0


# ── wrong symbol not filled ────────────────────────────────────────────────────

class TestSymbolIsolation:
    def test_quote_for_different_symbol_does_not_fill(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        order = _place_order(mgr, OrderSide.BUY, price=1255.0)
        e.on_quote(_quote(bid=1249.0, ask=1250.0, symbol="HPG"))
        assert order.status == OrderStatus.NEW

    def test_atc_fill_on_session_close(self):
        mgr = _order_mgr()
        e = PaperEngine(mgr)
        e.start()
        req = OrderRequest(
            symbol="VN30F2506",
            side=OrderSide.SELL,
            order_type=OrderType.ATC,
            price=0.0,
            quantity=1,
            market_type="DERIVATIVE",
        )
        order = mgr.submit(req)
        e.on_session_close(close_price=1255.0, symbol="VN30F2506")
        assert order.status == OrderStatus.FILLED
        assert order.avg_fill_price == pytest.approx(1255.0)
