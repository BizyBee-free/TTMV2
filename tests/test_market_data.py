"""Tests for Phase 2: Market Data (DataBuffer + MarketDataManager)."""

import asyncio
from collections import deque
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_buffer import DataBuffer, OhlcBuffer, QuoteBuffer, TradeBuffer
from vendor.dnse.trading_websocket.models import Ohlc, Quote, Trade, PriceLevel


# ---------------------------------------------------------------------------
#  Fixtures: sample SDK model instances
# ---------------------------------------------------------------------------

def make_quote(symbol: str = "HPG", bid_price: float = 25.0, ask_price: float = 25.1) -> Quote:
    return Quote(
        marketId=1,
        boardId=1,
        symbol=symbol,
        isin="VN000000HPG0",
        bid=[PriceLevel(price=bid_price, quantity=1000)],
        offer=[PriceLevel(price=ask_price, quantity=500)],
        totalBidQtty=1000,
        totalOfferQtty=500,
    )


def make_trade(symbol: str = "HPG", price: float = 25.05, qty: int = 200) -> Trade:
    return Trade(
        marketId=1,
        boardId=1,
        isin="VN000000HPG0",
        symbol=symbol,
        price=price,
        quantity=qty,
        totalVolumeTraded=10000,
        grossTradeAmount=250000.0,
        highestPrice=26.0,
        lowestPrice=24.5,
        openPrice=25.0,
        tradingSessionId=2,
    )


def make_ohlc(symbol: str = "HPG", ts: int = 1000, close: float = 25.1) -> Ohlc:
    return Ohlc(
        symbol=symbol,
        resolution=1,
        open=Decimal("25.0"),
        high=Decimal("25.5"),
        low=Decimal("24.8"),
        close=Decimal(str(close)),
        volume=5000,
        time=ts,
        lastUpdated=ts + 1,
        type="1",
    )


# ===========================================================================
#  QuoteBuffer tests
# ===========================================================================

class TestQuoteBuffer:
    @pytest.mark.asyncio
    async def test_update_and_get(self):
        buf = QuoteBuffer()
        q = make_quote("HPG")
        await buf.update(q)

        result = await buf.get("HPG")
        assert result is not None
        assert result.symbol == "HPG"
        assert result.best_bid == (25.0, 1000)
        assert result.best_ask == (25.1, 500)

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        buf = QuoteBuffer()
        assert await buf.get("MISSING") is None

    @pytest.mark.asyncio
    async def test_overwrites_on_update(self):
        buf = QuoteBuffer()
        await buf.update(make_quote("HPG", bid_price=20.0))
        await buf.update(make_quote("HPG", bid_price=21.0))

        result = await buf.get("HPG")
        assert result.best_bid[0] == 21.0
        assert buf.count == 1

    @pytest.mark.asyncio
    async def test_multiple_symbols(self):
        buf = QuoteBuffer()
        await buf.update(make_quote("HPG"))
        await buf.update(make_quote("VNM"))

        assert buf.count == 2
        syms = await buf.symbols()
        assert set(syms) == {"HPG", "VNM"}

    @pytest.mark.asyncio
    async def test_get_all(self):
        buf = QuoteBuffer()
        await buf.update(make_quote("HPG"))
        await buf.update(make_quote("FPT"))

        all_q = await buf.get_all()
        assert len(all_q) == 2
        assert "HPG" in all_q
        assert "FPT" in all_q

    @pytest.mark.asyncio
    async def test_spread_property(self):
        buf = QuoteBuffer()
        await buf.update(make_quote("HPG", bid_price=25.0, ask_price=25.5))
        q = await buf.get("HPG")
        assert q.spread == pytest.approx(0.5)


# ===========================================================================
#  TradeBuffer tests
# ===========================================================================

class TestTradeBuffer:
    @pytest.mark.asyncio
    async def test_append_and_get_latest(self):
        buf = TradeBuffer(maxlen=5)
        await buf.append(make_trade("HPG", price=25.0))
        await buf.append(make_trade("HPG", price=25.1))

        latest = await buf.get_latest("HPG")
        assert latest.price == 25.1

    @pytest.mark.asyncio
    async def test_get_latest_missing(self):
        buf = TradeBuffer()
        assert await buf.get_latest("NOPE") is None

    @pytest.mark.asyncio
    async def test_ring_buffer_maxlen(self):
        buf = TradeBuffer(maxlen=3)
        for i in range(5):
            await buf.append(make_trade("HPG", price=float(i)))

        history = await buf.get_history("HPG")
        assert len(history) == 3
        assert history[0].price == 2.0  # oldest surviving
        assert history[-1].price == 4.0

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self):
        buf = TradeBuffer(maxlen=10)
        for i in range(7):
            await buf.append(make_trade("HPG", price=float(i)))

        last_3 = await buf.get_history("HPG", n=3)
        assert len(last_3) == 3
        assert last_3[-1].price == 6.0

    @pytest.mark.asyncio
    async def test_multiple_symbols(self):
        buf = TradeBuffer(maxlen=5)
        await buf.append(make_trade("HPG"))
        await buf.append(make_trade("VNM"))

        assert buf.trade_count("HPG") == 1
        assert buf.trade_count("VNM") == 1
        syms = await buf.symbols()
        assert set(syms) == {"HPG", "VNM"}


# ===========================================================================
#  OhlcBuffer tests
# ===========================================================================

class TestOhlcBuffer:
    @pytest.mark.asyncio
    async def test_append_and_get_latest(self):
        buf = OhlcBuffer(maxlen=10)
        await buf.append(make_ohlc("HPG", ts=1000, close=25.0))
        await buf.append(make_ohlc("HPG", ts=2000, close=25.5))

        latest = await buf.get_latest("HPG")
        assert latest.time == 2000

    @pytest.mark.asyncio
    async def test_replaces_same_timestamp(self):
        """Current bar updates should overwrite, not append."""
        buf = OhlcBuffer(maxlen=10)
        await buf.append(make_ohlc("HPG", ts=1000, close=25.0))
        await buf.append(make_ohlc("HPG", ts=1000, close=25.3))

        history = await buf.get_history("HPG")
        assert len(history) == 1
        assert history[0].close == Decimal("25.3")

    @pytest.mark.asyncio
    async def test_ring_buffer_maxlen(self):
        buf = OhlcBuffer(maxlen=3)
        for i in range(5):
            await buf.append(make_ohlc("HPG", ts=1000 * i, close=float(i)))

        history = await buf.get_history("HPG")
        assert len(history) == 3
        assert history[0].time == 2000

    @pytest.mark.asyncio
    async def test_get_latest_missing(self):
        buf = OhlcBuffer()
        assert await buf.get_latest("NOPE") is None


# ===========================================================================
#  DataBuffer (unified facade) tests
# ===========================================================================

class TestDataBuffer:
    @pytest.mark.asyncio
    async def test_on_quote_routes_to_buffer(self):
        db = DataBuffer()
        await db.on_quote(make_quote("HPG"))

        q = await db.get_latest_quote("HPG")
        assert q is not None
        assert q.symbol == "HPG"

    @pytest.mark.asyncio
    async def test_on_trade_routes_to_buffer(self):
        db = DataBuffer()
        await db.on_trade(make_trade("HPG"))

        t = await db.get_latest_trade("HPG")
        assert t is not None
        assert t.symbol == "HPG"

    @pytest.mark.asyncio
    async def test_on_ohlc_routes_to_buffer(self):
        db = DataBuffer()
        await db.on_ohlc(make_ohlc("HPG", ts=1000))

        history = await db.get_ohlc_history("HPG")
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_update_count(self):
        db = DataBuffer()
        assert db.update_count == 0

        await db.on_quote(make_quote("HPG"))
        await db.on_trade(make_trade("HPG"))
        await db.on_ohlc(make_ohlc("HPG"))

        assert db.update_count == 3

    @pytest.mark.asyncio
    async def test_stats(self):
        db = DataBuffer()
        await db.on_quote(make_quote("HPG"))
        await db.on_quote(make_quote("VNM"))
        await db.on_trade(make_trade("HPG"))
        await db.on_trade(make_trade("HPG"))
        await db.on_ohlc(make_ohlc("HPG", ts=1000))

        stats = db.stats()
        assert stats.total_quotes == 2
        assert stats.total_trades == 2
        assert stats.total_ohlc == 1
        assert stats.last_update_ts > 0

    @pytest.mark.asyncio
    async def test_get_all_quotes(self):
        db = DataBuffer()
        await db.on_quote(make_quote("HPG"))
        await db.on_quote(make_quote("FPT"))

        all_q = await db.get_all_quotes()
        assert len(all_q) == 2

    @pytest.mark.asyncio
    async def test_trade_history(self):
        db = DataBuffer(trade_maxlen=5)
        for i in range(7):
            await db.on_trade(make_trade("HPG", price=float(i)))

        history = await db.get_trade_history("HPG")
        assert len(history) == 5

        last_2 = await db.get_trade_history("HPG", n=2)
        assert len(last_2) == 2
        assert last_2[-1].price == 6.0


def test_normalize_dnse_ws_url_adds_stream_path_and_encoding():
    from src.market_data import _normalize_dnse_ws_url

    u = _normalize_dnse_ws_url("wss://ws-openapi.dnse.com.vn", "json")
    assert "v1/stream" in u
    assert "encoding=json" in u


def test_message_decoder_json_mode_falls_back_to_msgpack():
    import msgpack

    from vendor.dnse.trading_websocket.encoding import MessageDecoder

    dec = MessageDecoder("json")
    assert dec.decode(b'{"action":"pong"}')["action"] == "pong"
    packed = msgpack.packb({"T": "t", "symbol": "HPG", "price": 25.5}, use_bin_type=True)
    out = dec.decode(packed)
    assert out.get("T") == "t" or out.get("t") == "t"
    assert out.get("symbol") == "HPG"


# ===========================================================================
#  MarketDataManager tests (mocked WebSocket)
# ===========================================================================

class TestMarketDataManager:
    @pytest.fixture
    def mock_settings(self):
        with patch.dict(os.environ, {
            "DNSE_API_KEY": "test-key",
            "DNSE_API_SECRET": "test-secret",
            "DNSE_ACCOUNT_NO": "0001000115",
            "DNSE_WS_URL": "wss://test.example.com",
            "WS_ENCODING": "msgpack",
            "PAPER_MODE": "true",
        }):
            from src.config import Settings
            yield Settings()

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, mock_settings):
        with patch("src.market_data.TradingClient") as MockWS:
            instance = MockWS.return_value
            instance.connect = AsyncMock()
            instance.disconnect = AsyncMock()
            instance._session_id = "test-session"

            from src.market_data import MarketDataManager
            mgr = MarketDataManager(settings=mock_settings)

            await mgr.connect()
            assert mgr.is_connected
            instance.connect.assert_awaited_once()

            await mgr.disconnect()
            assert not mgr.is_connected
            instance.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subscribe_quotes(self, mock_settings):
        with patch("src.market_data.TradingClient") as MockWS:
            instance = MockWS.return_value
            instance.connect = AsyncMock()
            instance.subscribe_quotes = AsyncMock()
            instance._session_id = "test-session"

            from src.market_data import MarketDataManager
            mgr = MarketDataManager(settings=mock_settings)
            await mgr.connect()
            await mgr.subscribe_quotes(["HPG", "VNM"])

            instance.subscribe_quotes.assert_awaited_once()
            call_kwargs = instance.subscribe_quotes.call_args
            assert set(call_kwargs.kwargs["symbols"]) == {"HPG", "VNM"}

    @pytest.mark.asyncio
    async def test_subscribe_dedup(self, mock_settings):
        """Re-subscribing same symbols should not re-send."""
        with patch("src.market_data.TradingClient") as MockWS:
            instance = MockWS.return_value
            instance.connect = AsyncMock()
            instance.subscribe_quotes = AsyncMock()
            instance._session_id = "test-session"

            from src.market_data import MarketDataManager
            mgr = MarketDataManager(settings=mock_settings)
            await mgr.connect()

            await mgr.subscribe_quotes(["HPG"])
            await mgr.subscribe_quotes(["HPG"])

            assert instance.subscribe_quotes.await_count == 1

    @pytest.mark.asyncio
    async def test_callback_routes_to_buffer(self, mock_settings):
        with patch("src.market_data.TradingClient") as MockWS:
            instance = MockWS.return_value
            instance.connect = AsyncMock()
            instance._session_id = "test-session"

            from src.market_data import MarketDataManager
            mgr = MarketDataManager(settings=mock_settings)
            await mgr.connect()

            q = make_quote("HPG")
            mgr._on_quote(q)
            # Allow the ensure_future to complete
            await asyncio.sleep(0.05)

            result = await mgr.get_latest_quote("HPG")
            assert result is not None
            assert result.symbol == "HPG"

    @pytest.mark.asyncio
    async def test_external_callbacks(self, mock_settings):
        with patch("src.market_data.TradingClient") as MockWS:
            instance = MockWS.return_value
            instance.connect = AsyncMock()
            instance._session_id = "test-session"

            from src.market_data import MarketDataManager
            mgr = MarketDataManager(settings=mock_settings)
            await mgr.connect()

            received = []
            mgr.on_quote(lambda q: received.append(q))

            mgr._on_quote(make_quote("HPG"))
            mgr._on_quote(make_quote("VNM"))

            assert len(received) == 2
            assert received[0].symbol == "HPG"
            assert received[1].symbol == "VNM"

    @pytest.mark.asyncio
    async def test_msg_count(self, mock_settings):
        with patch("src.market_data.TradingClient") as MockWS:
            instance = MockWS.return_value
            instance.connect = AsyncMock()
            instance._session_id = "test-session"

            from src.market_data import MarketDataManager
            mgr = MarketDataManager(settings=mock_settings)
            await mgr.connect()

            mgr._on_quote(make_quote("HPG"))
            mgr._on_trade(make_trade("HPG"))
            mgr._on_ohlc(make_ohlc("HPG"))

            assert mgr.msg_count == 3

    @pytest.mark.asyncio
    async def test_get_subscriptions(self, mock_settings):
        with patch("src.market_data.TradingClient") as MockWS:
            instance = MockWS.return_value
            instance.connect = AsyncMock()
            instance.subscribe_quotes = AsyncMock()
            instance.subscribe_trades = AsyncMock()
            instance._session_id = "test-session"

            from src.market_data import MarketDataManager
            mgr = MarketDataManager(settings=mock_settings)
            await mgr.connect()

            await mgr.subscribe_quotes(["HPG", "VNM"])
            await mgr.subscribe_trades(["HPG"])

            subs = mgr.get_subscriptions()
            assert subs["quotes"] == ["HPG", "VNM"]
            assert subs["trades"] == ["HPG"]
            assert subs["ohlc"] == []
            assert subs["sec_def"] == []
