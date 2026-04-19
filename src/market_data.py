"""Real-time market data manager wrapping the DNSE WebSocket SDK.

Provides a high-level async interface to subscribe to quotes, trades,
and OHLC candles.  Incoming data is routed to:
  1. The internal DataBuffer (ring buffers for strategy consumption)
  2. Any externally registered callbacks (e.g. strategy signal functions)
"""

import asyncio
import sys
import os
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "dnse"))

from trading_websocket.client import TradingClient
from trading_websocket.models import Ohlc, Quote, Trade

from src.config import Settings, canonical_data_symbol, get_settings
from src.data_buffer import DataBuffer
from src.hmm.oi_ws_cache import record_open_interest_from_ws
from src.logger import get_logger
from src.ohlc_synth import (
    TradeOhlcSynth,
    resolution_param_to_minutes_int,
    resolution_param_to_seconds,
)

logger = get_logger("market_data")

Callback = Callable[..., Any]
AsyncCallback = Callable[..., Coroutine[Any, Any, None]]


def _normalize_dnse_ws_url(url: str, encoding: str) -> str:
    """
    DNSE OpenAPI WebSocket bắt buộc path ``/v1/stream`` (PyPI ``dnse`` dùng ``WS_BASE_URL`` tương tự).
    URL chỉ host (không path) trả HTTP 404 khi upgrade WebSocket.
    Thêm ``?encoding=`` nếu chưa có (server gửi welcome message theo encoding).
    """
    u = (url or "").strip().rstrip("/")
    if u in ("wss://ws-openapi.dnse.com.vn", "ws://ws-openapi.dnse.com.vn"):
        u = "wss://ws-openapi.dnse.com.vn/v1/stream"
    elif "ws-openapi.dnse.com.vn" in u and "/v1/stream" not in u:
        u = "wss://ws-openapi.dnse.com.vn/v1/stream"
    enc = (encoding or "msgpack").strip().lower()
    if enc not in ("json", "msgpack"):
        enc = "msgpack"
    if "?" not in u:
        u = f"{u}?encoding={enc}"
    return u


class MarketDataManager:
    """Manages WebSocket subscriptions and routes data to buffers + strategies.

    Usage::

        mgr = MarketDataManager(settings)
        await mgr.connect()
        await mgr.subscribe_quotes(["HPG", "VNM", "FPT"])
        await mgr.subscribe_trades(["HPG"])
        await mgr.subscribe_ohlc(["HPG"], resolution="1", board_id="G1")

        # Read from buffer any time
        quote = await mgr.buffer.get_latest_quote("HPG")

        # Graceful shutdown
        await mgr.disconnect()
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        buffer: Optional[DataBuffer] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.buffer = buffer or DataBuffer()

        ws_url = _normalize_dnse_ws_url(
            self._settings.DNSE_WS_URL,
            self._settings.WS_ENCODING,
        )
        self._ws_client = TradingClient(
            api_key=self._settings.DNSE_API_KEY,
            api_secret=self._settings.DNSE_API_SECRET,
            base_url=ws_url,
            encoding=self._settings.WS_ENCODING,
            subscribe_use_session_id=self._settings.DNSE_WS_SUBSCRIBE_USE_SESSION_ID,
            pipeline_debug=self._settings.DNSE_WS_PIPELINE_DEBUG,
        )

        self._connected = False
        self._subscribed_quotes: Set[Tuple[str, str]] = set()
        self._subscribed_trades: Set[Tuple[str, str]] = set()
        self._subscribed_ohlc: Set[str] = set()

        # External strategy callbacks
        self._quote_callbacks: List[Callback] = []
        self._trade_callbacks: List[Callback] = []
        self._ohlc_callbacks: List[Callback] = []
        self._subscribed_sec_def: Set[Tuple[str, str]] = set()

        # Metrics
        self._connect_ts: float = 0.0
        self._msg_count: int = 0
        self._last_msg_ts: float = 0.0
        self._latencies: list[float] = []
        self._ohlc_rx_session: int = 0
        self._trade_ohlc_synth: Optional[TradeOhlcSynth] = None
        self._synth_lock = asyncio.Lock()
        self._synth_suppressed: bool = False
        self._synth_ohlc_session: int = 0
        self._pipeline_trade_count: int = 0
        self._sec_def_rx_count: int = 0

    @staticmethod
    def _normalize_incoming_symbol(obj: Any) -> None:
        """DNSE WS có thể gửi trade_symbol (41…); thống nhất data_symbol cho buffer/strategy."""
        raw = getattr(obj, "symbol", None)
        if raw is None or str(raw).strip() == "":
            return
        canon = canonical_data_symbol(str(raw))
        if canon:
            obj.symbol = canon

    # ---- Connection lifecycle ----

    async def connect(self) -> None:
        """Connect and authenticate with the DNSE WebSocket gateway."""
        logger.info(
            "Connecting to DNSE WebSocket...",
            extra={"url": self._ws_client._base_url, "encoding": self._settings.WS_ENCODING},
        )
        t0 = time.time()
        await self._ws_client.connect()
        elapsed = (time.time() - t0) * 1000

        self._connected = True
        self._connect_ts = time.time()
        self._ohlc_rx_session = 0
        self._synth_ohlc_session = 0
        self._pipeline_trade_count = 0
        n_sec = int(self._settings.DNSE_WS_PIPELINE_ASSERT_MARKET_SEC or 0)
        if n_sec > 0:

            async def _assert_market_pipeline() -> None:
                await asyncio.sleep(float(n_sec))
                disp = self._ws_client.market_message_dispatches
                raw_b = self._ws_client.raw_ws_bytes_received
                dec_ok = self._ws_client.raw_ws_frames_decoded
                derr = self._ws_client.decode_error_count
                if disp <= 0:
                    raise RuntimeError(
                        "DNSE_WS_PIPELINE_ASSERT_MARKET_SEC: market_message_dispatches==0 after "
                        f"{n_sec}s (raw_recv={raw_b}, decoded_ok={dec_ok}, decode_errors={derr}). "
                        "Bật DNSE_WS_PIPELINE_DEBUG=true để xem RAW_WS → DECODED → SUB OK → MARKET DATA."
                    )

            asyncio.create_task(_assert_market_pipeline())
        logger.info(
            "WebSocket connected",
            extra={"elapsed_ms": round(elapsed, 1), "session_id": self._ws_client._session_id},
        )

    async def disconnect(self) -> None:
        """Gracefully disconnect from the WebSocket gateway."""
        if not self._connected:
            return
        logger.info("Disconnecting from WebSocket...")
        await self._ws_client.disconnect()
        self._connected = False
        self._trade_ohlc_synth = None
        self._synth_suppressed = False

        stats = self.buffer.stats()
        logger.info(
            "WebSocket disconnected",
            extra={
                "total_messages": self._msg_count,
                "uptime_s": round(self.buffer.uptime_seconds, 1),
                "quotes_tracked": stats.total_quotes,
                "trades_buffered": stats.total_trades,
                "ohlc_buffered": stats.total_ohlc,
            },
        )

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ---- Subscription API ----

    async def subscribe_quotes(
        self,
        symbols: List[str],
        board_id: str = "G1",
    ) -> None:
        """Subscribe to real-time BBO quotes for the given symbols."""
        bid = (board_id or "G1").strip() or "G1"
        new_symbols = [s for s in symbols if (s, bid) not in self._subscribed_quotes]
        if not new_symbols:
            logger.debug("All symbols already subscribed for quotes")
            return

        await self._ws_client.subscribe_quotes(
            symbols=new_symbols,
            on_quote=self._on_quote,
            encoding=self._settings.WS_ENCODING,
            board_id=bid,
        )
        self._subscribed_quotes.update((s, bid) for s in new_symbols)
        logger.info("Subscribed to quotes", extra={"symbols": new_symbols})

    async def subscribe_trades(
        self,
        symbols: List[str],
        board_id: str = "G1",
    ) -> None:
        """Subscribe to real-time trade (match) updates."""
        bid = (board_id or "G1").strip() or "G1"
        new_symbols = [s for s in symbols if (s, bid) not in self._subscribed_trades]
        if not new_symbols:
            return

        await self._ws_client.subscribe_trades(
            symbols=new_symbols,
            on_trade=self._on_trade,
            encoding=self._settings.WS_ENCODING,
            board_id=bid,
        )
        self._subscribed_trades.update((s, bid) for s in new_symbols)
        logger.info("Subscribed to trades", extra={"symbols": new_symbols})

    async def subscribe_ohlc(
        self,
        symbols: List[str],
        resolution: str = "1",
        board_id: str = "G1",
    ) -> None:
        """Subscribe to OHLC candle updates. Resolution: 1,3,5,15,30,1H,1D,1W.

        ``board_id`` mặc định ``G1`` (phái sinh), cùng quy ước với :meth:`subscribe_quotes`.
        """
        new_symbols = [s for s in symbols if s not in self._subscribed_ohlc]
        if not new_symbols:
            return

        await self._ws_client.subscribe_ohlc(
            symbols=new_symbols,
            resolution=resolution,
            on_ohlc=self._on_ohlc,
            encoding=self._settings.WS_ENCODING,
            board_id=board_id,
            board_in_ohlc_channel=self._settings.DNSE_WS_OHLC_BOARD_IN_CHANNEL,
            prefer_closed=self._settings.DNSE_WS_OHLC_PREFER_CLOSED,
        )
        self._subscribed_ohlc.update(new_symbols)
        if self._settings.DNSE_OHLC_SYNTH_FROM_TRADES:
            self._synth_suppressed = False
            self._trade_ohlc_synth = TradeOhlcSynth(
                period_sec=resolution_param_to_seconds(resolution),
                resolution_minutes=resolution_param_to_minutes_int(resolution),
                symbols=set(self._subscribed_ohlc),
                push=self._push_synthetic_ohlc,
            )
        logger.info(
            "Subscribed to OHLC",
            extra={
                "symbols": new_symbols,
                "resolution": resolution,
                "board_id": board_id,
                "ohlc_board_in_channel": self._settings.DNSE_WS_OHLC_BOARD_IN_CHANNEL,
                "prefer_ohlc_closed": self._settings.DNSE_WS_OHLC_PREFER_CLOSED,
                "synth_from_trades": self._settings.DNSE_OHLC_SYNTH_FROM_TRADES,
            },
        )

    async def subscribe_sec_def(self, symbols: List[str], board_id: str = "G1") -> None:
        """Subscribe security definition (OI, limits, …). Giúp xác nhận WS có đẩy dữ liệu."""
        bid = (board_id or "G1").strip() or "G1"
        new_symbols = [s for s in symbols if (s, bid) not in self._subscribed_sec_def]
        if not new_symbols:
            return
        await self._ws_client.subscribe_sec_def(
            symbols=new_symbols,
            on_sec_def=self._on_sec_def,
            encoding=self._settings.WS_ENCODING,
            board_id=bid,
        )
        self._subscribed_sec_def.update((s, bid) for s in new_symbols)
        logger.info("Subscribed to sec_def", extra={"symbols": new_symbols})

    # ---- External callback registration ----

    def on_quote(self, callback: Callback) -> None:
        """Register an external callback for quote updates."""
        self._quote_callbacks.append(callback)

    def on_trade(self, callback: Callback) -> None:
        """Register an external callback for trade updates."""
        self._trade_callbacks.append(callback)

    def on_ohlc(self, callback: Callback) -> None:
        """Register an external callback for OHLC updates."""
        self._ohlc_callbacks.append(callback)

    # ---- Internal dispatch (called by SDK TradingClient) ----

    def _on_quote(self, quote: Quote) -> None:
        """Handle incoming quote from the SDK."""
        self._normalize_incoming_symbol(quote)
        self._msg_count += 1
        self._last_msg_ts = time.time()

        asyncio.ensure_future(self.buffer.on_quote(quote))

        for cb in self._quote_callbacks:
            try:
                result = cb(quote)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                logger.error("Quote callback error", extra={"error": str(e), "symbol": quote.symbol})

    async def _push_synthetic_ohlc(self, ohlc: Ohlc) -> None:
        """Đưa nến tổng hợp từ trade vào buffer + callback (không tính vào ws_ohlc_rx)."""
        self._normalize_incoming_symbol(ohlc)
        self._msg_count += 1
        self._last_msg_ts = time.time()
        self._synth_ohlc_session += 1
        if self._synth_ohlc_session == 1:
            logger.info(
                "OHLC tổng hợp từ trade (synth_trade): WS có thể không đẩy kênh ohlc — "
                "xem enum Market Data / hỏi DNSE support.",
                extra={
                    "symbol": getattr(ohlc, "symbol", None),
                    "resolution": getattr(ohlc, "resolution", None),
                },
            )
        await self.buffer.on_ohlc(ohlc)
        for cb in self._ohlc_callbacks:
            try:
                result = cb(ohlc)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                logger.error(
                    "OHLC callback error (synthetic)",
                    extra={"error": str(e), "symbol": getattr(ohlc, "symbol", None)},
                )

    async def _synth_on_trade(self, trade: Trade) -> None:
        if self._trade_ohlc_synth is None or self._synth_suppressed:
            return
        async with self._synth_lock:
            await self._trade_ohlc_synth.on_trade(trade)

    def _on_trade(self, trade: Trade) -> None:
        """Handle incoming trade from the SDK."""
        self._normalize_incoming_symbol(trade)
        self._msg_count += 1
        self._last_msg_ts = time.time()
        if self._settings.DNSE_WS_PIPELINE_DEBUG:
            self._pipeline_trade_count += 1
            n = self._pipeline_trade_count
            if n <= 40 or n % 100 == 0:
                print(
                    f"TICK(pipeline_trade) #{n} symbol={trade.symbol!r} price={trade.price!r}",
                    flush=True,
                )

        asyncio.ensure_future(self.buffer.on_trade(trade))
        if self._settings.DNSE_OHLC_SYNTH_FROM_TRADES:
            asyncio.ensure_future(self._synth_on_trade(trade))

        for cb in self._trade_callbacks:
            try:
                result = cb(trade)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                logger.error("Trade callback error", extra={"error": str(e), "symbol": trade.symbol})

    def _on_sec_def(self, secdef) -> None:
        """Internal: sec_def stream — OI cho TTM/HMM (REST secdef phái sinh thường không có openInterestQuantity)."""
        self._normalize_incoming_symbol(secdef)
        self._msg_count += 1
        self._last_msg_ts = time.time()
        self._sec_def_rx_count += 1
        try:
            sym = getattr(secdef, "symbol", None)
            if sym and str(sym).strip():
                oi_raw = getattr(secdef, "openInterestQuantity", 0)
                try:
                    oi_val = int(oi_raw)
                except (TypeError, ValueError):
                    oi_val = 0
                record_open_interest_from_ws(str(sym).strip(), oi_val)
                if self._sec_def_rx_count <= 3 or oi_val > 0:
                    logger.info(
                        "sec_def received (OI cache)",
                        extra={
                            "symbol": str(sym).strip(),
                            "openInterestQuantity": oi_val,
                            "rx_count": self._sec_def_rx_count,
                        },
                    )
        except Exception as ex:
            logger.warning("sec_def OI cache failed", extra={"error": str(ex)})
        if not getattr(self, "_sec_def_logged_once", False):
            self._sec_def_logged_once = True
            logger.info(
                "First sec_def received on WebSocket",
                extra={"symbol": getattr(secdef, "symbol", None)},
            )

    def _on_ohlc(self, ohlc: Ohlc) -> None:
        """Handle incoming OHLC candle from the SDK."""
        self._normalize_incoming_symbol(ohlc)
        typ = str(getattr(ohlc, "type", "") or "")
        if self._settings.DNSE_OHLC_SYNTH_STOP_WHEN_WS_OHLC and typ != "synth_trade":
            self._synth_suppressed = True
        self._msg_count += 1
        self._last_msg_ts = time.time()
        self._ohlc_rx_session += 1
        if self._ohlc_rx_session == 1:
            logger.info(
                "First OHLC received on WebSocket (feed OK for JSONL / TTM bars)",
                extra={
                    "symbol": getattr(ohlc, "symbol", None),
                    "resolution": getattr(ohlc, "resolution", None),
                    "time": getattr(ohlc, "time", None),
                },
            )

        asyncio.ensure_future(self.buffer.on_ohlc(ohlc))

        for cb in self._ohlc_callbacks:
            try:
                result = cb(ohlc)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                logger.error(
                    "OHLC callback error",
                    extra={"error": str(e), "symbol": getattr(ohlc, "symbol", None)},
                )

    # ---- Convenience read-through to buffer ----

    async def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        return await self.buffer.get_latest_quote(symbol)

    async def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        return await self.buffer.get_latest_trade(symbol)

    async def get_ohlc_history(self, symbol: str, n: Optional[int] = None) -> list:
        return await self.buffer.get_ohlc_history(symbol, n)

    # ---- Metrics ----

    @property
    def msg_count(self) -> int:
        return self._msg_count

    @property
    def ohlc_rx_count(self) -> int:
        """Số message OHLC đã nhận trên WebSocket trong phiên kết nối hiện tại."""
        return self._ohlc_rx_session

    @property
    def sec_def_rx_count(self) -> int:
        """Số frame security_definition / sec_def đã dispatch (OI từ WS)."""
        return self._sec_def_rx_count

    @property
    def ohlc_synth_count(self) -> int:
        """Số bản ghi OHLC tổng hợp từ trade (``synth_trade``) trong phiên hiện tại."""
        return self._synth_ohlc_session

    @property
    def ws_raw_frames(self) -> int:
        """Mọi frame WS decode được (kể cả ack); nếu >0 mà msg_count=0 thì xem log [dnse-ws]."""
        return getattr(self._ws_client, "raw_ws_frames_decoded", 0)

    @property
    def ws_market_dispatches(self) -> int:
        """Số callback market (quote/trade/ohlc/...) đã gọi; nên gần msg_count."""
        return getattr(self._ws_client, "market_message_dispatches", 0)

    @property
    def ws_raw_bytes_received(self) -> int:
        return getattr(self._ws_client, "raw_ws_bytes_received", 0)

    @property
    def ws_decode_errors(self) -> int:
        return getattr(self._ws_client, "decode_error_count", 0)

    @property
    def uptime_seconds(self) -> float:
        if self._connect_ts == 0.0:
            return 0.0
        return time.time() - self._connect_ts

    def get_subscriptions(self) -> Dict[str, List[str]]:
        def _pair_lines(pairs: Set[Tuple[str, str]]) -> List[str]:
            return sorted(f"{s}@{b}" for s, b in pairs)

        return {
            "quotes": _pair_lines(self._subscribed_quotes),
            "trades": _pair_lines(self._subscribed_trades),
            "ohlc": sorted(self._subscribed_ohlc),
            "sec_def": _pair_lines(self._subscribed_sec_def),
        }
