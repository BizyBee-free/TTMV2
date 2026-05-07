"""TradingClient - Async WebSocket client for DNSE market data.

This is a simplified version based on the official DNSE SDK.
The full SDK client.py (~25KB) handles all subscription types.
"""

import asyncio
import json
import logging
import sys
from typing import Callable, Dict, List, Optional, Any, Set

from .auth import AuthManager
from .connection import WebSocketConnection
from .encoding import MessageEncoder, MessageDecoder
from .models import Quote, Trade, Ohlc, ExpectedPrice, SecurityDefinition, TradeExtra, MarketIndex

logger = logging.getLogger(__name__)


def _normalize_ws_channel_name(channel: str) -> str:
    """Map ``ohlc.1m.json`` / ``ohlc.1m.msgpack`` → ``ohlc`` để khớp :meth:`_dispatch` model_map."""
    c = (channel or "").strip().lower()
    if c.startswith("ohlc."):
        return "ohlc"
    if c.startswith("ohlc_closed."):
        return "ohlc"
    if c.startswith("candles.") or c.startswith("kline."):
        return "ohlc"
    return c


def _inject_ohlc_timeframe_from_channel(data: Dict[str, Any], channel: Any) -> Dict[str, Any]:
    """Bổ sung ``timeframe`` từ tên kênh (vd. ``ohlc.1m.json``) khi payload thiếu — để ``resolution`` đúng."""
    if not isinstance(data, dict):
        return data
    if data.get("timeframe") is not None:
        return data
    ch = str(channel or "").strip().lower()
    if not (ch.startswith("ohlc.") or ch.startswith("ohlc_closed.")):
        return data
    parts = ch.split(".")
    if len(parts) < 2:
        return data
    seg = parts[1]
    out = dict(data)
    if seg.isdigit():
        out["timeframe"] = f"{int(seg)}m"
    elif seg in ("1m", "3m", "5m", "15m", "30m", "1h", "1d", "1w"):
        out["timeframe"] = seg
    return out


def _normalize_stream_t_key(raw: Any) -> str:
    """Chuẩn hóa ``T``/``t`` từ JSON hoặc msgpack (bytes, ký tự ASCII dạng int).

    Khớp openapi-sdk ``_dispatch_message``: msgpack có thể để ``T`` là bytes
    (``b'sd'``); phải decode trước khi so khớp ``stream_map`` / ``SecurityDefinition``.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("ascii", errors="replace").lower()
    if isinstance(raw, int) and 32 <= raw <= 126:
        return chr(raw).lower()
    return str(raw).lower()


def _coerce_stream_type_fields(message: Dict[str, Any]) -> None:
    """Ghi đè ``T``/``t`` bằng chuỗi đã chuẩn (``sd``, ``q``, …) để model khớp openapi-sdk."""
    for k in ("T", "t"):
        if k not in message:
            continue
        raw = message[k]
        if raw is None:
            continue
        norm = _normalize_stream_t_key(raw)
        if norm:
            message[k] = norm
    inner = message.get("data") or message.get("Data")
    if isinstance(inner, dict):
        _coerce_stream_type_fields(inner)


def _merge_ws_nested_data(message: Dict[str, Any]) -> Dict[str, Any]:
    """Một số gateway gửi ``T``/``t`` và fields nến chỉ trong ``data``/``Data``."""
    inner = message.get("data") or message.get("Data")
    if not isinstance(inner, dict):
        return message
    if message.get("T") is not None or message.get("t") is not None:
        return message
    inner_t = inner.get("T") is not None or inner.get("t") is not None
    inner_ohlc = inner.get("open") is not None and inner.get("high") is not None
    if not inner_t and not inner_ohlc:
        return message
    merged = dict(message)
    merged.update(inner)
    return merged


def _pipeline_validate_stream_payload(t_val: str, d: Dict[str, Any]) -> Optional[str]:
    """DNSE stream uses ``T``/``t`` + symbol + price (not a single ``price`` top-level for all frames)."""
    if t_val == "t":
        if d.get("symbol") is None and d.get("Symbol") is None:
            return "T=t trade missing symbol/Symbol"
        # msgpack thường camelCase: matchPrice (phái sinh); cổ phiếu có thể price/Price
        if (
            d.get("price") is None
            and d.get("Price") is None
            and d.get("matchPrice") is None
            and d.get("MatchPrice") is None
            and d.get("match_price") is None
        ):
            return "T=t trade missing price/Price/matchPrice"
    if t_val == "q":
        if d.get("symbol") is None and d.get("Symbol") is None:
            return "T=q quote missing symbol/Symbol"
    if t_val == "b":
        if d.get("symbol") is None and d.get("Symbol") is None:
            return "T=b ohlc missing symbol/Symbol"
    return None


def _resolution_to_dnse_ohlc_timeframe(resolution: Any) -> str:
    """Map 1,3,5,15,30,1H,... -> tên kênh ``ohlc.*.json`` (giống PyPI ``dnse``)."""
    if resolution is None:
        return "1m"
    s = str(resolution).strip().upper()
    m = {
        "1": "1m",
        "3": "3m",
        "5": "5m",
        "15": "15m",
        "30": "30m",
        "1H": "1h",
        "1D": "1d",
        "1W": "1w",
    }
    if s in m:
        return m[s]
    sl = str(resolution).strip().lower()
    if len(sl) >= 2 and sl[-1] in ("m", "h", "d", "w"):
        return sl
    if str(resolution).isdigit():
        return f"{int(resolution)}m"
    return "1m"


class TradingClient:
    """Async client for DNSE WebSocket market data streams."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "wss://ws-openapi.dnse.com.vn/v1/stream",
        encoding: str = "msgpack",
        *,
        subscribe_use_session_id: bool = False,
        pipeline_debug: bool = False,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url
        self._encoding = encoding
        self._subscribe_use_session_id = subscribe_use_session_id
        self._pipeline_debug = bool(pipeline_debug)

        self._auth = AuthManager(api_key, api_secret)
        self._connection = WebSocketConnection(base_url)
        self._encoder = MessageEncoder(encoding)
        self._decoder = MessageDecoder(encoding)

        self._session_id: Optional[str] = None
        self._running = False
        self._callbacks: dict[str, Callable] = {}
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._rx_frame_count: int = 0
        self._raw_recv_count: int = 0
        self._decode_error_count: int = 0
        self._rx_market_dispatches: int = 0
        self._pipeline_raw_log_i: int = 0
        self._unknown_frame_log_count: int = 0
        self._stream_no_cb_logged: set[str] = set()
        # Gửi lại sau reconnect (1006, …)
        self._replay_quote_by_board: Dict[str, Dict[str, Any]] = {}
        self._replay_trade_by_board: Dict[str, Dict[str, Any]] = {}
        self._replay_ohlc: Optional[Dict[str, Any]] = None
        self._replay_sec_def_by_board: Dict[str, Dict[str, Any]] = {}
        self._replay_expected_price_by_board: Dict[str, Dict[str, Any]] = {}
        # ``tick.{board}.*`` — khớp openapi-sdk: chỉ trade trên tick (quote = top_price, …).
        self._tick_symbols_by_board: Dict[str, Set[str]] = {}
        self._top_price_symbols_by_board: Dict[str, Set[str]] = {}
        self._sec_def_symbols_by_board: Dict[str, Set[str]] = {}
        self._exp_price_symbols_by_board: Dict[str, Set[str]] = {}
        # Tránh gửi lại cùng một payload subscribe (gateway có thể đóng 1006).
        self._tick_last_sent: Dict[str, tuple] = {}
        self._top_price_last_sent: Dict[str, tuple] = {}
        self._sec_def_last_sent: Dict[str, tuple] = {}
        self._exp_price_last_sent: Dict[str, tuple] = {}

    def _ws_stream_suffix(self) -> str:
        """Hậu tố tên kênh theo encoding — openapi-sdk dùng ``.msgpack`` khi encoding=msgpack."""
        e = (self._encoding or "msgpack").strip().lower()
        return ".msgpack" if e == "msgpack" else ".json"

    def _with_session(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Chỉ gắn session khi bật — nhiều gateway market chỉ cần auth."""
        out = dict(msg)
        if self._subscribe_use_session_id and self._session_id:
            sid = self._session_id
            out["sessionId"] = sid
            out["SessionId"] = sid
        return out

    def _log_raw_frame(self, tag: str, raw: bytes) -> None:
        if not self._pipeline_debug:
            return
        self._pipeline_raw_log_i += 1
        n = self._pipeline_raw_log_i
        if n > 60 and n % 25 != 0:
            return
        head = raw[:200]
        suffix = f" … total_len={len(raw)}" if len(raw) > 200 else ""
        print(f"{tag}{suffix} {head!r}", flush=True)

    async def _send_dict(self, msg: Dict[str, Any], tag: str) -> None:
        if self._pipeline_debug:
            try:
                redacted = dict(msg)
                for k in list(redacted.keys()):
                    lk = str(k).lower()
                    if lk in ("signature", "apisecret", "api_secret", "secret", "password", "token"):
                        redacted[k] = "<redacted>"
                safe = json.dumps(redacted, default=str)
                if len(safe) > 800:
                    safe = safe[:800] + "…"
                print(f"SENDING {tag}: {safe}", flush=True)
            except Exception:
                print(f"SENDING {tag}: keys={list(msg.keys())}", flush=True)
        await self._connection.send(self._encoder.encode(msg))

    def _mark_dispatch(self, cb_key: str, obj: Any) -> None:
        self._rx_market_dispatches += 1
        if not self._pipeline_debug:
            return
        n = self._rx_market_dispatches
        if n <= 30 or n % 50 == 0:
            sym = getattr(obj, "symbol", None)
            print(f"MARKET DATA OK #{n}: {cb_key} symbol={sym!r}", flush=True)

    def _log_decoded_summary(self, decoded: Dict[str, Any]) -> None:
        if not self._pipeline_debug:
            return
        keys = list(decoded.keys())[:18]
        act = decoded.get("action") or decoded.get("Action") or ""
        ch = decoded.get("channel") or decoded.get("Channel") or ""
        t = decoded.get("T") or decoded.get("t")
        print(
            f"DECODED keys={keys} action={act!r} channel={ch!r} T/t={t!r}",
            flush=True,
        )

    async def _flush_tick_subscribe(self, board_id: str) -> None:
        syms = self._tick_symbols_by_board.get(board_id) or set()
        if not syms:
            return
        sig = tuple(sorted(syms))
        if self._tick_last_sent.get(board_id) == sig:
            return
        self._tick_last_sent[board_id] = sig
        ext = self._ws_stream_suffix()
        ch = f"tick.{board_id}{ext}"
        msg = self._with_session(
            {
                "action": "subscribe",
                "channels": [{"name": ch, "symbols": list(sig)}],
            }
        )
        await self._send_dict(msg, "SUB_TICK")

    async def _flush_top_price_subscribe(self, board_id: str) -> None:
        syms = self._top_price_symbols_by_board.get(board_id) or set()
        if not syms:
            return
        sig = tuple(sorted(syms))
        if self._top_price_last_sent.get(board_id) == sig:
            return
        self._top_price_last_sent[board_id] = sig
        ext = self._ws_stream_suffix()
        ch = f"top_price.{board_id}{ext}"
        msg = self._with_session(
            {
                "action": "subscribe",
                "channels": [{"name": ch, "symbols": list(sig)}],
            }
        )
        await self._send_dict(msg, "SUB_TOP_PRICE")

    async def _flush_sec_def_channel_subscribe(self, board_id: str) -> None:
        syms = self._sec_def_symbols_by_board.get(board_id) or set()
        if not syms:
            return
        sig = tuple(sorted(syms))
        if self._sec_def_last_sent.get(board_id) == sig:
            return
        self._sec_def_last_sent[board_id] = sig
        ext = self._ws_stream_suffix()
        ch = f"security_definition.{board_id}{ext}"
        msg = self._with_session(
            {
                "action": "subscribe",
                "channels": [{"name": ch, "symbols": list(sig)}],
            }
        )
        await self._send_dict(msg, "SUB_SEC_DEF")

    async def _flush_expected_price_subscribe(self, board_id: str) -> None:
        syms = self._exp_price_symbols_by_board.get(board_id) or set()
        if not syms:
            return
        sig = tuple(sorted(syms))
        if self._exp_price_last_sent.get(board_id) == sig:
            return
        self._exp_price_last_sent[board_id] = sig
        ext = self._ws_stream_suffix()
        ch = f"expected_price.{board_id}{ext}"
        msg = self._with_session(
            {
                "action": "subscribe",
                "channels": [{"name": ch, "symbols": list(sig)}],
            }
        )
        await self._send_dict(msg, "SUB_EXPECTED_PRICE")

    async def _read_until_reconnect_auth_success(self) -> None:
        """Đọc frame sau AUTH(reconnect) cho đến ``auth_success``.

        Gateway đôi khi trả ``NOT_AUTHENTICATED`` (race / thứ tự) trước ``auth_success``.
        Chỉ đọc một ``receive()`` sẽ raise sớm → không gọi ``_replay_all_subscriptions``,
        kết nối vẫn sống nhưng mất market stream.
        """
        from .exceptions import AuthenticationError as _AuthErr

        transient_na = 0
        max_transient_na = 6
        for _ in range(48):
            raw = await self._connection.receive()
            self._raw_recv_count += 1
            self._log_raw_frame("RAW_WS(reconnect_auth)", raw)
            try:
                decoded = self._decoder.decode(raw)
            except Exception:
                continue
            batch = decoded if isinstance(decoded, list) else [decoded]
            for item in batch:
                if not isinstance(item, dict):
                    continue
                ract = str(item.get("action") or item.get("Action") or "").lower()
                if ract == "auth_success":
                    self._session_id = (
                        item.get("session_id")
                        or item.get("SessionId")
                        or self._session_id
                    )
                    logger.info("Reconnect WebSocket auth_success")
                    return
                if ract == "ping":
                    await self._send_dict({"action": "pong"}, "PONG_REPLY")
                    continue
                if ract == "pong":
                    continue
                if ract in ("auth_error",):
                    raise _AuthErr(f"Reconnect auth failed: {item}")
                if ract == "error":
                    code = str(item.get("code") or "").upper()
                    msg_l = str(item.get("message") or "").lower()
                    if code == "NOT_AUTHENTICATED" or "authenticate first" in msg_l:
                        transient_na += 1
                        if transient_na > max_transient_na:
                            raise _AuthErr(
                                f"Reconnect auth failed (too many NOT_AUTHENTICATED): {item}"
                            )
                        logger.warning(
                            "Reconnect: transient NOT_AUTHENTICATED (%s/%s); awaiting auth_success",
                            transient_na,
                            max_transient_na,
                        )
                        continue
                    raise _AuthErr(f"Reconnect auth failed: {item}")
                if str(item.get("status") or "").lower() == "error":
                    raise _AuthErr(f"Reconnect auth failed: {item}")
        raise _AuthErr("Reconnect auth: no auth_success within read budget")

    async def _on_auth_success_maybe_replay(self) -> None:
        """Sau auth_success, gửi lại subscription (an toàn khi frame lẻ không qua handshake reconnect)."""
        try:
            await self._replay_all_subscriptions()
        except Exception as sub_ex:
            logger.error(f"Re-subscribe after auth_success failed: {sub_ex}", exc_info=True)
            print(
                f"[dnse-ws] Re-subscribe after auth_success failed: {sub_ex}",
                file=sys.stderr,
                flush=True,
            )

    async def _replay_all_subscriptions(self) -> None:
        """Sau reconnect phải subscribe lại; nếu không, ws_frames tăng nhưng không có quote/OHLC."""
        did = False
        if self._replay_quote_by_board and self._callbacks.get("quote"):
            for bid, spec in self._replay_quote_by_board.items():
                await self.subscribe_quotes(
                    spec["symbols"],
                    self._callbacks["quote"],
                    encoding=spec["encoding"],
                    board_id=bid,
                )
            did = True
        if self._replay_trade_by_board and self._callbacks.get("trade"):
            for bid, spec in self._replay_trade_by_board.items():
                await self.subscribe_trades(
                    spec["symbols"],
                    self._callbacks["trade"],
                    encoding=spec["encoding"],
                    board_id=bid,
                )
            did = True
        if self._replay_ohlc and self._callbacks.get("ohlc"):
            r = self._replay_ohlc
            await self.subscribe_ohlc(
                r["symbols"],
                str(r["resolution"]),
                self._callbacks["ohlc"],
                encoding=r["encoding"],
                board_id=r["board_id"],
                board_in_ohlc_channel=bool(r.get("board_in_ohlc_channel", False)),
            )
            did = True
        if self._replay_sec_def_by_board and self._callbacks.get("sec_def"):
            for bid, spec in self._replay_sec_def_by_board.items():
                await self.subscribe_sec_def(
                    spec["symbols"],
                    self._callbacks["sec_def"],
                    encoding=spec["encoding"],
                    board_id=bid,
                )
            did = True
        if self._replay_expected_price_by_board and self._callbacks.get("expected_price"):
            for bid, spec in self._replay_expected_price_by_board.items():
                await self.subscribe_expected_price(
                    spec["symbols"],
                    self._callbacks["expected_price"],
                    encoding=spec["encoding"],
                    board_id=bid,
                )
            did = True
        if did:
            print(
                "[dnse-ws] Re-sent WebSocket subscriptions after reconnect",
                file=sys.stderr,
                flush=True,
            )

    @property
    def raw_ws_frames_decoded(self) -> int:
        """Mọi frame decode thành dict thành công (kể cả ack / không có channel market)."""
        return self._rx_frame_count

    @property
    def raw_ws_bytes_received(self) -> int:
        """Mọi lần ``recv`` từ socket (kể cả decode lỗi)."""
        return self._raw_recv_count

    @property
    def decode_error_count(self) -> int:
        return self._decode_error_count

    @property
    def market_message_dispatches(self) -> int:
        """Số lần dispatch ra quote / trade / ohlc / ... (callback đã gọi)."""
        return self._rx_market_dispatches

    async def connect(self) -> None:
        """Connect and authenticate with the WebSocket gateway."""
        await self._connection.connect()

        self._tick_last_sent = {}
        self._top_price_last_sent = {}
        self._sec_def_last_sent = {}
        self._exp_price_last_sent = {}
        self._rx_frame_count = 0
        self._raw_recv_count = 0
        self._decode_error_count = 0
        self._pipeline_raw_log_i = 0
        self._rx_market_dispatches = 0
        self._unknown_frame_log_count = 0
        self._stream_no_cb_logged = set()

        welcome_raw = await self._connection.receive()
        self._raw_recv_count += 1
        self._log_raw_frame("RAW_WS(welcome)", welcome_raw)
        welcome = self._decoder.decode(welcome_raw)
        self._session_id = welcome.get("session_id") or welcome.get("SessionId")
        if self._pipeline_debug:
            print(
                f"DECODED(welcome) session_id={self._session_id!r} keys={list(welcome.keys())[:12]}",
                flush=True,
            )

        auth_msg = self._auth.create_auth_message()
        await self._send_dict(auth_msg, "AUTH")

        response = await self._connection.receive()
        self._raw_recv_count += 1
        self._log_raw_frame("RAW_WS(auth_response)", response)
        decoded = self._decoder.decode(response)
        if self._pipeline_debug:
            print(
                f"AUTH_RESPONSE action={decoded.get('action')!r} status={decoded.get('status')!r} "
                f"keys={list(decoded.keys())[:14]}",
                flush=True,
            )

        act = str(decoded.get("action") or decoded.get("Action") or "").lower()
        if act in ("auth_error", "error"):
            from .exceptions import AuthenticationError
            raise AuthenticationError(f"Auth failed: {decoded}")
        if decoded.get("status") == "error":
            from .exceptions import AuthenticationError
            raise AuthenticationError(f"Auth failed: {decoded}")

        self._session_id = (
            decoded.get("session_id") or decoded.get("SessionId") or self._session_id
        )
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Gửi ping ứng dụng định kỳ (giống PyPI ``dnse``) để tránh gateway đóng khi ít bản tin."""
        while self._running:
            try:
                await asyncio.sleep(25)
                if not self._running:
                    break
                if self._connection.is_connected:
                    await self._send_dict({"action": "ping"}, "APP_PING")
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(3)

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket gateway."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        await self._connection.close()

    async def subscribe_quotes(
        self,
        symbols: List[str],
        on_quote: Callable[[Quote], Any],
        encoding: str = "msgpack",
        board_id: str = "G1",
    ) -> None:
        """Subscribe to quote (BBO) updates for given symbols."""
        self._callbacks["quote"] = on_quote
        bset = self._top_price_symbols_by_board.setdefault(board_id, set())
        bset.update(symbols)
        self._replay_quote_by_board[board_id] = {
            "symbols": sorted(bset),
            "encoding": encoding,
        }
        await self._flush_top_price_subscribe(board_id)

    async def subscribe_trades(
        self,
        symbols: List[str],
        on_trade: Callable[[Trade], Any],
        encoding: str = "msgpack",
        board_id: str = "G1",
    ) -> None:
        """Subscribe to trade updates for given symbols."""
        self._callbacks["trade"] = on_trade
        bset = self._tick_symbols_by_board.setdefault(board_id, set())
        bset.update(symbols)
        self._replay_trade_by_board[board_id] = {
            "symbols": sorted(bset),
            "encoding": encoding,
        }
        await self._flush_tick_subscribe(board_id)

    async def subscribe_ohlc(
        self,
        symbols: List[str],
        resolution: str,
        on_ohlc: Callable[[Ohlc], Any],
        encoding: str = "msgpack",
        board_id: str = "G1",
        board_in_ohlc_channel: bool = False,
        prefer_closed: bool = False,
    ) -> None:
        """Subscribe to OHLC candle updates. Resolution: 1,3,5,15,30,1H,1D,1W.

        Tên kênh ``ohlc.{timeframe}{.msgpack|.json}`` như openapi-sdk (không chèn board).
        ``board_in_ohlc_channel`` giữ tương thích API; không đổi tên kênh.
        """
        self._callbacks["ohlc"] = on_ohlc
        res = resolution
        if isinstance(res, str) and res.isdigit():
            res = int(res)
        self._replay_ohlc = {
            "symbols": list(symbols),
            "resolution": res,
            "encoding": encoding,
            "board_id": board_id,
            "board_in_ohlc_channel": board_in_ohlc_channel,
        }
        tf = _resolution_to_dnse_ohlc_timeframe(res)
        ext = self._ws_stream_suffix()
        channels = []
        if prefer_closed:
            channels.append(f"ohlc_closed.{tf}{ext}")
        channels.append(f"ohlc.{tf}{ext}")
        # giữ thứ tự + unique
        seen = set()
        ordered = []
        for ch in channels:
            if ch not in seen:
                seen.add(ch)
                ordered.append(ch)
        msg = self._with_session(
            {
                "action": "subscribe",
                "channels": [{"name": ch, "symbols": list(symbols)} for ch in ordered],
            }
        )
        await self._send_dict(
            msg,
            "SUB_OHLC_CLOSED+LIVE" if prefer_closed else "SUB_OHLC",
        )

    async def subscribe_expected_price(
        self,
        symbols: List[str],
        on_expected_price: Callable[[ExpectedPrice], Any],
        encoding: str = "msgpack",
        board_id: str = "G1",
    ) -> None:
        """Subscribe to expected price updates (ATO/ATC)."""
        self._callbacks["expected_price"] = on_expected_price
        bset = self._exp_price_symbols_by_board.setdefault(board_id, set())
        bset.update(symbols)
        self._replay_expected_price_by_board[board_id] = {
            "symbols": sorted(bset),
            "encoding": encoding,
        }
        await self._flush_expected_price_subscribe(board_id)

    async def subscribe_sec_def(
        self,
        symbols: List[str],
        on_sec_def: Callable[[SecurityDefinition], Any],
        encoding: str = "msgpack",
        board_id: str = "G1",
    ) -> None:
        """Subscribe to security definition updates."""
        self._callbacks["sec_def"] = on_sec_def
        bset = self._sec_def_symbols_by_board.setdefault(board_id, set())
        bset.update(symbols)
        self._replay_sec_def_by_board[board_id] = {
            "symbols": sorted(bset),
            "encoding": encoding,
        }
        await self._flush_sec_def_channel_subscribe(board_id)

    async def _listen_loop(self) -> None:
        """Background task that receives and dispatches messages."""
        from .exceptions import ConnectionClosed

        while self._running:
            try:
                raw = await self._connection.receive()
                self._raw_recv_count += 1
                self._log_raw_frame("RAW_WS", raw)
                try:
                    decoded = self._decoder.decode(raw)
                except Exception as e:
                    self._decode_error_count += 1
                    if self._pipeline_debug:
                        print(f"DECODE_ERROR: {e!r} len={len(raw)}", flush=True)
                    continue
                if isinstance(decoded, list):
                    for item in decoded:
                        if not isinstance(item, dict):
                            continue
                        self._rx_frame_count += 1
                        self._log_decoded_summary(item)
                        act_low = str(item.get("action") or item.get("Action") or "").lower()
                        if act_low == "auth_success":
                            self._session_id = (
                                item.get("session_id")
                                or item.get("SessionId")
                                or self._session_id
                            )
                            await self._on_auth_success_maybe_replay()
                            continue
                        if act_low == "ping":
                            if self._pipeline_debug:
                                print("PING RECEIVED (server → client)", flush=True)
                            await self._send_dict({"action": "pong"}, "PONG_REPLY")
                            continue
                        if act_low == "pong":
                            if self._pipeline_debug:
                                print("PONG RECEIVED (server → client)", flush=True)
                            continue
                        self._dispatch(item)
                    continue
                if not isinstance(decoded, dict):
                    self._decode_error_count += 1
                    if self._pipeline_debug:
                        print(
                            f"DECODE_SKIP: expected dict, got {type(decoded).__name__}",
                            flush=True,
                        )
                    continue
                self._rx_frame_count += 1
                self._log_decoded_summary(decoded)
                act_low = str(decoded.get("action") or decoded.get("Action") or "").lower()
                if act_low == "auth_success":
                    self._session_id = (
                        decoded.get("session_id")
                        or decoded.get("SessionId")
                        or self._session_id
                    )
                    await self._on_auth_success_maybe_replay()
                    continue
                if act_low == "ping":
                    if self._pipeline_debug:
                        print("PING RECEIVED (server → client)", flush=True)
                    await self._send_dict({"action": "pong"}, "PONG_REPLY")
                    continue
                if act_low == "pong":
                    if self._pipeline_debug:
                        print("PONG RECEIVED (server → client)", flush=True)
                    continue
                self._dispatch(decoded)
            except ConnectionClosed as e:
                if e.recoverable and self._running:
                    logger.warning("Connection lost, attempting reconnect...")
                    try:
                        await self._connection.connect()
                        welcome_raw = await self._connection.receive()
                        self._raw_recv_count += 1
                        self._log_raw_frame("RAW_WS(reconnect_welcome)", welcome_raw)
                        welcome = self._decoder.decode(welcome_raw)
                        self._session_id = welcome.get("session_id") or welcome.get("SessionId")
                        auth_msg = self._auth.create_auth_message()
                        await self._send_dict(auth_msg, "AUTH(reconnect)")
                        await self._read_until_reconnect_auth_success()
                        self._tick_last_sent = {}
                        self._top_price_last_sent = {}
                        self._sec_def_last_sent = {}
                        self._exp_price_last_sent = {}
                        logger.info("Reconnected successfully")
                        try:
                            await self._replay_all_subscriptions()
                        except Exception as sub_ex:
                            logger.error(f"Re-subscribe after reconnect failed: {sub_ex}", exc_info=True)
                            print(
                                f"[dnse-ws] Re-subscribe failed: {sub_ex}",
                                file=sys.stderr,
                                flush=True,
                            )
                    except Exception as re:
                        logger.error(f"Reconnect failed: {re}")
                        await asyncio.sleep(5)
                else:
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                err = f"Error in listen loop: {e}"
                logger.error(err, exc_info=True)
                print(f"[dnse-ws] {err}", file=sys.stderr, flush=True)
                await asyncio.sleep(1)

    def _dispatch(self, message: dict) -> None:
        """Route incoming message to the correct callback."""
        raw = dict(message)
        message = _merge_ws_nested_data(raw)
        _coerce_stream_type_fields(message)
        act0 = str(message.get("action") or message.get("Action") or "").lower()
        if act0 == "subscribed":
            if self._pipeline_debug:
                try:
                    snippet = json.dumps(message, default=str)
                    if len(snippet) > 700:
                        snippet = snippet[:700] + "…"
                    print(f"SUB OK: {snippet}", flush=True)
                except Exception:
                    print(f"SUB OK keys={list(message.keys())}", flush=True)
            return
        if act0 == "success" and self._pipeline_debug:
            try:
                snippet = json.dumps(message, default=str)
                if len(snippet) > 500:
                    snippet = snippet[:500] + "…"
                print(f"SERVER success frame: {snippet}", flush=True)
            except Exception:
                print(f"SERVER success keys={list(message.keys())}", flush=True)
        # Một số bản gateway gửi nến không có ``T``/``t`` (chỉ có timeframe + OHLC).
        tv_raw = (
            message.get("T")
            or message.get("t")
            or message.get("Type")
            or message.get("type")
        )
        tv_norm_early = _normalize_stream_t_key(tv_raw)
        if (
            not tv_raw
            and self._callbacks.get("ohlc")
            and (
                message.get("timeframe") is not None
                or message.get("resolution") is not None
                or message.get("Resolution") is not None
            )
        ):
            sym = message.get("symbol") or message.get("Symbol")
            o = message.get("open")
            h = message.get("high")
            if sym and o is not None and h is not None:
                try:
                    obj = Ohlc.from_dict(message)
                    self._callbacks["ohlc"](obj)
                    self._mark_dispatch("ohlc", obj)
                except Exception as e:
                    err = f"Error processing implicit OHLC message: {e}"
                    logger.error(err, exc_info=True)
                    print(f"[dnse-ws] {err}", file=sys.stderr, flush=True)
                return
        t_val = tv_norm_early if tv_raw is not None else ""
        stream_map = {
            "q": ("quote", Quote),
            "t": ("trade", Trade),
            "b": ("ohlc", Ohlc),
            "e": ("expected_price", ExpectedPrice),
            "sd": ("sec_def", SecurityDefinition),
            "te": ("trade_extra", TradeExtra),
        }
        if t_val in stream_map:
            cb_key, model_cls = stream_map[t_val]
            callback = self._callbacks.get(cb_key)
            if callback:
                pw = _pipeline_validate_stream_payload(t_val, message)
                if pw and self._pipeline_debug:
                    print(f"PARSE_WARN: {pw}", flush=True)
                try:
                    obj = model_cls.from_dict(message)
                    callback(obj)
                    self._mark_dispatch(cb_key, obj)
                except Exception as e:
                    err = f"Error processing stream T={t_val!r} message: {e}"
                    logger.error(err, exc_info=True)
                    print(f"[dnse-ws] {err}", file=sys.stderr, flush=True)
            else:
                if t_val not in self._stream_no_cb_logged:
                    self._stream_no_cb_logged.add(t_val)
                    print(
                        f"[dnse-ws] No callback for stream T={t_val!r} "
                        f"(cần subscribe_* tương ứng, ví dụ trade khi chỉ nhận T=t)",
                        file=sys.stderr,
                        flush=True,
                    )
            return

        ch_raw = str(message.get("channel") or message.get("Channel") or "")
        ch_low = ch_raw.strip().lower()
        if ch_low.startswith("top_price."):
            channel = "quote"
        elif ch_low.startswith("security_definition."):
            channel = "sec_def"
        elif ch_low.startswith("expected_price."):
            channel = "expected_price"
        else:
            channel = _normalize_ws_channel_name(ch_raw)
            if not channel:
                channel = ch_raw.lower()
        _aliases = {
            "quotes": "quote",
            "trades": "trade",
            "candles": "ohlc",
            "kline": "ohlc",
            "securitydefinition": "sec_def",
            "security_definition": "sec_def",
        }
        if channel in _aliases:
            channel = _aliases[channel]

        model_map = {
            "quote": ("quote", Quote),
            "trade": ("trade", Trade),
            "ohlc": ("ohlc", Ohlc),
            "expected_price": ("expected_price", ExpectedPrice),
            "sec_def": ("sec_def", SecurityDefinition),
            "trade_extra": ("trade_extra", TradeExtra),
            "market_index": ("market_index", MarketIndex),
        }

        if channel in model_map:
            cb_key, model_cls = model_map[channel]
            callback = self._callbacks.get(cb_key)
            if callback:
                data = message.get("data") or message.get("Data") or message
                if model_cls is Ohlc and isinstance(data, dict):
                    data = _inject_ohlc_timeframe_from_channel(
                        data,
                        message.get("channel") or message.get("Channel"),
                    )
                if model_cls is SecurityDefinition and isinstance(data, dict) and isinstance(
                    message, dict
                ):
                    # DNSE đôi khi đặt openInterestQuantity ở root frame, không trong ``data`` lồng.
                    merged = dict(data)
                    for _k in (
                        "openInterestQuantity",
                        "open_interest_quantity",
                        "OpenInterestQuantity",
                    ):
                        if _k in message and message.get(_k) is not None:
                            merged[_k] = message[_k]
                    data = merged
                try:
                    obj = model_cls.from_dict(data)
                    callback(obj)
                    self._mark_dispatch(cb_key, obj)
                except Exception as e:
                    err = f"Error processing {channel} message: {e}"
                    logger.error(err, exc_info=True)
                    print(f"[dnse-ws] {err}", file=sys.stderr, flush=True)
            else:
                if self._unknown_frame_log_count < 8:
                    self._unknown_frame_log_count += 1
                    print(
                        f"[dnse-ws] No callback for channel={channel!r} (subscribe order?)",
                        file=sys.stderr,
                        flush=True,
                    )
            return

        # Không phải kênh market đã map — log vài frame đầu để debug (ack, lỗi, schema lạ).
        act_low = str(message.get("action") or message.get("Action") or "").lower()
        if act_low == "error":
            print(
                f"[dnse-ws] subscribe/API error: code={message.get('code')!r} "
                f"message={message.get('message')!r} raw_keys={list(message.keys())}",
                file=sys.stderr,
                flush=True,
            )
            return

        st = str(message.get("status") or message.get("Status") or "").lower()
        if st == "error" or message.get("error") or message.get("Error"):
            snippet = repr(message)
            if len(snippet) > 800:
                snippet = snippet[:800] + "..."
            print(f"[dnse-ws] Server error frame: {snippet}", file=sys.stderr, flush=True)
            return
        if self._unknown_frame_log_count < 12:
            self._unknown_frame_log_count += 1
            keys = list(message.keys())[:14]
            act = message.get("action") or message.get("Action") or ""
            print(
                f"[dnse-ws] Non-market frame ch={channel!r} action={act!r} keys={keys}",
                file=sys.stderr,
                flush=True,
            )
