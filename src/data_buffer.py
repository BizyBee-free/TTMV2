"""Thread-safe ring buffers for real-time market data.

Stores the latest quotes (BBO), recent trades, and OHLC candles
per symbol using bounded deques with asyncio locks for safety.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from vendor.dnse.trading_websocket.models import Ohlc, Quote, Trade


def _sym_key(symbol: Optional[str]) -> str:
    """Khóa dict thống nhất (DNSE có thể trả symbol khác hoa/thường)."""
    return (symbol or "").strip().upper()


@dataclass
class BufferStats:
    """Snapshot of buffer utilization."""
    total_quotes: int = 0
    total_trades: int = 0
    total_ohlc: int = 0
    symbols_tracked: int = 0
    last_update_ts: float = 0.0


class QuoteBuffer:
    """Keeps the latest BBO quote per symbol (overwrites on update)."""

    def __init__(self) -> None:
        self._data: Dict[str, Quote] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def update(self, quote: Quote) -> None:
        async with self._lock:
            key = _sym_key(getattr(quote, "symbol", None))
            if not key:
                return
            self._data[key] = quote
            self._timestamps[key] = time.time()

    async def get(self, symbol: str) -> Optional[Quote]:
        async with self._lock:
            return self._data.get(_sym_key(symbol))

    async def get_timestamp(self, symbol: str) -> Optional[float]:
        async with self._lock:
            return self._timestamps.get(_sym_key(symbol))

    async def get_all(self) -> Dict[str, Quote]:
        async with self._lock:
            return dict(self._data)

    async def symbols(self) -> List[str]:
        async with self._lock:
            return list(self._data.keys())

    @property
    def count(self) -> int:
        return len(self._data)


class TradeBuffer:
    """Ring buffer storing the last N trades per symbol."""

    def __init__(self, maxlen: int = 100) -> None:
        self._maxlen = maxlen
        self._data: Dict[str, Deque[Trade]] = {}
        self._lock = asyncio.Lock()

    async def append(self, trade: Trade) -> None:
        async with self._lock:
            key = _sym_key(getattr(trade, "symbol", None))
            if not key:
                return
            if key not in self._data:
                self._data[key] = deque(maxlen=self._maxlen)
            self._data[key].append(trade)

    async def get_latest(self, symbol: str) -> Optional[Trade]:
        async with self._lock:
            buf = self._data.get(_sym_key(symbol))
            if buf and len(buf) > 0:
                return buf[-1]
            return None

    async def get_history(self, symbol: str, n: Optional[int] = None) -> List[Trade]:
        async with self._lock:
            buf = self._data.get(_sym_key(symbol))
            if not buf:
                return []
            if n is None:
                return list(buf)
            return list(buf)[-n:]

    async def symbols(self) -> List[str]:
        async with self._lock:
            return list(self._data.keys())

    def trade_count(self, symbol: str) -> int:
        buf = self._data.get(_sym_key(symbol))
        return len(buf) if buf else 0


class OhlcBuffer:
    """Ring buffer storing the last N OHLC candles per symbol."""

    def __init__(self, maxlen: int = 50) -> None:
        self._maxlen = maxlen
        self._data: Dict[str, Deque[Ohlc]] = {}
        self._lock = asyncio.Lock()

    async def append(self, ohlc: Ohlc) -> None:
        async with self._lock:
            key = _sym_key(getattr(ohlc, "symbol", None))
            if not key:
                return
            if key not in self._data:
                self._data[key] = deque(maxlen=self._maxlen)
            buf = self._data[key]
            # Replace the last candle if same timestamp (live update of current bar)
            if buf and buf[-1].time == ohlc.time:
                buf[-1] = ohlc
            else:
                buf.append(ohlc)

    async def get_latest(self, symbol: str) -> Optional[Ohlc]:
        async with self._lock:
            buf = self._data.get(_sym_key(symbol))
            if buf and len(buf) > 0:
                return buf[-1]
            return None

    async def get_history(self, symbol: str, n: Optional[int] = None) -> List[Ohlc]:
        async with self._lock:
            buf = self._data.get(_sym_key(symbol))
            if not buf:
                return []
            if n is None:
                return list(buf)
            return list(buf)[-n:]

    async def symbols(self) -> List[str]:
        async with self._lock:
            return list(self._data.keys())

    def candle_count(self, symbol: str) -> int:
        buf = self._data.get(_sym_key(symbol))
        return len(buf) if buf else 0


class DataBuffer:
    """Unified facade over Quote / Trade / OHLC buffers.

    This is the single entry point that MarketDataManager writes to
    and strategies read from.
    """

    def __init__(
        self,
        trade_maxlen: int = 100,
        ohlc_maxlen: int = 50,
    ) -> None:
        self.quotes = QuoteBuffer()
        self.trades = TradeBuffer(maxlen=trade_maxlen)
        self.ohlc = OhlcBuffer(maxlen=ohlc_maxlen)

        self._update_count: int = 0
        self._first_update_ts: float = 0.0
        self._last_update_ts: float = 0.0

    # ---- Write API (called by MarketDataManager) ----

    async def on_quote(self, quote: Quote) -> None:
        await self.quotes.update(quote)
        self._touch()

    async def on_trade(self, trade: Trade) -> None:
        await self.trades.append(trade)
        self._touch()

    async def on_ohlc(self, ohlc: Ohlc) -> None:
        await self.ohlc.append(ohlc)
        self._touch()

    # ---- Read API (called by strategies / scripts) ----

    async def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        return await self.quotes.get(symbol)

    async def get_latest_trade(self, symbol: str) -> Optional[Trade]:
        return await self.trades.get_latest(symbol)

    async def get_ohlc_history(
        self, symbol: str, n: Optional[int] = None
    ) -> List[Ohlc]:
        return await self.ohlc.get_history(symbol, n)

    async def get_trade_history(
        self, symbol: str, n: Optional[int] = None
    ) -> List[Trade]:
        return await self.trades.get_history(symbol, n)

    async def get_all_quotes(self) -> Dict[str, Quote]:
        return await self.quotes.get_all()

    # ---- Stats ----

    def stats(self) -> BufferStats:
        return BufferStats(
            total_quotes=self.quotes.count,
            total_trades=sum(
                self.trades.trade_count(s)
                for s in (self.trades._data.keys())
            ),
            total_ohlc=sum(
                self.ohlc.candle_count(s)
                for s in (self.ohlc._data.keys())
            ),
            symbols_tracked=self.quotes.count,
            last_update_ts=self._last_update_ts,
        )

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def uptime_seconds(self) -> float:
        if self._first_update_ts == 0.0:
            return 0.0
        return time.time() - self._first_update_ts

    def _touch(self) -> None:
        now = time.time()
        self._update_count += 1
        self._last_update_ts = now
        if self._first_update_ts == 0.0:
            self._first_update_ts = now
