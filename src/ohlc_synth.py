"""Tổng hợp nến OHLC từ khớp lệnh WebSocket khi gateway không đẩy kênh ``ohlc.*``.

Tham chiếu REST (SDK DNSE): ``getOhlc('STOCK', { symbol, resolution, from, to })`` —
luồng realtime có thể chỉ có ``T=t`` (tick) mà không có ``T=b`` / channel ohlc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Awaitable, Callable, Dict, Optional, Set

from vendor.dnse.trading_websocket.models import Ohlc, Trade


def resolution_param_to_seconds(resolution: str) -> int:
    r = str(resolution).strip().upper()
    m = {"1": 60, "3": 180, "5": 300, "15": 900, "30": 1800, "1H": 3600, "1D": 86400, "1W": 604800}
    return max(m.get(r, 60), 1)


def resolution_param_to_minutes_int(resolution: str) -> int:
    r = str(resolution).strip().upper()
    m = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "1H": 60, "1D": 1440, "1W": 10080}
    return max(m.get(r, 1), 1)


@dataclass
class _Agg:
    bucket: int
    o: Decimal
    h: Decimal
    l: Decimal
    c: Decimal
    vol: int


class TradeOhlcSynth:
    """Cập nhật nến theo ``period_sec``; đẩy bản ghi có ``Ohlc.type == synth_trade`` (ghi đè bucket hiện tại trong buffer)."""

    def __init__(
        self,
        period_sec: int,
        resolution_minutes: int,
        symbols: Set[str],
        push: Callable[[Ohlc], Awaitable[None]],
    ) -> None:
        self.period_sec = max(int(period_sec), 1)
        self.resolution_minutes = int(resolution_minutes)
        self.symbols = {s.strip().upper() for s in symbols if s and str(s).strip()}
        self._push = push
        self._by_sym: Dict[str, _Agg] = {}

    def _emit(self, sym: str, agg: _Agg) -> Ohlc:
        return Ohlc(
            symbol=sym,
            resolution=self.resolution_minutes,
            open=agg.o,
            high=agg.h,
            low=agg.l,
            close=agg.c,
            volume=agg.vol,
            time=agg.bucket,
            lastUpdated=agg.bucket,
            type="synth_trade",
        )

    async def on_trade(self, trade: Trade) -> None:
        sym = (trade.symbol or "").strip().upper()
        if not sym or sym not in self.symbols:
            return
        px = float(trade.price or 0.0)
        if px <= 0:
            return
        qty = int(trade.quantity or 0)
        ts = int(getattr(trade, "event_time_sec", 0) or 0) or int(time.time())
        bucket = (ts // self.period_sec) * self.period_sec
        dpx = Decimal(str(px))
        cur = self._by_sym.get(sym)
        if cur is None:
            self._by_sym[sym] = _Agg(bucket, dpx, dpx, dpx, dpx, qty)
            await self._push(self._emit(sym, self._by_sym[sym]))
            return
        if cur.bucket != bucket:
            await self._push(self._emit(sym, cur))
            self._by_sym[sym] = _Agg(bucket, dpx, dpx, dpx, dpx, qty)
            await self._push(self._emit(sym, self._by_sym[sym]))
            return
        cur.c = dpx
        if dpx > cur.h:
            cur.h = dpx
        if dpx < cur.l:
            cur.l = dpx
        cur.vol += qty
        await self._push(self._emit(sym, cur))
