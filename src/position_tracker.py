"""Position tracking with P&L calculation.

Tracks open positions per symbol, calculates unrealized P&L from
live market quotes, and accumulates realized P&L from fills.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import OrderSide
from src.logger import get_logger
from src.order_manager import ManagedOrder

logger = get_logger("position_tracker")


@dataclass
class Position:
    """A single position in one symbol."""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    updated_at: float = field(default_factory=time.time)

    @property
    def cost_basis(self) -> float:
        return self.avg_price * abs(self.quantity)

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def unrealized_pnl(self, market_price: float) -> float:
        if self.quantity == 0:
            return 0.0
        return (market_price - self.avg_price) * self.quantity

    def unrealized_pnl_pct(self, market_price: float) -> float:
        if self.avg_price == 0 or self.quantity == 0:
            return 0.0
        return ((market_price - self.avg_price) / self.avg_price) * 100.0

    def market_value(self, market_price: float) -> float:
        return market_price * abs(self.quantity)


@dataclass
class DailyPnL:
    """Aggregated P&L for the trading day."""
    realized: float = 0.0
    commission: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    started_at: float = field(default_factory=time.time)

    @property
    def net(self) -> float:
        return self.realized - self.commission

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return (self.win_count / self.trade_count) * 100.0


class PositionTracker:
    """Tracks positions and P&L across all symbols.

    Call on_fill() whenever an order is filled (partial or full).
    Query positions and P&L at any time.

    Usage::

        tracker = PositionTracker()
        order_mgr.on_fill(tracker.on_fill)

        pos = tracker.get_position("HPG")
        pnl = pos.unrealized_pnl(current_price)
        daily = tracker.daily_pnl
    """

    def __init__(self) -> None:
        self._positions: Dict[str, Position] = {}
        self._daily_pnl = DailyPnL()
        self._trade_log: List[Dict] = []

    def on_fill(self, order: ManagedOrder) -> None:
        """Process a fill event from OrderManager.

        Handles both partial and full fills. Updates position and
        calculates realized P&L when reducing a position.
        """
        if order.filled_qty <= 0:
            return

        pos = self._get_or_create(order.symbol)
        fill_qty = order.filled_qty
        fill_price = order.avg_fill_price

        prev_qty = pos.quantity

        if order.side == OrderSide.BUY:
            self._apply_buy(pos, fill_qty, fill_price)
        else:
            self._apply_sell(pos, fill_qty, fill_price)

        pos.updated_at = time.time()

        self._trade_log.append({
            "time": time.time(),
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "qty": fill_qty,
            "price": fill_price,
            "prev_qty": prev_qty,
            "new_qty": pos.quantity,
        })

        logger.info("Position updated", extra={
            "symbol": order.symbol,
            "side": order.side.value,
            "fill_qty": fill_qty,
            "fill_price": fill_price,
            "position_qty": pos.quantity,
            "avg_price": round(pos.avg_price, 2),
            "realized_pnl": round(pos.realized_pnl, 2),
        })

    def get_position(self, symbol: str) -> Position:
        return self._get_or_create(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    def get_open_positions(self) -> Dict[str, Position]:
        return {s: p for s, p in self._positions.items() if not p.is_flat}

    @property
    def daily_pnl(self) -> DailyPnL:
        return self._daily_pnl

    @property
    def trade_log(self) -> List[Dict]:
        return list(self._trade_log)

    def total_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate total unrealized P&L given current market prices."""
        total = 0.0
        for symbol, pos in self._positions.items():
            if not pos.is_flat and symbol in prices:
                total += pos.unrealized_pnl(prices[symbol])
        return total

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of trading day)."""
        self._daily_pnl = DailyPnL()
        self._trade_log.clear()
        for pos in self._positions.values():
            pos.realized_pnl = 0.0
        logger.info("Daily P&L reset")

    # ---- Internal ----

    def _get_or_create(self, symbol: str) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol=symbol)
        return self._positions[symbol]

    def _apply_buy(self, pos: Position, qty: int, price: float) -> None:
        if pos.quantity >= 0:
            # Adding to long or opening new long
            total_cost = pos.avg_price * pos.quantity + price * qty
            pos.quantity += qty
            pos.avg_price = total_cost / pos.quantity if pos.quantity > 0 else 0.0
        else:
            # Covering short position
            cover_qty = min(qty, abs(pos.quantity))
            realized = (pos.avg_price - price) * cover_qty
            pos.realized_pnl += realized
            self._record_realized(realized)

            remaining = qty - cover_qty
            pos.quantity += qty
            if pos.quantity > 0 and remaining > 0:
                pos.avg_price = price
            elif pos.quantity == 0:
                pos.avg_price = 0.0

    def _apply_sell(self, pos: Position, qty: int, price: float) -> None:
        if pos.quantity <= 0:
            # Adding to short or opening new short
            total_cost = pos.avg_price * abs(pos.quantity) + price * qty
            pos.quantity -= qty
            pos.avg_price = total_cost / abs(pos.quantity) if pos.quantity != 0 else 0.0
        else:
            # Selling long position
            sell_qty = min(qty, pos.quantity)
            realized = (price - pos.avg_price) * sell_qty
            pos.realized_pnl += realized
            self._record_realized(realized)

            remaining = qty - sell_qty
            pos.quantity -= qty
            if pos.quantity < 0 and remaining > 0:
                pos.avg_price = price
            elif pos.quantity == 0:
                pos.avg_price = 0.0

    def _record_realized(self, pnl: float) -> None:
        self._daily_pnl.realized += pnl
        self._daily_pnl.trade_count += 1
        if pnl > 0:
            self._daily_pnl.win_count += 1
        elif pnl < 0:
            self._daily_pnl.loss_count += 1
