"""Abstract base class for trading strategies.

Provides the lifecycle hooks that every strategy must implement,
plus wiring to MarketDataManager, OrderManager, and PositionTracker.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from src.config import OrderSide, OrderType, Settings, get_settings
from src.data_buffer import DataBuffer
from src.logger import get_logger
from src.order_manager import ManagedOrder, OrderManager, OrderRequest
from src.position_tracker import PositionTracker

logger = get_logger("strategy")


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalEvent:
    """Output of a strategy's signal generation."""
    signal: Signal
    symbol: str
    price: float = 0.0
    quantity: int = 0
    confidence: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class StrategyBase(ABC):
    """Base class for all trading strategies.

    Subclasses must implement:
        - on_tick()       -- called on every market data update
        - on_signal()     -- called when generate_signal returns BUY/SELL

    Optional overrides:
        - on_fill()       -- called when an order fills
        - on_start()      -- called once when strategy starts
        - on_stop()       -- called once when strategy stops

    Usage::

        class MyStrategy(StrategyBase):
            def on_tick(self, symbol, quote, trade):
                if some_condition:
                    return SignalEvent(Signal.BUY, symbol, price=25.0, quantity=100)
                return None

            def on_signal(self, event):
                self.place_order(event.symbol, OrderSide.BUY,
                                 OrderType.LO, event.price, event.quantity)

        strategy = MyStrategy("my_strat", symbols=["HPG"])
        strategy.start(order_mgr, position_tracker, data_buffer)
    """

    def __init__(
        self,
        name: str,
        symbols: List[str],
        settings: Optional[Settings] = None,
    ) -> None:
        self.name = name
        # DNSE WS có thể trả symbol khác hoa/thường; chuẩn hóa để quote/OHLC khớp.
        self.symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        self._settings = settings or get_settings()

        self._order_mgr: Optional[OrderManager] = None
        self._tracker: Optional[PositionTracker] = None
        self._buffer: Optional[DataBuffer] = None

        self._running = False
        self._started_at: float = 0.0
        self._tick_count: int = 0
        self._signal_count: int = 0
        self._order_count: int = 0

    # ---- Lifecycle ----

    def start(
        self,
        order_mgr: OrderManager,
        tracker: PositionTracker,
        buffer: DataBuffer,
    ) -> None:
        """Wire up dependencies and start the strategy."""
        self._order_mgr = order_mgr
        self._tracker = tracker
        self._buffer = buffer

        self._order_mgr.on_fill(self._handle_fill)

        self._running = True
        self._started_at = time.time()

        self.on_start()
        logger.info(f"Strategy started: {self.name}", extra={
            "symbols": self.symbols,
            "paper_mode": self._order_mgr.paper_mode,
        })

    def stop(self) -> None:
        """Stop the strategy."""
        self._running = False
        self.on_stop()
        logger.info(f"Strategy stopped: {self.name}", extra={
            "ticks": self._tick_count,
            "signals": self._signal_count,
            "orders": self._order_count,
            "uptime_s": round(time.time() - self._started_at, 1),
        })

    @property
    def is_running(self) -> bool:
        return self._running

    # ---- Market data handler (called by MarketDataManager callbacks) ----

    def handle_quote(self, quote) -> None:
        """Entry point for quote updates from MarketDataManager."""
        qs = (getattr(quote, "symbol", None) or "").strip().upper()
        if not self._running or qs not in self.symbols:
            return
        self._tick_count += 1
        signal = self.on_tick(qs, quote=quote, trade=None)
        if signal and signal.signal != Signal.HOLD:
            self._process_signal(signal)

    def handle_trade(self, trade) -> None:
        """Entry point for trade updates from MarketDataManager."""
        ts = (getattr(trade, "symbol", None) or "").strip().upper()
        if not self._running or ts not in self.symbols:
            return
        self._tick_count += 1
        signal = self.on_tick(ts, quote=None, trade=trade)
        if signal and signal.signal != Signal.HOLD:
            self._process_signal(signal)

    # ---- Abstract methods (implement in subclass) ----

    @abstractmethod
    def on_tick(self, symbol: str, quote=None, trade=None) -> Optional[SignalEvent]:
        """Process a market data tick. Return a SignalEvent or None."""
        ...

    @abstractmethod
    def on_signal(self, event: SignalEvent) -> None:
        """React to a generated signal (place orders, etc.)."""
        ...

    def on_fill(self, order: ManagedOrder) -> None:
        """Called when an order fills. Override for custom logic."""
        pass

    def on_start(self) -> None:
        """Called once when strategy starts. Override for init logic."""
        pass

    def on_stop(self) -> None:
        """Called once when strategy stops. Override for cleanup."""
        pass

    # ---- Helper methods for subclasses ----

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        quantity: int,
        loan_package_id: int = 0,
        market_type: str = "STOCK",
    ) -> ManagedOrder:
        """Submit an order through the OrderManager."""
        req = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            loan_package_id=loan_package_id,
            market_type=market_type,
        )
        order = self._order_mgr.submit(req)
        self._order_count += 1
        logger.info(f"[{self.name}] Order placed", extra={
            "order_id": order.order_id, "symbol": symbol,
            "side": side.value, "price": price, "qty": quantity,
            "status": order.status.value,
        })
        return order

    def get_position(self, symbol: str):
        """Get current position for a symbol."""
        return self._tracker.get_position(symbol)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[ManagedOrder]:
        """Get active orders, optionally filtered by symbol."""
        return self._order_mgr.get_active_orders(symbol)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "running": self._running,
            "symbols": self.symbols,
            "ticks": self._tick_count,
            "signals": self._signal_count,
            "orders": self._order_count,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
        }

    # ---- Internal ----

    def _process_signal(self, event: SignalEvent) -> None:
        self._signal_count += 1
        logger.info(f"[{self.name}] Signal generated", extra={
            "signal": event.signal.value, "symbol": event.symbol,
            "price": event.price, "confidence": event.confidence,
            "reason": event.reason,
        })
        self.on_signal(event)

    def _handle_fill(self, order: ManagedOrder) -> None:
        if order.symbol in self.symbols:
            self.on_fill(order)
