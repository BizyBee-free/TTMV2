"""Paper trading engine: auto-fill paper orders against live quotes.

Wired between MarketDataManager and OrderManager:

    MarketDataManager.on_quote -> PaperEngine.on_quote
        -> scans active paper orders
        -> calls OrderManager.paper_fill() when price condition met

Fill logic (Level-1 order book matching):
    LO BUY  : fills when quote.best_ask <= order.price
    LO SELL : fills when quote.best_bid >= order.price
    ATO/MTL : fills at current mid-price (market-like for paper)
    ATC     : queued until session close trigger, then fills at close

Slippage:
    Optional tick_slippage parameter shifts the fill price by N ticks
    in the direction adverse to the trader (more realistic simulation).
    Default = 0 ticks (clean fills).

Session management:
    start()  -> enable auto-fill
    stop()   -> disable auto-fill, log session summary
    reset()  -> clear all counters (for a new session)

Usage::

    engine = PaperEngine(order_mgr, tick_size=0.1, slippage_ticks=0)
    market_data_mgr.on_quote(engine.on_quote)
    engine.start()
    # ... trading ...
    report = engine.get_session_report()
    engine.stop()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.config import OrderType, Settings, get_settings
from src.logger import get_logger
from src.order_manager import ManagedOrder, OrderManager

logger = get_logger("paper_engine")


@dataclass
class FillRecord:
    """A record of a single paper fill event."""
    order_id: str
    symbol: str
    fill_price: float
    fill_qty: int
    side: str
    filled_at: float = field(default_factory=time.time)


@dataclass
class SessionReport:
    """Summary statistics for a completed paper session."""
    fills: int
    total_volume: int
    symbols_traded: List[str]
    elapsed_s: float
    quote_ticks_processed: int
    orders_checked: int


class PaperEngine:
    """Auto-fills paper orders against live market quotes.

    Args:
        order_mgr:       The OrderManager to call paper_fill() on.
        tick_size:       Minimum price increment for the instrument.
                         Used for slippage calculation.
        slippage_ticks:  Number of ticks of adverse slippage to apply
                         on each fill (default 0 = clean fills).
        settings:        Application settings (optional).
    """

    def __init__(
        self,
        order_mgr: OrderManager,
        tick_size: float = 0.1,
        slippage_ticks: int = 0,
        settings: Optional[Settings] = None,
    ) -> None:
        self._order_mgr = order_mgr
        self._tick_size = tick_size
        self._slippage_ticks = slippage_ticks
        self._settings = settings or get_settings()

        self._running = False
        self._started_at: float = 0.0

        # Session counters
        self._fills: List[FillRecord] = []
        self._quote_ticks: int = 0
        self._orders_checked: int = 0

    # ------------------------------------------------------------------ #
    # Session lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Enable auto-fill. Call once before quotes start flowing."""
        self._running = True
        self._started_at = time.time()
        logger.info("PaperEngine started", extra={
            "tick_size": self._tick_size, "slippage_ticks": self._slippage_ticks,
        })

    def stop(self) -> None:
        """Disable auto-fill and log session summary."""
        self._running = False
        report = self.get_session_report()
        logger.info("PaperEngine stopped", extra={
            "fills": report.fills,
            "total_volume": report.total_volume,
            "elapsed_s": round(report.elapsed_s, 1),
            "quote_ticks": report.quote_ticks_processed,
        })

    def reset(self) -> None:
        """Clear session counters (call at start of a new trading session)."""
        self._fills.clear()
        self._quote_ticks = 0
        self._orders_checked = 0
        self._started_at = time.time()
        logger.info("PaperEngine reset")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ #
    # Core fill logic -- called by MarketDataManager on every quote
    # ------------------------------------------------------------------ #

    def on_quote(self, quote) -> None:
        """Process an incoming quote and attempt to fill pending paper orders.

        Args:
            quote: A Quote object from the DNSE SDK (best_bid, best_ask, symbol).
        """
        if not self._running:
            return

        self._quote_ticks += 1
        symbol = quote.symbol

        active_orders = self._order_mgr.get_active_orders(symbol)
        if not active_orders:
            return

        for order in active_orders:
            if not order.is_paper:
                continue
            self._orders_checked += 1
            fill_price = self._check_fill(order, quote)
            if fill_price is not None:
                self._execute_fill(order, fill_price)

    def on_session_close(self, close_price: float, symbol: str) -> None:
        """Force-fill any remaining ATC paper orders at session close.

        Args:
            close_price: Official session closing price.
            symbol:      Symbol to close orders for.
        """
        active_orders = self._order_mgr.get_active_orders(symbol)
        for order in active_orders:
            if not order.is_paper:
                continue
            if order.order_type in (OrderType.ATC,):
                fill_price = close_price + self._slippage(order)
                self._execute_fill(order, fill_price)
                logger.info("ATC paper order filled at session close", extra={
                    "order_id": order.order_id, "close_price": close_price,
                })

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #

    def get_session_report(self) -> SessionReport:
        elapsed = time.time() - self._started_at if self._started_at > 0 else 0.0
        symbols = list({r.symbol for r in self._fills})
        total_vol = sum(r.fill_qty for r in self._fills)
        return SessionReport(
            fills=len(self._fills),
            total_volume=total_vol,
            symbols_traded=symbols,
            elapsed_s=round(elapsed, 1),
            quote_ticks_processed=self._quote_ticks,
            orders_checked=self._orders_checked,
        )

    @property
    def fill_history(self) -> List[FillRecord]:
        return list(self._fills)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _check_fill(self, order: ManagedOrder, quote) -> Optional[float]:
        """Determine if an order should fill, and return the fill price.

        Returns:
            Fill price if order should fill, else None.
        """
        order_type = order.order_type

        # Market-like orders: fill immediately at current market price
        if order_type in (OrderType.ATO, OrderType.MTL, OrderType.MOK, OrderType.MAK):
            mid = self._safe_mid(quote)
            return mid + self._slippage(order) if mid > 0 else None

        # Limit orders: price-conditional fill
        if order_type == OrderType.LO:
            from src.config import OrderSide
            if order.side == OrderSide.BUY:
                # Buyer wants ask_price <= order.price
                ask = getattr(quote, "best_ask", 0.0) or 0.0
                if 0 < ask <= order.price:
                    return ask + self._slippage(order)
            else:
                # Seller wants bid_price >= order.price
                bid = getattr(quote, "best_bid", 0.0) or 0.0
                if bid > 0 and bid >= order.price:
                    return bid + self._slippage(order)

        # ATC handled separately in on_session_close
        return None

    def _execute_fill(self, order: ManagedOrder, fill_price: float) -> None:
        """Call OrderManager.paper_fill() and record the result."""
        fill_price = max(0.0, fill_price)
        success = self._order_mgr.paper_fill(order.order_id, fill_price)
        if success:
            record = FillRecord(
                order_id=order.order_id,
                symbol=order.symbol,
                fill_price=fill_price,
                fill_qty=order.filled_qty,
                side=order.side.value,
            )
            self._fills.append(record)
            logger.info("PaperEngine fill", extra={
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "fill_price": fill_price,
                "qty": order.filled_qty,
                "status": order.status.value,
            })

    def _slippage(self, order: ManagedOrder) -> float:
        """Return signed slippage amount (adverse to trader)."""
        if self._slippage_ticks == 0:
            return 0.0
        from src.config import OrderSide
        # BUY gets filled slightly higher (more expensive)
        # SELL gets filled slightly lower (less profitable)
        direction = 1 if order.side == OrderSide.BUY else -1
        return direction * self._slippage_ticks * self._tick_size

    def _safe_mid(self, quote) -> float:
        """Return mid-price from quote, or 0 if unavailable."""
        bid = getattr(quote, "best_bid", 0.0) or 0.0
        ask = getattr(quote, "best_ask", 0.0) or 0.0
        if bid > 0 and ask > 0:
            return (bid + ask) / 2.0
        if bid > 0:
            return bid
        if ask > 0:
            return ask
        return getattr(quote, "last_price", 0.0) or 0.0
