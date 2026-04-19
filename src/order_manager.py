"""Order lifecycle management with paper-mode fill simulation.

Handles order submission, state tracking, and fill events.
In paper mode, orders are filled against live market data from DataBuffer
instead of being sent to DNSE.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.config import OrderSide, OrderStatus, OrderType, Settings, get_settings
from src.dnse_client import BeeTradeClient
from src.logger import get_logger
from src.utils import validate_lot_size

logger = get_logger("order_manager")


class OrderRejectReason(str, Enum):
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    RATE_LIMITED = "RATE_LIMITED"
    API_ERROR = "API_ERROR"
    PAPER_REJECTED = "PAPER_REJECTED"


@dataclass
class OrderRequest:
    """Intent to place an order -- validated before submission."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    loan_package_id: int = 0
    market_type: str = "STOCK"


@dataclass
class ManagedOrder:
    """An order tracked through its full lifecycle."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    market_type: str

    status: OrderStatus = OrderStatus.PENDING
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    reject_reason: str = ""

    is_paper: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    api_response: Optional[Dict[str, Any]] = None

    @property
    def remaining_qty(self) -> int:
        return self.quantity - self.filled_qty

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.DONE_FOR_DAY,
        )

    @property
    def is_active(self) -> bool:
        return not self.is_terminal

    @property
    def fill_value(self) -> float:
        return self.avg_fill_price * self.filled_qty


OrderCallback = Callable[[ManagedOrder], Any]


class OrderManager:
    """Manages order submission, tracking, and paper-mode simulation.

    Usage::

        mgr = OrderManager(client, paper_mode=True)
        mgr.on_fill(my_fill_handler)

        order = mgr.submit(OrderRequest(
            symbol="HPG", side=OrderSide.BUY,
            order_type=OrderType.LO, price=25.0, quantity=100,
        ))

        # Paper mode: simulate fill
        mgr.paper_fill(order.order_id, fill_price=25.0)

        # Check state
        print(order.status, order.filled_qty)
    """

    def __init__(
        self,
        client: Optional[BeeTradeClient] = None,
        settings: Optional[Settings] = None,
        paper_mode: Optional[bool] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._paper_mode = paper_mode if paper_mode is not None else self._settings.PAPER_MODE

        self._orders: Dict[str, ManagedOrder] = {}
        self._active_orders: Dict[str, ManagedOrder] = {}

        self._fill_callbacks: List[OrderCallback] = []
        self._update_callbacks: List[OrderCallback] = []

        self._total_submitted: int = 0
        self._total_filled: int = 0
        self._total_rejected: int = 0

    @property
    def paper_mode(self) -> bool:
        return self._paper_mode

    # ---- Callback registration ----

    def on_fill(self, callback: OrderCallback) -> None:
        self._fill_callbacks.append(callback)

    def on_update(self, callback: OrderCallback) -> None:
        self._update_callbacks.append(callback)

    # ---- Order submission ----

    def submit(self, req: OrderRequest) -> ManagedOrder:
        """Validate and submit an order. Returns the ManagedOrder for tracking."""
        # Pre-trade validation
        if not validate_lot_size(req.quantity, req.market_type):
            return self._reject_order(req, OrderRejectReason.INVALID_QUANTITY,
                                       f"Invalid lot size: {req.quantity} for {req.market_type}")

        if req.price <= 0 and req.order_type == OrderType.LO:
            return self._reject_order(req, OrderRejectReason.INVALID_PRICE,
                                       "LO order requires price > 0")

        if self._paper_mode:
            return self._submit_paper(req)
        else:
            return self._submit_live(req)

    def cancel(self, order_id: str) -> bool:
        """Cancel an active order. Returns True if cancellation was sent."""
        order = self._orders.get(order_id)
        if not order or order.is_terminal:
            logger.warning("Cannot cancel", extra={"order_id": order_id, "reason": "not found or terminal"})
            return False

        if self._paper_mode:
            self._update_status(order, OrderStatus.CANCELED)
            logger.info("Paper order canceled", extra={"order_id": order_id})
            return True

        if self._client is None:
            logger.error("No client for live cancel")
            return False

        try:
            result = self._client.cancel_order(order_id, order.market_type)
            if result["status"] == 200:
                self._update_status(order, OrderStatus.PENDING_CANCEL)
                return True
            else:
                logger.warning("Cancel API error", extra={"order_id": order_id, "result": result})
                return False
        except Exception as e:
            logger.error("Cancel exception", extra={"order_id": order_id, "error": str(e)})
            return False

    def modify(self, order_id: str, price: Optional[float] = None,
               quantity: Optional[int] = None) -> bool:
        """Modify an active order's price and/or quantity."""
        order = self._orders.get(order_id)
        if not order or order.is_terminal:
            return False

        if self._paper_mode:
            if price is not None:
                order.price = price
            if quantity is not None:
                order.quantity = quantity
            order.updated_at = time.time()
            self._update_status(order, OrderStatus.NEW)
            logger.info("Paper order modified", extra={"order_id": order_id, "price": price, "qty": quantity})
            return True

        if self._client is None:
            return False

        try:
            result = self._client.modify_order(order_id, price=price, quantity=quantity,
                                                market_type=order.market_type)
            if result["status"] == 200:
                self._update_status(order, OrderStatus.PENDING_REPLACE)
                return True
            return False
        except Exception as e:
            logger.error("Modify exception", extra={"order_id": order_id, "error": str(e)})
            return False

    # ---- Paper mode fill simulation ----

    def paper_fill(self, order_id: str, fill_price: float,
                   fill_qty: Optional[int] = None) -> bool:
        """Simulate a fill for a paper order.

        If fill_qty is None, fills the entire remaining quantity.
        Supports partial fills by specifying fill_qty < remaining.
        """
        order = self._orders.get(order_id)
        if not order or not order.is_paper or order.is_terminal:
            return False

        qty = fill_qty if fill_qty is not None else order.remaining_qty
        qty = min(qty, order.remaining_qty)
        if qty <= 0:
            return False

        total_cost = order.avg_fill_price * order.filled_qty + fill_price * qty
        order.filled_qty += qty
        order.avg_fill_price = total_cost / order.filled_qty
        order.updated_at = time.time()

        if order.remaining_qty == 0:
            self._update_status(order, OrderStatus.FILLED)
            self._total_filled += 1
        else:
            self._update_status(order, OrderStatus.PARTIALLY_FILLED)

        logger.info("Paper fill", extra={
            "order_id": order_id, "fill_price": fill_price,
            "fill_qty": qty, "total_filled": order.filled_qty,
            "status": order.status.value,
        })

        for cb in self._fill_callbacks:
            try:
                cb(order)
            except Exception as e:
                logger.error("Fill callback error", extra={"error": str(e)})

        return True

    # ---- Query ----

    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        return self._orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[ManagedOrder]:
        orders = list(self._active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    def get_all_orders(self) -> List[ManagedOrder]:
        return list(self._orders.values())

    def get_orders_by_status(self, status: OrderStatus) -> List[ManagedOrder]:
        return [o for o in self._orders.values() if o.status == status]

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_submitted": self._total_submitted,
            "total_filled": self._total_filled,
            "total_rejected": self._total_rejected,
            "active": len(self._active_orders),
            "all": len(self._orders),
        }

    # ---- Internal ----

    def _submit_paper(self, req: OrderRequest) -> ManagedOrder:
        order = ManagedOrder(
            order_id=f"PAPER-{uuid.uuid4().hex[:8].upper()}",
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            price=req.price,
            quantity=req.quantity,
            market_type=req.market_type,
            status=OrderStatus.NEW,
            is_paper=True,
        )
        self._register(order)
        self._total_submitted += 1
        logger.info("Paper order submitted", extra={
            "order_id": order.order_id, "symbol": req.symbol,
            "side": req.side.value, "price": req.price, "qty": req.quantity,
        })
        return order

    def _submit_live(self, req: OrderRequest) -> ManagedOrder:
        if self._client is None:
            return self._reject_order(req, OrderRejectReason.API_ERROR, "No BeeTradeClient configured")

        order = ManagedOrder(
            order_id="pending",
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            price=req.price,
            quantity=req.quantity,
            market_type=req.market_type,
            status=OrderStatus.PENDING,
            is_paper=False,
        )

        try:
            result = self._client.place_order(
                symbol=req.symbol,
                side=req.side.value,
                order_type=req.order_type.value,
                price=req.price,
                quantity=req.quantity,
                loan_package_id=req.loan_package_id,
                market_type=req.market_type,
            )
            order.api_response = result

            if result["status"] == 200:
                data = result["data"]
                if isinstance(data, dict):
                    order.order_id = str(data.get("orderId", data.get("id", order.order_id)))
                order.status = OrderStatus.PENDING_NEW
                self._register(order)
                self._total_submitted += 1
                logger.info("Live order submitted", extra={
                    "order_id": order.order_id, "symbol": req.symbol,
                    "api_status": result["status"],
                })
            else:
                error_code = ""
                error_msg = ""
                if isinstance(result["data"], dict):
                    error_code = result["data"].get("code", "")
                    error_msg = (
                        result["data"].get("message")
                        or result["data"].get("msg")
                        or result["data"].get("detail")
                        or ""
                    )
                order.status = OrderStatus.REJECTED
                order.reject_reason = f"API {result['status']}: {error_code} {error_msg}".strip()
                self._register(order)
                self._total_rejected += 1
                logger.warning("Live order rejected by API", extra={
                    "order_id": order.order_id, "status": result["status"],
                    "error_code": error_code,
                    "error_message": error_msg,
                    "api_data": result.get("data"),
                })

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.reject_reason = str(e)
            self._register(order)
            self._total_rejected += 1
            logger.error("Live order exception", extra={"error": str(e)})

        return order

    def _reject_order(self, req: OrderRequest, reason: OrderRejectReason, detail: str) -> ManagedOrder:
        order = ManagedOrder(
            order_id=f"REJ-{uuid.uuid4().hex[:8].upper()}",
            symbol=req.symbol,
            side=req.side,
            order_type=req.order_type,
            price=req.price,
            quantity=req.quantity,
            market_type=req.market_type,
            status=OrderStatus.REJECTED,
            reject_reason=f"{reason.value}: {detail}",
            is_paper=self._paper_mode,
        )
        self._orders[order.order_id] = order
        self._total_rejected += 1
        logger.warning("Order rejected pre-trade", extra={
            "order_id": order.order_id, "reason": reason.value, "detail": detail,
        })
        return order

    def _register(self, order: ManagedOrder) -> None:
        self._orders[order.order_id] = order
        if order.is_active:
            self._active_orders[order.order_id] = order

    def _update_status(self, order: ManagedOrder, new_status: OrderStatus) -> None:
        order.status = new_status
        order.updated_at = time.time()

        if order.is_terminal and order.order_id in self._active_orders:
            del self._active_orders[order.order_id]

        for cb in self._update_callbacks:
            try:
                cb(order)
            except Exception as e:
                logger.error("Update callback error", extra={"error": str(e)})
