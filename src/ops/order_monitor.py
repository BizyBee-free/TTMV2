"""Order lifecycle monitoring: ack/fill timeouts and rejections."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from src.config import OrderStatus, Settings, get_settings
from src.logger import get_logger

if TYPE_CHECKING:
    from src.ops.audit_logger import AuditLogger
    from src.ops.control_state import ControlStateManager
    from src.ops.telegram_alerts import TelegramAlertClient

logger = get_logger("ops.order_monitor")


@dataclass
class TrackedOrder:
    internal_id: str
    order_id: str
    submitted_at: float
    ack_at: Optional[float] = None
    filled_at: Optional[float] = None
    status: str = "submitted"
    meta: Dict[str, Any] = field(default_factory=dict)


class OrderMonitor:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        audit: Optional["AuditLogger"] = None,
        alerts: Optional["TelegramAlertClient"] = None,
        control: Optional["ControlStateManager"] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._audit = audit
        self._alerts = alerts
        self._control = control
        self._by_order_id: Dict[str, TrackedOrder] = {}
        self._by_internal: Dict[str, TrackedOrder] = {}

    def register_order_submitted(self, order_event: Dict[str, Any]) -> str:
        internal = str(order_event.get("internal_id") or uuid.uuid4().hex[:12])
        oid = str(order_event.get("order_id") or "pending")
        now = float(order_event.get("ts") or time.time())
        t = TrackedOrder(
            internal_id=internal,
            order_id=oid,
            submitted_at=now,
            meta=dict(order_event),
        )
        self._by_internal[internal] = t
        self._by_order_id[oid] = t
        return internal

    def register_order_ack(self, order_id: str, ack_event: Dict[str, Any]) -> None:
        t = self._by_order_id.get(order_id) or self._by_order_id.get(str(ack_event.get("prev_id", "")))
        if not t:
            return
        t.ack_at = float(ack_event.get("ts") or time.time())
        t.order_id = str(order_id or t.order_id)
        self._by_order_id[t.order_id] = t
        t.status = str(ack_event.get("status") or "acked")

    def register_order_fill(self, order_id: str, fill_event: Dict[str, Any]) -> None:
        t = self._by_order_id.get(order_id)
        if not t:
            return
        t.filled_at = float(fill_event.get("ts") or time.time())
        t.status = "filled"

    def register_order_reject(self, order_id: str, reject_event: Dict[str, Any]) -> None:
        t = self._by_order_id.get(order_id)
        if not t:
            t = TrackedOrder(
                internal_id="adhoc",
                order_id=order_id,
                submitted_at=float(reject_event.get("ts") or time.time()),
            )
            self._by_order_id[order_id] = t
        t.status = "rejected"
        self._emit_critical(
            "order_rejected",
            {"order_id": order_id, **reject_event},
        )

    def register_broker_disconnected(self, detail: str = "") -> None:
        self._emit_critical("broker_disconnected", {"detail": detail})

    def register_cancel_failed(self, order_id: str, detail: str = "") -> None:
        self._emit_critical("order_cancel_failed", {"order_id": order_id, "detail": detail})

    def register_position_mismatch(self, detail: str) -> None:
        self._emit_critical("local_broker_position_mismatch", {"detail": detail})

    def _emit_critical(self, kind: str, payload: Dict[str, Any]) -> None:
        if self._audit:
            self._audit.log_risk_event(
                {"event_type": kind, "severity": "CRITICAL", **payload},
            )
        if self._alerts:
            self._alerts.send_critical(f"{kind}: {payload}")
        if self._control and getattr(self._settings, "AUTO_FLATTEN_ON_CRITICAL", False):
            try:
                self._control.request_flatten(updated_by="auto_flatten_critical")
            except Exception:
                pass
        if (
            self._control
            and getattr(self._settings, "PAUSE_ON_ORDER_TIMEOUT", True)
            and kind
            in (
                "order_ack_timeout",
                "order_fill_timeout",
                "order_rejected",
                "broker_disconnected",
                "local_broker_position_mismatch",
            )
        ):
            self._control.set_pause(f"order_monitor:{kind}", updated_by="order_monitor")

    def check_timeouts(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        if not getattr(self._settings, "ORDER_MONITOR_ENABLED", True):
            return []
        ts = now if now is not None else time.time()
        ack_sec = max(1, int(getattr(self._settings, "ORDER_ACK_TIMEOUT_SEC", 5)))
        fill_sec = max(1, int(getattr(self._settings, "ORDER_FILL_TIMEOUT_SEC", 10)))
        out: List[Dict[str, Any]] = []
        for t in list(self._by_order_id.values()):
            if t.status in ("filled", "rejected", "canceled"):
                continue
            if t.ack_at is None and ts - t.submitted_at > ack_sec:
                ev = {
                    "event_type": "order_ack_timeout",
                    "order_id": t.order_id,
                    "submitted_at": t.submitted_at,
                    "severity": "CRITICAL",
                }
                out.append(ev)
                t.status = "canceled"
                self._emit_critical("order_ack_timeout", ev)
            elif t.ack_at is not None and t.filled_at is None and ts - t.ack_at > fill_sec:
                ev = {
                    "event_type": "order_fill_timeout",
                    "order_id": t.order_id,
                    "ack_at": t.ack_at,
                    "severity": "CRITICAL",
                }
                out.append(ev)
                t.status = "canceled"
                self._emit_critical("order_fill_timeout", ev)
        return out

    def handle_managed_order_update(self, order: Any) -> None:
        """Hook for OrderManager ManagedOrder updates."""
        try:
            oid = str(getattr(order, "order_id", "") or "")
            st = getattr(order, "status", None)
            if hasattr(st, "value"):
                st_val = st.value
            else:
                st_val = str(st or "")
            ts = time.time()
            if st_val in (OrderStatus.PENDING_NEW.value, OrderStatus.NEW.value):
                self.register_order_ack(oid, {"ts": ts, "status": st_val})
            if st_val == OrderStatus.PARTIALLY_FILLED.value:
                self.register_order_fill(oid, {"ts": ts})
            # Terminal REJECTED / FILLED handled in OpsController.on_order_terminal
        except Exception as e:
            logger.error("handle_managed_order_update error", extra={"error": str(e)})
