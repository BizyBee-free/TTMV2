"""Facade: wires ops subsystems to the live runner (no strategy logic)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.config import OrderStatus, Settings, get_settings
from src.logger import get_logger

if TYPE_CHECKING:
    from src.order_manager import ManagedOrder, OrderRequest

from src.ops.audit_logger import AuditLogger
from src.ops.control_state import ControlStateManager
from src.ops.health_monitor import HealthMonitor
from src.ops.order_monitor import OrderMonitor
from src.ops.otp_manager import OTPManager
from src.ops.telegram_alerts import TelegramAlertClient
from src.ops.telegram_control import OpsControlBot
from src.telegram_notifier import TelegramNotifier

logger = get_logger("ops.ops_controller")


class OpsController:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        runner: Any = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._runner = runner
        self.audit = AuditLogger(settings=self._settings)
        self.alerts = TelegramAlertClient(settings=self._settings)
        self.control_state = ControlStateManager(settings=self._settings)
        self.order_monitor = OrderMonitor(
            settings=self._settings,
            audit=self.audit,
            alerts=self.alerts,
            control=self.control_state,
        )
        self.health = HealthMonitor(
            settings=self._settings,
            audit=self.audit,
            alerts=self.alerts,
            control=self.control_state,
        )
        self._notifier = TelegramNotifier(settings=self._settings)
        self._otp: Optional[OTPManager] = None
        if getattr(self._settings, "OTP_TELEGRAM_ENABLED", False):
            self._otp = OTPManager(
                settings=self._settings,
                alerts=self.alerts,
                notifier=self._notifier,
            )
        self._control_bot = OpsControlBot(
            control=self.control_state,
            audit=self.audit,
            alerts=self.alerts,
            notifier=self._notifier,
            settings=self._settings,
            runner=runner,
            otp_manager=self._otp,
        )
        self._started = False
        self._block_warned: Dict[str, float] = {}
        self._warn_cooldown_sec = 300.0

    def attach_runner(self, runner: Any) -> None:
        self._runner = runner
        self._control_bot.set_runner(runner)

    def otp_provider(self) -> Optional[str]:
        """Callback for ``BeeTradeClient`` / ``TradingTokenManager``."""
        if not self._otp:
            return None
        return self._otp.request_otp("trading_token", timeout_sec=120)

    def start(self) -> None:
        if self._started:
            return
        if not getattr(self._settings, "OPS_ENABLED", False):
            return
        self._started = True
        try:
            self.alerts.send_message("Bot started", level="INFO")
            self.audit.log_control_event({"event_type": "bot_started", "detail": "ops_start"})
        except Exception as e:
            logger.error("ops start notify failed", extra={"error": str(e)})
        if getattr(self._settings, "CONTROL_ENABLED", True) and getattr(
            self._settings,
            "TELEGRAM_ENABLED",
            False,
        ):
            try:
                self._control_bot.start_background()
            except Exception as e:
                logger.error("OpsControlBot start failed", extra={"error": str(e)})

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            self._control_bot.stop()
        except Exception as e:
            logger.error("OpsControlBot stop failed", extra={"error": str(e)})
        try:
            self.alerts.send_message("Bot stopped", level="INFO")
            self.audit.log_control_event({"event_type": "bot_stopped", "detail": "ops_stop"})
        except Exception as e:
            logger.error("ops stop notify failed", extra={"error": str(e)})

    def health_snapshot(self) -> Dict[str, Any]:
        return self.health.snapshot()

    def after_run_tick(
        self,
        *,
        bar_unix_ts: Optional[int] = None,
        rt_tick_ts: Optional[float] = None,
    ) -> None:
        """Call once per runner decision cycle (e.g. end of ``run_once``)."""
        if not getattr(self._settings, "OPS_ENABLED", False):
            return
        now = time.time()
        if bar_unix_ts is not None:
            self.health.update_bar(unix_ts=int(bar_unix_ts))
        if rt_tick_ts is not None:
            self.health.update_tick(rt_tick_ts)
        self.health.update_decision(now)
        try:
            self.order_monitor.check_timeouts(now)
            self.health.check_health(now)
        except Exception as e:
            logger.error("ops tick failed", extra={"error": str(e)})

    def on_order_blocked(self, req: "OrderRequest", reason: str) -> None:
        if not getattr(self._settings, "OPS_ENABLED", False):
            return
        try:
            self.audit.log_control_event(
                {
                    "event_type": "order_blocked",
                    "detail": reason,
                    "symbol": req.symbol,
                    "side": str(req.side),
                    "qty": req.quantity,
                },
            )
            now = time.time()
            last = self._block_warned.get(reason, 0.0)
            if now - last >= self._warn_cooldown_sec:
                self._block_warned[reason] = now
                self.alerts.send_message(
                    f"Order blocked by control: {reason} ({req.symbol} {req.side.value})",
                    level="WARNING",
                )
        except Exception as e:
            logger.error("on_order_blocked failed", extra={"error": str(e)})

    def on_order_submitted(self, order: "ManagedOrder", req: "OrderRequest") -> None:
        if not getattr(self._settings, "OPS_ENABLED", False):
            return
        try:
            ev = {
                "event_type": "order_submitted",
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "price": order.price,
                "status": order.status.value if hasattr(order.status, "value") else str(order.status),
            }
            self.audit.log_order_event(ev)
            if order.status != OrderStatus.REJECTED:
                self.alerts.send_order_event(
                    {
                        "event_type": "ORDER_SUBMITTED",
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": order.quantity,
                        "price": order.price,
                    },
                )
            self.order_monitor.register_order_submitted(
                {"order_id": order.order_id, "ts": time.time(), **ev},
            )
            self.order_monitor.handle_managed_order_update(order)
        except Exception as e:
            logger.error("on_order_submitted failed", extra={"error": str(e)})

    def on_order_update(self, order: "ManagedOrder") -> None:
        if not getattr(self._settings, "OPS_ENABLED", False):
            return
        try:
            self.order_monitor.handle_managed_order_update(order)
        except Exception as e:
            logger.error("on_order_update failed", extra={"error": str(e)})

    def on_order_terminal(self, order: "ManagedOrder") -> None:
        if not getattr(self._settings, "OPS_ENABLED", False):
            return
        try:
            st = order.status.value if hasattr(order.status, "value") else str(order.status)
            if st == "Rejected":
                self.audit.log_order_event(
                    {
                        "event_type": "order_rejected",
                        "order_id": order.order_id,
                        "reason": order.reject_reason,
                    },
                )
                self.alerts.send_order_event(
                    {
                        "event_type": "ORDER_REJECTED",
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "detail": order.reject_reason,
                    },
                )
                self.order_monitor.register_order_reject(
                    order.order_id,
                    {"reason": order.reject_reason},
                )
            elif st == "Filled":
                self.audit.log_order_event(
                    {
                        "event_type": "order_filled",
                        "order_id": order.order_id,
                        "avg": order.avg_fill_price,
                        "qty": order.filled_qty,
                    },
                )
                self.alerts.send_order_event(
                    {
                        "event_type": "ORDER_FILLED",
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": order.filled_qty,
                        "price": order.avg_fill_price,
                    },
                )
                self.order_monitor.register_order_fill(order.order_id, {})
        except Exception as e:
            logger.error("on_order_terminal failed", extra={"error": str(e)})

    def register_order_callbacks(self, order_manager: Any) -> None:
        """Register ``on_update`` so fills/rejects notify ops without duplicating fill handlers."""

        def _upd(o: "ManagedOrder") -> None:
            self.on_order_update(o)
            if getattr(o, "is_terminal", False):
                self.on_order_terminal(o)

        try:
            order_manager.on_update(_upd)
        except Exception as e:
            logger.error("register on_update failed", extra={"error": str(e)})
