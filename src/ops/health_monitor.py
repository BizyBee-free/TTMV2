"""Health signals: stale data, WS, decision loop."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.config import Settings, get_settings
from src.logger import get_logger

if TYPE_CHECKING:
    from src.ops.audit_logger import AuditLogger
    from src.ops.control_state import ControlStateManager
    from src.ops.telegram_alerts import TelegramAlertClient

logger = get_logger("ops.health_monitor")


class HealthMonitor:
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
        self.last_tick_ts: Optional[float] = None
        self.last_quote_ts: Optional[float] = None
        self.last_bar_ts: Optional[float] = None
        self.last_decision_ts: Optional[float] = None
        self.websocket_connected: bool = True
        self.broker_connected: bool = True
        self._ws_disconnect_times: List[float] = []
        self._reconnect_storm_threshold = 5
        self._reconnect_storm_window_sec = 60.0
        self._health_action_debounce_sec = 120.0
        self._last_health_action: Dict[str, float] = {}

    def update_tick(self, ts: Optional[float] = None) -> None:
        self.last_tick_ts = ts if ts is not None else time.time()

    def update_quote(self, ts: Optional[float] = None) -> None:
        self.last_quote_ts = ts if ts is not None else time.time()

    def update_bar(self, unix_ts: Optional[int] = None, ts: Optional[float] = None) -> None:
        if unix_ts is not None:
            self.last_bar_ts = float(unix_ts)
        elif ts is not None:
            self.last_bar_ts = ts
        else:
            self.last_bar_ts = time.time()

    def update_decision(self, ts: Optional[float] = None) -> None:
        self.last_decision_ts = ts if ts is not None else time.time()

    def set_websocket_connected(self, ok: bool) -> None:
        if self.websocket_connected and not ok:
            self._ws_disconnect_times.append(time.time())
        self.websocket_connected = ok

    def set_broker_connected(self, ok: bool) -> None:
        self.broker_connected = ok

    def snapshot(self) -> Dict[str, Any]:
        cpu = mem = None
        try:
            import psutil  # type: ignore

            cpu = float(psutil.cpu_percent(interval=None))
            mem = float(psutil.virtual_memory().percent)
        except Exception:
            pass
        return {
            "last_tick_ts": self.last_tick_ts,
            "last_quote_ts": self.last_quote_ts,
            "last_bar_ts": self.last_bar_ts,
            "last_decision_ts": self.last_decision_ts,
            "websocket_connected": self.websocket_connected,
            "broker_connected": self.broker_connected,
            "cpu_percent": cpu,
            "memory_percent": mem,
        }

    def check_health(self, now: Optional[float] = None) -> List[Dict[str, Any]]:
        if not getattr(self._settings, "HEALTH_MONITOR_ENABLED", True):
            return []
        ts = now if now is not None else time.time()
        events: List[Dict[str, Any]] = []
        stale_data = float(getattr(self._settings, "DATA_STALE_SEC", 10))
        bar_stale = float(getattr(self._settings, "BAR_STALE_SEC", 70))
        dec_stale = float(getattr(self._settings, "DECISION_STALE_SEC", 90))

        if self.last_tick_ts and ts - self.last_tick_ts > stale_data:
            events.append(self._crit("tick_stale", f"tick stale > {stale_data}s"))
        if self.last_quote_ts and ts - self.last_quote_ts > stale_data:
            events.append(self._crit("quote_stale", f"quote stale > {stale_data}s"))
        if self.last_bar_ts and ts - self.last_bar_ts > bar_stale:
            events.append(self._crit("bar_stale", f"bar stale > {bar_stale}s"))
        if self.last_decision_ts and ts - self.last_decision_ts > dec_stale:
            events.append(self._crit("decision_stale", f"decision loop stale > {dec_stale}s"))

        cutoff = ts - self._reconnect_storm_window_sec
        recent = [x for x in self._ws_disconnect_times if x >= cutoff]
        self._ws_disconnect_times = recent
        if len(recent) >= self._reconnect_storm_threshold:
            events.append(self._crit("ws_reconnect_storm", f"{len(recent)} disconnects in {int(self._reconnect_storm_window_sec)}s"))

        if not self.broker_connected:
            events.append(self._crit("broker_disconnected", "broker flag disconnected"))

        acted: List[Dict[str, Any]] = []
        for ev in events:
            key = str(ev.get("event_type") or "health")
            last = self._last_health_action.get(key, 0.0)
            if ts - last < self._health_action_debounce_sec:
                continue
            self._last_health_action[key] = ts
            acted.append(ev)
            if self._audit:
                self._audit.log_health_event(ev)
            if self._alerts:
                self._alerts.send_health_event(ev)
            if self._control and getattr(self._settings, "PAUSE_ON_DATA_STALE", True):
                self._control.set_pause(f"health:{key}", updated_by="health_monitor")
        return acted

    def _crit(self, kind: str, detail: str) -> Dict[str, Any]:
        return {
            "event_type": kind,
            "severity": "CRITICAL",
            "detail": detail,
        }
