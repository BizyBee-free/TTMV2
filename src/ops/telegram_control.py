"""Telegram inbound control commands (long-polling) — ops layer."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING, Any, Dict, Optional

from src.config import Settings, get_settings
from src.logger import get_logger
from src.ops.audit_logger import AuditLogger
from src.ops.control_state import ControlStateManager
from src.ops.otp_manager import OTPManager
from src.ops.telegram_alerts import TelegramAlertClient
from src.telegram_notifier import TelegramNotifier

if TYPE_CHECKING:
    pass

logger = get_logger("ops.telegram_control")

_OTP_RE = re.compile(r"^OTP\s*(\d{4,10})\s*$", re.IGNORECASE)


class OpsControlBot:
    """Background thread: Telegram commands for ops (non-blocking for trading loop)."""

    def __init__(
        self,
        *,
        control: ControlStateManager,
        audit: AuditLogger,
        alerts: TelegramAlertClient,
        notifier: TelegramNotifier,
        settings: Optional[Settings] = None,
        runner: Any = None,
        otp_manager: Optional[OTPManager] = None,
        poll_interval_sec: float = 2.0,
        confirm_window_sec: float = 30.0,
    ) -> None:
        self._control = control
        self._audit = audit
        self._alerts = alerts
        self._notifier = notifier
        self._settings = settings or get_settings()
        self._runner = runner
        self._otp_manager = otp_manager
        self._poll_interval = poll_interval_sec
        self._confirm_window = confirm_window_sec
        self._offset: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pending_flatten_until: float = 0.0
        self._pending_kill_until: float = 0.0

    def set_runner(self, runner: Any) -> None:
        self._runner = runner

    def start_background(self) -> None:
        if not getattr(self._settings, "CONTROL_ENABLED", True):
            logger.info("OpsControlBot: CONTROL_ENABLED=false, skip")
            return
        if not (self._settings.TELEGRAM_BOT_TOKEN or "").strip():
            logger.info("OpsControlBot: no bot token, skip")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="OpsTelegramControl", daemon=True)
        self._thread.start()
        logger.info("OpsControlBot started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8.0)

    def _run_loop(self) -> None:
        token = self._settings.TELEGRAM_BOT_TOKEN
        base = f"https://api.telegram.org/bot{token}/getUpdates"
        while not self._stop.is_set():
            try:
                params: Dict[str, Any] = {"timeout": 25}
                if self._offset is not None:
                    params["offset"] = self._offset
                url = base + "?" + urllib.parse.urlencode({k: str(v) for k, v in params.items()})
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=35) as resp:
                    raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                if not data.get("ok"):
                    logger.warning("getUpdates not ok", extra={"data": data})
                    time.sleep(self._poll_interval)
                    continue
                for upd in data.get("result", []):
                    self._offset = int(upd["update_id"]) + 1
                    msg = upd.get("message") or {}
                    chat = msg.get("chat") or {}
                    cid = str(chat.get("id", ""))
                    text = (msg.get("text") or "").strip()
                    if not cid or not text:
                        continue
                    if not self._notifier.is_chat_allowed(cid):
                        logger.warning("Ops command from disallowed chat", extra={"chat_id": cid})
                        continue
                    if self._otp_manager and self._try_otp_message(text, cid):
                        continue
                    self._handle_command(cid, text)
            except urllib.error.URLError as e:
                logger.warning("getUpdates URLError", extra={"error": str(e)})
                time.sleep(self._poll_interval)
            except Exception as e:
                logger.error("getUpdates loop error", extra={"error": str(e)})
                time.sleep(self._poll_interval)

    def _try_otp_message(self, text: str, cid: str) -> bool:
        if not self._otp_manager:
            return False
        m = _OTP_RE.match(text.strip())
        if not m:
            return False
        ok = self._otp_manager.submit_otp_from_message(text, cid)
        if ok:
            self._audit.log_control_event(
                {"event_type": "otp_received", "detail": "masked_otp", "chat_id": cid},
            )
        return True

    def _reply(self, chat_id: str, msg: str) -> None:
        try:
            self._notifier.send_message(msg, chat_id=chat_id, timeout_sec=5.0)
        except Exception as e:
            logger.error("OpsControl reply failed", extra={"error": str(e)})

    def _handle_command(self, chat_id: str, text: str) -> None:
        parts = text.split()
        cmd = (parts[0] or "").strip()
        now = time.time()

        if cmd.startswith("/"):
            cmd_l = cmd.lower()
        else:
            return

        # Two-step confirmations
        if cmd_l == "/flatten":
            if len(parts) >= 2 and parts[1].upper() == "CONFIRM":
                if now > self._pending_flatten_until:
                    self._reply(chat_id, "No pending /flatten request (expired). Send /flatten first.")
                    return
                self._pending_flatten_until = 0.0
                st = self._control.request_flatten(updated_by=f"telegram:{chat_id}")
                self._audit.log_control_event(
                    {
                        "event_type": "flatten_confirmed",
                        "detail": "force_flatten_requested",
                        "state": self._control.to_dict(),
                    },
                )
                self._alerts.send_message("Flatten requested: new entries disabled.", level="WARNING")
                self._reply(chat_id, "Flatten flag set. New entries disabled; close positions per strategy.")
                return
            self._pending_flatten_until = now + self._confirm_window
            self._reply(
                chat_id,
                f"Send `/flatten CONFIRM` within {int(self._confirm_window)} seconds to arm flatten.",
            )
            return

        if cmd_l == "/kill":
            if len(parts) >= 2 and parts[1].upper() == "CONFIRM":
                if now > self._pending_kill_until:
                    self._reply(chat_id, "No pending /kill request (expired). Send /kill first.")
                    return
                self._pending_kill_until = 0.0
                st = self._control.kill(updated_by=f"telegram:{chat_id}")
                self._audit.log_control_event(
                    {"event_type": "kill_confirmed", "detail": "killed", "state": self._control.to_dict()},
                )
                self._alerts.send_critical("KILL confirmed: all trading stopped.")
                self._reply(chat_id, "KILL confirmed. Trading disabled.")
                return
            self._pending_kill_until = now + self._confirm_window
            self._reply(
                chat_id,
                f"Send `/kill CONFIRM` within {int(self._confirm_window)} seconds to kill bot trading.",
            )
            return

        if cmd_l == "/pause":
            st = self._control.set_pause("telegram_pause", updated_by=f"telegram:{chat_id}")
            self._audit.log_control_event(
                {"event_type": "pause", "detail": st.paused_reason or "", "state": self._control.to_dict()},
            )
            self._reply(chat_id, "Paused: new positions disabled (exits still allowed if trading enabled).")
            return

        if cmd_l == "/resume":
            st = self._control.set_resume(updated_by=f"telegram:{chat_id}")
            self._audit.log_control_event(
                {"event_type": "resume", "detail": "resumed", "state": self._control.to_dict()},
            )
            self._reply(chat_id, "Resumed: trading and new positions enabled (unless killed).")
            return

        if cmd_l == "/status":
            st = self._control.get_state()
            lines = [
                f"trade_enabled={st.trade_enabled}",
                f"open_new_position_enabled={st.open_new_position_enabled}",
                f"force_flatten={st.force_flatten_requested}",
                f"killed={st.killed}",
                f"paused_reason={st.paused_reason!r}",
                f"can_send_order={self._control.can_send_order()}",
                f"can_open_new={self._control.can_open_new_position()}",
            ]
            self._reply(chat_id, "\n".join(lines))
            return

        if cmd_l == "/health":
            oc = getattr(self._runner, "ops", None) or getattr(self._runner, "_ops", None)
            snap = {}
            if oc is not None and hasattr(oc, "health_snapshot"):
                try:
                    snap = oc.health_snapshot()
                except Exception:
                    snap = {}
            if snap:
                self._reply(chat_id, json.dumps(snap, indent=2, default=str)[:3500])
            else:
                self._reply(chat_id, "Health snapshot not available (ops not wired on runner).")
            return

        if cmd_l == "/position":
            reply = self._safe_position_summary()
            self._reply(chat_id, reply)
            return

        if cmd_l == "/pnl":
            reply = self._safe_pnl_summary()
            self._reply(chat_id, reply)
            return

        if cmd_l == "/risk":
            reply = self._safe_risk_summary()
            self._reply(chat_id, reply)
            return

    def _safe_position_summary(self) -> str:
        r = self._runner
        if r is None:
            return "Runner not attached; position unknown."
        try:
            tr = getattr(r, "_tracker", None)
            if tr is None:
                return "No position tracker."
            order_sym = None
            settings = getattr(r, "_settings", None)
            sym = getattr(settings, "HMM_SYMBOL", "VN30F1M") if settings else "VN30F1M"
            from src.live.hmm_live_runner import _derivative_order_symbol  # local import

            order_sym = _derivative_order_symbol(sym)
            p = tr.get_position(order_sym)
            if p is None or p.is_flat:
                return f"Flat ({order_sym})."
            return f"{order_sym} qty={p.quantity} avg={getattr(p, 'avg_price', '')}"
        except Exception as e:
            return f"Position unavailable: {e}"

    def _safe_pnl_summary(self) -> str:
        r = self._runner
        if r is None:
            return "Runner not attached; PnL unknown."
        try:
            tr = getattr(r, "_tracker", None)
            risk = getattr(r, "_risk", None)
            if tr is None:
                return "No tracker."
            snap = risk.stats if risk is not None else None
            daily = getattr(tr, "daily_pnl", None)
            dnet = getattr(daily, "net", None) if daily is not None else None
            parts = []
            if dnet is not None:
                parts.append(f"daily_pnl_net={dnet}")
            if snap is not None:
                parts.append(f"halted={snap.is_halted} reason={snap.halt_reason!r}")
            return "\n".join(parts) if parts else "PnL snapshot unavailable."
        except Exception as e:
            return f"PnL unavailable: {e}"

    def _safe_risk_summary(self) -> str:
        r = self._runner
        if r is None:
            return "Runner not attached."
        try:
            risk = getattr(r, "_risk", None)
            if risk is None:
                return "No risk manager."
            snap = risk.stats
            return (
                f"halted={snap.is_halted} halt_reason={snap.halt_reason!r}\n"
                f"external_enabled={risk.external_trading_enabled}\n"
                f"daily_pnl={snap.daily_pnl_net}"
            )
        except Exception as e:
            return f"Risk unavailable: {e}"
