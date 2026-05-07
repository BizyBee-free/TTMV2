"""OTP collection via Telegram — in-memory only, masked logging."""

from __future__ import annotations

import re
import threading
import time
from typing import TYPE_CHECKING, Optional

from src.logger import get_logger

if TYPE_CHECKING:
    from src.config import Settings
    from src.ops.telegram_alerts import TelegramAlertClient
    from src.telegram_notifier import TelegramNotifier

logger = get_logger("ops.otp_manager")

_OTP_MSG_RE = re.compile(r"^OTP\s*(\d{4,10})\s*$", re.IGNORECASE)


def mask_otp(otp: str) -> str:
    o = (otp or "").strip()
    if len(o) <= 2:
        return "OTP****"
    return f"OTP{o[:2]}****"


class OTPManager:
    def __init__(
        self,
        *,
        settings: "Settings",
        alerts: "TelegramAlertClient",
        notifier: "TelegramNotifier",
        max_attempts: int = 3,
    ) -> None:
        self._settings = settings
        self._alerts = alerts
        self._notifier = notifier
        self._max_attempts = max_attempts
        self._lock = threading.Lock()
        self._pending = False
        self._deadline = 0.0
        self._attempts = 0
        self._otp_result: Optional[str] = None
        self._done = threading.Event()
        self._reason = ""

    def has_pending_request(self) -> bool:
        with self._lock:
            return self._pending and time.time() < self._deadline

    def clear(self) -> None:
        with self._lock:
            self._pending = False
            self._otp_result = None
            self._done.set()
            self._attempts = 0

    def request_otp(self, reason: str, timeout_sec: int = 120) -> Optional[str]:
        if not getattr(self._settings, "OTP_TELEGRAM_ENABLED", False):
            logger.warning("OTP_TELEGRAM_ENABLED=false, cannot collect OTP")
            return None
        with self._lock:
            self._pending = True
            self._deadline = time.time() + max(10, int(timeout_sec))
            self._attempts = 0
            self._otp_result = None
            self._done.clear()
            self._reason = reason
        msg = (
            f"OTP required for broker auth ({reason}). "
            f"Send OTP as OTPXXXXXX within {timeout_sec} seconds."
        )
        try:
            self._alerts.send_message(msg, level="WARNING")
        except Exception as e:
            logger.error("OTP prompt telegram failed", extra={"error": str(e)})
        # Wait
        while True:
            remaining = self._deadline - time.time()
            if remaining <= 0:
                break
            if self._done.wait(timeout=min(remaining, 0.5)):
                break
        with self._lock:
            self._pending = False
            out = self._otp_result
            self._otp_result = None
        if out:
            return out
        try:
            self._alerts.send_critical("Broker auth failed: OTP timeout.")
        except Exception:
            pass
        logger.error("OTP request expired or failed", extra={"reason": reason})
        return None

    def submit_otp_from_message(self, message_text: str, chat_id: str) -> bool:
        if not getattr(self._settings, "OTP_TELEGRAM_ENABLED", False):
            return False
        if not self._notifier.is_chat_allowed(str(chat_id).strip()):
            logger.warning("OTP rejected: chat not allowed", extra={"chat_id": chat_id})
            return False
        m = _OTP_MSG_RE.match((message_text or "").strip())
        if not m:
            return False
        otp = m.group(1)
        with self._lock:
            if not self._pending or time.time() > self._deadline:
                logger.info("OTP ignored: no pending request")
                return False
            self._attempts += 1
            if self._attempts > self._max_attempts:
                logger.error("OTP max attempts exceeded", extra={"masked": mask_otp(otp)})
                self._pending = False
                self._done.set()
                return False
        logger.info("OTP received", extra={"masked": mask_otp(otp)})
        with self._lock:
            self._otp_result = otp
            self._pending = False
            self._done.set()
        return True
