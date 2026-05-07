"""Outbound Telegram alerts for ops layer — short timeout, never raises, masks secrets."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.config import Settings, get_settings
from src.logger import get_logger
from src.telegram_notifier import TelegramNotifier, _outbound_chat_id

logger = get_logger("ops.telegram_alerts")

_DEFAULT_TIMEOUT_SEC = 5.0


def resolved_outbound_chat_id(settings: Settings) -> str:
    """Same resolution as TelegramNotifier outbound destination."""
    return _outbound_chat_id(settings)


def mask_secrets_in_text(text: str) -> str:
    """Redact bot-token-like strings and obvious secrets in message bodies."""
    if not text:
        return text
    # Bot tokens: 123456:ABC-DEF...
    text = re.sub(
        r"\b\d{8,10}:[A-Za-z0-9_-]{30,}\b",
        "<token>",
        text,
    )
    # OTP123456 style (keep prefix for context)
    text = re.sub(r"\bOTP\d{4,}\b", "OTP******", text, flags=re.IGNORECASE)
    return text


class TelegramAlertClient:
    """Ops-facing Telegram client: formatted tags, non-blocking semantics (returns bool)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._notifier = TelegramNotifier(settings=self._settings)
        self._warned_disabled = False

    def is_enabled(self) -> bool:
        s = self._settings
        if not getattr(s, "TELEGRAM_ENABLED", False):
            return False
        if not (s.TELEGRAM_BOT_TOKEN or "").strip():
            return False
        if not resolved_outbound_chat_id(s):
            return False
        return True

    def send_message(self, text: str, level: str = "INFO") -> bool:
        """Send ``[LEVEL] ...`` message. Never raises."""
        try:
            if not self.is_enabled():
                if not self._warned_disabled:
                    logger.warning(
                        "Telegram ops alerts disabled or missing TELEGRAM_BOT_TOKEN / chat id",
                    )
                    self._warned_disabled = True
                return False
            body = mask_secrets_in_text(text)
            line = f"[{level}] {body}"
            cid = resolved_outbound_chat_id(self._settings)
            return bool(
                self._notifier.send_message(
                    line[:4000],
                    chat_id=cid,
                    timeout_sec=_DEFAULT_TIMEOUT_SEC,
                ),
            )
        except Exception as e:
            logger.error("TelegramAlertClient.send_message swallowed error", extra={"error": str(e)})
            return False

    def send_critical(self, text: str) -> bool:
        return self.send_message(text, level="CRITICAL")

    def send_order_event(self, event: Dict[str, Any]) -> bool:
        """Format order notification e.g. [ORDER_FILLED] LONG 1 VN30F1M @ ..."""
        try:
            et = str(event.get("event_type") or event.get("type") or "ORDER").upper()
            side = str(event.get("side") or "")
            qty = event.get("quantity", "")
            sym = str(event.get("symbol") or "")
            price = event.get("price", "")
            oid = str(event.get("order_id") or "")
            parts = [et, side, str(qty), sym]
            if price != "":
                parts.append("@")
                parts.append(str(price))
            msg = " ".join(p for p in parts if p)
            if oid:
                msg = f"{msg} id={oid}"
            return self.send_message(msg, level=et if et in ("ORDER_FILLED", "ORDER_REJECTED", "ORDER_SUBMITTED") else "ORDER")
        except Exception as e:
            logger.error("send_order_event error", extra={"error": str(e)})
            return False

    def send_health_event(self, event: Dict[str, Any]) -> bool:
        try:
            label = str(event.get("event_type") or "HEALTH")
            sev = str(event.get("severity") or "INFO").upper()
            detail = str(event.get("detail") or event.get("message") or "")
            return self.send_message(f"{label}: {detail}", level=sev)
        except Exception as e:
            logger.error("send_health_event error", extra={"error": str(e)})
            return False
