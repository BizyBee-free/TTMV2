"""Outbound Telegram notifications (Bot API). Failures never raise to callers."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger("telegram_notifier")


def _parse_allowed_chat_ids(raw: str, default_chat: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return [default_chat] if default_chat else []
    return [x.strip() for x in raw.split(",") if x.strip()]


class TelegramNotifier:
    """Send messages via https://api.telegram.org/bot<token>/sendMessage."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    @property
    def is_configured(self) -> bool:
        s = self._settings
        return bool(
            s.TELEGRAM_ENABLED
            and s.TELEGRAM_BOT_TOKEN
            and s.TELEGRAM_CHAT_ID
        )

    def send_message(self, text: str, chat_id: Optional[str] = None) -> bool:
        """Send plain text. Returns True on HTTP 200 from Telegram."""
        if not self.is_configured:
            logger.debug("Telegram disabled or not configured, skip send")
            return False
        cid = chat_id or self._settings.TELEGRAM_CHAT_ID
        token = self._settings.TELEGRAM_BOT_TOKEN
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode(
            {"chat_id": cid, "text": text[:4000]}
        ).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                ok = resp.status == 200
                if not ok:
                    logger.warning(
                        "Telegram sendMessage non-200",
                        extra={"status": resp.status},
                    )
                return ok
        except urllib.error.HTTPError as e:
            logger.error("Telegram HTTPError", extra={"code": e.code})
            return False
        except urllib.error.URLError as e:
            logger.error("Telegram URLError", extra={"error": str(e)})
            return False
        except Exception as e:
            logger.error("Telegram unexpected error", extra={"error": str(e)})
            return False

    def send_alert(self, severity: str, title: str, body: str = "") -> bool:
        """Human-readable alert with severity prefix (INFO, CRITICAL, HALT)."""
        text = f"[{severity}] {title}"
        if body:
            text = f"{text}\n{body}"
        return self.send_message(text)

    def is_chat_allowed(self, chat_id: str) -> bool:
        """Return True if chat_id may send /pause /resume commands."""
        allowed = _parse_allowed_chat_ids(
            self._settings.TELEGRAM_ALLOWED_CHAT_IDS,
            self._settings.TELEGRAM_CHAT_ID,
        )
        if not allowed:
            return False
        return str(chat_id).strip() in allowed
