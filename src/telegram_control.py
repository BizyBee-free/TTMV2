"""Telegram inbound commands via getUpdates long polling (/pause, /resume, /status)."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional

from src.config import Settings, get_settings
from src.logger import get_logger
from src.risk_manager import RiskManager
from src.telegram_notifier import TelegramNotifier

logger = get_logger("telegram_control")

CommandHandler = Callable[[str, str, Dict[str, Any]], None]


class TelegramControlBot:
    """Background thread: poll Telegram for commands and update RiskManager."""

    def __init__(
        self,
        risk: RiskManager,
        notifier: Optional[TelegramNotifier] = None,
        settings: Optional[Settings] = None,
        poll_interval_sec: float = 2.0,
    ) -> None:
        self._risk = risk
        self._notifier = notifier or TelegramNotifier(settings=settings)
        self._settings = settings or get_settings()
        self._poll_interval = poll_interval_sec
        self._offset: Optional[int] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start_background(self) -> None:
        if not self._notifier.is_configured:
            logger.info("TelegramControlBot: notifier not configured, skip control thread")
            return
        if not self._settings.TELEGRAM_BOT_TOKEN:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="TelegramControl", daemon=True)
        self._thread.start()
        logger.info("TelegramControlBot started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

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
                        logger.warning("Telegram command from disallowed chat", extra={"chat_id": cid})
                        continue
                    self._handle_command(cid, text.lower())
            except urllib.error.URLError as e:
                logger.warning("getUpdates URLError", extra={"error": str(e)})
                time.sleep(self._poll_interval)
            except Exception as e:
                logger.error("getUpdates loop error", extra={"error": str(e)})
                time.sleep(self._poll_interval)

    def _handle_command(self, chat_id: str, text: str) -> None:
        parts = text.split()
        cmd = parts[0] if parts else ""
        reply = ""

        if cmd in ("/pause", "/stop"):
            self._risk.set_external_trading_enabled(False)
            reply = "Trading paused (no new orders). Positions unchanged."
        elif cmd in ("/resume", "/start"):
            self._risk.set_external_trading_enabled(True)
            reply = "Trading enabled for new signals (subject to risk halt)."
        elif cmd in ("/status",):
            snap = self._risk.stats
            reply = (
                f"halted={snap.is_halted} reason={snap.halt_reason!r}\n"
                f"external_enabled={self._risk.external_trading_enabled}\n"
                f"daily_pnl={snap.daily_pnl_net}"
            )
        else:
            return

        self._notifier.send_message(reply, chat_id=chat_id)
