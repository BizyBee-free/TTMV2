"""Append-only JSONL audit logs with daily rotation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger("ops.audit_logger")

_SENSITIVE_KEY_RE = re.compile(r"(otp|token|secret|password|api_key|api_secret|trading_token)", re.I)


def _scrub_value(key: str, value: Any) -> Any:
    if _SENSITIVE_KEY_RE.search(key):
        if value is None:
            return None
        s = str(value)
        if "otp" in key.lower():
            return "OTP******" if len(s) > 4 else "OTP****"
        return "***"
    if isinstance(value, dict):
        return scrub_event_dict(value)
    if isinstance(value, list):
        return [_scrub_value(key, v) if not isinstance(v, dict) else scrub_event_dict(v) for v in value]
    return value


def scrub_event_dict(event: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in event.items():
        if isinstance(v, dict):
            out[k] = scrub_event_dict(v)
        else:
            out[k] = _scrub_value(k, v)
    return out


class AuditLogger:
    """Writes ``logs/<name>_YYYYMMDD.jsonl`` with required metadata fields."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        log_dir = self._settings.LOG_DIR
        p = Path(log_dir)
        if not p.is_absolute():
            root = Path(__file__).resolve().parent.parent.parent
            p = root / p
        self._log_dir = p
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, base_name: str) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self._log_dir / f"{base_name}_{day}.jsonl"

    def _append(self, base_name: str, event: Dict[str, Any]) -> None:
        path = self._path_for(base_name)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.error("audit append failed", extra={"path": str(path), "error": str(e)})

    def _enrich(
        self,
        event: Dict[str, Any],
        *,
        default_type: str,
        default_source: str,
        default_severity: str,
    ) -> Dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        raw = dict(event)
        merged = {
            "timestamp_utc": raw.pop("timestamp_utc", ts),
            "event_type": raw.pop("event_type", default_type),
            "source": raw.pop("source", default_source),
            "severity": raw.pop("severity", default_severity),
            **raw,
        }
        return scrub_event_dict(merged)

    def log_control_event(self, event: Dict[str, Any]) -> None:
        self._append("control_events", self._enrich(event, default_type="control", default_source="ops", default_severity="INFO"))

    def log_order_event(self, event: Dict[str, Any]) -> None:
        self._append("order_events", self._enrich(event, default_type="order", default_source="ops", default_severity="INFO"))

    def log_risk_event(self, event: Dict[str, Any]) -> None:
        self._append("risk_events", self._enrich(event, default_type="risk", default_source="ops", default_severity="WARNING"))

    def log_health_event(self, event: Dict[str, Any]) -> None:
        self._append("health_events", self._enrich(event, default_type="health", default_source="ops", default_severity="INFO"))
