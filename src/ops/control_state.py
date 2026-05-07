"""Persistent control state — gates only; does not affect strategy signals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import Settings, get_settings
from src.logger import get_logger

logger = get_logger("ops.control_state")


def _default_state_path(settings: Settings) -> Path:
    root = Path(__file__).resolve().parent.parent.parent
    return root / "runtime" / "control_state.json"


@dataclass
class ControlState:
    trade_enabled: bool
    open_new_position_enabled: bool
    force_flatten_requested: bool
    killed: bool
    paused_reason: Optional[str]
    updated_at: str
    updated_by: Optional[str]


class ControlStateManager:
    """Thread-safe enough for single-writer process; reload from disk each get if needed."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._path = state_path or _default_state_path(self._settings)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write_initial()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _defaults(self) -> ControlState:
        trade_on = bool(getattr(self._settings, "CONTROL_TRADE_ENABLED_DEFAULT", False))
        return ControlState(
            trade_enabled=trade_on,
            open_new_position_enabled=trade_on,
            force_flatten_requested=False,
            killed=False,
            paused_reason=None,
            updated_at=self._now_iso(),
            updated_by=None,
        )

    def _write_initial(self) -> None:
        st = self._defaults()
        self._atomic_write(st)

    def _atomic_write(self, state: ControlState) -> None:
        try:
            payload = json.dumps(asdict(state), indent=2, ensure_ascii=False)
            tmp = self._path.with_suffix(".json.tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._path)
        except Exception as e:
            logger.error("control_state write failed", extra={"error": str(e)})

    def _load(self) -> ControlState:
        if not self._path.exists():
            return self._defaults()
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return ControlState(
                trade_enabled=bool(raw.get("trade_enabled", False)),
                open_new_position_enabled=bool(raw.get("open_new_position_enabled", raw.get("trade_enabled", False))),
                force_flatten_requested=bool(raw.get("force_flatten_requested", False)),
                killed=bool(raw.get("killed", False)),
                paused_reason=raw.get("paused_reason"),
                updated_at=str(raw.get("updated_at") or self._now_iso()),
                updated_by=raw.get("updated_by"),
            )
        except Exception as e:
            logger.warning("control_state load failed, using safe defaults", extra={"error": str(e)})
            return self._defaults()

    def get_state(self) -> ControlState:
        return self._load()

    def _save(self, st: ControlState) -> None:
        self._atomic_write(st)

    def set_pause(self, reason: str, updated_by: Optional[str] = None) -> ControlState:
        st = self._load()
        if st.killed:
            return st
        st.open_new_position_enabled = False
        st.paused_reason = reason
        st.updated_at = self._now_iso()
        st.updated_by = updated_by
        self._save(st)
        return st

    def set_resume(self, updated_by: Optional[str] = None) -> ControlState:
        st = self._load()
        if st.killed:
            return st
        st.trade_enabled = True
        st.open_new_position_enabled = True
        st.paused_reason = None
        st.updated_at = self._now_iso()
        st.updated_by = updated_by
        self._save(st)
        return st

    def request_flatten(self, updated_by: Optional[str] = None) -> ControlState:
        st = self._load()
        if st.killed:
            return st
        st.force_flatten_requested = True
        st.open_new_position_enabled = False
        st.updated_at = self._now_iso()
        st.updated_by = updated_by
        self._save(st)
        return st

    def clear_flatten(self, updated_by: Optional[str] = None) -> ControlState:
        st = self._load()
        st.force_flatten_requested = False
        st.updated_at = self._now_iso()
        st.updated_by = updated_by
        self._save(st)
        return st

    def kill(self, updated_by: Optional[str] = None) -> ControlState:
        st = self._load()
        st.killed = True
        st.trade_enabled = False
        st.open_new_position_enabled = False
        st.paused_reason = "killed"
        st.updated_at = self._now_iso()
        st.updated_by = updated_by
        self._save(st)
        return st

    def enable_trading(self, updated_by: Optional[str] = None) -> ControlState:
        """Explicitly enable trading (e.g. first-time go-live)."""
        st = self._load()
        if st.killed:
            return st
        st.trade_enabled = True
        st.open_new_position_enabled = True
        st.paused_reason = None
        st.updated_at = self._now_iso()
        st.updated_by = updated_by
        self._save(st)
        return st

    def can_open_new_position(self) -> bool:
        st = self._load()
        if st.killed or not st.trade_enabled:
            return False
        if not st.open_new_position_enabled:
            return False
        if st.force_flatten_requested:
            return False
        return True

    def can_send_order(self) -> bool:
        st = self._load()
        if st.killed:
            return False
        if not st.trade_enabled:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.get_state())
