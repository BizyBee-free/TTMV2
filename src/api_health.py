"""Track consecutive DNSE API failures and signal when to halt trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.logger import get_logger

logger = get_logger("api_health")

HaltCallback = Callable[[str], None]


@dataclass
class ApiHealthMonitor:
    """Increments on failure, resets on success; fires halt when threshold reached."""

    threshold: int = 3
    _consecutive_failures: int = 0
    _halt_callbacks: List[HaltCallback] = field(default_factory=list)

    def on_halt(self, callback: HaltCallback) -> None:
        self._halt_callbacks.append(callback)

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self, reason: str) -> bool:
        """Return True if halt threshold reached."""
        self._consecutive_failures += 1
        logger.warning(
            "ApiHealthMonitor failure",
            extra={"consecutive": self._consecutive_failures, "reason": reason},
        )
        if self._consecutive_failures >= self.threshold:
            msg = (
                f"DNSE_API_FAIL_HALT: {self._consecutive_failures} consecutive failures "
                f"(last={reason})"
            )
            for cb in self._halt_callbacks:
                try:
                    cb(msg)
                except Exception as e:
                    logger.error("ApiHealth halt callback error", extra={"error": str(e)})
            self._consecutive_failures = 0
            return True
        return False

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def reset(self) -> None:
        self._consecutive_failures = 0
