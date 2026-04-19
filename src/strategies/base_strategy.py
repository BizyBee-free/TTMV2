"""Abstract base for signal-only strategies (LONG/SHORT/EXIT/HOLD)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, TypedDict


class SignalDict(TypedDict, total=False):
    """Standard output of :meth:`BaseStrategy.generate_signal`."""

    action: str  # "LONG" | "SHORT" | "EXIT" | "HOLD"
    confidence: float
    reason: str
    strategy: str


class BaseStrategy(ABC):
    """Minimal interface for backtest/live signal generation."""

    @abstractmethod
    def generate_signal(self, state: Any) -> Dict[str, Any]:
        """
        Args:
            state: Strategy-specific state (often includes OHLC bars, OI, position).

        Returns:
            {
                "action": "LONG" | "SHORT" | "EXIT" | "HOLD",
                "confidence": float,
                "reason": str,
                "strategy": str,
            }
        """
        raise NotImplementedError
