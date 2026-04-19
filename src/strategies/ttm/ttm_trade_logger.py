"""In-memory rolling trade log for TTM adaptive learning (no disk I/O on hot path)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class TradeRecord:
    entry_time: str
    exit_time: str
    side: str  # "LONG" | "SHORT"
    entry_price: float
    exit_price: float
    pnl: float
    holding_bars: int
    features_at_entry: Dict[str, float]
    score_long: float
    score_short: float
    prob_long: float
    prob_short: float
    regime: int
    extra: Dict[str, Any] = field(default_factory=dict)


class TradeLogger:
    """Rolling window of completed trades (FIFO eviction when maxlen exceeded)."""

    def __init__(self, maxlen: int = 100) -> None:
        self._maxlen = max(1, int(maxlen))
        self._q: Deque[TradeRecord] = deque(maxlen=self._maxlen)

    def log_trade(self, record: TradeRecord) -> None:
        self._q.append(record)

    def get_recent_trades(self, n: int) -> List[TradeRecord]:
        if n <= 0:
            return []
        items = list(self._q)
        return items[-n:] if len(items) > n else items

    def __len__(self) -> int:
        return len(self._q)
