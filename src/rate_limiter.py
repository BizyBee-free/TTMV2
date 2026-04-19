"""Sliding window rate limiter for order submission.

Uses a deque of timestamps to enforce a maximum number of actions
within a rolling time window. Designed to protect against accidental
order floods before they reach the DNSE API.
"""

import time
from collections import deque

from src.logger import get_logger

logger = get_logger("rate_limiter")


class RateLimitExceeded(Exception):
    """Raised when an action is blocked by the rate limiter."""

    def __init__(self, limit: int, window: float, retry_after: float):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded: {limit} actions per {window:.0f}s. "
            f"Retry after {retry_after:.1f}s"
        )


class SlidingWindowRateLimiter:
    """Sliding window rate limiter.

    Tracks timestamps of recent actions in a deque.  Before each new
    action, expired entries (older than *window* seconds) are pruned.
    If the remaining count >= *limit*, the action is rejected.

    Args:
        limit:  Maximum number of actions allowed in the window.
        window: Window size in seconds (default 60 = 1 minute).
    """

    def __init__(self, limit: int, window: float = 60.0) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window <= 0:
            raise ValueError("window must be positive")
        self._limit = limit
        self._window = window
        self._timestamps: deque[float] = deque()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def window(self) -> float:
        return self._window

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def acquire(self, now: float | None = None) -> None:
        """Record an action. Raises RateLimitExceeded if over limit."""
        now = now or time.time()
        self._prune(now)

        if len(self._timestamps) >= self._limit:
            oldest = self._timestamps[0]
            retry_after = oldest + self._window - now
            logger.warning(
                "Rate limit hit",
                extra={
                    "limit": self._limit,
                    "window": self._window,
                    "retry_after": round(retry_after, 2),
                },
            )
            raise RateLimitExceeded(self._limit, self._window, retry_after)

        self._timestamps.append(now)

    def can_acquire(self, now: float | None = None) -> bool:
        """Check if an action is allowed without consuming a slot."""
        now = now or time.time()
        self._prune(now)
        return len(self._timestamps) < self._limit

    @property
    def remaining(self) -> int:
        """How many more actions are allowed right now."""
        self._prune(time.time())
        return max(0, self._limit - len(self._timestamps))

    @property
    def count(self) -> int:
        """Number of actions recorded in the current window."""
        self._prune(time.time())
        return len(self._timestamps)

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()

    def retry_after(self) -> float:
        """Seconds until the next slot opens. Returns 0 if available now."""
        now = time.time()
        self._prune(now)
        if len(self._timestamps) < self._limit:
            return 0.0
        oldest = self._timestamps[0]
        return max(0.0, oldest + self._window - now)
