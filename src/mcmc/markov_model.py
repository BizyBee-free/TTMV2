"""2-State Markov Chain model for derivatives session direction prediction.

States:
    UP   (1) -- session closes higher than open (close > open)
    DOWN (0) -- session closes equal or lower than open (close <= open)

The transition matrix P is estimated from the last N observed sessions
(rolling window) using maximum likelihood (frequency counting):

    P[i][j] = count(i -> j) / count(i -> any)

where i, j ∈ {DOWN, UP}.

Usage::

    model = MarkovModel(window=50)
    model.feed_ohlc_history(candles)          # bulk-load historical OHLC
    p_up, p_down = model.predict()            # predict next session
    state = model.update(open_p, close_p)     # record new session result
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, List, Optional, Tuple

import numpy as np

from src.logger import get_logger

logger = get_logger("markov_model")


class MarkovState(IntEnum):
    DOWN = 0
    UP = 1


@dataclass
class TransitionStats:
    """Snapshot of the transition matrix and observation counts."""
    matrix: np.ndarray          # shape (2, 2) -- P[from][to]
    counts: np.ndarray          # shape (2, 2) -- raw transition counts
    state_counts: np.ndarray    # shape (2,)   -- total times in each state
    n_sessions: int
    last_state: Optional[MarkovState]


class MarkovModel:
    """Rolling-window 2-state Markov Chain.

    Args:
        window: Maximum number of historical sessions retained for
                transition matrix estimation. Older sessions are
                discarded automatically.
    """

    # Minimum sessions needed before predictions are meaningful
    MIN_SESSIONS = 5

    def __init__(self, window: int = 50) -> None:
        if window < self.MIN_SESSIONS:
            raise ValueError(f"window must be >= {self.MIN_SESSIONS}")
        self._window = window

        # Ring buffer of observed states (one per session)
        self._states: Deque[int] = deque(maxlen=window + 1)

        # Cached transition matrix -- invalidated when _states changes
        self._matrix_cache: Optional[np.ndarray] = None
        self._cache_valid = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def window(self) -> int:
        return self._window

    @property
    def n_sessions(self) -> int:
        """Number of sessions currently in the rolling window."""
        # At least 2 states needed to form 1 transition
        return max(0, len(self._states) - 1)

    @property
    def is_ready(self) -> bool:
        """True when enough data exists for reliable predictions."""
        return self.n_sessions >= self.MIN_SESSIONS

    @property
    def last_state(self) -> Optional[MarkovState]:
        """The most recently observed session state."""
        if not self._states:
            return None
        return MarkovState(self._states[-1])

    def feed_ohlc_history(self, candles: list) -> int:
        """Bulk-load OHLC candles to initialise the rolling window.

        Args:
            candles: List of objects with .open and .close attributes
                     (compatible with vendor DNSE Ohlc model), sorted
                     oldest-first.

        Returns:
            Number of sessions loaded.
        """
        loaded = 0
        for c in candles:
            try:
                open_p = float(c.open)
                close_p = float(c.close)
            except (AttributeError, TypeError, ValueError):
                continue
            state = MarkovState.UP if close_p > open_p else MarkovState.DOWN
            self._states.append(int(state))
            loaded += 1

        self._cache_valid = False
        logger.debug("MarkovModel: loaded OHLC history", extra={"sessions": loaded, "window": self._window})
        return loaded

    def update(self, open_price: float, close_price: float) -> MarkovState:
        """Record the result of a completed session.

        Args:
            open_price:  Session open price.
            close_price: Session close price.

        Returns:
            The classified state for this session.
        """
        state = MarkovState.UP if close_price > open_price else MarkovState.DOWN
        self._states.append(int(state))
        self._cache_valid = False
        logger.debug(
            "MarkovModel: session recorded",
            extra={"state": state.name, "open": open_price, "close": close_price},
        )
        return state

    def predict(self, current_state: Optional[MarkovState] = None) -> Tuple[float, float]:
        """Predict next-session probabilities given the current state.

        Args:
            current_state: The state to condition on. If None, uses the
                           most recently observed state.

        Returns:
            (p_up, p_down) -- probabilities summing to 1.0.

        Raises:
            RuntimeError: If fewer than MIN_SESSIONS have been observed.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Not enough data: need {self.MIN_SESSIONS} sessions, "
                f"have {self.n_sessions}"
            )

        matrix = self._get_transition_matrix()
        state = current_state if current_state is not None else self.last_state
        if state is None:
            state = MarkovState.DOWN

        row = matrix[int(state)]
        p_up = float(row[MarkovState.UP])
        p_down = float(row[MarkovState.DOWN])
        return p_up, p_down

    def get_stats(self) -> TransitionStats:
        """Return a full snapshot of the model's current state."""
        matrix = self._get_transition_matrix() if self.is_ready else np.full((2, 2), 0.5)
        counts = self._count_transitions()
        state_counts = np.array([
            int(np.sum(counts[0])),
            int(np.sum(counts[1])),
        ])
        return TransitionStats(
            matrix=matrix,
            counts=counts,
            state_counts=state_counts,
            n_sessions=self.n_sessions,
            last_state=self.last_state,
        )

    def reset(self) -> None:
        """Clear all historical data."""
        self._states.clear()
        self._matrix_cache = None
        self._cache_valid = False

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _count_transitions(self) -> np.ndarray:
        """Count transitions between states. Returns shape (2, 2) int array."""
        counts = np.zeros((2, 2), dtype=np.int64)
        states = list(self._states)
        for i in range(len(states) - 1):
            from_s = states[i]
            to_s = states[i + 1]
            counts[from_s][to_s] += 1
        return counts

    def _get_transition_matrix(self) -> np.ndarray:
        """Compute (and cache) the MLE transition probability matrix."""
        if self._cache_valid and self._matrix_cache is not None:
            return self._matrix_cache

        counts = self._count_transitions()
        matrix = np.zeros((2, 2), dtype=np.float64)

        for i in range(2):
            row_total = counts[i].sum()
            if row_total > 0:
                matrix[i] = counts[i] / row_total
            else:
                # Uniform prior when a state has never been observed
                matrix[i] = np.array([0.5, 0.5])

        self._matrix_cache = matrix
        self._cache_valid = True
        return matrix

    def get_log_returns(self, candles: list) -> np.ndarray:
        """Extract log-returns from a list of OHLC candles.

        Convenience method used by MCMCEngine to get the return series
        without needing to import OHLC models separately.

        Returns:
            1-D numpy array of log(close/open) per candle, oldest first.
        """
        returns: List[float] = []
        for c in candles:
            try:
                open_p = float(c.open)
                close_p = float(c.close)
                if open_p > 0:
                    returns.append(float(np.log(close_p / open_p)))
            except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                continue
        return np.array(returns, dtype=np.float64)

    def empirical_state_mean_log_returns(self, candles: list) -> np.ndarray:
        """Mean log(close/open) per Markov state (0=DOWN, 1=UP) over ``candles``.

        Sessions without a valid open/close are skipped. States with no samples
        get mean 0.0 for that coordinate.
        """
        sums = np.zeros(2, dtype=np.float64)
        counts = np.zeros(2, dtype=np.int64)
        for c in candles:
            try:
                open_p = float(c.open)
                close_p = float(c.close)
            except (AttributeError, TypeError, ValueError):
                continue
            if open_p <= 0:
                continue
            st = int(MarkovState.UP if close_p > open_p else MarkovState.DOWN)
            sums[st] += float(np.log(close_p / open_p))
            counts[st] += 1
        out = np.zeros(2, dtype=np.float64)
        for i in range(2):
            if counts[i] > 0:
                out[i] = sums[i] / counts[i]
        return out
