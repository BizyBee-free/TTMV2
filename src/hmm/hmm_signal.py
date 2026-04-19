"""HMM signal generator: maps regime state probabilities to trade direction.

Trading rules (from HMMplan.yaml):
    BULL state  →  +1 (Long)
    BEAR state  →  -1 (Short)
    FLAT state  →   0 (No trade / hold flat)

Confidence filter:
    Only open a position if the probability of the most likely state
    exceeds `confidence_threshold`.  This avoids trading when the model
    is uncertain (e.g. during regime transitions).

Usage::

    from src.hmm.hmm_signal import HMMSignalGenerator
    gen = HMMSignalGenerator(state_labels={0:"BULL",1:"FLAT",2:"BEAR"})
    direction = gen.get_direction(proba, confidence_threshold=0.60)
    # direction: +1 (Long), -1 (Short), or 0 (Flat)
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.hmm.regime_model import STATE_BEAR, STATE_BULL, STATE_FLAT

# Direction constants
LONG  =  1
SHORT = -1
FLAT  =  0


class HMMSignalGenerator:
    """Maps posterior state probabilities to a trade direction.

    Args:
        state_labels: Mapping from internal state index to label string
                      (e.g. ``{0: "BULL", 1: "FLAT", 2: "BEAR"}``).
                      Obtained from HMMRegimeModel.state_labels after fitting.
    """

    def __init__(self, state_labels: Dict[int, str]) -> None:
        self._labels = state_labels

    def get_direction(
        self,
        proba: np.ndarray,
        confidence_threshold: float = 0.60,
    ) -> int:
        """Return trade direction for a single bar.

        Args:
            proba:                Shape ``(k_states,)`` posterior probabilities.
            confidence_threshold: Minimum probability of the dominant state
                                  required to open a trade. Default 0.60.

        Returns:
            +1 (Long), -1 (Short), or 0 (Flat/no trade).
        """
        if proba.size == 0:
            return FLAT

        best_state = int(np.argmax(proba))
        best_prob  = float(proba[best_state])

        if best_prob < confidence_threshold:
            return FLAT

        label = self._labels.get(best_state, STATE_FLAT)
        if label == STATE_BULL:
            return LONG
        if label == STATE_BEAR:
            return SHORT
        return FLAT   # FLAT / ACCUMULATION

    def dominant_state_label(self, proba: np.ndarray) -> str:
        """Return the label of the most probable state (ignores threshold)."""
        if proba.size == 0:
            return STATE_FLAT
        best = int(np.argmax(proba))
        return self._labels.get(best, STATE_FLAT)

    def dominant_state_proba(self, proba: np.ndarray) -> float:
        """Return the probability of the most likely state."""
        if proba.size == 0:
            return 0.0
        return float(np.max(proba))
