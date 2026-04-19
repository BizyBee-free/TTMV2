"""Sum of empirical expected returns → alpha (automate_tunning.md §3)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.strategies.ttm.empirical.calibration import CalibrationState
from src.strategies.ttm.empirical.feature_store import EMPIRICAL_FEATURE_KEYS, extract_empirical_features_from_last


def compute_empirical_alpha(
    last: Mapping[str, Any],
    state: Optional[CalibrationState],
    *,
    clip_abs: float = 0.003,
) -> tuple[float, Dict[str, float]]:
    """
    ``alpha = sum_f curve_f(bin(f))``; clip to ``[-clip_abs, clip_abs]``.

    Returns (alpha, per_feature_contribution).
    """
    if state is None:
        return 0.0, {k: 0.0 for k in EMPIRICAL_FEATURE_KEYS}
    feats = extract_empirical_features_from_last(last)
    parts: Dict[str, float] = {}
    s = 0.0
    for k in EMPIRICAL_FEATURE_KEYS:
        v = float(feats[k])
        exp = state.expected_return(k, v)
        parts[k] = float(exp)
        s += exp
    cap = float(clip_abs)
    if cap > 0 and np.isfinite(cap):
        s = float(np.clip(s, -cap, cap))
    return float(s), parts


def empirical_alpha_to_score_delta(
    alpha: float,
    *,
    scale: float = 500.0,
    max_delta: float = 2.0,
) -> float:
    """Map small return alpha to a bounded score perturbation for blending with V2 score."""
    if not np.isfinite(alpha):
        return 0.0
    d = float(alpha) * float(scale)
    return float(np.clip(d, -float(max_delta), float(max_delta)))
