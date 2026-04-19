"""TTM model V2: bridge to :mod:`ttm_score` (backward-compatible helpers)."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from src.strategies.ttm.ttm_score import compute_score, probs_valid, scores_to_probs
from src.strategies.ttm.ttm_types import SCORING_FEATURE_KEYS, TTMFeatureVector


def sigmoid_score(x: float, temperature: float = 1.0) -> float:
    """Map scalar to (0,1); kept for tests / legacy callers."""
    if not np.isfinite(x):
        return float("nan")
    t = max(1e-12, float(temperature))
    z = float(x) / t
    return float(1.0 / (1.0 + np.exp(-z)))


def compute_directional_probs_at_last(
    *,
    breakout_strength: float,
    fu: bool,
    fd: bool,
    basis_norm: float,
    oi_signal: float,
    use_oi: bool,
    config: Mapping[str, Any],
) -> Tuple[float, float, float, float, bool]:
    """
    Legacy wrapper for tests: minimal :class:`TTMFeatureVector` (no basis_delta / vol_regime).
    """
    oi = oi_signal if use_oi else 0.0
    fs = max(float(fu), float(fd))
    vm = {k: True for k in SCORING_FEATURE_KEYS}
    if not use_oi:
        vm["oi_signal"] = False
    fv = TTMFeatureVector(
        breakout_strength=breakout_strength,
        failure_strength=fs,
        basis_norm=basis_norm,
        basis_delta=0.0,
        oi_signal=oi,
        vol_regime=0,
        valid_mask=vm,
    )
    sl, ss, _ = compute_score(fv, config)
    pl, ps = scores_to_probs(sl, ss, config)
    ok = probs_valid(pl, ps)
    return float(sl), float(ss), float(pl), float(ps), ok
