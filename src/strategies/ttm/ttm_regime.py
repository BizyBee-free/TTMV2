"""Vol-spike based regime detection for TTM (deterministic)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np


def detect_regime(features: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> int:
    """
    Map last-bar ``vol_spike`` to {-1, 0, 1}: low / neutral / high vol.

    Uses configurable thresholds (defaults 0.8 and 1.2).
    """
    cfg = config or {}
    lo_th = float(cfg.get("ttm_regime_low_threshold", 0.8))
    hi_th = float(cfg.get("ttm_regime_high_threshold", 1.2))
    raw = features.get("vol_spike")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = 1.0
    if not np.isfinite(v):
        v = 1.0
    if v < lo_th:
        return -1
    if v > hi_th:
        return 1
    return 0
