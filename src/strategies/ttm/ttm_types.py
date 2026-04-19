"""Strict feature contract for TTM scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

# Keys used by :func:`src.strategies.ttm.ttm_score.compute_score`
SCORING_FEATURE_KEYS = (
    "breakout_strength",
    "failure_strength",
    "basis_norm",
    "basis_delta",
    "oi_signal",
    "vol_regime",
)


@dataclass(frozen=True)
class TTMFeatureVector:
    breakout_strength: float
    failure_strength: float
    basis_norm: float
    basis_delta: float
    oi_signal: float
    vol_regime: int
    valid_mask: Dict[str, bool]


def _clamp_vol_regime(x: float) -> int:
    if not np.isfinite(x):
        return 0
    if x <= -0.5:
        return -1
    if x >= 0.5:
        return 1
    return 0


def feature_vector_from_last(
    last: Mapping[str, Any],
    *,
    use_oi: bool,
    oi_signal_live: Optional[float] = None,
) -> TTMFeatureVector:
    """
    Build a :class:`TTMFeatureVector` from ``features_last_row`` output.

    All scalar inputs must already be finite (0.0 when invalid); ``valid_mask`` encodes trust.
    ``oi_signal_live`` overrides ``last['oi_signal']`` when ``use_oi`` (e.g. signed_oi from closes).
    """
    vm = last.get("feature_valid_mask")
    if not isinstance(vm, dict):
        vm = {k: True for k in SCORING_FEATURE_KEYS}

    def _mask(key: str) -> bool:
        v = vm.get(key)
        return bool(v) if v is not None else True

    bs = float(last.get("breakout_strength") or 0.0)
    fs = float(last.get("failure_strength") or 0.0)
    bn = float(last.get("basis_norm") or 0.0)
    bd = float(last.get("basis_delta") or 0.0)
    oi_raw = oi_signal_live if use_oi and oi_signal_live is not None else float(last.get("oi_signal") or 0.0)
    oi = oi_raw if use_oi else 0.0
    vr_f = float(last.get("vol_regime") or 0.0)

    mask_oi = bool(_mask("oi_signal")) if use_oi else False

    valid_mask = {
        "breakout_strength": _mask("breakout_strength"),
        "failure_strength": _mask("failure_strength"),
        "basis_norm": _mask("basis_norm"),
        "basis_delta": _mask("basis_delta"),
        "oi_signal": mask_oi,
        "vol_regime": _mask("vol_regime"),
    }

    return TTMFeatureVector(
        breakout_strength=bs,
        failure_strength=fs,
        basis_norm=bn,
        basis_delta=bd,
        oi_signal=oi,
        vol_regime=_clamp_vol_regime(vr_f),
        valid_mask=valid_mask,
    )
