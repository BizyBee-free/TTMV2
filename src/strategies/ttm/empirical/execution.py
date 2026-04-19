"""Cost-aware execution helpers for empirical alpha (automate_tunning.md §4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Side = Literal["LONG", "SHORT", "SKIP"]


@dataclass
class ExecutionDecision:
    side: Side
    size: float
    alpha: float
    threshold_used: float
    cost_used: float
    margin_used: float
    reason: str


def position_size_from_alpha(
    alpha: float,
    *,
    max_alpha_ref: float = 0.003,
    size_min: float = 0.5,
    size_max: float = 2.0,
) -> float:
    """``size = clip(|alpha| / max_alpha_ref, size_min, size_max)`` — doc §4.3 style."""
    if not np.isfinite(alpha) or max_alpha_ref <= 1e-12:
        return float(size_min)
    ra = float(max_alpha_ref)
    raw = abs(float(alpha)) / ra
    return float(np.clip(raw, float(size_min), float(size_max)))


def decide_execution(
    alpha: float,
    *,
    cost_roundtrip: float = 0.0004,
    margin: float = 0.0001,
    no_trade_abs: float = 0.0,
    max_alpha_ref: float = 0.003,
    size_min: float = 0.5,
    size_max: float = 2.0,
) -> ExecutionDecision:
    """
    LONG if alpha > cost + margin; SHORT if alpha < -(cost + margin); else SKIP.
    ``no_trade_abs``: extra dead-zone on |alpha| (doc §4.2).
    """
    c = float(cost_roundtrip) + float(margin)
    thr = c
    a = float(alpha) if np.isfinite(alpha) else 0.0
    nz = float(no_trade_abs)
    if nz > 0 and abs(a) < nz:
        return ExecutionDecision(
            "SKIP",
            0.0,
            a,
            thr,
            float(cost_roundtrip),
            float(margin),
            "no_trade_zone",
        )
    if a > thr:
        sz = position_size_from_alpha(a, max_alpha_ref=max_alpha_ref, size_min=size_min, size_max=size_max)
        return ExecutionDecision("LONG", sz, a, thr, float(cost_roundtrip), float(margin), "alpha_gt_cost")
    if a < -thr:
        sz = position_size_from_alpha(abs(a), max_alpha_ref=max_alpha_ref, size_min=size_min, size_max=size_max)
        return ExecutionDecision("SHORT", sz, a, thr, float(cost_roundtrip), float(margin), "alpha_lt_neg_cost")
    return ExecutionDecision("SKIP", 0.0, a, thr, float(cost_roundtrip), float(margin), "inside_spread")
