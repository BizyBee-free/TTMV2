"""Post-refactor 20260511: LONG gate labels, exit reason refinement, entry-block classification."""

from __future__ import annotations

import numpy as np

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_features import features_last_row
from src.strategies.ttm.ttm_parallel_runner import (
    _classify_v2_blocked,
    _refine_v2_trade_exit_reason,
)
from src.strategies.ttm.ttm_score import compute_score_v2_alpha
from src.strategies.ttm.ttm_signal_v2 import generate_ttm_signal_v2


def test_classify_v2_blocked_score_long_below() -> None:
    assert _classify_v2_blocked("HOLD", "score_long_below_threshold", {}) == "score_long_below_threshold"


def test_classify_v2_blocked_long_gate() -> None:
    assert _classify_v2_blocked("HOLD", "long_gate_no_valid_breakout", {}) == "long_gate"


def test_refine_exit_score_decay_vs_no_breakout_after_entry() -> None:
    assert (
        _refine_v2_trade_exit_reason(
            "score_decay",
            entry_crowd_phase="ignition",
            exit_crowd_phase="no_breakout",
            dbg_exit={},
        )
        == "no_breakout_after_entry"
    )
    assert (
        _refine_v2_trade_exit_reason(
            "score_decay",
            entry_crowd_phase="ignition",
            exit_crowd_phase="ignition",
            dbg_exit={},
        )
        == "score_decay_exit"
    )


def test_refine_exit_failed_breakout_suffix() -> None:
    assert (
        _refine_v2_trade_exit_reason(
            "failed_breakout",
            entry_crowd_phase="ignition",
            exit_crowd_phase="failed_breakout",
            dbg_exit={},
        )
        == "failed_breakout_exit"
    )


def test_v3_score_long_uses_v3_effective_strength_not_legacy_feature() -> None:
    n = 30
    bu = np.zeros(n, dtype=bool)
    bu[-1] = True
    feats = {
        "n": n,
        "close": np.linspace(100.0, 103.0, n),
        "breakout_up": bu,
        "breakout_strength_base": np.zeros(n),
        "raw_strength": np.full(n, -3.0),
        # Legacy feature is strongly positive; v3 should ignore it when enabled.
        "effective_strength": np.full(n, 3.0),
        "momentum_1_z": np.zeros(n),
        "basis_signal": np.zeros(n),
        "basis_norm": np.zeros(n),
        "price_z": np.zeros(n),
        "extension": np.zeros(n),
        "last_bar_return": np.zeros(n),
        "failure_depth_up": np.zeros(n),
        "short_score": np.zeros(n),
        "exhaustion_confirm": np.zeros(n, dtype=bool),
    }
    last = features_last_row(feats)
    cfg = {**TTM_CONFIG, "ttm_v2_use_effective_strength_v3": True}

    score_long, _score_short, comp = compute_score_v2_alpha(feats, last, cfg)

    assert score_long < 0.0
    assert abs(score_long - float(comp["score_long"])) < 1e-9
    assert abs(float(comp["score_long"]) - np.tanh(float(comp["effective_strength_v3"]))) < 1e-9
    assert float(comp["effective_strength_last"]) < 0.0


def test_short_no_short_context_never_enters_even_with_exhaustion_signal() -> None:
    n = 30
    feats = {
        "n": n,
        "close": np.linspace(100.0, 99.0, n),
        "breakout_up": np.zeros(n, dtype=bool),
        "breakout_strength_base": np.zeros(n),
        "raw_strength": np.zeros(n),
        "effective_strength": np.zeros(n),
        "momentum_1_z": np.zeros(n),
        "basis_signal": np.zeros(n),
        "basis_norm": np.zeros(n),
        "price_z": np.zeros(n),
        "extension": np.zeros(n),
        "last_bar_return": np.full(n, -0.001),
        "failure_depth_up": np.zeros(n),
        "short_score": np.ones(n),
        "exhaustion_confirm": np.ones(n, dtype=bool),
    }
    cfg = {
        **TTM_CONFIG,
        "ttm_v2_enable_short_trading": True,
        "ttm_v2_use_short_effective_strength_v3": False,
    }

    sig = generate_ttm_signal_v2(feats, cfg)

    assert sig["action"] == "HOLD"
    dbg = sig.get("debug") or {}
    assert dbg.get("short_phase") == "no_short_context"
    assert dbg.get("short_signal") is False
    assert sig.get("reason") in {"no_prior_upside_breakout", "insufficient_setup"}
