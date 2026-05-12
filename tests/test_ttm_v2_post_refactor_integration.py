"""Post-refactor 20260511: LONG gate labels, exit reason refinement, entry-block classification."""

from __future__ import annotations

from src.strategies.ttm.ttm_parallel_runner import (
    _classify_v2_blocked,
    _refine_v2_trade_exit_reason,
)


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
