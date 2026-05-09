"""Unit tests for refactor_4 continuation-aware exit ordering."""

from __future__ import annotations

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_v2_exit_continuation import (
    decide_long_exit_continuation,
    map_parallel_runner_reason_to_canonical,
)


def test_long_hard_stop_before_soft_min():
    cfg = {
        **TTM_CONFIG,
        "ttm_v2_enable_soft_min_hold": True,
        "ttm_v2_soft_min_hold_bars": 3,
        "ttm_v2_sl_return": -0.001,
    }
    last = {"breakout_up": True, "failure_strength": 0.0, "exhaustion_confirm": False}
    comp = {"continuation_confirm": 0.9, "crowd_phase": "early_continuation", "rejection_confirm": 0.0}
    act, reason, _ = decide_long_exit_continuation(
        ok=True,
        pl=0.9,
        prob_exit=0.4,
        score_long=2.0,
        last=last,
        components=comp,
        cfg=cfg,
        holding_bars=0,
        unrealized_return=-0.02,
        entry_snapshot={"entry_score_long": 3.0},
    )
    assert act == "EXIT"
    assert reason == "hard_stop"


def test_soft_min_hold_blocks_score_decay_when_continuation_valid():
    cfg = {
        **TTM_CONFIG,
        "ttm_v2_enable_soft_min_hold": True,
        "ttm_v2_soft_min_hold_bars": 3,
        "ttm_v2_continuation_exit_threshold": 0.0,
    }
    last = {"breakout_up": True, "failure_strength": 0.0, "exhaustion_confirm": False}
    comp = {"continuation_confirm": 0.8, "crowd_phase": "early_continuation", "rejection_confirm": 0.0}
    act, reason, extra = decide_long_exit_continuation(
        ok=True,
        pl=0.1,
        prob_exit=0.4,
        score_long=2.0,
        last=last,
        components=comp,
        cfg=cfg,
        holding_bars=1,
        unrealized_return=-0.0001,
        entry_snapshot={"entry_score_long": 3.0},
    )
    assert act == "HOLD"
    assert reason == "soft_min_hold_continuation"
    assert extra.get("v2_hold_reason") == "soft_min_hold_continuation"


def test_map_parallel_runner_long_stop_to_hard_stop():
    assert (
        map_parallel_runner_reason_to_canonical(
            "LONG", "ttm_exit_stop_loss_parallel", "stop_loss_return"
        )
        == "hard_stop"
    )


def test_map_prob_decay_long_to_score_decay():
    assert map_parallel_runner_reason_to_canonical("LONG", "prob_decay", "prob") == "score_decay"
