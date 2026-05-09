from __future__ import annotations

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_v2_short import (
    build_exhaustion_short_entry_meta,
    build_short_opportunity_entry_meta,
    check_exhaustion_short_exit,
    check_short_v3_structural_exit,
    is_exhaustion_short_meta,
)


def test_build_exhaustion_short_entry_meta_from_confirmed_setup() -> None:
    feat = {
        "exhaustion_confirm": True,
        "short_setup_high": 110.0,
        "short_setup_atr": 2.0,
        "short_score": 1.5,
        "short_setup_raw_strength": 3.5,
        "short_setup_cap": 2.0,
        "short_setup_bar_index": 77.0,
    }
    meta = build_exhaustion_short_entry_meta(feat, 108.0, TTM_CONFIG)
    assert meta is not None
    assert is_exhaustion_short_meta(meta) is True
    assert meta["short_stop_loss_price"] == 111.0
    assert meta["short_take_profit_price"] == 105.0
    assert meta["short_time_stop_bars"] == int(TTM_CONFIG["ttm_v2_exhaustion_short_time_stop_bars"])


def test_build_short_opportunity_entry_meta_requires_flag_and_fields() -> None:
    feat = {
        "ttm_v2_short_setup": "short_opportunity_v3",
        "rolling_high": 112.0,
        "atr": 1.5,
    }
    meta = build_short_opportunity_entry_meta(feat, 109.0, TTM_CONFIG)
    assert meta is not None
    assert meta["ttm_v2_short_setup"] == "short_opportunity_v3"
    assert meta["short_ref_high_at_entry"] == 112.0
    assert meta["short_stop_loss_price"] > 112.0


def test_exhaustion_short_exit_also_applies_to_short_opportunity_v3_meta() -> None:
    feat = {
        "ttm_v2_short_setup": "short_opportunity_v3",
        "rolling_high": 110.0,
        "atr": 2.0,
    }
    meta = build_short_opportunity_entry_meta(feat, 108.0, TTM_CONFIG)
    assert meta is not None
    assert check_exhaustion_short_exit(meta, meta["short_stop_loss_price"] + 0.1, 1)["trigger"] == "SL"


def test_short_v3_structural_reclaim_exit() -> None:
    meta = {
        "ttm_v2_short_setup": "short_opportunity_v3",
        "short_ref_high_at_entry": 100.0,
    }
    last = {"close": 100.8, "breakout_up": False, "momentum_1_z": 0.0}
    out = check_short_v3_structural_exit(meta, 100.5, 1, last, {**TTM_CONFIG, "ttm_v2_short_downside_decay_threshold": 0.0})
    assert out is not None
    assert out["trigger"] == "RECLAIM"


def test_check_exhaustion_short_exit_rules() -> None:
    meta = build_exhaustion_short_entry_meta(
        {
            "exhaustion_confirm": True,
            "short_setup_high": 110.0,
            "short_setup_atr": 2.0,
            "short_score": 1.5,
        },
        108.0,
        TTM_CONFIG,
    )
    assert meta is not None
    assert check_exhaustion_short_exit(meta, 111.25, 1)["reason"] == "ttm_exit_stop_loss_short_atr"
    assert check_exhaustion_short_exit(meta, 104.5, 1)["reason"] == "ttm_exit_take_profit_short_atr"
    assert check_exhaustion_short_exit(meta, 107.0, int(meta["short_time_stop_bars"]))["reason"] == "ttm_exit_max_bars_short_atr"
