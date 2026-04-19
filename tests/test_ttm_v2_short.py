from __future__ import annotations

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_v2_short import (
    build_exhaustion_short_entry_meta,
    check_exhaustion_short_exit,
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
