"""Smoke tests for post-audit TTM refinements (alignment, failure depth, adaptive logging, presets)."""

from __future__ import annotations

import numpy as np
import pytest

from src.backtest.data_fetcher import OhlcBar
from src.strategies.ttm.config import (
    TTM_CONFIG,
    build_ttm_live_adaptive_config,
    build_ttm_research_parallel_config,
    is_ttm_adaptive_learning_enabled,
)
from src.strategies.ttm.ttm_alignment import TTMAlignmentError, validate_ttm_input_alignment
from src.strategies.ttm.ttm_features import (
    compute_ttm_features,
    compute_ttm_features_from_config,
    features_last_row,
)
from src.strategies.ttm.ttm_signal_v2 import generate_ttm_signal_v2
from src.strategies.ttm.ttm_adaptive_context import TTMAdaptiveContext, build_trade_record_from_snapshot


def _bars_uptrend(n: int) -> list[OhlcBar]:
    out: list[OhlcBar] = []
    p = 1000.0
    for i in range(n):
        p += 0.5
        out.append(
            OhlcBar(
                symbol="X",
                time=str(i),
                open=p - 0.2,
                high=p + 0.5,
                low=p - 0.5,
                close=p,
                volume=1.0,
                unix_ts=1_700_000_000 + i * 60,
            )
        )
    return out


def test_strict_alignment_rejects_missing_basis() -> None:
    bars = _bars_uptrend(25)
    data = {"bars": bars}
    with pytest.raises(TTMAlignmentError, match="basis"):
        compute_ttm_features(
            dict(data),
            breakout_window=10,
            failure_window=3,
            strict_basis_oi=True,
            use_open_interest=True,
        )


def test_strict_alignment_requires_oi_length() -> None:
    bars = _bars_uptrend(25)
    n = len(bars)
    data = {
        "bars": bars,
        "basis": [0.01] * n,
        "open_interest": [1000.0, 1000.0],
    }
    with pytest.raises(TTMAlignmentError, match="open_interest length"):
        compute_ttm_features(
            dict(data),
            breakout_window=10,
            failure_window=3,
            strict_basis_oi=True,
            use_open_interest=True,
        )


def test_failure_strength_continuous_not_only_zero_one() -> None:
    """After a breakout, pullback below the breakout bar's rolling high yields fractional depth."""
    bars: list[OhlcBar] = []
    # Plateau so rolling high is stable, then spike above it, then close back inside range.
    plateau = 1000.0
    for i in range(15):
        bars.append(
            OhlcBar(
                symbol="X",
                time=str(i),
                open=plateau,
                high=plateau + 0.5,
                low=plateau - 0.5,
                close=plateau,
                volume=1.0,
                unix_ts=1_700_000_000 + i * 60,
            )
        )
    bars.append(
        OhlcBar(
            symbol="X",
            time="brk",
            open=plateau,
            high=plateau + 5.0,
            low=plateau,
            close=plateau + 3.0,
            volume=1.0,
            unix_ts=1_700_000_000 + 20 * 60,
        )
    )
    bars.append(
        OhlcBar(
            symbol="X",
            time="fail",
            open=plateau,
            high=plateau + 0.5,
            low=plateau - 0.5,
            close=plateau,
            volume=1.0,
            unix_ts=1_700_000_000 + 21 * 60,
        )
    )
    n = len(bars)
    data = {
        "bars": bars,
        "basis": [0.0] * n,
        "open_interest": [1000.0 + i for i in range(n)],
    }
    feats = compute_ttm_features(
        dict(data),
        breakout_window=8,
        failure_window=5,
        strict_basis_oi=False,
    )
    fs = feats["failure_strength"]
    assert float(fs[-1]) > 0.0
    assert float(fs[-1]) < 0.99


def test_generate_v2_includes_fingerprint_and_profile() -> None:
    bars = _bars_uptrend(30)
    n = len(bars)
    cfg = {
        **TTM_CONFIG,
        "ttm_config_profile": "test_profile",
        "breakout_window": 10,
    }
    data = {
        "bars": bars,
        "basis": [0.01] * n,
        "open_interest": [1000.0 + i * 0.1 for i in range(n)],
    }
    out = generate_ttm_signal_v2(data, cfg, position_side=None, live_mode=False)
    dbg = out.get("debug") or {}
    assert dbg.get("ttm_config_profile") == "test_profile"
    assert isinstance(dbg.get("input_aligned_fingerprint"), str)
    assert len(dbg.get("input_aligned_fingerprint", "")) >= 8


def test_research_parallel_preset() -> None:
    c = build_ttm_research_parallel_config()
    assert c["ttm_config_profile"] == "research_parallel"
    assert c["adaptive_enabled"] is False
    assert c["ttm_adaptive_enabled"] is False
    assert is_ttm_adaptive_learning_enabled(c) is False
    assert c["ttm_strict_basis_oi"] is False


def test_live_adaptive_preset_strict() -> None:
    c = build_ttm_live_adaptive_config()
    assert c["ttm_config_profile"] == "live_adaptive"
    assert c["adaptive_enabled"] is True
    assert c["ttm_adaptive_enabled"] is True
    assert is_ttm_adaptive_learning_enabled(c) is True
    assert c["ttm_strict_basis_oi"] is True
    assert c["ttm_strict_feature_finite"] is True


def test_adaptive_learning_disabled_by_default() -> None:
    assert TTM_CONFIG.get("adaptive_enabled") is False
    assert TTM_CONFIG.get("ttm_adaptive_enabled") is False
    assert is_ttm_adaptive_learning_enabled(TTM_CONFIG) is False


def test_ttm_adaptive_key_overrides_alias() -> None:
    assert (
        is_ttm_adaptive_learning_enabled({"adaptive_enabled": True, "ttm_adaptive_enabled": False}) is False
    )
    assert is_ttm_adaptive_learning_enabled({"adaptive_enabled": False, "ttm_adaptive_enabled": True}) is True


def test_adaptive_update_debug_includes_regime_counts() -> None:
    cfg = {
        **build_ttm_live_adaptive_config(
            {"ttm_adaptive_min_trades": 5, "ttm_adaptive_update_every_n_trades": 4}
        ),
    }
    ctx = TTMAdaptiveContext(cfg)
    base = {
        "entry_price": 100.0,
        "side": "LONG",
        "debug": {
            "score_long": 0.1,
            "score_short": -0.1,
            "prob_long": 0.6,
            "prob_short": 0.4,
            "ttm_regime_id": 0,
            "score_components": {"momentum": 0.5, "basis_effect": 0.0, "oi_effect": 0.0},
        },
        "entry_time": "",
    }
    for _ in range(4):
        ctx.on_trade_closed(
            build_trade_record_from_snapshot(base, exit_price=101.0, pnl=1.0, holding_bars=1)
        )
    dbg = ctx.get_last_update_debug()
    assert dbg is not None
    assert "regime_trade_counts_window" in dbg
    assert dbg["regime_trade_counts_window"].get(0, 0) >= 4


def test_validate_ttm_input_alignment_passes() -> None:
    bars = _bars_uptrend(10)
    n = len(bars)
    validate_ttm_input_alignment(
        {"bars": bars, "basis": [0.0] * n, "open_interest": [1.0] * n},
        strict_basis_oi=True,
        use_open_interest=True,
    )


def test_compute_from_config_matches_strict_flag() -> None:
    bars = _bars_uptrend(22)
    n = len(bars)
    cfg = {**TTM_CONFIG, "ttm_strict_basis_oi": True}
    data = {"bars": bars, "basis": [0.0] * n, "open_interest": [1.0] * n}
    feats = compute_ttm_features_from_config(dict(data), cfg)
    assert int(feats["n"]) == n


def test_oi_signal_tanh_has_spread_when_oi_levels_vary() -> None:
    """Varying OI levels → non-degenerate oi_zscore / oi_signal (level z-score, not bar Δ)."""
    bars = _bars_uptrend(60)
    n = len(bars)
    oi = 1000.0 + 50.0 * np.sin(np.linspace(0.0, 4.0, n))
    data = {"bars": bars, "basis": [0.0] * n, "open_interest": oi}
    feats = compute_ttm_features_from_config(dict(data), {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True})
    sig = np.asarray(feats["oi_signal"], dtype=np.float64)
    assert float(np.nanstd(sig)) > 1e-6
    last = features_last_row(feats)
    assert "oi_pipeline" in last
    assert set(last["oi_pipeline"].keys()) == {"oi", "oi_zscore", "oi_signal"}
    assert abs(last["oi_delta"] - (last["oi_level"] - last["prev_oi"])) < 1e-9


def test_strict_score_rejects_nan_features() -> None:
    from src.strategies.ttm.ttm_types import TTMFeatureVector, SCORING_FEATURE_KEYS
    from src.strategies.ttm.ttm_score import compute_score

    vm = {k: True for k in SCORING_FEATURE_KEYS}
    fv = TTMFeatureVector(
        float("nan"),
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        vm,
    )
    with pytest.raises(ValueError, match="non-finite"):
        compute_score(fv, {**TTM_CONFIG, "ttm_strict_feature_finite": True})
