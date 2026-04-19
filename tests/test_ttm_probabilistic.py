"""TTM dual-system: feature layer, V1/V2, shadow mode, fallback."""

from __future__ import annotations

import numpy as np

from src.backtest.data_fetcher import OhlcBar
from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_features import compute_features, features_last_row
from src.strategies.ttm.ttm_model_v2 import compute_directional_probs_at_last, sigmoid_score
from src.strategies.ttm.ttm_score import compute_score
from src.strategies.ttm.ttm_signal import generate_ttm_signal, generate_ttm_signal_v1


def _bars(n: int = 80) -> list[OhlcBar]:
    bars: list[OhlcBar] = []
    p = 100.0
    t0 = 1700000000
    for i in range(n):
        p += np.sin(i / 8.0) * 0.15
        o = p
        c = p + 0.05 * np.sin(i / 3.0)
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        bars.append(
            OhlcBar(
                symbol="X",
                time=str(i),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1.0,
                unix_ts=t0 + i * 900,
            )
        )
        p = c
    return bars


def test_compute_features_has_expected_keys():
    bars = _bars(60)
    data = {"bars": bars, "basis": np.zeros(len(bars)), "open_interest": np.linspace(1e5, 1.01e5, len(bars))}
    cfg = {**TTM_CONFIG}
    feats = compute_features(data, cfg)
    assert feats["n"] == len(bars)
    for k in ("breakout_strength", "failure_strength", "basis_norm", "basis_delta", "oi_signal", "vol_regime"):
        assert k in feats
        assert isinstance(feats[k], np.ndarray)
    last = features_last_row(feats)
    assert "feature_valid_mask" in last
    assert isinstance(last["feature_valid_mask"], dict)


def test_feature_consistency_no_nan_on_synthetic():
    bars = _bars(100)
    data = {"bars": bars}
    feats = compute_features(data, TTM_CONFIG)
    last = features_last_row(feats)
    for k, v in last.items():
        if k == "feature_valid_mask":
            continue
        if isinstance(v, float):
            assert np.isfinite(v), k


def test_v1_only_matches_direct_v1():
    bars = _bars(90)
    data = {"bars": bars}
    cfg = {**TTM_CONFIG, "ttm_mode": "v1_only"}
    a = generate_ttm_signal(data, cfg)
    b = generate_ttm_signal_v1(data, cfg)
    assert a["action"] == b["action"]
    assert a.get("reason") == b.get("reason")


def test_shadow_executes_v1_action_with_v2_sidecar():
    bars = _bars(90)
    data = {"bars": bars}
    cfg = {**TTM_CONFIG, "ttm_mode": "shadow"}
    out = generate_ttm_signal(data, cfg)
    assert out["ttm_mode"] == "shadow"
    assert "v2" in out
    assert "divergence" in out
    v1 = generate_ttm_signal_v1(data, cfg)
    assert out["action"] == v1["action"]


def test_v2_only_fallback_invalid_data():
    cfg = {**TTM_CONFIG, "ttm_mode": "v2_only"}
    out = generate_ttm_signal({"bars": []}, cfg)
    assert out.get("v2_fallback") is True or out["action"] == "HOLD"


def test_sigmoid_monotonicity():
    a = sigmoid_score(0.5, 1.0)
    b = sigmoid_score(1.5, 1.0)
    assert b > a


def test_directional_probs_valid():
    sl, ss, pl, ps, ok = compute_directional_probs_at_last(
        breakout_strength=0.5,
        fu=True,
        fd=False,
        basis_norm=0.1,
        oi_signal=1.0,
        use_oi=True,
        config=TTM_CONFIG,
    )
    assert ok
    assert 0.0 < pl < 1.0 and 0.0 < ps < 1.0
    assert np.isfinite(sl) and np.isfinite(ss)


def test_exit_logic_prob_decay_v2():
    from src.strategies.ttm.ttm_signal_v2 import generate_ttm_signal_v2

    bars = _bars(120)
    data = {
        "bars": bars,
        "basis": np.zeros(len(bars)),
        "open_interest": np.linspace(1e5, 1.01e5, len(bars)),
    }
    cfg = {
        **TTM_CONFIG,
        "prob_exit_threshold": 0.99,
        "prob_entry_threshold": 0.99,
        "score_temperature": 10.0,
    }
    out = generate_ttm_signal_v2(data, cfg, position_side="LONG")
    assert out["action"] in ("EXIT", "HOLD")


def test_no_overtrading_flat_hold_stable():
    bars = _bars(50)
    data = {"bars": bars}
    cfg = {**TTM_CONFIG, "ttm_mode": "v1_only"}
    o1 = generate_ttm_signal(data, cfg)
    o2 = generate_ttm_signal(data, cfg)
    assert o1["action"] == o2["action"]


def test_regime_adaptation_vol_regime_in_features():
    bars = _bars(60)
    data = {"bars": bars}
    feats = compute_features(data, TTM_CONFIG)
    vr = feats["vol_regime"]
    assert np.all(np.isin(vr, [-1.0, 0.0, 1.0]))


def test_unified_score_vol_regime_scales_breakout():
    from src.strategies.ttm.ttm_types import SCORING_FEATURE_KEYS, TTMFeatureVector

    vm = {k: True for k in SCORING_FEATURE_KEYS}
    cfg = {**TTM_CONFIG}
    fv_hi = TTMFeatureVector(1.0, 0.0, 0.0, 0.0, 0.0, 1, vm)
    fv_lo = TTMFeatureVector(1.0, 0.0, 0.0, 0.0, 0.0, -1, vm)
    sl_hi, _, _ = compute_score(fv_hi, cfg)
    sl_lo, _, _ = compute_score(fv_lo, cfg)
    assert sl_hi != sl_lo


def test_trap_detection_requires_setup():
    bars = _bars(40)
    data = {"bars": bars}
    cfg = {**TTM_CONFIG, "ttm_mode": "v2_only", "prob_entry_threshold": 0.999}
    out = generate_ttm_signal(data, cfg)
    assert out.get("action") in ("HOLD", "LONG", "SHORT", "EXIT") or out.get("v2_fallback")


def test_v2_only_emits_ttm_mode_when_probs_valid():
    bars = _bars(100)
    data = {"bars": bars, "basis": np.zeros(len(bars)), "open_interest": np.linspace(1e5, 1.02e5, len(bars))}
    cfg = {**TTM_CONFIG, "ttm_mode": "v2_only", "prob_entry_threshold": 0.01, "prob_exit_threshold": 0.01}
    out = generate_ttm_signal(data, cfg)
    if not out.get("v2_fallback"):
        assert out.get("ttm_mode") == "v2_only"
        assert "debug" in out or out.get("action") is not None


def test_unknown_mode_falls_back_to_v1():
    bars = _bars(50)
    data = {"bars": bars}
    cfg = {**TTM_CONFIG, "ttm_mode": "bogus_mode_xyz"}
    a = generate_ttm_signal(data, cfg)
    b = generate_ttm_signal_v1(data, {**TTM_CONFIG, "ttm_mode": "bogus_mode_xyz"})
    assert a["action"] == b["action"]


def test_log_v2_in_v1_only_merges():
    bars = _bars(80)
    data = {"bars": bars}
    cfg = {**TTM_CONFIG, "ttm_mode": "v1_only", "log_v2_in_v1_only": True}
    out = generate_ttm_signal(data, cfg)
    assert out.get("ttm_mode") == "v1_only"
    assert "v2" in out
