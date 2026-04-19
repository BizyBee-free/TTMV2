"""Tests for TTM empirical feature store, calibration, engine, and alpha."""

from __future__ import annotations

import numpy as np
import pytest

from src.strategies.ttm.empirical.adaptive_engine import EmpiricalAlphaEngine
from src.strategies.ttm.empirical.alpha import compute_empirical_alpha, empirical_alpha_to_score_delta
from src.strategies.ttm.empirical.calibration import (
    assign_bin,
    build_curves_from_labeled_rows,
    smooth_curves_ema,
)
from src.strategies.ttm.empirical.execution import decide_execution, position_size_from_alpha
from src.strategies.ttm.empirical.feature_store import (
    FeatureStoreRow,
    RollingFeatureStore,
    extract_empirical_features_from_last,
    last_dict_from_decision_features,
)


def test_extract_empirical_features_signed_breakout() -> None:
    last = {
        "breakout_up": True,
        "breakout_down": False,
        "breakout_strength_base": 0.5,
        "basis_signal": 0.1,
        "positioning_strength": -0.2,
        "vol_zscore": 1.0,
    }
    f = extract_empirical_features_from_last(last)
    assert f["breakout"] == pytest.approx(0.5)
    assert f["basis"] == pytest.approx(0.1)
    assert f["vol"] == pytest.approx(1.0)


def test_rolling_store_labels_no_future_leak() -> None:
    st = RollingFeatureStore(window_size=100, forward_horizon=4)
    last = {
        "breakout_up": False,
        "breakout_down": False,
        "breakout_strength_base": 0.0,
        "basis_signal": 0.0,
        "positioning_strength": 0.0,
        "vol_zscore": 0.0,
    }
    for i in range(10):
        c = 1000.0 + float(i)
        st.append(i, i, last, c)
    labeled = st.labeled_rows()
    assert len(labeled) == 6
    for r in labeled:
        assert r.bar_index <= 5
        assert r.is_labeled()
        exp = (1000.0 + float(r.bar_index + 4) - (1000.0 + float(r.bar_index))) / abs(
            1000.0 + float(r.bar_index)
        )
        assert r.forward_return_4 == pytest.approx(exp, rel=1e-9)


def test_build_curves_and_assign_bin() -> None:
    rows: list[FeatureStoreRow] = []
    for i in range(80):
        rows.append(
            FeatureStoreRow(
                bar_index=i,
                timestamp=i,
                breakout=float(i),
                basis=0.0,
                positioning=0.0,
                vol=0.0,
                close=1000.0,
                forward_return_4=float(np.sin(i / 10.0)) * 0.001,
            )
        )
    state = build_curves_from_labeled_rows(rows, n_bins=5, min_bin_samples=5)
    assert state.n_labeled == 80
    e = state.edges["breakout"]
    assert e.size == 4
    b = assign_bin(50.0, e)
    assert 0 <= b < 5


def test_smooth_curves_ema() -> None:
    rows = [
        FeatureStoreRow(0, 0, 1.0, 0.0, 0.0, 0.0, 100.0, 0.001),
        FeatureStoreRow(1, 1, -1.0, 0.0, 0.0, 0.0, 100.0, -0.001),
    ] * 40
    a = build_curves_from_labeled_rows(rows, n_bins=5, min_bin_samples=3)
    b = build_curves_from_labeled_rows(rows, n_bins=5, min_bin_samples=3)
    s = smooth_curves_ema(a, b, alpha_old=0.5)
    assert s.n_bins == 5


def test_empirical_alpha_engine_rebuild() -> None:
    cfg = {
        "ttm_v2_empirical_window_size": 120,
        "ttm_v2_empirical_forward_horizon": 4,
        "ttm_v2_empirical_n_bins": 5,
        "ttm_v2_empirical_min_bin_samples": 5,
        "ttm_v2_empirical_update_every_bars": 10,
        "ttm_v2_empirical_ema_old_weight": 0.5,
    }
    eng = EmpiricalAlphaEngine(cfg)
    last = {
        "breakout_up": False,
        "breakout_down": False,
        "breakout_strength_base": 0.1,
        "basis_signal": 0.0,
        "positioning_strength": 0.0,
        "vol_zscore": 0.0,
    }
    for i in range(50):
        eng.on_bar(i, i, last, 1000.0 + 0.01 * i)
    eng.force_rebuild()
    assert eng.calibration is not None
    assert eng.calibration.n_labeled >= 1


def test_compute_empirical_alpha_none_state() -> None:
    a, p = compute_empirical_alpha({"breakout_strength_base": 1.0}, None)
    assert a == 0.0
    assert p["breakout"] == 0.0


def test_empirical_alpha_to_score_delta() -> None:
    d = empirical_alpha_to_score_delta(0.002, scale=500.0, max_delta=2.0)
    assert d == pytest.approx(1.0)


def test_decide_execution_and_position_size() -> None:
    d_long = decide_execution(0.002, cost_roundtrip=0.0004, margin=0.0001)
    assert d_long.side == "LONG"
    assert d_long.size >= 0.5
    d_skip = decide_execution(0.0001, cost_roundtrip=0.0004, margin=0.0001)
    assert d_skip.side == "SKIP"
    sz = position_size_from_alpha(0.003, max_alpha_ref=0.003)
    assert sz == pytest.approx(1.0)


def test_last_dict_from_decision_features_minimal() -> None:
    feats = {
        "is_breakout_up": True,
        "is_breakout_down": False,
        "breakout_strength_base": 0.3,
        "positioning_log": {"positioning_strength": 0.5, "basis_signal": 0.1},
        "volume_participation_log": {"vol_zscore": 0.8},
    }
    last = last_dict_from_decision_features(feats)
    assert last["breakout_up"] is True
    assert extract_empirical_features_from_last(last)["vol"] == pytest.approx(0.8)
