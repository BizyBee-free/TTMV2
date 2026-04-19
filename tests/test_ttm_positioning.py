"""Tests for positioning_strength (basis + volume; OI not in core blend)."""

from __future__ import annotations

import numpy as np
import pytest

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_features import (
    compute_positioning_strength,
    compute_ttm_features_from_config,
    features_last_row,
)


def _bars_from_closes(closes: np.ndarray) -> list[dict[str, float]]:
    return [{"high": float(c) + 0.5, "low": float(c) - 0.5, "close": float(c)} for c in closes]


def test_positioning_strength_distribution_not_degenerate() -> None:
    """Must not be all zeros; must include both positive and negative values."""
    n = 120
    t = np.linspace(0.0, 6.28, n)
    closes = 100.0 + 0.4 * np.arange(n) + 2.0 * np.sin(t)
    oi = 10_000.0 + 400.0 * np.sin(0.7 * t)
    basis = 3.0 * np.cos(0.3 * t)
    data = {"bars": _bars_from_closes(closes), "basis": basis.tolist(), "open_interest": oi.tolist()}
    feats = compute_ttm_features_from_config(dict(data), {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True})
    ps = np.asarray(feats["positioning_strength"], dtype=np.float64)
    assert ps.size == n
    assert float(np.nanstd(ps)) > 1e-6
    assert float(np.max(ps)) > 1e-6
    assert float(np.min(ps)) < -1e-6


def test_positioning_log_last_row_schema() -> None:
    n = 40
    closes = 100.0 + np.linspace(0.0, 2.0, n)
    oi = 5000.0 + 100.0 * np.sin(np.linspace(0, 3.0, n))
    data = {"bars": _bars_from_closes(closes), "basis": [0.1] * n, "open_interest": oi.tolist()}
    feats = compute_ttm_features_from_config(dict(data), TTM_CONFIG)
    last = features_last_row(feats)
    assert "positioning_log" in last
    keys = {
        "price_change",
        "price_signal",
        "oi_zscore",
        "oi_signal",
        "pos_core",
        "basis_norm",
        "basis_signal",
        "basis_effect",
        "positioning_strength",
    }
    assert set(last["positioning_log"].keys()) == keys


def test_compute_positioning_strength_basis_vol_blend() -> None:
    """positioning = 0.6 * tanh(clip(basis_norm)) + 0.4 * vol_signal; pos_core unused (zeros)."""
    close = np.array([100.0, 101.0, 100.0, 99.0], dtype=np.float64)
    basis_norm = np.array([0.0, 1.0, -1.0, 0.0], dtype=np.float64)
    vol_signal = np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float64)
    oi_sig = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    oi_z = np.zeros(4)
    out = compute_positioning_strength(
        close, basis_norm, vol_signal, oi_sig, oi_z, w_basis=0.6, w_vol=0.4
    )
    assert out["price_signal"][1] == 1.0
    assert out["price_signal"][2] == -1.0
    assert np.allclose(out["pos_core"], 0.0)
    b1 = float(np.tanh(1.0))
    assert abs(out["positioning_strength"][1] - (0.6 * b1 + 0.4 * 0.5)) < 1e-9
    b2 = float(np.tanh(-1.0))
    assert abs(out["positioning_strength"][2] - (0.6 * b2 + 0.4 * (-0.5))) < 1e-9


def test_correlation_forward_return_not_identically_zero() -> None:
    """Offline sanity: Pearson corr vs 1-bar forward return is finite and not exactly 0."""
    n = 200
    rng = np.random.default_rng(42)
    walk = rng.standard_normal(n).cumsum()
    closes = 100.0 + walk + 0.02 * np.arange(n)
    oi = 8000.0 + 200.0 * np.sin(np.linspace(0, 12.0, n)) + rng.standard_normal(n) * 5.0
    basis = 2.0 * np.sin(np.linspace(0, 8.0, n))
    data = {"bars": _bars_from_closes(closes), "basis": basis.tolist(), "open_interest": oi.tolist()}
    feats = compute_ttm_features_from_config(dict(data), {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True})
    close = np.asarray(feats["close"], dtype=np.float64).ravel()
    ps = np.asarray(feats["positioning_strength"], dtype=np.float64).ravel()
    fwd = np.diff(close) / np.maximum(close[:-1], 1e-12)
    a, b = ps[:-1], fwd
    if a.std() < 1e-12 or b.std() < 1e-12:
        pytest.skip("degenerate variance in synthetic draw")
    r = float(np.corrcoef(a, b)[0, 1])
    assert np.isfinite(r)
    assert abs(r) > 1e-12


def test_oi_context_flag_spike() -> None:
    n = 30
    closes = 100.0 + np.linspace(0, 0.5, n)
    oi = [5000.0 + i * 10.0 for i in range(n)]
    data = {"bars": _bars_from_closes(np.asarray(closes)), "basis": [0.0] * n, "open_interest": oi}
    cfg = {
        **TTM_CONFIG,
        "ttm_oi_assert_nonzero_std": True,
        "oi_day_change_threshold": 100.0,
    }
    raw = dict(data)
    raw["oi_prev_day_close"] = float(oi[-1]) - 500.0
    feats = compute_ttm_features_from_config(raw, cfg)
    assert feats.get("oi_context_flag") == "OI_SPIKE"


def test_oi_context_flag_normal_without_prev() -> None:
    n = 20
    closes = 100.0 + np.linspace(0, 0.2, n)
    oi = [4000.0] * n
    data = {"bars": _bars_from_closes(np.asarray(closes)), "basis": [0.0] * n, "open_interest": oi}
    feats = compute_ttm_features_from_config(dict(data), TTM_CONFIG)
    assert feats.get("oi_context_flag") == "NORMAL"
