"""Trade-volume z-score, participation, breakout/positioning boost for TTM."""

from __future__ import annotations

import numpy as np

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_features import compute_ttm_features_from_config, features_last_row


def _bars(closes: np.ndarray, volumes: np.ndarray) -> list[dict[str, float]]:
    out = []
    for c, v in zip(closes, volumes):
        x = float(c)
        out.append({"high": x + 0.5, "low": x - 0.5, "close": x, "volume": float(v)})
    return out


def test_vol_signal_not_flat_with_varying_volume() -> None:
    n = 120
    t = np.linspace(0.0, 8.0, n)
    closes = 100.0 + 0.1 * np.arange(n)
    vol = 1_000.0 + 300.0 * np.sin(t) + 50.0 * np.cos(1.3 * t)
    oi = 5000.0 + 200.0 * np.sin(0.5 * t)
    data = {"bars": _bars(closes, vol), "basis": [0.0] * n, "open_interest": oi.tolist()}
    feats = compute_ttm_features_from_config(dict(data), {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True})
    vs = np.asarray(feats["vol_signal"], dtype=np.float64)
    assert vs.size == n
    assert float(np.nanstd(vs)) > 1e-5


def test_vol_signal_distribution_roughly_symmetric() -> None:
    """Oscillating volume around a stable level → rolling z near symmetric → mean(vol_signal) ~ 0 mid-series."""
    n = 200
    rng = np.random.default_rng(7)
    closes = 100.0 + rng.standard_normal(n).cumsum() * 0.02
    center = 5000.0
    vol = center + 800.0 * np.sin(np.linspace(0, 14.0, n))
    oi = center + 100.0 * np.sin(np.linspace(0, 3.0, n))
    data = {"bars": _bars(closes, vol), "basis": [0.0] * n, "open_interest": oi.tolist()}
    feats = compute_ttm_features_from_config(dict(data), {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True})
    vs = np.asarray(feats["vol_signal"], dtype=np.float64)
    mid = vs[80:160]
    assert abs(float(np.mean(mid))) < 0.25


def test_volume_participation_log_schema() -> None:
    n = 60
    closes = 100.0 + np.linspace(0, 1.0, n)
    vol = 2000.0 + 100.0 * np.arange(n)
    oi = 3000.0 + 10.0 * np.arange(n)
    data = {"bars": _bars(closes, vol), "basis": [0.05] * n, "open_interest": oi.tolist()}
    feats = compute_ttm_features_from_config(dict(data), TTM_CONFIG)
    last = features_last_row(feats)
    assert "volume_participation_log" in last
    assert set(last["volume_participation_log"].keys()) == {
        "volume",
        "vol_zscore",
        "vol_signal",
        "participation_strength",
    }


def test_breakout_strength_splits_continuation_from_exhaustion() -> None:
    n = 120
    closes = 100.0 + 1.0 * np.arange(n)
    closes[-1] += 8.0
    vol = 1000.0 + 100.0 * np.sin(np.linspace(0, 4.0, n))
    oi = 2000.0 + 20.0 * np.sin(np.linspace(0, 2.0, n))
    data = {"bars": _bars(closes, vol), "basis": [0.0] * n, "open_interest": oi.tolist()}
    cfg = {
        **TTM_CONFIG,
        "breakout_strength_min": 0.2,
        "ttm_breakout_exhaustion_cap_window": 20,
        "ttm_breakout_exhaustion_cap_q": 0.8,
    }
    feats = compute_ttm_features_from_config(dict(data), cfg)
    vm = feats["feature_valid_mask"]["breakout_strength"]

    assert bool(feats["breakout_up"][n - 2])
    assert bool(vm[n - 2])
    assert np.isclose(
        float(feats["breakout_strength"][n - 2]),
        float(feats["breakout_strength_base"][n - 2]),
    )

    assert bool(feats["exhaustion_candidate"][n - 1])
    assert not bool(feats["breakout_up"][n - 1])
    assert not bool(vm[n - 1])
    assert float(feats["breakout_strength"][n - 1]) == 0.0


def test_exhaustion_confirm_and_short_score_shift_to_confirm_bar() -> None:
    n = 120
    closes = 100.0 + 1.0 * np.arange(n)
    closes[-2] += 8.0
    closes[-1] = closes[-2] - 4.0
    vol = 1000.0 + 100.0 * np.sin(np.linspace(0, 4.0, n))
    oi = 2000.0 + 20.0 * np.sin(np.linspace(0, 2.0, n))
    data = {"bars": _bars(closes, vol), "basis": [0.0] * n, "open_interest": oi.tolist()}
    cfg = {
        **TTM_CONFIG,
        "breakout_strength_min": 0.2,
        "ttm_breakout_exhaustion_cap_window": 20,
        "ttm_breakout_exhaustion_cap_q": 0.8,
    }
    feats = compute_ttm_features_from_config(dict(data), cfg)
    assert bool(feats["exhaustion_candidate"][n - 2])
    assert bool(feats["exhaustion_confirm"][n - 1])
    assert float(feats["short_score"][n - 1]) > 0.0
    assert np.isclose(float(feats["short_setup_high"][n - 1]), float(closes[n - 2] + 0.5))
    assert np.isclose(float(feats["short_setup_atr"][n - 1]), float(feats["atr"][n - 2]))
    assert np.isclose(float(feats["short_setup_raw_strength"][n - 1]), float(feats["raw_strength"][n - 2]))
    assert np.isclose(float(feats["short_setup_cap"][n - 1]), float(feats["cap"][n - 2]))


def test_volume_features_disabled_is_noop() -> None:
    n = 50
    closes = 100.0 + np.arange(n)
    vol = 5000.0 + 500.0 * np.sin(np.linspace(0, 3.0, n))
    oi = 1000.0 + np.arange(n)
    data = {"bars": _bars(closes, vol), "basis": [0.0] * n, "open_interest": oi.tolist()}
    cfg = {**TTM_CONFIG, "ttm_volume_features_enabled": False, "ttm_oi_assert_nonzero_std": True}
    feats = compute_ttm_features_from_config(dict(data), cfg)
    assert np.allclose(feats["vol_signal"], 0.0)
    assert np.allclose(feats["breakout_strength"], feats["breakout_strength_base"])
