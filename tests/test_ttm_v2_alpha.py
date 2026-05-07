"""TTM V2 alpha: continuation-long and exhaustion-short use separate alpha legs."""

from __future__ import annotations

import numpy as np

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_features import compute_ttm_features_from_config, features_last_row
from src.strategies.ttm.ttm_score import compute_score_v2_alpha


def _bars(closes: np.ndarray, volumes: np.ndarray) -> list[dict[str, float]]:
    out = []
    for c, v in zip(closes, volumes):
        x = float(c)
        out.append({"high": x + 0.5, "low": x - 0.5, "close": x, "volume": float(v)})
    return out


def _feats_up_to(feats: dict, i: int) -> dict:
    """Prefix features so bar index i is the last bar (for historical replay)."""
    out: dict = {}
    for k, v in feats.items():
        if k == "n":
            continue
        if isinstance(v, np.ndarray):
            out[k] = np.asarray(v[: i + 1]).copy()
        else:
            out[k] = v
    out["n"] = int(i + 1)
    return out


def test_v2_alpha_ignores_oi_for_score() -> None:
    """Different OI paths → identical V2 score when price/vol/basis series match."""
    n = 80
    t = np.linspace(0, 4.0, n)
    closes = 100.0 + 0.2 * np.arange(n)
    vol = 2000.0 + 500.0 * np.sin(t)
    basis = 0.5 * np.sin(0.5 * t)
    oi_a = [3000.0 + 50.0 * i for i in range(n)]
    oi_b = [8000.0 + 80.0 * i for i in range(n)]
    cfg = {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True}
    data_a = {"bars": _bars(closes, vol), "basis": basis.tolist(), "open_interest": oi_a}
    data_b = {"bars": _bars(closes, vol), "basis": basis.tolist(), "open_interest": oi_b}
    feats_a = compute_ttm_features_from_config(dict(data_a), cfg)
    feats_b = compute_ttm_features_from_config(dict(data_b), cfg)
    la = features_last_row(feats_a)
    lb = features_last_row(feats_b)
    sla, ssa, _ = compute_score_v2_alpha(feats_a, la, cfg)
    slb, ssb, _ = compute_score_v2_alpha(feats_b, lb, cfg)
    assert abs(sla - slb) < 1e-6
    assert abs(ssa - ssb) < 1e-6
    assert np.isfinite(sla) and np.isfinite(slb)
    assert np.isfinite(ssa) and np.isfinite(ssb)


def test_score_distribution_not_saturated_on_series() -> None:
    """LONG score remains bounded and finite under diverse histories."""
    n = 150
    rng = np.random.default_rng(99)
    closes = 100.0 + rng.standard_normal(n).cumsum() * 0.15
    vol = 1000.0 + np.abs(rng.standard_normal(n)) * 200.0
    basis = rng.standard_normal(n) * 0.3
    oi = (5000.0 + np.cumsum(rng.standard_normal(n) * 3.0)).tolist()
    cfg = {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True}
    data = {"bars": _bars(closes, vol), "basis": basis.tolist(), "open_interest": oi}
    feats = compute_ttm_features_from_config(dict(data), cfg)
    scores = []
    for i in range(50, n):
        sub = _feats_up_to(feats, i)
        last = features_last_row(sub)
        sl, _, _ = compute_score_v2_alpha(sub, last, cfg)
        scores.append(sl)
    arr = np.asarray(scores, dtype=np.float64)
    assert np.all(np.isfinite(arr))
    assert np.all(np.abs(arr) <= 1.0 + 1e-9)


def test_v2_short_score_uses_dedicated_exhaustion_leg() -> None:
    """Confirmed exhaustion should contribute to short alpha without forcing score symmetry."""
    n = 120
    closes = 100.0 + 1.0 * np.arange(n)
    closes[-2] += 8.0
    closes[-1] = closes[-2] - 4.0
    vol = 1500.0 + 100.0 * np.sin(np.linspace(0, 5.0, n))
    basis = np.zeros(n)
    oi = [4000.0 + i * 3.0 for i in range(n)]
    cfg = {
        **TTM_CONFIG,
        "ttm_oi_assert_nonzero_std": True,
        "breakout_strength_min": 0.2,
        "ttm_breakout_exhaustion_cap_window": 20,
        "ttm_breakout_exhaustion_cap_q": 0.8,
    }
    data = {"bars": _bars(closes, vol), "basis": basis.tolist(), "open_interest": oi}
    feats = compute_ttm_features_from_config(dict(data), cfg)
    last = features_last_row(feats)
    sl, ss, comp = compute_score_v2_alpha(feats, last, cfg)
    assert bool(last["exhaustion_confirm"])
    assert float(last["short_score"]) > 0.0
    assert ss > 0.0
    assert ss > sl
    assert float(comp["breakout_short_last"]) > 0.0
    assert abs(float(comp["score"]) - sl) < 1e-9


def test_v2_alpha_component_keys() -> None:
    """Expose separate long/short alpha diagnostics plus legacy aliases."""
    n = 100
    closes = np.linspace(100.0, 102.0, n)
    vol = np.full(n, 5000.0)
    basis = np.zeros(n)
    oi = [3000.0 + float(i) for i in range(n)]
    cfg = {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True}
    data = {"bars": _bars(closes, vol), "basis": basis.tolist(), "open_interest": oi}
    feats = compute_ttm_features_from_config(dict(data), cfg)
    last = features_last_row(feats)
    _, _, comp = compute_score_v2_alpha(feats, last, cfg)
    assert "alpha_raw" in comp and "alpha_rank" in comp and "score" in comp
    assert "alpha_raw_short" in comp and "alpha_rank_short" in comp
    assert abs(comp.get("vol_breakout_interaction_factor", 0.0) - 1.0) < 1e-9
    assert 0.0 - 1e-6 <= comp["alpha_rank"] <= 1.0 + 1e-6
    assert np.isclose(comp["momentum"], comp["score_long"] - comp["score_short"])


def test_v2_vol_breakout_interaction_optional() -> None:
    """LONG score is driven by effective_strength only; vol-breakout interaction no longer changes it."""
    n = 100
    closes = np.linspace(100.0, 101.0, n)
    vol = np.linspace(500.0, 50_000.0, n)
    basis = np.zeros(n)
    oi = [3000.0 + float(i) for i in range(n)]
    cfg0 = {**TTM_CONFIG, "ttm_oi_assert_nonzero_std": True, "ttm_v2_vol_breakout_interaction_k": 0.0}
    cfg1 = {
        **TTM_CONFIG,
        "ttm_oi_assert_nonzero_std": True,
        "ttm_v2_vol_breakout_interaction_k": 0.5,
    }
    data = {"bars": _bars(closes, vol), "basis": basis.tolist(), "open_interest": oi}
    feats = compute_ttm_features_from_config(dict(data), cfg0)
    last = features_last_row(feats)
    sl0, _, c0 = compute_score_v2_alpha(feats, last, cfg0)
    feats1 = compute_ttm_features_from_config(dict(data), cfg1)
    last1 = features_last_row(feats1)
    sl1, _, c1 = compute_score_v2_alpha(feats1, last1, cfg1)
    assert c0["vol_breakout_interaction_factor"] == 1.0
    assert c1["vol_breakout_interaction_factor"] == 1.0
    assert abs(sl0 - sl1) < 1e-9
