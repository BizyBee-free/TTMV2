"""TTM V2 effective_strength calibration (feature layer + V2 alpha input)."""

from __future__ import annotations

import numpy as np

from src.backtest.data_fetcher import OhlcBar
from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_features import compute_ttm_features_from_config
from src.strategies.ttm.ttm_score import compute_score_v2_alpha


def _bars_uptrend(n: int = 40) -> list[OhlcBar]:
    out: list[OhlcBar] = []
    p = 100.0
    for i in range(n):
        p = p + 0.05
        o, c = p, p + 0.02
        h, l = max(o, c) + 0.05, min(o, c) - 0.05
        out.append(
            OhlcBar(
                symbol="X",
                time=str(i),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1.0,
                unix_ts=1_700_000_000 + i * 60,
            )
        )
    return out


def test_effective_strength_matches_raw_when_k_zero() -> None:
    bars = _bars_uptrend(35)
    raw = {"bars": bars, "basis": np.zeros(len(bars)), "open_interest": np.linspace(1e5, 1.01e5, len(bars))}
    cfg = {**TTM_CONFIG, "ttm_v2_effective_strength_k_extension": 0.0, "ttm_v2_effective_strength_k_lastret": 0.0}
    feats = compute_ttm_features_from_config(raw, cfg)
    rs = np.asarray(feats["raw_strength"], dtype=np.float64)
    eff = np.asarray(feats["effective_strength"], dtype=np.float64)
    m = np.isfinite(rs) & np.isfinite(eff)
    assert m.sum() > 0
    np.testing.assert_allclose(eff[m], rs[m], rtol=0, atol=1e-9)


def test_last_bar_return_no_future_close() -> None:
    bars = _bars_uptrend(10)
    raw = {"bars": bars, "basis": np.zeros(len(bars)), "open_interest": np.linspace(1e5, 1.01e5, len(bars))}
    feats = compute_ttm_features_from_config(raw, {**TTM_CONFIG})
    lb = np.asarray(feats["last_bar_return"], dtype=np.float64)
    close = np.asarray([float(b.close) for b in bars], dtype=np.float64)
    # Bar 2: prior completed bar return uses close[0], close[1] only (no close[2]).
    assert np.isfinite(lb[2])
    c0, c1 = float(close[0]), float(close[1])
    assert abs(float(lb[2]) - (c1 - c0) / c0) < 1e-9


def test_compute_score_v2_alpha_reads_effective_strength_key() -> None:
    n = 30
    close = np.linspace(100.0, 103.0, n)
    bu = np.zeros(n, dtype=bool)
    bu[20] = True
    brk_base = np.zeros(n, dtype=np.float64)
    brk_base[20] = 1.0
    eff = np.zeros(n, dtype=np.float64)
    eff[20] = 0.25
    feats = {
        "n": n,
        "close": close,
        "breakout_strength_base": brk_base,
        "breakout_up": bu,
        "short_score": np.full(n, np.nan),
        "exhaustion_confirm": np.zeros(n, dtype=bool),
        "momentum_1_z": np.zeros(n),
        "basis_signal": np.zeros(n),
        "vol_zscore": np.zeros(n),
        "effective_strength": eff,
    }
    last = {
        "breakout_up": True,
        "breakout_strength_base": 1.0,
        "momentum_1_z": 0.0,
        "basis_signal": 0.0,
        "vol_zscore": 0.0,
        "exhaustion_confirm": False,
        "short_score": float("nan"),
    }
    sl, ss, comp = compute_score_v2_alpha(feats, last, {**TTM_CONFIG}, bar_index=20)
    assert np.isfinite(sl) and np.isfinite(ss)
    assert abs(float(comp.get("effective_strength_last", 0.0)) - 0.25) < 1e-9


def test_breakout_validation_prefers_effective_strength_in_features() -> None:
    import importlib

    tv = importlib.import_module("src.strategies.ttm.ttm_validation")

    closes = np.array([100.0, 100.5, 101.0, 101.5, 102.0, 102.5, 103.0, 103.5], dtype=np.float64)
    decisions = []
    for i in range(len(closes)):
        f = {
            "is_breakout_up": True,
            "breakout_up_filtered_last": True,
            "raw_strength": 0.5,
            "rolling_high": 99.0,
            "effective_strength": 0.1 + 0.01 * i,
            "exhaustion_candidate": False,
            "breakout_strength": 0.5,
        }
        decisions.append({"bar_index": i, "features": f})
    br = tv.test_breakout(decisions, closes, forward_h=2, min_samples=3)
    ev = (br.metrics or {}).get("breakout_up_eval") or {}
    assert ev.get("strength_axis") == "effective_strength"
    assert int(ev.get("n_bucket_rows_using_effective_strength", 0)) > 0
