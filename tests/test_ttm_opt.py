"""Tests for TTM optimization stack (features, engine, walk-forward)."""

from __future__ import annotations

import numpy as np

from src.backtest.ttm_opt.engine import run_ttm_opt_backtest
from src.backtest.ttm_opt.features import compute_feature_bundle, entry_masks_long_short
from src.backtest.ttm_opt.optimize import stability_score, walk_forward_objective
from src.backtest.ttm_opt.params import TtmOptParams, perturb_params
from src.backtest.ttm_opt.walk_forward import walk_forward_splits
from src.backtest.data_fetcher import OhlcBar


def _synth_bars(n: int = 200) -> list[OhlcBar]:
    bars: list[OhlcBar] = []
    p = 100.0
    t0 = 1700000000
    for i in range(n):
        p += np.sin(i / 8.0) * 0.15 + (0.01 if i % 17 == 0 else 0)
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


def test_features_bundle_runs():
    n = 120
    c = np.cumsum(np.random.randn(n) * 0.1) + 100.0
    h = c + 0.2
    l = c - 0.2
    b = np.zeros(n)
    oi = np.linspace(1000, 1100, n)
    p = TtmOptParams(warmup_bars=60)
    feats = compute_feature_bundle(c, h, l, b, oi, p)
    assert feats["n"] == n
    assert len(feats["squeeze_fire"]) == n


def test_entry_masks_shape():
    n = 100
    c = np.linspace(100, 102, n)
    h = c + 0.05
    l = c - 0.05
    feats = compute_feature_bundle(c, h, l, np.zeros(n), np.ones(n) * 1e5, TtmOptParams(warmup_bars=40))
    lo, sh = entry_masks_long_short(feats, TtmOptParams(warmup_bars=40, use_squeeze=False))
    assert len(lo) == n and len(sh) == n


def test_engine_smoke():
    bars = _synth_bars(180)
    n = len(bars)
    o = np.array([b.open for b in bars])
    h = np.array([b.high for b in bars])
    l = np.array([b.low for b in bars])
    c = np.array([b.close for b in bars])
    tl = [b.time for b in bars]
    basis = np.random.randn(n) * 0.5
    oi = np.linspace(1e5, 1.01e5, n)
    p = TtmOptParams(
        warmup_bars=80,
        use_squeeze=False,
        regime_filter="none",
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    tr, eq, pack = run_ttm_opt_backtest(o, h, l, c, tl, basis, oi, p, bar_type="15")
    assert len(eq) == n
    assert "metrics" in pack
    m = pack["metrics"]
    assert m.n_bars == n
    assert stability_score(m) is not None


def test_walk_forward_splits():
    bars = _synth_bars(5000)
    splits = walk_forward_splits(bars, train_months=1.0, test_months=0.5, resolution="15")
    assert isinstance(splits, list)


def test_perturb_params_numpy_rng():
    p = TtmOptParams()
    rng = np.random.default_rng(0)
    q = perturb_params(p, rng, frac=0.1)
    assert q.bb_len >= 1


def test_wf_objective_trivial():
    bars = _synth_bars(400)
    n = len(bars)
    o = np.array([b.open for b in bars])
    h = np.array([b.high for b in bars])
    l = np.array([b.low for b in bars])
    c = np.array([b.close for b in bars])
    tl = [b.time for b in bars]
    basis = np.zeros(n)
    oi = np.zeros(n)
    p = TtmOptParams(warmup_bars=60, use_squeeze=False, regime_filter="none")
    splits = walk_forward_splits(bars, train_months=0.25, test_months=0.15, resolution="15")
    agg, tr, te = walk_forward_objective(p, splits, o, h, l, c, tl, basis, oi, "15")
    assert isinstance(agg, float)
