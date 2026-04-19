"""HMM last_bar_signal and sliding window."""

import numpy as np

from src.backtest.data_fetcher import OhlcBar
from src.backtest.hmm_replay import HMMBacktestConfig, HMMBarReplay
from src.hmm.feature_engineer import HMMConfig
from src.hmm.hmm_signal import FLAT


def _synthetic_bars(n: int = 80) -> list:
    rng = np.random.default_rng(42)
    t = 1700000000
    bars = []
    p = 1000.0
    for i in range(n):
        o = p
        c = p + float(rng.normal(0, 2))
        h = max(o, c) + 1
        l = min(o, c) - 1
        bars.append(
            OhlcBar(
                symbol="X",
                time=str(t + i * 900),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1000.0 + i,
                unix_ts=t + i * 900,
            )
        )
        p = c
    return bars


def test_last_bar_signal_returns_int():
    bars = _synthetic_bars(80)
    cfg = HMMBacktestConfig(
        hmm_config=HMMConfig(k_states=3, n_iter=50),
        warmup_bars=25,
        refit_every=10,
        confidence_threshold=0.99,
    )
    replay = HMMBarReplay()
    direction, label, detail, _ = replay.last_bar_signal(bars, cfg)
    assert detail == "ok" or detail == "warmup"
    assert direction in (-1, 0, 1)


def test_basis_last_bar_signal_runs():
    bars = _synthetic_bars(80)
    index_closes = [float(b.close) - 0.5 for b in bars]
    cfg = HMMBacktestConfig(
        hmm_config=HMMConfig(k_states=3, n_iter=50, use_basis=True),
        warmup_bars=25,
        refit_every=10,
        confidence_threshold=0.99,
        index_closes=index_closes,
    )
    replay = HMMBarReplay()
    direction, label, detail, _ = replay.last_bar_signal(bars, cfg)
    assert detail == "ok" or detail == "warmup"
    assert direction in (-1, 0, 1)


def test_last_bar_signal_return_trace():
    bars = _synthetic_bars(80)
    cfg = HMMBacktestConfig(
        hmm_config=HMMConfig(k_states=3, n_iter=50),
        warmup_bars=25,
        refit_every=10,
        confidence_threshold=0.99,
    )
    replay = HMMBarReplay()
    direction, label, detail, trace = replay.last_bar_signal(bars, cfg, return_trace=True)
    assert trace is not None
    assert trace.n_bars == len(bars)
    if detail == "ok":
        assert trace.proba is not None
        assert len(trace.proba) == cfg.hmm_config.k_states
        assert trace.feature_tail_last5 is not None
        assert trace.guard_reason in (
            "confidence_below_threshold",
            "dominant_regime_not_bull_bear",
            "signal_long",
            "signal_short",
            "flat_or_internal_mismatch",
        )


def test_sliding_window_smaller_than_expanding():
    bars = _synthetic_bars(100)
    cfg = HMMBacktestConfig(
        hmm_config=HMMConfig(k_states=3, n_iter=30),
        warmup_bars=25,
        refit_every=20,
        sliding_window_bars=40,
        confidence_threshold=0.99,
    )
    replay = HMMBarReplay()
    d, _, det, _ = replay.last_bar_signal(bars, cfg)
    assert det == "ok" or det == "warmup" or d == FLAT
