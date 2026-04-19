"""Compare HMM walk-forward metrics: 3 features (no basis) vs 4 features (with basis).

Uses synthetic OHLC + aligned index closes (no network). Reports Sharpe, P&L,
MDD, timeframe label, and calendar span — same fields as ``scripts/compare_hmm_basis.py``.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.backtest.data_fetcher import OhlcBar
from src.hmm.basis_performance_compare import (
    compare_basis_vs_no_basis,
    run_hmm_perf_snapshot,
    snapshots_to_markdown,
)
from src.hmm.feature_engineer import HMMConfig


def _make_bar(
    t: str,
    open_: float,
    close: float,
    high: float,
    low: float,
    vol: float,
    unix_ts: int,
) -> OhlcBar:
    return OhlcBar(
        symbol="SYN",
        time=t,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=vol,
        unix_ts=unix_ts,
    )


def _make_trend_bars(n: int = 120, seed: int = 7) -> list:
    """Synthetic 15m-style timestamps; strong drift so HMM tends to trade."""
    rng = np.random.default_rng(seed)
    bars = []
    price = 1500.0
    ts0 = 1_700_000_000
    for i in range(n):
        drift = 0.0012
        noise = rng.normal(0, 0.004)
        prev = price
        close = max(10.0, prev * (1.0 + drift + noise))
        o = prev
        hi = max(o, close) * 1.002
        lo = min(o, close) * 0.998
        day = 1 + (i // 96)
        hh = (i % 24)
        mm = (i % 4) * 15
        t = f"202601{day:02d}{hh:02d}{mm:02d}"
        bars.append(
            _make_bar(
                t=t,
                open_=round(o, 4),
                close=round(close, 4),
                high=round(hi, 4),
                low=round(lo, 4),
                vol=float(5000 + i * 10),
                unix_ts=ts0 + i * 900,
            )
        )
        price = close
    return bars


def _index_closes_aligned(bars: list, basis_pts: float = 3.5) -> list:
    """Index level so basis ≈ constant (future_close - index_close)."""
    return [float(b.close) - basis_pts for b in bars]


@pytest.fixture
def hmm_base() -> HMMConfig:
    # Lower n_iter keeps CI fast; still exercises full walk-forward replay.
    return HMMConfig(k_states=3, n_iter=60, zscore_window=20, random_state=42)


def test_compare_produces_two_snapshots_with_required_fields(hmm_base: HMMConfig):
    bars = _make_trend_bars(110)
    ic = _index_closes_aligned(bars)
    a, b = compare_basis_vs_no_basis(
        bars,
        ic,
        base_hmm_config=hmm_base,
        timeframe="15m",
        bar_type="15",
        symbol="SYN",
        warmup_bars=35,
        confidence_threshold=0.55,
        refit_every=3,
    )
    for s in (a, b):
        assert s.n_bars == len(bars)
        assert s.period_start == bars[0].date_str
        assert s.period_end == bars[-1].date_str
        assert s.timeframe == "15m"
        assert math.isfinite(s.sharpe_ratio)
        assert math.isfinite(s.max_drawdown_pct)
        assert math.isfinite(s.total_pnl)
        assert math.isfinite(s.total_return_pct)
        assert s.n_trades >= 0
    assert "no basis" in a.variant.lower()
    assert "basis" in b.variant.lower()


def test_markdown_table_includes_sharpe_pnl_mdd_timeframe(hmm_base: HMMConfig):
    bars = _make_trend_bars(100)
    ic = _index_closes_aligned(bars)
    a, b = compare_basis_vs_no_basis(
        bars,
        ic,
        base_hmm_config=hmm_base,
        timeframe="15m",
        bar_type="15",
        warmup_bars=35,
        confidence_threshold=0.55,
        refit_every=2,
    )
    md = snapshots_to_markdown(a, b)
    assert "Sharpe" in md
    assert "Total P&L" in md
    assert "MDD" in md
    assert "15m" in md
    assert a.period_start in md and a.period_end in md


def test_daily_bar_type_annualisation(hmm_base: HMMConfig):
    """Smoke: bar_type D uses daily-style metrics path."""
    bars = _make_trend_bars(90)
    ic = _index_closes_aligned(bars)
    a, b = compare_basis_vs_no_basis(
        bars,
        ic,
        base_hmm_config=hmm_base,
        timeframe="Daily 1D",
        bar_type="D",
        warmup_bars=30,
        confidence_threshold=0.55,
    )
    assert a.sharpe_ratio == a.sharpe_ratio
    assert b.total_pnl == b.total_pnl


def test_run_single_variant_requires_index_when_basis(hmm_base: HMMConfig):
    bars = _make_trend_bars(40)
    cfg = replace(hmm_base, use_basis=True)
    with pytest.raises(ValueError, match="index_closes"):
        run_hmm_perf_snapshot(
            bars,
            variant="x",
            timeframe="15m",
            bar_type="15",
            hmm_config=cfg,
            index_closes=None,
        )
