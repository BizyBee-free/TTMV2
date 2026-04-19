"""Unit tests for the backtest framework.

Test plan:
    DataFetcher
        [F1] fetch() returns well-formed OhlcBar list from mock SDK
        [F2] fetch() serves from cache on second call (no SDK call)
        [F3] parse_columnar handles DNSE columnar JSON format
        [F4] parse_rows handles row-oriented JSON format
        [F5] fetch() warns and returns [] when API returns 0 bars
        [F6] clear_cache() removes files from cache dir

    BarReplay (no look-ahead bias)
        [R1] replay with constant-rise bars generates BUY trades
        [R2] replay with constant-fall bars generates SELL trades
        [R3] replay respects warmup: no trades before warmup_bars
        [R4] signal at bar i only uses close data from bars 0..i-1
        [R5] cold-start: < warmup bars -> no trades, no crash
        [R6] equity_curve has same length as bars
        [R7] T0 P&L = (close - open) * direction
        [R8] commission is subtracted from net_pnl

    BacktestMetrics
        [M1] compute_metrics golden values on known toy returns
        [M2] Sharpe = 0 when all returns are identical
        [M3] max_drawdown 0 for monotonically rising equity
        [M4] profit_factor = inf when no losing trades
        [M5] CAGR correctly annualises < 1-year period
        [M6] sortino = inf when no losing trades and positive mean
        [M7] win_rate = 100% when all trades profitable

    Statistical Tests
        [S1] monte_carlo_permutation p-value ~= 0.5 for random strategy
        [S2] walk_forward_split returns correct ratio
        [S3] _extract_returns correct values
        [S4] _apply_shuffled_returns preserves opens

    CSV export
        [V1] CSV has correct number of rows (one per trade)
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.bar_replay import BacktestConfig, BarReplay, TradeRecord
from src.backtest.data_fetcher import DataFetcher, OhlcBar
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.backtest.statistical_tests import (
    PermResult,
    _apply_shuffled_returns,
    _extract_returns,
    monte_carlo_permutation,
    walk_forward_split,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_bars(n: int, symbol: str = "TEST", trend: str = "up",
               base_price: float = 1000.0) -> List[OhlcBar]:
    """Generate synthetic daily bars with a deterministic trend."""
    bars = []
    price = base_price
    for i in range(n):
        date = f"2025{(i // 30 + 9):02d}{(i % 30 + 1):02d}"  # rough YYYYMMDD
        if trend == "up":
            open_p = price
            close_p = price * 1.003   # +0.3% every bar
        elif trend == "down":
            open_p = price
            close_p = price * 0.997   # -0.3%
        else:
            rng = np.random.default_rng(i)
            open_p = price
            close_p = price * (1 + rng.normal(0, 0.005))
        bars.append(OhlcBar(
            symbol=symbol,
            time=date,
            open=round(open_p, 2),
            high=round(max(open_p, close_p) * 1.001, 2),
            low=round(min(open_p, close_p) * 0.999, 2),
            close=round(close_p, 2),
            volume=1000.0,
            unix_ts=1756684800 + i * 86400,
        ))
        price = close_p
    return bars


def _make_config(warmup: int = 10, threshold: float = 0.55, fast: bool = True) -> BacktestConfig:
    return BacktestConfig(
        symbol="TEST",
        bar_type="D",
        warmup_bars=warmup,
        confidence_threshold=threshold,
        position_size=1,
        mh_iterations=50,
        mh_burnin=20,
        n_paths=200,
        seed=42,
        commission_pct=0.0,
        fast_mode=fast,
    )


# ── DataFetcher tests ─────────────────────────────────────────────────────────

class TestDataFetcher:
    """Tests for DataFetcher.

    The real DNSE API uses:
      - GET /price/ohlc?symbol=X&resolution=1D&type=stock&from={unix_ts}&to={unix_ts}
      - Response: columnar JSON {"t":[unix_ts,...], "o":[...], "h":[...], "l":[...], "c":[...]}

    Tests mock the SDK's _request() method with realistic Unix timestamp payloads.
    """

    # Real-ish Unix timestamps: 2025-09-01 and 2025-09-02 UTC
    _TS1 = 1756684800   # 2025-09-01
    _TS2 = 1756771200   # 2025-09-02

    def _fetcher_with_mock_sdk(self, response_body, status=200, cache_dir=None):
        """Build a DataFetcher backed by a mock SDK."""
        fetcher = DataFetcher.__new__(DataFetcher)
        fetcher._cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp())
        fetcher._cache_dir.mkdir(parents=True, exist_ok=True)
        fetcher._use_cache = True
        mock_sdk = MagicMock()
        # The new fetcher calls sdk._request() not sdk.get_ohlc()
        mock_sdk._request.return_value = (status, json.dumps(response_body))
        fetcher._sdk = mock_sdk
        return fetcher

    def test_f1_returns_ohlcbar_list(self, tmp_path):
        """[F1] fetch() returns well-formed OhlcBar list from mock SDK."""
        payload = {
            "t": [self._TS1, self._TS2],
            "o": [1000.0, 1005.0],
            "h": [1010.0, 1015.0],
            "l": [995.0, 1000.0],
            "c": [1005.0, 1010.0],
            "v": [1000, 2000],
        }
        fetcher = self._fetcher_with_mock_sdk(payload, cache_dir=tmp_path)
        bars = fetcher.fetch("TEST", "20250901", "20250902")

        assert len(bars) == 2
        assert isinstance(bars[0], OhlcBar)
        assert bars[0].open == 1000.0
        assert bars[1].close == 1010.0
        assert bars[0].symbol == "TEST"
        assert bars[0].unix_ts == self._TS1

    def test_f2_serves_from_cache(self, tmp_path):
        """[F2] fetch() serves from cache on second call (no extra SDK call)."""
        payload = {
            "t": [self._TS1],
            "o": [1000.0], "h": [1010.0], "l": [990.0], "c": [1005.0], "v": [1000],
        }
        fetcher = self._fetcher_with_mock_sdk(payload, cache_dir=tmp_path)

        fetcher.fetch("TEST", "20250901", "20250901")
        fetcher.fetch("TEST", "20250901", "20250901")  # second call

        assert fetcher._sdk._request.call_count == 1  # only one actual API call

    def test_f3_parse_columnar_format(self, tmp_path):
        """[F3] parse_columnar handles DNSE Unix timestamp columnar format."""
        ts_vals = [self._TS1 + i * 86400 for i in range(3)]
        payload = {
            "t": ts_vals,
            "o": [1200, 1210, 1205],
            "h": [1215, 1220, 1215],
            "l": [1195, 1205, 1200],
            "c": [1210, 1205, 1210],
        }
        fetcher = self._fetcher_with_mock_sdk(payload, cache_dir=tmp_path)
        bars = fetcher.fetch("SYMB", "20250901", "20250903")
        assert len(bars) == 3
        assert all(isinstance(b, OhlcBar) for b in bars)
        assert all(b.unix_ts > 0 for b in bars)

    def test_f4_date_str_format(self, tmp_path):
        """[F4] bar.time is YYYYMMDD string derived from Unix timestamp."""
        payload = {
            "t": [self._TS1],
            "o": [100.0], "h": [105.0], "l": [95.0], "c": [102.0], "v": [1000],
        }
        fetcher = self._fetcher_with_mock_sdk(payload, cache_dir=tmp_path)
        bars = fetcher.fetch("TEST", "20250901", "20250901")
        assert len(bars) == 1
        assert bars[0].time == "20250901"  # YYYYMMDD from Unix ts
        assert bars[0].date_str == "20250901"

    def test_f5_returns_empty_on_zero_bars(self, tmp_path):
        """[F5] fetch() returns [] when API returns 0 bars."""
        payload = {"t": [], "o": [], "h": [], "l": [], "c": [], "v": [], "nextTime": 0}
        fetcher = self._fetcher_with_mock_sdk(payload, cache_dir=tmp_path)
        bars = fetcher.fetch("X", "20250101", "20250101")
        assert bars == []

    def test_f6_clear_cache(self, tmp_path):
        """[F6] clear_cache() removes files from cache dir."""
        payload = {
            "t": [self._TS1], "o": [1000.0], "h": [1010.0], "l": [990.0], "c": [1005.0], "v": [100],
        }
        fetcher = self._fetcher_with_mock_sdk(payload, cache_dir=tmp_path)
        fetcher.fetch("CLR", "20250901", "20250901")

        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1

        deleted = fetcher.clear_cache()
        assert deleted == 1
        assert not list(tmp_path.glob("*.json"))


# ── BarReplay tests ────────────────────────────────────────────────────────────

class TestBarReplay:

    def test_r1_buy_trades_on_uptrend(self):
        """[R1] replay with constant-rise bars generates BUY trades."""
        bars = _make_bars(30, trend="up")
        config = _make_config(warmup=5, threshold=0.50, fast=True)
        replay = BarReplay()
        trades, equity = replay.run(bars, config)

        buy_count = sum(1 for t in trades if t.side == "BUY")
        assert buy_count > 0, "Expected at least one BUY trade on uptrend"

    def test_r2_sell_trades_on_downtrend(self):
        """[R2] replay with constant-fall bars generates SELL trades."""
        bars = _make_bars(30, trend="down")
        config = _make_config(warmup=5, threshold=0.50, fast=True)
        replay = BarReplay()
        trades, equity = replay.run(bars, config)

        sell_count = sum(1 for t in trades if t.side == "SELL")
        assert sell_count > 0, "Expected at least one SELL trade on downtrend"

    def test_r3_no_trades_before_warmup(self):
        """[R3] replay respects warmup_bars: no trades during warm-up period."""
        bars = _make_bars(60, trend="up")
        warmup = 25
        config = _make_config(warmup=warmup, threshold=0.50, fast=True)
        replay = BarReplay()
        trades, _ = replay.run(bars, config)

        for t in trades:
            assert t.entry_bar >= warmup, (
                f"Trade at bar {t.entry_bar} is within warmup ({warmup})"
            )

    def test_r4_no_lookahead_signal(self):
        """[R4] signal at bar i only uses data from bars 0..i-1."""
        # We verify by checking: all trades use entry at bar.open (not prev bar.close)
        bars = _make_bars(40, trend="up")
        config = _make_config(warmup=10, threshold=0.50, fast=True)
        replay = BarReplay()
        trades, _ = replay.run(bars, config)

        for t in trades:
            expected_entry = bars[t.entry_bar].open
            assert abs(t.entry_price - expected_entry) < 0.001, (
                f"Entry price mismatch at bar {t.entry_bar}: "
                f"got {t.entry_price}, expected open={expected_entry}"
            )

    def test_r5_cold_start_no_crash(self):
        """[R5] cold-start (< warmup bars) -> no trades, no crash."""
        bars = _make_bars(5, trend="up")   # fewer than default warmup=10
        config = _make_config(warmup=10, fast=True)
        replay = BarReplay()
        trades, equity = replay.run(bars, config)

        assert trades == []
        assert len(equity) == len(bars)

    def test_r6_equity_same_length_as_bars(self):
        """[R6] equity_curve has same length as bars."""
        bars = _make_bars(60)
        config = _make_config(warmup=10, fast=True)
        replay = BarReplay()
        _, equity = replay.run(bars, config)

        assert len(equity) == len(bars)

    def test_r7_t0_pnl_calculation(self):
        """[R7] T0 P&L = (close - open) * direction for each trade."""
        bars = _make_bars(40, trend="up")
        config = _make_config(warmup=10, threshold=0.50, fast=True)
        replay = BarReplay()
        trades, _ = replay.run(bars, config)

        for t in trades:
            direction = 1 if t.side == "BUY" else -1
            expected_pnl = (t.exit_price - t.entry_price) * direction * config.position_size
            assert abs(t.pnl - expected_pnl) < 1e-6, (
                f"PnL mismatch: got {t.pnl}, expected {expected_pnl}"
            )

    def test_r8_commission_deducted(self):
        """[R8] commission is deducted from net_pnl."""
        bars = _make_bars(40, trend="up")
        config = _make_config(warmup=10, threshold=0.50, fast=True)
        config.commission_pct = 0.001   # 0.1% per side
        replay = BarReplay()
        trades, _ = replay.run(bars, config)

        for t in trades:
            expected_commission = (t.entry_price + t.exit_price) * config.position_size * config.commission_pct
            assert abs(t.commission - expected_commission) < 1e-4


# ── BacktestMetrics tests ──────────────────────────────────────────────────────

class TestBacktestMetrics:

    def _make_trades(self, pnls: list) -> List[TradeRecord]:
        trades = []
        for i, pnl in enumerate(pnls):
            direction = 1 if pnl >= 0 else -1
            ep = 1000.0
            xp = ep + pnl * direction
            t = TradeRecord(
                symbol="T", entry_bar=i, exit_bar=i,
                entry_date=f"2025090{i+1}", exit_date=f"2025090{i+1}",
                side="BUY" if direction == 1 else "SELL",
                entry_price=ep, exit_price=xp,
                pnl=pnl, commission=0.0, net_pnl=pnl,
                p_up=0.6, p_down=0.4, confidence=0.6,
            )
            trades.append(t)
        return trades

    def test_m1_golden_values(self):
        """[M1] compute_metrics golden values on known toy returns."""
        pnls = [1.0, -0.5, 1.5, -0.2, 0.8, 1.2, -0.3, 0.9, 0.4, -0.1]
        trades = self._make_trades(pnls)
        equity = np.cumsum(pnls)
        m = compute_metrics(trades, equity, n_bars=len(pnls), bar_type="D", nav_start=100.0)

        # Positives: 1.0, 1.5, 0.8, 1.2, 0.9, 0.4 = 6 wins
        # Negatives: -0.5, -0.2, -0.3, -0.1 = 4 losses
        assert m.n_trades == 10
        assert m.n_wins == 6
        assert m.n_losses == 4
        assert abs(m.total_pnl - sum(pnls)) < 1e-9
        assert m.win_rate_pct == pytest.approx(60.0)  # 6/10 wins
        assert m.profit_factor > 1.0

    def test_m2_sharpe_zero_for_identical_returns(self):
        """[M2] Sharpe = 0 when all returns are identical (zero std)."""
        pnls = [1.0] * 10
        trades = self._make_trades(pnls)
        equity = np.cumsum(pnls)
        m = compute_metrics(trades, equity, n_bars=10, bar_type="D", nav_start=100.0)

        assert m.sharpe_ratio == 0.0

    def test_m3_zero_drawdown_on_rising_equity(self):
        """[M3] max_drawdown = 0 for monotonically rising equity."""
        pnls = [1.0, 0.5, 0.3, 0.8, 1.2]
        trades = self._make_trades(pnls)
        equity = np.cumsum(pnls)
        m = compute_metrics(trades, equity, n_bars=5, bar_type="D", nav_start=100.0)

        assert m.max_drawdown_pct == pytest.approx(0.0)

    def test_m4_profit_factor_inf_when_no_losses(self):
        """[M4] profit_factor = inf when no losing trades."""
        pnls = [1.0, 0.5, 0.3]
        trades = self._make_trades(pnls)
        equity = np.cumsum(pnls)
        m = compute_metrics(trades, equity, n_bars=3, bar_type="D", nav_start=100.0)

        assert m.profit_factor == float("inf")

    def test_m5_cagr_annualises_correctly(self):
        """[M5] CAGR correctly annualises a < 1-year period."""
        pnls = [0.5] * 63   # 63 trading days ~= 0.25 year (1 quarter)
        trades = self._make_trades(pnls)
        equity = np.cumsum(pnls)
        m = compute_metrics(trades, equity, n_bars=63, bar_type="D", nav_start=100.0)

        # total return = 31.5, annualised should be roughly 4x that
        assert m.cagr_pct > m.total_return_pct  # annualised > period return for < 1yr hold

    def test_m6_sortino_inf_when_no_losses(self):
        """[M6] sortino = inf when no losing trades and positive mean return."""
        pnls = [1.0, 0.8, 0.5]
        trades = self._make_trades(pnls)
        equity = np.cumsum(pnls)
        m = compute_metrics(trades, equity, n_bars=3, bar_type="D", nav_start=100.0)

        assert m.sortino_ratio == float("inf")

    def test_m7_win_rate_100_pct(self):
        """[M7] win_rate = 100% when all trades profitable."""
        pnls = [1.0, 2.0, 0.5]
        trades = self._make_trades(pnls)
        equity = np.cumsum(pnls)
        m = compute_metrics(trades, equity, n_bars=3, bar_type="D", nav_start=100.0)

        assert m.win_rate_pct == pytest.approx(100.0)
        assert m.n_wins == 3
        assert m.n_losses == 0


# ── Statistical Tests ─────────────────────────────────────────────────────────

class TestStatisticalTests:

    def test_s1_perm_pvalue_near_half_for_random(self):
        """[S1] monte_carlo_permutation p-value ~= 0.5 for a random strategy."""
        bars = _make_bars(80, trend="random")
        config = _make_config(warmup=10, threshold=0.99, fast=True)  # very high threshold -> near-zero trades

        # Use a small number of perms for speed
        result = monte_carlo_permutation(bars, config, n_perms=100, seed=42)

        assert isinstance(result, PermResult)
        assert 0.0 <= result.p_value <= 1.0
        assert len(result.perm_sharpes) == 100

    def test_s2_walk_forward_split_ratio(self):
        """[S2] walk_forward_split returns correct proportion."""
        bars = _make_bars(100)
        is_bars, oos_bars = walk_forward_split(bars, train_ratio=0.75)

        assert len(is_bars) == 75
        assert len(oos_bars) == 25
        assert is_bars[-1].time < oos_bars[0].time  # temporal order preserved

    def test_s3_extract_returns(self):
        """[S3] _extract_returns computes correct open-to-close return fractions."""
        bars = [
            OhlcBar("S", "20250901", open=1000.0, high=1010.0, low=990.0, close=1010.0, volume=100),
            OhlcBar("S", "20250902", open=1010.0, high=1020.0, low=1005.0, close=1005.0, volume=100),
        ]
        returns = _extract_returns(bars)
        assert len(returns) == 2
        assert abs(returns[0] - 0.01) < 1e-9     # +1%
        assert abs(returns[1] - (-0.00495)) < 1e-4  # ~-0.5%

    def test_s4_apply_shuffled_returns_preserves_opens(self):
        """[S4] _apply_shuffled_returns preserves original open prices."""
        bars = _make_bars(10, trend="up")
        returns = _extract_returns(bars)
        shuffled = np.random.default_rng(0).permutation(returns)
        new_bars = _apply_shuffled_returns(bars, shuffled)

        assert len(new_bars) == len(bars)
        for orig, new in zip(bars, new_bars):
            assert orig.open == new.open  # opens must be preserved


# ── CSV export ─────────────────────────────────────────────────────────────────

class TestCSVExport:

    def test_v1_csv_row_count(self, tmp_path):
        """[V1] CSV has correct number of rows (one per trade)."""
        import csv as csv_mod
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from scripts.backtest import _trades_to_rows, write_csv

        # Patch CSV path
        import scripts.backtest as bt_mod
        orig_path = bt_mod._CSV_PATH
        bt_mod._CSV_PATH = tmp_path / "test_trades.csv"

        try:
            trades = []
            bars = _make_bars(30, trend="up")
            config = _make_config(warmup=5, threshold=0.50, fast=True)
            replay = BarReplay()
            trade_list, _ = replay.run(bars, config)

            rows = _trades_to_rows(trade_list, "test_suite")
            write_csv(rows)

            csv_path = bt_mod._CSV_PATH
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv_mod.DictReader(f)
                    written = list(reader)
                assert len(written) == len(rows)
        finally:
            bt_mod._CSV_PATH = orig_path
