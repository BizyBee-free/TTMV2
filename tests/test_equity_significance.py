"""Tests for equity curve increment permutation p-value."""

import numpy as np

from src.backtest.equity_significance import equity_increment_permutation_pvalue


def test_pvalue_in_unit_interval():
    rng = np.random.default_rng(0)
    equity = np.cumsum(rng.normal(0.01, 1.0, 120))
    p, s = equity_increment_permutation_pvalue(
        equity, n_bars=len(equity), bars_per_year=252 * 4 * 12, n_perms=200, seed=1
    )
    assert 0.0 < p <= 1.0
    assert isinstance(s, float)
