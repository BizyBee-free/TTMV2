"""Permutation p-value for equity curve bar increments.

Tests whether the *ordering* of per-bar P&L contributions (diffs of cumulative
equity) is consistent with random reordering — a lightweight proxy for
"path significance" without re-fitting HMM on every permutation.

H0: shuffling bar increments does not reduce a Sharpe-like statistic on those
increments. One-sided p-value: fraction of permutations with Sharpe >= actual.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _increment_sharpe(
    increments: np.ndarray,
    n_bars: int,
    bars_per_year: int,
    nav_start: float = 100.0,
) -> float:
    """Annualized Sharpe treating each bar's increment as a return on nav_start."""
    inc = np.asarray(increments, dtype=float)
    if inc.size < 2:
        return 0.0
    ret = inc / nav_start if nav_start > 0 else inc
    std = float(np.std(ret, ddof=1))
    if std < 1e-12:
        return 0.0
    mean = float(np.mean(ret))
    years = n_bars / bars_per_year if bars_per_year > 0 else 1.0
    bars_per_yr = n_bars / years if years > 0 else float(n_bars)
    return (mean / std) * math.sqrt(bars_per_yr)


def equity_increment_permutation_pvalue(
    equity_curve: np.ndarray,
    n_bars: int,
    bars_per_year: int,
    n_perms: int = 1000,
    seed: int = 42,
    nav_start: float = 100.0,
) -> Tuple[float, float]:
    """Return (p_value, actual_sharpe_on_increments).

    p_value is one-sided: (1 + count(perm_sharpe >= actual)) / (n_perms + 1).
    """
    eq = np.asarray(equity_curve, dtype=float)
    if eq.size < 2 or n_perms < 1:
        return 1.0, 0.0

    inc = np.zeros_like(eq)
    inc[1:] = eq[1:] - eq[:-1]
    actual = _increment_sharpe(inc, n_bars=n_bars, bars_per_year=bars_per_year, nav_start=nav_start)

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perms):
        perm = rng.permutation(inc)
        s = _increment_sharpe(perm, n_bars=n_bars, bars_per_year=bars_per_year, nav_start=nav_start)
        if s >= actual:
            ge += 1

    p_val = (ge + 1) / (n_perms + 1)
    return float(p_val), float(actual)
