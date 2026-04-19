"""Nested walk-forward splits (train / test) by calendar time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.backtest.data_fetcher import OhlcBar


@dataclass
class WFSplit:
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


def _bar_period_seconds(resolution: str) -> int:
    r = (resolution or "15").upper()
    if r in {"1", "1M", "1MIN"}:
        return 60
    if r in {"5", "5M"}:
        return 300
    if r in {"15", "15M"}:
        return 900
    if r in {"30", "30M"}:
        return 1800
    if r in {"60", "1H", "H"}:
        return 3600
    if r in {"240", "4H"}:
        return 14400
    if r in {"1D", "D"}:
        return 86400
    return 900


def walk_forward_splits(
    bars: Sequence[OhlcBar],
    *,
    train_months: float = 6.0,
    test_months: float = 2.0,
    resolution: str = "15",
) -> List[WFSplit]:
    """
    Build rolling windows using bar timestamps (unix_ts).
    Approximate month = 30 * 86400 seconds.
    """
    n = len(bars)
    if n < 10:
        return []

    ts = np.array([int(b.unix_ts or 0) for b in bars], dtype=np.int64)
    if np.all(ts == 0):
        # Fallback: equal bar spacing
        sec = _bar_period_seconds(resolution)
        ts = np.arange(n, dtype=np.int64) * sec

    train_sec = int(train_months * 30.0 * 86400.0)
    test_sec = int(test_months * 30.0 * 86400.0)

    splits: List[WFSplit] = []
    t0 = int(ts[0])
    t_end_data = int(ts[-1])

    cur = t0
    while True:
        tr_e = cur + train_sec
        te_e = tr_e + test_sec
        if te_e > t_end_data + 1:
            break
        # map times to indices
        i_tr_s = int(np.searchsorted(ts, cur, side="left"))
        i_tr_e = int(np.searchsorted(ts, tr_e, side="right"))
        i_te_s = int(np.searchsorted(ts, tr_e, side="left"))
        i_te_e = int(np.searchsorted(ts, te_e, side="right"))
        if i_tr_e <= i_tr_s + 20 or i_te_e <= i_te_s + 5:
            cur += test_sec
            continue
        splits.append(
            WFSplit(
                train_start=i_tr_s,
                train_end=i_tr_e,
                test_start=i_te_s,
                test_end=i_te_e,
            )
        )
        cur += test_sec

    return splits


def hold_out_split(
    n: int,
    holdout_frac: float = 0.25,
    min_train: int = 200,
) -> Tuple[slice, slice]:
    """Last ``holdout_frac`` of bars for OOS."""
    if n < min_train + 50:
        return slice(0, n), slice(n, n)
    k = int(n * (1.0 - holdout_frac))
    k = max(min_train, min(k, n - 30))
    return slice(0, k), slice(k, n)
