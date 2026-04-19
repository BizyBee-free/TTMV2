"""Regime labels per bar for reporting (trend / vol / basis / OI)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.backtest.ttm_opt.params import TtmOptParams


def label_regimes(feats: Dict[str, Any], p: TtmOptParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        trend_tag: 1 trending, 0 range, -1 unknown
        vol_tag: 1 high, 0 mid, -1 low, -2 unknown
    """
    n = int(feats["n"])
    adx = feats["adx"]
    atr_pct = feats["atr_pct"]
    thr = float(p.adx_trend_min)

    trend = np.full(n, -1, dtype=np.int8)
    vol = np.full(n, -2, dtype=np.int8)
    for i in range(n):
        if np.isfinite(adx[i]):
            trend[i] = 1 if float(adx[i]) >= thr else 0
        if np.isfinite(atr_pct[i]):
            ap = float(atr_pct[i])
            if ap >= 0.67:
                vol[i] = 1
            elif ap <= 0.33:
                vol[i] = -1
            else:
                vol[i] = 0
    return trend, vol


def regime_name(trend_i: int, vol_i: int) -> str:
    parts = []
    if trend_i == 1:
        parts.append("trend")
    elif trend_i == 0:
        parts.append("range")
    else:
        parts.append("trend_unk")
    if vol_i == 1:
        parts.append("hi_vol")
    elif vol_i == -1:
        parts.append("lo_vol")
    elif vol_i == 0:
        parts.append("mid_vol")
    else:
        parts.append("vol_unk")
    return "+".join(parts)
