"""Vectorized TTM Squeeze, momentum, basis, OI features for optimization backtests."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from src.backtest.ttm_opt.params import TtmOptParams


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (float(span) + 1.0)
    out[0] = float(x[0])
    for i in range(1, n):
        out[i] = alpha * float(x[i]) + (1.0 - alpha) * out[i - 1]
    return out


def _sma(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = x[lo : i + 1]
        if len(seg) >= w:
            out[i] = float(np.mean(seg))
    return out


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = x[lo : i + 1]
        if len(seg) >= w:
            out[i] = float(np.std(seg, ddof=0))
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = float(high[0] - low[0])
    for i in range(1, n):
        tr[i] = max(
            float(high[i] - low[i]),
            abs(float(high[i] - close[i - 1])),
            abs(float(low[i] - close[i - 1])),
        )
    return tr


def _wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (RMA): y[i] = (y[i-1]*(p-1) + x[i]) / p."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or period < 1 or n < period:
        return out
    p = float(period)
    out[period - 1] = float(np.sum(x[:period])) / p
    for i in range(period, n):
        out[i] = (out[i - 1] * (p - 1) + x[i]) / p
    return out


def _atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    tr = _true_range(high, low, close)
    return _wilder_smooth(tr, period)


def _adx_di(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (adx, plus_di, minus_di) length n, NaN during warmup."""
    n = len(close)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = float(high[i] - high[i - 1])
        down = float(low[i - 1] - low[i])
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    tr = _true_range(high, low, close)
    tr_s = _wilder_smooth(tr, period)
    p_dm = _wilder_smooth(plus_dm, period)
    m_dm = _wilder_smooth(minus_dm, period)
    pdi = np.full(n, np.nan, dtype=np.float64)
    mdi = np.full(n, np.nan, dtype=np.float64)
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isfinite(tr_s[i]) or tr_s[i] <= 1e-12:
            continue
        pdi[i] = 100.0 * p_dm[i] / tr_s[i]
        mdi[i] = 100.0 * m_dm[i] / tr_s[i]
        s = pdi[i] + mdi[i]
        if s > 1e-12:
            dx[i] = 100.0 * abs(pdi[i] - mdi[i]) / s
    adx = _wilder_smooth(np.nan_to_num(dx, nan=0.0), period)
    return adx, pdi, mdi


def _macd_hist(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    n = len(close)
    hist = np.full(n, np.nan, dtype=np.float64)
    if n < slow + signal:
        return hist
    ef = _ema(close, fast)
    es = _ema(close, slow)
    macd = ef - es
    sig = _ema(macd, signal)
    hist[:] = macd - sig
    return hist


def _percentile_rank_last(window_values: np.ndarray) -> float:
    """Percentile rank of last value in window [0,1]."""
    w = window_values[np.isfinite(window_values)]
    if len(w) < 2:
        return 0.5
    last = w[-1]
    return float(np.mean(w <= last))


def compute_feature_bundle(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    basis: np.ndarray,
    oi: np.ndarray,
    p: TtmOptParams,
) -> Dict[str, Any]:
    """
    All arrays length n. ``basis`` and ``oi`` may be zeros if unavailable.
    """
    n = len(close)
    c = np.asarray(close, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    b = np.asarray(basis, dtype=np.float64) if basis is not None else np.zeros(n, dtype=np.float64)
    o = np.asarray(oi, dtype=np.float64) if oi is not None else np.zeros(n, dtype=np.float64)

    bb_len = max(2, int(p.bb_len))
    bb_std = float(p.bb_std)
    mid_bb = _sma(c, bb_len)
    sd = _rolling_std(c, bb_len)
    upper_bb = mid_bb + bb_std * sd
    lower_bb = mid_bb - bb_std * sd

    kc_len = max(2, int(p.kc_len))
    atr_p = max(2, int(p.atr_len))
    kc_mid = _ema(c, kc_len)
    atr = _atr_wilder(h, l, c, atr_p)
    mult = float(p.kc_atr_mult)
    upper_kc = kc_mid + mult * atr
    lower_kc = kc_mid - mult * atr

    squeeze = np.zeros(n, dtype=bool)
    for i in range(n):
        if not (
            np.isfinite(upper_bb[i])
            and np.isfinite(lower_bb[i])
            and np.isfinite(upper_kc[i])
            and np.isfinite(lower_kc[i])
        ):
            continue
        squeeze[i] = bool(upper_bb[i] < upper_kc[i] and lower_bb[i] > lower_kc[i])

    squeeze_fire = np.zeros(n, dtype=bool)
    for i in range(1, n):
        squeeze_fire[i] = bool(squeeze[i - 1] and (not squeeze[i]))

    # Momentum (primary)
    mh = _macd_hist(c)
    # HTF proxy: slower MACD on same closes
    scale = max(1, int(p.htf_momentum_scale))
    mh_htf = _macd_hist(c, fast=12 * scale, slow=26 * scale, signal=9 * scale)

    # Basis helpers
    bd_w = max(2, int(p.basis_delta_window))
    bz_w = max(5, int(p.basis_z_window))
    basis_delta_sum = np.zeros(n, dtype=np.float64)
    for i in range(bd_w, n):
        basis_delta_sum[i] = float(b[i] - b[i - bd_w])
    basis_z = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - bz_w + 1)
        seg = b[lo : i + 1]
        if len(seg) >= 5:
            m = float(np.mean(seg))
            s = float(np.std(seg, ddof=0))
            basis_z[i] = (b[i] - m) / s if s > 1e-12 else 0.0

    # OI
    oi_ma_w = max(2, int(p.oi_ma_window))
    oi_d_w = max(1, int(p.oi_delta_window))
    oi_sl_w = max(2, int(p.oi_slope_window))
    oi_ma = _sma(o, oi_ma_w)
    oi_delta = np.zeros(n, dtype=np.float64)
    for i in range(oi_d_w, n):
        oi_delta[i] = float(o[i] - o[i - oi_d_w])
    oi_slope = np.zeros(n, dtype=np.float64)
    for i in range(oi_sl_w, n):
        oi_slope[i] = (float(o[i]) - float(o[i - oi_sl_w])) / float(oi_sl_w)

    # Regime: ATR percentile, ADX
    atr_reg_w = max(10, int(p.atr_regime_window))
    pct_win = max(20, int(p.atr_percentile_window))
    atr_pct = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - pct_win + 1)
        seg = atr[lo : i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 10:
            atr_pct[i] = _percentile_rank_last(seg)

    adx, pdi, mdi = _adx_di(h, l, c, max(2, int(p.adx_period)))

    return {
        "n": n,
        "squeeze_fire": squeeze_fire,
        "momentum_hist": mh,
        "momentum_hist_htf": mh_htf,
        "basis": b,
        "basis_delta_sum": basis_delta_sum,
        "basis_z": basis_z,
        "oi": o,
        "oi_ma": oi_ma,
        "oi_delta": oi_delta,
        "oi_slope": oi_slope,
        "atr": atr,
        "atr_pct": atr_pct,
        "adx": adx,
        "plus_di": pdi,
        "minus_di": mdi,
    }


def entry_masks_long_short(feats: Dict[str, Any], p: TtmOptParams) -> Tuple[np.ndarray, np.ndarray]:
    """Boolean arrays length n: signal known at bar close index i."""
    n = int(feats["n"])
    sf = feats["squeeze_fire"]
    mh = feats["momentum_hist"]
    mh_htf = feats["momentum_hist_htf"]
    b = feats["basis"]
    bds = feats["basis_delta_sum"]
    o = feats["oi"]
    o_ma = feats["oi_ma"]
    o_d = feats["oi_delta"]
    adx = feats["adx"]
    atr_pct = feats["atr_pct"]

    thr_m = float(p.momentum_threshold)
    bt = float(p.basis_threshold)

    long_m = np.zeros(n, dtype=bool)
    short_m = np.zeros(n, dtype=bool)
    for i in range(n):
        if p.use_squeeze and not bool(sf[i]):
            continue
        mom_ok_l = (not p.use_momentum) or (
            np.isfinite(mh[i]) and float(mh[i]) > thr_m and np.isfinite(mh_htf[i]) and float(mh_htf[i]) > 0
        )
        mom_ok_s = (not p.use_momentum) or (
            np.isfinite(mh[i]) and float(mh[i]) < -thr_m and np.isfinite(mh_htf[i]) and float(mh_htf[i]) < 0
        )
        basis_ok_l = (not p.use_basis) or (float(b[i]) > bt or float(bds[i]) > 0)
        basis_ok_s = (not p.use_basis) or (float(b[i]) < -bt or float(bds[i]) < 0)
        oi_ok_l = (not p.use_oi) or (
            (np.isfinite(o_ma[i]) and float(o[i]) > float(o_ma[i])) or float(o_d[i]) > 0
        )
        oi_ok_s = (not p.use_oi) or (
            (np.isfinite(o_ma[i]) and float(o[i]) < float(o_ma[i])) or float(o_d[i]) < 0
        )

        rf = str(p.regime_filter)
        if rf == "basis_oi_align":
            reg_ok_l = float(bds[i]) >= 0 and float(o_d[i]) >= 0
            reg_ok_s = float(bds[i]) <= 0 and float(o_d[i]) <= 0
            long_m[i] = bool(mom_ok_l and basis_ok_l and oi_ok_l and reg_ok_l)
            short_m[i] = bool(mom_ok_s and basis_ok_s and oi_ok_s and reg_ok_s)
            continue

        reg_ok = True
        if rf == "trend_only":
            reg_ok = bool(np.isfinite(adx[i]) and float(adx[i]) >= float(p.adx_trend_min))
        elif rf == "range_only":
            reg_ok = bool(np.isfinite(adx[i]) and float(adx[i]) < float(p.adx_trend_min))
        elif rf == "skip_high_vol":
            reg_ok = bool(np.isfinite(atr_pct[i]) and float(atr_pct[i]) < 0.75)
        elif rf == "skip_low_vol":
            reg_ok = bool(np.isfinite(atr_pct[i]) and float(atr_pct[i]) > 0.25)

        long_m[i] = bool(mom_ok_l and basis_ok_l and oi_ok_l and reg_ok)
        short_m[i] = bool(mom_ok_s and basis_ok_s and oi_ok_s and reg_ok)

    return long_m, short_m
