"""TTM feature engineering: OHLC, basis, open interest, trap score."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.backtest.data_fetcher import OhlcBar
from src.logger import get_logger
from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_alignment import (
    TTMAlignmentError,
    validate_ohlc_close_alignment,
    validate_ttm_input_alignment,
)
from src.strategies.ttm.ttm_types import SCORING_FEATURE_KEYS

logger = get_logger("ttm_features")

def _optional_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def signed_oi_signal(close_now: Any, close_prev: Any, oi_delta: Any) -> float:
    """Price vs OI alignment in {-1, 0, 1}; shared by features + signal."""
    if (
        close_now is None
        or close_prev is None
        or oi_delta is None
        or not np.isfinite(close_now)
        or not np.isfinite(close_prev)
        or not np.isfinite(oi_delta)
    ):
        return 0.0
    p_up = float(close_now) > float(close_prev)
    p_dn = float(close_now) < float(close_prev)
    oi_up = float(oi_delta) > 0.0
    oi_dn = float(oi_delta) < 0.0
    if p_up and oi_up:
        return 1.0
    if p_up and oi_dn:
        return -1.0
    if p_dn and oi_up:
        return -1.0
    if p_dn and oi_dn:
        return 1.0
    return 0.0


def rolling_zscore_levels(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score of *levels* at each index (trailing window)."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if window < 2 or n == 0:
        return out
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = x[lo : i + 1]
        if len(seg) < 2:
            continue
        m = float(np.mean(seg))
        s = float(np.std(seg, ddof=0))
        if s > 1e-12:
            out[i] = (float(x[i]) - m) / s
    return out


def _oi_level_zscore_tanh(
    oi_levels: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Trailing window over OI *levels*:

    ``oi_zscore[i] = (oi[i] - mean(window)) / (std(window) + 1e-6)``,
    ``oi_signal[i] = tanh(oi_zscore[i])``.

    Returns ``(oi_zscore, oi_signal, oi_level_std)`` (same length as ``oi_levels``).
    """
    oi_levels = np.asarray(oi_levels, dtype=np.float64)
    n = len(oi_levels)
    oi_zscore = np.zeros(n, dtype=np.float64)
    oi_signal = np.zeros(n, dtype=np.float64)
    oi_level_std = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return oi_zscore, oi_signal, oi_level_std
    w = max(2, int(window))
    eps = 1e-6
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = oi_levels[lo : i + 1]
        if len(seg) < 2:
            continue
        m = float(np.mean(seg))
        s = float(np.std(seg, ddof=0))
        oi_level_std[i] = s
        z = (float(oi_levels[i]) - m) / (s + eps)
        oi_zscore[i] = z
        oi_signal[i] = float(np.tanh(z))
    if not np.all(np.isfinite(oi_zscore)):
        oi_zscore = np.nan_to_num(oi_zscore, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(oi_signal)):
        oi_signal = np.nan_to_num(oi_signal, nan=0.0, posinf=0.0, neginf=0.0)
    return oi_zscore, oi_signal, oi_level_std


def _volume_zscore_tanh(volumes: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling z-score of *trade volume* per bar, then ``tanh``.

    ``vol_zscore[i] = (vol[i] - mean(window)) / (std(window) + 1e-6)``,
    ``vol_signal[i] = tanh(vol_zscore[i])``.
    """
    volumes = np.asarray(volumes, dtype=np.float64).ravel()
    n = len(volumes)
    vol_zscore = np.zeros(n, dtype=np.float64)
    vol_signal = np.zeros(n, dtype=np.float64)
    if n == 0:
        return vol_zscore, vol_signal
    w = max(2, int(window))
    eps = 1e-6
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = volumes[lo : i + 1]
        if len(seg) < 2:
            continue
        m = float(np.mean(seg))
        s = float(np.std(seg, ddof=0))
        z = (float(volumes[i]) - m) / (s + eps)
        vol_zscore[i] = z
        vol_signal[i] = float(np.tanh(z))
    if not np.all(np.isfinite(vol_zscore)):
        vol_zscore = np.nan_to_num(vol_zscore, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(vol_signal)):
        vol_signal = np.nan_to_num(vol_signal, nan=0.0, posinf=0.0, neginf=0.0)
    return vol_zscore, vol_signal


def _build_oi_proxy_levels(
    close: np.ndarray,
    bar_volume: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    """
    Build a synthetic OI level when real OI is absent/flat.

    Proxy intent: create smooth participation-aware level changes while avoiding
    explosive cumulative drift.
    """
    c = np.asarray(close, dtype=np.float64).ravel()
    v = np.asarray(bar_volume, dtype=np.float64).ravel()
    n = c.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    if n == 1:
        out[0] = max(1.0, abs(float(v[0])) + 1.0)
        return out
    safe_v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    vol_z = rolling_zscore_levels(safe_v, max(5, min(50, n)))
    vol_z = np.nan_to_num(vol_z, nan=0.0, posinf=0.0, neginf=0.0)
    ret = np.zeros(n, dtype=np.float64)
    prev = np.maximum(np.abs(c[:-1]), 1e-12)
    ret[1:] = np.nan_to_num((c[1:] - c[:-1]) / prev, nan=0.0, posinf=0.0, neginf=0.0)
    base = max(1.0, float(np.nanmean(np.abs(safe_v))) + 1.0)
    k = max(0.0, float(strength))
    # Use volume *level* (not cumulative) to preserve up/down transitions.
    ema = np.zeros(n, dtype=np.float64)
    ema[0] = safe_v[0]
    alpha = 2.0 / max(4.0, min(25.0, float(n) / 3.0))
    for i in range(1, n):
        ema[i] = alpha * safe_v[i] + (1.0 - alpha) * ema[i - 1]
    for i in range(1, n):
        signed_bias = np.tanh(ret[i] * 250.0) * np.tanh(vol_z[i])
        level = ema[i] * (1.0 + 0.20 * k * signed_bias)
        out[i] = max(1.0, base + float(level))
    out[0] = out[1] if n > 1 else base
    return out


def compute_positioning_strength(
    close: np.ndarray,
    basis_norm: np.ndarray,
    vol_signal: np.ndarray,
    oi_signal: np.ndarray,
    oi_zscore: np.ndarray,
    *,
    basis_norm_clip: float = 3.0,
    w_basis: float = 0.6,
    w_vol: float = 0.4,
    w_oi: float = 0.2,
) -> Dict[str, np.ndarray]:
    """
    Positioning alpha in ``[-1, 1]`` from basis + volume + OI context.

    ``basis_signal = tanh(clip(basis_norm))`` and
    ``positioning_strength = clip(w_basis*basis + w_vol*vol + w_oi*oi, -1, 1)``.
    """
    close = np.asarray(close, dtype=np.float64).ravel()
    basis_norm = np.asarray(basis_norm, dtype=np.float64).ravel()
    vol_signal = np.asarray(vol_signal, dtype=np.float64).ravel()
    oi_signal = np.asarray(oi_signal, dtype=np.float64).ravel()
    oi_zscore = np.asarray(oi_zscore, dtype=np.float64).ravel()
    n = close.size
    if n == 0:
        z = np.array([], dtype=np.float64)
        return {
            "price_change": z,
            "price_signal": z,
            "pos_core": z,
            "basis_signal": z,
            "basis_effect": z,
            "positioning_strength": z,
            "oi_zscore": z,
            "oi_signal": z,
        }
    if (
        basis_norm.size != n
        or vol_signal.size != n
        or oi_signal.size != n
        or oi_zscore.size != n
    ):
        raise ValueError("close, basis_norm, vol_signal, oi_signal, oi_zscore must have the same length")

    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close[0]
    if n >= 2:
        prev_close[1:] = close[:-1]
    price_change = close - prev_close

    price_signal = np.zeros(n, dtype=np.float64)
    price_signal[price_change > 0.0] = 1.0
    price_signal[price_change < 0.0] = -1.0

    clip = max(1e-9, float(basis_norm_clip))
    bn = np.clip(basis_norm, -clip, clip)
    basis_signal = np.tanh(bn)
    wb = float(w_basis)
    wv = float(w_vol)
    wo = float(w_oi)
    total = wb + wv + max(0.0, wo)
    if total > 1e-12:
        wb = wb / total
        wv = wv / total
        wo = max(0.0, wo) / total
    oi_core = np.tanh(np.asarray(oi_zscore, dtype=np.float64))
    pos_core = oi_core.copy()
    positioning_strength = np.clip(
        wb * basis_signal + wv * vol_signal + wo * oi_core,
        -1.0,
        1.0,
    )
    # Log field ``basis_effect`` mirrors ``basis_signal`` for downstream JSONL schema stability.
    basis_effect = basis_signal

    if not np.all(np.isfinite(positioning_strength)):
        positioning_strength = np.nan_to_num(positioning_strength, nan=0.0, posinf=1.0, neginf=-1.0)

    return {
        "price_change": price_change,
        "price_signal": price_signal,
        "pos_core": pos_core,
        "basis_signal": basis_signal,
        "basis_effect": basis_effect,
        "positioning_strength": positioning_strength,
        "oi_zscore": oi_zscore,
        "oi_signal": oi_signal,
    }


def compute_oi_features(oi_series: np.ndarray, window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    ``oi_delta`` bar-over-bar change in OI (0 at index 0).
    ``oi_z`` rolling z-score of OI *levels* with ``window`` (legacy trap / confirmation).
    """
    oi_series = np.asarray(oi_series, dtype=np.float64)
    n = len(oi_series)
    oi_delta = np.zeros(n, dtype=np.float64)
    if n >= 2:
        oi_delta[1:] = np.diff(oi_series)
    oi_z = rolling_zscore_levels(oi_series, max(2, int(window)))
    return oi_delta, oi_z


def compute_basis_features(basis_series: np.ndarray) -> np.ndarray:
    """First difference of basis; index 0 is 0."""
    b = np.asarray(basis_series, dtype=np.float64)
    n = len(b)
    basis_change = np.zeros(n, dtype=np.float64)
    if n >= 2:
        basis_change[1:] = np.diff(b)
    return basis_change


def compute_trap_score(features: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    """
    Weighted score from price failure, OI expansion / z-score, basis move and reversal.

    ``features`` must expose: failure (bool), oi_delta, oi_z, basis_change, basis_reversal (bool).
    """
    score = 0.0
    if bool(features.get("failure")):
        score += 1.0
    oi_inc_thr = float(config.get("oi_increase_threshold", 0))
    oi_z_thr = float(config.get("oi_z_threshold", 0.8))
    div_thr = float(config.get("basis_divergence_threshold", 0.2))
    br_thr = float(config.get("basis_reversal_threshold", 0.1))

    oi_ch = features.get("oi_delta")
    if oi_ch is not None and np.isfinite(oi_ch) and float(oi_ch) > oi_inc_thr:
        score += 0.8

    oi_z = features.get("oi_z")
    if oi_z is not None and np.isfinite(oi_z) and float(oi_z) > oi_z_thr:
        score += 0.7

    bc = features.get("basis_change")
    if bc is not None and np.isfinite(bc) and abs(float(bc)) > div_thr:
        score += 0.5

    if bool(features.get("basis_reversal")):
        score += 0.7

    return float(score)


def _rolling_max(a: np.ndarray, window: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n == 0:
        return out
    for i in range(n):
        lo = max(0, i - window)
        if i == 0:
            continue
        seg = a[lo:i]
        if len(seg) > 0:
            out[i] = float(np.max(seg))
    return out


def _rolling_min(a: np.ndarray, window: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n == 0:
        return out
    for i in range(n):
        lo = max(0, i - window)
        if i == 0:
            continue
        seg = a[lo:i]
        if len(seg) > 0:
            out[i] = float(np.min(seg))
    return out


def _detect_breakout_down(
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    breakout_down detection:
    1) rolling_low = low.rolling(window).min().shift(1)
    2) is_breakout_down = close < rolling_low
    Logs breakout_down count and ratio over total bars.
    """
    rolling_low = _rolling_min(np.asarray(low, dtype=np.float64), int(window))
    is_breakout_down = np.isfinite(rolling_low) & (np.asarray(close, dtype=np.float64) < rolling_low)
    n = int(is_breakout_down.size)
    count = int(np.sum(is_breakout_down))
    ratio = float(count / n) if n > 0 else 0.0
    logger.info(
        "TTM breakout_down stats",
        extra={
            "breakout_down_count": count,
            "breakout_down_ratio": ratio,
            "total_bars": n,
        },
    )
    return rolling_low, is_breakout_down


def _rolling_std(a: np.ndarray, window: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 1 or n == 0:
        return out
    for i in range(n):
        lo = max(0, i - window)
        seg = a[lo : i + 1]
        if len(seg) >= 2:
            out[i] = float(np.std(seg, ddof=0))
    return out


def _rolling_quantile_past(
    a: np.ndarray,
    window: int,
    q: float,
    *,
    mask: Optional[np.ndarray] = None,
    min_count: int = 5,
) -> np.ndarray:
    """Past-only rolling quantile: bar ``i`` uses ``[max(0, i-window), i)``."""
    arr = np.asarray(a, dtype=np.float64)
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out
    w = max(1, int(window))
    qq = float(np.clip(q, 0.0, 1.0))
    mc = max(1, int(min_count))
    use_mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - w)
        seg = arr[lo:i]
        seg_mask = use_mask[lo:i]
        vals = seg[seg_mask & np.isfinite(seg)]
        if vals.size >= mc:
            out[i] = float(np.quantile(vals, qq))
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average (same length as ``x``)."""
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


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = float(high[0]) - float(low[0])
    for i in range(1, n):
        hl = float(high[i]) - float(low[i])
        hc = abs(float(high[i]) - float(close[i - 1]))
        lc = abs(float(low[i]) - float(close[i - 1]))
        tr[i] = max(hl, hc, lc)
    return tr


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Simple ATR: rolling mean of true range."""
    tr = _true_range(high, low, close)
    n = len(tr)
    out = np.zeros(n, dtype=np.float64)
    p = max(1, int(period))
    for i in range(n):
        lo = max(0, i - p + 1)
        seg = tr[lo : i + 1]
        if len(seg) > 0:
            out[i] = float(np.mean(seg))
    return out


def _macd_histogram(close: np.ndarray) -> np.ndarray:
    """MACD histogram (line − signal); shorter periods for short series."""
    n = len(close)
    hist = np.full(n, np.nan, dtype=np.float64)
    if n < 3:
        return hist
    fast, slow, sigp = 12, 26, 9
    if n < 40:
        fast, slow, sigp = 5, 13, 5
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd_line = ema_f - ema_s
    signal_line = _ema(macd_line, sigp)
    hist[:] = macd_line - signal_line
    return hist


def _keltner_bands(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int,
    atr_period: int,
    mult: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = _ema(close, max(2, int(ema_period)))
    atr = _atr(high, low, close, max(1, int(atr_period)))
    upper = mid + float(mult) * atr
    lower = mid - float(mult) * atr
    return mid, upper, lower


def _finalize_ttm_feature_arrays(feats: Dict[str, Any]) -> None:
    """
    Replace non-finite values with 0.0 and record ``feature_valid_mask`` (per key, per bar).
    Boolean arrays are unchanged.
    """
    mask: Dict[str, np.ndarray] = {}
    for k, v in list(feats.items()):
        if k in ("n",):
            continue
        if isinstance(v, np.ndarray) and v.dtype != bool and np.issubdtype(v.dtype, np.number):
            arr = np.asarray(v, dtype=np.float64)
            mask[k] = np.isfinite(arr)
            feats[k] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    feats["feature_valid_mask"] = mask
    if __debug__:
        for k, v in feats.items():
            if k in ("n", "feature_valid_mask"):
                continue
            if isinstance(v, np.ndarray) and v.dtype != bool and np.issubdtype(v.dtype, np.number):
                assert np.all(np.isfinite(v)), f"non-finite remains in export: {k}"


def _apply_trap_memory_failure_boost(
    result: Dict[str, Any],
    lookback: int,
    *,
    min_boost: float,
    breakout_scale: float,
) -> None:
    """Lift failure_strength on the last bar when V1-style trap memory fires (lazy import)."""
    n = int(result.get("n", 0))
    if n <= 0 or lookback <= 0:
        return
    from src.strategies.ttm.ttm_signal import replay_trap_memory_for_last_bar

    ls, ll, mem = replay_trap_memory_for_last_bar(result, lookback)
    fs = np.asarray(result["failure_strength"], dtype=np.float64).copy()
    i = n - 1
    brk_arr = np.asarray(result["breakout_strength"], dtype=np.float64)
    if ls and mem.breakout_up_index is not None:
        j = int(mem.breakout_up_index)
        brk = float(brk_arr[j]) if 0 <= j < brk_arr.size else 0.0
        b = float(max(min_boost, breakout_scale * brk))
        fs[i] = max(fs[i], b)
    if ll and mem.breakout_down_index is not None:
        j = int(mem.breakout_down_index)
        brk = float(brk_arr[j]) if 0 <= j < brk_arr.size else 0.0
        b = float(max(min_boost, breakout_scale * brk))
        fs[i] = max(fs[i], b)
    result["failure_strength"] = fs


def compute_ttm_features(
    data: Dict[str, Any],
    *,
    breakout_window: int = 20,
    failure_window: int = 3,
    vol_short_window: int = 5,
    vol_long_window: int = 20,
    oi_z_window: int = 20,
    vol_regime_bound: float = 1.2,
    strict_basis_oi: bool = False,
    use_open_interest: bool = True,
    trap_memory_lookback: Optional[int] = None,
    trap_memory_min_boost: float = 0.02,
    trap_memory_breakout_scale: float = 0.5,
    failure_strength_cap: float = 3.0,
    oi_signal_level_window: int = 100,
    ttm_oi_assert_nonzero_std: bool = True,
    positioning_basis_norm_clip: float = 3.0,
    positioning_w_basis: float = 0.6,
    positioning_w_vol: float = 0.4,
    volume_z_window: int = 50,
    volume_breakout_strength_threshold: float = 0.0,
    ttm_volume_features_enabled: bool = True,
    oi_day_change_threshold: Optional[float] = None,
    oi_prev_day_close: Optional[float] = None,
    breakout_strength_min: float = 0.5,
    breakout_exhaustion_cap_window: int = 200,
    breakout_exhaustion_cap_q: float = 0.8,
    breakout_debug_stats: bool = False,
    oi_proxy_from_volume_enabled: bool = True,
    oi_proxy_strength: float = 0.35,
    breakout_exhaustion_penalty: float = 0.30,
    breakout_quality_momentum_bonus: float = 0.20,
    positioning_w_oi: float = 0.20,
    ttm_v2_effective_strength_k_extension: float = 0.0,
    ttm_v2_effective_strength_k_lastret: float = 0.0,
    ttm_v2_effective_strength_use_abs_price_z: bool = True,
) -> Dict[str, Any]:
    """
    ``data`` keys:
        - ``bars``: OHLC bars
        - ``open_interest``: optional unless ``strict_basis_oi`` and ``use_open_interest``
        - ``basis``: optional unless ``strict_basis_oi`` (then required, aligned)
        - ``oi_prev_day_close``: optional; with ``oi_day_change_threshold`` in config sets ``oi_context_flag``

    When ``strict_basis_oi`` is True, missing/misaligned/non-finite series raise
    :class:`~src.strategies.ttm.ttm_alignment.TTMAlignmentError` (no zero-fill).

    **Breakout:** ``breakout_up_raw`` / ``breakout_down_raw`` = close vs prior-window
    rolling high/low (no filters). ``breakout_up`` / ``breakout_down`` = tradable:
    raw AND ATR-normalized strength above ``breakout_strength_min``.
    """
    bars: Sequence[Any] = data.get("bars") or []
    close_in = data.get("close")
    if close_in is not None:
        validate_ohlc_close_alignment(close_in, bars)
    n = len(bars)
    if n == 0:
        return {"n": 0}

    validate_ttm_input_alignment(
        data,
        strict_basis_oi=strict_basis_oi,
        use_open_interest=use_open_interest,
    )

    high = np.zeros(n, dtype=np.float64)
    low = np.zeros(n, dtype=np.float64)
    close = np.zeros(n, dtype=np.float64)
    bar_volume = np.zeros(n, dtype=np.float64)

    for i, b in enumerate(bars):
        if isinstance(b, OhlcBar):
            high[i] = float(b.high)
            low[i] = float(b.low)
            close[i] = float(b.close)
            bar_volume[i] = float(getattr(b, "volume", 0.0) or 0.0)
        else:
            high[i] = float(b.get("high", 0) or 0)
            low[i] = float(b.get("low", 0) or 0)
            close[i] = float(b.get("close", 0) or 0)
            bar_volume[i] = float(b.get("volume", 0) or 0)

    rh_excl = _rolling_max(high, breakout_window)
    rl_excl, is_breakout_down = _detect_breakout_down(low, close, breakout_window)

    ret = np.zeros(n, dtype=np.float64)
    ret[1:] = np.diff(close) / np.maximum(close[:-1], 1e-12)

    std_s = _rolling_std(ret, vol_short_window)
    std_l = _rolling_std(ret, vol_long_window)
    vol_spike = np.ones(n, dtype=np.float64)
    mask_vs = (std_l > 1e-12) & ~np.isnan(std_l) & ~np.isnan(std_s)
    vol_spike[mask_vs] = std_s[mask_vs] / std_l[mask_vs]

    # --- OI ---
    oi_raw: Optional[Sequence[Any]] = data.get("open_interest")
    oi = np.zeros(n, dtype=np.float64)
    if strict_basis_oi and use_open_interest:
        oi[:] = np.asarray(oi_raw, dtype=np.float64).ravel()
    elif oi_raw is not None:
        oi_list = list(oi_raw)
        if len(oi_list) == n:
            oi[:] = np.asarray(oi_list, dtype=np.float64)
        elif len(oi_list) > 0 and not strict_basis_oi:
            oi[:] = float(oi_list[-1])
        elif len(oi_list) > 0:
            raise TTMAlignmentError(
                "strict_basis_oi: open_interest must have length equal to bars (no broadcast)"
            )

    oi_proxy_used = False
    oi_ff = oi.copy()
    for i in range(1, n):
        if not np.isfinite(oi_ff[i]):
            oi_ff[i] = oi_ff[i - 1]
    # Validation requires meaningful OI up/down splits; for sparse OI streams use
    # a volume/participation proxy to avoid degenerate positioning statistics.
    oi_std = float(np.nanstd(oi_ff)) if oi_ff.size > 0 else 0.0
    if (
        bool(oi_proxy_from_volume_enabled)
        and (oi_raw is None or oi_ff.size == 0 or oi_std <= 1e-12)
    ):
        oi_ff = _build_oi_proxy_levels(
            close,
            bar_volume,
            strength=float(oi_proxy_strength),
        )
        oi_proxy_used = True

    oi_delta, oi_z = compute_oi_features(oi_ff, window=max(2, min(oi_z_window, n)))
    # prev_oi[t] = oi_ff[t-1]; prev_oi[0] = oi_ff[0] (no prior bar)
    prev_oi = np.zeros(n, dtype=np.float64)
    if n >= 1:
        prev_oi[0] = oi_ff[0]
    if n >= 2:
        prev_oi[1:] = oi_ff[:-1]

    oi_raw_len = len(np.asarray(oi_raw).ravel()) if oi_raw is not None else 0
    if (
        ttm_oi_assert_nonzero_std
        and use_open_interest
        and oi_raw is not None
        and oi_raw_len == n
        and n >= 50
    ):
        lvl_50 = float(np.std(oi_ff[-50:], ddof=0))
        if lvl_50 <= 1e-18:
            raise RuntimeError("OI pipeline broken: std(open_interest levels over 50 bars) == 0")

    lw = max(2, int(oi_signal_level_window))
    oi_zscore, oi_signal, oi_level_std = _oi_level_zscore_tanh(oi_ff, lw)

    vz_w = max(2, int(volume_z_window))
    if bool(ttm_volume_features_enabled):
        vol_zscore, vol_signal = _volume_zscore_tanh(bar_volume, vz_w)
    else:
        vol_zscore = np.zeros(n, dtype=np.float64)
        vol_signal = np.zeros(n, dtype=np.float64)
    # Intraday participation for logs: volume only (OI kept in oi_pipeline for context).
    participation_strength = np.asarray(vol_signal, dtype=np.float64).copy()

    # --- Basis ---
    basis_in = data.get("basis")
    if strict_basis_oi:
        basis = np.asarray(basis_in, dtype=np.float64).ravel()
    elif basis_in is not None and len(list(basis_in)) == n:
        basis = np.asarray(basis_in, dtype=np.float64).ravel()
    else:
        basis = np.zeros(n, dtype=np.float64)

    basis_change = compute_basis_features(basis)
    basis_delta = basis_change.copy()
    atr = _atr(high, low, close, period=max(5, min(14, n)))
    basis_norm = np.zeros(n, dtype=np.float64)
    atr_mask = atr > 1e-12
    basis_norm[atr_mask] = basis[atr_mask] / atr[atr_mask]

    # --- Breakout: raw (price vs prior rolling extrema) vs tradable (raw + strength threshold) ---
    range_n = np.maximum(rh_excl - rl_excl, 0.0)
    break_strength_up = np.zeros(n, dtype=np.float64)
    break_strength_down = np.zeros(n, dtype=np.float64)
    close_pos = np.zeros(n, dtype=np.float64)

    hl = np.maximum(high - low, 1e-12)
    close_pos[:] = np.clip((close - low) / hl, 0.0, 1.0)

    brk_str_min = float(breakout_strength_min)

    valid_atr = atr > 1e-12
    range_expansion = np.zeros(n, dtype=np.float64)
    range_expansion[valid_atr] = range_n[valid_atr] / atr[valid_atr]
    compression = range_expansion.copy()

    # Short-only close position: near low => stronger downside continuation context.
    close_pos_down = np.zeros(n, dtype=np.float64)
    close_pos_down[:] = (high - close) / (np.maximum(high - low, 0.0) + 1e-6)
    close_pos_down = np.clip(close_pos_down, 0.0, 1.0)

    valid_up = valid_atr & np.isfinite(rh_excl)
    valid_dn = valid_atr & np.isfinite(rl_excl)
    break_strength_up[valid_up] = (close[valid_up] - rh_excl[valid_up]) / atr[valid_up]
    break_strength_down[valid_dn] = (rl_excl[valid_dn] - close[valid_dn]) / atr[valid_dn]

    is_breakout_up = np.isfinite(rh_excl) & (close > rh_excl)

    breakout_up_raw = is_breakout_up
    breakout_down_raw = is_breakout_down
    raw_strength_up = np.full(n, np.nan, dtype=np.float64)
    raw_strength_up[valid_up] = break_strength_up[valid_up]
    breakout_strength_cap = _rolling_quantile_past(
        raw_strength_up,
        max(2, int(breakout_exhaustion_cap_window)),
        float(breakout_exhaustion_cap_q),
        mask=breakout_up_raw,
        min_count=5,
    )
    cap_eval = np.where(np.isfinite(breakout_strength_cap), breakout_strength_cap, np.inf)
    exhaustion_candidate = np.isfinite(raw_strength_up) & (raw_strength_up > cap_eval)
    breakout_up = (
        breakout_up_raw
        & np.isfinite(raw_strength_up)
        & (raw_strength_up > brk_str_min)
        & (~exhaustion_candidate)
    )
    exhaustion_confirm = np.zeros(n, dtype=bool)
    short_score = np.full(n, np.nan, dtype=np.float64)
    short_setup_high = np.full(n, np.nan, dtype=np.float64)
    short_setup_atr = np.full(n, np.nan, dtype=np.float64)
    short_setup_bar_index = np.full(n, np.nan, dtype=np.float64)
    short_setup_raw_strength = np.full(n, np.nan, dtype=np.float64)
    short_setup_cap = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        j = i - 1
        if not bool(exhaustion_candidate[j]):
            continue
        if not (np.isfinite(close[i]) and np.isfinite(close[j]) and float(close[i]) < float(close[j])):
            continue
        exhaustion_confirm[i] = True
        short_score[i] = float(raw_strength_up[j] - cap_eval[j])
        short_setup_high[i] = float(high[j])
        short_setup_atr[i] = float(atr[j])
        short_setup_bar_index[i] = float(j)
        short_setup_raw_strength[i] = float(raw_strength_up[j])
        short_setup_cap[i] = float(cap_eval[j])
    breakout_down = is_breakout_down & (break_strength_down > brk_str_min)

    strength = np.zeros(n, dtype=np.float64)
    strength[breakout_up] = raw_strength_up[breakout_up]
    strength[breakout_down] = break_strength_down[breakout_down]

    breakout_flag = breakout_up | breakout_down
    if breakout_debug_stats:
        n_raw_u = int(np.sum(breakout_up_raw))
        n_filt_u = int(np.sum(breakout_up))
        n_exh_u = int(np.sum(exhaustion_candidate))
        n_raw_d = int(np.sum(breakout_down_raw))
        n_filt_d = int(np.sum(breakout_down))
        print("breakout_up_raw_count:", n_raw_u)
        print("breakout_up_filtered_count:", n_filt_u)
        print("breakout_up_exhaustion_count:", n_exh_u)
        print("breakout_up_raw_ratio:", float(np.mean(breakout_up_raw)))
        print("breakout_up_filtered_ratio:", float(np.mean(breakout_up)))
        print("breakout_down_raw_count:", n_raw_d)
        print("breakout_down_filtered_count:", n_filt_d)
        re_fin = range_expansion[valid_atr]
        bs_up_fin = raw_strength_up[breakout_up_raw & np.isfinite(raw_strength_up)]
        cap_fin = breakout_strength_cap[np.isfinite(breakout_strength_cap)]
        if re_fin.size > 0:
            re_q = np.quantile(re_fin, [0.1, 0.5, 0.9]).tolist()
        else:
            re_q = [0.0, 0.0, 0.0]
        if bs_up_fin.size > 0:
            bs_q = np.quantile(bs_up_fin, [0.1, 0.5, 0.8, 0.9, 0.95]).tolist()
        else:
            bs_q = [0.0, 0.0, 0.0, 0.0, 0.0]
        if cap_fin.size > 0:
            cap_q = np.quantile(cap_fin, [0.1, 0.5, 0.9]).tolist()
            cap_last = float(cap_fin[-1])
        else:
            cap_q = [0.0, 0.0, 0.0]
            cap_last = float("nan")
        logger.info(
            "TTM breakout debug stats",
            extra={
                "n_bars": n,
                "breakout_count_filtered": int(np.sum(breakout_flag)),
                "breakout_up_raw_count": n_raw_u,
                "breakout_up_filtered_count": n_filt_u,
                "breakout_up_exhaustion_count": n_exh_u,
                "breakout_down_raw_count": n_raw_d,
                "breakout_down_filtered_count": n_filt_d,
                "breakout_up_raw_ratio": float(np.mean(breakout_up_raw)),
                "breakout_up_filtered_ratio": float(np.mean(breakout_up)),
                "range_expansion_q10_q50_q90": [float(x) for x in re_q],
                "raw_strength_up_q10_q50_q80_q90_q95": [float(x) for x in bs_q],
                "breakout_strength_cap_q10_q50_q90": [float(x) for x in cap_q],
                "breakout_strength_cap_last": cap_last,
            },
        )

    # Reversal: SHORT — prior bar expanded basis, current weakens (bearish for basis)
    basis_reversal_short = np.zeros(n, dtype=bool)
    basis_reversal_long = np.zeros(n, dtype=bool)
    div_hint = 0.2
    rev_hint = 0.05
    for i in range(1, n):
        b_prev = basis_change[i - 1]
        b_now = basis_change[i]
        if b_prev > div_hint and b_now < -rev_hint:
            basis_reversal_short[i] = True
        if b_prev < -div_hint and b_now > rev_hint:
            basis_reversal_long[i] = True

    # Continuous failure depth (same relative scale as breakout_strength: vs rolling level).
    failure_depth_up = np.zeros(n, dtype=np.float64)
    failure_depth_down = np.zeros(n, dtype=np.float64)
    K = max(1, int(failure_window))
    for i in range(1, n):
        start = max(0, i - K + 1)
        best_u = 0.0
        best_d = 0.0
        for j in range(start, i):
            if breakout_up[j]:
                rh = float(rh_excl[j])
                if close[i] < rh and rh > 1e-12:
                    depth = (rh - close[i]) / rh
                    if depth > best_u:
                        best_u = depth
            if breakout_down[j]:
                rl = float(rl_excl[j])
                if close[i] > rl and rl > 1e-12:
                    depth = (close[i] - rl) / rl
                    if depth > best_d:
                        best_d = depth
        failure_depth_up[i] = best_u
        failure_depth_down[i] = best_d

    cap_fs = max(1e-9, float(failure_strength_cap))
    failure_strength = np.clip(
        np.maximum(failure_depth_up, failure_depth_down), 0.0, cap_fs
    )
    eps_fail = 1e-12
    failure_up = failure_depth_up > eps_fail
    failure_down = failure_depth_down > eps_fail

    ema_p = 20 if n >= 20 else max(3, n // 2)
    atr_p = 10 if n >= 10 else max(2, n // 3)
    macd_hist = _macd_histogram(close)
    kc_mid, kc_upper, kc_lower = _keltner_bands(high, low, close, ema_p, atr_p, 2.0)

    vt = float(vol_regime_bound)
    vt = max(vt, 1e-9)
    vol_regime = np.zeros(n, dtype=np.float64)
    vs_fin = np.isfinite(vol_spike)
    vol_regime[(vol_spike < 1.0 / vt) & vs_fin] = -1.0
    vol_regime[(vol_spike > vt) & vs_fin] = 1.0

    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close[0]
    if n >= 2:
        prev_close[1:] = close[:-1]
    momentum_1 = close - prev_close
    momentum_1_z = rolling_zscore_levels(momentum_1, max(2, int(breakout_window)))
    range_expansion_z = rolling_zscore_levels(range_expansion, max(2, int(breakout_window)))
    close_pos_down_z = rolling_zscore_levels(close_pos_down, max(2, int(breakout_window)))
    if not np.all(np.isfinite(momentum_1_z)):
        momentum_1_z = np.nan_to_num(momentum_1_z, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(range_expansion_z)):
        range_expansion_z = np.nan_to_num(range_expansion_z, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(close_pos_down_z)):
        close_pos_down_z = np.nan_to_num(close_pos_down_z, nan=0.0, posinf=0.0, neginf=0.0)

    breakout_strength_down = (
        0.5 * range_expansion_z
        + 0.3 * close_pos_down_z
        + 0.2 * vol_zscore
    )
    if not np.all(np.isfinite(breakout_strength_down)):
        breakout_strength_down = np.nan_to_num(
            breakout_strength_down, nan=0.0, posinf=0.0, neginf=0.0
        )

    breakout_strength_base = np.asarray(strength, dtype=np.float64).copy()
    strength_z = rolling_zscore_levels(breakout_strength_base, max(2, int(breakout_window)))
    if not np.all(np.isfinite(strength_z)):
        strength_z = np.nan_to_num(strength_z, nan=0.0, posinf=0.0, neginf=0.0)
    strength_validated = np.zeros(n, dtype=np.float64)
    strength_validated[breakout_up] = raw_strength_up[breakout_up]
    strength_validated[breakout_down] = breakout_strength_down[breakout_down]
    strength_validated[exhaustion_candidate] = np.nan
    _ = float(volume_breakout_strength_threshold)  # legacy config key; no longer gates volume boost

    pos_pack = compute_positioning_strength(
        close,
        basis_norm,
        vol_signal,
        oi_signal,
        oi_zscore,
        basis_norm_clip=float(positioning_basis_norm_clip),
        w_basis=float(positioning_w_basis),
        w_vol=float(positioning_w_vol),
        w_oi=float(positioning_w_oi),
    )

    # --- V2-only calibration inputs (causal): extension from price level z; prior-bar return; effective_strength ---
    w_px = max(2, int(breakout_window))
    price_z = rolling_zscore_levels(close, w_px)
    if bool(ttm_v2_effective_strength_use_abs_price_z):
        extension = np.abs(np.asarray(price_z, dtype=np.float64))
    else:
        extension = np.asarray(price_z, dtype=np.float64).copy()
    last_bar_return = np.full(n, np.nan, dtype=np.float64)
    for _i in range(2, n):
        c0 = float(close[_i - 2])
        if abs(c0) > 1e-12 and np.isfinite(close[_i - 1]):
            last_bar_return[_i] = (float(close[_i - 1]) - c0) / c0
    k_ext = float(ttm_v2_effective_strength_k_extension)
    k_lr = float(ttm_v2_effective_strength_k_lastret)
    effective_strength = np.full(n, np.nan, dtype=np.float64)
    for _i in range(n):
        rs_i = raw_strength_up[_i]
        if not np.isfinite(rs_i):
            continue
        ext_i = float(extension[_i]) if np.isfinite(extension[_i]) else 0.0
        lb = last_bar_return[_i]
        pen_ret = k_lr * max(0.0, float(lb)) if np.isfinite(lb) else 0.0
        effective_strength[_i] = float(rs_i) - k_ext * ext_i - pen_ret

    result: Dict[str, Any] = {
        "n": n,
        "close": close,
        "bar_volume": np.asarray(bar_volume, dtype=np.float64).copy(),
        "vol_zscore": vol_zscore,
        "vol_signal": vol_signal,
        "participation_strength": participation_strength,
        "oi_level": np.asarray(oi_ff, dtype=np.float64).copy(),
        "prev_oi": prev_oi,
        "oi_level_std": oi_level_std,
        "oi_proxy_used": np.full(n, oi_proxy_used, dtype=bool),
        "rolling_high": rh_excl,
        "rolling_low": rl_excl,
        "range_n": range_n,
        "range_expansion": range_expansion,
        "compression": compression,
        "close_pos": close_pos,
        "breakout_flag": breakout_flag,
        "break_strength": strength_validated,
        "raw_strength": raw_strength_up,
        "price_z": np.asarray(price_z, dtype=np.float64).copy(),
        "extension": np.asarray(extension, dtype=np.float64).copy(),
        "last_bar_return": last_bar_return,
        "effective_strength": effective_strength,
        "cap": breakout_strength_cap,
        "breakout_up_raw": breakout_up_raw,
        "breakout_down_raw": breakout_down_raw,
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
        "exhaustion_candidate": exhaustion_candidate,
        "exhaustion_confirm": exhaustion_confirm,
        "short_score": short_score,
        "short_setup_high": short_setup_high,
        "short_setup_atr": short_setup_atr,
        "short_setup_bar_index": short_setup_bar_index,
        "short_setup_raw_strength": short_setup_raw_strength,
        "short_setup_cap": short_setup_cap,
        "breakout_strength_base": breakout_strength_base,
        "breakout_strength_cap": breakout_strength_cap,
        "breakout_strength_z": strength_z,
        "breakout_strength": strength_validated,
        "momentum_1": momentum_1,
        "momentum_1_z": momentum_1_z,
        "range_expansion_z": range_expansion_z,
        "close_pos_down": close_pos_down,
        "close_pos_down_z": close_pos_down_z,
        "breakout_strength_down": breakout_strength_down,
        "failure_strength": failure_strength,
        "vol_regime": vol_regime,
        "oi_signal": oi_signal,
        "vol_spike": vol_spike,
        "oi_delta": oi_delta,
        "oi_zscore": oi_zscore,
        "oi_z": oi_z,
        "basis": basis,
        "basis_change": basis_change,
        "basis_delta": basis_delta,
        "basis_norm": basis_norm,
        "atr": atr,
        "basis_reversal_short": basis_reversal_short,
        "basis_reversal_long": basis_reversal_long,
        "failure_up": failure_up,
        "failure_down": failure_down,
        "macd_hist": macd_hist,
        "kc_upper": kc_upper,
        "kc_lower": kc_lower,
        "kc_mid": kc_mid,
        "failure_depth_up": failure_depth_up,
        "failure_depth_down": failure_depth_down,
        "price_change": pos_pack["price_change"],
        "price_signal": pos_pack["price_signal"],
        "pos_core": pos_pack["pos_core"],
        "basis_signal": pos_pack["basis_signal"],
        "basis_effect": pos_pack["basis_effect"],
        "positioning_strength": pos_pack["positioning_strength"],
        "positioning_oi_core": pos_pack["pos_core"],
    }
    if trap_memory_lookback is not None and int(trap_memory_lookback) > 0:
        _apply_trap_memory_failure_boost(
            result,
            int(trap_memory_lookback),
            min_boost=float(trap_memory_min_boost),
            breakout_scale=float(trap_memory_breakout_scale),
        )
    brk_f = np.asarray(result["breakout_strength"], dtype=np.float64)
    vs_f = np.asarray(result["vol_signal"], dtype=np.float64)
    fs_f = np.asarray(result["failure_strength"], dtype=np.float64)
    result["alpha_trap_score"] = brk_f * (1.0 - vs_f) * fs_f
    _finalize_ttm_feature_arrays(result)
    oi_ctx = "NORMAL"
    thr_d = oi_day_change_threshold
    if (
        thr_d is not None
        and oi_prev_day_close is not None
        and np.isfinite(thr_d)
        and np.isfinite(float(oi_prev_day_close))
        and n > 0
    ):
        oi_last = float(oi_ff[n - 1])
        if np.isfinite(oi_last) and abs(oi_last - float(oi_prev_day_close)) > float(thr_d):
            oi_ctx = "OI_SPIKE"
    result["oi_context_flag"] = oi_ctx
    return result


def features_last_row(features: Dict[str, Any]) -> Dict[str, Any]:
    n = int(features.get("n", 0))
    if n <= 0:
        return {}
    i = n - 1
    out: Dict[str, Any] = {}
    for key in (
        "rolling_high",
        "rolling_low",
        "range_n",
        "range_expansion",
        "range_expansion_z",
        "close_pos_down",
        "close_pos_down_z",
        "breakout_strength_down",
        "compression",
        "close_pos",
        "break_strength",
        "raw_strength",
        "price_z",
        "extension",
        "last_bar_return",
        "effective_strength",
        "cap",
        "breakout_strength",
        "breakout_strength_base",
        "breakout_strength_cap",
        "breakout_strength_z",
        "short_score",
        "short_setup_high",
        "short_setup_atr",
        "short_setup_bar_index",
        "short_setup_raw_strength",
        "short_setup_cap",
        "bar_volume",
        "vol_zscore",
        "vol_signal",
        "participation_strength",
        "failure_strength",
        "alpha_trap_score",
        "vol_spike",
        "vol_regime",
        "oi_signal",
        "oi_delta",
        "oi_zscore",
        "oi_z",
        "basis",
        "basis_change",
        "basis_delta",
        "basis_norm",
        "atr",
        "positioning_strength",
        "price_change",
        "price_signal",
        "pos_core",
        "basis_signal",
        "basis_effect",
        "positioning_oi_core",
        "momentum_1",
        "momentum_1_z",
    ):
        arr = features.get(key)
        if isinstance(arr, np.ndarray) and arr.size > i:
            out[key] = float(arr[i])
        else:
            out[key] = 0.0

    fvm = features.get("feature_valid_mask") or {}
    out_vm: Dict[str, bool] = {}
    for key in SCORING_FEATURE_KEYS:
        marr = fvm.get(key) if isinstance(fvm, dict) else None
        if isinstance(marr, np.ndarray) and marr.size > i:
            out_vm[key] = bool(marr[i])
        else:
            out_vm[key] = True
    for key in (
        "raw_strength",
        "cap",
        "breakout_strength_cap",
        "short_score",
        "short_setup_high",
        "short_setup_atr",
        "short_setup_raw_strength",
        "short_setup_cap",
        "price_z",
        "extension",
        "last_bar_return",
        "effective_strength",
    ):
        marr = fvm.get(key) if isinstance(fvm, dict) else None
        if isinstance(marr, np.ndarray) and marr.size > i:
            out_vm[key] = bool(marr[i])
        else:
            out_vm[key] = True
    out["feature_valid_mask"] = out_vm
    for key in (
        "breakout_flag",
        "breakout_up_raw",
        "breakout_down_raw",
        "breakout_up",
        "breakout_down",
        "exhaustion_candidate",
        "exhaustion_confirm",
        "failure_up",
        "failure_down",
        "basis_reversal_short",
        "basis_reversal_long",
    ):
        arr = features.get(key)
        if isinstance(arr, np.ndarray) and arr.size > i:
            out[key] = bool(arr[i])
        else:
            out[key] = False

    close = features.get("close")
    mh = features.get("macd_hist")
    kcu = features.get("kc_upper")
    kcl = features.get("kc_lower")
    bc = features.get("basis")
    bch = features.get("basis_change")
    mom = None
    mom_prev = None
    if isinstance(mh, np.ndarray) and mh.size > i:
        v = mh[i]
        mom = float(v) if np.isfinite(v) else None
    if isinstance(mh, np.ndarray) and mh.size > i and i >= 1:
        v = mh[i - 1]
        mom_prev = float(v) if np.isfinite(v) else None
    out["momentum"] = mom
    out["macd_hist_prev"] = mom_prev

    c_now = float(close[i]) if isinstance(close, np.ndarray) and close.size > i else None
    c_prev = None
    if isinstance(close, np.ndarray) and close.size > i and i >= 1:
        c_prev = float(close[i - 1])
    out["close_prev"] = c_prev

    squeeze_fail_exit_long = False
    squeeze_fail_exit_short = False
    if (
        isinstance(close, np.ndarray)
        and close.size > i
        and i >= 1
        and isinstance(kcu, np.ndarray)
        and kcu.size > i
        and isinstance(kcl, np.ndarray)
        and kcl.size > i
    ):
        squeeze_fail_exit_long = bool(close[i - 1] > kcu[i - 1] and close[i] < kcu[i])
        squeeze_fail_exit_short = bool(close[i - 1] < kcl[i - 1] and close[i] > kcl[i])
    out["squeeze_fail_exit_long"] = squeeze_fail_exit_long
    out["squeeze_fail_exit_short"] = squeeze_fail_exit_short

    for key in ("oi_level", "prev_oi", "oi_level_std"):
        arr = features.get(key)
        if isinstance(arr, np.ndarray) and arr.size > i:
            v = float(arr[i])
            out[key] = v if np.isfinite(v) else 0.0
        else:
            out[key] = 0.0
    arr_proxy = features.get("oi_proxy_used")
    if isinstance(arr_proxy, np.ndarray) and arr_proxy.size > i:
        out["oi_proxy_used"] = bool(arr_proxy[i])
    else:
        out["oi_proxy_used"] = False
    out["oi_pipeline"] = {
        "oi": float(out.get("oi_level", 0.0)),
        "oi_zscore": float(out.get("oi_zscore", 0.0)),
        "oi_signal": float(out.get("oi_signal", 0.0)),
        "oi_proxy_used": bool(out.get("oi_proxy_used", False)),
    }
    out["volume_participation_log"] = {
        "volume": float(out.get("bar_volume", 0.0)),
        "vol_zscore": float(out.get("vol_zscore", 0.0)),
        "vol_signal": float(out.get("vol_signal", 0.0)),
        "participation_strength": float(out.get("participation_strength", 0.0)),
    }
    out["positioning_log"] = {
        "price_change": float(out.get("price_change", 0.0)),
        "price_signal": float(out.get("price_signal", 0.0)),
        "oi_zscore": float(out.get("oi_zscore", 0.0)),
        "oi_signal": float(out.get("oi_signal", 0.0)),
        "pos_core": float(out.get("pos_core", 0.0)),
        "basis_norm": float(out.get("basis_norm", 0.0)),
        "basis_signal": float(out.get("basis_signal", 0.0)),
        "basis_effect": float(out.get("basis_effect", 0.0)),
        "positioning_oi_core": float(out.get("positioning_oi_core", 0.0)),
        "oi_proxy_used": bool(out.get("oi_proxy_used", False)),
        "positioning_strength": float(out.get("positioning_strength", 0.0)),
    }
    out["oi_context_flag"] = str(features.get("oi_context_flag") or "NORMAL")

    return out


def compute_ttm_features_from_config(raw: Dict[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    """Single entry for signal paths: maps TTM config keys to :func:`compute_ttm_features`."""
    fail_w = int(config.get("failure_window", 3))
    trap_lb = int(config.get("trap_lookback_bars", 5))
    strict = bool(config.get("ttm_strict_basis_oi", False))
    use_oi = bool(config.get("use_open_interest", True))
    tm_lb = int(trap_lb) if bool(config.get("ttm_v2_use_trap_memory", False)) else None
    return compute_ttm_features(
        raw,
        breakout_window=int(config.get("breakout_window", 20)),
        failure_window=max(fail_w, trap_lb),
        vol_short_window=int(config.get("vol_short_window", 5)),
        vol_long_window=int(config.get("vol_long_window", 20)),
        oi_z_window=int(config.get("oi_z_window", 20)),
        vol_regime_bound=float(config.get("vol_regime_bound", config.get("vol_threshold", 1.2))),
        strict_basis_oi=strict,
        use_open_interest=use_oi,
        trap_memory_lookback=tm_lb,
        trap_memory_min_boost=float(config.get("ttm_trap_memory_min_boost", 0.02)),
        trap_memory_breakout_scale=float(config.get("ttm_trap_memory_breakout_scale", 0.5)),
        failure_strength_cap=float(config.get("ttm_failure_strength_cap", 3.0)),
        oi_signal_level_window=int(
            config.get(
                "oi_signal_level_window",
                config.get("oi_signal_std_window", 100),
            )
        ),
        ttm_oi_assert_nonzero_std=bool(config.get("ttm_oi_assert_nonzero_std", True)),
        positioning_basis_norm_clip=float(config.get("positioning_basis_norm_clip", 3.0)),
        positioning_w_basis=float(config.get("positioning_w_basis", 0.6)),
        positioning_w_vol=float(config.get("positioning_w_vol", 0.4)),
        volume_z_window=int(config.get("volume_z_window", 50)),
        volume_breakout_strength_threshold=float(config.get("volume_breakout_strength_threshold", 0.0)),
        ttm_volume_features_enabled=bool(config.get("ttm_volume_features_enabled", True)),
        oi_day_change_threshold=_optional_float(config.get("oi_day_change_threshold")),
        oi_prev_day_close=_optional_float(raw.get("oi_prev_day_close")),
        breakout_strength_min=float(config.get("breakout_strength_min", 0.5)),
        breakout_exhaustion_cap_window=int(config.get("ttm_breakout_exhaustion_cap_window", 200)),
        breakout_exhaustion_cap_q=float(config.get("ttm_breakout_exhaustion_cap_q", 0.8)),
        breakout_debug_stats=bool(config.get("ttm_breakout_debug_stats", False)),
        oi_proxy_from_volume_enabled=bool(config.get("ttm_oi_proxy_from_volume_enabled", True)),
        oi_proxy_strength=float(config.get("ttm_oi_proxy_strength", 0.35)),
        breakout_exhaustion_penalty=float(config.get("ttm_breakout_exhaustion_penalty", 0.30)),
        breakout_quality_momentum_bonus=float(config.get("ttm_breakout_quality_momentum_bonus", 0.20)),
        positioning_w_oi=float(config.get("positioning_w_oi", 0.20)),
        ttm_v2_effective_strength_k_extension=float(
            config.get("ttm_v2_effective_strength_k_extension", 0.0)
        ),
        ttm_v2_effective_strength_k_lastret=float(config.get("ttm_v2_effective_strength_k_lastret", 0.0)),
        ttm_v2_effective_strength_use_abs_price_z=bool(
            config.get("ttm_v2_effective_strength_use_abs_price_z", True)
        ),
    )


def compute_features(data: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build the full TTM feature bundle from ``data`` (``bars``, optional ``basis``, ``open_interest``)."""
    cfg = {**TTM_CONFIG, **(dict(config) if config else {})}
    return compute_ttm_features_from_config(dict(data), cfg)
