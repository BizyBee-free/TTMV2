"""
Unified TTM scoring: bounded blocks, symmetric long/short, softmax probs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from src.strategies.ttm.ttm_types import TTMFeatureVector
from src.strategies.ttm.ttm_weights import RegimeWeights


def _v2_feat_slice(feats: Mapping[str, Any], key: str, n: int) -> np.ndarray:
    v = feats.get(key)
    if not isinstance(v, np.ndarray):
        return np.zeros(n, dtype=np.float64)
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size >= n:
        return np.nan_to_num(a[:n].copy(), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(n, dtype=np.float64)
    out[: a.size] = np.nan_to_num(a[: a.size], nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _v2_bool_slice(feats: Mapping[str, Any], key: str, n: int) -> np.ndarray:
    v = feats.get(key)
    if not isinstance(v, np.ndarray):
        return np.zeros(n, dtype=bool)
    a = np.asarray(v).ravel()
    if a.size >= n:
        return a[:n].astype(bool, copy=False)
    out = np.zeros(n, dtype=bool)
    out[: a.size] = a.astype(bool, copy=False)
    return out


def _forward_returns_close(close: np.ndarray, h: int) -> np.ndarray:
    """Bar-i forward return to i+h (fraction); NaN where undefined."""
    c = np.asarray(close, dtype=np.float64).ravel()
    n = c.size
    out = np.full(n, np.nan, dtype=np.float64)
    if h <= 0 or n == 0:
        return out
    for i in range(n - h):
        a, b = float(c[i]), float(c[i + h])
        if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-12:
            out[i] = (b - a) / abs(a)
    return out


def _pearson_corr(x: np.ndarray, y: np.ndarray, min_n: int) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < max(5, int(min_n)):
        return 0.0
    xa = x[m].astype(np.float64, copy=False)
    ya = y[m].astype(np.float64, copy=False)
    if float(np.std(xa)) < 1e-12:
        return 0.0
    r = float(np.corrcoef(xa, ya)[0, 1])
    if not np.isfinite(r):
        return 0.0
    return r


def _alignment_factor(
    corr: float,
    *,
    guard_enabled: bool,
    flip_max: float,
) -> float:
    """
    Convert trailing correlation into a bounded sign-alignment factor.

    - guard disabled: legacy hard sign flip (+1 / -1)
    - guard enabled: never fully invert on weak evidence; bounded in ``[1-flip_max, 1+flip_max]``.
    """
    if not np.isfinite(corr):
        return 1.0
    if not guard_enabled:
        return -1.0 if corr < 0.0 else 1.0
    c = float(np.clip(corr, -1.0, 1.0))
    f = 1.0 + float(np.clip(c, -abs(flip_max), abs(flip_max)))
    return float(np.clip(f, 1.0 - abs(flip_max), 1.0 + abs(flip_max)))


def _percentile_rank_in_window(window: np.ndarray, value: float) -> float:
    w = window[np.isfinite(window)]
    if w.size == 0:
        return 0.5
    v = float(value)
    sw = np.sort(w)
    k = int(np.searchsorted(sw, v, side="right"))
    return float(k / max(sw.size, 1))


def compute_score_v2_alpha(
    feats: Mapping[str, Any],
    last: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    bar_index: Optional[int] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    TTM V2 unified symmetric alpha (no standalone vol_z term; vol gate / sizing in signal layer).

    - Breakout leg uses ``breakout_strength_base`` (ATR breakout only, no volume in that scalar).
    - Optional: ``breakout_signed *= (1 + k * tanh(vol_z))`` when ``ttm_v2_vol_breakout_interaction_k != 0``.
    - ``alpha_raw = w1*breakout + w2*price_momentum + w3*basis_mom`` on signed / aligned features.
    - Feature signs vs forward return: Pearson correlation on a trailing window; flip if rho < 0.
    - ``alpha_rank`` = percentile rank of current ``alpha_raw`` in trailing window.
    - ``score`` = clip((2*alpha_rank - 1) * cap, -cap, cap); logits for probs are ``(score, -score)``
      via existing :func:`scores_to_probs`.
    """
    if bool(config.get("ttm_strict_feature_finite", False)):
        for name in (
            "breakout_strength_base",
            "breakout_strength",
            "momentum_1_z",
            "basis_signal",
        ):
            v = last.get(name)
            if v is not None and not np.isfinite(float(v)):
                raise ValueError(f"ttm_strict_feature_finite: non-finite {name}={v!r}")

    n_tot = int(feats.get("n", 0))
    bi = int(bar_index) if bar_index is not None else max(0, n_tot - 1)
    n = bi + 1
    if n_tot <= 0 or n <= 0:
        return 0.0, 0.0, {
            "alpha_raw": 0.0,
            "alpha_rank": 0.5,
            "score": 0.0,
            "vol_breakout_interaction_factor": 1.0,
            "total_long": 0.0,
            "total_short": 0.0,
        }
    if bi >= n_tot:
        bi = n_tot - 1
        n = bi + 1

    cap = float(config.get("ttm_v2_alpha_score_cap", 5.0))
    w1 = float(config.get("ttm_v2_w_breakout", 1.0 / 3.0))
    w2 = float(config.get("ttm_v2_w_momentum", 1.0 / 3.0))
    w3 = float(config.get("ttm_v2_w_basis", 1.0 / 3.0))
    wsum = w1 + w2 + w3
    if wsum > 1e-12:
        w1, w2, w3 = w1 / wsum, w2 / wsum, w3 / wsum

    fwd_h = max(1, int(config.get("ttm_v2_sign_corr_forward_bars", 4)))
    rank_win = max(8, int(config.get("ttm_v2_rank_window", 120)))
    min_n_corr = int(config.get("ttm_v2_sign_corr_min_n", 25))
    sign_guard = bool(config.get("ttm_v2_sign_flip_guard_enabled", True))
    flip_max = float(config.get("ttm_v2_sign_flip_max", 0.35))

    close = _v2_feat_slice(feats, "close", n)
    brk = _v2_feat_slice(feats, "breakout_strength_base", n)
    if float(np.max(np.abs(brk))) < 1e-15 and "breakout_strength" in feats:
        brk = _v2_feat_slice(feats, "breakout_strength", n)
    bu = _v2_bool_slice(feats, "breakout_up", n)
    use_eff = "effective_strength" in feats and isinstance(
        feats.get("effective_strength"), np.ndarray
    )
    eff_slice = _v2_feat_slice(feats, "effective_strength", n) if use_eff else None
    brk_for_long = eff_slice if eff_slice is not None else brk
    long_breakout = np.where(bu, brk_for_long, 0.0)
    short_breakout = _v2_feat_slice(feats, "short_score", n)
    short_confirm = _v2_bool_slice(feats, "exhaustion_confirm", n)
    short_breakout = np.where(short_confirm, short_breakout, 0.0)

    k_vol_brk = float(config.get("ttm_v2_vol_breakout_interaction_k", 0.0))
    v_int_arr = np.ones(n, dtype=np.float64)
    if abs(k_vol_brk) > 1e-15:
        vol_z_arr = _v2_feat_slice(feats, "vol_zscore", n)
        v_int_arr = 1.0 + k_vol_brk * np.tanh(vol_z_arr)
        long_breakout = long_breakout * v_int_arr
        short_breakout = short_breakout * v_int_arr

    mom = _v2_feat_slice(feats, "momentum_1_z", n)
    basis_mom = _v2_feat_slice(feats, "basis_signal", n)
    if float(np.max(np.abs(basis_mom))) < 1e-12 and "basis_effect" in feats:
        basis_mom = _v2_feat_slice(feats, "basis_effect", n)

    short_mom = -mom
    short_basis_mom = -basis_mom
    fwd = _forward_returns_close(close, fwd_h)
    neg_fwd = -fwd

    c1_long = _pearson_corr(long_breakout, fwd, min_n_corr)
    c2_long = _pearson_corr(mom, fwd, min_n_corr)
    c3_long = _pearson_corr(basis_mom, fwd, min_n_corr)
    c1_short = _pearson_corr(short_breakout, neg_fwd, min_n_corr)
    c2_short = _pearson_corr(short_mom, neg_fwd, min_n_corr)
    c3_short = _pearson_corr(short_basis_mom, neg_fwd, min_n_corr)

    s1_long = _alignment_factor(c1_long, guard_enabled=sign_guard, flip_max=flip_max)
    s2_long = _alignment_factor(c2_long, guard_enabled=sign_guard, flip_max=flip_max)
    s3_long = _alignment_factor(c3_long, guard_enabled=sign_guard, flip_max=flip_max)
    s1_short = _alignment_factor(c1_short, guard_enabled=sign_guard, flip_max=flip_max)
    s2_short = _alignment_factor(c2_short, guard_enabled=sign_guard, flip_max=flip_max)
    s3_short = _alignment_factor(c3_short, guard_enabled=sign_guard, flip_max=flip_max)

    alpha_series_long = w1 * (long_breakout * s1_long) + w2 * (mom * s2_long) + w3 * (basis_mom * s3_long)
    alpha_series_short = (
        w1 * (short_breakout * s1_short)
        + w2 * (short_mom * s2_short)
        + w3 * (short_basis_mom * s3_short)
    )
    alpha_raw_long = float(alpha_series_long[bi]) if np.isfinite(alpha_series_long[bi]) else 0.0
    alpha_raw_short = float(alpha_series_short[bi]) if np.isfinite(alpha_series_short[bi]) else 0.0

    lo = max(0, n - rank_win)
    alpha_window_long = alpha_series_long[lo : bi + 1]
    alpha_window_short = alpha_series_short[lo : bi + 1]
    alpha_rank_long = _percentile_rank_in_window(alpha_window_long, alpha_raw_long)
    alpha_rank_short = _percentile_rank_in_window(alpha_window_short, alpha_raw_short)
    use_rank_score = bool(config.get("ttm_v2_use_rank_score", False))
    if use_rank_score:
        score_long = float(np.clip((2.0 * alpha_rank_long - 1.0) * cap, -cap, cap))
        score_short = float(np.clip((2.0 * alpha_rank_short - 1.0) * cap, -cap, cap))
    else:
        score_long = float(np.clip(np.tanh(alpha_raw_long) * cap, -cap, cap))
        score_short = float(np.clip(np.tanh(alpha_raw_short) * cap, -cap, cap))

    v_int_last = float(v_int_arr[bi]) if v_int_arr.size > bi else 1.0
    components: Dict[str, float] = {
        "alpha_raw": alpha_raw_long,
        "alpha_rank": float(alpha_rank_long),
        "alpha_raw_long": alpha_raw_long,
        "alpha_raw_short": alpha_raw_short,
        "alpha_rank_long": float(alpha_rank_long),
        "alpha_rank_short": float(alpha_rank_short),
        "score": score_long,
        "score_long": score_long,
        "score_short": score_short,
        "vol_breakout_interaction_factor": v_int_last,
        "sign_breakout_corr": s1_long,
        "sign_momentum_corr": s2_long,
        "sign_basis_corr": s3_long,
        "sign_breakout_corr_long": s1_long,
        "sign_momentum_corr_long": s2_long,
        "sign_basis_corr_long": s3_long,
        "sign_breakout_corr_short": s1_short,
        "sign_momentum_corr_short": s2_short,
        "sign_basis_corr_short": s3_short,
        "corr_breakout": c1_long,
        "corr_momentum": c2_long,
        "corr_basis": c3_long,
        "corr_breakout_long": c1_long,
        "corr_momentum_long": c2_long,
        "corr_basis_long": c3_long,
        "corr_breakout_short": c1_short,
        "corr_momentum_short": c2_short,
        "corr_basis_short": c3_short,
        "w_breakout": w1,
        "w_momentum": w2,
        "w_basis": w3,
        "breakout_signed_last": float(long_breakout[bi] - short_breakout[bi]),
        "breakout_long_last": float(long_breakout[bi]),
        "effective_strength_last": float(eff_slice[bi]) if eff_slice is not None else float(brk[bi]),
        "breakout_short_last": float(short_breakout[bi]),
        "price_momentum_last": float(mom[bi]),
        "basis_mom_last": float(basis_mom[bi]),
        "total_long": score_long,
        "total_short": score_short,
        # Legacy key for follow_design / diagnostics: sign carries side preference.
        "momentum": score_long - score_short,
    }
    if bool(config.get("ttm_strict_asserts", False)):
        if not (np.isfinite(score_long) and np.isfinite(score_short)):
            raise ValueError("non-finite score from compute_score_v2_alpha")
    return score_long, score_short, components


def _regime_row(config: Mapping[str, Any], vol_regime: int) -> Dict[str, float]:
    rw = config.get("ttm_regime_weights") or {}
    row = rw.get(vol_regime)
    if row is None:
        row = rw.get(0, {"m": 1.0, "b": 1.0, "o": 1.0})
    return {k: float(v) for k, v in row.items()}


def compute_score(
    features: TTMFeatureVector,
    config: Mapping[str, Any],
) -> Tuple[float, float, Dict[str, float]]:
    """
    Bounded composition only (tanh / clip). Returns ``score_short = -score_long`` (symmetric).

    Inputs must be finite; use :class:`TTMFeatureVector` + ``valid_mask`` for missing data.
    """
    if bool(config.get("ttm_strict_feature_finite", False)):
        for name, val in (
            ("breakout_strength", features.breakout_strength),
            ("failure_strength", features.failure_strength),
            ("basis_norm", features.basis_norm),
            ("basis_delta", features.basis_delta),
            ("oi_signal", features.oi_signal),
        ):
            if not np.isfinite(val):
                raise ValueError(f"ttm_strict_feature_finite: non-finite {name}={val!r}")

    k1 = float(config.get("ttm_score_k1", 1.0))
    k2 = float(config.get("ttm_score_k2", 1.0))
    k3 = float(config.get("ttm_score_k3", 1.0))
    k4 = float(config.get("ttm_score_k4", 1.0))
    k5 = float(config.get("ttm_score_k5", 1.0))
    cap = float(config.get("ttm_score_cap", 5.0))
    regime_id = int(config.get("_ttm_regime_id", features.vol_regime))
    rw_obj = config.get("_ttm_regime_weight_set")
    if isinstance(rw_obj, RegimeWeights):
        ws = rw_obj.get(regime_id)
        w_m, w_b, w_o = ws.momentum, ws.basis, ws.oi
    else:
        w_m = float(config.get("ttm_w_momentum", 1.0))
        w_b = float(config.get("ttm_w_basis", 1.0))
        w_o = float(config.get("ttm_w_oi", 0.8))

    vm = features.valid_mask
    bs = features.breakout_strength if vm.get("breakout_strength", True) else 0.0
    fs = features.failure_strength if vm.get("failure_strength", True) else 0.0
    bn = features.basis_norm if vm.get("basis_norm", True) else 0.0
    bd = features.basis_delta if vm.get("basis_delta", True) else 0.0
    oi = features.oi_signal if vm.get("oi_signal", True) else 0.0

    momentum = float(np.tanh(k1 * bs - k2 * fs))
    basis_effect = float(np.tanh(k3 * bn + k4 * bd))
    oi_effect = float(np.clip(oi * k5, -1.0, 1.0))

    rr = _regime_row(config, regime_id)
    rm, rb, ro = rr["m"], rr["b"], rr["o"]

    raw = w_m * momentum * rm + w_b * basis_effect * rb + w_o * oi_effect * ro
    raw = float(np.clip(raw, -cap, cap))

    score_long = raw
    score_short = -raw

    components: Dict[str, float] = {
        "momentum": momentum,
        "basis_effect": basis_effect,
        "oi_effect": oi_effect,
        "w_momentum": w_m,
        "w_basis": w_b,
        "w_oi": w_o,
        "regime_id": float(regime_id),
        "regime_m": rm,
        "regime_b": rb,
        "regime_o": ro,
        "raw": raw,
        "total_long": score_long,
        "total_short": score_short,
    }

    if bool(config.get("ttm_strict_asserts", False)):
        if not (np.isfinite(score_long) and np.isfinite(score_short)):
            raise ValueError("non-finite score from compute_score")
    return score_long, score_short, components


def softmax_two(z_long: float, z_short: float, temperature: float = 1.0) -> Tuple[float, float]:
    """Pairwise softmax over (long, short) logits; probabilities sum to 1."""
    t = max(1e-12, float(temperature))
    a = float(z_long) / t
    b = float(z_short) / t
    m = max(a, b, 0.0)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    s = ea + eb
    if s <= 0.0 or not math.isfinite(s):
        return 0.5, 0.5
    return float(ea / s), float(eb / s)


def scores_to_probs(
    score_long: float,
    score_short: float,
    config: Mapping[str, Any],
) -> Tuple[float, float]:
    """Map scores to (prob_long, prob_short); optional extra tanh on logits before softmax."""
    if not np.isfinite(score_long) or not np.isfinite(score_short):
        return float("nan"), float("nan")

    use_tanh = bool(config.get("score_use_tanh_logits", False))
    if use_tanh:
        zl = float(np.tanh(score_long))
        zs = float(np.tanh(score_short))
    else:
        zl, zs = float(score_long), float(score_short)

    temp = float(config.get("softmax_temperature", 1.0))
    pl, ps = softmax_two(zl, zs, temp)

    if bool(config.get("ttm_strict_asserts", False)):
        if not np.isfinite(pl) or not np.isfinite(ps):
            raise ValueError("non-finite prob")
        if abs(pl + ps - 1.0) > 1e-5:
            raise ValueError(f"prob sum != 1: {pl + ps}")
    return pl, ps


def probs_valid(prob_long: float, prob_short: float) -> bool:
    return bool(np.isfinite(prob_long) and np.isfinite(prob_short))
