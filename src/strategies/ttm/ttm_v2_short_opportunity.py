"""
TTM V2 SHORT crowd-long unwind (refactor 3).

Gated by ``ttm_v2_use_short_effective_strength_v3`` in :func:`~src.strategies.ttm.ttm_score.compute_score_v2_alpha`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.strategies.ttm.ttm_v2_phases import classify_crowd_phase


def _fin(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def _last_prior_breakout_bar(feats: Mapping[str, Any], bi: int, lookback: int) -> Optional[int]:
    """Last bar index <= bi with tradable or raw upside breakout in lookback window."""
    start = max(0, int(bi) - max(1, int(lookback)) + 1)
    bu_f = feats.get("breakout_up")
    bu_r = feats.get("breakout_up_raw")
    for j in range(int(bi), start - 1, -1):
        if isinstance(bu_f, np.ndarray) and j < bu_f.size and bool(bu_f[j]):
            return int(j)
        if isinstance(bu_r, np.ndarray) and j < bu_r.size and bool(bu_r[j]):
            return int(j)
    return None


def compute_short_opportunity_v3(
    feats: Mapping[str, Any],
    last: Mapping[str, Any],
    bar_index: int,
    config: Mapping[str, Any],
    long_components: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Conditional SHORT alpha: unwind after prior upside / crowded-long context weakens.

    ``long_components`` should contain LONG v3 diagnostics when enabled (``crowd_phase``, etc.).
    """
    bi = int(bar_index)
    lookback = max(5, int(config.get("ttm_v2_short_prior_breakout_lookback", 40)))
    last_j = _last_prior_breakout_bar(feats, bi, lookback)
    prior_upside_breakout_exists = last_j is not None
    last_upside_breakout_bar_index = int(last_j) if last_j is not None else None
    bars_since_upside_breakout = int(bi - last_j) if last_j is not None else None

    crowd_phase = str(long_components.get("crowd_phase") or classify_crowd_phase(last, config))
    cp = crowd_phase

    fd_arr = feats.get("failure_depth_up")
    fd_up = _fin(fd_arr[bi] if isinstance(fd_arr, np.ndarray) and fd_arr.size > bi else 0.0)
    mom_z = _fin(last.get("momentum_1_z"))
    cont_confirm = _fin(long_components.get("continuation_confirm"))
    score_long = _fin(long_components.get("score_long"))
    pos_lb = _fin(long_components.get("positive_last_bar_return"))
    late_pen = _fin(long_components.get("late_phase_penalty"))
    ext = _fin(long_components.get("extension"), _fin(last.get("extension") or last.get("price_z")))
    ext_sq = float(ext * ext)
    lb_raw = _fin(last.get("last_bar_return"))
    rh = _fin(last.get("rolling_high"))
    close_px = _fin(last.get("close"))
    exhaustion_cand = bool(last.get("exhaustion_candidate"))
    bs_min = int(config.get("ttm_v2_bars_since_breakout_min_for_short", 1))
    bs_max = int(config.get("ttm_v2_bars_since_breakout_max_for_short", 20))
    bsb = bars_since_upside_breakout if bars_since_upside_breakout is not None else 999

    bars_ok = bsb >= bs_min and bsb <= bs_max
    crowded_long_pressure = float(
        np.clip(
            0.22 * float(np.tanh(ext / 2.5))
            + 0.22 * float(np.tanh(ext_sq / 8.0))
            + 0.18 * float(np.tanh(pos_lb * 2.0))
            + 0.18 * float(np.tanh(late_pen * 2.0))
            + (0.12 if exhaustion_cand else 0.0)
            + (0.08 if bars_ok else 0.0),
            0.0,
            1.2,
        )
    )

    continuation_decay = float(
        np.clip(
            0.45 * float(np.clip(1.0 - cont_confirm, 0.0, 1.0))
            + 0.30 * float(np.tanh(max(0.0, -mom_z) / 2.0))
            + 0.25 * float(np.tanh(fd_up * 18.0)),
            0.0,
            1.2,
        )
    )

    below_ref = rh > 1e-12 and close_px > 0 and close_px < rh * 0.998
    rejection_confirm = float(
        np.clip(
            (0.55 if below_ref else 0.0)
            + 0.35 * float(np.tanh(max(0.0, -lb_raw) * 12.0))
            + 0.25 * float(np.tanh(fd_up * 15.0)),
            0.0,
            1.2,
        )
    )

    failed_breakout_confirm = float(
        np.clip(
            0.55 * float(np.tanh(fd_up * 20.0))
            + (0.45 if cp == "failed_breakout" else 0.0),
            0.0,
            1.2,
        )
    )

    downside_momentum_confirm = float(
        np.clip(
            0.55 * float(np.tanh(max(0.0, -mom_z) / 2.0))
            + 0.45 * float(np.tanh(max(0.0, -lb_raw) * 10.0)),
            0.0,
            1.2,
        )
    )

    early_continuation_still_alive = float(
        np.clip(
            0.35 * float(np.clip(cont_confirm, 0.0, 1.0))
            + 0.30 * float(np.clip(max(0.0, score_long), 0.0, 1.0))
            + (0.25 if cp in ("ignition", "early_continuation") else 0.0)
            + (0.20 if rh > 1e-12 and close_px >= rh * 0.999 else 0.0),
            0.0,
            1.2,
        )
    )

    chase_thr_cfg = float(config.get("ttm_v2_short_chase_risk_threshold", 0.7))
    short_chase_risk = float(
        np.clip(
            0.45 * float(np.tanh(max(0.0, -lb_raw) * 14.0))
            + 0.35 * float(np.tanh(max(0.0, -_fin(last.get("price_z"))) / 2.0))
            + 0.20 * float(np.tanh(max(0.0, downside_momentum_confirm - 0.85) * 6.0)),
            0.0,
            1.2,
        )
    )
    chase_flag = bool(
        bool(config.get("ttm_v2_enable_short_anti_chase", True)) and short_chase_risk >= chase_thr_cfg
    )

    w_cl = float(config.get("ttm_v2_w_crowded_long_pressure", 1.0))
    w_cd = float(config.get("ttm_v2_w_continuation_decay", 0.8))
    w_rc = float(config.get("ttm_v2_w_rejection_confirm", 0.8))
    w_fb = float(config.get("ttm_v2_w_failed_breakout_confirm", 0.8))
    w_dm = float(config.get("ttm_v2_w_downside_momentum_confirm", 0.5))
    w_ec = float(config.get("ttm_v2_w_early_continuation_still_alive", 1.0))
    w_sc = float(config.get("ttm_v2_w_short_chase_risk", 0.8))

    short_effective_strength_raw = float(
        w_cl * crowded_long_pressure
        + w_cd * continuation_decay
        + w_rc * rejection_confirm
        + w_fb * failed_breakout_confirm
        + w_dm * downside_momentum_confirm
        - w_ec * early_continuation_still_alive
        - w_sc * short_chase_risk
    )
    short_effective_strength = short_effective_strength_raw
    short_score = float(np.tanh(short_effective_strength_raw))

    crowd_ok = cp in ("late_fomo", "exhaustion", "failed_breakout")
    rej_or_fail = rejection_confirm > 0.18 or failed_breakout_confirm > 0.18
    short_candidate = bool(
        prior_upside_breakout_exists
        and crowd_ok
        and crowded_long_pressure > 0.28
        and continuation_decay > 0.22
        and rej_or_fail
        and cp not in ("ignition", "early_continuation", "no_breakout")
    )

    short_phase = "no_short_context"
    short_block_reason: Optional[str] = None
    short_reason: Optional[str] = None

    if not prior_upside_breakout_exists:
        short_phase = "no_short_context"
        short_block_reason = "no_prior_upside_breakout"
    elif cp in ("ignition", "early_continuation"):
        short_phase = "short_invalid"
        short_block_reason = "long_continuation_early"
    elif chase_flag:
        short_phase = "short_chase_risk"
        short_block_reason = "short_chase_risk"
    elif not short_candidate:
        if prior_upside_breakout_exists and crowded_long_pressure > 0.38 and not rej_or_fail:
            short_phase = "crowded_long_watch"
            short_block_reason = "rejection_not_confirmed"
        elif crowded_long_pressure > 0.30 and continuation_decay > 0.18:
            short_phase = "short_setup"
            short_block_reason = "rejection_weak"
        else:
            short_phase = "no_short_context"
            short_block_reason = "insufficient_setup"
    else:
        trig = float(config.get("ttm_v2_short_trigger_threshold", 0.5))
        entry_thr = float(config.get("ttm_v2_short_entry_threshold", 0.0))
        alive_cut = float(config.get("ttm_v2_short_continuation_recovery_threshold", 0.5))
        if early_continuation_still_alive > alive_cut:
            short_phase = "short_invalid"
            short_block_reason = "long_continuation_recovered"
        elif short_score <= max(trig, entry_thr):
            short_phase = "short_setup"
            short_block_reason = "below_trigger_threshold"
        else:
            short_phase = "short_trigger"
            short_reason = "crowd_unwind_short_trigger"
            short_block_reason = None

    return {
        "short_candidate": bool(short_candidate),
        "short_score": float(short_score),
        "short_effective_strength_raw": float(short_effective_strength_raw),
        "short_effective_strength": float(short_effective_strength),
        "short_phase": short_phase,
        "short_reason": short_reason,
        "short_block_reason": short_block_reason,
        "prior_upside_breakout_exists": bool(prior_upside_breakout_exists),
        "last_upside_breakout_bar_index": last_upside_breakout_bar_index,
        "bars_since_upside_breakout": bars_since_upside_breakout,
        "crowded_long_pressure": float(crowded_long_pressure),
        "continuation_decay": float(continuation_decay),
        "rejection_confirm": float(rejection_confirm),
        "failed_breakout_confirm": float(failed_breakout_confirm),
        "downside_momentum_confirm": float(downside_momentum_confirm),
        "early_continuation_still_alive": float(early_continuation_still_alive),
        "short_chase_risk": float(short_chase_risk),
        "_short_chase_flag": chase_flag,
    }
