"""
TTM V2 LONG effective_strength_v3 + crowd phase (refactor 2).

Gated by ``ttm_v2_use_effective_strength_v3`` in :func:`~src.strategies.ttm.ttm_score.compute_score_v2_alpha`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


def _bars_since_last_breakout_up(feats: Mapping[str, Any], bi: int) -> int:
    bu = feats.get("breakout_up")
    if not isinstance(bu, np.ndarray) or bi < 0:
        return 999
    for j in range(int(bi), -1, -1):
        if j < bu.size and bool(bu[j]):
            return int(bi) - j
    return 999


def _persisted_last_breakout_up(last: Mapping[str, Any], bi: int, max_age: int) -> Optional[int]:
    raw = last.get("ttm_v2_persist_last_upside_breakout_bar_index")
    if raw is None:
        return None
    try:
        ix = int(raw)
    except (TypeError, ValueError):
        return None
    if ix < 0 or ix > int(bi):
        return None
    if int(bi) - ix > max(1, int(max_age)):
        return None
    return ix


def compute_effective_strength_v3(
    feats: Mapping[str, Any],
    last: Mapping[str, Any],
    bar_index: int,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    LONG opportunity proxy: weighted conviction/continuation minus extension/FOMO penalties.

    ``valid_breakout`` follows tradable continuation breakout (``breakout_up_filtered_last`` or ``breakout_up``).
    """
    bi = int(bar_index)
    valid_breakout = bool(last.get("breakout_up_filtered_last", last.get("breakout_up")))
    max_ctx = max(1, int(config.get("ttm_v2_bars_since_breakout_max_for_short", 20)))
    persisted_ix = _persisted_last_breakout_up(last, bi, max_ctx)
    if persisted_ix is not None:
        bsb = int(bi) - int(persisted_ix)
    else:
        bsb = _bars_since_last_breakout_up(feats, bi)

    fd_arr = feats.get("failure_depth_up")
    fd_up = float(fd_arr[bi]) if isinstance(fd_arr, np.ndarray) and fd_arr.size > bi else 0.0
    if not np.isfinite(fd_up):
        fd_up = 0.0

    empty: Dict[str, Any] = {
        "effective_strength_raw": 0.0,
        "effective_strength": 0.0,
        "score_long": 0.0,
        "breakout_conviction": 0.0,
        "continuation_confirm": 0.0,
        "basis_confirm": 0.0,
        "extension": 0.0,
        "extension_sq": 0.0,
        "last_bar_return_raw": 0.0,
        "last_bar_return_norm": 0.0,
        "positive_last_bar_return": 0.0,
        "late_phase_penalty": 0.0,
        "crowd_phase": "no_breakout",
        "phase_reason": "no_valid_breakout",
        "late_fomo_flag": False,
        "entry_block_reason": None,
    }

    if not valid_breakout:
        fail_min = float(config.get("ttm_v2_failed_breakout_failure_min", 0.35))
        failed_recent = (
            1 <= int(bsb) <= max_ctx
            and (
                fd_up > 0.03
                or bool(last.get("failure_up"))
                or float(last.get("failure_strength") or 0.0) >= fail_min
                or float(last.get("last_bar_return") or 0.0) < 0.0
            )
        )
        if failed_recent:
            empty["crowd_phase"] = "failed_breakout"
            empty["phase_reason"] = "recent_upside_breakout_lost_validity"
            empty["entry_block_reason"] = "failed_breakout"
        else:
            empty["entry_block_reason"] = "no_breakout"
        return empty

    rs = float(last.get("raw_strength") or 0.0)
    if not np.isfinite(rs):
        rs = 0.0
    breakout_conviction = float(np.tanh(rs / 2.5))

    mom_z = float(last.get("momentum_1_z") or 0.0)
    if not np.isfinite(mom_z):
        mom_z = 0.0
    continuation_confirm = float(np.tanh(mom_z / 2.0)) * float(np.clip(1.0 - fd_up * 25.0, 0.0, 1.0))
    continuation_confirm = float(np.clip(continuation_confirm, -0.25, 1.0))

    bn = float(last.get("basis_norm") or 0.0)
    if not np.isfinite(bn):
        bn = 0.0
    basis_confirm = float(np.tanh(bn))

    ext_line = float(last.get("extension") or last.get("price_z") or 0.0)
    if not np.isfinite(ext_line):
        ext_line = 0.0
    if bool(config.get("ttm_v2_effective_strength_use_abs_price_z", True)):
        extension = abs(ext_line)
    else:
        extension = max(0.0, ext_line)
    extension_sq = float(extension * extension)

    lb_raw = last.get("last_bar_return")
    if lb_raw is None or not np.isfinite(float(lb_raw)):
        lb_raw_f = 0.0
    else:
        lb_raw_f = float(lb_raw)
    last_bar_return_raw = lb_raw_f
    last_bar_return_norm = float(np.tanh(lb_raw_f * 6.0))
    positive_last_bar_return = max(last_bar_return_norm, 0.0)

    bs_ref = max(1.0, float(config.get("ttm_v2_bars_since_breakout_max_for_short", 20)))
    bars_norm = min(1.0, float(bsb) / bs_ref)
    ext_norm = float(np.tanh(extension / 3.0))
    pos_lb_pen = float(np.tanh(max(0.0, lb_raw_f) * 10.0))
    exhaust = 1.0 if bool(last.get("exhaustion_candidate")) else 0.0
    failed_ft = 1.0 if fd_up > 0.03 else 0.0
    late_phase_penalty = float(
        np.clip(
            0.22 * bars_norm
            + 0.22 * ext_norm
            + 0.22 * pos_lb_pen
            + 0.22 * exhaust
            + 0.22 * failed_ft,
            0.0,
            1.2,
        )
    )

    w_bc = float(config.get("ttm_v2_w_breakout_conviction", 1.0))
    w_cc = float(config.get("ttm_v2_w_continuation_confirm", 0.5))
    w_basis = float(config.get("ttm_v2_w_basis_confirm", 0.2))
    w_ext = float(config.get("ttm_v2_w_extension", 0.5))
    w_ext2 = float(config.get("ttm_v2_w_extension_sq", 0.2))
    w_plr = float(config.get("ttm_v2_w_positive_last_bar_return", 0.5))
    w_late = float(config.get("ttm_v2_w_late_phase_penalty", 0.7))

    eff_raw = float(
        w_bc * breakout_conviction
        + w_cc * continuation_confirm
        + w_basis * basis_confirm
        - w_ext * extension
        - w_ext2 * extension_sq
        - w_plr * positive_last_bar_return
        - w_late * late_phase_penalty
    )
    effective_strength = eff_raw
    score_long = float(np.tanh(eff_raw))

    ext_thr = config.get("ttm_v2_extension_late_threshold")
    lb_thr = config.get("ttm_v2_last_bar_return_spike_threshold")
    late_fomo_thr = float(config.get("ttm_v2_late_fomo_threshold", 0.7))
    late_fomo_flag = False
    if ext_thr is not None and float(extension) > float(ext_thr):
        late_fomo_flag = True
    if lb_thr is not None and positive_last_bar_return > float(lb_thr):
        late_fomo_flag = True
    if bool(last.get("exhaustion_candidate")):
        late_fomo_flag = True
    if late_phase_penalty > late_fomo_thr:
        late_fomo_flag = True

    crowd_phase = "no_breakout"
    phase_reason = "no_valid_breakout"
    if bool(last.get("exhaustion_confirm")) or bool(last.get("exhaustion_candidate")):
        crowd_phase = "exhaustion"
        phase_reason = "exhaustion_flag"
    elif late_fomo_flag:
        crowd_phase = "late_fomo"
        phase_reason = "late_fomo_penalty_or_spike"
    elif fd_up > 0.08:
        crowd_phase = "failed_breakout"
        phase_reason = "failure_depth_up"
    elif bsb == 0:
        crowd_phase = "ignition"
        phase_reason = "fresh_valid_breakout"
    elif continuation_confirm >= float(config.get("ttm_v2_early_continuation_threshold", 0.2)) and extension < 3.0:
        crowd_phase = "early_continuation"
        phase_reason = "valid_breakout_continuation_confirmed"
    else:
        crowd_phase = "late_fomo"
        phase_reason = "valid_breakout_but_continuation_weak_or_extended"

    entry_block_reason: Optional[str] = None
    if crowd_phase in ("no_breakout", "late_fomo", "exhaustion", "failed_breakout"):
        entry_block_reason = crowd_phase

    return {
        "effective_strength_raw": eff_raw,
        "effective_strength": effective_strength,
        "score_long": score_long,
        "breakout_conviction": breakout_conviction,
        "continuation_confirm": continuation_confirm,
        "basis_confirm": basis_confirm,
        "extension": extension,
        "extension_sq": extension_sq,
        "last_bar_return_raw": last_bar_return_raw,
        "last_bar_return_norm": last_bar_return_norm,
        "positive_last_bar_return": positive_last_bar_return,
        "late_phase_penalty": late_phase_penalty,
        "crowd_phase": crowd_phase,
        "phase_reason": phase_reason,
        "late_fomo_flag": late_fomo_flag,
        "entry_block_reason": entry_block_reason,
    }
