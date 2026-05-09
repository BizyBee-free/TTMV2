"""Continuation-aware V2 exit ordering (refactor_4); no future data — bar-close only."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from src.strategies.ttm.ttm_v2_short import check_exhaustion_short_exit, check_short_v3_structural_exit

LONG_EXIT_REASONS = [
    "hard_stop",
    "failed_breakout",
    "exhaustion_rejection",
    "continuation_decay",
    "score_decay",
    "time_stop",
    "take_profit",
    "soft_min_hold_continuation",
    "unknown",
]

SHORT_EXIT_REASONS = [
    "short_hard_stop",
    "long_continuation_recovered",
    "short_failed_price_reclaim",
    "short_downside_decay",
    "short_chase_risk_after_entry",
    "short_time_stop",
    "short_take_profit",
    "unknown",
]


def _fin(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def continuation_still_valid_long(
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    entry_snapshot: Optional[Mapping[str, Any]],
    score_long: float,
) -> bool:
    """True when soft-min-hold may defer a score/prob-based exit."""
    thr = float(cfg.get("ttm_v2_continuation_exit_threshold", 0.0))
    if "continuation_confirm" in components:
        cc = _fin(components.get("continuation_confirm"))
        if cc < thr:
            return False
    if bool(last.get("exhaustion_confirm")):
        return False
    fail_min = float(cfg.get("ttm_v2_failed_breakout_failure_min", 0.35))
    if not bool(last.get("breakout_up")) and _fin(last.get("failure_strength")) >= fail_min:
        return False
    phase = str(components.get("crowd_phase") or "")
    cc = _fin(components.get("continuation_confirm")) if "continuation_confirm" in components else thr
    if phase in ("late_exhaustion", "exhaustion", "distribution") and cc < thr + 0.15:
        return False
    ent_sl = entry_snapshot.get("entry_score_long") if entry_snapshot else None
    if ent_sl is not None:
        try:
            if float(score_long) < float(ent_sl) - 2.0:
                return False
        except (TypeError, ValueError):
            pass
    return True


def decide_long_exit_continuation(
    *,
    ok: bool,
    pl: float,
    prob_exit: float,
    score_long: float,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    holding_bars: Optional[int],
    unrealized_return: Optional[float],
    entry_snapshot: Optional[Mapping[str, Any]],
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (action, canonical_exit_or_hold_reason, debug_extra).
    action in ("EXIT", "HOLD").
    For EXIT, second value is a LONG_EXIT_REASONS label; for soft-min HOLD use soft_min_hold_continuation.
    """
    extra: Dict[str, Any] = {"v2_exit_path": "continuation_long"}
    sl_ret = float(cfg.get("ttm_v2_sl_return", -0.0007))
    tp_ret = float(cfg.get("ttm_v2_tp_return", 0.0015))
    max_hold_cfg = cfg.get("ttm_v2_max_hold_bars")
    max_hold = int(max_hold_cfg) if max_hold_cfg is not None else int(cfg.get("ttm_v2_time_stop_bars", 4))

    if unrealized_return is not None and np.isfinite(unrealized_return) and float(unrealized_return) <= sl_ret:
        return "EXIT", "hard_stop", {**extra, "v2_exit_reason": "hard_stop"}

    fail_min = float(cfg.get("ttm_v2_failed_breakout_failure_min", 0.35))
    if not bool(last.get("breakout_up")) and _fin(last.get("failure_strength")) >= fail_min:
        return "EXIT", "failed_breakout", {**extra, "v2_exit_reason": "failed_breakout"}

    rej_thr = float(cfg.get("ttm_v2_exhaustion_rejection_threshold", 0.25))
    if bool(last.get("exhaustion_confirm")) and _fin(components.get("rejection_confirm")) >= rej_thr:
        return "EXIT", "exhaustion_rejection", {**extra, "v2_exit_reason": "exhaustion_rejection"}

    thr_cc = float(cfg.get("ttm_v2_continuation_exit_threshold", 0.0))
    if "continuation_confirm" in components and _fin(components.get("continuation_confirm")) < thr_cc:
        return "EXIT", "continuation_decay", {**extra, "v2_exit_reason": "continuation_decay"}

    decay_prob = bool(ok and pl < prob_exit)
    rel = float(cfg.get("ttm_v2_score_decay_relative", 0.65))
    decay_score = False
    if entry_snapshot and entry_snapshot.get("entry_score_long") is not None:
        try:
            decay_score = float(score_long) < float(entry_snapshot["entry_score_long"]) * rel
        except (TypeError, ValueError):
            decay_score = False
    would_decay = decay_prob or decay_score

    if would_decay:
        if bool(cfg.get("ttm_v2_enable_soft_min_hold")) and holding_bars is not None:
            smin = max(0, int(cfg.get("ttm_v2_soft_min_hold_bars", 2)))
            if int(holding_bars) < smin and continuation_still_valid_long(
                last,
                components,
                cfg,
                entry_snapshot=entry_snapshot,
                score_long=score_long,
            ):
                return "HOLD", "soft_min_hold_continuation", {
                    **extra,
                    "v2_hold_reason": "soft_min_hold_continuation",
                    "target_alpha_hold_bars": int(cfg.get("ttm_v2_target_alpha_hold_bars", 3)),
                }
        return "EXIT", "score_decay", {**extra, "v2_exit_reason": "score_decay"}

    if (
        holding_bars is not None
        and max_hold > 0
        and int(holding_bars) >= max_hold
    ):
        return "EXIT", "time_stop", {**extra, "v2_exit_reason": "time_stop"}

    if unrealized_return is not None and np.isfinite(unrealized_return) and float(unrealized_return) >= tp_ret:
        return "EXIT", "take_profit", {**extra, "v2_exit_reason": "take_profit"}

    return "HOLD", "", {**extra, "v2_hold_reason": "no_exit_signal"}


def decide_short_exit_continuation(
    *,
    ok: bool,
    ps: float,
    prob_exit: float,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    holding_bars: Optional[int],
    unrealized_return: Optional[float],
    position_meta: Optional[Mapping[str, Any]],
    current_price: float,
) -> Tuple[str, str, Dict[str, Any]]:
    extra: Dict[str, Any] = {"v2_exit_path": "continuation_short"}
    sl_ret = float(cfg.get("ttm_v2_sl_return_short", cfg.get("ttm_v2_sl_return", -0.0007)))
    tp_ret = float(cfg.get("ttm_v2_tp_return_short", cfg.get("ttm_v2_tp_return", 0.0015)))
    max_hold_cfg = cfg.get("ttm_v2_short_max_hold_bars")
    max_hold = int(max_hold_cfg) if max_hold_cfg is not None else int(cfg.get("ttm_v2_time_stop_bars_short", cfg.get("ttm_v2_time_stop_bars", 4)))

    close_px = float(current_price) if current_price and np.isfinite(current_price) else _fin(last.get("close"))

    ex = check_exhaustion_short_exit(position_meta, close_px, int(holding_bars or 0))
    if ex is not None:
        tr = str(ex.get("trigger", "")).upper()
        if tr == "SL":
            return "EXIT", "short_hard_stop", {**extra, "v2_exit_reason": "short_hard_stop", "struct": ex}
        if tr == "TP":
            return "EXIT", "short_take_profit", {**extra, "v2_exit_reason": "short_take_profit", "struct": ex}
        if tr == "TIME":
            return "EXIT", "short_time_stop", {**extra, "v2_exit_reason": "short_time_stop", "struct": ex}

    if unrealized_return is not None and np.isfinite(unrealized_return) and float(unrealized_return) <= sl_ret:
        return "EXIT", "short_hard_stop", {**extra, "v2_exit_reason": "short_hard_stop"}

    v3 = check_short_v3_structural_exit(
        position_meta,
        close_px,
        int(holding_bars or 0),
        last,
        cfg,
    )
    if v3 is not None:
        trig = str(v3.get("trigger", "")).upper()
        if trig == "RECLAIM":
            return "EXIT", "short_failed_price_reclaim", {**extra, "v2_exit_reason": "short_failed_price_reclaim", "struct": v3}
        if trig == "RECOVER":
            return "EXIT", "long_continuation_recovered", {**extra, "v2_exit_reason": "long_continuation_recovered", "struct": v3}
        if trig == "DECAY":
            return "EXIT", "short_downside_decay", {**extra, "v2_exit_reason": "short_downside_decay", "struct": v3}

    chase_thr = float(cfg.get("ttm_v2_short_chase_risk_threshold", 0.7))
    if _fin(components.get("short_chase_risk")) >= chase_thr and _fin(components.get("early_continuation_still_alive")) < 0.2:
        return "EXIT", "short_chase_risk_after_entry", {**extra, "v2_exit_reason": "short_chase_risk_after_entry"}

    if ok and ps < prob_exit:
        return "EXIT", "short_downside_decay", {**extra, "v2_exit_reason": "short_downside_decay"}

    if holding_bars is not None and max_hold > 0 and int(holding_bars) >= max_hold:
        return "EXIT", "short_time_stop", {**extra, "v2_exit_reason": "short_time_stop"}

    if unrealized_return is not None and np.isfinite(unrealized_return) and float(unrealized_return) >= tp_ret:
        return "EXIT", "short_take_profit", {**extra, "v2_exit_reason": "short_take_profit"}

    return "HOLD", "", {**extra, "v2_hold_reason": "no_exit_signal"}


def map_parallel_runner_reason_to_canonical(side: str, reason: str, exit_channel: str) -> str:
    """Map legacy runner/strategy reason strings to refactor_4 labels for trade JSONL."""
    r = str(reason or "").lower()
    ch = str(exit_channel or "").lower()
    su = str(side).upper()
    if su == "LONG":
        if "stop_loss" in r or ch == "stop_loss_return":
            return "hard_stop"
        if "take_profit" in r or ch == "take_profit_return":
            return "take_profit"
        if "max_bars" in r or ch == "max_bars_in_trade":
            return "time_stop"
        if "prob_decay" in r:
            return "score_decay"
        if r in LONG_EXIT_REASONS:
            return r
        return "unknown"
    if su == "SHORT":
        if "reclaim" in r:
            return "short_failed_price_reclaim"
        if "recovered" in r or "long_continuation" in r:
            return "long_continuation_recovered"
        if "momentum_decay" in r or "downside_decay" in r:
            return "short_downside_decay"
        if "stop_loss" in r and "short" in r:
            return "short_hard_stop"
        if "take_profit" in r and "short" in r:
            return "short_take_profit"
        if "max_bars" in r and "short" in r:
            return "short_time_stop"
        if "prob_decay" in r:
            return "short_downside_decay"
        if "short_exhaustion_sl" in ch or ch.endswith("_sl"):
            return "short_hard_stop"
        if "short_exhaustion_tp" in ch or ch.endswith("_tp"):
            return "short_take_profit"
        if "short_exhaustion_time" in ch or ch.endswith("_time"):
            return "short_time_stop"
        return "unknown"
    return "unknown"
