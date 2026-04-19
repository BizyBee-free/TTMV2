"""TTM entry rules (trap + OI/basis) and signal-driven exit rules."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from src.logger import get_logger
from src.strategies.ttm.ttm_features import (
    compute_ttm_features_from_config,
    features_last_row,
)

logger = get_logger("ttm_signal")


class TTMStateMemory:
    """Stateful breakout tracking for multi-bar trap detection (replay over full series)."""

    def __init__(self) -> None:
        self.last_breakout_up = False
        self.last_breakout_down = False
        self.breakout_up_index: Optional[int] = None
        self.breakout_down_index: Optional[int] = None

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "last_breakout_up": self.last_breakout_up,
            "last_breakout_down": self.last_breakout_down,
            "breakout_up_index": self.breakout_up_index,
            "breakout_down_index": self.breakout_down_index,
        }


def replay_trap_memory_for_last_bar(
    feats: Mapping[str, Any], max_lookback: int
) -> Tuple[bool, bool, TTMStateMemory]:
    """
    Replay bars 0..n-1: expire breakouts older than ``max_lookback``, record new breakouts,
    then trap SHORT when ``failure_up`` with prior breakout still in window; LONG symmetric.
    """
    n = int(feats.get("n", 0))
    mem = TTMStateMemory()
    if n <= 0:
        return False, False, mem

    bu = np.asarray(feats["breakout_up"], dtype=bool)
    bd = np.asarray(feats["breakout_down"], dtype=bool)
    fu = np.asarray(feats["failure_up"], dtype=bool)
    fd = np.asarray(feats["failure_down"], dtype=bool)

    last_short = False
    last_long = False

    for i in range(n):
        if mem.last_breakout_up and mem.breakout_up_index is not None:
            if i - mem.breakout_up_index > max_lookback:
                mem.last_breakout_up = False
                mem.breakout_up_index = None
        if mem.last_breakout_down and mem.breakout_down_index is not None:
            if i - mem.breakout_down_index > max_lookback:
                mem.last_breakout_down = False
                mem.breakout_down_index = None

        if bu[i]:
            mem.last_breakout_up = True
            mem.breakout_up_index = i
        if bd[i]:
            mem.last_breakout_down = True
            mem.breakout_down_index = i

        in_up = (
            mem.last_breakout_up
            and mem.breakout_up_index is not None
            and (i - mem.breakout_up_index) <= max_lookback
        )
        in_dn = (
            mem.last_breakout_down
            and mem.breakout_down_index is not None
            and (i - mem.breakout_down_index) <= max_lookback
        )
        if i == n - 1:
            if in_up and fu[i]:
                last_short = True
            if in_dn and fd[i]:
                last_long = True

    return last_short, last_long, mem


def _norm_confidence_01(raw: float, cap: float = 5.0) -> float:
    if not np.isfinite(raw):
        return 0.0
    return float(max(0.0, min(1.0, float(raw) / cap)))


def _trace_safe(v: Any) -> Any:
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        x = float(v)
        return x if np.isfinite(x) else None
    if isinstance(v, np.bool_):
        return bool(v)
    return repr(v)


def log_decision_trace(data: Mapping[str, Any], *, print_to_stdout: bool = False) -> None:
    """Structured TTM decision trace: optional stdout (live) + logger ``ttm_decision_trace``."""
    lines = ["", "[TTM DECISION TRACE]"]
    safe: Dict[str, Any] = {}
    for k, v in data.items():
        sv = _trace_safe(v)
        safe[str(k)] = sv
        lines.append(f"{k}: {sv}")
    text = "\n".join(lines)
    if print_to_stdout:
        print(text)
    logger.info("TTM decision trace", extra={"strategy": "TTM", "ttm_decision_trace": safe})


def _trap_window_last_bar(
    trap_mem: TTMStateMemory, bar_index: int, trap_lb: int
) -> Tuple[bool, bool]:
    in_up = (
        trap_mem.last_breakout_up
        and trap_mem.breakout_up_index is not None
        and (bar_index - trap_mem.breakout_up_index) <= trap_lb
    )
    in_dn = (
        trap_mem.last_breakout_down
        and trap_mem.breakout_down_index is not None
        and (bar_index - trap_mem.breakout_down_index) <= trap_lb
    )
    return in_up, in_dn


def _classify_no_setup(
    fu: bool,
    fd: bool,
    in_up: bool,
    in_dn: bool,
    trap_mem: TTMStateMemory,
) -> str:
    if not fu and not fd:
        if trap_mem.last_breakout_up or trap_mem.last_breakout_down:
            return "breakout_but_no_failure"
        return "no_breakout"
    if fu and not in_up:
        return "failure_but_no_breakout"
    if fd and not in_dn:
        return "failure_but_no_breakout"
    return "no_trap_setup"


def _emit_signal(
    out: Dict[str, Any],
    trace: Dict[str, Any],
    print_trace: bool,
) -> Dict[str, Any]:
    log_decision_trace(trace, print_to_stdout=print_trace)
    merged = dict(out)
    merged["decision_trace"] = trace
    return merged


def _signal_features_dict(last: Dict[str, Any]) -> Dict[str, float]:
    """Schema: momentum, basis, oi_delta (+ optional diagnostics)."""
    m = last.get("momentum")
    b = last.get("basis")
    oi = last.get("oi_delta")
    out: Dict[str, float] = {
        "momentum": float(m) if m is not None and np.isfinite(m) else 0.0,
        "basis": float(b) if b is not None and np.isfinite(b) else 0.0,
        "oi_delta": float(oi) if oi is not None and np.isfinite(oi) else 0.0,
    }
    mp = last.get("macd_hist_prev")
    if mp is not None and np.isfinite(mp):
        out["momentum_prev"] = float(mp)
    if last.get("squeeze_fail_exit_long") is not None:
        out["squeeze_fail_long"] = 1.0 if bool(last.get("squeeze_fail_exit_long")) else 0.0
        out["squeeze_fail_short"] = 1.0 if bool(last.get("squeeze_fail_exit_short")) else 0.0
    return out


def _evaluate_exit_long(last: Dict[str, Any], config: Mapping[str, Any]) -> Optional[str]:
    """Return exit reason or None. Priority: momentum → squeeze_fail → basis → OI div."""
    mom = last.get("momentum")
    mom_prev = last.get("macd_hist_prev")
    if (
        mom_prev is not None
        and mom is not None
        and np.isfinite(mom_prev)
        and np.isfinite(mom)
        and float(mom_prev) > 0
        and float(mom) < 0
    ):
        return "momentum_reversal"
    if bool(last.get("squeeze_fail_exit_long")):
        return "squeeze_fail"
    sharp = float(config.get("exit_basis_sharp_drop", 0.15))
    bc = last.get("basis_change")
    b = last.get("basis")
    if b is not None and np.isfinite(b) and float(b) < 0:
        return "basis_flip"
    if bc is not None and np.isfinite(bc) and float(bc) < -sharp:
        return "basis_deterioration"
    cp = last.get("close_prev")
    cn = None
    if last.get("close") is not None:
        cn = last.get("close")
    oi_ch = last.get("oi_delta")
    if (
        cp is not None
        and cn is not None
        and oi_ch is not None
        and np.isfinite(cp)
        and np.isfinite(cn)
        and np.isfinite(oi_ch)
        and float(cn) > float(cp)
        and float(oi_ch) < 0
    ):
        return "oi_divergence"
    return None


def _evaluate_exit_short(last: Dict[str, Any], config: Mapping[str, Any]) -> Optional[str]:
    mom = last.get("momentum")
    mom_prev = last.get("macd_hist_prev")
    if (
        mom_prev is not None
        and mom is not None
        and np.isfinite(mom_prev)
        and np.isfinite(mom)
        and float(mom_prev) < 0
        and float(mom) > 0
    ):
        return "momentum_reversal"
    if bool(last.get("squeeze_fail_exit_short")):
        return "squeeze_fail"
    sharp = float(config.get("exit_basis_sharp_drop", 0.15))
    bc = last.get("basis_change")
    b = last.get("basis")
    if b is not None and np.isfinite(b) and float(b) > 0:
        return "basis_flip"
    if bc is not None and np.isfinite(bc) and float(bc) > sharp:
        return "basis_deterioration"
    cp = last.get("close_prev")
    cn = last.get("close")
    oi_ch = last.get("oi_delta")
    if (
        cp is not None
        and cn is not None
        and oi_ch is not None
        and np.isfinite(cp)
        and np.isfinite(cn)
        and np.isfinite(oi_ch)
        and float(cn) < float(cp)
        and float(oi_ch) > 0
    ):
        return "oi_divergence"
    return None


def _merge_last_with_ohlc(last: Dict[str, Any], feats: Dict[str, Any]) -> Dict[str, Any]:
    """Attach last bar close for OI divergence checks."""
    n = int(feats.get("n", 0))
    if n <= 0:
        return last
    close = feats.get("close")
    if isinstance(close, np.ndarray) and close.size >= n:
        v = float(close[n - 1])
        last = dict(last)
        last["close"] = v
    return last


def _oi_confirms(last: Dict[str, Any], config: Mapping[str, Any]) -> bool:
    """OI mở rộng hoặc z-score mức OI bất thường (cao)."""
    oi_inc = float(config.get("oi_increase_threshold", 0))
    oi_z_thr = float(config.get("oi_z_threshold", 0.8))
    oi_ch = last.get("oi_delta")
    oi_z = last.get("oi_z")
    if oi_ch is not None and np.isfinite(oi_ch) and float(oi_ch) > oi_inc:
        return True
    if oi_z is not None and np.isfinite(oi_z) and float(oi_z) > oi_z_thr:
        return True
    return False


def _basis_confirmation(last: Dict[str, Any], config: Mapping[str, Any]) -> bool:
    div = float(config.get("basis_divergence_threshold", 0.2))
    bc = last.get("basis_change")
    if bc is not None and np.isfinite(bc) and abs(float(bc)) > div:
        return True
    if last.get("basis_reversal_short") or last.get("basis_reversal_long"):
        return True
    return False


def _validate_input_series_lengths(data: Mapping[str, Any]) -> Optional[str]:
    """``basis`` / ``open_interest`` must match ``bars`` length when present."""
    bars = data.get("bars")
    if not bars:
        return None
    nb = len(bars)
    b = data.get("basis")
    if b is not None and len(np.asarray(b, dtype=np.float64).ravel()) != nb:
        return "misaligned_data"
    oi = data.get("open_interest")
    if oi is not None and len(np.asarray(oi, dtype=np.float64).ravel()) != nb:
        return "misaligned_data"
    return None


def _soft_entry_weights(config: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "breakout": float(config.get("score_weight_breakout", 1.0)),
        "failure": float(config.get("score_weight_failure", 1.2)),
        "basis": float(config.get("score_weight_basis", 0.8)),
        "oi": float(config.get("score_weight_oi", 0.6)),
    }


def _directional_soft_score(
    *,
    side: str,
    breakout_strength: float,
    failure_strength: float,
    basis_norm: float,
    oi_signal: float,
    config: Mapping[str, Any],
) -> Tuple[float, Dict[str, float]]:
    w = _soft_entry_weights(config)
    side_u = side.upper()
    side_sign = 1.0 if side_u == "SHORT" else -1.0
    basis_dir = float(np.tanh(side_sign * float(basis_norm)))
    oi_dir = float(oi_signal) if side_u == "LONG" else -float(oi_signal)
    brk = max(0.0, float(breakout_strength))
    fail = max(0.0, float(failure_strength))
    comp = {
        "breakout": float(w["breakout"] * brk),
        "failure": float(w["failure"] * fail),
        "basis": float(w["basis"] * basis_dir),
        "oi": float(w["oi"] * oi_dir),
    }
    return float(sum(comp.values())), comp


def _adaptive_threshold(
    score_history: np.ndarray,
    config: Mapping[str, Any],
    fallback: float,
) -> float:
    if score_history.size == 0:
        return float(fallback)
    p = float(config.get("dynamic_threshold_percentile", 75.0))
    p = min(99.0, max(50.0, p))
    win = int(config.get("score_history_window", 120))
    arr = score_history[-win:] if win > 0 else score_history
    arr = arr[np.isfinite(arr)]
    if arr.size < 8:
        return float(fallback)
    return float(np.percentile(arr, p))


def _score_history_for_side(
    feats: Mapping[str, Any], side: str, config: Mapping[str, Any], use_oi: bool
) -> np.ndarray:
    n = int(feats.get("n", 0))
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    brk = np.asarray(feats.get("breakout_strength", np.zeros(n)), dtype=np.float64)
    bnorm = np.asarray(feats.get("basis_norm", np.zeros(n)), dtype=np.float64)
    if side.upper() == "SHORT":
        fail_raw = np.asarray(feats.get("failure_up", np.zeros(n, dtype=bool)), dtype=bool)
    else:
        fail_raw = np.asarray(feats.get("failure_down", np.zeros(n, dtype=bool)), dtype=bool)
    fail = fail_raw.astype(np.float64)
    oi_sig = np.asarray(feats.get("oi_signal", np.zeros(n)), dtype=np.float64)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s, _ = _directional_soft_score(
            side=side,
            breakout_strength=brk[i],
            failure_strength=fail[i],
            basis_norm=bnorm[i],
            oi_signal=oi_sig[i],
            config=config,
        )
        scores[i] = s
    return scores


def generate_ttm_signal_v1(
    features_or_data: Mapping[str, Any] | Dict[str, Any],
    config: Mapping[str, Any],
    *,
    position_side: Optional[str] = None,
    live_mode: bool = False,
) -> Dict[str, Any]:
    """
    TTM V1 — rule-based + adaptive score (unchanged behavior).

    ``position_side``: ``None`` (flat) → entry-only LONG/SHORT/HOLD.
    ``LONG`` / ``SHORT`` → evaluate EXIT first, then HOLD (no new entry).
    ``live_mode``: when True, print full :func:`log_decision_trace` to stdout (and log).
    """
    print_trace = bool(live_mode) or bool(config.get("decision_trace_print", False))

    if "bars" in features_or_data and features_or_data.get("bars") is not None:
        raw = dict(features_or_data)
        if not bool(config.get("ttm_strict_basis_oi", False)):
            bad = _validate_input_series_lengths(raw)
            if bad:
                logger.warning(
                    "TTM: dữ liệu không căn chỉnh (bars vs basis/OI)",
                    extra={"ly_do": bad, "n_bars": len(raw.get("bars") or [])},
                )
                tr_mis = {
                    "stage": "validate",
                    "bar_index": None,
                    "action": "HOLD",
                    "reason": bad,
                    "trap_score": 0.0,
                    "min_score": float(config.get("confirm_trap_min_score", 2.0)),
                }
                return _emit_signal(
                    {
                        "action": "HOLD",
                        "confidence": 0.0,
                        "reason": bad,
                        "strategy": "TTM",
                        "trap_score": 0.0,
                        "features": {"momentum": 0.0, "basis": 0.0, "oi_delta": 0.0},
                        "debug": {"misaligned": True},
                    },
                    tr_mis,
                    print_trace,
                )
        feats = compute_ttm_features_from_config(raw, config)
    else:
        feats = dict(features_or_data)

    n = int(feats.get("n", 0))
    if n <= 0:
        tr_nd = {
            "stage": "features",
            "bar_index": None,
            "action": "HOLD",
            "reason": "no_data",
            "trap_score": 0.0,
            "min_score": float(config.get("confirm_trap_min_score", 2.0)),
        }
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "no_data",
                "strategy": "TTM",
                "features": {"momentum": 0.0, "basis": 0.0, "oi_delta": 0.0},
            },
            tr_nd,
            print_trace,
        )

    last = features_last_row(feats)
    last = _merge_last_with_ohlc(last, feats)
    feat_pack = _signal_features_dict(last)

    side_u = (position_side or "").strip().upper()
    bar_index_exit = n - 1
    if side_u == "LONG":
        r = _evaluate_exit_long(last, config)
        if r is not None:
            out = {
                "action": "EXIT",
                "confidence": 0.85,
                "reason": r,
                "strategy": "TTM",
                "features": feat_pack,
            }
            logger.info(
                "TTM: tín hiệu EXIT (LONG)",
                extra={"strategy": "TTM", "action": "EXIT", "ly_do": r, "features": feat_pack},
            )
            tr_ex = {
                "stage": "exit",
                "bar_index": bar_index_exit,
                "position_side": "LONG",
                "basis": feat_pack.get("basis"),
                "action": "EXIT",
                "reason": r,
            }
            return _emit_signal(out, tr_ex, print_trace)
        tr_hex = {
            "stage": "exit",
            "bar_index": bar_index_exit,
            "position_side": "LONG",
            "basis": feat_pack.get("basis"),
            "action": "HOLD",
            "reason": "no_exit_signal",
        }
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "no_exit_signal",
                "strategy": "TTM",
                "features": feat_pack,
            },
            tr_hex,
            print_trace,
        )

    if side_u == "SHORT":
        r = _evaluate_exit_short(last, config)
        if r is not None:
            out = {
                "action": "EXIT",
                "confidence": 0.85,
                "reason": r,
                "strategy": "TTM",
                "features": feat_pack,
            }
            logger.info(
                "TTM: tín hiệu EXIT (SHORT)",
                extra={"strategy": "TTM", "action": "EXIT", "ly_do": r, "features": feat_pack},
            )
            tr_exs = {
                "stage": "exit",
                "bar_index": bar_index_exit,
                "position_side": "SHORT",
                "basis": feat_pack.get("basis"),
                "action": "EXIT",
                "reason": r,
            }
            return _emit_signal(out, tr_exs, print_trace)
        tr_hxs = {
            "stage": "exit",
            "bar_index": bar_index_exit,
            "position_side": "SHORT",
            "basis": feat_pack.get("basis"),
            "action": "HOLD",
            "reason": "no_exit_signal",
        }
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "no_exit_signal",
                "strategy": "TTM",
                "features": feat_pack,
            },
            tr_hxs,
            print_trace,
        )

    if side_u not in ("", "NONE"):
        tr_inv = {
            "stage": "validate",
            "bar_index": bar_index_exit,
            "position_side": side_u,
            "action": "HOLD",
            "reason": "invalid_position_side",
        }
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "invalid_position_side",
                "strategy": "TTM",
                "features": feat_pack,
            },
            tr_inv,
            print_trace,
        )

    # --- ENTRY (flat): stage 1 multi-bar trap, stage 2 adaptive soft-score ---
    min_entry = float(config.get("entry_score_min", config.get("confirm_trap_min_score", 2.0)))
    trap_lb = int(config.get("trap_lookback_bars", 5))

    bu = bool(last.get("breakout_up"))
    bd = bool(last.get("breakout_down"))
    fu = bool(last.get("failure_up"))
    fd = bool(last.get("failure_down"))
    br_s = bool(last.get("basis_reversal_short"))
    br_l = bool(last.get("basis_reversal_long"))

    oi_ch = last.get("oi_delta")
    oi_z = last.get("oi_z")
    bc = last.get("basis_change")

    setup_short, setup_long, trap_mem = replay_trap_memory_for_last_bar(feats, trap_lb)
    use_oi = bool(config.get("use_open_interest", True))
    oi_signal_last = float(last.get("oi_signal") or 0.0) if use_oi else 0.0
    basis_norm_last = last.get("basis_norm")
    basis_norm_last_f = (
        float(basis_norm_last) if basis_norm_last is not None and np.isfinite(basis_norm_last) else 0.0
    )
    breakout_strength_last = last.get("breakout_strength")
    breakout_strength_last_f = (
        float(breakout_strength_last)
        if breakout_strength_last is not None and np.isfinite(breakout_strength_last)
        else 0.0
    )
    if setup_short and not setup_long:
        score_pts, score_components = _directional_soft_score(
            side="SHORT",
            breakout_strength=breakout_strength_last_f,
            failure_strength=1.0 if fu else 0.0,
            basis_norm=basis_norm_last_f,
            oi_signal=oi_signal_last if use_oi else 0.0,
            config=config,
        )
        score_hist = _score_history_for_side(feats, "SHORT", config, use_oi)
    elif setup_long and not setup_short:
        score_pts, score_components = _directional_soft_score(
            side="LONG",
            breakout_strength=breakout_strength_last_f,
            failure_strength=1.0 if fd else 0.0,
            basis_norm=basis_norm_last_f,
            oi_signal=oi_signal_last if use_oi else 0.0,
            config=config,
        )
        score_hist = _score_history_for_side(feats, "LONG", config, use_oi)
    else:
        score_pts = 0.0
        score_components = {"breakout": 0.0, "failure": 0.0, "basis": 0.0, "oi": 0.0}
        score_hist = np.asarray([], dtype=np.float64)
    dynamic_threshold = _adaptive_threshold(score_hist, config, fallback=min_entry)

    bar_index = n - 1
    in_up, in_dn = _trap_window_last_bar(trap_mem, bar_index, trap_lb)
    current_basis = feat_pack.get("basis")
    basis_ok_long = (basis_norm_last_f <= 0.0) or br_l
    basis_ok_short = (basis_norm_last_f >= 0.0) or br_s
    oi_confirm: Any = oi_signal_last if use_oi else "skipped"
    min_confirm = float(config.get("confirm_trap_min_score", 2.0))
    entry_require_basis = bool(config.get("entry_require_basis", False))

    def _entry_full_trace(action: str, reason: str) -> Dict[str, Any]:
        age_up = (
            bar_index - trap_mem.breakout_up_index
            if trap_mem.breakout_up_index is not None
            else None
        )
        age_dn = (
            bar_index - trap_mem.breakout_down_index
            if trap_mem.breakout_down_index is not None
            else None
        )
        return {
            "stage": "entry",
            "bar_index": bar_index,
            "breakout_up": bu,
            "breakout_down": bd,
            "failure_up": fu,
            "failure_down": fd,
            "in_trap_window_up": in_up,
            "in_trap_window_down": in_dn,
            "last_breakout_up": trap_mem.last_breakout_up,
            "last_breakout_down": trap_mem.last_breakout_down,
            "breakout_up_index": trap_mem.breakout_up_index,
            "breakout_down_index": trap_mem.breakout_down_index,
            "breakout_age_up": age_up,
            "breakout_age_down": age_dn,
            "basis": current_basis,
            "basis_ok_long": basis_ok_long,
            "basis_ok_short": basis_ok_short,
            "oi_confirm": oi_confirm,
            "use_open_interest": use_oi,
            "trap_score": float(score_pts),
            "min_score": min_confirm,
            "entry_score_min": min_entry,
            "dynamic_threshold": float(dynamic_threshold),
            "score_components": dict(score_components),
            "setup_short": setup_short,
            "setup_long": setup_long,
            "action": action,
            "reason": reason,
        }

    debug_log: Dict[str, Any] = {
        "bar_index": bar_index,
        "breakout_up": bu,
        "failure_up": fu,
        "breakout_down": bd,
        "failure_down": fd,
        "memory": trap_mem.to_debug_dict(),
        "trap_score": float(score_pts),
    }
    logger.info(
        "TTM trap debug",
        extra={"strategy": "TTM", "ttm_trap_debug": debug_log},
    )

    debug_payload: Dict[str, Any] = {
        "bar_index": bar_index,
        "trap_lookback_bars": trap_lb,
        "breakout_up": bu,
        "breakout_down": bd,
        "failure_up": fu,
        "failure_down": fd,
        "setup_short": setup_short,
        "setup_long": setup_long,
        "trap_memory": trap_mem.to_debug_dict(),
        "basis": feat_pack.get("basis"),
        "basis_norm": basis_norm_last_f,
        "basis_change": bc,
        "basis_delta": last.get("basis_delta"),
        "oi_delta": oi_ch,
        "oi_z": oi_z,
        "oi_signal": oi_signal_last,
        "momentum": last.get("momentum"),
        "vol_spike": last.get("vol_spike"),
        "score": score_pts,
        "dynamic_threshold": float(dynamic_threshold),
        "score_components": score_components,
        "basis_reversal_short": br_s,
        "basis_reversal_long": br_l,
        "ttm_trap_debug": debug_log,
    }
    logger.info("TTM debug", extra={"strategy": "TTM", "ttm_debug": debug_payload})

    def _log_hold(reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {**debug_payload, **(extra or {})}
        tr = _entry_full_trace("HOLD", reason)
        if extra:
            tr = {**tr, **extra}
        logger.info(
            "TTM: không vào lệnh",
            extra={
                "strategy": "TTM",
                "oi": last.get("oi_level"),
                "oi_zscore": last.get("oi_zscore"),
                "oi_signal": last.get("oi_signal"),
                "oi_z": oi_z,
                "basis_change": bc,
                "action": "HOLD",
                "ly_do": reason,
                "ttm_debug": payload,
            },
        )
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": reason,
                "strategy": "TTM",
                "trap_score": float(score_pts),
                "features": feat_pack,
                "debug": payload,
            },
            tr,
            print_trace,
        )

    if not setup_short and not setup_long:
        reason_ns = _classify_no_setup(fu, fd, in_up, in_dn, trap_mem)
        tr = _entry_full_trace("HOLD", reason_ns)
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": reason_ns,
                "strategy": "TTM",
                "trap_score": float(score_pts),
                "features": feat_pack,
                "debug": {**debug_payload, "reason_detail": reason_ns},
            },
            tr,
            print_trace,
        )

    if entry_require_basis:
        if setup_short and not setup_long and not basis_ok_short:
            return _log_hold("basis_filter_failed", {"blocked_side": "SHORT"})
        if setup_long and not setup_short and not basis_ok_long:
            return _log_hold("basis_filter_failed", {"blocked_side": "LONG"})

    if setup_short and setup_long:
        if score_pts >= dynamic_threshold:
            return _log_hold("ambiguous_setup")
        return _log_hold(
            "score_too_low",
            {"min_entry": float(min_entry), "dynamic_threshold": float(dynamic_threshold)},
        )

    if score_pts < dynamic_threshold:
        return _log_hold(
            "score_too_low",
            {"min_entry": float(min_entry), "dynamic_threshold": float(dynamic_threshold)},
        )

    conf = _norm_confidence_01(float(score_pts), cap=4.0)
    if setup_short:
        out = {
            "action": "SHORT",
            "confidence": conf,
            "reason": "score_entry_short",
            "strategy": "TTM",
            "trap_score": float(score_pts),
            "features": feat_pack,
            "debug": debug_payload,
        }
        logger.info(
            "TTM: tín hiệu SHORT (điểm)",
            extra={"strategy": "TTM", "action": "SHORT", "score": score_pts, "ttm_debug": debug_payload},
        )
        tr_s = _entry_full_trace("SHORT", "score_entry_short")
        return _emit_signal(out, tr_s, print_trace)

    out = {
        "action": "LONG",
        "confidence": conf,
        "reason": "score_entry_long",
        "strategy": "TTM",
        "trap_score": float(score_pts),
        "features": feat_pack,
        "debug": debug_payload,
    }
    logger.info(
        "TTM: tín hiệu LONG (điểm)",
        extra={"strategy": "TTM", "action": "LONG", "score": score_pts, "ttm_debug": debug_payload},
    )
    tr_l = _entry_full_trace("LONG", "score_entry_long")
    return _emit_signal(out, tr_l, print_trace)


def _normalize_action(a: Any) -> str:
    return str(a or "HOLD").upper()


def _merge_shadow_style(
    out_v1: Dict[str, Any],
    out_v2: Dict[str, Any],
    feats_last: Optional[Dict[str, Any]],
    *,
    mode_label: str,
) -> Dict[str, Any]:
    a1 = _normalize_action(out_v1.get("action"))
    a2 = _normalize_action(out_v2.get("action"))
    div = a1 != a2
    out = dict(out_v1)
    d2 = out_v2.get("debug") or {}
    out["ttm_mode"] = mode_label
    out["v2"] = {
        "action": a2,
        "reason": out_v2.get("reason"),
        "confidence": out_v2.get("confidence"),
        "prob_long": d2.get("prob_long"),
        "prob_short": d2.get("prob_short"),
        "score_long": d2.get("score_long"),
        "score_short": d2.get("score_short"),
        "trap_score": out_v2.get("trap_score"),
    }
    out["divergence"] = div
    out["divergence_detail"] = f"v1={a1} v2={a2}" if div else "aligned"
    dbg = dict(out_v1.get("debug") or {})
    if feats_last is not None:
        dbg["features_snapshot"] = feats_last
    dbg["v2_debug"] = d2
    out["debug"] = dbg
    return out


def generate_ttm_signal(
    features_or_data: Mapping[str, Any] | Dict[str, Any],
    config: Mapping[str, Any],
    *,
    position_side: Optional[str] = None,
    live_mode: bool = False,
) -> Dict[str, Any]:
    """Dispatch V1 / V2 / shadow per ``ttm_mode`` in config (default: v1_only)."""
    from src.strategies.ttm.ttm_features import compute_features
    from src.strategies.ttm.ttm_signal_v2 import generate_ttm_signal_v2

    cfg = dict(config)
    mode = str(cfg.get("ttm_mode", "v1_only")).lower().strip()
    feats_last: Optional[Dict[str, Any]] = None
    if "bars" in features_or_data and features_or_data.get("bars") is not None:
        try:
            feats_full = compute_features(dict(features_or_data), cfg)
            feats_last = features_last_row(feats_full)
        except Exception:
            feats_last = None

    out_v1 = generate_ttm_signal_v1(
        features_or_data, config, position_side=position_side, live_mode=live_mode
    )

    def _run_v2_safe() -> Optional[Dict[str, Any]]:
        try:
            return generate_ttm_signal_v2(
                features_or_data, config, position_side=position_side, live_mode=live_mode
            )
        except Exception as e:
            logger.warning("TTM V2 failed", extra={"err": str(e)})
            return None

    if mode == "v1_only":
        if not cfg.get("log_v2_in_v1_only"):
            return out_v1
        out_v2 = _run_v2_safe()
        if out_v2 is None:
            return out_v1
        return _merge_shadow_style(out_v1, out_v2, feats_last, mode_label="v1_only")

    if mode == "shadow":
        out_v2 = _run_v2_safe()
        if out_v2 is None:
            o = dict(out_v1)
            o["ttm_mode"] = "shadow"
            o["v2_error"] = "v2_failed"
            return o
        return _merge_shadow_style(out_v1, out_v2, feats_last, mode_label="shadow")

    if mode == "v2_only":
        out_v2 = _run_v2_safe()
        if out_v2 is None:
            o = dict(out_v1)
            o["ttm_mode"] = "v2_only"
            o["v2_fallback"] = True
            o["v2_fallback_reason"] = "exception"
            return o
        reason = str(out_v2.get("reason") or "")
        dbg = out_v2.get("debug") or {}
        if reason == "v2_invalid_probs" or dbg.get("probs_valid") is False:
            o = dict(out_v1)
            o["ttm_mode"] = "v2_only"
            o["v2_fallback"] = True
            o["v2_fallback_reason"] = "invalid_probs"
            o["debug"] = {**(o.get("debug") or {}), "v2_skipped": dbg}
            return o
        out_v2["ttm_mode"] = "v2_only"
        return out_v2

    logger.warning("TTM: unknown ttm_mode, using v1", extra={"mode": mode})
    return out_v1
