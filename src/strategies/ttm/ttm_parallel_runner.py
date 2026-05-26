"""
Chạy song song TTM V1 (rule-based) và TTM V2 (scoring) trên cùng dòng dữ liệu (paper / replay).

Chỉ dùng với mã phái sinh (hợp đồng tương lai); không áp dụng cổ phiếu — ``paper_test`` chặn mã không phải derivative.

- Ghi JSONL: quyết định mỗi bar (kèm ``forward_return``, ``vol_z``, ``regime_tag``), một dòng trade
  ``CLOSED`` khi đóng vị thế (entry+exit, ``position_size``, ``realized_return``), cơ hội bỏ lỡ.
- Hai vị thế giả lập độc lập (v1 / v2).
- Metrics: agreement, block reasons, follow_design (V2), báo cáo cuối phiên.

Hành vi deterministic nếu cùng config và cùng chuỗi bar.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, TextIO, Tuple

import numpy as np

from src.logger import get_logger
from src.strategies.ttm.config import (
    TTM_CONFIG,
    build_ttm_live_adaptive_config,
    build_ttm_paper_live_config,
    build_ttm_research_parallel_config,
)
from src.strategies.ttm.ttm_alignment import fingerprint_aligned_tail
from src.strategies.ttm.ttm_features import (
    compute_ttm_features_from_config,
    features_last_row,
    features_row_at,
)
from src.strategies.ttm.ttm_regime import detect_regime
from src.strategies.ttm.ttm_signal import generate_ttm_signal_v1
from src.strategies.ttm.empirical.adaptive_engine import EmpiricalAlphaEngine
from src.strategies.ttm.ttm_score import compute_score_v2_alpha
from src.strategies.ttm.ttm_signal_v2 import generate_ttm_signal_v2
from src.strategies.ttm.ttm_v2_gates import (
    ResearchGateSessionState,
    entry_confirm_mode_for_gate,
    eval_data_stale,
    record_research_trade_if_applicable,
    resolve_gate_mode,
)
from src.strategies.ttm.ttm_v2_phases import classify_crowd_phase
from src.strategies.ttm.ttm_v2_exit_continuation import map_parallel_runner_reason_to_canonical
from src.strategies.ttm.ttm_strategy import TTMDerivativesStrategy
from src.strategies.ttm.ttm_v2_short import (
    build_exhaustion_short_entry_meta,
    build_short_opportunity_entry_meta,
    check_exhaustion_short_exit,
    check_short_v3_structural_exit,
)
from src.strategies.ttm.ttm_execution_realism import (
    ExecutionRealismConfig,
    adjust_fill_price,
    entry_due_unix,
    slippage_abs_at_bar,
)

# Bật in chi tiết (lý do không vào lệnh, điều kiện, score vs ngưỡng)
DEBUG_VERBOSE: bool = False

logger = get_logger("ttm_parallel_runner")


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _finite_or_none(obj: Any) -> Optional[float]:
    try:
        v = float(obj)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _v2_long_entry_calibration_from_features(feat: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Snapshot V2 LONG calibration scalars from the entry signal bar (for trade JSONL)."""
    if not feat:
        return None
    return {
        "raw_strength": _json_safe(feat.get("raw_strength")),
        "extension": _json_safe(feat.get("extension")),
        "effective_strength": _json_safe(feat.get("effective_strength")),
        "last_bar_return": _json_safe(feat.get("last_bar_return")),
    }


def _build_v2_entry_snapshot(
    sig_v2: Mapping[str, Any],
    side: str,
    config: Mapping[str, Any],
    *,
    signal_bar_snapshot: Optional[Mapping[str, Any]] = None,
    signal_bar_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Persist score/phase/continuation at entry for CLOSED trade flattening (refactor_4)."""
    dbg = dict(sig_v2.get("debug") or {})
    sc = dbg.get("v2_score_components") if isinstance(dbg.get("v2_score_components"), dict) else {}
    sh = dbg.get("v2_short_components") if isinstance(dbg.get("v2_short_components"), dict) else {}
    _ = str(side).upper()
    snap: Dict[str, Any] = {
        "entry_score_long": dbg.get("score_long"),
        "entry_crowd_phase": dbg.get("crowd_phase") or sc.get("crowd_phase"),
        "continuation_score_at_entry": sc.get("continuation_confirm"),
        "short_entry_phase": sh.get("short_phase") or dbg.get("short_phase"),
        "short_entry_score": sh.get("short_score"),
        "short_entry_reason": sh.get("short_reason"),
        "entry_prob_long": dbg.get("prob_long"),
        "entry_prob_short": dbg.get("prob_short"),
        "target_alpha_hold_bars": int(config.get("ttm_v2_target_alpha_hold_bars", 3)),
    }
    if signal_bar_index is not None:
        snap["signal_bar_index"] = int(signal_bar_index)
    if isinstance(signal_bar_snapshot, dict):
        for k, v in signal_bar_snapshot.items():
            if v is not None or k == "signal_late_fomo_flag":
                snap[str(k)] = v
    return {k: _json_safe(v) for k, v in snap.items()}


def _snapshot_feature_value(last: Mapping[str, Any], key: str) -> Any:
    vm = last.get("feature_valid_mask") or {}
    if isinstance(vm, Mapping) and not bool(vm.get(key, True)):
        return None
    return _json_safe(last.get(key))


def _norm_signal(action: Any) -> str:
    a = str(action or "HOLD").upper()
    if a in ("LONG", "SHORT"):
        return a
    return "NONE"


def _regime_id_to_tag(regime_id: int) -> str:
    """Human-readable tag from :func:`detect_regime` {-1,0,1}."""
    rid = int(regime_id)
    if rid < 0:
        return "vol_low"
    if rid > 0:
        return "vol_high"
    return "vol_neutral"


def _forward_return_from_closes(closes: List[float], bar_index: int, h: int) -> Optional[float]:
    """Same definition as ``ttm_validation._forward_return`` (fraction, h bars ahead)."""
    if h <= 0 or bar_index < 0:
        return None
    i = int(bar_index)
    if i + h >= len(closes):
        return None
    try:
        a = float(closes[i])
        b = float(closes[i + h])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(a) or not np.isfinite(b) or abs(a) < 1e-12:
        return None
    return float((b - a) / abs(a))


def _classify_v2_blocked(
    action: str,
    reason: str,
    dbg: Mapping[str, Any],
) -> str:
    """Map lý do V2 -> threshold | risk | regime | data | none."""
    if action in ("LONG", "SHORT"):
        return "none"
    r = str(reason or "").lower()
    if "misaligned" in r or "no_data" in r:
        return "data"
    if "invalid_prob" in r or "v2_invalid" in r:
        return "threshold"
    if r == "score_long_below_threshold" or "score_long_below_threshold" in r:
        return "score_long_below_threshold"
    if "long_gate_" in r or r == "long_gate_block":
        return "long_gate"
    if r in (
        "no_prior_upside_breakout",
        "long_continuation_early",
        "short_chase_risk",
        "rejection_not_confirmed",
        "rejection_weak",
        "insufficient_setup",
        "below_trigger_threshold",
        "long_continuation_recovered",
    ) or r.startswith("short_phase_") or r.startswith("short_gate_"):
        return r
    if "below_entry" in r or "threshold" in r or "weak" in r or "tie" in r:
        return "threshold"
    if "basis_filter" in r or "entry_require" in r:
        return "risk"
    rid = dbg.get("ttm_regime_id")
    if rid is not None and "regime" in r:
        return "regime"
    return "none"


# Placeholders for crowd-alpha breakdown (commit 1: all None; populated in later commits).
_V2_SCORE_COMPONENTS_EMPTY: Dict[str, Any] = {
    "breakout_conviction": None,
    "continuation_confirm": None,
    "basis_confirm": None,
    "extension": None,
    "extension_sq": None,
    "last_bar_return_raw": None,
    "last_bar_return_norm": None,
    "positive_last_bar_return": None,
    "late_phase_penalty": None,
    "effective_strength_raw": None,
    "effective_strength": None,
    "score_long": None,
    "crowd_phase": None,
    "phase_reason": None,
    "late_fomo_flag": None,
    "entry_block_reason": None,
}

_V2_SHORT_COMPONENTS_EMPTY: Dict[str, Any] = {
    "prior_upside_breakout_exists": None,
    "last_upside_breakout_bar_index": None,
    "last_upside_breakout_score": None,
    "last_upside_breakout_phase": None,
    "last_upside_breakout_extension": None,
    "bars_since_upside_breakout": None,
    "crowded_long_pressure": None,
    "continuation_decay": None,
    "rejection_confirm": None,
    "failed_breakout_confirm": None,
    "downside_momentum_confirm": None,
    "early_continuation_still_alive": None,
    "short_chase_risk": None,
    "short_effective_strength_raw": None,
    "short_effective_strength": None,
    "short_score": None,
    "short_phase": None,
    "short_candidate": None,
    "short_block_reason": None,
    "short_reason": None,
}

def _merge_v2_score_components_log(dbg: Mapping[str, Any]) -> Dict[str, Any]:
    sc = {k: _json_safe(v) for k, v in _V2_SCORE_COMPONENTS_EMPTY.items()}
    packed = dbg.get("v2_score_components")
    if isinstance(packed, dict):
        for k in sc:
            if k in packed:
                sc[k] = _json_safe(packed[k])
    return sc


def _merge_v2_short_components_log(dbg: Mapping[str, Any]) -> Dict[str, Any]:
    sc = {k: _json_safe(v) for k, v in _V2_SHORT_COMPONENTS_EMPTY.items()}
    packed = dbg.get("v2_short_components")
    if isinstance(packed, dict):
        for k in sc:
            if k in packed:
                sc[k] = _json_safe(packed[k])
    return sc


def _v2_signal_bar_snapshot(sig_v2: Mapping[str, Any], feat: Mapping[str, Any]) -> Dict[str, Any]:
    """Scalars + score_components from the signal bar (execution may be next bar)."""
    dbg = dict(sig_v2.get("debug") or {})
    sc = dbg.get("v2_score_components") if isinstance(dbg.get("v2_score_components"), dict) else {}
    eff = dbg.get("effective_strength")
    if eff is None:
        eff = feat.get("effective_strength")
    lf = dbg.get("late_fomo_flag")
    if lf is None:
        lf = sc.get("late_fomo_flag")
    ebr = dbg.get("entry_block_reason")
    if ebr is None:
        ebr = sc.get("entry_block_reason")
    out: Dict[str, Any] = {
        "signal_score_long": dbg.get("score_long"),
        "signal_effective_strength": eff,
        "signal_crowd_phase": dbg.get("crowd_phase") or sc.get("crowd_phase"),
        "signal_late_fomo_flag": lf,
        "signal_entry_block_reason": ebr,
    }
    v2sc = dbg.get("v2_score_components")
    if isinstance(v2sc, dict):
        out["signal_score_components"] = _json_safe(dict(v2sc))
    return {k: _json_safe(v) for k, v in out.items()}


def _v2_entry_confirm_row(
    feats: Mapping[str, Any],
    execution_last_row: Mapping[str, Any],
    *,
    signal_bar_index: Optional[int] = None,
) -> Tuple[Dict[str, Any], str, Optional[int]]:
    """
    Causal confirm row for signal@N → fill@N+1 backtest: use closed features at signal bar,
    not the execution bar row (which includes the full N+1 candle — lookahead).
    """
    if signal_bar_index is not None:
        try:
            sbi = int(signal_bar_index)
            n = int(feats.get("n", 0) or 0)
            if 0 <= sbi < n:
                return dict(features_row_at(dict(feats), sbi)), "signal_bar", sbi
        except (TypeError, ValueError):
            pass
    return dict(execution_last_row), "execution_bar", None


def _v2_entry_execution_long_confirm(
    feats: Mapping[str, Any],
    last_row: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    signal_only_exec: bool,
    signal_crowd_phase: Optional[str] = None,
    signal_long_candidate: bool = False,
    entry_confirm_mode: str = "strict",
    signal_bar_index: Optional[int] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    At execution bar: optional block for LONG when confirmation enabled.
    Returns (blocked, block_reason, fields_to_log).
    """
    if not bool(config.get("ttm_v2_enable_entry_confirmation", False)):
        return False, "", {}
    confirm_row, confirm_src, confirm_bi = _v2_entry_confirm_row(
        feats, last_row, signal_bar_index=signal_bar_index
    )
    sl, _ss, comp = compute_score_v2_alpha(feats, confirm_row, config)
    cp = str(comp.get("crowd_phase") or "")
    thr = float(config.get("ttm_v2_score_long_entry_threshold", 0.0))
    from src.strategies.ttm.ttm_v2_gates import research_score_long_floor

    research_floor = research_score_long_floor(config)
    bu_ok = bool(
        confirm_row.get("breakout_up_filtered_last", confirm_row.get("breakout_up"))
    )
    sig_cp = str(signal_crowd_phase or "").strip().lower()
    fields: Dict[str, Any] = {
        "entry_confirm_score_long": float(sl),
        "entry_confirm_phase": cp,
        "entry_confirm_breakout_valid": bool(bu_ok),
        "entry_confirm_mode": str(entry_confirm_mode),
        "signal_crowd_phase": signal_crowd_phase,
        "signal_long_candidate": bool(signal_long_candidate),
        "confirm_data_source": confirm_src,
        "confirm_bar_index": confirm_bi,
    }
    mode_u = str(entry_confirm_mode or "strict").strip().lower()
    if mode_u == "research":
        if bool(confirm_row.get("exhaustion_confirm")) or cp == "exhaustion":
            return True, "entry_confirm_exhaustion_confirm", fields
        if bool(comp.get("late_fomo_flag")) or cp == "late_fomo":
            return True, "entry_confirm_late_fomo_extreme", fields
        adv_thr = float(config.get("ttm_v2_research_entry_hard_adverse_return", -0.0015))
        try:
            lbv = float(confirm_row.get("last_bar_return") or 0.0)
        except (TypeError, ValueError):
            lbv = 0.0
        if lbv < adv_thr:
            return True, "entry_confirm_hard_adverse_move", fields
        if bool(comp.get("failed_breakout_confirm")) or str(comp.get("crowd_phase") or "") == "failed_breakout":
            return True, "entry_confirm_failed_breakout", fields
        if eval_data_stale(confirm_row, comp):
            return True, "entry_confirm_data_stale", fields
        if not bu_ok and not signal_only_exec and not signal_long_candidate:
            if sig_cp not in ("ignition", "early_continuation"):
                return True, "entry_confirm_no_breakout_at_fill", fields
        fields["entry_confirm_result"] = "pass"
        fields["entry_confirm_block_reason"] = None
        return False, "", fields
    bad_phases = ("no_breakout", "late_fomo", "exhaustion", "failed_breakout")
    if cp in bad_phases:
        return True, "entry_confirm_bad_phase", fields
    if float(sl) <= float(thr):
        return True, "entry_confirm_score_below_threshold", fields
    if not bu_ok and not signal_only_exec:
        return True, "entry_confirm_no_breakout_at_fill", fields
    fields["entry_confirm_result"] = "pass"
    fields["entry_confirm_block_reason"] = None
    return False, "", fields


def _refine_v2_trade_exit_reason(
    canon: str,
    *,
    entry_crowd_phase: Any,
    exit_crowd_phase: Any,
    dbg_exit: Mapping[str, Any],
) -> str:
    """Finer trade-log labels without changing exit ordering (refactor 20260511)."""
    c = str(canon or "").strip().lower()
    ent_cp = str(entry_crowd_phase or "").strip().lower()
    ex_cp = str(exit_crowd_phase or "").strip().lower()
    if c == "failed_breakout":
        return "failed_breakout_exit"
    if c == "score_decay":
        if ent_cp and ent_cp != "no_breakout" and ex_cp == "no_breakout":
            return "no_breakout_after_entry"
        return "score_decay_exit"
    if c == "hard_stop":
        return "hard_stop"
    if c == "exhaustion_rejection":
        return "exhaustion_rejection"
    if c == "continuation_decay":
        return "continuation_decay"
    if c == "time_stop":
        return "time_stop"
    if c == "take_profit":
        return "take_profit"
    v2xr = str(dbg_exit.get("v2_exit_reason") or "")
    if c == "unknown" and v2xr:
        return v2xr
    return c if c else "unknown"


# Flat CLOSED trade row extensions for V2 (refactor_1 foundation).
_V2_CLOSED_TRADE_FLAT_DEFAULTS: Dict[str, Any] = {
    "signal_bar_index": None,
    "signal_score_long": None,
    "signal_effective_strength": None,
    "signal_crowd_phase": None,
    "signal_late_fomo_flag": None,
    "signal_entry_block_reason": None,
    "signal_score_components": None,
    "entry_confirm_score_long": None,
    "entry_confirm_phase": None,
    "entry_confirm_breakout_valid": None,
    "entry_crowd_phase": None,
    "exit_crowd_phase": None,
    "entry_blocked": None,
    "entry_block_reason": None,
    "exit_reason": None,
    "holding_bars": None,
    "entry_score_long": None,
    "exit_score_long": None,
    "continuation_score_at_entry": None,
    "continuation_score_at_exit": None,
    "mfe": None,
    "mae": None,
    "bars_to_mfe": None,
    "short_entry_phase": None,
    "short_exit_phase": None,
    "short_entry_score": None,
    "short_exit_score": None,
    "short_entry_reason": None,
    "short_exit_reason": None,
    "prior_upside_breakout_bar": None,
    "bars_since_upside_breakout_at_entry": None,
    "crowded_long_pressure_at_entry": None,
    "continuation_decay_at_entry": None,
    "rejection_confirm_at_entry": None,
    "short_chase_risk_at_entry": None,
}


def _build_v1_block(
    sig: Mapping[str, Any],
    config: Mapping[str, Any],
    last_row: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    dbg = dict(sig.get("debug") or {})
    action = str(sig.get("action", "HOLD")).upper()
    signal = _norm_signal(action)
    reason_block = str(sig.get("reason") or "") if signal == "NONE" else ""
    setup_s = bool(dbg.get("setup_short", False))
    setup_l = bool(dbg.get("setup_long", False))
    breakout = setup_s or setup_l
    vol_spike = float(last_row.get("vol_spike") or 0.0) if last_row else 0.0
    vt = float(config.get("vol_threshold", config.get("vol_regime_bound", 1.2)))
    volume = bool(np.isfinite(vol_spike) and vol_spike >= vt)
    score_pts = float(dbg.get("score", 0.0) or 0.0)
    dthr = float(dbg.get("dynamic_threshold", 0.0) or 0.0)
    filter_pass = bool(breakout and score_pts >= dthr)
    return {
        "signal": signal if signal != "NONE" else "NONE",
        "reason_block": reason_block,
        "conditions": {
            "breakout": breakout,
            "volume": volume,
            "filter_pass": filter_pass,
        },
    }


def _build_v2_block(
    sig: Mapping[str, Any],
    config: Mapping[str, Any],
    position_side: Optional[str] = None,
    *,
    execution_position_open: bool = False,
) -> Dict[str, Any]:
    dbg = dict(sig.get("debug") or {})
    action = str(sig.get("action", "HOLD")).upper()
    comp = dict(dbg.get("score_components") or {})
    pos_u = (position_side or "").strip().upper()
    entry_thr = float(config.get("entry_threshold", 0.6))

    # Chỉ coi là "blocked_by: position" khi runner thực sự đang có vị thế mở (sau ENTRY).
    in_execution = execution_position_open and pos_u in ("LONG", "SHORT")
    if in_execution and action == "HOLD":
        decision = "HOLD"
        blocked_by = "position"
    elif action == "EXIT":
        decision = "EXIT"
        blocked_by = "none"
    elif action in ("LONG", "SHORT"):
        decision = action
        blocked_by = "none"
    else:
        decision = "NONE"
        blocked_by = _classify_v2_blocked(action, str(sig.get("reason") or ""), dbg)

    pl = dbg.get("prob_long")
    ps = dbg.get("prob_short")
    sl = dbg.get("score_long")
    ss = dbg.get("score_short")
    if DEBUG_VERBOSE and decision in ("NONE", "HOLD"):
        print(
            f"[TTM parallel V2] action={action} decision={decision} reason={sig.get('reason')} "
            f"pl={pl} ps={ps} entry_thr={entry_thr} sl={sl} ss={ss} blocked={blocked_by}"
        )
    return {
        "score_long": _json_safe(sl),
        "score_short": _json_safe(ss),
        "prob_long": _json_safe(pl),
        "prob_short": _json_safe(ps),
        "decision": decision,
        "components": {
            "alpha_raw": _json_safe(comp.get("alpha_raw")),
            "alpha_rank": _json_safe(comp.get("alpha_rank")),
            "score": _json_safe(comp.get("score")),
            "edge": _json_safe(comp.get("edge")),
            "signed_breakout": _json_safe(comp.get("breakout_signed_last", comp.get("signed_breakout"))),
            "positioning": _json_safe(comp.get("price_momentum_last", comp.get("positioning_strength"))),
            "basis_mom": _json_safe(comp.get("basis_mom_last")),
            "trap": _json_safe(comp.get("alpha_trap_score")),
            # Legacy keys for older report parsers (V2 alpha does not use OI).
            "momentum": _json_safe(comp.get("momentum")),
            "basis": _json_safe(comp.get("basis_mom_last", comp.get("positioning_strength"))),
            "oi": 0.0,
        },
        "blocked_by": blocked_by,
        "score_components": _merge_v2_score_components_log(dbg),
        "short_components": _merge_v2_short_components_log(dbg),
        "gate_diagnostics": _json_safe(dbg.get("gate_diagnostics") or dbg.get("v2_gate_diagnostics")),
        "log": {
            "decision_source": dbg.get("decision_source", "threshold"),
            "threshold": _json_safe(dbg.get("entry_threshold", entry_thr)),
            "input_aligned_fingerprint": dbg.get("input_aligned_fingerprint"),
            "ttm_config_profile": dbg.get("ttm_config_profile"),
            "crowd_phase": dbg.get("crowd_phase"),
            "short_phase": dbg.get("short_phase"),
            "gate_mode": dbg.get("gate_mode") or (dbg.get("gate_diagnostics") or {}).get("gate_mode"),
        },
    }


def _features_snapshot(last: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "breakout_strength": _snapshot_feature_value(last, "breakout_strength"),
        "breakout_strength_cap": _snapshot_feature_value(last, "breakout_strength_cap"),
        "raw_strength": _snapshot_feature_value(last, "raw_strength"),
        "cap": _snapshot_feature_value(last, "cap"),
        "failure_strength": _json_safe(last.get("failure_strength")),
        "basis_norm": _json_safe(last.get("basis_norm")),
        "oi_signal": _json_safe(last.get("oi_signal")),
        "positioning_oi_core": _json_safe(last.get("positioning_oi_core")),
        "vol_regime": _json_safe(last.get("vol_regime")),
        "positioning_strength": _json_safe(last.get("positioning_strength")),
        "exhaustion_candidate": _json_safe(last.get("exhaustion_candidate")),
        "exhaustion_confirm": _json_safe(last.get("exhaustion_confirm")),
        "short_score": _snapshot_feature_value(last, "short_score"),
        "extension": _snapshot_feature_value(last, "extension"),
        "last_bar_return": _snapshot_feature_value(last, "last_bar_return"),
        "effective_strength_pre_gate": _snapshot_feature_value(last, "effective_strength_pre_gate"),
        "effective_strength_active": _json_safe(last.get("effective_strength_active")),
        "effective_strength": _snapshot_feature_value(last, "effective_strength"),
    }


def _oi_data_for_jsonl(
    data: Mapping[str, Any],
    *,
    last: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Chuỗi OI thô + meta enricher — giải thích khi ``oi_signal`` = 0 (thiếu DNSE vs không đổi)."""
    oi_raw = data.get("open_interest")
    last_lv = None
    max_lv = None
    if oi_raw is not None:
        oi_list = list(oi_raw)
        if oi_list:
            try:
                floats = [float(x) for x in oi_list]
                last_lv = floats[-1]
                max_lv = max(floats)
            except (TypeError, ValueError):
                pass
    meta = data.get("oi_enricher_meta") if isinstance(data.get("oi_enricher_meta"), dict) else {}
    oi_proxy_used = bool(last.get("oi_proxy_used")) if isinstance(last, dict) else False
    if oi_proxy_used and last is not None:
        try:
            lv = float(last.get("oi_level", 0.0))
            if np.isfinite(lv):
                last_lv = lv
        except (TypeError, ValueError):
            pass
    elif last_lv is None and last is not None:
        # Fallback to feature-level OI (real or proxy) when raw stream is absent.
        try:
            lv = float(last.get("oi_level", 0.0))
            if np.isfinite(lv):
                last_lv = lv
        except (TypeError, ValueError):
            pass
    return {
        "open_interest_last": _json_safe(last_lv),
        "open_interest_max": _json_safe(max_lv),
        "oi_proxy_used": oi_proxy_used,
        "oi_source": _json_safe(meta.get("oi_source")),
        "openInterestQuantity_rest": _json_safe(meta.get("openInterestQuantity_rest")),
        "openInterestQuantity_final": _json_safe(meta.get("openInterestQuantity")),
        "secdef_symbol_used": _json_safe(meta.get("secdef_symbol_used")),
        "dnse_secdef_http_status": _json_safe(meta.get("status")),
    }


def _merge_features(
    last: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    feats: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out = _features_snapshot(last)
    pipe = last.get("oi_pipeline")
    if isinstance(pipe, dict):
        out["oi_pipeline"] = {k: _json_safe(pipe.get(k)) for k in ("oi", "oi_zscore", "oi_signal")}
    plog = last.get("positioning_log")
    if isinstance(plog, dict):
        out["positioning_log"] = {k: _json_safe(plog.get(k)) for k in (
            "price_change",
            "price_signal",
            "oi_zscore",
            "oi_signal",
            "pos_core",
            "basis_norm",
            "basis_signal",
            "basis_effect",
            "positioning_strength",
        )}
    vlog = last.get("volume_participation_log")
    if isinstance(vlog, dict):
        out["volume_participation_log"] = {k: _json_safe(vlog.get(k)) for k in (
            "volume",
            "vol_zscore",
            "vol_signal",
            "participation_strength",
        )}
    out["oi_data"] = _oi_data_for_jsonl(data, last=last)
    if feats is not None:
        for raw_key, count_key in (
            ("breakout_up_raw", "breakout_up_raw_count"),
            ("breakout_up", "breakout_up_filtered_count"),
            ("breakout_down_raw", "breakout_down_raw_count"),
            ("breakout_down", "breakout_down_filtered_count"),
        ):
            arr = feats.get(raw_key)
            if isinstance(arr, np.ndarray):
                out[count_key] = int(np.sum(arr))
        out["breakout_up_raw_last"] = bool(last.get("breakout_up_raw"))
        out["breakout_up_filtered_last"] = bool(last.get("breakout_up"))
        out["breakout_down_raw_last"] = bool(last.get("breakout_down_raw"))
        out["breakout_down_filtered_last"] = bool(last.get("breakout_down"))
        out["is_breakout_up"] = bool(last.get("breakout_up_raw"))
        out["is_breakout_down"] = bool(last.get("breakout_down_raw"))
    return out


@dataclass
class ParallelPosition:
    """Trạng thái vị thế mô phỏng cho một model (v1 hoặc v2)."""

    is_open: bool = False
    side: Optional[str] = None
    entry_price: float = 0.0
    entry_bar_index: int = -1
    holding_bars: int = 0
    quantity: int = 1
    entry_time: str = ""
    paper_exec_meta: Optional[Dict[str, Any]] = None
    entry_calibration: Optional[Dict[str, Any]] = None
    entry_v2_meta: Optional[Dict[str, Any]] = None
    mfe_return: float = 0.0
    mae_return: float = 0.0
    bars_to_mfe: int = -1
    v2_extrema_initialized: bool = False

    def assert_consistent(self) -> None:
        if self.is_open:
            assert self.side in ("LONG", "SHORT")
        else:
            assert self.side is None

    def effective_position_side(self) -> Optional[str]:
        """Chỉ trả về LONG/SHORT khi thực sự có vị thế mở (tránh stale side)."""
        return self.side if self.is_open else None

    def state(self) -> str:
        if not self.is_open:
            return "FLAT"
        return str(self.side or "FLAT")

    def open_position(
        self,
        side: str,
        price: float,
        bar_ix: int,
        quantity: int = 1,
        entry_time: str = "",
        *,
        paper_exec_meta: Optional[Dict[str, Any]] = None,
        entry_calibration: Optional[Dict[str, Any]] = None,
        entry_v2_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert not self.is_open
        su = str(side).strip().upper()
        assert su in ("LONG", "SHORT")
        self.is_open = True
        self.side = su
        self.entry_price = float(price)
        self.entry_bar_index = bar_ix
        self.holding_bars = 0
        self.quantity = max(1, int(quantity))
        self.entry_time = str(entry_time or "")
        self.paper_exec_meta = dict(paper_exec_meta) if paper_exec_meta else None
        self.entry_calibration = dict(entry_calibration) if entry_calibration else None
        self.entry_v2_meta = dict(entry_v2_meta) if entry_v2_meta else None
        self.mfe_return = 0.0
        self.mae_return = 0.0
        self.bars_to_mfe = -1
        self.v2_extrema_initialized = False
        self.assert_consistent()

    def close_position(self) -> None:
        assert self.is_open
        self.is_open = False
        self.side = None
        self.entry_price = 0.0
        self.entry_bar_index = -1
        self.holding_bars = 0
        self.quantity = 1
        self.entry_time = ""
        self.paper_exec_meta = None
        self.entry_calibration = None
        self.entry_v2_meta = None
        self.mfe_return = 0.0
        self.mae_return = 0.0
        self.bars_to_mfe = -1
        self.v2_extrema_initialized = False
        self.assert_consistent()


def _new_parallel_positions() -> Dict[str, ParallelPosition]:
    """V1 và V2: hai instance độc lập, không dùng chung một object."""
    return {"v1": ParallelPosition(), "v2": ParallelPosition()}


@dataclass
class _MissedWatch:
    bar_index: int
    who_missed: str  # "v1" | "v2"
    other_entered: str  # "LONG" | "SHORT"
    entry_price: float
    bars_remaining: int


@dataclass
class ParallelRunner:
    """
    Gọi mỗi bar với ``data`` đầy đủ (cùng format :meth:`generate_ttm_signal_v1`).

    - ``data`` phải chứa ``bars`` (danh sách OHLC tích lũy tới bar hiện tại).
    - Tính feature một lần / bar; V1 và V2 nhận cùng dict feature (không tính lại trong signal nếu không có ``bars``).
    """

    config: Dict[str, Any]
    decision_log_path: Optional[str] = None
    trade_log_path: Optional[str] = None
    execution_near_bars: int = 5
    missed_forward_bars: int = 10
    missed_move_threshold: float = 2.0
    adaptive_context: Any = None
    empirical_engine: Any = None
    debug_terminal: bool = False
    execution_realism: Optional[ExecutionRealismConfig] = None
    execution_bar_seconds: int = 900

    _positions: Dict[str, ParallelPosition] = field(
        default_factory=_new_parallel_positions, init=False
    )

    _agree_bars: int = field(default=0, init=False)
    _total_bars: int = field(default=0, init=False)
    _exec_same_dir_events: int = field(default=0, init=False)
    _exec_checked: int = field(default=0, init=False)

    _blocked_v2: Counter = field(default_factory=Counter, init=False)
    _v2_trades_total: int = field(default=0, init=False)
    _v2_follow_design: int = field(default=0, init=False)

    _v1_trades: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _v2_trades: List[Dict[str, Any]] = field(default_factory=list, init=False)

    _missed_watches: deque = field(default_factory=deque, init=False)
    _missed_v2_count: int = field(default=0, init=False)
    _false_pos_v2: int = field(default=0, init=False)

    _last_logged_bar_ix: int = field(default=-1, init=False)
    _pending_entries: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {"v1": {}, "v2": {}}, init=False
    )

    _decision_fp: Optional[TextIO] = field(default=None, init=False)
    _trade_fp: Optional[TextIO] = field(default=None, init=False)

    # Per-bar close aligned to decision JSONL ``bar_index`` (for ttm_validation --closes).
    _session_closes: List[float] = field(default_factory=list, init=False)
    _last_bars: List[Any] = field(default_factory=list, init=False)
    _last_feat_row: Optional[Dict[str, Any]] = field(default=None, init=False)
    # Last bar index with upside breakout / early crowd context (for SHORT diagnostics persistence).
    _v2_last_upside_breakout_bar_index: Optional[int] = field(default=None, init=False)
    _v2_last_upside_breakout_score: Optional[float] = field(default=None, init=False)
    _v2_last_upside_breakout_phase: Optional[str] = field(default=None, init=False)
    _v2_last_upside_breakout_extension: Optional[float] = field(default=None, init=False)
    _research_gate_state: ResearchGateSessionState = field(
        default_factory=ResearchGateSessionState, init=False
    )

    def __post_init__(self) -> None:
        if self.decision_log_path:
            self._decision_fp = open(self.decision_log_path, "a", encoding="utf-8")
        if self.trade_log_path:
            self._trade_fp = open(self.trade_log_path, "a", encoding="utf-8")

    def _record_bar_close(self, bar_index: int, price: float) -> None:
        """Dense list index = bar_index; extend with NaN if needed (should be sequential in paper)."""
        bi = int(bar_index)
        while len(self._session_closes) <= bi:
            self._session_closes.append(float("nan"))
        self._session_closes[bi] = float(price)

    def _closes_export_path(self) -> Optional[Path]:
        """``ttm_parallel_decisions_<sym>_<stamp>.jsonl`` -> ``closes_<sym>_<stamp>.json`` (same folder)."""
        if not self.decision_log_path:
            return None
        p = Path(self.decision_log_path)
        name = p.name
        if not name.endswith(".jsonl"):
            return p.with_suffix(".closes.json")
        core = name[: -len(".jsonl")]
        if core.startswith("ttm_parallel_decisions_"):
            rest = core[len("ttm_parallel_decisions_") :]
            return p.parent / f"closes_{rest}.json"
        return p.parent / f"closes_{core}.json"

    def _export_closes_json(self) -> Tuple[Optional[str], int]:
        """Write ``closes`` JSON array; length == max(bar_index)+1 for this session."""
        out = self._closes_export_path()
        if out is None:
            return None, 0
        arr = list(self._session_closes)
        # Trim trailing NaNs only (keep leading valid prefix).
        while arr and (math.isnan(arr[-1]) or (isinstance(arr[-1], float) and np.isnan(arr[-1]))):
            arr.pop()
        if not arr:
            return None, 0
        if any(
            (isinstance(x, float) and (math.isnan(x) or np.isnan(x))) for x in arr
        ):
            logger.warning(
                "TTM parallel: closes series has gaps (NaN); validation may fail — check bar_index order",
                extra={"path": str(out), "len": len(arr), "decision_log": self.decision_log_path},
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps([float(x) for x in arr], ensure_ascii=False), encoding="utf-8")
        tmp.replace(out)
        n = len(arr)
        logger.info(
            "TTM parallel: exported closes for validation",
            extra={"path": str(out), "n_closes": n},
        )
        return str(out.resolve()), n

    def close(self) -> Dict[str, Any]:
        rep = self.summary()
        cpath, cn = self._export_closes_json()
        if cpath:
            rep["closes_export_path"] = cpath
            rep["closes_n"] = cn
        if self._total_bars == 0 and self.decision_log_path:
            logger.warning(
                "TTM parallel: đóng runner với total_bars=0 — không có dòng event_type=decision trong JSONL "
                "(ttm_validation cần ít nhất một nến OHLC: WS ohlc hoặc synth_trade). "
                "Kiểm tra Session status ws_ohlc_received / ohlc_synth_bars.",
                extra={
                    "decision_log_path": self.decision_log_path,
                    "total_bars": 0,
                },
            )
        if self._decision_fp:
            self._decision_fp.close()
            self._decision_fp = None
        if self._trade_fp:
            self._trade_fp.close()
            self._trade_fp = None
        return rep

    def _write_jsonl(self, fp: Optional[TextIO], record: Dict[str, Any]) -> None:
        if fp is None:
            return
        line = json.dumps(_json_safe(record), ensure_ascii=False)
        fp.write(line + "\n")
        fp.flush()

    @property
    def decision_bar_count(self) -> int:
        """Số lần :meth:`on_new_bar` đã chạy (mỗi lần ~ một dòng decisions JSONL)."""
        return self._total_bars

    def _position_state_snapshot(self) -> Dict[str, Any]:
        p1 = self._positions["v1"]
        p2 = self._positions["v2"]
        p1.assert_consistent()
        p2.assert_consistent()
        return {
            "v1_is_open": p1.is_open,
            "v1_side": p1.side,
            "v1_state": p1.state(),
            "v1_position_size": int(p1.quantity) if p1.is_open else 0,
            "v2_is_open": p2.is_open,
            "v2_side": p2.side,
            "v2_state": p2.state(),
            "v2_position_size": int(p2.quantity) if p2.is_open else 0,
        }

    def _debug_print_decision(
        self,
        bar_index: int,
        timestamp: str,
        n_feat: int,
        rec: Dict[str, Any],
    ) -> None:
        if not self.debug_terminal:
            return
        v1 = rec.get("v1") or {}
        v2 = rec.get("v2") or {}
        s1 = v1.get("signal", v1.get("reason_block", "?"))
        s2 = v2.get("decision", v2.get("blocked_by", "?"))
        print(
            f"[TTM parallel] #{self._total_bars} bar_ix={bar_index} n_feat={n_feat} "
            f"v1={s1} v2={s2} ts={timestamp}",
            flush=True,
        )

    @staticmethod
    def _truncate_feats(feats: Mapping[str, Any], bar_index: int) -> Dict[str, Any]:
        """Causal prefix of precomputed feature arrays (for O(n) backtest replay)."""
        n_full = int(feats.get("n", 0) or 0)
        n = min(n_full, int(bar_index) + 1)
        if n <= 0:
            return {"n": 0}
        out: Dict[str, Any] = {"n": n}
        for k, v in feats.items():
            if k == "n":
                continue
            if isinstance(v, np.ndarray) and v.size >= n:
                out[k] = v[:n]
            else:
                out[k] = v
        return out

    def _compute_feats(self, data: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raw = dict(data)
        if "bars" not in raw or raw.get("bars") is None:
            return {}, {}
        pc = raw.get("precomputed_feats")
        if isinstance(pc, Mapping):
            bi = raw.get("feat_bar_index")
            if bi is None:
                bi = max(0, int(pc.get("n", 0)) - 1)
            feats = self._truncate_feats(pc, int(bi))
            last = features_last_row(feats) if int(feats.get("n", 0)) > 0 else {}
            return feats, last
        feats = compute_ttm_features_from_config(raw, self.config)
        last = features_last_row(feats) if int(feats.get("n", 0)) > 0 else {}
        return feats, last

    def _update_v2_extrema(self, pos: ParallelPosition, price_ref: float) -> None:
        if not pos.is_open or price_ref <= 0:
            return
        ep = float(pos.entry_price)
        if ep <= 1e-12:
            return
        if pos.side == "LONG":
            u = (float(price_ref) - ep) / ep
        elif pos.side == "SHORT":
            u = (ep - float(price_ref)) / ep
        else:
            return
        if not pos.v2_extrema_initialized:
            pos.mfe_return = float(u)
            pos.mae_return = float(u)
            pos.bars_to_mfe = int(pos.holding_bars)
            pos.v2_extrema_initialized = True
            return
        if u > pos.mfe_return:
            pos.mfe_return = float(u)
            pos.bars_to_mfe = int(pos.holding_bars)
        if u < pos.mae_return:
            pos.mae_return = float(u)

    def _build_v2_closed_flat(
        self,
        pos: ParallelPosition,
        sig_v2: Mapping[str, Any],
        side: str,
    ) -> Dict[str, Any]:
        dbg = dict(sig_v2.get("debug") or {})
        ent = dict(pos.entry_v2_meta or {})
        v2sc = dbg.get("v2_score_components") if isinstance(dbg.get("v2_score_components"), dict) else {}
        v2sh = dbg.get("v2_short_components") if isinstance(dbg.get("v2_short_components"), dict) else {}
        raw_reason = str(sig_v2.get("reason") or "")
        ch = str(dbg.get("exit_channel") or "")
        if dbg.get("parallel_runner_override"):
            canon = map_parallel_runner_reason_to_canonical(side, raw_reason, ch)
        elif ch == "continuation":
            canon = raw_reason if raw_reason else "unknown"
        elif ch == "prob":
            canon = "score_decay" if str(side).upper() == "LONG" else "short_downside_decay"
        else:
            canon = map_parallel_runner_reason_to_canonical(side, raw_reason, ch)
        exit_crowd = v2sc.get("crowd_phase") or dbg.get("crowd_phase")
        exit_lbl = _refine_v2_trade_exit_reason(
            canon,
            entry_crowd_phase=ent.get("entry_crowd_phase"),
            exit_crowd_phase=exit_crowd,
            dbg_exit=dbg,
        )
        sig_sl = ent.get("signal_score_long")
        if sig_sl is None:
            sig_sl = ent.get("entry_score_long")
        return {
            "signal_bar_index": ent.get("signal_bar_index"),
            "signal_score_long": sig_sl,
            "signal_effective_strength": ent.get("signal_effective_strength"),
            "signal_crowd_phase": ent.get("signal_crowd_phase"),
            "signal_late_fomo_flag": ent.get("signal_late_fomo_flag"),
            "signal_entry_block_reason": ent.get("signal_entry_block_reason"),
            "signal_score_components": ent.get("signal_score_components"),
            "entry_confirm_score_long": ent.get("entry_confirm_score_long"),
            "entry_confirm_phase": ent.get("entry_confirm_phase"),
            "entry_confirm_breakout_valid": ent.get("entry_confirm_breakout_valid"),
            "entry_blocked": ent.get("entry_blocked"),
            "entry_block_reason": ent.get("entry_block_reason"),
            "exit_reason": exit_lbl,
            "entry_crowd_phase": ent.get("entry_crowd_phase"),
            "exit_crowd_phase": exit_crowd,
            "entry_score_long": ent.get("entry_score_long"),
            "exit_score_long": dbg.get("score_long"),
            "continuation_score_at_entry": ent.get("continuation_score_at_entry"),
            "continuation_score_at_exit": v2sc.get("continuation_confirm"),
            "short_entry_phase": ent.get("short_entry_phase"),
            "short_exit_phase": v2sh.get("short_phase") or dbg.get("short_phase"),
            "short_entry_score": ent.get("short_entry_score"),
            "short_exit_score": v2sh.get("short_score"),
            "short_entry_reason": ent.get("short_entry_reason"),
            "short_exit_reason": v2sh.get("short_reason"),
            "mfe": float(pos.mfe_return),
            "mae": float(pos.mae_return),
            "bars_to_mfe": int(pos.bars_to_mfe) if pos.bars_to_mfe >= 0 else None,
        }

    def on_new_bar(
        self,
        bar_index: int,
        timestamp: str,
        data: Mapping[str, Any],
        *,
        close_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Một bar mới: chạy V1 + V2, ghi log, cập nhật vị thế giả lập.

        ``close_price``: giá đóng để đánh giá missed opportunity (mặc định lấy từ bar cuối trong ``data['bars']``).
        """
        if bar_index == self._last_logged_bar_ix:
            return {
                "event_type": "skipped",
                "bar_index": bar_index,
                "reason": "duplicate_bar_index",
                "position_state": self._position_state_snapshot(),
            }
        self._last_logged_bar_ix = bar_index

        self._total_bars += 1
        self._last_bars = list(data.get("bars") or [])
        feats, last = self._compute_feats(data)
        feats_v2: Mapping[str, Any] = feats
        if isinstance(feats, dict) and last:
            feats_v2 = dict(feats)
            cp_anchor = str(classify_crowd_phase(last, self.config))
            bu_anchor = bool(last.get("breakout_up_filtered_last", last.get("breakout_up")))
            if bu_anchor or cp_anchor in ("ignition", "early_continuation"):
                self._v2_last_upside_breakout_bar_index = int(bar_index)
                self._v2_last_upside_breakout_phase = cp_anchor
                self._v2_last_upside_breakout_extension = _finite_or_none(last.get("extension"))
            max_short_ctx = max(1, int(self.config.get("ttm_v2_bars_since_breakout_max_for_short", 20)))
            if self._v2_last_upside_breakout_bar_index is not None:
                age = int(bar_index) - int(self._v2_last_upside_breakout_bar_index)
                if 0 <= age <= max_short_ctx:
                    feats_v2["ttm_v2_cross_bar_state"] = {
                        "ttm_v2_persist_last_upside_breakout_bar_index": int(
                            self._v2_last_upside_breakout_bar_index
                        ),
                        "ttm_v2_persist_last_upside_breakout_score": _json_safe(
                            self._v2_last_upside_breakout_score
                        ),
                        "ttm_v2_persist_last_upside_breakout_phase": _json_safe(
                            self._v2_last_upside_breakout_phase
                        ),
                        "ttm_v2_persist_last_upside_breakout_extension": _json_safe(
                            self._v2_last_upside_breakout_extension
                        ),
                    }
        n = int(feats.get("n", 0))

        if close_price is None and n > 0:
            bars = data.get("bars") or []
            if bars:
                close_price = float(getattr(bars[-1], "close", 0) or 0)

        price_ref = float(close_price or 0.0)
        self._record_bar_close(bar_index, price_ref)

        # Không có feature -> log tối thiểu, không thiếu key
        empty_features = {
            "breakout_strength": None,
            "failure_strength": None,
            "basis_norm": None,
            "oi_signal": None,
            "vol_regime": None,
        }

        if n <= 0:
            ef = {**empty_features, "oi_data": _oi_data_for_jsonl(data)}
            rec = {
                "event_type": "decision",
                "timestamp": timestamp,
                "bar_index": bar_index,
                "features": ef,
                "v1": {
                    "signal": "NONE",
                    "reason_block": "no_data",
                    "conditions": {"breakout": False, "volume": False, "filter_pass": False},
                },
                "v2": {
                    "score_long": None,
                    "score_short": None,
                    "prob_long": None,
                    "prob_short": None,
                    "decision": "NONE",
                    "components": {"momentum": None, "basis": None, "oi": None},
                    "blocked_by": "data",
                },
                "position_state": self._position_state_snapshot(),
                "forward_return": None,
                "forward_horizon_bars": max(1, int(self.config.get("ttm_parallel_forward_return_bars", 4))),
                "vol_z": None,
                "regime_tag": None,
                "ttm_regime_id": None,
            }
            self._write_jsonl(self._decision_fp, rec)
            self._debug_print_decision(bar_index, timestamp, n, rec)
            return rec

        pos_v1 = self._positions["v1"]
        pos_v2 = self._positions["v2"]
        side_v1 = pos_v1.effective_position_side()
        side_v2 = pos_v2.effective_position_side()
        self._last_feat_row = dict(last) if last else {}
        self._update_v2_extrema(pos_v2, price_ref)

        _holding_v2 = int(pos_v2.holding_bars) if pos_v2.is_open else None
        _u_ret_for_sig: Optional[float] = None
        _maxb_long = int(self.config.get("ttm_v2_time_stop_bars", 4))
        _sl_ret_long = float(self.config.get("ttm_v2_sl_return", -0.0007))
        _tp_ret_long = float(self.config.get("ttm_v2_tp_return", 0.0015))
        _maxb_short = int(self.config.get("ttm_v2_time_stop_bars_short", _maxb_long))
        _sl_ret_short = float(self.config.get("ttm_v2_sl_return_short", _sl_ret_long))
        _tp_ret_short = float(self.config.get("ttm_v2_tp_return_short", _tp_ret_long))
        _ent_ix = int(pos_v2.entry_bar_index) if pos_v2.is_open else -1
        _bars_since_entry = (bar_index - _ent_ix) if pos_v2.is_open and _ent_ix >= 0 else -1
        # Khớp live TTM: thoát SL/TP/max bars; prob-only V2 thường không đủ khi prob_long ở ~0.5–0.7.
        _forced_max = bool(
            pos_v2.is_open
            and (
                (pos_v2.side == "LONG" and _maxb_long > 0)
                or (pos_v2.side == "SHORT" and _maxb_short > 0)
            )
            and (
                (pos_v2.side == "LONG" and (pos_v2.holding_bars >= _maxb_long or _bars_since_entry >= _maxb_long))
                or (pos_v2.side == "SHORT" and (pos_v2.holding_bars >= _maxb_short or _bars_since_entry >= _maxb_short))
            )
        )
        _u_pnl = 0.0
        _u_ret = 0.0
        _forced_sl = False
        _forced_tp = False
        if pos_v2.is_open and price_ref > 0 and pos_v2.side == "LONG":
            _u_pnl = float(price_ref) - float(pos_v2.entry_price)
            if float(pos_v2.entry_price) > 0:
                _u_ret = _u_pnl / float(pos_v2.entry_price)
            _u_ret_for_sig = float(_u_ret)
            _forced_sl = _u_ret <= _sl_ret_long
            _forced_tp = _u_ret >= _tp_ret_long
        elif pos_v2.is_open and price_ref > 0 and pos_v2.side == "SHORT":
            _u_pnl = float(pos_v2.entry_price) - float(price_ref)
            if float(pos_v2.entry_price) > 0:
                _u_ret = _u_pnl / float(pos_v2.entry_price)
            _u_ret_for_sig = float(_u_ret)
            _forced_sl = _u_ret <= _sl_ret_short
            _forced_tp = _u_ret >= _tp_ret_short

        _v2_exit_reason = ""
        _v2_exit_channel = ""
        _short_exit = None
        if pos_v2.is_open and pos_v2.side == "SHORT":
            _hb_short = max(int(pos_v2.holding_bars), int(_bars_since_entry))
            _short_exit = check_exhaustion_short_exit(
                pos_v2.paper_exec_meta,
                price_ref,
                _hb_short,
            )
            if _short_exit is None:
                _last_for_short = features_last_row(feats)
                _short_exit = check_short_v3_structural_exit(
                    pos_v2.paper_exec_meta,
                    price_ref,
                    _hb_short,
                    _last_for_short,
                    self.config,
                )
        if _short_exit is not None:
            _v2_exit_reason = str(_short_exit["reason"])
            _tr = str(_short_exit["trigger"]).lower()
            if _tr in ("sl", "tp", "time"):
                _v2_exit_channel = f"short_exhaustion_{_tr}"
            else:
                _v2_exit_channel = f"short_v3_{_tr}"
        elif _forced_sl:
            _v2_exit_reason = "ttm_exit_stop_loss_parallel"
            _v2_exit_channel = "stop_loss_return"
        elif _forced_tp:
            _v2_exit_reason = "ttm_exit_take_profit_parallel"
            _v2_exit_channel = "take_profit_return"
        elif _forced_max:
            _v2_exit_reason = "ttm_exit_max_bars_parallel"
            _v2_exit_channel = "max_bars_in_trade"

        sig_v1 = generate_ttm_signal_v1(
            feats, self.config, position_side=side_v1, live_mode=False
        )
        _ts_int: Optional[int] = None
        try:
            _ts_int = int(timestamp) if str(timestamp).isdigit() else None
        except (TypeError, ValueError):
            _ts_int = None
        _live_mode = str(self.config.get("ttm_config_profile", "") or "") == "live_adaptive"
        _session_date = None
        try:
            if str(timestamp).isdigit():
                from datetime import datetime, timezone, timedelta

                dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc) + timedelta(hours=7)
                _session_date = dt.strftime("%Y%m%d")
        except (TypeError, ValueError, OSError):
            _session_date = None
        sig_v2 = generate_ttm_signal_v2(
            feats_v2,
            self.config,
            position_side=side_v2,
            live_mode=_live_mode,
            adaptive=self.adaptive_context,
            empirical_engine=self.empirical_engine,
            bar_timestamp=_ts_int,
            holding_bars=_holding_v2,
            position_unrealized_return=_u_ret_for_sig,
            entry_snapshot=pos_v2.entry_v2_meta if pos_v2.is_open else None,
            position_meta=pos_v2.paper_exec_meta if pos_v2.is_open else None,
            current_price=float(price_ref) if price_ref > 0 else None,
            research_gate_state=self._research_gate_state,
            session_date=_session_date,
        )

        _dbg2_pre = sig_v2.get("debug") or {}
        _pl0 = _dbg2_pre.get("prob_long")
        _ps0 = _dbg2_pre.get("prob_short")

        if _v2_exit_reason and str(sig_v2.get("action", "HOLD")).upper() != "EXIT":
            sig_v2 = {
                **dict(sig_v2),
                "action": "EXIT",
                "reason": _v2_exit_reason,
                "confidence": float(_pl0 or _ps0 or 0.0),
            }
            _nd = dict(sig_v2.get("debug") or _dbg2_pre)
            _nd["exit_channel"] = _v2_exit_channel
            _nd["parallel_runner_override"] = True
            _nd["bars_since_entry"] = _bars_since_entry
            _nd["unrealized_pnl_points"] = _u_pnl
            _nd["unrealized_pnl_return"] = _u_ret
            sig_v2["debug"] = _nd

        _dbgx = sig_v2.get("debug") or {}
        _scp = str(_dbgx.get("crowd_phase") or "")
        if last and (
            bool(last.get("breakout_up_raw") or last.get("breakout_up"))
            or _scp in ("ignition", "early_continuation")
        ):
            self._v2_last_upside_breakout_bar_index = int(bar_index)
            self._v2_last_upside_breakout_score = _finite_or_none(_dbgx.get("score_long"))
            self._v2_last_upside_breakout_phase = _scp or None
            self._v2_last_upside_breakout_extension = _finite_or_none(_dbgx.get("extension"))
        elif self._v2_last_upside_breakout_bar_index is not None:
            self._v2_last_upside_breakout_score = _finite_or_none(
                _dbgx.get("score_long") or self._v2_last_upside_breakout_score
            )

        last_snap = _merge_features(last, data, feats=feats) if last else empty_features
        v1_struct = _build_v1_block(sig_v1, self.config, last)
        v2_struct = _build_v2_block(
            sig_v2,
            self.config,
            position_side=side_v2,
            execution_position_open=pos_v2.is_open,
        )
        fp_bar = fingerprint_aligned_tail(dict(data))
        log_v2 = v2_struct.get("log") if isinstance(v2_struct.get("log"), dict) else {}
        if fp_bar and not log_v2.get("input_aligned_fingerprint"):
            log_v2 = {**log_v2, "input_aligned_fingerprint": fp_bar}
            v2_struct = {**v2_struct, "log": log_v2}

        raw_a1 = str(sig_v1.get("action", "HOLD")).upper()
        raw_a2 = str(sig_v2.get("action", "HOLD")).upper()
        if raw_a1 == raw_a2:
            self._agree_bars += 1

        self._blocked_v2[str(v2_struct["blocked_by"])] += 1

        decision_record = {
            "event_type": "decision",
            "timestamp": timestamp,
            "bar_index": bar_index,
            "features": last_snap,
            "v1": v1_struct,
            "v2": v2_struct,
            "position_state": self._position_state_snapshot(),
        }
        fw_h = max(1, int(self.config.get("ttm_parallel_forward_return_bars", 4)))
        fr = _forward_return_from_closes(self._session_closes, bar_index, fw_h)
        decision_record["forward_return"] = _json_safe(fr)
        decision_record["forward_horizon_bars"] = fw_h
        vz_raw = None
        if last:
            vz_raw = last.get("vol_zscore", last.get("vol_z"))
        try:
            vz_f = float(vz_raw) if vz_raw is not None else float("nan")
            decision_record["vol_z"] = _json_safe(vz_f) if np.isfinite(vz_f) else None
        except (TypeError, ValueError):
            decision_record["vol_z"] = None
        rid = int(detect_regime(last or {}, self.config)) if last else 0
        decision_record["ttm_regime_id"] = rid
        decision_record["regime_tag"] = _regime_id_to_tag(rid)
        self._write_jsonl(self._decision_fp, decision_record)
        self._debug_print_decision(bar_index, timestamp, n, decision_record)

        self._simulate_pair(
            bar_index,
            timestamp,
            price_ref,
            sig_v1,
            sig_v2,
            data,
            feats_v2,
            dict(last) if last else {},
        )
        self._process_missed_watches(bar_index, timestamp, price_ref)
        self._check_missed_opportunities(bar_index, timestamp, raw_a1, raw_a2, price_ref)

        return decision_record

    def _emit_trade(
        self,
        model: str,
        event: str,
        timestamp: str,
        price: float,
        side: str,
        pnl: Optional[float],
        bar_index: int,
        *,
        entry_time: str = "",
        exit_time: str = "",
        holding_period: Optional[int] = None,
        entry_bar_index: Optional[int] = None,
        entry_price: Optional[float] = None,
        position_size: Optional[int] = None,
        realized_return: Optional[float] = None,
        signal_price: Optional[float] = None,
        execution_audit: Optional[Dict[str, Any]] = None,
        entry_calibration: Optional[Dict[str, Any]] = None,
        v2_closed_flat: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        One JSONL row per round-trip: skip ENTRY; on EXIT emit ``event: CLOSED`` with entry+exit fields.
        Legacy ENTRY/EXIT pairs are no longer written (validation accepts both styles).
        """
        if event == "ENTRY":
            return
        if event == "EXIT":
            row: Dict[str, Any] = {
                "event_type": "trade",
                "model": model,
                "event": "CLOSED",
                "timestamp": timestamp,
                "bar_index": bar_index,
                "exit_bar_index": int(bar_index),
                "entry_bar_index": int(entry_bar_index) if entry_bar_index is not None else None,
                "entry_price": _json_safe(entry_price),
                "exit_price": _json_safe(price),
                "side": side,
                "position_size": _json_safe(position_size),
                "pnl": _json_safe(pnl),
                "net_pnl": _json_safe(pnl),
                "realized_return": _json_safe(realized_return),
                "entry_time": entry_time or None,
                "exit_time": exit_time or None,
                "holding_period": _json_safe(holding_period),
            }
            if signal_price is not None:
                row["signal_exit_price"] = _json_safe(signal_price)
            if execution_audit:
                row["execution_audit"] = _json_safe(execution_audit)
            if entry_calibration:
                row["entry_calibration"] = _json_safe(entry_calibration)
            if str(model).lower() == "v2":
                v2f = dict(_V2_CLOSED_TRADE_FLAT_DEFAULTS)
                if holding_period is not None:
                    v2f["holding_bars"] = int(holding_period)
                if v2_closed_flat:
                    for k, v in v2_closed_flat.items():
                        if k in v2f:
                            v2f[k] = v
                if v2f.get("signal_bar_index") is None and entry_bar_index is not None:
                    v2f["signal_bar_index"] = int(entry_bar_index) - 1
                row.update({k: _json_safe(v) for k, v in v2f.items()})
            self._write_jsonl(self._trade_fp, row)
            return
        self._write_jsonl(
            self._trade_fp,
            {
                "event_type": "trade",
                "model": model,
                "event": event,
                "timestamp": timestamp,
                "bar_index": bar_index,
                "price": _json_safe(price),
                "side": side,
                "pnl": _json_safe(pnl),
                "entry_time": entry_time or None,
                "exit_time": exit_time or None,
                "holding_period": _json_safe(holding_period),
            },
        )

    def _emit_v2_entry_blocked(
        self,
        bar_index: int,
        timestamp: str,
        reason: str,
        fields: Mapping[str, Any],
        *,
        signal_bar_index: Optional[int],
    ) -> None:
        if not self._trade_fp:
            return
        row = {
            "event_type": "ENTRY_BLOCKED",
            "model": "v2",
            "timestamp": str(timestamp),
            "bar_index": int(bar_index),
            "reason": str(reason),
            "signal_bar_index": _json_safe(signal_bar_index),
            **_json_safe(dict(fields)),
        }
        self._write_jsonl(self._trade_fp, row)

    def _simulate_pair(
        self,
        bar_index: int,
        timestamp: str,
        price: float,
        sig_v1: Mapping[str, Any],
        sig_v2: Mapping[str, Any],
        data: Mapping[str, Any],
        feats: Mapping[str, Any],
        last_row: Mapping[str, Any],
    ) -> None:
        _live_mode = str(self.config.get("ttm_config_profile", "") or "") == "live_adaptive"
        a1 = str(sig_v1.get("action", "HOLD")).upper()
        a2 = str(sig_v2.get("action", "HOLD")).upper()
        bars = self._last_bars
        bs = int(self.execution_bar_seconds)

        if self._pending_entries["v1"].get("ready_on_bar") == bar_index and not self._positions["v1"].is_open:
            pe1 = dict(self._pending_entries["v1"])
            self._positions["v1"].open_position(
                str(pe1.get("side", "LONG")),
                float(pe1.get("entry_price", price)),
                bar_index,
                quantity=int(pe1.get("quantity", 1)),
                entry_time=str(timestamp),
            )
            self._pending_entries["v1"] = {}

        # V2: legacy next-bar fill vs execution realism (latency + slippage)
        pos2 = self._positions["v2"]
        pe2 = dict(self._pending_entries["v2"])
        if self.execution_realism is not None and pe2.get("due_unix") is not None and bars and not pos2.is_open:
            due_u = float(pe2["due_unix"])
            b0 = bars[bar_index]
            if float(getattr(b0, "unix_ts", 0)) + 1e-9 >= due_u:
                side_pe = str(pe2.get("side", "LONG"))
                base_open = float(getattr(b0, "open", 0) or 0)
                if base_open <= 0:
                    base_open = float(price)
                cfg_e = self.execution_realism
                slip_e = slippage_abs_at_bar(
                    bars,
                    bar_index,
                    cfg_e.slippage_mode,
                    worst_case_multiplier=cfg_e.worst_case_slippage_multiplier,
                )
                entry_px = adjust_fill_price(side_pe, is_entry=True, base_price=base_open, slippage_abs=slip_e)
                sig_px = float(pe2.get("signal_close_price", price))
                exec_ts = str(int(getattr(b0, "unix_ts", 0)))
                meta = {
                    "signal_timestamp": str(pe2.get("signal_timestamp", "")),
                    "execution_timestamp": exec_ts,
                    "signal_price": sig_px,
                    "execution_price": float(entry_px),
                    "latency_ms": int(cfg_e.latency_ms),
                    "slippage_mode": str(cfg_e.slippage_mode),
                    "slippage_abs_entry": float(slip_e),
                    "entry_bar_signal": int(pe2.get("signal_bar_index", -1)),
                    "entry_bar_execution": int(bar_index),
                }
                feat_sig = pe2.get("features") if isinstance(pe2.get("features"), dict) else {}
                short_meta = build_exhaustion_short_entry_meta(feat_sig, entry_px, self.config)
                if short_meta is None:
                    short_meta = build_short_opportunity_entry_meta(feat_sig, entry_px, self.config)
                if short_meta:
                    meta.update(short_meta)
                psig = pe2.get("pending_sig_v2")
                if not isinstance(psig, dict):
                    psig = dict(sig_v2)
                snap_meta = _build_v2_entry_snapshot(
                    psig,
                    str(side_pe),
                    self.config,
                    signal_bar_snapshot=pe2.get("signal_bar_snapshot"),
                    signal_bar_index=(
                        int(pe2["signal_bar_index"]) if pe2.get("signal_bar_index") is not None else None
                    ),
                )
                blocked_open = False
                blk_reason = ""
                if str(side_pe).upper() == "LONG":
                    _snap = pe2.get("signal_bar_snapshot") if isinstance(pe2.get("signal_bar_snapshot"), dict) else {}
                    _sc = _snap.get("signal_score_components") if isinstance(_snap.get("signal_score_components"), dict) else {}
                    _gd = _snap.get("gate_diagnostics") if isinstance(_snap.get("gate_diagnostics"), dict) else {}
                    if not _gd and isinstance(psig.get("debug"), dict):
                        _gd = (psig.get("debug") or {}).get("gate_diagnostics") or {}
                    _ecm = entry_confirm_mode_for_gate(resolve_gate_mode(self.config, live_mode=_live_mode))
                    blocked_open, blk_reason, cfld = _v2_entry_execution_long_confirm(
                        feats,
                        last_row,
                        self.config,
                        signal_only_exec=bool(
                            self.config.get("ttm_v2_allow_signal_only_long_execution", False)
                        ),
                        signal_crowd_phase=str(_sc.get("crowd_phase") or _snap.get("signal_crowd_phase") or ""),
                        signal_long_candidate=bool(_gd.get("long_candidate")),
                        entry_confirm_mode=_ecm,
                        signal_bar_index=pe2.get("signal_bar_index"),
                    )
                    snap_meta.update(cfld)
                    snap_meta["entry_blocked"] = bool(blocked_open)
                    snap_meta["entry_block_reason"] = blk_reason if blocked_open else None
                if blocked_open:
                    self._emit_v2_entry_blocked(
                        bar_index,
                        exec_ts,
                        blk_reason,
                        snap_meta,
                        signal_bar_index=pe2.get("signal_bar_index"),
                    )
                    self._pending_entries["v2"] = {}
                else:
                    pos2.open_position(
                        side_pe,
                        entry_px,
                        bar_index,
                        quantity=int(pe2.get("quantity", 1)),
                        entry_time=exec_ts,
                        paper_exec_meta=meta,
                        entry_calibration=(
                            pe2.get("entry_calibration")
                            if str(side_pe).upper() == "LONG"
                            else None
                        ),
                        entry_v2_meta=snap_meta,
                    )
                    record_research_trade_if_applicable(
                        cfg=self.config,
                        side=str(side_pe),
                        bar_index=int(bar_index),
                        session_state=self._research_gate_state,
                        live_mode=_live_mode,
                    )
                    self._record_v2_follow_design(sig_v2, side_pe)
                    self._pending_entries["v2"] = {}
        elif self._pending_entries["v2"].get("ready_on_bar") == bar_index and not pos2.is_open:
            pe2 = dict(self._pending_entries["v2"])
            base_open = float(getattr(bars[bar_index], "open", 0) or 0) if bars and bar_index < len(bars) else 0.0
            entry_px = float(base_open if base_open > 0 else pe2.get("entry_price", price))
            paper_exec_meta = None
            feat_sig = pe2.get("features") if isinstance(pe2.get("features"), dict) else {}
            short_meta = build_exhaustion_short_entry_meta(feat_sig, entry_px, self.config)
            if short_meta is None:
                short_meta = build_short_opportunity_entry_meta(feat_sig, entry_px, self.config)
            if short_meta:
                paper_exec_meta = dict(short_meta)
            side_pe2 = str(pe2.get("side", "LONG"))
            psig2 = pe2.get("pending_sig_v2")
            if not isinstance(psig2, dict):
                psig2 = dict(sig_v2)
            snap_meta2 = _build_v2_entry_snapshot(
                psig2,
                side_pe2,
                self.config,
                signal_bar_snapshot=pe2.get("signal_bar_snapshot"),
                signal_bar_index=(
                    int(pe2["signal_bar_index"]) if pe2.get("signal_bar_index") is not None else None
                ),
            )
            blocked2 = False
            blk2 = ""
            if side_pe2.upper() == "LONG":
                _snap2 = pe2.get("signal_bar_snapshot") if isinstance(pe2.get("signal_bar_snapshot"), dict) else {}
                _sc2 = _snap2.get("signal_score_components") if isinstance(_snap2.get("signal_score_components"), dict) else {}
                _gd2 = _snap2.get("gate_diagnostics") if isinstance(_snap2.get("gate_diagnostics"), dict) else {}
                if not _gd2 and isinstance(psig2.get("debug"), dict):
                    _gd2 = (psig2.get("debug") or {}).get("gate_diagnostics") or {}
                _ecm2 = entry_confirm_mode_for_gate(resolve_gate_mode(self.config, live_mode=_live_mode))
                blocked2, blk2, cf2 = _v2_entry_execution_long_confirm(
                    feats,
                    last_row,
                    self.config,
                    signal_only_exec=bool(
                        self.config.get("ttm_v2_allow_signal_only_long_execution", False)
                    ),
                    signal_crowd_phase=str(_sc2.get("crowd_phase") or _snap2.get("signal_crowd_phase") or ""),
                    signal_long_candidate=bool(_gd2.get("long_candidate")),
                    entry_confirm_mode=_ecm2,
                    signal_bar_index=pe2.get("signal_bar_index"),
                )
                snap_meta2.update(cf2)
                snap_meta2["entry_blocked"] = bool(blocked2)
                snap_meta2["entry_block_reason"] = blk2 if blocked2 else None
            if blocked2:
                self._emit_v2_entry_blocked(
                    bar_index,
                    str(timestamp),
                    blk2,
                    snap_meta2,
                    signal_bar_index=pe2.get("signal_bar_index"),
                )
                self._pending_entries["v2"] = {}
            else:
                pos2.open_position(
                    side_pe2,
                    entry_px,
                    bar_index,
                    quantity=int(pe2.get("quantity", 1)),
                    entry_time=str(timestamp),
                    paper_exec_meta=paper_exec_meta,
                    entry_calibration=(
                        pe2.get("entry_calibration")
                        if side_pe2.upper() == "LONG"
                        else None
                    ),
                    entry_v2_meta=snap_meta2,
                )
                record_research_trade_if_applicable(
                    cfg=self.config,
                    side=str(side_pe2),
                    bar_index=int(bar_index),
                    session_state=self._research_gate_state,
                    live_mode=_live_mode,
                )
                self._record_v2_follow_design(sig_v2, side_pe2)
                self._pending_entries["v2"] = {}

        for model, a in (("v1", a1), ("v2", a2)):
            pos = self._positions[model]
            if a == "EXIT" and pos.is_open:
                ep = float(pos.entry_price)
                side = pos.side or "LONG"
                exit_signal_px = float(price)
                exit_exec_px = exit_signal_px
                slip_x = 0.0
                sig_for_emit: Optional[float] = None
                audit: Optional[Dict[str, Any]] = None
                if model == "v2" and self.execution_realism is not None and bars:
                    cfg_e = self.execution_realism
                    slip_x = slippage_abs_at_bar(
                        bars,
                        bar_index,
                        cfg_e.slippage_mode,
                        worst_case_multiplier=cfg_e.worst_case_slippage_multiplier,
                    )
                    exit_exec_px = adjust_fill_price(
                        str(side), is_entry=False, base_price=exit_signal_px, slippage_abs=slip_x
                    )
                    sig_for_emit = exit_signal_px
                    ent = dict(pos.paper_exec_meta or {})
                    audit = {
                        **ent,
                        "signal_exit_price": exit_signal_px,
                        "execution_exit_price": float(exit_exec_px),
                        "slippage_abs_exit": float(slip_x),
                        "exit_bar_index": int(bar_index),
                        "exit_signal_timestamp": str(timestamp),
                        "exit_execution_timestamp": str(timestamp),
                        "exit_logic_note": (
                            "SL/TP/max-bars use bar close (signal path); fill price applies slippage only."
                        ),
                    }
                if side == "LONG":
                    pnl = (exit_exec_px - ep) * float(pos.quantity)
                else:
                    pnl = (ep - exit_exec_px) * float(pos.quantity)
                if model == "v2" and pnl < 0:
                    self._false_pos_v2 += 1
                qty = int(pos.quantity)
                rr: Optional[float] = None
                if ep > 1e-12:
                    if side == "LONG":
                        rr = (float(exit_exec_px) - ep) / ep
                    else:
                        rr = (ep - float(exit_exec_px)) / ep
                v2_flat = None
                if model == "v2":
                    v2_flat = self._build_v2_closed_flat(pos, sig_v2, str(side))
                self._emit_trade(
                    model,
                    "EXIT",
                    timestamp,
                    float(exit_exec_px),
                    side,
                    pnl,
                    bar_index,
                    entry_time=str(pos.entry_time),
                    exit_time=str(timestamp),
                    holding_period=int(pos.holding_bars),
                    entry_bar_index=int(pos.entry_bar_index),
                    entry_price=ep,
                    position_size=qty,
                    realized_return=rr,
                    signal_price=sig_for_emit,
                    execution_audit=audit,
                    entry_calibration=(
                        pos.entry_calibration
                        if model == "v2" and str(side).upper() == "LONG"
                        else None
                    ),
                    v2_closed_flat=v2_flat,
                )
                if model == "v1":
                    self._v1_trades.append({"pnl": pnl, "side": side})
                else:
                    self._v2_trades.append({"pnl": pnl, "side": side})
                pos.close_position()
            elif a in ("LONG", "SHORT") and not pos.is_open:
                qty = 1
                if model == "v2":
                    feat = sig_v2.get("features") if isinstance(sig_v2.get("features"), dict) else {}
                    qty = 1
                if model == "v2" and self.execution_realism is not None and bars:
                    sb = bars[bar_index]
                    cfg_e = self.execution_realism
                    due_u = entry_due_unix(sb, int(cfg_e.latency_ms), bs)
                    self._pending_entries["v2"] = {
                        "side": a,
                        "due_unix": float(due_u),
                        "quantity": qty,
                        "signal_bar_index": int(bar_index),
                        "signal_close_price": float(price),
                        "signal_timestamp": str(timestamp),
                        "features": feat,
                        "pending_sig_v2": dict(sig_v2),
                        "signal_bar_snapshot": _v2_signal_bar_snapshot(sig_v2, feat),
                        "entry_calibration": (
                            _v2_long_entry_calibration_from_features(feat) if a == "LONG" else None
                        ),
                    }
                else:
                    self._pending_entries[model] = {
                        "side": a,
                        "entry_price": float(price),
                        "ready_on_bar": int(bar_index) + 1,
                        "quantity": qty,
                        "features": feat if model == "v2" else {},
                        **(
                            {
                                "entry_calibration": _v2_long_entry_calibration_from_features(feat)
                                if a == "LONG"
                                else None,
                                "signal_bar_index": int(bar_index),
                                "pending_sig_v2": dict(sig_v2),
                                "signal_bar_snapshot": _v2_signal_bar_snapshot(sig_v2, feat),
                            }
                            if model == "v2"
                            else {}
                        ),
                    }
            elif pos.is_open:
                if (pos.side == "LONG" and a == "SHORT") or (pos.side == "SHORT" and a == "LONG"):
                    logger.info(
                        "Execution state machine: ignore direct reversal",
                        extra={"model": model, "state": pos.state(), "signal": a, "bar_index": bar_index},
                    )
                pos.holding_bars += 1
                if pos.holding_bars > 10:
                    print(
                        "WARNING: POSITION STUCK",
                        {"model": model, "bars": pos.holding_bars, "side": pos.side},
                        flush=True,
                    )
                    logger.warning(
                        "WARNING: POSITION STUCK (open > 10 bars)",
                        extra={
                            "model": model,
                            "holding_bars": pos.holding_bars,
                            "side": pos.side,
                            "bar_index": bar_index,
                        },
                    )
            pos.assert_consistent()

        # Execution agreement: cùng bar vào cùng hướng
        if a1 in ("LONG", "SHORT") and a2 in ("LONG", "SHORT") and a1 == a2:
            self._exec_same_dir_events += 1
        self._exec_checked += 1

    def _record_v2_follow_design(self, sig_v2: Mapping[str, Any], direction: str) -> None:
        """Đánh giá 1 lần mỗi ENTRY V2: prob vs ngưỡng, regime, momentum cùng hướng."""
        dbg2 = sig_v2.get("debug") or {}
        pl = float(dbg2.get("prob_long") or 0.0)
        ps = float(dbg2.get("prob_short") or 0.0)
        pe = float(
            self.config.get("entry_threshold", self.config.get("prob_entry_threshold", 0.6))
        )
        rid = dbg2.get("ttm_regime_id")
        comp = dbg2.get("score_components") or {}
        mom = float(comp.get("momentum") or 0.0)
        if direction == "LONG":
            ok_prob = pl >= pe
            align = mom >= 0.0
        else:
            ok_prob = ps >= pe
            align = mom <= 0.0
        regime_ok = rid is not None
        valid_design = bool(ok_prob and align and regime_ok)
        self._v2_trades_total += 1
        if valid_design:
            self._v2_follow_design += 1

    def _check_missed_opportunities(
        self,
        bar_index: int,
        timestamp: str,
        raw_a1: str,
        raw_a2: str,
        price: float,
    ) -> None:
        """V1 vào mà V2 không (hoặc ngược lại) -> theo dõi forward MFE."""
        if raw_a1 in ("LONG", "SHORT") and raw_a2 == "HOLD":
            self._missed_watches.append(
                _MissedWatch(
                    bar_index=bar_index,
                    who_missed="v2",
                    other_entered=raw_a1,
                    entry_price=price,
                    bars_remaining=self.missed_forward_bars,
                )
            )
        if raw_a2 in ("LONG", "SHORT") and raw_a1 == "HOLD":
            self._missed_watches.append(
                _MissedWatch(
                    bar_index=bar_index,
                    who_missed="v1",
                    other_entered=raw_a2,
                    entry_price=price,
                    bars_remaining=self.missed_forward_bars,
                )
            )

    def _process_missed_watches(self, bar_index: int, timestamp: str, price: float) -> None:
        remaining: List[_MissedWatch] = []
        for w in self._missed_watches:
            if w.other_entered == "LONG":
                mfe = price - w.entry_price
            else:
                mfe = w.entry_price - price
            w.bars_remaining -= 1
            if mfe >= self.missed_move_threshold:
                exp_pnl = float(mfe)
                self._missed_v2_count += 1 if w.who_missed == "v2" else 0
                self._write_jsonl(
                    self._trade_fp,
                    {
                        "event_type": "MISSED_TRADE",
                        "timestamp": timestamp,
                        "bar_index": bar_index,
                        "type": "MISSED_TRADE",
                        "who_missed": w.who_missed,
                        "other_side": w.other_entered,
                        "entry_reference_bar": w.bar_index,
                        "expected_pnl": exp_pnl,
                    },
                )
                continue
            if w.bars_remaining > 0:
                remaining.append(w)
        self._missed_watches = deque(remaining)

    def summary(self) -> Dict[str, Any]:
        def _stats(trades: List[Dict[str, Any]]) -> Tuple[int, float, float]:
            if not trades:
                return 0, 0.0, 0.0
            pnls = [float(t["pnl"]) for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            return len(trades), wins / max(len(trades), 1), float(sum(pnls))

        n1, wr1, pnl1 = _stats(self._v1_trades)
        n2, wr2, pnl2 = _stats(self._v2_trades)

        agreement_rate = self._agree_bars / max(self._total_bars, 1)
        exec_agree = self._exec_same_dir_events / max(self._exec_checked, 1)
        follow_v2 = self._v2_follow_design / max(self._v2_trades_total, 1)

        top_block = "none"
        if self._blocked_v2:
            top_block = self._blocked_v2.most_common(1)[0][0]

        rep: Dict[str, Any] = {
            "event_type": "summary",
            "total_bars": self._total_bars,
            "v1": {
                "trades": n1,
                "winrate": wr1,
                "pnl": pnl1,
            },
            "v2": {
                "trades": n2,
                "winrate": wr2,
                "pnl": pnl2,
            },
            "agreement_rate": agreement_rate,
            "execution_same_direction_rate": exec_agree,
            "missed_trades_logged": self._missed_v2_count,
            "missed_trades_v2": self._missed_v2_count,
            "false_positive_v2_exits_loss": self._false_pos_v2,
            "top_block_reason_v2": top_block,
            "blocked_by_reason_count_v2": dict(self._blocked_v2),
            "follow_design_rate_v2": follow_v2,
        }
        if self.execution_realism is not None:
            er = self.execution_realism
            rep["execution_realism"] = {
                "latency_ms": int(er.latency_ms),
                "slippage_mode": str(er.slippage_mode),
                "bar_seconds": int(self.execution_bar_seconds),
            }
        self._write_jsonl(self._decision_fp, rep)
        return rep


def replay_bars(
    data_series: List[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]] = None,
    *,
    timestamps: Optional[List[str]] = None,
    decision_log_path: Optional[str] = None,
    trade_log_path: Optional[str] = None,
    execution_realism: Optional[ExecutionRealismConfig] = None,
    execution_bar_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Replay danh sách ``data`` đã tích lũy từng bước (mỗi phần tử là ``{"bars": bars[:i+1], ...}``).

    Trả về summary sau khi đóng runner.

    ``execution_realism``: optional V2-only execution layer (latency + slippage); ``None`` preserves legacy fills.
    ``execution_bar_seconds``: bar length in seconds for signal/execution time alignment (default 900).
    """
    cfg = dict(TTM_CONFIG if config is None else config)
    bar_sec = int(execution_bar_seconds) if execution_bar_seconds is not None else int(
        cfg.get("ttm_execution_bar_seconds", 900)
    )
    empirical_engine = None
    if bool(cfg.get("ttm_v2_empirical_alpha_enabled", False)):
        empirical_engine = EmpiricalAlphaEngine(cfg)
    runner = ParallelRunner(
        cfg,
        decision_log_path=decision_log_path,
        trade_log_path=trade_log_path,
        execution_realism=execution_realism,
        execution_bar_seconds=bar_sec,
        empirical_engine=empirical_engine,
    )
    for i, d in enumerate(data_series):
        ts = (timestamps[i] if timestamps and i < len(timestamps) else str(i))
        runner.on_new_bar(i, ts, d)
    return runner.close()


class TTMDerivativesParallelStrategy(TTMDerivativesStrategy):
    """
    Giống :class:`~src.strategies.ttm.ttm_strategy.TTMDerivativesStrategy` nhưng sau mỗi nến
    gọi :class:`ParallelRunner` (log song song V1 + V2). Dùng với ``scripts/paper_test.py``.
    """

    def __init__(
        self,
        name: str = "ttm_deriv",
        symbols: Optional[List[str]] = None,
        settings: Any = None,
        ttm_config: Optional[Dict[str, Any]] = None,
        *,
        parallel_runner: ParallelRunner,
    ) -> None:
        super().__init__(name=name, symbols=symbols, settings=settings, ttm_config=ttm_config)
        self._parallel_runner = parallel_runner

    def handle_ohlc(self, ohlc) -> None:
        super().handle_ohlc(ohlc)
        sym = (getattr(ohlc, "symbol", None) or "").strip().upper()
        raw_sym = getattr(ohlc, "symbol", None)
        if not sym:
            return
        if sym not in self.symbols:
            if not getattr(self, "_warned_ohlc_symbol_mismatch", False):
                self._warned_ohlc_symbol_mismatch = True
                logger.warning(
                    "OHLC symbol không khớp strategy.symbols — parallel JSONL sẽ trống",
                    extra={
                        "ohlc_symbol": raw_sym,
                        "strategy_symbols": list(self.symbols),
                    },
                )
            return
        if not getattr(self, "_logged_first_parallel_ohlc", False):
            self._logged_first_parallel_ohlc = True
            logger.info(
                "TTM parallel: OHLC khớp symbol, ghi decisions JSONL mỗi nến (on_new_bar)",
                extra={
                    "symbol": sym,
                    "bars_in_history": len(self._ohlc_history),
                    "decision_log": getattr(
                        self._parallel_runner, "decision_log_path", None
                    ),
                },
            )
        st = self._build_state(sym)
        ts = str(getattr(ohlc, "time", "") or "")
        bar_ix = max(0, len(self._ohlc_history) - 1)
        self._parallel_runner.on_new_bar(bar_ix, ts, st)
