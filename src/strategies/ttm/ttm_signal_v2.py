"""TTM V2 — unified bounded score + softmax; default prob-only exits, optional continuation-aware ordering."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from src.logger import get_logger
from src.strategies.ttm.ttm_alignment import fingerprint_aligned_tail
from src.strategies.ttm.ttm_features import compute_ttm_features_from_config, features_last_row
from src.strategies.ttm.ttm_regime import detect_regime
from src.strategies.ttm.empirical.alpha import compute_empirical_alpha, empirical_alpha_to_score_delta
from src.strategies.ttm.ttm_score import compute_score_v2_alpha, probs_valid, scores_to_probs
from src.strategies.ttm.ttm_v2_exit_continuation import (
    decide_long_exit_continuation,
    decide_short_exit_continuation,
)
from src.strategies.ttm.ttm_v2_gates import (
    ResearchGateSessionState,
    build_gate_diagnostics,
    entry_confirm_mode_for_gate,
    is_research_gate_mode,
    long_gate_for_trading,
    resolve_gate_mode,
    resolve_trading_gate_mode,
    short_signal_for_trading,
    valid_breakout_up,
)
from src.strategies.ttm.ttm_v2_phases import classify_crowd_phase, classify_short_phase
from src.strategies.ttm.ttm_signal import (
    _emit_signal,
    _merge_last_with_ohlc,
    _norm_confidence_01,
    _signal_features_dict,
    _validate_input_series_lengths,
)
logger = get_logger("ttm_signal_v2")

# Flat entry: so sánh trực tiếp prob với ngưỡng (long ưu tiên nếu cả hai vượt).
ENTRY_THRESHOLD = 0.6


def _v2_eval_hard_long_gate(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg_work: Mapping[str, Any],
    score_long: float,
) -> Tuple[bool, str]:
    """Strict LONG gate (backward-compatible wrapper)."""
    from src.strategies.ttm.ttm_v2_gates import GATE_MODE_STRICT, eval_long_gate

    ok, reason = eval_long_gate(
        last=last,
        components=components,
        cfg=cfg_work,
        score_long=float(score_long),
        mode=GATE_MODE_STRICT,
    )
    return ok, reason


def _merge_ttm_config(config: Mapping[str, Any], adaptive: Optional[Any]) -> Dict[str, Any]:
    merged = dict(config)
    if adaptive is not None:
        merged.update(adaptive.get_config_overlay())
    return merged


def _abs_rankable(x: Any) -> float:
    try:
        return abs(float(x))
    except (TypeError, ValueError):
        return 0.0


def _top_component_names(components: Mapping[str, Any], n: int = 3) -> list[str]:
    skip = (
        "total_long",
        "total_short",
        "raw",
        "prob_long",
        "prob_short",
        "momentum",
        "vol_breakout_interaction_factor",
        "crowd_phase",
        "entry_block_reason",
        "late_fomo_flag",
        "short_candidate",
        "short_chase_risk",
        "crowded_long_pressure",
        "continuation_decay",
        "rejection_confirm",
        "failed_breakout_confirm",
        "downside_momentum_confirm",
        "early_continuation_still_alive",
        "short_effective_strength_raw",
        "short_effective_strength",
        "short_score_unit",
        "prior_upside_breakout_exists",
        "phase_reason",
    )
    keys = [k for k in components if k not in skip]
    ranked = sorted(keys, key=lambda k: _abs_rankable(components.get(k, 0.0)), reverse=True)
    return ranked[:n]


def _pack_v2_short_components(components: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "prior_upside_breakout_exists": components.get("prior_upside_breakout_exists"),
        "last_upside_breakout_bar_index": components.get("last_upside_breakout_bar_index"),
        "last_upside_breakout_score": components.get("last_upside_breakout_score"),
        "last_upside_breakout_phase": components.get("last_upside_breakout_phase"),
        "last_upside_breakout_extension": components.get("last_upside_breakout_extension"),
        "bars_since_upside_breakout": components.get("bars_since_upside_breakout"),
        "crowded_long_pressure": components.get("crowded_long_pressure"),
        "continuation_decay": components.get("continuation_decay"),
        "rejection_confirm": components.get("rejection_confirm"),
        "failed_breakout_confirm": components.get("failed_breakout_confirm"),
        "downside_momentum_confirm": components.get("downside_momentum_confirm"),
        "early_continuation_still_alive": components.get("early_continuation_still_alive"),
        "short_chase_risk": components.get("short_chase_risk"),
        "short_effective_strength_raw": components.get("short_effective_strength_raw"),
        "short_effective_strength": components.get("short_effective_strength"),
        "short_score": components.get("short_score_unit", components.get("short_score")),
        "short_phase": components.get("short_phase"),
        "short_candidate": components.get("short_candidate"),
        "short_block_reason": components.get("short_block_reason"),
        "short_reason": components.get("short_reason"),
    }


def _pack_v2_score_components(components: Mapping[str, Any]) -> Dict[str, Any]:
    eraw = components.get("effective_strength_v3_raw", components.get("alpha_raw_long"))
    return {
        "breakout_conviction": components.get("breakout_conviction"),
        "continuation_confirm": components.get("continuation_confirm"),
        "basis_confirm": components.get("basis_confirm"),
        "extension": components.get("extension"),
        "extension_sq": components.get("extension_sq"),
        "last_bar_return_raw": components.get("last_bar_return_raw"),
        "last_bar_return_norm": components.get("last_bar_return_norm"),
        "positive_last_bar_return": components.get("positive_last_bar_return"),
        "late_phase_penalty": components.get("late_phase_penalty"),
        "effective_strength_raw": eraw,
        "effective_strength": components.get("effective_strength_v3", components.get("effective_strength_last")),
        "score_long": components.get("score_long"),
        "crowd_phase": components.get("crowd_phase"),
        "phase_reason": components.get("phase_reason"),
        "late_fomo_flag": components.get("late_fomo_flag"),
        "entry_block_reason": components.get("entry_block_reason"),
    }


def _v2_vol_confirm_confidence(
    conf: float,
    vol_z: float,
    *,
    breakout_align: bool,
    cfg: Mapping[str, Any],
) -> tuple[float, bool]:
    """Scale confidence when volume confirms a directional breakout (vol_z > 0); does not flip side."""
    if not bool(cfg.get("ttm_v2_vol_confirm_enabled", True)):
        return conf, False
    if not breakout_align or not np.isfinite(vol_z) or float(vol_z) <= 0.0:
        return conf, False
    gain = float(cfg.get("ttm_v2_vol_confirm_gain", 0.12))
    boosted = float(conf) * (1.0 + gain * float(np.tanh(float(vol_z))))
    return min(1.0, boosted), True


def _masked_last_float(last: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    vm = last.get("feature_valid_mask") or {}
    if isinstance(vm, Mapping) and not bool(vm.get(key, True)):
        return float(default)
    v = last.get(key)
    try:
        return float(v) if v is not None and np.isfinite(float(v)) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _masked_last_optional(last: Mapping[str, Any], key: str) -> Optional[float]:
    vm = last.get("feature_valid_mask") or {}
    if isinstance(vm, Mapping) and not bool(vm.get(key, True)):
        return None
    v = last.get(key)
    try:
        fv = float(v)
        return fv if np.isfinite(fv) else None
    except (TypeError, ValueError):
        return None


def generate_ttm_signal_v2(
    features_or_data: Mapping[str, Any] | Dict[str, Any],
    config: Mapping[str, Any],
    *,
    position_side: Optional[str] = None,
    live_mode: bool = False,
    adaptive: Optional[Any] = None,
    empirical_engine: Optional[Any] = None,
    bar_timestamp: Optional[int] = None,
    holding_bars: Optional[int] = None,
    position_unrealized_return: Optional[float] = None,
    entry_snapshot: Optional[Mapping[str, Any]] = None,
    position_meta: Optional[Mapping[str, Any]] = None,
    current_price: Optional[float] = None,
    research_gate_state: Optional[ResearchGateSessionState] = None,
    session_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Intraday alpha from :func:`~src.strategies.ttm.ttm_score.compute_score_v2_alpha` (breakout uses
    volume-free base strength; optional vol×breakout interaction via config), then symmetric softmax.
    Volume: flat-entry gate (low vol_z), optional confidence boost on breakout + vol_z > 0, and sizing
    (see ``ttm_v2_position_size_vol_k`` on features passed to execution).

    Flat entry: ``prob_long`` / ``prob_short`` vs ``entry_threshold`` (mặc định :data:`ENTRY_THRESHOLD`).
    In-position exit: **prob-only** by default; with ``ttm_v2_enable_continuation_aware_exit``,
    structured ordering (refactor_4) using optional ``holding_bars``, ``position_unrealized_return``,
    ``entry_snapshot`` / ``position_meta`` when provided by the runner/strategy.

    Optional ``empirical_engine`` (:class:`~src.strategies.ttm.empirical.EmpiricalAlphaEngine`): rolling
    bin curves vs forward return — **separate** from :class:`~src.strategies.ttm.ttm_adaptive_context.TTMAdaptiveContext`
    (PnL-driven regime weights). Enable via ``ttm_v2_empirical_alpha_enabled``; log-only vs blend via
    ``ttm_v2_empirical_log_only`` / ``ttm_v2_empirical_blend``.
    """
    print_trace = bool(live_mode) or bool(config.get("decision_trace_print", False))
    prob_exit = float(config.get("prob_exit_threshold", 0.4))
    entry_thr = float(config.get("entry_threshold", ENTRY_THRESHOLD))

    if "bars" in features_or_data and features_or_data.get("bars") is not None:
        raw = dict(features_or_data)
        if not bool(config.get("ttm_strict_basis_oi", False)):
            bad = _validate_input_series_lengths(raw)
            if bad:
                logger.warning(
                    "TTM V2: misaligned series",
                    extra={"ly_do": bad, "n_bars": len(raw.get("bars") or [])},
                )
                tr_mis = {
                    "stage": "validate",
                    "bar_index": None,
                    "action": "HOLD",
                    "reason": bad,
                    "trap_score": 0.0,
                    "model_version": "v2",
                }
                return _emit_signal(
                    {
                        "action": "HOLD",
                        "confidence": 0.0,
                        "reason": bad,
                        "strategy": "TTM",
                        "trap_score": 0.0,
                        "features": {"momentum": 0.0, "basis": 0.0, "oi_delta": 0.0},
                        "debug": {"misaligned": True, "model_version": "v2"},
                    },
                    tr_mis,
                    print_trace,
                )
        feats = compute_ttm_features_from_config(raw, config)
    else:
        feats = dict(features_or_data)

    n = int(feats.get("n", 0))
    input_fp = ""
    if "bars" in features_or_data and features_or_data.get("bars") is not None:
        input_fp = fingerprint_aligned_tail(dict(features_or_data))
    if n <= 0:
        tr_nd = {
            "stage": "features",
            "bar_index": None,
            "action": "HOLD",
            "reason": "no_data",
            "trap_score": 0.0,
            "model_version": "v2",
        }
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "no_data",
                "strategy": "TTM",
                "features": {"momentum": 0.0, "basis": 0.0, "oi_delta": 0.0},
                "debug": {"model_version": "v2"},
            },
            tr_nd,
            print_trace,
        )

    last = features_last_row(feats)
    last = _merge_last_with_ohlc(last, feats)
    bu_arr = feats.get("breakout_up")
    if isinstance(bu_arr, np.ndarray) and n > 0:
        last = dict(last)
        last["breakout_up_filtered_last"] = bool(bu_arr[-1])
    xst = feats.get("ttm_v2_cross_bar_state") if isinstance(feats, dict) else None
    if isinstance(xst, dict):
        last = dict(last)
        for key in (
            "ttm_v2_persist_last_upside_breakout_bar_index",
            "ttm_v2_persist_last_upside_breakout_score",
            "ttm_v2_persist_last_upside_breakout_phase",
            "ttm_v2_persist_last_upside_breakout_extension",
        ):
            if key in xst:
                last[key] = xst.get(key)
        ix = last.get("ttm_v2_persist_last_upside_breakout_bar_index")
        if ix is not None:
            try:
                last["ttm_v2_persist_last_upside_breakout_bar_index"] = int(ix)
            except (TypeError, ValueError):
                last.pop("ttm_v2_persist_last_upside_breakout_bar_index", None)
    vol_z = float(last.get("vol_z", last.get("vol_zscore", 0.0)) or 0.0)
    vol_z_thr = float(config.get("ttm_v2_vol_z_filter_threshold", -1.0))

    feat_pack: Dict[str, Any] = dict(_signal_features_dict(last))
    feat_pack["breakout_strength"] = float(
        last.get("breakout_strength_base", last.get("breakout_strength", 0.0)) or 0.0
    )
    feat_pack["raw_strength"] = _masked_last_optional(last, "raw_strength")
    feat_pack["cap"] = _masked_last_optional(last, "cap")
    feat_pack["exhaustion_candidate"] = bool(last.get("exhaustion_candidate"))
    feat_pack["exhaustion_confirm"] = bool(last.get("exhaustion_confirm"))
    feat_pack["short_score"] = _masked_last_optional(last, "short_score")
    feat_pack["short_setup_high"] = _masked_last_optional(last, "short_setup_high")
    feat_pack["short_setup_atr"] = _masked_last_optional(last, "short_setup_atr")
    feat_pack["short_setup_bar_index"] = _masked_last_optional(last, "short_setup_bar_index")
    feat_pack["short_setup_raw_strength"] = _masked_last_optional(last, "short_setup_raw_strength")
    feat_pack["short_setup_cap"] = _masked_last_optional(last, "short_setup_cap")
    feat_pack["extension"] = _masked_last_optional(last, "extension")
    feat_pack["last_bar_return"] = _masked_last_optional(last, "last_bar_return")
    feat_pack["rolling_high"] = _masked_last_optional(last, "rolling_high")
    feat_pack["atr"] = _masked_last_optional(last, "atr")
    feat_pack["effective_strength_pre_gate"] = _masked_last_optional(last, "effective_strength_pre_gate")
    feat_pack["effective_strength_active"] = bool(last.get("effective_strength_active"))
    feat_pack["effective_strength"] = _masked_last_optional(last, "effective_strength")
    if np.isfinite(vol_z):
        feat_pack["vol_zscore"] = float(vol_z)

    cfg_work = _merge_ttm_config(config, adaptive)
    regime_id = detect_regime(last, cfg_work)
    cfg_work["_ttm_regime_id"] = regime_id

    sl, ss, components = compute_score_v2_alpha(feats, last, cfg_work)
    pl, ps = scores_to_probs(sl, ss, cfg_work)
    ok = probs_valid(pl, ps)
    edge = float(pl - ps) if ok else float("nan")
    components = {**dict(components), "edge": edge}

    side_u = (position_side or "").strip().upper()
    bar_index_exit = n - 1

    empirical_debug: Dict[str, Any] = {}
    if bool(cfg_work.get("ttm_v2_empirical_alpha_enabled", False)) and empirical_engine is not None:
        close_v = float(last.get("close") or 0.0)
        ts = int(bar_timestamp) if bar_timestamp is not None else int(bar_index_exit)
        empirical_engine.on_bar(bar_index_exit, ts, last, close_v)
        clip_e = float(cfg_work.get("ttm_v2_empirical_alpha_clip", 0.003))
        a_emp, parts = compute_empirical_alpha(
            last, empirical_engine.calibration, clip_abs=clip_e
        )
        empirical_debug = {
            "empirical_alpha": a_emp,
            "empirical_alpha_parts": parts,
            "empirical_n_labeled": (
                empirical_engine.calibration.n_labeled
                if empirical_engine.calibration is not None
                else 0
            ),
            "empirical_log_only": bool(cfg_work.get("ttm_v2_empirical_log_only", True)),
            "empirical_blend": float(cfg_work.get("ttm_v2_empirical_blend", 0.0)),
        }
        log_only = bool(cfg_work.get("ttm_v2_empirical_log_only", True))
        blend = float(cfg_work.get("ttm_v2_empirical_blend", 0.0))
        if (
            not log_only
            and blend > 1e-12
            and empirical_engine.calibration is not None
            and empirical_engine.calibration.n_labeled > 0
        ):
            scale = float(cfg_work.get("ttm_v2_empirical_score_scale", 500.0))
            max_d = float(cfg_work.get("ttm_v2_empirical_max_score_delta", 2.0))
            delta = (
                empirical_alpha_to_score_delta(a_emp, scale=scale, max_delta=max_d) * blend
            )
            cap = float(cfg_work.get("ttm_v2_alpha_score_cap", 5.0))
            sl = float(np.clip(sl + delta, -cap, cap))
            ss = float(np.clip(ss - delta, -cap, cap))
            pl, ps = scores_to_probs(sl, ss, cfg_work)
            ok = probs_valid(pl, ps)
            edge = float(pl - ps) if ok else float("nan")
            components = {
                **dict(components),
                "edge": edge,
                "empirical_blend_delta": float(delta),
                "score_long": float(sl),
                "score": float(sl),
                "total_long": float(sl),
            }

    breakout_strength_down = float(last.get("breakout_strength_down") or 0.0)
    short_score_last = _masked_last_optional(last, "short_score")
    enable_short = bool(cfg_work.get("ttm_v2_enable_short_trading", True))
    use_short_v3 = bool(cfg_work.get("ttm_v2_use_short_effective_strength_v3", False))
    short_phase_now = str(components.get("short_phase") or "")
    short_score_now = float(components.get("short_score_unit", components.get("short_score", 0.0)) or 0.0)
    short_entry_thr = float(cfg_work.get("ttm_v2_short_entry_threshold", 0.0))
    short_chase_flag = bool(components.get("_short_chase_flag")) or short_phase_now == "short_chase_risk"
    alive_cut = float(cfg_work.get("ttm_v2_short_continuation_recovery_threshold", 0.5))
    early_alive = float(components.get("early_continuation_still_alive") or 0.0)
    short_ok, _short_br = short_signal_for_trading(
        components=components,
        cfg=cfg_work,
        short_phase=short_phase_now,
        short_score=short_score_now,
        short_chase_flag=short_chase_flag,
        early_alive=early_alive,
        enable_short=enable_short,
        live_mode=live_mode,
        session_state=research_gate_state,
        bar_index=int(n - 1) if n > 0 else None,
    )
    short_signal = bool(short_ok)
    long_signal = valid_breakout_up(last)

    def _debug_core() -> Dict[str, Any]:
        return {
            "model_version": "v2",
            "input_aligned_fingerprint": input_fp,
            "ttm_config_profile": str(config.get("ttm_config_profile", "") or ""),
            "oi_pipeline": last.get("oi_pipeline"),
            "volume_participation_log": last.get("volume_participation_log"),
            "positioning_log": last.get("positioning_log"),
            "ttm_regime_id": regime_id,
            "score_long": sl,
            "score_short": ss,
            "alpha_raw": float(components.get("alpha_raw", 0.0)),
            "alpha_rank": float(components.get("alpha_rank", 0.5)),
            "score": float(components.get("score", sl)),
            "edge": float(components.get("edge", edge)),
            "prob_long": pl,
            "prob_short": ps,
            "vol_z": vol_z,
            "vol_z_filter_threshold": vol_z_thr,
            "probs_valid": ok,
            "entry_threshold": entry_thr,
            "prob_entry_threshold": entry_thr,
            "prob_exit_threshold": prob_exit,
            "score_components": components,
            "breakout_strength_down": breakout_strength_down,
            "raw_strength": feat_pack.get("raw_strength"),
            "cap": feat_pack.get("cap"),
            "exhaustion_candidate": feat_pack.get("exhaustion_candidate"),
            "exhaustion_confirm": feat_pack.get("exhaustion_confirm"),
            "short_score": feat_pack.get("short_score"),
            "short_signal": short_signal,
            "short_gate_reason": (
                None
                if short_signal
                else (components.get("short_block_reason") or f"short_phase_{short_phase_now or 'unknown'}")
            ),
            "long_signal": long_signal,
            "extension": feat_pack.get("extension"),
            "last_bar_return": feat_pack.get("last_bar_return"),
            "effective_strength": feat_pack.get("effective_strength"),
            "effective_strength_pre_gate": feat_pack.get("effective_strength_pre_gate"),
            "effective_strength_active": feat_pack.get("effective_strength_active"),
            "feature_valid_mask": dict(last.get("feature_valid_mask") or {}),
            "top_components": _top_component_names(components, 3),
            "oi_context_flag": last.get("oi_context_flag"),
            "crowd_phase": components.get("crowd_phase") or classify_crowd_phase(last, cfg_work),
            "short_phase": components.get("short_phase") or classify_short_phase(last, cfg_work),
            "v2_score_components": _pack_v2_score_components(components),
            "v2_short_components": _pack_v2_short_components(components),
            **empirical_debug,
        }

    if side_u == "LONG":
        if bool(cfg_work.get("ttm_v2_enable_continuation_aware_exit")):
            act_l, canon_l, x_l = decide_long_exit_continuation(
                ok=ok,
                pl=float(pl),
                prob_exit=prob_exit,
                score_long=float(sl),
                last=last,
                components=components,
                cfg=cfg_work,
                holding_bars=holding_bars,
                unrealized_return=position_unrealized_return,
                entry_snapshot=entry_snapshot,
            )
            if act_l == "EXIT":
                out = {
                    "action": "EXIT",
                    "confidence": float(pl),
                    "reason": str(canon_l),
                    "strategy": "TTM",
                    "features": feat_pack,
                    "debug": {**_debug_core(), "exit_channel": "continuation", **x_l},
                }
                tr_pd = {
                    "stage": "exit",
                    "bar_index": bar_index_exit,
                    "position_side": "LONG",
                    "action": "EXIT",
                    "reason": str(canon_l),
                    "model_version": "v2",
                }
                return _emit_signal(out, tr_pd, print_trace)
            if act_l == "HOLD" and canon_l == "soft_min_hold_continuation":
                tr_sm = {
                    "stage": "exit",
                    "bar_index": bar_index_exit,
                    "position_side": "LONG",
                    "action": "HOLD",
                    "reason": "soft_min_hold_continuation",
                    "model_version": "v2",
                }
                return _emit_signal(
                    {
                        "action": "HOLD",
                        "confidence": float(pl),
                        "reason": "soft_min_hold_continuation",
                        "strategy": "TTM",
                        "features": feat_pack,
                        "debug": {**_debug_core(), **x_l},
                    },
                    tr_sm,
                    print_trace,
                )
            tr_hex = {
                "stage": "exit",
                "bar_index": bar_index_exit,
                "position_side": "LONG",
                "action": "HOLD",
                "reason": "no_exit_signal",
                "model_version": "v2",
            }
            return _emit_signal(
                {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reason": "no_exit_signal",
                    "strategy": "TTM",
                    "features": feat_pack,
                    "debug": {**_debug_core(), **x_l},
                },
                tr_hex,
                print_trace,
            )
        if ok and pl < prob_exit:
            out = {
                "action": "EXIT",
                "confidence": float(pl),
                "reason": "prob_decay",
                "strategy": "TTM",
                "features": feat_pack,
                "debug": {**_debug_core(), "exit_channel": "prob"},
            }
            tr_pd = {
                "stage": "exit",
                "bar_index": bar_index_exit,
                "position_side": "LONG",
                "action": "EXIT",
                "reason": "prob_decay",
                "model_version": "v2",
            }
            return _emit_signal(out, tr_pd, print_trace)
        tr_hex = {
            "stage": "exit",
            "bar_index": bar_index_exit,
            "position_side": "LONG",
            "action": "HOLD",
            "reason": "no_exit_signal",
            "model_version": "v2",
        }
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "no_exit_signal",
                "strategy": "TTM",
                "features": feat_pack,
                "debug": _debug_core(),
            },
            tr_hex,
            print_trace,
        )

    if side_u == "SHORT":
        if bool(cfg_work.get("ttm_v2_enable_continuation_aware_exit")):
            px_s = float(current_price) if current_price is not None else float(last.get("close") or 0.0)
            act_s, canon_s, x_s = decide_short_exit_continuation(
                ok=ok,
                ps=float(ps),
                prob_exit=prob_exit,
                last=last,
                components=components,
                cfg=cfg_work,
                holding_bars=holding_bars,
                unrealized_return=position_unrealized_return,
                position_meta=position_meta,
                current_price=px_s,
            )
            if act_s == "EXIT":
                out = {
                    "action": "EXIT",
                    "confidence": float(ps),
                    "reason": str(canon_s),
                    "strategy": "TTM",
                    "features": feat_pack,
                    "debug": {**_debug_core(), "exit_channel": "continuation", **x_s},
                }
                tr_pds = {
                    "stage": "exit",
                    "bar_index": bar_index_exit,
                    "position_side": "SHORT",
                    "action": "EXIT",
                    "reason": str(canon_s),
                    "model_version": "v2",
                }
                return _emit_signal(out, tr_pds, print_trace)
            tr_hxs = {
                "stage": "exit",
                "bar_index": bar_index_exit,
                "position_side": "SHORT",
                "action": "HOLD",
                "reason": "no_exit_signal",
                "model_version": "v2",
            }
            return _emit_signal(
                {
                    "action": "HOLD",
                    "confidence": 0.0,
                    "reason": "no_exit_signal",
                    "strategy": "TTM",
                    "features": feat_pack,
                    "debug": {**_debug_core(), **x_s},
                },
                tr_hxs,
                print_trace,
            )
        if ok and ps < prob_exit:
            out = {
                "action": "EXIT",
                "confidence": float(ps),
                "reason": "prob_decay",
                "strategy": "TTM",
                "features": feat_pack,
                "debug": {**_debug_core(), "exit_channel": "prob"},
            }
            tr_pds = {
                "stage": "exit",
                "bar_index": bar_index_exit,
                "position_side": "SHORT",
                "action": "EXIT",
                "reason": "prob_decay",
                "model_version": "v2",
            }
            return _emit_signal(out, tr_pds, print_trace)
        tr_hxs = {
            "stage": "exit",
            "bar_index": bar_index_exit,
            "position_side": "SHORT",
            "action": "HOLD",
            "reason": "no_exit_signal",
            "model_version": "v2",
        }
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "no_exit_signal",
                "strategy": "TTM",
                "features": feat_pack,
                "debug": _debug_core(),
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
            "model_version": "v2",
        }
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": "invalid_position_side",
                "strategy": "TTM",
                "features": feat_pack,
                "debug": _debug_core(),
            },
            tr_inv,
            print_trace,
        )

    # --- ENTRY (flat): softmax probs vs threshold ---
    debug_payload: Dict[str, Any] = {
        "bar_index": n - 1,
        "input_aligned_fingerprint": input_fp,
        "ttm_config_profile": str(config.get("ttm_config_profile", "") or ""),
        "basis": feat_pack.get("basis"),
        "breakout_strength": float(last.get("breakout_strength") or 0.0),
        "failure_strength": float(last.get("failure_strength") or 0.0),
        "basis_norm": float(last.get("basis_norm") or 0.0),
        "basis_delta": float(last.get("basis_delta") or 0.0),
        "vol_signal": float(last.get("vol_signal") or 0.0),
        "alpha_trap_score": float(last.get("alpha_trap_score") or 0.0),
        "oi_signal": float(last.get("oi_signal") or 0.0),
        "oi_pipeline": last.get("oi_pipeline"),
        "oi_context_flag": last.get("oi_context_flag"),
        "vol_regime": float(last.get("vol_regime") or 0.0),
        "vol_z": vol_z,
        "vol_z_filter_threshold": vol_z_thr,
        "ttm_regime_id": regime_id,
        "score_long": sl,
        "score_short": ss,
        "alpha_raw": float(components.get("alpha_raw", 0.0)),
        "alpha_rank": float(components.get("alpha_rank", 0.5)),
        "score": float(components.get("score", sl)),
        "edge": float(components.get("edge", edge)),
        "prob_long": pl,
        "prob_short": ps,
        "score_components": components,
        "breakout_strength_down": breakout_strength_down,
        "raw_strength": feat_pack.get("raw_strength"),
        "cap": feat_pack.get("cap"),
        "exhaustion_candidate": feat_pack.get("exhaustion_candidate"),
        "exhaustion_confirm": feat_pack.get("exhaustion_confirm"),
        "short_score": feat_pack.get("short_score"),
        "short_signal": short_signal,
        "long_signal": long_signal,
        "extension": feat_pack.get("extension"),
        "last_bar_return": feat_pack.get("last_bar_return"),
        "effective_strength": feat_pack.get("effective_strength"),
        "effective_strength_pre_gate": feat_pack.get("effective_strength_pre_gate"),
        "effective_strength_active": feat_pack.get("effective_strength_active"),
        "feature_valid_mask": dict(last.get("feature_valid_mask") or {}),
        "top_components": _top_component_names(components, 3),
        "model_version": "v2",
        "probs_valid": ok,
        "crowd_phase": components.get("crowd_phase") or classify_crowd_phase(last, cfg_work),
        "short_phase": components.get("short_phase") or classify_short_phase(last, cfg_work),
        "v2_score_components": _pack_v2_score_components(components),
        "v2_short_components": _pack_v2_short_components(components),
    }
    _bar_ix = int(n - 1) if n > 0 else None
    if session_date is None and bar_timestamp is not None:
        try:
            from datetime import datetime, timezone, timedelta

            dt = datetime.fromtimestamp(int(bar_timestamp), tz=timezone.utc) + timedelta(hours=7)
            session_date = dt.strftime("%Y%m%d")
        except (TypeError, ValueError, OSError):
            session_date = None
    _gate_diag = build_gate_diagnostics(
        last=last,
        components=components,
        cfg=cfg_work,
        score_long=float(sl),
        short_phase=short_phase_now,
        short_score=short_score_now,
        short_chase_flag=short_chase_flag,
        early_alive=early_alive,
        prob_long=float(pl),
        entry_threshold=float(entry_thr),
        live_mode=live_mode,
        bar_index=_bar_ix,
        session_state=research_gate_state,
        session_date=session_date,
    )
    debug_payload["gate_diagnostics"] = _gate_diag
    debug_payload["v2_gate_diagnostics"] = _gate_diag
    _gate_mode = resolve_gate_mode(cfg_work, live_mode=live_mode)
    _trading_gate_mode = resolve_trading_gate_mode(cfg_work, live_mode=live_mode)
    _research_long_entry = bool(_gate_diag.get("research_long_entry_allowed"))
    logger.info(
        "TTM V2 unified score",
        extra={
            "strategy": "TTM",
            "score_long": sl,
            "score_short": ss,
            "prob_long": pl,
            "prob_short": ps,
            "top_components": debug_payload["top_components"],
            "feature_valid_mask": debug_payload["feature_valid_mask"],
        },
    )

    def _log_hold(reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {**debug_payload, **(extra or {})}
        tr = {
            "stage": "entry",
            "bar_index": n - 1,
            "action": "HOLD",
            "reason": reason,
            "model_version": "v2",
        }
        if extra:
            tr = {**tr, **extra}
        return _emit_signal(
            {
                "action": "HOLD",
                "confidence": 0.0,
                "reason": reason,
                "strategy": "TTM",
                "trap_score": max(abs(sl), abs(ss)),
                "features": feat_pack,
                "debug": payload,
            },
            tr,
            print_trace,
        )

    if not ok:
        return _log_hold("v2_invalid_probs", {"v2_fallback": True})

    if np.isfinite(vol_z) and vol_z < vol_z_thr:
        return _log_hold(
            "v2_vol_z_filter",
            {
                "vol_z": vol_z,
                "vol_z_filter_threshold": vol_z_thr,
                "decision_source": "vol_z_gate",
            },
        )

    # --- Flat entry: exhaustion-confirm SHORT overrides continuation LONG ---
    decision_entry = "NONE"
    if short_signal:
        if ps > entry_thr:
            decision_entry = "SHORT"
        else:
            decision_entry = "NONE"
    elif is_research_gate_mode(_trading_gate_mode) and not live_mode:
        if _research_long_entry:
            decision_entry = "LONG"
    elif long_signal and pl > entry_thr:
        decision_entry = "LONG"
    elif ps > entry_thr and short_signal:
        decision_entry = "SHORT"
    else:
        decision_entry = "NONE"

    logger.info("TTM V2 FINAL DECISION: %s (pl=%s ps=%s thr=%s)", decision_entry, pl, ps, entry_thr)
    if print_trace:
        print(f"FINAL DECISION: {decision_entry}", flush=True)

    if decision_entry == "LONG" and bool(cfg_work.get("ttm_v2_hard_long_gate_enabled", True)):
        ok_gate, gate_reason, gate_lbl = long_gate_for_trading(
            last=last,
            components=components,
            cfg=cfg_work,
            score_long=float(sl),
            live_mode=live_mode,
            bar_index=_bar_ix,
            session_state=research_gate_state,
        )
        if not ok_gate:
            return _log_hold(
                gate_reason,
                {
                    "score_long": float(sl),
                    "ttm_v2_score_long_entry_threshold": float(
                        cfg_work.get("ttm_v2_score_long_entry_threshold", 0.0)
                    ),
                    "ttm_v2_research_score_long_floor": float(
                        cfg_work.get("ttm_v2_research_score_long_floor", -0.30)
                    ),
                    "crowd_phase": str(components.get("crowd_phase") or ""),
                    "decision_source": gate_lbl,
                    "long_gate_reason": gate_reason,
                    "gate_mode": resolve_gate_mode(cfg_work, live_mode=live_mode),
                },
            )

    if decision_entry == "LONG":
        if bool(cfg_work.get("ttm_v2_use_effective_strength_v3")) and not bool(
            cfg_work.get("ttm_v2_hard_long_gate_enabled", True)
        ):
            cp = str(components.get("crowd_phase") or "")
            if cp and cp not in ("ignition", "early_continuation"):
                return _log_hold(
                    "crowd_phase_block",
                    {
                        "crowd_phase": cp,
                        "entry_block_reason": components.get("entry_block_reason"),
                        "decision_source": "crowd_phase_gate",
                    },
                )
        if bool(cfg_work.get("ttm_v2_enable_late_fomo_filter")):
            lf = components.get("late_fomo_flag")
            if lf is True:
                return _log_hold(
                    "late_fomo_filter",
                    {"late_fomo_flag": True, "decision_source": "late_fomo_gate"},
                )
        if bool(cfg_work.get("ttm_v2_enable_entry_confirmation")):
            _ecm_sig = entry_confirm_mode_for_gate(_gate_mode)
            _entry_confirm_result = "pass"
            _entry_confirm_block_reason = None
            if _ecm_sig != "research":
                thr_sl = float(cfg_work.get("ttm_v2_score_long_entry_threshold", 0.0))
                if float(sl) < thr_sl:
                    return _log_hold(
                        "score_below_threshold",
                        {"score_long": sl, "ttm_v2_score_long_entry_threshold": thr_sl},
                    )
            if bool(last.get("exhaustion_confirm")):
                return _log_hold("exhaustion_confirm_block", {})
            pos_lb = components.get("positive_last_bar_return")
            if pos_lb is None:
                lbv = float(last.get("last_bar_return") or 0.0)
                pos_lb = max(0.0, float(np.tanh(lbv * 6.0)))
            else:
                pos_lb = float(pos_lb)
            lb_thr = cfg_work.get("ttm_v2_last_bar_return_spike_threshold")
            if lb_thr is not None and pos_lb > float(lb_thr):
                return _log_hold(
                    "last_bar_return_spike",
                    {"positive_last_bar_return": pos_lb, "threshold": float(lb_thr)},
                )
            if bool(components.get("late_fomo_flag")):
                return _log_hold(
                    "late_fomo_entry_confirm",
                    {
                        "entry_confirm_mode": _ecm_sig,
                        "entry_confirm_result": "block",
                        "entry_confirm_block_reason": "late_fomo_entry_confirm",
                    },
                )
            debug_payload["entry_confirm_mode"] = _ecm_sig
            debug_payload["entry_confirm_result"] = _entry_confirm_result
            debug_payload["entry_confirm_block_reason"] = _entry_confirm_block_reason

    if ok and decision_entry == "LONG" and not is_research_gate_mode(_trading_gate_mode) and pl <= entry_thr:
        raise RuntimeError(
            "INCONSISTENT: LONG but prob_long not above threshold "
            f"(pl={pl}, thr={entry_thr})"
        )
    if ok and decision_entry == "LONG" and is_research_gate_mode(_trading_gate_mode) and not _research_long_entry:
        raise RuntimeError(
            "INCONSISTENT: research LONG without research_long_entry_allowed "
            f"(pl={pl}, gate_mode={_trading_gate_mode})"
        )
    if ok and decision_entry == "LONG" and not is_research_gate_mode(_trading_gate_mode) and not long_signal:
        raise RuntimeError(
            "INCONSISTENT: strict LONG while breakout_up is False "
            f"(pl={pl}, breakout_up={long_signal})"
        )
    if ok and decision_entry == "SHORT" and ps <= entry_thr:
        raise RuntimeError(
            "INCONSISTENT: SHORT but prob_short not above threshold "
            f"(ps={ps}, thr={entry_thr})"
        )
    if ok and decision_entry == "SHORT" and not short_signal:
        raise RuntimeError(
            "INCONSISTENT: SHORT while short_signal is False "
            f"(exhaustion_confirm={bool(last.get('exhaustion_confirm'))}, short_v3={use_short_v3}, "
            f"short_phase={components.get('short_phase')}, short_candidate={components.get('short_candidate')})"
        )

    if decision_entry == "NONE":
        no_entry_reason = "below_entry_threshold"
        if is_research_gate_mode(_trading_gate_mode) and not live_mode:
            if bool(_gate_diag.get("blocked_by_hard_block")):
                no_entry_reason = str(_gate_diag.get("hard_block_reason") or "hard_block_long")
            elif bool(_gate_diag.get("quality_research_long_allowed")) or bool(
                _gate_diag.get("exploratory_research_long_allowed")
            ):
                if bool(_gate_diag.get("blocked_by_prob_gate")):
                    no_entry_reason = "blocked_by_prob_gate"
                elif bool(_gate_diag.get("blocked_by_trade_cap")):
                    no_entry_reason = str(
                        _gate_diag.get("research_trade_cap_block_reason") or "blocked_by_trade_cap"
                    )
                elif bool(_gate_diag.get("long_candidate")):
                    no_entry_reason = str(
                        _gate_diag.get("quality_research_block_reason")
                        or _gate_diag.get("exploratory_research_block_reason")
                        or "research_tier_gate_block"
                    )
        blocked_short_phases = {
            "no_short_context",
            "crowded_long_watch",
            "short_setup",
            "short_chase_risk",
            "short_invalid",
        }
        if short_phase_now in blocked_short_phases:
            no_entry_reason = str(components.get("short_block_reason") or f"short_phase_{short_phase_now}")
        elif enable_short and ps > entry_thr and not short_signal:
            no_entry_reason = str(components.get("short_block_reason") or "short_gate_block")
        if short_signal and ps <= entry_thr:
            no_entry_reason = "short_below_entry_threshold"
        return _log_hold(
            no_entry_reason,
            {
                "prob_long": pl,
                "prob_short": ps,
                "entry_threshold": entry_thr,
                "decision_source": "threshold",
            },
        )

    if decision_entry == "LONG":
        conf = _norm_confidence_01(float(pl), cap=1.0)
        conf, vol_conf_applied = _v2_vol_confirm_confidence(
            conf,
            vol_z,
            breakout_align=bool(last.get("breakout_up")),
            cfg=cfg_work,
        )
        _gm = resolve_gate_mode(cfg_work, live_mode=live_mode)
        _long_reason = (
            "research_softmax_entry_long"
            if is_research_gate_mode(_gm)
            else "softmax_entry_long"
        )
        out = {
            "action": "LONG",
            "confidence": conf,
            "reason": _long_reason,
            "strategy": "TTM",
            "trap_score": float(sl),
            "features": feat_pack,
            "debug": {
                **debug_payload,
                "entry": "softmax",
                "decision_source": "threshold",
                "entry_threshold": entry_thr,
                "vol_confirm_applied": vol_conf_applied,
                "gate_mode": _gm,
                "research_gate_allowed": bool(_gate_diag.get("research_long_allowed")),
                "research_entry_source": _gate_diag.get("research_entry_source"),
                "research_prob_gate_bypassed": _gate_diag.get("research_prob_gate_bypassed"),
            },
        }
        tr_l = {
            "stage": "entry",
            "bar_index": n - 1,
            "action": "LONG",
            "reason": _long_reason,
            "model_version": "v2",
        }
        return _emit_signal(out, tr_l, print_trace)

    # decision_entry == "SHORT"
    conf = _norm_confidence_01(float(ps), cap=1.0)
    conf, vol_conf_applied = _v2_vol_confirm_confidence(
        conf,
        vol_z,
        breakout_align=bool(short_signal),
        cfg=cfg_work,
    )
    short_reason_lbl = (
        "softmax_entry_short_crowd_unwind" if use_short_v3 else "softmax_entry_short_exhaustion"
    )
    short_ds = "short_opportunity_v3" if use_short_v3 else "exhaustion_confirm"
    feat_out = dict(feat_pack)
    if use_short_v3:
        feat_out["ttm_v2_short_setup"] = "short_opportunity_v3"
    out = {
        "action": "SHORT",
        "confidence": conf,
        "reason": short_reason_lbl,
        "strategy": "TTM",
        "trap_score": float(ss),
        "features": feat_out,
        "debug": {
            **debug_payload,
            "entry": "softmax",
            "decision_source": short_ds,
            "entry_threshold": entry_thr,
            "vol_confirm_applied": vol_conf_applied,
        },
    }
    tr_s = {
        "stage": "entry",
        "bar_index": n - 1,
        "action": "SHORT",
        "reason": short_reason_lbl,
        "model_version": "v2",
    }
    return _emit_signal(out, tr_s, print_trace)
