"""TTM strategy parameters."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def is_ttm_adaptive_learning_enabled(cfg: Mapping[str, Any]) -> bool:
    """
    True when PnL-driven regime weight updates are allowed.

    Prefer explicit ``ttm_adaptive_enabled`` when present; otherwise ``adaptive_enabled``.
    """
    if "ttm_adaptive_enabled" in cfg:
        return bool(cfg["ttm_adaptive_enabled"])
    return bool(cfg.get("adaptive_enabled", False))


TTM_CONFIG: Dict[str, Any] = {
    # --- MODE: v1_only (default) | shadow | v2_only ---
    "ttm_mode": "v1_only",
    "entry_threshold": 0.6,
    "prob_entry_threshold": 0.6,
    "prob_exit_threshold": 0.4,
    "score_temperature": 1.0,
    # Vol regime: vol_spike below 1/vol_regime_bound -> -1, above bound -> 1
    "vol_regime_bound": 1.2,
    # --- CORE LOGIC ---
    "breakout_window": 20,
    # Tradable breakout = raw (close > rolling_high / close < rolling_low) & ATR-normalized strength > min.
    "breakout_strength_min": 0.5,
    # Continuation-only LONG breakout: bars above this rolling p-quantile cap are tagged exhaustion_candidate.
    "ttm_breakout_exhaustion_cap_window": 200,
    "ttm_breakout_exhaustion_cap_q": 0.8,
    "failure_window": 3,
    "trap_lookback_bars": 3,
    "vol_threshold": 1.2,
    "vol_short_window": 5,
    "vol_long_window": 20,
    # Trade volume (DNSE bar volume): z-score window, breakout gate, V2 momentum boost
    "ttm_volume_features_enabled": True,
    "volume_z_window": 50,
    "volume_breakout_strength_threshold": 0.0,
    "oi_z_window": 20,
    # Rolling window (bars) for OI *level* mean/std → oi_zscore → tanh → oi_signal (positioning).
    "oi_signal_level_window": 100,
    # Deprecated alias: if set, falls back when oi_signal_level_window missing in older configs.
    "oi_signal_std_window": 100,
    # OI assert off by default: keep runtime stable when OI stream is missing/flat.
    "ttm_oi_assert_nonzero_std": False,
    # OI / basis confirmation (upgraded TTM)
    "oi_z_threshold": 0.8,
    "oi_increase_threshold": 0,
    "basis_divergence_threshold": 0.0,
    "basis_reversal_threshold": 0.1,
    "confirm_trap_min_score": 2.0,
    "use_open_interest": False,
    # When true and OI stream is missing/flat, synthesize a smooth proxy from volume participation
    # so positioning splits remain statistically meaningful in validation/backtests.
    "ttm_oi_proxy_from_volume_enabled": True,
    "ttm_oi_proxy_strength": 0.35,
    # Positioning (intraday core): positioning_strength = w_basis * tanh(clip(basis_norm)) + w_vol * vol_signal
    "positioning_basis_norm_clip": 3.0,
    "positioning_w_basis": 0.6,
    "positioning_w_vol": 0.4,
    # Deprecated: ignored by new positioning (kept for older configs).
    "positioning_basis_coeff": 0.3,
    # Optional: daily OI context only (not used in V2 score). Set threshold + pass oi_prev_day_close on data.
    "oi_day_change_threshold": None,
    # V2 alpha: w1*breakout + w2*price_momentum + w3*basis_mom (vol_z not in blend; gate below).
    "ttm_v2_w_breakout": 0.4,
    "ttm_v2_w_momentum": 0.35,
    "ttm_v2_w_basis": 0.25,
    # Deprecated (ignored by compute_score_v2_alpha): kept for older JSON configs.
    "ttm_v2_w_positioning": 0.4,
    "ttm_v2_w_trap": 0.2,
    "ttm_v2_alpha_score_cap": 5.0,
    # Flat entry: hold when vol_z < threshold (low participation vs rolling window).
    "ttm_v2_vol_z_filter_threshold": -1.0,
    # After entry prob: if directional breakout and vol_z > 0, scale confidence by (1 + gain * tanh(vol_z)).
    "ttm_v2_vol_confirm_enabled": True,
    "ttm_v2_vol_confirm_gain": 0.12,
    # V2 sizing: multiply strength-based size by (1 + k * tanh(vol_z)).
    "ttm_v2_position_size_vol_k": 0.25,
    # Optional: scale breakout leg only by (1 + k * tanh(vol_z)); 0 = off (default).
    "ttm_v2_vol_breakout_interaction_k": 0.0,
    # V2 calibration only (TTM V2 alpha long breakout leg + validation buckets): raw_strength minus penalties.
    # Defaults 0 = identical to raw ATR-normalized strength until tuned (no CI surprise).
    "ttm_v2_effective_strength_k_extension": 0.0,
    "ttm_v2_effective_strength_k_lastret": 0.0,
    # When True, extension = abs(rolling_zscore(close)); when False, extension = signed price_z.
    "ttm_v2_effective_strength_use_abs_price_z": True,
    # Exhaustion-confirm SHORT (TTM V2 only): SL above setup-bar high, TP from actual entry.
    "ttm_v2_exhaustion_short_sl_atr_mult": 0.5,
    "ttm_v2_exhaustion_short_tp_atr_mult": 1.5,
    "ttm_v2_exhaustion_short_time_stop_bars": 5,
    "ttm_v2_sign_corr_forward_bars": 4,
    "ttm_v2_rank_window": 120,
    "ttm_v2_use_rank_score": False,
    "ttm_v2_sign_corr_min_n": 25,
    # Guard dynamic sign flips: keep historical adaptation but block full inversion on weak evidence.
    "ttm_v2_sign_flip_guard_enabled": True,
    "ttm_v2_sign_flip_max": 0.35,
    # Legacy LONG quality-shaping knobs kept for config compatibility after continuation/exhaustion refactor.
    "ttm_breakout_exhaustion_penalty": 0.30,
    "ttm_breakout_quality_momentum_bonus": 0.20,
    # Entry scoring (stage 2): need score >= entry_score_min
    "entry_score_min": 2,
    "entry_momentum_abs_min": 0.01,
    "dynamic_threshold_percentile": 75.0,
    "score_history_window": 120,
    "score_weight_breakout": 1.0,
    "score_weight_failure": 1.2,
    "score_weight_basis": 0.8,
    "score_weight_basis_delta": 0.5,
    "score_weight_oi": 0.0,
    "score_weight_vol_regime": 0.4,
    # Multiplies breakout_strength: (1 + k * vol_regime); vr in {-1,0,1}
    "vol_regime_breakout_k": 0.5,
    "score_use_tanh_logits": False,
    "softmax_temperature": 1.0,
    "ttm_score_cap": 5.0,
    "ttm_score_k1": 1.0,
    "ttm_score_k2": 1.0,
    "ttm_score_k3": 1.0,
    "ttm_score_k4": 1.0,
    "ttm_score_k5": 1.0,
    "ttm_w_momentum": 1.0,
    "ttm_w_basis": 1.0,
    "ttm_w_oi": 0.0,
    "ttm_regime_weights": {
        -1: {"m": 0.85, "b": 0.9, "o": 0.9},
        0: {"m": 1.0, "b": 1.0, "o": 1.0},
        1: {"m": 1.15, "b": 1.1, "o": 1.1},
    },
    "ttm_strict_asserts": False,
    # --- RISK ---
    "stop_loss_points": 8.0,
    "take_profit_points": 12.0,
    "max_bars_in_trade": 10,
    # Ép đóng / chặn vào mới gần cuối phiên VN (cần unix_ts trên nến). Bật trong build_ttm_paper_live_config.
    "session_flatten_enabled": False,
    "session_flatten_vn_morning_hhmm": "11:25",
    "session_flatten_vn_afternoon_hhmm": "14:38",
    # Exit / reverse (runner)
    "allow_reverse": False,
    "exit_basis_sharp_drop": 0.15,
    # Live debugging: print full decision trace to stdout (also use live_mode on TTMStrategy)
    "decision_trace_print": False,
    # If True, block entry when directional basis_ok_* is false (optional strict gate)
    "entry_require_basis": False,
    # Extra logging when mode is v1_only (normally V2 is skipped)
    "log_v2_in_v1_only": False,
    # --- ADAPTIVE (learned weights from realized trades; default off) ---
    # ``adaptive_enabled`` is a short alias; ``ttm_adaptive_enabled`` wins when both are set.
    "adaptive_enabled": False,
    "ttm_adaptive_enabled": False,
    "ttm_trade_log_maxlen": 100,
    "ttm_adaptive_min_trades": 30,
    "ttm_adaptive_alpha": 0.1,
    "ttm_adaptive_max_delta": 0.2,
    "ttm_adaptive_pnl_variance_max": 1e12,
    "ttm_adaptive_update_every_n_trades": 20,
    "ttm_adaptive_use_recency_weight": True,
    "ttm_adaptive_recency_halflife_trades": 40,
    "ttm_regime_low_threshold": 0.8,
    "ttm_regime_high_threshold": 1.2,
    "ttm_adaptive_rollback_eps": 1e-9,
    # Paper live song song V1+V2: build_ttm_paper_live_config() tắt adaptive + trace stdout;
    # chỉnh prob_entry_threshold / prob_exit_threshold tại đây hoặc truyền extra vào build_ttm_paper_live_config.
    # --- ALIGNMENT / AUDIT (post-audit refinements) ---
    # Empty string = disabled. "research_parallel" | "live_adaptive" when using preset builders.
    "ttm_config_profile": "",
    # Require basis + OI series same length as bars; fail-fast (no zero-fill). Use live_adaptive preset.
    "ttm_strict_basis_oi": False,
    # Reject non-finite feature scalars before tanh/softmax in compute_score.
    "ttm_strict_feature_finite": False,
    # If True with ttm_adaptive_enabled, TTMDerivativesStrategy must wire TTMAdaptiveContext or raise.
    "ttm_assert_adaptive_wired": False,
    # Optional path: atomic JSON with closes, basis, OI per bar (last built state).
    "ttm_persist_aligned_bundle_path": "",
    # V2: blend V1-style trap memory into failure_strength on last bar (see ttm_features).
    "ttm_v2_use_trap_memory": False,
    "ttm_trap_memory_min_boost": 0.02,
    "ttm_trap_memory_breakout_scale": 0.5,
    # Cap continuous failure_strength (same relative units as breakout_strength).
    "ttm_failure_strength_cap": 3.0,
    # --- EMPIRICAL ALPHA (rolling bin curves; see automate_tunning.md + src/strategies/ttm/empirical/) ---
    # When True, pass EmpiricalAlphaEngine from strategy / ParallelRunner into generate_ttm_signal_v2.
    "ttm_v2_empirical_alpha_enabled": False,
    # If True, only log empirical_alpha in debug; if False and blend>0, adjust score_long/short before softmax.
    "ttm_v2_empirical_log_only": True,
    # Blend empirical score delta: 0 = V2 score only, 1 = full delta from empirical_alpha_to_score_delta.
    "ttm_v2_empirical_blend": 0.0,
    "ttm_v2_empirical_alpha_clip": 0.003,
    "ttm_v2_empirical_score_scale": 500.0,
    "ttm_v2_empirical_max_score_delta": 2.0,
    "ttm_v2_empirical_window_size": 300,
    "ttm_v2_empirical_forward_horizon": 4,
    "ttm_v2_empirical_n_bins": 5,
    "ttm_v2_empirical_min_bin_samples": 20,
    "ttm_v2_empirical_update_every_bars": 30,
    "ttm_v2_empirical_ema_old_weight": 0.7,
    # Backtest / execution helpers (scripts/ttm_empirical_research.py)
    "ttm_v2_empirical_cost_roundtrip": 0.0004,
    "ttm_v2_empirical_cost_margin": 0.0001,
    # Breakout monotonic validator: require strict high > mid > low on mean ret_4 (set False only with PM sign-off).
    "ttm_validation_breakout_require_strict_triplet": True,
    "ttm_v2_empirical_no_trade_abs": 0.0,
    "ttm_v2_empirical_size_min": 0.5,
    "ttm_v2_empirical_size_max": 2.0,
    # --- TTM V2 REFACTOR (crowd-alpha model) ---
    # Core feature toggles (off by default = legacy behavior)
    "ttm_v2_use_effective_strength_v3": False,
    "ttm_v2_enable_late_fomo_filter": False,
    "ttm_v2_enable_entry_confirmation": False,
    # Hard LONG entry gate (score + phase + breakout); paper preset enables below.
    "ttm_v2_hard_long_gate_enabled": False,
    # If True, allow LONG fill when only signal-bar breakout was valid (entry bar may fail filtered breakout).
    "ttm_v2_allow_signal_only_long_execution": False,
    "ttm_v2_enable_soft_min_hold": False,
    # SHORT paper optimization
    "ttm_v2_enable_short_trading": True,
    "ttm_v2_enable_short_trading_default_paper": True,
    "ttm_v2_enable_short_trading_default_live": False,
    "ttm_v2_enable_short_candidate_logging": True,
    "ttm_v2_use_short_effective_strength_v3": False,
    "ttm_v2_enable_short_anti_chase": True,
    # LONG weights
    "ttm_v2_w_breakout_conviction": 1.0,
    "ttm_v2_w_continuation_confirm": 0.5,
    "ttm_v2_w_basis_confirm": 0.2,
    "ttm_v2_w_extension": 0.5,
    "ttm_v2_w_extension_sq": 0.2,
    "ttm_v2_w_positive_last_bar_return": 0.5,
    "ttm_v2_w_late_phase_penalty": 0.7,
    # SHORT weights
    "ttm_v2_w_crowded_long_pressure": 1.0,
    "ttm_v2_w_continuation_decay": 0.8,
    "ttm_v2_w_rejection_confirm": 0.8,
    "ttm_v2_w_failed_breakout_confirm": 0.8,
    "ttm_v2_w_downside_momentum_confirm": 0.5,
    "ttm_v2_w_early_continuation_still_alive": 1.0,
    "ttm_v2_w_short_chase_risk": 0.8,
    # LONG thresholds
    "ttm_v2_score_long_entry_threshold": 0.0,
    "ttm_v2_late_fomo_threshold": 0.7,
    "ttm_v2_last_bar_return_spike_threshold": None,
    "ttm_v2_extension_late_threshold": None,
    "ttm_v2_continuation_exit_threshold": 0.0,
    "ttm_v2_soft_min_hold_bars": 2,
    "ttm_v2_max_hold_bars": None,
    # Continuation-aware exit (refactor_4); False = legacy prob-only in-position exit in signal.
    "ttm_v2_enable_continuation_aware_exit": False,
    "ttm_v2_target_alpha_hold_bars": 3,
    "ttm_v2_failed_breakout_failure_min": 0.35,
    "ttm_v2_exhaustion_rejection_threshold": 0.25,
    "ttm_v2_score_decay_relative": 0.65,
    # Bar-close return exits (parallel runner / strategy); defaults match runner fallbacks.
    "ttm_v2_sl_return": -0.0007,
    "ttm_v2_tp_return": 0.0015,
    "ttm_v2_time_stop_bars": 4,
    "ttm_v2_sl_return_short": -0.0007,
    "ttm_v2_tp_return_short": 0.0015,
    "ttm_v2_time_stop_bars_short": 4,
    # SHORT thresholds
    "ttm_v2_short_entry_threshold": 0.0,
    "ttm_v2_short_trigger_threshold": 0.5,
    "ttm_v2_short_chase_risk_threshold": 0.7,
    "ttm_v2_short_max_hold_bars": None,
    "ttm_v2_short_continuation_recovery_threshold": 0.5,
    "ttm_v2_short_downside_decay_threshold": 0.0,
    "ttm_v2_bars_since_breakout_min_for_short": 1,
    "ttm_v2_bars_since_breakout_max_for_short": 20,
    "ttm_v2_short_prior_breakout_lookback": 40,
}


def build_ttm_research_parallel_config(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Parallel research / paper JSONL: both models, **no** PnL learning, permissive alignment.

    Does not require basis/OI on every bar (zeros allowed when series absent).
    """
    cfg = dict(TTM_CONFIG)
    cfg["ttm_config_profile"] = "research_parallel"
    cfg["adaptive_enabled"] = False
    cfg["ttm_adaptive_enabled"] = False
    cfg["ttm_strict_basis_oi"] = False
    cfg["ttm_strict_feature_finite"] = False
    cfg["ttm_assert_adaptive_wired"] = False
    cfg["decision_trace_print"] = False
    cfg["session_flatten_enabled"] = True
    cfg["ttm_v2_enable_short_trading"] = bool(
        cfg.get("ttm_v2_enable_short_trading_default_paper", True)
    )
    # Paper / parallel JSONL: stricter LONG logging + entry-time confirmation (no weight changes).
    cfg["ttm_v2_hard_long_gate_enabled"] = True
    cfg["ttm_v2_enable_entry_confirmation"] = True
    if extra:
        cfg.update(dict(extra))
    return cfg


def build_ttm_live_adaptive_config(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """
    Production-style path: strict bar alignment, finite feature asserts, adaptive learning on
    closed trades with bounded updates + rollback (see ttm_online_learning).
    """
    cfg = dict(TTM_CONFIG)
    cfg["ttm_config_profile"] = "live_adaptive"
    cfg["adaptive_enabled"] = True
    cfg["ttm_adaptive_enabled"] = True
    cfg["ttm_strict_basis_oi"] = True
    cfg["ttm_strict_feature_finite"] = True
    cfg["ttm_assert_adaptive_wired"] = True
    cfg["decision_trace_print"] = False
    cfg["ttm_v2_enable_short_trading"] = bool(
        cfg.get("ttm_v2_enable_short_trading_default_live", False)
    )
    if extra:
        cfg.update(dict(extra))
    return cfg


def build_ttm_paper_live_config(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Backward-compatible alias: same as :func:`build_ttm_research_parallel_config`."""
    return build_ttm_research_parallel_config(extra)
