from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


def _to_finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _fin(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def build_exhaustion_short_entry_meta(
    features: Mapping[str, Any],
    entry_price: float,
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if not bool(features.get("exhaustion_confirm")):
        return None
    entry_px = _to_finite_float(entry_price)
    setup_high = _to_finite_float(features.get("short_setup_high"))
    setup_atr = _to_finite_float(features.get("short_setup_atr"))
    if entry_px is None or entry_px <= 0.0 or setup_high is None or setup_atr is None or setup_atr <= 1e-12:
        return None
    sl_mult = max(0.0, float(config.get("ttm_v2_exhaustion_short_sl_atr_mult", 0.5)))
    tp_mult = max(0.0, float(config.get("ttm_v2_exhaustion_short_tp_atr_mult", 1.5)))
    time_bars = max(1, int(config.get("ttm_v2_exhaustion_short_time_stop_bars", 5)))
    return {
        "ttm_v2_short_setup": "exhaustion_confirm",
        "short_setup_high": float(setup_high),
        "short_setup_atr": float(setup_atr),
        "short_setup_raw_strength": _to_finite_float(features.get("short_setup_raw_strength")),
        "short_setup_cap": _to_finite_float(features.get("short_setup_cap")),
        "short_score": _to_finite_float(features.get("short_score")),
        "short_setup_bar_index": _to_finite_float(features.get("short_setup_bar_index")),
        "short_entry_price": float(entry_px),
        "short_stop_loss_price": float(setup_high + sl_mult * setup_atr),
        "short_take_profit_price": float(entry_px - tp_mult * setup_atr),
        "short_time_stop_bars": int(time_bars),
    }


def is_exhaustion_short_meta(meta: Optional[Mapping[str, Any]]) -> bool:
    return bool(meta) and str((meta or {}).get("ttm_v2_short_setup", "")).lower() == "exhaustion_confirm"


def is_short_exit_meta(meta: Optional[Mapping[str, Any]]) -> bool:
    """SL/TP/time exit model for exhaustion-confirm or crowd-unwind SHORT."""
    if not meta:
        return False
    s = str((meta or {}).get("ttm_v2_short_setup", "")).lower()
    return s in ("exhaustion_confirm", "short_opportunity_v3")


def build_short_opportunity_entry_meta(
    features: Mapping[str, Any],
    entry_price: float,
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    if str(features.get("ttm_v2_short_setup", "")).lower() != "short_opportunity_v3":
        return None
    entry_px = _to_finite_float(entry_price)
    rh = _to_finite_float(features.get("rolling_high"))
    atr = _to_finite_float(features.get("atr"))
    if atr is None or atr <= 1e-12:
        atr = _to_finite_float(features.get("short_setup_atr"))
    if entry_px is None or entry_px <= 0.0 or rh is None or atr is None or atr <= 1e-12:
        return None
    sl_mult = max(0.0, float(config.get("ttm_v2_exhaustion_short_sl_atr_mult", 0.5)))
    tp_mult = max(0.0, float(config.get("ttm_v2_exhaustion_short_tp_atr_mult", 1.5)))
    max_b = config.get("ttm_v2_short_max_hold_bars")
    if max_b is None:
        max_b = int(config.get("ttm_v2_exhaustion_short_time_stop_bars", 5))
    time_bars = max(1, int(max_b))
    return {
        "ttm_v2_short_setup": "short_opportunity_v3",
        "short_setup_high": float(rh),
        "short_setup_atr": float(atr),
        "short_entry_price": float(entry_px),
        "short_stop_loss_price": float(rh + sl_mult * atr),
        "short_take_profit_price": float(entry_px - tp_mult * atr),
        "short_time_stop_bars": int(time_bars),
        "short_ref_high_at_entry": float(rh),
    }


def check_short_v3_structural_exit(
    meta: Optional[Mapping[str, Any]],
    current_price: float,
    holding_bars: int,
    last: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Reclaim / long recovery / downside decay (refactor 3); after SL/TP/time."""
    if not meta or str(meta.get("ttm_v2_short_setup", "")).lower() != "short_opportunity_v3":
        return None
    px = _to_finite_float(current_price)
    rh_ref = _to_finite_float((meta or {}).get("short_ref_high_at_entry"))
    close_px = _to_finite_float(last.get("close"))
    if px is None:
        return None
    if rh_ref is not None and close_px is not None and close_px > float(rh_ref) * 1.0005:
        return {"reason": "ttm_exit_short_price_reclaim", "trigger": "RECLAIM"}
    if bool(last.get("breakout_up")) and _fin(last.get("momentum_1_z")) > 0.12:
        return {"reason": "ttm_exit_short_long_continuation_recovered", "trigger": "RECOVER"}
    dec_thr = float(config.get("ttm_v2_short_downside_decay_threshold", 0.0))
    if dec_thr > 1e-12 and _fin(last.get("momentum_1_z")) > dec_thr:
        return {"reason": "ttm_exit_short_downside_momentum_decay", "trigger": "DECAY"}
    return None


def check_exhaustion_short_exit(
    meta: Optional[Mapping[str, Any]],
    current_price: float,
    holding_bars: int,
) -> Optional[Dict[str, Any]]:
    if not is_short_exit_meta(meta):
        return None
    px = _to_finite_float(current_price)
    if px is None:
        return None
    sl = _to_finite_float((meta or {}).get("short_stop_loss_price"))
    tp = _to_finite_float((meta or {}).get("short_take_profit_price"))
    max_bars = max(1, int((meta or {}).get("short_time_stop_bars", 5)))
    if sl is not None and px >= sl:
        return {"reason": "ttm_exit_stop_loss_short_atr", "trigger": "SL"}
    if tp is not None and px <= tp:
        return {"reason": "ttm_exit_take_profit_short_atr", "trigger": "TP"}
    if int(holding_bars) >= max_bars:
        return {"reason": "ttm_exit_max_bars_short_atr", "trigger": "TIME"}
    return None
