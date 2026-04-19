from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np


def _to_finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


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


def check_exhaustion_short_exit(
    meta: Optional[Mapping[str, Any]],
    current_price: float,
    holding_bars: int,
) -> Optional[Dict[str, Any]]:
    if not is_exhaustion_short_meta(meta):
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
