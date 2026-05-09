"""TTM V2 crowd / short phase labels (string constants; stub classifiers until refactor commits 2–3)."""

from __future__ import annotations

from typing import Any, Mapping

CROWD_PHASES = [
    "no_breakout",
    "ignition",
    "early_continuation",
    "late_fomo",
    "exhaustion",
    "failed_breakout",
]

SHORT_PHASES = [
    "no_short_context",
    "crowded_long_watch",
    "short_setup",
    "short_trigger",
    "short_chase_risk",
    "short_invalid",
]


def classify_crowd_phase(features: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Crowd phase: uses scorer output when present, else lightweight breakout heuristics."""
    cp = features.get("crowd_phase")
    if isinstance(cp, str) and cp:
        return cp
    if bool(config.get("ttm_v2_use_effective_strength_v3")):
        return "no_breakout"
    valid = bool(features.get("breakout_up_filtered_last", features.get("breakout_up")))
    if not valid:
        return "no_breakout"
    if bool(features.get("exhaustion_confirm")) or bool(features.get("exhaustion_candidate")):
        return "exhaustion"
    return "ignition"


def classify_short_phase(features: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Uses scorer output when present; else minimal context from features."""
    sp = features.get("short_phase")
    if isinstance(sp, str) and sp:
        return sp
    if bool(config.get("ttm_v2_use_short_effective_strength_v3")):
        return "no_short_context"
    if bool(features.get("exhaustion_confirm")):
        return "short_trigger"
    return "no_short_context"
