"""TTM V2 phase constants and stub classifiers (foundation commit)."""

from __future__ import annotations

from src.strategies.ttm.ttm_v2_phases import (
    CROWD_PHASES,
    SHORT_PHASES,
    classify_crowd_phase,
    classify_short_phase,
)


def test_crowd_and_short_phase_lists() -> None:
    assert "no_breakout" in CROWD_PHASES
    assert "failed_breakout" in CROWD_PHASES
    assert "no_short_context" in SHORT_PHASES
    assert "short_invalid" in SHORT_PHASES


def test_stub_classifiers_return_defaults() -> None:
    assert classify_crowd_phase({}, {}) == "no_breakout"
    assert classify_short_phase({}, {}) == "no_short_context"
