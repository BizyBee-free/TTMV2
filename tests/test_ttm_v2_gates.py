"""TTM V2 gate modes: strict vs tiered research vs shadow diagnostics."""

from __future__ import annotations

import numpy as np

from src.strategies.ttm.config import (
    TTM_CONFIG,
    build_ttm_paper_live_config,
    build_ttm_research_parallel_config,
)
from src.strategies.ttm.ttm_features import features_last_row
from src.strategies.ttm.ttm_signal_v2 import generate_ttm_signal_v2
from src.strategies.ttm.ttm_v2_gates import (
    GATE_MODE_EXPLORATORY_RESEARCH,
    GATE_MODE_QUALITY_RESEARCH,
    GATE_MODE_RESEARCH_PAPER,
    GATE_MODE_STRICT,
    ResearchGateSessionState,
    build_gate_diagnostics,
    entry_confirm_mode_for_gate,
    eval_exploratory_research_long_allowed,
    eval_long_candidate_universe,
    eval_quality_research_long_allowed,
    eval_strict_long_allowed,
    long_gate_for_trading,
    normalize_gate_mode,
    resolve_gate_mode,
    resolve_trading_gate_mode,
)


def _feats_with_breakout(n: int = 30, *, score_neg: bool = True) -> dict:
    bu = np.zeros(n, dtype=bool)
    bu[-1] = True
    raw = np.full(n, -2.5 if score_neg else 1.5)
    return {
        "n": n,
        "close": np.linspace(100.0, 103.0, n),
        "breakout_up": bu,
        "breakout_strength_base": np.full(n, 0.8),
        "raw_strength": raw,
        "effective_strength": np.full(n, 3.0 if not score_neg else -3.0),
        "momentum_1_z": np.zeros(n),
        "basis_signal": np.zeros(n),
        "basis_norm": np.zeros(n),
        "price_z": np.zeros(n),
        "extension": np.full(n, 1.2),
        "last_bar_return": np.zeros(n),
        "failure_depth_up": np.zeros(n),
        "short_score": np.zeros(n),
        "exhaustion_confirm": np.zeros(n, dtype=bool),
    }


def test_resolve_gate_mode_live_forces_strict() -> None:
    cfg = {**TTM_CONFIG, "ttm_v2_gate_mode": GATE_MODE_QUALITY_RESEARCH}
    assert resolve_gate_mode(cfg, live_mode=True) == GATE_MODE_STRICT


def test_research_paper_alias_maps_to_quality() -> None:
    assert normalize_gate_mode(GATE_MODE_RESEARCH_PAPER) == GATE_MODE_QUALITY_RESEARCH


def test_research_parallel_config_uses_quality_research_mode() -> None:
    cfg = build_ttm_research_parallel_config()
    assert cfg["ttm_v2_gate_mode"] == "quality_research"
    assert float(cfg["ttm_v2_research_score_floor_abs"]) == -0.50


def test_paper_live_config_quality_execution_exploratory_shadow() -> None:
    cfg = build_ttm_paper_live_config()
    assert cfg["ttm_config_profile"] == "paper_live"
    assert cfg["ttm_v2_gate_mode"] == "quality_research"
    assert resolve_trading_gate_mode(cfg, live_mode=False) == GATE_MODE_QUALITY_RESEARCH
    feats = _feats_with_breakout(score_neg=False)
    sig = generate_ttm_signal_v2(feats, cfg, live_mode=False)
    gd = (sig.get("debug") or {}).get("gate_diagnostics") or {}
    assert gd.get("gate_mode") == GATE_MODE_QUALITY_RESEARCH
    assert "exploratory_research_long_allowed" in gd
    assert gd.get("research_entry_source") in (None, "quality_research")


def test_strict_blocks_negative_score_research_allows_with_percentile() -> None:
    last = {
        "breakout_up": True,
        "breakout_up_filtered_last": True,
        "exhaustion_confirm": False,
        "ttm_v2_persist_last_upside_breakout_bar_index": 10,
    }
    comp = {"crowd_phase": "ignition", "late_fomo_flag": False}
    cfg = {**TTM_CONFIG, "ttm_v2_research_score_floor_abs": -0.50}
    sl = -0.1
    ok_strict, _ = eval_strict_long_allowed(
        last=last, components=comp, cfg=cfg, score_long=float(sl)
    )
    state = ResearchGateSessionState()
    ok_qual, _, ctx = eval_quality_research_long_allowed(
        last=last,
        components=comp,
        cfg=cfg,
        score_long=float(sl),
        bar_index=10,
        session_state=state,
    )
    assert not ok_strict
    assert ok_qual
    assert ctx.get("long_candidate")


def test_long_candidate_includes_recent_ignition_without_current_breakout() -> None:
    last = {
        "breakout_up": False,
        "breakout_up_filtered_last": False,
        "exhaustion_confirm": False,
        "ttm_v2_persist_last_upside_breakout_bar_index": 8,
        "ttm_v2_persist_last_upside_breakout_phase": "ignition",
        "ttm_v2_persist_last_upside_breakout_extension": 1.5,
        "last_bar_return": 0.0001,
        "extension": 1.2,
    }
    comp = {"crowd_phase": "no_breakout", "late_fomo_flag": False, "extension": 1.2}
    cfg = {**TTM_CONFIG, "ttm_v2_research_candidate_window_bars": 4}
    cand = eval_long_candidate_universe(last=last, components=comp, cfg=cfg, bar_index=10)
    assert cand["recent_ignition_within_n_bars"]
    assert cand["pullback_after_ignition"]
    assert cand["long_candidate"]


def test_exploratory_wider_than_quality_with_same_scores() -> None:
    last = {
        "breakout_up": True,
        "breakout_up_filtered_last": True,
        "exhaustion_confirm": False,
        "ttm_v2_persist_last_upside_breakout_bar_index": 0,
    }
    comp = {"crowd_phase": "ignition", "late_fomo_flag": False}
    cfg = {**TTM_CONFIG, "ttm_v2_research_score_floor_abs": -0.50}
    state = ResearchGateSessionState()
    for i, sc in enumerate([-0.8, -0.6, -0.4, -0.2]):
        eval_quality_research_long_allowed(
            last=last,
            components=comp,
            cfg=cfg,
            score_long=float(sc),
            bar_index=i,
            session_state=state,
        )
    ok_q, _, _ = eval_quality_research_long_allowed(
        last=last, components=comp, cfg=cfg, score_long=-0.55, bar_index=4, session_state=state
    )
    ok_e, _, _ = eval_exploratory_research_long_allowed(
        last=last, components=comp, cfg=cfg, score_long=-0.55, bar_index=4, session_state=state
    )
    assert ok_e or ok_q
    assert ok_e >= ok_q


def test_gate_diagnostics_present_on_signal() -> None:
    feats = _feats_with_breakout(score_neg=False)
    cfg = build_ttm_research_parallel_config()
    sig = generate_ttm_signal_v2(feats, cfg, live_mode=False)
    dbg = sig.get("debug") or {}
    gd = dbg.get("gate_diagnostics") or {}
    assert "strict_long_allowed" in gd
    assert "quality_research_long_allowed" in gd
    assert "exploratory_research_long_allowed" in gd
    assert "long_candidate" in gd
    assert gd.get("entry_confirm_mode") == entry_confirm_mode_for_gate(GATE_MODE_QUALITY_RESEARCH)


def test_live_adaptive_config_strict_gate_mode() -> None:
    from src.strategies.ttm.config import build_ttm_live_adaptive_config

    cfg = build_ttm_live_adaptive_config()
    assert cfg["ttm_v2_gate_mode"] == "strict"


def test_research_prob_bypass_allows_low_prob_candidate() -> None:
    from src.strategies.ttm.ttm_v2_gates import build_research_entry_diagnostics

    gate_diag = {
        "quality_research_long_allowed": True,
        "exploratory_research_long_allowed": True,
        "hard_block_long": False,
        "strict_long_allowed": False,
    }
    cfg = {
        **TTM_CONFIG,
        "ttm_v2_research_use_prob_gate": True,
        "ttm_v2_research_allow_candidate_without_prob_gate": True,
        "ttm_v2_quality_research_prob_floor": 0.20,
    }
    out = build_research_entry_diagnostics(
        cfg=cfg,
        gate_mode=GATE_MODE_QUALITY_RESEARCH,
        gate_diag=gate_diag,
        prob_long=0.05,
    )
    assert out["research_prob_gate_passed"]
    assert out["research_prob_gate_bypassed"]
    assert out["research_long_entry_allowed"]
    assert out["research_entry_source"] == "quality_research"


def test_strict_prob_gate_unchanged() -> None:
    from src.strategies.ttm.ttm_v2_gates import eval_research_prob_gate

    cfg = {**TTM_CONFIG, "entry_threshold": 0.6}
    ok, bypass, thr, _ = eval_research_prob_gate(
        cfg=cfg,
        gate_mode=GATE_MODE_STRICT,
        prob_long=0.62,
        tier_allowed=True,
        live_mode=False,
    )
    assert ok and not bypass and thr == 0.6
    ok2, _, _, br = eval_research_prob_gate(
        cfg=cfg,
        gate_mode=GATE_MODE_STRICT,
        prob_long=0.10,
        tier_allowed=True,
        live_mode=False,
    )
    assert not ok2 and br == "blocked_by_prob_gate"


def test_research_trade_cap_blocks_after_limit() -> None:
    state = ResearchGateSessionState()
    cfg = {
        **TTM_CONFIG,
        "ttm_v2_max_research_trades_per_day": 1,
        "ttm_v2_min_bars_between_research_trades": 0,
    }
    state.record_research_trade(side="LONG", bar_index=1)
    ok, reason = state.research_trade_allowed(side="LONG", bar_index=2, cfg=cfg)
    assert not ok
    assert reason == "research_max_trades_per_day"
