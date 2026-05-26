"""TTM V2 entry gate modes: strict (live), tiered research (backtest), shadow_only (diagnostics)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from src.strategies.ttm.ttm_v2_phases import classify_crowd_phase

GATE_MODE_STRICT = "strict"
GATE_MODE_RESEARCH_PAPER = "research_paper"  # backward-compat alias -> quality_research
GATE_MODE_QUALITY_RESEARCH = "quality_research"
GATE_MODE_EXPLORATORY_RESEARCH = "exploratory_research"
GATE_MODE_SHADOW_ONLY = "shadow_only"

_RESEARCH_GATE_MODES = frozenset(
    {
        GATE_MODE_RESEARCH_PAPER,
        GATE_MODE_QUALITY_RESEARCH,
        GATE_MODE_EXPLORATORY_RESEARCH,
    }
)
_VALID_GATE_MODES = frozenset(
    {
        GATE_MODE_STRICT,
        GATE_MODE_RESEARCH_PAPER,
        GATE_MODE_QUALITY_RESEARCH,
        GATE_MODE_EXPLORATORY_RESEARCH,
        GATE_MODE_SHADOW_ONLY,
    }
)


def normalize_gate_mode(raw: str) -> str:
    mode = str(raw or GATE_MODE_STRICT).strip().lower()
    if mode == GATE_MODE_RESEARCH_PAPER:
        return GATE_MODE_QUALITY_RESEARCH
    return mode


def is_research_gate_mode(gate_mode: str) -> bool:
    return normalize_gate_mode(gate_mode) in (
        GATE_MODE_QUALITY_RESEARCH,
        GATE_MODE_EXPLORATORY_RESEARCH,
    )


def resolve_gate_mode(cfg: Mapping[str, Any], *, live_mode: bool = False) -> str:
    """Resolve active gate mode. Live always uses strict (never research silently)."""
    if live_mode:
        return GATE_MODE_STRICT
    raw = str(cfg.get("ttm_v2_gate_mode", GATE_MODE_STRICT) or GATE_MODE_STRICT).strip().lower()
    if raw not in _VALID_GATE_MODES:
        return GATE_MODE_STRICT
    return normalize_gate_mode(raw)


def resolve_trading_gate_mode(cfg: Mapping[str, Any], *, live_mode: bool = False) -> str:
    """Gate mode used for actual trade permission (shadow_only -> strict)."""
    mode = resolve_gate_mode(cfg, live_mode=live_mode)
    if mode == GATE_MODE_SHADOW_ONLY:
        return GATE_MODE_STRICT
    return mode


def entry_confirm_mode_for_gate(gate_mode: str) -> str:
    if is_research_gate_mode(gate_mode) or gate_mode == GATE_MODE_SHADOW_ONLY:
        return "research"
    return "strict"


def valid_breakout_up(last: Mapping[str, Any]) -> bool:
    """Aligned with crowd-phase / hard LONG gate (filtered breakout when present)."""
    return bool(last.get("breakout_up_filtered_last", last.get("breakout_up")))


def _valid_breakout(last: Mapping[str, Any]) -> bool:
    return valid_breakout_up(last)


def research_score_long_floor(cfg: Mapping[str, Any]) -> float:
    """Config-driven research absolute fallback floor."""
    base = float(cfg.get("ttm_v2_research_score_floor_abs", cfg.get("ttm_v2_research_score_long_floor", -0.50)))
    if bool(cfg.get("ttm_v2_use_effective_strength_v3")):
        return float(cfg.get("ttm_v2_research_score_long_floor_when_v3", base))
    return base


def _fin(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def _bars_since_persisted_breakout(last: Mapping[str, Any], bar_index: Optional[int]) -> Optional[int]:
    raw = last.get("ttm_v2_persist_last_upside_breakout_bar_index")
    if raw is None or bar_index is None:
        return None
    try:
        ix = int(raw)
        bi = int(bar_index)
    except (TypeError, ValueError):
        return None
    if ix < 0 or ix > bi:
        return None
    return int(bi) - int(ix)


def eval_data_stale(last: Mapping[str, Any], components: Mapping[str, Any]) -> bool:
    if bool(last.get("data_stale")) or bool(components.get("data_stale")):
        return True
    vm = last.get("feature_valid_mask") or {}
    if isinstance(vm, Mapping):
        for key in ("close", "breakout_up"):
            if key in vm and not bool(vm.get(key)):
                return True
    return False


def eval_long_hard_block(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Hard blocks that must NOT be relaxed in any research tier."""
    if eval_data_stale(last, components):
        return True, "data_stale"
    cp = str(components.get("crowd_phase") or classify_crowd_phase(last, cfg))
    if bool(components.get("late_fomo_flag")) or cp == "late_fomo":
        return True, "late_fomo_extreme"
    if bool(last.get("exhaustion_confirm")) or cp == "exhaustion":
        return True, "exhaustion_confirm"
    fb = _fin(components.get("failed_breakout_confirm"))
    if fb > 0.35 or cp == "failed_breakout":
        return True, "failed_breakout_confirm"
    adv_thr = float(cfg.get("ttm_v2_research_entry_hard_adverse_return", -0.0015))
    lbv = _fin(last.get("last_bar_return"))
    if lbv < adv_thr:
        return True, "hard_adverse_move"
    return False, ""


def eval_recent_ignition_within_window(
    *,
    last: Mapping[str, Any],
    cfg: Mapping[str, Any],
    bar_index: Optional[int],
) -> bool:
    window = max(1, int(cfg.get("ttm_v2_research_candidate_window_bars", 4)))
    age = _bars_since_persisted_breakout(last, bar_index)
    if age is None:
        return False
    if age > window:
        return False
    phase = str(last.get("ttm_v2_persist_last_upside_breakout_phase") or "")
    if age == 0:
        return _valid_breakout(last) or phase in ("ignition", "early_continuation")
    return True


def eval_pullback_after_ignition(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    bar_index: Optional[int],
) -> bool:
    if _valid_breakout(last):
        return False
    age = _bars_since_persisted_breakout(last, bar_index)
    window = max(1, int(cfg.get("ttm_v2_research_candidate_window_bars", 4)))
    if age is None or age < 1 or age > window:
        return False
    cp = str(components.get("crowd_phase") or classify_crowd_phase(last, cfg))
    if cp in ("failed_breakout", "exhaustion"):
        return False
    blocked, _ = eval_long_hard_block(last=last, components=components, cfg=cfg)
    if blocked:
        return False
    persist_ext = last.get("ttm_v2_persist_last_upside_breakout_extension")
    cur_ext = _fin(components.get("extension"), _fin(last.get("extension") or last.get("price_z")))
    ext_cooled = True
    if persist_ext is not None and np.isfinite(float(persist_ext)):
        ext_cooled = cur_ext <= float(persist_ext) * 1.08
    adv_thr = float(cfg.get("ttm_v2_research_entry_hard_adverse_return", -0.0015))
    lb_not_strongly_adverse = _fin(last.get("last_bar_return")) >= adv_thr
    return ext_cooled or lb_not_strongly_adverse


def eval_long_candidate_universe(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    bar_index: Optional[int] = None,
) -> Dict[str, bool]:
    vb = _valid_breakout(last)
    recent = eval_recent_ignition_within_window(last=last, cfg=cfg, bar_index=bar_index)
    pullback = eval_pullback_after_ignition(
        last=last, components=components, cfg=cfg, bar_index=bar_index
    )
    return {
        "valid_breakout": bool(vb),
        "recent_ignition_within_n_bars": bool(recent),
        "pullback_after_ignition": bool(pullback),
        "long_candidate": bool(vb or recent or pullback),
    }


def eval_strict_long_allowed(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    score_long: float,
) -> Tuple[bool, str]:
    cp = str(components.get("crowd_phase") or classify_crowd_phase(last, cfg))
    blocked, br = eval_long_hard_block(last=last, components=components, cfg=cfg)
    if blocked:
        return False, br
    if not _valid_breakout(last):
        return False, "long_gate_no_valid_breakout"
    if cp not in ("ignition", "early_continuation"):
        return False, "long_gate_crowd_phase"
    thr = float(cfg.get("ttm_v2_score_long_entry_threshold", 0.0))
    if float(score_long) <= thr:
        return False, "score_long_below_threshold"
    return True, ""


def _percentile_threshold(scores: List[float], q: float) -> Optional[float]:
    if not scores:
        return None
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def _score_percentile_rank(scores: List[float], score: float) -> Optional[float]:
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr <= float(score)))


class ResearchGateSessionState:
    """Session rolling candidate score distributions + research trade frequency caps."""

    def __init__(self) -> None:
        self._long_candidate_scores: List[float] = []
        self._short_candidate_scores: List[float] = []
        self._session_date: Optional[str] = None
        self._research_trades_today: int = 0
        self._research_long_trades_today: int = 0
        self._research_short_trades_today: int = 0
        self._last_research_trade_bar: Optional[int] = None

    def reset_session_if_new_day(self, session_date: Optional[str]) -> None:
        sd = str(session_date or "").strip()
        if not sd:
            return
        if self._session_date == sd:
            return
        self._session_date = sd
        self._long_candidate_scores = []
        self._short_candidate_scores = []
        self._research_trades_today = 0
        self._research_long_trades_today = 0
        self._research_short_trades_today = 0
        self._last_research_trade_bar = None

    def long_percentile_context(
        self,
        *,
        score_long: float,
        is_long_candidate: bool,
        cfg: Mapping[str, Any],
    ) -> Dict[str, Any]:
        p40_q = float(cfg.get("ttm_v2_rolling_candidate_score_percentile_40", 0.40))
        p60_q = float(cfg.get("ttm_v2_rolling_candidate_score_percentile_60", 0.60))
        prior = list(self._long_candidate_scores)
        p40 = _percentile_threshold(prior, p40_q)
        p60 = _percentile_threshold(prior, p60_q)
        pct_rank = _score_percentile_rank(prior, float(score_long)) if prior else None
        fallback_used = False
        floor = research_score_long_floor(cfg)
        min_prior = int(cfg.get("ttm_v2_research_percentile_min_prior_candidates", 3))
        if len(prior) < min_prior:
            fallback_used = True
            if p40 is None:
                p40 = floor
            if p60 is None:
                p60 = floor
        if is_long_candidate:
            self._long_candidate_scores.append(float(score_long))
        return {
            "candidate_score_p40": p40,
            "candidate_score_p60": p60,
            "score_long_candidate_percentile": pct_rank,
            "fallback_used": bool(fallback_used),
            "research_score_floor_abs": float(floor),
        }

    def short_percentile_context(
        self,
        *,
        short_score: float,
        is_short_candidate: bool,
        cfg: Mapping[str, Any],
    ) -> Dict[str, Any]:
        p40_q = float(cfg.get("ttm_v2_rolling_candidate_score_percentile_40", 0.40))
        p60_q = float(cfg.get("ttm_v2_rolling_candidate_score_percentile_60", 0.60))
        prior = list(self._short_candidate_scores)
        p40 = _percentile_threshold(prior, p40_q)
        p60 = _percentile_threshold(prior, p60_q)
        pct_rank = _score_percentile_rank(prior, float(short_score)) if prior else None
        fallback_used = False
        floor = float(cfg.get("ttm_v2_research_short_score_floor", 0.0))
        min_prior = int(cfg.get("ttm_v2_research_percentile_min_prior_candidates", 3))
        if len(prior) < min_prior:
            fallback_used = True
            if p40 is None:
                p40 = floor
            if p60 is None:
                p60 = floor
        if is_short_candidate:
            self._short_candidate_scores.append(float(short_score))
        return {
            "short_candidate_score_p40": p40,
            "short_candidate_score_p60": p60,
            "short_score_candidate_percentile": pct_rank,
            "short_fallback_used": bool(fallback_used),
        }

    def research_trade_allowed(
        self,
        *,
        side: str,
        bar_index: int,
        cfg: Mapping[str, Any],
    ) -> Tuple[bool, str]:
        max_total = int(cfg.get("ttm_v2_max_research_trades_per_day", 3))
        max_long = int(cfg.get("ttm_v2_max_research_long_trades_per_day", 2))
        max_short = int(cfg.get("ttm_v2_max_research_short_trades_per_day", 1))
        min_gap = int(cfg.get("ttm_v2_min_bars_between_research_trades", 3))
        su = str(side or "").strip().upper()
        if self._last_research_trade_bar is not None:
            if int(bar_index) - int(self._last_research_trade_bar) < min_gap:
                return False, "research_min_bars_between_trades"
        if self._research_trades_today >= max_total:
            return False, "research_max_trades_per_day"
        if su == "LONG" and self._research_long_trades_today >= max_long:
            return False, "research_max_long_trades_per_day"
        if su == "SHORT" and self._research_short_trades_today >= max_short:
            return False, "research_max_short_trades_per_day"
        return True, ""

    def record_research_trade(self, *, side: str, bar_index: int) -> None:
        su = str(side or "").strip().upper()
        self._research_trades_today += 1
        if su == "LONG":
            self._research_long_trades_today += 1
        elif su == "SHORT":
            self._research_short_trades_today += 1
        self._last_research_trade_bar = int(bar_index)


def eval_research_long_allowed(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    score_long: float,
    bar_index: Optional[int],
    session_state: Optional[ResearchGateSessionState],
    percentile_q: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    cand = eval_long_candidate_universe(
        last=last, components=components, cfg=cfg, bar_index=bar_index
    )
    blocked, br = eval_long_hard_block(last=last, components=components, cfg=cfg)
    pct_ctx: Dict[str, Any] = {}
    if session_state is not None:
        pct_ctx = session_state.long_percentile_context(
            score_long=float(score_long),
            is_long_candidate=bool(cand["long_candidate"]),
            cfg=cfg,
        )
    if not cand["long_candidate"]:
        return False, "not_long_candidate", {**cand, **pct_ctx}
    if blocked:
        return False, br, {**cand, **pct_ctx}
    p_key = "candidate_score_p60" if percentile_q >= 0.55 else "candidate_score_p40"
    thr = pct_ctx.get(p_key)
    if thr is None:
        thr = research_score_long_floor(cfg)
    if float(score_long) < float(thr):
        return False, "research_score_below_percentile", {**cand, **pct_ctx}
    return True, "", {**cand, **pct_ctx}


def eval_quality_research_long_allowed(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    score_long: float,
    bar_index: Optional[int] = None,
    session_state: Optional[ResearchGateSessionState] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    q = float(cfg.get("ttm_v2_rolling_candidate_score_percentile_60", 0.60))
    return eval_research_long_allowed(
        last=last,
        components=components,
        cfg=cfg,
        score_long=float(score_long),
        bar_index=bar_index,
        session_state=session_state,
        percentile_q=q,
    )


def eval_exploratory_research_long_allowed(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    score_long: float,
    bar_index: Optional[int] = None,
    session_state: Optional[ResearchGateSessionState] = None,
) -> Tuple[bool, str, Dict[str, Any]]:
    q = float(cfg.get("ttm_v2_rolling_candidate_score_percentile_40", 0.40))
    return eval_research_long_allowed(
        last=last,
        components=components,
        cfg=cfg,
        score_long=float(score_long),
        bar_index=bar_index,
        session_state=session_state,
        percentile_q=q,
    )


def eval_signal_only_long(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    prob_long: float,
    entry_threshold: float,
    bar_index: Optional[int] = None,
) -> Tuple[bool, str]:
    if float(prob_long) <= float(entry_threshold):
        return False, "signal_only_below_prob_threshold"
    cand = eval_long_candidate_universe(
        last=last, components=components, cfg=cfg, bar_index=bar_index
    )
    if not cand["long_candidate"]:
        return False, "signal_only_not_long_candidate"
    blocked, br = eval_long_hard_block(last=last, components=components, cfg=cfg)
    if blocked:
        return False, f"signal_only_{br}"
    return True, ""


def eval_long_gate(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    score_long: float,
    mode: str,
    bar_index: Optional[int] = None,
    session_state: Optional[ResearchGateSessionState] = None,
) -> Tuple[bool, str]:
    """Backward-compatible eval; mode may be strict / quality / exploratory / research_paper."""
    m = normalize_gate_mode(mode)
    if m == GATE_MODE_STRICT:
        ok, reason = eval_strict_long_allowed(
            last=last, components=components, cfg=cfg, score_long=float(score_long)
        )
        return ok, reason
    if m == GATE_MODE_EXPLORATORY_RESEARCH:
        ok, reason, _ = eval_exploratory_research_long_allowed(
            last=last,
            components=components,
            cfg=cfg,
            score_long=float(score_long),
            bar_index=bar_index,
            session_state=session_state,
        )
        return ok, reason
    ok, reason, _ = eval_quality_research_long_allowed(
        last=last,
        components=components,
        cfg=cfg,
        score_long=float(score_long),
        bar_index=bar_index,
        session_state=session_state,
    )
    return ok, reason


def eval_short_research_candidate(
    *,
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    short_phase: str,
    short_chase_flag: bool,
    early_alive: float,
) -> Tuple[bool, str]:
    if not bool(components.get("prior_upside_breakout_exists")):
        return False, "no_prior_upside_breakout"
    sp = str(short_phase or "")
    if sp in ("no_short_context", "short_invalid", "short_chase_risk"):
        return False, str(components.get("short_block_reason") or f"short_phase_{sp}")
    allowed_phases = ("short_setup", "short_trigger", "post_ignition_failure")
    cp = str(components.get("crowd_phase") or "")
    if sp not in allowed_phases:
        if cp == "failed_breakout" and bool(components.get("prior_upside_breakout_exists")):
            sp = "post_ignition_failure"
        else:
            return False, str(components.get("short_block_reason") or f"short_phase_{sp}")
    cont_decay = _fin(components.get("continuation_decay"))
    decay_thr = float(cfg.get("ttm_v2_short_downside_decay_threshold", 0.0))
    if cont_decay <= max(0.12, decay_thr):
        return False, "continuation_decay_insufficient"
    if short_chase_flag:
        return False, "short_chase_risk"
    alive_cut = float(cfg.get("ttm_v2_short_continuation_recovery_threshold", 0.5))
    if float(early_alive) > alive_cut:
        return False, "early_continuation_still_alive"
    return True, ""


def eval_short_gate_strict(
    *,
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    short_phase: str,
    short_score: float,
    short_chase_flag: bool,
    early_alive: float,
) -> Tuple[bool, str]:
    short_entry_thr = float(cfg.get("ttm_v2_short_entry_threshold", 0.0))
    alive_cut = float(cfg.get("ttm_v2_short_continuation_recovery_threshold", 0.5))
    if not bool(components.get("prior_upside_breakout_exists")):
        return False, "no_prior_upside_breakout"
    if str(short_phase) != "short_trigger":
        return False, str(components.get("short_block_reason") or f"short_phase_{short_phase}")
    if not bool(components.get("short_candidate")):
        return False, "short_not_candidate"
    if float(short_score) <= short_entry_thr:
        return False, "below_trigger_threshold"
    if short_chase_flag:
        return False, "short_chase_risk"
    if float(early_alive) > alive_cut:
        return False, "long_continuation_recovered"
    return True, ""


def eval_short_gate_research(
    *,
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    short_phase: str,
    short_score: float,
    short_chase_flag: bool,
    early_alive: float,
    session_state: Optional[ResearchGateSessionState] = None,
    exploratory: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok_cand, br = eval_short_research_candidate(
        components=components,
        cfg=cfg,
        short_phase=str(short_phase),
        short_chase_flag=bool(short_chase_flag),
        early_alive=float(early_alive),
    )
    pct_ctx: Dict[str, Any] = {}
    if session_state is not None:
        pct_ctx = session_state.short_percentile_context(
            short_score=float(short_score),
            is_short_candidate=bool(ok_cand),
            cfg=cfg,
        )
    if not ok_cand:
        return False, br, pct_ctx
    p_key = "short_candidate_score_p40" if exploratory else "short_candidate_score_p60"
    thr = pct_ctx.get(p_key)
    if thr is None:
        thr = float(cfg.get("ttm_v2_research_short_score_floor", 0.0))
    if float(short_score) < float(thr):
        return False, "research_short_score_below_percentile", pct_ctx
    return True, "", pct_ctx


def build_gate_diagnostics(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    score_long: float,
    short_phase: str,
    short_score: float,
    short_chase_flag: bool,
    early_alive: float,
    prob_long: float,
    entry_threshold: float,
    live_mode: bool = False,
    bar_index: Optional[int] = None,
    session_state: Optional[ResearchGateSessionState] = None,
    session_date: Optional[str] = None,
) -> Dict[str, Any]:
    if session_state is not None:
        session_state.reset_session_if_new_day(session_date)

    gate_mode = resolve_gate_mode(cfg, live_mode=live_mode)
    cp = str(components.get("crowd_phase") or classify_crowd_phase(last, cfg))
    cand = eval_long_candidate_universe(
        last=last, components=components, cfg=cfg, bar_index=bar_index
    )
    hard_block, hard_block_reason = eval_long_hard_block(last=last, components=components, cfg=cfg)

    strict_ok, strict_br = eval_strict_long_allowed(
        last=last, components=components, cfg=cfg, score_long=float(score_long)
    )
    qual_ok, qual_br, qual_ctx = eval_quality_research_long_allowed(
        last=last,
        components=components,
        cfg=cfg,
        score_long=float(score_long),
        bar_index=bar_index,
        session_state=session_state,
    )
    expl_ok, expl_br, expl_ctx = eval_exploratory_research_long_allowed(
        last=last,
        components=components,
        cfg=cfg,
        score_long=float(score_long),
        bar_index=bar_index,
        session_state=session_state,
    )
    sig_ok, sig_br = eval_signal_only_long(
        last=last,
        components=components,
        cfg=cfg,
        prob_long=float(prob_long),
        entry_threshold=float(entry_threshold),
        bar_index=bar_index,
    )
    strict_short_ok, strict_short_br = eval_short_gate_strict(
        components=components,
        cfg=cfg,
        short_phase=str(short_phase),
        short_score=float(short_score),
        short_chase_flag=bool(short_chase_flag),
        early_alive=float(early_alive),
    )
    qual_short_ok, qual_short_br, qual_short_ctx = eval_short_gate_research(
        components=components,
        cfg=cfg,
        short_phase=str(short_phase),
        short_score=float(short_score),
        short_chase_flag=bool(short_chase_flag),
        early_alive=float(early_alive),
        session_state=session_state,
        exploratory=False,
    )
    expl_short_ok, expl_short_br, expl_short_ctx = eval_short_gate_research(
        components=components,
        cfg=cfg,
        short_phase=str(short_phase),
        short_score=float(short_score),
        short_chase_flag=bool(short_chase_flag),
        early_alive=float(early_alive),
        session_state=session_state,
        exploratory=True,
    )

    pct_ctx = qual_ctx or expl_ctx or {}
    research_floor = research_score_long_floor(cfg)
    research_short_floor = float(cfg.get("ttm_v2_research_short_score_floor", 0.0))

    base_diag = {
        "gate_mode": gate_mode,
        "entry_confirm_mode": entry_confirm_mode_for_gate(gate_mode),
        "long_candidate": bool(cand["long_candidate"]),
        "valid_breakout": bool(cand["valid_breakout"]),
        "recent_ignition_within_n_bars": bool(cand["recent_ignition_within_n_bars"]),
        "pullback_after_ignition": bool(cand["pullback_after_ignition"]),
        "hard_block_long": bool(hard_block),
        "hard_block_reason": hard_block_reason or None,
        "strict_long_allowed": bool(strict_ok),
        "quality_research_long_allowed": bool(qual_ok),
        "exploratory_research_long_allowed": bool(expl_ok),
        "research_long_allowed": bool(qual_ok),
        "signal_only_long_allowed": bool(sig_ok),
        "strict_short_allowed": bool(strict_short_ok),
        "quality_research_short_allowed": bool(qual_short_ok),
        "exploratory_research_short_allowed": bool(expl_short_ok),
        "research_short_allowed": bool(qual_short_ok),
        "strict_block_reason": strict_br or None,
        "quality_research_block_reason": qual_br or None,
        "exploratory_research_block_reason": expl_br or None,
        "research_block_reason": qual_br or None,
        "signal_only_block_reason": sig_br or None,
        "strict_short_block_reason": strict_short_br or None,
        "quality_research_short_block_reason": qual_short_br or None,
        "exploratory_research_short_block_reason": expl_short_br or None,
        "research_short_block_reason": qual_short_br or None,
        "score_long": float(score_long),
        "score_long_candidate_percentile": pct_ctx.get("score_long_candidate_percentile"),
        "candidate_score_p40": pct_ctx.get("candidate_score_p40"),
        "candidate_score_p60": pct_ctx.get("candidate_score_p60"),
        "percentile_fallback_used": pct_ctx.get("fallback_used"),
        "research_score_long_floor": research_floor,
        "research_short_score_floor": research_short_floor,
        "short_score_candidate_percentile": qual_short_ctx.get("short_score_candidate_percentile"),
        "short_candidate_score_p40": qual_short_ctx.get("short_candidate_score_p40"),
        "short_candidate_score_p60": qual_short_ctx.get("short_candidate_score_p60"),
        "crowd_phase": cp,
    }
    entry_diag = build_research_entry_diagnostics(
        cfg=cfg,
        gate_mode=gate_mode,
        gate_diag=base_diag,
        prob_long=float(prob_long),
        live_mode=live_mode,
        bar_index=bar_index,
        session_state=session_state,
    )
    return {**base_diag, **entry_diag}


def long_gate_for_trading(
    *,
    last: Mapping[str, Any],
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    score_long: float,
    live_mode: bool = False,
    bar_index: Optional[int] = None,
    session_state: Optional[ResearchGateSessionState] = None,
) -> Tuple[bool, str, str]:
    """Returns (allowed, block_reason, applied_gate_label). shadow_only uses strict for trades."""
    gate_mode = resolve_gate_mode(cfg, live_mode=live_mode)
    trading_mode = resolve_trading_gate_mode(cfg, live_mode=live_mode)
    if not bool(cfg.get("ttm_v2_hard_long_gate_enabled", False)):
        return True, "", "disabled"
    ok, reason = eval_long_gate(
        last=last,
        components=components,
        cfg=cfg,
        score_long=float(score_long),
        mode=trading_mode,
        bar_index=bar_index,
        session_state=session_state,
    )
    if bar_index is not None and session_state is not None and is_research_gate_mode(trading_mode):
        cap_ok, cap_reason = session_state.research_trade_allowed(
            side="LONG", bar_index=int(bar_index), cfg=cfg
        )
        if ok and not cap_ok:
            ok, reason = False, cap_reason
    label = f"{trading_mode}_long_gate"
    if ok and is_research_gate_mode(trading_mode):
        label = "research_long_gate"
    elif ok:
        label = "strict_long_gate"
    return ok, reason, label


def research_prob_floor_for_mode(cfg: Mapping[str, Any], gate_mode: str) -> float:
    mode = normalize_gate_mode(gate_mode)
    if mode == GATE_MODE_EXPLORATORY_RESEARCH:
        return float(cfg.get("ttm_v2_exploratory_research_prob_floor", 0.05))
    if is_research_gate_mode(mode):
        return float(cfg.get("ttm_v2_quality_research_prob_floor", 0.20))
    return float(cfg.get("entry_threshold", 0.6))


def tier_long_allowed_for_mode(gate_diag: Mapping[str, Any], gate_mode: str) -> bool:
    mode = normalize_gate_mode(gate_mode)
    if mode == GATE_MODE_EXPLORATORY_RESEARCH:
        return bool(gate_diag.get("exploratory_research_long_allowed"))
    if mode == GATE_MODE_QUALITY_RESEARCH:
        return bool(gate_diag.get("quality_research_long_allowed"))
    return bool(gate_diag.get("strict_long_allowed"))


def eval_research_prob_gate(
    *,
    cfg: Mapping[str, Any],
    gate_mode: str,
    prob_long: float,
    tier_allowed: bool,
    live_mode: bool = False,
) -> Tuple[bool, bool, float, Optional[str]]:
    """
    Research softmax entry gate. Returns (passed, bypassed, threshold_used, block_reason).
    Strict/live keeps entry_threshold behavior.
    """
    if live_mode or not is_research_gate_mode(gate_mode):
        thr = float(cfg.get("entry_threshold", 0.6))
        passed = float(prob_long) > thr
        return passed, False, thr, None if passed else "blocked_by_prob_gate"

    if not tier_allowed:
        return False, False, research_prob_floor_for_mode(cfg, gate_mode), "tier_gate_not_allowed"

    allow_bypass = bool(cfg.get("ttm_v2_research_allow_candidate_without_prob_gate", False))
    use_prob = bool(cfg.get("ttm_v2_research_use_prob_gate", True))
    floor = research_prob_floor_for_mode(cfg, gate_mode)
    pl = float(prob_long)

    if allow_bypass:
        bypassed = bool(use_prob and pl < floor)
        return True, bypassed, floor, None

    if not use_prob:
        thr = float(cfg.get("entry_threshold", 0.6))
        passed = pl > thr
        return passed, False, thr, None if passed else "blocked_by_prob_gate"

    if pl >= floor:
        return True, False, floor, None
    return False, False, floor, "blocked_by_prob_gate"


def build_research_entry_diagnostics(
    *,
    cfg: Mapping[str, Any],
    gate_mode: str,
    gate_diag: Mapping[str, Any],
    prob_long: float,
    live_mode: bool = False,
    bar_index: Optional[int] = None,
    session_state: Optional[ResearchGateSessionState] = None,
) -> Dict[str, Any]:
    """Prob-gate + trade-cap diagnostics for research entry (also logs blocked candidates)."""
    mode = resolve_gate_mode(cfg, live_mode=live_mode) if live_mode else normalize_gate_mode(gate_mode)
    tier_allowed = tier_long_allowed_for_mode(gate_diag, mode)
    hard_block = bool(gate_diag.get("hard_block_long"))
    prob_pass, prob_bypass, thr_used, prob_br = eval_research_prob_gate(
        cfg=cfg,
        gate_mode=mode,
        prob_long=float(prob_long),
        tier_allowed=bool(tier_allowed),
        live_mode=live_mode,
    )

    blocked_cap = False
    cap_br: Optional[str] = None
    if (
        bar_index is not None
        and session_state is not None
        and is_research_gate_mode(mode)
        and not live_mode
        and tier_allowed
        and prob_pass
        and not hard_block
    ):
        cap_ok, cap_br = session_state.research_trade_allowed(
            side="LONG", bar_index=int(bar_index), cfg=cfg
        )
        blocked_cap = not cap_ok

    entry_source: Optional[str] = None
    research_entry_allowed = False
    if live_mode or mode == GATE_MODE_STRICT:
        if tier_allowed and prob_pass and not hard_block and not blocked_cap:
            research_entry_allowed = True
            entry_source = "strict"
    elif mode == GATE_MODE_EXPLORATORY_RESEARCH:
        if tier_allowed and prob_pass and not hard_block and not blocked_cap:
            research_entry_allowed = True
            entry_source = "exploratory_research"
    elif is_research_gate_mode(mode):
        if tier_allowed and prob_pass and not hard_block and not blocked_cap:
            research_entry_allowed = True
            entry_source = "quality_research"
    elif mode == GATE_MODE_SHADOW_ONLY:
        if bool(gate_diag.get("signal_only_long_allowed")):
            entry_source = "signal_only_shadow"

    allow_bypass = bool(cfg.get("ttm_v2_research_allow_candidate_without_prob_gate", False))
    blocked_prob = bool(
        tier_allowed
        and not hard_block
        and not prob_pass
        and not (allow_bypass and tier_allowed)
    )

    return {
        "prob_long": float(prob_long),
        "prob_long_threshold_used": float(thr_used),
        "research_prob_gate_passed": bool(prob_pass),
        "research_prob_gate_bypassed": bool(prob_bypass),
        "research_entry_source": entry_source,
        "research_long_entry_allowed": bool(research_entry_allowed),
        "blocked_by_prob_gate": bool(blocked_prob),
        "blocked_by_trade_cap": bool(blocked_cap),
        "blocked_by_hard_block": bool(hard_block),
        "blocked_by_entry_confirmation": False,
        "research_prob_block_reason": prob_br,
        "research_trade_cap_block_reason": cap_br,
    }


def research_long_allowed_for_mode(gate_diag: Mapping[str, Any], gate_mode: str) -> bool:
    """Tier gate only (ignores research prob relaxation). Prefer research_long_entry_allowed for execution."""
    return tier_long_allowed_for_mode(gate_diag, gate_mode)


def research_long_entry_allowed_from_diag(gate_diag: Mapping[str, Any]) -> bool:
    return bool(gate_diag.get("research_long_entry_allowed"))


def short_signal_for_trading(
    *,
    components: Mapping[str, Any],
    cfg: Mapping[str, Any],
    short_phase: str,
    short_score: float,
    short_chase_flag: bool,
    early_alive: float,
    enable_short: bool,
    live_mode: bool = False,
    session_state: Optional[ResearchGateSessionState] = None,
    bar_index: Optional[int] = None,
) -> Tuple[bool, str]:
    if not enable_short:
        return False, "short_disabled"
    trading_mode = resolve_trading_gate_mode(cfg, live_mode=live_mode)
    if is_research_gate_mode(trading_mode):
        exploratory = normalize_gate_mode(trading_mode) == GATE_MODE_EXPLORATORY_RESEARCH
        ok, br, _ = eval_short_gate_research(
            components=components,
            cfg=cfg,
            short_phase=str(short_phase),
            short_score=float(short_score),
            short_chase_flag=bool(short_chase_flag),
            early_alive=float(early_alive),
            session_state=session_state,
            exploratory=exploratory,
        )
    else:
        ok, br = eval_short_gate_strict(
            components=components,
            cfg=cfg,
            short_phase=str(short_phase),
            short_score=float(short_score),
            short_chase_flag=bool(short_chase_flag),
            early_alive=float(early_alive),
        )
    if bar_index is not None and session_state is not None and is_research_gate_mode(trading_mode):
        cap_ok, cap_reason = session_state.research_trade_allowed(
            side="SHORT", bar_index=int(bar_index), cfg=cfg
        )
        if ok and not cap_ok:
            return False, cap_reason
    return ok, br


def record_research_trade_if_applicable(
    *,
    cfg: Mapping[str, Any],
    side: str,
    bar_index: int,
    session_state: Optional[ResearchGateSessionState],
    live_mode: bool = False,
) -> None:
    if session_state is None:
        return
    if not is_research_gate_mode(resolve_gate_mode(cfg, live_mode=live_mode)):
        return
    session_state.record_research_trade(side=str(side), bar_index=int(bar_index))
