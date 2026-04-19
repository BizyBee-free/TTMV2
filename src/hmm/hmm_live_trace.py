"""Debug trace for HMM live: state probs, guards, feature tail, timestamps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.hmm.feature_engineer import HMMConfig
from src.hmm.regime_model import STATE_BEAR, STATE_BULL, STATE_FLAT


def feature_column_names(hmm_cfg: HMMConfig) -> List[str]:
    names = ["log_ret", "range_ratio"]
    if getattr(hmm_cfg, "use_vol_change", False):
        names.append("vol_change")
    if hmm_cfg.use_basis:
        names.append("basis_spread")
    if hmm_cfg.use_open_interest:
        names.append("oi_rel_delta")
    return names


@dataclass
class HMMTraceSnapshot:
    """One 15m tick diagnostic for live HMM (no PII)."""

    n_bars: int
    detail: str
    direction: int
    dominant_label: str
    confidence_threshold: float
    k_states: int
    use_basis: bool
    use_open_interest: bool
    # Set when detail == "ok" and model ran
    early_exit: Optional[str] = None
    proba: Optional[List[float]] = None
    state_labels: Optional[Dict[str, str]] = None
    max_prob: float = 0.0
    dominant_state_idx: int = 0
    confidence_pass: bool = False
    guard_reason: str = ""
    feature_names: List[str] = field(default_factory=list)
    feature_tail_last5: Optional[List[List[float]]] = None
    feature_tail_has_nan: bool = False
    feature_tail_has_all_zero_row: bool = False
    last_bar_unix_ts: int = 0
    last_bar_time_str: str = ""
    exec_utc_iso: str = ""
    clock_note: str = ""
    state_label_stats: Optional[str] = None

    def to_jsonl_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    def format_console(self) -> str:
        lines = [
            "--- HMM_LIVE_TRACE ---",
            f"n_bars={self.n_bars} detail={self.detail} direction={self.direction} label={self.dominant_label}",
            f"threshold={self.confidence_threshold} k={self.k_states} basis={self.use_basis} oi={self.use_open_interest}",
        ]
        if self.early_exit:
            lines.append(f"early_exit={self.early_exit}")
        if self.proba is not None:
            lines.append(f"proba={['%.4f' % p for p in self.proba]}")
        if self.state_labels:
            lines.append(f"state_labels={self.state_labels}")
        if self.state_label_stats:
            lines.append(self.state_label_stats)
        lines.append(
            f"max_prob={self.max_prob:.6f} argmax_state={self.dominant_state_idx} "
            f"confidence_pass={self.confidence_pass} guard={self.guard_reason}"
        )
        lines.append(f"last_bar_ts={self.last_bar_unix_ts} bar_time={self.last_bar_time_str}")
        lines.append(f"exec_utc={self.exec_utc_iso} {self.clock_note}")
        if self.feature_names and self.feature_tail_last5:
            lines.append("feature_tail_last5 (rows oldest->newest within tail):")
            for row in self.feature_tail_last5:
                pairs = [f"{n}={v:.6g}" for n, v in zip(self.feature_names, row)]
                lines.append("  " + " | ".join(pairs))
            lines.append(f"  tail_has_nan={self.feature_tail_has_nan} tail_all_zero_row={self.feature_tail_has_all_zero_row}")
        lines.append("--- end HMM_LIVE_TRACE ---")
        return "\n".join(lines)


def _clock_note_bar_vs_exec(ts_bar: int) -> str:
    """So sánh thời điểm nến cuối với thời điểm chạy (UTC + giờ VN cho PM).

    ``ts_bar`` là Unix epoch giây (chuẩn UTC); không có độ lệch 18h do “quy đổi sai” —
    lệch lớn thường do API/cache trả nến cũ hoặc khoảng ``from``/``to`` cắt mất phiên.
    """
    if not ts_bar:
        return "bar_unix_ts=0 (thiếu mốc thời gian)"
    now_utc = datetime.now(timezone.utc)
    skew_utc = now_utc.timestamp() - ts_bar
    try:
        from src.vn_time import get_vn_tzinfo

        vn = get_vn_tzinfo()
        bar_vn = datetime.fromtimestamp(ts_bar, tz=vn)
        exec_vn = datetime.now(vn)
        skew_vn = (exec_vn - bar_vn).total_seconds()
        note = (
            f"skew_UTC≈{skew_utc:.0f}s skew_VN≈{skew_vn:.0f}s | "
            f"nến_VN={bar_vn.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"máy_VN={exec_vn.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if abs(skew_vn) > 900:
            note = "WARN: nến quá cũ hoặc lỗi mốc API — " + note
        return note
    except Exception:
        return f"skew_UTC≈{skew_utc:.0f}s (không format được giờ VN)"


def _state_label_stats_text(
    state_means: Optional[np.ndarray],
    state_labels_map: Dict[int, str],
) -> str:
    if state_means is None or state_means.size == 0:
        return ""
    lines = ["Bảng nhãn trạng thái (mean cột log_ret theo cửa sổ fit gần nhất):"]
    for i in range(state_means.shape[0]):
        lab = state_labels_map.get(i, "?")
        m0 = float(state_means[i, 0])
        lines.append(f"  state {i} → {lab}  mean_log_ret={m0:.6f}")
    lines.append(
        "Quy tắc: xếp mean_log_ret → thấp nhất=BEAR, cao nhất=BULL, giữa=FLAT (K≥3)."
    )
    return "\n".join(lines)


def build_trace_from_ok(
    *,
    n: int,
    detail: str,
    direction: int,
    label: str,
    confidence_threshold: float,
    hmm_cfg: HMMConfig,
    proba: np.ndarray,
    state_labels_map: Dict[int, str],
    X_pred: np.ndarray,
    last_bar: Any,
    state_means: Optional[np.ndarray] = None,
) -> HMMTraceSnapshot:
    """Build trace when last_bar_signal reached scoring (detail=='ok')."""
    proba = np.asarray(proba).ravel()
    k_prob = proba.size
    dom_idx = int(np.argmax(proba)) if k_prob else 0
    max_p = float(proba[dom_idx]) if k_prob else 0.0
    conf_pass = max_p >= confidence_threshold
    lab = state_labels_map.get(dom_idx, STATE_FLAT)

    if not conf_pass:
        guard = "confidence_below_threshold"
    elif lab == STATE_FLAT or lab not in (STATE_BULL, STATE_BEAR):
        guard = "dominant_regime_not_bull_bear"
    elif lab == STATE_BULL and direction == 1:
        guard = "signal_long"
    elif lab == STATE_BEAR and direction == -1:
        guard = "signal_short"
    else:
        guard = "flat_or_internal_mismatch"

    tail_n = min(5, X_pred.shape[0])
    tail = X_pred[-tail_n:].copy() if X_pred.size else np.zeros((0, 0))
    names = feature_column_names(hmm_cfg)
    has_nan = bool(np.isnan(tail).any()) if tail.size else False
    all_zero_row = False
    if tail.size:
        all_zero_row = any(np.allclose(r, 0.0) for r in tail)

    now = datetime.now(timezone.utc)
    ts_bar = int(getattr(last_bar, "unix_ts", 0) or 0)
    note = _clock_note_bar_vs_exec(ts_bar)

    sl = {str(k): v for k, v in state_labels_map.items()}
    sl_stats = _state_label_stats_text(state_means, state_labels_map)

    return HMMTraceSnapshot(
        n_bars=n,
        detail=detail,
        direction=direction,
        dominant_label=label,
        confidence_threshold=confidence_threshold,
        k_states=hmm_cfg.k_states,
        use_basis=hmm_cfg.use_basis,
        use_open_interest=hmm_cfg.use_open_interest,
        early_exit=None,
        proba=[float(x) for x in proba.ravel()],
        state_labels=sl,
        max_prob=max_p,
        dominant_state_idx=dom_idx,
        confidence_pass=conf_pass,
        guard_reason=guard,
        feature_names=names,
        feature_tail_last5=tail.tolist() if tail.size else None,
        feature_tail_has_nan=has_nan,
        feature_tail_has_all_zero_row=all_zero_row,
        last_bar_unix_ts=ts_bar,
        last_bar_time_str=str(getattr(last_bar, "time", "")),
        exec_utc_iso=now.isoformat(),
        clock_note=note,
        state_label_stats=sl_stats or None,
    )


def trace_early_exit(
    *,
    n_bars: int,
    detail: str,
    hmm_cfg: HMMConfig,
    last_bar: Any,
    reason: str,
    confidence_threshold: float = 0.0,
) -> HMMTraceSnapshot:
    now = datetime.now(timezone.utc)
    ts_bar = int(getattr(last_bar, "unix_ts", 0) or 0) if last_bar is not None else 0
    clk = _clock_note_bar_vs_exec(ts_bar) if ts_bar else f"early_exit: {reason}"
    return HMMTraceSnapshot(
        n_bars=n_bars,
        detail=detail,
        direction=0,
        dominant_label="",
        confidence_threshold=confidence_threshold,
        k_states=hmm_cfg.k_states,
        use_basis=hmm_cfg.use_basis,
        use_open_interest=hmm_cfg.use_open_interest,
        early_exit=reason,
        last_bar_unix_ts=ts_bar,
        last_bar_time_str=str(getattr(last_bar, "time", "")) if last_bar is not None else "",
        exec_utc_iso=now.isoformat(),
        clock_note=clk,
    )


def append_trace_jsonl(path: Any, snap: HMMTraceSnapshot) -> None:
    """Append one JSON line to debug file (path: pathlib.Path)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(snap.to_jsonl_dict(), ensure_ascii=False) + "\n")
