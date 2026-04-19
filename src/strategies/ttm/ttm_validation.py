"""
TTM V2 validation against four hypotheses using paper / backtest JSONL logs.

- Decisions: ``event_type == "decision"`` (parallel runner format).
- Trades: ``event_type == "trade"``, filter ``model == "v2"`` for adaptive block.
  Prefer one row per round-trip: ``event == "CLOSED"`` with ``entry_*`` / ``exit_*`` / ``position_size``;
  legacy ``ENTRY`` + ``EXIT`` pairs are still accepted.

**Close prices (required):** supply ``closes_path`` from real OHLC source
(JSON list of floats, or JSON ``{bar_index: close}``, or CSV ``bar_index,close``),
strictly aligned to merged ``bar_index``. Trade-price inference is disabled.

Exit code / ``overall_valid`` is False if any block fails or data are insufficient.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.strategies.ttm.config import TTM_CONFIG

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


# --- loaders -----------------------------------------------------------------


def load_decisions_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("event_type") == "decision":
            rows.append(obj)
    rows.sort(key=lambda r: int(r.get("bar_index", -1)))
    return rows


def load_trades_jsonl(path: Path, model: str = "v2") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("event_type") != "trade":
            continue
        if str(obj.get("model", "")).lower() != model.lower():
            continue
        rows.append(obj)
    rows.sort(key=lambda r: (int(r.get("bar_index", 0)), str(r.get("event", ""))))
    return rows


def _remap_trade_bar_indices(
    tt: Dict[str, Any],
    idx_map: Dict[int, int],
    bar_offset: int,
) -> Dict[str, Any]:
    """Shift ``bar_index`` / ``entry_bar_index`` / ``exit_bar_index`` when merging split sessions."""
    out = dict(tt)
    old_t = int(out.get("bar_index", -1))
    out["bar_index"] = idx_map.get(old_t, old_t + bar_offset)
    for k in ("entry_bar_index", "exit_bar_index"):
        if k in out and out[k] is not None:
            try:
                o = int(out[k])
                out[k] = idx_map.get(o, o + bar_offset)
            except (TypeError, ValueError):
                pass
    return out


def _expand_dated_jsonl_group(path: Path) -> List[Path]:
    """Resolve a single JSONL file or a day-prefix group ``..._YYYYMMDD.jsonl``.

    Examples:
      - exact file exists -> [that file]
      - ``ttm_parallel_decisions_VN30F1M_20260408.jsonl`` (not exists) ->
        all ``ttm_parallel_decisions_VN30F1M_20260408_*.jsonl`` sorted by name.
    """
    if path.is_file():
        return [path]
    m = re.match(r"^(?P<prefix>.+_\d{8})\.jsonl$", path.name)
    if not m:
        raise FileNotFoundError(f"Input not found: {path}")
    prefix = m.group("prefix")
    grp = sorted(path.parent.glob(f"{prefix}_*.jsonl"))
    if not grp:
        raise FileNotFoundError(
            f"No split files found for day prefix: {path.parent / (prefix + '_*.jsonl')}"
        )
    return grp


def _load_merged_logs(
    decisions_input: Path,
    trades_input: Path,
    *,
    model: str = "v2",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Load and merge decisions/trades from exact file or day-prefix group.

    When grouping by day, each split file often starts ``bar_index`` from 0.
    We remap bar indices by cumulative offset per file to create one continuous session.
    """
    decision_files = _expand_dated_jsonl_group(decisions_input)
    trade_files = _expand_dated_jsonl_group(trades_input)

    decisions_all: List[Dict[str, Any]] = []
    trades_all: List[Dict[str, Any]] = []
    bar_offset = 0

    n_pairs = min(len(decision_files), len(trade_files))
    for i in range(n_pairs):
        dfile = decision_files[i]
        tfile = trade_files[i]
        dec = load_decisions_jsonl(dfile)
        trd = load_trades_jsonl(tfile, model=model)

        # Build per-file mapping old->new bar index.
        idx_map: Dict[int, int] = {}
        for r in dec:
            old = int(r.get("bar_index", -1))
            new = old + bar_offset
            idx_map[old] = new
            rr = dict(r)
            rr["bar_index"] = new
            decisions_all.append(rr)

        for t in trd:
            trades_all.append(_remap_trade_bar_indices(dict(t), idx_map, bar_offset))

        # Next file starts after current file max bar.
        if idx_map:
            bar_offset = max(idx_map.values()) + 1

    meta = {
        "decision_files": [str(p) for p in decision_files],
        "trade_files": [str(p) for p in trade_files],
        "decision_files_count": len(decision_files),
        "trade_files_count": len(trade_files),
        "paired_files_count": n_pairs,
        "bar_offset_final": bar_offset,
    }
    if len(decision_files) != len(trade_files):
        meta["merge_warning"] = (
            f"decision_files ({len(decision_files)}) != trade_files ({len(trade_files)}); "
            f"used {n_pairs} paired files by sorted order."
        )

    decisions_all.sort(key=lambda r: int(r.get("bar_index", -1)))
    trades_all.sort(key=lambda r: (int(r.get("bar_index", -1)), str(r.get("event", ""))))
    return decisions_all, trades_all, meta


def _iter_dates_yyyymmdd(start_date: str, end_date: str) -> List[str]:
    sd = datetime.strptime(start_date, "%Y%m%d").date()
    ed = datetime.strptime(end_date, "%Y%m%d").date()
    if ed < sd:
        raise ValueError(f"end_date {end_date} < start_date {start_date}")
    out: List[str] = []
    d = sd
    while d <= ed:
        out.append(d.strftime("%Y%m%d"))
        d = d + timedelta(days=1)
    return out


def _load_merged_logs_multiday(
    reports_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    model: str = "v2",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], List[str]]:
    decisions_all: List[Dict[str, Any]] = []
    trades_all: List[Dict[str, Any]] = []
    warns: List[str] = []
    bar_offset = 0
    decision_files_used: List[str] = []
    trade_files_used: List[str] = []

    for ymd in _iter_dates_yyyymmdd(start_date, end_date):
        dec_files = sorted(reports_dir.glob(f"ttm_parallel_decisions_{symbol}_{ymd}_*.jsonl"))
        if not dec_files:
            warns.append(f"multiday: no decisions files for {ymd}")
        for dfile in dec_files:
            tfile = reports_dir / dfile.name.replace("decisions", "trades", 1)
            dec = load_decisions_jsonl(dfile)
            trd = load_trades_jsonl(tfile, model=model) if tfile.is_file() else []
            if not tfile.is_file():
                warns.append(f"multiday: missing trade file for {dfile.name}")

            idx_map: Dict[int, int] = {}
            for r in dec:
                old = int(r.get("bar_index", -1))
                new = old + bar_offset
                idx_map[old] = new
                rr = dict(r)
                rr["bar_index"] = new
                decisions_all.append(rr)

            for t in trd:
                trades_all.append(_remap_trade_bar_indices(dict(t), idx_map, bar_offset))

            if idx_map:
                bar_offset = max(idx_map.values()) + 1
            decision_files_used.append(str(dfile))
            if tfile.is_file():
                trade_files_used.append(str(tfile))

    decisions_all.sort(key=lambda r: int(r.get("bar_index", -1)))
    trades_all.sort(key=lambda r: (int(r.get("bar_index", 0)), str(r.get("event", ""))))
    meta = {
        "aggregate_mode": "multiday",
        "aggregate_start_date": start_date,
        "aggregate_end_date": end_date,
        "reports_dir": str(reports_dir),
        "symbol": symbol,
        "decision_files": decision_files_used,
        "trade_files": trade_files_used,
        "decision_files_count": len(decision_files_used),
        "trade_files_count": len(trade_files_used),
        "bar_offset_final": bar_offset,
    }
    return decisions_all, trades_all, meta, warns


def load_closes_file(path: Path, n_bars: int) -> np.ndarray:
    """Load length n_bars close series from JSON array, JSON dict bar->close, or CSV."""
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".csv":
        closes = np.full(n_bars, np.nan, dtype=np.float64)
        seen_idx: set[int] = set()
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                bi = int(row["bar_index"])
                if bi in seen_idx:
                    raise ValueError(f"duplicate close bar_index in CSV: {bi}")
                seen_idx.add(bi)
                closes[bi] = float(row["close"])
        return closes
    data = json.loads(text)
    if isinstance(data, list):
        arr = np.array([float(x) for x in data], dtype=np.float64)
        if len(arr) < n_bars:
            raise ValueError(f"closes list len {len(arr)} < n_bars {n_bars}")
        return arr[:n_bars]
    if isinstance(data, dict):
        closes = np.full(n_bars, np.nan, dtype=np.float64)
        seen_idx: set[int] = set()
        for k, v in data.items():
            bi = int(k)
            if bi in seen_idx:
                raise ValueError(f"duplicate close bar_index in JSON object: {bi}")
            seen_idx.add(bi)
            closes[bi] = float(v)
        return closes
    raise ValueError("closes file must be JSON array or object or CSV")


def _zero_return_ratio(closes: np.ndarray) -> float:
    c = np.asarray(closes, dtype=np.float64)
    if c.size < 2:
        return float("nan")
    prev = c[:-1]
    curr = c[1:]
    valid = np.isfinite(prev) & np.isfinite(curr) & (np.abs(prev) > 1e-12)
    if not np.any(valid):
        return float("nan")
    ret = (curr[valid] - prev[valid]) / np.abs(prev[valid])
    return float(np.mean(np.abs(ret) <= 1e-12))


def build_close_array(
    decisions: Sequence[Dict[str, Any]],
    trades: Sequence[Dict[str, Any]],
    closes_path: Optional[Path],
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    if not decisions:
        return np.array([]), ["no decision rows"], {}
    n_bars = int(max(int(d["bar_index"]) for d in decisions)) + 1
    if closes_path is None:
        raise ValueError(
            "--closes is required. Validation only accepts real OHLC closes aligned to bar_index."
        )
    if not closes_path.is_file():
        raise FileNotFoundError(f"closes path not found: {closes_path}")
    closes = load_closes_file(closes_path, n_bars)
    if len(closes) != n_bars:
        raise ValueError(
            f"closes length mismatch: expected {n_bars} bars from decisions, got {len(closes)}"
        )
    if len(decisions) != n_bars:
        raise ValueError(
            f"close/features alignment mismatch: len(close)={n_bars} != len(features)={len(decisions)}"
        )
    missing_count = int(np.sum(np.isnan(closes)))
    missing_ratio = float(missing_count / n_bars) if n_bars > 0 else 0.0
    if missing_ratio > 0.05:
        raise ValueError(
            f"missing_ratio too high: {missing_ratio:.4f} (> 0.05), "
            f"missing={missing_count}, total={n_bars}"
        )
    meta = {
        "close_source": "external_closes_file",
        "close_missing_count": missing_count,
        "close_missing_ratio": missing_ratio,
        "zero_return_ratio": _zero_return_ratio(closes),
        "total_bars": n_bars,
    }
    return closes, [], meta


def build_close_array_multiday_from_cache(
    decisions: Sequence[Dict[str, Any]],
    *,
    cache_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    warnings: List[str] = []
    ts_to_close: Dict[int, float] = {}
    cache_files_used: List[str] = []

    for ymd in _iter_dates_yyyymmdd(start_date, end_date):
        cands = sorted(cache_dir.glob(f"{symbol}_{symbol}_derivative_1_{ymd}_{ymd}_*.json"))
        if not cands:
            cands = sorted(cache_dir.glob(f"{symbol}_{symbol}_derivative_*_{ymd}_{ymd}_*.json"))
        if not cands:
            warnings.append(f"multiday_closes: missing cache file for {ymd}")
            continue
        cfile = cands[0]
        cache_files_used.append(str(cfile))
        try:
            rows = json.loads(cfile.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"multiday_closes: failed reading {cfile.name}: {exc}")
            continue
        for row in rows:
            try:
                ts = int(row.get("unix_ts"))
                close = float(row.get("close"))
                sym = str(row.get("symbol", "")).strip().upper()
            except Exception:
                continue
            if sym and sym != str(symbol).upper():
                continue
            if np.isfinite(close):
                if ts in ts_to_close:
                    raise ValueError(f"duplicate OHLC timestamp detected in cache source: {ts}")
                ts_to_close[ts] = close

    n_bars = int(max(int(d.get("bar_index", -1)) for d in decisions) + 1) if decisions else 0
    closes = np.full(n_bars, np.nan, dtype=np.float64)
    matched = 0
    missing = 0
    for d in decisions:
        bi = int(d.get("bar_index", -1))
        if bi < 0 or bi >= n_bars:
            continue
        try:
            ts = int(d.get("timestamp"))
        except Exception:
            ts = -1
        if ts in ts_to_close:
            closes[bi] = ts_to_close[ts]
            matched += 1
        else:
            missing += 1

    if n_bars > 0 and np.isnan(closes).all():
        raise ValueError("multiday_closes: no timestamps matched cache closes")
    if len(decisions) != n_bars:
        raise ValueError(
            f"close/features alignment mismatch: len(close)={n_bars} != len(features)={len(decisions)}"
        )
    missing_ratio = float(missing / n_bars) if n_bars > 0 else 0.0
    if missing_ratio > 0.05:
        raise ValueError(
            f"multiday_closes: missing_ratio too high {missing_ratio:.4f} (> 0.05), "
            f"missing={missing}, total={n_bars}"
        )
    meta = {
        "close_source": "cache_vn30f1m_derivative",
        "cache_dir": str(cache_dir),
        "symbol": symbol,
        "cache_files_used": cache_files_used,
        "cache_files_count": len(cache_files_used),
        "close_match_count": matched,
        "close_missing_count": missing,
        "close_missing_ratio": missing_ratio,
        "zero_return_ratio": _zero_return_ratio(closes),
        "total_bars": n_bars,
    }
    return closes, warnings, meta


def _forward_return(closes: np.ndarray, i: int, h: int) -> float:
    if i + h >= len(closes):
        return float("nan")
    a, b = closes[i], closes[i + h]
    if np.isnan(a) or np.isnan(b) or abs(a) < 1e-12:
        return float("nan")
    return float((b - a) / abs(a))


def _feat(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    f = d.get("features") or {}
    v = f.get(key)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _decision_has_raw_breakout_feature_keys(d: Dict[str, Any]) -> bool:
    f = d.get("features") or {}
    return any(
        k in f for k in ("is_breakout_up", "breakout_up_raw", "breakout_up_raw_last")
    )


def _is_breakout_up_from_features(d: Dict[str, Any]) -> Optional[bool]:
    """
    Raw up-breakout from JSONL if keys exist; ``None`` if this row carries no flag
    (caller may fall back to close proxy when the whole file lacks flags).
    """
    f = d.get("features") or {}
    if "is_breakout_up" in f:
        return bool(f.get("is_breakout_up"))
    if "breakout_up_raw" in f:
        return bool(f.get("breakout_up_raw"))
    if "breakout_up_raw_last" in f:
        return bool(f.get("breakout_up_raw_last"))
    return None


def _is_breakout_down_from_features(d: Dict[str, Any]) -> Optional[bool]:
    """
    Raw down-breakout from JSONL if keys exist; ``None`` if this row carries no flag.
    """
    f = d.get("features") or {}
    if "is_breakout_down" in f:
        return bool(f.get("is_breakout_down"))
    if "breakout_down_raw" in f:
        return bool(f.get("breakout_down_raw"))
    if "breakout_down_raw_last" in f:
        return bool(f.get("breakout_down_raw_last"))
    return None


def _is_breakout_up_close_proxy(closes: np.ndarray, bi: int, window: int) -> bool:
    """
    Legacy logs without raw-breakout flags: ``close[bi] > max(close[bi-window:bi])``.

    Proxy only — live features use rolling **high**; use when JSONL has no ``is_breakout_up``.
    """
    if bi < 1 or bi >= len(closes):
        return False
    c = closes[bi]
    if not np.isfinite(c):
        return False
    lo = max(0, bi - int(window))
    prev = closes[lo:bi]
    prev = prev[np.isfinite(prev)]
    if prev.size == 0:
        return False
    return float(c) > float(np.max(prev))


def _oi_level(d: Dict[str, Any]) -> float:
    f = d.get("features") or {}
    oi = f.get("oi_data") or {}
    v = oi.get("open_interest_last")
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _feature_float(d: Dict[str, Any], key: str) -> float:
    f = d.get("features") or {}
    try:
        v = f.get(key)
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _v2_score_long(d: Dict[str, Any]) -> float:
    v2 = d.get("v2") or {}
    try:
        return float(v2.get("score_long", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _v2_score_short(d: Dict[str, Any]) -> float:
    short_score = _feature_float(d, "short_score")
    if np.isfinite(short_score):
        return float(short_score)
    v2 = d.get("v2") or {}
    try:
        return float(v2.get("score_short", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


# --- statistical helpers -----------------------------------------------------


def _mann_whitney(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Returns (statistic, p_value two-sided) or (None, None) if scipy missing or too small."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return None, None
    if scipy_stats is None:
        return None, None
    try:
        r = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
        return float(r.statistic), float(r.pvalue)
    except ValueError:
        return None, None


def _pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return float("nan"), None
    if scipy_stats is not None:
        r, p = scipy_stats.pearsonr(x, y)
        return float(r), float(p)
    r = np.corrcoef(x, y)[0, 1]
    return float(r), None


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[float]]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return float("nan"), None
    if scipy_stats is not None:
        r, p = scipy_stats.spearmanr(x, y)
        return float(r), float(p)
    return float("nan"), None


# --- tests -------------------------------------------------------------------


def summarize_v2_trade_holding(trades: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Holding-period stats for V2 CLOSED rows (trades list is typically already model=v2)."""
    rows = [
        t
        for t in trades
        if str(t.get("event_type", "")).lower() == "trade"
        and str(t.get("event", "")).upper() == "CLOSED"
        and str(t.get("model", "")).lower() == "v2"
    ]
    long_rows = [t for t in rows if str(t.get("side", "")).upper() == "LONG"]
    hp_list: List[int] = []
    for t in long_rows:
        h = t.get("holding_period")
        if h is None:
            continue
        try:
            hp_list.append(int(h))
        except (TypeError, ValueError):
            continue
    if not hp_list:
        return {
            "n_v2_long_closed": 0,
            "n_v2_closed_all_sides": len(rows),
            "pct_holding_period_eq_1": None,
            "holding_period_hist": {},
        }
    hist: Dict[str, int] = {}
    for h in hp_list:
        k = str(int(h))
        hist[k] = hist.get(k, 0) + 1
    eq1 = sum(1 for h in hp_list if h == 1)
    return {
        "n_v2_long_closed": len(hp_list),
        "n_v2_closed_all_sides": len(rows),
        "pct_holding_period_eq_1": float(eq1 / len(hp_list)),
        "holding_period_hist": hist,
    }


@dataclass
class BlockResult:
    name: str
    passed: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def test_breakout(
    decisions: Sequence[Dict[str, Any]],
    closes: np.ndarray,
    *,
    forward_h: int = 4,
    alpha: float = 0.05,
    min_samples: int = 15,
    breakout_window: int = 20,
    trap_threshold: float = 0.0005,
    require_strict_triplet: Optional[bool] = None,
) -> BlockResult:
    """
    Raw up-breakout (``is_breakout_up`` / ``close > rolling_high``) vs non-breakout on
    forward returns. ``breakout_strength`` buckets and monotonicity use **only** those
    raw-breakout rows (not tradable-filtered ``breakout_up``).

    If no decision row includes raw-breakout keys, uses a **close-only** rolling proxy
    (see :func:`_is_breakout_up_close_proxy`) and logs a warning.

    Strength buckets prefer ``features.effective_strength`` when finite; otherwise
    ``breakout_strength``. Set ``require_strict_triplet=False`` (or config
    ``ttm_validation_breakout_require_strict_triplet``) only with PM sign-off.
    """
    triplet_req = (
        bool(require_strict_triplet)
        if require_strict_triplet is not None
        else bool(TTM_CONFIG.get("ttm_validation_breakout_require_strict_triplet", True))
    )
    wr: List[str] = []
    er: List[str] = []
    fwd_b: List[float] = []
    fwd_nb: List[float] = []
    breakout_flags: List[bool] = []
    true_break: List[bool] = []
    detected_traps = 0
    actual_traps = 0
    raw_up_total = 0
    valid_up_total = 0
    up_rows: List[Tuple[float, float, float, float]] = []  # (strength_axis_value, ret_2, ret_3, ret_4)
    down_rows: List[Tuple[float, float]] = []  # (breakout_strength_down, ret_4_short)
    strength_rows_used_eff = 0

    use_close_proxy = not any(_decision_has_raw_breakout_feature_keys(d) for d in decisions)
    if use_close_proxy:
        wr.append(
            "breakout: JSONL has no is_breakout_up / breakout_up_raw; "
            f"using close > max(prev closes, W={int(breakout_window)}) proxy (not rolling HIGH)"
        )

    for i, d in enumerate(decisions):
        bi = int(d["bar_index"])
        if bi != i:
            wr.append(f"bar_index mismatch at row {i}: expected {i}, got {bi}")
        f = d.get("features") or {}
        if use_close_proxy:
            is_up = _is_breakout_up_close_proxy(closes, bi, int(breakout_window))
        else:
            ex = _is_breakout_up_from_features(d)
            is_up = bool(ex) if ex is not None else False
        eff_bs = _feature_float(d, "effective_strength")
        if np.isfinite(eff_bs):
            bs = float(eff_bs)
        else:
            try:
                bs = (
                    float(f["breakout_strength"])
                    if f.get("breakout_strength") is not None
                    else float("nan")
                )
            except (TypeError, ValueError):
                bs = float("nan")
        exhaustion_candidate = bool(f.get("exhaustion_candidate"))
        fr = _forward_return(closes, bi, forward_h)
        r2 = _forward_return(closes, bi, 2)
        r3 = _forward_return(closes, bi, 3)
        r4 = _forward_return(closes, bi, 4)
        ret_4_short = -float(r4) if np.isfinite(r4) else float("nan")
        # Trap definition (requested):
        # trap = (ret_2 < -0.0005) OR (close[t+1] < rolling_high[t])
        next_close = closes[bi + 1] if (bi + 1) < len(closes) else float("nan")
        rolling_high = _feat(d, "rolling_high", default=float("nan"))
        trap_ret2 = bool(np.isfinite(r2) and (float(r2) < -abs(float(trap_threshold))))
        trap_reject_next = bool(
            np.isfinite(next_close) and np.isfinite(rolling_high) and (float(next_close) < float(rolling_high))
        )
        trap_signal = bool(trap_ret2 or trap_reject_next)
        rs = _feature_float(d, "raw_strength")
        tradable_extra = True
        if "breakout_up_filtered_last" in f:
            tradable_extra = bool(f.get("breakout_up_filtered_last"))
        valid_breakout = bool(
            is_up
            and (not trap_signal)
            and np.isfinite(rs)
            and (float(rs) > 0.0)
            and tradable_extra
        )

        # Strength buckets: only bars that pass valid_breakout filter.
        if valid_breakout and (not exhaustion_candidate) and np.isfinite(bs):
            if np.isfinite(eff_bs):
                strength_rows_used_eff += 1
            up_rows.append((float(bs), float(r2), float(r3), float(r4)))
        is_dn_ex = _is_breakout_down_from_features(d)
        is_dn = bool(is_dn_ex) if is_dn_ex is not None else False
        bsd = _feat(d, "breakout_strength_down", default=float("nan"))
        if is_dn and np.isfinite(bsd) and np.isfinite(ret_4_short):
            down_rows.append((float(bsd), float(ret_4_short)))
        if np.isnan(fr):
            continue
        if is_up:
            raw_up_total += 1
            if trap_signal:
                detected_traps += 1
            if valid_breakout:
                valid_up_total += 1
                fwd_b.append(fr)
            else:
                # Filtered trap breakouts are treated as "non-valid-breakout" bucket.
                fwd_nb.append(fr)
            true_break.append(fr > 0)
            breakout_flags.append(True)
            if trap_signal:
                actual_traps += 1
        else:
            fwd_nb.append(fr)
            breakout_flags.append(False)

    n_b = len(fwd_b)
    n_nb = len(fwd_nb)
    tbr = float(np.mean(true_break)) if true_break else float("nan")
    tdr = float(detected_traps / raw_up_total) if raw_up_total else float("nan")

    passed = True
    stat = None
    pval = None

    if tbr < 0.45 and n_b >= 10:
        wr.append(f"breakout: true_breakout_rate low ({tbr:.3f})")
    if raw_up_total >= 10 and tdr < 0.1:
        wr.append(f"breakout: trap_detection_rate weak ({tdr:.3f} on {raw_up_total} raw up-breakouts)")

    # --- Raw up-breakout + strength (buckets only within is_breakout_up bars) ---
    up_arr = np.array(up_rows, dtype=np.float64) if up_rows else np.empty((0, 4), dtype=np.float64)
    n_raw_up_strength = int(up_arr.shape[0])
    up_eval: Dict[str, Any] = {
        "n_is_breakout_up": n_raw_up_strength,
        "n_breakout_up": n_raw_up_strength,
        "ret_2_mean": None,
        "ret_3_mean": None,
        "ret_4_mean": None,
        "ret_4_winrate": None,
        "ret_4_sharpe": None,
        "bucket_table": [],
        "monotonic_low_mid_high": None,
        "strength_validation_mode": None,
        "mean_ret_4_high_minus_low": None,
        "strength_axis": (
            "effective_strength" if strength_rows_used_eff > 0 else "breakout_strength_fallback"
        ),
        "n_bucket_rows_using_effective_strength": int(strength_rows_used_eff),
        "mean_ret_low": None,
        "mean_ret_mid": None,
        "mean_ret_high": None,
        "mean_mid_ge_mean_high": None,
        "anti_inverted_u_ok": None,
    }
    if up_arr.shape[0] >= 3:
        bs = up_arr[:, 0]
        r2 = up_arr[:, 1]
        r3 = up_arr[:, 2]
        r4 = up_arr[:, 3]
        m2 = np.isfinite(r2)
        m3 = np.isfinite(r3)
        m4 = np.isfinite(r4)
        r2v = r2[m2]
        r3v = r3[m3]
        r4v = r4[m4]
        if r2v.size > 0:
            up_eval["ret_2_mean"] = float(np.mean(r2v))
        if r3v.size > 0:
            up_eval["ret_3_mean"] = float(np.mean(r3v))
        if r4v.size > 0:
            up_eval["ret_4_mean"] = float(np.mean(r4v))
            up_eval["ret_4_winrate"] = float(np.mean(r4v > 0.0))
            sd = float(np.std(r4v, ddof=0))
            up_eval["ret_4_sharpe"] = float(np.mean(r4v) / sd) if sd > 1e-12 else 0.0

        q1, q2 = np.quantile(bs, [1.0 / 3.0, 2.0 / 3.0])
        buckets: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
        for s, rr in zip(bs, r4):
            if not np.isfinite(rr):
                continue
            if s <= q1:
                buckets["low"].append(float(rr))
            elif s <= q2:
                buckets["mid"].append(float(rr))
            else:
                buckets["high"].append(float(rr))

        table: List[Dict[str, Any]] = []
        for k in ("low", "mid", "high"):
            arr = np.array(buckets[k], dtype=np.float64)
            if arr.size > 0:
                table.append(
                    {
                        "bucket": k,
                        "n": int(arr.size),
                        "mean_ret_4": float(np.mean(arr)),
                        "winrate_ret_4": float(np.mean(arr > 0.0)),
                    }
                )
            else:
                table.append(
                    {"bucket": k, "n": 0, "mean_ret_4": None, "winrate_ret_4": None}
                )
        up_eval["bucket_table"] = table

        low_arr = np.array(buckets["low"], dtype=np.float64)
        high_arr = np.array(buckets["high"], dtype=np.float64)
        mean_high = float(np.mean(high_arr)) if high_arr.size > 0 else float("nan")
        mean_low = float(np.mean(low_arr)) if low_arr.size > 0 else float("nan")
        up_eval["mean_ret_4_high_minus_low"] = (
            float(mean_high - mean_low) if np.isfinite(mean_high) and np.isfinite(mean_low) else None
        )
        if low_arr.size >= min_samples and high_arr.size >= min_samples:
            up_eval["strength_validation_mode"] = "strong_mannwhitney"
            stat, pval = _mann_whitney(high_arr, low_arr)
            if stat is None and scipy_stats is None:
                wr.append("scipy not installed; Mann-Whitney p-value unavailable")
            if pval is None:
                if mean_high <= mean_low:
                    passed = False
                    er.append(
                        "breakout_strength_test: high strength mean return <= low strength mean return "
                        "(no scipy)"
                    )
            else:
                if pval >= alpha:
                    passed = False
                    er.append(
                        "breakout_strength_test: no significant high-vs-low strength difference "
                        f"(p={pval:.4f} >= {alpha})"
                    )
                if mean_high <= mean_low:
                    passed = False
                    er.append(
                        "breakout_strength_test: significant p but high strength mean return "
                        "<= low strength mean return"
                    )
        else:
            up_eval["strength_validation_mode"] = "weak_monotonic_mean"
            wr.append(
                "breakout_strength_test: using weak validation due to limited samples "
                f"(low={int(low_arr.size)}, high={int(high_arr.size)}, min={min_samples})"
            )
            if not (np.isfinite(mean_high) and np.isfinite(mean_low)):
                passed = False
                er.append("breakout_strength_test: weak validation unavailable (missing low/high means)")
            elif mean_high <= mean_low:
                passed = False
                er.append(
                    "breakout_strength_test: weak validation failed "
                    "(require mean_ret_4(high) > mean_ret_4(low))"
                )

        means = {r["bucket"]: r["mean_ret_4"] for r in table}
        up_eval["mean_ret_low"] = means.get("low")
        up_eval["mean_ret_mid"] = means.get("mid")
        up_eval["mean_ret_high"] = means.get("high")
        mlow, mmid, mhigh = means.get("low"), means.get("mid"), means.get("high")
        up_eval["mean_mid_ge_mean_high"] = bool(
            mlow is not None
            and mmid is not None
            and mhigh is not None
            and float(mmid) >= float(mhigh)
        )
        up_eval["anti_inverted_u_ok"] = bool(up_eval["mean_mid_ge_mean_high"])
        mono = (
            means["low"] is not None
            and means["mid"] is not None
            and means["high"] is not None
            and (means["high"] > means["mid"] > means["low"])
        )
        up_eval["monotonic_low_mid_high"] = bool(mono)
        # Rule from spec: FAIL if high <= mid OR mid <= low (optional strict gate)
        if not mono:
            if triplet_req:
                passed = False
                er.append(
                    "breakout_up_eval: monotonic check failed on raw-breakout strength buckets "
                    "(require high > mid > low on mean ret_4)"
                )
            else:
                wr.append(
                    "breakout_up_eval: strict triplet check disabled "
                    "(ttm_validation_breakout_require_strict_triplet=false); not failing run on monotonic"
                )
        table_txt = " | ".join(
            [
                f"{r['bucket']}: n={r['n']}, mean={r['mean_ret_4']}, win={r['winrate_ret_4']}"
                for r in table
            ]
        )
        wr.append(f"breakout_up_eval table: {table_txt}")
    elif up_arr.shape[0] > 0:
        wr.append(
            f"breakout_up_eval: insufficient is_breakout_up samples for 3 strength buckets (n={up_arr.shape[0]})"
        )

    # --- Raw down-breakout + short strength validation ---
    dn_arr = np.array(down_rows, dtype=np.float64) if down_rows else np.empty((0, 2), dtype=np.float64)
    dn_eval: Dict[str, Any] = {
        "n_is_breakout_down": int(dn_arr.shape[0]),
        "ret_4_short_mean_return": None,
        "ret_4_short_winrate": None,
        "ret_4_short_sharpe": None,
        "bucket_table": [],
        "monotonic_low_mid_high": None,
    }
    if dn_arr.shape[0] >= 3:
        bsd = dn_arr[:, 0]
        r4s = dn_arr[:, 1]
        r4s = r4s[np.isfinite(r4s)]
        if r4s.size > 0:
            dn_eval["ret_4_short_mean_return"] = float(np.mean(r4s))
            dn_eval["ret_4_short_winrate"] = float(np.mean(r4s > 0.0))
            sd = float(np.std(r4s, ddof=0))
            dn_eval["ret_4_short_sharpe"] = float(np.mean(r4s) / sd) if sd > 1e-12 else 0.0

        q1d, q2d = np.quantile(bsd, [1.0 / 3.0, 2.0 / 3.0])
        dn_buckets: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
        for s, rr in dn_arr:
            if not np.isfinite(rr):
                continue
            if s <= q1d:
                dn_buckets["low"].append(float(rr))
            elif s <= q2d:
                dn_buckets["mid"].append(float(rr))
            else:
                dn_buckets["high"].append(float(rr))

        dn_table: List[Dict[str, Any]] = []
        for k in ("low", "mid", "high"):
            arr = np.array(dn_buckets[k], dtype=np.float64)
            if arr.size > 0:
                m = float(np.mean(arr))
                w = float(np.mean(arr > 0.0))
                s = float(np.std(arr, ddof=0))
                sh = float(m / s) if s > 1e-12 else 0.0
                dn_table.append(
                    {
                        "bucket": k,
                        "n": int(arr.size),
                        "mean_return": m,
                        "winrate": w,
                        "sharpe": sh,
                    }
                )
            else:
                dn_table.append(
                    {
                        "bucket": k,
                        "n": 0,
                        "mean_return": None,
                        "winrate": None,
                        "sharpe": None,
                    }
                )
        dn_eval["bucket_table"] = dn_table
        dn_means = {r["bucket"]: r["mean_return"] for r in dn_table}
        dn_mono = (
            dn_means["low"] is not None
            and dn_means["mid"] is not None
            and dn_means["high"] is not None
            and (dn_means["low"] < dn_means["mid"] < dn_means["high"])
        )
        dn_eval["monotonic_low_mid_high"] = bool(dn_mono)
        if not dn_mono:
            passed = False
            er.append(
                "breakout_down_eval: monotonic check failed "
                "(require low < mid < high on mean ret_4_short)"
            )
        dn_table_txt = " | ".join(
            [
                f"{r['bucket']}: n={r['n']}, mean={r['mean_return']}, win={r['winrate']}, sharpe={r['sharpe']}"
                for r in dn_table
            ]
        )
        wr.append(f"breakout_down_eval table: {dn_table_txt}")
    elif dn_arr.shape[0] > 0:
        wr.append(
            f"breakout_down_eval: insufficient is_breakout_down samples for 3 strength buckets (n={dn_arr.shape[0]})"
        )

    def _sample_metrics(arr: np.ndarray) -> Dict[str, Optional[float]]:
        if arr.size == 0:
            return {
                "n": 0,
                "mean_return": None,
                "winrate": None,
                "top_20_return_mean": None,
                "bottom_20_return_mean": None,
            }
        q80 = float(np.quantile(arr, 0.8))
        q20 = float(np.quantile(arr, 0.2))
        top = arr[arr >= q80]
        bottom = arr[arr <= q20]
        return {
            "n": int(arr.size),
            "mean_return": float(np.mean(arr)),
            "winrate": float(np.mean(arr > 0.0)),
            "top_20_return_mean": float(np.mean(top)) if top.size > 0 else None,
            "bottom_20_return_mean": float(np.mean(bottom)) if bottom.size > 0 else None,
        }

    def _return_distribution_stats(vals: Sequence[float]) -> Dict[str, Any]:
        arr = np.array(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        full = _sample_metrics(arr)
        non_zero = _sample_metrics(arr[np.abs(arr) > 1e-12])

        def _delta(k: str) -> Optional[float]:
            a = full.get(k)
            b = non_zero.get(k)
            if a is None or b is None:
                return None
            return float(b - a)

        return {
            "full_sample": full,
            "non_zero_sample": non_zero,
            "comparison": {
                "non_zero_minus_full_mean_return": _delta("mean_return"),
                "non_zero_minus_full_winrate": _delta("winrate"),
                "non_zero_minus_full_top_20_return_mean": _delta("top_20_return_mean"),
                "non_zero_minus_full_bottom_20_return_mean": _delta("bottom_20_return_mean"),
                "zero_return_count": int(full["n"] - non_zero["n"]),
            },
        }

    metrics = {
        "forward_horizon_bars": forward_h,
        "raw_breakout_source": "close_proxy" if use_close_proxy else "features",
        "breakout_window": int(breakout_window),
        "n_breakout_bars": n_b,
        "n_non_breakout_bars": n_nb,
        "n_raw_breakout_up_bars": raw_up_total,
        "n_valid_breakout_up_bars": valid_up_total,
        "raw_breakout_up_ratio": float(n_b / (n_b + n_nb)) if (n_b + n_nb) > 0 else None,
        "valid_breakout_up_ratio": float(valid_up_total / raw_up_total) if raw_up_total > 0 else None,
        "true_breakout_rate": tbr,
        "trap_detection_rate": tdr,
        "trap_threshold": abs(float(trap_threshold)),
        "total_adverse_breakouts": actual_traps,
        "detected_traps": detected_traps,
        "trap_candidates": raw_up_total,
        "mannwhitney_strength_high_vs_low_p_value": pval,
        "mannwhitney_strength_high_vs_low_stat": stat,
        "breakout_return_stats": _return_distribution_stats(fwd_b),
        "non_breakout_return_stats": _return_distribution_stats(fwd_nb),
        "breakout_up_eval": up_eval,
        "breakout_down_eval": dn_eval,
    }
    return BlockResult("breakout", passed, metrics, wr, er)


def test_positioning(
    decisions: Sequence[Dict[str, Any]],
    closes: np.ndarray,
    *,
    forward_h: int = 4,
    alpha: float = 0.05,
    min_per_cell: int = 8,
) -> BlockResult:
    wr: List[str] = []
    er: List[str] = []
    pu_oiu: List[float] = []
    pu_oid: List[float] = []
    basis_bucket: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}

    prev_close = float("nan")
    prev_oi = float("nan")

    for d in decisions:
        bi = int(d["bar_index"])
        c = closes[bi] if bi < len(closes) else float("nan")
        oi = _oi_level(d)
        bn = _feat(d, "basis_norm")
        fr = _forward_return(closes, bi, forward_h)
        if np.isnan(fr) or np.isnan(c):
            prev_close, prev_oi = c, oi
            continue
        if not np.isnan(prev_close) and not np.isnan(prev_oi) and not np.isnan(oi):
            p_up = c > prev_close
            oi_up = oi > prev_oi
            if p_up and oi_up:
                pu_oiu.append(fr)
            elif p_up and not oi_up:
                pu_oid.append(fr)
        # basis tertiles on this window
        if not np.isnan(bn):
            # global tertiles computed in second pass
            pass
        prev_close, prev_oi = c, oi

    bns = np.array([_feat(d, "basis_norm") for d in decisions], dtype=np.float64)
    bns = bns[np.isfinite(bns)]
    if len(bns) >= 9:
        q1, q2 = np.quantile(bns, [1 / 3, 2 / 3])
        for d in decisions:
            bi = int(d["bar_index"])
            bn = _feat(d, "basis_norm")
            fr = _forward_return(closes, bi, forward_h)
            if np.isnan(fr) or not np.isfinite(bn):
                continue
            if bn <= q1:
                basis_bucket["low"].append(fr)
            elif bn <= q2:
                basis_bucket["mid"].append(fr)
            else:
                basis_bucket["high"].append(fr)

    passed = True
    if len(pu_oiu) < min_per_cell or len(pu_oid) < min_per_cell:
        passed = False
        er.append(
            f"positioning: insufficient price↑ splits (PU_OIU={len(pu_oiu)}, PU_OID={len(pu_oid)}, min={min_per_cell})"
        )
    _, p_po = _mann_whitney(np.array(pu_oiu), np.array(pu_oid))
    if len(pu_oiu) >= min_per_cell and len(pu_oid) >= min_per_cell:
        if p_po is None:
            m1, m2 = float(np.mean(pu_oiu)), float(np.mean(pu_oid))
            if abs(m1 - m2) < 1e-9:
                passed = False
                er.append("positioning: PU_OIU vs PU_OID means identical (no scipy)")
        elif p_po >= alpha:
            passed = False
            er.append(f"positioning: PU_OIU vs PU_OID not significantly different (p={p_po:.4f})")

    # basis: require some spread of means across buckets
    means = {k: float(np.mean(v)) if v else float("nan") for k, v in basis_bucket.items()}
    counts = {k: len(v) for k, v in basis_bucket.items()}
    if min(counts.values()) < min_per_cell:
        passed = False
        er.append(f"positioning: basis bucket counts too small: {counts}")
    else:
        arr = [means["low"], means["mid"], means["high"]]
        if all(np.isfinite(arr)):
            if max(arr) - min(arr) < 1e-8:
                passed = False
                er.append("positioning: basis buckets have near-identical mean forward returns")

    metrics = {
        "forward_horizon_bars": forward_h,
        "mean_fwd_price_up_oi_up": float(np.mean(pu_oiu)) if pu_oiu else None,
        "mean_fwd_price_up_oi_down": float(np.mean(pu_oid)) if pu_oid else None,
        "n_price_up_oi_up": len(pu_oiu),
        "n_price_up_oi_down": len(pu_oid),
        "pu_oiu_vs_oid_mannwhitney_p": p_po,
        "basis_bucket_counts": counts,
        "basis_bucket_mean_fwd": means,
    }
    if scipy_stats is None:
        wr.append("positioning: scipy missing; some p-values omitted")
    return BlockResult("positioning", passed, metrics, wr, er)


def test_scoring(
    decisions: Sequence[Dict[str, Any]],
    closes: np.ndarray,
    *,
    forward_h: int = 4,
    alpha: float = 0.05,
    min_samples: int = 20,
    direction: str = "long",
) -> BlockResult:
    """
    Correlate V2 score with forward return aligned to that side.

    - ``long``: ``y = forward_return`` vs ``score_long`` (higher score → higher fwd).
    - ``short``: ``y = -forward_return`` vs ``score_short`` (higher short score → price drop).
    """
    dir_u = str(direction or "long").strip().lower()
    if dir_u not in ("long", "short"):
        return BlockResult(
            "scoring",
            False,
            {},
            [],
            [f"scoring: invalid direction {direction!r} (use 'long' or 'short')"],
        )

    wr: List[str] = []
    er: List[str] = []
    scores: List[float] = []
    fwds: List[float] = []
    uses_short_feature_score = False
    if dir_u == "short":
        uses_short_feature_score = any(np.isfinite(_feature_float(d, "short_score")) for d in decisions)

    for d in decisions:
        bi = int(d["bar_index"])
        if dir_u == "long":
            sc = _v2_score_long(d)
        else:
            if uses_short_feature_score:
                sc = _feature_float(d, "short_score")
                if not np.isfinite(sc):
                    continue
            else:
                sc = _v2_score_short(d)
        fr = _forward_return(closes, bi, forward_h)
        if not np.isnan(fr):
            y = float(fr) if dir_u == "long" else float(-fr)
            scores.append(sc)
            fwds.append(y)

    sx = np.array(scores, dtype=np.float64)
    fy = np.array(fwds, dtype=np.float64)
    r, p_pear = _pearson(sx, fy)
    rho, p_spear = _spearman(sx, fy)

    passed = True
    block_name = f"scoring_{dir_u}"
    label = "long" if dir_u == "long" else "short"
    neg_msg = (
        f"scoring_{label}: significantly negative Pearson r={r:.4f} (contradicts long-score edge)"
        if dir_u == "long"
        else f"scoring_{label}: significantly negative Pearson r={r:.4f} (contradicts short-score edge)"
    )

    if len(sx) < min_samples:
        passed = False
        er.append(f"scoring_{label}: insufficient samples ({len(sx)} < {min_samples})")
    elif p_pear is not None and p_pear >= alpha:
        passed = False
        er.append(
            f"scoring_{label}: Pearson correlation not significant (r={r:.4f}, p={p_pear:.4f})"
        )
    elif p_pear is None and abs(r) < 0.08:
        passed = False
        er.append(
            f"scoring_{label}: weak Pearson |r|={abs(r):.4f} (no p-value; scipy missing?)"
        )
    elif p_pear is not None and abs(r) < 0.05:
        wr.append(f"scoring_{label}: significant but very small effect |r|={abs(r):.4f}")
    if p_pear is not None and p_pear < alpha and r < 0:
        passed = False
        er.append(neg_msg)

    # bucket monotonicity: high score -> higher mean aligned return
    q1, q2 = np.quantile(sx, [1 / 3, 2 / 3]) if len(sx) >= 9 else (0.0, 0.0)
    buckets = {"low": [], "mid": [], "high": []}
    for s, f in zip(sx, fy):
        if s <= q1:
            buckets["low"].append(f)
        elif s <= q2:
            buckets["mid"].append(f)
        else:
            buckets["high"].append(f)
    means = [float(np.mean(buckets[k])) if buckets[k] else float("nan") for k in ("low", "mid", "high")]
    tie_break = abs(q2 - q1) < 1e-12
    mono_ok = tie_break or (
        len(buckets["low"]) >= 5
        and len(buckets["mid"]) >= 5
        and len(buckets["high"]) >= 5
        and means[0] <= means[1] <= means[2]
    )
    if len(sx) >= 15 and not tie_break and not mono_ok:
        passed = False
        er.append(
            f"scoring_{label}: bucket mean aligned returns not monotonic low→mid→high: {means}"
        )
    elif p_spear is not None and p_spear < alpha and rho < 0:
        passed = False
        er.append(f"scoring_{label}: Spearman negative ({rho:.4f})")

    metrics = {
        "direction": dir_u,
        "score_source": "features.short_score" if (dir_u == "short" and uses_short_feature_score) else f"v2.score_{dir_u}",
        "aligned_target": "forward_return" if dir_u == "long" else "neg_forward_return",
        "n_samples": len(sx),
        "pearson_r": r,
        "pearson_p": p_pear,
        "spearman_rho": rho,
        "spearman_p": p_spear,
        "bucket_mean_aligned_return": {"low": means[0], "mid": means[1], "high": means[2]},
    }
    if scipy_stats is None:
        wr.append("scoring: install scipy for rigorous correlation p-values")
    return BlockResult(block_name, passed, metrics, wr, er)


def test_adaptive(
    trades: Sequence[Dict[str, Any]],
    *,
    model: str = "v2",
    min_trades: int = 10,
) -> BlockResult:
    wr: List[str] = []
    er: List[str] = []
    # One row per round-trip (event CLOSED) or legacy ENTRY + EXIT pair.
    pnls: List[Tuple[int, float]] = []
    i = 0
    tlist = [t for t in trades if str(t.get("model", "")).lower() == model.lower()]
    while i < len(tlist):
        t = tlist[i]
        ev = str(t.get("event", "")).upper()
        if ev == "CLOSED":
            pnl = t.get("pnl")
            bi = int(t.get("entry_bar_index", t.get("bar_index", 0)))
            if pnl is not None:
                try:
                    pnls.append((bi, float(pnl)))
                except (TypeError, ValueError):
                    pass
            i += 1
            continue
        if t.get("event") == "ENTRY":
            if i + 1 < len(tlist) and tlist[i + 1].get("event") == "EXIT":
                ex = tlist[i + 1]
                pnl = ex.get("pnl")
                bi = int(t.get("bar_index", 0))
                if pnl is not None:
                    try:
                        pnls.append((bi, float(pnl)))
                    except (TypeError, ValueError):
                        pass
                i += 2
                continue
        i += 1

    if len(pnls) < min_trades:
        return BlockResult(
            "adaptive",
            False,
            {"n_closed_trades": len(pnls)},
            wr,
            [f"adaptive: need >= {min_trades} closed trades, got {len(pnls)}"],
        )

    pnls.sort(key=lambda x: x[0])
    values = np.array([p for _, p in pnls], dtype=np.float64)
    mid = len(values) // 2
    early, late = values[:mid], values[mid:]
    if len(early) < 3 or len(late) < 3:
        return BlockResult(
            "adaptive",
            False,
            {"n_closed_trades": len(pnls)},
            wr,
            ["adaptive: early/late split too small"],
        )

    def _stat(a: np.ndarray) -> Dict[str, float]:
        return {
            "pnl_sum": float(np.sum(a)),
            "winrate": float(np.mean(a > 0)),
            "sharpe_proxy": float(np.mean(a) / np.std(a)) if np.std(a) > 1e-12 else float("nan"),
        }

    se, sl = _stat(early), _stat(late)
    passed = True
    # improvement: late winrate or sharpe or pnl better
    improved = (
        sl["winrate"] > se["winrate"] + 0.05
        or sl["sharpe_proxy"] > se["sharpe_proxy"] + 0.05
        or sl["pnl_sum"] > se["pnl_sum"]
    )
    if not improved:
        passed = False
        er.append("adaptive: late segment not better than early on winrate/sharpe_proxy/pnl_sum")

    _, pnl_p = _mann_whitney(early, late)
    if pnl_p is not None and pnl_p >= 0.2 and not improved:
        wr.append(f"adaptive: Mann-Whitney on early vs late PnL p={pnl_p:.3f} (no strong shift)")

    metrics = {
        "n_closed_trades": len(pnls),
        "early": se,
        "late": sl,
        "mannwhitney_early_late_pnl_p": pnl_p,
    }
    return BlockResult("adaptive", passed, metrics, wr, er)


# --- orchestration -----------------------------------------------------------


def _merge_scoring_into_result(
    base: Dict[str, Any],
    *,
    scoring_mode: str,
    b_long: BlockResult,
    b_short: BlockResult,
) -> Dict[str, Any]:
    """Attach scoring block(s) per ``scoring_mode``. ``base`` must already set ``overall_valid``."""
    mode = str(scoring_mode or "long").strip().lower()
    out = dict(base)
    if mode == "long":
        out["scoring"] = b_long.as_dict()
    elif mode == "short":
        out["scoring"] = b_short.as_dict()
    elif mode in ("all", "both"):
        out["scoring_long"] = b_long.as_dict()
        out["scoring_short"] = b_short.as_dict()
        combined_passed = bool(b_long.passed and b_short.passed)
        out["scoring_combined"] = {
            "passed": combined_passed,
            "long_passed": bool(b_long.passed),
            "short_passed": bool(b_short.passed),
        }
        out["scoring"] = b_long.as_dict()
    else:
        out["scoring"] = b_long.as_dict()
        out.setdefault("global_warnings", []).append(f"unknown scoring_mode={scoring_mode!r}; using long")

    out.setdefault("meta", {})["scoring_mode"] = mode
    return out


def run_validation(
    decisions_path: Path,
    trades_path: Path,
    *,
    closes_path: Optional[Path] = None,
    forward_horizon: int = 4,
    alpha: float = 0.05,
    min_samples_breakout: int = 15,
    min_samples_positioning: int = 8,
    min_samples_scoring: int = 20,
    min_trades_adaptive: int = 10,
    breakout_window: int = 20,
    scoring_mode: str = "long",
) -> Dict[str, Any]:
    decisions, trades, merge_meta = _load_merged_logs(
        decisions_path, trades_path, model="v2"
    )
    closes, cw, close_meta = build_close_array(decisions, trades, closes_path)

    global_warnings = list(cw)
    if len(closes) == 0:
        out = {
            "breakout": BlockResult("breakout", False, {}, global_warnings, ["no closes"]).as_dict(),
            "positioning": BlockResult("positioning", False, {}, [], ["no closes"]).as_dict(),
            "scoring": BlockResult("scoring", False, {}, [], ["no closes"]).as_dict(),
            "adaptive": BlockResult("adaptive", False, {}, [], ["no closes"]).as_dict(),
            "overall_valid": False,
            "global_warnings": global_warnings,
        }
        return out

    b1 = test_breakout(
        decisions,
        closes,
        forward_h=forward_horizon,
        alpha=alpha,
        min_samples=min_samples_breakout,
        breakout_window=breakout_window,
    )
    b1.metrics["v2_trade_holding"] = summarize_v2_trade_holding(trades)
    b1.warnings.extend(global_warnings)

    b2 = test_positioning(
        decisions,
        closes,
        forward_h=forward_horizon,
        alpha=alpha,
        min_per_cell=min_samples_positioning,
    )

    b3_long = test_scoring(
        decisions,
        closes,
        forward_h=forward_horizon,
        alpha=alpha,
        min_samples=min_samples_scoring,
        direction="long",
    )
    b3_short = test_scoring(
        decisions,
        closes,
        forward_h=forward_horizon,
        alpha=alpha,
        min_samples=min_samples_scoring,
        direction="short",
    )

    b4 = test_adaptive(trades, min_trades=min_trades_adaptive)

    mode = str(scoring_mode or "long").strip().lower()
    if mode == "short":
        b3_active = b3_short
    else:
        b3_active = b3_long
    blocks = {"breakout": b1, "positioning": b2, "scoring": b3_active, "adaptive": b4}
    overall = all(b.passed for b in blocks.values())
    if mode in ("all", "both"):
        overall = overall and b3_long.passed and b3_short.passed

    # emit warnings to stderr for visibility
    for name, b in blocks.items():
        for w in b.warnings:
            print(f"[ttm_validation WARNING] {name}: {w}", file=sys.stderr)
        for e in b.errors:
            print(f"[ttm_validation ERROR] {name}: {e}", file=sys.stderr)
    if mode in ("all", "both"):
        for w in b3_short.warnings:
            print(f"[ttm_validation WARNING] scoring_short: {w}", file=sys.stderr)
        for e in b3_short.errors:
            print(f"[ttm_validation ERROR] scoring_short: {e}", file=sys.stderr)
    print(
        "[ttm_validation INFO] close_quality "
        f"missing_ratio={close_meta.get('close_missing_ratio')} "
        f"zero_return_ratio={close_meta.get('zero_return_ratio')} "
        f"total_bars={close_meta.get('total_bars')}",
        file=sys.stderr,
    )

    base_result = {
        "breakout": b1.as_dict(),
        "positioning": b2.as_dict(),
        "adaptive": b4.as_dict(),
        "overall_valid": overall,
        "global_warnings": global_warnings,
        "meta": {
            "decisions_path": str(decisions_path),
            "trades_path": str(trades_path),
            "closes_path": str(closes_path) if closes_path else None,
            "n_decisions": len(decisions),
            "n_bars_close": len(closes),
            "forward_horizon": forward_horizon,
            "alpha": alpha,
            "breakout_window": int(breakout_window),
            "scipy_available": scipy_stats is not None,
            **merge_meta,
            **close_meta,
        },
    }
    return _merge_scoring_into_result(
        base_result,
        scoring_mode=mode,
        b_long=b3_long,
        b_short=b3_short,
    )


def run_validation_multiday(
    *,
    reports_dir: Path,
    cache_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    forward_horizon: int = 4,
    alpha: float = 0.05,
    min_samples_breakout: int = 15,
    min_samples_positioning: int = 8,
    min_samples_scoring: int = 20,
    min_trades_adaptive: int = 10,
    breakout_window: int = 20,
    scoring_mode: str = "long",
    closes_path: Optional[Path] = None,
) -> Dict[str, Any]:
    decisions, trades, merge_meta, load_warnings = _load_merged_logs_multiday(
        reports_dir=reports_dir,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        model="v2",
    )
    if closes_path is not None:
        closes, close_warnings, close_meta = build_close_array(
            decisions, trades, closes_path
        )
        close_meta = {
            **close_meta,
            "close_source": "external_closes_file",
            "aggregate_closes_path": str(closes_path),
        }
    else:
        closes, close_warnings, close_meta = build_close_array_multiday_from_cache(
            decisions,
            cache_dir=cache_dir,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    global_warnings = list(load_warnings) + list(close_warnings)
    if len(closes) == 0:
        out = {
            "breakout": BlockResult("breakout", False, {}, global_warnings, ["no closes"]).as_dict(),
            "positioning": BlockResult("positioning", False, {}, [], ["no closes"]).as_dict(),
            "scoring": BlockResult("scoring", False, {}, [], ["no closes"]).as_dict(),
            "adaptive": BlockResult("adaptive", False, {}, [], ["no closes"]).as_dict(),
            "overall_valid": False,
            "global_warnings": global_warnings,
        }
        return out

    b1 = test_breakout(
        decisions,
        closes,
        forward_h=forward_horizon,
        alpha=alpha,
        min_samples=min_samples_breakout,
        breakout_window=breakout_window,
    )
    b1.metrics["v2_trade_holding"] = summarize_v2_trade_holding(trades)
    b1.warnings.extend(global_warnings)
    b2 = test_positioning(
        decisions,
        closes,
        forward_h=forward_horizon,
        alpha=alpha,
        min_per_cell=min_samples_positioning,
    )
    b3_long = test_scoring(
        decisions,
        closes,
        forward_h=forward_horizon,
        alpha=alpha,
        min_samples=min_samples_scoring,
        direction="long",
    )
    b3_short = test_scoring(
        decisions,
        closes,
        forward_h=forward_horizon,
        alpha=alpha,
        min_samples=min_samples_scoring,
        direction="short",
    )
    b4 = test_adaptive(trades, min_trades=min_trades_adaptive)

    mode = str(scoring_mode or "long").strip().lower()
    b3_active = b3_short if mode == "short" else b3_long
    blocks = {"breakout": b1, "positioning": b2, "scoring": b3_active, "adaptive": b4}
    overall = all(b.passed for b in blocks.values())
    if mode in ("all", "both"):
        overall = overall and b3_long.passed and b3_short.passed

    for name, b in blocks.items():
        for w in b.warnings:
            print(f"[ttm_validation WARNING] {name}: {w}", file=sys.stderr)
        for e in b.errors:
            print(f"[ttm_validation ERROR] {name}: {e}", file=sys.stderr)
    if mode in ("all", "both"):
        for w in b3_short.warnings:
            print(f"[ttm_validation WARNING] scoring_short: {w}", file=sys.stderr)
        for e in b3_short.errors:
            print(f"[ttm_validation ERROR] scoring_short: {e}", file=sys.stderr)
    print(
        "[ttm_validation INFO] close_quality "
        f"missing_ratio={close_meta.get('close_missing_ratio')} "
        f"zero_return_ratio={close_meta.get('zero_return_ratio')} "
        f"total_bars={close_meta.get('total_bars')}",
        file=sys.stderr,
    )

    total_n_breakout = int((b1.metrics or {}).get("n_breakout_bars", 0))
    print(f"[ttm_validation INFO] total_n_breakout={total_n_breakout}", file=sys.stderr)

    base_result = {
        "breakout": b1.as_dict(),
        "positioning": b2.as_dict(),
        "adaptive": b4.as_dict(),
        "overall_valid": overall,
        "global_warnings": global_warnings,
        "meta": {
            "aggregate_mode": "multiday",
            "reports_dir": str(reports_dir),
            "cache_dir": str(cache_dir),
            "symbol": symbol,
            "aggregate_start_date": start_date,
            "aggregate_end_date": end_date,
            "n_decisions": len(decisions),
            "n_bars_close": len(closes),
            "forward_horizon": forward_horizon,
            "alpha": alpha,
            "breakout_window": int(breakout_window),
            "scipy_available": scipy_stats is not None,
            "total_n_breakout": total_n_breakout,
            **merge_meta,
            **close_meta,
        },
    }
    return _merge_scoring_into_result(
        base_result,
        scoring_mode=mode,
        b_long=b3_long,
        b_short=b3_short,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Validate TTM V2 hypotheses from paper JSONL logs")
    p.add_argument("--decisions", required=False, type=Path)
    p.add_argument("--trades", required=False, type=Path)
    p.add_argument(
        "--closes",
        type=Path,
        default=None,
        help="JSON array, JSON dict, or CSV bar_index,close. With --aggregate-start-date/--aggregate-end-date, "
        "one array length = merged bars (see scripts/ttm_merge_multiday_closes.py).",
    )
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--symbol", type=str, default="VN30F1M")
    p.add_argument("--aggregate-start-date", type=str, default=None, help="YYYYMMDD")
    p.add_argument("--aggregate-end-date", type=str, default=None, help="YYYYMMDD")
    p.add_argument("--forward-horizon", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-breakout", type=int, default=15)
    p.add_argument("--min-positioning", type=int, default=8)
    p.add_argument("--min-scoring", type=int, default=20)
    p.add_argument("--min-trades-adaptive", type=int, default=10)
    p.add_argument(
        "--breakout-window",
        type=int,
        default=20,
        help="Rolling window for close-proxy raw breakout when JSONL lacks is_breakout_up",
    )
    p.add_argument(
        "--scoring-mode",
        type=str,
        default="long",
        choices=("long", "short", "all", "both"),
        help="Scoring hypothesis: long (score_long vs fwd), short (score_short vs -fwd), "
        "all/both (include scoring_long + scoring_short + scoring_combined; overall needs both)",
    )
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.aggregate_start_date or args.aggregate_end_date:
        if not args.aggregate_start_date or not args.aggregate_end_date:
            raise ValueError("both --aggregate-start-date and --aggregate-end-date are required")
        result = run_validation_multiday(
            reports_dir=args.reports_dir,
            cache_dir=args.cache_dir,
            symbol=args.symbol,
            start_date=args.aggregate_start_date,
            end_date=args.aggregate_end_date,
            forward_horizon=args.forward_horizon,
            alpha=args.alpha,
            min_samples_breakout=args.min_breakout,
            min_samples_positioning=args.min_positioning,
            min_samples_scoring=args.min_scoring,
            min_trades_adaptive=args.min_trades_adaptive,
            breakout_window=args.breakout_window,
            scoring_mode=args.scoring_mode,
            closes_path=args.closes,
        )
    else:
        if args.decisions is None or args.trades is None:
            raise ValueError("--decisions and --trades are required when not using aggregate date mode")
        result = run_validation(
            args.decisions,
            args.trades,
            closes_path=args.closes,
            forward_horizon=args.forward_horizon,
            alpha=args.alpha,
            min_samples_breakout=args.min_breakout,
            min_samples_positioning=args.min_positioning,
            min_samples_scoring=args.min_scoring,
            min_trades_adaptive=args.min_trades_adaptive,
            breakout_window=args.breakout_window,
            scoring_mode=args.scoring_mode,
        )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    return 0 if result["overall_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
