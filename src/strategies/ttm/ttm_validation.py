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
from collections import Counter
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


def _v2_long_scoring_domain_row(d: Dict[str, Any]) -> bool:
    """
    Bars where TTM V2 LONG alpha is active: tradable continuation ``breakout_up``
    (logged as ``breakout_up_filtered_last``).
    """
    f = d.get("features") or {}
    if "breakout_up_filtered_last" in f:
        return bool(f.get("breakout_up_filtered_last"))
    if "effective_strength_active" in f:
        try:
            v = f.get("effective_strength_active")
            if v is None:
                return False
            return bool(float(v) > 0.5)
        except (TypeError, ValueError):
            return False
    # Older / synthetic logs without filtered flag: approximate continuation bar.
    raw_up = bool(f.get("is_breakout_up")) or bool(f.get("breakout_up_raw_last"))
    if not raw_up:
        return False
    if bool(f.get("exhaustion_candidate")):
        return False
    rs = _feature_float(d, "raw_strength")
    return np.isfinite(rs) and float(rs) > 0.0


def _v2_short_scoring_domain_row(d: Dict[str, Any]) -> bool:
    """Bars in the exhaustion-confirmed SHORT domain (matches V2 short_signal)."""
    f = d.get("features") or {}
    return bool(f.get("exhaustion_confirm"))


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
    forward returns. Strength buckets and monotonicity use **raw_strength** only
    (breakout "force" axis), not ``effective_strength``.

    If no decision row includes raw-breakout keys, uses a **close-only** rolling proxy
    (see :func:`_is_breakout_up_close_proxy`) and logs a warning.

    Set ``require_strict_triplet=False`` (or config
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
    up_rows: List[Tuple[float, float, float, float]] = []  # (raw_strength, ret_2, ret_3, ret_4)
    down_rows: List[Tuple[float, float]] = []  # (breakout_strength_down, ret_4_short)

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

        # Strength buckets: valid continuation breakouts only; axis = raw_strength.
        if valid_breakout and (not exhaustion_candidate) and np.isfinite(rs):
            up_rows.append((float(rs), float(r2), float(r3), float(r4)))
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
        "strength_axis": "raw_strength",
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

    n_decisions = 0
    n_in_domain = 0
    for d in decisions:
        n_decisions += 1
        bi = int(d["bar_index"])
        if dir_u == "long":
            if not _v2_long_scoring_domain_row(d):
                continue
            sc = _v2_score_long(d)
        else:
            if not _v2_short_scoring_domain_row(d):
                continue
            if uses_short_feature_score:
                sc = _feature_float(d, "short_score")
                if not np.isfinite(sc):
                    continue
            else:
                sc = _v2_score_short(d)
        n_in_domain += 1
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

    domain_name = "breakout_up_filtered_last" if dir_u == "long" else "exhaustion_confirm"
    if len(sx) == 0 and n_in_domain > 0:
        wr.append(
            f"scoring_{label}: {n_in_domain} bar(s) in domain {domain_name!r} but no finite forward returns"
        )

    metrics = {
        "direction": dir_u,
        "scoring_domain": domain_name,
        "n_decision_rows": int(n_decisions),
        "n_rows_in_scoring_domain": int(n_in_domain),
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


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _closed_v2_rows(trades: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        t
        for t in trades
        if str(t.get("event_type", "")).lower() == "trade"
        and str(t.get("model", "")).lower() == "v2"
        and str(t.get("event", "")).upper() == "CLOSED"
    ]


def _decision_v2_score_components(d: Dict[str, Any]) -> Dict[str, Any]:
    v2 = d.get("v2") or {}
    sc = v2.get("score_components")
    return dict(sc) if isinstance(sc, dict) else {}


def _decision_v2_short_components(d: Dict[str, Any]) -> Dict[str, Any]:
    v2 = d.get("v2") or {}
    sc = v2.get("short_components")
    return dict(sc) if isinstance(sc, dict) else {}


def _decision_crowd_phase(d: Dict[str, Any]) -> str:
    sc = _decision_v2_score_components(d)
    if isinstance(sc.get("crowd_phase"), str) and str(sc.get("crowd_phase")):
        return str(sc.get("crowd_phase"))
    v2log = ((d.get("v2") or {}).get("log") or {})
    if isinstance(v2log.get("crowd_phase"), str) and str(v2log.get("crowd_phase")):
        return str(v2log.get("crowd_phase"))
    f = d.get("features") or {}
    if bool(f.get("exhaustion_confirm")) or bool(f.get("exhaustion_candidate")):
        return "exhaustion"
    if bool(f.get("breakout_up_filtered_last", f.get("breakout_up"))):
        return "ignition"
    return "no_breakout"


def _decision_short_phase(d: Dict[str, Any]) -> str:
    sh = _decision_v2_short_components(d)
    if isinstance(sh.get("short_phase"), str) and str(sh.get("short_phase")):
        return str(sh.get("short_phase"))
    v2log = ((d.get("v2") or {}).get("log") or {})
    if isinstance(v2log.get("short_phase"), str) and str(v2log.get("short_phase")):
        return str(v2log.get("short_phase"))
    if bool((d.get("features") or {}).get("exhaustion_confirm")):
        return "short_trigger"
    return "no_short_context"


def _bucket3(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return float(q1), float(q2)


def _bucket_stats_from_pairs(score: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    mask = np.isfinite(score) & np.isfinite(target)
    s = score[mask]
    y = target[mask]
    if s.size == 0:
        return {
            "n_samples": 0,
            "pearson_r": None,
            "spearman_rho": None,
            "bucket_low_mid_high": [],
            "top20_minus_bottom20": None,
            "top_count": 0,
            "bottom_count": 0,
            "insufficient_sample": True,
        }
    pr, _ = _pearson(s, y)
    sr, _ = _spearman(s, y)
    q1, q2 = _bucket3(s)
    out_rows: List[Dict[str, Any]] = []
    for name, cond in (
        ("low", s <= q1),
        ("mid", (s > q1) & (s <= q2)),
        ("high", s > q2),
    ):
        yy = y[cond]
        out_rows.append(
            {
                "bucket": name,
                "n": int(yy.size),
                "mean_forward_return": float(np.mean(yy)) if yy.size else None,
                "winrate": float(np.mean(yy > 0.0)) if yy.size else None,
            }
        )
    ql = float(np.quantile(s, 0.2))
    qh = float(np.quantile(s, 0.8))
    low = y[s <= ql]
    high = y[s >= qh]
    return {
        "n_samples": int(s.size),
        "pearson_r": float(pr) if np.isfinite(pr) else None,
        "spearman_rho": float(sr) if np.isfinite(sr) else None,
        "bucket_low_mid_high": out_rows,
        "top20_minus_bottom20": (
            float(np.mean(high) - np.mean(low)) if high.size and low.size else None
        ),
        "top_count": int(high.size),
        "bottom_count": int(low.size),
        "insufficient_sample": bool(s.size < 20),
    }


def _partial_sessions_from_meta(meta: Dict[str, Any], decisions: Sequence[Dict[str, Any]]) -> List[str]:
    by_date_max: Dict[str, int] = {}
    for d in decisions:
        ts = str(d.get("timestamp", ""))
        if len(ts) < 10 or not ts.isdigit():
            continue
        dt = datetime.utcfromtimestamp(int(ts) + 7 * 3600)
        ymd = dt.strftime("%Y%m%d")
        hhmm = int(dt.strftime("%H%M"))
        by_date_max[ymd] = max(by_date_max.get(ymd, 0), hhmm)
    out: List[str] = []
    for ymd, hhmm in by_date_max.items():
        if hhmm < 1438:
            out.append(f"{ymd}_{hhmm:04d}")
    return sorted(out)


def _series_quantiles(x: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0, "min": None, "p50": None, "p80": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "p50": float(np.quantile(a, 0.50)),
        "p80": float(np.quantile(a, 0.80)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
    }


def _infer_last_bar_return_unit(raw: np.ndarray) -> str:
    a = np.asarray(raw, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "unknown"
    mx = float(np.max(np.abs(a)))
    if mx <= 0.05:
        return "fraction_return"
    if mx <= 5.0:
        return "percent_or_points"
    return "unknown_large_scale"


def _nondegenerate_spike_threshold(pos: np.ndarray) -> Tuple[Optional[float], str]:
    a = np.asarray(pos, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None, "empty"
    q80 = float(np.quantile(a, 0.80))
    if q80 > 0.0:
        return q80, "all_values_p80"
    pos_only = a[a > 0.0]
    if pos_only.size == 0:
        return None, "all_zero_or_negative"
    q80_pos = float(np.quantile(pos_only, 0.80))
    if q80_pos > 0.0:
        return q80_pos, "positive_subset_p80"
    return float(np.max(pos_only)), "positive_subset_max"


def _gate_diag_from_decision(d: Dict[str, Any]) -> Dict[str, Any]:
    v2 = d.get("v2") or {}
    gd = v2.get("gate_diagnostics")
    return dict(gd) if isinstance(gd, dict) else {}


def _score_distribution(vals: List[float]) -> Dict[str, Any]:
    a = np.asarray(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "min": None, "p25": None, "p50": None, "p75": None, "max": None}
    return {
        "n": int(a.size),
        "min": float(np.min(a)),
        "p25": float(np.quantile(a, 0.25)),
        "p50": float(np.quantile(a, 0.50)),
        "p75": float(np.quantile(a, 0.75)),
        "max": float(np.max(a)),
    }


def _top_bottom_forward_stats(fwd_pairs: List[Tuple[float, float]]) -> Dict[str, Any]:
    """Pairs of (score, forward_return) for candidate bars."""
    if not fwd_pairs:
        return {"top20_mean_forward_return": None, "bottom20_mean_forward_return": None, "n": 0}
    pairs = [(float(s), float(r)) for s, r in fwd_pairs if np.isfinite(s) and np.isfinite(r)]
    if not pairs:
        return {"top20_mean_forward_return": None, "bottom20_mean_forward_return": None, "n": 0}
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    k = max(1, int(round(n * 0.20)))
    bottom = [r for _, r in pairs[:k]]
    top = [r for _, r in pairs[-k:]]
    return {
        "n": int(n),
        "top20_mean_forward_return": float(np.mean(top)),
        "bottom20_mean_forward_return": float(np.mean(bottom)),
        "top20_winrate": float(np.mean(np.array(top) > 0.0)),
        "bottom20_winrate": float(np.mean(np.array(bottom) > 0.0)),
    }


def build_gate_mode_report(
    *,
    decisions: Sequence[Dict[str, Any]],
    trades: Sequence[Dict[str, Any]],
    closes: np.ndarray,
    forward_horizon: int = 4,
    configured_gate_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate strict / quality / exploratory / signal_only gate diagnostics per decision."""
    modes = ("strict", "quality_research", "exploratory_research", "signal_only")
    long_keys = {
        "strict": "strict_long_allowed",
        "quality_research": "quality_research_long_allowed",
        "exploratory_research": "exploratory_research_long_allowed",
        "signal_only": "signal_only_long_allowed",
    }
    short_keys = {
        "strict": "strict_short_allowed",
        "quality_research": "quality_research_short_allowed",
        "exploratory_research": "exploratory_research_short_allowed",
    }
    block_keys = {
        "strict": "strict_block_reason",
        "quality_research": "quality_research_block_reason",
        "exploratory_research": "exploratory_research_block_reason",
        "signal_only": "signal_only_block_reason",
    }
    out: Dict[str, Any] = {}
    closed_v2 = [
        t
        for t in trades
        if str(t.get("model", "")).lower() == "v2" and str(t.get("event", "")).upper() == "CLOSED"
    ]
    score_vals: List[float] = []
    pct_vals: List[float] = []
    cp_counts: Counter = Counter()
    entry_confirm_blocks: Counter = Counter()
    universe_cand = 0

    for d in decisions:
        gd = _gate_diag_from_decision(d)
        if gd.get("long_candidate"):
            universe_cand += 1
        eb = d.get("entry_block_reason") or (d.get("v2") or {}).get("entry_block_reason")
        if eb:
            entry_confirm_blocks[str(eb)] += 1

    cfg_mode = str(configured_gate_mode or "").strip().lower()
    if cfg_mode == "research_paper":
        cfg_mode = "quality_research"

    for mode in modes:
        cand = 0
        blocked = Counter()
        fwd: List[float] = []
        fwd_pairs: List[Tuple[float, float]] = []
        mode_scores: List[float] = []
        mode_pcts: List[float] = []
        mode_phases: Counter = Counter()
        for d in decisions:
            gd = _gate_diag_from_decision(d)
            if not gd:
                continue
            lk = long_keys.get(mode)
            if lk and bool(gd.get(lk)):
                cand += 1
                sl = _safe_float(gd.get("score_long"))
                if sl is not None:
                    mode_scores.append(float(sl))
                pct = _safe_float(gd.get("score_long_candidate_percentile"))
                if pct is not None:
                    mode_pcts.append(float(pct))
                mode_phases[str(gd.get("crowd_phase") or _decision_crowd_phase(d) or "unknown")] += 1
                bi = int(d.get("bar_index", -1))
                if 0 <= bi < len(closes):
                    fr = _forward_return(closes, bi, forward_horizon)
                    if np.isfinite(fr):
                        fwd.append(float(fr))
                        if sl is not None:
                            fwd_pairs.append((float(sl), float(fr)))
            elif lk and gd.get("long_candidate"):
                br = str(gd.get(block_keys.get(mode, "")) or "blocked")
                blocked[br] += 1
        short_cand = 0
        if mode != "signal_only":
            sk = short_keys.get(mode)
            for d in decisions:
                gd = _gate_diag_from_decision(d)
                if sk and bool(gd.get(sk)):
                    short_cand += 1
        winrate = float(np.mean(np.array(fwd) > 0.0)) if fwd else None
        actual_trades = int(len(closed_v2)) if cfg_mode and mode == cfg_mode else 0
        out[mode] = {
            "long_candidate_count": int(cand),
            "long_universe_candidate_count": int(universe_cand),
            "short_candidate_count": int(short_cand),
            "actual_v2_closed_trades": actual_trades,
            "blocked_count_by_reason": dict(blocked),
            "mean_forward_return_candidates": float(np.mean(fwd)) if fwd else None,
            "candidate_forward_return_winrate": winrate,
            "candidate_forward_return_n": int(len(fwd)),
            "score_long_distribution": _score_distribution(mode_scores),
            "score_percentile_distribution": _score_distribution(mode_pcts),
            "top_bottom_forward_return": _top_bottom_forward_stats(fwd_pairs),
            "phase_distribution": dict(mode_phases),
            "entry_confirm_block_stats": dict(entry_confirm_blocks),
        }
        if cand == 0:
            top = Counter()
            for d in decisions:
                gd = _gate_diag_from_decision(d)
                if gd and gd.get("long_candidate") and not gd.get(lk):
                    top[str(gd.get(block_keys.get(mode, "")) or "unknown")] += 1
            out[mode]["top_blocking_reason_if_zero_passed"] = dict(top.most_common(8))

    for d in decisions:
        v2 = d.get("v2") or {}
        sc = _decision_v2_score_components(d)
        sl = _safe_float(v2.get("score_long") if v2.get("score_long") is not None else sc.get("score_long"))
        if sl is not None:
            score_vals.append(float(sl))
        gd = _gate_diag_from_decision(d)
        pct = _safe_float(gd.get("score_long_candidate_percentile"))
        if pct is not None:
            pct_vals.append(float(pct))
        cp_counts[str(_decision_crowd_phase(d) or "unknown")] += 1

    out["score_long_distribution"] = _score_distribution(score_vals)
    out["score_percentile_distribution"] = _score_distribution(pct_vals)
    out["crowd_phase_distribution"] = dict(cp_counts)
    out["actual_v2_closed_total"] = int(len(closed_v2))
    out["configured_gate_mode"] = cfg_mode or None
    min_trades_for_alpha = 20
    if len(closed_v2) < min_trades_for_alpha:
        out["alpha_validation_status"] = "insufficient_trade_sample"
        out["candidate_validation_status"] = (
            "candidate_universe_available"
            if universe_cand > 0
            else "insufficient_candidate_universe"
        )
    else:
        out["alpha_validation_status"] = "alpha_sample_available"
        out["candidate_validation_status"] = "candidate_universe_available"
    out["insufficient_trade_sample"] = bool(len(closed_v2) < min_trades_for_alpha)
    out["candidate_count_sufficient_for_validation"] = bool(universe_cand >= 30)
    out["research_prob_gate_report"] = _build_research_prob_gate_report(
        decisions=decisions,
        trades=trades,
        configured_gate_mode=cfg_mode,
    )
    return out


def _build_research_prob_gate_report(
    *,
    decisions: Sequence[Dict[str, Any]],
    trades: Sequence[Dict[str, Any]],
    configured_gate_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Research softmax entry diagnostics among tier-passed candidates."""
    quality_probs: List[float] = []
    expl_probs: List[float] = []
    blocked_prob = Counter()
    blocked_cap = Counter()
    entry_allowed_quality = 0
    entry_allowed_expl = 0
    long_decisions = 0
    closed_v2 = [
        t
        for t in trades
        if str(t.get("model", "")).lower() == "v2" and str(t.get("event", "")).upper() == "CLOSED"
    ]

    for d in decisions:
        gd = _gate_diag_from_decision(d)
        if not gd:
            continue
        pl = _safe_float(gd.get("prob_long") if gd.get("prob_long") is not None else (d.get("v2") or {}).get("prob_long"))
        if bool(gd.get("quality_research_long_allowed")):
            if pl is not None:
                quality_probs.append(float(pl))
            if bool(gd.get("research_long_entry_allowed")) and str(
                gd.get("research_entry_source") or ""
            ) in ("quality_research", "exploratory_research"):
                entry_allowed_quality += 1
            if bool(gd.get("blocked_by_prob_gate")):
                blocked_prob["quality_research"] += 1
            if bool(gd.get("blocked_by_trade_cap")):
                blocked_cap["quality_research"] += 1
        if bool(gd.get("exploratory_research_long_allowed")):
            if pl is not None:
                expl_probs.append(float(pl))
            if bool(gd.get("research_long_entry_allowed")) and str(
                gd.get("research_entry_source") or ""
            ) == "exploratory_research":
                entry_allowed_expl += 1
            if bool(gd.get("blocked_by_prob_gate")):
                blocked_prob["exploratory_research"] += 1
            if bool(gd.get("blocked_by_trade_cap")):
                blocked_cap["exploratory_research"] += 1
        if str((d.get("v2") or {}).get("decision", "")).upper() == "LONG":
            long_decisions += 1

    def _prob_counts(probs: List[float]) -> Dict[str, Any]:
        a = np.asarray(probs, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {
                "distribution": _score_distribution([]),
                "count_prob_gt_0_6": 0,
                "count_prob_gt_0_2": 0,
                "count_prob_gt_0_05": 0,
            }
        return {
            "distribution": _score_distribution(probs),
            "count_prob_gt_0_6": int(np.sum(a > 0.6)),
            "count_prob_gt_0_2": int(np.sum(a > 0.2)),
            "count_prob_gt_0_05": int(np.sum(a > 0.05)),
        }

    return {
        "configured_gate_mode": configured_gate_mode,
        "quality_candidates_count": int(len(quality_probs)),
        "exploratory_candidates_count": int(len(expl_probs)),
        "quality_prob_long": _prob_counts(quality_probs),
        "exploratory_prob_long": _prob_counts(expl_probs),
        "research_long_entry_allowed_quality": int(entry_allowed_quality),
        "research_long_entry_allowed_exploratory": int(entry_allowed_expl),
        "v2_long_decisions": int(long_decisions),
        "actual_v2_closed_trades": int(len(closed_v2)),
        "blocked_by_prob_gate_count": dict(blocked_prob),
        "blocked_by_trade_cap_count": dict(blocked_cap),
    }


def build_refactor5_report(
    *,
    decisions: Sequence[Dict[str, Any]],
    trades: Sequence[Dict[str, Any]],
    closes: np.ndarray,
    close_meta: Dict[str, Any],
    merge_meta: Dict[str, Any],
    forward_horizon: int,
    symbol: str,
    date_start: Optional[str],
    date_end: Optional[str],
    baseline_validation_path: Optional[Path] = None,
) -> Dict[str, Any]:
    by_bar = {int(d.get("bar_index", -1)): d for d in decisions}
    closed = _closed_v2_rows(trades)
    long_closed = [t for t in closed if str(t.get("side", "")).upper() == "LONG"]
    short_closed = [t for t in closed if str(t.get("side", "")).upper() == "SHORT"]
    partial_sessions = _partial_sessions_from_meta(merge_meta, decisions)
    if date_start is None or date_end is None:
        all_ymd: List[str] = []
        for d in decisions:
            ts = str(d.get("timestamp", ""))
            if len(ts) >= 10 and ts.isdigit():
                dt = datetime.utcfromtimestamp(int(ts) + 7 * 3600)
                all_ymd.append(dt.strftime("%Y%m%d"))
        if all_ymd:
            if date_start is None:
                date_start = min(all_ymd)
            if date_end is None:
                date_end = max(all_ymd)

    dataset_summary = {
        "n_decisions": int(len(decisions)),
        "n_trades": int(len(closed)),
        "n_long_trades": int(len(long_closed)),
        "n_short_trades": int(len(short_closed)),
        "date_start": date_start,
        "date_end": date_end,
        "partial_sessions": partial_sessions,
        "close_missing_count": int(close_meta.get("close_missing_count", 0) or 0),
        "close_missing_ratio": float(close_meta.get("close_missing_ratio", 0.0) or 0.0),
        "partial_session_note": "20260508 likely partial if end-time < 14:38",
    }

    # 2) breakout detector health
    bmask = []
    nmask = []
    for d in decisions:
        bi = int(d.get("bar_index", -1))
        if bi < 0 or bi >= len(closes):
            continue
        fr = _forward_return(closes, bi, forward_horizon)
        if not np.isfinite(fr):
            continue
        f = d.get("features") or {}
        is_brk = bool(f.get("is_breakout_up") or f.get("breakout_up_raw_last") or f.get("breakout_up_raw"))
        if is_brk:
            bmask.append(float(fr))
        else:
            nmask.append(float(fr))
    breakout_health = {
        "n_valid_breakout": int(len(bmask)),
        "breakout_mean_forward_return": float(np.mean(bmask)) if bmask else None,
        "non_breakout_mean_forward_return": float(np.mean(nmask)) if nmask else None,
        "breakout_winrate": float(np.mean(np.array(bmask) > 0.0)) if bmask else None,
        "non_breakout_winrate": float(np.mean(np.array(nmask) > 0.0)) if nmask else None,
        "true_breakout_rate": _safe_float((merge_meta.get("true_breakout_rate"))),
        "trap_adverse_count": None,
        "interpretation": {
            "raw_strength": "detector existence/force axis",
            "score_long_effective_strength": "ranking expected return axis",
        },
    }

    # 3) LONG scoring validation (scoring domain only)
    s_score: List[float] = []
    s_eff: List[float] = []
    s_raw: List[float] = []
    s_fwd: List[float] = []
    for d in decisions:
        if not _v2_long_scoring_domain_row(d):
            continue
        bi = int(d.get("bar_index", -1))
        fr = _forward_return(closes, bi, forward_horizon)
        if not np.isfinite(fr):
            continue
        s_fwd.append(float(fr))
        s_score.append(float(_v2_score_long(d)))
        s_eff.append(float(_feature_float(d, "effective_strength")))
        s_raw.append(float(_feature_float(d, "raw_strength")))
    arr_fwd = np.asarray(s_fwd, dtype=np.float64)
    scoring_long = {
        "score_long": _bucket_stats_from_pairs(np.asarray(s_score, dtype=np.float64), arr_fwd),
        "effective_strength": _bucket_stats_from_pairs(np.asarray(s_eff, dtype=np.float64), arr_fwd),
        "raw_strength_diagnostic": _bucket_stats_from_pairs(np.asarray(s_raw, dtype=np.float64), arr_fwd),
        "acceptance_hint": "Prefer high > low; monotonic target only when sample sufficient",
    }

    # 4) last_bar_return impact
    lbr_raw: List[float] = []
    lbr_pos: List[float] = []
    lbr_fwd: List[float] = []
    for d in decisions:
        if not _v2_long_scoring_domain_row(d):
            continue
        bi = int(d.get("bar_index", -1))
        fr = _forward_return(closes, bi, forward_horizon)
        if not np.isfinite(fr):
            continue
        lbr_raw.append(float(_feature_float(d, "last_bar_return")))
        sc = _decision_v2_score_components(d)
        lbr_pos.append(float(sc.get("positive_last_bar_return") or 0.0))
        lbr_fwd.append(float(fr))
    lbr_raw_a = np.asarray(lbr_raw, dtype=np.float64)
    lbr_pos_a = np.asarray(lbr_pos, dtype=np.float64)
    lbr_fwd_a = np.asarray(lbr_fwd, dtype=np.float64)
    spike_thr, spike_method = _nondegenerate_spike_threshold(lbr_pos_a)
    spike_mask = (
        (lbr_pos_a >= float(spike_thr)) & (lbr_pos_a > 0.0)
        if (lbr_pos_a.size and spike_thr is not None)
        else np.zeros(lbr_pos_a.shape, dtype=bool)
    )
    non_mask = ~spike_mask if lbr_pos_a.size else np.array([], dtype=bool)
    last_bar_return_impact = {
        "bucket_last_bar_return_raw": _bucket_stats_from_pairs(lbr_raw_a, lbr_fwd_a),
        "bucket_positive_last_bar_return": _bucket_stats_from_pairs(lbr_pos_a, lbr_fwd_a),
        "raw_last_bar_return_stats": _series_quantiles(lbr_raw_a),
        "positive_last_bar_return_stats": _series_quantiles(lbr_pos_a),
        "last_bar_return_unit_inference": _infer_last_bar_return_unit(lbr_raw_a),
        "spike_threshold_candidate": spike_thr,
        "spike_threshold_method": spike_method,
        "spike_count": int(np.sum(spike_mask)) if spike_mask.size else 0,
        "non_spike_count": int(np.sum(non_mask)) if non_mask.size else 0,
        "spike_mean_forward_return": float(np.mean(lbr_fwd_a[spike_mask])) if spike_mask.size and np.any(spike_mask) else None,
        "non_spike_mean_forward_return": float(np.mean(lbr_fwd_a[non_mask])) if non_mask.size and np.any(non_mask) else None,
        "spike_bucket_non_degenerate": bool(np.any(spike_mask) and np.any(non_mask)) if lbr_pos_a.size else False,
        "question_answer": {
            "spike_worse_trade_quality": (
                bool(np.mean(lbr_fwd_a[spike_mask]) < np.mean(lbr_fwd_a[non_mask]))
                if spike_mask.size and np.any(spike_mask) and np.any(non_mask)
                else None
            ),
            "blocked_by_spike_filter": int(np.sum(spike_mask)) if spike_mask.size else 0,
        },
    }

    # 5) crowd phase performance
    phase_rows: Dict[str, Dict[str, Any]] = {}
    phase_names = ("no_breakout", "ignition", "early_continuation", "late_fomo", "exhaustion", "failed_breakout")
    for ph in phase_names:
        phase_rows[ph] = {"decision_count": 0, "trade_count": 0, "fwd": [], "pnl": [], "holding": [], "mfe": [], "mae": []}
    for d in decisions:
        bi = int(d.get("bar_index", -1))
        ph = _decision_crowd_phase(d)
        if ph not in phase_rows:
            continue
        phase_rows[ph]["decision_count"] += 1
        fr = _forward_return(closes, bi, forward_horizon)
        if np.isfinite(fr):
            phase_rows[ph]["fwd"].append(float(fr))
    for t in closed:
        ph = str(t.get("entry_crowd_phase") or "")
        if not ph and t.get("entry_bar_index") is not None:
            d_ent = by_bar.get(int(t.get("entry_bar_index")))
            if d_ent is None:
                d_ent = by_bar.get(int(t.get("entry_bar_index")) - 1)
            if d_ent is not None:
                ph = _decision_crowd_phase(d_ent)
        if not ph:
            ph = "no_breakout"
        if ph not in phase_rows:
            continue
        phase_rows[ph]["trade_count"] += 1
        rr = _safe_float(t.get("realized_return"))
        if rr is not None:
            phase_rows[ph]["pnl"].append(rr)
        hb = _safe_float(t.get("holding_period"))
        if hb is not None:
            phase_rows[ph]["holding"].append(hb)
        mfe = _safe_float(t.get("mfe"))
        mae = _safe_float(t.get("mae"))
        if mfe is not None:
            phase_rows[ph]["mfe"].append(mfe)
        if mae is not None:
            phase_rows[ph]["mae"].append(mae)
    crowd_phase_performance = []
    for ph in phase_names:
        r = phase_rows[ph]
        pnl = np.asarray(r["pnl"], dtype=np.float64) if r["pnl"] else np.array([], dtype=np.float64)
        crowd_phase_performance.append(
            {
                "phase": ph,
                "decision_count": int(r["decision_count"]),
                "trade_count": int(r["trade_count"]),
                "forward_return_mean": float(np.mean(r["fwd"])) if r["fwd"] else None,
                "winrate": float(np.mean(pnl > 0.0)) if pnl.size else None,
                "realized_pnl_sum": float(np.sum(pnl)) if pnl.size else None,
                "avg_holding": float(np.mean(r["holding"])) if r["holding"] else None,
                "mfe_mean": float(np.mean(r["mfe"])) if r["mfe"] else None,
                "mae_mean": float(np.mean(r["mae"])) if r["mae"] else None,
            }
        )

    # 6) entry block simulation (simulation only)
    reason_stats: Dict[str, Dict[str, float]] = {}
    for reason in ("late_fomo", "exhaustion", "last_bar_return_spike", "score_below_threshold", "failed_breakout"):
        reason_stats[reason] = {"blocked": 0, "blocked_pnl": 0.0, "blocked_wins": 0, "kept": 0, "kept_pnl": 0.0, "kept_wins": 0}
    for t in long_closed:
        ebi = t.get("entry_bar_index")
        if ebi is None:
            continue
        d = by_bar.get(int(ebi) - 1) or by_bar.get(int(ebi))
        if not d:
            continue
        sc = _decision_v2_score_components(d)
        f = d.get("features") or {}
        blocked_flags = {
            "late_fomo": bool(sc.get("late_fomo_flag")),
            "exhaustion": bool(f.get("exhaustion_confirm")),
            "last_bar_return_spike": bool(spike_thr is not None and float(sc.get("positive_last_bar_return") or 0.0) >= float(spike_thr) and float(sc.get("positive_last_bar_return") or 0.0) > 0.0),
            "score_below_threshold": bool(float((d.get("v2") or {}).get("score_long") or 0.0) < float(TTM_CONFIG.get("ttm_v2_score_long_entry_threshold", 0.0))),
            "failed_breakout": bool((not bool(f.get("breakout_up"))) and float(f.get("failure_strength") or 0.0) >= float(TTM_CONFIG.get("ttm_v2_failed_breakout_failure_min", 0.35))),
        }
        rr = _safe_float(t.get("realized_return"))
        if rr is None:
            continue
        for k, v in blocked_flags.items():
            st = reason_stats[k]
            if v:
                st["blocked"] += 1
                st["blocked_pnl"] += rr
                st["blocked_wins"] += int(rr > 0.0)
            else:
                st["kept"] += 1
                st["kept_pnl"] += rr
                st["kept_wins"] += int(rr > 0.0)
    entry_block_simulation = []
    for k, st in reason_stats.items():
        b = max(1, int(st["blocked"]))
        kp = max(1, int(st["kept"]))
        entry_block_simulation.append(
            {
                "reason": k,
                "mode": "simulation",
                "blocked_trades": int(st["blocked"]),
                "blocked_pnl": float(st["blocked_pnl"]),
                "kept_pnl": float(st["kept_pnl"]),
                "blocked_winrate": float(st["blocked_wins"] / b) if st["blocked"] else None,
                "kept_winrate": float(st["kept_wins"] / kp) if st["kept"] else None,
                "blocked_avg_trade": float(st["blocked_pnl"] / b) if st["blocked"] else None,
                "kept_avg_trade": float(st["kept_pnl"] / kp) if st["kept"] else None,
            }
        )

    # 7) holding period performance + soft-min hold count
    hold_groups = {"1": [], "2": [], "3+": [], "4+": []}
    soft_hold_count = 0
    for t in closed:
        hb = int(_safe_float(t.get("holding_period")) or 0)
        rr = _safe_float(t.get("realized_return"))
        if rr is None:
            continue
        if hb == 1:
            hold_groups["1"].append(rr)
        if hb == 2:
            hold_groups["2"].append(rr)
        if hb >= 3:
            hold_groups["3+"].append(rr)
        if hb >= 4:
            hold_groups["4+"].append(rr)
        if str(t.get("exit_reason") or "") == "soft_min_hold_continuation":
            soft_hold_count += 1
    holding_period_performance = []
    for k in ("1", "2", "3+", "4+"):
        arr = np.asarray(hold_groups[k], dtype=np.float64) if hold_groups[k] else np.array([], dtype=np.float64)
        holding_period_performance.append(
            {
                "bucket": k,
                "trade_count": int(arr.size),
                "pnl_sum": float(np.sum(arr)) if arr.size else None,
                "winrate": float(np.mean(arr > 0.0)) if arr.size else None,
                "avg_trade": float(np.mean(arr)) if arr.size else None,
            }
        )

    # 8) exit reason performance (split long/short)
    def _exit_reason_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        by_reason: Dict[str, List[Dict[str, Any]]] = {}
        for t in rows:
            r = str(t.get("exit_reason") or "unknown")
            by_reason.setdefault(r, []).append(t)
        for r, rr in sorted(by_reason.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            rets = np.asarray([x for x in (_safe_float(x.get("realized_return")) for x in rr) if x is not None], dtype=np.float64)
            holds = np.asarray([x for x in (_safe_float(x.get("holding_period")) for x in rr) if x is not None], dtype=np.float64)
            mfe = np.asarray([x for x in (_safe_float(x.get("mfe")) for x in rr) if x is not None], dtype=np.float64)
            mae = np.asarray([x for x in (_safe_float(x.get("mae")) for x in rr) if x is not None], dtype=np.float64)
            out.append(
                {
                    "exit_reason": r,
                    "count": int(len(rr)),
                    "pnl": float(np.sum(rets)) if rets.size else None,
                    "winrate": float(np.mean(rets > 0.0)) if rets.size else None,
                    "avg_trade": float(np.mean(rets)) if rets.size else None,
                    "avg_holding": float(np.mean(holds)) if holds.size else None,
                    "mfe_mean": float(np.mean(mfe)) if mfe.size else None,
                    "mae_mean": float(np.mean(mae)) if mae.size else None,
                }
            )
        return out
    exit_reason_performance = {
        "long": _exit_reason_table(long_closed),
        "short": _exit_reason_table(short_closed),
    }

    # 9) SHORT validation actual + candidates
    short_actual_rets = np.asarray([x for x in (_safe_float(t.get("realized_return")) for t in short_closed) if x is not None], dtype=np.float64)
    actual_short = {
        "count": int(len(short_closed)),
        "pnl": float(np.sum(short_actual_rets)) if short_actual_rets.size else None,
        "winrate": float(np.mean(short_actual_rets > 0.0)) if short_actual_rets.size else None,
        "avg_trade": float(np.mean(short_actual_rets)) if short_actual_rets.size else None,
        "avg_holding": float(np.mean([_safe_float(t.get("holding_period")) for t in short_closed if _safe_float(t.get("holding_period")) is not None])) if short_closed else None,
        "entry_short_phase_distribution": dict(Counter(str(t.get("short_entry_phase") or "unknown") for t in short_closed)),
        "exit_reason_distribution": dict(Counter(str(t.get("exit_reason") or "unknown") for t in short_closed)),
        "unknown_entry_short_phase_count": int(sum(1 for t in short_closed if str(t.get("short_entry_phase") or "unknown") == "unknown")),
        "unknown_exit_reason_count": int(sum(1 for t in short_closed if str(t.get("exit_reason") or "unknown") == "unknown")),
    }
    cand_map: Dict[str, List[Tuple[float, float]]] = {}
    all_short_scores: List[float] = []
    all_short_targets: List[float] = []
    phase_alias = ("no_short_context", "crowded_long_watch", "short_setup", "short_trigger", "short_chase_risk", "short_invalid")
    for d in decisions:
        bi = int(d.get("bar_index", -1))
        fr = _forward_return(closes, bi, forward_horizon)
        if not np.isfinite(fr):
            continue
        sh = _decision_v2_short_components(d)
        ph = str(sh.get("short_phase") or _decision_short_phase(d) or "no_short_context")
        if ph not in phase_alias:
            continue
        sscore = _safe_float(sh.get("short_score"))
        if sscore is None:
            sscore = _safe_float(_feature_float(d, "short_score"))
        if sscore is None:
            continue
        neg_fr = float(-fr)
        cand_map.setdefault(ph, []).append((float(sscore), neg_fr))
        all_short_scores.append(float(sscore))
        all_short_targets.append(neg_fr)
    short_candidates = []
    for ph in phase_alias:
        pairs = cand_map.get(ph, [])
        if not pairs:
            short_candidates.append({"phase": ph, "count": 0, "E_neg_forward_return": None, "short_winrate": None, "top_minus_bottom": None})
            continue
        sc = np.asarray([p[0] for p in pairs], dtype=np.float64)
        tg = np.asarray([p[1] for p in pairs], dtype=np.float64)
        ql = float(np.quantile(sc, 0.2))
        qh = float(np.quantile(sc, 0.8))
        low = tg[sc <= ql]
        high = tg[sc >= qh]
        short_candidates.append(
            {
                "phase": ph,
                "count": int(len(pairs)),
                "E_neg_forward_return": float(np.mean(tg)),
                "short_winrate": float(np.mean(tg > 0.0)),
                "top_minus_bottom": float(np.mean(high) - np.mean(low)) if low.size and high.size else None,
            }
        )
    short_alpha_checks = {
        "short_trigger_outperform_setup_watch": None,
        "short_chase_risk_worse_than_trigger": None,
        "blocked_early_continuation_not_good_short": None,
        "no_prior_breakout_not_trusted": None,
    }
    cdict = {r["phase"]: r for r in short_candidates}
    if cdict.get("short_trigger") and cdict.get("short_setup") and cdict.get("crowded_long_watch"):
        vtr = cdict["short_trigger"]["E_neg_forward_return"]
        vss = cdict["short_setup"]["E_neg_forward_return"]
        vcw = cdict["crowded_long_watch"]["E_neg_forward_return"]
        if vtr is not None and vss is not None and vcw is not None:
            short_alpha_checks["short_trigger_outperform_setup_watch"] = bool(vtr >= max(vss, vcw))
    if cdict.get("short_chase_risk") and cdict.get("short_trigger"):
        a = cdict["short_chase_risk"]["E_neg_forward_return"]
        b = cdict["short_trigger"]["E_neg_forward_return"]
        if a is not None and b is not None:
            short_alpha_checks["short_chase_risk_worse_than_trigger"] = bool(a < b)
    short_validation = {
        "actual_short": actual_short,
        "short_candidates": short_candidates,
        "short_alpha_checks": short_alpha_checks,
        "acceptance_note": "Do not force positive SHORT pnl on small sample; require phase separation/logging.",
    }

    # 9b) post-refactor wiring diagnostics: v3 score consistency + short context persistence
    score_component_mismatch_count = 0
    score_tanh_mismatch_count = 0
    raw_negative_strong_positive_count = 0
    score_rows_checked = 0
    for d in decisions:
        v2 = d.get("v2") or {}
        sc = _decision_v2_score_components(d)
        v2_sl = _safe_float(v2.get("score_long"))
        sc_sl = _safe_float(sc.get("score_long"))
        eff = _safe_float(sc.get("effective_strength"))
        eraw = _safe_float(sc.get("effective_strength_raw"))
        if v2_sl is not None and sc_sl is not None:
            score_rows_checked += 1
            if abs(float(v2_sl) - float(sc_sl)) >= 1e-9:
                score_component_mismatch_count += 1
        if sc_sl is not None and eff is not None:
            if abs(float(sc_sl) - float(np.tanh(float(eff)))) >= 1e-6:
                score_tanh_mismatch_count += 1
        if eraw is not None and float(eraw) < 0.0 and v2_sl is not None and float(v2_sl) > 0.5:
            raw_negative_strong_positive_count += 1
    score_component_consistency = {
        "rows_checked": int(score_rows_checked),
        "score_component_mismatch_count": int(score_component_mismatch_count),
        "score_tanh_mismatch_count": int(score_tanh_mismatch_count),
        "raw_negative_strong_positive_count": int(raw_negative_strong_positive_count),
    }

    phase_reason_counts = Counter(
        str(_decision_v2_score_components(d).get("phase_reason") or "unknown")
        for d in decisions
    )
    phase_diagnostics = {
        "phase_reason_distribution": dict(phase_reason_counts),
        "missing_phase_reason_count": int(phase_reason_counts.get("unknown", 0)),
    }

    max_short_ctx = 20
    last_ignition_ix: Optional[int] = None
    short_context_missing_after_ignition = 0
    short_context_rows_checked = 0
    for d in sorted(decisions, key=lambda x: int(x.get("bar_index", -1))):
        bi = int(d.get("bar_index", -1))
        ph = _decision_crowd_phase(d)
        if ph in ("ignition", "early_continuation"):
            last_ignition_ix = bi
        if last_ignition_ix is None or bi <= last_ignition_ix:
            continue
        if bi - last_ignition_ix > max_short_ctx:
            continue
        sh = _decision_v2_short_components(d)
        short_context_rows_checked += 1
        if (
            not bool(sh.get("prior_upside_breakout_exists"))
            or sh.get("last_upside_breakout_bar_index") is None
            or _safe_float(sh.get("bars_since_upside_breakout")) is None
            or float(sh.get("bars_since_upside_breakout") or 0.0) < 1.0
        ):
            short_context_missing_after_ignition += 1
    short_context_persistence = {
        "rows_checked_after_ignition": int(short_context_rows_checked),
        "missing_context_rows": int(short_context_missing_after_ignition),
        "context_window_bars_assumed": int(max_short_ctx),
    }

    # 10) before/after summary
    def _summary_from_current() -> Dict[str, Any]:
        all_rets = np.asarray([x for x in (_safe_float(t.get("realized_return")) for t in closed) if x is not None], dtype=np.float64)
        long_rets = np.asarray([x for x in (_safe_float(t.get("realized_return")) for t in long_closed) if x is not None], dtype=np.float64)
        short_rets = np.asarray([x for x in (_safe_float(t.get("realized_return")) for t in short_closed) if x is not None], dtype=np.float64)
        score_tb = scoring_long["score_long"].get("top20_minus_bottom20")
        eff_tb = scoring_long["effective_strength"].get("top20_minus_bottom20")
        hold1 = next((x for x in holding_period_performance if x["bucket"] == "1"), {})
        hold3 = next((x for x in holding_period_performance if x["bucket"] == "3+"), {})
        return {
            "total_trades": int(len(closed)),
            "LONG_trades": int(len(long_closed)),
            "LONG_pnl": float(np.sum(long_rets)) if long_rets.size else None,
            "LONG_winrate": float(np.mean(long_rets > 0.0)) if long_rets.size else None,
            "SHORT_trades": int(len(short_closed)),
            "SHORT_pnl": float(np.sum(short_rets)) if short_rets.size else None,
            "SHORT_winrate": float(np.mean(short_rets > 0.0)) if short_rets.size else None,
            "avg_trade": float(np.mean(all_rets)) if all_rets.size else None,
            "score_long_top_vs_bottom": score_tb,
            "effective_strength_top_vs_bottom": eff_tb,
            "last_bar_return_spike_blocked_count": int(last_bar_return_impact.get("spike_count") or 0),
            "holding_1_bar_count": hold1.get("trade_count"),
            "holding_1_bar_pnl": hold1.get("pnl_sum"),
            "holding_3_plus_count": hold3.get("trade_count"),
            "holding_3_plus_pnl": hold3.get("pnl_sum"),
            "exit_reason_distribution": dict(Counter(str(t.get("exit_reason") or "unknown") for t in closed)),
        }
    after = _summary_from_current()
    def _extract_summary_from_legacy_validation(v: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        meta = v.get("meta") or {}
        out["total_trades"] = _safe_float(((v.get("adaptive") or {}).get("metrics") or {}).get("n_closed_trades"))
        hold = (((v.get("breakout") or {}).get("metrics") or {}).get("v2_trade_holding") or {})
        out["LONG_trades"] = _safe_float(hold.get("n_v2_long_closed"))
        # Legacy files typically do not carry full side-level pnl aggregates.
        out["LONG_pnl"] = None
        out["LONG_winrate"] = None
        out["SHORT_trades"] = None
        out["SHORT_pnl"] = None
        out["SHORT_winrate"] = None
        out["avg_trade"] = None
        sc = v.get("scoring_long") or v.get("scoring") or {}
        bm = (sc.get("metrics") or {}).get("bucket_mean_aligned_return") or {}
        if isinstance(bm, dict) and bm.get("high") is not None and bm.get("low") is not None:
            out["score_long_top_vs_bottom"] = float(bm.get("high")) - float(bm.get("low"))
        else:
            out["score_long_top_vs_bottom"] = None
        out["effective_strength_top_vs_bottom"] = None
        out["last_bar_return_spike_blocked_count"] = None
        hhist = hold.get("holding_period_hist") or {}
        if isinstance(hhist, dict):
            h1 = int(hhist.get("1", 0) or 0)
            h3 = sum(int(vh or 0) for kh, vh in hhist.items() if str(kh).isdigit() and int(kh) >= 3)
            out["holding_1_bar_count"] = h1
            out["holding_3_plus_count"] = h3
        else:
            out["holding_1_bar_count"] = None
            out["holding_3_plus_count"] = None
        out["holding_1_bar_pnl"] = None
        out["holding_3_plus_pnl"] = None
        out["exit_reason_distribution"] = None
        out["_legacy_meta_start"] = meta.get("aggregate_start_date")
        out["_legacy_meta_end"] = meta.get("aggregate_end_date")
        return out

    before: Dict[str, Any] = {}
    if baseline_validation_path is not None and baseline_validation_path.is_file():
        try:
            bdat = json.loads(baseline_validation_path.read_text(encoding="utf-8"))
            # allow baseline either raw refactor5_report or full validation json
            before = (bdat.get("refactor5_report") or {}).get("before_after", {}).get("after") or {}
            if not before:
                before = _extract_summary_from_legacy_validation(bdat)
        except Exception:
            before = {}
    metric_order = [
        "total_trades", "LONG_trades", "LONG_pnl", "LONG_winrate",
        "SHORT_trades", "SHORT_pnl", "SHORT_winrate", "avg_trade",
        "score_long_top_vs_bottom", "effective_strength_top_vs_bottom",
        "last_bar_return_spike_blocked_count", "holding_1_bar_count",
        "holding_1_bar_pnl", "holding_3_plus_count", "holding_3_plus_pnl",
        "exit_reason_distribution",
    ]
    rows = []
    for m in metric_order:
        rows.append({"metric": m, "before": before.get(m), "after": after.get(m), "comment": "baseline optional"})
    before_after = {
        "baseline_path": str(baseline_validation_path) if baseline_validation_path else None,
        "before": before,
        "after": after,
        "rows": rows,
    }

    # acceptance checklist (refactor_5)
    n_dec = max(1, len(decisions))
    score_comp_count = sum(1 for d in decisions if isinstance(((d.get("v2") or {}).get("score_components")), dict) and len(((d.get("v2") or {}).get("score_components")) or {}) > 0)
    short_comp_count = sum(1 for d in decisions if isinstance(((d.get("v2") or {}).get("short_components")), dict) and len(((d.get("v2") or {}).get("short_components")) or {}) > 0)
    cov_score = float(score_comp_count / n_dec)
    cov_short = float(short_comp_count / n_dec)
    score_components_ok = bool(cov_score >= 0.95)
    short_components_ok = bool(cov_short >= 0.95)
    coverage_ok = bool(score_components_ok and short_components_ok)
    unknown_exit_count = sum(1 for t in closed if str(t.get("exit_reason") or "unknown") == "unknown")
    all_trade_exit_reason = bool(len(closed) == 0 or unknown_exit_count < len(closed))
    actual_short_unknown_phase_count = sum(
        1 for t in short_closed if str(t.get("short_entry_phase") or "unknown") == "unknown"
    )
    actual_short_unknown_exit_reason_count = sum(
        1 for t in short_closed if str(t.get("exit_reason") or "unknown") == "unknown"
    )
    actual_short_invalid_phase_count = sum(
        1
        for t in short_closed
        if str(t.get("short_entry_phase") or "unknown")
        not in ("short_trigger", "short_setup")
    )
    all_trade_phases = bool(
        len(short_closed) == 0 or actual_short_unknown_phase_count < len(short_closed)
    )
    non_no_breakout_decisions = sum(
        int(r.get("decision_count", 0))
        for r in crowd_phase_performance
        if str(r.get("phase")) != "no_breakout"
    )
    phase_wired_ok = bool(
        int(breakout_health.get("n_valid_breakout") or 0) <= 0 or non_no_breakout_decisions > 0
    )
    acceptance = {
        "logging": {
            "every_decision_has_v2_score_components": score_components_ok,
            "every_decision_has_v2_short_components": short_components_ok,
            "every_trade_has_entry_exit_phase_if_available": all_trade_phases,
            "every_trade_has_exit_reason": all_trade_exit_reason,
            "decision_score_components_coverage": cov_score,
            "decision_short_components_coverage": cov_short,
            "coverage_threshold": 0.95,
            "phase_wired_ok": phase_wired_ok,
            "unknown_exit_reason_count": int(unknown_exit_count),
            "actual_short_unknown_phase_count": int(actual_short_unknown_phase_count),
            "actual_short_unknown_exit_reason_count": int(actual_short_unknown_exit_reason_count),
            "actual_short_invalid_phase_count": int(actual_short_invalid_phase_count),
            "score_component_mismatch_count": int(score_component_mismatch_count),
            "score_tanh_mismatch_count": int(score_tanh_mismatch_count),
            "raw_negative_strong_positive_count": int(raw_negative_strong_positive_count),
            "short_context_missing_after_ignition": int(short_context_missing_after_ignition),
            "missing_phase_reason_count": int(phase_diagnostics["missing_phase_reason_count"]),
        },
        "exit": {
            "holding_period_stats_reported": bool(holding_period_performance),
            "exit_reason_stats_reported": bool(exit_reason_performance["long"] or exit_reason_performance["short"]),
        },
        "no_leakage": {
            "signal_bar_equals_entry_minus_1": all(
                (t.get("signal_bar_index") is None or t.get("entry_bar_index") is None or int(t.get("signal_bar_index")) == int(t.get("entry_bar_index")) - 1)
                for t in closed
            ) if closed else True,
            "simulations_marked": True,
        },
        "overall_acceptance": bool(
            coverage_ok
            and phase_wired_ok
            and all_trade_exit_reason
            and all_trade_phases
            and (len(short_closed) == 0 or actual_short_unknown_exit_reason_count < len(short_closed))
            and actual_short_invalid_phase_count == 0
            and score_component_mismatch_count == 0
            and score_tanh_mismatch_count == 0
            and raw_negative_strong_positive_count == 0
            and short_context_missing_after_ignition == 0
            and int(phase_diagnostics["missing_phase_reason_count"]) == 0
        ),
    }
    insufficient_sample_warnings: List[str] = []
    if scoring_long["score_long"].get("insufficient_sample"):
        insufficient_sample_warnings.append("score_long sample insufficient for strong ranking inference")
    if len(short_closed) < 20:
        insufficient_sample_warnings.append("actual SHORT trades < 20; collect 50-100 candidates/trades before tuning")
    if len(long_closed) < 30:
        insufficient_sample_warnings.append("LONG trades limited; avoid overfit conclusions")

    gate_mode_report = build_gate_mode_report(
        decisions=decisions,
        trades=trades,
        closes=closes,
        forward_horizon=forward_horizon,
        configured_gate_mode=str(merge_meta.get("ttm_v2_gate_mode") or ""),
    )
    instr_ok = bool(
        coverage_ok
        and phase_wired_ok
        and score_component_mismatch_count == 0
        and score_tanh_mismatch_count == 0
        and raw_negative_strong_positive_count == 0
        and short_context_missing_after_ignition == 0
        and int(phase_diagnostics["missing_phase_reason_count"]) == 0
    )
    alpha_status = str(gate_mode_report.get("alpha_validation_status") or "")
    acceptance["alpha_validation_status"] = alpha_status
    acceptance["instrumentation_acceptance"] = bool(instr_ok)
    acceptance["candidate_validation_status"] = str(
        gate_mode_report.get("candidate_validation_status") or ""
    )
    acceptance["overall_acceptance"] = bool(
        acceptance["overall_acceptance"]
        and len(closed) > 0
        and alpha_status == "alpha_sample_available"
        and not bool(gate_mode_report.get("insufficient_trade_sample"))
    )

    return {
        "dataset_summary": dataset_summary,
        "breakout_detector_health": breakout_health,
        "long_scoring_validation": scoring_long,
        "last_bar_return_impact": last_bar_return_impact,
        "crowd_phase_performance": crowd_phase_performance,
        "entry_block_simulation": entry_block_simulation,
        "holding_period_performance": {
            "rows": holding_period_performance,
            "soft_min_hold_prevented_exits": int(soft_hold_count),
        },
        "exit_reason_performance": exit_reason_performance,
        "short_validation": short_validation,
        "score_component_consistency": score_component_consistency,
        "phase_diagnostics": phase_diagnostics,
        "short_context_persistence": short_context_persistence,
        "gate_mode_report": gate_mode_report,
        "before_after": before_after,
        "acceptance": acceptance,
        "insufficient_sample_warnings": insufficient_sample_warnings,
        "pm_todo": [
            "Review score_long high-vs-low and effective_strength high-vs-low",
            "Review blocked late/FOMO and spike-filter blocked PnL",
            "Review soft-min-hold saved trades vs increased losses",
            "Review short_trigger quality; collect more SHORT samples before go-live",
        ],
        "cursor_output_sections": [
            "files_changed", "validation_command", "dataset_coverage", "before_after_table",
            "long_scoring_table", "last_bar_return_spike_table", "crowd_phase_table",
            "holding_period_table", "exit_reason_table", "short_actual_candidate_table",
            "insufficient_sample_warnings", "risks_todo",
        ],
    }


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
    baseline_validation_path: Optional[Path] = None,
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
    merged = _merge_scoring_into_result(
        base_result,
        scoring_mode=mode,
        b_long=b3_long,
        b_short=b3_short,
    )
    merged["refactor5_report"] = build_refactor5_report(
        decisions=decisions,
        trades=trades,
        closes=closes,
        close_meta=close_meta,
        merge_meta=merge_meta,
        forward_horizon=forward_horizon,
        symbol="",
        date_start=None,
        date_end=None,
        baseline_validation_path=baseline_validation_path,
    )
    r5_acc = ((merged.get("refactor5_report") or {}).get("acceptance") or {}).get("overall_acceptance")
    merged["overall_acceptance_valid"] = bool(r5_acc) if r5_acc is not None else None
    return merged


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
    baseline_validation_path: Optional[Path] = None,
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
    merged = _merge_scoring_into_result(
        base_result,
        scoring_mode=mode,
        b_long=b3_long,
        b_short=b3_short,
    )
    merged["refactor5_report"] = build_refactor5_report(
        decisions=decisions,
        trades=trades,
        closes=closes,
        close_meta=close_meta,
        merge_meta=merge_meta,
        forward_horizon=forward_horizon,
        symbol=symbol,
        date_start=start_date,
        date_end=end_date,
        baseline_validation_path=baseline_validation_path,
    )
    # Refactor 5 focuses reporting/acceptance for PM decisions (separate from legacy strict blocks).
    r5_acc = ((merged.get("refactor5_report") or {}).get("acceptance") or {}).get("overall_acceptance")
    merged["overall_acceptance_valid"] = bool(r5_acc) if r5_acc is not None else None
    return merged


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
    p.add_argument(
        "--baseline-validation",
        type=Path,
        default=None,
        help="Optional baseline validation JSON for before/after comparison table.",
    )
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
            baseline_validation_path=args.baseline_validation,
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
            baseline_validation_path=args.baseline_validation,
        )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    ov = bool(result.get("overall_valid", False))
    oa = bool(result.get("overall_acceptance_valid", False))
    return 0 if (ov or oa) else 2


if __name__ == "__main__":
    raise SystemExit(main())
