#!/usr/bin/env python3
"""Offline grid research for TTM empirical alpha (automate_tunning.md).

Replays paper ``decisions`` JSONL + aligned closes array; aligns alpha at bar ``i`` (pre-``on_bar``)
with ``forward_return_4`` labeled when horizon elapses.

Example:
  python scripts/ttm_empirical_research.py \\
    --decisions reports/ttm_parallel_decisions_VN30F1M_20260410_0734.jsonl \\
    --closes reports/closes_VN30F1M_20260410_0734.json \\
    --json-out reports/ttm_empirical_research_out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.strategies.ttm.config import TTM_CONFIG  # noqa: E402
from src.strategies.ttm.empirical.adaptive_engine import EmpiricalAlphaEngine  # noqa: E402
from src.strategies.ttm.empirical.alpha import compute_empirical_alpha  # noqa: E402
from src.strategies.ttm.empirical.execution import decide_execution  # noqa: E402
from src.strategies.ttm.empirical.feature_store import last_dict_from_decision_features  # noqa: E402
from src.strategies.ttm.ttm_validation import load_decisions_jsonl  # noqa: E402


def _load_closes_list(path: Path) -> List[float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("closes file must be JSON array")
    return [float(x) for x in raw]


def _max_drawdown(cumulative: np.ndarray) -> float:
    if cumulative.size == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative)
    dd = peak - cumulative
    return float(np.max(dd)) if dd.size else 0.0


def run_one(
    decisions: List[Dict[str, Any]],
    closes: List[float],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    n_bars = len(closes)
    if len(decisions) != n_bars:
        raise ValueError(f"len(decisions)={len(decisions)} != len(closes)={n_bars}")

    engine = EmpiricalAlphaEngine(cfg)
    clip_e = float(cfg.get("ttm_v2_empirical_alpha_clip", 0.003))
    alpha_by_bi: Dict[int, float] = {}

    cost = float(cfg.get("ttm_v2_empirical_cost_roundtrip", 0.0004))
    margin = float(cfg.get("ttm_v2_empirical_cost_margin", 0.0001))
    nz = float(cfg.get("ttm_v2_empirical_no_trade_abs", 0.0))
    exec_counts = {"LONG": 0, "SHORT": 0, "SKIP": 0}

    for d in decisions:
        bi = int(d["bar_index"])
        if bi < 0 or bi >= n_bars:
            continue
        last = last_dict_from_decision_features(d.get("features") or {})
        last["close"] = float(closes[bi])
        try:
            ts = int(d.get("timestamp", bi))
        except (TypeError, ValueError):
            ts = bi
        a_pre, _ = compute_empirical_alpha(last, engine.calibration, clip_abs=clip_e)
        alpha_by_bi[bi] = float(a_pre)
        ex = decide_execution(
            a_pre,
            cost_roundtrip=cost,
            margin=margin,
            no_trade_abs=nz,
        )
        exec_counts[str(ex.side)] = exec_counts.get(str(ex.side), 0) + 1
        engine.on_bar(bi, ts, last, float(closes[bi]))

    engine.force_rebuild()
    labeled = engine.store.labeled_rows()
    xs: List[float] = []
    ys: List[float] = []
    for r in labeled:
        if r.bar_index not in alpha_by_bi:
            continue
        xs.append(alpha_by_bi[r.bar_index])
        ys.append(r.forward_return_4)

    pearson = float("nan")
    if len(xs) >= 5:
        xa = np.asarray(xs, dtype=np.float64)
        ya = np.asarray(ys, dtype=np.float64)
        if np.std(xa) > 1e-15 and np.std(ya) > 1e-15:
            pearson = float(np.corrcoef(xa, ya)[0, 1])

    toy = np.asarray(ys, dtype=np.float64) * np.sign(np.asarray(xs, dtype=np.float64))
    cum = np.cumsum(toy) if toy.size else np.array([])
    max_dd = _max_drawdown(cum) if cum.size else 0.0

    return {
        "n_bars": n_bars,
        "n_labeled": len(labeled),
        "pearson_alpha_fwd": pearson,
        "toy_pnl_sum_sign": float(np.sum(toy)) if toy.size else 0.0,
        "toy_max_drawdown_abs_sum": max_dd,
        "execution_counts": exec_counts,
        "calibration_meta": engine.get_last_debug(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="TTM empirical alpha offline research")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--closes", type=Path, required=True)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--grid-window",
        type=str,
        default="300",
        help="Comma-separated ttm_v2_empirical_window_size values",
    )
    p.add_argument(
        "--grid-update",
        type=str,
        default="30",
        help="Comma-separated ttm_v2_empirical_update_every_bars values",
    )
    args = p.parse_args()

    decisions = load_decisions_jsonl(args.decisions)
    closes = _load_closes_list(args.closes)

    windows = [int(x.strip()) for x in args.grid_window.split(",") if x.strip()]
    updates = [int(x.strip()) for x in args.grid_update.split(",") if x.strip()]
    results: List[Dict[str, Any]] = []

    for w in windows:
        for u in updates:
            cfg = dict(TTM_CONFIG)
            cfg["ttm_v2_empirical_window_size"] = w
            cfg["ttm_v2_empirical_update_every_bars"] = max(1, u)
            try:
                m = run_one(decisions, closes, cfg)
            except ValueError as e:
                m = {"error": str(e)}
            m["grid_window"] = w
            m["grid_update_every"] = u
            results.append(m)

    out = {"results": results, "decisions": str(args.decisions), "closes": str(args.closes)}
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
