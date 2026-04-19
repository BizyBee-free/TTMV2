"""Export top configs, IS/OOS metrics, equity curves, feature ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.backtest.data_fetcher import OhlcBar
from src.backtest.metrics import BacktestMetrics, compare_metrics
from src.backtest.ttm_opt.engine import run_ttm_opt_backtest
from src.backtest.ttm_opt.optimize import bars_to_arrays
from src.backtest.ttm_opt.params import TtmOptParams
from src.backtest.ttm_opt.walk_forward import hold_out_split


def feature_ablation_sharpe(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    tl: List[str],
    basis: np.ndarray,
    oi: np.ndarray,
    p: TtmOptParams,
    bar_type: str,
) -> Dict[str, float]:
    """Relative contribution proxy: Sharpe when toggling feature groups off."""
    variants = [
        ("full", {}),
        ("no_squeeze", {"use_squeeze": False}),
        ("no_momentum", {"use_momentum": False}),
        ("no_basis", {"use_basis": False}),
        ("no_oi", {"use_oi": False}),
        ("price_only", {"use_basis": False, "use_oi": False}),
    ]
    out: Dict[str, float] = {}
    base = p.to_dict()
    for name, overrides in variants:
        d = {**base, **overrides}
        pp = TtmOptParams.from_dict(d)
        _tr, _eq, pack = run_ttm_opt_backtest(o, h, l, c, tl, basis, oi, pp, bar_type=bar_type)
        out[name] = float(pack["metrics"].sharpe_ratio)
    return out


def evaluate_is_oos(
    bars: Sequence[OhlcBar],
    basis: np.ndarray,
    oi: np.ndarray,
    p: TtmOptParams,
    bar_type: str,
    holdout_frac: float = 0.25,
) -> Tuple[BacktestMetrics, BacktestMetrics, np.ndarray, np.ndarray]:
    o, h, l, c, tl = bars_to_arrays(bars)
    n = len(c)
    if len(basis) != n:
        basis = np.zeros(n, dtype=np.float64)
    if len(oi) != n:
        oi = np.zeros(n, dtype=np.float64)
    sl_is, sl_oos = hold_out_split(n, holdout_frac=holdout_frac)
    tl_is = tl[sl_is.start : sl_is.stop]
    tl_oos = tl[sl_oos.start : sl_oos.stop]
    _, eq_is, pk_is = run_ttm_opt_backtest(
        o[sl_is], h[sl_is], l[sl_is], c[sl_is], tl_is,
        basis[sl_is], oi[sl_is], p, bar_type=bar_type,
    )
    _, eq_oos, pk_oos = run_ttm_opt_backtest(
        o[sl_oos], h[sl_oos], l[sl_oos], c[sl_oos], tl_oos,
        basis[sl_oos], oi[sl_oos], p, bar_type=bar_type,
    )
    return pk_is["metrics"], pk_oos["metrics"], eq_is, eq_oos


def write_top_report(
    ranked: List[Tuple[float, TtmOptParams, Dict[str, Any]]],
    bars: Sequence[OhlcBar],
    basis: np.ndarray,
    oi: np.ndarray,
    bar_type: str,
    out_dir: Path,
    top_k: int = 5,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    o, h, l, c, tl = bars_to_arrays(bars)
    n = len(c)
    if len(basis) != n:
        basis = np.zeros(n, dtype=np.float64)
    if len(oi) != n:
        oi = np.zeros(n, dtype=np.float64)

    seen: set = set()
    rows: List[Dict[str, Any]] = []
    for agg, p, meta in ranked[: top_k * 3]:
        key = json.dumps(p.to_dict(), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        m_is, m_oos, eq_is, eq_oos = evaluate_is_oos(bars, basis, oi, p, bar_type)
        abl = feature_ablation_sharpe(o, h, l, c, tl, basis, oi, p, bar_type)
        row = {
            "rank": len(rows) + 1,
            "aggregate_walk_forward_score": agg,
            "params": p.to_dict(),
            "fold_train_scores": meta.get("train_scores", []),
            "fold_test_scores": meta.get("test_scores", []),
            "walk_forward_diag": meta.get("walk_forward_diag", {}),
            "in_sample_metrics": m_is.to_dict(),
            "out_of_sample_metrics": m_oos.to_dict(),
            "feature_ablation_sharpe": abl,
            "oos_vs_is_md": compare_metrics(m_is, m_oos),
        }
        rows.append(row)
        rk = len(rows)
        np.save(out_dir / f"equity_rank{rk}_is.npy", eq_is)
        np.save(out_dir / f"equity_rank{rk}_oos.npy", eq_oos)
        if len(rows) >= top_k:
            break

    path = out_dir / "ttm_opt_top.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    return path
