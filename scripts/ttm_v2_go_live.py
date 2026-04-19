#!/usr/bin/env python3
"""End-to-end TTM V2 optimization + go-live acceptance report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtest.ttm_opt.engine import run_ttm_opt_backtest
from src.backtest.ttm_opt.optimize import bars_to_arrays, run_optimization
from src.backtest.ttm_opt.robustness import (
    monte_carlo_worst_case,
    net_to_gross_ratio,
    regime_stability_score_5,
)
from src.strategies.ttm.ttm_validation import run_validation, run_validation_multiday
from scripts.ttm_optimize import _bar_type_from_resolution, load_bars_basis_oi


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTM V2 go-live optimizer + acceptance gate")
    p.add_argument("--symbol", default="VN30F1M")
    p.add_argument("--from-date", default="20230101")
    p.add_argument("--to-date", default="20260414")
    p.add_argument("--resolution", default="15")
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-months", type=float, default=27.0)
    p.add_argument("--test-months", type=float, default=12.0)
    p.add_argument("--mc-sims", type=int, default=10000)
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    p.add_argument("--validation-start-date", type=str, default="20260413")
    p.add_argument("--validation-end-date", type=str, default="20260414")
    p.add_argument("--validation-closes", type=Path, default=Path("reports/closes_VN30F1M_20260413_20260414_merged.json"))
    p.add_argument("--validation-decisions", type=Path, default=None)
    p.add_argument("--validation-trades", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _criterion(ok: bool, value: Any, target: str) -> Dict[str, Any]:
    return {"passed": bool(ok), "value": value, "target": target}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = _parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    bars, basis, oi = load_bars_basis_oi(args.symbol, args.from_date, args.to_date, args.resolution)
    if len(bars) < 300:
        raise RuntimeError(f"insufficient bars for optimization: {len(bars)}")

    bar_type = _bar_type_from_resolution(args.resolution)
    ranked = run_optimization(
        bars,
        basis,
        oi,
        bar_type=bar_type,
        resolution=args.resolution,
        n_trials=int(args.trials),
        seed=int(args.seed),
        train_months=float(args.train_months),
        test_months=float(args.test_months),
        use_optuna=True,
    )
    if not ranked:
        raise RuntimeError("run_optimization returned no candidates")

    best_score, best_params, best_meta = ranked[0]
    o, h, l, c, tl = bars_to_arrays(bars)
    trades, _eq, pack = run_ttm_opt_backtest(o, h, l, c, tl, basis, oi, best_params, bar_type=bar_type)
    metrics = pack["metrics"]

    pnls = np.array([float(t.net_pnl) for t in trades], dtype=np.float64)
    mc = monte_carlo_worst_case(pnls, n_sims=int(args.mc_sims), seed=int(args.seed))
    regime = regime_stability_score_5(pack.get("regime_pnl", {}))
    net_gross = net_to_gross_ratio(trades)
    wf_diag = best_meta.get("walk_forward_diag", {})

    if args.validation_decisions and args.validation_trades:
        validation_result = run_validation(
            args.validation_decisions,
            args.validation_trades,
            closes_path=args.validation_closes if args.validation_closes and args.validation_closes.exists() else None,
            scoring_mode="all",
        )
    else:
        validation_result = run_validation_multiday(
            reports_dir=args.reports_dir,
            cache_dir=args.cache_dir,
            symbol=args.symbol,
            start_date=args.validation_start_date,
            end_date=args.validation_end_date,
            scoring_mode="all",
            closes_path=args.validation_closes if args.validation_closes and args.validation_closes.exists() else None,
        )

    criteria = {
        "validation_blocks": _criterion(
            bool(validation_result.get("overall_valid", False)),
            bool(validation_result.get("overall_valid", False)),
            "all of breakout/positioning/scoring/adaptive pass",
        ),
        "walk_forward_efficiency": _criterion(
            float(wf_diag.get("walk_forward_efficiency", 0.0)) >= 0.75,
            float(wf_diag.get("walk_forward_efficiency", 0.0)),
            ">= 0.75",
        ),
        "monte_carlo_dd_95": _criterion(
            float(mc.get("dd_pct_p95", 999.0)) <= 18.0,
            float(mc.get("dd_pct_p95", 999.0)),
            "<= 18.0",
        ),
        "oos_degradation": _criterion(
            float(wf_diag.get("oos_degradation", 1.0)) <= 0.25,
            float(wf_diag.get("oos_degradation", 1.0)),
            "<= 0.25",
        ),
        "sharpe_full_cycle": _criterion(float(metrics.sharpe_ratio) >= 1.8, float(metrics.sharpe_ratio), ">= 1.8"),
        "profit_factor": _criterion(float(metrics.profit_factor) >= 2.0, float(metrics.profit_factor), ">= 2.0"),
        "trades_and_expectancy": _criterion(
            int(metrics.n_trades) >= 300 and float(metrics.expectancy) > 0.0,
            {"n_trades": int(metrics.n_trades), "expectancy": float(metrics.expectancy)},
            "n_trades >= 300 and expectancy > 0",
        ),
        "regime_stability": _criterion(
            float(regime.get("score", 0.0)) >= 0.85,
            float(regime.get("score", 0.0)),
            ">= 0.85",
        ),
        "net_after_costs": _criterion(float(net_gross) >= 0.70, float(net_gross), ">= 0.70 gross"),
    }
    go_live_ready = all(bool(v.get("passed")) for v in criteria.values())

    out: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "symbol": args.symbol,
            "from_date": args.from_date,
            "to_date": args.to_date,
            "resolution": args.resolution,
            "trials": int(args.trials),
            "train_months": float(args.train_months),
            "test_months": float(args.test_months),
            "mc_sims": int(args.mc_sims),
        },
        "best_candidate": {
            "aggregate_score": float(best_score),
            "params": best_params.to_dict(),
            "walk_forward_diag": wf_diag,
        },
        "full_cycle_metrics": metrics.to_dict(),
        "robustness": {
            "monte_carlo": mc,
            "regime_stability": regime,
            "net_to_gross_ratio": float(net_gross),
            "regime_pnl": pack.get("regime_pnl", {}),
        },
        "validation": validation_result,
        "criteria": criteria,
        "go_live_ready": bool(go_live_ready),
    }

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.out = args.reports_dir / f"ttm_v2_go_live_summary_{args.symbol}_{stamp}.json"
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"go_live_ready": go_live_ready, "out": str(args.out)}, ensure_ascii=False))
    return 0 if go_live_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())

