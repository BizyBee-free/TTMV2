#!/usr/bin/env python3
"""
TTM Optimization v2 — TTM Squeeze + Basis + OI, walk-forward, Optuna (hoặc random search).

Ví dụ:
  python scripts/ttm_optimize.py --from-date 20240101 --to-date 20260325 --resolution 15 --trials 40
  python scripts/ttm_optimize.py --no-optuna --trials 80 --out-dir reports/ttm_opt_run1

Đầu ra: ``reports/ttm_opt/`` (hoặc ``--out-dir``): ``ttm_opt_top.json``, equity *.npy, log tiếng Việt trên terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtest.data_fetcher import DataFetcher, OhlcBar
from src.backtest.ttm_opt.optimize import run_optimization
from src.backtest.ttm_opt.report import write_top_report
from src.backtest.ttm_opt.robustness import monte_carlo_equity, monte_carlo_worst_case
from src.config import get_settings
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm.oi_data import OICache, align_open_interest_to_bars
def _bar_type_from_resolution(resolution: str) -> str:
    r = (resolution or "15").upper()
    if r in {"1D", "D"}:
        return "D"
    if r in {"60", "1H", "H"}:
        return "60"
    if r == "30":
        return "30"
    if r == "15":
        return "15"
    if r in {"5", "1"}:
        return "5"
    if r in {"240", "4H"}:
        return "240"
    return "15"


def _oi_cache_path(symbol: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in symbol)
    return _ROOT / "data" / "cache" / "oi" / f"{safe}.jsonl"


def load_bars_basis_oi(
    symbol: str,
    from_date: str,
    to_date: str,
    resolution: str,
) -> tuple[list[OhlcBar], np.ndarray, np.ndarray]:
    fetcher = DataFetcher(settings=get_settings())
    settings = get_settings()
    basis = np.array([], dtype=np.float64)
    oi = np.array([], dtype=np.float64)

    if settings.HMM_USE_BASIS:
        print("[TTM-OPT] Đang tải future + index để tính basis…", flush=True)
        bars, index_closes, _ = fetch_aligned_future_index(
            fetcher, symbol, settings.HMM_INDEX_SYMBOL, from_date, to_date, resolution
        )
        if index_closes and len(index_closes) == len(bars):
            basis = np.array(
                [float(b.close) - float(ic) for b, ic in zip(bars, index_closes)],
                dtype=np.float64,
            )
    else:
        print("[TTM-OPT] HMM_USE_BASIS=false — basis = 0.", flush=True)
        bars = fetcher.fetch(
            symbol,
            from_date,
            to_date,
            resolution=resolution,
            asset_type="derivative",
        )
        if not bars:
            bars = fetcher.fetch(
                "VN30",
                from_date,
                to_date,
                resolution=resolution,
                asset_type="index",
            )
        basis = np.zeros(len(bars), dtype=np.float64)

    if not bars:
        return [], basis, oi

    cache = OICache(_oi_cache_path(symbol))
    pts = cache.load_points()
    if pts:
        oi_list = align_open_interest_to_bars(bars, pts)
        oi = np.array(oi_list, dtype=np.float64) if oi_list else np.zeros(len(bars), dtype=np.float64)
    else:
        print("[TTM-OPT] Không có cache OI — dùng OI=0 (chỉ squeeze + basis + giá).", flush=True)
        oi = np.zeros(len(bars), dtype=np.float64)

    if len(basis) != len(bars):
        basis = np.zeros(len(bars), dtype=np.float64)
    if len(oi) != len(bars):
        oi = np.zeros(len(bars), dtype=np.float64)

    return bars, basis, oi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTM walk-forward + Bayesian/random optimization")
    p.add_argument("--symbol", default="VN30F1M")
    p.add_argument("--from-date", default="20240101")
    p.add_argument("--to-date", default="20260325")
    p.add_argument("--resolution", default="15", help="DNSE: 1,5,15,30,60,1D")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-months", type=float, default=27.0)
    p.add_argument("--test-months", type=float, default=12.0)
    p.add_argument("--no-optuna", action="store_true", help="Random search thay vì Optuna")
    p.add_argument("--out-dir", default=None, help="Mặc định: reports/ttm_opt")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--mc-sims", type=int, default=10000, help="Monte Carlo trên chuỗi PnL lệnh (top-1)")
    p.add_argument("--gate-sharpe-min", type=float, default=1.8)
    p.add_argument("--gate-pf-min", type=float, default=2.0)
    p.add_argument("--gate-dd-max", type=float, default=18.0)
    p.add_argument("--gate-trades-min", type=float, default=300.0)
    p.add_argument("--gate-expectancy-min", type=float, default=0.0)
    p.add_argument("--gate-wfe-min", type=float, default=0.75)
    p.add_argument("--gate-oos-degradation-max", type=float, default=0.25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "reports" / "ttm_opt"
    out_dir.mkdir(parents=True, exist_ok=True)

    bars, basis, oi = load_bars_basis_oi(args.symbol, args.from_date, args.to_date, args.resolution)
    if len(bars) < 200:
        print(f"[TTM-OPT] Lỗi: quá ít nến ({len(bars)}). Kiểm tra symbol / ngày / resolution.", flush=True)
        return 1

    bar_type = _bar_type_from_resolution(args.resolution)
    print(
        f"[TTM-OPT] Đã nạp {len(bars)} nến | res={args.resolution} | bar_type={bar_type} | "
        f"basis_nonzero={np.any(np.abs(basis) > 1e-9)} | oi_nonzero={np.any(oi > 0)}",
        flush=True,
    )

    use_optuna = not args.no_optuna
    try:
        import optuna  # noqa: F401
    except ImportError:
        if use_optuna:
            print("[TTM-OPT] Optuna chưa cài — chuyển sang random search (pip install optuna).", flush=True)
        use_optuna = False

    ranked = run_optimization(
        bars,
        basis,
        oi,
        bar_type=bar_type,
        resolution=args.resolution,
        n_trials=args.trials,
        seed=args.seed,
        train_months=args.train_months,
        test_months=args.test_months,
        use_optuna=use_optuna,
        hard_gates={
            "sharpe_min": float(args.gate_sharpe_min),
            "profit_factor_min": float(args.gate_pf_min),
            "max_drawdown_pct_max": float(args.gate_dd_max),
            "trades_min": float(args.gate_trades_min),
            "expectancy_min": float(args.gate_expectancy_min),
            "wfe_min": float(args.gate_wfe_min),
            "oos_degradation_max": float(args.gate_oos_degradation_max),
        },
    )

    if not ranked:
        print("[TTM-OPT] Không có kết quả trial hợp lệ.", flush=True)
        return 1

    top_path = write_top_report(ranked, bars, basis, oi, bar_type, out_dir, top_k=args.top_k)
    print(f"[TTM-OPT] Đã ghi top configs: {top_path}", flush=True)

    # Monte Carlo trên PnL giao dịch của top-1 (độ bền đường equity)
    from src.backtest.ttm_opt.engine import run_ttm_opt_backtest
    from src.backtest.ttm_opt.optimize import bars_to_arrays

    _agg0, p0, _meta0 = ranked[0]
    o, h, l, c, tl = bars_to_arrays(bars)
    tr, _, _pk = run_ttm_opt_backtest(o, h, l, c, tl, basis, oi, p0, bar_type=bar_type)
    pnls = np.array([t.net_pnl for t in tr], dtype=np.float64)
    mc = monte_carlo_equity(pnls, n_sims=args.mc_sims, seed=args.seed)
    mc_wc = monte_carlo_worst_case(pnls, n_sims=args.mc_sims, seed=args.seed)
    print(
        f"[TTM-OPT] Monte Carlo PnL (top-1): median_final={mc['median_final']:.4f} "
        f"p5={mc['p5_final']:.4f} p95={mc['p95_final']:.4f}",
        flush=True,
    )
    print(
        f"[TTM-OPT] Monte Carlo worst-case DD: p95={mc_wc['dd_pct_p95']:.2f}% "
        f"p99={mc_wc['dd_pct_p99']:.2f}% worst={mc_wc['dd_pct_worst']:.2f}%",
        flush=True,
    )

    best = ranked[0]
    wf_diag = best[2].get("walk_forward_diag", {})
    print(
        f"[TTM-OPT] Tốt nhất: điểm WF={best[0]:.4f} | regime={best[1].regime_filter} | "
        f"SL={best[1].stop_loss_pct:.4f} TP={best[1].take_profit_pct:.4f}",
        flush=True,
    )
    print(
        "[TTM-OPT] Hard-gate status: "
        f"passed={wf_diag.get('hard_gate_passed')} "
        f"wfe={wf_diag.get('walk_forward_efficiency')} "
        f"oos_deg={wf_diag.get('oos_degradation')} "
        f"reasons={wf_diag.get('hard_gate_reasons')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
