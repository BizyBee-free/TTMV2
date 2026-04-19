#!/usr/bin/env python3
"""
Grid backtest: TTM V2 execution realism (latency × slippage) on historical OHLC.

Does not change signal generation — only V2 simulated fills (see ``ttm_execution_realism``).

Example:
  python scripts/ttm_v2_execution_grid.py --from-date 20260301 --to-date 20260331 --resolution 15

Output: ``reports/ttm_v2_execution_grid_<symbol>_<from>_<to>_<stamp>.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtest.data_fetcher import DataFetcher, OhlcBar
from src.config import get_settings
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm.oi_data import OICache, align_open_interest_to_bars
from src.strategies.ttm.ttm_execution_realism import (
    ExecutionRealismConfig,
    resolution_to_bar_seconds,
    v2_metrics_from_trade_jsonl_rows,
)
from src.strategies.ttm.ttm_parallel_runner import build_ttm_paper_live_config, replay_bars


def _oi_cache_path(symbol: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in symbol)
    return _ROOT / "data" / "cache" / "oi" / f"{safe}.jsonl"


def load_bars_basis_oi(
    symbol: str,
    from_date: str,
    to_date: str,
    resolution: str,
    fetcher: DataFetcher,
    settings,
) -> Tuple[List[OhlcBar], np.ndarray, np.ndarray]:
    basis = np.array([], dtype=np.float64)
    oi = np.array([], dtype=np.float64)

    if settings.HMM_USE_BASIS:
        print("[TTM V2 grid] Loading future + index (basis)...", flush=True)
        bars, index_closes, _ = fetch_aligned_future_index(
            fetcher, symbol, settings.HMM_INDEX_SYMBOL, from_date, to_date, resolution
        )
        if index_closes and len(index_closes) == len(bars):
            basis = np.array(
                [float(b.close) - float(ic) for b, ic in zip(bars, index_closes)],
                dtype=np.float64,
            )
        if not bars:
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
    else:
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
        oi = np.zeros(len(bars), dtype=np.float64)

    if len(basis) != len(bars):
        basis = np.zeros(len(bars), dtype=np.float64)
    if len(oi) != len(bars):
        oi = np.zeros(len(bars), dtype=np.float64)

    return bars, basis, oi


def _read_trade_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTM V2 execution realism grid (latency × slippage)")
    p.add_argument("--symbol", default="VN30F1M")
    p.add_argument("--from-date", default="20260301")
    p.add_argument("--to-date", default="20260331")
    p.add_argument("--resolution", default="15")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--latencies",
        default="0,100,300,500",
        help="Comma-separated latency_ms values",
    )
    p.add_argument(
        "--slippage-modes",
        default="none,base,worst_case",
        help="Comma-separated: none | base | worst_case",
    )
    return p.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    settings = get_settings()
    fetcher = DataFetcher(settings=settings, use_cache=not args.no_cache)

    bars, basis, oi = load_bars_basis_oi(
        args.symbol,
        args.from_date,
        args.to_date,
        args.resolution,
        fetcher,
        settings,
    )
    n = len(bars)
    if n == 0:
        print("[TTM V2 grid] Error: no bars.", flush=True)
        return 1

    bar_sec = resolution_to_bar_seconds(args.resolution)
    basis_list = basis.tolist()
    oi_list = oi.tolist()
    data_series: List[dict] = []
    timestamps: List[str] = []
    for i in range(n):
        data_series.append(
            {
                "bars": bars[: i + 1],
                "basis": basis_list[: i + 1],
                "open_interest": oi_list[: i + 1],
            }
        )
        timestamps.append(str(bars[i].unix_ts))

    latencies = [int(x.strip()) for x in args.latencies.split(",") if x.strip()]
    slip_modes = [x.strip() for x in args.slippage_modes.split(",") if x.strip()]
    cfg_base = build_ttm_paper_live_config()
    cfg_base["ttm_execution_bar_seconds"] = int(bar_sec)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_sym = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.symbol)
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_results: List[Dict[str, Any]] = []

    for latency_ms in latencies:
        for slip in slip_modes:
            er = ExecutionRealismConfig(
                latency_ms=int(latency_ms),
                slippage_mode=slip,  # type: ignore[arg-type]
                bar_seconds=int(bar_sec),
            )
            trd_path = (
                out_dir
                / f"_ttm_v2_grid_{safe_sym}_{args.from_date}_{args.to_date}_lat{latency_ms}_{slip}_{stamp}.jsonl"
            )
            summary = replay_bars(
                data_series,
                config=cfg_base,
                timestamps=timestamps,
                trade_log_path=str(trd_path),
                execution_realism=er,
                execution_bar_seconds=int(bar_sec),
            )
            trade_rows = _read_trade_rows(trd_path)
            metrics = v2_metrics_from_trade_jsonl_rows(
                trade_rows,
                n_bars=n,
                bar_type=args.resolution,
                nav_start=100.0,
            )
            grid_results.append(
                {
                    "latency_ms": int(latency_ms),
                    "slippage_mode": str(slip),
                    "summary_v2": summary.get("v2"),
                    "total_pnl_v2": float(metrics.total_pnl),
                    "sharpe_ratio": float(metrics.sharpe_ratio),
                    "max_drawdown_pct": float(metrics.max_drawdown_pct),
                    "n_trades_v2": int(metrics.n_trades),
                    "trades_jsonl": str(trd_path),
                }
            )

    out_path = out_dir / f"ttm_v2_execution_grid_{safe_sym}_{args.from_date}_{args.to_date}_{stamp}.json"
    payload = {
        "symbol": args.symbol,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "resolution": args.resolution,
        "bar_seconds": int(bar_sec),
        "n_bars": n,
        "grid": grid_results,
        "notes": {
            "latency": "Entry fill at first bar open with bar.unix_ts >= signal_close + latency_ms/1000.",
            "slippage": "0.5*bid_ask_proxy + 0.1*vol; worst_case = 2x base. Exit triggers unchanged (bar close); fill applies slippage.",
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"[TTM V2 grid] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
