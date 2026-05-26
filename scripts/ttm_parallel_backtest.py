#!/usr/bin/env python3
"""
Backtest song song TTM V1 + V2 (JSONL) trên OHLC lịch sử DNSE.

Dùng cùng nguồn dữ liệu như ``ttm_optimize.py``: basis (future − index nếu
``HMM_USE_BASIS``), OI từ JSONL cache (``data/cache/oi/``) nếu có, không thì OI=0.

Ví dụ (tháng 3/2026):
  python scripts/ttm_parallel_backtest.py --from-date 20260301 --to-date 20260331
  python scripts/ttm_parallel_backtest.py --from-date 20260301 --to-date 20260331 --resolution 15 --symbol VN30F1M

Đầu ra: ``reports/ttm_parallel_backtest_<symbol>_<from>_<to>_<stamp>.{jsonl}``
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.backtest.data_fetcher import DataFetcher, OhlcBar
from src.config import get_settings
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm.oi_data import OICache, align_open_interest_to_bars
from src.strategies.ttm.empirical.adaptive_engine import EmpiricalAlphaEngine
from src.strategies.ttm.ttm_features import compute_ttm_features_from_config
from src.strategies.ttm.ttm_parallel_runner import (
    ParallelRunner,
    build_ttm_paper_live_config,
    build_ttm_research_parallel_config,
)
from src.strategies.ttm.ttm_v2_gates import (
    GATE_MODE_EXPLORATORY_RESEARCH,
    GATE_MODE_QUALITY_RESEARCH,
    GATE_MODE_RESEARCH_PAPER,
    GATE_MODE_SHADOW_ONLY,
    GATE_MODE_STRICT,
)


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
    """Giống ``scripts/ttm_optimize.load_bars_basis_oi`` (tách để script độc lập)."""
    basis = np.array([], dtype=np.float64)
    oi = np.array([], dtype=np.float64)

    if settings.HMM_USE_BASIS:
        print("[TTM parallel BT] Loading future + index (basis)...", flush=True)
        bars, index_closes, _ = fetch_aligned_future_index(
            fetcher, symbol, settings.HMM_INDEX_SYMBOL, from_date, to_date, resolution
        )
        if index_closes and len(index_closes) == len(bars):
            basis = np.array(
                [float(b.close) - float(ic) for b, ic in zip(bars, index_closes)],
                dtype=np.float64,
            )
        if not bars:
            print(
                "[TTM parallel BT] Future+index empty; fallback derivative/VN30 index, basis=0.",
                flush=True,
            )
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
        print("[TTM parallel BT] HMM_USE_BASIS=false; basis=0.", flush=True)
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
        print("[TTM parallel BT] No OI cache; using OI=0.", flush=True)
        oi = np.zeros(len(bars), dtype=np.float64)

    if len(basis) != len(bars):
        basis = np.zeros(len(bars), dtype=np.float64)
    if len(oi) != len(bars):
        oi = np.zeros(len(bars), dtype=np.float64)

    return bars, basis, oi


def resolution_bar_seconds(resolution: str) -> int:
    """DNSE resolution string -> bar length in seconds (aligns holding_period with wall clock)."""
    key = str(resolution or "15").strip().upper()
    return {
        "1": 60,
        "5": 300,
        "15": 900,
        "30": 1800,
        "60": 3600,
        "1D": 86400,
    }.get(key, 900)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest TTM parallel V1+V2 trên OHLC lịch sử")
    p.add_argument("--symbol", default="VN30F1M")
    p.add_argument("--from-date", default="20260301", help="YYYYMMDD")
    p.add_argument("--to-date", default="20260331", help="YYYYMMDD")
    p.add_argument("--resolution", default="15", help="DNSE: 1,5,15,30,60,1D")
    p.add_argument("--no-cache", action="store_true", help="Không đọc/ghi cache OHLC (gọi API mỗi lần)")
    p.add_argument("--out-dir", default=None, help="Mặc định: reports/")
    p.add_argument(
        "--gate-mode",
        default="quality_research",
        choices=(
            GATE_MODE_STRICT,
            GATE_MODE_QUALITY_RESEARCH,
            GATE_MODE_EXPLORATORY_RESEARCH,
            GATE_MODE_RESEARCH_PAPER,
            GATE_MODE_SHADOW_ONLY,
        ),
        help="TTM V2 gate mode (default: quality_research for paper/backtest preset)",
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
        print(
            "[TTM parallel BT] Error: no bars. Check API/credentials, symbol, dates, resolution.",
            flush=True,
        )
        return 1

    print(
        f"[TTM parallel BT] {n} nến | {args.from_date}–{args.to_date} | res={args.resolution} | "
        f"basis_nonzero={bool(np.any(np.abs(basis) > 1e-9))} | oi_nonzero={bool(np.any(oi > 0))}",
        flush=True,
    )

    basis_list = basis.tolist()
    oi_list = oi.tolist()
    timestamps = [str(bars[i].unix_ts) for i in range(n)]
    full_raw = {"bars": bars, "basis": basis_list, "open_interest": oi_list}

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_sym = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.symbol)
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    dec_path = out_dir / f"ttm_parallel_backtest_{safe_sym}_{args.from_date}_{args.to_date}_{stamp}_decisions.jsonl"
    trd_path = out_dir / f"ttm_parallel_backtest_{safe_sym}_{args.from_date}_{args.to_date}_{stamp}_trades.jsonl"

    if args.gate_mode == GATE_MODE_STRICT:
        cfg = build_ttm_research_parallel_config(
            {
                "ttm_v2_gate_mode": GATE_MODE_STRICT,
                "ttm_v2_allow_signal_only_long_execution": False,
            }
        )
        print(f"[TTM parallel BT] gate_mode={GATE_MODE_STRICT}", flush=True)
    elif args.gate_mode == GATE_MODE_SHADOW_ONLY:
        cfg = build_ttm_research_parallel_config(
            {
                "ttm_v2_gate_mode": GATE_MODE_SHADOW_ONLY,
                "ttm_v2_allow_signal_only_long_execution": True,
            }
        )
        print(f"[TTM parallel BT] gate_mode={GATE_MODE_SHADOW_ONLY}", flush=True)
    else:
        gm = GATE_MODE_QUALITY_RESEARCH if args.gate_mode == GATE_MODE_RESEARCH_PAPER else args.gate_mode
        cfg = build_ttm_paper_live_config({"ttm_v2_gate_mode": gm})
        print(f"[TTM parallel BT] gate_mode={gm}", flush=True)
    bar_sec = resolution_bar_seconds(args.resolution)
    cfg["ttm_execution_bar_seconds"] = int(bar_sec)
    print(f"[TTM parallel BT] bar_seconds={bar_sec}", flush=True)
    print("[TTM parallel BT] Precomputing features (one pass)...", flush=True)
    full_feats = compute_ttm_features_from_config(full_raw, cfg)
    empirical_engine = None
    if bool(cfg.get("ttm_v2_empirical_alpha_enabled", False)):
        empirical_engine = EmpiricalAlphaEngine(cfg)
    runner = ParallelRunner(
        cfg,
        decision_log_path=str(dec_path),
        trade_log_path=str(trd_path),
        execution_bar_seconds=bar_sec,
        empirical_engine=empirical_engine,
    )
    progress_every = max(500, n // 20)
    for i in range(n):
        if i > 0 and i % progress_every == 0:
            print(f"[TTM parallel BT] replay {i}/{n} bars...", flush=True)
        runner.on_new_bar(
            i,
            timestamps[i],
            {
                **full_raw,
                "precomputed_feats": full_feats,
                "feat_bar_index": i,
            },
        )
    summary = runner.close()

    meta_path = out_dir / f"ttm_parallel_backtest_{safe_sym}_{args.from_date}_{args.to_date}_{stamp}_summary.json"
    meta = {
        "symbol": args.symbol,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "resolution": args.resolution,
        "n_bars": n,
        "decisions_jsonl": str(dec_path),
        "trades_jsonl": str(trd_path),
        "summary": summary,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[TTM parallel BT] decisions: {dec_path}", flush=True)
    print(f"[TTM parallel BT] trades:    {trd_path}", flush=True)
    print(f"[TTM parallel BT] summary:   {meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
