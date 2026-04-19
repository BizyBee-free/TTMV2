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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest TTM parallel V1+V2 trên OHLC lịch sử")
    p.add_argument("--symbol", default="VN30F1M")
    p.add_argument("--from-date", default="20260301", help="YYYYMMDD")
    p.add_argument("--to-date", default="20260331", help="YYYYMMDD")
    p.add_argument("--resolution", default="15", help="DNSE: 1,5,15,30,60,1D")
    p.add_argument("--no-cache", action="store_true", help="Không đọc/ghi cache OHLC (gọi API mỗi lần)")
    p.add_argument("--out-dir", default=None, help="Mặc định: reports/")
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_sym = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.symbol)
    out_dir = Path(args.out_dir) if args.out_dir else _ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    dec_path = out_dir / f"ttm_parallel_backtest_{safe_sym}_{args.from_date}_{args.to_date}_{stamp}_decisions.jsonl"
    trd_path = out_dir / f"ttm_parallel_backtest_{safe_sym}_{args.from_date}_{args.to_date}_{stamp}_trades.jsonl"

    cfg = build_ttm_paper_live_config()
    summary = replay_bars(
        data_series,
        config=cfg,
        timestamps=timestamps,
        decision_log_path=str(dec_path),
        trade_log_path=str(trd_path),
    )

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
