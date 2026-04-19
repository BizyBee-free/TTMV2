"""Run HMM walk-forward on a date range and print Sharpe, P&L, MDD (and context).

Uses ``fetch_hmm_bars`` (basis + optional OI from ``data/cache/oi/{symbol}.jsonl``).

Examples::

    python scripts/hmm_report_period.py --from 20260101 --to 20260328 --resolution 15
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _bar_type(resolution: str) -> str:
    if resolution == "1D":
        return "D"
    if resolution == "1H":
        return "60"
    if resolution == "30":
        return "30"
    if resolution == "15":
        return "15"
    if resolution == "5":
        return "5"
    return "D"


def main() -> int:
    p = argparse.ArgumentParser(description="HMM metrics for a custom period")
    p.add_argument("--from", dest="from_d", default="20260101", help="YYYYMMDD")
    p.add_argument("--to", dest="to_d", default="20260328", help="YYYYMMDD")
    p.add_argument("--resolution", default="15", help="1D, 1H, 30, 15, 5")
    p.add_argument("--symbol", default=None, help="Default: HMM_SYMBOL from settings")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--confidence", type=float, default=0.65)
    p.add_argument("--hmm-states", type=int, default=3)
    p.add_argument("--hmm-iter", type=int, default=120, help="Baum-Welch iterations (lower = faster)")
    p.add_argument(
        "--refit-every",
        type=int,
        default=15,
        help="Higher = faster on long 15m histories (default 15)",
    )
    p.add_argument("--basis", dest="use_basis", action="store_true", default=True)
    p.add_argument("--no-basis", dest="use_basis", action="store_false")
    p.add_argument("--oi", dest="use_oi", action="store_true", default=True)
    p.add_argument("--no-oi", dest="use_oi", action="store_false")
    args = p.parse_args()

    os.environ["HMM_USE_BASIS"] = "true" if args.use_basis else "false"
    os.environ["HMM_USE_OPEN_INTEREST"] = "true" if args.use_oi else "false"

    logging.getLogger("hmmlearn").setLevel(logging.ERROR)

    from scripts.backtest import fetch_hmm_bars
    from src.backtest.data_fetcher import DataFetcher
    from src.backtest.hmm_replay import HMMBarReplay
    from src.backtest.metrics import compute_metrics
    from src.config import get_settings

    settings = get_settings()
    sym = args.symbol or settings.HMM_SYMBOL

    ns = SimpleNamespace(
        symbol=sym,
        warmup=args.warmup,
        confidence=args.confidence,
        hmm_states=args.hmm_states,
        hmm_iter=args.hmm_iter,
        refit_every=args.refit_every,
        sliding_window_bars=settings.HMM_LIVE_TRAIN_WINDOW_BARS,
        commission=0.0,
    )

    fetcher = DataFetcher(settings=settings)
    bars, cfg = fetch_hmm_bars(fetcher, ns, args.from_d, args.to_d, args.resolution, symbol=sym)

    oi_on = cfg.hmm_config.use_open_interest and cfg.open_interest is not None
    basis_on = cfg.hmm_config.use_basis
    nf = 3 + (1 if basis_on else 0) + (1 if cfg.hmm_config.use_open_interest else 0)

    print()
    print("=== HMM backtest report ===")
    print(f"  Symbol:      {sym}")
    print(f"  Period:      {args.from_d} -> {args.to_d}")
    print(f"  Resolution:  {args.resolution}  (metrics bar_type={_bar_type(args.resolution)})")
    print(f"  Bars:        {len(bars)}")
    print(f"  Basis:       {basis_on}")
    print(f"  OI feature:  {oi_on}  (requires data/cache/oi/*.jsonl)")
    print(f"  n_features:  {nf}")

    if len(bars) < 3:
        print("  ERROR: insufficient bars.")
        return 1

    replay = HMMBarReplay()
    trades, equity = replay.run(bars, cfg)
    m = compute_metrics(trades, equity, len(bars), bar_type=_bar_type(args.resolution))

    print()
    print("--- Results ---")
    print(f"  Sharpe ratio:     {m.sharpe_ratio:.6f}")
    print(f"  Total P&L:        {m.total_pnl:.6f}")
    print(f"  Max drawdown %:   {m.max_drawdown_pct:.4f}")
    print(f"  Total return %:   {m.total_return_pct:.4f}")
    print(f"  Trades:           {m.n_trades}")
    print(f"  Win rate %:       {m.win_rate_pct:.2f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
