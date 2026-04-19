"""Compare HMM backtest metrics: without basis (3 features) vs with basis (4 features).

Fetches aligned future + index OHLC, runs the same walk-forward replay twice,
and prints a Markdown table with Sharpe, total P&L, max drawdown %, timeframe,
and calendar period.

Usage::

    python scripts/compare_hmm_basis.py --from 20250901 --to 20251231 --resolution 1D
    python scripts/compare_hmm_basis.py --from 20260101 --to 20260320 --resolution 15 --symbol VN30F1M

Requires DNSE credentials in environment (same as ``scripts/backtest.py``).
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

from src.backtest.data_fetcher import DataFetcher
from src.config import get_settings
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm.basis_performance_compare import compare_basis_vs_no_basis, snapshots_to_markdown
from src.hmm.feature_engineer import HMMConfig


def _bar_type_from_resolution(resolution: str) -> str:
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


def _tf_label(resolution: str) -> str:
    if resolution == "1D":
        return "Daily 1D"
    if resolution == "1H":
        return "Hourly 1H"
    if resolution == "30":
        return "30m"
    if resolution == "15":
        return "15m"
    if resolution == "5":
        return "5m"
    return resolution


def main() -> int:
    p = argparse.ArgumentParser(description="Compare HMM with vs without basis (same data window)")
    p.add_argument("--from", dest="from_d", required=True, help="YYYYMMDD start")
    p.add_argument("--to", dest="to_d", required=True, help="YYYYMMDD end")
    p.add_argument("--resolution", default="1D", help="1D, 1H, 30, 15, 5")
    p.add_argument("--symbol", default=None, help="Future / derivative symbol (default: settings or VN30F1M)")
    p.add_argument("--index-symbol", default=None, help="Index for basis (default: HMM_INDEX_SYMBOL)")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--confidence", type=float, default=0.65)
    p.add_argument("--hmm-states", type=int, default=3)
    p.add_argument("--hmm-iter", type=int, default=200)
    p.add_argument("--refit-every", type=int, default=1)
    p.add_argument("--commission", type=float, default=0.0)
    args = p.parse_args()

    settings = get_settings()
    fut = args.symbol or settings.HMM_SYMBOL
    idx = args.index_symbol or settings.HMM_INDEX_SYMBOL

    fetcher = DataFetcher()
    bars, index_closes, _ = fetch_aligned_future_index(
        fetcher, fut, idx, args.from_d, args.to_d, args.resolution
    )
    if len(bars) < 30:
        print(f"ERROR: insufficient bars ({len(bars)}). Check symbol, dates, and API cache.")
        return 1

    bar_type = _bar_type_from_resolution(args.resolution)
    tf = _tf_label(args.resolution)

    base = HMMConfig(
        k_states=args.hmm_states,
        n_iter=args.hmm_iter,
        zscore_window=20,
        random_state=42,
        use_basis=False,
    )

    a, b = compare_basis_vs_no_basis(
        bars,
        index_closes,
        base_hmm_config=base,
        timeframe=tf,
        bar_type=bar_type,
        symbol=fut,
        warmup_bars=args.warmup,
        confidence_threshold=args.confidence,
        refit_every=args.refit_every,
        commission_pct=args.commission,
    )

    print()
    print(f"## HMM basis comparison — {fut} vs index {idx}")
    print(f"- **Resolution:** {args.resolution} ({tf})")
    print(f"- **Window:** {args.from_d} → {args.to_d}")
    print(f"- **Bars:** {len(bars)}")
    print()
    print(snapshots_to_markdown(a, b))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
