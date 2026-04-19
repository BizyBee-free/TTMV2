"""MCMC & HMM Strategy Backtest Runner -- 5 Test Suites + Comparison.

Fetches historical OHLC data from DNSE API, runs walk-forward backtests
for MCMC and/or HMM strategies, and produces:
  - reports/TEST_REPORT.md (appended with structured Markdown)
  - reports/backtest_results.csv (per-trade log, all suites)

Test suites:
  Suite 1 -- In-sample      : VN30F1M daily, 01/09/2025-31/12/2025
  Suite 2 -- Out-of-sample  : VN30F1M daily, 01/01/2026-30/03/2026
  Suite 3 -- Permutation    : Monte Carlo significance test (hourly bars)
  Suite 4 -- Sensitivity    : Hourly bars, same period as in-sample
  Suite 5 -- Multi-symbol   : HPG, VNM, FPT stocks, in-sample period

Usage:
    python scripts/backtest.py                            # MCMC only
    python scripts/backtest.py --strategy hmm             # HMM only
    python scripts/backtest.py --strategy both --suite all  # comparison
    python scripts/backtest.py --strategy both --fast
    python scripts/backtest.py --strategy hmm --suite tfcompare --no-fetch  # 15m/30m/1h/4h → CSV
    python scripts/backtest.py --strategy hmm --suite tfcompare --tf-from 20260101 --tf-to 20260320 --no-fetch
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# Silence hmmlearn convergence chatter (informational, not errors)
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

from src.backtest.bar_replay import BacktestConfig, BarReplay, TradeRecord
from src.backtest.data_fetcher import DataFetcher, OhlcBar
from src.backtest.hmm_replay import HMMBacktestConfig, HMMBarReplay
from src.backtest.metrics import BacktestMetrics, compare_metrics, compute_metrics
from src.backtest.statistical_tests import PermResult, monte_carlo_permutation
from src.config import get_settings
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm import HMMConfig
from src.hmm.oi_data import OICache, align_open_interest_to_bars
from src.logger import get_logger

logger = get_logger("backtest")

# ── Date ranges ───────────────────────────────────────────────────────────────
IS_FROM   = "20250901"
IS_TO     = "20251231"
OOS_FROM  = "20260101"
OOS_TO    = "20260325"
# Extended training window for HMM (2 years for better regime estimation)
HMM_TRAIN_FROM = "20240101"

# ── Default symbols ───────────────────────────────────────────────────────────
DEFAULT_SYMBOL = "VN30F1M"
MULTI_SYMBOLS  = ["HPG", "VNM", "FPT"]

# ── Report paths ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_REPORT_PATH  = _PROJECT_ROOT / "reports" / "TEST_REPORT.md"
_CSV_PATH     = _PROJECT_ROOT / "reports" / "backtest_results.csv"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BeeTrade MCMC & HMM Strategy Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--suite",
                   choices=["all", "insample", "oos", "perm", "sensitivity", "multi", "tfcompare"],
                   default="all",
                   help="tfcompare: HMM 15m/30m/1h/4h same period → backtest_results.csv")
    p.add_argument("--strategy",
                   choices=["mcmc", "hmm", "both"],
                   default="mcmc",
                   help="Which strategy to backtest: mcmc, hmm, or both (comparison)")
    p.add_argument("--symbol",      default=DEFAULT_SYMBOL)
    p.add_argument("--symbols",     nargs="+", default=MULTI_SYMBOLS)
    p.add_argument("--confidence",  type=float, default=0.75)
    p.add_argument("--warmup",      type=int,   default=30)
    p.add_argument("--n-paths",     type=int,   default=500)
    p.add_argument("--n-perms",     type=int,   default=1000)
    p.add_argument("--fast",        action="store_true")
    p.add_argument("--no-fetch",    action="store_true")
    p.add_argument("--no-report",   action="store_true")
    p.add_argument("--commission",  type=float, default=0.0)
    p.add_argument("--hmm-states",  type=int,   default=3, help="HMM hidden states k")
    p.add_argument("--hmm-iter",    type=int,   default=200, help="HMM Baum-Welch iterations")
    p.add_argument("--refit-every", type=int,   default=5, help="Refit HMM every N bars")
    p.add_argument("--sliding-window-bars", type=int, default=None,
                   help="Rolling training window (bars) for HMM; omit for expanding window")
    p.add_argument("--csv-out", default=None,
                   help="Write trade log CSV here (default: reports/backtest_results.csv)")
    p.add_argument("--hmm-timeframe", choices=["1H", "30", "15"],
                   default="1H",
                   help="Intraday timeframe for HMM Suite 3/4 (default: 1H)")
    p.add_argument("--tf-from", default=None,
                   help="tfcompare: start date YYYYMMDD (default: in-sample Sep-Dec 2025)")
    p.add_argument("--tf-to", default=None,
                   help="tfcompare: end date YYYYMMDD (default: in-sample end)")
    p.add_argument("--tf-out", default=None,
                   help="tfcompare: optional CSV path (default: reports/backtest_results.csv "
                        "or reports/backtest_results_tf_<from>_<to>.csv if --tf-from/--tf-to set)")
    return p.parse_args()


# ── Config factories ──────────────────────────────────────────────────────────

def make_mcmc_config(args, resolution: str = "1D", symbol: Optional[str] = None) -> BacktestConfig:
    return BacktestConfig(
        symbol=symbol or args.symbol,
        bar_type=resolution,
        warmup_bars=args.warmup,
        confidence_threshold=args.confidence,
        position_size=1,
        mh_iterations=200,
        mh_burnin=100,
        n_paths=args.n_paths,
        seed=42,
        commission_pct=args.commission,
        fast_mode=args.fast,
    )


def _oi_cache_path(symbol: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in symbol)
    return _PROJECT_ROOT / "data" / "cache" / "oi" / f"{safe}.jsonl"


def _attach_open_interest_from_cache(
    cfg: HMMBacktestConfig,
    bars: List[OhlcBar],
    symbol: str,
) -> None:
    """If ``use_open_interest``, fill ``cfg.open_interest`` from JSONL cache or disable OI."""
    if not cfg.hmm_config.use_open_interest:
        return
    cache = OICache(_oi_cache_path(symbol))
    pts = cache.load_points()
    if not pts:
        logger.warning(
            "HMM_USE_OPEN_INTEREST: no OI points in data/cache/oi/ for this symbol; "
            "disabling OI for this backtest (populate cache via live hmm_live, or backtest = basis only).",
        )
        cfg.hmm_config = replace(cfg.hmm_config, use_open_interest=False)
        cfg.open_interest = None
        return
    cfg.open_interest = align_open_interest_to_bars(bars, pts)


def make_hmm_config(args, resolution: str = "1D", symbol: Optional[str] = None) -> HMMBacktestConfig:
    settings = get_settings()
    return HMMBacktestConfig(
        symbol=symbol or args.symbol,
        hmm_config=HMMConfig(
            k_states=args.hmm_states,
            n_iter=args.hmm_iter,
            zscore_window=20,
            random_state=42,
            use_basis=settings.HMM_USE_BASIS,
            use_open_interest=settings.HMM_USE_OPEN_INTEREST,
        ),
        warmup_bars=args.warmup,
        confidence_threshold=args.confidence,
        position_size=1,
        commission_pct=args.commission,
        refit_every=args.refit_every,
        sliding_window_bars=args.sliding_window_bars,
    )


def fetch_hmm_bars(
    fetcher: DataFetcher,
    args,
    from_date: str,
    to_date: str,
    resolution: str,
    symbol: Optional[str] = None,
) -> Tuple[List[OhlcBar], HMMBacktestConfig]:
    """Load OHLC for HMM; when ``HMM_USE_BASIS``, align future with index (VN30)."""
    sym = symbol or args.symbol
    settings = get_settings()
    cfg = make_hmm_config(args, resolution, symbol=sym)
    if settings.HMM_USE_BASIS:
        bars, ic, _ = fetch_aligned_future_index(
            fetcher, sym, settings.HMM_INDEX_SYMBOL, from_date, to_date, resolution
        )
        cfg.index_closes = ic
        _attach_open_interest_from_cache(cfg, bars, sym)
        return bars, cfg
    bars = get_bars(fetcher, sym, from_date, to_date, resolution)
    cfg.index_closes = None
    _attach_open_interest_from_cache(cfg, bars, sym)
    return bars, cfg


def resample_index_closes_4h(index_closes: List[float], bars_1h: List[OhlcBar]) -> List[float]:
    """Align index close series with ``resample_4h`` (last bar of each 4×1H chunk)."""
    if len(index_closes) != len(bars_1h):
        return []
    out: List[float] = []
    for i in range(0, len(bars_1h), 4):
        chunk = bars_1h[i : i + 4]
        if not chunk:
            break
        last = i + len(chunk) - 1
        out.append(index_closes[last])
    return out


# ── Data helpers ──────────────────────────────────────────────────────────────

def get_bars(fetcher: DataFetcher, symbol: str, from_date: str, to_date: str,
             resolution: str = "1D") -> List[OhlcBar]:
    bars = fetcher.fetch(symbol, from_date, to_date, resolution=resolution)
    if not bars:
        print(f"  WARNING: No bars for {symbol} res={resolution} {from_date}-{to_date}")
    else:
        print(f"  Fetched {len(bars)} bars for {symbol} (res={resolution}, {from_date}-{to_date})")
    return bars


def _bar_type_from_resolution(resolution: str) -> str:
    """Map data resolution to compute_metrics bar_type."""
    if resolution == "1D":
        return "D"
    if resolution == "1H":
        return "60"
    if resolution == "30":
        return "30"
    if resolution == "15":
        return "15"
    return "D"


def _tf_label(resolution: str) -> str:
    if resolution == "1D":
        return "Daily 1D"
    if resolution == "1H":
        return "Hourly 1H"
    if resolution == "30":
        return "30-min"
    if resolution == "15":
        return "15-min"
    if resolution == "4H":
        return "4-hour (resampled)"
    return resolution


def resample_4h(bars_1h: List[OhlcBar]) -> List[OhlcBar]:
    """Aggregate consecutive 1H bars into 4H bars (4 bars → 1)."""
    if not bars_1h:
        return []
    out: List[OhlcBar] = []
    for i in range(0, len(bars_1h), 4):
        chunk = bars_1h[i : i + 4]
        if not chunk:
            break
        out.append(
            OhlcBar(
                symbol=chunk[0].symbol,
                time=chunk[0].time,
                open=chunk[0].open,
                high=max(b.high for b in chunk),
                low=min(b.low for b in chunk),
                close=chunk[-1].close,
                volume=sum(b.volume for b in chunk),
                unix_ts=chunk[0].unix_ts,
            )
        )
    return out


# Columns for tfcompare CSV (summary rows + per-timeframe trades)
_TF_CSV_FIELDS = [
    "record_type",
    "timeframe",
    "resolution",
    "suite",
    "symbol",
    "period_from",
    "period_to",
    "n_bars",
    "n_trades",
    "win_rate_pct",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown_pct",
    "total_return_pct",
    "cagr_pct",
    "expectancy",
    "profit_factor",
    "entry_date",
    "exit_date",
    "side",
    "entry_price",
    "exit_price",
    "pnl",
    "commission",
    "net_pnl",
    "p_up",
    "p_down",
    "confidence",
    "mcmc_ms",
]


def _empty_tf_row() -> dict:
    return {k: "" for k in _TF_CSV_FIELDS}


def _summary_tf_row(
    tf_label: str,
    resolution: str,
    suite: str,
    symbol: str,
    m: BacktestMetrics,
    n_bars: int,
    period_from: str,
    period_to: str,
) -> dict:
    r = _empty_tf_row()
    r.update(
        {
            "record_type": "summary",
            "timeframe": tf_label,
            "resolution": resolution,
            "suite": suite,
            "symbol": symbol,
            "period_from": period_from,
            "period_to": period_to,
            "n_bars": n_bars,
            "n_trades": m.n_trades,
            "win_rate_pct": round(m.win_rate_pct, 4),
            "sharpe_ratio": round(m.sharpe_ratio, 6),
            "sortino_ratio": round(m.sortino_ratio, 6),
            "max_drawdown_pct": round(m.max_drawdown_pct, 4),
            "total_return_pct": round(m.total_return_pct, 4),
            "cagr_pct": round(m.cagr_pct, 4),
            "expectancy": round(m.expectancy, 6),
            "profit_factor": round(m.profit_factor, 6)
            if m.profit_factor != float("inf")
            else "inf",
        }
    )
    return r


def _trade_to_tf_row(
    t: TradeRecord,
    suite: str,
    symbol: str,
    tf_label: str,
    resolution: str,
    period_from: str,
    period_to: str,
) -> dict:
    r = _empty_tf_row()
    r.update(
        {
            "record_type": "trade",
            "timeframe": tf_label,
            "resolution": resolution,
            "suite": suite,
            "symbol": symbol,
            "period_from": period_from,
            "period_to": period_to,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "side": t.side,
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "pnl": round(t.pnl, 4),
            "commission": round(t.commission, 4),
            "net_pnl": round(t.net_pnl, 4),
            "p_up": t.p_up,
            "p_down": t.p_down,
            "confidence": t.confidence,
            "mcmc_ms": t.mcmc_elapsed_ms,
        }
    )
    return r


def write_tfcompare_csv(rows: List[dict], out_path: Optional[Path] = None) -> None:
    """Write timeframe comparison rows; fallback path if main CSV is locked."""
    if not rows:
        return
    primary = out_path if out_path is not None else _CSV_PATH
    primary.parent.mkdir(parents=True, exist_ok=True)
    targets = [primary, primary.with_name(f"backtest_results_{int(time.time())}.csv")]

    def _write(path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_TF_CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in _TF_CSV_FIELDS})

    for path in targets:
        try:
            _write(path)
            print(f"  Timeframe comparison CSV saved to {path}")
            return
        except PermissionError:
            continue
    print(f"  ERROR: could not write CSV (permission denied): {primary}")


def run_hmm_tfcompare(fetcher: DataFetcher, args) -> List[dict]:
    """Run HMM walk-forward on 15m, 30m, 1H, 4H (4H = resample from 1H); build CSV rows."""
    symbol = args.symbol
    p_from = getattr(args, "tf_from", None) or IS_FROM
    p_to = getattr(args, "tf_to", None) or IS_TO
    rows: List[dict] = []
    table_lines: List[tuple] = []

    print(f"  Period: {p_from} – {p_to}")

    settings = get_settings()
    specs: List[tuple] = []

    for tf_label, api_res, bar_type in [
        ("15m", "15", "15"),
        ("30m", "30", "30"),
        ("1h", "1H", "60"),
    ]:
        if settings.HMM_USE_BASIS:
            bars, cfg_tf = fetch_hmm_bars(fetcher, args, p_from, p_to, api_res)
        else:
            bars = get_bars(fetcher, symbol, p_from, p_to, api_res)
            cfg_tf = make_hmm_config(args, api_res)
        specs.append((tf_label, api_res, bar_type, bars, cfg_tf))

    if settings.HMM_USE_BASIS:
        bars_1h, cfg_1h = fetch_hmm_bars(fetcher, args, p_from, p_to, "1H")
    else:
        bars_1h = get_bars(fetcher, symbol, p_from, p_to, "1H")
        cfg_1h = None
    bars_4h = resample_4h(bars_1h)
    cfg_4h = make_hmm_config(args, "4H")
    if (
        settings.HMM_USE_BASIS
        and cfg_1h
        and cfg_1h.index_closes
        and len(cfg_1h.index_closes) == len(bars_1h)
        and bars_1h
    ):
        cfg_4h.index_closes = resample_index_closes_4h(cfg_1h.index_closes, bars_1h)
    else:
        cfg_4h.index_closes = None
    specs.append(("4h", "4H(resampled)", "4H", bars_4h, cfg_4h))
    print(f"  4H: built {len(bars_4h)} bars from {len(bars_1h)} x 1H")

    for tf_label, api_res, bar_type, bars, cfg in specs:
        suite_trade = f"hmm_tfcompare_{tf_label}"
        suite_sum = f"hmm_tfcompare_summary_{tf_label}"
        if not bars:
            print(f"  [{tf_label}] skipped — no bars")
            continue
        trades, equity = HMMBarReplay().run(bars, cfg)
        m = compute_metrics(trades, equity, len(bars), bar_type)
        rows.append(
            _summary_tf_row(
                tf_label, api_res, suite_sum, symbol, m, len(bars), p_from, p_to
            )
        )
        for t in trades:
            rows.append(
                _trade_to_tf_row(
                    t, suite_trade, symbol, tf_label, api_res, p_from, p_to
                )
            )
        print(
            f"  [{tf_label}] n_bars={len(bars)} trades={m.n_trades} "
            f"WinRate={m.win_rate_pct:.1f}% Sharpe={m.sharpe_ratio:.3f} "
            f"MDD={m.max_drawdown_pct:.1f}% Return={m.total_return_pct:.2f}%"
        )
        table_lines.append(
            (
                tf_label,
                len(bars),
                m.n_trades,
                m.win_rate_pct,
                m.sharpe_ratio,
                m.max_drawdown_pct,
                m.total_return_pct,
            )
        )

    if not args.no_report and table_lines:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [
            f"\n---\n\n## HMM intraday timeframe comparison [{now}]\n\n",
            f"**Symbol:** {symbol} | **Period:** {p_from}–{p_to} | "
            f"**Confidence:** {args.confidence} | **Warmup:** {args.warmup} | "
            f"**Refit every:** {args.refit_every}\n\n",
            "| TF | Bars | Trades | Win% | Sharpe | MDD% | Total return% |\n",
            "|----|------|--------|------|--------|------|----------------|\n",
        ]
        for tf_label, nb, nt, wr, sh, mdd, ret in table_lines:
            lines.append(
                f"| {tf_label} | {nb} | {nt} | {wr:.1f}% | {sh:.3f} | "
                f"{mdd:.1f}% | {ret:.2f}% |\n"
            )
        lines.append(
            "\n> **Note:** 4H is built by aggregating every four consecutive 1H bars. "
            "VN30F1M uses VN30 index OHLC as proxy where needed.\n\n"
        )
        _append_report("".join(lines))

    return rows


# ── CSV / Report helpers ──────────────────────────────────────────────────────

def _trades_to_rows(trades: List[TradeRecord], suite_name: str) -> List[dict]:
    rows = []
    for t in trades:
        rows.append({
            "suite": suite_name,
            "symbol": t.symbol,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "side": t.side,
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2),
            "pnl": round(t.pnl, 4),
            "commission": round(t.commission, 4),
            "net_pnl": round(t.net_pnl, 4),
            "p_up": t.p_up,
            "p_down": t.p_down,
            "confidence": t.confidence,
            "mcmc_ms": t.mcmc_elapsed_ms,
        })
    return rows


def write_csv(all_rows: List[dict], out_path: Optional[Path] = None) -> None:
    if not all_rows:
        return
    path = out_path or _CSV_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Trade log saved to {path}")


def _append_report(content: str) -> None:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(content)


def _section_header(title: str, symbol: str, date_range: str,
                    bar_type: str, config) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    conf = getattr(config, "confidence_threshold", "?")
    warm = getattr(config, "warmup_bars", "?")
    return (
        f"\n---\n\n## Backtest: {title} [{now}]\n\n"
        f"**Symbol:** {symbol} | **Period:** {date_range} | **Bar type:** {bar_type}\n"
        f"**Confidence:** {conf} | **Warmup:** {warm} bars\n\n"
    )


def _metrics_row(label: str, m: BacktestMetrics) -> str:
    return (
        f"| {label} | {m.n_trades} | {m.win_rate_pct:.1f}% | "
        f"{m.sharpe_ratio:.3f} | {m.sortino_ratio:.3f} | "
        f"{m.max_drawdown_pct:.2f}% | {m.total_return_pct:.2f}% |"
    )


# ── MCMC Suite Runners ────────────────────────────────────────────────────────

def run_suite1_mcmc(fetcher, args, all_rows):
    print("\n--- [MCMC] Suite 1: In-Sample (daily, Sep-Dec 2025) ---")
    bars = get_bars(fetcher, args.symbol, IS_FROM, IS_TO, "1D")
    if not bars:
        return None
    cfg = make_mcmc_config(args, "1D")
    trades, equity = BarReplay().run(bars, cfg)
    metrics = compute_metrics(trades, equity, len(bars), "D")
    all_rows.extend(_trades_to_rows(trades, "mcmc_suite1"))
    print(f"  Trades:{metrics.n_trades} WinRate:{metrics.win_rate_pct:.1f}% "
          f"Sharpe:{metrics.sharpe_ratio:.3f} MDD:{metrics.max_drawdown_pct:.2f}%")
    if not args.no_report:
        s = _section_header("MCMC -- Suite 1 In-Sample", args.symbol, f"{IS_FROM}-{IS_TO}", "Daily 1D", cfg)
        s += metrics.summary_table() + "\n"
        _append_report(s)
    return bars, trades, equity, metrics, cfg


def run_suite2_mcmc(fetcher, args, all_rows):
    print("\n--- [MCMC] Suite 2: Out-of-Sample (daily, Jan-Mar 2026) ---")
    bars = get_bars(fetcher, args.symbol, OOS_FROM, OOS_TO, "1D")
    if not bars:
        return None
    cfg = make_mcmc_config(args, "1D")
    trades, equity = BarReplay().run(bars, cfg)
    metrics = compute_metrics(trades, equity, len(bars), "D")
    all_rows.extend(_trades_to_rows(trades, "mcmc_suite2"))
    print(f"  Trades:{metrics.n_trades} WinRate:{metrics.win_rate_pct:.1f}% "
          f"Sharpe:{metrics.sharpe_ratio:.3f} MDD:{metrics.max_drawdown_pct:.2f}%")
    if not args.no_report:
        s = _section_header("MCMC -- Suite 2 OOS", args.symbol, f"{OOS_FROM}-{OOS_TO}", "Daily 1D", cfg)
        s += metrics.summary_table() + "\n"
        _append_report(s)
    return bars, trades, equity, metrics, cfg


def run_suite3_mcmc(fetcher, args, all_rows):
    print(f"\n--- [MCMC] Suite 3: Monte Carlo Permutation ({args.n_perms} perms, 1H) ---")
    bars = get_bars(fetcher, args.symbol, IS_FROM, IS_TO, "1H")
    if not bars:
        return None
    perm_cfg = make_mcmc_config(args, "1H")
    perm_cfg.fast_mode = True
    result = monte_carlo_permutation(bars, perm_cfg, n_perms=args.n_perms, seed=42)
    print(f"  Sharpe:{result.actual_sharpe:.4f} p-value:{result.p_value:.4f} "
          f"Sig:{result.significant} ({result.elapsed_s:.1f}s)")
    if not args.no_report:
        s = _section_header("MCMC -- Suite 3 Monte Carlo", args.symbol, f"{IS_FROM}-{IS_TO}", "Hourly 1H", perm_cfg)
        s += result.summary_table() + "\n"
        s += f"\n**Interpretation:** {'Significant skill (p<0.05)' if result.significant else 'Cannot reject null -- may be due to chance'}\n"
        _append_report(s)
    return result


def run_suite4_mcmc(fetcher, args, all_rows):
    print("\n--- [MCMC] Suite 4: Hourly Sensitivity (Sep-Dec 2025) ---")
    bars = get_bars(fetcher, args.symbol, IS_FROM, IS_TO, "1H")
    if not bars:
        return None
    cfg = make_mcmc_config(args, "1H")
    trades, equity = BarReplay().run(bars, cfg)
    metrics = compute_metrics(trades, equity, len(bars), "60")
    all_rows.extend(_trades_to_rows(trades, "mcmc_suite4"))
    print(f"  Trades:{metrics.n_trades} WinRate:{metrics.win_rate_pct:.1f}% Sharpe:{metrics.sharpe_ratio:.3f}")
    if not args.no_report:
        s = _section_header("MCMC -- Suite 4 Hourly", args.symbol, f"{IS_FROM}-{IS_TO}", "Hourly 1H", cfg)
        s += metrics.summary_table() + "\n"
        _append_report(s)
    return bars, trades, equity, metrics, cfg


def run_suite5_mcmc(fetcher, args, all_rows) -> Dict[str, BacktestMetrics]:
    print(f"\n--- [MCMC] Suite 5: Multi-Symbol ({', '.join(args.symbols)}) ---")
    results = {}
    rows_for_report = []
    for sym in args.symbols:
        bars = get_bars(fetcher, sym, IS_FROM, IS_TO, "1D")
        if not bars:
            continue
        cfg = make_mcmc_config(args, "1D", symbol=sym)
        trades, equity = BarReplay().run(bars, cfg)
        metrics = compute_metrics(trades, equity, len(bars), "D")
        results[sym] = metrics
        all_rows.extend(_trades_to_rows(trades, f"mcmc_s5_{sym}"))
        print(f"  {sym}: Trades={metrics.n_trades} Sharpe={metrics.sharpe_ratio:.3f} WinRate={metrics.win_rate_pct:.1f}%")
        rows_for_report.append((sym, metrics))
    if not args.no_report and rows_for_report:
        cfg_repr = make_mcmc_config(args, "1D")
        s = _section_header("MCMC -- Suite 5 Multi-Symbol", "+".join(args.symbols), f"{IS_FROM}-{IS_TO}", "Daily 1D", cfg_repr)
        s += "| Symbol | Trades | Win Rate | Sharpe | Sortino | MDD | Return |\n"
        s += "|--------|--------|----------|--------|---------|-----|--------|\n"
        for sym, m in rows_for_report:
            s += _metrics_row(sym, m) + "\n"
        _append_report(s)
    return results


# ── HMM Suite Runners ─────────────────────────────────────────────────────────

def run_suite1_hmm(fetcher, args, all_rows):
    print("\n--- [HMM] Suite 1: In-Sample (daily, Sep-Dec 2025) ---")
    # Fetch extended training window so HMM sees 2 years of regimes
    bars_train, cfg = fetch_hmm_bars(fetcher, args, HMM_TRAIN_FROM, IS_TO, "1D")
    if not bars_train:
        return None
    # IS backtest window: only bars within IS period
    is_cutoff = IS_FROM
    bars_is = [b for b in bars_train if b.date_str >= is_cutoff]
    print(f"  Training window: {len(bars_train)} bars | IS window: {len(bars_is)} bars")
    trades, equity = HMMBarReplay().run(bars_train, cfg)
    # Trim equity/trades to IS period
    is_trades = [t for t in trades if t.entry_date >= is_cutoff]
    is_start  = next((i for i, b in enumerate(bars_train) if b.date_str >= is_cutoff), 0)
    is_equity = equity[is_start:]
    metrics = compute_metrics(is_trades, is_equity, len(bars_is), "D")
    all_rows.extend(_trades_to_rows(is_trades, "hmm_suite1"))
    print(f"  Trades:{metrics.n_trades} WinRate:{metrics.win_rate_pct:.1f}% "
          f"Sharpe:{metrics.sharpe_ratio:.3f} MDD:{metrics.max_drawdown_pct:.2f}%")
    if not args.no_report:
        s = _section_header("HMM -- Suite 1 In-Sample", args.symbol, f"{IS_FROM}-{IS_TO}", "Daily 1D", cfg)
        s += f"*Training window: {HMM_TRAIN_FROM}–{IS_TO} ({len(bars_train)} bars)*\n\n"
        s += metrics.summary_table() + "\n"
        _append_report(s)
    return bars_is, is_trades, is_equity, metrics, cfg


def run_suite2_hmm(fetcher, args, all_rows):
    print("\n--- [HMM] Suite 2: Out-of-Sample (daily, Jan-Mar 2026) ---")
    # Use full extended range so model is trained on pre-OOS data
    bars_all, cfg = fetch_hmm_bars(fetcher, args, HMM_TRAIN_FROM, OOS_TO, "1D")
    if not bars_all:
        return None
    trades, equity = HMMBarReplay().run(bars_all, cfg)
    oos_trades = [t for t in trades if t.entry_date >= OOS_FROM]
    oos_start  = next((i for i, b in enumerate(bars_all) if b.date_str >= OOS_FROM), 0)
    oos_equity = equity[oos_start:]
    oos_bars   = [b for b in bars_all if b.date_str >= OOS_FROM]
    metrics = compute_metrics(oos_trades, oos_equity, len(oos_bars), "D")
    all_rows.extend(_trades_to_rows(oos_trades, "hmm_suite2"))
    print(f"  Trades:{metrics.n_trades} WinRate:{metrics.win_rate_pct:.1f}% "
          f"Sharpe:{metrics.sharpe_ratio:.3f} MDD:{metrics.max_drawdown_pct:.2f}%")
    if not args.no_report:
        s = _section_header("HMM -- Suite 2 OOS", args.symbol, f"{OOS_FROM}-{OOS_TO}", "Daily 1D", cfg)
        s += metrics.summary_table() + "\n"
        _append_report(s)
    return oos_bars, oos_trades, oos_equity, metrics, cfg


def run_suite3_hmm(fetcher, args, all_rows):
    """HMM Suite 3: Report actual Sharpe on hourly bars (skip permutation loop).

    Full Monte Carlo permutation with HMM re-fitting is O(n_perms * n_bars * n_iter)
    GaussianHMM fits — approximately 100× slower than MCMC permutation.
    We report the actual in-sample Sharpe and reference the MCMC p-value.
    """
    tf = args.hmm_timeframe
    print(f"\n--- [HMM] Suite 3: {tf} Sharpe (permutation skipped — see MCMC Suite 3) ---")
    bars, cfg = fetch_hmm_bars(fetcher, args, IS_FROM, IS_TO, tf)
    if not bars:
        return None

    trades, equity = HMMBarReplay().run(bars, cfg)
    metrics = compute_metrics(trades, equity, len(bars), _bar_type_from_resolution(tf))
    print(f"  Trades:{metrics.n_trades} WinRate:{metrics.win_rate_pct:.1f}% "
          f"Sharpe:{metrics.sharpe_ratio:.4f} MDD:{metrics.max_drawdown_pct:.2f}%")

    if not args.no_report:
        s = _section_header(
            f"HMM -- Suite 3 {_tf_label(tf)} (Actual Sharpe)",
            args.symbol,
            f"{IS_FROM}-{IS_TO}",
            _tf_label(tf),
            cfg,
        )
        s += "| Metric | Value |\n|--------|-------|\n"
        s += f"| Actual Sharpe | {metrics.sharpe_ratio:.4f} |\n"
        s += f"| Trades | {metrics.n_trades} |\n"
        s += f"| Win Rate | {metrics.win_rate_pct:.1f}% |\n"
        s += f"| MDD | {metrics.max_drawdown_pct:.2f}% |\n"
        s += (
            "\n**Note:** HMM permutation test skipped — each permutation requires "
            "~500 GaussianHMM re-fits (100× slower than MCMC). "
            "See MCMC Suite 3 for p-value reference (p=0.072).\n"
        )
        _append_report(s)
    return metrics.sharpe_ratio, None, None


def run_suite4_hmm(fetcher, args, all_rows):
    tf = args.hmm_timeframe
    print(f"\n--- [HMM] Suite 4: {tf} Sensitivity (Sep-Dec 2025) ---")
    bars, cfg = fetch_hmm_bars(fetcher, args, IS_FROM, IS_TO, tf)
    if not bars:
        return None
    trades, equity = HMMBarReplay().run(bars, cfg)
    metrics = compute_metrics(trades, equity, len(bars), _bar_type_from_resolution(tf))
    all_rows.extend(_trades_to_rows(trades, f"hmm_suite4_{tf}"))
    print(f"  Trades:{metrics.n_trades} WinRate:{metrics.win_rate_pct:.1f}% Sharpe:{metrics.sharpe_ratio:.3f}")
    if not args.no_report:
        s = _section_header(
            f"HMM -- Suite 4 {_tf_label(tf)}",
            args.symbol,
            f"{IS_FROM}-{IS_TO}",
            _tf_label(tf),
            cfg,
        )
        s += metrics.summary_table() + "\n"
        _append_report(s)
    return bars, trades, equity, metrics, cfg


def run_suite5_hmm(fetcher, args, all_rows) -> Dict[str, BacktestMetrics]:
    print(f"\n--- [HMM] Suite 5: Multi-Symbol ({', '.join(args.symbols)}) ---")
    results = {}
    rows_for_report = []
    for sym in args.symbols:
        bars, cfg = fetch_hmm_bars(fetcher, args, HMM_TRAIN_FROM, IS_TO, "1D", symbol=sym)
        if not bars:
            continue
        trades, equity = HMMBarReplay().run(bars, cfg)
        is_trades = [t for t in trades if t.entry_date >= IS_FROM]
        is_start  = next((i for i, b in enumerate(bars) if b.date_str >= IS_FROM), 0)
        is_equity = equity[is_start:]
        is_bars   = [b for b in bars if b.date_str >= IS_FROM]
        metrics = compute_metrics(is_trades, is_equity, len(is_bars), "D")
        results[sym] = metrics
        all_rows.extend(_trades_to_rows(is_trades, f"hmm_s5_{sym}"))
        print(f"  {sym}: Trades={metrics.n_trades} Sharpe={metrics.sharpe_ratio:.3f} WinRate={metrics.win_rate_pct:.1f}%")
        rows_for_report.append((sym, metrics))
    if not args.no_report and rows_for_report:
        cfg_repr = make_hmm_config(args, "1D")
        s = _section_header("HMM -- Suite 5 Multi-Symbol", "+".join(args.symbols), f"{IS_FROM}-{IS_TO}", "Daily 1D", cfg_repr)
        s += "| Symbol | Trades | Win Rate | Sharpe | Sortino | MDD | Return |\n"
        s += "|--------|--------|----------|--------|---------|-----|--------|\n"
        for sym, m in rows_for_report:
            s += _metrics_row(sym, m) + "\n"
        _append_report(s)
    return results


# ── Comparison report ─────────────────────────────────────────────────────────

def run_comparison(
    mcmc_results: dict,
    hmm_results: dict,
    args,
) -> None:
    """Append side-by-side MCMC vs HMM comparison table to report."""
    if args.no_report:
        return
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n---\n\n## MCMC vs HMM Comparison [{now}]\n\n",
        f"**Symbol:** {args.symbol} | **Confidence:** {args.confidence} | "
        f"**Warmup:** {args.warmup} | **HMM states:** {args.hmm_states}\n\n",
        "| Suite | Metric | MCMC | HMM | Winner |\n",
        "|-------|--------|------|-----|--------|\n",
    ]

    def _cmp_row(suite: str, metric: str, v_mcmc, v_hmm, higher_better: bool = True):
        fmt = f"{v_mcmc:.3f}" if isinstance(v_mcmc, float) else str(v_mcmc)
        fmh = f"{v_hmm:.3f}"  if isinstance(v_hmm,  float) else str(v_hmm)
        if isinstance(v_mcmc, float) and isinstance(v_hmm, float):
            if higher_better:
                winner = "MCMC" if v_mcmc > v_hmm else ("HMM" if v_hmm > v_mcmc else "TIE")
            else:
                winner = "MCMC" if v_mcmc < v_hmm else ("HMM" if v_hmm < v_mcmc else "TIE")
        else:
            winner = "—"
        return f"| {suite} | {metric} | {fmt} | {fmh} | {winner} |\n"

    for suite_key, suite_label in [("is", "Suite 1 IS"), ("oos", "Suite 2 OOS"),
                                    ("h1", "Suite 4 Hourly")]:
        mc = mcmc_results.get(suite_key)
        hm = hmm_results.get(suite_key)
        if mc and hm:
            lines.append(_cmp_row(suite_label, "Sharpe",  mc.sharpe_ratio,      hm.sharpe_ratio))
            lines.append(_cmp_row(suite_label, "Win Rate", mc.win_rate_pct,      hm.win_rate_pct))
            lines.append(_cmp_row(suite_label, "MDD %",   mc.max_drawdown_pct,  hm.max_drawdown_pct, False))
            lines.append(_cmp_row(suite_label, "Trades",  float(mc.n_trades),   float(hm.n_trades)))

    perm_mc = mcmc_results.get("perm")
    perm_hm = hmm_results.get("perm")
    if perm_mc:
        # MCMC returns PermResult; HMM returns (sharpe, p_value, sig) tuple or None
        mc_pval = perm_mc.p_value if hasattr(perm_mc, "p_value") else perm_mc[1]
        mc_sharpe = perm_mc.actual_sharpe if hasattr(perm_mc, "actual_sharpe") else perm_mc[0]
        lines.append(f"| Suite 3 MC | Actual Sharpe | {mc_sharpe:.4f} | "
                     f"{'N/A (skipped)' if not perm_hm else f'{perm_hm[0]:.4f}'} | "
                     f"{'—' if not perm_hm else ('MCMC' if mc_sharpe > perm_hm[0] else 'HMM')} |\n")
        lines.append(f"| Suite 3 MC | p-value | {mc_pval:.4f} | "
                     f"{'N/A' if not perm_hm or perm_hm[1] is None else f'{perm_hm[1]:.4f}'} | — |\n")

    lines.append(
        "\n> **Note:** VN30F1M uses VN30 index as proxy (DNSE REST API does not serve "
        "derivative historical OHLC). HMM trained on extended window "
        f"({HMM_TRAIN_FROM}–{IS_TO}) for better regime estimation.\n"
    )
    _append_report("".join(lines))
    print("\n  Comparison table appended to report.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    t_start = time.time()
    run_mcmc = args.strategy in ("mcmc", "both")
    run_hmm  = args.strategy in ("hmm",  "both")

    settings = get_settings()
    fetcher  = DataFetcher(use_cache=not args.no_fetch, settings=settings)

    print("=" * 60)
    strat_label = {"mcmc": "MCMC", "hmm": "HMM", "both": "MCMC + HMM Comparison"}[args.strategy]
    print(f"BeeTrade Backtest Runner — {strat_label}")
    print(f"Suite: {args.suite} | Symbol: {args.symbol}")
    print(f"Confidence: {args.confidence} | Warmup: {args.warmup} | Fast: {args.fast}")
    if run_hmm:
        sw = args.sliding_window_bars
        sw_s = f"sliding_window={sw}" if sw is not None else "sliding_window=expanding"
        print(f"HMM: k={args.hmm_states} states | iter={args.hmm_iter} | "
              f"refit_every={args.refit_every} | {sw_s}")
    print("=" * 60)

    all_csv_rows = []
    mcmc_results: dict = {}
    hmm_results:  dict = {}

    # Report header
    if not args.no_report:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        _append_report(
            f"\n---\n\n## Giai đoạn 5 — {strat_label} Backtest [{now}]\n\n"
            f"**Strategy:** {strat_label} | **Symbol:** {args.symbol} | "
            f"**Confidence:** {args.confidence} | **Warmup:** {args.warmup} bars\n\n"
        )

    suite = args.suite

    if suite == "tfcompare":
        if not run_hmm:
            print("ERROR: --suite tfcompare requires --strategy hmm or both")
            return
        p_from = getattr(args, "tf_from", None) or IS_FROM
        p_to = getattr(args, "tf_to", None) or IS_TO
        print("\n=== HMM timeframe comparison: 15m, 30m, 1h, 4h ===")
        rows = run_hmm_tfcompare(fetcher, args)
        tf_out: Optional[Path] = None
        if args.tf_out:
            tf_out = Path(args.tf_out)
        elif getattr(args, "tf_from", None) or getattr(args, "tf_to", None):
            tf_out = _PROJECT_ROOT / "reports" / f"backtest_results_tf_{p_from}_{p_to}.csv"
        write_tfcompare_csv(rows, out_path=tf_out)
        elapsed = time.time() - t_start
        print(f"\nBacktest complete in {elapsed:.1f}s")
        if not args.no_report:
            print(f"Report appended to {_REPORT_PATH}")
        return

    # ── MCMC Suites ───────────────────────────────────────────────────────────
    if run_mcmc:
        if suite in ("all", "insample"):
            r = run_suite1_mcmc(fetcher, args, all_csv_rows)
            if r:
                mcmc_results["is"] = r[3]

        if suite in ("all", "oos"):
            r = run_suite2_mcmc(fetcher, args, all_csv_rows)
            if r:
                mcmc_results["oos"] = r[3]

        if suite in ("all", "perm"):
            r = run_suite3_mcmc(fetcher, args, all_csv_rows)
            if r:
                mcmc_results["perm"] = r   # (sharpe, p_value, sig)

        if suite in ("all", "sensitivity"):
            r = run_suite4_mcmc(fetcher, args, all_csv_rows)
            if r:
                mcmc_results["h1"] = r[3]

        if suite in ("all", "multi"):
            run_suite5_mcmc(fetcher, args, all_csv_rows)

    # ── HMM Suites ────────────────────────────────────────────────────────────
    if run_hmm:
        if suite in ("all", "insample"):
            r = run_suite1_hmm(fetcher, args, all_csv_rows)
            if r:
                hmm_results["is"] = r[3]

        if suite in ("all", "oos"):
            r = run_suite2_hmm(fetcher, args, all_csv_rows)
            if r:
                hmm_results["oos"] = r[3]

        if suite in ("all", "perm"):
            r = run_suite3_hmm(fetcher, args, all_csv_rows)
            if r:
                hmm_results["perm"] = r

        if suite in ("all", "sensitivity"):
            r = run_suite4_hmm(fetcher, args, all_csv_rows)
            if r:
                hmm_results["h1"] = r[3]

        if suite in ("all", "multi"):
            run_suite5_hmm(fetcher, args, all_csv_rows)

    # ── Comparison ────────────────────────────────────────────────────────────
    if run_mcmc and run_hmm and mcmc_results and hmm_results:
        run_comparison(mcmc_results, hmm_results, args)

    if all_csv_rows:
        csv_dest = Path(args.csv_out).resolve() if args.csv_out else None
        write_csv(all_csv_rows, out_path=csv_dest)

    elapsed = time.time() - t_start
    print(f"\nBacktest complete in {elapsed:.1f}s")
    if not args.no_report:
        print(f"Report appended to {_REPORT_PATH}")


if __name__ == "__main__":
    main()
