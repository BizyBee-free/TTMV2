#!/usr/bin/env python3
"""HMM configuration grid search from YAML (objective + constraints + significance).

Usage::
    python scripts/hmm_optimize_loop.py
    python scripts/hmm_optimize_loop.py --config config/config.yaml --no-fetch

Reads ``config/config.yaml`` (or ``--config`` path): grid over
(window, k_states, timeframe), optional **basis** (future vs index) and **OI**
(from ``data/cache/oi/{symbol}.jsonl``), writes CSV + Markdown under ``reports/``.

Examples::

    python scripts/hmm_optimize_loop.py
    python scripts/hmm_optimize_loop.py --config config/hmm_optimization.yaml
    python scripts/hmm_optimize_loop.py --only-timeframe 15m 1h
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

from src.backtest.data_fetcher import DataFetcher
from src.backtest.equity_significance import equity_increment_permutation_pvalue
from src.backtest.hmm_replay import HMMBacktestConfig, HMMBarReplay
from src.backtest.metrics import _bars_per_year_from_bar_type, compute_metrics
from src.config import get_settings
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm.feature_engineer import HMMConfig
from src.hmm.oi_data import OICache, align_open_interest_to_bars
from src.logger import get_logger

logger = get_logger("hmm_optimize")

_MIN_FIT_BARS = 20  # matches hmm_replay.HMMBarReplay warmup logic

TF_TO_API: Dict[str, str] = {"5m": "5", "15m": "15", "1h": "1H"}
TF_TO_METRIC_BAR: Dict[str, str] = {"5m": "5", "15m": "15", "1h": "60"}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fetch_bars_timeout(
    fetcher: DataFetcher,
    symbol: str,
    d0: str,
    d1: str,
    resolution: str,
    timeout_s: float = 45.0,
) -> list:
    """Avoid hanging indefinitely on unsupported resolutions (e.g. 5m)."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fetcher.fetch, symbol, d0, d1, resolution)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout:
            logger.warning(
                "fetch timeout",
                extra={"symbol": symbol, "resolution": resolution, "timeout_s": timeout_s},
            )
            return []


def _fetch_aligned_timeout(
    fetcher: DataFetcher,
    fut_symbol: str,
    idx_symbol: str,
    d0: str,
    d1: str,
    resolution: str,
    timeout_s: float = 45.0,
) -> Tuple[list, List[float]]:
    """Aligned future + index bars (inner join on unix_ts)."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(
            fetch_aligned_future_index, fetcher, fut_symbol, idx_symbol, d0, d1, resolution
        )
        try:
            bars, idx, _ = fut.result(timeout=timeout_s)
            return bars, idx
        except FuturesTimeout:
            logger.warning(
                "fetch_aligned timeout",
                extra={
                    "fut_symbol": fut_symbol,
                    "idx_symbol": idx_symbol,
                    "resolution": resolution,
                    "timeout_s": timeout_s,
                },
            )
            return [], []


def _resolve_oi_cache_path(doc: dict, symbol: str) -> Path:
    feats = doc.get("features") or {}
    raw = feats.get("oi_cache_path")
    if raw:
        p = Path(str(raw))
        return p if p.is_absolute() else (_ROOT / p).resolve()
    safe_sym = "".join(c if c.isalnum() or c in "._-" else "_" for c in symbol)
    return _ROOT / "data" / "cache" / "oi" / f"{safe_sym}.jsonl"


def _trim_tail(
    bars: list,
    index_closes: Optional[List[float]],
    open_interest: Optional[List[float]],
    max_bars: Optional[int],
) -> Tuple[list, Optional[List[float]], Optional[List[float]]]:
    if not max_bars or not bars or len(bars) <= int(max_bars):
        return bars, index_closes, open_interest
    n = int(max_bars)
    b = bars[-n:]
    ic = index_closes[-n:] if index_closes is not None else None
    oi = open_interest[-n:] if open_interest is not None else None
    return b, ic, oi


def _clip(x: float, cap: float) -> float:
    if not math.isfinite(x):
        return cap
    if x > cap:
        return cap
    if x < -cap:
        return -cap
    return x


def _score(
    calmar: float,
    sharpe: float,
    pf: float,
    obj: dict,
) -> float:
    cc = float(obj.get("calmar_cap", 10.0))
    sc = float(obj.get("sharpe_cap", 10.0))
    pc = float(obj.get("profit_factor_cap", 10.0))
    cw = float(obj.get("calmar_weight", 0.4))
    sw = float(obj.get("sharpe_weight", 0.3))
    pw = float(obj.get("profit_factor_weight", 0.3))

    c = _clip(calmar, cc)
    s = _clip(sharpe, sc)
    pfv = min(pf, pc) if math.isfinite(pf) else pc
    return cw * c + sw * s + pw * pfv


def _passes_pf(m, gt: float) -> bool:
    pf = m.profit_factor
    if not math.isfinite(pf):
        return m.n_losses == 0 and m.n_wins > 0
    return pf > gt


def evaluate_config(
    bars,
    window: int,
    k_states: int,
    tf_key: str,
    cfg_doc: dict,
    index_closes: Optional[List[float]] = None,
    open_interest: Optional[List[float]] = None,
) -> Dict[str, Any]:
    refit = int(cfg_doc.get("optimization_refit_every", cfg_doc["refit_every"]))
    hm = cfg_doc["hmm"]
    bt = cfg_doc["backtest"]
    cons = cfg_doc["constraints"]
    sig = cfg_doc["significance"]
    obj = cfg_doc["objective"]
    feats = cfg_doc.get("features") or {}
    use_basis = bool(feats.get("use_basis", False))
    use_oi = bool(feats.get("use_open_interest", False))

    bar_type = TF_TO_METRIC_BAR[tf_key]
    bars_per_year = _bars_per_year_from_bar_type(bar_type)

    symbol = str(cfg_doc["symbol"])
    hmm_cfg = HMMConfig(
        k_states=k_states,
        n_iter=int(hm["n_iter"]),
        zscore_window=int(hm["zscore_window"]),
        random_state=int(hm.get("random_state", 42)),
        use_basis=use_basis,
        use_open_interest=use_oi,
    )
    hcfg = HMMBacktestConfig(
        symbol=symbol,
        hmm_config=hmm_cfg,
        warmup_bars=int(bt["warmup_bars"]),
        confidence_threshold=float(bt["confidence"]),
        position_size=int(bt.get("position_size", 1)),
        commission_pct=float(bt.get("commission_pct", 0.0)),
        refit_every=refit,
        sliding_window_bars=window,
        index_closes=index_closes if use_basis else None,
        open_interest=open_interest if use_oi else None,
    )

    t0 = time.perf_counter()
    trades, equity = HMMBarReplay().run(bars, hcfg)
    elapsed = time.perf_counter() - t0

    n_bars = len(bars)
    m = compute_metrics(trades, equity, n_bars=n_bars, bar_type=bar_type)

    row: Dict[str, Any] = {
        "grid_refit_every": refit,
        "use_basis": use_basis,
        "use_open_interest": use_oi,
        "window_size": window,
        "k_states": k_states,
        "timeframe": tf_key,
        "n_bars": n_bars,
        "n_trades": m.n_trades,
        "sharpe": round(m.sharpe_ratio, 6),
        "calmar": round(m.calmar_ratio, 6) if math.isfinite(m.calmar_ratio) else None,
        "profit_factor": round(m.profit_factor, 6) if math.isfinite(m.profit_factor) else None,
        "max_drawdown_pct": round(m.max_drawdown_pct, 4),
        "total_return_pct": round(m.total_return_pct, 4),
        "run_seconds": round(elapsed, 3),
        "valid": False,
        "fail_reason": "",
        "objective_score": None,
        "p_value": None,
        "increment_sharpe": None,
    }

    fail: List[str] = []
    if use_basis and (index_closes is None or len(index_closes) != n_bars):
        fail.append("basis_align_mismatch")
    if use_oi and (open_interest is None or len(open_interest) != n_bars):
        fail.append("oi_length_mismatch")

    if n_bars < max(int(bt["warmup_bars"]), _MIN_FIT_BARS) + 5:
        fail.append("insufficient_bars")

    mdd_lt = float(cons["max_drawdown_pct_lt"])
    if m.max_drawdown_pct >= mdd_lt:
        fail.append(f"mdd>={mdd_lt}")

    if m.n_trades <= int(cons["min_trades"]):
        fail.append(f"trades<={cons['min_trades']}")

    if not _passes_pf(m, float(cons["profit_factor_gt"])):
        fail.append(f"pf<={cons['profit_factor_gt']}")

    p_val: Optional[float] = None
    inc_sh: Optional[float] = None
    if not fail:
        p_val, inc_sh = equity_increment_permutation_pvalue(
            equity,
            n_bars=n_bars,
            bars_per_year=bars_per_year,
            n_perms=int(sig["n_permutations"]),
            seed=int(sig.get("seed", 42)),
        )
        row["p_value"] = round(p_val, 6)
        row["increment_sharpe"] = round(inc_sh, 6)

        if p_val >= float(cons["p_value_lt"]):
            fail.append(f"p>={cons['p_value_lt']}")

        if cons.get("require_positive_increment_sharpe", True) and inc_sh <= 0:
            fail.append("increment_sharpe<=0")

    row["fail_reason"] = ";".join(fail)
    row["valid"] = len(fail) == 0

    calmar_u = m.calmar_ratio if math.isfinite(m.calmar_ratio) else 0.0
    sharpe_u = m.sharpe_ratio if math.isfinite(m.sharpe_ratio) else 0.0
    pf_u = m.profit_factor if math.isfinite(m.profit_factor) else float(obj.get("profit_factor_cap", 10.0))

    sc = _score(calmar_u, sharpe_u, pf_u, obj)
    row["objective_score"] = round(sc, 6) if row["valid"] else None
    row["objective_score_unconstrained"] = round(sc, 6)

    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="HMM optimization loop from YAML")
    ap.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config (default: config/config.yaml)",
    )
    ap.add_argument("--no-fetch", action="store_true", help="Use only local OHLC cache")
    ap.add_argument(
        "--only-timeframe",
        nargs="*",
        metavar="TF",
        help="Run only these timeframes (e.g. 15m 1h). Default: all from YAML.",
    )
    args = ap.parse_args()

    cfg_path = (_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    doc = _load_yaml(cfg_path)

    symbol = doc["symbol"]
    d0, d1 = doc["from_date"], doc["to_date"]
    windows: List[int] = doc["search_space"]["window_size"]
    ks: List[int] = doc["search_space"]["k_states"]
    tfs: List[str] = doc["search_space"]["timeframe"]
    if args.only_timeframe:
        allowed = set(args.only_timeframe)
        tfs = [t for t in tfs if t in allowed]
        if not tfs:
            print("ERROR: --only-timeframe produced empty list (check labels: 5m 15m 1h)")
            return 1

    settings = get_settings()
    # Same convention as scripts/backtest.py: --no-fetch disables reading cache (forces API).
    fetcher = DataFetcher(use_cache=not args.no_fetch, settings=settings)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    print("=" * 60)
    print("HMM Optimization Loop")
    print(f"Symbol={symbol}  Range={d0}-{d1}")
    print(f"Grid: {len(windows)} x {len(ks)} x {len(tfs)} = {len(windows)*len(ks)*len(tfs)} runs")
    print("=" * 60)

    fetch_timeout = float(doc.get("fetch_timeout_seconds", 45.0))
    max_bars = doc.get("max_bars")

    feats = doc.get("features") or {}
    use_basis = bool(feats.get("use_basis", False))
    use_oi = bool(feats.get("use_open_interest", False))
    idx_symbol = str(feats.get("index_symbol", "VN30"))
    oi_cache_path = _resolve_oi_cache_path(doc, symbol)
    print(
        f"Features: OHLCV + basis={use_basis} (index={idx_symbol if use_basis else '—'}) "
        f"+ OI={use_oi} ({oi_cache_path.name if use_oi else '—'})"
    )

    bars_by_tf: Dict[str, Tuple[list, Optional[List[float]], Optional[List[float]]]] = {}
    oi_warned = False
    for tf in tfs:
        api_res = TF_TO_API[tf]
        if use_basis:
            bars, idx = _fetch_aligned_timeout(
                fetcher, symbol, idx_symbol, d0, d1, api_res, timeout_s=fetch_timeout
            )
            index_closes: Optional[List[float]] = idx if idx else None
        else:
            bars = _fetch_bars_timeout(fetcher, symbol, d0, d1, api_res, timeout_s=fetch_timeout)
            index_closes = None

        bars, index_closes, _ = _trim_tail(bars, index_closes, None, max_bars)

        open_interest: Optional[List[float]] = None
        if use_oi and bars:
            pts = OICache(oi_cache_path).load_points()
            if not pts and not oi_warned:
                logger.warning(
                    "OI cache empty; OI feature uses zeros until JSONL is populated",
                    extra={"path": str(oi_cache_path)},
                )
                oi_warned = True
            open_interest = align_open_interest_to_bars(bars, pts)

        bars_by_tf[tf] = (bars, index_closes, open_interest)
        extra = ""
        if use_basis:
            extra = f" [basis vs {idx_symbol}]"
        if use_oi:
            extra += f" [OI→{len(open_interest or [])} pts]"
        print(f"  Fetched {tf} ({api_res}): {len(bars)} bars{extra}")

    rows: List[Dict[str, Any]] = []
    for tf in tfs:
        bars, index_closes, open_interest = bars_by_tf[tf]
        if not bars:
            for w in windows:
                for k in ks:
                    rows.append(
                        {
                            "grid_refit_every": int(doc.get("optimization_refit_every", doc["refit_every"])),
                            "window_size": w,
                            "k_states": k,
                            "timeframe": tf,
                            "n_bars": 0,
                            "n_trades": 0,
                            "valid": False,
                            "fail_reason": "no_bars",
                            "objective_score": None,
                            "objective_score_unconstrained": None,
                        }
                    )
            continue
        for w in windows:
            for k in ks:
                row = evaluate_config(
                    bars, w, k, tf, doc,
                    index_closes=index_closes,
                    open_interest=open_interest,
                )
                rows.append(row)
                v = "OK" if row["valid"] else row["fail_reason"][:40]
                print(
                    f"  {tf} W={w} K={k} trades={row['n_trades']} "
                    f"score={row.get('objective_score_unconstrained')} valid={v}"
                )

    valid_rows = [r for r in rows if r.get("valid")]
    valid_rows.sort(key=lambda r: float(r["objective_score"]), reverse=True)
    top5 = valid_rows[:5]

    # Unconstrained ranking for diagnostics
    rows_scored = sorted(
        [r for r in rows if r.get("objective_score_unconstrained") is not None],
        key=lambda r: float(r["objective_score_unconstrained"]),
        reverse=True,
    )
    top5_any = rows_scored[:5]

    out_csv = _ROOT / doc["output"]["csv"]
    out_md = _ROOT / doc["output"]["markdown"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        fields = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})

    live_tf = doc.get("defaults_from_timeframe", "15m")

    def _pick_live(valid_list: List[dict]) -> Optional[dict]:
        sub = [r for r in valid_list if r.get("timeframe") == live_tf]
        return sub[0] if sub else None

    live_pick = _pick_live(valid_rows)
    global_pick = valid_rows[0] if valid_rows else None

    obj = doc.get("objective") or {}
    oc = float(obj.get("calmar_weight", 0.4))
    os_ = float(obj.get("sharpe_weight", 0.3))
    op = float(obj.get("profit_factor_weight", 0.3))
    objective_md = (
        f"Score = {oc}×Calmar + {os_}×Sharpe + {op}×ProfitFactor (clipped per YAML).\n\n"
    )

    lines_md = [
        "# HMM optimization summary\n",
        f"Config: `{cfg_path.relative_to(_ROOT)}`\n",
        f"Symbol **{symbol}** | {d0}–{d1}\n\n",
        "## Objective\n",
        objective_md,
        "## Features (this run)\n",
        f"- Basis (future vs index): **{use_basis}**"
        + (f" (`index_symbol={idx_symbol}`)\n" if use_basis else "\n"),
        f"- Open interest from cache: **{use_oi}** (`{oi_cache_path}`)\n\n",
        "## Constraints\n",
        f"- Max drawdown < {doc['constraints']['max_drawdown_pct_lt']}%\n",
        f"- Trades > {doc['constraints']['min_trades']}\n",
        f"- Profit factor > {doc['constraints']['profit_factor_gt']}\n",
        f"- Increment-shuffle p-value < {doc['constraints']['p_value_lt']}\n\n",
        "## Top 5 valid configs\n",
    ]

    if not top5:
        lines_md.append("*No configuration satisfied all constraints.*\n\n")
    else:
        lines_md.append(
            "| # | W | K | TF | Score | Trades | Sharpe | Calmar | PF | MDD% | p-value |\n"
        )
        lines_md.append("|---:|---:|---:|:---|------:|-------:|-------:|-------:|---:|-----:|--------:|\n")
        for i, r in enumerate(top5, 1):
            lines_md.append(
                f"| {i} | {r['window_size']} | {r['k_states']} | {r['timeframe']} | "
                f"{r['objective_score']} | {r['n_trades']} | {r.get('sharpe')} | "
                f"{r.get('calmar')} | {r.get('profit_factor')} | {r.get('max_drawdown_pct')} | "
                f"{r.get('p_value')} |\n"
            )

    lines_md.append("\n## Top 5 by raw score (constraints ignored)\n")
    if top5_any:
        lines_md.append(
            "| # | W | K | TF | Score* | Valid | Trades | MDD% | reason |\n"
        )
        lines_md.append("|---:|---:|---:|:---|-------:|:-----|-------:|-----:|:-------|\n")
        for i, r in enumerate(top5_any, 1):
            lines_md.append(
                f"| {i} | {r.get('window_size')} | {r.get('k_states')} | {r.get('timeframe')} | "
                f"{r.get('objective_score_unconstrained')} | {r.get('valid')} | "
                f"{r.get('n_trades')} | {r.get('max_drawdown_pct')} | {str(r.get('fail_reason', ''))[:30]} |\n"
            )
        lines_md.append("\n*Score* = same weighted formula even if invalid.\n")

    lines_md.append("\n## Suggested app defaults\n")
    if live_pick:
        lines_md.append(
            f"**Use for live (timeframe filter `{live_tf}`):** "
            f"`HMM_LIVE_TRAIN_WINDOW_BARS={live_pick['window_size']}`, "
            f"`HMM_LIVE_REFIT_EVERY={doc['refit_every']}` (grid used refit_every={doc.get('optimization_refit_every', doc['refit_every'])}), "
            f"`HMM_LIVE_K_STATES={live_pick['k_states']}`\n"
        )
    elif global_pick:
        lines_md.append(
            f"No valid run on `{live_tf}`; best valid overall is **{global_pick['timeframe']}** "
            f"W={global_pick['window_size']} K={global_pick['k_states']}. "
            "Align live bar resolution with that TF before copying defaults.\n"
        )
    else:
        lines_md.append(
            "No config passed **all** constraints. Consider longer history (drop `max_bars`), "
            "relax thresholds, or validate on a different symbol/period.\n\n"
        )
        sub_live = [r for r in rows_scored if r.get("timeframe") == live_tf]
        if sub_live:
            pick = sub_live[0]
            lines_md.append(
                "**Best-effort (highest raw Score* on live TF, constraints ignored):** "
                f"`HMM_LIVE_TRAIN_WINDOW_BARS={pick['window_size']}`, "
                f"`HMM_LIVE_REFIT_EVERY={doc['refit_every']}`, "
                f"`HMM_LIVE_K_STATES={pick['k_states']}`. "
                f"Observed MDD≈{pick.get('max_drawdown_pct')}%, PF≈{pick.get('profit_factor')} — "
                "review before live use.\n"
            )
        else:
            lines_md.append("No rows for `defaults_from_timeframe`; keep existing `.env` values.\n")

    lines_md.append(f"\nFull grid: `{out_csv.relative_to(_ROOT)}`\n")

    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines_md)

    print("\n" + "=" * 60)
    print("  TOP 5 VALID CONFIGS (by objective score)")
    print("=" * 60)
    if top5:
        print(
            f"  {'#':<3} {'W':>5} {'K':>3} {'TF':>5} {'Score':>10} "
            f"{'Trades':>8} {'Sharpe':>9} {'Calmar':>9} {'PF':>8} {'MDD%':>8} {'p-val':>8}"
        )
        for i, r in enumerate(top5, 1):
            print(
                f"  {i:<3} {r['window_size']:>5} {r['k_states']:>3} {r['timeframe']:>5} "
                f"{r['objective_score']:>10} {r['n_trades']:>8} {r.get('sharpe'):>9} "
                f"{r.get('calmar')!s:>9} {r.get('profit_factor')!s:>8} "
                f"{r.get('max_drawdown_pct'):>8} {r.get('p_value'):>8}"
            )
    else:
        print("  (none — relax constraints or extend history / OI cache)")

    print("\nWrote", out_csv)
    print("Wrote", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
