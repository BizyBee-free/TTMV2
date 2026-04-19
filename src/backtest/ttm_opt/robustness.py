"""Parameter perturbation and institutional robustness metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.backtest.bar_replay import TradeRecord
from src.backtest.ttm_opt.params import TtmOptParams, perturb_params


def perturbation_stability(
    base_metrics_fn,
    p: TtmOptParams,
    n_draws: int,
    seed: int = 42,
) -> Tuple[float, List[float]]:
    """
    ``base_metrics_fn(TtmOptParams) -> BacktestMetrics``.
    Returns mean(stability_score) and list of scores under ±10% param noise.
    """
    from src.backtest.ttm_opt.optimize import stability_score

    rng = np.random.default_rng(seed)
    scores: List[float] = []
    for _ in range(n_draws):
        pp = perturb_params(p, rng, frac=0.10)
        m = base_metrics_fn(pp)
        scores.append(stability_score(m))
    return float(np.mean(scores)) if scores else 0.0, scores


def monte_carlo_equity(
    trade_net_pnls: Sequence[float],
    *,
    n_sims: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """Shuffle trade order / bootstrap equity terminal stats."""
    rng = np.random.default_rng(seed)
    pnls = np.array(trade_net_pnls, dtype=np.float64)
    if len(pnls) == 0:
        return {"median_final": 0.0, "p5_final": 0.0, "p95_final": 0.0}
    finals: List[float] = []
    for _ in range(n_sims):
        order = rng.permutation(len(pnls))
        path = np.cumsum(pnls[order])
        finals.append(float(path[-1]))
    arr = np.array(finals, dtype=np.float64)
    return {
        "median_final": float(np.median(arr)),
        "p5_final": float(np.percentile(arr, 5)),
        "p95_final": float(np.percentile(arr, 95)),
    }


def walk_forward_efficiency(train_scores: Sequence[float], test_scores: Sequence[float]) -> float:
    """Median OOS/IS score efficiency across folds, clipped to [0, 1.5]."""
    pairs = []
    for tr, te in zip(train_scores, test_scores):
        a = float(tr)
        b = float(te)
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if abs(a) <= 1e-9:
            continue
        pairs.append(float(np.clip(b / a, 0.0, 1.5)))
    if not pairs:
        return 0.0
    return float(np.median(np.asarray(pairs, dtype=np.float64)))


def oos_degradation_ratio(
    *,
    is_value: float,
    oos_value: float,
) -> float:
    """Relative degradation max(0, IS-OOS)/|IS|, robust to near-zero IS."""
    a = float(is_value)
    b = float(oos_value)
    if not np.isfinite(a) or abs(a) <= 1e-9:
        return 0.0
    return float(max(0.0, (a - b) / (abs(a) + 1e-9)))


def monte_carlo_worst_case(
    trade_net_pnls: Sequence[float],
    *,
    n_sims: int = 10_000,
    seed: int = 42,
    nav_start: float = 100.0,
) -> Dict[str, float]:
    """
    Randomize trade order and evaluate worst-case risk.

    Returns percentile metrics on final PnL and max drawdown percentage.
    """
    rng = np.random.default_rng(seed)
    pnls = np.asarray(trade_net_pnls, dtype=np.float64)
    if pnls.size == 0:
        return {
            "n_sims": int(n_sims),
            "median_final": 0.0,
            "p5_final": 0.0,
            "p95_final": 0.0,
            "dd_pct_p95": 0.0,
            "dd_pct_p99": 0.0,
            "dd_pct_worst": 0.0,
        }
    finals = np.zeros(int(n_sims), dtype=np.float64)
    dd_pcts = np.zeros(int(n_sims), dtype=np.float64)
    for i in range(int(n_sims)):
        path = np.cumsum(pnls[rng.permutation(pnls.size)])
        finals[i] = float(path[-1])
        nav = float(nav_start) + path
        peak = np.maximum.accumulate(nav)
        dd = np.where(peak > 1e-12, (peak - nav) / peak, 0.0)
        dd_pcts[i] = float(np.max(dd) * 100.0)
    return {
        "n_sims": int(n_sims),
        "median_final": float(np.median(finals)),
        "p5_final": float(np.percentile(finals, 5)),
        "p95_final": float(np.percentile(finals, 95)),
        "dd_pct_p95": float(np.percentile(dd_pcts, 95)),
        "dd_pct_p99": float(np.percentile(dd_pcts, 99)),
        "dd_pct_worst": float(np.max(dd_pcts)),
    }


def net_to_gross_ratio(trades: Sequence[TradeRecord]) -> float:
    gross = 0.0
    net = 0.0
    for t in trades:
        gp = float(getattr(t, "pnl", 0.0) or 0.0)
        npnl = float(getattr(t, "net_pnl", gp) or gp)
        gross += gp
        net += npnl
    if abs(gross) <= 1e-9:
        return 0.0
    return float(net / gross)


def regime_stability_score_5(regime_pnl: Mapping[str, float]) -> Dict[str, Any]:
    """
    Build 5-bucket regime stability score in [0, 1].

    Mapping is approximate from available regime tags in ttm_opt engine.
    """
    get = lambda k: float(regime_pnl.get(k, 0.0))
    buckets = {
        "BullVol": get("trend+hi_vol"),
        "BearLowVol": get("trend+lo_vol"),
        "SidewaysHighVol": get("range+hi_vol"),
        "Crisis": get("trend_unk+hi_vol") + get("trend+vol_unk"),
        "Normal": get("range+mid_vol") + get("trend+mid_vol"),
    }
    vals = np.asarray(list(buckets.values()), dtype=np.float64)
    abs_vals = np.abs(vals)
    total = float(np.sum(abs_vals))
    if total <= 1e-9:
        return {"score": 0.0, "buckets": buckets}
    share = abs_vals / total
    # Lower dispersion across regimes => higher stability.
    cv = float(np.std(share) / (np.mean(share) + 1e-12))
    score = float(np.clip(1.0 - cv, 0.0, 1.0))
    return {"score": score, "buckets": buckets}


def synthetic_metrics_from_pnls(pnls: np.ndarray, n_bars: int, bar_type: str) -> BacktestMetrics:
    """Rebuild metrics after MC shuffle (approximate bar count)."""
    eq = np.cumsum(pnls)
    trades: List[TradeRecord] = []
    for i, x in enumerate(pnls):
        trades.append(
            TradeRecord(
                symbol="MC",
                entry_bar=i,
                exit_bar=i + 1,
                entry_date=str(i),
                exit_date=str(i + 1),
                side="BUY",
                entry_price=1.0,
                exit_price=1.0 + x,
                pnl=x,
                commission=0.0,
                net_pnl=x,
                p_up=0.0,
                p_down=0.0,
                confidence=0.0,
                mcmc_elapsed_ms=0.0,
            )
        )
    return compute_metrics(trades, eq, n_bars=n_bars, bar_type=bar_type, nav_start=100.0)
