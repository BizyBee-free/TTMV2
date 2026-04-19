"""HMM Live-Readiness Gate
=========================
Full test campaign orchestrator for the HMM strategy.  Runs rolling OOS
windows, multi-timeframe tests, Baum-Welch / LL diagnostics, hidden-state
separation analysis, and a latency benchmark.  Appends a structured Go/No-Go
scorecard to reports/TEST_REPORT.md.

Sections written to TEST_REPORT.md
  1. Rolling OOS Robustness  (5 quarterly windows, 1D)
  2. Multi-Timeframe Results (1H, 4H-resampled, 1D)
  3. Baum-Welch / Log-Likelihood Diagnostics
  4. Hidden-State Separation and Labeling
  5. Performance Benchmarks
  6. Go/No-Go Scorecard

Usage:
    python scripts/hmm_live_gate.py               # full run (uses cache)
    python scripts/hmm_live_gate.py --no-report   # run only, no report write
    python scripts/hmm_live_gate.py --fast        # reduced iterations (quicker)
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")

from src.backtest.data_fetcher import DataFetcher, OhlcBar
from src.backtest.hmm_replay import HMMBacktestConfig, HMMBarReplay
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.hmm.feature_engineer import HMMConfig, HMMFeatureEngineer
from src.hmm.hmm_signal import HMMSignalGenerator
from src.hmm.regime_model import HMMRegimeModel, STATE_BEAR, STATE_BULL, STATE_FLAT
from src.logger import get_logger

logger = get_logger("hmm_live_gate")

# ── File paths ────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent.parent
_REPORT_PATH = _ROOT / "reports" / "TEST_REPORT.md"

# ── Rolling OOS window definitions ────────────────────────────────────────────
# Training always starts 20240101.  Each OOS window is one quarter.
ROLLING_WINDOWS = [
    ("20240101", "20241231", "20250101", "20250331", "Q1-2025"),
    ("20240101", "20250331", "20250401", "20250630", "Q2-2025"),
    ("20240101", "20250630", "20250701", "20250930", "Q3-2025"),
    ("20240101", "20250930", "20251001", "20251231", "Q4-2025"),
    ("20240101", "20251231", "20260101", "20260325", "Q1-2026"),
]

# ── Go/No-Go thresholds ───────────────────────────────────────────────────────
GATE = {
    "oos_sharpe_median":         0.80,  # median Sharpe across OOS windows
    "oos_mdd_max_pct":          30.0,   # max MDD % across any OOS window
    "oos_positive_windows_min":  3,     # minimum windows with Sharpe > 0
    "state_separation_min":      0.80,  # min pairwise Mahalanobis distance
    "seed_ll_cv_max":            0.25,  # max CV of final LL over 5 seeds
    "fit_latency_p90_ms":     3000.0,  # fit() p90 latency
    "predict_latency_p95_ms":  200.0,  # predict_proba() p95 latency
    "fit_failure_rate_max":      0.05,  # max fraction of fit failures
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_bars(
    fetcher: DataFetcher,
    symbol: str,
    from_date: str,
    to_date: str,
    resolution: str = "1D",
) -> List[OhlcBar]:
    bars = fetcher.fetch(symbol, from_date, to_date, resolution=resolution)
    if bars:
        print(f"  Loaded {len(bars)} {resolution} bars for {symbol} ({from_date}–{to_date})")
    else:
        print(f"  WARNING: No bars for {symbol} {resolution} {from_date}–{to_date}")
    return bars


def resample_4h(bars_1h: List[OhlcBar]) -> List[OhlcBar]:
    """Aggregate 1H bars into 4H bars (group every 4 consecutive bars)."""
    result = []
    for i in range(0, len(bars_1h), 4):
        chunk = bars_1h[i: i + 4]
        if not chunk:
            break
        result.append(OhlcBar(
            symbol=chunk[0].symbol,
            time=chunk[0].time,
            open=chunk[0].open,
            high=max(b.high for b in chunk),
            low=min(b.low for b in chunk),
            close=chunk[-1].close,
            volume=sum(b.volume for b in chunk),
            unix_ts=chunk[0].unix_ts,
        ))
    return result


def filter_bars(bars: List[OhlcBar], from_date: str, to_date: str) -> List[OhlcBar]:
    return [b for b in bars if from_date <= b.date_str <= to_date]


def aic_bic(ll: float, n: int, k: int, d: int = 3) -> Tuple[float, float]:
    n_params = k * (k - 1) + (k - 1) + k * d + k * d * (d + 1) // 2
    aic = -2.0 * ll + 2.0 * n_params
    bic = -2.0 * ll + n_params * math.log(max(n, 2))
    return aic, bic


def mahalanobis(mu1: np.ndarray, mu2: np.ndarray, cov: np.ndarray) -> float:
    diff = mu1 - mu2
    try:
        inv = np.linalg.inv(cov + np.eye(len(cov)) * 1e-8)
        return float(np.sqrt(max(0.0, diff @ inv @ diff)))
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(diff))


# ── Section 1: Rolling OOS ────────────────────────────────────────────────────

@dataclass
class WindowResult:
    label: str
    n_bars: int
    n_trades: int
    sharpe: float
    mdd_pct: float
    win_rate_pct: float
    expectancy: float
    total_return_pct: float
    ll_final: float = float("-inf")


def run_rolling_oos(
    all_bars: List[OhlcBar],
    hmm_cfg: HMMConfig,
    confidence: float = 0.75,
    warmup: int = 30,
    refit_every: int = 5,
) -> List[WindowResult]:
    results = []
    for train_from, train_to, oos_from, oos_to, label in ROLLING_WINDOWS:
        train_bars = filter_bars(all_bars, train_from, train_to)
        oos_bars   = filter_bars(all_bars, oos_from, oos_to)
        if len(train_bars) < 50 or len(oos_bars) < 5:
            print(f"  [{label}] skipped — insufficient data "
                  f"(train={len(train_bars)}, oos={len(oos_bars)})")
            continue
        # Use full span bars so walk-forward engine builds history properly
        span_bars = filter_bars(all_bars, train_from, oos_to)
        replay_cfg = HMMBacktestConfig(
            symbol="VN30F1M",
            hmm_config=hmm_cfg,
            warmup_bars=warmup,
            confidence_threshold=confidence,
            position_size=1,
            commission_pct=0.0,
            refit_every=refit_every,
        )
        trades_all, equity_all = HMMBarReplay().run(span_bars, replay_cfg)
        oos_trades = [t for t in trades_all if t.entry_date >= oos_from]
        oos_start  = next(
            (i for i, b in enumerate(span_bars) if b.date_str >= oos_from), 0
        )
        oos_equity = equity_all[oos_start:]
        m = compute_metrics(oos_trades, oos_equity, len(oos_bars), "D")

        # Collect LL from the last fit on training data
        eng   = HMMFeatureEngineer(hmm_cfg)
        X_tr  = eng.compute(train_bars)
        model = HMMRegimeModel(hmm_cfg)
        model.fit(X_tr)
        ll = model.log_likelihood

        wr = WindowResult(
            label=label, n_bars=len(oos_bars), n_trades=m.n_trades,
            sharpe=m.sharpe_ratio, mdd_pct=m.max_drawdown_pct,
            win_rate_pct=m.win_rate_pct, expectancy=m.expectancy,
            total_return_pct=m.total_return_pct, ll_final=ll,
        )
        results.append(wr)
        print(f"  [{label}] OOS: trades={m.n_trades}, Sharpe={m.sharpe_ratio:.3f}, "
              f"MDD={m.max_drawdown_pct:.1f}%, WinRate={m.win_rate_pct:.1f}%, "
              f"LL={ll:.1f}")
    return results


# ── Section 2: Multi-Timeframe ────────────────────────────────────────────────

@dataclass
class TFResult:
    timeframe: str
    n_bars: int
    n_trades: int
    sharpe: float
    mdd_pct: float
    win_rate_pct: float
    expectancy: float


def run_timeframe(
    bars: List[OhlcBar],
    tf_label: str,
    hmm_cfg: HMMConfig,
    confidence: float = 0.75,
    warmup: int = 30,
    refit_every: int = 5,
    bar_type: str = "D",
) -> TFResult:
    replay_cfg = HMMBacktestConfig(
        symbol="VN30F1M",
        hmm_config=hmm_cfg,
        warmup_bars=warmup,
        confidence_threshold=confidence,
        position_size=1,
        commission_pct=0.0,
        refit_every=refit_every,
    )
    trades, equity = HMMBarReplay().run(bars, replay_cfg)
    m = compute_metrics(trades, equity, len(bars), bar_type)
    print(f"  [{tf_label}] n_bars={len(bars)}, trades={m.n_trades}, "
          f"Sharpe={m.sharpe_ratio:.3f}, MDD={m.max_drawdown_pct:.1f}%, "
          f"WinRate={m.win_rate_pct:.1f}%")
    return TFResult(
        timeframe=tf_label, n_bars=len(bars), n_trades=m.n_trades,
        sharpe=m.sharpe_ratio, mdd_pct=m.max_drawdown_pct,
        win_rate_pct=m.win_rate_pct, expectancy=m.expectancy,
    )


# ── Section 3 & 4: Diagnostics and State Separation ──────────────────────────

@dataclass
class DiagResult:
    n_samples: int
    ll_final: float
    ll_history_len: int
    ll_monotonic: bool
    ll_per_sample: float
    seed_cv: float        # CV of LL across 5 seeds
    aic_k2: float
    aic_k3: float
    aic_k4: float
    bic_k3: float
    k3_aic_gap_pct: float  # gap from best AIC


@dataclass
class StateSepResult:
    bull_mean_ret: float
    flat_mean_ret: float
    bear_mean_ret: float
    semantics_correct: bool
    min_mahal: float          # minimum pairwise Mahalanobis distance
    self_transitions: List[float]  # diagonal of transmat
    n_persistent_states: int  # states with self-transition > 0.30


def collect_diagnostics(
    bars: List[OhlcBar],
    hmm_cfg: HMMConfig,
    n_seeds: int = 5,
) -> Tuple[DiagResult, StateSepResult]:
    eng = HMMFeatureEngineer(hmm_cfg)
    X = eng.compute(bars)

    # Primary fit (fixed seed)
    model = HMMRegimeModel(hmm_cfg)
    model.fit(X)

    # LL monotonicity
    history = model.ll_history
    monotonic = all(
        history[i] >= history[i - 1] - 1e-3
        for i in range(1, len(history))
    ) if len(history) > 1 else True

    # Multi-seed LL stability
    lls = []
    for seed in range(n_seeds):
        cfg_s = HMMConfig(
            k_states=hmm_cfg.k_states, n_iter=hmm_cfg.n_iter,
            zscore_window=hmm_cfg.zscore_window, random_state=seed,
        )
        m = HMMRegimeModel(cfg_s)
        if m.fit(X) and m.log_likelihood > float("-inf"):
            lls.append(m.log_likelihood)
    mean_ll = statistics.mean(lls) if lls else float("-inf")
    std_ll  = statistics.stdev(lls) if len(lls) > 1 else 0.0
    seed_cv = abs(std_ll / mean_ll) if abs(mean_ll) > 1e-9 else 0.0

    # AIC/BIC for K = 2, 3, 4
    aic_vals: Dict[int, float] = {}
    bic_vals: Dict[int, float] = {}
    for k in [2, 3, 4]:
        cfg_k = HMMConfig(
            k_states=k, n_iter=hmm_cfg.n_iter,
            zscore_window=hmm_cfg.zscore_window, random_state=42,
        )
        mk = HMMRegimeModel(cfg_k)
        if mk.fit(X) and mk.log_likelihood > float("-inf"):
            a, b = aic_bic(mk.log_likelihood, X.shape[0], k)
            aic_vals[k] = a
            bic_vals[k] = b

    best_aic    = min(aic_vals.values()) if aic_vals else 0.0
    k3_aic      = aic_vals.get(3, float("inf"))
    aic_gap_pct = abs(k3_aic - best_aic) / abs(best_aic) if abs(best_aic) > 1e-6 else 0.0

    diag = DiagResult(
        n_samples=X.shape[0],
        ll_final=model.log_likelihood,
        ll_history_len=len(history),
        ll_monotonic=monotonic,
        ll_per_sample=model.log_likelihood / X.shape[0] if X.shape[0] > 0 else float("-inf"),
        seed_cv=seed_cv,
        aic_k2=aic_vals.get(2, float("nan")),
        aic_k3=aic_vals.get(3, float("nan")),
        aic_k4=aic_vals.get(4, float("nan")),
        bic_k3=bic_vals.get(3, float("nan")),
        k3_aic_gap_pct=aic_gap_pct,
    )

    # State separation
    labels = model.state_labels
    means  = model.state_means    # (k, d)
    covars = model.state_covars   # (k, d, d)
    T      = model.transition_matrix

    bull_mean = flat_mean = bear_mean = float("nan")
    semantics_correct = False
    min_mahal = float("nan")
    self_trans: List[float] = []
    n_persistent = 0

    if means is not None and covars is not None and T is not None:
        bull_idx = next((i for i, v in labels.items() if v == STATE_BULL), None)
        flat_idx = next((i for i, v in labels.items() if v == STATE_FLAT), None)
        bear_idx = next((i for i, v in labels.items() if v == STATE_BEAR), None)

        if bull_idx is not None:
            bull_mean = float(means[bull_idx, 0])
        if flat_idx is not None:
            flat_mean = float(means[flat_idx, 0])
        if bear_idx is not None:
            bear_mean = float(means[bear_idx, 0])

        if bull_idx is not None and bear_idx is not None:
            semantics_correct = bull_mean >= bear_mean

        # Pairwise Mahalanobis
        k = means.shape[0]
        dists = []
        for i in range(k):
            for j in range(i + 1, k):
                pooled = (covars[i] + covars[j]) / 2.0
                dists.append(mahalanobis(means[i], means[j], pooled))
        min_mahal = min(dists) if dists else float("nan")

        # State persistence
        self_trans = list(T.diagonal())
        n_persistent = int(np.sum(T.diagonal() > 0.30))

    sep = StateSepResult(
        bull_mean_ret=bull_mean, flat_mean_ret=flat_mean, bear_mean_ret=bear_mean,
        semantics_correct=semantics_correct,
        min_mahal=min_mahal,
        self_transitions=self_trans,
        n_persistent_states=n_persistent,
    )
    return diag, sep


# ── Section 5: Performance Benchmarks ────────────────────────────────────────

@dataclass
class PerfResult:
    fit_latencies_ms: List[float]
    predict_latencies_ms: List[float]
    fit_p50_ms: float
    fit_p90_ms: float
    fit_p95_ms: float
    predict_p50_ms: float
    predict_p90_ms: float
    predict_p95_ms: float
    fit_failure_rate: float


def benchmark_performance(
    bars: List[OhlcBar],
    hmm_cfg: HMMConfig,
    n_fit_trials: int = 10,
    n_predict_trials: int = 50,
) -> PerfResult:
    eng = HMMFeatureEngineer(hmm_cfg)
    X = eng.compute(bars)
    X_single = X[-1:]

    # Fit latency
    fit_lat = []
    failures = 0
    for seed in range(n_fit_trials):
        cfg_s = HMMConfig(
            k_states=hmm_cfg.k_states, n_iter=hmm_cfg.n_iter,
            zscore_window=hmm_cfg.zscore_window, random_state=seed,
        )
        m = HMMRegimeModel(cfg_s)
        t0 = time.perf_counter()
        ok = m.fit(X)
        fit_lat.append((time.perf_counter() - t0) * 1000)
        if not ok:
            failures += 1
    fit_lat.sort()

    # Predict latency (use a fitted model)
    model_ref = HMMRegimeModel(hmm_cfg)
    model_ref.fit(X)
    pred_lat = []
    for _ in range(n_predict_trials):
        t0 = time.perf_counter()
        model_ref.predict_proba(X_single)
        pred_lat.append((time.perf_counter() - t0) * 1000)
    pred_lat.sort()

    def pct(lst, p):
        return lst[min(int(len(lst) * p), len(lst) - 1)]

    return PerfResult(
        fit_latencies_ms=fit_lat,
        predict_latencies_ms=pred_lat,
        fit_p50_ms=pct(fit_lat, 0.50),
        fit_p90_ms=pct(fit_lat, 0.90),
        fit_p95_ms=pct(fit_lat, 0.95),
        predict_p50_ms=pct(pred_lat, 0.50),
        predict_p90_ms=pct(pred_lat, 0.90),
        predict_p95_ms=pct(pred_lat, 0.95),
        fit_failure_rate=failures / n_fit_trials,
    )


# ── Section 6: Go/No-Go Scorecard ────────────────────────────────────────────

@dataclass
class GateResult:
    gate: str
    measured: str
    threshold: str
    passed: bool
    severity: str   # "HARD" or "WARN"
    note: str = ""


def evaluate_gates(
    oos_results: List[WindowResult],
    diag: DiagResult,
    sep: StateSepResult,
    perf: PerfResult,
) -> Tuple[str, List[GateResult]]:
    gates: List[GateResult] = []

    def gate(name, measured_val, threshold_str, passed, severity="HARD", note=""):
        gates.append(GateResult(
            gate=name, measured=str(measured_val),
            threshold=threshold_str, passed=passed,
            severity=severity, note=note,
        ))

    # ── OOS Robustness ───────────────────────────────────────────────────────
    if oos_results:
        sharpes = [r.sharpe for r in oos_results]
        mdds    = [r.mdd_pct for r in oos_results]
        median_sharpe  = statistics.median(sharpes)
        max_mdd        = max(mdds)
        n_positive     = sum(1 for s in sharpes if s > 0)

        gate(
            "OOS Sharpe (median)",
            f"{median_sharpe:.3f}",
            f"≥ {GATE['oos_sharpe_median']}",
            median_sharpe >= GATE["oos_sharpe_median"],
            "HARD",
        )
        gate(
            "OOS Max Drawdown",
            f"{max_mdd:.1f}%",
            f"≤ {GATE['oos_mdd_max_pct']}%",
            max_mdd <= GATE["oos_mdd_max_pct"],
            "HARD",
        )
        gate(
            "OOS Positive Windows",
            f"{n_positive}/{len(oos_results)}",
            f"≥ {GATE['oos_positive_windows_min']}",
            n_positive >= GATE["oos_positive_windows_min"],
            "HARD",
        )
    else:
        gate("OOS Robustness", "NO DATA", "≥3 windows", False, "HARD", "No OOS data loaded")

    # ── Model Quality ────────────────────────────────────────────────────────
    gate(
        "Baum-Welch LL monotonic",
        "YES" if diag.ll_monotonic else "NO",
        "YES",
        diag.ll_monotonic,
        "HARD",
        "EM convergence violated — implementation error",
    )
    gate(
        "Multi-seed LL stability (CV)",
        f"{diag.seed_cv:.3f}",
        f"≤ {GATE['seed_ll_cv_max']}",
        diag.seed_cv <= GATE["seed_ll_cv_max"],
        "WARN",
        "High CV means results are seed-dependent",
    )
    gate(
        "K=3 AIC gap from best",
        f"{diag.k3_aic_gap_pct:.1%}",
        "≤ 10%",
        diag.k3_aic_gap_pct <= 0.10,
        "WARN",
        "If K=2 or K=4 is much better, the 3-state assumption is unjustified",
    )

    # ── State Separation ─────────────────────────────────────────────────────
    gate(
        "Regime semantics (BULL>BEAR)",
        "CORRECT" if sep.semantics_correct else "INVERTED",
        "CORRECT",
        sep.semantics_correct,
        "HARD",
        "Inverted labels → all live trades in wrong direction",
    )
    min_mahal_ok = (not math.isnan(sep.min_mahal)) and sep.min_mahal >= GATE["state_separation_min"]
    gate(
        "Min state Mahal. distance",
        f"{sep.min_mahal:.3f}" if not math.isnan(sep.min_mahal) else "N/A",
        f"≥ {GATE['state_separation_min']}",
        min_mahal_ok,
        "WARN",
        "Low separation = states overlap = unreliable regime labels",
    )
    gate(
        "Regime persistence (states > 0.30)",
        f"{sep.n_persistent_states}/3",
        "≥ 2",
        sep.n_persistent_states >= 2,
        "WARN",
        "Flickering states cannot sustain multi-bar holds",
    )

    # ── Performance ──────────────────────────────────────────────────────────
    gate(
        "Fit latency p90",
        f"{perf.fit_p90_ms:.0f}ms",
        f"≤ {GATE['fit_latency_p90_ms']:.0f}ms",
        perf.fit_p90_ms <= GATE["fit_latency_p90_ms"],
        "WARN",
        "Slow fit blocks live refit frequency",
    )
    gate(
        "Predict latency p95",
        f"{perf.predict_p95_ms:.1f}ms",
        f"≤ {GATE['predict_latency_p95_ms']:.0f}ms",
        perf.predict_p95_ms <= GATE["predict_latency_p95_ms"],
        "WARN",
        "High predict latency causes missed entries",
    )
    gate(
        "Fit failure rate",
        f"{perf.fit_failure_rate:.1%}",
        f"≤ {GATE['fit_failure_rate_max']:.0%}",
        perf.fit_failure_rate <= GATE["fit_failure_rate_max"],
        "WARN",
    )

    # ── Decision ─────────────────────────────────────────────────────────────
    hard_fails = [g for g in gates if not g.passed and g.severity == "HARD"]
    warn_fails = [g for g in gates if not g.passed and g.severity == "WARN"]

    if not hard_fails and not warn_fails:
        decision = "Go"
    elif not hard_fails and len(warn_fails) <= 2:
        decision = "Conditional Go"
    elif not hard_fails:
        decision = "Conditional Go (multiple warnings)"
    else:
        decision = "No-Go"

    return decision, gates


# ── Report helpers ────────────────────────────────────────────────────────────

def _append(txt: str) -> None:
    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(txt)


def build_report(
    oos_results: List[WindowResult],
    tf_results: List[TFResult],
    diag: DiagResult,
    sep: StateSepResult,
    perf: PerfResult,
    decision: str,
    gates: List[GateResult],
    fast: bool,
    hmm_cfg: HMMConfig,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"\n---\n\n# HMM Live-Readiness Gate Report [{now}]\n",
        f"**HMM config:** k_states={hmm_cfg.k_states} | n_iter={hmm_cfg.n_iter} | "
        f"zscore_window={hmm_cfg.zscore_window} | fast={fast}\n\n",
    ]

    # Section 1: Rolling OOS
    lines.append("## 1. Rolling OOS Robustness (1D, VN30F1M proxy)\n\n")
    if oos_results:
        lines.append("| Window | Trades | Sharpe | MDD% | WinRate% | Return% | LL (IS) |\n")
        lines.append("|--------|--------|--------|------|----------|---------|----------|\n")
        for r in oos_results:
            lines.append(
                f"| {r.label} | {r.n_trades} | {r.sharpe:.3f} | "
                f"{r.mdd_pct:.1f}% | {r.win_rate_pct:.1f}% | "
                f"{r.total_return_pct:.2f}% | {r.ll_final:.1f} |\n"
            )
        sharpes = [r.sharpe for r in oos_results]
        lines.append(
            f"\n**Median Sharpe:** {statistics.median(sharpes):.3f} | "
            f"**Min:** {min(sharpes):.3f} | **Max:** {max(sharpes):.3f}\n\n"
        )
    else:
        lines.append("*No OOS data available.*\n\n")

    # Section 2: Multi-Timeframe
    lines.append("## 2. Multi-Timeframe Results (Sep–Dec 2025)\n\n")
    if tf_results:
        lines.append("| Timeframe | Bars | Trades | Sharpe | MDD% | WinRate% |\n")
        lines.append("|-----------|------|--------|--------|------|----------|\n")
        for r in tf_results:
            lines.append(
                f"| {r.timeframe} | {r.n_bars} | {r.n_trades} | "
                f"{r.sharpe:.3f} | {r.mdd_pct:.1f}% | {r.win_rate_pct:.1f}% |\n"
            )
        lines.append("\n")
    else:
        lines.append("*No intraday data available.*\n\n")

    # Section 3: Baum-Welch / LL Diagnostics
    lines.append("## 3. Baum-Welch / Log-Likelihood Diagnostics\n\n")
    lines.append("| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| Training samples | {diag.n_samples} |\n")
    lines.append(f"| Final LL | {diag.ll_final:.2f} |\n")
    lines.append(f"| LL / sample | {diag.ll_per_sample:.4f} |\n")
    lines.append(f"| LL history length (iters) | {diag.ll_history_len} |\n")
    lines.append(f"| LL monotonic | {'YES' if diag.ll_monotonic else 'NO'} |\n")
    lines.append(f"| Multi-seed LL CV | {diag.seed_cv:.3f} |\n")
    lines.append(f"| AIC (K=2) | {diag.aic_k2:.1f} |\n")
    lines.append(f"| AIC (K=3) | {diag.aic_k3:.1f} |\n")
    lines.append(f"| AIC (K=4) | {diag.aic_k4:.1f} |\n")
    lines.append(f"| BIC (K=3) | {diag.bic_k3:.1f} |\n")
    lines.append(f"| K=3 AIC gap from best | {diag.k3_aic_gap_pct:.1%} |\n\n")

    # Section 4: State Separation
    lines.append("## 4. Hidden-State Separation and Labeling\n\n")
    lines.append("| Metric | Value |\n|--------|-------|\n")
    lines.append(f"| Regime semantics correct | {'YES' if sep.semantics_correct else 'NO — INVERTED'} |\n")
    lines.append(f"| BULL mean log-return | {sep.bull_mean_ret:.5f} |\n")
    lines.append(f"| FLAT mean log-return | {sep.flat_mean_ret:.5f} |\n")
    lines.append(f"| BEAR mean log-return | {sep.bear_mean_ret:.5f} |\n")
    lines.append(f"| Min pairwise Mahal. distance | {sep.min_mahal:.3f} |\n")
    st_str = ", ".join(f"{v:.3f}" for v in sep.self_transitions)
    lines.append(f"| Self-transition probs | [{st_str}] |\n")
    lines.append(f"| Persistent states (>0.30) | {sep.n_persistent_states}/3 |\n\n")

    # Section 5: Performance
    lines.append("## 5. Performance Benchmarks\n\n")
    lines.append("| Metric | p50 | p90 | p95 |\n|--------|-----|-----|-----|\n")
    lines.append(
        f"| Fit latency (ms) | {perf.fit_p50_ms:.0f} | {perf.fit_p90_ms:.0f} | "
        f"{perf.fit_p95_ms:.0f} |\n"
    )
    lines.append(
        f"| Predict latency (ms) | {perf.predict_p50_ms:.1f} | {perf.predict_p90_ms:.1f} | "
        f"{perf.predict_p95_ms:.1f} |\n"
    )
    lines.append(f"\n**Fit failure rate:** {perf.fit_failure_rate:.1%}\n\n")

    # Section 6: Scorecard
    lines.append("## 6. Go/No-Go Scorecard\n\n")
    lines.append("| Gate | Measured | Threshold | Result | Severity |\n")
    lines.append("|------|----------|-----------|--------|----------|\n")
    for g in gates:
        result_str = "PASS" if g.passed else "FAIL"
        lines.append(
            f"| {g.gate} | {g.measured} | {g.threshold} | "
            f"**{result_str}** | {g.severity} |\n"
        )

    pass_count = sum(1 for g in gates if g.passed)
    total      = len(gates)
    lines.append(f"\n**Score: {pass_count}/{total} gates passed.**\n\n")

    dec_icon = {"Go": "GO", "Conditional Go": "CONDITIONAL GO",
                "Conditional Go (multiple warnings)": "CONDITIONAL GO",
                "No-Go": "NO-GO"}.get(decision, decision)
    lines.append(f"### Decision: {dec_icon}\n\n")

    hard_fails = [g for g in gates if not g.passed and g.severity == "HARD"]
    warn_fails = [g for g in gates if not g.passed and g.severity == "WARN"]

    if hard_fails:
        lines.append("**Hard failures (must fix before live):**\n")
        for g in hard_fails:
            lines.append(f"- `{g.gate}`: {g.measured} vs {g.threshold}. {g.note}\n")
        lines.append("\n")
    if warn_fails:
        lines.append("**Warnings (monitor closely):**\n")
        for g in warn_fails:
            lines.append(f"- `{g.gate}`: {g.measured} vs {g.threshold}. {g.note}\n")
        lines.append("\n")

    if decision == "Go":
        lines.append(
            "> All gates passed. Proceed to **2-4 week paper-trading shadow run** "
            "before deploying small capital.\n\n"
        )
    elif "Conditional" in decision:
        lines.append(
            "> Warnings present. Proceed to paper-trading but monitor flagged "
            "metrics daily. Do NOT increase position size until warnings resolve.\n\n"
        )
    else:
        lines.append(
            "> Hard failures block live deployment. Fix hard failures and rerun "
            "this gate script before any live capital exposure.\n\n"
        )

    lines.append(
        "> **Proxy note:** VN30F1M historical data is proxied via VN30 index "
        "(DNSE REST API does not serve derivative historical OHLC). "
        "Live futures may diverge from index in high-volatility periods.\n\n"
    )

    return "".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HMM Live-Readiness Gate")
    p.add_argument("--fast",      action="store_true",
                   help="Fewer iterations (quicker run, less precise)")
    p.add_argument("--no-report", action="store_true",
                   help="Do not append to TEST_REPORT.md")
    p.add_argument("--confidence", type=float, default=0.75)
    p.add_argument("--warmup",     type=int,   default=30)
    p.add_argument("--refit-every",type=int,   default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    n_iter    = 100 if not args.fast else 50
    hmm_cfg   = HMMConfig(k_states=3, n_iter=n_iter, zscore_window=20, random_state=42)
    fetcher   = DataFetcher(use_cache=True)

    print("=" * 65)
    print("HMM Live-Readiness Gate")
    print(f"n_iter={n_iter} | confidence={args.confidence} | "
          f"warmup={args.warmup} | refit_every={args.refit_every}")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[Data] Loading VN30F1M 1D (2024-01-01 – 2026-03-25)…")
    daily_bars = load_bars(fetcher, "VN30F1M", "20240101", "20260325", "1D")

    print("\n[Data] Loading VN30F1M 1H (2025-09-01 – 2025-12-31)…")
    hourly_bars = load_bars(fetcher, "VN30F1M", "20250901", "20251231", "1H")
    bars_4h     = resample_4h(hourly_bars) if hourly_bars else []
    if bars_4h:
        print(f"  Resampled to {len(bars_4h)} 4H bars")

    # ── Section 1: Rolling OOS ────────────────────────────────────────────────
    print("\n[1/6] Rolling OOS Robustness…")
    oos_results: List[WindowResult] = []
    if daily_bars:
        oos_results = run_rolling_oos(
            daily_bars, hmm_cfg,
            confidence=args.confidence,
            warmup=args.warmup,
            refit_every=args.refit_every,
        )
    else:
        print("  Skipped — no daily data")

    # ── Section 2: Multi-Timeframe ────────────────────────────────────────────
    print("\n[2/6] Multi-Timeframe Tests…")
    tf_results: List[TFResult] = []

    if daily_bars:
        bars_1d_is = filter_bars(daily_bars, "20250901", "20251231")
        if bars_1d_is:
            tf_results.append(run_timeframe(
                bars_1d_is, "1D", hmm_cfg,
                confidence=args.confidence, warmup=args.warmup,
                refit_every=args.refit_every, bar_type="D",
            ))

    if hourly_bars:
        tf_results.append(run_timeframe(
            hourly_bars, "1H", hmm_cfg,
            confidence=args.confidence, warmup=args.warmup,
            refit_every=args.refit_every, bar_type="60",
        ))
    if bars_4h:
        tf_results.append(run_timeframe(
            bars_4h, "4H", hmm_cfg,
            confidence=args.confidence, warmup=args.warmup,
            refit_every=args.refit_every, bar_type="60",
        ))

    # ── Sections 3 & 4: Diagnostics ──────────────────────────────────────────
    print("\n[3-4/6] Baum-Welch / LL Diagnostics & State Separation…")
    diag_bars = (filter_bars(daily_bars, "20240101", "20251231")
                 if daily_bars else [])
    if diag_bars:
        diag, sep = collect_diagnostics(
            diag_bars, hmm_cfg, n_seeds=5 if not args.fast else 3
        )
        print(f"  LL={diag.ll_final:.2f} | LL/sample={diag.ll_per_sample:.4f} | "
              f"monotonic={diag.ll_monotonic} | seed_CV={diag.seed_cv:.3f}")
        print(f"  AIC K2={diag.aic_k2:.0f} K3={diag.aic_k3:.0f} K4={diag.aic_k4:.0f} "
              f"| K3-gap={diag.k3_aic_gap_pct:.1%}")
        print(f"  Semantics correct={sep.semantics_correct} | "
              f"min Mahal={sep.min_mahal:.3f} | "
              f"persistent={sep.n_persistent_states}/3")
    else:
        # Fallback diagnostics on empty data
        from dataclasses import fields as dc_fields
        diag = DiagResult(0, float("-inf"), 0, True, float("-inf"), 0.0,
                          float("nan"), float("nan"), float("nan"), float("nan"), 0.0)
        sep  = StateSepResult(float("nan"), float("nan"), float("nan"),
                              False, float("nan"), [], 0)
        print("  Skipped — no daily data for diagnostics")

    # ── Section 5: Performance ────────────────────────────────────────────────
    print("\n[5/6] Performance Benchmarks…")
    bench_bars = (filter_bars(daily_bars, "20240101", "20251231")[:200]
                  if daily_bars else [])
    if bench_bars and len(bench_bars) >= 30:
        perf = benchmark_performance(
            bench_bars, hmm_cfg,
            n_fit_trials=10 if not args.fast else 5,
            n_predict_trials=50 if not args.fast else 20,
        )
        print(f"  Fit  p50={perf.fit_p50_ms:.0f}ms  p90={perf.fit_p90_ms:.0f}ms  "
              f"p95={perf.fit_p95_ms:.0f}ms  failure={perf.fit_failure_rate:.1%}")
        print(f"  Pred p50={perf.predict_p50_ms:.1f}ms  p90={perf.predict_p90_ms:.1f}ms  "
              f"p95={perf.predict_p95_ms:.1f}ms")
    else:
        perf = PerfResult([], [], float("inf"), float("inf"), float("inf"),
                          float("inf"), float("inf"), float("inf"), 1.0)
        print("  Skipped — no data for benchmarks")

    # ── Section 6: Scorecard ──────────────────────────────────────────────────
    print("\n[6/6] Go/No-Go Scorecard…")
    decision, gates = evaluate_gates(oos_results, diag, sep, perf)

    pass_count = sum(1 for g in gates if g.passed)
    print(f"\n  Gates: {pass_count}/{len(gates)} passed")
    for g in gates:
        icon = "PASS" if g.passed else "FAIL"
        thresh = g.threshold.replace("\u2265", ">=").replace("\u2264", "<=")
        print(f"    [{icon}] {g.gate}: {g.measured} (threshold {thresh})")
    print(f"\n  DECISION: {decision}")

    # ── Report ────────────────────────────────────────────────────────────────
    if not args.no_report:
        report = build_report(
            oos_results, tf_results, diag, sep, perf, decision, gates,
            args.fast, hmm_cfg,
        )
        _append(report)
        print(f"\n  Report appended to {_REPORT_PATH}")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    main()
