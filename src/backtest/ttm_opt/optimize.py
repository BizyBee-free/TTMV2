"""Bayesian (Optuna) or random search over TtmOptParams with walk-forward scoring."""

from __future__ import annotations

import json
import math
import random
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.backtest.data_fetcher import OhlcBar
from src.backtest.metrics import BacktestMetrics
from src.backtest.ttm_opt.engine import run_ttm_opt_backtest
from src.backtest.ttm_opt.params import (
    ADX_THRESHOLD_GRID,
    ATR_REGIME_WIN_GRID,
    BB_LEN_GRID,
    BB_STD_GRID,
    BASIS_DELTA_WIN_GRID,
    BASIS_THRESHOLD_GRID,
    BASIS_Z_WIN_GRID,
    KC_ATR_MULT_GRID,
    KC_LEN_GRID,
    OI_DELTA_WIN_GRID,
    OI_MA_WIN_GRID,
    OI_SLOPE_WIN_GRID,
    POSITION_RISK_PCT_GRID,
    STOP_LOSS_PCT_GRID,
    TAKE_PROFIT_PCT_GRID,
    TtmOptParams,
    TRAILING_PCT_GRID,
)
from src.backtest.ttm_opt.robustness import oos_degradation_ratio, walk_forward_efficiency
from src.backtest.ttm_opt.walk_forward import WFSplit, walk_forward_splits


def bars_to_arrays(bars: Sequence[OhlcBar]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    o = np.array([float(b.open) for b in bars], dtype=np.float64)
    h = np.array([float(b.high) for b in bars], dtype=np.float64)
    l = np.array([float(b.low) for b in bars], dtype=np.float64)
    c = np.array([float(b.close) for b in bars], dtype=np.float64)
    tl = [str(getattr(b, "time", "") or "") for b in bars]
    return o, h, l, c, tl


def _cap(x: float, lim: float = 20.0) -> float:
    if not math.isfinite(x):
        return 0.0
    return float(max(-lim, min(lim, x)))


def stability_score(m: BacktestMetrics) -> float:
    """
    Risk-adjusted composite (NOT raw profit): Sharpe + Calmar − drawdown penalty.
    Penalize very sparse trading.
    """
    if m.n_trades < 3:
        return -2.0 + 0.05 * m.n_trades
    cal = m.calmar_ratio if math.isfinite(m.calmar_ratio) else 0.0
    cal = min(cal, 15.0)
    dd = m.max_drawdown_pct / 100.0
    dd_pen = min(dd, 0.5) * 1.5
    sh = _cap(m.sharpe_ratio, 8.0)
    return 0.45 * sh + 0.25 * cal - 0.35 * dd_pen + 0.05 * min(m.n_trades / 50.0, 1.0)


def evaluate_on_slice(
    p: TtmOptParams,
    sl: slice,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    tl: List[str],
    basis: np.ndarray,
    oi: np.ndarray,
    bar_type: str,
) -> Dict[str, Any]:
    o_s = o[sl]
    h_s = h[sl]
    l_s = l[sl]
    c_s = c[sl]
    n = len(c_s)
    tl_s = tl[sl.start : sl.stop] if sl.stop <= len(tl) else [str(i) for i in range(n)]
    b_s = basis[sl] if len(basis) == len(c) else np.zeros(n, dtype=np.float64)
    oi_s = oi[sl] if len(oi) == len(c) else np.zeros(n, dtype=np.float64)
    if len(b_s) != n:
        b_s = np.zeros(n, dtype=np.float64)
    if len(oi_s) != n:
        oi_s = np.zeros(n, dtype=np.float64)
    _, _, pack = run_ttm_opt_backtest(
        o_s, h_s, l_s, c_s, tl_s, b_s, oi_s, p, bar_type=bar_type
    )
    return pack


def walk_forward_objective(
    p: TtmOptParams,
    splits: Sequence[WFSplit],
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    tl: List[str],
    basis: np.ndarray,
    oi: np.ndarray,
    bar_type: str,
    *,
    use_test_only: bool = False,
    hard_gates: Dict[str, float] | None = None,
) -> Tuple[float, List[float], List[float], Dict[str, Any]]:
    """
    Returns:
        aggregate_score, train_scores, test_scores (stability_score per fold)
    """
    tr_s: List[float] = []
    te_s: List[float] = []
    tr_metrics: List[BacktestMetrics] = []
    te_metrics: List[BacktestMetrics] = []
    for sp in splits:
        tr_sl = slice(sp.train_start, sp.train_end)
        te_sl = slice(sp.test_start, sp.test_end)
        p_tr = evaluate_on_slice(p, tr_sl, o, h, l, c, tl, basis, oi, bar_type)
        p_te = evaluate_on_slice(p, te_sl, o, h, l, c, tl, basis, oi, bar_type)
        m_tr = p_tr["metrics"]
        m_te = p_te["metrics"]
        tr_s.append(stability_score(m_tr))
        te_s.append(stability_score(m_te))
        tr_metrics.append(m_tr)
        te_metrics.append(m_te)
    if not te_s:
        pack = evaluate_on_slice(p, slice(0, len(c)), o, h, l, c, tl, basis, oi, bar_type)
        sc = stability_score(pack["metrics"])
        return sc, [sc], [sc], {
            "walk_forward_efficiency": 0.0,
            "oos_degradation": 0.0,
            "hard_gate_passed": False,
            "hard_gate_reasons": ["no_walk_forward_splits"],
        }
    te_arr = np.array(te_s, dtype=np.float64)
    med = float(np.median(te_arr))
    std = float(np.std(te_arr)) if len(te_arr) > 1 else 0.0
    deg = 0.0
    if len(tr_s) == len(te_s):
        for a, b in zip(tr_s, te_s):
            if a > 0.1:
                deg += max(0.0, (a - b) / (abs(a) + 1e-9))
        deg /= len(tr_s)
    agg = med - 0.4 * std - 0.25 * deg
    if use_test_only:
        agg = med - 0.4 * std
    wfe = walk_forward_efficiency(tr_s, te_s)
    # Degradation measured on median fold stability.
    deg_ratio = oos_degradation_ratio(is_value=float(np.median(np.array(tr_s))), oos_value=med)
    gate_cfg = {
        "sharpe_min": 1.8,
        "profit_factor_min": 2.0,
        "max_drawdown_pct_max": 18.0,
        "trades_min": 300.0,
        "expectancy_min": 0.0,
        "wfe_min": 0.75,
        "oos_degradation_max": 0.25,
    }
    if hard_gates:
        gate_cfg.update({k: float(v) for k, v in hard_gates.items()})
    hard_reasons: List[str] = []
    test_m = te_metrics[-1] if te_metrics else None
    if test_m is None:
        hard_reasons.append("missing_test_metrics")
    else:
        if float(test_m.sharpe_ratio) < gate_cfg["sharpe_min"]:
            hard_reasons.append("sharpe_below_min")
        if float(test_m.profit_factor) < gate_cfg["profit_factor_min"]:
            hard_reasons.append("profit_factor_below_min")
        if float(test_m.max_drawdown_pct) > gate_cfg["max_drawdown_pct_max"]:
            hard_reasons.append("max_drawdown_above_max")
        if float(test_m.n_trades) < gate_cfg["trades_min"]:
            hard_reasons.append("trades_below_min")
        if float(test_m.expectancy) <= gate_cfg["expectancy_min"]:
            hard_reasons.append("expectancy_non_positive")
    if float(wfe) < gate_cfg["wfe_min"]:
        hard_reasons.append("wfe_below_min")
    if float(deg_ratio) > gate_cfg["oos_degradation_max"]:
        hard_reasons.append("oos_degradation_above_max")
    hard_pass = len(hard_reasons) == 0
    if not hard_pass:
        agg -= 5.0 + 0.25 * len(hard_reasons)
    return agg, tr_s, te_s, {
        "walk_forward_efficiency": float(wfe),
        "oos_degradation": float(deg_ratio),
        "hard_gate_passed": bool(hard_pass),
        "hard_gate_reasons": hard_reasons,
        "last_train_metrics": tr_metrics[-1].to_dict() if tr_metrics else {},
        "last_test_metrics": te_metrics[-1].to_dict() if te_metrics else {},
        "hard_gate_config": gate_cfg,
    }


def suggest_params_optuna(trial: Any) -> TtmOptParams:
    """Optuna Trial — categorical + float suggestions."""
    return TtmOptParams(
        bb_len=trial.suggest_categorical("bb_len", BB_LEN_GRID),
        bb_std=trial.suggest_categorical("bb_std", BB_STD_GRID),
        kc_len=trial.suggest_categorical("kc_len", KC_LEN_GRID),
        kc_atr_mult=trial.suggest_categorical("kc_atr_mult", KC_ATR_MULT_GRID),
        atr_len=trial.suggest_categorical("atr_len", ATR_REGIME_WIN_GRID),
        momentum_threshold=trial.suggest_float("mom_thr", -1.5, 1.5),
        htf_momentum_scale=trial.suggest_categorical("htf_scale", [1, 2, 4]),
        basis_threshold=trial.suggest_categorical("basis_thr", BASIS_THRESHOLD_GRID),
        basis_delta_window=trial.suggest_categorical("basis_d_w", BASIS_DELTA_WIN_GRID),
        basis_z_window=trial.suggest_categorical("basis_z_w", BASIS_Z_WIN_GRID),
        oi_ma_window=trial.suggest_categorical("oi_ma_w", OI_MA_WIN_GRID),
        oi_delta_window=trial.suggest_categorical("oi_d_w", OI_DELTA_WIN_GRID),
        oi_slope_window=trial.suggest_categorical("oi_sl_w", OI_SLOPE_WIN_GRID),
        atr_regime_window=trial.suggest_categorical("atr_reg_w", ATR_REGIME_WIN_GRID),
        atr_percentile_window=trial.suggest_int("atr_pct_w", 60, 200),
        adx_period=14,
        adx_trend_min=trial.suggest_categorical("adx_min", ADX_THRESHOLD_GRID),
        regime_filter=trial.suggest_categorical(
            "regime_filter",
            ["none", "trend_only", "range_only", "skip_high_vol", "skip_low_vol", "basis_oi_align"],
        ),
        stop_loss_pct=trial.suggest_categorical("sl", STOP_LOSS_PCT_GRID),
        take_profit_pct=trial.suggest_categorical("tp", TAKE_PROFIT_PCT_GRID),
        trailing_stop=trial.suggest_categorical("trail_on", [False, True]),
        trailing_pct=trial.suggest_categorical("trail_pct", TRAILING_PCT_GRID),
        commission_pct=0.0003,
        slippage_pct=trial.suggest_float("slip", 0.0001, 0.001, log=True),
        position_risk_pct=trial.suggest_categorical("risk_pct", POSITION_RISK_PCT_GRID),
        warmup_bars=120,
    )


def params_from_optuna_dict(d: Dict[str, Any]) -> TtmOptParams:
    return TtmOptParams(
        bb_len=int(d.get("bb_len", 20)),
        bb_std=float(d.get("bb_std", 2.0)),
        kc_len=int(d.get("kc_len", 20)),
        kc_atr_mult=float(d.get("kc_atr_mult", 1.5)),
        atr_len=int(d.get("atr_len", 14)),
        momentum_threshold=float(d.get("mom_thr", 0.0)),
        htf_momentum_scale=int(d.get("htf_scale", 1)),
        basis_threshold=float(d.get("basis_thr", 0.0)),
        basis_delta_window=int(d.get("basis_d_w", 5)),
        basis_z_window=int(d.get("basis_z_w", 20)),
        oi_ma_window=int(d.get("oi_ma_w", 20)),
        oi_delta_window=int(d.get("oi_d_w", 5)),
        oi_slope_window=int(d.get("oi_sl_w", 10)),
        atr_regime_window=int(d.get("atr_reg_w", 20)),
        atr_percentile_window=int(d.get("atr_pct_w", 100)),
        adx_trend_min=float(d.get("adx_min", 25.0)),
        regime_filter=str(d.get("regime_filter", "none")),
        stop_loss_pct=float(d.get("sl", 0.01)),
        take_profit_pct=float(d.get("tp", 0.02)),
        trailing_stop=bool(d.get("trail_on", False)),
        trailing_pct=float(d.get("trail_pct", 0.01)),
        slippage_pct=float(d.get("slip", 0.0003)),
        position_risk_pct=float(d.get("risk_pct", 0.02)),
    )


def random_params(rng: random.Random) -> TtmOptParams:
    return TtmOptParams(
        bb_len=rng.choice(BB_LEN_GRID),
        bb_std=rng.choice(BB_STD_GRID),
        kc_len=rng.choice(KC_LEN_GRID),
        kc_atr_mult=rng.choice(KC_ATR_MULT_GRID),
        atr_len=rng.choice(ATR_REGIME_WIN_GRID),
        momentum_threshold=rng.uniform(-1.5, 1.5),
        htf_momentum_scale=rng.choice([1, 2, 4]),
        basis_threshold=rng.choice(BASIS_THRESHOLD_GRID),
        basis_delta_window=rng.choice(BASIS_DELTA_WIN_GRID),
        basis_z_window=rng.choice(BASIS_Z_WIN_GRID),
        oi_ma_window=rng.choice(OI_MA_WIN_GRID),
        oi_delta_window=rng.choice(OI_DELTA_WIN_GRID),
        oi_slope_window=rng.choice(OI_SLOPE_WIN_GRID),
        atr_regime_window=rng.choice(ATR_REGIME_WIN_GRID),
        atr_percentile_window=rng.randint(60, 200),
        adx_trend_min=rng.choice(ADX_THRESHOLD_GRID),
        regime_filter=rng.choice(
            ["none", "trend_only", "range_only", "skip_high_vol", "skip_low_vol", "basis_oi_align"]
        ),
        stop_loss_pct=rng.choice(STOP_LOSS_PCT_GRID),
        take_profit_pct=rng.choice(TAKE_PROFIT_PCT_GRID),
        trailing_stop=rng.choice([True, False]),
        trailing_pct=rng.choice(TRAILING_PCT_GRID),
        slippage_pct=10 ** rng.uniform(-4, -3),
        position_risk_pct=rng.choice(POSITION_RISK_PCT_GRID),
    )


def run_optimization(
    bars: Sequence[OhlcBar],
    basis: np.ndarray,
    oi: np.ndarray,
    *,
    bar_type: str,
    resolution: str,
    n_trials: int = 40,
    seed: int = 42,
    train_months: float = 6.0,
    test_months: float = 2.0,
    use_optuna: bool = True,
    hard_gates: Dict[str, float] | None = None,
) -> List[Tuple[float, TtmOptParams, Dict[str, Any]]]:
    """
    Returns sorted list (desc): (aggregate_score, params, meta with fold scores).
    """
    o, h, l, c, tl = bars_to_arrays(bars)
    n = len(c)
    if len(basis) != n:
        basis = np.zeros(n, dtype=np.float64)
    if len(oi) != n:
        oi = np.zeros(n, dtype=np.float64)

    splits = walk_forward_splits(bars, train_months=train_months, test_months=test_months, resolution=resolution)
    results: List[Tuple[float, TtmOptParams, Dict[str, Any]]] = []

    if use_optuna:
        try:
            import optuna
        except ImportError:
            use_optuna = False

    if use_optuna:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def obj(trial: Any) -> float:
            p = suggest_params_optuna(trial)
            agg, tr_l, te_l, diag = walk_forward_objective(
                p, splits, o, h, l, c, tl, basis, oi, bar_type, hard_gates=hard_gates
            )
            trial.set_user_attr("train_scores", tr_l)
            trial.set_user_attr("test_scores", te_l)
            trial.set_user_attr("walk_forward_diag", diag)
            return agg

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
                continue
            p = params_from_optuna_dict(dict(t.params))
            results.append(
                (
                    float(t.value),
                    p,
                    {
                        "train_scores": t.user_attrs.get("train_scores", []),
                        "test_scores": t.user_attrs.get("test_scores", []),
                        "walk_forward_diag": t.user_attrs.get("walk_forward_diag", {}),
                    },
                )
            )
    else:
        rng = random.Random(seed)
        for _ in range(n_trials):
            p = random_params(rng)
            agg, tr_l, te_l, diag = walk_forward_objective(
                p, splits, o, h, l, c, tl, basis, oi, bar_type, hard_gates=hard_gates
            )
            results.append((agg, p, {"train_scores": tr_l, "test_scores": te_l, "walk_forward_diag": diag}))

    results.sort(key=lambda x: -x[0])

    seen: set = set()
    deduped: List[Tuple[float, TtmOptParams, Dict[str, Any]]] = []
    for item in results:
        key = json.dumps(item[1].to_dict(), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
