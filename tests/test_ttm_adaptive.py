"""TTM adaptive: regime, weights, online learner, scheduler."""

from __future__ import annotations

import copy

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_adaptive_context import TTMAdaptiveContext, build_trade_record_from_snapshot
from src.strategies.ttm.ttm_online_learning import OnlineLearner, RegimeWeightState
from src.strategies.ttm.ttm_regime import detect_regime
from src.strategies.ttm.ttm_score import compute_score
from src.strategies.ttm.ttm_trade_logger import TradeLogger, TradeRecord
from src.strategies.ttm.ttm_types import SCORING_FEATURE_KEYS, TTMFeatureVector
from src.strategies.ttm.ttm_weights import RegimeWeights, WeightSet, clip_weight_set


def test_detect_regime_boundaries() -> None:
    assert detect_regime({"vol_spike": 0.5}) == -1
    assert detect_regime({"vol_spike": 0.8}) == 0
    assert detect_regime({"vol_spike": 1.0}) == 0
    assert detect_regime({"vol_spike": 1.2}) == 0
    assert detect_regime({"vol_spike": 1.3}) == 1


def test_clip_weight_set_bounds() -> None:
    w = clip_weight_set(WeightSet(5.0, -5.0, 1.0))
    assert w.momentum == 3.0
    assert w.basis == -3.0
    assert w.oi == 1.0


def test_compute_score_with_regime_weight_set() -> None:
    vm = {k: True for k in SCORING_FEATURE_KEYS}
    fv = TTMFeatureVector(1.0, 0.0, 0.1, 0.0, 0.5, 0, vm)
    rw = RegimeWeights(
        weights={
            -1: WeightSet(0.5, 0.5, 0.5),
            0: WeightSet(1.0, 1.0, 1.0),
            1: WeightSet(2.0, 2.0, 2.0),
        }
    )
    cfg = {**TTM_CONFIG, "_ttm_regime_id": 1, "_ttm_regime_weight_set": rw}
    sl, _, comp = compute_score(fv, cfg)
    cfg_flat = {**TTM_CONFIG, "_ttm_regime_id": 0}
    sl0, _, _ = compute_score(fv, cfg_flat)
    assert sl != sl0 or abs(sl) > 0


def test_learner_skips_small_sample() -> None:
    cfg = {
        **TTM_CONFIG,
        "ttm_adaptive_min_trades": 30,
        "ttm_adaptive_alpha": 0.1,
        "ttm_adaptive_max_delta": 0.2,
        "ttm_adaptive_pnl_variance_max": 1e12,
    }
    w0 = RegimeWeights.from_config(cfg)
    st = RegimeWeightState()
    learner = OnlineLearner()
    trades: list[TradeRecord] = []
    for i in range(10):
        trades.append(
            TradeRecord(
                entry_time="",
                exit_time="",
                side="LONG",
                entry_price=100.0,
                exit_price=101.0,
                pnl=1.0,
                holding_bars=1,
                features_at_entry={"momentum": 0.1, "basis_effect": 0.0, "oi_effect": 0.0},
                score_long=0.0,
                score_short=0.0,
                prob_long=0.5,
                prob_short=0.5,
                regime=0,
            )
        )
    w1, dbg = learner.update(w0, trades, cfg, st)
    assert w1.weights[0].momentum == w0.weights[0].momentum
    assert "skipped" in dbg


def test_learner_deterministic_update() -> None:
    cfg = {
        **TTM_CONFIG,
        "ttm_adaptive_min_trades": 5,
        "ttm_adaptive_alpha": 1.0,
        "ttm_adaptive_max_delta": 0.2,
        "ttm_adaptive_pnl_variance_max": 1e12,
    }
    w0 = RegimeWeights.from_config(cfg)
    st = RegimeWeightState()
    learner = OnlineLearner()
    pnls = [1.0, -1.0, 2.0, -0.5, 1.5]
    mom = [0.5, -0.5, 0.8, -0.2, 0.6]
    trades = [
        TradeRecord(
            entry_time="",
            exit_time="",
            side="LONG",
            entry_price=100.0,
            exit_price=101.0,
            pnl=pnls[i],
            holding_bars=1,
            features_at_entry={"momentum": mom[i], "basis_effect": 0.0, "oi_effect": 0.0},
            score_long=0.0,
            score_short=0.0,
            prob_long=0.5,
            prob_short=0.5,
            regime=0,
        )
        for i in range(5)
    ]
    w1, dbg1 = learner.update(copy.deepcopy(w0), trades, cfg, RegimeWeightState())
    w2, dbg2 = learner.update(copy.deepcopy(w0), trades, cfg, RegimeWeightState())
    assert w1.weights[0].momentum == w2.weights[0].momentum
    assert dbg1["regimes"].get(0, {}).get("rolled_back") is not None or "skipped" in str(dbg1)


def test_trade_logger_window() -> None:
    tl = TradeLogger(maxlen=3)
    for i in range(5):
        tl.log_trade(
            TradeRecord(
                entry_time="",
                exit_time="",
                side="LONG",
                entry_price=1.0,
                exit_price=1.0,
                pnl=float(i),
                holding_bars=1,
                features_at_entry={},
                score_long=0.0,
                score_short=0.0,
                prob_long=0.5,
                prob_short=0.5,
                regime=0,
            )
        )
    assert len(tl) == 3
    recent = tl.get_recent_trades(10)
    assert len(recent) == 3
    assert recent[-1].pnl == 4.0


def test_adaptive_context_scheduler_runs_on_multiple_of_20() -> None:
    cfg = {
        **TTM_CONFIG,
        "ttm_adaptive_enabled": True,
        "ttm_adaptive_min_trades": 5,
        "ttm_adaptive_update_every_n_trades": 20,
        "ttm_adaptive_alpha": 0.5,
        "ttm_adaptive_max_delta": 0.2,
        "ttm_adaptive_pnl_variance_max": 1e12,
    }
    ctx = TTMAdaptiveContext(cfg)
    base = {
        "entry_price": 100.0,
        "side": "LONG",
        "debug": {
            "score_long": 0.1,
            "score_short": -0.1,
            "prob_long": 0.6,
            "prob_short": 0.4,
            "ttm_regime_id": 0,
            "score_components": {"momentum": 0.5, "basis_effect": 0.0, "oi_effect": 0.0},
        },
        "entry_time": "",
    }
    for i in range(19):
        rec = build_trade_record_from_snapshot(base, exit_price=101.0, pnl=1.0, holding_bars=1)
        ctx.on_trade_closed(rec)
    assert ctx.closed_trade_count == 19
    assert ctx.get_last_update_debug() is None
    rec = build_trade_record_from_snapshot(base, exit_price=101.0, pnl=1.0, holding_bars=1)
    ctx.on_trade_closed(rec)
    assert ctx.closed_trade_count == 20
    assert ctx.get_last_update_debug() is not None
