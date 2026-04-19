"""
Unit tests for OnlineLearner: sample-size gate, correlation sign, delta cap, determinism.

Synthetic TradeRecords only; fail with explicit messages on violation.
"""

from __future__ import annotations

import copy

from src.strategies.ttm.ttm_online_learning import OnlineLearner, RegimeWeightState
from src.strategies.ttm.ttm_trade_logger import TradeRecord
from src.strategies.ttm.ttm_weights import RegimeWeights, WeightSet

REGIME = 0

_BASE_CFG = {
    "ttm_adaptive_min_trades": 30,
    "ttm_adaptive_alpha": 0.1,
    "ttm_adaptive_max_delta": 0.2,
    "ttm_adaptive_pnl_variance_max": 1e12,
    "ttm_adaptive_rollback_eps": 1e-9,
}


def _uniform_weights() -> RegimeWeights:
    u = WeightSet(1.0, 1.0, 1.0)
    return RegimeWeights(weights={-1: u, 0: u, 1: u})


def _record(
    *,
    pnl: float,
    momentum: float,
    basis_effect: float = 0.0,
    oi_effect: float = 0.0,
    regime: int = REGIME,
) -> TradeRecord:
    return TradeRecord(
        entry_time="",
        exit_time="",
        side="LONG",
        entry_price=100.0,
        exit_price=101.0,
        pnl=float(pnl),
        holding_bars=1,
        features_at_entry={
            "momentum": float(momentum),
            "basis_effect": float(basis_effect),
            "oi_effect": float(oi_effect),
        },
        score_long=0.0,
        score_short=0.0,
        prob_long=0.5,
        prob_short=0.5,
        regime=regime,
    )


def test_online_learner_no_update_when_trades_less_than_30() -> None:
    """1. No weight change when regime sample size < min_trades (30)."""
    cfg = {**_BASE_CFG, "ttm_adaptive_min_trades": 30}
    w0 = _uniform_weights()
    st = RegimeWeightState()
    learner = OnlineLearner()
    n = 29
    trades = [_record(pnl=float(i), momentum=float(i) / 29.0) for i in range(n)]
    w1, dbg = learner.update(copy.deepcopy(w0), trades, cfg, st)

    assert any(
        s.get("reason") == "min_trades" and s.get("n") == n for s in dbg.get("skipped", [])
    ), (
        "FAIL: expected skip with reason=min_trades and n=29 in debug['skipped']; "
        f"got skipped={dbg.get('skipped')}"
    )
    assert w1.weights[REGIME].momentum == w0.weights[REGIME].momentum, (
        "FAIL: momentum weight must not change when trades < 30; "
        f"before={w0.weights[REGIME].momentum} after={w1.weights[REGIME].momentum}"
    )
    assert w1.weights[REGIME].basis == w0.weights[REGIME].basis, (
        "FAIL: basis weight must not change when trades < 30"
    )
    assert w1.weights[REGIME].oi == w0.weights[REGIME].oi, (
        "FAIL: oi weight must not change when trades < 30"
    )


def test_online_learner_positive_correlation_increases_momentum_weight() -> None:
    """2. Momentum weight increases when momentum feature is positively correlated with pnl."""
    cfg = {**_BASE_CFG, "ttm_adaptive_min_trades": 30, "ttm_adaptive_alpha": 0.1}
    w0 = _uniform_weights()
    st = RegimeWeightState()
    learner = OnlineLearner()
    n = 30
    # Perfect positive association: larger momentum -> larger pnl; basis/oi held constant -> corr 0 for those dims
    trades = [
        _record(pnl=float(i), momentum=float(i) / max(n - 1, 1), basis_effect=0.0, oi_effect=0.0)
        for i in range(n)
    ]
    old_m = w0.weights[REGIME].momentum
    w1, dbg = learner.update(copy.deepcopy(w0), trades, cfg, st)

    assert REGIME in dbg.get("regimes", {}), (
        f"FAIL: expected update for regime {REGIME}; debug={dbg}"
    )
    assert dbg["regimes"][REGIME].get("rolled_back") is False, (
        f"FAIL: unexpected rollback on first update; {dbg['regimes'][REGIME]}"
    )
    new_m = w1.weights[REGIME].momentum
    assert new_m > old_m, (
        "FAIL: positive correlation with pnl must increase momentum weight; "
        f"old_m={old_m} new_m={new_m} debug={dbg['regimes'][REGIME]}"
    )
    # Pearson ~1 -> delta_m = alpha * 1 = 0.1 (within cap 0.2)
    d0 = dbg["regimes"][REGIME]["delta"][0]
    assert abs(d0 - 0.1) < 1e-6, (
        f"FAIL: expected momentum delta ~0.1 for corr≈1 and alpha=0.1; got delta[0]={d0}"
    )
    assert abs(new_m - (old_m + d0)) < 1e-9, (
        f"FAIL: new momentum should equal old + delta; old={old_m} delta={d0} new={new_m}"
    )


def test_online_learner_negative_correlation_decreases_momentum_weight() -> None:
    """3. Momentum weight decreases when momentum feature is negatively correlated with pnl."""
    cfg = {**_BASE_CFG, "ttm_adaptive_min_trades": 30, "ttm_adaptive_alpha": 0.1}
    w0 = _uniform_weights()
    st = RegimeWeightState()
    learner = OnlineLearner()
    n = 30
    # Larger momentum -> smaller pnl (perfect negative association on momentum dimension)
    trades = [
        _record(pnl=-float(i), momentum=float(i) / max(n - 1, 1), basis_effect=0.0, oi_effect=0.0)
        for i in range(n)
    ]
    old_m = w0.weights[REGIME].momentum
    w1, dbg = learner.update(copy.deepcopy(w0), trades, cfg, st)

    assert REGIME in dbg.get("regimes", {}), f"FAIL: expected regime update; debug={dbg}"
    new_m = w1.weights[REGIME].momentum
    assert new_m < old_m, (
        "FAIL: negative correlation with pnl must decrease momentum weight; "
        f"old_m={old_m} new_m={new_m} debug={dbg['regimes'][REGIME]}"
    )
    d0 = dbg["regimes"][REGIME]["delta"][0]
    assert d0 < 0, f"FAIL: expected negative momentum delta; got delta[0]={d0}"
    assert abs(d0 + 0.1) < 1e-6, (
        f"FAIL: expected momentum delta ~-0.1 for corr≈-1 and alpha=0.1; got {d0}"
    )


def test_online_learner_delta_bounded_by_max_delta() -> None:
    """4. Per-dimension |delta| must not exceed ttm_adaptive_max_delta (0.2)."""
    cfg = {
        **_BASE_CFG,
        "ttm_adaptive_min_trades": 30,
        "ttm_adaptive_alpha": 10.0,
        "ttm_adaptive_max_delta": 0.2,
    }
    w0 = _uniform_weights()
    st = RegimeWeightState()
    learner = OnlineLearner()
    n = 30
    trades = [
        _record(pnl=float(i), momentum=float(i) / max(n - 1, 1), basis_effect=0.0, oi_effect=0.0)
        for i in range(n)
    ]
    _, dbg = learner.update(copy.deepcopy(w0), trades, cfg, st)

    assert REGIME in dbg.get("regimes", {}), f"FAIL: no regime update; dbg={dbg}"
    deltas = dbg["regimes"][REGIME]["delta"]
    for name, d in zip(("momentum", "basis", "oi"), deltas):
        assert abs(d) <= 0.2 + 1e-9, (
            f"FAIL: |delta| must be <= 0.2 for {name}; delta={d} all_deltas={deltas}"
        )


def test_online_learner_deterministic() -> None:
    """5. Same weights + same trades + fresh state -> identical output weights."""
    cfg = {**_BASE_CFG, "ttm_adaptive_min_trades": 30, "ttm_adaptive_alpha": 0.1}
    w0 = _uniform_weights()
    learner = OnlineLearner()
    n = 30
    trades = [
        _record(pnl=float(i), momentum=float(i) / max(n - 1, 1), basis_effect=0.0, oi_effect=0.0)
        for i in range(n)
    ]

    w_a, dbg_a = learner.update(copy.deepcopy(w0), trades, cfg, RegimeWeightState())
    w_b, dbg_b = learner.update(copy.deepcopy(w0), trades, cfg, RegimeWeightState())

    for r in (-1, 0, 1):
        assert w_a.weights[r].momentum == w_b.weights[r].momentum, (
            f"FAIL: deterministic mismatch momentum regime {r}: {w_a.weights[r].momentum} vs {w_b.weights[r].momentum}"
        )
        assert w_a.weights[r].basis == w_b.weights[r].basis, (
            f"FAIL: deterministic mismatch basis regime {r}"
        )
        assert w_a.weights[r].oi == w_b.weights[r].oi, (
            f"FAIL: deterministic mismatch oi regime {r}"
        )

    assert dbg_a == dbg_b, (
        f"FAIL: debug dict must match for identical inputs; diff keys a={dbg_a.keys()} b={dbg_b.keys()}"
    )


def test_online_learner_max_jump_applies_even_with_high_alpha() -> None:
    """Extra: weight step magnitude equals max_delta when alpha * |corr| would exceed cap."""
    cfg = {
        **_BASE_CFG,
        "ttm_adaptive_min_trades": 30,
        "ttm_adaptive_alpha": 10.0,
        "ttm_adaptive_max_delta": 0.2,
    }
    w0 = _uniform_weights()
    st = RegimeWeightState()
    learner = OnlineLearner()
    n = 30
    trades = [
        _record(pnl=float(i), momentum=float(i) / max(n - 1, 1), basis_effect=0.0, oi_effect=0.0)
        for i in range(n)
    ]
    old_m = w0.weights[REGIME].momentum
    w1, dbg = learner.update(copy.deepcopy(w0), trades, cfg, st)
    new_m = w1.weights[REGIME].momentum
    d0 = dbg["regimes"][REGIME]["delta"][0]
    assert abs(d0 - 0.2) < 1e-9, (
        f"FAIL: with corr≈1 and alpha=10, delta must clip to 0.2; got delta[0]={d0}"
    )
    assert abs(new_m - old_m - 0.2) < 1e-9, (
        f"FAIL: momentum should move by exactly max_delta; old={old_m} new={new_m}"
    )
