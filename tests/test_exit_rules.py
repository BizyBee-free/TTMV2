"""Unit tests for config-driven exit rules."""

import numpy as np

from src.strategies.exit_rules import (
    compute_expected_return_one_hot,
    expected_return_exit,
    exit_config_from_settings,
    risk_exit,
    should_exit,
    state_flip_exit,
    time_exit,
)
from src.config import Settings


def _settings(**overrides) -> Settings:
    base = {
        "DNSE_API_KEY": "k",
        "DNSE_API_SECRET": "s",
        "DNSE_ACCOUNT_NO": "0001",
    }
    base.update(overrides)
    return Settings(**base)


def test_exit_config_from_settings():
    cfg = _settings(
        MCMC_STOP_LOSS_POINTS=2.5,
        MCMC_TAKE_PROFIT_POINTS=5.0,
        MCMC_MAX_BARS_IN_TRADE=8,
        MCMC_STATE_FLIP_THRESHOLD=0.6,
        MCMC_EXPECTED_RETURN_THRESHOLD=0.1,
        MCMC_TRANSACTION_COST_POINTS=0.2,
        MCMC_EXIT_BULLISH_STATES="2, 3",
        MCMC_EXIT_BEARISH_STATES="[0, 1]",
    )
    d = exit_config_from_settings(cfg)
    assert d["stop_loss_points"] == 2.5
    assert d["bullish_states"] == [2, 3]
    assert d["bearish_states"] == [0, 1]


def test_risk_exit_long_short_same_pnl_rule():
    cfg = {
        "stop_loss_points": 2.5,
        "take_profit_points": 5.0,
        "max_bars_in_trade": 99,
        "state_flip_threshold": 1.0,
        "expected_return_threshold": 0.0,
        "transaction_cost_points": 0.0,
        "bullish_states": [1],
        "bearish_states": [0],
    }
    st = {"pnl_points": -3.0, "posterior": np.array([0.5, 0.5]), "bars_in_trade": 0}
    assert risk_exit({"direction": "LONG"}, st, cfg)[0] is True
    assert risk_exit({"direction": "SHORT"}, st, cfg)[0] is True


def test_state_flip_long_bearish():
    cfg = {
        "stop_loss_points": 1e9,
        "take_profit_points": 1e9,
        "max_bars_in_trade": 999,
        "state_flip_threshold": 0.6,
        "expected_return_threshold": 0.0,
        "transaction_cost_points": 0.0,
        "bullish_states": [1],
        "bearish_states": [0],
    }
    st = {"pnl_points": 0.0, "posterior": np.array([0.7, 0.3]), "bars_in_trade": 0}
    hit, reason = state_flip_exit({"direction": "LONG"}, st, cfg)
    assert hit is True
    assert reason == "state_flip_bearish"


def test_state_flip_short_bullish():
    cfg = {
        "stop_loss_points": 1e9,
        "take_profit_points": 1e9,
        "max_bars_in_trade": 999,
        "state_flip_threshold": 0.6,
        "expected_return_threshold": 0.0,
        "transaction_cost_points": 0.0,
        "bullish_states": [1],
        "bearish_states": [0],
    }
    st = {"pnl_points": 0.0, "posterior": np.array([0.3, 0.7]), "bars_in_trade": 0}
    hit, reason = state_flip_exit({"direction": "SHORT"}, st, cfg)
    assert hit is True
    assert reason == "state_flip_bullish"


def test_expected_return_long():
    cfg = {
        "stop_loss_points": 1e9,
        "take_profit_points": 1e9,
        "max_bars_in_trade": 999,
        "state_flip_threshold": 1.0,
        "expected_return_threshold": 0.1,
        "transaction_cost_points": 0.2,
        "bullish_states": [1],
        "bearish_states": [0],
    }
    A = np.array([[0.5, 0.5], [0.4, 0.6]], dtype=float)
    mu = np.array([0.0, 0.0], dtype=float)
    st = {
        "pnl_points": 0.0,
        "posterior": np.array([0.5, 0.5]),
        "transition_matrix": A,
        "state_returns": mu,
        "bars_in_trade": 0,
        "last_state_index": 0,
    }
    hit, reason = expected_return_exit({"direction": "LONG"}, st, cfg)
    assert hit is True
    assert reason == "expected_return_long"


def test_expected_return_short():
    cfg = {
        "stop_loss_points": 1e9,
        "take_profit_points": 1e9,
        "max_bars_in_trade": 999,
        "state_flip_threshold": 1.0,
        "expected_return_threshold": 0.1,
        "transaction_cost_points": 0.2,
        "bullish_states": [1],
        "bearish_states": [0],
    }
    A = np.array([[0.5, 0.5], [0.4, 0.6]], dtype=float)
    mu = np.array([0.5, 0.5], dtype=float)
    st = {
        "pnl_points": 0.0,
        "posterior": np.array([0.5, 0.5]),
        "transition_matrix": A,
        "state_returns": mu,
        "bars_in_trade": 0,
        "last_state_index": 0,
    }
    hit, reason = expected_return_exit({"direction": "SHORT"}, st, cfg)
    assert hit is True
    assert reason == "expected_return_short"


def test_compute_expected_return_one_hot():
    A = np.array([[0.2, 0.8], [0.3, 0.7]], dtype=float)
    mu = np.array([0.01, -0.02], dtype=float)
    e = compute_expected_return_one_hot(0, A, mu)
    assert abs(e - (0.2 * 0.01 + 0.8 * (-0.02))) < 1e-9


def test_time_exit():
    cfg = {
        "stop_loss_points": 1e9,
        "take_profit_points": 1e9,
        "max_bars_in_trade": 8,
        "state_flip_threshold": 1.0,
        "expected_return_threshold": 0.0,
        "transaction_cost_points": 0.0,
        "bullish_states": [1],
        "bearish_states": [0],
    }
    st = {"pnl_points": 0.0, "bars_in_trade": 8}
    hit, reason = time_exit({"direction": "LONG"}, st, cfg)
    assert hit is True
    assert reason == "time_max_bars"


def test_should_exit_priority_risk_before_flip():
    cfg = {
        "stop_loss_points": 1.0,
        "take_profit_points": 100.0,
        "max_bars_in_trade": 1,
        "state_flip_threshold": 0.0,
        "expected_return_threshold": 0.0,
        "transaction_cost_points": 0.0,
        "bullish_states": [1],
        "bearish_states": [0],
    }
    st = {
        "pnl_points": -5.0,
        "posterior": np.array([1.0, 0.0]),
        "transition_matrix": np.eye(2),
        "state_returns": np.zeros(2),
        "bars_in_trade": 999,
        "last_state_index": 0,
    }
    out = should_exit({"direction": "LONG"}, st, cfg)
    assert out["exit"] is True
    assert out["reason"] == "risk_stop_loss"


def test_should_exit_priority_flip_before_time():
    cfg = {
        "stop_loss_points": 1e9,
        "take_profit_points": 1e9,
        "max_bars_in_trade": 1,
        "state_flip_threshold": 0.5,
        "expected_return_threshold": 0.0,
        "transaction_cost_points": 0.0,
        "bullish_states": [1],
        "bearish_states": [0],
    }
    st = {
        "pnl_points": 0.0,
        "posterior": np.array([0.9, 0.1]),
        "transition_matrix": np.eye(2),
        "state_returns": np.zeros(2),
        "bars_in_trade": 999,
        "last_state_index": 0,
    }
    out = should_exit({"direction": "LONG"}, st, cfg)
    assert out["exit"] is True
    assert out["reason"] == "state_flip_bearish"
