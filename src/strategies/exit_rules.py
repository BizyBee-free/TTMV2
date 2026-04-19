"""Config-driven derivative exit rules (LONG/SHORT).

Priority: risk_exit → state_flip_exit → expected_return_exit → time_exit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple, Union

import numpy as np

from src.config import parse_int_list

ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...]]


def _as_float_array(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


def _group_mass(posterior: np.ndarray, indices: List[int]) -> float:
    if posterior.size == 0 or not indices:
        return 0.0
    p = _as_float_array(posterior)
    valid = [i for i in indices if 0 <= i < p.size]
    if not valid:
        return 0.0
    return float(np.sum(p[valid]))


def compute_expected_return_one_hot(
    last_state_index: int,
    transition_matrix: np.ndarray,
    state_returns: np.ndarray,
) -> float:
    """E[r_{t+1}] = sum_i P(S_t=i) * sum_j A_ij * mu_j with P(S_t) = one-hot(last)."""
    A = np.asarray(transition_matrix, dtype=np.float64)
    mu = np.asarray(state_returns, dtype=np.float64).ravel()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return float("nan")
    if mu.size != A.shape[1]:
        return float("nan")
    if last_state_index < 0 or last_state_index >= A.shape[0]:
        return float("nan")
    row = A[int(last_state_index)]
    return float(np.dot(row, mu))


def exit_config_from_settings(settings: Any) -> Dict[str, Any]:
    """Build exit config dict from application Settings."""
    return {
        "stop_loss_points": float(settings.MCMC_STOP_LOSS_POINTS),
        "take_profit_points": float(settings.MCMC_TAKE_PROFIT_POINTS),
        "max_bars_in_trade": int(settings.MCMC_MAX_BARS_IN_TRADE),
        "state_flip_threshold": float(settings.MCMC_STATE_FLIP_THRESHOLD),
        "expected_return_threshold": float(settings.MCMC_EXPECTED_RETURN_THRESHOLD),
        "transaction_cost_points": float(settings.MCMC_TRANSACTION_COST_POINTS),
        "bullish_states": parse_int_list(settings.MCMC_EXIT_BULLISH_STATES),
        "bearish_states": parse_int_list(settings.MCMC_EXIT_BEARISH_STATES),
    }


def risk_exit(
    position: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Stop / take-profit in signed points (same rule for LONG and SHORT)."""
    pnl = float(state.get("pnl_points", 0.0))
    sl = float(config["stop_loss_points"])
    tp = float(config["take_profit_points"])
    if pnl <= -sl:
        return True, "risk_stop_loss"
    if pnl >= tp:
        return True, "risk_take_profit"
    return False, ""


def state_flip_exit(
    position: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[bool, str]:
    """LONG: exit if mass on bearish states > threshold; SHORT: bullish mass."""
    direction = str(position.get("direction", "")).upper()
    thr = float(config["state_flip_threshold"])
    post = state.get("posterior")
    if post is None:
        return False, ""
    p = _as_float_array(post)
    bearish = [int(x) for x in config.get("bearish_states", [])]
    bullish = [int(x) for x in config.get("bullish_states", [])]

    if direction == "LONG":
        mass = _group_mass(p, bearish)
        if mass > thr:
            return True, "state_flip_bearish"
    elif direction == "SHORT":
        mass = _group_mass(p, bullish)
        if mass > thr:
            return True, "state_flip_bullish"
    return False, ""


def expected_return_exit(
    position: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Compare (E[r_{t+1}] - transaction_cost) to threshold; LONG vs SHORT."""
    direction = str(position.get("direction", "")).upper()
    tc = float(config["transaction_cost_points"])
    thr = float(config["expected_return_threshold"])

    last_idx = state.get("last_state_index")
    A = state.get("transition_matrix")
    mu = state.get("state_returns")
    if last_idx is None or A is None or mu is None:
        return False, ""

    E = compute_expected_return_one_hot(int(last_idx), A, mu)
    if np.isnan(E):
        return False, ""

    adj = E - tc
    if direction == "LONG":
        if adj < thr:
            return True, "expected_return_long"
    elif direction == "SHORT":
        if adj > -thr:
            return True, "expected_return_short"
    return False, ""


def time_exit(
    position: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[bool, str]:
    bars = int(state.get("bars_in_trade", 0))
    mx = int(config["max_bars_in_trade"])
    if bars >= mx:
        return True, "time_max_bars"
    return False, ""


def should_exit(
    position: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return exit decision with priority: risk → flip → expected return → time."""
    out: Dict[str, Any] = {"exit": False, "reason": ""}

    E = float("nan")
    last_idx = state.get("last_state_index")
    A = state.get("transition_matrix")
    mu = state.get("state_returns")
    if last_idx is not None and A is not None and mu is not None:
        E = compute_expected_return_one_hot(int(last_idx), A, mu)

    for check in (risk_exit, state_flip_exit, expected_return_exit, time_exit):
        hit, reason = check(position, state, config)
        if hit:
            out["exit"] = True
            out["reason"] = reason
            break

    out["expected_return"] = E
    post = state.get("posterior")
    out["state_probabilities"] = (
        _as_float_array(post).tolist() if post is not None else []
    )
    return out
