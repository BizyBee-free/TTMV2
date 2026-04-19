"""Select signal strategy from ``STRATEGY_ALGO`` (.env / Settings)."""

from __future__ import annotations

from typing import Optional

from src.config import Settings, StrategyAlgo, get_settings
from src.strategies.base_strategy import BaseStrategy
from src.strategies.hmm_live_strategy import HMMStrategy
from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_strategy import TTMStrategy


def get_strategy(settings: Optional[Settings] = None) -> BaseStrategy:
    """
    ``STRATEGY_ALGO`` is read from Settings (``.env``).

    - ``TTM`` → :class:`~src.strategies.ttm.ttm_strategy.TTMStrategy`
    - ``HMM`` → :class:`~src.strategies.hmm_live_strategy.HMMStrategy` (live uses runner)
    """
    s = settings or get_settings()
    algo = s.STRATEGY_ALGO
    if algo == StrategyAlgo.TTM:
        return TTMStrategy(TTM_CONFIG)
    if algo == StrategyAlgo.HMM:
        return HMMStrategy()
    raise ValueError(f"Unsupported STRATEGY_ALGO for get_strategy(): {algo!r}")


def get_strategy_algo_name(settings: Optional[Settings] = None) -> str:
    """Normalized algo string for branching in scripts."""
    s = settings or get_settings()
    v = s.STRATEGY_ALGO
    if isinstance(v, StrategyAlgo):
        return v.value
    return str(v).strip().upper()


def live_session_logger_name(settings: Optional[Settings] = None) -> str:
    """Structured log name, e.g. ``ttm_live_session`` / ``hmm_live_session`` (matches STRATEGY_ALGO)."""
    s = settings or get_settings()
    return f"{s.STRATEGY_ALGO.value.lower()}_live_session"


def live_session_cli_prefix(settings: Optional[Settings] = None) -> str:
    """Console bracket tag aligned with :func:`live_session_logger_name`, e.g. ``[ttm_live_session]``."""
    return f"[{live_session_logger_name(settings)}]"


def live_15m_cli_prefix(settings: Optional[Settings] = None) -> str:
    """One-shot 15m runner tag, e.g. ``[ttm_live]`` / ``[hmm_live]``."""
    s = settings or get_settings()
    return f"[{s.STRATEGY_ALGO.value.lower()}_live]"


def live_15m_logger_name(settings: Optional[Settings] = None) -> str:
    """Logger name for ``scripts/hmm_live.py`` one-shot, e.g. ``ttm_live`` / ``hmm_live``."""
    s = settings or get_settings()
    return f"{s.STRATEGY_ALGO.value.lower()}_live"
