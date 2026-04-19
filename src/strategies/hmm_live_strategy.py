"""HMM live stack uses :class:`~src.live.hmm_live_runner.HmmLiveRunner`, not ``generate_signal``.

This class exists so :func:`src.strategy_factory.get_strategy` can return a stable type for
``STRATEGY_ALGO=HMM`` without importing anything under ``src.hmm`` here.
"""

from __future__ import annotations

from typing import Any, Dict

from src.strategies.base_strategy import BaseStrategy


class HMMStrategy(BaseStrategy):
    """Placeholder for routing; live execution stays in ``HmmLiveRunner.run_once``."""

    def generate_signal(self, state: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "HMM live is driven by HmmLiveRunner.run_once(), not generate_signal(). "
            "Use scripts/hmm_live.py with STRATEGY_ALGO=HMM."
        )
