"""Mutable adaptive state: trade log, learned weights, config overlay for TTM V2.

Independent from empirical bin curves (:mod:`src.strategies.ttm.empirical`): adaptive context
updates regime weights from **closed-trade PnL**; empirical engine fits **bar forward returns**.
Both may be enabled via separate config flags; watch for overlapping effects if blending scores.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional

from src.logger import get_logger
from src.strategies.ttm.config import is_ttm_adaptive_learning_enabled
from src.strategies.ttm.ttm_online_learning import OnlineLearner, RegimeWeightState
from src.strategies.ttm.ttm_trade_logger import TradeLogger, TradeRecord
from src.strategies.ttm.ttm_weights import RegimeWeights

logger = get_logger("ttm_adaptive")


class TTMAdaptiveContext:
    """
    Shared with :class:`~src.strategies.ttm.ttm_strategy.TTMStrategy` via ``adaptive_context``.

    ``get_config_overlay()`` is merged into the signal config before scoring (V2 uses
    :func:`~src.strategies.ttm.ttm_score.compute_score_v2_alpha`; legacy helpers may use
    :func:`~src.strategies.ttm.ttm_score.compute_score`).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config_ref = config
        self._weights = RegimeWeights.from_config(config)
        self._trade_logger = TradeLogger(maxlen=int(config.get("ttm_trade_log_maxlen", 100)))
        self._learner = OnlineLearner()
        self._state = RegimeWeightState()
        self._closed_trade_count = 0
        self._last_update_debug: Optional[Dict[str, Any]] = None

    @property
    def regime_weights(self) -> RegimeWeights:
        return self._weights

    @property
    def closed_trade_count(self) -> int:
        return self._closed_trade_count

    def get_config_overlay(self) -> Dict[str, Any]:
        if not is_ttm_adaptive_learning_enabled(self._config_ref):
            return {}
        return {"_ttm_regime_weight_set": self._weights}

    def get_last_update_debug(self) -> Optional[Dict[str, Any]]:
        return self._last_update_debug

    def on_trade_closed(self, record: TradeRecord) -> None:
        if not is_ttm_adaptive_learning_enabled(self._config_ref):
            return
        self._trade_logger.log_trade(record)
        self._closed_trade_count += 1
        every = int(self._config_ref.get("ttm_adaptive_update_every_n_trades", 20))
        if self._closed_trade_count > 0 and self._closed_trade_count % every == 0:
            self._run_update()

    def _run_update(self) -> None:
        trades = self._trade_logger.get_recent_trades(10_000)
        regime_counts = dict(Counter(int(t.regime) for t in trades))
        self._weights, dbg = self._learner.update(self._weights, trades, self._config_ref, self._state)
        self._last_update_debug = dbg
        dbg_out = {
            **dbg,
            "regime_trade_counts_window": regime_counts,
            "closed_trade_count": int(self._closed_trade_count),
        }
        self._config_ref["_ttm_adaptive_last_debug"] = dbg_out
        self._last_update_debug = dbg_out
        logger.info(
            "TTM adaptive weight update",
            extra={
                "ttm_adaptive_update": True,
                "regime_trade_counts_window": regime_counts,
                "get_last_update_debug": dbg_out,
            },
        )


def build_trade_record_from_snapshot(
    entry: Mapping[str, Any],
    *,
    exit_price: float,
    pnl: float,
    holding_bars: int,
    exit_time: str = "",
) -> TradeRecord:
    dbg = entry.get("debug") or {}
    comp = dbg.get("score_components") or {}
    mom = comp.get("price_momentum_last", comp.get("momentum", 0.0))
    basis_eff = comp.get("basis_mom_last", comp.get("basis_effect", 0.0))
    oi_eff = comp.get("positioning_oi_core", comp.get("oi_effect", 0.0))
    fe = {
        "momentum": float(mom or 0.0),
        "basis_effect": float(basis_eff or 0.0),
        "oi_effect": float(oi_eff or 0.0),
    }
    return TradeRecord(
        entry_time=str(entry.get("entry_time", "")),
        exit_time=str(exit_time),
        side=str(entry.get("side", "LONG")),
        entry_price=float(entry.get("entry_price", 0.0)),
        exit_price=float(exit_price),
        pnl=float(pnl),
        holding_bars=int(holding_bars),
        features_at_entry=fe,
        score_long=float(dbg.get("score_long", 0.0)),
        score_short=float(dbg.get("score_short", 0.0)),
        prob_long=float(dbg.get("prob_long", 0.0)),
        prob_short=float(dbg.get("prob_short", 0.0)),
        regime=int(dbg.get("ttm_regime_id", entry.get("regime", 0))),
    )
