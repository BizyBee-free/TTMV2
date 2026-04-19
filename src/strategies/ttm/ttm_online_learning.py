"""Deterministic online weight updates from realized trade outcomes (no price training)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from src.strategies.ttm.ttm_trade_logger import TradeRecord
from src.strategies.ttm.ttm_weights import REGIME_KEYS, RegimeWeights, WeightSet, clip_regime_weights, clip_weight_set


def _pearson_corr(x: List[float], y: List[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    n = a.size
    if n < 2 or b.size != n:
        return 0.0
    ca = a - a.mean()
    cb = b - b.mean()
    da = float(np.sum(ca**2))
    db = float(np.sum(cb**2))
    if da <= 1e-18 or db <= 1e-18:
        return 0.0
    return float(np.sum(ca * cb) / np.sqrt(da * db))


@dataclass
class RegimeWeightState:
    """Tracks last accepted batch mean PnL per regime for rollback."""

    last_mean_pnl: Dict[int, float] = field(default_factory=dict)


class OnlineLearner:
    """Stateless update step; caller holds :class:`RegimeWeightState` and :class:`RegimeWeights`."""

    FEATURE_KEYS = ("momentum", "basis_effect", "oi_effect")

    @staticmethod
    def _recency_weights(n: int, halflife: float) -> np.ndarray:
        if n <= 0:
            return np.array([], dtype=np.float64)
        h = max(1.0, float(halflife))
        idx = np.arange(n, dtype=np.float64)
        # Newer trades get larger weights.
        w = np.exp((idx - float(n - 1)) / h)
        s = float(np.sum(w))
        if s <= 1e-12:
            return np.full(n, 1.0 / n, dtype=np.float64)
        return (w / s).astype(np.float64)

    def update(
        self,
        weights: RegimeWeights,
        trades: List[TradeRecord],
        config: Mapping[str, Any],
        state: RegimeWeightState,
    ) -> Tuple[RegimeWeights, Dict[str, Any]]:
        min_trades = int(config.get("ttm_adaptive_min_trades", 30))
        alpha = float(config.get("ttm_adaptive_alpha", 0.1))
        max_delta = float(config.get("ttm_adaptive_max_delta", 0.2))
        var_max = float(config.get("ttm_adaptive_pnl_variance_max", 1e12))
        eps = float(config.get("ttm_adaptive_rollback_eps", 1e-9))
        use_recency = bool(config.get("ttm_adaptive_use_recency_weight", True))
        halflife = float(config.get("ttm_adaptive_recency_halflife_trades", 40))

        debug: Dict[str, Any] = {"regimes": {}, "skipped": []}
        new_map = {
            k: WeightSet(
                momentum=weights.weights[k].momentum,
                basis=weights.weights[k].basis,
                oi=weights.weights[k].oi,
            )
            for k in REGIME_KEYS
        }

        for r in REGIME_KEYS:
            sub = [t for t in trades if int(t.regime) == int(r)]
            n_sub = len(sub)
            if n_sub < min_trades:
                debug["skipped"].append({"regime": r, "reason": "min_trades", "n": n_sub})
                continue

            pnls = [float(t.pnl) for t in sub]
            pnl_arr = np.asarray(pnls, dtype=np.float64)
            rw = (
                self._recency_weights(len(sub), halflife)
                if use_recency
                else np.full(len(sub), 1.0 / max(1, len(sub)), dtype=np.float64)
            )
            var_p = float(np.var(pnl_arr, ddof=0))
            if var_p > var_max:
                debug["skipped"].append({"regime": r, "reason": "pnl_variance", "var": var_p})
                continue

            mean_pnl = float(np.sum(pnl_arr * rw))
            old_m = new_map[r].momentum
            old_b = new_map[r].basis
            old_o = new_map[r].oi

            m, b, o = old_m, old_b, old_o
            deltas: List[float] = []
            for i, fk in enumerate(self.FEATURE_KEYS):
                xs = [float((t.features_at_entry or {}).get(fk, 0.0)) for t in sub]
                corr = _pearson_corr(xs, pnls)
                d = float(max(-max_delta, min(max_delta, alpha * corr)))
                deltas.append(d)
                if i == 0:
                    m += d
                elif i == 1:
                    b += d
                else:
                    o += d

            cand = clip_weight_set(WeightSet(momentum=m, basis=b, oi=o))

            last_m = state.last_mean_pnl.get(r)
            if last_m is not None and np.isfinite(last_m) and mean_pnl < last_m - eps:
                debug["regimes"][r] = {
                    "rolled_back": True,
                    "performance_before": last_m,
                    "batch_mean_pnl": mean_pnl,
                    "old_weights": {"momentum": old_m, "basis": old_b, "oi": old_o},
                    "new_weights": {"momentum": old_m, "basis": old_b, "oi": old_o},
                    "delta": deltas,
                }
                continue

            new_map[r] = cand
            state.last_mean_pnl[r] = mean_pnl
            debug["regimes"][r] = {
                "rolled_back": False,
                "performance_before": last_m,
                "performance_after": mean_pnl,
                "batch_mean_pnl": mean_pnl,
                "old_weights": {"momentum": old_m, "basis": old_b, "oi": old_o},
                "new_weights": {"momentum": cand.momentum, "basis": cand.basis, "oi": cand.oi},
                "delta": deltas,
                "adaptive_recency_weighted": bool(use_recency),
                "adaptive_recency_halflife_trades": float(halflife),
            }

        out = RegimeWeights(weights=clip_regime_weights(new_map))
        return out, debug
