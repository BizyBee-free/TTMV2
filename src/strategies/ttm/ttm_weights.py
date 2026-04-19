"""Bounded per-regime weight sets for TTM adaptive scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

REGIME_KEYS = (-1, 0, 1)


@dataclass
class WeightSet:
    momentum: float
    basis: float
    oi: float


@dataclass
class RegimeWeights:
    """Three WeightSets keyed by vol regime {-1, 0, 1}."""

    weights: Dict[int, WeightSet]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RegimeWeights:
        init = config.get("ttm_adaptive_init_weights")
        if isinstance(init, dict):
            wmap: Dict[int, WeightSet] = {}
            for k in REGIME_KEYS:
                row = init.get(k) or init.get(str(k))
                if isinstance(row, Mapping):
                    wmap[k] = WeightSet(
                        momentum=float(row.get("momentum", row.get("m", 1.0))),
                        basis=float(row.get("basis", row.get("b", 1.0))),
                        oi=float(row.get("oi", row.get("o", 0.8))),
                    )
                else:
                    wmap[k] = WeightSet(
                        momentum=float(config.get("ttm_w_momentum", 1.0)),
                        basis=float(config.get("ttm_w_basis", 1.0)),
                        oi=float(config.get("ttm_w_oi", 0.8)),
                    )
            return cls(weights=clip_regime_weights(wmap))
        m = float(config.get("ttm_w_momentum", 1.0))
        b = float(config.get("ttm_w_basis", 1.0))
        o = float(config.get("ttm_w_oi", 0.8))
        ws = WeightSet(momentum=m, basis=b, oi=o)
        return cls(weights={k: WeightSet(momentum=ws.momentum, basis=ws.basis, oi=ws.oi) for k in REGIME_KEYS})

    def get(self, regime_id: int) -> WeightSet:
        return self.weights.get(int(regime_id), self.weights[0])


def clip_weight_set(ws: WeightSet, lo: float = -3.0, hi: float = 3.0) -> WeightSet:
    return WeightSet(
        momentum=float(min(hi, max(lo, ws.momentum))),
        basis=float(min(hi, max(lo, ws.basis))),
        oi=float(min(hi, max(lo, ws.oi))),
    )


def clip_regime_weights(wmap: Dict[int, WeightSet], lo: float = -3.0, hi: float = 3.0) -> Dict[int, WeightSet]:
    return {k: clip_weight_set(v, lo, hi) for k, v in wmap.items()}
