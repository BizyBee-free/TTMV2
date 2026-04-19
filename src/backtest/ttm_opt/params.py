"""Searchable parameters for TTM Squeeze + Basis + OI optimization (v2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional

# --- Discrete grids (prompt §3) ------------------------------------------------
RESOLUTION_CHOICES = ["1", "5", "15", "60", "240"]  # DNSE: 1m,5m,15m,1h,4h
HTF_CONFIRM_CHOICES: List[Optional[str]] = [None, "15", "60", "240", "1D"]

BB_LEN_GRID = [14, 20, 25]
BB_STD_GRID = [1.5, 2.0, 2.5]
KC_LEN_GRID = [14, 20, 30]
KC_ATR_MULT_GRID = [1.2, 1.5, 2.0]

BASIS_THRESHOLD_GRID = [-10.0, -5.0, 0.0, 5.0, 10.0]
BASIS_DELTA_WIN_GRID = [3, 5, 10]
BASIS_Z_WIN_GRID = [20, 50]

OI_MA_WIN_GRID = [10, 20, 50]
OI_DELTA_WIN_GRID = [3, 5, 10]
OI_SLOPE_WIN_GRID = [10, 20]

ATR_REGIME_WIN_GRID = [14, 20]
ADX_THRESHOLD_GRID = [20.0, 25.0, 30.0]

STOP_LOSS_PCT_GRID = [0.005, 0.01, 0.02, 0.03]
TAKE_PROFIT_PCT_GRID = [0.01, 0.02, 0.03, 0.05]
TRAILING_PCT_GRID = [0.005, 0.01, 0.015, 0.02]
POSITION_RISK_PCT_GRID = [0.01, 0.02, 0.03, 0.04, 0.05]

REGIME_FILTER_CHOICES = [
    "none",
    "trend_only",
    "range_only",
    "skip_high_vol",
    "skip_low_vol",
    "basis_oi_align",
]


@dataclass
class TtmOptParams:
    """Single configuration for TTM-opt backtest + search."""

    # Timing / squeeze
    bb_len: int = 20
    bb_std: float = 2.0
    kc_len: int = 20
    kc_atr_mult: float = 1.5
    atr_len: int = 14
    momentum_threshold: float = 0.0
    # Higher-TF: scale factor on bar index (1 = same as primary); confirmation of momentum sign
    htf_momentum_scale: int = 1  # e.g. 4 → EMA span ~4x on same series (proxy HTF)

    # Basis
    basis_threshold: float = 0.0
    basis_delta_window: int = 5
    basis_z_window: int = 20

    # OI
    oi_ma_window: int = 20
    oi_delta_window: int = 5
    oi_slope_window: int = 10

    # Regime
    atr_regime_window: int = 20
    atr_percentile_window: int = 100
    adx_period: int = 14
    adx_trend_min: float = 25.0
    regime_filter: str = "none"

    # Risk
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    trailing_stop: bool = False
    trailing_pct: float = 0.01
    commission_pct: float = 0.0003
    slippage_pct: float = 0.0003
    position_risk_pct: float = 0.02
    warmup_bars: int = 120

    # Ablation flags for feature importance
    use_squeeze: bool = True
    use_momentum: bool = True
    use_basis: bool = True
    use_oi: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TtmOptParams":
        valid = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in valid}
        return cls(**kw)


def perturb_params(p: TtmOptParams, rng: Any, frac: float = 0.1) -> TtmOptParams:
    """±frac on floats; ints nudged. ``rng``: ``numpy.random.Generator`` or ``random.Random``."""
    d = p.to_dict()
    out: Dict[str, Any] = {}

    def _rand_int(a: int, b: int) -> int:
        if hasattr(rng, "integers"):
            return int(rng.integers(a, b + 1))
        return int(rng.randint(a, b))

    def _rand01() -> float:
        return float(rng.random())

    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int) and k not in ("htf_momentum_scale",):
            delta = max(1, int(abs(v) * frac + 0.5))
            out[k] = max(1, v + _rand_int(-delta, delta))
        elif isinstance(v, float):
            noise = v * frac * (2 * _rand01() - 1)
            out[k] = max(1e-9, v + noise)
        else:
            out[k] = v
    return TtmOptParams.from_dict(out)
