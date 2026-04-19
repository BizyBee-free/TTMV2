"""Periodic rebuild of empirical curves from rolling store (automate_tunning.md §5)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.strategies.ttm.empirical.calibration import (
    CalibrationState,
    build_curves_from_labeled_rows,
    calibration_to_serializable,
    smooth_curves_ema,
)
from src.strategies.ttm.empirical.feature_store import RollingFeatureStore


class EmpiricalAlphaEngine:
    """
    Maintains :class:`RollingFeatureStore` and optional :class:`CalibrationState`.

    Rebuilds curves every ``update_every_bars`` bars with EMA smoothing vs previous state.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = dict(config or {})
        win = int(cfg.get("ttm_v2_empirical_window_size", 300))
        h = int(cfg.get("ttm_v2_empirical_forward_horizon", 4))
        self._store = RollingFeatureStore(window_size=win, forward_horizon=h)
        self._n_bins = int(cfg.get("ttm_v2_empirical_n_bins", 5))
        self._min_bin = int(cfg.get("ttm_v2_empirical_min_bin_samples", 20))
        self._update_every = max(1, int(cfg.get("ttm_v2_empirical_update_every_bars", 30)))
        self._ema_old = float(cfg.get("ttm_v2_empirical_ema_old_weight", 0.7))
        self._bars_since_update = 0
        self._bar_ticks = 0
        self._calib: Optional[CalibrationState] = None
        self._prev_for_ema: Optional[CalibrationState] = None
        self._last_debug: Dict[str, Any] = {}

    @property
    def store(self) -> RollingFeatureStore:
        return self._store

    @property
    def calibration(self) -> Optional[CalibrationState]:
        return self._calib

    def on_bar(
        self,
        bar_index: int,
        timestamp: int,
        last: Mapping[str, Any],
        close: float,
    ) -> None:
        self._store.append(bar_index, timestamp, last, close)
        self._bar_ticks += 1
        self._bars_since_update += 1
        if self._bars_since_update >= self._update_every:
            self._rebuild()
            self._bars_since_update = 0

    def _rebuild(self) -> None:
        rows = list(self._store.iter_rows())
        raw = build_curves_from_labeled_rows(
            rows,
            n_bins=self._n_bins,
            min_bin_samples=self._min_bin,
        )
        self._calib = smooth_curves_ema(self._prev_for_ema, raw, alpha_old=self._ema_old)
        self._prev_for_ema = self._calib
        self._last_debug = {
            "rebuilt": True,
            "n_rows_store": len(rows),
            "n_labeled": raw.n_labeled,
            "calibration": calibration_to_serializable(self._calib),
        }

    def force_rebuild(self) -> None:
        """Public hook for tests / research replay."""
        self._rebuild()
        self._bars_since_update = 0

    def get_last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    def reset(self) -> None:
        self._store.clear()
        self._calib = None
        self._prev_for_ema = None
        self._bars_since_update = 0
        self._bar_ticks = 0
        self._last_debug = {}
