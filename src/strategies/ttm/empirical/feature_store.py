"""Rolling feature store for empirical TTM calibration (no leakage on forward_return)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterator, List, Mapping, Optional

import numpy as np

# Canonical keys used by calibration / alpha (match automate_tunning.md schema).
EMPIRICAL_FEATURE_KEYS = ("breakout", "basis", "positioning", "vol")


@dataclass
class FeatureStoreRow:
    """One bar: features at t, close at t, label when horizon elapses."""

    bar_index: int
    timestamp: int
    breakout: float
    basis: float
    positioning: float
    vol: float
    close: float
    forward_return_4: float = field(default_factory=lambda: float("nan"))

    def features_dict(self) -> Dict[str, float]:
        return {
            "breakout": float(self.breakout),
            "basis": float(self.basis),
            "positioning": float(self.positioning),
            "vol": float(self.vol),
        }

    def is_labeled(self) -> bool:
        return math.isfinite(self.forward_return_4)


def last_dict_from_decision_features(features: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build a flat ``last``-like dict from one paper JSONL ``features`` object (nested logs).
    Used by offline research replay when full ``compute_ttm_features`` is not run.
    """
    f = dict(features or {})
    vplog = f.get("volume_participation_log") or {}
    plog = f.get("positioning_log") or {}
    return {
        "breakout_up": bool(f.get("is_breakout_up")),
        "breakout_down": bool(f.get("is_breakout_down")),
        "breakout_strength_base": float(
            f.get("breakout_strength_base", f.get("breakout_strength", 0.0)) or 0.0
        ),
        "breakout_strength": float(f.get("breakout_strength", 0.0) or 0.0),
        "basis_signal": float(plog.get("basis_signal", f.get("basis_signal", 0.0)) or 0.0),
        "basis_effect": float(plog.get("basis_effect", f.get("basis_effect", 0.0)) or 0.0),
        "positioning_strength": float(
            plog.get("positioning_strength", f.get("positioning_strength", 0.0)) or 0.0
        ),
        "vol_zscore": float(vplog.get("vol_zscore", f.get("vol_zscore", 0.0)) or 0.0),
    }


def extract_empirical_features_from_last(last: Mapping[str, Any]) -> Dict[str, float]:
    """
    Map TTM V2 ``features_last_row`` / decision ``features`` scalars to empirical axes.

    - breakout: signed strength (up positive, down negative) using base ATR strength.
    - basis: ``basis_signal`` with fallback ``basis_effect``.
    - positioning: ``positioning_strength``.
    - vol: ``vol_zscore`` (volume participation z).
    """
    bu = bool(last.get("breakout_up"))
    bd = bool(last.get("breakout_down"))
    base = float(last.get("breakout_strength_base", last.get("breakout_strength", 0.0)) or 0.0)
    sign = 1.0 if bu else (-1.0 if bd else 0.0)
    breakout = sign * base

    basis = float(last.get("basis_signal", last.get("basis_effect", 0.0)) or 0.0)
    if abs(basis) < 1e-15 and last.get("basis_effect") is not None:
        basis = float(last.get("basis_effect") or 0.0)

    positioning = float(last.get("positioning_strength", 0.0) or 0.0)
    vol = float(last.get("vol_zscore", last.get("vol_z", 0.0)) or 0.0)
    if not np.isfinite(vol):
        vol = 0.0
    if not np.isfinite(breakout):
        breakout = 0.0
    if not np.isfinite(basis):
        basis = 0.0
    if not np.isfinite(positioning):
        positioning = 0.0

    return {
        "breakout": breakout,
        "basis": basis,
        "positioning": positioning,
        "vol": vol,
    }


class RollingFeatureStore:
    """
    Rolling window of rows; labels ``forward_return_4`` only after ``horizon`` bars.

    When appending bar ``i`` with ``close_i``, any row at ``i - horizon`` gets
    ``(close_i - close_{i-h}) / |close_{i-h}|`` (no look-ahead for features at t).
    """

    def __init__(self, window_size: int = 300, forward_horizon: int = 4) -> None:
        self.window_size = max(8, int(window_size))
        self.forward_horizon = max(1, int(forward_horizon))
        self._rows: Deque[FeatureStoreRow] = deque(maxlen=self.window_size)

    def __len__(self) -> int:
        return len(self._rows)

    def append(
        self,
        bar_index: int,
        timestamp: int,
        last: Mapping[str, Any],
        close: float,
    ) -> None:
        """Append current bar; back-fill forward return for bar ``bar_index - horizon``."""
        feats = extract_empirical_features_from_last(last)
        row = FeatureStoreRow(
            bar_index=int(bar_index),
            timestamp=int(timestamp),
            breakout=feats["breakout"],
            basis=feats["basis"],
            positioning=feats["positioning"],
            vol=feats["vol"],
            close=float(close),
        )
        self._rows.append(row)

        h = self.forward_horizon
        target_bi = int(bar_index) - h
        if target_bi < 0:
            return
        c_now = float(close)
        if not np.isfinite(c_now) or abs(c_now) < 1e-12:
            return
        for r in self._rows:
            if r.bar_index == target_bi and not r.is_labeled():
                c_old = float(r.close)
                if np.isfinite(c_old) and abs(c_old) > 1e-12:
                    r.forward_return_4 = float((c_now - c_old) / abs(c_old))
                break

    def labeled_rows(self) -> List[FeatureStoreRow]:
        return [r for r in self._rows if r.is_labeled()]

    def iter_rows(self) -> Iterator[FeatureStoreRow]:
        return iter(self._rows)

    def get_dataset_snapshot(self) -> List[Dict[str, Any]]:
        """Serializable view for logging / research."""
        out: List[Dict[str, Any]] = []
        for r in self._rows:
            out.append(
                {
                    "bar_index": r.bar_index,
                    "timestamp": r.timestamp,
                    **r.features_dict(),
                    "close": r.close,
                    "forward_return_4": r.forward_return_4,
                }
            )
        return out

    def clear(self) -> None:
        self._rows.clear()
