"""Quantile binning and empirical expected-return curves (automate_tunning.md §2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.strategies.ttm.empirical.feature_store import EMPIRICAL_FEATURE_KEYS, FeatureStoreRow


def _quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Strictly increasing edges for ``n_bins`` buckets; len n_bins-1."""
    x = x[np.isfinite(x)]
    if x.size < n_bins:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(x, qs)
    # enforce strictly increasing
    eps = 1e-12 * (1.0 + np.max(np.abs(edges)) if edges.size else 1.0)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    return edges.astype(np.float64)


def assign_bin(value: float, edges: np.ndarray) -> int:
    """Map scalar to bin index in ``0 .. n_bins-1`` (n_bins = len(edges)+1)."""
    if edges.size == 0:
        return 0
    if not np.isfinite(value):
        return 0
    return int(np.searchsorted(edges, value, side="right"))


@dataclass
class CalibrationState:
    """Per-feature bin edges, mean forward return per bin, counts."""

    n_bins: int
    edges: Dict[str, np.ndarray]
    curve: Dict[str, List[float]]
    counts: Dict[str, List[int]]
    global_mean_return: float
    n_labeled: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def expected_return(self, key: str, feature_value: float) -> float:
        k = str(key)
        ed = self.edges.get(k)
        cr = self.curve.get(k)
        if ed is None or cr is None or len(cr) != self.n_bins:
            return float(self.global_mean_return)
        b = assign_bin(float(feature_value), ed)
        b = max(0, min(self.n_bins - 1, b))
        return float(cr[b])


def build_curves_from_labeled_rows(
    rows: Sequence[FeatureStoreRow],
    *,
    n_bins: int = 5,
    min_bin_samples: int = 20,
) -> CalibrationState:
    """
    Fit quantile edges on feature values in ``rows`` (labeled only) and mean ``forward_return_4`` per bin.
    Sparse bins fall back to global mean of labeled returns.
    """
    n_bins = max(2, int(n_bins))
    labeled = [r for r in rows if r.is_labeled()]
    y_all = np.array([r.forward_return_4 for r in labeled], dtype=np.float64)
    y_all = y_all[np.isfinite(y_all)]
    global_mean = float(np.mean(y_all)) if y_all.size > 0 else 0.0

    edges_map: Dict[str, np.ndarray] = {}
    curve_map: Dict[str, List[float]] = {}
    counts_map: Dict[str, List[int]] = {}

    for fk in EMPIRICAL_FEATURE_KEYS:
        xs = np.array([getattr(r, fk) for r in labeled], dtype=np.float64)
        ys = np.array([r.forward_return_4 for r in labeled], dtype=np.float64)
        m = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[m], ys[m]
        if xs.size < n_bins * 2:
            edges_map[fk] = np.array([], dtype=np.float64)
            curve_map[fk] = [global_mean] * n_bins
            counts_map[fk] = [0] * n_bins
            continue

        edges = _quantile_edges(xs, n_bins)
        if edges.size != n_bins - 1:
            edges_map[fk] = np.array([], dtype=np.float64)
            curve_map[fk] = [global_mean] * n_bins
            counts_map[fk] = [0] * n_bins
            continue

        edges_map[fk] = edges
        means: List[float] = []
        cnts: List[int] = []
        for b in range(n_bins):
            lo = -np.inf if b == 0 else float(edges[b - 1])
            hi = np.inf if b == n_bins - 1 else float(edges[b])
            if b == 0:
                mask = xs <= hi
            elif b == n_bins - 1:
                mask = xs > lo
            else:
                mask = (xs > lo) & (xs <= hi)
            n_b = int(np.sum(mask))
            cnts.append(n_b)
            if n_b < min_bin_samples:
                means.append(global_mean)
            else:
                means.append(float(np.mean(ys[mask])))
        curve_map[fk] = means
        counts_map[fk] = cnts

    return CalibrationState(
        n_bins=n_bins,
        edges=edges_map,
        curve=curve_map,
        counts=counts_map,
        global_mean_return=global_mean,
        n_labeled=len(labeled),
        meta={"min_bin_samples": min_bin_samples},
    )


def smooth_curves_ema(
    old: Optional[CalibrationState],
    new: CalibrationState,
    *,
    alpha_old: float = 0.7,
) -> CalibrationState:
    """``curve = alpha_old * old + (1-alpha_old) * new`` per bin; edges/counts from ``new``."""
    if old is None or old.n_bins != new.n_bins:
        return new
    a = float(np.clip(alpha_old, 0.0, 1.0))
    blended: Dict[str, List[float]] = {}
    for fk in EMPIRICAL_FEATURE_KEYS:
        oc = old.curve.get(fk) or []
        nc = new.curve.get(fk) or []
        if len(oc) != new.n_bins or len(nc) != new.n_bins:
            blended[fk] = list(nc)
            continue
        blended[fk] = [a * float(oc[i]) + (1.0 - a) * float(nc[i]) for i in range(new.n_bins)]
    return CalibrationState(
        n_bins=new.n_bins,
        edges=new.edges,
        curve=blended,
        counts=new.counts,
        global_mean_return=new.global_mean_return,
        n_labeled=new.n_labeled,
        meta={**new.meta, "ema_alpha_old": a},
    )


def calibration_to_serializable(state: CalibrationState) -> Dict[str, Any]:
    return {
        "n_bins": state.n_bins,
        "edges": {k: v.tolist() for k, v in state.edges.items()},
        "curve": dict(state.curve),
        "counts": dict(state.counts),
        "global_mean_return": state.global_mean_return,
        "n_labeled": state.n_labeled,
        "meta": dict(state.meta),
    }


def calibration_from_serializable(d: Mapping[str, Any]) -> CalibrationState:
    edges = {k: np.asarray(v, dtype=np.float64) for k, v in (d.get("edges") or {}).items()}
    return CalibrationState(
        n_bins=int(d.get("n_bins", 5)),
        edges=edges,
        curve={k: list(v) for k, v in (d.get("curve") or {}).items()},
        counts={k: list(v) for k, v in (d.get("counts") or {}).items()},
        global_mean_return=float(d.get("global_mean_return", 0.0)),
        n_labeled=int(d.get("n_labeled", 0)),
        meta=dict(d.get("meta") or {}),
    )
