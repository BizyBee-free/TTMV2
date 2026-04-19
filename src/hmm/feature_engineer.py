"""Feature engineering for HMM regime detection.

Transforms raw OhlcBar list into a stationary, normalised feature matrix
ready for hmmlearn's GaussianHMM.

Features (as per HMMplan.yaml):
    log_ret      = ln(close_t / close_{t-1})            -- price momentum
    range_ratio  = (high_t - low_t) / close_t           -- intraday volatility
    vol_change   = (vol_t - vol_{t-1}) / vol_{t-1}      -- volume momentum

Each feature is then z-score normalised over a rolling window so the HMM
receives stationary inputs regardless of the absolute price level.

Usage::

    from src.hmm.feature_engineer import HMMFeatureEngineer, HMMConfig
    cfg = HMMConfig()
    eng = HMMFeatureEngineer(cfg)
    X = eng.compute(bars)          # shape (n_bars-1, 3), first bar dropped
    X_latest = eng.compute_latest(bars)   # shape (1, 3) for prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.backtest.data_fetcher import OhlcBar


@dataclass
class HMMConfig:
    """Configuration shared by HMM feature engineering and model."""
    k_states: int = 3          # number of hidden states (Bull/Bear/Flat)
    n_iter: int = 1000         # Baum-Welch EM iterations
    covariance_type: str = "full"
    zscore_window: int = 20    # rolling window for z-score normalisation
    random_state: int = 42
    use_basis: bool = False    # extra col: (future_close − index_close) z-scored
    # extra col after basis (if any): relative OI change between consecutive bars
    use_open_interest: bool = False
    use_vol_change: bool = False
    state_smoothing_window: int = 3
    state_min_run_bars: int = 2
    state_scoring_mode: str = "argmax"  # argmax | strength
    state_strength_flat_quantile: float = 0.35


class HMMFeatureEngineer:
    """Computes the feature matrix from a list of OhlcBar.

    Default: 3 columns [log_ret, range_ratio, vol_change]. Optional columns (in order):
    basis spread (``use_basis``), then OI relative change (``use_open_interest``).
    All columns are rolling z-scored. Column 0 remains ``log_ret`` for state labelling.

    Args:
        config: HMMConfig instance; uses zscore_window, ``use_basis``, ``use_open_interest``.
    """

    _EPS = 1e-10   # prevent division by zero

    def __init__(self, config: HMMConfig) -> None:
        self._cfg = config
        self._w = config.zscore_window

    @property
    def n_features(self) -> int:
        n = 2 + (1 if self._cfg.use_vol_change else 0)
        if self._cfg.use_basis:
            n += 1
        if self._cfg.use_open_interest:
            n += 1
        return n

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compute(
        self,
        bars: List[OhlcBar],
        index_closes: Optional[List[float]] = None,
        open_interest: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Compute the feature matrix for all bars.

        Args:
            bars: List of OhlcBar, sorted oldest-first.  Must have >= 2 bars.
            index_closes: When ``config.use_basis`` is True, one float per bar
                (same order as ``bars``), index close at each bar timestamp.
            open_interest: When ``config.use_open_interest`` is True, one float per bar
                (OI level from DNSE secdef, aligned to ``bars``).

        Returns:
            numpy array of shape ``(len(bars) - 1, n_features)``. The first bar
            is consumed as a reference so the output has one fewer row.
        """
        n = len(bars)
        nf = self.n_features
        if n < 2:
            return np.empty((0, nf))

        if self._cfg.use_basis:
            if index_closes is None or len(index_closes) != n:
                raise ValueError(
                    "index_closes (len == len(bars)) required when use_basis=True"
                )
        if self._cfg.use_open_interest:
            if open_interest is None or len(open_interest) != n:
                raise ValueError(
                    "open_interest (len == len(bars)) required when use_open_interest=True"
                )

        log_ret    = np.empty(n - 1)
        range_r    = np.empty(n - 1)
        vol_change = np.empty(n - 1)
        basis_spread = np.empty(n - 1) if self._cfg.use_basis else None
        oi_delta = np.empty(n - 1) if self._cfg.use_open_interest else None

        for i in range(1, n):
            prev = bars[i - 1]
            curr = bars[i]

            # log return
            c_prev = prev.close if prev.close > 0 else self._EPS
            c_curr = curr.close if curr.close > 0 else self._EPS
            log_ret[i - 1] = np.log(c_curr / c_prev)

            # intraday range ratio
            range_r[i - 1] = (curr.high - curr.low) / (c_curr + self._EPS)

            # volume change
            v_prev = prev.volume if prev.volume > 0 else self._EPS
            v_curr = curr.volume if curr.volume >= 0 else 0.0
            vol_change[i - 1] = (v_curr - v_prev) / (abs(v_prev) + self._EPS)

            if basis_spread is not None:
                ic = index_closes[i] if index_closes is not None else 0.0
                basis_spread[i - 1] = curr.close - float(ic)

            if oi_delta is not None and open_interest is not None:
                o_prev = float(open_interest[i - 1])
                o_curr = float(open_interest[i])
                oi_delta[i - 1] = (o_curr - o_prev) / (abs(o_prev) + self._EPS)

        parts = [log_ret, range_r]
        if self._cfg.use_vol_change:
            parts.append(vol_change)
        if basis_spread is not None:
            parts.append(basis_spread)
        if oi_delta is not None:
            parts.append(oi_delta)
        raw = np.column_stack(parts)
        return self._rolling_zscore(raw)

    def compute_latest(
        self,
        bars: List[OhlcBar],
        index_closes: Optional[List[float]] = None,
        open_interest: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Return the feature vector for the last bar only (shape (1, n_features))."""
        X = self.compute(bars, index_closes=index_closes, open_interest=open_interest)
        if X.shape[0] == 0:
            return np.zeros((1, self.n_features))
        return X[-1:, :]

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _rolling_zscore(self, X: np.ndarray) -> np.ndarray:
        """Apply per-column rolling z-score normalisation.

        For each column and each row i:
            z_i = (x_i - mean(x_{i-w+1:i+1})) / (std(x_{i-w+1:i+1}) + eps)

        Rows with fewer than 2 samples in the window retain the raw value
        (z-score is undefined; this only affects the first few bars during
        warm-up and those bars are not traded anyway).
        """
        n, k = X.shape
        Z = np.empty_like(X)
        for col in range(k):
            for i in range(n):
                start = max(0, i - self._w + 1)
                window = X[start : i + 1, col]
                mu = window.mean()
                sigma = window.std(ddof=1) if len(window) > 1 else 0.0
                Z[i, col] = (X[i, col] - mu) / (sigma + self._EPS)
        return Z
