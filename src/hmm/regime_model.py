"""Hidden Markov Model wrapper for market regime detection.

Wraps hmmlearn's GaussianHMM with:
  - Automatic state labelling (BULL / FLAT / BEAR) by ranking per-state
    mean log-return so trading rules are stable across re-fits.
  - predict_proba() returns posterior state probabilities for every bar
    in the feature matrix (via the forward-backward algorithm).
  - Graceful handling of degenerate fits (too few samples, numerical issues).

Theory:
    The HMM has K hidden states Z_t ∈ {0, 1, ..., K-1}.
    Observations X_t ~ N(mu_k, Sigma_k) given Z_t = k.
    Transition probabilities A[i,j] = P(Z_t=j | Z_{t-1}=i).
    Baum-Welch EM maximises P(X | A, mu, Sigma) over the training window.

Usage::

    from src.hmm.regime_model import HMMRegimeModel
    from src.hmm.feature_engineer import HMMConfig

    cfg = HMMConfig(k_states=3)
    model = HMMRegimeModel(cfg)
    model.fit(X_train)             # X_train: (n_samples, n_features)
    proba = model.predict_proba(X) # shape (n_samples, k_states)
    labels = model.state_labels    # {0: "BULL", 1: "FLAT", 2: "BEAR"}
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np

from src.hmm.feature_engineer import HMMConfig

# Silence hmmlearn convergence chatter at module level
logging.getLogger("hmmlearn").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="hmmlearn")
from src.logger import get_logger

logger = get_logger("hmm_regime_model")

# State label names — assigned after fitting by mean log-return rank
STATE_BULL = "BULL"
STATE_FLAT = "FLAT"
STATE_BEAR = "BEAR"

# Minimum samples needed to attempt a fit
_MIN_SAMPLES = 10


class HMMRegimeModel:
    """GaussianHMM wrapper for market regime detection.

    Args:
        config: HMMConfig with k_states, n_iter, covariance_type, random_state.
    """

    def __init__(self, config: HMMConfig) -> None:
        self._cfg = config
        self._model: Optional[object] = None   # hmmlearn GaussianHMM
        self._state_labels: Dict[int, str] = {}
        self._is_fitted = False
        self._n_features: int = 3
        # Diagnostics captured after each fit
        self._log_likelihood: float = float("-inf")
        self._ll_history: List[float] = []
        self._transition_matrix: Optional[np.ndarray] = None
        self._state_means_arr: Optional[np.ndarray] = None
        self._state_covars_arr: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def state_labels(self) -> Dict[int, str]:
        """Map from internal state index → label (BULL/FLAT/BEAR)."""
        return dict(self._state_labels)

    def fit(self, X: np.ndarray) -> bool:
        """Fit (or re-fit) the HMM on feature matrix X via Baum-Welch EM.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            True if fit succeeded, False if skipped (too few samples or error).
        """
        if X.shape[0] < _MIN_SAMPLES:
            logger.debug(f"HMMRegimeModel: skip fit, only {X.shape[0]} samples (need {_MIN_SAMPLES})")
            return False

        Xf = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        from hmmlearn import hmm as hmmlib

        # "full" covariance often fails PD checks on short / collinear windows; retry diag + min_covar.
        plans: List[tuple] = [(self._cfg.covariance_type, 1e-3)]
        if self._cfg.covariance_type != "diag":
            plans.append(("diag", 1e-3))
        plans.append(("diag", 1e-2))
        plans.append(("diag", 0.1))

        seen = set()
        last_err: Optional[Exception] = None
        for cov_type, min_cov in plans:
            key = (cov_type, min_cov)
            if key in seen:
                continue
            seen.add(key)
            try:
                model = hmmlib.GaussianHMM(
                    n_components=self._cfg.k_states,
                    covariance_type=cov_type,
                    n_iter=self._cfg.n_iter,
                    min_covar=min_cov,
                    random_state=self._cfg.random_state,
                    verbose=False,
                )
                model.fit(Xf)
                try:
                    self._log_likelihood = float(model.score(Xf))
                except Exception:
                    self._log_likelihood = float("-inf")
                try:
                    self._ll_history = list(model.monitor_.history)
                except Exception:
                    self._ll_history = (
                        [self._log_likelihood] if self._log_likelihood > float("-inf") else []
                    )
                self._transition_matrix = model.transmat_.copy()
                self._state_means_arr = model.means_.copy()
                try:
                    self._state_covars_arr = model.covars_.copy()
                except Exception:
                    self._state_covars_arr = None
                self._model = model
                self._is_fitted = True
                self._label_states(Xf)
                if cov_type != self._cfg.covariance_type:
                    logger.info(
                        "HMMRegimeModel: fit used fallback covariance",
                        extra={"covariance_type": cov_type, "min_covar": min_cov},
                    )
                return True
            except Exception as e:
                last_err = e
                continue

        logger.warning(f"HMMRegimeModel: fit failed — {last_err}")
        self._is_fitted = False
        return False

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities for each row of X.

        Uses the forward-backward algorithm (model.predict_proba in hmmlearn).

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, k_states). Columns ordered by
            internal state index (use state_labels to map to BULL/FLAT/BEAR).
            Returns uniform probabilities if model is not fitted.
        """
        k = self._cfg.k_states
        if not self._is_fitted or self._model is None or X.shape[0] < 1:
            return np.full((X.shape[0], k), 1.0 / k)

        try:
            # hmmlearn exposes predict_proba via predict on posteriors
            # Use _compute_subnorm for posteriors (forward-backward)
            posteriors = self._model.predict_proba(X)   # (n, k)
            return posteriors
        except Exception as e:
            logger.warning(f"HMMRegimeModel: predict_proba failed — {e}")
            return np.full((X.shape[0], k), 1.0 / k)

    def predict_states(self, X: np.ndarray) -> np.ndarray:
        """Predict hidden states with optional causal smoothing/persistence guards.

        This keeps state transitions less noisy (useful for live trading and
        stability-oriented diagnostics): first apply a trailing majority vote,
        then enforce a minimum run length.
        """
        if X.shape[0] < 1:
            return np.empty((0,), dtype=int)
        k = max(1, int(self._cfg.k_states))
        if not self._is_fitted or self._model is None:
            return np.zeros((X.shape[0],), dtype=int)
        try:
            mode = str(getattr(self._cfg, "state_scoring_mode", "argmax")).strip().lower()
            if mode == "strength" and k == 3:
                raw = self._states_from_strength(X)
            else:
                raw = np.asarray(self._model.predict(X), dtype=int)
        except Exception as e:
            logger.warning(f"HMMRegimeModel: predict_states failed, fallback zeros — {e}")
            return np.zeros((X.shape[0],), dtype=int)

        w = max(1, int(getattr(self._cfg, "state_smoothing_window", 1)))
        out = raw.copy()
        if w > 1:
            sm = out.copy()
            for i in range(len(out)):
                seg = out[max(0, i - w + 1) : i + 1]
                binc = np.bincount(np.clip(seg, 0, k - 1), minlength=k)
                sm[i] = int(np.argmax(binc))
            out = sm

        min_run = max(1, int(getattr(self._cfg, "state_min_run_bars", 1)))
        if min_run > 1 and len(out) > 1:
            fixed = out.copy()
            run_state = int(fixed[0])
            run_len = 1
            for i in range(1, len(fixed)):
                cur = int(fixed[i])
                prev = int(fixed[i - 1])
                if cur == prev:
                    run_len += 1
                    run_state = cur
                    continue
                if run_len < min_run:
                    fixed[i] = run_state
                    run_len += 1
                else:
                    run_state = cur
                    run_len = 1
            out = fixed

        return out

    def _states_from_strength(self, X: np.ndarray) -> np.ndarray:
        """Map bars into 3 regimes by signed regime-strength score.

        score_t = sum_k posterior_t(k) * mean_log_ret_k
        Then quantile split:
          low  -> 0 (bear-like)
          mid  -> 1 (flat-like)
          high -> 2 (bull-like)
        """
        if self._model is None:
            return np.zeros((X.shape[0],), dtype=int)
        proba = self.predict_proba(X)  # (n, 3)
        means = self._model.means_
        if means.shape[1] < 1:
            return np.asarray(self._model.predict(X), dtype=int)
        mu = np.asarray(means[:, 0], dtype=float)  # mean log_ret per hidden state
        score = proba @ mu
        q = float(getattr(self._cfg, "state_strength_flat_quantile", 0.35))
        q = min(0.49, max(0.05, q))
        lo = float(np.quantile(score, q))
        hi = float(np.quantile(score, 1.0 - q))
        out = np.ones((len(score),), dtype=int)
        out[score <= lo] = 0
        out[score >= hi] = 2
        return out

    def get_latest_state_proba(self, X: np.ndarray) -> np.ndarray:
        """Return state probabilities for the last bar only (shape (k,))."""
        proba = self.predict_proba(X)
        return proba[-1]

    def label_of_state(self, state_idx: int) -> str:
        """Return the label (BULL/FLAT/BEAR) for an internal state index."""
        return self._state_labels.get(state_idx, STATE_FLAT)

    # ------------------------------------------------------------------ #
    # Diagnostics properties (populated after each successful fit)
    # ------------------------------------------------------------------ #

    @property
    def log_likelihood(self) -> float:
        """Final log-likelihood log P(X|theta) of training data under fitted model."""
        return self._log_likelihood

    @property
    def ll_history(self) -> List[float]:
        """Log-likelihood recorded by hmmlearn ConvergenceMonitor per EM iteration."""
        return list(self._ll_history)

    @property
    def transition_matrix(self) -> Optional[np.ndarray]:
        """Fitted transition matrix A[i,j]=P(Z_t=j|Z_{t-1}=i). Shape (k, k)."""
        return self._transition_matrix.copy() if self._transition_matrix is not None else None

    @property
    def state_means(self) -> Optional[np.ndarray]:
        """Fitted emission means per state. Shape (k, n_features)."""
        return self._state_means_arr.copy() if self._state_means_arr is not None else None

    @property
    def state_covars(self) -> Optional[np.ndarray]:
        """Fitted emission covariance matrices per state. Shape (k, d, d) for full cov."""
        return self._state_covars_arr.copy() if self._state_covars_arr is not None else None

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _label_states(self, X_train: np.ndarray) -> None:
        """Auto-label states by their mean log-return (first feature column).

        Ranking:
          highest mean log-return → BULL
          middle mean log-return  → FLAT (accumulation / range-bound)
          lowest mean log-return  → BEAR
        """
        if self._model is None:
            return

        k = self._cfg.k_states
        means = self._model.means_   # shape (k, n_features)

        # Use first feature column (log_ret) for ranking
        log_ret_means = means[:, 0]
        ranked = np.argsort(log_ret_means)  # ascending: ranked[0]=lowest, ranked[-1]=highest

        self._state_labels = {}
        if k == 1:
            self._state_labels[ranked[0]] = STATE_FLAT
        elif k == 2:
            self._state_labels[ranked[0]] = STATE_BEAR
            self._state_labels[ranked[1]] = STATE_BULL
        else:
            # k >= 3: label bottom as BEAR, top as BULL, middle(s) as FLAT
            self._state_labels[ranked[0]] = STATE_BEAR
            self._state_labels[ranked[-1]] = STATE_BULL
            for idx in ranked[1:-1]:
                self._state_labels[idx] = STATE_FLAT

        table = ", ".join(
            f"z{i}={self._state_labels[i]}(mean_log_ret={log_ret_means[i]:.6f})"
            for i in range(k)
        )
        logger.info(
            "HMMRegimeModel: gán nhãn theo mean log_ret (cột 0) — "
            f"{table}. K≥3: 3 hạng → BEAR / FLAT / BULL (middle=FLAT)."
        )
