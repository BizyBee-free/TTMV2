"""Bayesian MCMC engine for derivatives price simulation.

Two-phase computation:

Phase 1 -- Posterior sampling (Metropolis-Hastings):
    Given observed log-returns, sample the posterior distribution of
    (mu, sigma) -- the drift and volatility parameters of a GBM process.
    The sampler runs MCMC_MH_ITERATIONS steps; the first MCMC_MH_BURNIN
    samples are discarded.  Proposal distribution: isotropic Gaussian.

Phase 2 -- Vectorized path simulation:
    Each posterior sample (mu_i, sigma_i) drives a 1-step GBM projection:

        S_T = S_0 * exp((mu_regime - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    where mu_regime = mu * direction_factor, Z ~ N(0,1).
    direction_factor: weighted blend of UP(+1) and DOWN(-1) regimes
    from the Markov transition probabilities.

    All 10,000 paths are computed in a single numpy vectorized call,
    ensuring < 100 ms on typical hardware.

Phase 3 -- Aggregation:
    P(UP)   = count(S_T > S_0) / n_paths
    P(DOWN) = count(S_T < S_0) / n_paths
    mean_return, var_return from the simulated terminal distribution.

Usage::

    engine = MCMCEngine(
        n_iterations=1000, burnin=500,
        n_paths=10000, seed=42,
    )
    result = engine.run(
        log_returns=log_returns,      # 1-D numpy array
        current_price=1250.0,
        p_up_markov=0.62,
        dt=0.4,                       # fraction of session remaining
    )
    print(result.p_up, result.p_down, result.elapsed_ms)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.logger import get_logger

logger = get_logger("mcmc_engine")

# Small value to prevent log(0) and division-by-zero
_EPS = 1e-9


@dataclass
class MCMCResult:
    """Output of a single MCMCEngine.run() call."""
    p_up: float
    p_down: float
    mean_return: float
    var_return: float
    mu_posterior_mean: float
    sigma_posterior_mean: float
    n_paths: int
    elapsed_ms: float

    @property
    def dominant_signal(self) -> str:
        if self.p_up > self.p_down:
            return "UP"
        if self.p_down > self.p_up:
            return "DOWN"
        return "NEUTRAL"


class MCMCEngine:
    """Bayesian MCMC + GBM path simulator for a single session horizon.

    Args:
        n_iterations: Total MH sampler iterations (including burn-in).
        burnin:       Number of initial samples to discard.
        n_paths:      Number of GBM paths to simulate per call.
        seed:         Optional random seed for reproducibility.
    """

    # Hyperparameters for the Normal prior on mu
    _PRIOR_MU_MEAN = 0.0
    _PRIOR_MU_STD = 0.1

    # Hyperparameters for the HalfNormal prior on sigma
    _PRIOR_SIGMA_SCALE = 0.02

    # MH proposal step sizes
    _PROPOSAL_MU_STD = 0.005
    _PROPOSAL_SIGMA_STD = 0.003

    def __init__(
        self,
        n_iterations: int = 1000,
        burnin: int = 500,
        n_paths: int = 10000,
        seed: int | None = None,
    ) -> None:
        if burnin >= n_iterations:
            raise ValueError("burnin must be less than n_iterations")
        self._n_iterations = n_iterations
        self._burnin = burnin
        self._n_paths = n_paths
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        log_returns: np.ndarray,
        current_price: float,
        p_up_markov: float,
        dt: float = 1.0,
    ) -> MCMCResult:
        """Run the full MCMC + simulation pipeline.

        Args:
            log_returns:  1-D array of observed log-returns (e.g. last 50
                          sessions), oldest first.
            current_price: Current session's last traded price.
            p_up_markov:  Markov-predicted P(next session is UP),
                          in [0, 1].  P(DOWN) = 1 - p_up_markov.
            dt:           Fraction of a full session remaining (0, 1].
                          e.g. 2h left out of 5.75h total ≈ 0.348.

        Returns:
            MCMCResult with probabilities and diagnostics.
        """
        t_start = time.perf_counter()

        if len(log_returns) < 3:
            return self._fallback_result(p_up_markov, t_start)

        # Phase 1: posterior sampling
        mu_samples, sigma_samples = self._metropolis_hastings(log_returns)

        # Phase 2: vectorized path simulation
        terminal_prices = self._simulate_paths(
            current_price=current_price,
            mu_samples=mu_samples,
            sigma_samples=sigma_samples,
            p_up_markov=p_up_markov,
            dt=dt,
        )

        # Phase 3: aggregate
        result = self._aggregate(
            terminal_prices=terminal_prices,
            current_price=current_price,
            mu_samples=mu_samples,
            sigma_samples=sigma_samples,
            t_start=t_start,
        )

        logger.debug(
            "MCMCEngine.run complete",
            extra={
                "p_up": round(result.p_up, 4),
                "p_down": round(result.p_down, 4),
                "elapsed_ms": round(result.elapsed_ms, 1),
                "n_returns": len(log_returns),
                "dt": round(dt, 3),
            },
        )
        return result

    # ------------------------------------------------------------------ #
    # Phase 1: Metropolis-Hastings sampler
    # ------------------------------------------------------------------ #

    def _metropolis_hastings(
        self, log_returns: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample posterior of (mu, sigma) given log_returns.

        Returns:
            (mu_samples, sigma_samples) -- each shape (n_posterior,)
            where n_posterior = n_iterations - burnin.
        """
        n = len(log_returns)
        obs_mean = np.mean(log_returns)
        obs_std = np.std(log_returns) if np.std(log_returns) > _EPS else self._PRIOR_SIGMA_SCALE

        # Initialise chain at MLE estimates
        mu_cur = obs_mean
        sigma_cur = max(obs_std, _EPS)

        log_post_cur = self._log_posterior(mu_cur, sigma_cur, log_returns, n)

        mu_samples = np.empty(self._n_iterations, dtype=np.float64)
        sigma_samples = np.empty(self._n_iterations, dtype=np.float64)

        for step in range(self._n_iterations):
            # Propose
            mu_prop = mu_cur + self._rng.normal(0.0, self._PROPOSAL_MU_STD)
            sigma_prop = sigma_cur + self._rng.normal(0.0, self._PROPOSAL_SIGMA_STD)

            if sigma_prop <= 0:
                mu_samples[step] = mu_cur
                sigma_samples[step] = sigma_cur
                continue

            log_post_prop = self._log_posterior(mu_prop, sigma_prop, log_returns, n)

            # Accept / reject
            log_alpha = log_post_prop - log_post_cur
            if np.log(self._rng.uniform() + _EPS) < log_alpha:
                mu_cur = mu_prop
                sigma_cur = sigma_prop
                log_post_cur = log_post_prop

            mu_samples[step] = mu_cur
            sigma_samples[step] = sigma_cur

        # Discard burn-in
        return mu_samples[self._burnin:], sigma_samples[self._burnin:]

    def _log_posterior(
        self,
        mu: float,
        sigma: float,
        log_returns: np.ndarray,
        n: int,
    ) -> float:
        """Log posterior = log likelihood + log prior."""
        if sigma <= 0:
            return -np.inf

        # Gaussian log likelihood: sum over observed returns
        log_lik = (
            -n * np.log(sigma + _EPS)
            - 0.5 * np.sum(((log_returns - mu) / (sigma + _EPS)) ** 2)
        )

        # Normal prior on mu
        log_prior_mu = -0.5 * ((mu - self._PRIOR_MU_MEAN) / self._PRIOR_MU_STD) ** 2

        # HalfNormal prior on sigma (log scale)
        log_prior_sigma = -0.5 * (sigma / self._PRIOR_SIGMA_SCALE) ** 2

        return float(log_lik + log_prior_mu + log_prior_sigma)

    # ------------------------------------------------------------------ #
    # Phase 2: vectorized GBM path simulation
    # ------------------------------------------------------------------ #

    def _simulate_paths(
        self,
        current_price: float,
        mu_samples: np.ndarray,
        sigma_samples: np.ndarray,
        p_up_markov: float,
        dt: float,
    ) -> np.ndarray:
        """Simulate terminal prices for each posterior sample.

        Each posterior sample (mu_i, sigma_i) generates one path per
        (paths_per_sample) draws.  The regime-switching drift blends
        the UP and DOWN regimes according to Markov probabilities:

            mu_regime = mu * (p_up - p_down)

        This ensures the drift is UP-biased when p_up > 0.5 and
        DOWN-biased otherwise.

        Returns:
            1-D array of terminal prices, length n_paths.
        """
        n_posterior = len(mu_samples)
        paths_per_sample = max(1, self._n_paths // n_posterior)
        total_paths = n_posterior * paths_per_sample

        p_down_markov = 1.0 - p_up_markov
        direction_factor = p_up_markov - p_down_markov  # in [-1, 1]

        # Broadcast posterior params across paths
        mu_rep = np.repeat(mu_samples, paths_per_sample)
        sigma_rep = np.repeat(sigma_samples, paths_per_sample)

        # Regime-adjusted drift
        mu_regime = mu_rep * direction_factor

        # GBM step: vectorized
        z = self._rng.standard_normal(total_paths)
        log_return_sim = (mu_regime - 0.5 * sigma_rep ** 2) * dt + sigma_rep * np.sqrt(dt) * z
        terminal_prices = current_price * np.exp(log_return_sim)

        return terminal_prices

    # ------------------------------------------------------------------ #
    # Phase 3: aggregate results
    # ------------------------------------------------------------------ #

    def _aggregate(
        self,
        terminal_prices: np.ndarray,
        current_price: float,
        mu_samples: np.ndarray,
        sigma_samples: np.ndarray,
        t_start: float,
    ) -> MCMCResult:
        n = len(terminal_prices)
        p_up = float(np.sum(terminal_prices > current_price)) / n
        p_down = float(np.sum(terminal_prices < current_price)) / n

        log_ret = np.log(terminal_prices / (current_price + _EPS) + _EPS)
        mean_ret = float(np.mean(log_ret))
        var_ret = float(np.var(log_ret))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return MCMCResult(
            p_up=p_up,
            p_down=p_down,
            mean_return=mean_ret,
            var_return=var_ret,
            mu_posterior_mean=float(np.mean(mu_samples)),
            sigma_posterior_mean=float(np.mean(sigma_samples)),
            n_paths=n,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------ #
    # Fallback when insufficient data
    # ------------------------------------------------------------------ #

    def _fallback_result(self, p_up_markov: float, t_start: float) -> MCMCResult:
        """Return a neutral / Markov-only result when data is insufficient."""
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        p_down = 1.0 - p_up_markov
        logger.warning("MCMCEngine: insufficient data, falling back to Markov-only probabilities")
        return MCMCResult(
            p_up=p_up_markov,
            p_down=p_down,
            mean_return=0.0,
            var_return=0.0,
            mu_posterior_mean=0.0,
            sigma_posterior_mean=self._PRIOR_SIGMA_SCALE,
            n_paths=0,
            elapsed_ms=elapsed_ms,
        )
