"""Unit tests for MCMCEngine (Bayesian posterior sampling + GBM paths)."""

import time
import pytest
import numpy as np

from src.mcmc.mcmc_engine import MCMCEngine, MCMCResult


# ── fixtures ──────────────────────────────────────────────────────────────────

def _returns(n: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.001, 0.015, n)


def _engine(n_iter=200, burnin=100, n_paths=1000, seed=42) -> MCMCEngine:
    return MCMCEngine(n_iterations=n_iter, burnin=burnin, n_paths=n_paths, seed=seed)


# ── construction ──────────────────────────────────────────────────────────────

class TestMCMCEngineConstruction:
    def test_burnin_less_than_iterations(self):
        with pytest.raises(ValueError, match="burnin must be less than"):
            MCMCEngine(n_iterations=100, burnin=100)

    def test_default_params(self):
        e = MCMCEngine()
        assert e._n_iterations == 1000
        assert e._burnin == 500
        assert e._n_paths == 10000

    def test_custom_seed_reproducible(self):
        e1 = _engine(seed=7)
        e2 = _engine(seed=7)
        r1 = e1.run(_returns(), 1000.0, 0.6)
        r2 = e2.run(_returns(), 1000.0, 0.6)
        assert r1.p_up == pytest.approx(r2.p_up)
        assert r1.p_down == pytest.approx(r2.p_down)


# ── run() output structure ─────────────────────────────────────────────────────

class TestMCMCEngineRunOutput:
    def test_probabilities_sum_to_one_approx(self):
        e = _engine()
        r = e.run(_returns(), 1000.0, 0.6)
        # p_up + p_down <= 1.0 (some paths may be exactly at current price)
        assert r.p_up + r.p_down <= 1.0 + 1e-9
        assert r.p_up >= 0.0
        assert r.p_down >= 0.0

    def test_n_paths_positive(self):
        e = _engine()
        r = e.run(_returns(), 1000.0, 0.6)
        assert r.n_paths > 0

    def test_elapsed_ms_positive(self):
        e = _engine()
        r = e.run(_returns(), 1000.0, 0.6)
        assert r.elapsed_ms > 0.0

    def test_mu_sigma_reasonable(self):
        e = _engine()
        r = e.run(_returns(50, seed=1), 1000.0, 0.6)
        # Posterior mu should be small (log-return scale)
        assert abs(r.mu_posterior_mean) < 0.1
        # Posterior sigma should be positive and small
        assert 0 < r.sigma_posterior_mean < 0.5

    def test_dominant_signal_values(self):
        e = _engine(seed=99)
        r = e.run(_returns(), 1000.0, 0.7)
        assert r.dominant_signal in ("UP", "DOWN", "NEUTRAL")


# ── fallback when insufficient data ───────────────────────────────────────────

class TestMCMCEngineFallback:
    def test_too_few_returns_uses_markov(self):
        e = _engine()
        r = e.run(np.array([0.01, 0.02]), 1000.0, 0.65)
        assert r.n_paths == 0
        assert r.p_up == pytest.approx(0.65)
        assert r.p_down == pytest.approx(0.35)

    def test_empty_returns_fallback(self):
        e = _engine()
        r = e.run(np.array([]), 1000.0, 0.55)
        assert r.p_up == pytest.approx(0.55)


# ── direction sensitivity ─────────────────────────────────────────────────────

class TestMCMCEngineDirectionSensitivity:
    def test_high_p_up_markov_biases_up(self):
        """When Markov says 90% up, simulated P(UP) should exceed neutral."""
        e = _engine(seed=123)
        r_up = e.run(_returns(), 1000.0, 0.9, dt=0.5)
        e2 = _engine(seed=123)
        r_neutral = e2.run(_returns(), 1000.0, 0.5, dt=0.5)
        assert r_up.p_up >= r_neutral.p_up - 0.05  # allow small variance

    def test_high_p_down_markov_biases_down(self):
        e = _engine(seed=456)
        r_down = e.run(_returns(), 1000.0, 0.1, dt=0.5)  # p_up=0.1 -> strong down
        e2 = _engine(seed=456)
        r_neutral = e2.run(_returns(), 1000.0, 0.5, dt=0.5)
        assert r_down.p_down >= r_neutral.p_down - 0.05


# ── dt parameter ─────────────────────────────────────────────────────────────

class TestMCMCEngineDt:
    def test_dt_1_vs_dt_half(self):
        """Longer time horizon -> more variance in terminal prices."""
        e1 = _engine(seed=10)
        e2 = _engine(seed=10)
        r_full = e1.run(_returns(), 1000.0, 0.6, dt=1.0)
        r_half = e2.run(_returns(), 1000.0, 0.6, dt=0.5)
        # Larger dt -> larger variance in returns
        assert r_full.var_return >= r_half.var_return - 1e-6


# ── performance benchmark ─────────────────────────────────────────────────────

class TestMCMCEnginePerformance:
    def test_10k_paths_under_200ms(self):
        """Full 10k path run must complete < 200ms (target is 100ms, allow 2x margin)."""
        e = MCMCEngine(n_iterations=1000, burnin=500, n_paths=10000, seed=0)
        rets = _returns(50)
        t0 = time.perf_counter()
        r = e.run(rets, 1250.0, 0.62, dt=0.4)
        elapsed = (time.perf_counter() - t0) * 1000.0
        assert elapsed < 200.0, f"MCMC took {elapsed:.1f}ms (limit 200ms)"
        assert r.n_paths > 0
