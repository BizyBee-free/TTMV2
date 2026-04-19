"""Unit tests for MarkovModel (2-state Markov Chain)."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.mcmc.markov_model import MarkovModel, MarkovState


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_candle(open_p: float, close_p: float):
    c = MagicMock()
    c.open = open_p
    c.close = close_p
    return c


def _feed_alternating(model: MarkovModel, n: int) -> None:
    """Feed alternating UP/DOWN sessions."""
    for i in range(n):
        if i % 2 == 0:
            model.update(100.0, 101.0)  # UP
        else:
            model.update(101.0, 100.0)  # DOWN


# ── construction ──────────────────────────────────────────────────────────────

class TestMarkovModelConstruction:
    def test_default_window(self):
        m = MarkovModel()
        assert m.window == 50

    def test_custom_window(self):
        m = MarkovModel(window=30)
        assert m.window == 30

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError, match="window must be"):
            MarkovModel(window=2)

    def test_initial_state(self):
        m = MarkovModel(window=10)
        assert m.n_sessions == 0
        assert not m.is_ready
        assert m.last_state is None


# ── update / state classification ─────────────────────────────────────────────

class TestMarkovModelUpdate:
    def test_up_state(self):
        m = MarkovModel(window=10)
        state = m.update(100.0, 101.0)
        assert state == MarkovState.UP

    def test_down_state(self):
        m = MarkovModel(window=10)
        state = m.update(101.0, 100.0)
        assert state == MarkovState.DOWN

    def test_equal_prices_is_down(self):
        m = MarkovModel(window=10)
        state = m.update(100.0, 100.0)
        assert state == MarkovState.DOWN

    def test_session_count_grows(self):
        m = MarkovModel(window=10)
        for i in range(7):
            m.update(100.0, 101.0 if i % 2 == 0 else 99.0)
        assert m.n_sessions == 6  # N states -> N-1 transitions

    def test_rolling_window_drops_old(self):
        m = MarkovModel(window=5)
        for i in range(20):
            m.update(100.0, 101.0)
        # deque maxlen = window+1 = 6, so n_sessions <= 5
        assert m.n_sessions <= 5

    def test_last_state_tracked(self):
        m = MarkovModel(window=10)
        m.update(100.0, 101.0)
        assert m.last_state == MarkovState.UP
        m.update(101.0, 100.0)
        assert m.last_state == MarkovState.DOWN


# ── feed_ohlc_history ─────────────────────────────────────────────────────────

class TestFeedOhlcHistory:
    def test_bulk_load(self):
        m = MarkovModel(window=50)
        candles = [_make_candle(100.0, 101.0) for _ in range(20)]
        loaded = m.feed_ohlc_history(candles)
        assert loaded == 20
        assert m.n_sessions == 19

    def test_invalid_candles_skipped(self):
        m = MarkovModel(window=50)
        bad = MagicMock()
        bad.open = "bad"
        bad.close = "data"
        candles = [bad, _make_candle(100.0, 101.0)]
        loaded = m.feed_ohlc_history(candles)
        assert loaded == 1

    def test_respects_window(self):
        m = MarkovModel(window=5)
        candles = [_make_candle(100.0, 101.0) for _ in range(30)]
        m.feed_ohlc_history(candles)
        assert m.n_sessions <= 5


# ── transition matrix ─────────────────────────────────────────────────────────

class TestTransitionMatrix:
    def test_all_up_matrix(self):
        m = MarkovModel(window=50)
        for _ in range(20):
            m.update(100.0, 101.0)
        stats = m.get_stats()
        # All UP->UP, no DOWN
        assert stats.matrix[MarkovState.UP][MarkovState.UP] == pytest.approx(1.0)

    def test_all_down_matrix(self):
        m = MarkovModel(window=50)
        for _ in range(20):
            m.update(101.0, 100.0)
        stats = m.get_stats()
        assert stats.matrix[MarkovState.DOWN][MarkovState.DOWN] == pytest.approx(1.0)

    def test_alternating_matrix(self):
        m = MarkovModel(window=50)
        _feed_alternating(m, 20)
        stats = m.get_stats()
        # UP->DOWN and DOWN->UP should be high
        assert stats.matrix[MarkovState.UP][MarkovState.DOWN] > 0.8
        assert stats.matrix[MarkovState.DOWN][MarkovState.UP] > 0.8

    def test_matrix_rows_sum_to_one(self):
        m = MarkovModel(window=50)
        _feed_alternating(m, 20)
        stats = m.get_stats()
        for i in range(2):
            assert stats.matrix[i].sum() == pytest.approx(1.0, abs=1e-9)

    def test_unseen_state_defaults_to_uniform(self):
        m = MarkovModel(window=50)
        # Feed only UP sessions so DOWN state never seen as 'from' state
        for _ in range(20):
            m.update(100.0, 101.0)
        stats = m.get_stats()
        # DOWN row should be [0.5, 0.5] (uniform prior)
        assert stats.matrix[MarkovState.DOWN][MarkovState.UP] == pytest.approx(0.5)
        assert stats.matrix[MarkovState.DOWN][MarkovState.DOWN] == pytest.approx(0.5)


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_raises_when_not_ready(self):
        m = MarkovModel(window=50)
        with pytest.raises(RuntimeError, match="Not enough data"):
            m.predict()

    def test_probabilities_sum_to_one(self):
        m = MarkovModel(window=50)
        _feed_alternating(m, 20)
        p_up, p_down = m.predict()
        assert p_up + p_down == pytest.approx(1.0, abs=1e-9)

    def test_explicit_state(self):
        m = MarkovModel(window=50)
        _feed_alternating(m, 20)
        p_up_from_up, p_down_from_up = m.predict(current_state=MarkovState.UP)
        p_up_from_down, p_down_from_down = m.predict(current_state=MarkovState.DOWN)
        # Alternating sequence: from UP, next is DOWN
        assert p_down_from_up > p_up_from_up
        # From DOWN, next is UP
        assert p_up_from_down > p_down_from_down

    def test_uses_last_state_by_default(self):
        m = MarkovModel(window=50)
        _feed_alternating(m, 20)
        last = m.last_state
        p1 = m.predict(current_state=last)
        p2 = m.predict()
        assert p1[0] == pytest.approx(p2[0])


# ── get_log_returns ────────────────────────────────────────────────────────────

class TestGetLogReturns:
    def test_basic(self):
        m = MarkovModel(window=50)
        candles = [_make_candle(100.0, 110.0), _make_candle(110.0, 99.0)]
        ret = m.get_log_returns(candles)
        assert len(ret) == 2
        assert ret[0] == pytest.approx(np.log(110.0 / 100.0))
        assert ret[1] == pytest.approx(np.log(99.0 / 110.0))

    def test_skips_zero_open(self):
        m = MarkovModel(window=50)
        candles = [_make_candle(0.0, 100.0), _make_candle(100.0, 110.0)]
        ret = m.get_log_returns(candles)
        assert len(ret) == 1

    def test_empty_returns_empty_array(self):
        m = MarkovModel(window=50)
        ret = m.get_log_returns([])
        assert len(ret) == 0


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_state(self):
        m = MarkovModel(window=50)
        _feed_alternating(m, 20)
        assert m.n_sessions > 0
        m.reset()
        assert m.n_sessions == 0
        assert not m.is_ready
        assert m.last_state is None
