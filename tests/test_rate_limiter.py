"""Tests for SlidingWindowRateLimiter and its integration with BeeTradeClient."""

import json
import time

import pytest
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rate_limiter import SlidingWindowRateLimiter, RateLimitExceeded
from src.config import Settings


# ===========================================================================
#  SlidingWindowRateLimiter unit tests
# ===========================================================================

class TestSlidingWindowRateLimiter:
    def test_allows_up_to_limit(self):
        rl = SlidingWindowRateLimiter(limit=3, window=60.0)
        now = 1000.0
        rl.acquire(now=now)
        rl.acquire(now=now + 1)
        rl.acquire(now=now + 2)
        with patch("src.rate_limiter.time.time", return_value=now + 2):
            assert rl.count == 3

    def test_blocks_over_limit(self):
        rl = SlidingWindowRateLimiter(limit=3, window=60.0)
        now = 1000.0
        for i in range(3):
            rl.acquire(now=now + i)

        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.acquire(now=now + 3)

        assert exc_info.value.limit == 3
        assert exc_info.value.window == 60.0
        assert exc_info.value.retry_after > 0

    def test_window_slides(self):
        """After the window passes, old entries expire and new ones are allowed."""
        rl = SlidingWindowRateLimiter(limit=2, window=10.0)
        rl.acquire(now=100.0)
        rl.acquire(now=101.0)

        with pytest.raises(RateLimitExceeded):
            rl.acquire(now=105.0)

        # t=100 entry expires at t=110
        rl.acquire(now=110.1)
        with patch("src.rate_limiter.time.time", return_value=110.1):
            assert rl.count == 2  # t=101 + t=110.1

    def test_can_acquire_does_not_consume(self):
        rl = SlidingWindowRateLimiter(limit=1, window=60.0)
        assert rl.can_acquire(now=100.0) is True
        assert rl.can_acquire(now=100.0) is True  # still True, not consumed
        with patch("src.rate_limiter.time.time", return_value=100.0):
            assert rl.remaining == 1

        rl.acquire(now=100.0)
        assert rl.can_acquire(now=100.0) is False
        with patch("src.rate_limiter.time.time", return_value=100.0):
            assert rl.remaining == 0

    def test_remaining_property(self):
        rl = SlidingWindowRateLimiter(limit=5, window=60.0)
        assert rl.remaining == 5
        rl.acquire(now=100.0)
        rl.acquire(now=101.0)
        with patch("src.rate_limiter.time.time", return_value=101.0):
            assert rl.remaining == 3

    def test_reset(self):
        rl = SlidingWindowRateLimiter(limit=2, window=60.0)
        rl.acquire(now=100.0)
        rl.acquire(now=101.0)
        with patch("src.rate_limiter.time.time", return_value=101.0):
            assert rl.remaining == 0

        rl.reset()
        assert rl.remaining == 2
        assert rl.count == 0

    def test_retry_after(self):
        rl = SlidingWindowRateLimiter(limit=2, window=10.0)
        rl.acquire(now=100.0)
        rl.acquire(now=103.0)

        # Window full: oldest=100, expires at 110
        # Patch time.time to control retry_after
        with patch("src.rate_limiter.time.time", return_value=105.0):
            retry = rl.retry_after()
            assert retry == pytest.approx(5.0, abs=0.1)

    def test_retry_after_when_available(self):
        rl = SlidingWindowRateLimiter(limit=5, window=60.0)
        rl.acquire(now=100.0)
        with patch("src.rate_limiter.time.time", return_value=100.0):
            assert rl.retry_after() == 0.0

    def test_invalid_limit(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            SlidingWindowRateLimiter(limit=0, window=60.0)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window must be positive"):
            SlidingWindowRateLimiter(limit=5, window=0)

    def test_exception_message(self):
        rl = SlidingWindowRateLimiter(limit=1, window=60.0)
        rl.acquire(now=100.0)
        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.acquire(now=100.5)
        assert "1 actions per 60s" in str(exc_info.value)
        assert "Retry after" in str(exc_info.value)

    def test_burst_at_boundary(self):
        """All actions at exact same timestamp should respect limit."""
        rl = SlidingWindowRateLimiter(limit=3, window=60.0)
        t = 500.0
        rl.acquire(now=t)
        rl.acquire(now=t)
        rl.acquire(now=t)
        with pytest.raises(RateLimitExceeded):
            rl.acquire(now=t)


# ===========================================================================
#  Integration: BeeTradeClient + rate limiter
# ===========================================================================

class TestBeeTradeClientRateLimit:
    @pytest.fixture
    def mock_settings(self):
        with patch.dict(os.environ, {
            "DNSE_API_KEY": "test-key",
            "DNSE_API_SECRET": "test-secret",
            "DNSE_ACCOUNT_NO": "0001000115",
            "PAPER_MODE": "true",
            "MAX_ORDERS_PER_MINUTE": "3",
        }):
            yield Settings()

    @patch("src.dnse_client.DNSEClient")
    def test_rate_limiter_initialized_from_config(self, MockSDK, mock_settings):
        from src.dnse_client import BeeTradeClient
        client = BeeTradeClient(mock_settings)
        assert client.order_limiter.limit == 3
        assert client.order_limiter.window == 60.0

    @patch("src.dnse_client.DNSEClient")
    def test_place_order_blocked_by_rate_limit(self, MockSDK, mock_settings):
        instance = MockSDK.return_value
        instance.post_order.return_value = (200, json.dumps({"orderId": "123"}))

        from src.dnse_client import BeeTradeClient
        client = BeeTradeClient(mock_settings)
        client.token_manager.set_token("fake-token")

        for _ in range(3):
            client.place_order("HPG", "NB", "LO", 25.0, 100, 0)

        with pytest.raises(RateLimitExceeded):
            client.place_order("HPG", "NB", "LO", 25.0, 100, 0)

    @patch("src.dnse_client.DNSEClient")
    def test_modify_order_consumes_rate_limit(self, MockSDK, mock_settings):
        instance = MockSDK.return_value
        instance.put_order.return_value = (200, json.dumps({"status": "ok"}))

        from src.dnse_client import BeeTradeClient
        client = BeeTradeClient(mock_settings)
        client.token_manager.set_token("fake-token")

        for _ in range(3):
            client.modify_order("order-1", price=26.0)

        with pytest.raises(RateLimitExceeded):
            client.modify_order("order-2", price=27.0)

    @patch("src.dnse_client.DNSEClient")
    def test_cancel_order_consumes_rate_limit(self, MockSDK, mock_settings):
        instance = MockSDK.return_value
        instance.cancel_order.return_value = (200, json.dumps({"status": "ok"}))

        from src.dnse_client import BeeTradeClient
        client = BeeTradeClient(mock_settings)
        client.token_manager.set_token("fake-token")

        for _ in range(3):
            client.cancel_order(f"order-{_}")

        with pytest.raises(RateLimitExceeded):
            client.cancel_order("order-99")

    @patch("src.dnse_client.DNSEClient")
    def test_mixed_actions_share_limiter(self, MockSDK, mock_settings):
        """place + modify + cancel all count against the same limit."""
        instance = MockSDK.return_value
        instance.post_order.return_value = (200, json.dumps({"orderId": "1"}))
        instance.put_order.return_value = (200, json.dumps({"status": "ok"}))
        instance.cancel_order.return_value = (200, json.dumps({"status": "ok"}))

        from src.dnse_client import BeeTradeClient
        client = BeeTradeClient(mock_settings)
        client.token_manager.set_token("fake-token")

        client.place_order("HPG", "NB", "LO", 25.0, 100, 0)
        client.modify_order("order-1", price=26.0)
        client.cancel_order("order-1")

        assert client.order_limiter.remaining == 0

        with pytest.raises(RateLimitExceeded):
            client.place_order("HPG", "NB", "LO", 25.0, 100, 0)

    @patch("src.dnse_client.DNSEClient")
    def test_read_endpoints_not_rate_limited(self, MockSDK, mock_settings):
        """get_accounts, get_balances, etc. should NOT be rate limited."""
        instance = MockSDK.return_value
        instance.get_accounts.return_value = (200, json.dumps([]))
        instance.get_balances.return_value = (200, json.dumps({}))

        from src.dnse_client import BeeTradeClient
        client = BeeTradeClient(mock_settings)

        for _ in range(20):
            client.get_accounts()
            client.get_balances()

        assert client.order_limiter.remaining == 3  # untouched
