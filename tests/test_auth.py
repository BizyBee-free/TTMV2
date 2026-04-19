"""Tests for authentication and DNSE client wrapper."""

import json
import pytest
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dnse_client import BeeTradeClient, TradingTokenManager
from src.config import Settings


@pytest.fixture
def mock_settings():
    """Create test settings without requiring .env file."""
    with patch.dict(os.environ, {
        "DNSE_API_KEY": "test-key",
        "DNSE_API_SECRET": "test-secret",
        "DNSE_ACCOUNT_NO": "0001000115",
        "PAPER_MODE": "true",
    }):
        return Settings()


@pytest.fixture
def mock_sdk():
    """Create a mocked DNSEClient."""
    sdk = MagicMock()
    sdk.get_accounts.return_value = (200, json.dumps([{"accountNo": "0001000115"}]))
    sdk.get_balances.return_value = (200, json.dumps({"balance": 10000000}))
    sdk.send_email_otp.return_value = (200, json.dumps({"message": "OK"}))
    sdk.create_trading_token.return_value = (
        200,
        json.dumps({"tradingToken": "test-token-123"}),
    )
    return sdk


class TestTradingTokenManager:
    def test_initial_state(self, mock_sdk):
        mgr = TradingTokenManager(mock_sdk)
        assert mgr.token is None
        assert not mgr.is_valid

    def test_request_otp(self, mock_sdk):
        mgr = TradingTokenManager(mock_sdk)
        status, _ = mgr.request_otp()
        assert status == 200
        mock_sdk.send_email_otp.assert_called_once()

    def test_activate_token(self, mock_sdk):
        mgr = TradingTokenManager(mock_sdk)
        token = mgr.activate_token("123456")
        assert token == "test-token-123"
        assert mgr.is_valid
        mock_sdk.create_trading_token.assert_called_once_with(
            otp_type="email_otp", passcode="123456"
        )

    def test_set_token_manually(self, mock_sdk):
        mgr = TradingTokenManager(mock_sdk)
        mgr.set_token("manual-token")
        assert mgr.token == "manual-token"
        assert mgr.is_valid


class TestBeeTradeClient:
    @patch("src.dnse_client.DNSEClient")
    def test_get_accounts(self, MockSDK, mock_settings):
        instance = MockSDK.return_value
        instance.get_accounts.return_value = (
            200,
            json.dumps([{"accountNo": "0001000115"}]),
        )

        client = BeeTradeClient(mock_settings)
        result = client.get_accounts()

        assert result["status"] == 200
        assert isinstance(result["data"], list)
        assert result["elapsed_ms"] >= 0

    @patch("src.dnse_client.DNSEClient")
    def test_get_balances(self, MockSDK, mock_settings):
        instance = MockSDK.return_value
        instance.get_balances.return_value = (
            200,
            json.dumps({"balance": 5000000}),
        )

        client = BeeTradeClient(mock_settings)
        result = client.get_balances()

        assert result["status"] == 200
        assert result["data"]["balance"] == 5000000

    @patch("src.dnse_client.DNSEClient")
    def test_api_error_handling(self, MockSDK, mock_settings):
        instance = MockSDK.return_value
        instance.get_accounts.return_value = (
            401,
            json.dumps({"code": "OA-101", "message": "Unauthorized"}),
        )

        client = BeeTradeClient(mock_settings)
        result = client.get_accounts()

        assert result["status"] == 401
        assert result["data"]["code"] == "OA-101"
