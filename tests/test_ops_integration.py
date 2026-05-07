"""Integration-style tests for ops + broker hooks."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DNSE_API_KEY", "k")
    monkeypatch.setenv("DNSE_API_SECRET", "s")
    monkeypatch.setenv("DNSE_ACCOUNT_NO", "1")


def test_trading_token_uses_otp_provider(env):
    from src.dnse_client import TradingTokenManager

    sdk = MagicMock()
    sdk.create_trading_token.return_value = (200, json.dumps({"tradingToken": "abc"}))
    mgr = TradingTokenManager(sdk, otp_provider=lambda: "123456")
    tok = mgr.ensure_token()
    assert tok == "abc"


def test_ops_modules_isolated_from_ttm_signal():
    import src.ops.ops_controller as oc

    src = open(oc.__file__, "r", encoding="utf-8").read()
    assert "ttm_signal_v2" not in src
    assert "ttm_features" not in src
