"""Tests for ops TelegramAlertClient (mocked HTTP)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.ops.telegram_alerts import TelegramAlertClient, mask_secrets_in_text


@pytest.fixture
def env_min(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DNSE_API_KEY", "k")
    monkeypatch.setenv("DNSE_API_SECRET", "s")
    monkeypatch.setenv("DNSE_ACCOUNT_NO", "1")


def test_mask_secrets():
    t = mask_secrets_in_text("OTP123456 and token 12345678:ABC-DEFghijklmnopqrstuvwxyz1234567890AB")
    assert "OTP******" in t
    assert "<token>" in t


def test_send_message_disabled_no_crash(env_min, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TELEGRAM_ENABLED", "false")
    from src.config import Settings

    c = TelegramAlertClient(settings=Settings())
    assert c.send_message("hello") is False


@patch("src.telegram_notifier.urllib.request.urlopen")
def test_send_message_short_timeout(mock_urlopen, env_min, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TELEGRAM_ENABLED", "true")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=MagicMock(status=200))
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_ctx

    from src.config import Settings

    c = TelegramAlertClient(settings=Settings())
    assert c.send_message("x", level="INFO") is True
    call_kw = mock_urlopen.call_args[1]
    assert call_kw["timeout"] == 5.0
