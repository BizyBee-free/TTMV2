"""OTPManager behavior (no real Telegram)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.ops.otp_manager import OTPManager, mask_otp


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DNSE_API_KEY", "k")
    monkeypatch.setenv("DNSE_API_SECRET", "s")
    monkeypatch.setenv("DNSE_ACCOUNT_NO", "1")
    monkeypatch.setenv("OTP_TELEGRAM_ENABLED", "true")
    from src.config import Settings

    return Settings()


def test_mask_otp():
    assert "****" in mask_otp("123456")


def test_otp_only_when_pending(settings):
    alerts = MagicMock()
    notifier = MagicMock()
    notifier.is_chat_allowed = MagicMock(return_value=True)
    om = OTPManager(settings=settings, alerts=alerts, notifier=notifier)
    assert not om.submit_otp_from_message("OTP123456", "1")


def test_otp_rejects_wrong_chat(settings):
    alerts = MagicMock()
    notifier = MagicMock()
    notifier.is_chat_allowed = MagicMock(return_value=False)
    om = OTPManager(settings=settings, alerts=alerts, notifier=notifier)
    # force pending
    om._pending = True  # noqa: SLF001
    om._deadline = 9e12  # noqa: SLF001
    assert not om.submit_otp_from_message("OTP999999", "bad")


def test_otp_accepts_when_pending(settings):
    alerts = MagicMock()
    notifier = MagicMock()
    notifier.is_chat_allowed = MagicMock(return_value=True)
    om = OTPManager(settings=settings, alerts=alerts, notifier=notifier)
    om._pending = True  # noqa: SLF001
    om._deadline = 9e12  # noqa: SLF001
    om._done.clear()  # noqa: SLF001
    assert om.submit_otp_from_message("OTP123456", "1")
