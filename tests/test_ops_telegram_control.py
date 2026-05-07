"""Ops Telegram command handling (no network)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.ops.control_state import ControlStateManager
from src.ops.telegram_control import OpsControlBot


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("DNSE_API_KEY", "k")
    monkeypatch.setenv("DNSE_API_SECRET", "s")
    monkeypatch.setenv("DNSE_ACCOUNT_NO", "1")
    monkeypatch.setenv("CONTROL_TRADE_ENABLED_DEFAULT", "false")
    from src.config import Settings

    return Settings()


def _make_bot(settings, tmp_path):
    ctrl = ControlStateManager(settings=settings, state_path=tmp_path / "c.json")
    ctrl.enable_trading("test")
    audit = MagicMock()
    alerts = MagicMock()
    notifier = MagicMock()
    notifier.is_chat_allowed = MagicMock(return_value=True)
    notifier.send_message = MagicMock(return_value=True)
    bot = OpsControlBot(
        control=ctrl,
        audit=audit,
        alerts=alerts,
        notifier=notifier,
        settings=settings,
        runner=None,
        otp_manager=None,
    )
    return bot, ctrl, audit, notifier


def test_pause_resume(settings, tmp_path):
    bot, ctrl, _, _ = _make_bot(settings, tmp_path)
    bot._handle_command("1", "/pause")
    assert not ctrl.can_open_new_position()
    bot._handle_command("1", "/resume")
    assert ctrl.can_open_new_position()


def test_flatten_requires_confirm(settings, tmp_path):
    bot, ctrl, _, n = _make_bot(settings, tmp_path)
    bot._handle_command("1", "/flatten")
    assert n.send_message.called
    # without confirm, flatten not set
    assert not ctrl.get_state().force_flatten_requested


def test_kill_requires_confirm(settings, tmp_path):
    bot, ctrl, _, n = _make_bot(settings, tmp_path)
    bot._handle_command("1", "/kill")
    assert n.send_message.called
    assert not ctrl.get_state().killed
