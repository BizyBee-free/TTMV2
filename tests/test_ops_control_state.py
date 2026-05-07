"""ControlStateManager persistence tests."""

from __future__ import annotations

import json

import pytest

from src.ops.control_state import ControlStateManager


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DNSE_API_KEY", "k")
    monkeypatch.setenv("DNSE_API_SECRET", "s")
    monkeypatch.setenv("DNSE_ACCOUNT_NO", "1")
    monkeypatch.setenv("CONTROL_TRADE_ENABLED_DEFAULT", "false")
    from src.config import Settings

    return Settings()


def test_default_trade_off(settings, tmp_path):
    p = tmp_path / "control_state.json"
    m = ControlStateManager(settings=settings, state_path=p)
    assert not m.can_send_order()
    assert not m.can_open_new_position()


def test_resume_opens_trading(settings, tmp_path):
    p = tmp_path / "control_state.json"
    m = ControlStateManager(settings=settings, state_path=p)
    m.set_resume("test")
    assert m.can_send_order()
    assert m.can_open_new_position()


def test_pause_blocks_new_only(settings, tmp_path):
    p = tmp_path / "control_state.json"
    m = ControlStateManager(settings=settings, state_path=p)
    m.enable_trading("t")
    m.set_pause("p", "t")
    assert m.can_send_order()
    assert not m.can_open_new_position()


def test_kill_blocks_all(settings, tmp_path):
    p = tmp_path / "control_state.json"
    m = ControlStateManager(settings=settings, state_path=p)
    m.enable_trading("t")
    m.kill("t")
    assert not m.can_send_order()
    assert not m.can_open_new_position()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["killed"] is True
