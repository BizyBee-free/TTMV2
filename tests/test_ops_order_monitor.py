"""OrderMonitor timeouts and critical path."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.ops.order_monitor import OrderMonitor


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DNSE_API_KEY", "k")
    monkeypatch.setenv("DNSE_API_SECRET", "s")
    monkeypatch.setenv("DNSE_ACCOUNT_NO", "1")
    monkeypatch.setenv("ORDER_ACK_TIMEOUT_SEC", "5")
    monkeypatch.setenv("ORDER_FILL_TIMEOUT_SEC", "10")
    monkeypatch.setenv("PAUSE_ON_ORDER_TIMEOUT", "true")
    from src.config import Settings

    return Settings()


def test_ack_timeout_alerts_and_pause(settings):
    audit = MagicMock()
    alerts = MagicMock()
    control = MagicMock()
    om = OrderMonitor(settings=settings, audit=audit, alerts=alerts, control=control)
    om.register_order_submitted({"order_id": "O1", "ts": 1000.0})
    evs = om.check_timeouts(now=2000.0)
    assert any(e.get("event_type") == "order_ack_timeout" for e in evs)
    alerts.send_critical.assert_called()
    control.set_pause.assert_called()


def test_reject_invokes_register_order_reject(settings):
    audit = MagicMock()
    alerts = MagicMock()
    control = MagicMock()
    om = OrderMonitor(settings=settings, audit=audit, alerts=alerts, control=control)
    om.register_order_reject("Z1", {"reason": "bad"})
    alerts.send_critical.assert_called()
