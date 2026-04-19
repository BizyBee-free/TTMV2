"""Unit tests for VN30 KRX front-month resolution (calendar + /instruments parsing)."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config as cfg


def test_calendar_krx_matches_repo_examples() -> None:
    assert cfg.calendar_krx_vn30_index_symbol_for_expiry(2026, 4) == "41I1G4000"
    assert cfg.calendar_krx_vn30_index_symbol_for_expiry(2026, 5) == "41I1G5000"
    assert cfg.calendar_krx_vn30_index_symbol_for_expiry(2026, 6) == "41I1G6000"
    assert cfg.calendar_krx_vn30_index_symbol_for_expiry(2026, 10) == "41I1G1000"
    assert cfg.calendar_krx_vn30_index_symbol_for_expiry(2026, 1) == "41I1G0100"


def test_third_thursday_april_2026() -> None:
    d = cfg._third_thursday(2026, 4)
    assert d.weekday() == 3  # Thursday
    assert 15 <= d.day <= 21


def test_resolve_updates_when_api_matches_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg._VN30_F1M_RESOLVED = False
    cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] = "41I1G4000"

    vn_tz = timezone(timedelta(hours=7))

    def _fake_vn_now() -> datetime:
        # Sau đáo hạn tháng 4/2026 → front tháng 5, calendar = 41I1G5000
        return datetime(2026, 4, 20, 10, 0, 0, tzinfo=vn_tz)

    monkeypatch.setattr(cfg, "vn_now", _fake_vn_now)

    client = MagicMock()
    client.settings = MagicMock()
    client.settings.VN30_AUTO_RESOLVE_TRADE_SYMBOL = True
    client.get_instruments.return_value = {
        "status": 200,
        "data": {
            "rows": [
                {"symbol": "41I1G4000", "lastTradingDate": "2026-04-16", "totalVolumeTraded": 1e9},
                {"symbol": "41I1G5000", "lastTradingDate": "2026-05-21", "totalVolumeTraded": 1e6},
            ]
        },
    }

    cfg.ensure_vn30_f1m_trade_symbol_resolved(client)
    assert cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] == "41I1G5000"
    client.get_instruments.assert_called()


def test_resolve_exits_on_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg._VN30_F1M_RESOLVED = False
    cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] = "41I1G4000"

    vn_tz = timezone(timedelta(hours=7))
    monkeypatch.setattr(
        cfg,
        "vn_now",
        lambda: datetime(2026, 4, 20, 10, 0, 0, tzinfo=vn_tz),
    )

    client = MagicMock()
    client.settings = MagicMock()
    client.settings.VN30_AUTO_RESOLVE_TRADE_SYMBOL = True
    client.get_instruments.return_value = {
        "status": 200,
        "data": {
            "rows": [
                {"symbol": "41I1G6000", "lastTradingDate": "2026-06-18", "totalVolumeTraded": 1e12},
            ]
        },
    }

    with pytest.raises(SystemExit) as exc:
        cfg.ensure_vn30_f1m_trade_symbol_resolved(client)
    assert exc.value.code == 1


def test_resolve_skipped_when_disabled() -> None:
    cfg._VN30_F1M_RESOLVED = False
    cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] = "41I1G4000"
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.VN30_AUTO_RESOLVE_TRADE_SYMBOL = False
    cfg.ensure_vn30_f1m_trade_symbol_resolved(client)
    assert cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] == "41I1G4000"
    client.get_instruments.assert_not_called()


def test_env_trade_symbol_pins_without_api() -> None:
    cfg._VN30_F1M_RESOLVED = False
    cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] = "41I1G4000"
    client = MagicMock()
    client.settings = SimpleNamespace(
        VN30_F1M_TRADE_SYMBOL="41I1G5000",
        VN30_AUTO_RESOLVE_TRADE_SYMBOL=False,
    )
    client.get_instruments = MagicMock()
    cfg.ensure_vn30_f1m_trade_symbol_resolved(client)
    assert cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] == "41I1G5000"
    assert cfg._VN30_F1M_RESOLVED is True
    client.get_instruments.assert_not_called()


def test_env_trade_symbol_overrides_auto_resolve_no_api() -> None:
    cfg._VN30_F1M_RESOLVED = False
    cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] = "41I1G4000"
    client = MagicMock()
    client.settings = SimpleNamespace(
        VN30_F1M_TRADE_SYMBOL="41I1G5000",
        VN30_AUTO_RESOLVE_TRADE_SYMBOL=True,
    )
    client.get_instruments = MagicMock()
    cfg.ensure_vn30_f1m_trade_symbol_resolved(client)
    assert cfg.SYMBOL_MAP["VN30F1M"]["trade_symbol"] == "41I1G5000"
    client.get_instruments.assert_not_called()
