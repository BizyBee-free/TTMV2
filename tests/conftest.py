"""Pytest defaults: keep optional DNSE network behaviors off unless a test opts in."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_vn30_auto_contract_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VN30_AUTO_RESOLVE_TRADE_SYMBOL", "false")


@pytest.fixture(autouse=True)
def _clear_vn30_f1m_trade_symbol_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tránh .env máy dev làm lệch SYMBOL_MAP trong unit test (trừ khi test tự setenv)."""
    monkeypatch.delenv("VN30_F1M_TRADE_SYMBOL", raising=False)


@pytest.fixture(autouse=True)
def _reset_vn30_resolution_flag() -> None:
    import src.config as _cfg

    _cfg._VN30_F1M_RESOLVED = False
    yield
    _cfg._VN30_F1M_RESOLVED = False
