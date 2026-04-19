"""Tests for balance parsing and API health."""

import pytest

from src.api_health import ApiHealthMonitor
from src.balance_utils import extract_balance_vnd, validate_balance_response


def test_extract_balance_flat_dict():
    assert extract_balance_vnd({"netAssetValue": 1_000_000.0}) == 1_000_000.0
    assert extract_balance_vnd({"data": {"cashBalance": 500}}) == 500.0


def test_validate_balance_timeout():
    ok, bal, reason = validate_balance_response(
        {"status": 408, "data": {"code": "CLIENT_TIMEOUT"}, "elapsed_ms": 100.0}
    )
    assert not ok
    assert bal is None
    assert reason == "CLIENT_TIMEOUT"


def test_validate_balance_ok():
    ok, bal, reason = validate_balance_response(
        {"status": 200, "data": {"netAssetValue": 1e9}, "elapsed_ms": 50.0}
    )
    assert ok
    assert bal == 1e9
    assert reason == "OK"


def test_api_health_halt_once():
    halted = []

    def on_halt(msg: str) -> None:
        halted.append(msg)

    m = ApiHealthMonitor(threshold=2)
    m.on_halt(on_halt)
    assert not m.record_failure("a")
    assert m.record_failure("b")
    assert len(halted) == 1
    assert m.consecutive_failures == 0
