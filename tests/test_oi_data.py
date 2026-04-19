"""DNSE secdef OI parsing and forward-fill alignment."""

from __future__ import annotations

from src.backtest.data_fetcher import OhlcBar
from src.hmm.oi_data import (
    align_open_interest_to_bars,
    deep_scan_open_interest_quantity,
    norm_unix_ts_sec,
    parse_open_interest_from_secdef_payload,
)


def test_parse_oi_camel_case():
    d = {"symbol": "41I1G4000", "openInterestQuantity": 12345}
    assert parse_open_interest_from_secdef_payload(d) == 12345


def test_parse_oi_snake_case():
    d = {"open_interest_quantity": 999}
    assert parse_open_interest_from_secdef_payload(d) == 999


def test_parse_oi_nested_data():
    d = {"data": {"OpenInterestQuantity": 42}}
    assert parse_open_interest_from_secdef_payload(d) == 42


def test_parse_oi_nested_security_definition():
    d = {"securityDefinition": {"openInterestQuantity": 777}}
    assert parse_open_interest_from_secdef_payload(d) == 777


def test_deep_scan_total_open_interest_key():
    d = {"result": {"rows": [{"symbol": "41I1G4000", "totalOpenInterest": 88_888}]}}
    assert parse_open_interest_from_secdef_payload(d) == 88_888
    assert deep_scan_open_interest_quantity({"a": {"nestedOpenInterest": 42}}) == 42


def test_parse_list_first_empty_deep_scan_rest():
    d = [{"x": 1}, {"openInterestQuantity": 333}]
    assert parse_open_interest_from_secdef_payload(d) == 333


def test_norm_unix_ts_ms_to_sec():
    assert norm_unix_ts_sec(1_700_000_000) == 1_700_000_000
    assert norm_unix_ts_sec(1_700_000_000_000) == 1_700_000_000


def test_align_forward_fill():
    bars = [
        OhlcBar("X", "202601010900", 1, 1, 1, 1, 0, unix_ts=100),
        OhlcBar("X", "202601010915", 1, 1, 1, 1, 0, unix_ts=200),
        OhlcBar("X", "202601010930", 1, 1, 1, 1, 0, unix_ts=300),
    ]
    pts = [(100, 10.0), (250, 20.0)]
    oi = align_open_interest_to_bars(bars, pts)
    assert oi[0] == 10.0
    assert oi[1] == 10.0  # still before 250
    assert oi[2] == 20.0


def test_align_bar_ms_matches_cache_sec():
    """Bar unix_ts dạng ms phải khớp điểm cache (giây)."""
    bars = [
        OhlcBar("X", "t", 1, 1, 1, 1, 0, unix_ts=1_700_000_000_000),
    ]
    pts = [(1_700_000_000, 55.0)]
    oi = align_open_interest_to_bars(bars, pts)
    assert oi[0] == 55.0
