"""OI cache từ WebSocket sec_def."""

from __future__ import annotations

from src.hmm.oi_ws_cache import (
    clear_open_interest_cache,
    get_latest_open_interest,
    record_open_interest_from_ws,
)


def setup_function() -> None:
    clear_open_interest_cache()


def test_record_and_get_by_trade_and_data_symbol() -> None:
    record_open_interest_from_ws("41I1G4000", 50_000)
    assert get_latest_open_interest("41I1G4000") == 50_000
    assert get_latest_open_interest("VN30F1M") == 50_000


def test_security_definition_from_dict_camel_open_interest() -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "vendor" / "dnse"))
    from trading_websocket.models import SecurityDefinition

    d = {
        "symbol": "41I1G4000",
        "openInterestQuantity": 42_000,
        "basic_price": 100.0,
        "ceiling_price": 110.0,
        "floor_price": 90.0,
    }
    sd = SecurityDefinition.from_dict(d)
    assert sd.openInterestQuantity == 42_000


def test_security_definition_sd_stream_bytes_t_and_snake_oi() -> None:
    """openapi-sdk: T=sd + open_interest_quantity; msgpack có thể gửi T dạng bytes."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "vendor" / "dnse"))
    from trading_websocket.models import SecurityDefinition

    d = {
        "T": b"sd",
        "symbol": "41I1G4000",
        "open_interest_quantity": 99_001,
        "basic_price": 100.0,
        "ceiling_price": 110.0,
        "floor_price": 90.0,
    }
    sd = SecurityDefinition.from_dict(d)
    assert sd.openInterestQuantity == 99_001
