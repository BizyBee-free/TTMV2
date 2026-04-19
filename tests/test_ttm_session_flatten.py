"""Unit tests for VN session-end flatten window."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.session_flatten import session_flatten_bar_unix_ts
from src.vn_time import get_vn_tzinfo


def _ts_vn(y: int, month: int, day: int, h: int, m: int) -> int:
    tz = get_vn_tzinfo()
    return int(datetime(y, month, day, h, m, 0, tzinfo=tz).timestamp())


def test_session_flatten_disabled_by_default() -> None:
    ts = _ts_vn(2024, 6, 3, 14, 39)
    assert session_flatten_bar_unix_ts(ts, TTM_CONFIG) is False


def test_session_flatten_morning_window() -> None:
    cfg = {
        **TTM_CONFIG,
        "session_flatten_enabled": True,
        "session_flatten_vn_morning_hhmm": "11:25",
        "session_flatten_vn_afternoon_hhmm": "14:38",
    }
    assert session_flatten_bar_unix_ts(_ts_vn(2024, 6, 3, 10, 0), cfg) is False
    assert session_flatten_bar_unix_ts(_ts_vn(2024, 6, 3, 11, 26), cfg) is True
    assert session_flatten_bar_unix_ts(_ts_vn(2024, 6, 3, 13, 30), cfg) is False


def test_session_flatten_afternoon_window() -> None:
    cfg = {
        **TTM_CONFIG,
        "session_flatten_enabled": True,
        "session_flatten_vn_morning_hhmm": "11:25",
        "session_flatten_vn_afternoon_hhmm": "14:38",
    }
    assert session_flatten_bar_unix_ts(_ts_vn(2024, 6, 3, 14, 37), cfg) is False
    assert session_flatten_bar_unix_ts(_ts_vn(2024, 6, 3, 14, 38), cfg) is True


@pytest.mark.parametrize("unix_ts", [0, -1])
def test_session_flatten_invalid_ts(unix_ts: int) -> None:
    cfg = {**TTM_CONFIG, "session_flatten_enabled": True}
    assert session_flatten_bar_unix_ts(unix_ts, cfg) is False
