"""Ép đóng / chặn vào mới gần cuối phiên phái sinh VN (không giữ qua nghỉ trưa / qua đêm).

Cần ``unix_ts`` hợp lệ trên nến (epoch giây). Bật bằng ``session_flatten_enabled`` trong TTM config.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Optional

from src.vn_time import get_vn_tzinfo


def _parse_hhmm(s: Optional[str]) -> Optional[int]:
    """'HH:MM' hoặc 'HHMM' -> phút từ 00:00 trong ngày."""
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    t = t.replace(":", "")
    if len(t) != 4 or not t.isdigit():
        return None
    h, m = int(t[:2]), int(t[2:])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return h * 60 + m


def session_flatten_bar_unix_ts(unix_ts: int, config: Mapping[str, Any]) -> bool:
    """
    True nếu theo config cần coi là “sắp hết phiên”: ép thoát / không vào mới.

    - Buổi sáng: từ ``session_flatten_vn_morning_hhmm`` trở đi, trước 12:00 VN (trước phiên chiều).
    - Buổi chiều: từ ``session_flatten_vn_afternoon_hhmm`` trở đi cùng ngày.
    """
    if not bool(config.get("session_flatten_enabled", False)):
        return False
    tsi = int(unix_ts)
    if tsi <= 0:
        return False
    dt = datetime.fromtimestamp(tsi, tz=get_vn_tzinfo())
    m = dt.hour * 60 + dt.minute

    morn = _parse_hhmm(config.get("session_flatten_vn_morning_hhmm"))
    if morn is not None and m >= morn and dt.hour < 12:
        return True

    aft = _parse_hhmm(config.get("session_flatten_vn_afternoon_hhmm"))
    if aft is not None and m >= aft:
        return True

    return False
