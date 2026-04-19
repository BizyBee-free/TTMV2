"""Vietnam session calendar and Unix timestamp display (UTC+7).

DNSE OHLC returns ``unix_ts`` as **standard Unix epoch seconds** (same instant worldwide).
There is no separate "Vietnam Unix": ``datetime.fromtimestamp(ts, tz=UTC)`` and
``datetime.fromtimestamp(ts, tz=Asia/Ho_Chi_Minh)`` describe the **same** instant.

Historical confusion (~18h skew) was **not** from subtracting 18h in code; it came from:

1. **API range**: ``to`` previously ended at midnight UTC of ``to_date``, which cut off
   most intraday 15m bars for that calendar day (fixed in ``DataFetcher._fetch_from_api``).
2. **"Today" boundary**: using ``datetime.now(UTC).date()`` for live ``to_date`` can be
   **yesterday** in UTC while Vietnam is already in the next session day — wrong range/cache.
3. **Disk cache**: serving a snapshot from the first fetch of the day so ``last_bar_unix_ts``
   never advanced until cache was bypassed or deleted.

Use **Vietnam calendar** for live date ranges and **skip reading cache** when the request
includes today's VN date for intraday freshness.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Union


@lru_cache(maxsize=1)
def get_vn_tzinfo():
    """Asia/Ho_Chi_Minh, or fixed UTC+7 if ``tzdata`` is missing (Windows)."""
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Asia/Ho_Chi_Minh")
    except Exception:
        return timezone(timedelta(hours=7))


def vn_now() -> datetime:
    """Current local time in Vietnam."""
    return datetime.now(get_vn_tzinfo())


def vn_calendar_today_yyyymmdd() -> str:
    """Today's date string YYYYMMDD in Vietnam (for OHLC ``to_date`` / live range)."""
    return vn_now().strftime("%Y%m%d")


def unix_ts_to_vn_str(ts: Union[int, float]) -> str:
    """Format epoch seconds as Vietnam local wall-clock (for logs / PM review)."""
    if ts is None:
        return ""
    t = int(ts)
    if t <= 0:
        return ""
    return datetime.fromtimestamp(t, tz=get_vn_tzinfo()).strftime("%Y-%m-%d %H:%M:%S")


def unix_ts_to_utc_str(ts: Union[int, float]) -> str:
    """Format epoch seconds as UTC (for cross-check with bar_utc in logs)."""
    if ts is None:
        return ""
    t = int(ts)
    if t <= 0:
        return ""
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def unix_to_vietnam_time(ts: Union[int, float]) -> str:
    """Alias của :func:`unix_ts_to_vn_str` — hiển thị giờ tường VN từ epoch giây (UTC)."""
    return unix_ts_to_vn_str(ts)
