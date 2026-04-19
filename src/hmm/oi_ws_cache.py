"""Open interest từ WebSocket ``sec_def`` (DNSE REST ``/price/.../secdef`` có thể không trả OI cho phái sinh).

Paper/live enrich ưu tiên giá trị cache WS nếu đã nhận.
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, Optional

_lock = Lock()
_by_symbol: Dict[str, int] = {}


def _norm(s: str) -> str:
    return (s or "").strip().upper()


def record_open_interest_from_ws(symbol: str, oi: int) -> None:
    """Gọi từ MarketDataManager khi có SecurityDefinition (mọi alias symbol)."""
    sym = _norm(symbol)
    if not sym:
        return
    try:
        v = int(oi)
    except (TypeError, ValueError):
        return
    if v < 0:
        return
    with _lock:
        _by_symbol[sym] = v
        try:
            from src.config import resolve_symbol_profile

            p = resolve_symbol_profile(sym)
            for k in (p.get("data_symbol"), p.get("trade_symbol")):
                kk = _norm(str(k or ""))
                if kk:
                    _by_symbol[kk] = v
        except Exception:
            pass


def get_latest_open_interest(symbol: str) -> Optional[int]:
    """Trả về OI mới nhất từ WS cho symbol / alias (41… hoặc VN30F*); None nếu chưa có."""
    sym = _norm(symbol)
    if not sym:
        return None
    with _lock:
        if sym in _by_symbol:
            return _by_symbol[sym]
        try:
            from src.config import resolve_symbol_profile

            p = resolve_symbol_profile(sym)
            for k in (p.get("data_symbol"), p.get("trade_symbol")):
                kk = _norm(str(k or ""))
                if kk and kk in _by_symbol:
                    return _by_symbol[kk]
        except Exception:
            pass
        return None


def clear_open_interest_cache() -> None:
    """Chủ yếu cho test."""
    global _by_symbol
    with _lock:
        _by_symbol.clear()
