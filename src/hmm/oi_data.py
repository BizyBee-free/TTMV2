"""Open interest (OI) from DNSE security definition + per-bar alignment for HMM.

DNSE OpenAPI exposes contract OI on ``GET /price/{symbol}/secdef`` (REST + Python
client mirror `dnse-tech/openapi-sdk`: ``python/trading-api/get_security_definition.py``,
``python/dnse/... get_security_definition``). Real-time OI uses WebSocket
``security_definition.{board}.msgpack`` with stream type ``T=sd`` (see
``python/websocket-marketdata/sec_def.py`` and ``trading_websocket/models.py``
``SecurityDefinition`` in the SDK: https://github.com/dnse-tech/openapi-sdk ).

The JSON field is typically ``openInterestQuantity`` (camelCase) or
``open_interest_quantity`` (snake_case). The vendored websocket model
``SecurityDefinition`` maps this to ``openInterestQuantity`` — contract open
interest, not order book depth.

OHLC history does not include OI per bar; we persist ``(unix_ts, oi)`` in a JSONL
cache and forward-fill OI onto each bar for walk-forward features.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.backtest.data_fetcher import OhlcBar
from src.logger import get_logger

logger = get_logger("hmm_oi_data")


def norm_unix_ts_sec(ts: Union[int, float]) -> int:
    """DNSE có thể trả epoch giây hoặc ms; chuẩn hóa về giây để căn OI với cache."""
    t = int(ts)
    if t > 10_000_000_000:
        return t // 1000
    return t


def deep_scan_open_interest_quantity(obj: Any, depth: int = 0) -> Optional[int]:
    """Fallback: tìm số nguyên ≥0 trong cây JSON có key gợi ý OI (DNSE đổi tên field thỉnh thoảng)."""
    if depth > 14 or obj is None:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            low = ks.lower()
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if (
                    "openinterest" in low
                    or low in ("oi", "oiqty", "oiquantity", "totaloi", "totalopeninterest")
                ):
                    try:
                        iv = int(float(v))
                        if iv >= 0:
                            return iv
                    except (TypeError, ValueError):
                        pass
            got = deep_scan_open_interest_quantity(v, depth + 1)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for item in obj[:50]:
            got = deep_scan_open_interest_quantity(item, depth + 1)
            if got is not None:
                return got
    return None


def parse_open_interest_from_secdef_payload(payload: Any) -> Optional[int]:
    """Extract OI from secdef JSON (dict or first element of list).

    Matches DNSE / OpenAPI naming used in ``vendor/dnse/trading_websocket/models.py``
    ``SecurityDefinition.openInterestQuantity``.
    """
    if payload is None:
        return None
    if isinstance(payload, list) and len(payload) > 0:
        got = parse_open_interest_from_secdef_payload(payload[0])
        if got is not None:
            return got
        return deep_scan_open_interest_quantity(payload)
    if not isinstance(payload, dict):
        return None
    for key in (
        "openInterestQuantity",
        "open_interest_quantity",
        "OpenInterestQuantity",
    ):
        if key in payload and payload[key] is not None:
            try:
                return int(float(payload[key]))
            except (TypeError, ValueError):
                return None
    # Một số payload REST lồng object con
    for nested_key in ("securityDefinition", "SecurityDefinition", "secdef"):
        inner_obj = payload.get(nested_key)
        if isinstance(inner_obj, (dict, list)):
            got = parse_open_interest_from_secdef_payload(inner_obj)
            if got is not None:
                return got
    inner = payload.get("data")
    if isinstance(inner, (dict, list)):
        got = parse_open_interest_from_secdef_payload(inner)
        if got is not None:
            return got
    return deep_scan_open_interest_quantity(payload)


class OICache:
    """Append-only JSONL: one object per line ``{\"unix_ts\": int, \"oi\": float}``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, unix_ts: int, oi: float) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"unix_ts": int(unix_ts), "oi": float(oi)}) + "\n")

    def load_points(self) -> List[Tuple[int, float]]:
        """All points sorted by ``unix_ts``; duplicate timestamps keep last occurrence."""
        if not self._path.exists():
            return []
        by_ts: Dict[int, float] = {}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    by_ts[int(obj["unix_ts"])] = float(obj["oi"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning("OICache read failed", extra={"path": str(self._path), "error": str(e)})
            return []
        return sorted(by_ts.items(), key=lambda x: x[0])


def align_open_interest_to_bars(
    bars: Sequence[OhlcBar],
    points: Sequence[Tuple[int, float]],
) -> List[float]:
    """Forward-fill OI: for each bar use latest OI with ``point_ts <= bar.unix_ts``."""
    if not bars:
        return []
    if not points:
        return [0.0] * len(bars)
    ts_list = [p[0] for p in points]
    vals = [p[1] for p in points]
    out: List[float] = []
    for b in bars:
        ts = norm_unix_ts_sec(int(getattr(b, "unix_ts", 0) or 0))
        i = bisect.bisect_right(ts_list, ts) - 1
        if i < 0:
            out.append(0.0)
        else:
            out.append(float(vals[i]))
    return out


def build_open_interest_series_for_live(
    bars: List[OhlcBar],
    cache: OICache,
    current_oi: Optional[int],
) -> List[float]:
    """Merge cache + optional current OI at last bar timestamp, then align to ``bars``."""
    pts = list(cache.load_points())
    if bars and current_oi is not None:
        last_ts = norm_unix_ts_sec(int(getattr(bars[-1], "unix_ts", 0) or 0))
        pts.append((last_ts, float(current_oi)))
        pts.sort(key=lambda x: x[0])
        # de-dupe same ts (keep last)
        by_ts: Dict[int, float] = {}
        for t, v in pts:
            by_ts[t] = v
        pts = sorted(by_ts.items(), key=lambda x: x[0])
    return align_open_interest_to_bars(bars, pts)
