"""Historical OHLC data fetcher using DNSE REST API.

Fetches daily or hourly OHLC bars for a symbol from the DNSE API
and caches results locally as JSON to avoid repeated network calls.

DNSE OHLC endpoint (discovered via API probing 2026-03-25):
    GET /price/ohlc?symbol=X&resolution={res}&type={asset_type}&from={unix_ts}&to={unix_ts}

Correct parameters:
    resolution : "1D"  (daily), "1H"  (hourly), "15" (15-min), "30" (30-min)
    type       : "stock"  -- for individual stocks (HPG, VNM, FPT...)
                 "index"  -- for indices (VN30, VNINDEX, VN100...)
                 "derivative" -- for futures (DNSE returns empty data for historical)
    from / to  : Unix timestamp in SECONDS (NOT YYYYMMDD string)

Symbol mapping (for the backtest plan):
    "VN30F2506" -> use "VN30" index as proxy (derivative history unavailable)
    "VNM"       -> type="stock"
    "HPG"       -> type="stock"
    "VN30"      -> type="index"

Usage::

    from src.backtest.data_fetcher import DataFetcher, OhlcBar
    fetcher = DataFetcher()
    bars = fetcher.fetch("HPG", "20250901", "20251231")             # daily stock
    bars = fetcher.fetch("VN30", "20250901", "20251231")            # daily index
    bars = fetcher.fetch("HPG", "20251201", "20251231", "1H")       # hourly
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_settings, resolve_symbol_profile
from src.logger import get_logger
from src.vn_time import vn_calendar_today_yyyymmdd

logger = get_logger("data_fetcher")

# Default cache directory (project root / data / cache)
_DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"

# Approximate trading days per year in Vietnam
TRADING_DAYS_PER_YEAR = 252

# Approximate hourly bars per year (5.75h session * 252 days)
HOURLY_BARS_PER_YEAR = int(5.75 * TRADING_DAYS_PER_YEAR)

# DNSE resolution strings accepted by the API
RESOLUTION_DAILY  = "1D"
RESOLUTION_HOURLY = "1H"
RESOLUTION_30MIN  = "30"
RESOLUTION_15MIN  = "15"
RESOLUTION_5MIN   = "5"   # if DNSE returns empty, optimizer marks timeframe invalid

# Symbols whose OHLC is served via type="index"
_INDEX_SYMBOLS = {"VN30", "VNINDEX", "VN100", "VNCOND", "HNX30", "UPCOM"}

# Proxy fallback for derivative symbols when DNSE returns no historical data.
# DNSE REST API does not serve historical OHLC for derivative futures contracts.
# VN30 index is the closest available proxy for VN30F derivatives.
_DERIVATIVE_INDEX_PROXY = "VN30"


@dataclass
class OhlcBar:
    """A single OHLC price bar."""
    symbol: str
    time: str          # "YYYYMMDD" for daily, "YYYYMMDDHH" for intraday
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    unix_ts: int = 0   # raw Unix timestamp from API

    @property
    def date_str(self) -> str:
        return self.time[:8]


def resolve_dnse_symbol(symbol: str) -> tuple[str, str]:
    """Resolve symbol for trading/execution context (trade_symbol)."""
    p = resolve_symbol_profile(symbol)
    return p["trade_symbol"], p["type"]


def resolve_data_symbol(symbol: str) -> tuple[str, str]:
    """Resolve symbol for market-data context (data_symbol)."""
    p = resolve_symbol_profile(symbol)
    return p["data_symbol"], p["type"]


def _resolve_symbol(symbol: str) -> tuple[str, str]:
    """Resolve symbol for OHLC/WS data calls (must use data_symbol)."""
    return resolve_data_symbol(symbol)


def _infer_asset_type(symbol: str) -> str:
    """Infer the DNSE asset type string for a given symbol (legacy helper)."""
    _, asset_type = _resolve_symbol(symbol)
    return asset_type


def _yyyymmdd_to_unix(date_str: str) -> int:
    """Convert 'YYYYMMDD' string to Unix timestamp (seconds, midnight UTC)."""
    dt = datetime(
        int(date_str[:4]),
        int(date_str[4:6]),
        int(date_str[6:8]),
        tzinfo=timezone.utc,
    )
    return int(dt.timestamp())


def _unix_to_yyyymmdd(ts: int) -> str:
    """Convert Unix timestamp to 'YYYYMMDD' string (UTC — matches DNSE columnar labels)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")


def _unix_to_yymmddhh(ts: int) -> str:
    """Convert Unix timestamp to 'YYYYMMDDHH' for intraday (UTC — use ``unix_ts`` for ordering)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d%H")


class DataFetcher:
    """Fetches and caches historical OHLC bars from DNSE REST API.

    The DNSE API requires:
      - `resolution` parameter for bar size (not `type` for bar size)
      - `type` parameter for asset class (stock / index / derivative)
      - Unix timestamp (seconds) for `from` and `to`

    Args:
        cache_dir: Directory to store JSON cache files.
        use_cache: If True (default), serve from local cache when available.
        settings:  Application settings (API credentials).
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        settings=None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._use_cache = use_cache
        self._settings = settings or get_settings()
        self._sdk = self._build_sdk()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        resolution: str = RESOLUTION_DAILY,
        asset_type: Optional[str] = None,
        proxy_derivatives: bool = True,
    ) -> List[OhlcBar]:
        """Fetch OHLC bars for a symbol in the given date range.

        Args:
            symbol:            Trading symbol.  Friendly names are auto-resolved:
                               "VN30F1M" → DNSE code "41I1G4000"
                               "VN30F2M" → DNSE code "41I1G5000"
                               "VN30F1Q" → DNSE code "41I1G6000"
                               "VN30"    → index
                               "HPG"     → stock
            from_date:         Start date "YYYYMMDD".
            to_date:           End date   "YYYYMMDD".
            resolution:        "1D" (daily, default), "1H" (hourly), "30", "15", "5" (5-min if supported).
            asset_type:        Override asset type: "stock" | "index" | "derivative".
            proxy_derivatives: If True (default), fall back to VN30 index when the
                               derivative symbol returns 0 historical bars.  The
                               DNSE REST API does not serve historical OHLC for
                               futures contracts; VN30 is the closest proxy.

        Returns:
            List of OhlcBar sorted oldest-first.
        """
        dnse_code, inferred_type = _resolve_symbol(symbol)
        if asset_type is None:
            asset_type = inferred_type

        # v4: tách cache theo mã resolve + asset_type + proxy để tránh lẫn index/future.
        cache_key = (
            f"{symbol}_{dnse_code}_{asset_type}_{resolution}_{from_date}_{to_date}"
            f"_proxy{int(bool(proxy_derivatives))}_v4"
        )
        cache_file = self._cache_dir / f"{cache_key}.json"

        # Live: range includes **today (VN)** — disk cache would freeze intraday bars (duplicate_bar).
        vn_today = vn_calendar_today_yyyymmdd()
        skip_cache_read = bool(to_date >= vn_today)

        if self._use_cache and not skip_cache_read and cache_file.exists():
            bars = self._load_cache(cache_file)
            if bars:
                logger.info(
                    "DataFetcher: loaded from cache",
                    extra={"symbol": symbol, "resolution": resolution, "n_bars": len(bars)},
                )
                return bars
            logger.warning(
                "DataFetcher: cache file empty, refetching from API",
                extra={"path": str(cache_file), "symbol": symbol},
            )
            try:
                cache_file.unlink()
            except OSError:
                pass

        if skip_cache_read and self._use_cache:
            logger.info(
                "DataFetcher: bỏ qua đọc cache (to_date >= hôm nay VN — lấy OHLC mới từ API)",
                extra={
                    "symbol": symbol,
                    "to_date": to_date,
                    "vn_today": vn_today,
                    "resolution": resolution,
                },
            )
            print(
                f"[DataFetcher] Bỏ cache: to_date={to_date} >= vn_today={vn_today} → gọi API OHLC.",
                flush=True,
            )

        bars = self._fetch_from_api(dnse_code, from_date, to_date, resolution, asset_type)

        # Derivative fallback: if API returns 0 bars, use VN30 index as proxy
        if not bars and asset_type == "derivative" and proxy_derivatives:
            logger.warning(
                f"DataFetcher: {symbol} ({dnse_code}) has no historical OHLC data. "
                f"Falling back to '{_DERIVATIVE_INDEX_PROXY}' index as proxy.",
                extra={"original_symbol": symbol, "proxy": _DERIVATIVE_INDEX_PROXY},
            )
            bars = self._fetch_from_api(
                _DERIVATIVE_INDEX_PROXY, from_date, to_date, resolution, "index"
            )
            # Tag bars with the original requested symbol
            bars = [
                OhlcBar(
                    symbol=symbol, time=b.time, open=b.open, high=b.high,
                    low=b.low, close=b.close, volume=b.volume, unix_ts=b.unix_ts,
                )
                for b in bars
            ]

        if bars:
            self._save_cache(cache_file, bars)

        return bars

    def fetch_multi(
        self,
        symbols: List[str],
        from_date: str,
        to_date: str,
        resolution: str = RESOLUTION_DAILY,
    ) -> Dict[str, List[OhlcBar]]:
        """Fetch bars for multiple symbols."""
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch(sym, from_date, to_date, resolution)
            except Exception as e:
                logger.error(f"DataFetcher: failed {sym}", extra={"error": str(e)})
                result[sym] = []
        return result

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """Delete cache files."""
        pattern = f"{symbol}_*.json" if symbol else "*.json"
        deleted = 0
        for f in self._cache_dir.glob(pattern):
            f.unlink()
            deleted += 1
        logger.info(f"DataFetcher: cleared {deleted} cache files")
        return deleted

    # ------------------------------------------------------------------ #
    # Internal: API call
    # ------------------------------------------------------------------ #

    def _fetch_from_api(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
        resolution: str,
        asset_type: str,
    ) -> List[OhlcBar]:
        """Call DNSE REST API and return parsed OhlcBar list."""
        t_from = _yyyymmdd_to_unix(from_date)
        # Trước đây dùng nửa đêm UTC của to_date → cắt mất nến trong ngày (skew ~18h vs phiên VN).
        # Dùng mốc cuối ngày calendar (exclusive: 00:00 UTC ngày sau) để gồm đủ nến 15m trong to_date.
        t_to = _yyyymmdd_to_unix(to_date) + 86400

        # Warn about derivative data limitations
        if asset_type == "derivative":
            logger.warning(
                "DataFetcher: DNSE does not provide historical OHLC for derivative futures. "
                "Use 'VN30' (index) as a proxy for VN30F backtest.",
                extra={"symbol": symbol},
            )

        query = {
            "symbol": symbol,
            "resolution": resolution,
            "type": asset_type,
            "from": t_from,
            "to": t_to,
        }

        logger.info(
            "DataFetcher: fetching from API",
            extra={"symbol": symbol, "resolution": resolution, "asset_type": asset_type,
                   "from": from_date, "to": to_date},
        )

        try:
            status, body = self._sdk._request("GET", "/price/ohlc", query=query)
        except Exception as e:
            raise RuntimeError(f"DNSE API call failed for {symbol}: {e}") from e

        raw_preview = (body or "")[:800]
        logger.info(
            "DataFetcher: raw JSON preview",
            extra={
                "symbol": symbol,
                "asset_type": asset_type,
                "resolution": resolution,
                "status": status,
                "raw_json_preview": raw_preview,
            },
        )
        print(
            f"[API-RAW] symbol={symbol} type={asset_type} status={status} preview={raw_preview}",
            flush=True,
        )

        if status != 200:
            raise RuntimeError(
                f"DNSE API {status} for {symbol}: {(body or '')[:200]}"
            )

        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from DNSE API: {e}") from e

        bars = self._parse_columnar(data, symbol, resolution)

        if not bars:
            logger.warning(
                "DataFetcher: API returned 0 bars",
                extra={"symbol": symbol, "resolution": resolution, "from": from_date, "to": to_date},
            )
            return []

        bars.sort(key=lambda b: b.unix_ts)
        logger.info(
            "DataFetcher: fetched bars",
            extra={"symbol": symbol, "resolution": resolution, "n_bars": len(bars)},
        )
        return bars

    def _parse_columnar(self, data: dict, symbol: str, resolution: str) -> List[OhlcBar]:
        """Parse DNSE columnar response: {"t":[], "o":[], "h":[], "l":[], "c":[], "v":[]}."""
        if not isinstance(data, dict):
            logger.warning(f"DataFetcher: unexpected response type {type(data)} for {symbol}")
            return []

        times   = data.get("t", [])
        opens   = data.get("o", [])
        highs   = data.get("h", [])
        lows    = data.get("l", [])
        closes  = data.get("c", [])
        volumes = data.get("v", [0.0] * len(times))

        bars = []
        for i, ts in enumerate(times):
            try:
                ts_int = int(ts)
                # Một số API trả Unix **milliseconds**
                if ts_int > 10_000_000_000:
                    ts_int //= 1000
                if resolution == RESOLUTION_DAILY:
                    time_str = _unix_to_yyyymmdd(ts_int)
                else:
                    time_str = _unix_to_yymmddhh(ts_int)

                bars.append(OhlcBar(
                    symbol=symbol,
                    time=time_str,
                    open=float(opens[i]),
                    high=float(highs[i]),
                    low=float(lows[i]),
                    close=float(closes[i]),
                    volume=float(volumes[i]) if i < len(volumes) else 0.0,
                    unix_ts=ts_int,
                ))
            except (IndexError, ValueError, TypeError):
                continue
        return bars

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _save_cache(self, path: Path, bars: List[OhlcBar]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([asdict(b) for b in bars], f)
        except Exception as e:
            logger.warning(f"DataFetcher: could not write cache {path}: {e}")

    def _load_cache(self, path: Path) -> List[OhlcBar]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [OhlcBar(**r) for r in raw]

    # ------------------------------------------------------------------ #
    # SDK construction
    # ------------------------------------------------------------------ #

    def _build_sdk(self):
        """Build a DNSEClient from application settings."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "vendor" / "dnse"))
            from dnse import DNSEClient
            return DNSEClient(
                api_key=self._settings.DNSE_API_KEY,
                api_secret=self._settings.DNSE_API_SECRET,
                base_url=self._settings.DNSE_BASE_URL,
            )
        except Exception as e:
            logger.warning(f"DataFetcher: could not build DNSEClient: {e}")
            return None
