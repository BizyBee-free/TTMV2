"""Bổ sung basis + OI cho TTM paper (WebSocket) — cùng logic nguồn với ``ttm_live_runner``."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.backtest.data_fetcher import (
    RESOLUTION_15MIN,
    RESOLUTION_5MIN,
    DataFetcher,
    OhlcBar,
    resolve_data_symbol,
    resolve_dnse_symbol,
)
from src.config import Settings, get_settings
from src.dnse_client import BeeTradeClient
from src.hmm.oi_data import (
    OICache,
    build_open_interest_series_for_live,
    norm_unix_ts_sec,
    parse_open_interest_from_secdef_payload,
)
from src.hmm.oi_ws_cache import get_latest_open_interest
from src.logger import get_logger

logger = get_logger("ttm_paper_basis_oi")

# Thử 1m trước (paper dùng nến 1 phút), fallback 5m / 15m nếu API trả rỗng
_RES_TRIES: Tuple[str, ...] = ("1", RESOLUTION_5MIN, RESOLUTION_15MIN)


def _bars_yyyymmdd_range(bars: Sequence[OhlcBar]) -> Tuple[str, str]:
    from datetime import datetime, timezone

    ts_list = [norm_unix_ts_sec(int(b.unix_ts)) for b in bars if int(getattr(b, "unix_ts", 0) or 0) > 0]
    if not ts_list:
        d = datetime.now(timezone.utc).strftime("%Y%m%d")
        return d, d
    lo, hi = min(ts_list), max(ts_list)
    return (
        datetime.fromtimestamp(lo, tz=timezone.utc).strftime("%Y%m%d"),
        datetime.fromtimestamp(hi, tz=timezone.utc).strftime("%Y%m%d"),
    )


def _align_index_close_to_bars(
    bars: Sequence[OhlcBar],
    index_points: Sequence[Tuple[int, float]],
) -> List[float]:
    """Forward-fill: index close tại mốc ts mới nhất ≤ bar.unix_ts."""
    if not bars:
        return []
    if not index_points:
        return [0.0] * len(bars)
    ts_list = [int(p[0]) for p in index_points]
    vals = [float(p[1]) for p in index_points]
    out: List[float] = []
    for b in bars:
        ts = norm_unix_ts_sec(int(getattr(b, "unix_ts", 0) or 0))
        if ts <= 0:
            out.append(0.0)
            continue
        i = bisect.bisect_right(ts_list, ts) - 1
        out.append(vals[i] if i >= 0 else 0.0)
    return out


class TTMPaperEnricher:
    """
    Gọi sau mỗi nến OHLC mới: REST secdef (OI) + REST index OHLC (basis = future_close − index_close).

    Khớp hướng ``ttm_live_runner.run_once`` (chuỗi OI + basis_series).
    """

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        client: BeeTradeClient,
        fetcher: DataFetcher,
        derivative_symbol: str,
        debug_oi: bool = False,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._fetcher = fetcher
        self._derivative_symbol = (derivative_symbol or "").strip().upper()
        idx_sym = (self._settings.HMM_INDEX_SYMBOL or "VN30").strip().upper()
        self._idx_data, _ = resolve_data_symbol(idx_sym)
        self._trade_sym, _ = resolve_dnse_symbol(self._derivative_symbol)
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in self._trade_sym)
        self._oi_cache = OICache(
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "cache"
            / "oi"
            / f"{safe}_paper.jsonl"
        )
        self._last_oi_append_ts: Optional[int] = None
        self._idx_range: Optional[Tuple[str, str]] = None
        self._idx_points: List[Tuple[int, float]] = []
        self._debug_oi = bool(debug_oi)
        self._warned_secdef_no_oi = False
        self._logged_first_oi_ok = False
        self.last_secdef_snapshot: Dict[str, Any] = {}
        # Sau khi tìm được mã secdef có OI, chỉ gọi API với mã đó (tránh 2 request/nến).
        self._preferred_secdef_symbol: Optional[str] = None

    def _resolve_secdef_oi(self) -> Tuple[Dict[str, Any], Optional[int], str]:
        """Gọi GET /price/{symbol}/secdef (có boardId từ BeeTradeClient). Thử trade_symbol rồi data symbol."""
        candidates: List[str] = []
        if self._trade_sym:
            candidates.append(self._trade_sym)
        if self._derivative_symbol and self._derivative_symbol not in candidates:
            candidates.append(self._derivative_symbol)

        if not candidates:
            return {}, None, ""

        if self._preferred_secdef_symbol:
            sym = self._preferred_secdef_symbol
            secdef = self._client.get_security_definition(sym)
            payload = secdef.get("data") if isinstance(secdef, dict) else None
            oi_now = parse_open_interest_from_secdef_payload(payload)
            if oi_now is None and isinstance(secdef, dict):
                oi_now = parse_open_interest_from_secdef_payload(secdef)
            return secdef, oi_now, sym

        last: Dict[str, Any] = {}
        oi_now: Optional[int] = None
        for sym in candidates:
            secdef = self._client.get_security_definition(sym)
            last = secdef
            payload = secdef.get("data") if isinstance(secdef, dict) else None
            oi_now = parse_open_interest_from_secdef_payload(payload)
            if oi_now is None and isinstance(secdef, dict):
                oi_now = parse_open_interest_from_secdef_payload(secdef)
            if oi_now is not None:
                self._preferred_secdef_symbol = sym
                return secdef, oi_now, sym
            st = secdef.get("status") if isinstance(secdef, dict) else None
            if st is not None and int(st) >= 400:
                break

        self._preferred_secdef_symbol = candidates[0]
        return last, oi_now, candidates[0]

    def _refresh_index_points(self, from_d: str, to_d: str) -> None:
        if self._idx_range == (from_d, to_d) and self._idx_points:
            return
        pts: List[Tuple[int, float]] = []
        used_res: Optional[str] = None
        for res in _RES_TRIES:
            bars = self._fetcher.fetch(
                self._idx_data,
                from_d,
                to_d,
                resolution=res,
                asset_type="index",
            )
            if bars:
                used_res = res
                pts = [(norm_unix_ts_sec(int(b.unix_ts)), float(b.close)) for b in bars]
                pts.sort(key=lambda x: x[0])
                break
        self._idx_range = (from_d, to_d)
        self._idx_points = pts
        logger.info(
            "TTM paper: index OHLC cho basis",
            extra={
                "index": self._idx_data,
                "from": from_d,
                "to": to_d,
                "resolution": used_res,
                "n": len(pts),
            },
        )

    def enrich(self, bars: List[OhlcBar]) -> Dict[str, Any]:
        """Trả về ``open_interest`` và tùy chọn ``basis`` (cùng độ dài ``bars``)."""
        if not bars:
            return {}
        out: Dict[str, Any] = {}
        n = len(bars)

        # --- OI: WebSocket sec_def (ưu tiên) — REST /price/.../secdef phái sinh thường KHÔNG có openInterestQuantity ---
        try:
            secdef, oi_rest, sym_used = self._resolve_secdef_oi()
            st = secdef.get("status") if isinstance(secdef, dict) else None
            payload = secdef.get("data") if isinstance(secdef, dict) else None

            oi_ws = get_latest_open_interest(self._derivative_symbol)
            if oi_ws is None:
                oi_ws = get_latest_open_interest(self._trade_sym)
            if oi_ws is not None:
                oi_now = oi_ws
                oi_source = "websocket"
            else:
                oi_now = oi_rest
                oi_source = "rest" if oi_rest is not None else "none"

            snap: Dict[str, Any] = {
                "status": st,
                "trade_symbol": self._trade_sym,
                "derivative_symbol": self._derivative_symbol,
                "secdef_symbol_used": sym_used,
                "board_id": (getattr(self._settings, "DNSE_WS_BOARD_ID", "") or "").split(",")[0].strip(),
                "openInterestQuantity_rest": oi_rest,
                "openInterestQuantity": oi_now,
                "oi_source": oi_source,
            }
            if isinstance(payload, dict):
                snap["payload_top_keys"] = list(payload.keys())[:30]
            self.last_secdef_snapshot = snap

            if st is not None and int(st) >= 400:
                logger.warning(
                    "TTM paper: secdef API lỗi HTTP",
                    extra={"status": st, "secdef_symbol": sym_used, "data": payload},
                )
            elif oi_now is None and not self._warned_secdef_no_oi:
                self._warned_secdef_no_oi = True
                pk = list(payload.keys())[:25] if isinstance(payload, dict) else None
                logger.warning(
                    "TTM paper: chưa có openInterestQuantity (REST phái sinh thường không trả OI; "
                    "cần WebSocket sec_def — đảm bảo paper_test đã subscribe_sec_def và đợi vài nến)",
                    extra={
                        "status": st,
                        "secdef_symbol": sym_used,
                        "payload_keys": pk,
                    },
                )
            elif oi_now is not None and not self._logged_first_oi_ok:
                self._logged_first_oi_ok = True
                logger.info(
                    "TTM paper: openInterestQuantity OK",
                    extra={
                        "openInterestQuantity": oi_now,
                        "oi_source": oi_source,
                        "secdef_symbol": sym_used,
                    },
                )

            oi_seq = build_open_interest_series_for_live(bars, self._oi_cache, oi_now)
            if len(oi_seq) != n:
                oi_seq = (oi_seq + [0.0] * n)[:n]
            out["open_interest"] = oi_seq

            if self._debug_oi:
                tail = float(oi_seq[-1]) if oi_seq else 0.0
                print(
                    f"[TTM OI] source={oi_source} rest={oi_rest} ws_cache={oi_ws} "
                    f"final={oi_now} aligned_last={tail} http={st} sym={sym_used}",
                    flush=True,
                )

            if oi_now is not None:
                last_ts = norm_unix_ts_sec(int(bars[-1].unix_ts))
                if last_ts > 0 and last_ts != self._last_oi_append_ts:
                    self._oi_cache.append(last_ts, float(oi_now))
                    self._last_oi_append_ts = last_ts
        except Exception as e:
            logger.warning("TTM paper: OI enrich failed", extra={"error": str(e)})
            out["open_interest"] = [0.0] * n

        # --- Basis (future close − index close) ---
        if not self._settings.HMM_USE_BASIS:
            return out

        try:
            from_d, to_d = _bars_yyyymmdd_range(bars)
            self._refresh_index_points(from_d, to_d)
            idx_close = _align_index_close_to_bars(bars, self._idx_points)
            basis: List[float] = []
            for b, ic in zip(bars, idx_close):
                fc = float(b.close)
                basis.append(fc - float(ic))
            out["basis"] = basis
        except Exception as e:
            logger.warning("TTM paper: basis enrich failed", extra={"error": str(e)})

        return out
