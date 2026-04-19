"""Application configuration loaded from environment variables and .env file."""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.logger import get_logger
from src.vn_time import vn_now

_config_log = get_logger("config")


def parse_int_list(value: Any) -> List[int]:
    """Parse env or JSON/comma-separated string into a list of ints."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("["):
        return [int(x) for x in json.loads(s)]
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class MarketType(str, Enum):
    STOCK = "STOCK"
    DERIVATIVE = "DERIVATIVE"


class OrderSide(str, Enum):
    """DNSE uses NB/NS, not BUY/SELL."""
    BUY = "NB"
    SELL = "NS"


class OrderType(str, Enum):
    LO = "LO"       # Limit Order (all exchanges)
    ATO = "ATO"      # At The Open (HOSE)
    ATC = "ATC"      # At The Close (HOSE, HNX)
    MTL = "MTL"      # Market To Limit (HOSE, HNX)
    MOK = "MOK"      # Match Or Kill (HNX)
    MAK = "MAK"      # Match And Kill (HNX)
    PLO = "PLO"      # Post Limit Order (HNX)


class OrderStatus(str, Enum):
    """Order lifecycle statuses from DNSE API."""
    PENDING = "Pending"
    PENDING_NEW = "PendingNew"
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    PENDING_REPLACE = "PendingReplace"
    PENDING_CANCEL = "PendingCancel"
    CANCELED = "Canceled"
    REJECTED = "Rejected"
    EXPIRED = "Expired"
    DONE_FOR_DAY = "DoneForDay"


class StrategyAlgo(str, Enum):
    """Which strategy stack the deployment uses (routing / docs; see STRATEGY_ALGO)."""
    MCMC = "MCMC"
    HMM = "HMM"
    TTM = "TTM"


# Dual-symbol mapping for DNSE KRX compatibility:
# - data_symbol: used for OHLC / websocket market data
# - trade_symbol: used for order placement / account / secdef
SYMBOL_MAP: Dict[str, Dict[str, str]] = {
    "VN30F1M": {
        "data_symbol": "VN30F1M",
        "trade_symbol": "41I1G4000",
        "type": "derivative",
    },
    "VN30": {
        "data_symbol": "VN30",
        "trade_symbol": "VN30",
        "type": "index",
    },
}


def resolve_symbol_profile(symbol: str) -> Dict[str, str]:
    """Resolve symbol into dual-mapping profile."""
    s = (symbol or "").strip().upper()
    if not s:
        return {"data_symbol": symbol, "trade_symbol": symbol, "type": "stock"}
    if s in SYMBOL_MAP:
        return dict(SYMBOL_MAP[s])
    # Reverse-lookup by trade symbol (9-char KRX) so legacy callers still work
    for profile in SYMBOL_MAP.values():
        if s == profile.get("trade_symbol", "").upper():
            return dict(profile)
    # Direct 9-char KRX derivative code (unknown mapping)
    if len(s) == 9 and s.isalnum() and s.startswith("41"):
        return {"data_symbol": s, "trade_symbol": s, "type": "derivative"}
    # Generic derivative-like symbols fallback
    if s.startswith("VN30F") or s.startswith("V100F"):
        return {"data_symbol": s, "trade_symbol": s, "type": "derivative"}
    return {"data_symbol": s, "trade_symbol": s, "type": "stock"}


def canonical_data_symbol(symbol: str) -> str:
    """Thống nhất mã hiển thị/REST (data_symbol): map 41… / alias → VN30F1M, v.v.)."""
    p = resolve_symbol_profile((symbol or "").strip())
    return (p.get("data_symbol") or symbol or "").strip().upper()


def ws_subscribe_symbol_list(sym: str) -> List[str]:
    """Ký hiệu subscribe WS: phái sinh có thể cần cả data_symbol và trade_symbol (KRX 9 ký tự)."""
    s = (sym or "").strip().upper()
    prof = resolve_symbol_profile(s)
    if str(prof.get("type") or "").lower() != "derivative":
        return [s] if s else []
    ds = (prof.get("data_symbol") or s).strip().upper()
    ts = (prof.get("trade_symbol") or "").strip().upper()
    out: List[str] = []
    if ds:
        out.append(ds)
    if ts and ts != ds:
        out.append(ts)
    return out if out else ([s] if s else [])


# --- VN30F1M: đối chiếu mã KRX front-month (API /instruments vs lịch đáo hạn), 1 lần / process ---

_VN30_F1M_RESOLVED: bool = False
_VN30_KRX_TRADE_RE = re.compile(r"^41I1G\d{4}$")


def _env_vn30_f1m_trade_symbol(settings: Optional[Any]) -> str:
    """Chuỗi KRX từ ``Settings.VN30_F1M_TRADE_SYMBOL`` (chỉ str; bỏ qua MagicMock trong test)."""
    if settings is None:
        return ""
    v = getattr(settings, "VN30_F1M_TRADE_SYMBOL", "")
    if v is None or not isinstance(v, str):
        return ""
    return v.strip().upper()


def apply_vn30_f1m_trade_symbol_from_settings(settings: Optional[Any]) -> None:
    """Nếu ``VN30_F1M_TRADE_SYMBOL`` khác rỗng và khớp ``41I1Gdddd``: ghi ``SYMBOL_MAP['VN30F1M']['trade_symbol']``."""
    sym = _env_vn30_f1m_trade_symbol(settings)
    if not sym:
        return
    if not _VN30_KRX_TRADE_RE.match(sym):
        _config_log.warning(
            "VN30_F1M_TRADE_SYMBOL ignored (expected pattern 41I1G + 4 digits)",
            extra={"VN30_F1M_TRADE_SYMBOL": sym},
        )
        return
    SYMBOL_MAP["VN30F1M"]["trade_symbol"] = sym


def _third_thursday(year: int, month: int) -> date:
    """Thứ Năm tuần thứ ba của tháng (calendar month, local date arithmetic)."""
    first = date(year, month, 1)
    offset = (3 - first.weekday()) % 7  # Monday=0, Thursday=3
    first_thu = first + timedelta(days=offset)
    return first_thu + timedelta(days=14)


def _next_month(y: int, m: int) -> Tuple[int, int]:
    if m == 12:
        return y + 1, 1
    return y, m + 1


def _calendar_seed_month(today: date) -> Tuple[int, int]:
    """Tháng bắt đầu quét: trước tháng 4/2026 coi front chuẩn là tháng 4 (chu kỳ VN30 user mô tả)."""
    if today.year == 2026 and today.month < 4:
        return 2026, 4
    return today.year, today.month


def _front_expiry_month_nearest(today: date) -> Tuple[int, int]:
    """Chọn (năm, tháng) có third_thursday >= today gần nhất, quét từ seed_month."""
    y0, m0 = _calendar_seed_month(today)
    best: Optional[Tuple[int, int, date]] = None
    y, m = y0, m0
    for _ in range(0, 24):
        exp = _third_thursday(y, m)
        if exp >= today:
            if best is None or exp < best[2]:
                best = (y, m, exp)
        y, m = _next_month(y, m)
    if best is None:
        raise RuntimeError("VN30 calendar: không tìm được tháng đáo hạn trong 24 tháng tới")
    return best[0], best[1]


def calendar_krx_vn30_index_symbol_for_expiry(exp_year: int, exp_month: int) -> str:
    """Heuristic mã KRX 9 ký tự từ tháng đáo hạn (rủi ro sai nếu quy ước DNSE đổi).

    Khớp các ví dụ trong repo: 41I1G4000 (T4), 41I1G5000 (T5), 41I1G6000 (T6).
    """
    if exp_month < 1 or exp_month > 12:
        raise ValueError(f"invalid expiry month {exp_month}")
    if 4 <= exp_month <= 9:
        return f"41I1G{exp_month}000"
    if 10 <= exp_month <= 12:
        return f"41I1G{exp_month}00"
    return f"41I1G{exp_month:02d}00"


def _extract_instrument_rows(data: Any) -> List[Dict[str, Any]]:
    """Lấy list dict từ payload /instruments (nhiều dạng lồng nhau)."""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    for key in ("rows", "instruments", "items", "results"):
        v = data.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    res = data.get("result")
    if isinstance(res, dict):
        for key in ("rows", "instruments", "items"):
            v = res.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _parse_iso_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        ts = int(v)
        if ts > 10_000_000_000:  # ms
            ts //= 1000
        try:
            return datetime.utcfromtimestamp(ts).date()
        except (OSError, OverflowError, ValueError):
            return None
    s = str(v).strip()
    if not s:
        return None
    s10 = s[:10]
    try:
        return date.fromisoformat(s10)
    except ValueError:
        pass
    if len(s) >= 8 and s[:8].isdigit():
        try:
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        except ValueError:
            return None
    return None


def _row_symbol(row: Dict[str, Any]) -> str:
    for k in ("symbol", "ticker", "secCd", "sec_cd", "isin"):
        v = row.get(k)
        if v is not None and str(v).strip():
            sym = str(v).strip().upper()
            if sym.startswith("VN41"):
                continue
            return sym
    return ""


def _row_expiry(row: Dict[str, Any]) -> Optional[date]:
    for k in (
        "lastTradingDate",
        "lastTradeDate",
        "maturityDate",
        "expiryDate",
        "dueDate",
        "expirationDate",
        "endDate",
    ):
        d = _parse_iso_date(row.get(k))
        if d:
            return d
    return None


def _row_volume(row: Dict[str, Any]) -> float:
    for k in ("totalVolumeTraded", "totalVolume", "volume", "totalQty", "openInterestQuantity"):
        v = row.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def _vn30_krx_from_instruments_payload(data: Any, today: date) -> str:
    rows = _extract_instrument_rows(data)
    cands: List[Tuple[str, Optional[date], float]] = []
    for row in rows:
        sym = _row_symbol(row)
        if not _VN30_KRX_TRADE_RE.fullmatch(sym):
            continue
        if not sym.startswith("41I1"):
            continue
        exp = _row_expiry(row)
        vol = _row_volume(row)
        cands.append((sym, exp, vol))
    if not cands:
        raise RuntimeError("VN30 API: không có hàng nào khớp mã KRX 41I1Gxxxx (9 ký tự) trong /instruments")

    with_exp = [(s, e, v) for s, e, v in cands if e is not None]
    fut = [(s, e, v) for s, e, v in with_exp if e >= today]
    if fut:
        sym = min(fut, key=lambda t: t[1])[0]
        _config_log.info("VN30 API: chọn hợp đồng nearest expiry >= today", extra={"symbol": sym})
        return sym
    # Hết hạn trên payload: chọn volume lớn nhất trong các mã 41I1Gxxxx
    sym = max(cands, key=lambda t: t[2])[0]
    _config_log.warning(
        "VN30 API: không có expiry >= today; fallback volume max",
        extra={"symbol": sym},
    )
    return sym


def _vn30_trade_symbol_via_api(client: Any) -> str:
    """Gọi DNSE /instruments (qua BeeTradeClient), lọc mã 41I1Gxxxx."""
    for kwargs in (
        {"market_id": "DVX", "limit": 500},
        {"limit": 500},
    ):
        res = client.get_instruments(**kwargs)
        st = int(res.get("status") or 0)
        if st != 200:
            continue
        data = res.get("data")
        try:
            return _vn30_krx_from_instruments_payload(data, vn_now().date())
        except RuntimeError:
            continue
    raise RuntimeError("VN30 API: get_instruments không trả dữ liệu hợp lệ để suy ra mã 41I1Gxxxx")


def ensure_vn30_f1m_trade_symbol_resolved(client: Any) -> None:
    """Một lần / process: API vs lịch; khớp thì ghi SYMBOL_MAP['VN30F1M']['trade_symbol'], lệch thì exit(1).

    Nếu ``VN30_F1M_TRADE_SYMBOL`` (.env) hợp lệ: dùng giá trị đó, đánh dấu đã resolve, không gọi API.

    Phải gọi với ``BeeTradeClient`` vừa khởi tạo (không tạo client mới bên trong — tránh đệ quy).
    """
    global _VN30_F1M_RESOLVED
    if _VN30_F1M_RESOLVED:
        return

    settings = getattr(client, "settings", None)
    apply_vn30_f1m_trade_symbol_from_settings(settings)
    env_sym = _env_vn30_f1m_trade_symbol(settings)
    if env_sym and _VN30_KRX_TRADE_RE.match(env_sym):
        _VN30_F1M_RESOLVED = True
        _config_log.info(
            "VN30F1M trade_symbol from VN30_F1M_TRADE_SYMBOL (.env / settings)",
            extra={"trade_symbol": env_sym},
        )
        return

    if settings is None or not bool(getattr(settings, "VN30_AUTO_RESOLVE_TRADE_SYMBOL", False)):
        # Không đánh dấu _VN30_F1M_RESOLVED: để lần sau (vd. paper_test prime) vẫn có thể resolve.
        return

    today = vn_now().date()
    api_sym = _vn30_trade_symbol_via_api(client)

    y, m = _front_expiry_month_nearest(today)
    cal_sym = calendar_krx_vn30_index_symbol_for_expiry(y, m)

    if api_sym != cal_sym:
        print(
            "FATAL: VN30 front-month KRX mismatch between DNSE /instruments and calendar heuristic.\n"
            f"  API (instruments): {api_sym}\n"
            f"  Calendar (third Thu): {cal_sym}\n"
            "Set VN30_AUTO_RESOLVE_TRADE_SYMBOL=false to use static SYMBOL_MAP, or fix mapping logic.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    old = SYMBOL_MAP["VN30F1M"].get("trade_symbol")
    SYMBOL_MAP["VN30F1M"]["trade_symbol"] = api_sym
    _VN30_F1M_RESOLVED = True
    _config_log.info(
        "VN30F1M trade_symbol resolved",
        extra={"old_trade_symbol": old, "new_trade_symbol": api_sym, "expiry_month": f"{y}-{m:02d}"},
    )


def paper_test_prime_vn30f1m_krx(settings: Any) -> Any:
    """Dùng từ ``scripts/paper_test.py``: reset cờ một lần + tạo ``BeeTradeClient`` để resolve VN30F1M trước subscribe.

    Trả về ``Settings`` đã bật ``VN30_AUTO_RESOLVE_TRADE_SYMBOL`` cho phần còn lại phiên.
    ``SystemExit`` / lỗi API–lịch lệch nhau: ném ra ngoài (dừng paper).
    """
    global _VN30_F1M_RESOLVED
    _VN30_F1M_RESOLVED = False
    s = settings.model_copy(update={"VN30_AUTO_RESOLVE_TRADE_SYMBOL": True})
    from src.dnse_client import BeeTradeClient

    BeeTradeClient(settings=s)
    return s


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
    )

    # DNSE API credentials
    DNSE_API_KEY: str = Field(..., description="DNSE OpenAPI Key")
    DNSE_API_SECRET: str = Field(..., description="DNSE OpenAPI Secret")
    DNSE_ACCOUNT_NO: str = Field(..., description="Trading sub-account number")
    DNSE_TRADING_TOKEN: str = Field(
        default="",
        description=(
            "Trading token sau OTP (đặt lệnh live). "
            "Không prompt OTP trong live; cần set sẵn token ở .env hoặc DNSE_TRADING_TOKEN_FILE."
        ),
    )
    DNSE_TRADING_TOKEN_FILE: str = Field(
        default="",
        description="Đường dẫn file chứa một dòng trading token (ưu tiên sau khi để trống DNSE_TRADING_TOKEN).",
    )
    DNSE_POSITIONS_FALLBACK_TO_DEALS: bool = Field(
        default=True,
        description="Nếu GET /accounts/.../positions trả 404, gọi lại endpoint legacy /deals.",
    )

    # API endpoints
    DNSE_BASE_URL: str = Field(
        default="https://openapi.dnse.com.vn",
        description="DNSE REST API base URL",
    )
    DNSE_WS_URL: str = Field(
        default="wss://ws-openapi.dnse.com.vn/v1/stream",
        description="DNSE WebSocket URL (bắt buộc /v1/stream; encoding thêm trong MarketDataManager)",
    )

    # WebSocket settings
    WS_ENCODING: str = Field(
        default="msgpack",
        description=(
            "WebSocket encoding: 'msgpack' (mặc định, khớp stream DNSE) hoặc 'json'. "
            "Với 'json', decoder vẫn fallback msgpack nếu frame là nhị phân."
        ),
    )
    DNSE_WS_BOARD_ID: str = Field(
        default="G1",
        description=(
            "boardId cho subscribe quote / trade / sec_def (WebSocket). "
            "Nhiều board: dấu phẩy, ví dụ G1,AL (khớp openapi-sdk subscribe nhiều kênh tick/top_price). "
            "OHLC dùng kênh ``ohlc.{tf}.msgpack`` (không board)."
        ),
    )
    DNSE_WS_SUBSCRIBE_USE_SESSION_ID: bool = Field(
        default=False,
        description=(
            "Nếu true: gắn sessionId/SessionId vào payload subscribe. "
            "Một số bản DNSE market WS chỉ cần auth; gắn session vào subscribe có thể khiến không nhận quote/OHLC."
        ),
    )
    DNSE_WS_OHLC_BOARD_IN_CHANNEL: bool = Field(
        default=False,
        description=(
            "Tương thích API: client đã gộp về kênh ``ohlc.{resolution}.*`` giống openapi-sdk "
            "(không chèn board trong tên kênh). Nếu true vẫn được ghi log nhưng không đổi tên kênh."
        ),
    )
    DNSE_WS_OHLC_PREFER_CLOSED: bool = Field(
        default=True,
        description=(
            "Ưu tiên subscribe kênh OHLC Closed (theo changelog 2026-04-07) để lấy nến đã đóng. "
            "Client vẫn gửi thêm kênh ohlc chuẩn để fallback tương thích gateway cũ."
        ),
    )
    DNSE_WS_PIPELINE_DEBUG: bool = Field(
        default=False,
        description=(
            "In chuỗi quan sát WS: RAW_WS, SENDING SUB/AUTH, DECODED, SUB OK, PING/PONG, "
            "DECODE_ERROR, MARKET DATA OK, TICK(pipeline_trade). Bật khi debug pipeline."
        ),
    )
    DNSE_WS_PIPELINE_ASSERT_MARKET_SEC: int = Field(
        default=0,
        description=(
            "Nếu > 0: sau N giây kể từ connect, assert ``market_message_dispatches > 0`` "
            "(RuntimeError nếu không có bản tin market). 0 = tắt."
        ),
    )
    DNSE_OHLC_SYNTH_FROM_TRADES: bool = Field(
        default=True,
        description=(
            "Khi WS không đẩy nến (chỉ có trade T=t), tổng hợp OHLC từ giá khớp + sending_time "
            "(bản ghi ``type=synth_trade``). Khớp hướng REST getOhlc cho STOCK khi không có stream nến."
        ),
    )
    DNSE_OHLC_SYNTH_STOP_WHEN_WS_OHLC: bool = Field(
        default=True,
        description="Ngừng tổng hợp từ trade sau khi nhận nến WS thật (type khác synth_trade).",
    )

    # Trading mode
    PAPER_MODE: bool = Field(
        default=True,
        description="Paper trading mode (no real orders sent)",
    )

    STRATEGY_ALGO: StrategyAlgo = Field(
        default=StrategyAlgo.HMM,
        description=(
            "MCMC: Chiến lược MCMC derivatives (Markov + MCMC engine; không phải HMM). "
            "HMM: Chiến lược Live HMM (regime) + kill switch / DNSE robustness / hmm_live (mặc định). "
            "TTM: Trapped Trader Model (src/strategies/ttm) + ttm_live runner."
        ),
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_DIR: str = Field(default="logs")
    LOG_MAX_BYTES: int = Field(default=50 * 1024 * 1024)  # 50MB
    LOG_BACKUP_COUNT: int = Field(default=7)

    # Risk management defaults
    MAX_DAILY_LOSS_PCT: float = Field(
        default=2.0,
        description="Max daily loss as percentage of NAV before halting",
    )
    MAX_ORDERS_PER_MINUTE: int = Field(default=10)
    STOPLOSS_DEFAULT_PCT: float = Field(default=3.0)

    # MCMC Derivatives strategy settings
    MCMC_DERIVATIVE_SYMBOL: str = Field(
        default="VN30F2506",
        description="Active futures contract symbol to trade",
    )
    MCMC_NUM_PATHS: int = Field(
        default=10000,
        description="Number of GBM simulation paths",
    )
    MCMC_CONFIDENCE_THRESHOLD: float = Field(
        default=0.75,
        description="Minimum P(UP) or P(DOWN) probability to generate a signal",
    )
    MCMC_ROLLING_WINDOW: int = Field(
        default=50,
        description="Number of historical sessions for Markov transition matrix",
    )
    MCMC_MH_ITERATIONS: int = Field(
        default=1000,
        description="Metropolis-Hastings sampler total iterations",
    )
    MCMC_MH_BURNIN: int = Field(
        default=500,
        description="MH burn-in period (discarded samples)",
    )
    MCMC_FORCE_CLOSE_MINUTE: int = Field(
        default=865,
        description="Minute-of-day to force close T0 position (865 = 14:25)",
    )
    MCMC_POSITION_SIZE: int = Field(
        default=1,
        description="Number of contracts per order (1 = safest for paper test)",
    )
    MCMC_COOLDOWN_TICKS: int = Field(
        default=10,
        description="Quote ticks to wait after a round-trip before next signal",
    )

    # MCMC derivatives — config-driven exit (points / states)
    MCMC_STOP_LOSS_POINTS: float = Field(
        default=2.5,
        description="Signed PnL in index points: exit when pnl <= -this",
    )
    MCMC_TAKE_PROFIT_POINTS: float = Field(
        default=5.0,
        description="Signed PnL in index points: exit when pnl >= this",
    )
    MCMC_MAX_BARS_IN_TRADE: int = Field(
        default=8,
        description="Exit when bars_in_trade >= this (completed OHLC bars while open)",
    )
    MCMC_STATE_FLIP_THRESHOLD: float = Field(
        default=0.75,
        description="Exit when sum of posterior mass on opposite-side states exceeds this",
    )
    MCMC_EXPECTED_RETURN_THRESHOLD: float = Field(
        default=0.1,
        description="Expected next-bar return threshold (after transaction cost) for exit",
    )
    MCMC_TRANSACTION_COST_POINTS: float = Field(
        default=0.2,
        description="Subtracted from expected return before threshold comparison",
    )
    MCMC_EXIT_BULLISH_STATES: str = Field(
        default="1",
        description="Comma-separated or JSON list of state indices (Markov: 1=UP)",
    )
    MCMC_EXIT_BEARISH_STATES: str = Field(
        default="0",
        description="Comma-separated or JSON list of state indices (Markov: 0=DOWN)",
    )

    # --- Live HMM / kill switch ---
    MIN_BALANCE: Optional[float] = Field(
        default=None,
        description="If set, halt live trading when API account balance (VND) is below this",
    )
    DNSE_HTTP_TIMEOUT_SEC: float = Field(
        default=30.0,
        description="Per-request timeout for DNSE SDK calls (seconds)",
    )
    DNSE_API_FAIL_HALT_THRESHOLD: int = Field(
        default=3,
        description="Consecutive API failures (timeout/invalid schema) before halt",
    )
    RECONCILE_BALANCE_EPS_VND: float = Field(
        default=50_000.0,
        description="Max acceptable |expected - actual| cash delta after a fill (VND)",
    )
    DERIVATIVE_FEE_RATE: float = Field(
        default=0.0,
        description="Expected fee as fraction of notional (0 = skip fee check); set from DNSE schedule",
    )

    HMM_LIVE_TRAIN_WINDOW_BARS: Optional[int] = Field(
        default=1000,
        description="Sliding window length (bars) for live HMM fit; None uses expanding window",
    )
    HMM_LIVE_REFIT_EVERY: int = Field(
        default=5,
        description="Refit HMM every N bars in live/sliding mode",
    )
    HMM_LIVE_K_STATES: int = Field(
        default=3,
        description="HMM hidden states k (must match offline tuning; see config/hmm_optimization.yaml)",
    )
    HMM_LIVE_CONFIDENCE: float = Field(
        default=0.65,
        description="HMM signal confidence threshold for live",
    )
    HMM_LIVE_WARMUP_BARS: int = Field(
        default=30,
        description="Minimum bars before first live signal",
    )
    HMM_LIVE_DAYS_BACK: int = Field(
        default=120,
        description="Calendar days of OHLC for hmm_live when --days not passed (need enough 15m bars vs window)",
    )
    HMM_LIVE_DEBUG_TRACE: bool = Field(
        default=False,
        description="Log HMM state probs, guards, feature tail, timestamps each live tick (verbose)",
    )
    HMM_LIVE_ALLOW_INCOMPLETE_BAR: bool = Field(
        default=False,
        description=(
            "If true, do not block duplicate last_bar_unix_ts in live run_once; "
            "allows intrabar updates on the same 15m candle."
        ),
    )
    HMM_LIVE_MAX_BAR_AGE_SEC: int = Field(
        default=900,
        description=(
            "Không cho đặt lệnh nếu nến cuối cũ hơn N giây (stale data guard). "
            "Mặc định 900s = 15 phút."
        ),
    )
    HMM_SYMBOL: str = Field(
        default="VN30F1M",
        description="Symbol OHLC (15m) cho HMM live / scripts/hmm_live (front-month hoặc proxy index)",
    )
    VN30_AUTO_RESOLVE_TRADE_SYMBOL: bool = Field(
        default=False,
        description=(
            "Khi true: lần đầu khởi tạo BeeTradeClient, gọi /instruments lọc VN30 KRX 41I1Gxxxx và đối chiếu "
            "với heuristic lịch (thứ Năm tuần 3). Hai nguồn phải giống nhau mới cập nhật SYMBOL_MAP['VN30F1M']['trade_symbol']; "
            "lệch nhau → thoát process. Tắt (mặc định) cho pytest/offline."
        ),
    )
    VN30_F1M_TRADE_SYMBOL: str = Field(
        default="",
        description=(
            "Mã KRX 9 ký tự cho hợp đồng VN30 front-month (vd. 41I1G5000). Đặt trong .env: áp vào "
            "SYMBOL_MAP['VN30F1M']['trade_symbol'] khi load get_settings() / khởi tạo BeeTradeClient; "
            "ưu tiên hơn VN30_AUTO_RESOLVE (không gọi /instruments). Để trống = không ép từ env."
        ),
    )
    HMM_USE_BASIS: bool = Field(
        default=True,
        description="Include basis (future close − index close) as 4th HMM feature; requires index OHLC",
    )
    HMM_INDEX_SYMBOL: str = Field(
        default="VN30",
        description="Index symbol for basis (OHLC same resolution as HMM_SYMBOL)",
    )
    HMM_USE_OPEN_INTEREST: bool = Field(
        default=False,
        description="Add OI feature from DNSE secdef (openInterestQuantity); requires JSONL cache + live secdef",
    )
    HMM_USE_VOL_CHANGE: bool = Field(
        default=False,
        description="Bật/tắt feature vol_change trong HMM (khuyến nghị tắt khi cần giảm nhiễu).",
    )
    HMM_STATE_SMOOTHING_WINDOW: int = Field(
        default=3,
        description="Làm mượt state bằng majority vote theo cửa sổ trailing N bars (>=1).",
    )
    HMM_STATE_MIN_RUN_BARS: int = Field(
        default=2,
        description="Yêu cầu state mới phải giữ tối thiểu N bars để xác nhận (>=1).",
    )
    HMM_STATE_SCORING_MODE: str = Field(
        default="argmax",
        description="Cách quy đổi state: argmax (mặc định) hoặc strength (theo regime strength).",
    )
    HMM_STATE_STRENGTH_FLAT_QUANTILE: float = Field(
        default=0.35,
        description="Với scoring=strength: quantile biên vùng FLAT (0.05..0.49).",
    )
    HMM_DERIV_LO_MIN_QTY: int = Field(
        default=10,
        description=(
            "Phái sinh HMM live: số hợp đồng >= mức này dùng LO (chờ khớp); "
            "dưới mức dùng MTL. Đặt 11 nếu chỉ muốn LO từ 11 hợp đồng trở lên."
        ),
    )

    # --- Telegram ---
    TELEGRAM_ENABLED: bool = Field(default=False)
    TELEGRAM_BOT_TOKEN: str = Field(default="", description="Bot token from @BotFather")
    TELEGRAM_CHAT_ID: str = Field(
        default="",
        description="Default chat id for outbound notifications",
    )
    TELEGRAM_ALLOWED_CHAT_IDS: str = Field(
        default="",
        description="Comma-separated chat ids allowed for /pause /resume (empty = same as CHAT_ID)",
    )

    # Cached trading token (set at runtime, not from env)
    _trading_token: Optional[str] = None


def get_settings() -> Settings:
    """Get singleton settings instance."""
    s = Settings()
    apply_vn30_f1m_trade_symbol_from_settings(s)
    return s
