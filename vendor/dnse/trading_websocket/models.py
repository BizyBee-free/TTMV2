"""Data models for market data and private channel updates.

All models support parsing from both abbreviated (MessagePack) and full (JSON) field names.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any, Tuple


def parse_timestamp(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        seconds = v.get("Seconds", v.get("seconds", 0))
        nanos = v.get("Nanos", v.get("nanos", 0))
        return float(seconds) + float(nanos) * 1e-9
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def proto_timestamp_to_str(v: Any) -> Optional[str]:
    if isinstance(v, dict):
        seconds = v.get("Seconds", v.get("seconds", 0))
        nanos = v.get("Nanos", v.get("nanos", 0))
        dt = datetime.fromtimestamp(seconds + nanos / 1_000_000_000, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return None


def _dnse_stream_ts_to_epoch_int(ts: Any) -> int:
    """Stream DNSE (PyPI dnse) thường gửi ``timestamp`` ms; Ohlc.time dùng int (giây)."""
    if ts is None:
        return 0
    try:
        v = int(float(ts))
    except (TypeError, ValueError):
        return 0
    if v > 10_000_000_000:
        return v // 1000
    return v


def _dnse_wire_event_seconds(data: Dict[str, Any]) -> int:
    """Lấy epoch giây từ ``sending_time`` / ``multicast_receive_time`` (protobuf map trên wire DNSE)."""
    for key in ("sending_time", "SendingTime", "multicast_receive_time", "MulticastReceiveTime"):
        d = data.get(key)
        if isinstance(d, dict):
            s = d.get("seconds")
            if s is not None:
                try:
                    return int(s)
                except (TypeError, ValueError):
                    continue
    return 0


def _timeframe_to_resolution_minutes(tf: Any) -> int:
    """``timeframe`` kiểu 1m / 15m / 1h -> số phút cho :class:`Ohlc`.resolution."""
    if tf is None:
        return 0
    s = str(tf).strip().lower()
    if not s:
        return 0
    try:
        if s.endswith("m"):
            base = s[:-1]
            return int(float(base)) if base else 1
        if s.endswith("h"):
            base = s[:-1]
            return int(float(base) * 60) if base else 60
        if s.endswith("d"):
            base = s[:-1]
            return int(float(base) * 60 * 24) if base else 1440
        if s.endswith("w"):
            base = s[:-1]
            return int(float(base) * 60 * 24 * 7) if base else 10080
    except ValueError:
        return 0
    return 0


@dataclass
class PriceLevel:
    price: float
    quantity: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriceLevel":
        return cls(
            price=data.get("Price") or data.get("price"),
            quantity=data.get("qtty") or data.get("Qtty"),
        )


@dataclass
class Trade:
    marketId: int
    boardId: int
    isin: str
    symbol: str
    price: float
    quantity: int
    totalVolumeTraded: int
    grossTradeAmount: float
    highestPrice: float
    lowestPrice: float
    openPrice: float
    tradingSessionId: int
    event_time_sec: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        ev = _dnse_wire_event_seconds(data)
        t_val = str(data.get("T") or data.get("t") or "")
        # WebSocket msgpack: camelCase (matchPrice, matchQtty). Legacy: PascalCase / snake_case.
        mp = data.get("matchPrice")
        if mp is None:
            mp = data.get("MatchPrice")
        if mp is None:
            mp = data.get("match_price")
        simple_px = data.get("price")
        if simple_px is None:
            simple_px = data.get("Price")
        # Nhánh tối giản: T=t chỉ có price/Price (không có match*)
        if t_val == "t" and mp is None and simple_px is not None:
            vol = int(data.get("volume") or data.get("Volume") or 0)
            px = float(simple_px or 0.0)
            return cls(
                marketId=0,
                boardId=0,
                isin="",
                symbol=str(data.get("symbol") or ""),
                price=px,
                quantity=vol,
                totalVolumeTraded=vol,
                grossTradeAmount=px * vol,
                highestPrice=px,
                lowestPrice=px,
                openPrice=px,
                tradingSessionId=0,
                event_time_sec=ev,
            )
        if t_val == "t" and mp is None and simple_px is None:
            vol = int(data.get("volume") or data.get("Volume") or 0)
            return cls(
                marketId=0,
                boardId=0,
                isin="",
                symbol=str(data.get("symbol") or ""),
                price=0.0,
                quantity=vol,
                totalVolumeTraded=vol,
                grossTradeAmount=0.0,
                highestPrice=0.0,
                lowestPrice=0.0,
                openPrice=0.0,
                tradingSessionId=0,
                event_time_sec=ev,
            )
        mq = data.get("matchQtty")
        if mq is None:
            mq = data.get("MatchQtty")
        if mq is None:
            mq = data.get("match_qtty")
        if mq is None:
            mq = data.get("volume") or data.get("Volume") or 0
        px_final = float(mp) if mp is not None else float(simple_px or 0.0)
        return cls(
            marketId=data.get("market_id", 0) or data.get("MarketId", 0),
            boardId=data.get("board_id", 0) or data.get("BoardId", 0),
            isin=data.get("isin", "") or data.get("Isin", ""),
            symbol=data.get("Symbol") or data.get("symbol"),
            price=px_final,
            quantity=int(mq or 0),
            totalVolumeTraded=data.get("TotalVolumeTraded", 0) or data.get("total_volume_traded", 0),
            grossTradeAmount=data.get("GrossTradeAmount", 0) or data.get("gross_trade_amount", 0),
            highestPrice=data.get("HighestPrice", 0) or data.get("highest_price", 0),
            lowestPrice=data.get("LowestPrice", 0) or data.get("lowest_price", 0),
            openPrice=data.get("OpenPrice", 0) or data.get("open_price", 0),
            tradingSessionId=data.get("TradingSessionId", 0) or data.get("trading_session_id", 0),
            event_time_sec=ev,
        )


@dataclass
class TradeExtra:
    marketId: int
    boardId: int
    isin: str
    symbol: str
    price: float
    quantity: int
    side: int
    avgPrice: float
    totalVolumeTraded: int
    grossTradeAmount: float
    highestPrice: float
    lowestPrice: float
    openPrice: float
    tradingSessionId: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradeExtra":
        return cls(
            marketId=data.get("market_id", 0) or data.get("MarketId", 0),
            boardId=data.get("board_id", 0) or data.get("BoardId", 0),
            isin=data.get("isin", "") or data.get("Isin", ""),
            symbol=data.get("Symbol") or data.get("symbol"),
            price=data.get("MatchPrice", 0.0) or data.get("match_price", 0.0),
            quantity=data.get("MatchQtty", 0) or data.get("match_qtty", 0),
            side=data.get("Side", 0) or data.get("side", 0),
            avgPrice=data.get("AvgPrice", 0) or data.get("avg_price", 0),
            totalVolumeTraded=data.get("TotalVolumeTraded", 0) or data.get("total_volume_traded", 0),
            grossTradeAmount=data.get("GrossTradeAmount", 0) or data.get("gross_trade_amount", 0),
            highestPrice=data.get("HighestPrice", 0) or data.get("highest_price", 0),
            lowestPrice=data.get("LowestPrice", 0) or data.get("lowest_price", 0),
            openPrice=data.get("OpenPrice", 0) or data.get("open_price", 0),
            tradingSessionId=data.get("TradingSessionId", 0) or data.get("trading_session_id", 0),
        )


@dataclass
class MarketIndex:
    index_name: str
    changed_ratio: float
    changed_value: float
    fluctuation_steadiness_issue_count: int
    fluctuation_down_issue_count: int
    fluctuation_up_issue_count: int
    fluctuation_lower_limit_issue_count: int
    fluctuation_upper_limit_issue_count: int
    fluctuation_down_issue_volume: int
    fluctuation_up_issue_volume: int
    fluctuation_steadiness_issue_volume: int
    currency_code: str
    index_type_code: str
    lowest_value_indexes: float
    highest_value_indexes: float
    prior_value_indexes: float
    value_indexes: float
    contauct_acc_trd_val: float
    contauct_acc_trd_vol: int
    blk_trd_acc_trd_val: float
    blk_trd_acc_trd_vol: int
    gross_trade_amount: float
    total_volume_traded: int
    market_index_class: int
    market_id: int
    trading_session_id: int
    transact_time: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketIndex":
        return cls(
            index_name=data.get("IndexName") or data.get("index_name"),
            changed_ratio=data.get("ChangedRatio") or data.get("changed_ratio"),
            changed_value=data.get("ChangedValue") or data.get("changed_value"),
            fluctuation_steadiness_issue_count=data.get("FluctuationSteadinessIssueCount") or data.get("fluctuation_steadiness_issue_count"),
            fluctuation_down_issue_count=data.get("FluctuationDownIssueCount") or data.get("fluctuation_down_issue_count"),
            fluctuation_up_issue_count=data.get("FluctuationUpIssueCount") or data.get("fluctuation_up_issue_count"),
            fluctuation_lower_limit_issue_count=data.get("FluctuationLowerLimitIssueCount") or data.get("fluctuation_lower_limit_issue_count"),
            fluctuation_upper_limit_issue_count=data.get("FluctuationUpperLimitIssueCount") or data.get("fluctuation_upper_limit_issue_count"),
            fluctuation_down_issue_volume=data.get("FluctuationDownIssueVolume") or data.get("fluctuation_down_issue_volume"),
            fluctuation_up_issue_volume=data.get("FluctuationUpIssueVolume") or data.get("fluctuation_up_issue_volume"),
            fluctuation_steadiness_issue_volume=data.get("FluctuationSteadinessIssueVolume") or data.get("fluctuation_steadiness_issue_volume"),
            currency_code=data.get("CurrencyCode") or data.get("currency_code"),
            index_type_code=data.get("IndexTypeCode") or data.get("index_type_code"),
            lowest_value_indexes=data.get("LowestValueIndexes") or data.get("lowest_value_indexes"),
            highest_value_indexes=data.get("HighestValueIndexes") or data.get("highest_value_indexes"),
            prior_value_indexes=data.get("PriorValueIndexes") or data.get("prior_value_indexes"),
            value_indexes=data.get("ValueIndexes") or data.get("value_indexes"),
            contauct_acc_trd_val=data.get("ContauctAccTrdVal") or data.get("contauct_acc_trd_val"),
            contauct_acc_trd_vol=data.get("ContauctAccTrdVol") or data.get("contauct_acc_trd_vol"),
            blk_trd_acc_trd_val=data.get("BlkTrdAccTrdVal") or data.get("blk_trd_acc_trd_val"),
            blk_trd_acc_trd_vol=data.get("BlkTrdAccTrdVol") or data.get("blk_trd_acc_trd_vol"),
            gross_trade_amount=data.get("GrossTradeAmount") or data.get("gross_trade_amount"),
            total_volume_traded=data.get("TotalVolumeTraded") or data.get("total_volume_traded"),
            market_index_class=data.get("MarketIndexClass") or data.get("market_index_class"),
            market_id=data.get("MarketId") or data.get("market_id"),
            trading_session_id=data.get("TradingSessionId") or data.get("trading_session_id"),
            transact_time=proto_timestamp_to_str(data.get("TransactTime") or data.get("transact_time")),
        )


@dataclass
class ExpectedPrice:
    marketId: int
    boardId: int
    isin: str
    symbol: str
    closePrice: float
    expectedTradePrice: float
    expectedTradeQuantity: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedPrice":
        t_val = str(data.get("T") or data.get("t") or "")
        if t_val == "e":
            v = int(data.get("volume") or 0)
            return cls(
                marketId=0,
                boardId=0,
                isin="",
                symbol=str(data.get("symbol") or ""),
                closePrice=0.0,
                expectedTradePrice=float(data.get("price") or 0.0),
                expectedTradeQuantity=v,
            )
        return cls(
            marketId=data.get("market_id", 0) or data.get("MarketId", 0),
            boardId=data.get("board_id", 0) or data.get("BoardId", 0),
            isin=data.get("isin", "") or data.get("Isin", ""),
            symbol=data.get("Symbol") or data.get("symbol"),
            closePrice=data.get("close_price", 0.0) or data.get("ClosePrice", 0),
            expectedTradePrice=data.get("expected_trade_price", 0.0) or data.get("ExpectedTradePrice", 0.0),
            expectedTradeQuantity=data.get("expected_trade_quantity", 0) or data.get("ExpectedTradeQuantity", 0),
        )


def _normalize_t_stream(raw: Any) -> str:
    """Giống ``_normalize_stream_t_key`` trong client: msgpack có thể gửi ``T`` dạng bytes."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("ascii", errors="replace").lower()
    if isinstance(raw, int) and 32 <= raw <= 126:
        return chr(raw).lower()
    return str(raw).lower()


def _open_interest_int_from_payload(data: Dict[str, Any]) -> int:
    """OI từ frame secdef — ưu tiên ``open_interest_quantity`` như openapi-sdk Python."""
    for key in (
        "open_interest_quantity",
        "openInterestQuantity",
        "OpenInterestQuantity",
    ):
        if key in data and data[key] is not None:
            try:
                return int(float(data[key]))
            except (TypeError, ValueError):
                return 0
    return 0


@dataclass
class SecurityDefinition:
    marketId: int
    boardId: int
    symbol: str
    isin: str
    productGrpId: int
    securityGroupId: int
    basicPrice: float
    ceilingPrice: float
    floorPrice: float
    openInterestQuantity: int
    securityStatus: int
    symbolAdminStatusCode: int
    symbolTradingMethodStatusCode: int
    symbolTradingSanctionStatusCode: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityDefinition":
        t_norm = _normalize_t_stream(
            data.get("T") or data.get("t") or data.get("Type") or data.get("type")
        )
        if t_norm == "sd":
            _oi_sd_i = _open_interest_int_from_payload(data)
            return cls(
                symbol=str(data.get("symbol") or data.get("Symbol") or ""),
                marketId=int(data.get("market_id") or data.get("MarketId") or 0),
                boardId=int(data.get("board_id") or data.get("BoardId") or 0),
                isin=str(data.get("isin") or data.get("Isin") or ""),
                productGrpId=int(data.get("product_grp_id") or data.get("ProductGrpId") or 0),
                securityGroupId=int(data.get("security_group_id") or data.get("SecurityGroupId") or 0),
                basicPrice=float(
                    data.get("ref_price")
                    or data.get("basic_price")
                    or data.get("BasicPrice")
                    or 0.0
                ),
                ceilingPrice=float(
                    data.get("ceiling")
                    or data.get("ceiling_price")
                    or data.get("CeilingPrice")
                    or 0.0
                ),
                floorPrice=float(
                    data.get("floor")
                    or data.get("floor_price")
                    or data.get("FloorPrice")
                    or 0.0
                ),
                openInterestQuantity=_oi_sd_i,
                securityStatus=int(data.get("security_status") or data.get("SecurityStatus") or 0),
                symbolAdminStatusCode=int(
                    data.get("symbol_admin_status_code") or data.get("SymbolAdminStatusCode") or 0
                ),
                symbolTradingMethodStatusCode=int(
                    data.get("symbol_trading_method_status_code")
                    or data.get("SymbolTradingMethodStatusCode")
                    or 0
                ),
                symbolTradingSanctionStatusCode=int(
                    data.get("symbol_trading_sanction_status_code")
                    or data.get("SymbolTradingSanctionStatusCode")
                    or 0
                ),
            )
        _oi_i = _open_interest_int_from_payload(data)
        return cls(
            symbol=data.get("symbol") or data.get("Symbol"),
            marketId=data.get("market_id", 0) or data.get("MarketId", 0),
            boardId=data.get("board_id", 0) or data.get("BoardId", 0),
            isin=data.get("isin", "") or data.get("Isin", ""),
            productGrpId=data.get("product_grp_id", 0) or data.get("ProductGrpId", 0),
            securityGroupId=data.get("security_group_id", 0) or data.get("SecurityGroupId", 0),
            basicPrice=data.get("basic_price", 0.0) or data.get("BasicPrice", 0.0),
            ceilingPrice=data.get("ceiling_price", 0.0) or data.get("CeilingPrice", 0.0),
            floorPrice=data.get("floor_price", 0.0) or data.get("FloorPrice", 0.0),
            openInterestQuantity=_oi_i,
            securityStatus=data.get("security_status", 0) or data.get("SecurityStatus", 0),
            symbolAdminStatusCode=data.get("symbol_admin_status_code", 0) or data.get("SymbolAdminStatusCode", 0),
            symbolTradingMethodStatusCode=data.get("symbol_trading_method_status_code", 0) or data.get("SymbolTradingMethodStatusCode", 0),
            symbolTradingSanctionStatusCode=data.get("symbol_trading_sanction_status_code", 0) or data.get("SymbolTradingSanctionStatusCode", 0),
        )


@dataclass
class Quote:
    marketId: int
    boardId: int
    symbol: str
    isin: str
    bid: List[PriceLevel]
    offer: List[PriceLevel]
    totalOfferQtty: float
    totalBidQtty: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Quote":
        t_val = str(data.get("T") or data.get("t") or "")
        if t_val == "q" or (
            data.get("bid_price") is not None
            and data.get("ask_price") is not None
            and not (data.get("Bid") or data.get("bid"))
        ):
            bp = data.get("bid_price")
            ap = data.get("ask_price")
            bv = int(data.get("bid_volume") or 0)
            av = int(data.get("ask_volume") or 0)
            bids = [PriceLevel(price=float(bp), quantity=bv)] if bp is not None else []
            offers = [PriceLevel(price=float(ap), quantity=av)] if ap is not None else []
            return cls(
                marketId=0,
                boardId=0,
                symbol=str(data.get("symbol") or ""),
                isin="",
                bid=bids,
                offer=offers,
                totalOfferQtty=float(av),
                totalBidQtty=float(bv),
            )
        bids_data = data.get("Bid") or data.get("bid") or []
        bids = [PriceLevel.from_dict(level) for level in bids_data]

        offer_data = data.get("Offer") or data.get("offer") or []
        offers = [PriceLevel.from_dict(level) for level in offer_data]

        return cls(
            symbol=data.get("Symbol") or data.get("symbol"),
            marketId=data.get("market_id", 0) or data.get("MarketId", 0),
            boardId=data.get("board_id", 0) or data.get("BoardId", 0),
            isin=data.get("isin", "") or data.get("Isin", ""),
            bid=bids,
            offer=offers,
            totalOfferQtty=data.get("total_offer_qtty") or data.get("TotalOfferQtty"),
            totalBidQtty=data.get("total_bid_qtty") or data.get("TotalBidQtty"),
        )

    @property
    def best_bid(self) -> Optional[Tuple[float, int]]:
        if not self.bid:
            return None
        return self.bid[0].price, self.bid[0].quantity

    @property
    def best_ask(self) -> Optional[Tuple[float, int]]:
        if not self.offer:
            return None
        return self.offer[0].price, self.offer[0].quantity

    @property
    def spread(self) -> Optional[float]:
        bid = self.best_bid
        offer = self.best_ask
        if bid and offer:
            return offer[0] - bid[0]
        return None


@dataclass
class Ohlc:
    symbol: str
    resolution: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    time: int
    lastUpdated: int
    type: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Ohlc":
        def _dec(v: Any) -> Decimal:
            if v is None:
                return Decimal("0")
            if isinstance(v, Decimal):
                return v
            try:
                return Decimal(str(v))
            except Exception:
                return Decimal("0")

        def _iv(v: Any) -> int:
            if v is None:
                return 0
            try:
                return int(float(v))
            except (TypeError, ValueError):
                return 0

        def _norm_t(v: Any) -> str:
            if v is None:
                return ""
            if isinstance(v, bytes):
                return v.decode("ascii", errors="replace").lower()
            if isinstance(v, int) and 32 <= v <= 126:
                return chr(v).lower()
            return str(v).lower()

        _tr = data.get("T")
        if _tr is None:
            _tr = data.get("t")
        t_val = _norm_t(_tr)
        o_stream = data.get("open")
        if o_stream is None:
            o_stream = data.get("Open")
        h_stream = data.get("high")
        if h_stream is None:
            h_stream = data.get("High")
        l_stream = data.get("low")
        if l_stream is None:
            l_stream = data.get("Low")
        c_stream = data.get("close")
        if c_stream is None:
            c_stream = data.get("Close")
        if t_val == "b" or (
            data.get("timeframe") is not None
            and o_stream is not None
            and data.get("Resolution") is None
            and data.get("resolution") is None
        ):
            tf = data.get("timeframe")
            tsi = _dnse_stream_ts_to_epoch_int(
                data.get("timestamp") or data.get("time") or data.get("Time")
            )
            res_min = _timeframe_to_resolution_minutes(tf)
            return cls(
                symbol=str(data.get("symbol") or data.get("Symbol") or "").strip(),
                resolution=res_min,
                open=_dec(o_stream),
                high=_dec(h_stream),
                low=_dec(l_stream),
                close=_dec(c_stream),
                volume=_iv(data.get("volume") or data.get("Volume")),
                time=tsi,
                lastUpdated=tsi,
                type=str(tf or ""),
            )

        sym = (data.get("symbol") or data.get("Symbol") or "") or ""
        sym = str(sym).strip()
        res = data.get("resolution") or data.get("Resolution")
        if res is not None and not isinstance(res, int):
            try:
                res = int(res)
            except (TypeError, ValueError):
                res = 0
        if res is None:
            res = 0
        t_raw = data.get("time") or data.get("Time")
        lu_raw = data.get("lastUpdated") or data.get("LastUpdated")
        try:
            t_i = int(t_raw) if t_raw is not None else 0
        except (TypeError, ValueError):
            t_i = 0
        try:
            lu_i = int(lu_raw) if lu_raw is not None else 0
        except (TypeError, ValueError):
            lu_i = 0
        typ = data.get("type") or data.get("Type")
        typ_s = "" if typ is None else str(typ)

        return cls(
            symbol=sym,
            resolution=res,
            open=_dec(data.get("open") or data.get("Open")),
            high=_dec(data.get("high") or data.get("High")),
            low=_dec(data.get("low") or data.get("Low")),
            close=_dec(data.get("close") or data.get("Close")),
            volume=_iv(data.get("volume") or data.get("Volume")),
            time=t_i,
            lastUpdated=lu_i,
            type=typ_s,
        )


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str
    order_type: str
    status: str
    quantity: int
    filled_quantity: int
    price: Optional[Decimal]
    average_fill_price: Optional[Decimal]
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        return cls(
            order_id=data.get("oid") or data.get("order_id"),
            symbol=data.get("S") or data.get("symbol"),
            side=data.get("sd") or data.get("side"),
            order_type=data.get("ot") or data.get("order_type"),
            status=data.get("st") or data.get("status"),
            quantity=data.get("q") or data.get("quantity"),
            filled_quantity=data.get("fq") or data.get("filled_quantity"),
            price=Decimal(str(data["p"])) if (data.get("p") or data.get("price")) else None,
            average_fill_price=Decimal(str(data["ap"])) if (data.get("ap") or data.get("average_fill_price")) else None,
            timestamp=datetime.fromtimestamp((data.get("t") or data.get("timestamp")) / 1000),
        )


@dataclass
class Position:
    symbol: str
    quantity: int
    average_price: Decimal
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pl: Decimal
    unrealized_pl_percent: Decimal
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        return cls(
            symbol=data.get("S") or data.get("symbol"),
            quantity=data.get("q") or data.get("quantity"),
            average_price=Decimal(str(data.get("ap") or data.get("average_price"))),
            market_value=Decimal(str(data.get("mv") or data.get("market_value"))),
            cost_basis=Decimal(str(data.get("cb") or data.get("cost_basis"))),
            unrealized_pl=Decimal(str(data.get("upl") or data.get("unrealized_pl"))),
            unrealized_pl_percent=Decimal(str(data.get("uplp") or data.get("unrealized_pl_percent"))),
            timestamp=datetime.fromtimestamp((data.get("t") or data.get("timestamp")) / 1000),
        )


@dataclass
class AccountUpdate:
    cash: Decimal
    buying_power: Decimal
    portfolio_value: Decimal
    equity: Decimal
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccountUpdate":
        return cls(
            cash=Decimal(str(data.get("c") or data.get("cash"))),
            buying_power=Decimal(str(data.get("bp") or data.get("buying_power"))),
            portfolio_value=Decimal(str(data.get("pv") or data.get("portfolio_value"))),
            equity=Decimal(str(data.get("eq") or data.get("equity"))),
            timestamp=datetime.fromtimestamp((data.get("t") or data.get("timestamp")) / 1000),
        )
