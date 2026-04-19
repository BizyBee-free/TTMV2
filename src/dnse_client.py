"""High-level wrapper around the DNSE SDK with logging, error handling,
and TradingToken lifecycle management."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config import Settings, get_settings
from src.logger import get_logger
from src.rate_limiter import SlidingWindowRateLimiter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor", "dnse"))
from dnse import DNSEClient

logger = get_logger("dnse_client")


# DNSE API error codes that indicate retryable conditions
RETRYABLE_CODES = {"OA-104", "TIME_OUT", "TOO_MANY_REQUESTS"}


class TradingTokenManager:
    """Manages the OTP-based trading token lifecycle.

    Trading tokens are required for order placement/modification/cancellation.
    They are obtained by:
      1. Calling send_email_otp() to trigger OTP delivery
      2. User provides the OTP passcode
      3. Calling create_trading_token() with the passcode
    """

    def __init__(self, sdk_client: DNSEClient):
        self._client = sdk_client
        self._token: Optional[str] = None
        self._token_time: float = 0
        self._token_ttl: float = 3600  # assume 1h validity; adjust if DNSE docs specify

    @property
    def token(self) -> Optional[str]:
        return self._token

    @property
    def is_valid(self) -> bool:
        if not self._token:
            return False
        return (time.time() - self._token_time) < self._token_ttl

    def request_otp(self) -> Tuple[int, str]:
        """Send OTP to registered email. Returns (status, body)."""
        logger.info("Requesting email OTP...")
        status, body = self._client.send_email_otp()
        if status == 200:
            logger.info("OTP sent to email successfully")
        else:
            logger.error("Failed to send OTP", extra={"status": status, "body": body})
        return status, body

    def activate_token(self, passcode: str, otp_type: str = "email_otp") -> str:
        """Exchange OTP passcode for a trading token."""
        logger.info("Creating trading token...")
        status, body = self._client.create_trading_token(
            otp_type=otp_type, passcode=passcode
        )
        if status == 200:
            data = json.loads(body) if isinstance(body, str) else body
            self._token = data if isinstance(data, str) else data.get("tradingToken", data.get("token", body))
            self._token_time = time.time()
            logger.info("Trading token activated successfully")
            return self._token
        else:
            logger.error("Failed to create trading token", extra={"status": status, "body": body})
            raise RuntimeError(f"Trading token creation failed: {status} {body}")

    def set_token(self, token: str) -> None:
        """Manually set a trading token (e.g. from saved session)."""
        self._token = token
        self._token_time = time.time()

    def ensure_token(self) -> str:
        """Return trading token for order APIs (non-interactive).

        Live runner must never block on OTP prompt. We only use:
        1) token preloaded from settings/file, or
        2) token already activated by a dedicated OTP script.
        """
        if self._token:
            if not self.is_valid:
                logger.warning(
                    "Trading token age exceeded local TTL; still using current token "
                    "and letting DNSE API validate it.",
                )
            return self._token

        msg = (
            "DNSE trading token is missing. Auto OTP prompt is disabled in live mode. "
            "Run scripts/request_dnse_trading_token.py and set DNSE_TRADING_TOKEN "
            "(or DNSE_TRADING_TOKEN_FILE) before starting --submit."
        )
        logger.error(msg)
        raise RuntimeError(msg)


class BeeTradeClient:
    """Main client wrapping DNSEClient with logging, error handling, and token management."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        self._sdk = DNSEClient(
            api_key=self.settings.DNSE_API_KEY,
            api_secret=self.settings.DNSE_API_SECRET,
            base_url=self.settings.DNSE_BASE_URL,
        )

        self.token_manager = TradingTokenManager(self._sdk)
        self._order_limiter = SlidingWindowRateLimiter(
            limit=self.settings.MAX_ORDERS_PER_MINUTE,
            window=60.0,
        )
        self._load_trading_token_from_settings()
        logger.info(
            "BeeTradeClient initialized",
            extra={
                "base_url": self.settings.DNSE_BASE_URL,
                "account": self.settings.DNSE_ACCOUNT_NO,
                "paper_mode": self.settings.PAPER_MODE,
            },
        )
        try:
            from src import config as _bee_cfg

            _bee_cfg.ensure_vn30_f1m_trade_symbol_resolved(self)
        except SystemExit:
            raise
        except Exception as e:
            logger.error("VN30 auto contract resolve failed", extra={"error": str(e)})
            raise

    def _load_trading_token_from_settings(self) -> None:
        """Pre-load token from env or file so live runs do not require console OTP."""
        tok = (getattr(self.settings, "DNSE_TRADING_TOKEN", None) or "").strip()
        if not tok:
            path_str = (getattr(self.settings, "DNSE_TRADING_TOKEN_FILE", None) or "").strip()
            if path_str:
                p = Path(path_str)
                if p.is_file():
                    try:
                        tok = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
                    except OSError as e:
                        logger.warning(
                            "Could not read DNSE_TRADING_TOKEN_FILE",
                            extra={"path": path_str, "error": str(e)},
                        )
        if tok:
            self.token_manager.set_token(tok)
            logger.info("Trading token loaded from settings (env or file)")

    @property
    def sdk(self) -> DNSEClient:
        """Direct access to underlying SDK client."""
        return self._sdk

    @property
    def account_no(self) -> str:
        return self.settings.DNSE_ACCOUNT_NO

    @property
    def order_limiter(self) -> SlidingWindowRateLimiter:
        return self._order_limiter

    # ---- Account endpoints ----

    def get_accounts(self) -> Dict[str, Any]:
        """Get list of trading accounts."""
        return self._call("get_accounts")

    def get_balances(self) -> Dict[str, Any]:
        """Get account balances."""
        return self._call("get_balances", self.account_no)

    def get_positions(self, market_type: str = "STOCK") -> Dict[str, Any]:
        """List positions (OpenAPI V2). Prefer this over legacy *deals* naming."""
        return self._call("get_positions", self.account_no, market_type)

    def get_deals(self, market_type: str = "STOCK") -> Dict[str, Any]:
        """List portfolio positions — tries ``/positions`` first, then ``/deals`` if 404."""
        result = self.get_positions(market_type)
        if (
            self.settings.DNSE_POSITIONS_FALLBACK_TO_DEALS
            and result.get("status") == 404
        ):
            logger.info(
                "get_positions returned 404, falling back to get_deals",
                extra={"market_type": market_type},
            )
            return self._call("get_deals", self.account_no, market_type)
        return result

    def get_ppse(self, symbol: str, price: float, loan_package_id: int,
                 market_type: str = "STOCK") -> Dict[str, Any]:
        """Get purchasing power / selling estimation."""
        return self._call(
            "get_ppse", self.account_no, market_type, symbol, price, loan_package_id
        )

    def get_loan_packages(self, market_type: str = "STOCK",
                          symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get available loan packages."""
        return self._call("get_loan_packages", self.account_no, market_type, symbol)

    def resolve_loan_package_id(
        self,
        market_type: str = "STOCK",
        symbol: Optional[str] = None,
    ) -> int:
        """Resolve a valid loanPackageId from DNSE loan-packages endpoint.

        Returns first available package id. Raises RuntimeError when no id is found.
        """
        res = self.get_loan_packages(market_type=market_type, symbol=symbol)
        st = res.get("status")
        data = res.get("data")
        if st is None or (isinstance(st, int) and st >= 400):
            raise RuntimeError(f"get_loan_packages failed: HTTP_{st} data={data}")

        packs = []
        if isinstance(data, dict):
            lp = data.get("loanPackages")
            if isinstance(lp, list):
                packs = lp
        elif isinstance(data, list):
            packs = data

        for item in packs:
            if isinstance(item, dict):
                pid = item.get("id")
                if isinstance(pid, int):
                    return pid
                if isinstance(pid, str) and pid.isdigit():
                    return int(pid)

        raise RuntimeError(
            f"No valid loan package id for market_type={market_type} symbol={symbol}. data={data}"
        )

    # ---- Order endpoints ----

    def get_orders(self, market_type: str = "STOCK") -> Dict[str, Any]:
        """Get today's orders."""
        return self._call("get_orders", self.account_no, market_type)

    def get_order_detail(self, order_id: str, market_type: str = "STOCK") -> Dict[str, Any]:
        """Get specific order details."""
        return self._call("get_order_detail", self.account_no, order_id, market_type)

    def get_order_history(self, market_type: str = "STOCK",
                          from_date: Optional[str] = None,
                          to_date: Optional[str] = None) -> Dict[str, Any]:
        """Get historical orders."""
        return self._call(
            "get_order_history", self.account_no, market_type,
            from_date=from_date, to_date=to_date
        )

    def place_order(self, symbol: str, side: str, order_type: str, price: float,
                    quantity: int, loan_package_id: int,
                    market_type: str = "STOCK") -> Dict[str, Any]:
        """Place a new order. side should be 'NB' (buy) or 'NS' (sell)."""
        self._order_limiter.acquire()
        token = self.token_manager.ensure_token()
        payload = {
            "accountNo": self.account_no,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "price": price,
            "quantity": quantity,
            "loanPackageId": loan_package_id,
        }
        logger.info("Placing order", extra={"payload": payload, "market_type": market_type})
        return self._call(
            "post_order", market_type, payload, token, "NORMAL"
        )

    def modify_order(self, order_id: str, price: Optional[float] = None,
                     quantity: Optional[int] = None,
                     market_type: str = "STOCK") -> Dict[str, Any]:
        """Modify an existing order. For DERIVATIVE, only price OR quantity per request."""
        self._order_limiter.acquire()
        token = self.token_manager.ensure_token()
        payload = {}
        if price is not None:
            payload["price"] = price
        if quantity is not None:
            payload["quantity"] = quantity

        logger.info("Modifying order", extra={"order_id": order_id, "payload": payload})
        return self._call(
            "put_order", self.account_no, order_id, market_type, payload, token, "NORMAL"
        )

    def cancel_order(self, order_id: str, market_type: str = "STOCK") -> Dict[str, Any]:
        """Cancel an existing order."""
        self._order_limiter.acquire()
        token = self.token_manager.ensure_token()
        logger.info("Canceling order", extra={"order_id": order_id})
        return self._call(
            "cancel_order", self.account_no, order_id, market_type, token, "NORMAL"
        )

    def close_position(self, position_id: str,
                       market_type: str = "DERIVATIVE") -> Dict[str, Any]:
        """Close a derivative position (T0)."""
        token = self.token_manager.ensure_token()
        logger.info("Closing position", extra={"position_id": position_id})
        return self._call(
            "close_position", position_id, market_type, {}, token
        )

    # ---- Price / Market data endpoints ----

    def get_security_definition(
        self,
        symbol: str,
        board_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """GET ``/price/{symbol}/secdef`` — ceiling/floor/basic price, **open interest** (OI).

        Derivative OI is in ``openInterestQuantity`` / ``open_interest_quantity`` in the JSON body
        (see ``vendor/dnse/trading_websocket/models.py`` ``SecurityDefinition``).

        Phái sinh thường cần ``boardId`` (ví dụ G1). Nếu ``board_id`` không truyền, lấy phần đầu
        của ``DNSE_WS_BOARD_ID`` trong settings (trước dấu phẩy).
        """
        bid = (board_id or "").strip() if board_id else ""
        if not bid:
            raw = (getattr(self.settings, "DNSE_WS_BOARD_ID", None) or "").strip()
            if raw:
                bid = raw.split(",")[0].strip()
        bid_arg: Optional[str] = bid if bid else None
        return self._call("get_security_definition", symbol, bid_arg)

    def get_ohlc(self, bar_type: str, **kwargs) -> Dict[str, Any]:
        """Get OHLC data."""
        return self._call("get_ohlc", bar_type, kwargs if kwargs else None)

    def get_instruments(self, **kwargs) -> Dict[str, Any]:
        """Get instruments list."""
        return self._call("get_instruments", **kwargs)

    def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        """Get latest trade for a symbol."""
        return self._call("get_latest_trade", symbol)

    # ---- Internal helpers ----

    def _call(self, method_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Call SDK method with logging, optional per-request timeout, and error parsing."""
        t0 = time.time()
        fn = getattr(self._sdk, method_name)
        timeout = float(self.settings.DNSE_HTTP_TIMEOUT_SEC)

        def _invoke() -> Tuple[Any, Any]:
            return fn(*args, **kwargs)

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_invoke)
                try:
                    status, body = fut.result(timeout=timeout)
                except FuturesTimeout:
                    elapsed = (time.time() - t0) * 1000
                    logger.error(
                        f"API timeout: {method_name}",
                        extra={"elapsed_ms": round(elapsed, 1), "timeout_sec": timeout},
                    )
                    return {
                        "status": 408,
                        "data": {"code": "CLIENT_TIMEOUT", "method": method_name},
                        "elapsed_ms": round(elapsed, 1),
                    }

            elapsed = (time.time() - t0) * 1000

            logger.debug(
                f"API call: {method_name}",
                extra={"status": status, "elapsed_ms": round(elapsed, 1)},
            )

            parsed = self._parse_response(status, body)

            if status and status >= 400:
                error_code = ""
                if isinstance(parsed, dict):
                    error_code = parsed.get("code", "")
                logger.warning(
                    f"API error: {method_name}",
                    extra={"status": status, "error_code": error_code, "body": body},
                )

            return {"status": status, "data": parsed, "elapsed_ms": round(elapsed, 1)}

        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            logger.error(
                f"API exception: {method_name}",
                extra={"error": str(e), "elapsed_ms": round(elapsed, 1)},
            )
            raise

    @staticmethod
    def _parse_response(status: Optional[int], body: Optional[str]) -> Any:
        """Parse JSON response body, returning raw string on failure."""
        if body is None:
            return None
        try:
            return json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return body
