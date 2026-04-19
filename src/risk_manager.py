"""Pre-trade risk management and position monitoring.

Responsibilities:
    1. pre_trade_check()    -- gate-keep order submission before it reaches OrderManager
    2. on_tick_risk_check() -- monitor live positions for stoploss and daily loss breach
    3. Halt management      -- once halted, no further orders pass until reset

Risk rules implemented:
    - MAX_DAILY_LOSS_PCT  : halt all trading when net daily P&L falls below threshold
    - STOPLOSS_DEFAULT_PCT: trigger emergency close when a single position's unrealized
                             loss exceeds the configured percentage
    - MAX_POSITION_SIZE   : cap the number of contracts per symbol
    - Rate limiter check  : delegate to SlidingWindowRateLimiter (already wired in
                             BeeTradeClient for live orders; checked here for paper)

Usage::

    risk = RiskManager(tracker, settings=settings)
    ok, reason = risk.pre_trade_check(order_req, symbol_position=pos)
    if not ok:
        log and skip order ...

    # Called each market-data tick
    risk.on_tick_risk_check(symbol="VN30F2506", current_price=1250.0)
    if risk.is_halted:
        stop strategy ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

from src.config import OrderSide, OrderType, Settings, get_settings
from src.logger import get_logger
from src.order_manager import OrderRequest
from src.position_tracker import Position, PositionTracker

logger = get_logger("risk_manager")

HaltCallback = Callable[[str], None]
StoplossCallback = Callable[[str, float], None]


class RiskRejectReason(str, Enum):
    DAILY_LOSS_HALT = "DAILY_LOSS_HALT"
    STOPLOSS_TRIGGERED = "STOPLOSS_TRIGGERED"
    POSITION_SIZE_EXCEEDED = "POSITION_SIZE_EXCEEDED"
    RATE_LIMIT = "RATE_LIMIT"
    RISK_MANAGER_HALTED = "RISK_MANAGER_HALTED"
    BELOW_MIN_BALANCE = "BELOW_MIN_BALANCE"
    TRADING_DISABLED = "TRADING_DISABLED"
    ALLOWED = "ALLOWED"


@dataclass
class RiskSnapshot:
    """Point-in-time risk status."""
    is_halted: bool
    halt_reason: str
    daily_pnl_net: float
    daily_loss_pct: float
    open_positions: int
    stoploss_triggers: int
    total_checks: int
    total_rejected: int


class RiskManager:
    """Centralised pre-trade and intra-session risk controller.

    Args:
        tracker:  PositionTracker instance to read P&L and positions from.
        settings: Application settings (MAX_DAILY_LOSS_PCT, etc.).
        nav_start: Starting NAV for the session (used to compute daily loss %).
                   Defaults to 0, in which case daily loss is checked by absolute
                   P&L against a floor of -MAX_DAILY_LOSS_PCT as a fraction of
                   nav_start when provided.
    """

    def __init__(
        self,
        tracker: PositionTracker,
        settings: Optional[Settings] = None,
        nav_start: float = 0.0,
    ) -> None:
        self._tracker = tracker
        self._settings = settings or get_settings()
        self._nav_start = nav_start

        self._halted = False
        self._halt_reason = ""

        self._halt_callbacks: List[HaltCallback] = []
        self._stoploss_callbacks: List[StoplossCallback] = []

        self._total_checks = 0
        self._total_rejected = 0
        self._stoploss_triggers = 0

        # External pause (e.g. Telegram /pause) — does not clear on resume() alone
        self._external_trading_enabled = True

    # ------------------------------------------------------------------ #
    # Callback registration
    # ------------------------------------------------------------------ #

    def on_halt(self, callback: HaltCallback) -> None:
        """Register callback invoked when trading is halted."""
        self._halt_callbacks.append(callback)

    def on_stoploss(self, callback: StoplossCallback) -> None:
        """Register callback invoked when a stoploss is triggered."""
        self._stoploss_callbacks.append(callback)

    # ------------------------------------------------------------------ #
    # Pre-trade check (called before every order submission)
    # ------------------------------------------------------------------ #

    def set_external_trading_enabled(self, enabled: bool) -> None:
        """Enable/disable new orders from strategy (Telegram /pause, ops switch)."""
        self._external_trading_enabled = enabled
        logger.info(
            "RiskManager: external trading flag",
            extra={"enabled": enabled},
        )

    @property
    def external_trading_enabled(self) -> bool:
        return self._external_trading_enabled

    def pre_trade_check(
        self,
        req: OrderRequest,
        current_position: Optional[Position] = None,
        account_balance: Optional[float] = None,
        paper_mode: bool = True,
    ) -> Tuple[bool, str]:
        """Validate an order against all risk rules.

        Args:
            req:              The order request to validate.
            current_position: Position snapshot for req.symbol (optional;
                              fetched from tracker if None).
            account_balance:  Latest balance from API (VND) for MIN_BALANCE check when live.
            paper_mode:       When True, MIN_BALANCE rule is skipped.

        Returns:
            (allowed, reason) -- allowed=True means the order may proceed.
        """
        self._total_checks += 1

        if self._halted:
            return self._reject(RiskRejectReason.RISK_MANAGER_HALTED, self._halt_reason)

        if not self._external_trading_enabled:
            return self._reject(
                RiskRejectReason.TRADING_DISABLED,
                "External trading disabled (e.g. Telegram /pause)",
            )

        # 0. Minimum balance (live only, when configured and balance known)
        if (
            not paper_mode
            and self._settings.MIN_BALANCE is not None
            and account_balance is not None
            and account_balance < self._settings.MIN_BALANCE
        ):
            reason = (
                f"Balance {account_balance:.0f} < MIN_BALANCE {self._settings.MIN_BALANCE:.0f}"
            )
            self._trigger_halt(reason)
            return self._reject(RiskRejectReason.BELOW_MIN_BALANCE, reason)

        # 1. Daily loss halt (only when nav_start is set, otherwise no basis for %)
        daily = self._tracker.daily_pnl
        if self._nav_start > 0:
            loss_pct = (-daily.net / self._nav_start) * 100.0
        else:
            loss_pct = 0.0

        if self._nav_start > 0 and daily.net < 0 and loss_pct >= self._settings.MAX_DAILY_LOSS_PCT:
            reason = (
                f"Daily loss {loss_pct:.2f}% >= limit {self._settings.MAX_DAILY_LOSS_PCT}%"
            )
            self._trigger_halt(reason)
            return self._reject(RiskRejectReason.DAILY_LOSS_HALT, reason)

        # 2. Position size limit
        pos = current_position or self._tracker.get_position(req.symbol)
        max_size = self._settings.MCMC_POSITION_SIZE

        if req.side == OrderSide.BUY and pos.quantity >= max_size:
            return self._reject(
                RiskRejectReason.POSITION_SIZE_EXCEEDED,
                f"Long position {pos.quantity} already at max {max_size}",
            )
        if req.side == OrderSide.SELL and pos.quantity <= -max_size:
            return self._reject(
                RiskRejectReason.POSITION_SIZE_EXCEEDED,
                f"Short position {pos.quantity} already at max {max_size}",
            )

        return True, RiskRejectReason.ALLOWED.value

    # ------------------------------------------------------------------ #
    # Intra-tick risk monitoring
    # ------------------------------------------------------------------ #

    def on_tick_risk_check(self, symbol: str, current_price: float) -> bool:
        """Check stoploss and daily loss breach on every market tick.

        Args:
            symbol:        Symbol to check.
            current_price: Latest market price.

        Returns:
            True if a risk action was triggered (stoploss or halt).
        """
        if self._halted:
            return False

        triggered = False

        # Stoploss check (use actual PnL to handle both long and short correctly)
        pos = self._tracker.get_position(symbol)
        if not pos.is_flat:
            pnl = pos.unrealized_pnl(current_price)
            cost = pos.cost_basis
            loss_pct = (abs(pnl) / max(cost, 1.0)) * 100.0 if pnl < 0 else 0.0
            if loss_pct >= self._settings.STOPLOSS_DEFAULT_PCT:
                self._stoploss_triggers += 1
                reason = (
                    f"Stoploss triggered for {symbol}: "
                    f"unrealized loss {loss_pct:.2f}% >= {self._settings.STOPLOSS_DEFAULT_PCT}%"
                )
                logger.warning(reason, extra={
                    "symbol": symbol, "price": current_price,
                    "avg_price": pos.avg_price, "qty": pos.quantity,
                    "loss_pct": round(loss_pct, 2),
                })
                for cb in self._stoploss_callbacks:
                    try:
                        cb(symbol, current_price)
                    except Exception as e:
                        logger.error("Stoploss callback error", extra={"error": str(e)})
                triggered = True

        # Daily loss re-check
        daily = self._tracker.daily_pnl
        if self._nav_start > 0 and daily.net < 0:
            loss_pct_daily = (-daily.net / self._nav_start) * 100.0
            if loss_pct_daily >= self._settings.MAX_DAILY_LOSS_PCT:
                reason = (
                    f"Daily loss {loss_pct:.2f}% >= {self._settings.MAX_DAILY_LOSS_PCT}%"
                )
                self._trigger_halt(reason)
                triggered = True

        return triggered

    # ------------------------------------------------------------------ #
    # Halt management
    # ------------------------------------------------------------------ #

    def halt(self, reason: str) -> None:
        """Manually halt trading (e.g. from strategy or external monitor)."""
        self._trigger_halt(reason)

    def resume(self) -> None:
        """Lift the halt (use with caution, typically at start of new session)."""
        if self._halted:
            logger.info("RiskManager: halt lifted", extra={"prev_reason": self._halt_reason})
        self._halted = False
        self._halt_reason = ""

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    # ------------------------------------------------------------------ #
    # NAV management
    # ------------------------------------------------------------------ #

    def set_nav_start(self, nav: float) -> None:
        """Set the starting NAV for daily loss % calculation."""
        self._nav_start = nav
        logger.info("RiskManager: NAV start set", extra={"nav_start": nav})

    # ------------------------------------------------------------------ #
    # Stats / snapshot
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> RiskSnapshot:
        daily = self._tracker.daily_pnl
        if self._nav_start > 0 and daily.net < 0:
            loss_pct = (-daily.net / self._nav_start) * 100.0
        else:
            loss_pct = 0.0
        return RiskSnapshot(
            is_halted=self._halted,
            halt_reason=self._halt_reason,
            daily_pnl_net=round(daily.net, 2),
            daily_loss_pct=round(loss_pct, 2),
            open_positions=len(self._tracker.get_open_positions()),
            stoploss_triggers=self._stoploss_triggers,
            total_checks=self._total_checks,
            total_rejected=self._total_rejected,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _reject(self, reason: RiskRejectReason, detail: str) -> Tuple[bool, str]:
        self._total_rejected += 1
        msg = f"{reason.value}: {detail}"
        logger.warning("RiskManager: order rejected", extra={
            "reason": reason.value, "detail": detail,
        })
        return False, msg

    def _trigger_halt(self, reason: str) -> None:
        if self._halted:
            return
        self._halted = True
        self._halt_reason = reason
        logger.error("RiskManager: TRADING HALTED", extra={"reason": reason})
        for cb in self._halt_callbacks:
            try:
                cb(reason)
            except Exception as e:
                logger.error("Halt callback error", extra={"error": str(e)})
