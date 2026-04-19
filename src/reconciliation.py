"""Post-fill reconciliation: balance delta and optional fee sanity check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from src.balance_utils import extract_balance_vnd, validate_balance_response
from src.config import Settings, get_settings
from src.logger import get_logger
from src.order_manager import ManagedOrder

logger = get_logger("reconciliation")

BalanceGetter = Callable[[], Dict[str, Any]]


@dataclass
class ReconcileResult:
    ok: bool
    reason: str
    balance_before: Optional[float] = None
    balance_after: Optional[float] = None


def expected_fee_vnd(
    order: ManagedOrder,
    fee_rate: float,
) -> float:
    """Rough notional fee = rate * (entry+exit approx using limit price)."""
    if fee_rate <= 0:
        return 0.0
    notional = abs(order.price * order.quantity)
    return notional * fee_rate


def reconcile_after_fill(
    order: ManagedOrder,
    balance_before: Optional[float],
    get_balances: BalanceGetter,
    settings: Optional[Settings] = None,
) -> ReconcileResult:
    """After terminal fill, refresh balance and compare to eps (optional).

    Fee check is informational when DERIVATIVE_FEE_RATE > 0.
    """
    settings = settings or get_settings()
    result = get_balances()
    ok_b, bal_after, reason = validate_balance_response(result)
    if not ok_b or bal_after is None:
        logger.error(
            "reconcile_after_fill: could not read balance",
            extra={"reason": reason, "order_id": order.order_id},
        )
        return ReconcileResult(ok=False, reason=f"balance_read_failed:{reason}")

    eps = settings.RECONCILE_BALANCE_EPS_VND
    if balance_before is not None:
        delta = bal_after - balance_before
        logger.info(
            "reconcile_after_fill balance snapshot",
            extra={
                "order_id": order.order_id,
                "balance_before": balance_before,
                "balance_after": bal_after,
                "delta": delta,
            },
        )
        # We do not predict exact delta without fills ledger; only log.

    fee_exp = expected_fee_vnd(order, settings.DERIVATIVE_FEE_RATE)
    if fee_exp > 0 and order.api_response:
        logger.debug(
            "expected_fee_vnd",
            extra={"order_id": order.order_id, "expected_fee": fee_exp},
        )

    return ReconcileResult(
        ok=True,
        reason="OK",
        balance_before=balance_before,
        balance_after=bal_after,
    )


def extract_fee_from_order_api(order: ManagedOrder) -> Optional[float]:
    """Best-effort fee from stored API response (structure varies)."""
    if not order.api_response:
        return None
    data = order.api_response.get("data")
    if isinstance(data, dict):
        for k in ("fee", "commission", "transactionFee"):
            if k in data and isinstance(data[k], (int, float)):
                return float(data[k])
    return None
