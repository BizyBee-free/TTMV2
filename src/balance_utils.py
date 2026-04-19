"""Parse DNSE balance payloads and validate minimum fields."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from src.logger import get_logger

logger = get_logger("balance_utils")

# Common DNSE / broker JSON keys for NAV or cash (structure varies by account type)
_BALANCE_NUMERIC_KEYS = (
    "netAssetValue",
    "totalAsset",
    "totalBalance",
    "cashBalance",
    "availableCash",
    "balance",
    "withdrawable",
    "pp0",  # purchasing power variants
    "netCash",
)


def _to_float(val: Any) -> Optional[float]:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace(",", ""))
        except ValueError:
            return None
    return None


def extract_derivative_remain_secure(payload: Any) -> Optional[float]:
    """Prefer derivative collateral for futures trading (DNSE: remainSecure)."""
    if not isinstance(payload, dict):
        return None
    der = payload.get("derivative")
    if isinstance(der, dict):
        v = _to_float(der.get("remainSecure"))
        if v is not None:
            return v
    data = payload.get("data")
    if isinstance(data, dict):
        der = data.get("derivative")
        if isinstance(der, dict):
            v = _to_float(der.get("remainSecure"))
            if v is not None:
                return v
    return None


def extract_balance_vnd(payload: Any) -> Optional[float]:
    """Best-effort extract a single scalar balance (VND) from nested API JSON.

    Returns None if no plausible numeric field is found.
    """
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, str):
        return _to_float(payload)
    if isinstance(payload, dict):
        # Futures account collateral is the primary source for derivative live trading.
        der_secure = extract_derivative_remain_secure(payload)
        if der_secure is not None:
            return der_secure
        for key in _BALANCE_NUMERIC_KEYS:
            if key in payload:
                got = _to_float(payload[key])
                if got is not None:
                    return got
        # Nested "data" / "balances" lists
        if "data" in payload:
            got = extract_balance_vnd(payload["data"])
            if got is not None:
                return got
        if "balances" in payload and isinstance(payload["balances"], list):
            total = 0.0
            found = False
            for item in payload["balances"]:
                if isinstance(item, dict):
                    for key in _BALANCE_NUMERIC_KEYS:
                        if key in item and isinstance(item[key], (int, float)):
                            total += float(item[key])
                            found = True
                            break
            if found:
                return total
        for v in payload.values():
            if isinstance(v, (dict, list)):
                got = extract_balance_vnd(v)
                if got is not None:
                    return got
    if isinstance(payload, list):
        for item in payload:
            got = extract_balance_vnd(item)
            if got is not None:
                return got
    return None


def validate_balance_response(
    api_result: Dict[str, Any],
) -> Tuple[bool, Optional[float], str]:
    """Validate get_balances-style unified client result.

    Args:
        api_result: ``{"status": int, "data": parsed body, "elapsed_ms": float}``

    Returns:
        (ok, balance_or_none, reason)
    """
    status = api_result.get("status")
    if status == 408:
        return False, None, "CLIENT_TIMEOUT"
    if status is None or (isinstance(status, int) and status >= 400):
        return False, None, f"HTTP_{status}"
    data = api_result.get("data")
    der_secure = extract_derivative_remain_secure(data)
    if der_secure is not None:
        return True, der_secure, "OK_DERIV_REMAIN_SECURE"
    bal = extract_balance_vnd(data)
    if bal is None:
        return False, None, "INCOMPLETE_SCHEMA_NO_BALANCE"
    return True, bal, "OK"
