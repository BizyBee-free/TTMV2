"""Shared utility functions and decorators."""

import functools
import time
from typing import Callable, TypeVar, Any

from src.logger import get_logger

logger = get_logger("utils")

F = TypeVar("F", bound=Callable[..., Any])


def retry(max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """Decorator for exponential backoff retry on exceptions.

    Usage:
        @retry(max_attempts=3)
        def fragile_call():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt == max_attempts:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}",
                            extra={"error": str(e)},
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}, "
                        f"retrying in {delay:.1f}s",
                        extra={"error": str(e)},
                    )
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def format_price(price: float, decimals: int = 0) -> str:
    """Format price for display (Vietnamese stock prices are integers)."""
    if decimals == 0:
        return f"{int(price):,}"
    return f"{price:,.{decimals}f}"


def validate_lot_size(quantity: int, market_type: str = "STOCK") -> bool:
    """Validate order quantity per DNSE rules.

    STOCK: Must be round lot (100, 200, ...) or odd lot (1-99).
           Mixed lots (101, 102, ...) are invalid.
    DERIVATIVE: Any positive integer.
    """
    if quantity <= 0:
        return False
    if market_type == "DERIVATIVE":
        return True
    if quantity < 100:
        return True  # odd lot
    return quantity % 100 == 0  # round lot


def is_terminal_status(status: str) -> bool:
    """Check if an order status is terminal (no more changes expected)."""
    return status in ("Filled", "Canceled", "Rejected", "Expired", "DoneForDay")
