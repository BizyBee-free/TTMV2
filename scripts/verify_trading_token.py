#!/usr/bin/env python3
"""Verify DNSE_TRADING_TOKEN: API accepts the token (no real order).

Probe 1: DELETE cancel fake order id (some DNSE builds return 500 here — inconclusive).
Probe 2: POST order with invalid payload (qty=0) — expect 4xx OA-* if token OK, 401 if bad.

Usage::
    python scripts/verify_trading_token.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.dnse_client import BeeTradeClient
from src.logger import setup_logging


def _is_auth_failure(status: int, body: str) -> bool:
    if status in (401, 403):
        return True
    if not body:
        return False
    low = body.lower()
    return "unauthorized" in low or "invalid token" in low or "trading-token" in low and "invalid" in low


def _probe_post_rejected(client: BeeTradeClient, tok: str, account_no: str) -> tuple[int, str]:
    """POST derivative order that should be rejected by validation (no fill)."""
    payload = {
        "accountNo": account_no,
        "symbol": "__BEETRADE_TOKEN_VERIFY__",
        "side": "NB",
        "orderType": "LO",
        "price": 1.0,
        "quantity": 0,
        "loanPackageId": 0,
    }
    return client.sdk.post_order("DERIVATIVE", payload, tok, dry_run=False)


def _mask(tok: str) -> str:
    if not tok or len(tok) < 8:
        return "(too short or empty)"
    return f"{tok[:4]}...{tok[-4:]} (len={len(tok)})"


def main() -> int:
    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

    raw = (getattr(settings, "DNSE_TRADING_TOKEN", None) or "").strip()
    if not raw:
        path = (getattr(settings, "DNSE_TRADING_TOKEN_FILE", None) or "").strip()
        if path:
            p = Path(path)
            if p.is_file():
                try:
                    raw = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
                except OSError:
                    pass

    print("DNSE trading token check")
    print(f"  PAPER_MODE={settings.PAPER_MODE}")

    if not raw:
        print("\nFAIL: DNSE_TRADING_TOKEN is empty (and file if configured).")
        print("Set DNSE_TRADING_TOKEN or run: python scripts/request_dnse_trading_token.py")
        return 1

    # Common mistake: 6-digit OTP is not the trading token
    if re.fullmatch(r"\d{6}", raw.strip()):
        print("  Token preview: ****** (6 digits = OTP from email, NOT trading token)")
        print(
            "\nFAIL: Put the long string from `request_dnse_trading_token.py` in DNSE_TRADING_TOKEN,\n"
            "      not the email OTP code."
        )
        return 1

    print(f"  Token from env/file: {_mask(raw)}")

    client = BeeTradeClient(settings)
    tok = client.token_manager.token
    if not tok:
        print("\nFAIL: Token manager did not load token (internal).")
        return 1

    acc = settings.DNSE_ACCOUNT_NO

    # Probe A: cancel non-existent order (DNSE may return 500 REMOTE_SERVER_ERROR — not proof of bad token)
    fake_id = "bee-verify-no-such-order"
    st_a, body_a = client.sdk.cancel_order(acc, fake_id, "DERIVATIVE", tok, dry_run=False)
    body_a = body_a or ""
    print(f"\nProbe A: cancel_order(fake id) -> HTTP {st_a}")
    if len(body_a) <= 500:
        print(f"  body: {body_a}")

    if _is_auth_failure(st_a, body_a):
        print("\nFAIL: Trading token rejected (unauthorized). Run: python scripts/request_dnse_trading_token.py")
        return 1

    if st_a in (404, 400, 422):
        print("\nOK: Token accepted (cancel probe returned expected client/business error).")
        return 0

    try:
        data_a = json.loads(body_a) if body_a else {}
        if isinstance(data_a, dict):
            code_a = str(data_a.get("code") or data_a.get("errorCode") or "")
            if code_a.startswith("OA-") and st_a != 401:
                print("\nOK: Token accepted (OA-* on cancel probe).")
                return 0
    except json.JSONDecodeError:
        pass

    # Probe B: invalid POST — should NOT create a real order; distinguishes auth vs validation
    print("\nProbe B: post_order(invalid: qty=0, fake symbol) ...")
    st_b, body_b = _probe_post_rejected(client, tok, acc)
    body_b = body_b or ""
    print(f"  -> HTTP {st_b}")
    if len(body_b) <= 500:
        print(f"  body: {body_b}")

    if _is_auth_failure(st_b, body_b):
        print("\nFAIL: Trading token rejected on order POST.")
        return 1

    if st_b in (400, 404, 422, 409):
        print("\nOK: Token is accepted for placing orders (request rejected by validation - no fill).")
        return 0

    try:
        data_b = json.loads(body_b) if body_b else {}
        if isinstance(data_b, dict):
            code_b = str(data_b.get("code") or data_b.get("errorCode") or "")
            if code_b.startswith("OA-") and st_b not in (401, 403):
                print("\nOK: Token accepted (OA-* business error on invalid order — expected).")
                return 0
    except json.JSONDecodeError:
        pass

    if st_b == 200:
        print(
            "\nWARN: HTTP 200 on validation probe — unexpected. Check DNSE dashboard for stray orders "
            "with symbol __BEETRADE_TOKEN_VERIFY__."
        )
        return 0

    if st_a == 500 and st_b == 500:
        print(
            "\nNOTE: Both probes returned HTTP 500 (REMOTE_SERVER_ERROR). This is often a DNSE backend issue,\n"
            "      not proof that your token is wrong. If you do NOT see 401/403 above, token may still be valid.\n"
            "      Try: python scripts/check_connection.py  and a small paper test before live."
        )
        return 0

    print(f"\nUNCLEAR: cancel={st_a}, post={st_b}. Refresh token if orders fail in live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
