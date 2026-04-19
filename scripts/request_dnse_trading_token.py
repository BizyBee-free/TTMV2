#!/usr/bin/env python3
"""Gửi OTP qua email DNSE và (tuỳ chọn) đổi mã OTP lấy trading token cho `.env`.

Bước 1 - DNSE gửi mã OTP tới email đã đăng ký tài khoản DNSE.
Bước 2 - Nhập mã OTP trong terminal; script in dong DNSE_TRADING_TOKEN=... de dan vao .env.

Cần: `DNSE_API_KEY`, `DNSE_API_SECRET` trong `.env` (cùng cách ký như OpenAPI).

Usage::

    # Chỉ yêu cầu gửi OTP (kiểm tra email)
    python scripts/request_dnse_trading_token.py --request-only

    # Gửi OTP + nhập mã ngay để lấy token (mặc định)
    python scripts/request_dnse_trading_token.py

    # Đã có OTP rồi, không gửi lại email
    python scripts/request_dnse_trading_token.py --otp 123456
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import get_settings
from src.dnse_client import BeeTradeClient
from src.logger import setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="DNSE: OTP email -> DNSE_TRADING_TOKEN")
    p.add_argument(
        "--request-only",
        action="store_true",
        help="Only call send-email-otp (no token exchange)",
    )
    p.add_argument(
        "--otp",
        default=None,
        help="OTP code from email (skips send-email-otp if set)",
    )
    p.add_argument(
        "--no-send",
        action="store_true",
        help="Do not call send-email-otp; use with --otp",
    )
    args = p.parse_args()

    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

    client = BeeTradeClient(settings)
    mgr = client.token_manager

    if not args.no_send and not args.otp:
        status, body = mgr.request_otp()
        print(f"[send-email-otp] HTTP {status} body={_short_body(body)}", flush=True)
        if status != 200:
            print("ERROR: send-email-otp failed. Check DNSE_API_KEY / DNSE_API_SECRET.", file=sys.stderr)
            return 1
        print("OK: OTP email requested. Check your DNSE-registered inbox.", flush=True)
        if args.request_only:
            return 0

    if args.request_only and args.otp:
        print("Hint: remove --request-only if you want to exchange --otp for a token.", file=sys.stderr)

    if args.request_only and not args.otp:
        return 0

    otp = (args.otp or "").strip()
    if not otp:
        try:
            otp = input("Enter OTP from DNSE email: ").strip()
        except EOFError:
            print("No OTP on stdin. Use: --otp CODE", file=sys.stderr)
            return 1
    if not otp:
        print("OTP cannot be empty.", file=sys.stderr)
        return 1

    try:
        token = mgr.activate_token(otp)
    except RuntimeError as e:
        print(f"Token exchange failed: {e}", file=sys.stderr)
        return 1

    if not token or not isinstance(token, str):
        print("Invalid token in API response.", file=sys.stderr)
        return 1

    print("\n--- Paste into project root .env (or your env file) ---\n", flush=True)
    print(f"DNSE_TRADING_TOKEN={token}", flush=True)
    print("\nToken is usually valid ~1 hour; re-run this script when it expires.\n", flush=True)
    return 0


def _short_body(body: object) -> str:
    if body is None:
        return ""
    s = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)[:500]
    return s[:500]


if __name__ == "__main__":
    raise SystemExit(main())
