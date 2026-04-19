#!/usr/bin/env python3
"""Kiểm tra kết nối DNSE: REST (giống HMM live / đặt lệnh) + WebSocket (giống paper_test).

**Không nhầm hai lớp:**

- **REST** — ``DNSE_BASE_URL`` (mặc định ``https://openapi.dnse.com.vn``): HMAC, tài khoản,
  số dư, lệnh, OTP → dùng :class:`src.dnse_client.BeeTradeClient` (vendor ``dnse`` HTTP),
  cùng luồng với ``scripts/hmm_live.py`` / đặt lệnh.
- **WebSocket** — ``DNSE_WS_URL``: luồng market data realtime, **host khác** với REST;
  bắt buộc path ``/v1/stream`` + ``?encoding=`` (xem ``src.market_data._normalize_dnse_ws_url``).
  Đây **không** phải “OpenAPI V2” REST; chỉ là endpoint stream của DNSE.

Usage::

    python scripts/check_connection.py
    python scripts/check_connection.py --skip-ws   # chỉ REST (nhanh)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Ensure project root is on the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import get_settings
from src.logger import setup_logging
from src.dnse_client import BeeTradeClient


async def _check_websocket_market_data() -> tuple[bool, str]:
    """Cùng stack với ``scripts/paper_test.py`` (:class:`src.market_data.MarketDataManager`)."""
    from src.data_buffer import DataBuffer
    from src.market_data import MarketDataManager

    settings = get_settings()
    mgr = MarketDataManager(settings=settings, buffer=DataBuffer())
    try:
        await mgr.connect()
        await mgr.disconnect()
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    p = argparse.ArgumentParser(description="DNSE REST + WebSocket connectivity check")
    p.add_argument(
        "--skip-ws",
        action="store_true",
        help="Chỉ kiểm tra REST (BeeTradeClient), bỏ qua WebSocket",
    )
    args = p.parse_args()

    settings = get_settings()
    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_dir=settings.LOG_DIR,
    )
    from src.market_data import _normalize_dnse_ws_url

    ws_resolved = _normalize_dnse_ws_url(settings.DNSE_WS_URL, settings.WS_ENCODING)

    print("=" * 60)
    print("  BeeTrade - DNSE Connection Check")
    print("=" * 60)
    print(f"  REST API:   {settings.DNSE_BASE_URL}  (HMM live / orders / OTP)")
    print(f"  WebSocket:  {ws_resolved}  (paper_test market stream; normalized)")
    print(f"  Account:    {settings.DNSE_ACCOUNT_NO}")
    print(f"  Paper Mode: {settings.PAPER_MODE}")
    print("=" * 60)

    client = BeeTradeClient(settings)

    # --- REST: giống hmm_live (BeeTradeClient) ---
    print("\n[1/4] REST — Fetching accounts...")
    result = client.get_accounts()
    if result["status"] != 200:
        print(f"  FAILED - Status {result['status']}")
        print(f"  Error: {result['data']}")
        return 1
    print(f"  OK - Status {result['status']} ({result['elapsed_ms']:.0f}ms)")
    if isinstance(result["data"], (list, dict)):
        print(f"  Data: {json.dumps(result['data'], indent=2, ensure_ascii=False)[:500]}")

    print(f"\n[2/4] REST — Balances ({settings.DNSE_ACCOUNT_NO})...")
    result = client.get_balances()
    if result["status"] != 200:
        print(f"  FAILED - Status {result['status']}")
        print(f"  Error: {result['data']}")
        return 1
    print(f"  OK - Status {result['status']} ({result['elapsed_ms']:.0f}ms)")

    print("\n[3/4] REST — Security definition HPG...")
    result = client.get_security_definition("HPG")
    if result["status"] != 200:
        print(f"  FAILED - Status {result['status']}")
        print(f"  Error: {result['data']}")
        return 1
    print(f"  OK - Status {result['status']} ({result['elapsed_ms']:.0f}ms)")

    if args.skip_ws:
        print("\n[4/4] WebSocket — skipped (--skip-ws)")
    else:
        print("\n[4/4] WebSocket — MarketDataManager.connect (same as paper_test)...")
        ok, detail = asyncio.run(_check_websocket_market_data())
        if not ok:
            print(f"  FAILED - {detail}")
            print(
                "  Hint: DNSE_WS_URL must include /v1/stream; encoding query is added automatically.\n"
                "  REST can work while WS fails if URL is wrong — check .env DNSE_WS_URL."
            )
            return 1
        print(f"  {detail}")

    print("\n" + "=" * 60)
    print("  All checks passed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
