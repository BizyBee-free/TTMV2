#!/usr/bin/env python3
"""Isolated DNSE WebSocket test: connect → auth → subscribe → log pipeline stages.

Chạy độc lập để phân biệt lỗi mạng/subscribe vs tích hợp app. Mặc định bật
``DNSE_WS_PIPELINE_DEBUG`` (RAW_WS, SENDING SUB, DECODED, SUB OK, MARKET DATA, …).

Usage:
    python scripts/ws_test.py --symbol HPG --minutes 2
    python scripts/ws_test.py --symbol VN30F1M --board G1 --minutes 1
    python scripts/ws_test.py --symbol HPG --no-pipeline-print   # chỉ tóm tắt cuối phiên
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DNSE WS raw / pipeline probe")
    p.add_argument("--symbol", type=str, required=True, help="Mã subscribe (vd. HPG, VN30F1M)")
    p.add_argument("--board", type=str, default="G1", help="Board tick.*.json (G1, AL, …)")
    p.add_argument("--minutes", type=float, default=3.0, help="Thời gian lắng nghe (phút)")
    p.add_argument(
        "--no-pipeline-print",
        action="store_true",
        help="Không set DNSE_WS_PIPELINE_DEBUG (chỉ in summary cuối)",
    )
    p.add_argument(
        "--assert-market-sec",
        type=int,
        default=None,
        metavar="N",
        help="Sau N giây assert market_message_dispatches>0 (giống DNSE_WS_PIPELINE_ASSERT_MARKET_SEC)",
    )
    return p.parse_args()


async def _run(args: argparse.Namespace) -> None:
    if not args.no_pipeline_print:
        os.environ["DNSE_WS_PIPELINE_DEBUG"] = "true"
    if args.assert_market_sec is not None:
        os.environ["DNSE_WS_PIPELINE_ASSERT_MARKET_SEC"] = str(args.assert_market_sec)

    sys.path.insert(0, str(_ROOT))
    from src.config import get_settings
    from src.market_data import MarketDataManager

    settings = get_settings()
    sym = args.symbol.strip().upper()
    board = args.board.strip() or "G1"

    mdm = MarketDataManager(settings)
    print(
        f"[ws_test] connect encoding={settings.WS_ENCODING!r} url_tail=… "
        f"board={board!r} symbol={sym!r}",
        flush=True,
    )
    try:
        await mdm.connect()
        await mdm.subscribe_quotes([sym], board_id=board)
        await mdm.subscribe_trades([sym], board_id=board)
        await mdm.subscribe_ohlc([sym], resolution="1", board_id=board)

        sec = max(0.0, float(args.minutes) * 60.0)
        print(f"[ws_test] listening {sec:.0f}s …", flush=True)
        await asyncio.sleep(sec)

        print(
            "[ws_test] summary "
            f"raw_recv={mdm.ws_raw_bytes_received} decode_err={mdm.ws_decode_errors} "
            f"decoded_ok={mdm.ws_raw_frames} ws_market_cb={mdm.ws_market_dispatches} "
            f"mdm_msg={mdm.msg_count}",
            flush=True,
        )
    finally:
        await mdm.disconnect()


def main() -> None:
    args = _parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
