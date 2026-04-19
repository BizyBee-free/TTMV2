#!/usr/bin/env python3
"""Phase 2 verification: measure WebSocket latency and throughput.

Connects to DNSE WebSocket, subscribes to quotes/trades for a set of
symbols, and collects timing statistics for a configurable duration.

Usage:
    python scripts/latency_test.py
    python scripts/latency_test.py --symbols HPG VNM FPT --duration 60
"""

import argparse
import asyncio
import os
import statistics
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import get_settings
from src.logger import setup_logging, get_logger
from src.market_data import MarketDataManager

logger = get_logger("latency_test")

# Stores receive timestamps for latency analysis
_receive_times: list[float] = []
_first_msg_ts: float = 0.0
_symbol_counts: dict[str, int] = {}


def on_quote_received(quote) -> None:
    global _first_msg_ts
    now = time.time()
    _receive_times.append(now)
    if _first_msg_ts == 0.0:
        _first_msg_ts = now
    _symbol_counts[quote.symbol] = _symbol_counts.get(quote.symbol, 0) + 1


def on_trade_received(trade) -> None:
    global _first_msg_ts
    now = time.time()
    _receive_times.append(now)
    if _first_msg_ts == 0.0:
        _first_msg_ts = now
    _symbol_counts[trade.symbol] = _symbol_counts.get(trade.symbol, 0) + 1


async def run_latency_test(
    symbols: list[str],
    duration: int,
    subscribe_trades: bool = True,
) -> None:
    settings = get_settings()
    mgr = MarketDataManager(settings)

    print("=" * 60)
    print("  BeeTrade - WebSocket Latency Test")
    print("=" * 60)
    print(f"  WS URL:     {settings.DNSE_WS_URL}")
    print(f"  Encoding:   {settings.WS_ENCODING}")
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Duration:   {duration}s")
    print("=" * 60)

    # Register external callbacks for timing
    mgr.on_quote(on_quote_received)
    if subscribe_trades:
        mgr.on_trade(on_trade_received)

    print("\n[1/4] Connecting to WebSocket...")
    t0 = time.time()
    try:
        await mgr.connect()
    except Exception as e:
        print(f"  FAILED: {e}")
        return
    connect_ms = (time.time() - t0) * 1000
    print(f"  Connected in {connect_ms:.0f}ms")

    print("\n[2/4] Subscribing to channels...")
    await mgr.subscribe_quotes(symbols)
    if subscribe_trades:
        await mgr.subscribe_trades(symbols)
    print(f"  Subscribed: quotes={len(symbols)}", end="")
    if subscribe_trades:
        print(f", trades={len(symbols)}", end="")
    print()

    print(f"\n[3/4] Collecting data for {duration}s...")
    print(f"  (Market hours: 09:00-11:30, 13:00-14:45 VN time)")
    print()

    start = time.time()
    last_report = start
    report_interval = 10

    try:
        while (time.time() - start) < duration:
            await asyncio.sleep(1)
            elapsed = time.time() - start

            if time.time() - last_report >= report_interval:
                total = len(_receive_times)
                rate = total / elapsed if elapsed > 0 else 0
                print(f"  [{elapsed:5.0f}s] Messages: {total:>6}  Rate: {rate:.1f} msg/s")
                last_report = time.time()

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    print(f"\n[4/4] Results")
    print("-" * 60)

    total_msgs = len(_receive_times)
    total_time = time.time() - start

    print(f"  Total messages received:   {total_msgs}")
    print(f"  Collection time:           {total_time:.1f}s")
    print(f"  Connect latency:           {connect_ms:.0f}ms")

    if total_msgs > 0:
        throughput = total_msgs / total_time
        print(f"  Throughput:                {throughput:.1f} msg/s")

        # Inter-message intervals (proxy for delivery latency)
        if total_msgs > 1:
            intervals = [
                (_receive_times[i] - _receive_times[i - 1]) * 1000
                for i in range(1, len(_receive_times))
            ]
            print(f"\n  Inter-message interval (ms):")
            print(f"    Min:    {min(intervals):.2f}")
            print(f"    Max:    {max(intervals):.2f}")
            print(f"    Mean:   {statistics.mean(intervals):.2f}")
            print(f"    Median: {statistics.median(intervals):.2f}")
            if len(intervals) > 1:
                print(f"    Stdev:  {statistics.stdev(intervals):.2f}")
            p95 = sorted(intervals)[int(len(intervals) * 0.95)]
            p99 = sorted(intervals)[min(int(len(intervals) * 0.99), len(intervals) - 1)]
            print(f"    P95:    {p95:.2f}")
            print(f"    P99:    {p99:.2f}")

        print(f"\n  Messages per symbol:")
        for sym in sorted(_symbol_counts.keys()):
            print(f"    {sym:<10} {_symbol_counts[sym]:>6}")
    else:
        print("  No messages received.")
        print("  Tip: Run during market hours (09:00-14:45 VN time)")

    # Buffer stats
    stats = mgr.buffer.stats()
    print(f"\n  Buffer state:")
    print(f"    Quotes tracked:  {stats.total_quotes}")
    print(f"    Trades buffered: {stats.total_trades}")
    print(f"    OHLC buffered:   {stats.total_ohlc}")

    print("-" * 60)

    await mgr.disconnect()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="BeeTrade WebSocket Latency Test")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["HPG", "VNM", "FPT", "VCB", "MBB"],
        help="Symbols to subscribe (default: HPG VNM FPT VCB MBB)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--no-trades",
        action="store_true",
        help="Skip trade subscriptions (quotes only)",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)

    asyncio.run(
        run_latency_test(
            symbols=args.symbols,
            duration=args.duration,
            subscribe_trades=not args.no_trades,
        )
    )


if __name__ == "__main__":
    main()
