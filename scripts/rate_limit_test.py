#!/usr/bin/env python3
"""DNSE API rate limit discovery test.

Sends progressively faster requests to find the minimum interval
before DNSE returns OA-103 (Too many requests).

Uses GET /price/{symbol}/secdef (SDK get_security_definition) -- lightweight, read-only, safe.

Usage:
    python scripts/rate_limit_test.py
    python scripts/rate_limit_test.py --start-interval 500 --min-interval 20 --step 50 --burst 5
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.config import get_settings
from src.logger import setup_logging, get_logger
from src.dnse_client import BeeTradeClient

logger = get_logger("rate_limit_test")

RATE_LIMIT_CODES = {"OA-103", "OA-400", "TOO_MANY_REQUESTS"}
RATE_LIMIT_HTTP_STATUS = 429

ENDPOINT_MAP = {
    "secdef": lambda c, sym: c.get_security_definition(sym),
    "accounts": lambda c, sym: c.get_accounts(),
    "balances": lambda c, sym: c.get_balances(),
    "latest_trade": lambda c, sym: c.get_latest_trade(sym),
}


def send_burst(
    client: BeeTradeClient,
    symbol: str,
    count: int,
    interval_ms: float,
    endpoint: str = "secdef",
) -> List[dict]:
    """Send `count` requests spaced by `interval_ms` milliseconds.

    Returns list of {seq, status, error_code, elapsed_ms, gap_ms}.
    """
    call_fn = ENDPOINT_MAP.get(endpoint, ENDPOINT_MAP["secdef"])
    results = []
    prev_ts = None

    for i in range(count):
        if i > 0:
            time.sleep(interval_ms / 1000.0)

        t0 = time.time()
        result = call_fn(client, symbol)
        elapsed = (time.time() - t0) * 1000

        gap = 0.0
        if prev_ts is not None:
            gap = (t0 - prev_ts) * 1000
        prev_ts = t0

        error_code = ""
        if isinstance(result["data"], dict):
            error_code = result["data"].get("code", "")

        results.append({
            "seq": i + 1,
            "status": result["status"],
            "error_code": error_code,
            "elapsed_ms": round(elapsed, 1),
            "gap_ms": round(gap, 1),
        })

        status_icon = "OK" if result["status"] == 200 else f"ERR {result['status']}"
        if error_code:
            status_icon += f" [{error_code}]"
        print(f"    #{i+1:>3}  {status_icon:<25} {elapsed:6.0f}ms  gap={gap:6.0f}ms")

        if error_code in RATE_LIMIT_CODES or result["status"] == RATE_LIMIT_HTTP_STATUS:
            break

    return results


def run_test(
    start_interval: int,
    min_interval: int,
    step: int,
    burst: int,
    cooldown: float,
    symbol: str,
    endpoint: str = "secdef",
) -> None:
    settings = get_settings()
    client = BeeTradeClient(settings)

    print("=" * 70)
    print("  BeeTrade - DNSE API Rate Limit Discovery Test")
    print("=" * 70)
    print(f"  API URL:        {settings.DNSE_BASE_URL}")
    print(f"  Endpoint:       {endpoint} (symbol={symbol})")
    print(f"  Start interval: {start_interval}ms")
    print(f"  Min interval:   {min_interval}ms")
    print(f"  Step:           -{step}ms per round")
    print(f"  Burst size:     {burst} requests per round")
    print(f"  Cooldown:       {cooldown}s between rounds")
    print("=" * 70)

    # Warmup: single request to verify connectivity
    print("\n[Warmup] Verifying API connectivity...")
    warmup = client.get_security_definition(symbol)
    if warmup["status"] != 200:
        print(f"  FAILED: status={warmup['status']} data={warmup['data']}")
        print("  Aborting test. Check API credentials / connectivity.")
        return
    print(f"  OK ({warmup['elapsed_ms']:.0f}ms)")

    # Progressive test
    found_limit = False
    limit_interval = None
    all_rounds: List[dict] = []
    interval = start_interval

    print(f"\n{'='*70}")
    print(f"  Starting rate limit scan: {start_interval}ms -> {min_interval}ms")
    print(f"{'='*70}")

    try:
        while interval >= min_interval:
            print(f"\n--- Round: interval = {interval}ms, burst = {burst} ---")

            results = send_burst(client, symbol, burst, interval, endpoint=endpoint)

            hit_limit = any(
                r["error_code"] in RATE_LIMIT_CODES or r["status"] == RATE_LIMIT_HTTP_STATUS
                for r in results
            )
            ok_count = sum(1 for r in results if r["status"] == 200)
            avg_elapsed = sum(r["elapsed_ms"] for r in results) / len(results)

            round_summary = {
                "interval_ms": interval,
                "burst": burst,
                "ok": ok_count,
                "total": len(results),
                "hit_limit": hit_limit,
                "avg_elapsed_ms": round(avg_elapsed, 1),
            }
            all_rounds.append(round_summary)

            if hit_limit:
                print(f"\n  >>> OA-103 DETECTED at interval = {interval}ms <<<")
                print(f"  >>> {ok_count}/{len(results)} requests succeeded before rate limit <<<")
                found_limit = True
                limit_interval = interval
                break

            print(f"  Summary: {ok_count}/{len(results)} OK, avg latency {avg_elapsed:.0f}ms")

            # Cooldown before next round
            if interval - step >= min_interval:
                print(f"  Cooling down {cooldown}s before next round...")
                time.sleep(cooldown)

            interval -= step

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")

    # Final report
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Interval':>10}  {'OK/Total':>10}  {'Avg Latency':>12}  {'Rate Limited':>13}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*13}")

    for r in all_rounds:
        flag = ">>> YES <<<" if r["hit_limit"] else "No"
        print(
            f"  {r['interval_ms']:>8}ms  "
            f"{r['ok']:>3}/{r['total']:<3}      "
            f"{r['avg_elapsed_ms']:>8.0f}ms    "
            f"{flag:>13}"
        )

    if found_limit:
        safe_interval = limit_interval + step
        print(f"\n  CONCLUSION:")
        print(f"    Rate limit (OA-103) hit at:  {limit_interval}ms between requests")
        print(f"    Last safe interval:          {safe_interval}ms")
        print(f"    Recommended MIN_REQUEST_INTERVAL: {safe_interval + step}ms (with safety buffer)")
        print(f"\n  This means DNSE allows roughly {1000/safe_interval:.1f} req/s sustained.")
    else:
        print(f"\n  CONCLUSION:")
        print(f"    No OA-103 detected down to {interval + step}ms interval.")
        print(f"    DNSE handles at least {1000/max(interval+step, 1):.1f} req/s for this endpoint.")
        print(f"    To test further, reduce --min-interval below {min_interval}ms.")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="DNSE API Rate Limit Discovery Test"
    )
    parser.add_argument(
        "--start-interval",
        type=int,
        default=500,
        help="Starting interval between requests in ms (default: 500)",
    )
    parser.add_argument(
        "--min-interval",
        type=int,
        default=20,
        help="Minimum interval to test in ms (default: 20)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=50,
        help="Reduce interval by this many ms each round (default: 50)",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=5,
        help="Number of requests per round (default: 5)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=3.0,
        help="Seconds to wait between rounds (default: 3)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="HPG",
        help="Symbol for secdef requests (default: HPG)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="secdef",
        choices=list(ENDPOINT_MAP.keys()),
        help="API endpoint to test (default: secdef)",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(log_level="WARNING", log_dir=settings.LOG_DIR)

    run_test(
        start_interval=args.start_interval,
        min_interval=args.min_interval,
        step=args.step,
        burst=args.burst,
        cooldown=args.cooldown,
        symbol=args.symbol,
        endpoint=args.endpoint,
    )


if __name__ == "__main__":
    main()
