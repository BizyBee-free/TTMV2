#!/usr/bin/env python3
"""Compare DNSE responses for secdef path variants (SDK upstream vs legacy).

Requires valid .env with DNSE_API_KEY, DNSE_API_SECRET (same signing as openapi-sdk).

Usage::

    python scripts/verify_secdef_paths.py
    python scripts/verify_secdef_paths.py --symbol HPG
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.path.insert(0, str(_ROOT / "vendor" / "dnse"))

from dnse import DNSEClient  # noqa: E402

from src.config import get_settings  # noqa: E402


def _preview(body: str, limit: int = 600) -> str:
    s = (body or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


def main() -> int:
    p = argparse.ArgumentParser(description="Verify secdef URL path against DNSE API")
    p.add_argument("--symbol", default="HPG", help="Symbol to query (default HPG)")
    args = p.parse_args()
    sym = args.symbol.strip()

    try:
        settings = get_settings()
    except Exception as e:
        print("ERROR: cannot load settings (.env with DNSE_*).", e, file=sys.stderr)
        return 2

    client = DNSEClient(
        api_key=settings.DNSE_API_KEY,
        api_secret=settings.DNSE_API_SECRET,
        base_url=settings.DNSE_BASE_URL,
    )

    paths = {
        "upstream_sdk": f"/price/{sym}/secdef",
        "legacy": f"/price/secdef/{sym}",
    }

    print(f"Base URL: {settings.DNSE_BASE_URL}")
    print(f"Symbol:   {sym}\n")

    results = {}
    for name, path in paths.items():
        status, body = client._request("GET", path, query=None, dry_run=False)
        results[name] = (status, body)
        print(f"=== {name} ===")
        print(f"Path:   GET {path}")
        print(f"Status: {status}")
        try:
            data = json.loads(body) if body else None
            pretty = json.dumps(data, ensure_ascii=False, indent=2) if data is not None else "(empty)"
            if len(pretty) > 800:
                pretty = pretty[:800] + "\n..."
            print("Body (JSON preview):\n", pretty)
        except json.JSONDecodeError:
            print("Body (raw preview):\n", _preview(body))
        print()

    s_up, _ = results["upstream_sdk"]
    s_leg, _ = results["legacy"]
    if s_up == 200 and s_leg != 200:
        print("CONCLUSION: Only /price/{symbol}/secdef returns HTTP 200 (use this).")
    elif s_leg == 200 and s_up != 200:
        print("CONCLUSION: Only /price/secdef/{symbol} returns HTTP 200 (legacy still needed).")
    elif s_up == 200 and s_leg == 200:
        print("CONCLUSION: Both return 200 — compare payloads; prefer upstream_sdk path.")
    else:
        print("CONCLUSION: Neither 200 — check credentials, symbol, or API availability.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
