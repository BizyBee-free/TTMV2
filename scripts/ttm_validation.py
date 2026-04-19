#!/usr/bin/env python3
"""CLI for TTM V2 log validation (hypotheses: breakout, positioning, scoring, adaptive).

Example:
  python scripts/ttm_validation.py \\
    --decisions reports/ttm_parallel_backtest_VN30F1M_20260301_20260331_xxx_decisions.jsonl \\
    --trades reports/ttm_parallel_backtest_VN30F1M_20260301_20260331_xxx_trades.jsonl \\
    --closes path/to/closes.json

Exit 0 if overall_valid, else 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.strategies.ttm.ttm_validation import main

if __name__ == "__main__":
    raise SystemExit(main())
