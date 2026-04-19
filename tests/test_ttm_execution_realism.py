"""Unit tests for TTM V2 execution realism helpers (latency / slippage)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.backtest.data_fetcher import OhlcBar
from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_execution_realism import (
    ExecutionRealismConfig,
    adjust_fill_price,
    entry_due_unix,
    resolution_to_bar_seconds,
    signal_timestamp_unix,
    slippage_abs_at_bar,
)
from src.strategies.ttm.ttm_parallel_runner import replay_bars


def _bars(n: int = 40) -> list[OhlcBar]:
    bars: list[OhlcBar] = []
    p = 100.0
    t0 = 1_700_000_000
    step = 900
    for i in range(n):
        p += np.sin(i / 8.0) * 0.15
        o = p
        c = p + 0.05 * np.sin(i / 3.0)
        h = max(o, c) + 0.1
        l = min(o, c) - 0.1
        bars.append(
            OhlcBar(
                symbol="X",
                time=str(i),
                open=o,
                high=h,
                low=l,
                close=c,
                volume=1.0,
                unix_ts=t0 + i * step,
            )
        )
        p = c
    return bars


def test_resolution_to_bar_seconds() -> None:
    assert resolution_to_bar_seconds("15") == 900
    assert resolution_to_bar_seconds("1H") == 3600


def test_entry_due_unix_monotonic() -> None:
    bars = _bars(5)
    b = bars[2]
    bs = 900
    t0 = entry_due_unix(b, 0, bs)
    t1 = entry_due_unix(b, 500, bs)
    assert t1 > t0
    assert signal_timestamp_unix(b, bs) == int(b.unix_ts) + bs


def test_slippage_modes() -> None:
    bars = _bars(30)
    s0 = slippage_abs_at_bar(bars, 10, "none")
    s1 = slippage_abs_at_bar(bars, 10, "base")
    s2 = slippage_abs_at_bar(bars, 10, "worst_case")
    assert s0 == 0.0
    assert s1 >= 0.0
    assert s2 >= s1


def test_adjust_fill_price_direction() -> None:
    slip = 0.5
    assert adjust_fill_price("LONG", is_entry=True, base_price=100.0, slippage_abs=slip) == 100.5
    assert adjust_fill_price("LONG", is_entry=False, base_price=100.0, slippage_abs=slip) == 99.5
    assert adjust_fill_price("SHORT", is_entry=True, base_price=100.0, slippage_abs=slip) == 99.5
    assert adjust_fill_price("SHORT", is_entry=False, base_price=100.0, slippage_abs=slip) == 100.5


def test_replay_with_execution_realism_smoke(tmp_path: Path) -> None:
    base = _bars(60)
    ds = [
        {"bars": base[: i + 1], "basis": np.zeros(i + 1), "open_interest": np.linspace(1e5, 1.01e5, i + 1)}
        for i in range(len(base))
    ]
    ts = [str(base[i].unix_ts) for i in range(len(base))]
    trd = tmp_path / "t.jsonl"
    er = ExecutionRealismConfig(latency_ms=100, slippage_mode="base", bar_seconds=900)
    cfg = {**TTM_CONFIG, "ttm_execution_bar_seconds": 900}
    rep = replay_bars(
        ds,
        config=cfg,
        timestamps=ts,
        trade_log_path=str(trd),
        execution_realism=er,
        execution_bar_seconds=900,
    )
    assert rep["total_bars"] == len(ds)
    assert "execution_realism" in rep
    if trd.is_file():
        lines = [json.loads(x) for x in trd.read_text(encoding="utf-8").splitlines() if x.strip()]
        closed_v2 = [x for x in lines if x.get("model") == "v2" and x.get("event") == "CLOSED"]
        for row in closed_v2:
            if "execution_audit" in row:
                assert "signal_exit_price" in row["execution_audit"] or row["execution_audit"].get(
                    "signal_price"
                ) is not None
