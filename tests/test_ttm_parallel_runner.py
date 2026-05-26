"""Smoke tests for TTM parallel paper runner (JSONL + deterministic replay)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.backtest.data_fetcher import OhlcBar
from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.ttm_parallel_runner import (
    ParallelRunner,
    _V2_CLOSED_TRADE_FLAT_DEFAULTS,
    _v2_entry_confirm_row,
    _v2_entry_execution_long_confirm,
    replay_bars,
)
from src.strategies.ttm.ttm_v2_gates import GATE_MODE_QUALITY_RESEARCH, entry_confirm_mode_for_gate


def _bars(n: int = 80) -> list[OhlcBar]:
    bars: list[OhlcBar] = []
    p = 100.0
    t0 = 1700000000
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
                unix_ts=t0 + i * 900,
            )
        )
        p = c
    return bars


def test_ttm_config_includes_refactor1_max_hold_bars() -> None:
    assert "ttm_v2_max_hold_bars" in TTM_CONFIG
    assert TTM_CONFIG["ttm_v2_max_hold_bars"] is None


def test_parallel_runner_jsonl_and_summary(tmp_path: Path) -> None:
    base = _bars(80)
    data_series = []
    for i in range(len(base)):
        data_series.append(
            {
                "bars": base[: i + 1],
                "basis": np.zeros(i + 1),
                "open_interest": np.linspace(1e5, 1.01e5, i + 1),
            }
        )
    dec = tmp_path / "dec.jsonl"
    trd = tmp_path / "tr.jsonl"
    cfg = {**TTM_CONFIG}
    rep = replay_bars(data_series, config=cfg, decision_log_path=str(dec), trade_log_path=str(trd))

    assert rep["total_bars"] == len(data_series)
    assert "agreement_rate" in rep
    assert "v1" in rep and "v2" in rep
    assert rep.get("closes_n") == len(data_series)
    cj = tmp_path / "closes_dec.json"
    assert cj.is_file()
    closes = json.loads(cj.read_text(encoding="utf-8"))
    assert len(closes) == len(data_series)
    assert all(isinstance(x, (int, float)) for x in closes)

    lines = dec.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= len(data_series)
    for line in lines[:-1]:
        obj = json.loads(line)
        assert "event_type" in obj
        assert obj["event_type"] == "decision"
        assert "timestamp" in obj
        assert "bar_index" in obj
        assert "features" in obj
        assert "v1" in obj and "v2" in obj
        assert "breakout_strength" in obj["features"]
        assert "exhaustion_candidate" in obj["features"]
        assert "raw_strength" in obj["features"]
        assert "cap" in obj["features"]
        assert "exhaustion_confirm" in obj["features"]
        assert "short_score" in obj["features"]
        assert "extension" in obj["features"]
        assert "last_bar_return" in obj["features"]
        assert "effective_strength" in obj["features"]
        assert "signal" in obj["v1"] or "reason_block" in obj["v1"]
        assert "blocked_by" in obj["v2"]
        assert "score_components" in obj["v2"]
        assert "short_components" in obj["v2"]
        assert isinstance(obj["v2"]["score_components"], dict)
        assert isinstance(obj["v2"]["short_components"], dict)
        assert "crowd_phase" in obj["v2"]["log"]
        assert "short_phase" in obj["v2"]["log"]
        assert "position_state" in obj
        ps = obj["position_state"]
        assert ps["v1_is_open"] is False or ps["v1_side"] in ("LONG", "SHORT")
        assert ps["v2_is_open"] is False or ps["v2_side"] in ("LONG", "SHORT")
        assert "v1_position_size" in ps and "v2_position_size" in ps
        assert "forward_return" in obj and "forward_horizon_bars" in obj
        assert "vol_z" in obj and "regime_tag" in obj and "ttm_regime_id" in obj
        if not ps["v2_is_open"]:
            assert ps["v2_side"] is None


def test_v2_exit_trade_rows_include_entry_trace_fields(tmp_path: Path) -> None:
    """CLOSED v2 rows remain traceable to entry_bar_index (and calibration when LONG)."""
    base = _bars(80)
    data_series = [
        {"bars": base[: i + 1], "basis": np.zeros(i + 1), "open_interest": np.linspace(1e5, 1.01e5, i + 1)}
        for i in range(len(base))
    ]
    dec = tmp_path / "dec2.jsonl"
    trd = tmp_path / "tr2.jsonl"
    # Disable strict 20260511 gates for this structural replay (synthetic bars rarely pass phase+score).
    cfg = {
        **TTM_CONFIG,
        "ttm_v2_hard_long_gate_enabled": False,
        "ttm_v2_enable_entry_confirmation": False,
    }
    replay_bars(data_series, config=cfg, decision_log_path=str(dec), trade_log_path=str(trd))
    closed_v2: list[dict] = []
    for line in trd.read_text(encoding="utf-8").strip().splitlines():
        o = json.loads(line)
        if o.get("event_type") != "trade" or str(o.get("model", "")).lower() != "v2":
            continue
        if str(o.get("event", "")).upper() != "CLOSED":
            continue
        closed_v2.append(o)
    if not closed_v2:
        pytest.skip("no v2 CLOSED trades in this replay sample")
    for o in closed_v2:
        assert "entry_bar_index" in o
        assert isinstance(o["entry_bar_index"], int)
        assert o["entry_bar_index"] >= 0
        for k in _V2_CLOSED_TRADE_FLAT_DEFAULTS:
            assert k in o
        assert o["signal_bar_index"] == int(o["entry_bar_index"]) - 1
        assert o["holding_bars"] == o.get("holding_period")


def test_v2_entry_confirm_row_uses_signal_bar_not_execution() -> None:
    """Execution confirm must not read the closed N+1 feature row (lookahead)."""
    feats = {
        "n": 3,
        "exhaustion_confirm": [False, False, True],
        "last_bar_return": [0.0, -0.002, 0.01],
        "breakout_up": [False, False, False],
        "breakout_up_filtered_last": [False, False, False],
    }
    exec_row = {"exhaustion_confirm": True, "last_bar_return": 0.01}
    sig_row, src, bi = _v2_entry_confirm_row(feats, exec_row, signal_bar_index=1)
    assert src == "signal_bar"
    assert bi == 1
    assert sig_row.get("exhaustion_confirm") is False

    cfg = {
        **TTM_CONFIG,
        "ttm_v2_enable_entry_confirmation": True,
        "ttm_v2_gate_mode": GATE_MODE_QUALITY_RESEARCH,
    }
    blocked_exec, reason_exec, _ = _v2_entry_execution_long_confirm(
        feats,
        exec_row,
        cfg,
        signal_only_exec=False,
        signal_long_candidate=True,
        entry_confirm_mode=entry_confirm_mode_for_gate(GATE_MODE_QUALITY_RESEARCH),
        signal_bar_index=None,
    )
    blocked_sig, reason_sig, fields_sig = _v2_entry_execution_long_confirm(
        feats,
        exec_row,
        cfg,
        signal_only_exec=False,
        signal_long_candidate=True,
        entry_confirm_mode=entry_confirm_mode_for_gate(GATE_MODE_QUALITY_RESEARCH),
        signal_bar_index=1,
    )
    assert blocked_exec and "exhaustion" in reason_exec
    assert not blocked_sig
    assert fields_sig.get("confirm_data_source") == "signal_bar"


def test_parallel_runner_deterministic() -> None:
    base = _bars(60)
    ds = [
        {"bars": base[: i + 1], "basis": np.zeros(i + 1), "open_interest": np.linspace(1e5, 1.01e5, i + 1)}
        for i in range(len(base))
    ]
    r1 = replay_bars(ds, config={**TTM_CONFIG})
    r2 = replay_bars(ds, config={**TTM_CONFIG})
    assert r1["total_bars"] == r2["total_bars"]
    assert r1["agreement_rate"] == r2["agreement_rate"]
