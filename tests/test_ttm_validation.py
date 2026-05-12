"""Tests for TTM JSONL validation (ttm_validation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.strategies.ttm.ttm_validation import load_decisions_jsonl, run_validation


def _write_jsonl(path: Path, rows: list) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def test_run_validation_empty_decisions(tmp_path: Path) -> None:
    dec = tmp_path / "d.jsonl"
    trd = tmp_path / "t.jsonl"
    dec.write_text("", encoding="utf-8")
    trd.write_text("", encoding="utf-8")
    r = run_validation(dec, trd)
    assert r["overall_valid"] is False
    assert r["breakout"]["passed"] is False


def test_synthetic_with_closes_engineered(tmp_path: Path) -> None:
    """Idealized series: breakouts align with larger h-step returns; scores correlate."""
    n = 100
    h = 4
    closes = np.array([100.0 + 0.08 * i for i in range(n)], dtype=np.float64)
    for i in range(0, n - h, 2):
        closes[i + h] += 4.0

    closes_path = tmp_path / "closes.json"
    closes_path.write_text(json.dumps(closes.tolist()), encoding="utf-8")

    decisions = []
    for i in range(n):
        brk = 1.0 if i % 2 == 0 else 0.0
        fail = 1.0 if (i % 11 == 0 and brk > 0 and closes[i + h] - closes[i] < 0) else 0.0
        oi = 1000.0 + (2 if i % 2 == 0 else -1) * i * 0.5
        sl = (i / max(n - 1, 1)) * 1.8 - 0.9
        decisions.append(
            {
                "event_type": "decision",
                "bar_index": i,
                "timestamp": str(1700000000 + i * 900),
                "features": {
                    "breakout_strength": brk,
                    "raw_strength": float(brk) if brk > 0 else None,
                    # Keep below next bar close so trap_reject_next rarely fires on this synthetic ramp.
                    "rolling_high": float(closes[i] - 50.0),
                    "failure_strength": fail,
                    "basis_norm": float(np.sin(i / 7.0)),
                    "oi_signal": 0.0,
                    "vol_regime": 0.0,
                    "is_breakout_up": bool(brk > 0),
                    "oi_pipeline": {"oi": 0.0, "oi_zscore": 0.0, "oi_signal": 0.0},
                    "oi_data": {"open_interest_last": oi},
                },
                "v1": {"conditions": {"breakout": brk > 0}},
                "v2": {
                    "score_long": sl,
                    "score_short": -sl,
                    "decision": "NONE",
                    "components": {},
                    "blocked_by": "threshold",
                    "log": {},
                },
                "position_state": {},
            }
        )

    # v2 trades: improving late (larger wins in second half); unified CLOSED rows
    trades = []
    for k in range(24):
        bi = 10 + k * 3
        side = "LONG"
        entry_p = float(100 + bi * 0.1)
        exit_p = entry_p + (1.5 if k >= 12 else 0.3) * (1 if k % 2 == 0 else -0.5)
        trades.append(
            {
                "event_type": "trade",
                "model": "v2",
                "event": "CLOSED",
                "bar_index": bi + 1,
                "exit_bar_index": bi + 1,
                "entry_bar_index": bi,
                "entry_price": entry_p,
                "exit_price": exit_p,
                "side": side,
                "position_size": 1,
                "pnl": float(exit_p - entry_p),
                "realized_return": float((exit_p - entry_p) / entry_p) if entry_p > 1e-12 else None,
                "holding_period": 2,
            }
        )

    dec = tmp_path / "dec.jsonl"
    trd = tmp_path / "tr.jsonl"
    _write_jsonl(dec, decisions)
    _write_jsonl(trd, trades)

    r = run_validation(
        dec,
        trd,
        closes_path=closes_path,
        forward_horizon=h,
        min_samples_breakout=12,
        min_samples_positioning=6,
        min_samples_scoring=18,
        min_trades_adaptive=8,
    )
    assert "breakout" in r and "scoring" in r
    assert r["breakout"]["metrics"]["n_breakout_bars"] >= 2
    assert "v2_trade_holding" in r["breakout"]["metrics"]


def test_load_decisions_sorts_by_bar_index(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    _write_jsonl(
        p,
        [
            {"event_type": "decision", "bar_index": 2, "features": {}},
            {"event_type": "decision", "bar_index": 0, "features": {}},
        ],
    )
    rows = load_decisions_jsonl(p)
    assert [r["bar_index"] for r in rows] == [0, 2]


def test_breakout_validation_skips_exhaustion_candidates(tmp_path: Path) -> None:
    n = 60
    closes = 100.0 + 0.01 * np.arange(n)
    breakout_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    strength_map = {
        0: 0.6,
        5: 0.7,
        10: 0.8,
        15: 1.2,
        20: 1.3,
        25: 1.4,
        30: 2.0,
        35: 2.1,
        40: 2.2,
        45: None,
    }
    jump_map = {
        0: 0.10,
        5: 0.11,
        10: 0.12,
        15: 0.20,
        20: 0.21,
        25: 0.22,
        30: 0.30,
        35: 0.31,
        40: 0.32,
        45: 0.05,
    }
    for bi, jump in jump_map.items():
        closes[bi + 4] += jump

    closes_path = tmp_path / "closes.json"
    closes_path.write_text(json.dumps(closes.tolist()), encoding="utf-8")

    decisions = []
    for i in range(n):
        strength = strength_map.get(i, 0.0)
        is_up = i in breakout_idx
        decisions.append(
            {
                "event_type": "decision",
                "bar_index": i,
                "timestamp": str(1700000000 + i * 300),
                "features": {
                    "breakout_strength": strength,
                    "raw_strength": float(strength) if strength is not None else 0.8,
                    "failure_strength": 0.0,
                    "basis_norm": 0.0,
                    "oi_signal": 0.0,
                    "vol_regime": 0.0,
                    "rolling_high": float(closes[i] - 0.01),
                    "is_breakout_up": is_up,
                    "exhaustion_candidate": i == 45,
                    "oi_pipeline": {"oi": 0.0, "oi_zscore": 0.0, "oi_signal": 0.0},
                    "oi_data": {"open_interest_last": 1000.0 + i},
                },
                "v1": {"conditions": {"breakout": is_up}},
                "v2": {
                    "score_long": 0.0,
                    "score_short": 0.0,
                    "decision": "NONE",
                    "components": {},
                    "blocked_by": "threshold",
                    "log": {},
                },
                "position_state": {},
            }
        )

    trades = []
    dec = tmp_path / "dec.jsonl"
    trd = tmp_path / "tr.jsonl"
    _write_jsonl(dec, decisions)
    _write_jsonl(trd, trades)

    r = run_validation(
        dec,
        trd,
        closes_path=closes_path,
        forward_horizon=4,
        min_samples_breakout=15,
        min_samples_positioning=4,
        min_samples_scoring=4,
        min_trades_adaptive=1,
    )
    assert r["breakout"]["passed"] is True
    assert r["breakout"]["metrics"]["trap_candidates"] == 10
    assert r["breakout"]["metrics"]["breakout_up_eval"]["n_is_breakout_up"] == 9
    assert r["breakout"]["metrics"]["breakout_up_eval"]["monotonic_low_mid_high"] is True


def test_scoring_short_prefers_feature_short_score(tmp_path: Path) -> None:
    n = 90
    h = 4
    closes = np.full(n, 100.0, dtype=np.float64)
    short_idx = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    score_map = {bi: 0.4 + 0.1 * k for k, bi in enumerate(short_idx)}
    jump_map = {bi: 0.05 + 0.03 * k for k, bi in enumerate(short_idx)}
    for bi, jump in jump_map.items():
        closes[bi + h] -= jump

    closes_path = tmp_path / "closes_short.json"
    closes_path.write_text(json.dumps(closes.tolist()), encoding="utf-8")

    decisions = []
    for i in range(n):
        short_score = score_map.get(i)
        decisions.append(
            {
                "event_type": "decision",
                "bar_index": i,
                "timestamp": str(1700000000 + i * 300),
                "features": {
                    "breakout_strength": 0.0,
                    "failure_strength": 0.0,
                    "basis_norm": 0.0,
                    "oi_signal": 0.0,
                    "vol_regime": 0.0,
                    "short_score": short_score,
                    "exhaustion_confirm": short_score is not None,
                    "raw_strength": (short_score + 1.0) if short_score is not None else None,
                    "cap": 1.0 if short_score is not None else None,
                    "oi_pipeline": {"oi": 0.0, "oi_zscore": 0.0, "oi_signal": 0.0},
                    "oi_data": {"open_interest_last": 1000.0 + i},
                },
                "v1": {"conditions": {"breakout": False}},
                "v2": {
                    "score_long": 0.0,
                    "score_short": 0.0,
                    "decision": "NONE",
                    "components": {},
                    "blocked_by": "threshold",
                    "log": {},
                },
                "position_state": {},
            }
        )

    dec = tmp_path / "dec_short.jsonl"
    trd = tmp_path / "tr_short.jsonl"
    _write_jsonl(dec, decisions)
    _write_jsonl(trd, [])

    r = run_validation(
        dec,
        trd,
        closes_path=closes_path,
        forward_horizon=h,
        min_samples_breakout=3,
        min_samples_positioning=3,
        min_samples_scoring=8,
        min_trades_adaptive=1,
        scoring_mode="all",
    )
    assert r["scoring_short"]["metrics"]["score_source"] == "features.short_score"
    assert r["scoring_short"]["passed"] is True


def test_refactor5_report_sections_present(tmp_path: Path) -> None:
    n = 40
    closes = [100.0 + 0.01 * i for i in range(n)]
    (tmp_path / "closes.json").write_text(json.dumps(closes), encoding="utf-8")
    decisions = []
    for i in range(n):
        decisions.append(
            {
                "event_type": "decision",
                "bar_index": i,
                "timestamp": str(1700000000 + i * 300),
                "features": {
                    "breakout_strength": 1.0 if i % 5 == 0 else 0.0,
                    "raw_strength": 1.0 if i % 5 == 0 else 0.0,
                    "effective_strength": 0.1 * (i % 5),
                    "last_bar_return": 0.001 * ((i % 3) - 1),
                    "breakout_up_filtered_last": bool(i % 5 == 0),
                },
                "v2": {
                    "score_long": 0.1 * (i % 7),
                    "score_short": -0.1 * (i % 7),
                    "score_components": {"positive_last_bar_return": 0.2, "crowd_phase": "ignition"},
                    "short_components": {"short_phase": "short_setup", "short_score": 0.3},
                    "log": {"crowd_phase": "ignition"},
                },
            }
        )
    trades = [
        {
            "event_type": "trade",
            "model": "v2",
            "event": "CLOSED",
            "bar_index": 10,
            "entry_bar_index": 8,
            "exit_bar_index": 10,
            "side": "LONG",
            "realized_return": 0.001,
            "holding_period": 2,
            "entry_crowd_phase": "ignition",
            "exit_reason": "score_decay",
            "signal_bar_index": 7,
        }
    ]
    dec = tmp_path / "dec.jsonl"
    trd = tmp_path / "tr.jsonl"
    _write_jsonl(dec, decisions)
    _write_jsonl(trd, trades)
    r = run_validation(dec, trd, closes_path=(tmp_path / "closes.json"), min_trades_adaptive=1)
    r5 = r.get("refactor5_report") or {}
    assert "dataset_summary" in r5
    assert "long_scoring_validation" in r5
    assert "exit_reason_performance" in r5
    assert "short_validation" in r5
    assert "acceptance" in r5


def test_refactor5_acceptance_fails_when_component_coverage_zero(tmp_path: Path) -> None:
    closes = [100.0 + 0.01 * i for i in range(20)]
    (tmp_path / "closes.json").write_text(json.dumps(closes), encoding="utf-8")
    decisions = [
        {
            "event_type": "decision",
            "bar_index": i,
            "timestamp": str(1700000000 + i * 300),
            "features": {"breakout_up_filtered_last": bool(i % 4 == 0)},
            "v2": {"score_long": 0.1, "score_short": -0.1, "log": {"crowd_phase": "no_breakout"}},
        }
        for i in range(20)
    ]
    trades = [
        {
            "event_type": "trade",
            "model": "v2",
            "event": "CLOSED",
            "bar_index": 5,
            "entry_bar_index": 4,
            "exit_bar_index": 5,
            "side": "LONG",
            "realized_return": 0.001,
            "holding_period": 1,
            "exit_reason": "unknown",
            "signal_bar_index": 3,
        }
    ]
    dec = tmp_path / "dec_zero_cov.jsonl"
    trd = tmp_path / "tr_zero_cov.jsonl"
    _write_jsonl(dec, decisions)
    _write_jsonl(trd, trades)
    r = run_validation(dec, trd, closes_path=(tmp_path / "closes.json"), min_trades_adaptive=1)
    acc = (r.get("refactor5_report") or {}).get("acceptance") or {}
    assert acc.get("overall_acceptance") is False
    assert ((acc.get("logging") or {}).get("decision_score_components_coverage")) == 0.0


def test_refactor5_last_bar_return_spike_bucket_non_degenerate(tmp_path: Path) -> None:
    closes = [100.0 + 0.01 * i for i in range(40)]
    (tmp_path / "closes.json").write_text(json.dumps(closes), encoding="utf-8")
    decisions = []
    for i in range(40):
        pos = 0.0 if i < 30 else 0.2 + 0.01 * (i - 30)
        decisions.append(
            {
                "event_type": "decision",
                "bar_index": i,
                "timestamp": str(1700000000 + i * 300),
                "features": {
                    "breakout_up_filtered_last": True,
                    "last_bar_return": 0.001 * (i - 20),
                    "effective_strength": 0.2,
                    "raw_strength": 1.0,
                },
                "v2": {
                    "score_long": 0.3,
                    "score_short": -0.3,
                    "score_components": {"positive_last_bar_return": pos, "crowd_phase": "ignition"},
                    "short_components": {"short_phase": "no_short_context"},
                    "log": {"crowd_phase": "ignition"},
                },
            }
        )
    dec = tmp_path / "dec_spike.jsonl"
    trd = tmp_path / "tr_spike.jsonl"
    _write_jsonl(dec, decisions)
    _write_jsonl(trd, [])
    r = run_validation(dec, trd, closes_path=(tmp_path / "closes.json"), min_trades_adaptive=1)
    lbr = ((r.get("refactor5_report") or {}).get("last_bar_return_impact") or {})
    assert lbr.get("spike_bucket_non_degenerate") is True
    assert (lbr.get("non_spike_count") or 0) > 0
    assert (lbr.get("spike_count") or 0) > 0


@pytest.mark.parametrize(
    "fname",
    [
        "ttm_parallel_backtest_VN30F1M_20260301_20260305_20260405_0649_decisions.jsonl",
    ],
)
def test_smoke_repo_backtest_if_present(fname: str) -> None:
    root = Path(__file__).resolve().parent.parent
    dec = root / "reports" / fname
    trd = root / "reports" / fname.replace("_decisions", "_trades")
    closes = root / "reports" / fname.replace("_decisions.jsonl", "_closes.json")
    if not dec.is_file() or not trd.is_file():
        pytest.skip("report pair not in workspace")
    if not closes.is_file():
        pytest.skip("strict validation requires aligned closes JSON (see --closes)")
    r = run_validation(
        dec,
        trd,
        closes_path=closes,
        min_samples_breakout=5,
        min_samples_scoring=5,
        min_trades_adaptive=4,
    )
    assert isinstance(r["overall_valid"], bool)
    assert set(r.keys()) >= {"breakout", "positioning", "scoring", "adaptive", "overall_valid"}
