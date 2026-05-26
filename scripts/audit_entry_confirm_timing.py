#!/usr/bin/env python3
"""Audit entry_confirm causality: signal bar N vs execution bar N+1 features."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.strategies.ttm.ttm_parallel_runner import (
    _v2_entry_execution_long_confirm,
    build_ttm_paper_live_config,
)
from src.strategies.ttm.ttm_v2_gates import (
    GATE_MODE_QUALITY_RESEARCH,
    entry_confirm_mode_for_gate,
    resolve_gate_mode,
)


def load_decisions(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event_type") == "decision":
            out[int(d["bar_index"])] = d
    return out


def load_blocked(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("event_type") == "ENTRY_BLOCKED":
            rows.append(ev)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--decisions", required=True)
    p.add_argument("--trades", required=True)
    args = p.parse_args()
    dec_by = load_decisions(Path(args.decisions))
    blocked = load_blocked(Path(args.trades))
    cfg = build_ttm_paper_live_config({"ttm_v2_gate_mode": GATE_MODE_QUALITY_RESEARCH})
    ecm = entry_confirm_mode_for_gate(resolve_gate_mode(cfg, live_mode=False))
    so = bool(cfg.get("ttm_v2_allow_signal_only_long_execution", False))

    def run_confirm(last: dict, ev: dict) -> tuple[bool, str]:
        return _v2_entry_execution_long_confirm(
            {},
            last,
            cfg,
            signal_only_exec=so,
            signal_crowd_phase=str(ev.get("signal_crowd_phase") or ""),
            signal_long_candidate=bool(ev.get("signal_long_candidate")),
            entry_confirm_mode=ecm,
        )[:2]

    pass_signal = pass_n2 = 0
    print(f"ENTRY_BLOCKED total: {len(blocked)}")
    for ev in blocked:
        sb = int(ev["signal_bar_index"])
        eb = int(ev["bar_index"])
        snap = dec_by.get(sb, {}).get("features") or {}
        execf = dec_by.get(eb, {}).get("features") or {}
        n2f = dec_by.get(eb + 1, {}).get("features") or {}
        b_exec, r_exec = run_confirm(execf, ev)
        b_sig, r_sig = run_confirm(snap, ev)
        b_n2, r_n2 = run_confirm(n2f, ev)
        if not b_sig:
            pass_signal += 1
        if not b_n2:
            pass_n2 += 1
        print(
            f"  {ev['reason']}: sig={sb} exec={eb} "
            f"lbr_sig={snap.get('last_bar_return')} lbr_exec={execf.get('last_bar_return')} "
            f"phase_exec_log={ev.get('entry_confirm_phase')} "
            f"re_sim_exec={r_exec} re_sim_sig={r_sig}"
        )
    print(f"would PASS with signal-bar features: {pass_signal}/{len(blocked)}")
    print(f"would PASS with N+2 bar features: {pass_n2}/{len(blocked)}")
    print("reasons:", dict(Counter(ev["reason"] for ev in blocked)))


if __name__ == "__main__":
    main()
