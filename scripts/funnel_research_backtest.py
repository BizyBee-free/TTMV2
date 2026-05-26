"""One-off funnel: research gate candidates -> closed trades."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def gd(d: dict) -> dict:
    v2 = d.get("v2") or {}
    g = v2.get("gate_diagnostics")
    return g if isinstance(g, dict) else {}


def load_decisions(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event_type") == "decision":
            out[int(d["bar_index"])] = d
    return out


def load_trades(path: Path) -> tuple[dict[int, str], set[int]]:
    blocked: dict[int, str] = {}
    closed: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if str(ev.get("model", "")).lower() != "v2":
            continue
        et = ev.get("event_type") or ev.get("event")
        if et == "ENTRY_BLOCKED":
            sbi = ev.get("signal_bar_index")
            if sbi is not None:
                blocked[int(sbi)] = str(ev.get("reason") or "unknown")
        is_closed = str(et).upper() == "CLOSED" or str(ev.get("event", "")).upper() == "CLOSED"
        if is_closed and str(ev.get("side", "")).upper() == "LONG":
            sbi = ev.get("signal_bar_index")
            if sbi is not None:
                closed.add(int(sbi))
    return blocked, closed


def analyze(
    dec_path: Path,
    tr_path: Path,
    gate_key: str,
    label: str,
) -> dict:
    dec_by = load_decisions(dec_path)
    blocked_sig, closed_sig = load_trades(tr_path)

    universe = passed = entry_allow = cap_block = prob_block = 0
    tier_block = Counter()
    for d in dec_by.values():
        g = gd(d)
        if g.get("long_candidate"):
            universe += 1
            if not g.get(gate_key):
                tier_block[
                    str(
                        g.get("quality_research_block_reason")
                        or g.get("exploratory_research_block_reason")
                        or "tier"
                    )
                ] += 1
        if g.get(gate_key):
            passed += 1
        if g.get("research_long_entry_allowed"):
            entry_allow += 1
        if g.get("blocked_by_trade_cap"):
            cap_block += 1
        if g.get("blocked_by_prob_gate"):
            prob_block += 1

    long_signals = [
        bi
        for bi, d in dec_by.items()
        if str((d.get("v2") or {}).get("decision", "")).upper() == "LONG"
    ]
    entry_allow_no_long = sum(
        1
        for d in dec_by.values()
        if gd(d).get("research_long_entry_allowed")
        and str((d.get("v2") or {}).get("decision", "")).upper() != "LONG"
    )

    outcomes = Counter()
    for sb in long_signals:
        if sb in closed_sig:
            outcomes["closed"] += 1
        elif sb in blocked_sig:
            outcomes[f"entry_confirm:{blocked_sig[sb]}"] += 1
        else:
            nb = dec_by.get(sb + 1, {})
            v2n = nb.get("v2") or {}
            dec = str(v2n.get("decision", "?")).upper()
            blk = str(v2n.get("blocked_by", "") or "")
            outcomes[f"other:{dec}/{blk or 'none'}"] += 1

    entry_confirm_total = sum(v for k, v in outcomes.items() if k.startswith("entry_confirm:"))
    entry_confirm_reasons = Counter(
        k.split(":", 1)[1] for k, v in outcomes.items() if k.startswith("entry_confirm:")
        for _ in range(v)
    )

    passed_no_long = Counter()
    for d in dec_by.values():
        g = gd(d)
        if not g.get(gate_key):
            continue
        if str((d.get("v2") or {}).get("decision", "")).upper() == "LONG":
            continue
        if g.get("blocked_by_trade_cap"):
            passed_no_long["trade_cap"] += 1
        elif not g.get("research_long_entry_allowed"):
            passed_no_long["no_entry_allowed"] += 1
        else:
            passed_no_long["entry_ok_but_HOLD_or_other"] += 1

    return {
        "label": label,
        "universe": universe,
        "passed": passed,
        "entry_allow": entry_allow,
        "cap_block": cap_block,
        "prob_block": prob_block,
        "decision_long": len(long_signals),
        "entry_confirm_blocked": entry_confirm_total,
        "closed": len(closed_sig),
        "outcomes": dict(outcomes),
        "entry_confirm_reasons": dict(entry_confirm_reasons),
        "passed_no_long": dict(passed_no_long),
        "entry_allow_no_long": entry_allow_no_long,
        "tier_block": dict(tier_block),
    }


def pct(n: int, denom: int) -> str:
    if denom <= 0:
        return "—"
    return f"{100.0 * n / denom:.1f}%"


def main() -> None:
    runs = [
        (
            Path("reports/quality_research_20260401_20260523")
            / "ttm_parallel_backtest_VN30F1M_20260401_20260523_20260526_0614_decisions.jsonl",
            Path("reports/quality_research_20260401_20260523")
            / "ttm_parallel_backtest_VN30F1M_20260401_20260523_20260526_0614_trades.jsonl",
            "quality_research_long_allowed",
            "quality_research",
        ),
        (
            Path("reports/exploratory_research_20260401_20260523")
            / "ttm_parallel_backtest_VN30F1M_20260401_20260523_20260526_0615_decisions.jsonl",
            Path("reports/exploratory_research_20260401_20260523")
            / "ttm_parallel_backtest_VN30F1M_20260401_20260523_20260526_0615_trades.jsonl",
            "exploratory_research_long_allowed",
            "exploratory_research",
        ),
    ]
    results = [analyze(d, t, gk, lb) for d, t, gk, lb in runs]
    for r in results:
        print("===", r["label"], "===")
        stages = [
            ("candidate_universe", r["universe"]),
            ("passed_score_gate", r["passed"]),
            ("research_entry_allowed", r["entry_allow"]),
            ("blocked_by_trade_cap", r["cap_block"]),
            ("blocked_by_prob_gate", r["prob_block"]),
            ("decision_LONG", r["decision_long"]),
            ("entry_confirm_BLOCKED", r["entry_confirm_blocked"]),
            ("CLOSED", r["closed"]),
        ]
        for name, n in stages:
            print(f"  {name:28} {n:4}  (retain {pct(n, r['universe'])})")
        print("  LONG trace:", r["outcomes"])
        print("  entry_confirm reasons:", r["entry_confirm_reasons"])
        print("  passed_gate not LONG:", r["passed_no_long"])
        print("  entry_allowed not LONG:", r["entry_allow_no_long"])
        print("  universe blocked (top):", Counter(r["tier_block"]).most_common(5))
        print()


if __name__ == "__main__":
    main()
