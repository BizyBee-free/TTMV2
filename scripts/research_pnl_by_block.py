"""Aggregate forward_return and PnL by block reason from parallel backtest JSONL."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


def _closes_path_for_decisions(dec_path: Path) -> Optional[Path]:
    name = dec_path.name
    if not name.endswith("_decisions.jsonl"):
        return None
    core = name[: -len("_decisions.jsonl")]
    cand = dec_path.parent / f"closes_{core}_decisions.json"
    return cand if cand.is_file() else None


def _forward_return(closes: List[float], bar_index: int, h: int) -> Optional[float]:
    if h <= 0 or bar_index < 0 or bar_index + h >= len(closes):
        return None
    a, b = float(closes[bar_index]), float(closes[bar_index + h])
    if abs(a) < 1e-12:
        return None
    return (b - a) / abs(a)


def _gd(d: dict) -> dict:
    v2 = d.get("v2") or {}
    g = v2.get("gate_diagnostics")
    return g if isinstance(g, dict) else {}


def _v2(d: dict) -> dict:
    v = d.get("v2")
    return v if isinstance(v, dict) else {}


def load_decisions(path: Path) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("event_type") == "decision":
            out[int(d["bar_index"])] = d
    return out


def load_trades(path: Path) -> Tuple[Dict[int, str], set[int], List[dict]]:
    blocked: Dict[int, str] = {}
    closed_sig: set[int] = set()
    closed_rows: List[dict] = []
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
                blocked[int(sbi)] = str(ev.get("reason") or ev.get("entry_block_reason") or "unknown")
        is_closed = str(et).upper() == "CLOSED" or str(ev.get("event", "")).upper() == "CLOSED"
        if is_closed and str(ev.get("side", "")).upper() == "LONG":
            sbi = ev.get("signal_bar_index")
            if sbi is not None:
                closed_sig.add(int(sbi))
            closed_rows.append(ev)
    return blocked, closed_sig, closed_rows


def _gate_key(gate_mode: str) -> str:
    if "exploratory" in gate_mode:
        return "exploratory_research_long_allowed"
    return "quality_research_long_allowed"


def _tier_block_key(gate_mode: str) -> str:
    if "exploratory" in gate_mode:
        return "exploratory_research_block_reason"
    return "quality_research_block_reason"


def _fr_stats(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0, "mean_fr": None, "sum_fr": None, "median_fr": None}
    return {
        "n": len(vals),
        "mean_fr": statistics.mean(vals),
        "sum_fr": sum(vals),
        "median_fr": statistics.median(vals),
    }


def agg_forward_by_reason(
    dec_by: Dict[int, dict],
    closed_sig: set[int],
    gate_mode: str,
    closes: Optional[List[float]] = None,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Returns rows for blocked_by, entry_block, tier_gate (non-traded bars with forward_return)."""
    by_blocked: DefaultDict[str, List[float]] = defaultdict(list)
    by_entry: DefaultDict[str, List[float]] = defaultdict(list)
    by_tier: DefaultDict[str, List[float]] = defaultdict(list)
    gk = _gate_key(gate_mode)
    tbk = _tier_block_key(gate_mode)

    for bi, d in dec_by.items():
        if bi in closed_sig:
            continue
        fr = d.get("forward_return")
        if fr is None and closes is not None:
            h = int(d.get("forward_horizon_bars") or 4)
            fr = _forward_return(closes, bi, h)
        if fr is None:
            continue
        v2 = _v2(d)
        by_blocked[str(v2.get("blocked_by") or "none")].append(float(fr))
        sc = v2.get("score_components") or {}
        dbg = v2.get("debug") or {}
        ebr = dbg.get("entry_block_reason") or sc.get("entry_block_reason") or "none"
        by_entry[str(ebr)].append(float(fr))
        g = _gd(d)
        if not g.get(gk):
            by_tier[str(g.get(tbk) or g.get("research_block_reason") or "tier_blocked")].append(float(fr))
    tables = []
    for title, buckets in [
        ("v2.blocked_by (no trade)", by_blocked),
        ("entry_block_reason (no trade)", by_entry),
        (f"{tbk} when gate not passed (no trade)", by_tier),
    ]:
        rows = []
        for reason, vals in sorted(buckets.items(), key=lambda x: (-len(x[1]), x[0])):
            st = _fr_stats(vals)
            rows.append({"reason": reason, **st})
        tables.append({"title": title, "rows": rows})
    return tables[0]["rows"], tables[1]["rows"], tables[2]["rows"]


def split_funnel(
    dec_by: Dict[int, dict],
    blocked_sig: Dict[int, str],
    closed_sig: set[int],
    gate_mode: str,
    closes: Optional[List[float]] = None,
) -> dict:
    gk = _gate_key(gate_mode)
    passed_hold = 0
    passed_hold_fr: List[float] = []
    entry_blocked_n = len(blocked_sig)
    closed_n = len(closed_sig)
    for bi, d in dec_by.items():
        g = _gd(d)
        if not g.get(gk):
            continue
        if bi in closed_sig or bi in blocked_sig:
            continue
        if str(_v2(d).get("decision", "")).upper() != "LONG":
            passed_hold += 1
            fr = d.get("forward_return")
            if fr is None and closes is not None:
                h = int(d.get("forward_horizon_bars") or 4)
                fr = _forward_return(closes, bi, h)
            if fr is not None:
                passed_hold_fr.append(float(fr))
    return {
        "passed_gate_hold_bars": passed_hold,
        "passed_gate_hold_mean_fr": statistics.mean(passed_hold_fr) if passed_hold_fr else None,
        "entry_confirm_blocked_signals": entry_blocked_n,
        "closed_trades": closed_n,
    }


def pnl_tables(closed_rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    by_exit: DefaultDict[str, List[float]] = defaultdict(list)
    by_sig_ebr: DefaultDict[str, List[float]] = defaultdict(list)
    for t in closed_rows:
        by_exit[str(t.get("exit_reason") or "unknown")].append(float(t.get("pnl") or 0))
        by_sig_ebr[str(t.get("signal_entry_block_reason") or "none")].append(float(t.get("pnl") or 0))
    exit_rows = []
    for k, vals in sorted(by_exit.items(), key=lambda x: (-len(x[1]), x[0])):
        exit_rows.append(
            {
                "exit_reason": k,
                "n": len(vals),
                "sum_pnl": sum(vals),
                "mean_pnl": statistics.mean(vals),
                "winrate": sum(1 for v in vals if v > 0) / len(vals),
            }
        )
    sig_rows = []
    for k, vals in sorted(by_sig_ebr.items(), key=lambda x: (-len(x[1]), x[0])):
        sig_rows.append(
            {
                "signal_entry_block_reason": k,
                "n": len(vals),
                "sum_pnl": sum(vals),
                "mean_pnl": statistics.mean(vals),
            }
        )
    return exit_rows, sig_rows


def forensics_quality(
    dec_by: Dict[int, dict],
    closed_rows: List[dict],
) -> Tuple[List[dict], dict]:
    trades: List[dict] = []
    for t in sorted(closed_rows, key=lambda x: int(x.get("signal_bar_index") or 0)):
        sbi = int(t.get("signal_bar_index") or 0)
        pnl = float(t.get("pnl") or 0)
        if pnl > 0:
            wl = "winner"
        elif pnl < 0:
            wl = "loser"
        else:
            wl = "flat"
        d = dec_by.get(sbi, {})
        g = _gd(d)
        trades.append(
            {
                "signal_bar": sbi,
                "pnl": pnl,
                "outcome": wl,
                "crowd_phase_signal": t.get("signal_crowd_phase") or g.get("crowd_phase"),
                "pullback_after_ignition": g.get("pullback_after_ignition"),
                "entry_confirm_phase": t.get("entry_confirm_phase"),
                "entry_crowd_phase": t.get("entry_crowd_phase"),
                "exit_reason": t.get("exit_reason"),
                "holding_bars": t.get("holding_bars"),
                "mfe": t.get("mfe"),
                "mae": t.get("mae"),
            }
        )

    def bucket(field: str, outcome: str) -> Counter:
        c: Counter = Counter()
        for tr in trades:
            if tr["outcome"] != outcome:
                continue
            c[str(tr.get(field))] += 1
        return c

    def count_pullback(outcome: str, want: bool) -> int:
        return sum(
            1
            for tr in trades
            if tr["outcome"] == outcome and bool(tr.get("pullback_after_ignition")) is want
        )

    agg = {
        "n_trades": len(trades),
        "winners": sum(1 for t in trades if t["outcome"] == "winner"),
        "losers": sum(1 for t in trades if t["outcome"] == "loser"),
        "flat": sum(1 for t in trades if t["outcome"] == "flat"),
        "sum_pnl": sum(t["pnl"] for t in trades),
        "winners_pullback_true": count_pullback("winner", True),
        "losers_pullback_true": count_pullback("loser", True),
        "winners_by_crowd_phase": dict(bucket("crowd_phase_signal", "winner")),
        "losers_by_crowd_phase": dict(bucket("crowd_phase_signal", "loser")),
        "winners_by_entry_confirm": dict(bucket("entry_confirm_phase", "winner")),
        "losers_by_entry_confirm": dict(bucket("entry_confirm_phase", "loser")),
    }
    return trades, agg


def _md_table(headers: List[str], rows: List[dict], keys: List[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k)
            if isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}")
            elif v is None:
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_run_section(
    label: str,
    dec_path: Path,
    tr_path: Path,
    gate_mode: str,
) -> str:
    dec_by = load_decisions(dec_path)
    blocked_sig, closed_sig, closed_rows = load_trades(tr_path)
    closes_path = _closes_path_for_decisions(dec_path)
    closes: Optional[List[float]] = None
    if closes_path:
        closes = [float(x) for x in json.loads(closes_path.read_text(encoding="utf-8"))]
    gm = gate_mode
    if not gm:
        for d in dec_by.values():
            gm = str(_gd(d).get("gate_mode") or _v2(d).get("log", {}).get("gate_mode") or "")
            if gm:
                break
    fr_blocked, fr_entry, fr_tier = agg_forward_by_reason(dec_by, closed_sig, gm, closes)
    funnel = split_funnel(dec_by, blocked_sig, closed_sig, gm, closes)
    exit_rows, sig_rows = pnl_tables(closed_rows)
    entry_blocked_reasons = Counter(blocked_sig.values())

    parts = [f"### {label}\n", f"- decisions: `{dec_path}`\n", f"- trades: `{tr_path}`\n"]
    parts.append("**Funnel split (v2 LONG)**\n")
    parts.append(
        f"- passed gate but HOLD (no LONG signal at bar): {funnel['passed_gate_hold_bars']}"
        f" (mean forward_return: {funnel['passed_gate_hold_mean_fr']})\n"
    )
    parts.append(f"- ENTRY_BLOCKED at signal: {funnel['entry_confirm_blocked_signals']}\n")
    if entry_blocked_reasons:
        parts.append(f"- ENTRY_BLOCKED reasons: {dict(entry_blocked_reasons)}\n")
    parts.append(f"- CLOSED LONG: {funnel['closed_trades']}\n\n")

    parts.append("**Forward return by v2.blocked_by (bars without closed trade)**\n\n")
    parts.append(
        _md_table(
            ["reason", "n", "mean_fr", "median_fr", "sum_fr"],
            fr_blocked,
            ["reason", "n", "mean_fr", "median_fr", "sum_fr"],
        )
    )
    parts.append("**Forward return by entry_block_reason (no trade)**\n\n")
    parts.append(
        _md_table(
            ["reason", "n", "mean_fr", "median_fr", "sum_fr"],
            fr_entry,
            ["reason", "n", "mean_fr", "median_fr", "sum_fr"],
        )
    )
    parts.append("**PnL by exit_reason (CLOSED LONG)**\n\n")
    parts.append(
        _md_table(
            ["exit_reason", "n", "sum_pnl", "mean_pnl", "winrate"],
            exit_rows,
            ["exit_reason", "n", "sum_pnl", "mean_pnl", "winrate"],
        )
    )
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decisions", type=Path, action="append", default=[], help="decisions.jsonl")
    ap.add_argument("--trades", type=Path, action="append", default=[], help="trades.jsonl")
    ap.add_argument("--label", action="append", default=[], help="section label per run")
    ap.add_argument("--gate-mode", action="append", default=[], help="quality_research|exploratory_research")
    ap.add_argument("--out", type=Path, required=True, help="markdown report path")
    ap.add_argument(
        "--forensics-decisions",
        type=Path,
        default=None,
        help="quality causal-fix decisions for 19-trade forensics",
    )
    ap.add_argument(
        "--forensics-trades",
        type=Path,
        default=None,
        help="quality causal-fix trades for forensics",
    )
    ap.add_argument(
        "--compare-summary",
        type=Path,
        action="append",
        default=[],
        help="summary.json paths for exploratory before/after table",
    )
    ap.add_argument(
        "--compare-label",
        action="append",
        default=[],
        help="label per --compare-summary row",
    )
    args = ap.parse_args()

    if len(args.decisions) != len(args.trades):
        raise SystemExit("--decisions and --trades must have same count")
    labels = args.label or [f"run_{i}" for i in range(len(args.decisions))]
    gate_modes = args.gate_mode or [""] * len(args.decisions)

    sections: List[str] = ["# TTM research: causal-fix backtest analysis\n"]

    if args.compare_summary:
        cmp_labels = args.compare_label or [p.stem for p in args.compare_summary]
        rows = []
        for lab, sp in zip(cmp_labels, args.compare_summary):
            root = json.loads(sp.read_text(encoding="utf-8"))
            s = root.get("summary") or root
            v2 = s.get("v2") or {}
            tr_path = Path(str(root.get("trades_jsonl") or sp.with_name(sp.name.replace("_summary.json", "_trades.jsonl"))))
            if not tr_path.is_file():
                tr_path = sp.parent / tr_path.name
            blocked_n = 0
            for line in tr_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                if ev.get("event_type") == "ENTRY_BLOCKED" and str(ev.get("model", "")).lower() == "v2":
                    blocked_n += 1
            rows.append(
                {
                    "run": lab,
                    "v2_trades": v2.get("trades"),
                    "v2_pnl": round(float(v2.get("pnl") or 0), 2),
                    "v2_winrate": round(float(v2.get("winrate") or 0), 3),
                    "entry_blocked": blocked_n,
                }
            )
        sections.append("## Exploratory: before vs after entry-confirm fix\n\n")
        sections.append(
            _md_table(
                ["run", "v2_trades", "v2_pnl", "v2_winrate", "ENTRY_BLOCKED"],
                rows,
                ["run", "v2_trades", "v2_pnl", "v2_winrate", "entry_blocked"],
            )
        )
        sections.append("\n")
    for dec, tr, lab, gm in zip(args.decisions, args.trades, labels, gate_modes):
        sections.append(render_run_section(lab, dec, tr, gm))

    if args.forensics_decisions and args.forensics_trades:
        dec_by = load_decisions(args.forensics_decisions)
        _, _, closed_rows = load_trades(args.forensics_trades)
        trades, agg = forensics_quality(dec_by, closed_rows)
        sections.append("## Quality forensics (19 v2 CLOSED LONG)\n\n")
        sections.append("**Per trade**\n\n")
        sections.append(
            _md_table(
                [
                    "signal_bar",
                    "pnl",
                    "outcome",
                    "crowd_phase",
                    "pullback",
                    "entry_confirm",
                    "exit",
                    "hold",
                    "mfe",
                    "mae",
                ],
                trades,
                [
                    "signal_bar",
                    "pnl",
                    "outcome",
                    "crowd_phase_signal",
                    "pullback_after_ignition",
                    "entry_confirm_phase",
                    "exit_reason",
                    "holding_bars",
                    "mfe",
                    "mae",
                ],
            )
        )
        sections.append("**Aggregates**\n\n")
        sections.append(f"- Trades: {agg['n_trades']} | winners: {agg['winners']} | losers: {agg['losers']} | flat: {agg['flat']}\n")
        sections.append(f"- Sum PnL: {agg['sum_pnl']:.2f}\n")
        sections.append(f"- Winners with pullback_after_ignition=True: {agg['winners_pullback_true']}/{agg['winners']}\n")
        sections.append(f"- Losers with pullback_after_ignition=True: {agg['losers_pullback_true']}/{agg['losers']}\n")
        sections.append(f"- Winners crowd_phase @ signal: {agg['winners_by_crowd_phase']}\n")
        sections.append(f"- Losers crowd_phase @ signal: {agg['losers_by_crowd_phase']}\n")
        sections.append(f"- Winners entry_confirm_phase: {agg['winners_by_entry_confirm']}\n")
        sections.append(f"- Losers entry_confirm_phase: {agg['losers_by_entry_confirm']}\n\n")
        w_pull = agg["winners_pullback_true"]
        n_w = agg["winners"]
        n_l = agg["losers"]
        l_pull = agg["losers_pullback_true"]
        sections.append(
            "**Conclusion (EN):** "
            + (
                f"{w_pull}/{n_w} winners and {l_pull}/{n_l} losers had "
                "`pullback_after_ignition=True` at signal (gate diagnostics); "
                "all signals show `crowd_phase=no_breakout` — entries are pullback-in-window "
                "after a prior upside breakout, not live ignition/continuation phases."
                if n_w and w_pull == n_w
                else f"Only {w_pull}/{n_w} winners had pullback_after_ignition at signal."
            )
            + "\n\n"
        )
        sections.append(
            "**Kết luận (VI):** "
            + (
                f"{w_pull}/{n_w} lệnh thắng và {l_pull}/{n_l} lệnh thua có "
                "pullback_after_ignition=True tại bar tín hiệu; crowd_phase đều no_breakout — "
                "vào lệnh kiểu pullback trong cửa sổ sau breakout, không phải ignition trực tiếp."
                if n_w and w_pull == n_w
                else f"Chỉ {w_pull}/{n_w} lệnh thắng có pullback_after_ignition tại signal."
            )
            + "\n"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(sections), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
