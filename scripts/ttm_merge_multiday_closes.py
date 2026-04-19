#!/usr/bin/env python3
"""Build one closes JSON array for TTM multiday validation (same merge order as ttm_validation).

Uses per-session ``closes_VN30F1M_{date}_{time}.json`` when present; for 20260408 subsessions
without individual files, slices ``closes_VN30F1M_20260408_reconstructed_209.json`` in glob order.
Otherwise reconstructs from ``features.positioning_log.price_change`` (anchor 1000.0) — exploratory only.

Example:
  python scripts/ttm_merge_multiday_closes.py --out reports/closes_VN30F1M_20260406_20260410_merged.json
  python scripts/ttm_validation.py --aggregate-start-date 20260406 --aggregate-end-date 20260410 \\
    --closes reports/closes_VN30F1M_20260406_20260410_merged.json --scoring-mode all ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.strategies.ttm.ttm_validation import (  # noqa: E402
    _iter_dates_yyyymmdd,
    load_decisions_jsonl,
)


def _reconstruct_from_decisions(dec_path: Path) -> list[float]:
    rows = load_decisions_jsonl(dec_path)
    if not rows:
        return []
    n = max(int(d["bar_index"]) for d in rows) + 1
    pc = [0.0] * n
    for d in rows:
        bi = int(d["bar_index"])
        pl = (d.get("features") or {}).get("positioning_log") or {}
        pc[bi] = float(pl.get("price_change") or 0.0)
    anchor = 1000.0
    out = [0.0] * n
    out[0] = anchor
    for i in range(1, n):
        out[i] = out[i - 1] + pc[i]
    return out


def _session_bars(dec_path: Path) -> int:
    rows = load_decisions_jsonl(dec_path)
    if not rows:
        return 0
    return max(int(d["bar_index"]) for d in rows) + 1


def build_merged_closes(
    *,
    reports_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> tuple[list[float], list[str]]:
    notes: list[str] = []
    merged: list[float] = []
    recon08: dict[str, object] | None = None

    for ymd in _iter_dates_yyyymmdd(start_date, end_date):
        dec_files = sorted(reports_dir.glob(f"ttm_parallel_decisions_{symbol}_{ymd}_*.jsonl"))
        for dfile in dec_files:
            rest = dfile.name.replace(f"ttm_parallel_decisions_{symbol}_", "").replace(".jsonl", "")
            cpath = reports_dir / f"closes_{symbol}_{rest}.json"
            nb = _session_bars(dfile)
            if nb == 0:
                continue

            part: list[float]
            if cpath.is_file():
                raw = json.loads(cpath.read_text(encoding="utf-8"))
                if not isinstance(raw, list):
                    raise ValueError(f"{cpath} must be a JSON array")
                part = [float(x) for x in raw]
                if len(part) != nb:
                    raise ValueError(f"{cpath}: len {len(part)} != session bars {nb} ({dfile.name})")
            elif ymd == "20260408":
                master = reports_dir / "closes_VN30F1M_20260408_reconstructed_209.json"
                if not master.is_file():
                    notes.append(f"reconstructed_positioning: {dfile.name}")
                    part = _reconstruct_from_decisions(dfile)
                else:
                    if recon08 is None:
                        full = json.loads(master.read_text(encoding="utf-8"))
                        if not isinstance(full, list) or len(full) != 209:
                            raise ValueError(f"{master} expected list of 209")
                        recon08 = {"full": [float(x) for x in full], "off": 0}
                    assert recon08 is not None
                    off = int(recon08["off"])  # type: ignore[arg-type]
                    full_list = recon08["full"]  # type: ignore[assignment]
                    part = full_list[off : off + nb]
                    recon08["off"] = off + nb
                    notes.append(f"sliced_209: {dfile.name} [{off}:{off + nb}]")
            else:
                notes.append(f"reconstructed_positioning: {dfile.name}")
                part = _reconstruct_from_decisions(dfile)

            merged.extend(part)

    return merged, notes


def main() -> int:
    p = argparse.ArgumentParser(description="Merge per-session closes for TTM multiday validation")
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    p.add_argument("--symbol", type=str, default="VN30F1M")
    p.add_argument("--start-date", type=str, required=True, help="YYYYMMDD")
    p.add_argument("--end-date", type=str, required=True, help="YYYYMMDD")
    p.add_argument("--out", type=Path, required=True, help="Output JSON array path")
    args = p.parse_args()

    merged, notes = build_merged_closes(
        reports_dir=args.reports_dir,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged), encoding="utf-8")
    print(f"wrote {len(merged)} closes -> {args.out}")
    for n in notes:
        print(f"note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
