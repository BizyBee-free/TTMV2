"""Strict bar alignment for basis / OI and optional persistence for replay."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


class TTMAlignmentError(RuntimeError):
    """Raised when strict basis/OI alignment is violated (fail-fast)."""


def validate_ohlc_close_alignment(close: Any, bars: Any) -> None:
    """
    Validate OHLC close alignment against bars.

    Required checks:
    - ``len(close) == len(bars)``
    - if both expose ``index``, then ``(close.index == bars.index).all()``

    Raises
    ------
    ValueError
        When length or index alignment is invalid.
    """
    if close is None or bars is None:
        return
    if len(close) != len(bars):
        raise ValueError(f"OHLC alignment mismatch: len(close)={len(close)} != len(bars)={len(bars)}")
    if hasattr(close, "index") and hasattr(bars, "index"):
        idx_ok = bool((close.index == bars.index).all())
        if not idx_ok:
            raise ValueError("OHLC alignment mismatch: close.index != bars.index")


def _bars_close_series(bars: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for b in bars:
        if hasattr(b, "close"):
            out.append(float(b.close))
        else:
            out.append(float((b or {}).get("close", 0) or 0))
    return out


def _bars_unix_ts(bars: Sequence[Any]) -> List[int]:
    out: List[int] = []
    for b in bars:
        ut = int(getattr(b, "unix_ts", 0) or 0)
        if ut == 0 and isinstance(b, dict):
            ut = int(b.get("unix_ts", 0) or 0)
        out.append(ut)
    return out


def validate_ttm_input_alignment(
    data: Mapping[str, Any],
    *,
    strict_basis_oi: bool,
    use_open_interest: bool,
) -> None:
    """
    When ``strict_basis_oi`` is True, ``basis`` and (if ``use_open_interest``) ``open_interest``
    must be present, length-equal to ``bars``, and all finite. No silent zeros.
    """
    if not strict_basis_oi:
        return
    bars = data.get("bars")
    if not bars:
        return
    n = len(bars)
    b = data.get("basis")
    if b is None:
        raise TTMAlignmentError("strict_basis_oi: missing 'basis' aligned to bars")
    ba = np.asarray(b, dtype=np.float64).ravel()
    if ba.size != n:
        raise TTMAlignmentError(
            f"strict_basis_oi: basis length {ba.size} != bars length {n}"
        )
    if not np.all(np.isfinite(ba)):
        raise TTMAlignmentError("strict_basis_oi: basis contains non-finite values")

    if use_open_interest:
        oi = data.get("open_interest")
        if oi is None:
            raise TTMAlignmentError("strict_basis_oi: missing 'open_interest' aligned to bars")
        oa = np.asarray(oi, dtype=np.float64).ravel()
        if oa.size != n:
            raise TTMAlignmentError(
                f"strict_basis_oi: open_interest length {oa.size} != bars length {n}"
            )
        if not np.all(np.isfinite(oa)):
            raise TTMAlignmentError("strict_basis_oi: open_interest contains non-finite values")


def fingerprint_aligned_tail(
    data: Mapping[str, Any],
    *,
    tail: int = 64,
) -> str:
    """SHA256 of last ``tail`` closes | basis | OI for reproducibility logging."""
    bars = data.get("bars") or []
    if not bars:
        return ""
    n = len(bars)
    k = min(tail, n)
    closes = np.asarray(_bars_close_series(bars[-k:]), dtype=np.float64)
    parts = [closes.tobytes()]
    b = data.get("basis")
    if b is not None:
        ba = np.asarray(b, dtype=np.float64).ravel()
        if ba.size >= k:
            parts.append(ba[-k:].tobytes())
    oi = data.get("open_interest")
    if oi is not None:
        oa = np.asarray(oi, dtype=np.float64).ravel()
        if oa.size >= k:
            parts.append(oa[-k:].tobytes())
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:16]


def write_aligned_market_bundle(
    path: str | Path,
    *,
    symbol: str,
    bars: Sequence[Any],
    basis: Sequence[float],
    open_interest: Optional[Sequence[float]],
    profile: str = "",
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Persist per-bar closes, basis, OI (and unix_ts) for validation / offline replay.

    Overwrites the target path atomically (single JSON file).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = len(bars)
    closes = _bars_close_series(bars)
    ts = _bars_unix_ts(bars)
    bb = np.asarray(basis, dtype=np.float64).ravel()
    if bb.size != n:
        raise TTMAlignmentError(
            f"persist bundle: basis length {bb.size} != bars {n}"
        )
    payload: Dict[str, Any] = {
        "schema": "ttm_aligned_market_bundle_v1",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "n_bars": n,
        "profile": profile,
        "closes": [float(x) for x in closes],
        "basis": [float(x) for x in bb],
        "unix_ts": ts,
    }
    if open_interest is not None:
        oo = np.asarray(open_interest, dtype=np.float64).ravel()
        if oo.size != n:
            raise TTMAlignmentError(
                f"persist bundle: open_interest length {oo.size} != bars {n}"
            )
        payload["open_interest"] = [float(x) for x in oo]
    if extra:
        payload["extra"] = dict(extra)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)
