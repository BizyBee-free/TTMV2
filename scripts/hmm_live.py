#!/usr/bin/env python3
"""Run 15m live: one shot **or** full session until afternoon close (default 15:00 VN).

Uses ``STRATEGY_ALGO`` from ``.env``: ``HMM`` (regime HMM) or ``TTM`` (trapped trader).

**Session loop (poll ~1 phút, dừng sau phiên chiều 15:00)** — mặc định khi ``--symbol VN30F1M``;
hoặc bật rõ bằng ``--session`` cho mọi symbol. Một lần duy nhất: ``--once``.

Examples::

    # Full phiên VN (VN30F1M → loop mặc định)
    python scripts/hmm_live.py --symbol VN30F1M

    # Cùng một lần chạy như trên, nhưng chỉ một tick rồi thoát
    python scripts/hmm_live.py --symbol VN30F1M --once

    # Session cho symbol khác (ví dụ từ .env)
    python scripts/hmm_live.py --session

    # Paper: submit simulated order (trong phiên)
    python scripts/hmm_live.py --symbol VN30F1M --submit

    # With Telegram control bot (same process)
    python scripts/hmm_live.py --symbol VN30F1M --telegram-control
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import StrategyAlgo, get_settings
from src.live.hmm_live_runner import HmmLiveRunner
from src.live.session_15m_loop import Session15mLoopParams, run_15m_session_loop
from src.live.ttm_live_runner import TtmLiveRunner
from src.logger import get_logger, setup_logging
from src.strategy_factory import live_15m_cli_prefix, live_15m_logger_name
from src.telegram_control import TelegramControlBot


class _TeeWriter:
    """Mirror terminal output to a debug log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _setup_terminal_tee(settings) -> None:
    debug_dir = _ROOT / "data" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ymd = datetime.now().strftime("%Y%m%d")
    # Mọi stdout/stderr của script này (HMM hoặc TTM) → cùng một file theo ngày.
    log_path = debug_dir / f"hmm_live_{ymd}.log"
    fp = open(log_path, "a", encoding="utf-8")
    setattr(sys, "_live_15m_log_fp", fp)
    setattr(sys, "_live_15m_log_path", str(log_path.resolve()))
    sys.stdout = _TeeWriter(sys.stdout, fp)
    sys.stderr = _TeeWriter(sys.stderr, fp)


def main() -> int:
    p = argparse.ArgumentParser(
        description="15m live: one shot or full VN session until 15:00 (HMM or TTM)",
    )
    p.add_argument(
        "--symbol",
        default=None,
        help="OHLC symbol (default: HMM_SYMBOL from .env)",
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Chỉ chạy một tick run_once rồi thoát (tắt vòng phiên mặc định của VN30F1M)",
    )
    p.add_argument(
        "--session",
        action="store_true",
        help="Chạy cả phiên (9h–15h VN, nghỉ trưa) cho mọi symbol; không cần VN30F1M",
    )
    p.add_argument(
        "--morning-start",
        default="09:00",
        help="[--session / phiên VN] Bắt đầu phiên sáng (HH:MM)",
    )
    p.add_argument(
        "--morning-end",
        default="11:30",
        help="[--session] Kết thúc sáng / nghỉ trưa (HH:MM)",
    )
    p.add_argument(
        "--afternoon-start",
        default="13:00",
        help="[--session] Bắt đầu phiên chiều (HH:MM)",
    )
    p.add_argument(
        "--session-end",
        default="15:00",
        help="[--session] Kết thúc phiên (HH:MM), không tick sau mốc này",
    )
    p.add_argument(
        "--bar-delay",
        type=float,
        default=2.0,
        help="[--session] Chờ N giây sau mỗi mốc poll trước khi gọi API",
    )
    p.add_argument(
        "--duplicate-retry-seconds",
        type=float,
        default=60.0,
        help="[--session] Khi duplicate_bar: chờ N giây rồi retry",
    )
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Lookback calendar days for OHLC fetch (default: HMM_LIVE_DAYS_BACK from .env)",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Submit order when signal non-flat (respects PAPER_MODE)",
    )
    p.add_argument(
        "--telegram-control",
        action="store_true",
        help="Background Telegram /pause /resume /status (requires TELEGRAM_* env)",
    )
    p.add_argument(
        "--ignore-strategy-algo",
        action="store_true",
        help="Do not warn when STRATEGY_ALGO is not HMM",
    )
    p.add_argument(
        "--debug-trace",
        action="store_true",
        help="Set HMM_LIVE_DEBUG_TRACE for this run (state probs, feature tail, JSONL)",
    )
    args = p.parse_args()

    if args.debug_trace:
        os.environ["HMM_LIVE_DEBUG_TRACE"] = "true"

    settings = get_settings()
    _setup_terminal_tee(settings)
    setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)
    live_log = get_logger(live_15m_logger_name(settings))
    _log_dir = Path(settings.LOG_DIR)
    if not _log_dir.is_absolute():
        _log_dir = _ROOT / _log_dir
    _trace_jsonl = _ROOT / "data" / "debug" / "hmm_live_trace.jsonl"
    algo = settings.STRATEGY_ALGO.value
    px = live_15m_cli_prefix(settings)
    print(f"[BeeTrade] Thư mục log (JSON): {_log_dir.resolve()}", flush=True)
    if settings.STRATEGY_ALGO == StrategyAlgo.HMM:
        print(
            f"[BeeTrade] File trace HMM khi bật HMM_LIVE_DEBUG_TRACE: {_trace_jsonl.resolve()}",
            flush=True,
        )
    elif settings.STRATEGY_ALGO == StrategyAlgo.TTM:
        dbg = getattr(sys, "_live_15m_log_path", "")
        print(
            "[BeeTrade] Chiến lược TTM — không dùng file trace JSONL của HMM; "
            f"log terminal (tee) ghi vào: {dbg or 'data/debug/hmm_live_YYYYMMDD.log'}",
            flush=True,
        )
    else:
        print(
            f"[BeeTrade] STRATEGY_ALGO={algo} — trace JSONL HMM chỉ dùng khi chạy HMM.",
            flush=True,
        )
    symbol = args.symbol or settings.HMM_SYMBOL
    use_session = args.session or (symbol.upper() == "VN30F1M" and not args.once)
    live_log.info(
        "15m live start",
        extra={
            "strategy_algo": algo,
            "log_dir": str(_log_dir.resolve()),
            "symbol": symbol,
            "session_loop": use_session,
        },
    )
    if settings.STRATEGY_ALGO in (StrategyAlgo.HMM, StrategyAlgo.TTM):
        from src.strategy_factory import get_strategy

        _sig_strat = get_strategy(settings)
        print(f"{px} get_strategy() -> {type(_sig_strat).__name__}", flush=True)
    if settings.STRATEGY_ALGO == StrategyAlgo.MCMC and not args.ignore_strategy_algo:
        print(
            "WARNING: STRATEGY_ALGO=MCMC; this script is for HMM/TTM 15m live. "
            "Set STRATEGY_ALGO=HMM or TTM, or pass --ignore-strategy-algo.",
            file=sys.stderr,
        )
    if settings.STRATEGY_ALGO == StrategyAlgo.TTM:
        runner = TtmLiveRunner(settings=settings)
    else:
        runner = HmmLiveRunner(settings=settings)

    if args.telegram_control:
        ctrl = TelegramControlBot(runner.risk, runner.notifier, settings=settings)
        ctrl.start_background()

    days_back = args.days if args.days is not None else settings.HMM_LIVE_DAYS_BACK

    if use_session:
        params = Session15mLoopParams(
            morning_start=args.morning_start,
            morning_end=args.morning_end,
            afternoon_start=args.afternoon_start,
            session_end=args.session_end,
            bar_delay=args.bar_delay,
            duplicate_retry_seconds=args.duplicate_retry_seconds,
        )
        print(
            f"{px} Chế độ phiên liên tục (dừng sau {args.session_end} VN) — symbol={symbol}",
            flush=True,
        )
        live_log.info(
            "15m live session mode (hmm_live.py)",
            extra={
                "strategy_algo": algo,
                "symbol": symbol,
                "session_end": args.session_end,
            },
        )
        return run_15m_session_loop(
            settings=settings,
            runner=runner,
            symbol=symbol,
            days_back=days_back,
            submit=args.submit,
            params=params,
        )

    result = runner.run_once(symbol=symbol, days_back=days_back, submit_order=args.submit)
    print(
        f"ok={result.ok} detail={result.detail} root_cause={result.root_cause} "
        f"direction={result.direction} label={result.state_label} order={result.order_id}"
    )
    ok_details = (
        "duplicate_bar",
        "warmup",
        "no_submit",
        "no_signal",
        "position_open_skip_entry",
    )
    return 0 if result.ok or result.detail in ok_details else 1


if __name__ == "__main__":
    raise SystemExit(main())
