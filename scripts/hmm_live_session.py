#!/usr/bin/env python3
"""Chạy live 15m (HMM hoặc TTM theo ``STRATEGY_ALGO``) **một phiên trong ngày**: poll mỗi 1 phút trong giờ sàn VN.

Khung mặc định (Asia/Ho_Chi_Minh):
  - Phiên sáng: 09:00 – 11:30
  - Nghỉ trưa:   11:30 – 13:00
  - Phiên chiều: 13:00 – 15:00

Chạy một lần duy nhất từ CMD; process thoát sau ~15:00.
Quyết định vẫn theo nến 15m (runner tự chống duplicate bar).

Examples::

    python scripts/hmm_live_session.py
    python scripts/hmm_live_session.py --submit
    python scripts/hmm_live_session.py --symbol VN30F1M --submit --telegram-control
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import StrategyAlgo, get_settings
from src.live.hmm_live_runner import HmmLiveRunner
from src.live.session_15m_loop import Session15mLoopParams, run_15m_session_loop
from src.live.ttm_live_runner import TtmLiveRunner
from src.logger import get_logger, setup_logging
from src.strategy_factory import live_session_logger_name
from src.telegram_control import TelegramControlBot


def _vietnam_tz():
    """Asia/Ho_Chi_Minh, or UTC+7 if IANA data missing (Windows without ``tzdata``)."""
    try:
        return ZoneInfo("Asia/Ho_Chi_Minh")
    except Exception:
        return timezone(timedelta(hours=7))


VN = _vietnam_tz()


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


def _setup_terminal_tee(session_day: date, settings) -> None:
    debug_dir = _ROOT / "data" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ymd = session_day.strftime("%Y%m%d")
    # Cùng quy ước với hmm_live.py: mọi log terminal → hmm_live_YYYYMMDD.log
    log_path = debug_dir / f"hmm_live_{ymd}.log"
    fp = open(log_path, "a", encoding="utf-8")
    # Keep handle alive for process lifetime
    setattr(sys, "_live_session_log_fp", fp)
    setattr(sys, "_live_session_log_path", str(log_path.resolve()))
    sys.stdout = _TeeWriter(sys.stdout, fp)
    sys.stderr = _TeeWriter(sys.stderr, fp)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Live 15m (HMM/TTM) — một phiên trong ngày (9h–15h, nghỉ trưa)",
    )
    p.add_argument("--symbol", default=None, help="Mặc định: HMM_SYMBOL từ .env")
    p.add_argument(
        "--days",
        type=int,
        default=None,
        help="Lookback OHLC (mặc định: HMM_LIVE_DAYS_BACK)",
    )
    p.add_argument("--submit", action="store_true", help="Gửi lệnh khi có tín hiệu (theo PAPER_MODE)")
    p.add_argument("--telegram-control", action="store_true", help="Bot Telegram /pause /resume (một process)")
    p.add_argument(
        "--ignore-strategy-algo",
        action="store_true",
        help="Bỏ cảnh báo STRATEGY_ALGO=MCMC (script này dùng HMM/TTM)",
    )
    p.add_argument(
        "--morning-start",
        default="09:00",
        help="Bắt đầu phiên sáng (HH:MM, VN)",
    )
    p.add_argument(
        "--morning-end",
        default="11:30",
        help="Kết thúc sáng / bắt đầu nghỉ trưa (HH:MM)",
    )
    p.add_argument(
        "--afternoon-start",
        default="13:00",
        help="Bắt đầu phiên chiều (HH:MM)",
    )
    p.add_argument(
        "--session-end",
        default="15:00",
        help="Kết thúc phiên (HH:MM), không chạy tick sau mốc này",
    )
    p.add_argument(
        "--bar-delay",
        type=float,
        default=2.0,
        help="Chờ thêm N giây sau mỗi mốc poll 1 phút trước khi gọi API",
    )
    p.add_argument(
        "--debug-trace",
        action="store_true",
        help="HMM_LIVE_DEBUG_TRACE: log state probs, guards, feature tail, timestamps + data/debug/hmm_live_trace.jsonl",
    )
    p.add_argument(
        "--duplicate-retry-seconds",
        type=float,
        default=60.0,
        help="Khi duplicate_bar: chờ N giây rồi retry (mặc định 60s để tránh spam API)",
    )
    args = p.parse_args()

    if args.debug_trace:
        os.environ["HMM_LIVE_DEBUG_TRACE"] = "true"

    settings = get_settings()
    # Tee stdout/stderr TRƯỚC setup_logging để FileHandler/console ghi vào cùng file debug.
    session_day = datetime.now(VN).date()
    _setup_terminal_tee(session_day, settings)
    setup_logging(log_level=settings.LOG_LEVEL, log_dir=settings.LOG_DIR)
    session_logger = get_logger(live_session_logger_name(settings))
    _log_dir = Path(settings.LOG_DIR)
    if not _log_dir.is_absolute():
        _log_dir = _ROOT / _log_dir
    _trace_jsonl = _ROOT / "data" / "debug" / "hmm_live_trace.jsonl"
    algo = settings.STRATEGY_ALGO.value
    print(f"[BeeTrade] Thư mục log (JSON): {_log_dir.resolve()}", flush=True)
    if settings.STRATEGY_ALGO == StrategyAlgo.HMM:
        print(
            f"[BeeTrade] File trace HMM khi bật HMM_LIVE_DEBUG_TRACE: {_trace_jsonl.resolve()}",
            flush=True,
        )
    elif settings.STRATEGY_ALGO == StrategyAlgo.TTM:
        dbg = getattr(sys, "_live_session_log_path", "")
        print(
            "[BeeTrade] Chiến lược TTM — log phiên (tee) ghi vào: "
            f"{dbg or 'data/debug/hmm_live_YYYYMMDD.log'}",
            flush=True,
        )
    else:
        print(
            f"[BeeTrade] STRATEGY_ALGO={algo} — trace JSONL HMM chỉ dùng khi chạy HMM.",
            flush=True,
        )
    session_logger.info(
        "Đường dẫn log và file trace (theo chiến lược)",
        extra={
            "strategy_algo": algo,
            "log_dir": str(_log_dir.resolve()),
            "hmm_trace_jsonl": str(_trace_jsonl.resolve()),
        },
    )
    symbol = args.symbol or settings.HMM_SYMBOL
    days_back = args.days if args.days is not None else settings.HMM_LIVE_DAYS_BACK

    if settings.STRATEGY_ALGO in (StrategyAlgo.HMM, StrategyAlgo.TTM):
        from src.strategy_factory import get_strategy

        _sig_strat = get_strategy(settings)
        print(f"[BeeTrade] get_strategy() -> {type(_sig_strat).__name__}", flush=True)
    if settings.STRATEGY_ALGO == StrategyAlgo.MCMC and not args.ignore_strategy_algo:
        print(
            "WARNING: STRATEGY_ALGO=MCMC; dùng STRATEGY_ALGO=HMM hoặc TTM, hoặc --ignore-strategy-algo.",
            file=sys.stderr,
        )

    if settings.STRATEGY_ALGO == StrategyAlgo.TTM:
        runner = TtmLiveRunner(settings=settings)
    else:
        runner = HmmLiveRunner(settings=settings)
    if getattr(settings, "OPS_ENABLED", False) and hasattr(runner, "stop_ops"):
        atexit.register(runner.stop_ops)
    if args.telegram_control and not getattr(settings, "OPS_ENABLED", False):
        TelegramControlBot(runner.risk, runner.notifier, settings=settings).start_background()

    params = Session15mLoopParams(
        morning_start=args.morning_start,
        morning_end=args.morning_end,
        afternoon_start=args.afternoon_start,
        session_end=args.session_end,
        bar_delay=args.bar_delay,
        duplicate_retry_seconds=args.duplicate_retry_seconds,
    )
    return run_15m_session_loop(
        settings=settings,
        runner=runner,
        symbol=symbol,
        days_back=days_back,
        submit=args.submit,
        params=params,
    )


if __name__ == "__main__":
    raise SystemExit(main())
