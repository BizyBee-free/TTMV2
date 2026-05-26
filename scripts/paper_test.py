"""Paper test session runner for MCMCDerivativesStrategy.

Orchestrates a full paper trading session:
  1. Load config and initialise all components
  2. Connect to DNSE WebSocket
  3. Subscribe to derivative symbol (quotes + OHLC)
  4. Run strategy for the configured duration or until market close
  5. Force-close any open positions at session end
  6. Print session report and append to reports/TEST_REPORT.md

**Output locations**

- **Terminal copy (full stdout/stderr):** ``data/debug/ttm_live_YYYYMMDD.txt`` (TTM) hoặc
  ``data/debug/paper_live_YYYYMMDD.txt`` (MCMC); append, UTF-8.
- **Structured app log:** ``logs/beetrade.jsonl`` (rotating; sau ``setup_logging``).
- **TTM parallel V1/V2 (JSONL):** ``reports/ttm_parallel_decisions_<symbol>_<stamp>.jsonl`` và ``..._trades_...``.
- **Closes (validation):** ``reports/closes_<symbol>_<stamp>.json`` — mảng close theo ``bar_index``, ghi khi đóng runner (khớp số bar với JSONL decisions).
- **Session summary:** append ``reports/TEST_REPORT.md`` (unless ``--no-report``).

Usage (during market hours 08:45-14:30):
    python scripts/paper_test.py
    python scripts/paper_test.py --symbol VN30F2506
    python scripts/paper_test.py --symbol VN30F2506 --debug-feed    # TTM: chỉ phái sinh
    python scripts/paper_test.py --symbol VN30F2506 --duration 60   # 60 minutes
    python scripts/paper_test.py --dry-run                           # offline mode
    # Windows:  .\\scripts\\run_ttm_paper_parallel.ps1 -Symbol VN30F1M -DebugFeed

Environment:
    PAPER_MODE=true          (should always be true for paper test)
    STRATEGY_ALGO=MCMC       (default MCMC stack) or STRATEGY_ALGO=TTM (HMM_SYMBOL / --symbol)
    MCMC_DERIVATIVE_SYMBOL   (MCMC; overridden by --symbol)
    MCMC_CONFIDENCE_THRESHOLD, MCMC_ROLLING_WINDOW, etc. (from .env)

TTM + song song V1/V2 (mặc định bật):
    STRATEGY_ALGO=TTM
    DNSE_WS_BOARD_ID=G1      (TTM phái sinh; MCMC/ws_test cổ phiếu có thể G1,AL)
    V2 gates: build_ttm_paper_live_config() → quality_research trades; exploratory tier shadow in JSONL gate_diagnostics.
    Mỗi nến: REST bổ sung basis (future−index, HMM_INDEX_SYMBOL) + OI (secdef) như ttm_live — cần API key hợp lệ.
    Ghi JSONL: reports/ttm_parallel_decisions_<symbol>_<stamp>.jsonl và ..._trades_...
    Tắt:  --no-ttm-parallel
    Debug terminal: --debug-feed (in từng nến [TTM parallel] + [feed] ~30s; mặc định vẫn có Session status ~60s)
    OI secdef: --debug-oi (mỗi nến [TTM OI]: HTTP status, mã trade, openInterestQuantity, OI sau align).
    Session status ~60s kèm dnse_open_interest_quantity khi STRATEGY_ALGO=TTM.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    StrategyAlgo,
    get_settings,
    paper_test_prime_vn30f1m_krx,
    resolve_symbol_profile,
    ws_subscribe_symbol_list,
)
from src.data_buffer import DataBuffer
from src.logger import get_logger, setup_logging
from src.market_data import MarketDataManager
from src.order_manager import OrderManager
from src.paper_engine import PaperEngine
from src.position_tracker import PositionTracker
from src.risk_manager import RiskManager
from src.strategies.mcmc_derivatives import MCMCDerivativesStrategy
from src.strategies.ttm.ttm_parallel_runner import (
    ParallelRunner,
    TTMDerivativesParallelStrategy,
    build_ttm_paper_live_config,
)
from src.strategies.ttm.ttm_strategy import TTMDerivativesStrategy

logger = get_logger("paper_test")

# OHLC resolution for derivatives (1-minute candles)
OHLC_RESOLUTION = "1"

_ROOT = Path(__file__).resolve().parent.parent


class _TeeWriter:
    """Mirror terminal output to a debug log file."""

    def __init__(self, *streams):
        self._streams = streams
        self._lock = threading.RLock()

    def write(self, data):
        with self._lock:
            for s in self._streams:
                s.write(data)
        return len(data)

    def flush(self):
        with self._lock:
            for s in self._streams:
                s.flush()


def _setup_terminal_tee(ymd: str, basename: str) -> str | None:
    """Append stdout/stderr to data/debug/<basename>_YYYYMMDD.txt. Call before setup_logging."""
    debug_dir = _ROOT / "data" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    log_path = debug_dir / f"{basename}_{ymd}.txt"
    fp = open(log_path, "a", encoding="utf-8")
    setattr(sys, "_paper_test_log_fp", fp)
    setattr(sys, "_paper_test_log_path", str(log_path.resolve()))
    sys.stdout = _TeeWriter(sys.__stdout__, fp)
    sys.stderr = _TeeWriter(sys.__stderr__, fp)
    return str(log_path.resolve())


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BeeTrade MCMC Paper Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Derivative symbol to trade (default: MCMC_DERIVATIVE_SYMBOL from .env)",
    )
    parser.add_argument(
        "--duration", type=int, default=None,
        help="Max session duration in minutes (default: run until 14:30 force-close)",
    )
    parser.add_argument(
        "--nav-start", type=float, default=0.0,
        help="Starting NAV for daily loss %% calculation (0 = disable)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Exit after component initialisation (no WebSocket connection)",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="Skip appending results to TEST_REPORT.md",
    )
    parser.add_argument(
        "--no-ttm-parallel",
        action="store_true",
        help="TTM only: tắt ghi song song V1+V2 (JSONL so sánh quyết định)",
    )
    parser.add_argument(
        "--no-terminal-log",
        action="store_true",
        help="Không mirror stdout/stderr vào data/debug/*_live_YYYYMMDD.txt",
    )
    parser.add_argument(
        "--debug-feed",
        action="store_true",
        help=(
            "TTM + parallel: in mỗi nến ra terminal [TTM parallel] ...; "
            "thêm heartbeat [feed] ~30s (ws_ohlc, buffer, jsonl_bars, ticks)"
        ),
    )
    parser.add_argument(
        "--debug-ws-pipeline",
        action="store_true",
        help=(
            "Bật DNSE_WS_PIPELINE_DEBUG: RAW_WS, SENDING SUB/AUTH, DECODED, SUB OK, "
            "PING/PONG, MARKET DATA OK, TICK(pipeline_trade)."
        ),
    )
    parser.add_argument(
        "--debug-oi",
        action="store_true",
        help="TTM: in mỗi nến [TTM OI] (HTTP status, trade symbol, openInterestQuantity, OI sau align).",
    )
    parser.add_argument(
        "--ws-assert-market-sec",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sau N giây từ connect, assert market_message_dispatches>0 (RuntimeError nếu không). "
            "Ghi đè DNSE_WS_PIPELINE_ASSERT_MARKET_SEC; 0 qua env để tắt."
        ),
    )
    return parser.parse_args()


# ── session report helpers ────────────────────────────────────────────────────

def _format_report(
    symbol: str,
    started_at: float,
    strategy: MCMCDerivativesStrategy | TTMDerivativesStrategy,
    tracker: PositionTracker,
    paper_engine: PaperEngine,
    risk: RiskManager,
) -> str:
    elapsed = time.time() - started_at
    daily = tracker.daily_pnl
    stats = strategy.extended_stats
    fill_report = paper_engine.get_session_report()
    risk_snap = risk.stats
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if isinstance(strategy, TTMDerivativesStrategy):
        cfg = strategy._ttm.config
        lines = [
            "",
            "---",
            f"## Paper Test Session (TTM) -- {symbol} [{now}]",
            "",
            "### Configuration",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Symbol | {symbol} |",
            f"| STRATEGY_ALGO | TTM |",
            f"| breakout_window | {cfg.get('breakout_window')} |",
            f"| failure_window | {cfg.get('failure_window')} |",
            f"| vol_threshold | {cfg.get('vol_threshold')} |",
            f"| oi_z_threshold | {cfg.get('oi_z_threshold')} |",
            f"| stop_loss_points | {cfg.get('stop_loss_points')} |",
            f"| take_profit_points | {cfg.get('take_profit_points')} |",
            f"| max_bars_in_trade | {cfg.get('max_bars_in_trade')} |",
            f"| use_open_interest | {cfg.get('use_open_interest')} |",
        ]
        pe = getattr(strategy, "_paper_enricher", None)
        oi_snap = getattr(pe, "last_secdef_snapshot", None) if pe is not None else None
        if isinstance(oi_snap, dict) and oi_snap:
            lines.extend(
                [
                    f"| DNSE secdef HTTP status (last) | {oi_snap.get('status', '—')} |",
                    f"| DNSE trade symbol (resolved) | {oi_snap.get('trade_symbol', '—')} |",
                    f"| secdef symbol used (API) | {oi_snap.get('secdef_symbol_used', '—')} |",
                    f"| boardId (secdef query) | {oi_snap.get('board_id', '—')} |",
                    f"| openInterestQuantity (last) | {oi_snap.get('openInterestQuantity', '—')} |",
                    f"| OI source (rest vs websocket) | {oi_snap.get('oi_source', '—')} |",
                ]
            )
        lines.extend(
            [
            "",
            "### Session Results",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Duration | {int(elapsed // 60)}m {int(elapsed % 60)}s |",
            f"| Signals generated | {stats['signals']} |",
            f"| Orders placed | {stats['orders']} |",
            f"| Paper fills | {fill_report.fills} |",
            f"| Realized P&L | {daily.realized:.2f} |",
            f"| Commission | {daily.commission:.2f} |",
            f"| Net P&L | {daily.net:.2f} |",
            f"| Win rate | {daily.win_rate:.1f}% ({daily.win_count}W / {daily.loss_count}L) |",
            f"| Risk halted | {risk_snap.is_halted} |",
            f"| Stoploss triggers | {risk_snap.stoploss_triggers} |",
            "",
            "### Evaluation",
            "- [ ] TTM signals logged with strategy/action/confidence/reason",
            "- [ ] No overlapping entries when flat",
            "- [ ] Risk manager correctly gated orders",
            "",
            ]
        )
        return "\n".join(lines)

    lines = [
        "",
        "---",
        f"## Paper Test Session -- {symbol} [{now}]",
        "",
        "### Configuration",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Symbol | {symbol} |",
        f"| Confidence threshold | {strategy._confidence} |",
        f"| MCMC paths | {strategy._settings.MCMC_NUM_PATHS} |",
        f"| MH iterations | {strategy._settings.MCMC_MH_ITERATIONS} |",
        f"| Rolling window | {strategy._settings.MCMC_ROLLING_WINDOW} sessions |",
        f"| Cooldown ticks | {strategy._cooldown_ticks} |",
        f"| T0 force close | minute {strategy._force_close_minute} ({strategy._force_close_minute // 60:02d}:{strategy._force_close_minute % 60:02d}) |",
        "",
        "### Session Results",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Duration | {int(elapsed // 60)}m {int(elapsed % 60)}s |",
        f"| MCMC computations | {stats['mcmc_calls']} |",
        f"| Avg MCMC latency | {stats['avg_mcmc_ms']:.1f} ms |",
        f"| Markov sessions loaded | {stats['markov_sessions']} |",
        f"| Signals generated | {stats['signals']} |",
        f"| Signals skipped (cooldown) | {stats['signals_skipped_cooldown']} |",
        f"| Signals skipped (risk) | {stats['signals_skipped_risk']} |",
        f"| Orders placed | {stats['orders']} |",
        f"| Paper fills | {fill_report.fills} |",
        f"| Round-trips | {stats['round_trips']} |",
        f"| Realized P&L | {daily.realized:.2f} |",
        f"| Commission | {daily.commission:.2f} |",
        f"| Net P&L | {daily.net:.2f} |",
        f"| Win rate | {daily.win_rate:.1f}% ({daily.win_count}W / {daily.loss_count}L) |",
        f"| Risk halted | {risk_snap.is_halted} |",
        f"| Stoploss triggers | {risk_snap.stoploss_triggers} |",
        f"| T0 force close fired | {stats['t0_closed']} |",
        "",
        "### Evaluation",
        "- [ ] MCMC latency < 100ms per computation",
        "- [ ] No orders placed outside trading hours",
        "- [ ] T0 position properly closed",
        "- [ ] Risk manager correctly gated orders",
        "",
    ]
    return "\n".join(lines)


def _append_to_test_report(content: str) -> None:
    report_path = Path(__file__).parent.parent / "reports" / "TEST_REPORT.md"
    if not report_path.parent.exists():
        report_path.parent.mkdir(parents=True)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Session report appended to {report_path}")


# ── main async runner ─────────────────────────────────────────────────────────

async def run_session(args: argparse.Namespace) -> None:
    settings = get_settings()
    algo = settings.STRATEGY_ALGO
    parallel_runner: ParallelRunner | None = None
    if algo == StrategyAlgo.TTM:
        symbol = args.symbol or settings.HMM_SYMBOL
    elif algo == StrategyAlgo.MCMC:
        symbol = args.symbol or settings.MCMC_DERIVATIVE_SYMBOL
    else:
        logger.error(
            "paper_test requires STRATEGY_ALGO=MCMC or TTM in .env (HMM uses scripts/hmm_live.py)."
        )
        return

    logger.info("=" * 60)
    logger.info(f"BeeTrade Paper Test Session ({algo.value})")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Paper mode: {settings.PAPER_MODE}")
    if algo == StrategyAlgo.MCMC:
        logger.info(f"Confidence threshold: {settings.MCMC_CONFIDENCE_THRESHOLD}")

    if algo == StrategyAlgo.TTM:
        sym_check = str(symbol).strip().upper()
        prof = resolve_symbol_profile(sym_check)
        if str(prof.get("type") or "").lower() != "derivative":
            err = (
                "TTM (V1/V2) chỉ dùng cho phái sinh (VN30F*, mã 41...). "
                f"Mã {sym_check!r} không phải phái sinh (type={prof.get('type')!r})."
            )
            logger.error(err)
            print(f"\n[paper_test] {err}\n", flush=True)
            return
        # paper_test + VN30F1M: resolve KRX (API vs lịch) trước mọi BeeTradeClient / subscribe (reset cờ trong prime).
        if sym_check == "VN30F1M" and not args.dry_run:
            settings = paper_test_prime_vn30f1m_krx(settings)
            _ws = ws_subscribe_symbol_list(sym_check)
            print(f"[paper_test] Subscribe symbols (post-resolve): {_ws}", flush=True)
            logger.info("paper_test VN30 subscribe symbols (post-resolve)", extra={"symbols": _ws})

    if not settings.PAPER_MODE:
        logger.error("PAPER_MODE is not enabled! Set PAPER_MODE=true in .env. Aborting.")
        return

    # ── Component initialisation ──────────────────────────────────────────────
    buffer = DataBuffer(trade_maxlen=200, ohlc_maxlen=200)
    order_mgr = OrderManager(client=None, settings=settings, paper_mode=True)
    tracker = PositionTracker()
    order_mgr.on_fill(tracker.on_fill)

    risk = RiskManager(tracker, settings, nav_start=args.nav_start)
    paper_engine = PaperEngine(order_mgr, tick_size=0.1, slippage_ticks=0)

    if algo == StrategyAlgo.TTM:
        from src.strategy_factory import get_strategy

        _ = get_strategy(settings)
        use_parallel = not args.no_ttm_parallel
        if use_parallel:
            report_dir = Path(__file__).resolve().parent.parent / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            safe_sym = "".join(c if c.isalnum() else "_" for c in symbol)[:32]
            dec_path = report_dir / f"ttm_parallel_decisions_{safe_sym}_{stamp}.jsonl"
            trd_path = report_dir / f"ttm_parallel_trades_{safe_sym}_{stamp}.jsonl"
            ttm_cfg = build_ttm_paper_live_config()
            from src.strategies.ttm.empirical.adaptive_engine import EmpiricalAlphaEngine

            _emp = (
                EmpiricalAlphaEngine(ttm_cfg)
                if bool(ttm_cfg.get("ttm_v2_empirical_alpha_enabled", False))
                else None
            )
            parallel_runner = ParallelRunner(
                ttm_cfg,
                decision_log_path=str(dec_path),
                trade_log_path=str(trd_path),
                debug_terminal=bool(args.debug_feed),
                empirical_engine=_emp,
            )
            strategy = TTMDerivativesParallelStrategy(
                name="ttm_paper_parallel",
                symbols=[symbol],
                settings=settings,
                ttm_config=ttm_cfg,
                parallel_runner=parallel_runner,
            )
            logger.info(
                "TTM parallel V1+V2 logging enabled",
                extra={
                    "ttm_config_profile": ttm_cfg.get("ttm_config_profile"),
                    "ttm_v2_gate_mode": ttm_cfg.get("ttm_v2_gate_mode"),
                    "decisions_jsonl": str(dec_path),
                    "trades_jsonl": str(trd_path),
                    "jsonl_validation": (
                        "Mỗi dòng event_type=decision cần một nến OHLC (WS ohlc hoặc synth_trade từ khớp). "
                        "Nếu ws_ohlc_received=0 và ohlc_synth_bars=0 lâu, file gần như chỉ có summary lúc thoát — "
                        "xem Session status: ttm_parallel_jsonl_validation."
                    ),
                },
            )
            print(
                f"\n[TTM] Song song V1+V2 -> JSONL:\n  {dec_path}\n  {trd_path}\n"
                f"  profile={ttm_cfg.get('ttm_config_profile')} "
                f"gate_mode={ttm_cfg.get('ttm_v2_gate_mode')} "
                f"(exploratory tier shadow in gate_diagnostics only)\n"
                f"  (khi thoát phiên: closes_<symbol>_<stamp>.json cùng thư mục — dùng --closes cho ttm_validation)\n"
            )
            if args.debug_feed:
                print(
                    "[TTM] --debug-feed: mỗi nến sẽ in một dòng [TTM parallel] ... ra terminal.\n"
                )
        else:
            strategy = TTMDerivativesStrategy(
                name="ttm_paper",
                symbols=[symbol],
                settings=settings,
            )
    else:
        strategy = MCMCDerivativesStrategy(
            name="mcmc_t0",
            symbols=[symbol],
            settings=settings,
        )
    strategy.set_risk_manager(risk)

    if algo == StrategyAlgo.TTM:
        try:
            from src.backtest.data_fetcher import DataFetcher
            from src.dnse_client import BeeTradeClient
            from src.strategies.ttm.paper_basis_oi import TTMPaperEnricher

            _pe = TTMPaperEnricher(
                settings=settings,
                client=BeeTradeClient(settings=settings),
                fetcher=DataFetcher(settings=settings),
                derivative_symbol=str(symbol).strip().upper(),
                debug_oi=bool(getattr(args, "debug_oi", False)),
            )
            strategy.set_paper_enricher(_pe)
            logger.info(
                "TTM paper: basis+OI enricher (REST, cùng nguồn ttm_live)",
                extra={"symbol": str(symbol).strip().upper()},
            )
            rp = resolve_symbol_profile(str(symbol).strip().upper())
            print(
                f"[paper_test] KRX trade_symbol (sau resolve): {rp.get('trade_symbol')} | "
                f"data_symbol={rp.get('data_symbol')}",
                flush=True,
            )
            logger.info(
                "TTM paper: SYMBOL_MAP VN30 derivative mapping",
                extra={"trade_symbol": rp.get("trade_symbol"), "data_symbol": rp.get("data_symbol")},
            )
        except Exception as e:
            logger.warning("TTM paper: không bật enricher basis/OI", extra={"error": str(e)})

    mdm = MarketDataManager(settings=settings, buffer=buffer)
    mdm.on_quote(strategy.handle_quote)
    mdm.on_ohlc(strategy.handle_ohlc)
    mdm.on_quote(paper_engine.on_quote)

    risk.on_halt(lambda reason: logger.error(f"HALT: {reason}"))

    if args.dry_run:
        if parallel_runner is not None:
            try:
                summ = parallel_runner.close()
                print("\n--- TTM parallel V1/V2 (dry-run) ---")
                print(json.dumps(summ, ensure_ascii=False, indent=2))
            except Exception as ex:
                logger.warning(f"TTM parallel close (dry-run): {ex}")
        logger.info("Dry run -- components initialised OK. Exiting.")
        return

    # ── Connect ───────────────────────────────────────────────────────────────
    started_at = time.time()
    try:
        await mdm.connect()
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        if parallel_runner is not None:
            try:
                parallel_runner.close()
            except Exception:
                pass
        return

    raw_boards = (settings.DNSE_WS_BOARD_ID or "G1").strip() or "G1"
    ws_boards = [b.strip() for b in raw_boards.split(",") if b.strip()] or ["G1"]
    sym_u = str(symbol).strip().upper()
    # Phái sinh: subscribe cả data_symbol + trade_symbol (KRX 41…) — gateway thường stream theo mã 41…
    ws_syms = ws_subscribe_symbol_list(sym_u)
    if len(ws_syms) > 1:
        logger.info("WS subscribe symbol list (derivative)", extra={"symbols": ws_syms})
    # openapi-sdk: khi board_id=None thì subscribe nhiều board; ở đây cho phép G1,AL trong .env
    # sec_def trước: một số gateway ổn định hơn; phái sinh ưu tiên mã KRX 41… trước VN30F*
    ws_syms_secdef = list(reversed(ws_syms)) if len(ws_syms) > 1 else list(ws_syms)
    for ws_board in ws_boards:
        await mdm.subscribe_sec_def(ws_syms_secdef, board_id=ws_board)
    for ws_board in ws_boards:
        await mdm.subscribe_quotes(ws_syms, board_id=ws_board)
        # tick.* đẩy trade (T=t) và quote (T=q) trên cùng board
        await mdm.subscribe_trades(ws_syms, board_id=ws_board)
    await mdm.subscribe_ohlc(ws_syms, resolution=OHLC_RESOLUTION, board_id=ws_boards[0])

    strategy.start(order_mgr, tracker, buffer)
    paper_engine.start()

    logger.info(
        f"Session started for {sym_u}. Press Ctrl+C to stop.",
        extra={
            "ws_board_ids": ws_boards,
            "heartbeat_s": 60,
            "hint": "Nếu parallel_jsonl_bars=0 nhưng ws_ohlc_received>0, kiểm tra symbol mismatch",
        },
    )
    print(
        f"\n[paper_test] Symbol={sym_u} boardIds={','.join(ws_boards)} | "
        f"Mỗi ~60s log 'Session status' kèm ws_ohlc_received, parallel_jsonl_bars, ticks, ohlc_in_buffer.\n"
        + (
            "[paper_test] --debug-feed: thêm in từng nến + [feed] ~30s.\n"
            if args.debug_feed
            else ""
        ),
        flush=True,
    )

    # ── Session loop ──────────────────────────────────────────────────────────
    stop_event = asyncio.Event()

    def _on_signal(sig, frame):
        # Không gọi logger ở đây: dễ reentrant stdout (TeeWriter) khi WS thread đang print.
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)

    deadline = None
    if args.duration:
        deadline = time.time() + args.duration * 60

    last_status_log = 0.0
    last_feed_print = 0.0
    warned_parallel_no_ohlc = False

    try:
        while not stop_event.is_set():
            await asyncio.sleep(5)

            if deadline and time.time() > deadline:
                logger.info(f"Duration limit ({args.duration}m) reached. Stopping.")
                break

            now = time.time()
            elapsed = int(now - started_at)

            ohlc_buf_n = len(await buffer.get_ohlc_history(sym_u, n=10_000) or [])
            par_bars = (
                parallel_runner.decision_bar_count if parallel_runner is not None else 0
            )

            if args.debug_feed and (now - last_feed_print) >= 30.0:
                last_feed_print = now
                print(
                    f"[feed] t+{elapsed}s raw_recv={mdm.ws_raw_bytes_received} "
                    f"decode_err={mdm.ws_decode_errors} "
                    f"ws_frames={mdm.ws_raw_frames} "
                    f"ws_market_cb={mdm.ws_market_dispatches} "
                    f"mdm_msg={mdm.msg_count} ws_ohlc={mdm.ohlc_rx_count} "
                    f"ohlc_synth={getattr(mdm, 'ohlc_synth_count', 0)} "
                    f"ohlc_buf={ohlc_buf_n} ticks={strategy.stats['ticks']} "
                    f"jsonl_bars={par_bars}",
                    flush=True,
                )

            if (now - last_status_log) >= 60.0:
                last_status_log = now
                snap = risk.stats
                daily = tracker.daily_pnl
                stats = strategy.extended_stats
                status_extra = {
                    "elapsed_s": elapsed,
                    "mcmc_calls": stats["mcmc_calls"],
                    "avg_mcmc_ms": stats["avg_mcmc_ms"],
                    "signals": stats["signals"],
                    "orders": stats["orders"],
                    "fills": paper_engine.get_session_report().fills,
                    "net_pnl": round(daily.net, 2),
                    "halted": snap.is_halted,
                    "strategy_ticks": stats["ticks"],
                    "ws_messages": mdm.msg_count,
                    "ws_raw_bytes_received": mdm.ws_raw_bytes_received,
                    "ws_decode_errors": mdm.ws_decode_errors,
                    "ws_raw_frames": mdm.ws_raw_frames,
                    "ws_market_dispatches": mdm.ws_market_dispatches,
                    "ws_ohlc_received": mdm.ohlc_rx_count,
                    "ws_sec_def_received": getattr(mdm, "sec_def_rx_count", 0),
                    "ohlc_synth_bars": getattr(mdm, "ohlc_synth_count", 0),
                    "ohlc_in_buffer": ohlc_buf_n,
                    "parallel_jsonl_bars": par_bars,
                }
                if (
                    mdm.ws_decode_errors == 0
                    and mdm.ws_market_dispatches == 0
                    and mdm.ws_raw_frames >= 2
                    and elapsed >= 60
                ):
                    status_extra["feed_diagnostic_hint"] = (
                        "WS: subscribe đã ack, decode OK, nhưng chưa có dispatch market. "
                        "Ngoài giờ giao dịch HOSE thường không có tick/OHLC live — chạy lại trong phiên. "
                        "Trong giờ mà vẫn 0: thử DNSE_WS_BOARD_ID=G1,AL (subscribe cả hai board như openapi-sdk)."
                    )
                if isinstance(strategy, TTMDerivativesStrategy):
                    _pe = getattr(strategy, "_paper_enricher", None)
                    _snap = getattr(_pe, "last_secdef_snapshot", None) if _pe is not None else None
                    if isinstance(_snap, dict):
                        status_extra["dnse_secdef_status"] = _snap.get("status")
                        status_extra["dnse_open_interest_quantity"] = _snap.get(
                            "openInterestQuantity"
                        )
                        status_extra["dnse_secdef_trade_symbol"] = _snap.get("trade_symbol")
                if parallel_runner is not None and elapsed >= 60:
                    ws_o = int(mdm.ohlc_rx_count)
                    syn = int(getattr(mdm, "ohlc_synth_count", 0) or 0)
                    if par_bars == 0 and ws_o == 0 and syn == 0:
                        status_extra["ttm_parallel_jsonl_validation"] = (
                            "NO_OHLC_BARS: chưa có nến từ WS (ws_ohlc_received=0) và chưa có "
                            "synth_trade (ohlc_synth_bars=0) — parallel JSONL không có dòng "
                            "event_type=decision (chỉ summary khi đóng). Nguyên nhân thường gặp: "
                            "ngoài giờ HOSE, feed OHLC tắt, hoặc tick thiếu giá (PARSE_WARN) nên không tổng hợp nến."
                        )
                        if not warned_parallel_no_ohlc:
                            warned_parallel_no_ohlc = True
                            logger.warning(
                                "TTM parallel JSONL: chưa có nến OHLC — decisions jsonl sẽ trống (validation)",
                                extra={
                                    "decisions_jsonl": getattr(
                                        parallel_runner, "decision_log_path", None
                                    ),
                                    "ws_ohlc_received": ws_o,
                                    "ohlc_synth_bars": syn,
                                    "ticks": stats["ticks"],
                                },
                            )
                logger.info("Session status", extra=status_extra)

        if stop_event.is_set():
            logger.info(
                "Session loop exit: stop requested (signal); parallel JSONL đã flush mỗi nến nếu có OHLC"
            )

    except Exception as e:
        logger.error(f"Session error: {e}", exc_info=True)
    finally:
        if parallel_runner is not None:
            try:
                summ = parallel_runner.close()
                logger.info("TTM parallel session summary", extra=summ)
                print("\n--- TTM parallel V1/V2 (sim) ---")
                print(json.dumps(summ, ensure_ascii=False, indent=2))
                cep = summ.get("closes_export_path")
                if cep:
                    print(f"\n[TTM] Closes export (ttm_validation --closes): {cep}\n", flush=True)
            except Exception as e:
                logger.warning(f"TTM parallel summary failed: {e}")
        # ── Force close ───────────────────────────────────────────────────────
        logger.info("Force closing any open positions...")
        pos = tracker.get_position(sym_u)
        if not pos.is_flat:
            quote = await mdm.get_latest_quote(sym_u)
            if quote:
                mid = ((quote.best_bid or 0) + (quote.best_ask or 0)) / 2
                paper_engine.on_session_close(mid, sym_u)

        strategy.stop()
        paper_engine.stop()
        await mdm.disconnect()

    # ── Print session report ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)

    report = _format_report(symbol, started_at, strategy, tracker, paper_engine, risk)
    print(report)

    if not args.no_report:
        _append_to_test_report(report)
        print("Report appended to reports/TEST_REPORT.md")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if getattr(args, "debug_ws_pipeline", False):
        os.environ["DNSE_WS_PIPELINE_DEBUG"] = "true"
    if getattr(args, "ws_assert_market_sec", None) is not None:
        os.environ["DNSE_WS_PIPELINE_ASSERT_MARKET_SEC"] = str(args.ws_assert_market_sec)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    settings = get_settings()
    ymd = datetime.now().strftime("%Y%m%d")
    tee_basename = "ttm_live" if settings.STRATEGY_ALGO == StrategyAlgo.TTM else "paper_live"
    tee_path: str | None = None
    if not args.no_terminal_log:
        tee_path = _setup_terminal_tee(ymd, tee_basename)

    setup_logging(
        log_level=settings.LOG_LEVEL,
        log_dir=settings.LOG_DIR,
    )

    if tee_path:
        print(
            f"[paper_test] Terminal log (tee): {tee_path}",
            flush=True,
        )
        logger.info("Terminal tee file", extra={"path": tee_path})

    try:
        asyncio.run(run_session(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
