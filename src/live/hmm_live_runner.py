"""Single-step HMM live evaluation: fetch OHLC, signal, optional paper/live order."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.api_health import ApiHealthMonitor
from src.balance_utils import validate_balance_response
from src.backtest.data_fetcher import (
    OhlcBar,
    RESOLUTION_15MIN,
    DataFetcher,
    resolve_data_symbol,
    resolve_dnse_symbol,
)
from src.backtest.hmm_replay import HMMBacktestConfig, HMMBarReplay
from src.config import OrderSide, OrderStatus, OrderType, Settings, get_settings, resolve_symbol_profile
from src.dnse_client import BeeTradeClient
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm.feature_engineer import HMMConfig
from src.hmm.oi_data import (
    OICache,
    build_open_interest_series_for_live,
    parse_open_interest_from_secdef_payload,
)
from src.hmm.hmm_signal import FLAT, LONG, SHORT
from src.logger import get_logger
from src.order_manager import ManagedOrder, OrderManager, OrderRequest
from src.position_tracker import PositionTracker
from src.reconciliation import reconcile_after_fill
from src.risk_manager import RiskManager
from src.telegram_notifier import TelegramNotifier
from src.vn_time import unix_ts_to_vn_str, vn_calendar_today_yyyymmdd

logger = get_logger("hmm_live_runner")


@dataclass
class LiveStepResult:
    ok: bool
    detail: str
    direction: Optional[int] = None
    state_label: Optional[str] = None
    order_id: Optional[str] = None
    root_cause: Optional[str] = None


def _default_date_range(days_back: int) -> Tuple[str, str]:
    """Dùng **ngày lịch Việt Nam** cho ``to_date`` để khớp phiên VN (tránh lệch ngày quanh nửa đêm UTC)."""
    end = vn_calendar_today_yyyymmdd()
    end_d = datetime.strptime(end, "%Y%m%d").date()
    start_d = end_d - timedelta(days=days_back)
    return start_d.strftime("%Y%m%d"), end


def _floor_15m_unix(ts_utc: int) -> int:
    dt = datetime.fromtimestamp(int(ts_utc), tz=timezone.utc)
    dt = dt.replace(minute=(dt.minute // 15) * 15, second=0, microsecond=0)
    return int(dt.timestamp())


def fetch_latest_index_price(client: BeeTradeClient, index_symbol: str) -> Optional[float]:
    """Latest trade price for index symbol (e.g. VN30) — same source as future RT, for basis alignment."""
    try:
        sym, _ = resolve_dnse_symbol(index_symbol)
        r = client.get_latest_trade(sym)
        if r.get("status") == 200:
            px, _ = _parse_latest_trade(r.get("data"))
            return px
    except Exception:
        pass
    return None


def _parse_latest_trade(data: object) -> Tuple[Optional[float], Optional[int]]:
    """Parse DNSE get_latest_trade payload into (price, unix_ts_utc)."""
    if not isinstance(data, dict):
        return None, None
    trades = data.get("trades")
    if not isinstance(trades, list) or not trades:
        return None, None
    t0 = trades[0] if isinstance(trades[0], dict) else None
    if not t0:
        return None, None
    p = t0.get("matchPrice")
    tstr = t0.get("time")
    try:
        price = float(p)
    except Exception:
        return None, None
    if not isinstance(tstr, str) or not tstr.strip():
        return price, None
    try:
        # DNSE returns UTC-like wall time in this endpoint for derivative board.
        dt = datetime.strptime(tstr.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return price, int(dt.timestamp())
    except Exception:
        return price, None


def _inject_realtime_15m_bar(
    bars: list[OhlcBar],
    index_closes: Optional[list[float]],
    rt_price: float,
    rt_ts_utc: int,
    rt_index_price: Optional[float] = None,
) -> Tuple[list[OhlcBar], Optional[list[float]], str, bool]:
    """
    Update/append latest 15m bar from realtime future trade.

    When ``index_closes`` is aligned (same length as ``bars``) and basis is used:
    - **Merge** into current slot: optional ``rt_index_price`` updates the last index close (same ts).
    - **Append** new 15m slot: prefer ``rt_index_price``; if missing, **reuse last index close**
      so the future bar is still appended (avoids stuck on one bar / no breakouts).

    Returns:
        ``(bars, index_closes, mode, skip_signal)``.
    """
    if not bars:
        return bars, index_closes, "bars_empty_skip", False
    slot_ts = _floor_15m_unix(rt_ts_utc)
    if slot_ts < bars[-1].unix_ts:
        return bars, index_closes, "realtime_older_than_last_bar_skip", False

    aligned_basis = (
        index_closes is not None
        and len(index_closes) == len(bars)
    )

    if slot_ts == bars[-1].unix_ts:
        b = bars[-1]
        b.high = max(float(b.high), float(rt_price))
        b.low = min(float(b.low), float(rt_price))
        b.close = float(rt_price)
        if aligned_basis and rt_index_price is not None:
            index_closes = list(index_closes)
            index_closes[-1] = float(rt_index_price)
        return bars, index_closes, "realtime_merged_into_current_slot", False

    # New 15m slot would extend future series.
    if index_closes is not None and len(index_closes) != len(bars):
        return bars, index_closes, "index_bars_mismatch_skip_inject", False

    if index_closes is not None and len(index_closes) == len(bars):
        prev_close = float(bars[-1].close)
        new_bar = OhlcBar(
            symbol=bars[-1].symbol,
            time=datetime.fromtimestamp(slot_ts, tz=timezone.utc).strftime("%Y%m%d%H"),
            open=prev_close,
            high=max(prev_close, float(rt_price)),
            low=min(prev_close, float(rt_price)),
            close=float(rt_price),
            volume=0.0,
            unix_ts=slot_ts,
        )
        if rt_index_price is None:
            last_index = float(index_closes[-1]) if index_closes else None
            bars = [*bars, new_bar]
            if last_index is not None:
                index_closes = [*index_closes, last_index]
            print(
                "[TTM] Append bar with index fallback (no realtime index)",
                flush=True,
            )
            return bars, index_closes, "realtime_appended_with_index_fallback", False
        bars = [*bars, new_bar]
        index_closes = [*index_closes, float(rt_index_price)]
        return bars, index_closes, "realtime_appended_new_slot", False

    prev_close = float(bars[-1].close)
    new_bar = OhlcBar(
        symbol=bars[-1].symbol,
        time=datetime.fromtimestamp(slot_ts, tz=timezone.utc).strftime("%Y%m%d%H"),
        open=prev_close,
        high=max(prev_close, float(rt_price)),
        low=min(prev_close, float(rt_price)),
        close=float(rt_price),
        volume=0.0,
        unix_ts=slot_ts,
    )
    bars = [*bars, new_bar]
    return bars, index_closes, "realtime_appended_new_slot", False


def _validate_bars(bars: list) -> Tuple[bool, str]:
    if len(bars) < 2:
        return False, "insufficient_bars"
    for b in bars[-5:]:
        if b.close <= 0 or b.high < b.low:
            return False, "invalid_ohlc"
    return True, "ok"


def _derivative_order_symbol(friendly: str) -> str:
    """DNSE derivative order: always use trade_symbol when available."""
    p = resolve_symbol_profile(friendly)
    ts = p.get("trade_symbol", friendly)
    if p.get("type") == "derivative" and len(ts) == 9 and ts.isalnum():
        return ts
    return friendly


def _root_cause_from_detail(detail: str) -> str:
    """Classify step outcome for easier post-session RCA."""
    if detail in {
        "ok",
        "warmup",
        "model_not_fitted",
        "no_features",
        "no_submit",
        "state_persistence_guard",
        "ttm_derivatives_only",
    }:
        return "strategy_or_model"
    if detail in {"insufficient_bars", "invalid_ohlc", "empty_bars", "basis_align_failed", "stale_bar_guard"}:
        return "market_data"
    if detail in {"fetch_failed", "balance_fetch_failed"} or detail.startswith("halted:"):
        return "system_or_api"
    if detail in {"duplicate_bar"}:
        return "idempotency_guard"
    if detail in {"trading_disabled_external"}:
        return "ops_control"
    if detail.startswith("API_") or detail in {
        "RATE_LIMIT",
        "RISK_MANAGER_HALTED",
        "BELOW_MIN_BALANCE",
        "DAILY_LOSS_HALT",
        "STOPLOSS_TRIGGERED",
        "POSITION_SIZE_EXCEEDED",
        "TRADING_DISABLED",
    }:
        return "risk_or_order_reject"
    return "unknown"


class HmmLiveRunner:
    """Orchestrates HMM signal on latest 15m bars and optional order submission."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[BeeTradeClient] = None,
        fetcher: Optional[DataFetcher] = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or BeeTradeClient(self._settings)
        self._fetcher = fetcher or DataFetcher(settings=self._settings)
        self._state_path = state_path or (
            Path(__file__).resolve().parent.parent.parent / "data" / "hmm_live_state.json"
        )
        self._tracker = PositionTracker()
        self._risk = RiskManager(self._tracker, settings=self._settings)
        self._orders = OrderManager(self._client, settings=self._settings)
        self._notifier = TelegramNotifier(settings=self._settings)
        self._api_health = ApiHealthMonitor(threshold=self._settings.DNSE_API_FAIL_HALT_THRESHOLD)
        self._balance_before: Optional[float] = None
        self._signal_streak_label: str = ""
        self._signal_streak_count: int = 0

        self._risk.on_halt(self._on_halt)
        self._api_health.on_halt(self._risk.halt)

        self._orders.on_fill(self._on_fill)

    @property
    def risk(self) -> RiskManager:
        return self._risk

    @property
    def notifier(self) -> TelegramNotifier:
        return self._notifier

    def _on_halt(self, reason: str) -> None:
        self._notifier.send_alert("CRITICAL", "HALT", reason)

    def _on_fill(self, order: ManagedOrder) -> None:
        if not order.is_terminal:
            return
        self._notifier.send_message(
            f"Fill: {order.symbol} {order.side.value} qty={order.filled_qty} "
            f"avg={order.avg_fill_price:.2f} status={order.status.value}"
        )
        try:
            reconcile_after_fill(
                order,
                self._balance_before,
                self._client.get_balances,
                settings=self._settings,
            )
        except Exception as e:
            logger.error("reconcile_after_fill error", extra={"error": str(e)})

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path.exists():
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_state(self, last_bar_ts: int) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump({"last_bar_unix_ts": last_bar_ts}, f)

    def fetch_account_balance(self) -> Tuple[Optional[float], bool]:
        """Return (balance, success). On failure, may increment API health."""
        res = self._client.get_balances()
        ok, bal, reason = validate_balance_response(res)
        if not ok:
            logger.error(
                "[Live] Lỗi đọc số dư DNSE",
                extra={"status": res.get("status"), "reason": reason, "data": res.get("data")},
            )
            self._api_health.record_failure(reason)
            return None, False
        logger.info(
            "[Live] Số dư dùng cho risk check",
            extra={"balance_vnd": bal, "balance_source": reason},
        )
        print(f"[Live] Số dư kiểm tra lệnh: {float(bal):.0f} VND (nguồn={reason})", flush=True)
        self._api_health.record_success()
        return bal, True

    def run_once(
        self,
        symbol: str,
        days_back: int = 14,
        submit_order: bool = False,
    ) -> LiveStepResult:
        """Fetch OHLC 15m, compute HMM signal for last bar; optionally submit order.

        When `submit_order` is True and not PAPER_MODE, places a real order (requires
        trading token). Idempotent per bar: skips if same bar already processed.
        """
        if not self._risk.external_trading_enabled:
            return LiveStepResult(
                ok=False,
                detail="trading_disabled_external",
                root_cause=_root_cause_from_detail("trading_disabled_external"),
            )

        if self._risk.is_halted:
            d = f"halted:{self._risk.halt_reason}"
            return LiveStepResult(ok=False, detail=d, root_cause=_root_cause_from_detail(d))

        fut_data, fut_type = resolve_data_symbol(symbol)
        fut_trade, _ = resolve_dnse_symbol(symbol)
        idx_data, idx_type = resolve_data_symbol(self._settings.HMM_INDEX_SYMBOL)
        msg_fut = (
            f"[Live] Mã phái sinh: input='{symbol}' data_symbol='{fut_data}' trade_symbol='{fut_trade}' "
            f"(loại={fut_type})."
            if fut_type == "derivative"
            else f"[Live] Symbol '{symbol}' → data='{fut_data}' trade='{fut_trade}' (loại={fut_type})."
        )
        msg_idx = (
            f"[Live] Chỉ số basis '{self._settings.HMM_INDEX_SYMBOL}' → data_symbol='{idx_data}' (loại={idx_type}); "
            "đây là chỉ số, không phải mã hợp đồng 9 ký tự."
            if idx_type == "index"
            else f"[Live] Cột index '{self._settings.HMM_INDEX_SYMBOL}' → '{idx_data}' (loại={idx_type})."
        )
        logger.info(msg_fut)
        logger.info(msg_idx)
        print(msg_fut, flush=True)
        print(msg_idx, flush=True)

        from_d, to_d = _default_date_range(days_back)
        index_closes: Optional[list] = None
        try:
            if self._settings.HMM_USE_BASIS:
                bars, index_closes, _ = fetch_aligned_future_index(
                    self._fetcher,
                    fut_data,
                    self._settings.HMM_INDEX_SYMBOL,
                    from_d,
                    to_d,
                    RESOLUTION_15MIN,
                )
            else:
                bars = self._fetcher.fetch(
                    fut_data,
                    from_d,
                    to_d,
                    resolution=RESOLUTION_15MIN,
                    asset_type="derivative" if fut_type == "derivative" else None,
                )
        except Exception as e:
            self._api_health.record_failure(f"fetch:{e}")
            logger.error("HMM live fetch failed", extra={"error": str(e)})
            return LiveStepResult(
                ok=False,
                detail="fetch_failed",
                root_cause=_root_cause_from_detail("fetch_failed"),
            )

        if not bars:
            self._api_health.record_failure("empty_bars")
            logger.critical(
                "CRITICAL: empty_bars during live session",
                extra={
                    "input_symbol": symbol,
                    "data_symbol": fut_data,
                    "trade_symbol": fut_trade,
                    "resolution": RESOLUTION_15MIN,
                    "from_date": from_d,
                    "to_date": to_d,
                },
            )
            print(
                f"[CRITICAL] Không có nến OHLC cho data_symbol={fut_data} "
                f"(input={symbol}, trade_symbol={fut_trade}) trong khoảng {from_d}->{to_d}.",
                flush=True,
            )
            return LiveStepResult(
                ok=False,
                detail="empty_bars",
                root_cause=_root_cause_from_detail("empty_bars"),
            )

        if self._settings.HMM_USE_BASIS:
            if not index_closes or len(index_closes) != len(bars):
                logger.error(
                    "HMM basis: index/future alignment failed",
                    extra={"n_bars": len(bars), "n_ic": len(index_closes or [])},
                )
                return LiveStepResult(
                    ok=False,
                    detail="basis_align_failed",
                    root_cause=_root_cause_from_detail("basis_align_failed"),
                )

        use_basis_eff = bool(self._settings.HMM_USE_BASIS)
        if not use_basis_eff:
            index_closes = None

        ok_b, why = _validate_bars(bars)
        if not ok_b:
            self._api_health.record_failure(why)
            return LiveStepResult(ok=False, detail=why, root_cause=_root_cause_from_detail(why))

        self._api_health.record_success()

        # Realtime patch: DNSE /price/{symbol}/trades/latest works with 9-char derivative symbol.
        # Merge into current 15m slot (or append slot) to prevent stale-day last bar in live.
        try:
            rt = self._client.get_latest_trade(fut_trade)
            if rt.get("status") == 200:
                rt_price, rt_ts = _parse_latest_trade(rt.get("data"))
                if rt_price is not None and rt_ts is not None:
                    rt_index_price = None
                    if index_closes is not None and self._settings.HMM_USE_BASIS:
                        rt_index_price = fetch_latest_index_price(
                            self._client, self._settings.HMM_INDEX_SYMBOL
                        )
                    bars, index_closes, mode, skip_sig = _inject_realtime_15m_bar(
                        bars, index_closes, rt_price, rt_ts, rt_index_price
                    )
                    if skip_sig:
                        logger.warning(
                            "[Live] Realtime: bỏ qua append nến mới — không có giá index cùng ts",
                            extra={"trade_symbol": fut_trade, "mode": mode},
                        )
                    logger.info(
                        "[Live] Realtime merge từ latest trade",
                        extra={
                            "trade_symbol": fut_trade,
                            "rt_price": rt_price,
                            "rt_ts_utc": rt_ts,
                            "rt_index": rt_index_price,
                            "merge_mode": mode,
                            "new_last_unix": bars[-1].unix_ts if bars else 0,
                        },
                    )
                    print(
                        f"[Live] Realtime merge: symbol={fut_trade} price={rt_price:.2f} "
                        f"ts_utc={rt_ts} mode={mode}",
                        flush=True,
                    )
                else:
                    logger.warning(
                        "[Live] latest trade trống/không parse được",
                        extra={"trade_symbol": fut_trade, "data": rt.get("data")},
                    )
            else:
                logger.warning(
                    "[Live] latest trade API lỗi",
                    extra={"trade_symbol": fut_trade, "status": rt.get("status"), "data": rt.get("data")},
                )
        except Exception as e:
            logger.warning("[Live] realtime merge failed", extra={"error": str(e), "trade_symbol": fut_trade})

        last_ts = bars[-1].unix_ts
        bar_utc = datetime.fromtimestamp(int(last_ts), tz=timezone.utc).isoformat() if last_ts else ""
        now_utc = datetime.now(timezone.utc)
        age_sec = int(now_utc.timestamp() - int(last_ts)) if last_ts else 0
        bar_vn = unix_ts_to_vn_str(int(last_ts)) if last_ts else ""
        ts_msg = (
            f"[Live] bar_unix={last_ts} bar_utc={bar_utc} bar_vn={bar_vn} "
            f"exec_utc={now_utc.isoformat()} age_sec={age_sec}"
        )
        logger.info(ts_msg)
        print(ts_msg, flush=True)

        max_age = max(0, int(self._settings.HMM_LIVE_MAX_BAR_AGE_SEC))
        if submit_order and max_age > 0 and age_sec > max_age:
            msg = (
                f"[Live][GUARD] Dữ liệu nến quá cũ: age_sec={age_sec} > max={max_age}. "
                "Bỏ qua đặt lệnh để tránh giao dịch trên dữ liệu stale."
            )
            logger.warning(
                msg,
                extra={"age_sec": age_sec, "max_age_sec": max_age, "last_bar_unix_ts": last_ts},
            )
            print(msg, flush=True)
            return LiveStepResult(
                ok=False,
                detail="stale_bar_guard",
                root_cause=_root_cause_from_detail("stale_bar_guard"),
            )

        st = self._load_state()
        if st.get("last_bar_unix_ts") == last_ts and submit_order:
            if self._settings.HMM_LIVE_ALLOW_INCOMPLETE_BAR:
                msg = (
                    "[Live] duplicate_bar check bypassed (HMM_LIVE_ALLOW_INCOMPLETE_BAR=true): "
                    "tiếp tục cập nhật trạng thái trên nến đang hình thành."
                )
                logger.warning(msg, extra={"unix_ts": last_ts})
                print(msg, flush=True)
            else:
                logger.info("skip duplicate bar", extra={"unix_ts": last_ts})
                return LiveStepResult(
                    ok=False,
                    detail="duplicate_bar",
                    root_cause=_root_cause_from_detail("duplicate_bar"),
                )

        open_interest_seq: Optional[list] = None
        if self._settings.HMM_USE_OPEN_INTEREST:
            oi_sym = _derivative_order_symbol(symbol)
            safe_sym = "".join(
                c if c.isalnum() or c in "._-" else "_" for c in oi_sym
            )
            oi_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "cache"
                / "oi"
                / f"{safe_sym}.jsonl"
            )
            oi_cache = OICache(oi_path)
            secdef_res = self._client.get_security_definition(oi_sym)
            payload = secdef_res.get("data") if isinstance(secdef_res, dict) else None
            oi_now = parse_open_interest_from_secdef_payload(payload)
            if oi_now is None:
                logger.warning(
                    "HMM OI: secdef has no openInterestQuantity (OI)",
                    extra={"symbol": oi_sym},
                )
            open_interest_seq = build_open_interest_series_for_live(bars, oi_cache, oi_now)
            if oi_now is not None:
                oi_cache.append(int(last_ts), float(oi_now))

        cfg = HMMBacktestConfig(
            symbol=fut_data,
            hmm_config=HMMConfig(
                k_states=self._settings.HMM_LIVE_K_STATES,
                use_basis=use_basis_eff,
                use_open_interest=self._settings.HMM_USE_OPEN_INTEREST,
                use_vol_change=self._settings.HMM_USE_VOL_CHANGE,
                state_smoothing_window=self._settings.HMM_STATE_SMOOTHING_WINDOW,
                state_min_run_bars=self._settings.HMM_STATE_MIN_RUN_BARS,
                state_scoring_mode=self._settings.HMM_STATE_SCORING_MODE,
                state_strength_flat_quantile=self._settings.HMM_STATE_STRENGTH_FLAT_QUANTILE,
            ),
            warmup_bars=self._settings.HMM_LIVE_WARMUP_BARS,
            confidence_threshold=self._settings.HMM_LIVE_CONFIDENCE,
            position_size=self._settings.MCMC_POSITION_SIZE,
            refit_every=self._settings.HMM_LIVE_REFIT_EVERY,
            sliding_window_bars=self._settings.HMM_LIVE_TRAIN_WINDOW_BARS,
            index_closes=index_closes,
            open_interest=open_interest_seq,
        )
        replay = HMMBarReplay()
        direction, label, detail, trace = replay.last_bar_signal(
            bars,
            cfg,
            return_trace=self._settings.HMM_LIVE_DEBUG_TRACE,
        )
        logger.info(
            "HMM last_bar_signal",
            extra={"direction": direction, "label": label, "detail": detail},
        )
        if trace is not None:
            logger.info(trace.format_console())
            try:
                from src.hmm.hmm_live_trace import append_trace_jsonl

                trace_path = (
                    Path(__file__).resolve().parent.parent.parent
                    / "data"
                    / "debug"
                    / "hmm_live_trace.jsonl"
                )
                append_trace_jsonl(trace_path, trace)
                p = trace_path.resolve()
                trace_msg = f"[Live] Đã ghi trace HMM (JSONL) vào: {p}"
                logger.info(trace_msg)
                print(trace_msg, flush=True)
            except OSError as e:
                logger.warning("HMM trace file write failed", extra={"error": str(e)})

        if direction == FLAT or detail != "ok":
            # Reset streak when model says no-trade.
            self._signal_streak_label = ""
            self._signal_streak_count = 0
            cause = _root_cause_from_detail(detail)
            logger.info(
                "HMM live decision",
                extra={
                    "action": "no_order",
                    "detail": detail,
                    "root_cause": cause,
                    "direction": direction,
                    "label": label,
                },
            )
            return LiveStepResult(
                ok=True,
                detail=detail,
                direction=direction,
                state_label=label,
                root_cause=cause,
            )

        if not submit_order:
            cause = _root_cause_from_detail("no_submit")
            logger.info(
                "HMM live decision",
                extra={
                    "action": "signal_no_submit",
                    "detail": "no_submit",
                    "root_cause": cause,
                    "direction": direction,
                    "label": label,
                },
            )
            return LiveStepResult(
                ok=True,
                detail="no_submit",
                direction=direction,
                state_label=label,
                root_cause=cause,
            )

        # Entry guard: chỉ cho phép đặt lệnh khi state BULL/BEAR giữ ổn định >= 2 bars liên tiếp.
        current_sig = "BULL" if direction == LONG else ("BEAR" if direction == SHORT else "")
        if current_sig and current_sig == self._signal_streak_label:
            self._signal_streak_count += 1
        elif current_sig:
            self._signal_streak_label = current_sig
            self._signal_streak_count = 1
        else:
            self._signal_streak_label = ""
            self._signal_streak_count = 0

        if self._signal_streak_count < 2:
            msg = (
                f"[Live][GUARD] State chưa đủ ổn định để vào lệnh: "
                f"label={current_sig} streak={self._signal_streak_count}/2."
            )
            logger.info(msg)
            print(msg, flush=True)
            return LiveStepResult(
                ok=False,
                detail="state_persistence_guard",
                direction=direction,
                state_label=label,
                root_cause=_root_cause_from_detail("state_persistence_guard"),
            )

        if not self._settings.PAPER_MODE:
            bal, ok_bal = self.fetch_account_balance()
            if not ok_bal:
                self._risk.halt("DNSE_API_FAIL: balance fetch failed")
                return LiveStepResult(
                    ok=False,
                    detail="balance_fetch_failed",
                    root_cause=_root_cause_from_detail("balance_fetch_failed"),
                )
            self._balance_before = bal
        else:
            bal = None

        side = OrderSide.BUY if direction == LONG else OrderSide.SELL
        last_bar = bars[-1]
        order_sym = _derivative_order_symbol(symbol)
        if order_sym != symbol:
            log_ord = (
                f"[Live] Đặt lệnh dùng mã hợp đồng DNSE: '{order_sym}' "
                f"(từ cấu hình '{symbol}')."
            )
            logger.info(log_ord)
            print(log_ord, flush=True)
        qty = int(self._settings.MCMC_POSITION_SIZE)
        lo_min = max(1, int(self._settings.HMM_DERIV_LO_MIN_QTY))
        order_type = OrderType.LO if qty >= lo_min else OrderType.MTL
        ot_msg = (
            f"[Live] Loại lệnh: qty={qty} — "
            f"{'LO (giới hạn, chờ khớp)' if order_type == OrderType.LO else 'MTL (Market To Limit)'} "
            f"(ngưỡng LO từ {lo_min} hợp đồng, HMM_DERIV_LO_MIN_QTY)."
        )
        logger.info(ot_msg)
        print(ot_msg, flush=True)
        loan_package_id = 0
        if not self._settings.PAPER_MODE:
            try:
                loan_package_id = self._client.resolve_loan_package_id(
                    market_type="DERIVATIVE",
                    symbol=order_sym,
                )
                lp_msg = (
                    f"[Live] loanPackageId dùng để đặt lệnh phái sinh: {loan_package_id} "
                    f"(symbol={order_sym})."
                )
                logger.info(lp_msg)
                print(lp_msg, flush=True)
            except Exception as e:
                err_msg = f"[Live][LỖI DNSE] Không lấy được loanPackageId: {e}"
                logger.error(err_msg)
                print(err_msg, flush=True)
                return LiveStepResult(
                    ok=False,
                    detail=f"loan_package_resolve_failed:{e}",
                    direction=direction,
                    state_label=label,
                    root_cause="system_or_api",
                )

        req = OrderRequest(
            symbol=order_sym,
            side=side,
            order_type=order_type,
            price=last_bar.close,
            quantity=qty,
            loan_package_id=loan_package_id,
            market_type="DERIVATIVE",
        )
        allowed, rreason = self._risk.pre_trade_check(
            req,
            paper_mode=self._settings.PAPER_MODE,
            account_balance=bal,
        )
        if not allowed:
            cause = _root_cause_from_detail(rreason)
            logger.warning(
                "HMM live decision",
                extra={
                    "action": "order_blocked",
                    "detail": rreason,
                    "root_cause": cause,
                    "direction": direction,
                    "label": label,
                },
            )
            return LiveStepResult(ok=False, detail=rreason, root_cause=cause)

        order = self._orders.submit(req)
        if order.status == OrderStatus.REJECTED:
            logger.error(
                "[Live] Đặt lệnh DNSE bị từ chối",
                extra={
                    "symbol": req.symbol,
                    "side": req.side.value,
                    "order_type": req.order_type.value,
                    "qty": req.quantity,
                    "price": req.price,
                    "reject_reason": order.reject_reason,
                    "api_response": order.api_response,
                },
            )
            print(
                f"[Live][LỖI DNSE] Đặt lệnh thất bại: {order.reject_reason}",
                flush=True,
            )
            return LiveStepResult(
                ok=False,
                detail=f"order_rejected:{order.reject_reason}",
                direction=direction,
                state_label=label,
                order_id=order.order_id,
                root_cause="risk_or_order_reject",
            )
        if last_ts:
            self._save_state(last_ts)
        logger.info(
            "HMM live decision",
            extra={
                "action": "order_submitted",
                "detail": "submitted",
                "root_cause": "executed",
                "direction": direction,
                "label": label,
                "order_id": order.order_id,
                "order_type": order_type.value,
                "quantity": qty,
            },
        )
        return LiveStepResult(
            ok=True,
            detail="submitted",
            direction=direction,
            state_label=label,
            order_id=order.order_id,
            root_cause="executed",
        )
