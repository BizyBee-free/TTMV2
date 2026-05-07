"""TTM live one-step runner: OHLC fetch, TTM signal, optional order (mirrors HMM live flow)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from src.api_health import ApiHealthMonitor
from src.balance_utils import validate_balance_response
from src.backtest.data_fetcher import (
    RESOLUTION_15MIN,
    DataFetcher,
    OhlcBar,
    resolve_data_symbol,
    resolve_dnse_symbol,
)
from src.config import OrderSide, OrderStatus, OrderType, Settings, get_settings, resolve_symbol_profile
from src.dnse_client import BeeTradeClient
from src.hmm.basis_data import fetch_aligned_future_index
from src.hmm.oi_data import (
    OICache,
    build_open_interest_series_for_live,
    parse_open_interest_from_secdef_payload,
)
from src.live.hmm_live_runner import (
    LiveStepResult,
    _default_date_range,
    _derivative_order_symbol,
    _floor_15m_unix,
    _inject_realtime_15m_bar,
    _parse_latest_trade,
    _root_cause_from_detail,
    _validate_bars,
    fetch_latest_index_price,
)
from src.logger import get_logger
from src.order_manager import ManagedOrder, OrderManager, OrderRequest
from src.position_tracker import PositionTracker
from src.reconciliation import reconcile_after_fill
from src.risk_manager import RiskManager
from src.strategies.ttm.config import TTM_CONFIG
from src.strategies.ttm.session_flatten import session_flatten_bar_unix_ts
from src.strategies.ttm.ttm_strategy import TTMState, TTMStrategy
from src.strategies.ttm.ttm_v2_short import build_exhaustion_short_entry_meta, check_exhaustion_short_exit
from src.ops.ops_controller import OpsController
from src.telegram_notifier import TelegramNotifier
from src.vn_time import unix_ts_to_vn_str

logger = get_logger("ttm_live_runner")


def _ttm_position_side_label(pos: Any) -> Optional[str]:
    if pos is None or getattr(pos, "is_flat", True):
        return None
    if getattr(pos, "is_long", False):
        return "LONG"
    if getattr(pos, "is_short", False):
        return "SHORT"
    return None


def _ttm_is_opposite_entry(action: str, pos: Any) -> bool:
    a = (action or "").strip().upper()
    if getattr(pos, "is_long", False) and a == "SHORT":
        return True
    if getattr(pos, "is_short", False) and a == "LONG":
        return True
    return False


class TtmLiveRunner:
    """Orchestrates TTM signal on latest 15m bars and optional order submission."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[BeeTradeClient] = None,
        fetcher: Optional[DataFetcher] = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._ops: Optional[OpsController] = None
        if self._settings.OPS_ENABLED:
            self._ops = OpsController(settings=self._settings, runner=self)
        _otp = self._ops.otp_provider if self._ops else None
        self._client = client or BeeTradeClient(self._settings, otp_provider=_otp)
        self._fetcher = fetcher or DataFetcher(settings=self._settings)
        self._state_path = state_path or (
            Path(__file__).resolve().parent.parent.parent / "data" / "ttm_live_state.json"
        )
        self._tracker = PositionTracker()
        self._risk = RiskManager(self._tracker, settings=self._settings)
        self._orders = OrderManager(self._client, settings=self._settings)
        self._notifier = TelegramNotifier(settings=self._settings)
        self._api_health = ApiHealthMonitor(threshold=self._settings.DNSE_API_FAIL_HALT_THRESHOLD)
        self._balance_before: Optional[float] = None
        self._ttm = TTMStrategy(TTM_CONFIG, live_mode=True)
        # Live: bắt buộc đóng trước nghỉ trưa / cuối phiên VN (theo unix_ts nến).
        self._ttm.config["session_flatten_enabled"] = True

        self._orders.on_fill(self._tracker.on_fill)
        self._orders.on_fill(self._on_fill)

        self._last_ops_bar_ts: Optional[int] = None
        self._last_ops_rt_ts: Optional[float] = None
        if self._ops:
            self._ops.attach_runner(self)
            self._ops.register_order_callbacks(self._orders)
            self._ops.start()

        self._risk.on_halt(self._on_halt)
        self._api_health.on_halt(self._risk.halt)

        self._last_seen_bar_ts: Optional[int] = None
        self._bars_in_trade: int = 0
        self._pending_short_entry: Optional[Dict[str, Any]] = None
        self._entry_meta_by_symbol: Dict[str, Dict[str, Any]] = {}

    @property
    def risk(self) -> RiskManager:
        return self._risk

    @property
    def notifier(self) -> TelegramNotifier:
        return self._notifier

    @property
    def ops(self) -> Optional[OpsController]:
        return self._ops

    def ops_after_session_tick(self) -> None:
        if self._ops:
            self._ops.after_run_tick(
                bar_unix_ts=self._last_ops_bar_ts,
                rt_tick_ts=self._last_ops_rt_ts,
            )

    def stop_ops(self) -> None:
        if self._ops:
            self._ops.stop()

    def _ops_before_submit(
        self,
        req: OrderRequest,
        *,
        opening_new: bool,
    ) -> Optional[LiveStepResult]:
        if not self._ops:
            return None
        if not self._ops.control_state.can_send_order():
            self._ops.on_order_blocked(req, "can_send_order")
            return LiveStepResult(
                ok=False,
                detail="ops_blocked:cannot_send_order",
                root_cause=_root_cause_from_detail("ops_blocked"),
            )
        if opening_new and not self._ops.control_state.can_open_new_position():
            self._ops.on_order_blocked(req, "can_open_new_position")
            return LiveStepResult(
                ok=False,
                detail="ops_blocked:cannot_open_new_position",
                root_cause=_root_cause_from_detail("ops_blocked"),
            )
        return None

    def _ops_after_submit(self, order: ManagedOrder, req: OrderRequest) -> None:
        if not self._ops:
            return
        self._ops.on_order_submitted(order, req)
        if order.status == OrderStatus.REJECTED:
            self._ops.on_order_terminal(order)

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

        pos = self._tracker.get_position(order.symbol)
        if pos is None or pos.is_flat:
            self._bars_in_trade = 0
            self._last_seen_bar_ts = None
            self._entry_meta_by_symbol.pop(order.symbol, None)

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
        res = self._client.get_balances()
        ok, bal, reason = validate_balance_response(res)
        if not ok:
            logger.error(
                "[TTM Live] balance read failed",
                extra={"status": res.get("status"), "reason": reason, "data": res.get("data")},
            )
            self._api_health.record_failure(reason)
            return None, False
        self._api_health.record_success()
        return bal, True

    def run_once(
        self,
        symbol: str,
        days_back: int = 14,
        submit_order: bool = False,
    ) -> LiveStepResult:
        if not self._risk.external_trading_enabled:
            return LiveStepResult(
                ok=False,
                detail="trading_disabled_external",
                root_cause=_root_cause_from_detail("trading_disabled_external"),
            )

        if self._risk.is_halted:
            d = f"halted:{self._risk.halt_reason}"
            return LiveStepResult(ok=False, detail=d, root_cause=_root_cause_from_detail(d))

        prof = resolve_symbol_profile(str(symbol).strip().upper())
        if str(prof.get("type") or "").lower() != "derivative":
            d = "ttm_derivatives_only"
            logger.error(
                "TTM live chỉ hỗ trợ phái sinh",
                extra={"symbol": symbol, "resolved_type": prof.get("type")},
            )
            return LiveStepResult(
                ok=False,
                detail=d,
                root_cause=_root_cause_from_detail(d),
            )

        fut_data, fut_type = resolve_data_symbol(symbol)
        fut_trade, _ = resolve_dnse_symbol(symbol)

        from_d, to_d = _default_date_range(days_back)
        index_closes: Optional[list] = None
        try:
            if self._settings.HMM_USE_BASIS:
                print(
                    "[TTM] Đang tải dữ liệu Basis (future − index) để xác nhận trap…",
                    flush=True,
                )
                bars, index_closes, _ = fetch_aligned_future_index(
                    self._fetcher,
                    fut_data,
                    self._settings.HMM_INDEX_SYMBOL,
                    from_d,
                    to_d,
                    RESOLUTION_15MIN,
                )
            else:
                print(
                    "[TTM] HMM_USE_BASIS=false — không dùng basis; tín hiệu chỉ dựa trên giá + OI.",
                    flush=True,
                )
                bars = self._fetcher.fetch(
                    fut_data,
                    from_d,
                    to_d,
                    resolution=RESOLUTION_15MIN,
                    asset_type="derivative" if fut_type == "derivative" else None,
                )
        except Exception as e:
            self._api_health.record_failure(f"fetch:{e}")
            logger.error("TTM live fetch failed", extra={"error": str(e)})
            return LiveStepResult(
                ok=False,
                detail="fetch_failed",
                root_cause=_root_cause_from_detail("fetch_failed"),
            )

        if not bars:
            self._api_health.record_failure("empty_bars")
            return LiveStepResult(
                ok=False,
                detail="empty_bars",
                root_cause=_root_cause_from_detail("empty_bars"),
            )

        if self._settings.HMM_USE_BASIS:
            if not index_closes or len(index_closes) != len(bars):
                logger.error(
                    "TTM basis: không khớp future/index",
                    extra={"n_bars": len(bars), "n_ic": len(index_closes or [])},
                )
                print(
                    "[TTM] Lỗi: không ghép được basis (future vs index). Kiểm tra HMM_INDEX_SYMBOL và dữ liệu.",
                    flush=True,
                )
                return LiveStepResult(
                    ok=False,
                    detail="basis_align_failed",
                    root_cause=_root_cause_from_detail("basis_align_failed"),
                )

        if not self._settings.HMM_USE_BASIS:
            index_closes = None

        ok_b, why = _validate_bars(bars)
        if not ok_b:
            self._api_health.record_failure(why)
            return LiveStepResult(ok=False, detail=why, root_cause=_root_cause_from_detail(why))

        self._api_health.record_success()

        try:
            rt = self._client.get_latest_trade(fut_trade)
            if rt.get("status") == 200:
                rt_price, rt_ts = _parse_latest_trade(rt.get("data"))
                if rt_price is not None and rt_ts is not None:
                    self._last_ops_rt_ts = float(rt_ts)
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
                            "[TTM Live] Không append nến future mới — thiếu giá index realtime (basis).",
                            extra={"mode": mode, "trade_symbol": fut_trade},
                        )
                        print(
                            "[TTM] Realtime: giữ nguyên chuỗi nến (không append slot mới vì không có index cùng ts).",
                            flush=True,
                        )
                    logger.info(
                        "[TTM Live] Ghép giá realtime vào nến 15m",
                        extra={
                            "trade_symbol": fut_trade,
                            "mode": mode,
                            "rt_index": rt_index_price,
                            "merge_skipped": skip_sig,
                        },
                    )
                    print(
                        f"[TTM] Đã cập nhật nến từ giá realtime (mode={mode}).",
                        flush=True,
                    )
        except Exception as e:
            logger.warning("[TTM Live] realtime merge failed", extra={"error": str(e)})

        last_ts = bars[-1].unix_ts
        self._last_ops_bar_ts = int(last_ts) if last_ts else None
        bar_utc = datetime.fromtimestamp(int(last_ts), tz=timezone.utc).isoformat() if last_ts else ""
        now_utc = datetime.now(timezone.utc)
        age_sec = int(now_utc.timestamp() - int(last_ts)) if last_ts else 0
        bar_vn = unix_ts_to_vn_str(int(last_ts)) if last_ts else ""
        logger.info(
            f"[TTM Live] bar_unix={last_ts} bar_utc={bar_utc} bar_vn={bar_vn} age_sec={age_sec}",
        )

        max_age = max(0, int(self._settings.HMM_LIVE_MAX_BAR_AGE_SEC))
        if submit_order and max_age > 0 and age_sec > max_age:
            return LiveStepResult(
                ok=False,
                detail="stale_bar_guard",
                root_cause=_root_cause_from_detail("stale_bar_guard"),
            )

        order_sym = _derivative_order_symbol(symbol)
        pos = self._tracker.get_position(order_sym)

        # Đếm nến khi đang có vị thế (trước khi bỏ qua trùng nến).
        if pos is not None and not pos.is_flat:
            if self._last_seen_bar_ts is not None and last_ts > self._last_seen_bar_ts:
                self._bars_in_trade += 1
            self._last_seen_bar_ts = int(last_ts)

        st = self._load_state()
        is_dup = st.get("last_bar_unix_ts") == last_ts and submit_order
        if is_dup and not self._settings.HMM_LIVE_ALLOW_INCOMPLETE_BAR:
            if pos is None or pos.is_flat:
                logger.info("TTM skip duplicate bar", extra={"unix_ts": last_ts})
                return LiveStepResult(
                    ok=False,
                    detail="duplicate_bar",
                    root_cause=_root_cause_from_detail("duplicate_bar"),
                )

        open_interest_seq: Optional[list] = None
        oi_sym = _derivative_order_symbol(symbol)
        safe_sym = "".join(c if c.isalnum() or c in "._-" else "_" for c in oi_sym)
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
        open_interest_seq = build_open_interest_series_for_live(bars, oi_cache, oi_now)
        if oi_now is not None:
            oi_cache.append(int(last_ts), float(oi_now))

        print("[TTM] Đang ghép chuỗi Open Interest theo từng nến (cache + secdef)…", flush=True)

        basis_series: Optional[list] = None
        if index_closes is not None and len(index_closes) == len(bars):
            basis_series = [float(b.close) - float(ic) for b, ic in zip(bars, index_closes)]
            print(
                f"[TTM] Basis nến cuối ≈ {basis_series[-1]:.4f} điểm (future close − index close).",
                flush=True,
            )
        else:
            print(
                "[TTM] Không có basis — dùng basis=0 trong mô hình (chỉ OI + giá).",
                flush=True,
            )

        data: Dict[str, Any] = {"bars": bars, "open_interest": open_interest_seq}
        if basis_series is not None:
            data["basis"] = basis_series

        last_bar = bars[-1]
        if pos is not None and not pos.is_flat:
            self._pending_short_entry = None
        elif self._pending_short_entry is not None:
            due_ts = int(self._pending_short_entry.get("signal_bar_ts", 0) or 0)
            if last_ts > due_ts:
                if not submit_order:
                    return LiveStepResult(
                        ok=True,
                        detail="pending_short_next_open",
                        state_label="SHORT_PENDING",
                        root_cause=_root_cause_from_detail("no_submit"),
                    )
                return self._submit_pending_short_entry(
                    order_sym=order_sym,
                    last_bar=last_bar,
                    last_ts=int(last_ts or 0),
                    pending=dict(self._pending_short_entry),
                )
        ps = _ttm_position_side_label(pos)
        sig = self._ttm.generate_signal(TTMState(data=data, symbol=symbol, position_side=ps))

        logger.info(
            "TTM live: kết quả sau khi tính trap (Basis + OI + giá)",
            extra={
                "strategy": sig.get("strategy", "TTM"),
                "action": sig.get("action"),
                "vi_the": ps,
                "confidence": sig.get("confidence"),
                "reason": sig.get("reason"),
                "trap_score": sig.get("trap_score"),
                "diem_trap": sig.get("trap_score"),
                "features": sig.get("features"),
            },
        )
        print(
            f"[TTM] Tín hiệu: action={sig.get('action')} | "
            f"trap_score={sig.get('trap_score', '—')} | lý do={sig.get('reason', '')}",
            flush=True,
        )
        trace = sig.get("decision_trace") if isinstance(sig.get("decision_trace"), dict) else {}
        dbg = sig.get("debug") if isinstance(sig.get("debug"), dict) else {}
        breakout_strength = trace.get("breakout_strength", dbg.get("breakout_strength"))
        basis_delta = trace.get("basis_delta", dbg.get("basis_delta"))
        score_breakdown = trace.get("score_components", dbg.get("score_components"))
        print(
            "[TTM] chi_tiet: "
            f"breakout_strength={breakout_strength if breakout_strength is not None else '—'} | "
            f"basis_delta={basis_delta if basis_delta is not None else '—'} | "
            f"score_breakdown={score_breakdown if score_breakdown is not None else '{}'}",
            flush=True,
        )

        action = str(sig.get("action", "HOLD")).upper()
        feat_snap = sig.get("features") if isinstance(sig.get("features"), dict) else {}
        pnl = float(pos.unrealized_pnl(float(last_bar.close))) if pos is not None and not pos.is_flat else 0.0
        sl = float(self._ttm.config.get("stop_loss_points", 2.5))
        tp = float(self._ttm.config.get("take_profit_points", 5.0))
        max_bars = int(self._ttm.config.get("max_bars_in_trade", 8))
        allow_rev = bool(self._ttm.config.get("allow_reverse", False))

        # --- Ưu tiên: cuối phiên VN → EXIT tín hiệu → đảo chiều → SL / TP / thời gian ---
        if pos is not None and not pos.is_flat:
            if session_flatten_bar_unix_ts(int(last_ts or 0), self._ttm.config):
                if not submit_order:
                    return LiveStepResult(
                        ok=True,
                        detail="no_submit",
                        state_label="EXIT",
                        root_cause=_root_cause_from_detail("no_submit"),
                    )
                return self._submit_exit(
                    order_sym=order_sym,
                    last_bar=last_bar,
                    last_ts=last_ts,
                    pos=pos,
                    detail_label="ttm_exit_session_end",
                    trigger_type="SESSION",
                    features_snapshot=feat_snap,
                    pnl_at_exit=pnl,
                )
            if action == "EXIT":
                if not submit_order:
                    return LiveStepResult(
                        ok=True,
                        detail="no_submit",
                        state_label="EXIT",
                        root_cause=_root_cause_from_detail("no_submit"),
                    )
                return self._submit_exit(
                    order_sym=order_sym,
                    last_bar=last_bar,
                    last_ts=last_ts,
                    pos=pos,
                    detail_label=f"ttm_exit_signal:{sig.get('reason', '')}",
                    trigger_type="SIGNAL",
                    features_snapshot=feat_snap,
                    pnl_at_exit=pnl,
                )

            sig_flat = self._ttm.generate_signal(TTMState(data=data, symbol=symbol, position_side=None))
            flat_action = str(sig_flat.get("action", "HOLD")).upper()
            if _ttm_is_opposite_entry(flat_action, pos):
                if not submit_order:
                    return LiveStepResult(
                        ok=True,
                        detail="no_submit",
                        state_label="EXIT",
                        root_cause=_root_cause_from_detail("no_submit"),
                    )
                ex = self._submit_exit(
                    order_sym=order_sym,
                    last_bar=last_bar,
                    last_ts=last_ts,
                    pos=pos,
                    detail_label="ttm_exit_opposite_signal",
                    trigger_type="REVERSE",
                    features_snapshot=sig_flat.get("features") if isinstance(sig_flat.get("features"), dict) else feat_snap,
                    pnl_at_exit=pnl,
                )
                if not ex.ok:
                    return ex
                if allow_rev and flat_action in ("LONG", "SHORT"):
                    self._bars_in_trade = 0
                    self._last_seen_bar_ts = int(last_ts)
                    return self._submit_entry_after_exit(
                        order_sym=order_sym,
                        last_bar=last_bar,
                        last_ts=last_ts,
                        action=flat_action,
                        sig=sig_flat,
                        prior_result=ex,
                    )
                return ex

            exit_reason = ""
            trigger_risk = ""
            ent_meta = self._entry_meta_by_symbol.get(order_sym) or {}
            short_exit = None
            if pos.is_short:
                short_exit = check_exhaustion_short_exit(ent_meta, float(last_bar.close), self._bars_in_trade)
            if short_exit is not None:
                exit_reason = str(short_exit["reason"])
                trigger_risk = str(short_exit["trigger"])
            elif pnl <= -sl:
                exit_reason, trigger_risk = "ttm_exit_stop_loss", "SL"
            elif pnl >= tp:
                exit_reason, trigger_risk = "ttm_exit_take_profit", "TP"
            elif self._bars_in_trade >= max_bars:
                exit_reason, trigger_risk = "ttm_exit_max_bars", "TIME"

            if exit_reason:
                if not submit_order:
                    return LiveStepResult(
                        ok=True,
                        detail="no_submit",
                        state_label="EXIT",
                        root_cause=_root_cause_from_detail("no_submit"),
                    )
                return self._submit_exit(
                    order_sym=order_sym,
                    last_bar=last_bar,
                    last_ts=last_ts,
                    pos=pos,
                    detail_label=exit_reason,
                    trigger_type=trigger_risk,
                    features_snapshot=feat_snap,
                    pnl_at_exit=pnl,
                )

            if not submit_order:
                return LiveStepResult(
                    ok=True,
                    detail="no_submit",
                    state_label=action,
                    root_cause=_root_cause_from_detail("no_submit"),
                )
            return LiveStepResult(
                ok=True,
                detail="holding",
                state_label=action,
                root_cause=_root_cause_from_detail("warmup"),
            )

        if action == "EXIT":
            if not submit_order:
                return LiveStepResult(
                    ok=True,
                    detail="no_submit",
                    state_label="HOLD",
                    root_cause=_root_cause_from_detail("no_submit"),
                )
            return LiveStepResult(
                ok=True,
                detail="flat_ignore_exit",
                state_label="HOLD",
                root_cause=_root_cause_from_detail("warmup"),
            )

        if action == "HOLD" or not submit_order:
            if not submit_order:
                return LiveStepResult(
                    ok=True,
                    detail="no_submit",
                    state_label=action,
                    root_cause=_root_cause_from_detail("no_submit"),
                )
            return LiveStepResult(
                ok=True,
                detail="warmup" if sig.get("reason") == "no_data" else "no_signal",
                state_label=action,
                root_cause=_root_cause_from_detail("warmup"),
            )

        if action == "SHORT" and bool(feat_snap.get("exhaustion_confirm")):
            self._pending_short_entry = {
                "signal_bar_ts": int(last_ts or 0),
                "features": dict(feat_snap),
                "reason": str(sig.get("reason", "")),
                "confidence": float(sig.get("confidence", 0.0) or 0.0),
            }
            if last_ts:
                self._save_state(int(last_ts))
            return LiveStepResult(
                ok=True,
                detail="queued_short_next_bar_open",
                direction=0,
                state_label="SHORT_PENDING",
                root_cause="executed",
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

        qty = int(self._settings.MCMC_POSITION_SIZE)
        lo_min = max(1, int(self._settings.HMM_DERIV_LO_MIN_QTY))
        order_type = OrderType.LO if qty >= lo_min else OrderType.MTL

        if action == "LONG":
            side = OrderSide.BUY
        elif action == "SHORT":
            side = OrderSide.SELL
        else:
            return LiveStepResult(ok=True, detail="no_signal", state_label=action)

        if session_flatten_bar_unix_ts(int(last_ts or 0), self._ttm.config):
            return LiveStepResult(
                ok=True,
                detail="session_flatten_no_new_entry",
                state_label="HOLD",
                root_cause=_root_cause_from_detail("warmup"),
            )

        loan_package_id = 0
        if not self._settings.PAPER_MODE:
            try:
                loan_package_id = self._client.resolve_loan_package_id(
                    market_type="DERIVATIVE",
                    symbol=order_sym,
                )
            except Exception as e:
                return LiveStepResult(
                    ok=False,
                    detail=f"loan_package_resolve_failed:{e}",
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
            return LiveStepResult(ok=False, detail=rreason, root_cause=_root_cause_from_detail(rreason))

        blocked = self._ops_before_submit(req, opening_new=True)
        if blocked is not None:
            return blocked

        order = self._orders.submit(req)
        self._ops_after_submit(order, req)
        if order.status == OrderStatus.REJECTED:
            return LiveStepResult(
                ok=False,
                detail=f"order_rejected:{order.reject_reason}",
                order_id=order.order_id,
                root_cause="risk_or_order_reject",
            )

        if last_ts:
            self._save_state(last_ts)
        self._bars_in_trade = 0
        self._last_seen_bar_ts = int(last_ts)

        return LiveStepResult(
            ok=True,
            detail="submitted",
            direction=1 if action == "LONG" else -1,
            state_label=action,
            order_id=order.order_id,
            root_cause="executed",
        )

    def _submit_entry_after_exit(
        self,
        order_sym: str,
        last_bar: OhlcBar,
        last_ts: int,
        action: str,
        sig: Dict[str, Any],
        prior_result: LiveStepResult,
    ) -> LiveStepResult:
        """Đặt lệnh vào mới sau khi đóng vị thế đảo chiều (cùng bước run_once)."""
        if session_flatten_bar_unix_ts(int(last_ts or 0), self._ttm.config):
            return prior_result
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

        qty = int(self._settings.MCMC_POSITION_SIZE)
        lo_min = max(1, int(self._settings.HMM_DERIV_LO_MIN_QTY))
        order_type = OrderType.LO if qty >= lo_min else OrderType.MTL
        if action == "LONG":
            side = OrderSide.BUY
        elif action == "SHORT":
            side = OrderSide.SELL
        else:
            return prior_result

        loan_package_id = 0
        if not self._settings.PAPER_MODE:
            try:
                loan_package_id = self._client.resolve_loan_package_id(
                    market_type="DERIVATIVE",
                    symbol=order_sym,
                )
            except Exception as e:
                return LiveStepResult(
                    ok=False,
                    detail=f"loan_package_resolve_failed:{e}",
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
            return LiveStepResult(ok=False, detail=rreason, root_cause=_root_cause_from_detail(rreason))

        blocked = self._ops_before_submit(req, opening_new=True)
        if blocked is not None:
            return blocked

        order = self._orders.submit(req)
        self._ops_after_submit(order, req)
        if order.status == OrderStatus.REJECTED:
            return LiveStepResult(
                ok=False,
                detail=f"order_rejected:{order.reject_reason}",
                order_id=order.order_id,
                root_cause="risk_or_order_reject",
            )

        print(
            f"[TTM] Đảo chiều: đã vào {action} sau khi đóng (order_id={order.order_id}).",
            flush=True,
        )
        logger.info(
            "TTM live: đảo chiều — vào lệnh mới sau EXIT",
            extra={
                "hanh_dong": action,
                "ly_do": sig.get("reason"),
                "order_id": order.order_id,
                "exit_order_id": prior_result.order_id,
            },
        )

        if last_ts:
            self._save_state(last_ts)

        return LiveStepResult(
            ok=True,
            detail="submitted_reverse",
            direction=1 if action == "LONG" else -1,
            state_label=action,
            order_id=order.order_id,
            root_cause="executed",
        )

    def _submit_pending_short_entry(
        self,
        order_sym: str,
        last_bar: OhlcBar,
        last_ts: int,
        pending: Mapping[str, Any],
    ) -> LiveStepResult:
        if session_flatten_bar_unix_ts(int(last_ts or 0), self._ttm.config):
            self._pending_short_entry = None
            return LiveStepResult(
                ok=True,
                detail="session_flatten_no_new_entry",
                state_label="HOLD",
                root_cause=_root_cause_from_detail("warmup"),
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

        qty = int(self._settings.MCMC_POSITION_SIZE)
        lo_min = max(1, int(self._settings.HMM_DERIV_LO_MIN_QTY))
        order_type = OrderType.LO if qty >= lo_min else OrderType.MTL
        loan_package_id = 0
        if not self._settings.PAPER_MODE:
            try:
                loan_package_id = self._client.resolve_loan_package_id(
                    market_type="DERIVATIVE",
                    symbol=order_sym,
                )
            except Exception as e:
                return LiveStepResult(
                    ok=False,
                    detail=f"loan_package_resolve_failed:{e}",
                    root_cause="system_or_api",
                )
        entry_price = float(getattr(last_bar, "open", 0.0) or 0.0)
        if entry_price <= 0.0:
            entry_price = float(getattr(last_bar, "close", 0.0) or 0.0)
        req = OrderRequest(
            symbol=order_sym,
            side=OrderSide.SELL,
            order_type=order_type,
            price=entry_price,
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
            return LiveStepResult(ok=False, detail=rreason, root_cause=_root_cause_from_detail(rreason))

        blocked = self._ops_before_submit(req, opening_new=True)
        if blocked is not None:
            return blocked

        order = self._orders.submit(req)
        self._ops_after_submit(order, req)
        if order.status == OrderStatus.REJECTED:
            return LiveStepResult(
                ok=False,
                detail=f"order_rejected:{order.reject_reason}",
                order_id=order.order_id,
                root_cause="risk_or_order_reject",
            )
        feat_sig = pending.get("features") if isinstance(pending.get("features"), dict) else {}
        short_meta = build_exhaustion_short_entry_meta(feat_sig, entry_price, self._ttm.config) or {}
        self._entry_meta_by_symbol[order_sym] = {
            "entry_price": float(entry_price),
            "entry_time": str(getattr(last_bar, "time", "") or ""),
            "side": "SHORT",
            **short_meta,
        }
        self._pending_short_entry = None
        if last_ts:
            self._save_state(last_ts)
        self._bars_in_trade = 0
        self._last_seen_bar_ts = int(last_ts)
        return LiveStepResult(
            ok=True,
            detail="submitted_pending_short_open",
            direction=-1,
            state_label="SHORT",
            order_id=order.order_id,
            root_cause="executed",
        )

    def _submit_exit(
        self,
        order_sym: str,
        last_bar: OhlcBar,
        last_ts: int,
        pos: Any,
        detail_label: str,
        *,
        trigger_type: str = "SIGNAL",
        features_snapshot: Optional[Dict[str, Any]] = None,
        pnl_at_exit: Optional[float] = None,
    ) -> LiveStepResult:
        close_side = OrderSide.SELL if pos.is_long else OrderSide.BUY
        qty = abs(int(pos.quantity))
        if qty <= 0:
            return LiveStepResult(ok=True, detail=detail_label, root_cause="strategy_or_model")

        bal = None
        if not self._settings.PAPER_MODE:
            bal, ok_bal = self.fetch_account_balance()
            if not ok_bal:
                return LiveStepResult(
                    ok=False,
                    detail="balance_fetch_failed",
                    root_cause=_root_cause_from_detail("balance_fetch_failed"),
                )
            self._balance_before = bal

        lo_min = max(1, int(self._settings.HMM_DERIV_LO_MIN_QTY))
        order_type = OrderType.LO if qty >= lo_min else OrderType.MTL
        loan_package_id = 0
        if not self._settings.PAPER_MODE:
            try:
                loan_package_id = self._client.resolve_loan_package_id(
                    market_type="DERIVATIVE",
                    symbol=order_sym,
                )
            except Exception as e:
                return LiveStepResult(
                    ok=False,
                    detail=f"loan_package_resolve_failed:{e}",
                    root_cause="system_or_api",
                )

        req = OrderRequest(
            symbol=order_sym,
            side=close_side,
            order_type=order_type,
            price=float(last_bar.close),
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
            return LiveStepResult(ok=False, detail=rreason, root_cause=_root_cause_from_detail(rreason))

        blocked = self._ops_before_submit(req, opening_new=False)
        if blocked is not None:
            return blocked

        order = self._orders.submit(req)
        self._ops_after_submit(order, req)
        if order.status == OrderStatus.REJECTED:
            return LiveStepResult(
                ok=False,
                detail=f"order_rejected:{order.reject_reason}",
                order_id=order.order_id,
                root_cause="risk_or_order_reject",
            )

        loai = str(trigger_type).upper()
        print(
            f"[TTM] Thoát lệnh: kích hoạt={loai} | {detail_label} | PnL≈{pnl_at_exit if pnl_at_exit is not None else '—'}",
            flush=True,
        )
        logger.info(
            "TTM live: thoát lệnh",
            extra={
                "strategy": "TTM",
                "action": "EXIT",
                "kich_hoat": loai,
                "ly_do": detail_label,
                "pnl_tai_thoat": pnl_at_exit,
                "dac_diem": features_snapshot or {},
            },
        )

        if last_ts:
            self._save_state(last_ts)

        return LiveStepResult(
            ok=True,
            detail=detail_label,
            direction=0,
            state_label="EXIT",
            order_id=order.order_id,
            root_cause="executed",
        )
