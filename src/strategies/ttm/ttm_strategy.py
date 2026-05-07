"""TTM strategy: features + signal + optional websocket/paper adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import MarketType, OrderSide, OrderType, Settings, get_settings
from src.backtest.data_fetcher import OhlcBar
from src.logger import get_logger
from src.order_manager import ManagedOrder, OrderRequest
from src.position_tracker import PositionTracker
from src.risk_manager import RiskManager
from src.strategy_base import Signal, SignalEvent, StrategyBase
from src.strategies.base_strategy import BaseStrategy
from src.strategies.ttm.config import TTM_CONFIG, is_ttm_adaptive_learning_enabled
from src.strategies.ttm.empirical.adaptive_engine import EmpiricalAlphaEngine
from src.strategies.ttm.session_flatten import session_flatten_bar_unix_ts
from src.strategies.ttm.ttm_adaptive_context import TTMAdaptiveContext, build_trade_record_from_snapshot
from src.strategies.ttm.ttm_alignment import validate_ttm_input_alignment, write_aligned_market_bundle
from src.strategies.ttm.ttm_signal_v2 import generate_ttm_signal_v2
from src.strategies.ttm.ttm_v2_short import build_exhaustion_short_entry_meta, check_exhaustion_short_exit

logger = get_logger("ttm_strategy")


@dataclass
class TTMState:
    """State passed to :meth:`TTMStrategy.generate_signal`."""

    data: Dict[str, Any]
    symbol: str = ""
    position_side: Optional[str] = None  # "LONG" | "SHORT" | None


class TTMStrategy(BaseStrategy):
    """Trapped Trader Model — signal from OHLC + optional OI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, live_mode: bool = False) -> None:
        self.config = dict(TTM_CONFIG if config is None else config)
        self.live_mode = live_mode
        self.adaptive_context: Optional[TTMAdaptiveContext] = None
        self.empirical_engine: Optional[EmpiricalAlphaEngine] = None

    def generate_signal(self, state: Any) -> Dict[str, Any]:
        if bool(self.config.get("ttm_assert_adaptive_wired", False)):
            if is_ttm_adaptive_learning_enabled(self.config) and self.adaptive_context is None:
                raise RuntimeError(
                    "ttm_adaptive_enabled with ttm_assert_adaptive_wired but adaptive_context is None — "
                    "PnL-driven learning will never run"
                )
        position_side: Optional[str] = None
        if isinstance(state, TTMState):
            data = state.data
            position_side = state.position_side
        elif isinstance(state, dict):
            data = state.get("data", state)
            position_side = state.get("position_side")
        else:
            data = getattr(state, "data", {})
            position_side = getattr(state, "position_side", None)

        bar_ts: Optional[int] = None
        if isinstance(data, dict):
            bars = data.get("bars") or []
            if bars:
                lb = bars[-1]
                if isinstance(lb, dict) and lb.get("unix_ts") is not None:
                    try:
                        bar_ts = int(lb["unix_ts"])
                    except (TypeError, ValueError):
                        bar_ts = None
        signal = generate_ttm_signal_v2(
            data,
            self.config,
            position_side=position_side,
            live_mode=self.live_mode,
            adaptive=self.adaptive_context,
            empirical_engine=self.empirical_engine,
            bar_timestamp=bar_ts,
        )
        signal["strategy"] = "TTM"
        logger.info(
            "TTM Strategy: hoàn tất sinh tín hiệu (Basis + OI + giá)",
            extra={
                "hanh_dong": signal.get("action"),
                "vi_the": position_side,
                "diem_trap": signal.get("trap_score"),
                "do_tin_cay": signal.get("confidence"),
                "ly_do": signal.get("reason"),
            },
        )
        return signal


class TTMDerivativesStrategy(StrategyBase):
    """Paper/live websocket driver: maps TTM signals to BUY/SELL with one-position guard."""

    def __init__(
        self,
        name: str = "ttm_deriv",
        symbols: Optional[List[str]] = None,
        settings: Optional[Settings] = None,
        ttm_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = settings or get_settings()
        sym = symbols or [cfg.HMM_SYMBOL]
        super().__init__(name=name, symbols=sym, settings=cfg)
        self._ttm = TTMStrategy(ttm_config or TTM_CONFIG)
        if is_ttm_adaptive_learning_enabled(self._ttm.config):
            self._ttm.adaptive_context = TTMAdaptiveContext(self._ttm.config)
        if bool(self._ttm.config.get("ttm_v2_empirical_alpha_enabled", False)):
            self._ttm.empirical_engine = EmpiricalAlphaEngine(self._ttm.config)
        if bool(self._ttm.config.get("ttm_assert_adaptive_wired", False)):
            if is_ttm_adaptive_learning_enabled(self._ttm.config) and self._ttm.adaptive_context is None:
                raise RuntimeError(
                    "TTMDerivativesStrategy: ttm_adaptive_enabled requires TTMAdaptiveContext"
                )
        self._adaptive_entry_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._risk_mgr: Optional[RiskManager] = None
        self._bars_in_trade = 0
        self._last_ohlc_time: Optional[float] = None
        self._ohlc_history: List[Any] = []
        self._position_size = int(cfg.MCMC_POSITION_SIZE)
        self._pending_entry: Optional[Dict[str, Any]] = None
        self._entry_meta_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._last_bar_open: float = 0.0
        self._last_bar_index: int = -1
        self._paper_enricher: Any = None
        self._paper_extra: Dict[str, Any] = {}

    def set_paper_enricher(self, enricher: Any) -> None:
        """REST basis + OI (``TTMPaperEnricher``) — paper_test TTM gần với live."""
        self._paper_enricher = enricher

    def set_risk_manager(self, risk_mgr: RiskManager) -> None:
        self._risk_mgr = risk_mgr

    def _last_bar_unix_ts(self) -> int:
        if not self._ohlc_history:
            return 0
        o = self._ohlc_history[-1]
        ut = int(getattr(o, "unix_ts", 0) or 0)
        if ut > 0:
            return ut
        t = getattr(o, "time", None)
        if isinstance(t, (int, float)) and int(t) > 1_000_000_000:
            return int(t)
        return 0

    def on_tick(self, symbol: str, quote=None, trade=None) -> Optional[SignalEvent]:
        if symbol not in self.symbols:
            return None
        current_price = self._extract_price(quote, trade)
        if current_price <= 0:
            return None

        if self._risk_mgr is not None:
            self._risk_mgr.on_tick_risk_check(symbol, current_price)
            if self._risk_mgr.is_halted:
                return None

        pos = self._tracker.get_position(symbol) if self._tracker else None
        is_flat = pos is None or pos.is_flat

        if is_flat and self._pending_entry is not None:
            pe = dict(self._pending_entry)
            if self._last_bar_index > int(pe.get("signal_bar_index", -1)):
                act = str(pe.get("action", "")).upper()
                qty = int(pe.get("quantity", self._position_size) or self._position_size)
                conf = float(pe.get("confidence", 0.0) or 0.0)
                rsn = str(pe.get("reason", "next_bar_open_entry"))
                sig_ref = pe.get("signal_ref") if isinstance(pe.get("signal_ref"), dict) else {}
                entry_price = float(self._last_bar_open if self._last_bar_open > 0 else current_price)
                self._pending_entry = None
                entry_time = str(getattr(self._ohlc_history[-1], "time", "") or "") if self._ohlc_history else ""
                feat_sig = sig_ref.get("features") if isinstance(sig_ref.get("features"), dict) else {}
                entry_meta = {
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "side": act,
                }
                if act == "SHORT":
                    short_meta = build_exhaustion_short_entry_meta(feat_sig, entry_price, self._ttm.config)
                    if short_meta:
                        entry_meta.update(short_meta)
                self._entry_meta_by_symbol[symbol] = entry_meta
                if act == "LONG":
                    self._capture_adaptive_entry(symbol, entry_price, "LONG", sig_ref)
                    return SignalEvent(
                        signal=Signal.BUY,
                        symbol=symbol,
                        price=entry_price,
                        quantity=qty,
                        confidence=conf,
                        reason=rsn,
                        metadata={
                            "strategy": "TTM",
                            "ttm_action": "LONG",
                            "trigger": "NEXT_BAR_OPEN",
                            "features": sig_ref.get("features"),
                            "position_sizing": "strength_scaled",
                            "entry_time": entry_time,
                        },
                    )
                if act == "SHORT":
                    self._capture_adaptive_entry(symbol, entry_price, "SHORT", sig_ref)
                    return SignalEvent(
                        signal=Signal.SELL,
                        symbol=symbol,
                        price=entry_price,
                        quantity=qty,
                        confidence=conf,
                        reason=rsn,
                        metadata={
                            "strategy": "TTM",
                            "ttm_action": "SHORT",
                            "trigger": "NEXT_BAR_OPEN",
                            "features": sig_ref.get("features"),
                            "position_sizing": "strength_scaled",
                            "entry_time": entry_time,
                        },
                    )

        min_bars = int(self._ttm.config.get("breakout_window", 20)) + 2
        if len(self._ohlc_history) < min_bars and (pos is None or pos.is_flat):
            return None

        state = self._build_state(symbol)

        if pos is not None and not pos.is_flat:
            if session_flatten_bar_unix_ts(self._last_bar_unix_ts(), self._ttm.config):
                sig_eod = {"reason": "ttm_exit_session_end", "features": {}}
                return self._emit_exit_ttm(symbol, pos, current_price, sig_eod, "SESSION")
            ps = "LONG" if pos.is_long else "SHORT"
            sig = self._ttm.generate_signal(TTMState(data=state, symbol=symbol, position_side=ps))
            self._log_signal(sig, symbol)
            if str(sig.get("action", "")).upper() == "EXIT":
                return self._emit_exit_ttm(symbol, pos, current_price, sig, "SIGNAL")

            exit_ev = self._maybe_exit_risk(symbol, pos, current_price)
            if exit_ev is not None:
                return exit_ev
            return None

        sig = self._ttm.generate_signal(TTMState(data=state, symbol=symbol, position_side=None))
        self._log_signal(sig, symbol)

        action = str(sig.get("action", "HOLD")).upper()
        conf = float(sig.get("confidence", 0.0) or 0.0)
        reason = str(sig.get("reason", ""))
        meta_base = {
            "strategy": "TTM",
            "ttm_action": action,
            "features": sig.get("features"),
        }

        if action in ("LONG", "SHORT") and session_flatten_bar_unix_ts(
            self._last_bar_unix_ts(), self._ttm.config
        ):
            return None

        if action in ("LONG", "SHORT"):
            base_size = max(1.0, float(self._position_size))
            qty = max(1, int(round(base_size)))
            self._pending_entry = {
                "action": action,
                "confidence": conf,
                "reason": reason or "next_bar_open_entry",
                "quantity": qty,
                "signal_bar_index": self._last_bar_index,
                "signal_ref": sig,
                "meta_base": meta_base,
            }
            return None
        return None

    def handle_ohlc(self, ohlc) -> None:
        o_sym = (getattr(ohlc, "symbol", None) or "").strip().upper()
        if not o_sym or o_sym not in self.symbols:
            return
        new_time = getattr(ohlc, "time", None)
        if new_time is not None and new_time == self._last_ohlc_time:
            return
        self._last_ohlc_time = new_time
        self._ohlc_history.append(ohlc)
        self._last_bar_index = len(self._ohlc_history) - 1
        self._last_bar_open = float(getattr(ohlc, "open", 0.0) or 0.0)
        max_keep = max(500, int(self._ttm.config.get("breakout_window", 20)) * 3)
        if len(self._ohlc_history) > max_keep:
            self._ohlc_history = self._ohlc_history[-max_keep:]

        if self._paper_enricher is not None and self.symbols:
            try:
                bars = self._bars_to_ohlc_bars(self.symbols[0])
                self._paper_extra = self._paper_enricher.enrich(bars) or {}
            except Exception as ex:
                logger.warning("TTM paper enrich (basis/OI) failed", extra={"error": str(ex)})
                self._paper_extra = {}

        if self._tracker is not None:
            p = self._tracker.get_position(o_sym)
            if p is not None and not p.is_flat:
                self._bars_in_trade += 1

    def on_signal(self, event: SignalEvent) -> None:
        if self._risk_mgr is not None:
            req = OrderRequest(
                symbol=event.symbol,
                side=OrderSide.BUY if event.signal == Signal.BUY else OrderSide.SELL,
                order_type=OrderType.LO,
                price=event.price,
                quantity=event.quantity,
                market_type=MarketType.DERIVATIVE.value,
            )
            allowed, reason = self._risk_mgr.pre_trade_check(
                req,
                paper_mode=self._settings.PAPER_MODE,
            )
            if not allowed:
                logger.info(f"[{self.name}] Signal blocked by risk: {reason}")
                return

        self.place_order(
            symbol=event.symbol,
            side=OrderSide.BUY if event.signal == Signal.BUY else OrderSide.SELL,
            order_type=OrderType.LO,
            price=event.price,
            quantity=event.quantity,
            market_type=MarketType.DERIVATIVE.value,
        )

    def on_fill(self, order: ManagedOrder) -> None:
        if order.symbol not in self.symbols:
            return
        pos = self._tracker.get_position(order.symbol) if self._tracker else None
        if pos is not None and pos.is_flat:
            self._bars_in_trade = 0
            self._entry_meta_by_symbol.pop(order.symbol, None)

    def _bars_to_ohlc_bars(self, symbol: str) -> List[OhlcBar]:
        bars: List[OhlcBar] = []
        for o in self._ohlc_history:
            t = getattr(o, "time", "") or ""
            ut = int(getattr(o, "unix_ts", 0) or 0)
            if ut == 0:
                ut = int(getattr(o, "time", 0) or 0) or int(getattr(o, "lastUpdated", 0) or 0)
            bars.append(
                OhlcBar(
                    symbol=symbol,
                    time=str(t),
                    open=float(getattr(o, "open", 0) or 0),
                    high=float(getattr(o, "high", 0) or 0),
                    low=float(getattr(o, "low", 0) or 0),
                    close=float(getattr(o, "close", 0) or 0),
                    volume=float(getattr(o, "volume", 0) or 0),
                    unix_ts=ut,
                )
            )
        return bars

    def _build_state(self, symbol: str) -> Dict[str, Any]:
        bars = self._bars_to_ohlc_bars(symbol)
        state: Dict[str, Any] = {"bars": bars}
        if self._paper_extra:
            state.update(self._paper_extra)
        pe = getattr(self, "_paper_enricher", None)
        if pe is not None:
            snap = getattr(pe, "last_secdef_snapshot", None)
            if isinstance(snap, dict):
                state["oi_enricher_meta"] = {
                    "status": snap.get("status"),
                    "oi_source": snap.get("oi_source"),
                    "openInterestQuantity": snap.get("openInterestQuantity"),
                    "openInterestQuantity_rest": snap.get("openInterestQuantity_rest"),
                    "secdef_symbol_used": snap.get("secdef_symbol_used"),
                }
        nb = len(bars)
        if nb > 0:
            validate_ttm_input_alignment(
                state,
                strict_basis_oi=bool(self._ttm.config.get("ttm_strict_basis_oi", False)),
                use_open_interest=bool(self._ttm.config.get("use_open_interest", True)),
            )
            persist_path = str(self._ttm.config.get("ttm_persist_aligned_bundle_path") or "").strip()
            if persist_path:
                b = state.get("basis")
                oi = state.get("open_interest")
                use_oi = bool(self._ttm.config.get("use_open_interest", True))
                if b is None:
                    logger.warning(
                        "TTM: skip aligned bundle persist (missing basis)",
                        extra={"path": persist_path, "symbol": symbol},
                    )
                else:
                    ba = np.asarray(b, dtype=np.float64).ravel()
                    if ba.size != nb:
                        logger.warning(
                            "TTM: skip aligned bundle persist (basis length mismatch)",
                            extra={"path": persist_path, "symbol": symbol, "n_bars": nb, "basis_len": ba.size},
                        )
                    else:
                        oa = None
                        skip_write = False
                        if use_oi:
                            if oi is None:
                                logger.warning(
                                    "TTM: skip aligned bundle persist (missing open_interest)",
                                    extra={"path": persist_path, "symbol": symbol},
                                )
                                skip_write = True
                            else:
                                oa = np.asarray(oi, dtype=np.float64).ravel()
                                if oa.size != nb:
                                    logger.warning(
                                        "TTM: skip aligned bundle persist (OI length mismatch)",
                                        extra={"path": persist_path, "symbol": symbol},
                                    )
                                    skip_write = True
                        if not skip_write:
                            try:
                                write_aligned_market_bundle(
                                    persist_path,
                                    symbol=symbol,
                                    bars=bars,
                                    basis=ba.tolist(),
                                    open_interest=oa.tolist() if oa is not None else None,
                                    profile=str(self._ttm.config.get("ttm_config_profile", "") or ""),
                                )
                            except Exception as ex:
                                logger.error(
                                    "TTM persist aligned bundle failed",
                                    extra={"error": str(ex), "path": persist_path, "symbol": symbol},
                                )
        return state

    def _emit_exit_ttm(
        self,
        symbol: str,
        pos: Any,
        current_price: float,
        sig: Optional[Dict[str, Any]],
        trigger: str,
    ) -> SignalEvent:
        reason = str((sig or {}).get("reason", "ttm_exit"))
        feat = (sig or {}).get("features") if isinstance((sig or {}).get("features"), dict) else {}
        pnl = float(pos.unrealized_pnl(current_price))
        exit_time = str(getattr(self._ohlc_history[-1], "time", "") or "") if self._ohlc_history else ""
        ent_meta = self._entry_meta_by_symbol.get(symbol) or {}
        entry_time = str(ent_meta.get("entry_time", "") or "")
        holding_period = int(self._bars_in_trade)
        meta = {
            "strategy": "TTM",
            "ttm_action": "EXIT",
            "trigger": str(trigger).upper(),
            "features": feat,
            "pnl_at_exit": pnl,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "holding_period": holding_period,
        }
        logger.info(
            "TTM (paper/ws): thoát lệnh",
            extra={
                "symbol": symbol,
                "kich_hoat": trigger,
                "ly_do": reason,
                "pnl": pnl,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "holding_period": holding_period,
                "dac_diem": feat,
            },
        )
        self._finalize_adaptive_exit(symbol, current_price, pnl, exit_time=exit_time)
        self._entry_meta_by_symbol.pop(symbol, None)

        if pos.is_long:
            return SignalEvent(
                signal=Signal.SELL,
                symbol=symbol,
                price=current_price,
                quantity=abs(pos.quantity),
                confidence=1.0,
                reason=reason,
                metadata=meta,
            )
        if pos.is_short:
            return SignalEvent(
                signal=Signal.BUY,
                symbol=symbol,
                price=current_price,
                quantity=abs(pos.quantity),
                confidence=1.0,
                reason=reason,
                metadata=meta,
            )
        raise RuntimeError("emit_exit_ttm: invalid position")

    def _maybe_exit_risk(
        self,
        symbol: str,
        pos: Any,
        current_price: float,
    ) -> Optional[SignalEvent]:
        ent_meta = self._entry_meta_by_symbol.get(symbol) or {}
        pnl = float(pos.unrealized_pnl(current_price))
        entry_price = float(getattr(pos, "avg_price", 0.0) or 0.0)
        if entry_price <= 0:
            return None
        if pos.is_long:
            pnl_ret = (float(current_price) - entry_price) / entry_price
        elif pos.is_short:
            pnl_ret = (entry_price - float(current_price)) / entry_price
        else:
            return None

        tp_ret_long = float(self._ttm.config.get("ttm_v2_tp_return", 0.0015))
        sl_ret_long = float(self._ttm.config.get("ttm_v2_sl_return", -0.0007))
        max_bars_long = int(self._ttm.config.get("ttm_v2_time_stop_bars", 4))
        tp_ret_short = float(self._ttm.config.get("ttm_v2_tp_return_short", tp_ret_long))
        sl_ret_short = float(self._ttm.config.get("ttm_v2_sl_return_short", sl_ret_long))
        max_bars_short = int(self._ttm.config.get("ttm_v2_time_stop_bars_short", max_bars_long))

        exit_needed = False
        reason = ""
        trig = ""
        if pos.is_long:
            if pnl_ret <= sl_ret_long:
                exit_needed, reason, trig = True, "ttm_exit_stop_loss", "SL"
            elif pnl_ret >= tp_ret_long:
                exit_needed, reason, trig = True, "ttm_exit_take_profit", "TP"
            elif self._bars_in_trade >= max_bars_long:
                exit_needed, reason, trig = True, "ttm_exit_max_bars", "TIME"
        elif pos.is_short:
            short_exit = check_exhaustion_short_exit(ent_meta, current_price, self._bars_in_trade)
            if short_exit is not None:
                exit_needed = True
                reason = str(short_exit["reason"])
                trig = str(short_exit["trigger"])
            elif pnl_ret <= sl_ret_short:
                exit_needed, reason, trig = True, "ttm_exit_stop_loss_short", "SL"
            elif pnl_ret >= tp_ret_short:
                exit_needed, reason, trig = True, "ttm_exit_take_profit_short", "TP"
            elif self._bars_in_trade >= max_bars_short:
                exit_needed, reason, trig = True, "ttm_exit_max_bars_short", "TIME"

        if not exit_needed:
            return None

        exit_time = str(getattr(self._ohlc_history[-1], "time", "") or "") if self._ohlc_history else ""
        entry_time = str(ent_meta.get("entry_time", "") or "")
        holding_period = int(self._bars_in_trade)
        self._finalize_adaptive_exit(symbol, current_price, pnl, exit_time=exit_time)
        self._entry_meta_by_symbol.pop(symbol, None)

        meta = {
            "strategy": "TTM",
            "ttm_action": "EXIT",
            "trigger": trig,
            "features": {},
            "pnl_at_exit": pnl,
            "pnl_return_at_exit": pnl_ret,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "holding_period": holding_period,
        }
        if pos.is_long:
            return SignalEvent(
                signal=Signal.SELL,
                symbol=symbol,
                price=current_price,
                quantity=abs(pos.quantity),
                confidence=1.0,
                reason=reason,
                metadata=meta,
            )
        if pos.is_short:
            return SignalEvent(
                signal=Signal.BUY,
                symbol=symbol,
                price=current_price,
                quantity=abs(pos.quantity),
                confidence=1.0,
                reason=reason,
                metadata=meta,
            )
        return None

    def _capture_adaptive_entry(self, symbol: str, price: float, side: str, sig: Dict[str, Any]) -> None:
        if not is_ttm_adaptive_learning_enabled(self._ttm.config):
            return
        if self._ttm.adaptive_context is None:
            return
        dbg = sig.get("debug") if isinstance(sig.get("debug"), dict) else {}
        et = ""
        if self._ohlc_history:
            et = str(getattr(self._ohlc_history[-1], "time", "") or "")
        self._adaptive_entry_by_symbol[symbol] = {
            "entry_price": float(price),
            "side": side,
            "debug": dbg,
            "entry_time": et,
        }

    def _finalize_adaptive_exit(self, symbol: str, exit_price: float, pnl: float, exit_time: str = "") -> None:
        if not is_ttm_adaptive_learning_enabled(self._ttm.config):
            logger.info(
                "TTM adaptive: learning disabled; trade exit (no weight update)",
                extra={
                    "symbol": symbol,
                    "pnl": float(pnl),
                    "exit_price": float(exit_price),
                    "ttm_adaptive_learning": False,
                },
            )
            return
        ctx = self._ttm.adaptive_context
        if ctx is None:
            return
        snap = self._adaptive_entry_by_symbol.pop(symbol, None)
        if not isinstance(snap, dict) or "debug" not in snap:
            return
        rec = build_trade_record_from_snapshot(
            snap,
            exit_price=float(exit_price),
            pnl=float(pnl),
            holding_bars=max(0, int(self._bars_in_trade)),
            exit_time=exit_time,
        )
        ctx.on_trade_closed(rec)

    def _log_signal(self, sig: Dict[str, Any], symbol: str) -> None:
        logger.info(
            "TTM (paper/ws): tín hiệu",
            extra={
                "strategy": sig.get("strategy", "TTM"),
                "action": sig.get("action"),
                "confidence": sig.get("confidence"),
                "trap_score": sig.get("trap_score"),
                "reason": sig.get("reason"),
                "symbol": symbol,
            },
        )

    def _extract_price(self, quote, trade) -> float:
        if quote is not None:
            mid = ((getattr(quote, "best_bid", 0) or 0) + (getattr(quote, "best_ask", 0) or 0))
            if mid > 0:
                return mid / 2.0
            return float(getattr(quote, "last_price", 0) or 0)
        if trade is not None:
            return float(getattr(trade, "price", 0) or 0)
        return 0.0

    @property
    def extended_stats(self) -> dict:
        """Shape compatible with paper_test status / report (MCMC fields zeroed)."""
        b = self.stats
        return {
            **b,
            "mcmc_calls": 0,
            "avg_mcmc_ms": 0.0,
            "markov_sessions": 0,
            "signals_skipped_cooldown": 0,
            "signals_skipped_risk": 0,
            "round_trips": 0,
            "t0_closed": False,
        }
