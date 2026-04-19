"""Bar replay for parameterized TTM-opt strategy (fees, slippage, SL/TP/trailing)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.backtest.bar_replay import TradeRecord
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.backtest.ttm_opt.features import compute_feature_bundle, entry_masks_long_short
from src.backtest.ttm_opt.params import TtmOptParams
from src.backtest.ttm_opt.regimes import label_regimes, regime_name


def _slip_in_long_entry(p: float, slip: float) -> float:
    return p * (1.0 + slip)


def _slip_out_long_exit(p: float, slip: float) -> float:
    return p * (1.0 - slip)


def _slip_in_short_entry(p: float, slip: float) -> float:
    return p * (1.0 - slip)


def _slip_out_short_exit(p: float, slip: float) -> float:
    return p * (1.0 + slip)


def run_ttm_opt_backtest(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    time_labels: List[str],
    basis: np.ndarray,
    oi: np.ndarray,
    p: TtmOptParams,
    *,
    bar_type: str = "15",
    nav_start: float = 100.0,
    entry_delay_bars: int = 0,
) -> Tuple[List[TradeRecord], np.ndarray, Dict[str, Any]]:
    """
    Signal at close of bar ``sig_i``; entry at open of bar ``sig_i + 1 + entry_delay_bars``.

    Returns:
        trade_log, equity_curve (cum net PnL per bar), extras (regime breakdown, etc.)
    """
    n = len(close)
    slip = float(p.slippage_pct)
    comm = float(p.commission_pct)
    eq = np.zeros(n, dtype=np.float64)
    trades: List[TradeRecord] = []

    feats = compute_feature_bundle(close, high, low, basis, oi, p)
    long_m, short_m = entry_masks_long_short(feats, p)
    trend_tag, vol_tag = label_regimes(feats, p)

    warmup = max(int(p.warmup_bars), 50)
    pos: Optional[str] = None  # "LONG" / "SHORT"
    entry_i = -1
    entry_px = 0.0
    peak_px = 0.0
    trough_px = 0.0
    cum = 0.0
    regime_pnl: Dict[str, float] = {}

    sl_pct = float(p.stop_loss_pct)
    tp_pct = float(p.take_profit_pct)
    trail = bool(p.trailing_stop)
    trail_pct = float(p.trailing_pct)

    i = warmup
    while i < n:
        o = float(open_[i])
        h = float(high[i])
        l = float(low[i])
        c = float(close[i])

        # --- Manage open position (stops use bar i range) ---
        if pos == "LONG":
            stop_lv = entry_px * (1.0 - sl_pct)
            tp_lv = entry_px * (1.0 + tp_pct)
            if trail:
                peak_px = max(peak_px, h)
                stop_lv = max(stop_lv, peak_px * (1.0 - trail_pct))
            exit_px = None
            reason = ""
            if l <= stop_lv:
                exit_px = _slip_out_long_exit(min(stop_lv, o), slip)
                reason = "sl"
            elif h >= tp_lv:
                exit_px = _slip_out_long_exit(max(tp_lv, o), slip)
                reason = "tp"
            if exit_px is not None:
                gross = exit_px - entry_px
                fee = comm * (abs(entry_px) + abs(exit_px))
                net = gross - fee
                cum += net
                eq[i] = cum
                ei = min(entry_i, n - 1)
                xei = min(i, n - 1)
                t0 = time_labels[ei] if ei < len(time_labels) else str(ei)
                t1 = time_labels[xei] if xei < len(time_labels) else str(xei)
                rn = regime_name(int(trend_tag[entry_i]), int(vol_tag[entry_i]))
                regime_pnl[rn] = regime_pnl.get(rn, 0.0) + net
                trades.append(
                    TradeRecord(
                        symbol="TTM_OPT",
                        entry_bar=entry_i,
                        exit_bar=i,
                        entry_date=t0,
                        exit_date=t1,
                        side="BUY",
                        entry_price=entry_px,
                        exit_price=exit_px,
                        pnl=gross,
                        commission=fee,
                        net_pnl=net,
                        p_up=1.0 if reason == "tp" else 0.0,
                        p_down=1.0 if reason == "sl" else 0.0,
                        confidence=0.0,
                        mcmc_elapsed_ms=0.0,
                    )
                )
                pos = None
                i += 1
                continue

        elif pos == "SHORT":
            stop_lv = entry_px * (1.0 + sl_pct)
            tp_lv = entry_px * (1.0 - tp_pct)
            if trail:
                trough_px = min(trough_px, l)
                stop_lv = min(stop_lv, trough_px * (1.0 + trail_pct))
            exit_px = None
            if h >= stop_lv:
                exit_px = _slip_out_short_exit(max(stop_lv, o), slip)
            elif l <= tp_lv:
                exit_px = _slip_out_short_exit(min(tp_lv, o), slip)
            if exit_px is not None:
                gross = entry_px - exit_px
                fee = comm * (abs(entry_px) + abs(exit_px))
                net = gross - fee
                cum += net
                eq[i] = cum
                ei = min(entry_i, n - 1)
                xei = min(i, n - 1)
                t0 = time_labels[ei] if ei < len(time_labels) else str(ei)
                t1 = time_labels[xei] if xei < len(time_labels) else str(xei)
                rn = regime_name(int(trend_tag[entry_i]), int(vol_tag[entry_i]))
                regime_pnl[rn] = regime_pnl.get(rn, 0.0) + net
                trades.append(
                    TradeRecord(
                        symbol="TTM_OPT",
                        entry_bar=entry_i,
                        exit_bar=i,
                        entry_date=t0,
                        exit_date=t1,
                        side="SELL",
                        entry_price=entry_px,
                        exit_price=exit_px,
                        pnl=gross,
                        commission=fee,
                        net_pnl=net,
                        p_up=0.0,
                        p_down=0.0,
                        confidence=0.0,
                        mcmc_elapsed_ms=0.0,
                    )
                )
                pos = None
                i += 1
                continue

        # --- New signals (flat): decided on previous bar close ---
        sig_i = i - 1 - max(0, int(entry_delay_bars))
        if sig_i >= warmup and pos is None:
            want_long = bool(long_m[sig_i])
            want_short = bool(short_m[sig_i])
            if want_long and not want_short:
                entry_px = _slip_in_long_entry(o, slip)
                pos = "LONG"
                entry_i = i
                peak_px = h
            elif want_short and not want_long:
                entry_px = _slip_in_short_entry(o, slip)
                pos = "SHORT"
                entry_i = i
                trough_px = l

        eq[i] = cum
        i += 1

    # Mark-to-close last open position
    if pos is not None and n > 0:
        last = float(close[-1])
        if pos == "LONG":
            exit_px = _slip_out_long_exit(last, slip)
            gross = exit_px - entry_px
        else:
            exit_px = _slip_out_short_exit(last, slip)
            gross = entry_px - exit_px
        fee = comm * (abs(entry_px) + abs(exit_px))
        net = gross - fee
        cum += net
        eq[-1] = cum
        ei = min(entry_i, n - 1)
        t0 = time_labels[ei] if ei < len(time_labels) else str(ei)
        t1 = time_labels[-1] if time_labels else str(n - 1)
        rn = regime_name(int(trend_tag[entry_i]), int(vol_tag[entry_i]))
        regime_pnl[rn] = regime_pnl.get(rn, 0.0) + net
        trades.append(
            TradeRecord(
                symbol="TTM_OPT",
                entry_bar=entry_i,
                exit_bar=n - 1,
                entry_date=t0,
                exit_date=t1,
                side="BUY" if pos == "LONG" else "SELL",
                entry_price=entry_px,
                exit_price=exit_px,
                pnl=gross,
                commission=fee,
                net_pnl=net,
                p_up=0.0,
                p_down=0.0,
                confidence=0.0,
                mcmc_elapsed_ms=0.0,
            )
        )

    metrics = compute_metrics(trades, eq, n_bars=n, bar_type=bar_type, nav_start=nav_start)
    extras = {
        "regime_pnl": regime_pnl,
        "features_meta": {"n_trades": len(trades)},
    }
    return trades, eq, {"metrics": metrics, **extras}


def metrics_from_result(res: Dict[str, Any]) -> BacktestMetrics:
    return res["metrics"]
