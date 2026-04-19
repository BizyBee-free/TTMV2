"""Align future and index OHLC for HMM basis = future_close − index_close (same bar)."""

from __future__ import annotations

from typing import List, Tuple

from src.backtest.data_fetcher import DataFetcher, OhlcBar, resolve_data_symbol, resolve_dnse_symbol
from src.logger import get_logger

logger = get_logger("basis_data")


def fetch_aligned_future_index(
    fetcher: DataFetcher,
    fut_symbol: str,
    idx_symbol: str,
    from_date: str,
    to_date: str,
    resolution: str,
) -> Tuple[List[OhlcBar], List[float], bool]:
    """Fetch future and index OHLC, inner-join on ``unix_ts``.

    Returns:
        (future_bars, index_closes, future_was_index_proxy) với cùng độ dài;
        ``index_closes[i]`` khớp ``future_bars[i]`` (cùng unix_ts).

        ``future_was_index_proxy=True`` khi DNSE không có OHLC phái sinh và ta
        đã copy nến chỉ số làm future → **basis_spread luôn ~ 0** (future=index).
        Gọi live nên tắt ``use_basis`` khi cờ này bật.
    """
    fut_data, fut_type = resolve_data_symbol(fut_symbol)
    fut_trade, _ = resolve_dnse_symbol(fut_symbol)
    idx_data, idx_type = resolve_data_symbol(idx_symbol)
    logger.info(
        "basis_data: data source mapping",
        extra={
            "future_input": fut_symbol,
            "future_data_symbol": fut_data,
            "future_trade_symbol": fut_trade,
            "future_type": fut_type,
            "index_input": idx_symbol,
            "index_data_symbol": idx_data,
            "index_type": idx_type,
        },
    )
    print(
        f"[Basis] Nguồn giá data Future={fut_data} (trade={fut_trade}), Index={idx_data}.",
        flush=True,
    )

    # Quan trọng: basis phải dùng 2 nguồn tách biệt (future derivative vs index),
    # không dùng proxy index cho future để tránh basis bị ghi đè về 0.
    fut = fetcher.fetch(
        fut_data,
        from_date,
        to_date,
        resolution=resolution,
        asset_type="derivative",
        proxy_derivatives=False,
    )
    idx = fetcher.fetch(
        idx_data,
        from_date,
        to_date,
        resolution=resolution,
        asset_type="index",
    )

    future_was_index_proxy = False

    if not fut:
        return [], [], future_was_index_proxy
    if not idx:
        logger.warning(
            "basis_data: index OHLC empty — cannot build basis",
            extra={"idx_symbol": idx_symbol, "fut_symbol": fut_symbol},
        )
        return [], [], future_was_index_proxy

    idx_map = {b.unix_ts: float(b.close) for b in idx}
    aligned_bars: List[OhlcBar] = []
    index_closes: List[float] = []

    for b in fut:
        c = idx_map.get(b.unix_ts)
        if c is not None:
            aligned_bars.append(b)
            index_closes.append(c)

    dropped = len(fut) - len(aligned_bars)
    if dropped > 0:
        logger.warning(
            "basis_data: dropped future bars without index match",
            extra={"dropped": dropped, "kept": len(aligned_bars), "fut_total": len(fut)},
        )

    if fut and len(aligned_bars) < len(fut) * 0.5:
        logger.warning(
            "basis_data: fewer than half of future bars matched index — check symbols/resolution",
            extra={"matched": len(aligned_bars), "fut_total": len(fut)},
        )

    if aligned_bars and index_closes:
        b_latest = float(aligned_bars[-1].close) - float(index_closes[-1])
        msg = (
            f"[Basis] Giá gần nhất: Future={aligned_bars[-1].close:.2f} "
            f"Index={index_closes[-1]:.2f} => Basis={b_latest:.2f}"
        )
        logger.info(msg)
        print(msg, flush=True)

    return aligned_bars, index_closes, future_was_index_proxy
