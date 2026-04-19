"""Vòng lặp live 15m trong phiên VN (sáng + chiều), dừng khi hết phiên (mặc định 15:00).

Dùng chung cho ``scripts/hmm_live.py`` (chế độ phiên) và ``scripts/hmm_live_session.py``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from typing import TYPE_CHECKING, Union

from src.logger import get_logger
from src.strategy_factory import live_session_cli_prefix, live_session_logger_name

if TYPE_CHECKING:
    from src.config import Settings
    from src.live.hmm_live_runner import HmmLiveRunner
    from src.live.ttm_live_runner import TtmLiveRunner

Runner = Union["HmmLiveRunner", "TtmLiveRunner"]


def _vietnam_tz():
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo("Asia/Ho_Chi_Minh")
    except Exception:
        return timezone(timedelta(hours=7))


VN = _vietnam_tz()


def _combine(d: date, t: dtime) -> datetime:
    return datetime.combine(d, t, tzinfo=VN)


def _floor_15m(dt: datetime) -> datetime:
    m = (dt.minute // 15) * 15
    return dt.replace(minute=m, second=0, microsecond=0)


def _next_1m_after(dt: datetime) -> datetime:
    flo = dt.replace(second=0, microsecond=0)
    if flo <= dt:
        flo = flo + timedelta(minutes=1)
    return flo


def _skip_lunch(dt: datetime, lunch_start: dtime, lunch_end: dtime) -> datetime:
    if lunch_start <= dt.time() < lunch_end:
        return dt.replace(
            hour=lunch_end.hour,
            minute=lunch_end.minute,
            second=0,
            microsecond=0,
        )
    return dt


def _sleep_until(target: datetime) -> None:
    now = datetime.now(VN)
    sec = (target - now).total_seconds()
    if sec > 0:
        time.sleep(sec)


def _parse_hhmm(s: str) -> dtime:
    a, b = s.strip().split(":")
    return dtime(int(a), int(b))


@dataclass(frozen=True)
class Session15mLoopParams:
    """Thời gian phiên VN (Asia/Ho_Chi_Minh)."""

    morning_start: str = "09:00"
    morning_end: str = "11:30"
    afternoon_start: str = "13:00"
    session_end: str = "15:00"
    bar_delay: float = 2.0
    duplicate_retry_seconds: float = 60.0


def run_15m_session_loop(
    *,
    settings: "Settings",
    runner: Runner,
    symbol: str,
    days_back: int,
    submit: bool,
    params: Session15mLoopParams | None = None,
) -> int:
    """
    Poll mỗi ~1 phút trong giờ sàn; mỗi tick gọi ``runner.run_once``.
    Thoát khi ``now.time() >= session_end`` (mặc định 15:00) hoặc qua ngày mới.
    """
    pp = params or Session15mLoopParams()
    morning_start = _parse_hhmm(pp.morning_start)
    lunch_start = _parse_hhmm(pp.morning_end)
    afternoon_start = _parse_hhmm(pp.afternoon_start)
    session_end = _parse_hhmm(pp.session_end)
    duplicate_retry_seconds = max(1.0, float(pp.duplicate_retry_seconds))

    algo = settings.STRATEGY_ALGO.value
    session_logger = get_logger(live_session_logger_name(settings))
    px = live_session_cli_prefix(settings)
    session_day = datetime.now(VN).date()

    now0 = datetime.now(VN)
    if now0.time() >= session_end:
        print(f"{px} Đã quá giờ kết phiên hôm nay — thoát.", flush=True)
        return 0

    print(
        f"{px} Ngày {session_day} (VN) | "
        f"{pp.morning_start}-{pp.morning_end}, nghỉ, {pp.afternoon_start}-{pp.session_end} | "
        f"symbol={symbol} submit={submit} days={days_back}",
        flush=True,
    )
    session_logger.info(
        f"{algo} live session started",
        extra={
            "strategy_algo": algo,
            "session_day": str(session_day),
            "symbol": symbol,
            "submit": submit,
            "days_back": days_back,
            "morning_start": pp.morning_start,
            "morning_end": pp.morning_end,
            "afternoon_start": pp.afternoon_start,
            "session_end": pp.session_end,
            "debug_trace": settings.HMM_LIVE_DEBUG_TRACE,
            "paper_mode": settings.PAPER_MODE,
        },
    )

    while True:
        now = datetime.now(VN)
        if now.date() != session_day:
            print(f"{px} Qua ngày mới — kết thúc phiên.", flush=True)
            break
        if now.time() >= session_end:
            print(f"{px} Hết phiên ({pp.session_end}).", flush=True)
            break

        if now.time() < morning_start:
            t = _combine(session_day, morning_start)
            print(f"{px} Chờ mở phiên {t} …", flush=True)
            _sleep_until(t)
            continue

        if lunch_start <= now.time() < afternoon_start:
            t = _combine(session_day, afternoon_start)
            print(f"{px} Nghỉ trưa — chờ {t} …", flush=True)
            _sleep_until(t)
            continue

        in_morning = morning_start <= now.time() < lunch_start
        in_afternoon = afternoon_start <= now.time() < session_end
        if not (in_morning or in_afternoon):
            time.sleep(1.0)
            continue

        time.sleep(max(0.0, pp.bar_delay))
        slot = _floor_15m(now)
        print(
            f"{px} Tick {now.strftime('%H:%M:%S')} (quyết định theo nến 15m {slot.strftime('%H:%M')}) — run_once …",
            flush=True,
        )
        try:
            result = runner.run_once(symbol=symbol, days_back=days_back, submit_order=submit)
        except Exception as e:
            session_logger.exception(
                f"{algo} session tick crashed",
                extra={
                    "strategy_algo": algo,
                    "slot": slot.strftime("%H:%M"),
                    "error": str(e),
                },
            )
            print(f"  ok=False detail=tick_exception err={e}", flush=True)
            result = None

        if result is not None and result.detail == "duplicate_bar":
            print(
                f"  detail=duplicate_bar -> thử gọi lại API sau {int(duplicate_retry_seconds)} giây …",
                flush=True,
            )
            time.sleep(duplicate_retry_seconds)
            try:
                retry = runner.run_once(symbol=symbol, days_back=days_back, submit_order=submit)
                print(
                    f"  retry ok={retry.ok} detail={retry.detail} dir={retry.direction} "
                    f"label={retry.state_label} order={retry.order_id}",
                    flush=True,
                )
                session_logger.info(
                    f"{algo} session duplicate retry result",
                    extra={
                        "strategy_algo": algo,
                        "ok": retry.ok,
                        "detail": retry.detail,
                        "direction": retry.direction,
                        "label": retry.state_label,
                        "order_id": retry.order_id,
                        "root_cause": retry.root_cause,
                    },
                )
                result = retry
            except Exception as e:
                session_logger.exception(
                    f"{algo} duplicate retry crashed",
                    extra={"strategy_algo": algo, "error": str(e)},
                )
                print(f"  retry ok=False detail=tick_exception err={e}", flush=True)
                result = None

        if result is None:
            nxt = _next_1m_after(datetime.now(VN))
            nxt = _skip_lunch(nxt, lunch_start, afternoon_start)
            if nxt.time() >= session_end or nxt.date() > session_day:
                break
            _sleep_until(nxt)
            continue
        print(
            f"  ok={result.ok} detail={result.detail} dir={result.direction} "
            f"label={result.state_label} order={result.order_id}",
            flush=True,
        )
        session_logger.info(
            f"{algo} session tick result",
            extra={
                "strategy_algo": algo,
                "slot": slot.strftime("%H:%M"),
                "ok": result.ok,
                "detail": result.detail,
                "direction": result.direction,
                "label": result.state_label,
                "order_id": result.order_id,
                "root_cause": result.root_cause,
            },
        )
        nxt = _next_1m_after(datetime.now(VN))
        nxt = _skip_lunch(nxt, lunch_start, afternoon_start)
        if nxt.time() >= session_end or nxt.date() > session_day:
            break
        _sleep_until(nxt)

    session_logger.info(
        f"{algo} live session ended",
        extra={"strategy_algo": algo, "session_day": str(session_day)},
    )
    return 0
