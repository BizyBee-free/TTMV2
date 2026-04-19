# BeeTrade -- Master Plan

> **Project:** Algorithmic Trading cho DNSE OpenAPI
> **Cập nhật:** 2026-03-25
> **Trạng thái:** GĐ1-3 HOÀN THÀNH, tiếp GĐ4

---

## Tổng quan kiến trúc

```
BeeTrade/
├── vendor/dnse/                    # DNSE Official SDK (vendored)
│   ├── dnse/                       #   REST client (DNSEClient, HMAC-SHA256)
│   └── trading_websocket/          #   WS client (TradingClient, msgpack)
├── src/
│   ├── config.py                   # Pydantic Settings, enums (OrderSide NB/NS, OrderStatus, OrderType)
│   ├── logger.py                   # JSON lines logging + rotating file
│   ├── utils.py                    # retry decorator, lot size validation, format_price
│   ├── dnse_client.py              # BeeTradeClient wrapper + TradingTokenManager (OTP flow)
│   ├── rate_limiter.py             # SlidingWindowRateLimiter (order throttling)
│   ├── data_buffer.py              # Ring buffers: QuoteBuffer, TradeBuffer, OhlcBuffer, DataBuffer
│   ├── market_data.py              # MarketDataManager (WS subscriptions, callback routing)
│   ├── order_manager.py            # OrderManager (lifecycle, paper fill, pre-trade validation)
│   ├── position_tracker.py         # PositionTracker (positions, realized/unrealized P&L, DailyPnL)
│   └── strategy_base.py            # StrategyBase ABC (on_tick, signals, execute, helpers)
├── tests/
│   ├── test_auth.py                # 7 tests  - GĐ1
│   ├── test_market_data.py         # 29 tests - GĐ2
│   ├── test_rate_limiter.py        # 18 tests - Rate limiter
│   └── test_order_manager.py       # 36 tests - GĐ3 (OrderManager + PositionTracker + StrategyBase + Integration)
├── scripts/
│   ├── check_connection.py         # GĐ1 verify
│   ├── latency_test.py             # GĐ2 WS latency measurement
│   └── rate_limit_test.py          # DNSE API rate limit discovery (multi-endpoint, progressive)
├── reports/
│   └── TEST_REPORT.md              # Báo cáo test chi tiết cho PM review
├── .env                            # Credentials (gitignored)
├── requirements.txt
└── README.md
```

---

## Tiến độ giai đoạn

### ✅ GĐ1 -- Core & Authentication (HOÀN THÀNH)
- BeeTradeClient wrapper: HMAC-SHA256 signing qua SDK, error handling, logging
- TradingTokenManager: OTP email → trading token lifecycle
- Config: Pydantic Settings, enums OrderSide(NB/NS), OrderType(LO/ATO/ATC/MTL/MOK/MAK/PLO), OrderStatus
- Tests: 7/7 PASS

### ✅ GĐ2 -- Market Data Real-time (HOÀN THÀNH)
- MarketDataManager: wrap TradingClient WS, subscribe quotes/trades/ohlc, callback routing
- DataBuffer: QuoteBuffer(latest BBO), TradeBuffer(deque maxlen=100), OhlcBuffer(deque maxlen=50, same-ts replace)
- asyncio.Lock cho thread safety, ensure_future cho async callbacks
- Tests: 29/29 PASS

### ✅ Rate Limiter (HOÀN THÀNH)
- SlidingWindowRateLimiter: deque timestamps, acquire/can_acquire/remaining/retry_after
- Tích hợp BeeTradeClient: place_order/modify_order/cancel_order đều check, read endpoints không bị limit
- Tests: 18/18 PASS

### ✅ GĐ3 -- Order Execution & Strategy Framework (HOÀN THÀNH)
- OrderManager: submit/cancel/modify, paper mode fill simulation (partial fills, weighted avg), pre-trade validation (lot size, price), callbacks
- PositionTracker: qty/avg_price per symbol, realized/unrealized P&L, DailyPnL (win/loss/rate), reset_daily
- StrategyBase ABC: on_tick → SignalEvent(BUY/SELL/HOLD) → on_signal → place_order, auto-wiring with MarketDataManager
- Tests: 36/36 PASS

### 🔲 GĐ4 -- Risk Manager & Paper Test Framework (TIẾP THEO)
**Mục tiêu:** Tránh "cháy" tài khoản do lỗi logic

**Cần tạo:**
1. `src/risk_manager.py`:
   - Pre-trade checks: kiểm tra số dư trước lệnh (GET /accounts/{id}/ppse)
   - Max position size per symbol
   - Max daily loss (config MAX_DAILY_LOSS_PCT=2.0) → halt trading khi vượt ngưỡng
   - Auto-stoploss per position (config STOPLOSS_DEFAULT_PCT=3.0)
   - Max orders per minute (đã có rate_limiter, cần wire vào risk_manager)
   - Integrate với OrderManager: reject order nếu vi phạm risk rules

2. `src/paper_engine.py`:
   - Paper trading engine: match orders against live market data
   - Auto-fill logic: LO order fills khi market price crosses order price
   - Wire MarketDataManager quotes → paper engine → OrderManager.paper_fill()
   - Simulate slippage (optional)
   - Session management: start/stop/reset

3. `tests/test_risk_manager.py`:
   - Test daily loss halt
   - Test stoploss trigger
   - Test position size limits
   - Test paper engine fill logic
   - Giả lập crash scenario → verify cắt lỗ đúng ngưỡng

**Acceptance criteria:**
- Giả lập thị trường sập mạnh → hệ thống cắt lỗ đúng ngưỡng STOPLOSS_DEFAULT_PCT
- MAX_DAILY_LOSS_PCT → halt all trading khi daily loss >= 2%
- Paper engine fill orders dựa trên real market data

### 🔲 GĐ5 -- Logging & Deployment (VPS)
**Mục tiêu:** Giám sát 24/7

**Cần tạo:**
1. Dockerfile + docker-compose.yml
2. Health check endpoint
3. Monitoring dashboard (logs/metrics)
4. Alert system (Telegram/email khi có lỗi critical)

**Acceptance criteria:**
- Chạy 4 giờ phiên (Paper mode), rà soát log không có lỗi critical

### 🔲 GĐ3b -- Advanced Strategies (SAU GĐ4+5)
**Mở rộng sau khi core ổn định:**

1. `src/strategies/mcmc_derivatives.py` -- T0 phái sinh:
   - MCMC ~10.000 paths
   - Input: giá realtime, volatility, imbalance
   - Output: BUY/SELL/HOLD + confidence threshold
   - Ràng buộc T0: mở + đóng trong phiên
   - Target: < 100ms cho 10k paths (numpy vectorized)

2. `src/strategies/cw_analytics.py` -- CW Analysis:
   - ATM/ITM/OTM classification
   - Premium/discount, delta xấp xỉ, time decay
   - Win rate lịch sử theo moneyness
   - Feed ngược cho Pair Trading strategy

3. `src/strategies/pair_trading.py` -- Pair Trading:
   - Atomic execution: CW khớp → xử lý mã cơ sở
   - Dựa trên CW analytics signals

---

## DNSE API Insights (từ stress test)

| Thông số | Giá trị |
|----------|---------|
| Market data endpoints (secdef, ohlc) | Không giới hạn (>100 req/s OK) |
| Authenticated endpoints (/accounts) | 100 req/window |
| Error code rate limit | `OA-400` + HTTP 429 (docs ghi OA-103 nhưng thực tế OA-400) |
| Cooldown sau khi bị ban | > 5 phút (có thể 10-15 phút) |
| BeeTrade config MAX_ORDERS_PER_MINUTE | 10 (an toàn, 10% ngưỡng thực tế) |

---

## Test Summary

| Giai đoạn | File test | Tests | Pass |
|-----------|-----------|-------|------|
| GĐ1: Core & Auth | test_auth.py | 7 | 7 |
| GĐ2: Market Data | test_market_data.py | 29 | 29 |
| Rate Limiter | test_rate_limiter.py | 18 | 18 |
| GĐ3: Execution | test_order_manager.py | 36 | 36 |
| **Tổng** | | **90** | **90** |

Chi tiết: `reports/TEST_REPORT.md`

---

## Config (.env)

```
DNSE_API_KEY=...
DNSE_API_SECRET=...
DNSE_ACCOUNT_NO=0003635024
DNSE_BASE_URL=https://openapi.dnse.com.vn
DNSE_WS_URL=wss://ws-openapi.dnse.com.vn
WS_ENCODING=msgpack
PAPER_MODE=true
LOG_LEVEL=INFO
MAX_DAILY_LOSS_PCT=2.0
MAX_ORDERS_PER_MINUTE=10
STOPLOSS_DEFAULT_PCT=3.0
```
