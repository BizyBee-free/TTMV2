# Prompt cho hội thoại mới -- Copy & Paste vào chat mới

---

Tiep tuc du an BeeTrade - Algorithmic Trading cho DNSE OpenAPI.

## Trang thai hien tai
- Giai doan 1-3 da HOAN THANH
- Plan chi tiet: .cursor/plans/beetrade_plan.md
- Bao cao test: reports/TEST_REPORT.md
- Codebase: d:\Chuong\BeeTrade\
- Tong tests: 90/90 PASS

## Cac file da hoan thanh:
- vendor/dnse/ -- DNSE SDK (REST: DNSEClient, WS: TradingClient)
- src/config.py -- Pydantic Settings, enums (OrderSide NB/NS, OrderStatus, OrderType)
- src/logger.py -- JSON lines logging + rotating file
- src/utils.py -- retry decorator, lot size validation
- src/dnse_client.py -- BeeTradeClient wrapper + TradingTokenManager (OTP flow)
- src/rate_limiter.py -- SlidingWindowRateLimiter (order throttling cho place/modify/cancel)
- src/data_buffer.py -- Ring buffers: QuoteBuffer, TradeBuffer, OhlcBuffer
- src/market_data.py -- MarketDataManager (WS subscribe, callback routing to buffer + strategies)
- src/order_manager.py -- OrderManager (lifecycle, paper fill simulation, pre-trade validation)
- src/position_tracker.py -- PositionTracker (qty/avg_price, realized/unrealized P&L, DailyPnL)
- src/strategy_base.py -- StrategyBase ABC (on_tick -> SignalEvent -> on_signal -> place_order)
- tests/ -- 4 test files, 90 tests ALL PASS
- scripts/ -- check_connection.py, latency_test.py, rate_limit_test.py

## Nhiem vu: Thuc hien Giai doan 4 (Risk Manager & Paper Test Framework)

Doc .cursor/plans/beetrade_plan.md section "GD4" de lay chi tiet, sau do tao:

1. src/risk_manager.py -- Pre-trade risk checks:
   - Max daily loss (MAX_DAILY_LOSS_PCT=2.0) -> halt trading khi vuot nguong
   - Auto-stoploss per position (STOPLOSS_DEFAULT_PCT=3.0)
   - Max position size per symbol
   - Integrate voi OrderManager: reject order neu vi pham risk rules
   - Wire voi PositionTracker de monitor P&L real-time

2. src/paper_engine.py -- Paper trading engine:
   - Auto-fill logic: LO order fills khi market price crosses order price
   - Wire MarketDataManager quotes -> paper engine -> OrderManager.paper_fill()
   - Session management: start/stop/reset
   - Fill simulation dua tren live market data tu DataBuffer

3. tests/test_risk_manager.py -- Unit tests:
   - Test daily loss halt
   - Test stoploss trigger
   - Test position size limits
   - Test paper engine fill logic
   - Gia lap crash scenario -> verify cat lo dung nguong

4. Cap nhat reports/TEST_REPORT.md voi ket qua test GD4

Luu y ky thuat:
- Config co san trong .env: MAX_DAILY_LOSS_PCT=2.0, STOPLOSS_DEFAULT_PCT=3.0, PAPER_MODE=true
- OrderManager da co paper_fill() method san
- PositionTracker da co daily_pnl, unrealized_pnl(), reset_daily()
- MarketDataManager co on_quote/on_trade callback registration
- Acceptance: gia lap thi truong sap manh -> he thong cat lo dung nguong

---
