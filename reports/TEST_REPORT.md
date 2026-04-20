# BeeTrade - Test Report

> **Project:** BeeTrade - Algorithmic Trading cho DNSE OpenAPI
> **Bắt đầu:** 2026-03-24
> **Cập nhật lần cuối:** 2026-04-19 (TTM V2 `effective_strength` calibration layer)
> **Tổng test cases:** 299+ unit tests (see sections below) + 1 live stress test
> **Trạng thái tổng thể:** ✅ ON TRACK

---

## Mục lục

1. [Giai đoạn 1 -- Core & Authentication](#1-giai-đoạn-1----core--authentication)
2. [Giai đoạn 2 -- Market Data Real-time](#2-giai-đoạn-2----market-data-real-time)
3. [Rate Limiter -- Order Throttling](#3-rate-limiter----order-throttling)
4. [DNSE API Stress Test -- Rate Limit Discovery](#4-dnse-api-stress-test----rate-limit-discovery)
5. [Giai đoạn 3 -- Order Execution & Strategy Framework](#5-giai-đoạn-3----order-execution--strategy-framework)
6. [Giai đoạn 3b -- MCMC Derivatives Core Trading](#6-giai-đoạn-3b----mcmc-derivatives-core-trading)

6c. [Config-driven derivative exit rules](#6c-config-driven-derivative-exit-rules-2026-03-31)
7. [Giai đoạn 4 -- Backtest Framework](#7-giai-đoạn-4----backtest-framework-mcmc-strategy-validation)
8. [Quy ước báo cáo test](#8-quy-ước-báo-cáo-test)

---

## 1. Giai đoạn 1 -- Core & Authentication

**Ngày test:** 2026-03-24
**File test:** `tests/test_auth.py`
**Kết quả:** 7/7 PASS

### 1.1 Test environment


| Thành phần | Chi tiết                |
| ---------- | ----------------------- |
| Python     | 3.10.7                  |
| OS         | Windows 10 (10.0.19045) |
| pytest     | 9.0.2                   |
| Loại test  | Unit test (mocked SDK)  |


### 1.2 Test cases


| #   | Test case                 | Module              | Mô tả                                                    | Kết quả |
| --- | ------------------------- | ------------------- | -------------------------------------------------------- | ------- |
| 1   | `test_initial_state`      | TradingTokenManager | Token khởi tạo = None, is_valid = False                  | ✅ PASS  |
| 2   | `test_request_otp`        | TradingTokenManager | Gọi send_email_otp(), verify status 200                  | ✅ PASS  |
| 3   | `test_activate_token`     | TradingTokenManager | Đổi OTP lấy trading token, verify token value + is_valid | ✅ PASS  |
| 4   | `test_set_token_manually` | TradingTokenManager | Set token thủ công, verify token + is_valid              | ✅ PASS  |
| 5   | `test_get_accounts`       | BeeTradeClient      | GET /accounts -> status 200, parse JSON list             | ✅ PASS  |
| 6   | `test_get_balances`       | BeeTradeClient      | GET /accounts/{id}/balances -> status 200, parse JSON    | ✅ PASS  |
| 7   | `test_api_error_handling` | BeeTradeClient      | Status 401 + error code OA-101 -> xử lý đúng             | ✅ PASS  |


### 1.3 Đánh giá

- Authentication flow (HMAC-SHA256 signing) hoạt động tốt qua SDK
- Trading Token lifecycle (request OTP → activate → validate TTL) đã cover
- Error handling cho HTTP 4xx đã verify
- **Chưa test:** Token expiry tự động (cần mock time), OTP flow end-to-end (cần email thật)

---

## 2. Giai đoạn 2 -- Market Data Real-time

**Ngày test:** 2026-03-25
**File test:** `tests/test_market_data.py`
**Kết quả:** 29/29 PASS

### 2.1 Test environment


| Thành phần | Chi tiết                         |
| ---------- | -------------------------------- |
| Python     | 3.10.7                           |
| pytest     | 9.0.2 + pytest-asyncio 1.3.0     |
| Loại test  | Unit test (mocked WebSocket SDK) |


### 2.2 Test cases -- QuoteBuffer (6 tests)


| #   | Test case                       | Mô tả                                                  | Kết quả |
| --- | ------------------------------- | ------------------------------------------------------ | ------- |
| 1   | `test_update_and_get`           | Update quote HPG, verify best_bid/best_ask/symbol      | ✅ PASS  |
| 2   | `test_get_missing_returns_none` | Get symbol không tồn tại -> None                       | ✅ PASS  |
| 3   | `test_overwrites_on_update`     | Update 2 lần cùng symbol -> giữ giá mới, count vẫn = 1 | ✅ PASS  |
| 4   | `test_multiple_symbols`         | Update HPG + VNM -> count=2, symbols đúng              | ✅ PASS  |
| 5   | `test_get_all`                  | Get all quotes -> dict đầy đủ                          | ✅ PASS  |
| 6   | `test_spread_property`          | bid=25.0, ask=25.5 -> spread=0.5                       | ✅ PASS  |


### 2.3 Test cases -- TradeBuffer (5 tests)


| #   | Test case                     | Mô tả                                                | Kết quả |
| --- | ----------------------------- | ---------------------------------------------------- | ------- |
| 1   | `test_append_and_get_latest`  | Append 2 trades -> get_latest trả trade cuối         | ✅ PASS  |
| 2   | `test_get_latest_missing`     | Symbol không có -> None                              | ✅ PASS  |
| 3   | `test_ring_buffer_maxlen`     | maxlen=3, append 5 -> giữ 3 mới nhất (eviction đúng) | ✅ PASS  |
| 4   | `test_get_history_with_limit` | 7 trades, get last 3 -> đúng 3 trade cuối            | ✅ PASS  |
| 5   | `test_multiple_symbols`       | HPG + VNM riêng biệt, count đúng từng symbol         | ✅ PASS  |


### 2.4 Test cases -- OhlcBuffer (4 tests)


| #   | Test case                      | Mô tả                                                   | Kết quả |
| --- | ------------------------------ | ------------------------------------------------------- | ------- |
| 1   | `test_append_and_get_latest`   | 2 candles khác timestamp -> latest = candle cuối        | ✅ PASS  |
| 2   | `test_replaces_same_timestamp` | 2 candles cùng timestamp -> overwrite (không duplicate) | ✅ PASS  |
| 3   | `test_ring_buffer_maxlen`      | maxlen=3, append 5 -> giữ 3 mới nhất                    | ✅ PASS  |
| 4   | `test_get_latest_missing`      | Symbol không có -> None                                 | ✅ PASS  |


### 2.5 Test cases -- DataBuffer facade (7 tests)


| #   | Test case                        | Mô tả                                                  | Kết quả |
| --- | -------------------------------- | ------------------------------------------------------ | ------- |
| 1   | `test_on_quote_routes_to_buffer` | on_quote() -> get_latest_quote() trả đúng              | ✅ PASS  |
| 2   | `test_on_trade_routes_to_buffer` | on_trade() -> get_latest_trade() trả đúng              | ✅ PASS  |
| 3   | `test_on_ohlc_routes_to_buffer`  | on_ohlc() -> get_ohlc_history() có 1 entry             | ✅ PASS  |
| 4   | `test_update_count`              | 3 updates -> update_count = 3                          | ✅ PASS  |
| 5   | `test_stats`                     | Mixed updates -> stats().total_quotes/trades/ohlc đúng | ✅ PASS  |
| 6   | `test_get_all_quotes`            | 2 symbols -> get_all_quotes() trả dict 2 entries       | ✅ PASS  |
| 7   | `test_trade_history`             | maxlen=5, append 7 -> history=5, last 2 đúng           | ✅ PASS  |


### 2.6 Test cases -- MarketDataManager (7 tests)


| #   | Test case                        | Mô tả                                                   | Kết quả |
| --- | -------------------------------- | ------------------------------------------------------- | ------- |
| 1   | `test_connect_disconnect`        | connect -> is_connected=True, disconnect -> False       | ✅ PASS  |
| 2   | `test_subscribe_quotes`          | subscribe_quotes(["HPG","VNM"]) -> SDK gọi đúng symbols | ✅ PASS  |
| 3   | `test_subscribe_dedup`           | Subscribe cùng symbol 2 lần -> SDK chỉ gọi 1 lần        | ✅ PASS  |
| 4   | `test_callback_routes_to_buffer` | _on_quote() -> buffer có data (ensure_future hoạt động) | ✅ PASS  |
| 5   | `test_external_callbacks`        | Register callback -> nhận đúng 2 quotes                 | ✅ PASS  |
| 6   | `test_msg_count`                 | 3 messages (quote+trade+ohlc) -> msg_count = 3          | ✅ PASS  |
| 7   | `test_get_subscriptions`         | Verify subscription tracking đúng per channel           | ✅ PASS  |


### 2.7 Đánh giá

- Tất cả buffer operations (CRUD, eviction, concurrency lock) hoạt động đúng
- WebSocket SDK wrapping đã verify qua mocked TradingClient
- Callback routing (sync + async via ensure_future) đã cover
- **Chưa test live:** Chạy `scripts/latency_test.py` trong giờ giao dịch (09:00-14:45) để đo throughput thực

### 2.8 VN30 KRX front-month auto-resolve (mock)

**Ngày test:** 2026-04-17
**File test:** `tests/test_vn30_contract_resolve.py`, `tests/conftest.py`
**Kết quả:** 7/7 PASS (`pytest tests/test_vn30_contract_resolve.py -v`)


| #   | Test case                                             | Mô tả                                                                        | Kết quả |
| --- | ----------------------------------------------------- | ---------------------------------------------------------------------------- | ------- |
| 1   | `test_calendar_krx_matches_repo_examples`             | Heuristic `41I1G4000/5000/6000` khớp ví dụ repo                              | ✅ PASS  |
| 2   | `test_third_thursday_april_2026`                      | Tuần 3 thứ Năm tháng 4 nằm 15–21                                             | ✅ PASS  |
| 3   | `test_resolve_updates_when_api_matches_calendar`      | API + lịch cùng `41I1G5000` → cập nhật `SYMBOL_MAP`                          | ✅ PASS  |
| 4   | `test_resolve_exits_on_mismatch`                      | API ≠ calendar → `SystemExit(1)`                                             | ✅ PASS  |
| 5   | `test_resolve_skipped_when_disabled`                  | `VN30_AUTO_RESOLVE_TRADE_SYMBOL=false` → không gọi API                       | ✅ PASS  |
| 6   | `test_env_trade_symbol_pins_without_api`              | `VN30_F1M_TRADE_SYMBOL` trên settings → map + resolved, không `/instruments` | ✅ PASS  |
| 7   | `test_env_trade_symbol_overrides_auto_resolve_no_api` | Cùng env pin + auto-resolve bật → vẫn không gọi API                          | ✅ PASS  |


**Ghi chú:** `tests/conftest.py` tự động set `VN30_AUTO_RESOLVE_TRADE_SYMBOL=false` và xóa `VN30_F1M_TRADE_SYMBOL` khỏi env cho mọi test. `get_settings()` + `BeeTradeClient` áp `VN30_F1M_TRADE_SYMBOL` từ `.env` vào `SYMBOL_MAP` (live + paper). `paper_test_prime_vn30f1m_krx` reset cờ + resolve trước subscribe cho `paper_test --symbol VN30F1M`.

**Đánh giá:** Chưa có integration live `/instruments` thật; heuristic tháng HNX có rủi ro sai — cần đối chiếu DNSE khi rollover.

---

## 3. Rate Limiter -- Order Throttling

**Ngày test:** 2026-03-25
**File test:** `tests/test_rate_limiter.py`
**Kết quả:** 18/18 PASS

### 3.1 Test cases -- SlidingWindowRateLimiter (12 tests)


| #   | Test case                           | Mô tả                                               | Kết quả |
| --- | ----------------------------------- | --------------------------------------------------- | ------- |
| 1   | `test_allows_up_to_limit`           | limit=3, acquire 3 lần -> count=3, không raise      | ✅ PASS  |
| 2   | `test_blocks_over_limit`            | limit=3, acquire 4 lần -> RateLimitExceeded ở lần 4 | ✅ PASS  |
| 3   | `test_window_slides`                | limit=2 window=10s, 2 req ở t=100 -> ok ở t=110.1   | ✅ PASS  |
| 4   | `test_can_acquire_does_not_consume` | can_acquire() check không tiêu slot                 | ✅ PASS  |
| 5   | `test_remaining_property`           | limit=5, 2 acquired -> remaining=3                  | ✅ PASS  |
| 6   | `test_reset`                        | reset() xoá hết timestamps                          | ✅ PASS  |
| 7   | `test_retry_after`                  | Window full -> retry_after trả đúng số giây còn lại | ✅ PASS  |
| 8   | `test_retry_after_when_available`   | Có slot trống -> retry_after = 0                    | ✅ PASS  |
| 9   | `test_invalid_limit`                | limit=0 -> ValueError                               | ✅ PASS  |
| 10  | `test_invalid_window`               | window=0 -> ValueError                              | ✅ PASS  |
| 11  | `test_exception_message`            | RateLimitExceeded message chứa limit + retry info   | ✅ PASS  |
| 12  | `test_burst_at_boundary`            | 3 req cùng timestamp -> limit=3 ok, lần 4 bị block  | ✅ PASS  |


### 3.2 Test cases -- BeeTradeClient integration (6 tests)


| #   | Test case                          | Mô tả                                                | Kết quả |
| --- | ---------------------------------- | ---------------------------------------------------- | ------- |
| 1   | `test_rate_limiter_initialized`    | Config MAX_ORDERS_PER_MINUTE=3 -> limiter.limit=3    | ✅ PASS  |
| 2   | `test_place_order_blocked`         | 3 place_order OK, lần 4 -> RateLimitExceeded         | ✅ PASS  |
| 3   | `test_modify_order_consumes`       | 3 modify_order OK, lần 4 -> RateLimitExceeded        | ✅ PASS  |
| 4   | `test_cancel_order_consumes`       | 3 cancel_order OK, lần 4 -> RateLimitExceeded        | ✅ PASS  |
| 5   | `test_mixed_actions_share_limiter` | place+modify+cancel = 3 -> lần 4 bất kỳ bị block     | ✅ PASS  |
| 6   | `test_read_endpoints_not_limited`  | 20x get_accounts + get_balances -> limiter untouched | ✅ PASS  |


### 3.3 Đánh giá

- Sliding window algorithm chính xác: cho phép đúng N actions trong window, tự mở khi entries cũ expire
- Integration với BeeTradeClient: place/modify/cancel chia chung 1 limiter, read endpoints không bị ảnh hưởng
- `RateLimitExceeded` exception có đầy đủ thông tin (limit, window, retry_after) cho caller xử lý

---

## 4. DNSE API Stress Test -- Rate Limit Discovery

**Ngày test:** 2026-03-25 (06:50 - 07:00 UTC)
**Script:** `scripts/rate_limit_test.py`
**Loại test:** Live test trực tiếp lên DNSE production API
**Mục tiêu:** Tìm ngưỡng rate limit thực tế của DNSE OpenAPI

### 4.1 Phương pháp

Giảm dần khoảng cách giữa 2 request (từ 500ms → 0ms) cho đến khi API trả lỗi rate limit. Sử dụng endpoint GET read-only để tránh ảnh hưởng tài khoản.

### 4.2 Kết quả -- Endpoint `/price/secdef/{symbol}` (Market Data)

**Kết luận: KHÔNG phát hiện rate limit**


| Interval | Burst   | OK/Total    | Avg Latency | Rate Limited |
| -------- | ------- | ----------- | ----------- | ------------ |
| 500ms    | 5       | 5/5         | 11ms        | Không        |
| 400ms    | 5       | 5/5         | 13ms        | Không        |
| 300ms    | 5       | 5/5         | 13ms        | Không        |
| 200ms    | 5       | 5/5         | 12ms        | Không        |
| 100ms    | 5       | 5/5         | 12ms        | Không        |
| 50ms     | 5       | 5/5         | 12ms        | Không        |
| 40ms     | 15      | 15/15       | 7ms         | Không        |
| 20ms     | 15      | 15/15       | 7ms         | Không        |
| 5ms      | 15      | 15/15       | 8ms         | Không        |
| **0ms**  | **15**  | **15/15**   | **7ms**     | **Không**    |
| **0ms**  | **100** | **100/100** | **14ms**    | **Không**    |


> Market data endpoints không áp dụng rate limit hoặc ngưỡng rất cao (>100 req/s).

### 4.3 Kết quả -- Endpoint `/accounts` (Authenticated)

**Kết luận: Rate limit = 100 requests/window, cooldown > 5 phút**


| Interval | Burst   | OK/Total    | Avg Latency | Rate Limited     |
| -------- | ------- | ----------- | ----------- | ---------------- |
| **0ms**  | **200** | **100/200** | **7ms**     | **CÓ - từ #101** |


Chi tiết request bị rate limit:

```
Request #1   → #100: HTTP 200 OK          (avg 8ms)
Request #101 → #200: HTTP 429 [OA-400]    (avg 7ms)
```

**Cooldown observation:**


| Thời gian sau khi bị ban | Probe /accounts | Kết quả    |
| ------------------------ | --------------- | ---------- |
| +35 giây                 | 1 request       | 429 OA-400 |
| +2.5 phút                | 1 request       | 429 OA-400 |
| +5 phút                  | 1 request       | 429 OA-400 |
| +7 phút (ước tính)       | Chưa test       | -          |


> **Quan trọng:** Cooldown rất dài (>5 phút). Một khi bị ban, toàn bộ API key bị block trên endpoint đó.

### 4.4 Endpoint đặt lệnh (Order)

**Chưa test trực tiếp** vì:

- Cần Trading Token (OTP qua email) -- không thể tự động hoá
- Đặt lệnh thật có thể ảnh hưởng tài khoản

**Suy luận:** Write endpoints (POST/PUT/DELETE) nhiều khả năng có rate limit bằng hoặc thấp hơn `/accounts` (100 req/window).

### 4.5 Phát hiện quan trọng


| #   | Phát hiện                                                                        | Tác động                                                |
| --- | -------------------------------------------------------------------------------- | ------------------------------------------------------- |
| 1   | Error code thực tế là `OA-400` + HTTP 429, không phải `OA-103` như docs DNSE ghi | Cần handle cả hai code trong production                 |
| 2   | Market data endpoints không bị rate limit                                        | Có thể poll market data tần suất cao mà không lo        |
| 3   | Authenticated endpoints limit = 100 req/window                                   | Config MAX_ORDERS_PER_MINUTE=10 là an toàn (10% ngưỡng) |
| 4   | Cooldown sau khi vi phạm > 5 phút                                                | **NGUY HIỂM** -- nếu bị ban thì mất quyền giao dịch lâu |
| 5   | Rate limit theo endpoint, không phải global                                      | `/price/secdef` vẫn hoạt động khi `/accounts` bị ban    |


### 4.6 Khuyến nghị cho PM

1. **Giữ MAX_ORDERS_PER_MINUTE = 10** (hiện tại) -- rất an toàn, chỉ 10% ngưỡng thực tế
2. **Thêm handle `OA-400` + HTTP 429** vào `RETRYABLE_CODES` trong `dnse_client.py`
3. **Không nên test rate limit trên production thường xuyên** -- cooldown quá dài, ảnh hưởng trading
4. **Cân nhắc exponential backoff** khi gặp 429 thay vì fail ngay
5. **Test order rate limit** nên thực hiện trên sandbox/paper mode nếu DNSE cung cấp

---

## 5. Giai đoạn 3 -- Order Execution & Strategy Framework

**Ngày test:** 2026-03-25
**File test:** `tests/test_order_manager.py`
**Kết quả:** 36/36 PASS

### 5.1 Test environment


| Thành phần | Chi tiết                            |
| ---------- | ----------------------------------- |
| Python     | 3.10.7                              |
| pytest     | 9.0.2                               |
| Loại test  | Unit test (mocked SDK + paper mode) |


### 5.2 Test cases -- OrderManager Paper Mode (15 tests)


| #   | Test case                           | Mô tả                                                   | Kết quả |
| --- | ----------------------------------- | ------------------------------------------------------- | ------- |
| 1   | `test_submit_paper_order`           | Submit -> status NEW, order_id PAPER-xxx, is_paper=True | ✅ PASS  |
| 2   | `test_paper_fill_full`              | Full fill -> FILLED, filled_qty=100, avg_price đúng     | ✅ PASS  |
| 3   | `test_paper_fill_partial`           | 80+120 partial fills -> avg price weighted đúng         | ✅ PASS  |
| 4   | `test_paper_cancel`                 | Cancel -> CANCELED, is_terminal, active_orders = 0      | ✅ PASS  |
| 5   | `test_paper_modify`                 | Modify price -> price updated                           | ✅ PASS  |
| 6   | `test_reject_invalid_lot_size`      | qty=150 (mixed lot) -> REJECTED + INVALID_QUANTITY      | ✅ PASS  |
| 7   | `test_reject_zero_price_limit`      | LO order price=0 -> REJECTED + INVALID_PRICE            | ✅ PASS  |
| 8   | `test_odd_lot_allowed`              | qty=50 (<100 odd lot) -> NEW (allowed)                  | ✅ PASS  |
| 9   | `test_derivative_any_qty`           | DERIVATIVE qty=7 -> NEW (any qty ok)                    | ✅ PASS  |
| 10  | `test_fill_callback`                | on_fill callback triggered on fill event                | ✅ PASS  |
| 11  | `test_update_callback`              | on_update callback triggered on status change           | ✅ PASS  |
| 12  | `test_get_active_orders`            | Filled order removed from active, unfilled stays        | ✅ PASS  |
| 13  | `test_get_orders_by_status`         | Filter by NEW vs REJECTED status                        | ✅ PASS  |
| 14  | `test_cannot_fill_terminal_order`   | Fill canceled order -> False                            | ✅ PASS  |
| 15  | `test_cannot_cancel_terminal_order` | Cancel filled order -> False                            | ✅ PASS  |


### 5.3 Test cases -- OrderManager Live Mode (2 tests)


| #   | Test case                    | Mô tả                                                    | Kết quả |
| --- | ---------------------------- | -------------------------------------------------------- | ------- |
| 1   | `test_submit_live_success`   | API 200 -> order_id từ response, PENDING_NEW             | ✅ PASS  |
| 2   | `test_submit_live_api_error` | API 400 OA-200 -> REJECTED + error code in reject_reason | ✅ PASS  |


### 5.4 Test cases -- PositionTracker (11 tests)


| #   | Test case                   | Mô tả                                                    | Kết quả |
| --- | --------------------------- | -------------------------------------------------------- | ------- |
| 1   | `test_buy_opens_long`       | Buy 100@25 -> qty=100, avg=25, is_long                   | ✅ PASS  |
| 2   | `test_sell_closes_long`     | Buy 100@25 + Sell 100@26 -> flat, realized=+100          | ✅ PASS  |
| 3   | `test_partial_sell`         | Buy 200@25 + Sell 100@26 -> qty=100, realized=+100       | ✅ PASS  |
| 4   | `test_add_to_long`          | Buy 100@25 + Buy 100@27 -> qty=200, avg=26               | ✅ PASS  |
| 5   | `test_unrealized_pnl`       | Long 100@25, mkt=26 -> +100; mkt=24 -> -100; pct=4%      | ✅ PASS  |
| 6   | `test_daily_pnl`            | Win trade -> daily realized=+100, win_count=1, rate=100% | ✅ PASS  |
| 7   | `test_losing_trade`         | Sell below cost -> realized=-100, loss_count=1           | ✅ PASS  |
| 8   | `test_total_unrealized_pnl` | 2 symbols -> tổng unrealized đúng                        | ✅ PASS  |
| 9   | `test_get_open_positions`   | Flat position excluded, open included                    | ✅ PASS  |
| 10  | `test_reset_daily`          | Reset -> realized=0, trade_count=0, log cleared          | ✅ PASS  |
| 11  | `test_market_value`         | market_value + cost_basis tính đúng                      | ✅ PASS  |


### 5.5 Test cases -- StrategyBase (6 tests)


| #   | Test case                            | Mô tả                                               | Kết quả |
| --- | ------------------------------------ | --------------------------------------------------- | ------- |
| 1   | `test_start_stop`                    | start -> running + on_start; stop -> on_stop        | ✅ PASS  |
| 2   | `test_handle_quote_generates_signal` | Quote with bid -> on_tick -> BUY signal generated   | ✅ PASS  |
| 3   | `test_ignores_unsubscribed_symbols`  | VNM quote to HPG strategy -> ignored                | ✅ PASS  |
| 4   | `test_place_order_helper`            | place_order() -> NEW order, order_count incremented | ✅ PASS  |
| 5   | `test_get_position_helper`           | get_position("HPG") -> flat position                | ✅ PASS  |
| 6   | `test_stats`                         | stats dict has name, running, symbols               | ✅ PASS  |


### 5.6 Test cases -- Integration (2 tests)


| #   | Test case                    | Mô tả                                                      | Kết quả |
| --- | ---------------------------- | ---------------------------------------------------------- | ------- |
| 1   | `test_full_trade_lifecycle`  | Buy→fill→position→sell→fill→flat, realized P&L correct     | ✅ PASS  |
| 2   | `test_strategy_driven_trade` | Strategy generates signal → places order → fill → position | ✅ PASS  |


### 5.7 Đánh giá

- **Order lifecycle** hoàn chỉnh: submit → new → partially_filled → filled (và cancel/reject paths)
- **Paper mode** hoạt động đầy đủ: simulate fill với weighted avg price, partial fills
- **Live mode** integration verified qua mocked SDK (success + error paths)
- **Pre-trade validation** enforce: lot size rules (STOCK 100-lot/odd-lot, DERIVATIVE any), LO price > 0
- **Position tracking** chính xác: add-to-long, partial-sell, close, realized/unrealized P&L, cost basis
- **Strategy framework** hoạt động: tick → signal → order → fill → position update (full loop)
- **Callbacks** đúng: fill callbacks, update callbacks, strategy fill notifications

**Chưa test:**

- Live order modification/cancellation (cần production API + trading token)
- Short selling P&L (hiện logic có nhưng chưa có test case -- áp dụng cho phái sinh)
- Multi-strategy concurrent execution
- Order timeout / expiry handling

---

## 6. Giai đoạn 3b -- MCMC Derivatives Core Trading

**Ngày test:** 2026-03-25
**File test:** `tests/test_markov_model.py`, `tests/test_mcmc_engine.py`, `tests/test_risk_manager.py`, `tests/test_paper_engine.py`, `tests/test_mcmc_strategy.py`
**Kết quả:** 94/94 PASS

### 6.1 Test environment


| Thành phần     | Chi tiết                            |
| -------------- | ----------------------------------- |
| Python         | 3.10.7                              |
| numpy          | >=1.24                              |
| pytest         | 9.0.2                               |
| Môi trường     | Local (Windows 10), PAPER_MODE=true |
| Thời gian chạy | 1.82s                               |


### 6.2 Các module mới


| Module                  | File                                 | Mô tả                                                            |
| ----------------------- | ------------------------------------ | ---------------------------------------------------------------- |
| MarkovModel             | `src/mcmc/markov_model.py`           | 2-state (UP/DOWN) Markov Chain, rolling window transition matrix |
| MCMCEngine              | `src/mcmc/mcmc_engine.py`            | Bayesian MH posterior sampling + 10k GBM path simulation         |
| RiskManager             | `src/risk_manager.py`                | Pre-trade checks: daily loss halt, stoploss, position size limit |
| PaperEngine             | `src/paper_engine.py`                | Auto-fill paper orders vs live quotes, slippage support          |
| MCMCDerivativesStrategy | `src/strategies/mcmc_derivatives.py` | T0 strategy, OHLC-triggered MCMC, force close, cooldown          |
| Paper test runner       | `scripts/paper_test.py`              | Session orchestrator, session report, append to TEST_REPORT.md   |


### 6.3 Kết quả chi tiết

#### test_markov_model.py (26 tests, 26 PASS)


| Nhóm test                                                    | Tests | Kết quả |
| ------------------------------------------------------------ | ----- | ------- |
| Construction (window, min-size validation)                   | 4     | PASS    |
| Update / state classification (UP/DOWN/equal)                | 6     | PASS    |
| feed_ohlc_history (bulk load, invalid skip, window)          | 3     | PASS    |
| Transition matrix (all-UP, all-DOWN, alternating, row sums)  | 5     | PASS    |
| Predict (not-ready error, sum-to-1, explicit state, default) | 4     | PASS    |
| get_log_returns (basic, skip-zero-open, empty)               | 3     | PASS    |
| Reset                                                        | 1     | PASS    |


#### test_mcmc_engine.py (14 tests, 14 PASS)


| Nhóm test                                                                  | Tests | Kết quả |
| -------------------------------------------------------------------------- | ----- | ------- |
| Construction (burnin validation, defaults, seed reproducibility)           | 3     | PASS    |
| Run output (probabilities, n_paths, elapsed_ms, mu/sigma, dominant signal) | 5     | PASS    |
| Fallback when data insufficient (Markov-only probs)                        | 2     | PASS    |
| Direction sensitivity (high p_up biases up, high p_down biases down)       | 2     | PASS    |
| dt parameter (full vs half session variance)                               | 1     | PASS    |
| Performance benchmark (10k paths < 200ms)                                  | 1     | PASS    |


**Kết quả benchmark:** 10,000 paths + 500 MH posterior samples = **~25-45ms** (well within 100ms target)

#### test_risk_manager.py (17 tests, 17 PASS)


| Nhóm test                                                               | Tests | Kết quả |
| ----------------------------------------------------------------------- | ----- | ------- |
| Daily loss halt (allow within, reject at breach, halt flag, resume)     | 7     | PASS    |
| Position size limit (flat allow, long at max reject, close allow)       | 4     | PASS    |
| Stoploss trigger (flat no-trigger, long loss, short loss, within-limit) | 4     | PASS    |
| Manual halt                                                             | 2     | PASS    |


**Lỗi fix trong quá trình test:**

- `nav_start=0` không có đủ denominator để tính %, fix: bỏ qua daily loss check khi `nav_start=0`
- Short position stoploss: `unrealized_pnl_pct()` không account direction, fix: dùng `unrealized_pnl()` để detect actual loss

#### test_paper_engine.py (18 tests, 18 PASS)


| Nhóm test                                                   | Tests | Kết quả |
| ----------------------------------------------------------- | ----- | ------- |
| Session lifecycle (start/stop/reset)                        | 4     | PASS    |
| LO BUY fill logic (ask <= price, ask > price, fill at ask)  | 3     | PASS    |
| LO SELL fill logic (bid >= price, bid < price, fill at bid) | 3     | PASS    |
| Slippage (buy adverse, sell adverse)                        | 2     | PASS    |
| Not running (before start, after stop)                      | 2     | PASS    |
| Session report (fill count, empty)                          | 2     | PASS    |
| Symbol isolation (wrong symbol no fill, ATC session close)  | 2     | PASS    |


#### test_mcmc_strategy.py (19 tests, 19 PASS)


| Nhóm test                                                           | Tests | Kết quả |
| ------------------------------------------------------------------- | ----- | ------- |
| Signal generation (BUY, SELL, HOLD, no-MCMC, wrong symbol)          | 5     | PASS    |
| Exit signals (exit long on p_down, exit short on p_up)              | 2     | PASS    |
| Cooldown (no signal, decrement, signal after expiry)                | 3     | PASS    |
| T0 force close (no signal after close time, SELL sent, fires once)  | 3     | PASS    |
| Risk manager integration (halt blocks tick, pre-trade blocks order) | 2     | PASS    |
| OHLC handler (new candle updates Markov, same-ts skip)              | 2     | PASS    |
| Extended stats                                                      | 1     | PASS    |


### 6.4 Đánh giá

**Strengths:**

- MCMC pipeline hoạt động đúng: MH sampler hội tụ, GBM paths trong khoảng hợp lý
- Performance benchmark: 10k paths ~25-45ms (target < 100ms) -- headroom 2-3x
- T0 safety net: force-close fires exactly once, even under multiple ticks
- Risk manager: halt propagates correctly, stoploss triggers on both long and short
- Paper engine: LO fill logic mirrors real exchange matching (price-time priority)

**Limitations (expected for paper phase):**

- MH sampler với 500 iterations có thể chưa hội tụ đầy đủ cho variance ước lượng; dùng nhiều hơn trong production
- OHLC history cần đủ `MCMC_ROLLING_WINDOW` sessions trước khi strategy có thể ra tín hiệu
- `asyncio.ensure_future` trong test context gây deprecation warning (no current event loop); không ảnh hưởng production khi chạy trong event loop

**Khuyến nghị:**

- Chạy `scripts/paper_test.py` trong giờ giao dịch ít nhất 1 phiên đầy đủ trước khi tăng confidence hoặc position size
- Theo dõi `avg_mcmc_ms` trong live session; nếu > 80ms, giảm `MCMC_MH_ITERATIONS` hoặc `MCMC_NUM_PATHS`

---

## 7. Quy ước báo cáo test

Từ Giai đoạn 2 trở đi, mọi test đều phải được ghi vào report này theo format:

### Format cho Unit Test

```
## [Tên module] -- [Mô tả ngắn]
- Ngày test:
- File test:
- Kết quả: X/Y PASS
- Bảng test cases: #, tên, mô tả, kết quả
- Đánh giá: những gì đã cover, chưa cover
```

### Format cho Live/Integration Test

```
## [Tên test] -- [Mục tiêu]
- Ngày test:
- Script:
- Loại test: Live / Sandbox / Mock
- Phương pháp:
- Bảng kết quả:
- Phát hiện quan trọng:
- Khuyến nghị:
```

### Nguyên tắc

- Mọi test PHẢI có evidence (output cụ thể, số liệu, timestamps)
- Test result PHẢI ghi rõ PASS/FAIL, không dùng từ mơ hồ
- Nếu test FAIL, ghi rõ root cause và action items
- Live tests ghi rõ thời gian chạy để có thể reproduce
- Report cập nhật cùng lúc với code changes

---

## Tổng hợp test coverage


| Giai đoạn             | File test                                                                              | Tests   | Pass    | Fail  | Coverage                                                                |
| --------------------- | -------------------------------------------------------------------------------------- | ------- | ------- | ----- | ----------------------------------------------------------------------- |
| GĐ1: Core & Auth      | `test_auth.py`                                                                         | 7       | 7       | 0     | Auth flow, API wrapper, error handling                                  |
| GĐ2: Market Data      | `test_market_data.py`                                                                  | 29      | 29      | 0     | Buffers, subscriptions, callbacks                                       |
| Rate Limiter          | `test_rate_limiter.py`                                                                 | 18      | 18      | 0     | Sliding window, integration                                             |
| GĐ3: Execution        | \ est_order_manager.py\                                                                | 36      | 36      | 0     | Order lifecycle, positions, P&L, strategy                               |
| GĐ3b: Markov Chain    | \ est_markov_model.py\                                                                 | 26      | 26      | 0     | 2-state chain, rolling window                                           |
| GĐ3b: MCMC Engine     | \ est_mcmc_engine.py\                                                                  | 14      | 14      | 0     | MH sampler, GBM paths, perf <200ms                                      |
| GĐ3b: Risk Manager    | \ est_risk_manager.py\                                                                 | 17      | 17      | 0     | Daily halt, stoploss, position size                                     |
| GĐ3b: Paper Engine    | \ est_paper_engine.py\                                                                 | 18      | 18      | 0     | LO fill logic, slippage, ATC                                            |
| GĐ3b: MCMC Strategy   | \ est_mcmc_strategy.py\                                                                | 19      | 19      | 0     | Signals, cooldown, T0 force close                                       |
| TTM breakout refactor | `test_ttm_parallel_runner.py`, `test_ttm_volume_features.py`, `test_ttm_validation.py` | 12      | 11      | 0     | Breakout continuation-only gate, exhaustion snapshot, validation filter |
| **Tổng unit tests**   |                                                                                        | **184** | **184** | **0** |                                                                         |
| Live: Rate Limit      | ate_limit_test.py\                                                                     | 1       | -       | -     | DNSE API rate limit discovery                                           |
| Live: Paper Test      | \paper_test.py\                                                                        | -       | -       | -     | MCMC paper session (market hours)                                       |


**Chạy lại toàn bộ test:**

```bash
python -m pytest tests/ -v
```

**Evidence (full suite, 2026-03-31):** `python -m pytest tests -q --tb=no` → **295 passed** in ~120s (26 warnings, asyncio event loop deprecation in tests).

---

## 6c. Config-driven derivative exit rules [2026-03-31]

**Ngày test:** 2026-03-31  
**File test:** `tests/test_exit_rules.py`, `tests/test_mcmc_strategy.py` (exit integration)  
**Kết quả:** 10/10 PASS (`test_exit_rules.py`) + 18/18 PASS (`test_mcmc_strategy.py`); full suite **295/295 PASS**

### 6c.1 Module


| Module     | File                                 | Mô tả                                                                                                                  |
| ---------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Exit rules | `src/strategies/exit_rules.py`       | `should_exit`, `risk_exit`, `state_flip_exit`, `expected_return_exit`, `time_exit`; priority risk → flip → E[r] → time |
| Settings   | `src/config.py`                      | `MCMC_`* exit params + `parse_int_list` cho bullish/bearish state lists                                                |
| Markov     | `src/mcmc/markov_model.py`           | `empirical_state_mean_log_returns()` cho vector `mu` theo state UP/DOWN                                                |
| Strategy   | `src/strategies/mcmc_derivatives.py` | `_maybe_exit_signal`, `bars_in_trade`, structured `exit_decision` logging                                              |


### 6c.2 Test cases -- test_exit_rules.py


| #   | Test case                                    | Mô tả                             | Kết quả |
| --- | -------------------------------------------- | --------------------------------- | ------- |
| 1   | `test_exit_config_from_settings`             | Settings → dict; JSON/comma lists | ✅ PASS  |
| 2   | `test_risk_exit_long_short_same_pnl_rule`    | SL/TP signed points, LONG/SHORT   | ✅ PASS  |
| 3   | `test_state_flip_long_bearish`               | LONG + mass bearish > threshold   | ✅ PASS  |
| 4   | `test_state_flip_short_bullish`              | SHORT + mass bullish > threshold  | ✅ PASS  |
| 5   | `test_expected_return_long`                  | E[r] − tc < threshold             | ✅ PASS  |
| 6   | `test_expected_return_short`                 | E[r] − tc > −threshold            | ✅ PASS  |
| 7   | `test_compute_expected_return_one_hot`       | one-hot @ A @ mu                  | ✅ PASS  |
| 8   | `test_time_exit`                             | max bars                          | ✅ PASS  |
| 9   | `test_should_exit_priority_risk_before_flip` | priority                          | ✅ PASS  |
| 10  | `test_should_exit_priority_flip_before_time` | priority                          | ✅ PASS  |


### 6c.3 Đánh giá

- Đã cover: priority, LONG/SHORT, transaction cost trong expected return, parse config lists.
- **Chưa cover:** end-to-end live order từ exit (giữ integration paper/live như trước).

---

## 7. Giai đoạn 4 -- Backtest Framework (MCMC Strategy Validation)

**Ngày:** 2026-03-25
**Trang thai:** HOAN THANH (unit tests 26/26 PASS, synthetic simulation 5/5 suites OK)

### Kien truc Backtest


| Module           | File                              | Mo ta                                               |
| ---------------- | --------------------------------- | --------------------------------------------------- |
| DataFetcher      | src/backtest/data_fetcher.py      | Fetch + cache DNSE OHLC REST API, JSON local cache  |
| BarReplay        | src/backtest/bar_replay.py        | Walk-forward engine, no look-ahead, T0 execution    |
| BacktestMetrics  | src/backtest/metrics.py           | Sharpe, Sortino, MDD, CAGR, win rate, profit factor |
| StatisticalTests | src/backtest/statistical_tests.py | Monte Carlo permutation (10k perms, p-value)        |
| CLI Runner       | scripts/backtest.py               | 5 test suites, CSV + TEST_REPORT.md output          |
| Unit Tests       | tests/test_backtest.py            | 26 unit tests, all PASS                             |


### Unit Test Results (26/26 PASS)


| Group                    | Tests  | Status       |
| ------------------------ | ------ | ------------ |
| DataFetcher [F1-F6]      | 6      | PASS         |
| BarReplay [R1-R8]        | 8      | PASS         |
| BacktestMetrics [M1-M7]  | 7      | PASS         |
| StatisticalTests [S1-S4] | 4      | PASS         |
| CSV Export [V1]          | 1      | PASS         |
| **Total**                | **26** | **ALL PASS** |


**Key bugs fixed during testing:**

- BarReplay loop: model was only updated once at i=0, then not during warm-up. Fixed by restructuring to update with bars[i-1] at the start of each iteration.
- Sharpe ratio: 
p.std(ddof=1) gave tiny non-zero float for identical returns, causing division by near-zero. Fixed with std_ret > 1e-10 guard.
- test_mcmc_strategy.py: 6 tests were time-of-day sensitive (failed after 14:25 Vietnam time due to T0 force-close triggered by real clock). Fixed by patching datetime.now() to 09:30.

---

## Backtest: Suite 1 -- In-Sample (Synthetic Simulation) [2026-03-25]

**Note:** Live backtest requires DNSE API credentials with historical derivatives data.
The results below are from a synthetic simulation to validate the framework logic.
Run python scripts/backtest.py --suite insample with valid credentials for production results.

**Symbol:** VN30F2506 (synthetic) | **Period:** 2025-09-01 to 2025-12-31 | **Bar type:** Daily
**Confidence:** 0.55 | **Warmup:** 20 bars | **Paths:** 500 | **Fast mode:** True (Markov-only)


| Metric        | Value         |
| ------------- | ------------- |
| Bars          | 90            |
| Trades        | 61            |
| Win rate      | 70.5%         |
| Total P&L     | 333.89 points |
| Total return  | 25.68%        |
| CAGR          | 89.66%        |
| Sharpe ratio  | 8.615         |
| Sortino ratio | 21.497        |
| Max drawdown  | 2.78%         |
| Calmar ratio  | 32.253        |
| Profit factor | 5.797         |
| Expectancy    | 5.474         |


**Pass criteria:** Sharpe > 0, Win rate > 45%, MDD < 20%
**Status:** PASS (synthetic data -- framework validated)

---

## Backtest: Suite 2 -- Out-of-Sample (Synthetic Simulation) [2026-03-25]

**Symbol:** VN30F2506 (synthetic) | **Period:** 2026-01-01 to 2026-03-30 | **Bar type:** Daily
**Confidence:** 0.55 | **Warmup:** 20 bars | **Paths:** 500 | **Fast mode:** True


| Metric       | Value         |
| ------------ | ------------- |
| Bars         | 60            |
| Trades       | 27            |
| Win rate     | 40.7%         |
| Total P&L    | -19.30 points |
| Sharpe ratio | -1.396        |
| Max drawdown | 3.27%         |


**OOS/IS Sharpe ratio:** -0.162 (target >= 0.50)
**Status:** NEEDS REVIEW -- expected degradation on random OOS data without persistent signal.
This is normal behavior for synthetic random data. Production backtest on real DNSE data required for valid evaluation.

---

## Backtest: Suite 3 -- Monte Carlo Permutation (Synthetic Simulation) [2026-03-25]

**Symbol:** VN30F2506 (synthetic) | **Period:** 2025-09-01 to 2025-12-31 | **Bar type:** Daily
**N permutations:** 200 (production: 10000) | **Fast mode:** True


| Statistic            | Value   |
| -------------------- | ------- |
| Actual Sharpe        | 8.6154  |
| Perm mean Sharpe     | -0.4691 |
| Perm std Sharpe      | 1.7483  |
| Perm 95th pctile     | 2.9174  |
| p-value              | 0.0000  |
| Significant (p<0.05) | YES     |
| N permutations       | 200     |
| Elapsed              | 0.8s    |


**Interpretation:** Strategy shows statistically significant skill on synthetic data (p<0.05).
The actual Sharpe (8.61) is far above the 95th percentile of permuted Sharpes (2.92).
On synthetic data this is expected -- the Markov model correctly detects the seeded autocorrelation.
Production significance test pending live data.

---

## Backtest: Suite 4 -- Hourly Sensitivity (Synthetic Simulation) [2026-03-25]

**Symbol:** VN30F2506 (synthetic) | **Period:** 2025-09-01 to 2025-12-31 | **Bar type:** 60min
**Confidence:** 0.55 | **Warmup:** 20 bars


| Metric       | Value  |
| ------------ | ------ |
| Bars         | 540    |
| Trades       | 513    |
| Win rate     | 95.3%  |
| Sharpe ratio | 43.708 |
| Max drawdown | 0.15%  |


**Purpose:** Compare strategy performance on hourly vs daily resolution.
**Observation (synthetic):** More trades, higher win rate. This reflects the synthetic data construction where hourly bars inherit the daily trend exactly.

---

## Backtest: Suite 5 -- Multi-Symbol (Synthetic Simulation) [2026-03-25]

**Symbols:** VN30F2507, VNM, HPG | **Period:** 2025-09-01 to 2025-12-31 | **Bar type:** Daily
**Confidence:** 0.55 | **Warmup:** 20 bars | **Paths:** 200 | **Fast mode:** True


| Symbol    | Trades | Win Rate | Sharpe | Sortino | MDD   | Total Return |
| --------- | ------ | -------- | ------ | ------- | ----- | ------------ |
| VN30F2507 | 49     | 51.0%    | 0.332  | 0.408   | 3.26% | 0.87%        |
| VNM       | 50     | 50.0%    | 0.193  | 0.248   | 3.25% | 0.22%        |
| HPG       | 52     | 46.2%    | -0.596 | -0.858  | 3.88% | -0.44%       |


**Purpose:** Validate Markov autocorrelation pattern generalizes across symbols.
**Observation (synthetic):** Moderate Sharpe on random data (near zero is expected for purely random series). Production backtest with real price data required.

---

## Ket luan Giai doan 4 -- Backtest Framework


| Hang muc         | Trang thai | Chi tiet                                                    |
| ---------------- | ---------- | ----------------------------------------------------------- |
| DataFetcher      | DONE       | DNSE REST API wrapper, JSON cache, columnar+row format      |
| BarReplay        | DONE       | Walk-forward, no look-ahead, T0 open/close execution        |
| BacktestMetrics  | DONE       | 10 metrics: Sharpe, Sortino, Calmar, MDD, CAGR, etc.        |
| StatisticalTests | DONE       | MC permutation 10k perms, p-value, walk_forward_split       |
| CLI backtest.py  | DONE       | 5 suites, --suite flag, CSV + TEST_REPORT.md output         |
| Unit tests       | DONE       | 26/26 PASS                                                  |
| Live backtest    | PENDING    | Requires valid DNSE API creds + active derivatives contract |


**Tong so test cases:** 295/295 PASS (full `pytest tests`; includes backtest + exit rules)

**Huong dan chay production backtest:**
`ash

# In-sample

python scripts/backtest.py --suite insample --symbol VN30F2506

# Tat ca 5 suites (chay nhanh hon voi --fast --n-paths 200)

python scripts/backtest.py --suite all --fast --n-paths 200

# Monte Carlo day du (10k perms, can ~30 phut)

python scripts/backtest.py --suite perm --n-perms 10000

# Cache san, khong goi API lan nua

python scripts/backtest.py --suite all --no-fetch
`

---

## Backtest Summary -- MCMC Derivatives Strategy [2026-03-25 19:42]

**Config:** VN30F2506 | Confidence=0.75 | Warmup=50 | Paths=200 | Fast=True | Commission=0.000%

---

## Backtest Summary -- MCMC Derivatives Strategy [2026-03-25 20:04]

**Config:** VN30 | Confidence=0.55 | Warmup=20 | Paths=200 | Fast=True | Commission=0.000%

---

## Backtest: Suite 1 -- In-Sample [2026-03-25 20:04]

**Symbol:** VN30 | **Period:** 20250901-20251231 | **Bar type:** Daily
**Confidence:** 0.55 | **Warmup:** 20 bars | **Paths:** 200 | **Fast mode:** True


| Metric        | Value    |
| ------------- | -------- |
| Bars          | 85       |
| Trades        | 48       |
| Win rate      | 47.9%    |
| Total P&L     | 44.76    |
| Total return  | 44.76%   |
| CAGR          | 199.42%  |
| Sharpe ratio  | 0.388    |
| Sortino ratio | 0.540    |
| Max drawdown  | 106.36%  |
| Calmar ratio  | 1.875    |
| Profit factor | 1.093    |
| Expectancy    | 0.9325   |
| Avg win       | 22.9496  |
| Avg loss      | -19.3232 |
| Avg MCMC ms   | 0.0      |


**Pass criteria:** Sharpe > 0, Win rate > 45%, MDD < 20%
**Status:** NEEDS REVIEW

---

## Backtest: Suite 2 -- Out-of-Sample [2026-03-25 20:04]

**Symbol:** VN30 | **Period:** 20260101-20260325 | **Bar type:** Daily
**Confidence:** 0.55 | **Warmup:** 20 bars | **Paths:** 200 | **Fast mode:** True


| Metric        | Value     |
| ------------- | --------- |
| Bars          | 52        |
| Trades        | 29        |
| Win rate      | 55.2%     |
| Total P&L     | 279.09    |
| Total return  | 279.09%   |
| CAGR          | 63678.51% |
| Sharpe ratio  | 3.469     |
| Sortino ratio | 7.724     |
| Max drawdown  | 22.67%    |
| Calmar ratio  | 2808.997  |
| Profit factor | 2.447     |
| Expectancy    | 9.6238    |
| Avg win       | 29.4944   |
| Avg loss      | -14.8323  |
| Avg MCMC ms   | 0.0       |


### IS vs OOS Comparison


| Metric        | In-Sample | Out-of-Sample |
| ------------- | --------- | ------------- |
| Sharpe ratio  | 0.388     | 3.469         |
| Sortino ratio | 0.540     | 7.724         |
| Max drawdown  | 106.36%   | 22.67%        |
| Win rate      | 47.9%     | 55.2%         |
| Total return  | 44.76%    | 279.09%       |
| CAGR          | 199.42%   | 63678.51%     |
| N trades      | 48        | 29            |
| Profit factor | 1.093     | 2.447         |


**OOS/IS Sharpe ratio**: 8.94 (target >= 0.50)

**Pass criteria:** OOS/IS Sharpe >= 0.50

---

## Backtest: Suite 3 -- Monte Carlo Permutation [2026-03-25 20:04]

**Symbol:** VN30 | **Period:** 20250901-20251231 | **Bar type:** Daily
**Confidence:** 0.55 | **Warmup:** 20 bars | **Paths:** 200 | **Fast mode:** True


| Statistic            | Value   |
| -------------------- | ------- |
| Actual Sharpe        | 0.3881  |
| Perm mean Sharpe     | -0.4215 |
| Perm std Sharpe      | 1.7009  |
| Perm 95th pctile     | 2.3273  |
| p-value              | 0.3400  |
| Significant (p<0.05) | NO      |
| N permutations       | 200     |
| Elapsed              | 0.9s    |


**Interpretation:** Cannot reject null hypothesis -- performance may be due to chance

---

## Backtest: Suite 4 -- Hourly Sensitivity [2026-03-25 20:04]

**Symbol:** VN30 | **Period:** 20250901-20251231 | **Bar type:** 60min
**Confidence:** 0.55 | **Warmup:** 20 bars | **Paths:** 200 | **Fast mode:** True


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 323     |
| Win rate      | 39.9%   |
| Total P&L     | -54.04  |
| Total return  | -54.04% |
| CAGR          | -92.94% |
| Sharpe ratio  | -0.497  |
| Sortino ratio | -0.598  |
| Max drawdown  | 124.55% |
| Calmar ratio  | -0.746  |
| Profit factor | 0.951   |
| Expectancy    | -0.1673 |
| Avg win       | 8.1972  |
| Avg loss      | -8.6161 |
| Avg MCMC ms   | 0.0     |


**Purpose:** Compare strategy performance on hourly vs daily resolution.

---

## Backtest: Suite 5 -- Multi-Symbol [2026-03-25 20:04]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily
**Confidence:** 0.55 | **Warmup:** 20 bars | **Paths:** 200 | **Fast mode:** True


| Symbol | Trades | Win Rate | Sharpe | Sortino | MDD    | Total Return |
| ------ | ------ | -------- | ------ | ------- | ------ | ------------ |
| HPG    | 56     | 58.9%    | 2.478  | 4.713   | 1.20%  | 4.65%        |
| VNM    | 48     | 45.8%    | -0.019 | -0.037  | 6.83%  | -0.09%       |
| FPT    | 61     | 57.4%    | 0.092  | 0.139   | 10.41% | 0.69%        |


**Purpose:** Validate Markov autocorrelation pattern generalizes across symbols.

---

## Backtest Summary -- MCMC Derivatives Strategy [2026-03-25 20:53]

**Config:** VN30F1M | Confidence=0.75 | Warmup=30 | Paths=500 | Fast=True | Commission=0.000%

---

## Backtest: Suite 3 -- Monte Carlo Permutation (Hourly) [2026-03-25 20:54]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly (1H)
**Confidence:** 0.75 | **Warmup:** 30 bars | **Paths:** 500 | **Fast mode:** True


| Statistic            | Value   |
| -------------------- | ------- |
| Actual Sharpe        | 1.9846  |
| Perm mean Sharpe     | -0.5675 |
| Perm std Sharpe      | 1.7960  |
| Perm 95th pctile     | 2.3342  |
| p-value              | 0.0780  |
| Significant (p<0.05) | NO      |
| N permutations       | 2000    |
| Elapsed              | 48.1s   |


**Interpretation:** Cannot reject null hypothesis -- performance may be due to chance

---

## Giai đoạn 5 — MCMC + HMM Comparison Backtest [2026-03-27 19:29]

**Strategy:** MCMC + HMM Comparison | **Symbol:** VN30F1M | **Confidence:** 0.75 | **Warmup:** 30 bars

---

## Backtest: MCMC -- Suite 1 In-Sample [2026-03-27 19:29]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 85     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 2 OOS [2026-03-27 19:29]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 52     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 3 Monte Carlo [2026-03-27 19:30]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Statistic            | Value   |
| -------------------- | ------- |
| Actual Sharpe        | 1.9846  |
| Perm mean Sharpe     | -0.5086 |
| Perm std Sharpe      | 1.6884  |
| Perm 95th pctile     | 2.2020  |
| p-value              | 0.0720  |
| Significant (p<0.05) | NO      |
| N permutations       | 500     |
| Elapsed              | 17.3s   |


**Interpretation:** Cannot reject null -- may be due to chance

---

## Backtest: MCMC -- Suite 4 Hourly [2026-03-27 19:30]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 29      |
| Win rate      | 34.5%   |
| Total P&L     | 53.97   |
| Total return  | 53.97%  |
| CAGR          | 335.56% |
| Sharpe ratio  | 1.985   |
| Sortino ratio | 4.812   |
| Max drawdown  | 18.17%  |
| Calmar ratio  | 18.468  |
| Profit factor | 1.810   |
| Expectancy    | 1.8610  |
| Avg win       | 12.0590 |
| Avg loss      | -5.1246 |
| Avg MCMC ms   | 0.0     |


---

## Backtest: MCMC -- Suite 5 Multi-Symbol [2026-03-27 19:30]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Symbol | Trades | Win Rate | Sharpe  | Sortino | MDD   | Return |
| ------ | ------ | -------- | ------- | ------- | ----- | ------ |
| HPG    | 9      | 33.3%    | -1.161  | -3.775  | 0.85% | -0.75% |
| VNM    | 4      | 25.0%    | -2.081  | -3.166  | 2.99% | -2.60% |
| FPT    | 4      | 0.0%     | -18.152 | -18.152 | 6.57% | -6.57% |


---

## Backtest: HMM -- Suite 1 In-Sample [2026-03-27 19:30]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars

*Training window: 20240101–20251231 (497 bars)*


| Metric        | Value    |
| ------------- | -------- |
| Bars          | 85       |
| Trades        | 25       |
| Win rate      | 48.0%    |
| Total P&L     | 79.11    |
| Total return  | 79.11%   |
| CAGR          | 462.89%  |
| Sharpe ratio  | 0.888    |
| Sortino ratio | 1.934    |
| Max drawdown  | 19.20%   |
| Calmar ratio  | 24.105   |
| Profit factor | 1.347    |
| Expectancy    | 3.1644   |
| Avg win       | 25.6083  |
| Avg loss      | -17.5531 |
| Avg MCMC ms   | 0.0      |


---

## Backtest: HMM -- Suite 2 OOS [2026-03-27 19:31]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value     |
| ------------- | --------- |
| Bars          | 52        |
| Trades        | 15        |
| Win rate      | 66.7%     |
| Total P&L     | 178.63    |
| Total return  | 178.63%   |
| CAGR          | 14244.11% |
| Sharpe ratio  | 2.470     |
| Sortino ratio | 5.281     |
| Max drawdown  | 13.65%    |
| Calmar ratio  | 1043.419  |
| Profit factor | 2.367     |
| Expectancy    | 11.9087   |
| Avg win       | 30.9340   |
| Avg loss      | -26.1420  |
| Avg MCMC ms   | 0.0       |


---

## Giai đoạn 5 — MCMC + HMM Comparison Backtest [2026-03-27 19:41]

**Strategy:** MCMC + HMM Comparison | **Symbol:** VN30F1M | **Confidence:** 0.75 | **Warmup:** 30 bars

---

## Backtest: MCMC -- Suite 1 In-Sample [2026-03-27 19:41]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 85     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 2 OOS [2026-03-27 19:41]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 52     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 3 Monte Carlo [2026-03-27 19:41]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Statistic            | Value   |
| -------------------- | ------- |
| Actual Sharpe        | 1.9846  |
| Perm mean Sharpe     | -0.5086 |
| Perm std Sharpe      | 1.6884  |
| Perm 95th pctile     | 2.2020  |
| p-value              | 0.0720  |
| Significant (p<0.05) | NO      |
| N permutations       | 500     |
| Elapsed              | 23.7s   |


**Interpretation:** Cannot reject null -- may be due to chance

---

## Backtest: MCMC -- Suite 4 Hourly [2026-03-27 19:41]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 29      |
| Win rate      | 34.5%   |
| Total P&L     | 53.97   |
| Total return  | 53.97%  |
| CAGR          | 335.56% |
| Sharpe ratio  | 1.985   |
| Sortino ratio | 4.812   |
| Max drawdown  | 18.17%  |
| Calmar ratio  | 18.468  |
| Profit factor | 1.810   |
| Expectancy    | 1.8610  |
| Avg win       | 12.0590 |
| Avg loss      | -5.1246 |
| Avg MCMC ms   | 0.0     |


---

## Backtest: MCMC -- Suite 5 Multi-Symbol [2026-03-27 19:41]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Symbol | Trades | Win Rate | Sharpe  | Sortino | MDD   | Return |
| ------ | ------ | -------- | ------- | ------- | ----- | ------ |
| HPG    | 9      | 33.3%    | -1.161  | -3.775  | 0.85% | -0.75% |
| VNM    | 4      | 25.0%    | -2.081  | -3.166  | 2.99% | -2.60% |
| FPT    | 4      | 0.0%     | -18.152 | -18.152 | 6.57% | -6.57% |


---

## Backtest: HMM -- Suite 1 In-Sample [2026-03-27 19:42]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars

*Training window: 20240101–20251231 (497 bars)*


| Metric        | Value    |
| ------------- | -------- |
| Bars          | 85       |
| Trades        | 25       |
| Win rate      | 48.0%    |
| Total P&L     | 79.11    |
| Total return  | 79.11%   |
| CAGR          | 462.89%  |
| Sharpe ratio  | 0.888    |
| Sortino ratio | 1.934    |
| Max drawdown  | 19.20%   |
| Calmar ratio  | 24.105   |
| Profit factor | 1.347    |
| Expectancy    | 3.1644   |
| Avg win       | 25.6083  |
| Avg loss      | -17.5531 |
| Avg MCMC ms   | 0.0      |


---

## Backtest: HMM -- Suite 2 OOS [2026-03-27 19:43]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value     |
| ------------- | --------- |
| Bars          | 52        |
| Trades        | 15        |
| Win rate      | 66.7%     |
| Total P&L     | 178.63    |
| Total return  | 178.63%   |
| CAGR          | 14244.11% |
| Sharpe ratio  | 2.470     |
| Sortino ratio | 5.281     |
| Max drawdown  | 13.65%    |
| Calmar ratio  | 1043.419  |
| Profit factor | 2.367     |
| Expectancy    | 11.9087   |
| Avg win       | 30.9340   |
| Avg loss      | -26.1420  |
| Avg MCMC ms   | 0.0       |


---

## Giai đoạn 5 — MCMC + HMM Comparison Backtest [2026-03-27 19:46]

**Strategy:** MCMC + HMM Comparison | **Symbol:** VN30F1M | **Confidence:** 0.75 | **Warmup:** 30 bars

---

## Backtest: MCMC -- Suite 1 In-Sample [2026-03-27 19:46]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 85     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 2 OOS [2026-03-27 19:46]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 52     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 3 Monte Carlo [2026-03-27 19:46]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Statistic            | Value   |
| -------------------- | ------- |
| Actual Sharpe        | 1.9846  |
| Perm mean Sharpe     | -0.5086 |
| Perm std Sharpe      | 1.6884  |
| Perm 95th pctile     | 2.2020  |
| p-value              | 0.0720  |
| Significant (p<0.05) | NO      |
| N permutations       | 500     |
| Elapsed              | 26.2s   |


**Interpretation:** Cannot reject null -- may be due to chance

---

## Backtest: MCMC -- Suite 4 Hourly [2026-03-27 19:46]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 29      |
| Win rate      | 34.5%   |
| Total P&L     | 53.97   |
| Total return  | 53.97%  |
| CAGR          | 335.56% |
| Sharpe ratio  | 1.985   |
| Sortino ratio | 4.812   |
| Max drawdown  | 18.17%  |
| Calmar ratio  | 18.468  |
| Profit factor | 1.810   |
| Expectancy    | 1.8610  |
| Avg win       | 12.0590 |
| Avg loss      | -5.1246 |
| Avg MCMC ms   | 0.0     |


---

## Backtest: MCMC -- Suite 5 Multi-Symbol [2026-03-27 19:46]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Symbol | Trades | Win Rate | Sharpe  | Sortino | MDD   | Return |
| ------ | ------ | -------- | ------- | ------- | ----- | ------ |
| HPG    | 9      | 33.3%    | -1.161  | -3.775  | 0.85% | -0.75% |
| VNM    | 4      | 25.0%    | -2.081  | -3.166  | 2.99% | -2.60% |
| FPT    | 4      | 0.0%     | -18.152 | -18.152 | 6.57% | -6.57% |


---

## Backtest: HMM -- Suite 1 In-Sample [2026-03-27 19:47]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars

*Training window: 20240101–20251231 (497 bars)*


| Metric        | Value    |
| ------------- | -------- |
| Bars          | 85       |
| Trades        | 25       |
| Win rate      | 48.0%    |
| Total P&L     | 79.11    |
| Total return  | 79.11%   |
| CAGR          | 462.89%  |
| Sharpe ratio  | 0.888    |
| Sortino ratio | 1.934    |
| Max drawdown  | 19.20%   |
| Calmar ratio  | 24.105   |
| Profit factor | 1.347    |
| Expectancy    | 3.1644   |
| Avg win       | 25.6083  |
| Avg loss      | -17.5531 |
| Avg MCMC ms   | 0.0      |


---

## Backtest: HMM -- Suite 2 OOS [2026-03-27 19:49]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value     |
| ------------- | --------- |
| Bars          | 52        |
| Trades        | 15        |
| Win rate      | 66.7%     |
| Total P&L     | 178.63    |
| Total return  | 178.63%   |
| CAGR          | 14244.11% |
| Sharpe ratio  | 2.470     |
| Sortino ratio | 5.281     |
| Max drawdown  | 13.65%    |
| Calmar ratio  | 1043.419  |
| Profit factor | 2.367     |
| Expectancy    | 11.9087   |
| Avg win       | 30.9340   |
| Avg loss      | -26.1420  |
| Avg MCMC ms   | 0.0       |


---

## Backtest: HMM -- Suite 3 Monte Carlo [2026-03-27 19:49]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric                | Value   |
| --------------------- | ------- |
| Actual Sharpe         | 1.3439  |
| Perm Sharpe mean      | -0.0962 |
| p-value (approx)      | 0.1600  |
| Significant (p<0.05)  | False   |
| Permutations (capped) | 50      |


**Note:** HMM permutation test capped at 50 perms for runtime. Use MCMC suite for rigorous 1000-perm test.

---

## Backtest: HMM -- Suite 4 Hourly [2026-03-27 19:50]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 216     |
| Win rate      | 42.1%   |
| Total P&L     | 3.24    |
| Total return  | 3.24%   |
| CAGR          | 11.48%  |
| Sharpe ratio  | 0.041   |
| Sortino ratio | 0.050   |
| Max drawdown  | 87.42%  |
| Calmar ratio  | 0.131   |
| Profit factor | 1.005   |
| Expectancy    | 0.0150  |
| Avg win       | 7.5789  |
| Avg loss      | -7.1504 |
| Avg MCMC ms   | 0.0     |


---

## Backtest: HMM -- Suite 5 Multi-Symbol [2026-03-27 19:52]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Symbol | Trades | Win Rate | Sharpe | Sortino | MDD    | Return |
| ------ | ------ | -------- | ------ | ------- | ------ | ------ |
| HPG    | 15     | 20.0%    | -3.976 | -4.484  | 3.07%  | -3.20% |
| VNM    | 24     | 50.0%    | 1.470  | 4.352   | 2.65%  | 5.61%  |
| FPT    | 22     | 31.8%    | -2.088 | -3.331  | 15.42% | -9.51% |


---

## Giai đoạn 5 — MCMC + HMM Comparison Backtest [2026-03-27 20:03]

**Strategy:** MCMC + HMM Comparison | **Symbol:** VN30F1M | **Confidence:** 0.75 | **Warmup:** 30 bars

---

## Backtest: MCMC -- Suite 1 In-Sample [2026-03-27 20:03]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 85     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 2 OOS [2026-03-27 20:03]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 52     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 3 Monte Carlo [2026-03-27 20:04]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Statistic            | Value   |
| -------------------- | ------- |
| Actual Sharpe        | 1.9846  |
| Perm mean Sharpe     | -0.5086 |
| Perm std Sharpe      | 1.6884  |
| Perm 95th pctile     | 2.2020  |
| p-value              | 0.0720  |
| Significant (p<0.05) | NO      |
| N permutations       | 500     |
| Elapsed              | 13.7s   |


**Interpretation:** Cannot reject null -- may be due to chance

---

## Backtest: MCMC -- Suite 4 Hourly [2026-03-27 20:04]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 29      |
| Win rate      | 34.5%   |
| Total P&L     | 53.97   |
| Total return  | 53.97%  |
| CAGR          | 335.56% |
| Sharpe ratio  | 1.985   |
| Sortino ratio | 4.812   |
| Max drawdown  | 18.17%  |
| Calmar ratio  | 18.468  |
| Profit factor | 1.810   |
| Expectancy    | 1.8610  |
| Avg win       | 12.0590 |
| Avg loss      | -5.1246 |
| Avg MCMC ms   | 0.0     |


---

## Backtest: MCMC -- Suite 5 Multi-Symbol [2026-03-27 20:04]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Symbol | Trades | Win Rate | Sharpe  | Sortino | MDD   | Return |
| ------ | ------ | -------- | ------- | ------- | ----- | ------ |
| HPG    | 9      | 33.3%    | -1.161  | -3.775  | 0.85% | -0.75% |
| VNM    | 4      | 25.0%    | -2.081  | -3.166  | 2.99% | -2.60% |
| FPT    | 4      | 0.0%     | -18.152 | -18.152 | 6.57% | -6.57% |


---

## Backtest: HMM -- Suite 1 In-Sample [2026-03-27 20:04]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars

*Training window: 20240101–20251231 (497 bars)*


| Metric        | Value    |
| ------------- | -------- |
| Bars          | 85       |
| Trades        | 25       |
| Win rate      | 48.0%    |
| Total P&L     | 79.11    |
| Total return  | 79.11%   |
| CAGR          | 462.89%  |
| Sharpe ratio  | 0.888    |
| Sortino ratio | 1.934    |
| Max drawdown  | 19.20%   |
| Calmar ratio  | 24.105   |
| Profit factor | 1.347    |
| Expectancy    | 3.1644   |
| Avg win       | 25.6083  |
| Avg loss      | -17.5531 |
| Avg MCMC ms   | 0.0      |


---

## Backtest: HMM -- Suite 2 OOS [2026-03-27 20:05]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value     |
| ------------- | --------- |
| Bars          | 52        |
| Trades        | 15        |
| Win rate      | 66.7%     |
| Total P&L     | 178.63    |
| Total return  | 178.63%   |
| CAGR          | 14244.11% |
| Sharpe ratio  | 2.470     |
| Sortino ratio | 5.281     |
| Max drawdown  | 13.65%    |
| Calmar ratio  | 1043.419  |
| Profit factor | 2.367     |
| Expectancy    | 11.9087   |
| Avg win       | 30.9340   |
| Avg loss      | -26.1420  |
| Avg MCMC ms   | 0.0       |


---

## Backtest: HMM -- Suite 3 Hourly (Actual Sharpe) [2026-03-27 20:05]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Actual Sharpe | 0.0409 |
| Trades        | 216    |
| Win Rate      | 42.1%  |
| MDD           | 87.42% |


**Note:** HMM permutation test skipped — each permutation requires ~500 GaussianHMM re-fits (100× slower than MCMC). See MCMC Suite 3 for p-value reference (p=0.072).

---

## Backtest: HMM -- Suite 4 Hourly [2026-03-27 20:06]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 216     |
| Win rate      | 42.1%   |
| Total P&L     | 3.24    |
| Total return  | 3.24%   |
| CAGR          | 11.48%  |
| Sharpe ratio  | 0.041   |
| Sortino ratio | 0.050   |
| Max drawdown  | 87.42%  |
| Calmar ratio  | 0.131   |
| Profit factor | 1.005   |
| Expectancy    | 0.0150  |
| Avg win       | 7.5789  |
| Avg loss      | -7.1504 |
| Avg MCMC ms   | 0.0     |


---

## Backtest: HMM -- Suite 5 Multi-Symbol [2026-03-27 20:07]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Symbol | Trades | Win Rate | Sharpe | Sortino | MDD    | Return |
| ------ | ------ | -------- | ------ | ------- | ------ | ------ |
| HPG    | 15     | 20.0%    | -3.976 | -4.484  | 3.07%  | -3.20% |
| VNM    | 24     | 50.0%    | 1.470  | 4.352   | 2.65%  | 5.61%  |
| FPT    | 22     | 31.8%    | -2.088 | -3.331  | 15.42% | -9.51% |


---

## Giai đoạn 5 — MCMC + HMM Comparison Backtest [2026-03-27 20:08]

**Strategy:** MCMC + HMM Comparison | **Symbol:** VN30F1M | **Confidence:** 0.75 | **Warmup:** 30 bars

---

## Backtest: MCMC -- Suite 1 In-Sample [2026-03-27 20:08]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 85     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 2 OOS [2026-03-27 20:08]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Bars          | 52     |
| Trades        | 0      |
| Win rate      | 0.0%   |
| Total P&L     | 0.00   |
| Total return  | 0.00%  |
| CAGR          | 0.00%  |
| Sharpe ratio  | 0.000  |
| Sortino ratio | 0.000  |
| Max drawdown  | 0.00%  |
| Calmar ratio  | 0.000  |
| Profit factor | 0.000  |
| Expectancy    | 0.0000 |
| Avg win       | 0.0000 |
| Avg loss      | 0.0000 |
| Avg MCMC ms   | 0.0    |


---

## Backtest: MCMC -- Suite 3 Monte Carlo [2026-03-27 20:09]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Statistic            | Value   |
| -------------------- | ------- |
| Actual Sharpe        | 1.9846  |
| Perm mean Sharpe     | -0.5086 |
| Perm std Sharpe      | 1.6884  |
| Perm 95th pctile     | 2.2020  |
| p-value              | 0.0720  |
| Significant (p<0.05) | NO      |
| N permutations       | 500     |
| Elapsed              | 14.6s   |


**Interpretation:** Cannot reject null -- may be due to chance

---

## Backtest: MCMC -- Suite 4 Hourly [2026-03-27 20:09]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 29      |
| Win rate      | 34.5%   |
| Total P&L     | 53.97   |
| Total return  | 53.97%  |
| CAGR          | 335.56% |
| Sharpe ratio  | 1.985   |
| Sortino ratio | 4.812   |
| Max drawdown  | 18.17%  |
| Calmar ratio  | 18.468  |
| Profit factor | 1.810   |
| Expectancy    | 1.8610  |
| Avg win       | 12.0590 |
| Avg loss      | -5.1246 |
| Avg MCMC ms   | 0.0     |


---

## Backtest: MCMC -- Suite 5 Multi-Symbol [2026-03-27 20:09]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Symbol | Trades | Win Rate | Sharpe  | Sortino | MDD   | Return |
| ------ | ------ | -------- | ------- | ------- | ----- | ------ |
| HPG    | 9      | 33.3%    | -1.161  | -3.775  | 0.85% | -0.75% |
| VNM    | 4      | 25.0%    | -2.081  | -3.166  | 2.99% | -2.60% |
| FPT    | 4      | 0.0%     | -18.152 | -18.152 | 6.57% | -6.57% |


---

## Backtest: HMM -- Suite 1 In-Sample [2026-03-27 20:09]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars

*Training window: 20240101–20251231 (497 bars)*


| Metric        | Value    |
| ------------- | -------- |
| Bars          | 85       |
| Trades        | 25       |
| Win rate      | 48.0%    |
| Total P&L     | 79.11    |
| Total return  | 79.11%   |
| CAGR          | 462.89%  |
| Sharpe ratio  | 0.888    |
| Sortino ratio | 1.934    |
| Max drawdown  | 19.20%   |
| Calmar ratio  | 24.105   |
| Profit factor | 1.347    |
| Expectancy    | 3.1644   |
| Avg win       | 25.6083  |
| Avg loss      | -17.5531 |
| Avg MCMC ms   | 0.0      |


---

## Backtest: HMM -- Suite 2 OOS [2026-03-27 20:10]

**Symbol:** VN30F1M | **Period:** 20260101-20260325 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value     |
| ------------- | --------- |
| Bars          | 52        |
| Trades        | 15        |
| Win rate      | 66.7%     |
| Total P&L     | 178.63    |
| Total return  | 178.63%   |
| CAGR          | 14244.11% |
| Sharpe ratio  | 2.470     |
| Sortino ratio | 5.281     |
| Max drawdown  | 13.65%    |
| Calmar ratio  | 1043.419  |
| Profit factor | 2.367     |
| Expectancy    | 11.9087   |
| Avg win       | 30.9340   |
| Avg loss      | -26.1420  |
| Avg MCMC ms   | 0.0       |


---

## Backtest: HMM -- Suite 3 Hourly (Actual Sharpe) [2026-03-27 20:10]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value  |
| ------------- | ------ |
| Actual Sharpe | 0.0409 |
| Trades        | 216    |
| Win Rate      | 42.1%  |
| MDD           | 87.42% |


**Note:** HMM permutation test skipped — each permutation requires ~500 GaussianHMM re-fits (100× slower than MCMC). See MCMC Suite 3 for p-value reference (p=0.072).

---

## Backtest: HMM -- Suite 4 Hourly [2026-03-27 20:11]

**Symbol:** VN30F1M | **Period:** 20250901-20251231 | **Bar type:** Hourly 1H
**Confidence:** 0.75 | **Warmup:** 30 bars


| Metric        | Value   |
| ------------- | ------- |
| Bars          | 425     |
| Trades        | 216     |
| Win rate      | 42.1%   |
| Total P&L     | 3.24    |
| Total return  | 3.24%   |
| CAGR          | 11.48%  |
| Sharpe ratio  | 0.041   |
| Sortino ratio | 0.050   |
| Max drawdown  | 87.42%  |
| Calmar ratio  | 0.131   |
| Profit factor | 1.005   |
| Expectancy    | 0.0150  |
| Avg win       | 7.5789  |
| Avg loss      | -7.1504 |
| Avg MCMC ms   | 0.0     |


---

## Backtest: HMM -- Suite 5 Multi-Symbol [2026-03-27 20:12]

**Symbol:** HPG+VNM+FPT | **Period:** 20250901-20251231 | **Bar type:** Daily 1D
**Confidence:** 0.75 | **Warmup:** 30 bars


| Symbol | Trades | Win Rate | Sharpe | Sortino | MDD    | Return |
| ------ | ------ | -------- | ------ | ------- | ------ | ------ |
| HPG    | 15     | 20.0%    | -3.976 | -4.484  | 3.07%  | -3.20% |
| VNM    | 24     | 50.0%    | 1.470  | 4.352   | 2.65%  | 5.61%  |
| FPT    | 22     | 31.8%    | -2.088 | -3.331  | 15.42% | -9.51% |


---

## MCMC vs HMM Comparison [2026-03-27 20:12]

**Symbol:** VN30F1M | **Confidence:** 0.75 | **Warmup:** 30 | **HMM states:** 3


| Suite          | Metric        | MCMC   | HMM     | Winner |
| -------------- | ------------- | ------ | ------- | ------ |
| Suite 1 IS     | Sharpe        | 0.000  | 0.888   | HMM    |
| Suite 1 IS     | Win Rate      | 0.000  | 48.000  | HMM    |
| Suite 1 IS     | MDD %         | 0.000  | 19.203  | MCMC   |
| Suite 1 IS     | Trades        | 0.000  | 25.000  | HMM    |
| Suite 2 OOS    | Sharpe        | 0.000  | 2.470   | HMM    |
| Suite 2 OOS    | Win Rate      | 0.000  | 66.667  | HMM    |
| Suite 2 OOS    | MDD %         | 0.000  | 13.651  | MCMC   |
| Suite 2 OOS    | Trades        | 0.000  | 15.000  | HMM    |
| Suite 4 Hourly | Sharpe        | 1.985  | 0.041   | MCMC   |
| Suite 4 Hourly | Win Rate      | 34.483 | 42.130  | HMM    |
| Suite 4 Hourly | MDD %         | 18.170 | 87.421  | MCMC   |
| Suite 4 Hourly | Trades        | 29.000 | 216.000 | HMM    |
| Suite 3 MC     | Actual Sharpe | 1.9846 | 0.0409  | MCMC   |
| Suite 3 MC     | p-value       | 0.0720 | N/A     | —      |


> **Note:** VN30F1M uses VN30 index as proxy (DNSE REST API does not serve derivative historical OHLC). HMM trained on extended window (20240101–20251231) for better regime estimation.

---

# HMM Live-Readiness Gate Report [2026-03-27 23:14]

**HMM config:** k_states=3 | n_iter=50 | zscore_window=20 | fast=True

## 1. Rolling OOS Robustness (1D, VN30F1M proxy)


| Window  | Trades | Sharpe | MDD%  | WinRate% | Return% | LL (IS) |
| ------- | ------ | ------ | ----- | -------- | ------- | ------- |
| Q1-2025 | 18     | 0.082  | 23.7% | 55.6%    | 1.30%   | -881.7  |
| Q2-2025 | 14     | 5.010  | 4.3%  | 64.3%    | 197.05% | -1129.4 |
| Q3-2025 | 18     | 1.243  | 20.5% | 66.7%    | 72.60%  | -1347.1 |
| Q4-2025 | 24     | 0.155  | 23.9% | 41.7%    | 12.33%  | -1592.1 |
| Q1-2026 | 14     | 2.003  | 14.8% | 64.3%    | 143.40% | -1850.6 |


**Median Sharpe:** 1.243 | **Min:** 0.082 | **Max:** 5.010

## 2. Multi-Timeframe Results (Sep–Dec 2025)


| Timeframe | Bars | Trades | Sharpe | MDD%   | WinRate% |
| --------- | ---- | ------ | ------ | ------ | -------- |
| 1D        | 86   | 30     | 1.793  | 44.2%  | 43.3%    |
| 1H        | 425  | 217    | -0.440 | 103.5% | 41.5%    |
| 4H        | 107  | 36     | 1.789  | 47.4%  | 61.1%    |


## 3. Baum-Welch / Log-Likelihood Diagnostics


| Metric                    | Value    |
| ------------------------- | -------- |
| Training samples          | 497      |
| Final LL                  | -1850.64 |
| LL / sample               | -3.7236  |
| LL history length (iters) | 50       |
| LL monotonic              | YES      |
| Multi-seed LL CV          | 0.006    |
| AIC (K=2)                 | 3845.5   |
| AIC (K=3)                 | 3771.3   |
| AIC (K=4)                 | 3710.1   |
| BIC (K=3)                 | 3918.6   |
| K=3 AIC gap from best     | 1.6%     |


## 4. Hidden-State Separation and Labeling


| Metric                       | Value                 |
| ---------------------------- | --------------------- |
| Regime semantics correct     | YES                   |
| BULL mean log-return         | 0.50456               |
| FLAT mean log-return         | 0.00691               |
| BEAR mean log-return         | -0.58256              |
| Min pairwise Mahal. distance | 1.466                 |
| Self-transition probs        | [0.672, 0.064, 0.610] |
| Persistent states (>0.30)    | 2/3                   |


## 5. Performance Benchmarks


| Metric               | p50 | p90 | p95 |
| -------------------- | --- | --- | --- |
| Fit latency (ms)     | 154 | 160 | 160 |
| Predict latency (ms) | 0.8 | 1.0 | 2.0 |


**Fit failure rate:** 0.0%

## 6. Go/No-Go Scorecard


| Gate                               | Measured | Threshold | Result   | Severity |
| ---------------------------------- | -------- | --------- | -------- | -------- |
| OOS Sharpe (median)                | 1.243    | ≥ 0.8     | **PASS** | HARD     |
| OOS Max Drawdown                   | 23.9%    | ≤ 30.0%   | **PASS** | HARD     |
| OOS Positive Windows               | 5/5      | ≥ 3       | **PASS** | HARD     |
| Baum-Welch LL monotonic            | YES      | YES       | **PASS** | HARD     |
| Multi-seed LL stability (CV)       | 0.006    | ≤ 0.25    | **PASS** | WARN     |
| K=3 AIC gap from best              | 1.6%     | ≤ 10%     | **PASS** | WARN     |
| Regime semantics (BULL>BEAR)       | CORRECT  | CORRECT   | **PASS** | HARD     |
| Min state Mahal. distance          | 1.466    | ≥ 0.8     | **PASS** | WARN     |
| Regime persistence (states > 0.30) | 2/3      | ≥ 2       | **PASS** | WARN     |
| Fit latency p90                    | 160ms    | ≤ 3000ms  | **PASS** | WARN     |
| Predict latency p95                | 2.0ms    | ≤ 200ms   | **PASS** | WARN     |
| Fit failure rate                   | 0.0%     | ≤ 5%      | **PASS** | WARN     |


**Score: 12/12 gates passed.**

### Decision: GO

> All gates passed. Proceed to **2-4 week paper-trading shadow run** before deploying small capital.

> **Proxy note:** VN30F1M historical data is proxied via VN30 index (DNSE REST API does not serve derivative historical OHLC). Live futures may diverge from index in high-volatility periods.

---

## HMM backtest — so sánh (W=500, S=5) vs (W=2000, S=50)

**Ngày:** 2026-03-28  
**Script:** `python scripts/backtest.py --strategy hmm --suite insample|oos`  
**Symbol:** VN30F1M daily | Cùng tham số: confidence=0.75, warmup=30, k=3, iter=200  
**CSV:** `reports/hmm_compare_W500_S5_*.csv`, `reports/hmm_compare_W2000_S50_*.csv`

**Lưu ý dữ liệu:** Train+test daily có ~497–550 bar; **W=2000 > số bar** nên cửa sổ 2000 bar *không bị cắt* — thực tế dùng toàn bộ chuỗi; **W=500** mới rolling 500 bar khi đủ dài. So sánh phản ánh cả **refit_every (5 vs 50)** và **độ rộng cửa sổ khi W=500**.


| Period                 | Config       | Trades | Win% | Sharpe | MDD%  |
| ---------------------- | ------------ | ------ | ---- | ------ | ----- |
| In-sample Sep–Dec 2025 | W=500, S=5   | 25     | 48.0 | 0.888  | 19.20 |
| In-sample Sep–Dec 2025 | W=2000, S=50 | 24     | 58.3 | 1.987  | 31.77 |
| OOS Jan–Mar 2026       | W=500, S=5   | 22     | 40.9 | -2.338 | 44.73 |
| OOS Jan–Mar 2026       | W=2000, S=50 | 20     | 50.0 | 2.791  | 10.08 |


**CLI mới:** `--sliding-window-bars`, `--csv-out` (ghi log trade ra file tùy chọn).

---

## HMM — optimization loop (`config/hmm_optimization.yaml`)

**Ngày:** 2026-03-29  
**Script:** `python scripts/hmm_optimize_loop.py`  
**Objective:** Score = 0.4×Calmar + 0.3×Sharpe + 0.3×PF (clipped); constraints: MDD < 20%, trades > 100, PF > 1.2, increment-shuffle p < 0.05.  
**Kết quả:** Trên mẫu VN30 proxy Sep–Dec 2025 (`max_bars: 650`, grid `optimization_refit_every=30`, `n_iter=40`), **0/36** cấu hình thỏa đủ ràng buộc; top raw score trên **15m**: W∈{1000,2000,3000}, K=3, Score*≈6.24 nhưng MDD≈45%.  
**Mặc định app:** `HMM_LIVE_TRAIN_WINDOW_BARS=1000`, `HMM_LIVE_REFIT_EVERY=5`, `HMM_LIVE_K_STATES=3` (best-effort theo 15m, có cảnh báo rủi ro).  
**Evidence:** `reports/hmm_optimization_results.csv`, `reports/hmm_optimization_summary.md`  
**Unit test:** `tests/test_equity_significance.py` — 1/1 PASS.

---

## TTM — dual-system (V1 / V2 / shadow)

**Ngày:** 2026-04-01  
**File test:** `tests/test_ttm_probabilistic.py`, `tests/test_ttm_opt.py`  
**Lệnh:** `python -m pytest tests/test_ttm_probabilistic.py tests/test_ttm_opt.py -q`  
**Kết quả:** 14/14 + 6/6 PASS (TTM stack)


| #   | Test case                         | Mô tả                                                      | Kết quả |
| --- | --------------------------------- | ---------------------------------------------------------- | ------- |
| 1   | compute_features keys             | `failure_strength`, `vol_regime`, `oi_signal` trong bundle | PASS    |
| 2   | v1_only vs generate_ttm_signal_v1 | Hành vi mặc định không đổi                                 | PASS    |
| 3   | shadow                            | `action` = V1; `v2` + `divergence` logged                  | PASS    |
| 4   | v2_only fallback                  | Dữ liệu rỗng → fallback V1                                 | PASS    |
| 5   | sigmoid / directional probs       | Model V2                                                   | PASS    |
| 6   | ttm_opt smoke                     | Backtest opt stack                                         | PASS    |


**Ghi chú:** `pytest-cov` thêm vào `requirements.txt`; coverage package `src/strategies/ttm` chạy `pytest --cov=src/strategies/ttm` khi cần audit.

### TTM V2 — execution realism (latency / slippage)

**Ngày:** 2026-04-10  
**File test:** `tests/test_ttm_parallel_runner.py`, `tests/test_ttm_execution_realism.py`  
**Lệnh:** `python -m pytest tests/test_ttm_parallel_runner.py tests/test_ttm_execution_realism.py -q`  
**Kết quả:** 7/7 PASS  


| #   | Test case                                  | Mô tả                                     | Kết quả |
| --- | ------------------------------------------ | ----------------------------------------- | ------- |
| 1   | `test_parallel_runner_jsonl_and_summary`   | JSONL + summary (legacy, không realism)   | ✅ PASS  |
| 2   | `test_parallel_runner_deterministic`       | Deterministic replay                      | ✅ PASS  |
| 3   | `test_resolution_to_bar_seconds`           | Map resolution → giây/nến                 | ✅ PASS  |
| 4   | `test_entry_due_unix_monotonic`            | `entry_due_unix` tăng theo latency        | ✅ PASS  |
| 5   | `test_slippage_modes`                      | none / base / worst_case                  | ✅ PASS  |
| 6   | `test_adjust_fill_price_direction`         | Chiều slippage long/short                 | ✅ PASS  |
| 7   | `test_replay_with_execution_realism_smoke` | `replay_bars(..., execution_realism=...)` | ✅ PASS  |


**Script grid:** `scripts/ttm_v2_execution_grid.py` — lưới `latency_ms × slippage_mode`, output JSON trong `reports/`.

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 07:42]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 8m 30s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Unit Tests -- TTM breakout continuation/exhaustion refactor [2026-04-16]

**Ngày test:** 2026-04-16  
**File test:** `tests/test_ttm_parallel_runner.py`, `tests/test_ttm_volume_features.py`, `tests/test_ttm_validation.py`  
**Kết quả:** 11/12 PASS, 0 FAIL, 1 SKIP

### Phương pháp / Evidence

- Lệnh chạy: `python -m pytest tests/test_ttm_parallel_runner.py tests/test_ttm_volume_features.py tests/test_ttm_validation.py`
- Kết quả thực tế: `11 passed, 1 skipped, 4 warnings in 5.64s`
- Warnings:
  - `DeprecationWarning` từ `vendor/dnse/trading_websocket/connection.py` và `websockets.legacy` (không liên quan refactor breakout)
  - `ConstantInputWarning` trong `test_breakout_validation_skips_exhaustion_candidates` vì dataset synthetic cố ý giữ score hằng để chỉ test breakout filter

### Test cases


| #   | Test case                                                    | Module                        | Mô tả                                                                                        | Kết quả |
| --- | ------------------------------------------------------------ | ----------------------------- | -------------------------------------------------------------------------------------------- | ------- |
| 1   | `test_parallel_runner_jsonl_and_summary`                     | `test_ttm_parallel_runner.py` | Verify replay xuất JSONL, closes export, và features snapshot có `exhaustion_candidate`      | ✅ PASS  |
| 2   | `test_parallel_runner_deterministic`                         | `test_ttm_parallel_runner.py` | Replay cùng input cho summary ổn định                                                        | ✅ PASS  |
| 3   | `test_vol_signal_not_flat_with_varying_volume`               | `test_ttm_volume_features.py` | Volume signal vẫn biến thiên khi volume thay đổi                                             | ✅ PASS  |
| 4   | `test_vol_signal_distribution_roughly_symmetric`             | `test_ttm_volume_features.py` | Volume signal mid-series gần đối xứng quanh 0                                                | ✅ PASS  |
| 5   | `test_volume_participation_log_schema`                       | `test_ttm_volume_features.py` | Schema log participation không đổi                                                           | ✅ PASS  |
| 6   | `test_breakout_strength_splits_continuation_from_exhaustion` | `test_ttm_volume_features.py` | LONG breakout dùng continuation-only; exhaustion bị loại khỏi `breakout_up` và mask strength | ✅ PASS  |
| 7   | `test_volume_features_disabled_is_noop`                      | `test_ttm_volume_features.py` | Tắt volume features không làm vỡ output cơ bản                                               | ✅ PASS  |
| 8   | `test_run_validation_empty_decisions`                        | `test_ttm_validation.py`      | Empty decisions vẫn fail an toàn, không crash                                                | ✅ PASS  |
| 9   | `test_synthetic_with_closes_engineered`                      | `test_ttm_validation.py`      | Synthetic closes vẫn chạy full validation pipeline                                           | ✅ PASS  |
| 10  | `test_load_decisions_sorts_by_bar_index`                     | `test_ttm_validation.py`      | Loader giữ thứ tự `bar_index` đúng                                                           | ✅ PASS  |
| 11  | `test_breakout_validation_skips_exhaustion_candidates`       | `test_ttm_validation.py`      | Validator loại `exhaustion_candidate` khỏi bucket monotonic nhưng giữ raw breakout stats     | ✅ PASS  |
| 12  | `test_smoke_repo_backtest_if_present`                        | `test_ttm_validation.py`      | Smoke test report pair trong workspace                                                       | ⏭️ SKIP |


### Đánh giá

- Đã cover: cap-causal cho raw breakout strength, `breakout_up = continuation only`, `exhaustion_candidate` snapshot vào JSONL, và filter validator để exhaustion không lọt bucket monotonic.
- Đã giữ tương thích ngược: log cũ không có `exhaustion_candidate` vẫn đi qua validator với default `False`.
- Chưa cover: regression OOS/WFE trên report lịch sử thật sau refactor, và full-suite `pytest tests/ -v` cho toàn repo.

---

## Unit Tests -- TTM V2 effective_strength calibration [2026-04-19]

**Ngày test:** 2026-04-19  
**File test:** `tests/test_ttm_effective_strength.py`, `tests/test_ttm_parallel_runner.py`, `tests/test_ttm_validation.py`, `tests/test_ttm_v2_alpha.py`  
**Kết quả:** 16/16 PASS (nhóm chạy), 1 SKIP từ `test_ttm_validation`

### Phương pháp / Evidence

- Lệnh: `python -m pytest tests/test_ttm_validation.py tests/test_ttm_effective_strength.py tests/test_ttm_parallel_runner.py tests/test_ttm_v2_alpha.py -q`
- Kết quả: `16 passed, 1 skipped`
- Thay đổi chính: `price_z` / `extension` / `last_bar_return` / `effective_strength` trong `ttm_features`; `compute_score_v2_alpha` đọc `effective_strength` khi có key; JSONL snapshot + `entry_calibration` trên trade V2 LONG đóng; `test_breakout` dùng trục `effective_strength` khi có, filter `raw_strength > 0` + optional `breakout_up_filtered_last`; metrics `mean_mid_ge_mean_high`, `anti_inverted_u_ok`, `v2_trade_holding`.

### Test cases (rút gọn)


| #   | Test case                                                         | Module                           | Mô tả                                                                 | Kết quả |
| --- | ----------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------- | ------- |
| 1   | `test_effective_strength_matches_raw_when_k_zero`                 | `test_ttm_effective_strength.py` | k_extension=k_lastret=0 thì effective = raw trên các bar finite       | ✅ PASS  |
| 2   | `test_last_bar_return_no_future_close`                            | `test_ttm_effective_strength.py` | Công thức return bar trước không dùng close hiện tại                  | ✅ PASS  |
| 3   | `test_compute_score_v2_alpha_reads_effective_strength_key`        | `test_ttm_effective_strength.py` | `effective_strength_last` trong components khớp slice tại `bar_index` | ✅ PASS  |
| 4   | `test_breakout_validation_prefers_effective_strength_in_features` | `test_ttm_effective_strength.py` | `strength_axis` = effective khi log có field                          | ✅ PASS  |
| 5   | `test_parallel_runner_jsonl_and_summary`                          | `test_ttm_parallel_runner.py`    | Snapshot có `extension` / `last_bar_return` / `effective_strength`    | ✅ PASS  |


### Đánh giá

- Defaults `ttm_v2_effective_strength_k_* = 0` giữ hành vi V2 gần như cũ cho đến khi PM tune k1/k2.
- Synthetic `test_ttm_validation` đã bổ sung `raw_strength` + `rolling_high` để khớp filter `valid_breakout` mới.

---

## Unit Tests -- TTM SHORT exhaustion alpha refactor [2026-04-16]

**Ngày test:** 2026-04-16  
**File test:** `tests/test_ttm_v2_alpha.py`, `tests/test_ttm_v2_short.py`, `tests/test_ttm_volume_features.py`, `tests/test_ttm_parallel_runner.py`, `tests/test_ttm_validation.py`  
**Kết quả:** 20/21 PASS, 0 FAIL, 1 SKIP

### Phương pháp / Evidence

- Lệnh chạy: `python -m pytest tests/test_ttm_v2_alpha.py tests/test_ttm_v2_short.py tests/test_ttm_volume_features.py tests/test_ttm_parallel_runner.py tests/test_ttm_validation.py -q`
- Kết quả thực tế: `20 passed, 1 skipped, 6 warnings in 6.84s`
- Warnings:
  - `DeprecationWarning` từ `vendor/dnse/trading_websocket/connection.py` và `websockets.legacy` (không liên quan refactor SHORT exhaustion)
  - `ConstantInputWarning` trong `test_breakout_validation_skips_exhaustion_candidates` và `test_scoring_short_prefers_feature_short_score` vì dataset synthetic cố ý giữ series hằng ngoài cột đang validate

### Test cases


| #   | Test case                                                      | Module                        | Mô tả                                                                                      | Kết quả |
| --- | -------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------ | ------- |
| 1   | `test_v2_alpha_ignores_oi_for_score`                           | `test_ttm_v2_alpha.py`        | V2 long/short alpha không phụ thuộc đường OI khi price/vol/basis giữ nguyên                | ✅ PASS  |
| 2   | `test_score_distribution_not_saturated_on_series`              | `test_ttm_v2_alpha.py`        | Score distribution không bị dính trần hàng loạt trên chuỗi dài                             | ✅ PASS  |
| 3   | `test_v2_short_score_uses_dedicated_exhaustion_leg`            | `test_ttm_v2_alpha.py`        | Confirmed exhaustion đi vào `score_short` riêng thay vì symmetry `-score_long`             | ✅ PASS  |
| 4   | `test_v2_alpha_component_keys`                                 | `test_ttm_v2_alpha.py`        | Component log xuất đủ `alpha_raw_short`, `alpha_rank_short`, và legacy aliases             | ✅ PASS  |
| 5   | `test_v2_vol_breakout_interaction_optional`                    | `test_ttm_v2_alpha.py`        | Hệ số vol-breakout interaction vẫn hoạt động sau refactor score                            | ✅ PASS  |
| 6   | `test_build_exhaustion_short_entry_meta_from_confirmed_setup`  | `test_ttm_v2_short.py`        | Helper chung dựng metadata SL/TP/time-stop từ exhaustion-confirm setup                     | ✅ PASS  |
| 7   | `test_check_exhaustion_short_exit_rules`                       | `test_ttm_v2_short.py`        | Helper chung trả đúng lý do thoát SL/TP/TIME cho SHORT exhaustion                          | ✅ PASS  |
| 8   | `test_vol_signal_not_flat_with_varying_volume`                 | `test_ttm_volume_features.py` | Volume signal vẫn biến thiên khi volume thay đổi                                           | ✅ PASS  |
| 9   | `test_vol_signal_distribution_roughly_symmetric`               | `test_ttm_volume_features.py` | Volume signal mid-series gần đối xứng quanh 0                                              | ✅ PASS  |
| 10  | `test_volume_participation_log_schema`                         | `test_ttm_volume_features.py` | Schema log participation không đổi                                                         | ✅ PASS  |
| 11  | `test_breakout_strength_splits_continuation_from_exhaustion`   | `test_ttm_volume_features.py` | LONG continuation-only không nhận exhaustion bar vào `breakout_up`                         | ✅ PASS  |
| 12  | `test_exhaustion_confirm_and_short_score_shift_to_confirm_bar` | `test_ttm_volume_features.py` | `exhaustion_confirm` và `short_score` được shift sang đúng bar xác nhận, không leak future | ✅ PASS  |
| 13  | `test_volume_features_disabled_is_noop`                        | `test_ttm_volume_features.py` | Tắt volume features không làm vỡ output cơ bản                                             | ✅ PASS  |
| 14  | `test_parallel_runner_jsonl_and_summary`                       | `test_ttm_parallel_runner.py` | Replay JSONL xuất đủ `raw_strength`, `cap`, `exhaustion_confirm`, `short_score`            | ✅ PASS  |
| 15  | `test_parallel_runner_deterministic`                           | `test_ttm_parallel_runner.py` | Replay cùng input cho summary ổn định                                                      | ✅ PASS  |
| 16  | `test_run_validation_empty_decisions`                          | `test_ttm_validation.py`      | Empty decisions vẫn fail an toàn, không crash                                              | ✅ PASS  |
| 17  | `test_synthetic_with_closes_engineered`                        | `test_ttm_validation.py`      | Synthetic closes vẫn chạy full validation pipeline                                         | ✅ PASS  |
| 18  | `test_load_decisions_sorts_by_bar_index`                       | `test_ttm_validation.py`      | Loader giữ thứ tự `bar_index` đúng                                                         | ✅ PASS  |
| 19  | `test_breakout_validation_skips_exhaustion_candidates`         | `test_ttm_validation.py`      | Breakout validator tiếp tục loại exhaustion khỏi bucket LONG                               | ✅ PASS  |
| 20  | `test_scoring_short_prefers_feature_short_score`               | `test_ttm_validation.py`      | `scoring_short` ưu tiên `features.short_score` thay vì `v2.score_short` đối xứng           | ✅ PASS  |
| 21  | `test_smoke_repo_backtest_if_present`                          | `test_ttm_validation.py`      | Smoke test report pair trong workspace                                                     | ⏭️ SKIP |


### Đánh giá

- Đã cover: feature primitives (`exhaustion_confirm`, `short_score`), signal SHORT mới, V2 score tách long/short leg, JSONL snapshot mới, validator đọc `features.short_score`, và helper ATR exits dùng chung cho paper/replay/live.
- Đã verify regression gần kề: `python -m pytest tests/test_ttm_probabilistic.py tests/test_ttm_adaptive.py -q` cho kết quả `22 passed, 2 warnings in 1.63s`.
- Chưa cover: end-to-end live runner queue `SHORT_PENDING -> next bar open` bằng test tự động, và replay/backtest lịch sử thật để đánh giá monotonic/PF/Sharpe sau refactor.

---

## Unit Tests -- TTM SHORT exhaustion alpha verification rerun [2026-04-16]

**Ngày test:** 2026-04-16  
**File test:** `tests/test_ttm_v2_alpha.py`, `tests/test_ttm_v2_short.py`, `tests/test_ttm_volume_features.py`, `tests/test_ttm_parallel_runner.py`, `tests/test_ttm_validation.py`  
**Kết quả:** 20/21 PASS, 0 FAIL, 1 SKIP

### Phương pháp / Evidence

- Lệnh chạy: `python -m pytest tests/test_ttm_v2_alpha.py tests/test_ttm_v2_short.py tests/test_ttm_volume_features.py tests/test_ttm_parallel_runner.py tests/test_ttm_validation.py -q`
- Kết quả thực tế: `20 passed, 1 skipped, 6 warnings in 8.76s`
- Warnings:
  - `DeprecationWarning` từ `vendor/dnse/trading_websocket/connection.py` và `websockets.legacy` (không liên quan refactor SHORT exhaustion)
  - `ConstantInputWarning` trong `test_breakout_validation_skips_exhaustion_candidates` và `test_scoring_short_prefers_feature_short_score` vì dataset synthetic cố ý giữ series hằng ngoài cột đang validate

### Đánh giá

- Rerun xác nhận toàn bộ suite focus của plan vẫn pass trên workspace hiện tại, bao gồm feature primitives, signal replacement, score split, JSONL snapshot, validation source cho `short_score`, và helper exits.
- Không phát hiện regression mới trong các file cốt lõi của plan SHORT exhaustion ở lần verify này.

---

## Unit Tests -- Extended regression spot-check [2026-04-16]

**Ngày test:** 2026-04-16  
**File test:** `tests/test_ttm_probabilistic.py`, `tests/test_ttm_adaptive.py`, `tests/test_ttm_audit_refinement.py`  
**Kết quả:** 34 PASS, 1 FAIL

### Phương pháp / Evidence

- Lệnh chạy: `python -m pytest tests/test_ttm_probabilistic.py tests/test_ttm_adaptive.py tests/test_ttm_audit_refinement.py -q`
- Kết quả thực tế: `1 failed, 34 passed, 2 warnings in 2.05s`

### FAIL / Root cause / Action items


| #   | Test case                                            | Module                         | Root cause                                                                                                                                      | Action items                                                                                                                                       | Kết quả |
| --- | ---------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| 1   | `test_oi_signal_tanh_has_spread_when_oi_levels_vary` | `test_ttm_audit_refinement.py` | Test đang assert exact key-set của `last["oi_pipeline"]`, nhưng output hiện có thêm `oi_proxy_used`; fail này không đụng logic SHORT exhaustion | Cần quyết định schema chuẩn cho `oi_pipeline`: hoặc update test để chấp nhận superset, hoặc bỏ `oi_proxy_used` khỏi snapshot nếu không muốn public | ❌ FAIL  |


---

## Real-data replay/validation -- TTM V2 continuation vs exhaustion [2026-04-16]

**Ngày test:** 2026-04-16  
**Script:** `scripts/ttm_parallel_backtest.py`, `scripts/ttm_validation.py`  
**Loại test:** Replay + validation trên dữ liệu thật 5m

### Phương pháp / Evidence

- Replay command: `python scripts/ttm_parallel_backtest.py --from-date 20260301 --to-date 20260414 --resolution 5`
- Validation command: `python scripts/ttm_validation.py --decisions reports/ttm_parallel_backtest_VN30F1M_20260301_20260414_20260416_1049_decisions.jsonl --trades reports/ttm_parallel_backtest_VN30F1M_20260301_20260414_20260416_1049_trades.jsonl --closes reports/closes_ttm_parallel_backtest_VN30F1M_20260301_20260414_20260416_1049_decisions.json --scoring-mode all --json-out reports/ttm_validation_VN30F1M_20260301_20260414_20260416_1049_short_check.json`
- Input sample: `1472` bar 5m, `20260301–20260414`, basis/OI có mặt
- Output files:
  - `reports/ttm_parallel_backtest_VN30F1M_20260301_20260414_20260416_1049_decisions.jsonl`
  - `reports/ttm_parallel_backtest_VN30F1M_20260301_20260414_20260416_1049_trades.jsonl`
  - `reports/ttm_parallel_backtest_VN30F1M_20260301_20260414_20260416_1049_summary.json`
  - `reports/ttm_validation_VN30F1M_20260301_20260414_20260416_1049_short_check.json`

### Kết quả chính


| Nhóm             | Chỉ số                     | Giá trị                                   |
| ---------------- | -------------------------- | ----------------------------------------- |
| Replay V2        | Closed trades              | `26`                                      |
| Replay V2        | Winrate                    | `46.15%`                                  |
| Replay V2        | PnL                        | `-21.6`                                   |
| Replay V2        | Profit Factor              | `0.697`                                   |
| Replay V2        | Sharpe proxy               | `-0.102`                                  |
| V2 long trades   | Count / PF / Sharpe        | `16 / 0.454 / -0.336`                     |
| V2 short trades  | Count / PF / Sharpe        | `10 / 1.059 / 0.186`                      |
| Exhaustion flow  | `exhaustion_confirm` bars  | `12`                                      |
| Exhaustion flow  | SHORT decisions            | `10`                                      |
| Validation short | score source               | `features.short_score`                    |
| Validation short | n samples                  | `12`                                      |
| Validation short | Pearson / Spearman         | `0.327 / 0.420`                           |
| Validation short | bucket mean aligned return | `low=-0.00815, mid=0.00219, high=0.00133` |


### Phát hiện quan trọng

- Cơ khí refactor SHORT exhaustion hoạt động end-to-end:
  - Không có `SHORT` nào xuất hiện khi `exhaustion_confirm = False`
  - Không có `LONG` nào xuất hiện trên bar có `exhaustion_confirm = True`
  - `scoring_short` đã đọc `features.short_score` thay vì score short đối xứng
- Chất lượng alpha chưa pass thống kê:
  - `scoring_short` fail vì thiếu mẫu (`12 < 20`)
  - Bucket `short_score` chưa monotonic hoàn toàn (`mid > high`)
  - `breakout` continuation vẫn fail trên sample này
- Có dấu hiệu validator breakout chưa align với semantic trade hiện tại:
  - replay cho thấy `89` raw breakout-up bar, `20` tradable continuation bar, `20` exhaustion candidate, `12` exhaustion confirm
  - nhưng validator breakout đang evaluate `34` bar LONG; kiểm tra riêng cho thấy `25` bar trong số đó là raw breakout không tradable với `breakout_strength = 0.0`

### Đánh giá

- Kết luận tạm thời: refactor continuation vs exhaustion **đã chạy đúng dây chuyền kỹ thuật**, và SHORT exhaustion đang cho trade-level metrics tốt hơn LONG continuation trên sample thật này.
- Tuy nhiên chưa thể kết luận “pass validate” hay “alpha đã ổn” vì:
  - `short_score` còn ít mẫu, chưa đủ mạnh để xác nhận monotonic
  - breakout validator hiện có dấu hiệu đang trộn thêm raw breakout zero-strength, nên kết quả breakout hiện tại chưa sạch theo semantic continuation-only
- Chưa thực hiện:
  - sửa validator/source logic (theo yêu cầu không sửa code)
  - so sánh delta PF/Sharpe pre/post trên cùng sample 5m và cùng commit cũ, vì workspace hiện không có baseline 5m cùng period để so sạch

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 09:12]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 77m 17s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 10:35]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 80m 48s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 11:07]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 19m 42s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 11:23]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 11m 6s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 11:26]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 3m 25s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 11:46]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 0m 20s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 13:21]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 8m 41s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 13:30]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 3m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 13:39]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 13:41]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 13:44]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 13:49]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-02 13:51]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-02 13:58]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-02 14:00]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-02 14:03]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 3m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-02 14:15]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 55s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-02 14:23]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 10s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-02 14:41]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 0m 25s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-02 16:51]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 3m 50s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-03 06:47]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 35s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-03 09:07]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 4m 25s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-03 09:15]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 5m 10s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-03 09:22]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 6m 31s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-03 09:25]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 2m 31s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- HPG [2026-04-03 09:26]

### Configuration


| Parameter          | Value |
| ------------------ | ----- |
| Symbol             | HPG   |
| STRATEGY_ALGO      | TTM   |
| breakout_window    | 20    |
| failure_window     | 3     |
| vol_threshold      | 1.2   |
| oi_z_threshold     | 0.8   |
| stop_loss_points   | 8.0   |
| take_profit_points | 12.0  |
| max_bars_in_trade  | 10    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 0m 25s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 09:38]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 4m 40s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 09:45]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 2m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 09:58]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 10m 1s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 10:06]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 5m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 10:21]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 11m 26s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 10:45]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 14m 15s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 11:25]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 23m 1s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 13:26]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 24m 36s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 13:34]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (secdef)     | 41I1G4000 |
| openInterestQuantity (last)    | None      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 10s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 13:38]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 35s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M--debug-oi [2026-04-03 13:47]

### Configuration


| Parameter          | Value             |
| ------------------ | ----------------- |
| Symbol             | VN30F1M--debug-oi |
| STRATEGY_ALGO      | TTM               |
| breakout_window    | 20                |
| failure_window     | 3                 |
| vol_threshold      | 1.2               |
| oi_z_threshold     | 0.8               |
| stop_loss_points   | 8.0               |
| take_profit_points | 12.0              |
| max_bars_in_trade  | 10                |
| use_open_interest  | True              |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 0m 10s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 13:51]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 3m 10s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 14:02]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 3m 45s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 14:20]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 0m 50s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 14:27]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 2m 5s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 14:38]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | True    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 5m 10s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-03 14:54]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 14m 51s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-05 06:46]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | True    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 1m 40s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-06 07:15]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | True    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 3m 55s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-06 15:45]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | 43848     |
| OI source (rest vs websocket)  | websocket |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 473m 50s       |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-07 08:58]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | True    |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 253m 5s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-07 09:29]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 30m 36s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-07 10:57]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 87m 44s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-07 14:16]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | True      |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 67m 27s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-07 15:22]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 59m 45s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-08 09:11]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | False   |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 4m 16s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-08 09:20]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | False   |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 2m 0s          |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-08 10:51]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 85m 47s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-09 12:29]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | 38990     |
| OI source (rest vs websocket)  | websocket |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 307m 4s        |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-09 14:59]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 144m 44s       |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-10 15:11]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | 34483     |
| OI source (rest vs websocket)  | websocket |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 457m 22s       |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-13 15:05]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | 35576     |
| OI source (rest vs websocket)  | websocket |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 428m 52s       |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-14 15:28]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | None      |
| OI source (rest vs websocket)  | none      |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 444m 13s       |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-15 15:00]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G4000 |
| secdef symbol used (API)       | 41I1G4000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | 23830     |
| OI source (rest vs websocket)  | websocket |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 440m 48s       |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-17 06:13]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | False   |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 0m 15s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-17 06:55]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | False   |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 0m 30s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-17 07:00]

### Configuration


| Parameter          | Value   |
| ------------------ | ------- |
| Symbol             | VN30F1M |
| STRATEGY_ALGO      | TTM     |
| breakout_window    | 20      |
| failure_window     | 3       |
| vol_threshold      | 1.2     |
| oi_z_threshold     | 0.8     |
| stop_loss_points   | 8.0     |
| take_profit_points | 12.0    |
| max_bars_in_trade  | 10      |
| use_open_interest  | False   |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 0m 20s         |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders

---

## Paper Test Session (TTM) -- VN30F1M [2026-04-17 15:02]

### Configuration


| Parameter                      | Value     |
| ------------------------------ | --------- |
| Symbol                         | VN30F1M   |
| STRATEGY_ALGO                  | TTM       |
| breakout_window                | 20        |
| failure_window                 | 3         |
| vol_threshold                  | 1.2       |
| oi_z_threshold                 | 0.8       |
| stop_loss_points               | 8.0       |
| take_profit_points             | 12.0      |
| max_bars_in_trade              | 10        |
| use_open_interest              | False     |
| DNSE secdef HTTP status (last) | 200       |
| DNSE trade symbol (resolved)   | 41I1G5000 |
| secdef symbol used (API)       | 41I1G5000 |
| boardId (secdef query)         | G1        |
| openInterestQuantity (last)    | 21242     |
| OI source (rest vs websocket)  | websocket |


### Session Results


| Metric            | Value          |
| ----------------- | -------------- |
| Duration          | 429m 37s       |
| Signals generated | 0              |
| Orders placed     | 0              |
| Paper fills       | 0              |
| Realized P&L      | 0.00           |
| Commission        | 0.00           |
| Net P&L           | 0.00           |
| Win rate          | 0.0% (0W / 0L) |
| Risk halted       | False          |
| Stoploss triggers | 0              |


### Evaluation

- TTM signals logged with strategy/action/confidence/reason
- No overlapping entries when flat
- Risk manager correctly gated orders


---
## Paper Test Session (TTM) -- VN30F1M [2026-04-19 22:44]

### Configuration
| Parameter | Value |
|-----------|-------|
| Symbol | VN30F1M |
| STRATEGY_ALGO | TTM |
| breakout_window | 20 |
| failure_window | 3 |
| vol_threshold | 1.2 |
| oi_z_threshold | 0.8 |
| stop_loss_points | 8.0 |
| take_profit_points | 12.0 |
| max_bars_in_trade | 10 |
| use_open_interest | False |

### Session Results
| Metric | Value |
|--------|-------|
| Duration | 0m 10s |
| Signals generated | 0 |
| Orders placed | 0 |
| Paper fills | 0 |
| Realized P&L | 0.00 |
| Commission | 0.00 |
| Net P&L | 0.00 |
| Win rate | 0.0% (0W / 0L) |
| Risk halted | False |
| Stoploss triggers | 0 |

### Evaluation
- [ ] TTM signals logged with strategy/action/confidence/reason
- [ ] No overlapping entries when flat
- [ ] Risk manager correctly gated orders

---
## Paper Test Session (TTM) -- VN30F1M [2026-04-20 15:19]

### Configuration
| Parameter | Value |
|-----------|-------|
| Symbol | VN30F1M |
| STRATEGY_ALGO | TTM |
| breakout_window | 20 |
| failure_window | 3 |
| vol_threshold | 1.2 |
| oi_z_threshold | 0.8 |
| stop_loss_points | 8.0 |
| take_profit_points | 12.0 |
| max_bars_in_trade | 10 |
| use_open_interest | False |
| DNSE secdef HTTP status (last) | 200 |
| DNSE trade symbol (resolved) | 41I1G5000 |
| secdef symbol used (API) | 41I1G5000 |
| boardId (secdef query) | G1 |
| openInterestQuantity (last) | 32596 |
| OI source (rest vs websocket) | websocket |

### Session Results
| Metric | Value |
|--------|-------|
| Duration | 502m 31s |
| Signals generated | 0 |
| Orders placed | 0 |
| Paper fills | 0 |
| Realized P&L | 0.00 |
| Commission | 0.00 |
| Net P&L | 0.00 |
| Win rate | 0.0% (0W / 0L) |
| Risk halted | False |
| Stoploss triggers | 0 |

### Evaluation
- [ ] TTM signals logged with strategy/action/confidence/reason
- [ ] No overlapping entries when flat
- [ ] Risk manager correctly gated orders
