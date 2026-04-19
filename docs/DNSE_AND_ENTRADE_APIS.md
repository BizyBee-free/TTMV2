# DNSE OpenAPI vs Entrade Smart Order (tài liệu nội bộ)

Tài liệu Word **«Tài liệu API Entrade»** gồm **hai nhóm** không thay thế lẫn nhau:

## 1. Entrade — Smart Order (điều kiện phái sinh)

- **Base URL (thật):** `https://services.entrade.com.vn/smart-order/orders`  
- **Paper:** `https://services.entrade.com.vn/papertrade-smart-order/orders`
- **Auth:** `Authorization: Bearer <trading-token>` (không phải flow HMAC của OpenAPI).
- **Body:** `investorId`, `investorAccountId`, `bankMarginPortfolioId`, `expiredTime` (ISO UTC 8601), `symbol`, `targetPrice`, `targetSide` (NB/NS), `type` (STOP, LIMIT, …), `targetQuantity`, `condition`, …

Đây là **API lệnh điều kiện** (Entrade), **không** trùng với client BeeTrade hiện tại (`openapi.dnse.com.vn` + HMAC + `post_order` tại `/accounts/orders`).

**BeeTrade:** chưa tích hợp Entrade Smart Order. Nếu cần lệnh STOP/điều kiện qua Entrade, phải thêm module riêng (Bearer, payload khác).

---

## 2. DNSE — OpenAPI V2 (đang dùng trong BeeTrade)

Thông báo trong tài liệu (cập nhật ~03/2026):

| Thay đổi | Cách xử lý trong BeeTrade |
|-----------|----------------------------|
| `get deals` → **get positions** (đổi tên) | SDK: thêm `get_positions` → `GET /accounts/{accountNo}/positions`. `BeeTradeClient.get_deals()` gọi **positions trước**, nếu **404** và `DNSE_POSITIONS_FALLBACK_TO_DEALS=true` thì gọi lại `/deals`. |
| Endpoint **secdef** cập nhật | Đã đồng bộ với [openapi-sdk Python](https://github.com/dnse-tech/openapi-sdk): `GET /price/{symbol}/secdef`. Chi tiết: [`DNSE_SDK_SYNC.md`](DNSE_SDK_SYNC.md). |
| OHLC / tick history / latest tick / close position / instruments | Phần lớn đã có trong SDK (`get_ohlc`, `get_trades`, `get_latest_trade`, `close_position`, `get_instruments`, `get_position_by_id`). |
| WebSocket msgpack: **bỏ viết tắt**, thêm channel **Market Index** | Cần cập nhật `vendor/dnse/trading_websocket/` khi có spec chi tiết; tạm thời có thể dùng `WS_ENCODING=json` để giảm rủi ro tương thích. |

---

## 3. Biến cấu hình liên quan

| Biến | Ý nghĩa |
|------|---------|
| `DNSE_BASE_URL` | OpenAPI DNSE (mặc định `https://openapi.dnse.com.vn`). **Không** dùng cho `services.entrade.com.vn`. |
| `DNSE_POSITIONS_FALLBACK_TO_DEALS` | Fallback sang endpoint cũ `/deals` khi `/positions` trả 404. |

---

## 4. Chuẩn bị live

1. Dùng **OpenAPI** cho đặt/hủy lệnh thường (đã có trong `BeeTradeClient`).
2. Xác nhận với DNSE: môi trường production đã bật **`/positions`**; nếu chỉ còn `/positions`, có thể tắt fallback sau khi ổn định.
3. Lệnh **điều kiện Entrade** — tách roadmap; không bắt buộc cho HMM live runner hiện tại.
