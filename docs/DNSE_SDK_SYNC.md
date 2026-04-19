# Đồng bộ với DNSE OpenAPI SDK (GitHub)

Tham chiếu: [dnse-tech/openapi-sdk](https://github.com/dnse-tech/openapi-sdk) (Python trong thư mục `python/`).

## Đã căn chỉnh (BeeTrade)

| Hạng mục | SDK upstream (main) | BeeTrade `vendor/dnse` |
|----------|------------------------|-------------------------|
| **Danh sách vị thế** | Chỉ `get_positions` → `/accounts/{accountNo}/positions` | Có `get_positions` giống upstream; **thêm** `get_deals` → `/deals` để fallback 404 (xem `BeeTradeClient.get_deals`). |
| **Security definition** | `GET /price/{symbol}/secdef` | Đã **sửa** từ bản cũ `/price/secdef/{symbol}` → **giống upstream**. **Kiểm chứng live:** `GET /price/HPG/secdef` → 200 (JSON); `GET /price/secdef/HPG` → 404. Script: `python scripts/verify_secdef_paths.py`. |
| **Còn lại** | `get_ohlc`, `get_trades`, `get_latest_trade`, orders, `close_position`, … | Cùng path/query với bản upstream đã fetch. |
| **`common.py`** | `build_signature`, `get_date_header_name` | Khớp logic. |

## Khuyến nghị bảo trì

- Định kỳ so sánh `vendor/dnse/dnse/client.py` với [client.py trên GitHub](https://github.com/dnse-tech/openapi-sdk/blob/main/python/dnse/client.py).
- Giữ `get_deals` trong fork chỉ khi vẫn cần fallback; khi DNSE ngừng hỗ trợ `/deals`, có thể xóa và chỉ dùng `get_positions`.
