# Câu hỏi gửi DNSE Support (ngắn gọn — PM / kỹ thuật)

*Dùng khi cần xác nhận với phía DNSE sau khi đối chiếu [openapi-sdk](https://github.com/dnse-tech/openapi-sdk).*

1. **Endpoint vị thế:** Production đã chuyển hẳn sang `GET /accounts/{accountNo}/positions` chưa? Endpoint cũ `/deals` còn được hỗ trợ đến khi nào?

2. **Secdef:** Xác nhận path chuẩn là `GET /price/{symbol}/secdef` (đúng với SDK Python trên GitHub) — có môi trường nào vẫn dùng path cũ không?

3. **WebSocket msgpack V2:** Có tài liệu chi tiết (field name đầy đủ, channel Market Index) để cập nhật client không?

4. **Entrade Smart Order** (`services.entrade.com.vn`) và **OpenAPI** (`openapi.dnse.com.vn`): Cùng `trading-token` hay quy trình auth khác nhau? Có roadmap gộp hay tách rõ use-case?

5. **Rate limit / quota:** Giới hạn request/phút cho REST khi chạy bot lặp (ví dụ mỗi 15 phút + polling lệnh)?

6. **WebSocket OHLC (Market Data):** Sau khi subscribe thành công (`action=subscribed`, channel dạng `ohlc.1m.json` hoặc `ohlc.1m.{BOARD}.json`), có trường hợp **chỉ nhận `T=t` (trade)** mà **không có bản tin nến** (`T=b` hoặc payload tương đương) trong phiên giao dịch không? Đây là hành vi đúng hay cần thêm tham số (ví dụ `marketId`, `boardId` trong body subscribe, tên kênh khác cho cổ phiếu HOSE vs phái sinh)?

7. **Định dạng bản tin nến realtime:** Vui lòng cung cấp **một ví dụ JSON/msgpack đầy đủ** (một nến 1 phút) cho mã cổ phiếu HOSE và một mã phái sinh — gồm toàn bộ key (kể cả `T`/`channel`/`data` nếu có).

8. **openapi-sdk GitHub:** Repo [dnse-tech/openapi-sdk](https://github.com/dnse-tech/openapi-sdk) hiện chỉ thấy REST (Python/JS); **WebSocket streaming** có SDK / sample chính thức nào khác (repo hoặc nhánh) để đối chiếu subscribe + decode không?
