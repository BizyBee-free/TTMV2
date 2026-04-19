# Ghi chú vận hành & cấu hình (BeeTrade — mặc định HMM 15m)

Tài liệu mô tả **ghi chú vận hành**, **biến môi trường**, và **ý nghĩa từng field** trong [`src/config.py`](../src/config.py). Override qua **`.env`** ở **thư mục gốc project**. Mẫu đầy đủ: [`config/.env.example`](../config/.env.example).

**Mặc định dự án:** `STRATEGY_ALGO=HMM` (Live HMM nến 15m + kill switch / DNSE robustness). Chiến lược MCMC chỉ khi đặt `STRATEGY_ALGO=MCMC`.

**Tài liệu API Entrade (Word) + OpenAPI V2:** đối chiếu và tích hợp positions/Entrade — xem [`DNSE_AND_ENTRADE_APIS.md`](DNSE_AND_ENTRADE_APIS.md).

---

## 0. Chọn thuật toán: `STRATEGY_ALGO`

| Giá trị | Ý nghĩa |
|---------|--------|
| **`HMM`** (mặc định) | **Chiến lược Live HMM** (regime HMM, `HMM_LIVE_*`, `HMM_SYMBOL`, `scripts/hmm_live.py`) + kill switch / DNSE (`MIN_BALANCE`, timeout, Telegram, …). |
| **`MCMC`** | **Chiến lược MCMC derivatives** (Markov + MCMC engine, tham số `MCMC_*`). Không dùng pipeline HMM làm stack chính. |

`python scripts/hmm_live.py` cảnh báo nếu bạn đặt **`STRATEGY_ALGO=MCMC`** (dùng `--ignore-strategy-algo` để tắt).

---

## 1. Chuẩn bị file `.env` (trước thứ 2)

1. Copy `config/.env.example` → `.env` tại root project.
2. **Bắt buộc:** `DNSE_API_KEY`, `DNSE_API_SECRET`, `DNSE_ACCOUNT_NO`.
3. **HMM 15m:** giữ `STRATEGY_ALGO=HMM`, chỉnh `HMM_SYMBOL` nếu đổi hợp đồng/proxy.
4. **Live thật** (`PAPER_MODE=false`): điền **`DNSE_TRADING_TOKEN`** (hoặc **`DNSE_TRADING_TOKEN_FILE`** trỏ đến file một dòng token), **`MIN_BALANCE`**, Telegram nếu cần.
5. **Paper / dry-run:** `PAPER_MODE=true`, có thể để trống token.

---

## 2. Checklist thứ 2 — bật app là chạy

| Trước phiên | Việc cần làm |
|-------------|----------------|
| `.env` | Đã copy từ `.env.example`, đủ `DNSE_*`. |
| Chiến lược | `STRATEGY_ALGO=HMM` (mặc định code + mẫu). |
| Symbol | `HMM_SYMBOL=VN30F1M` (hoặc mã bạn trade; đồng bộ với hợp đồng đang có). |
| Paper vs live | `PAPER_MODE=true` cho thử; **live** → `false` + token + `MIN_BALANCE`. |
| Token DNSE | Đặt `DNSE_TRADING_TOKEN=` hoặc file `DNSE_TRADING_TOKEN_FILE` (token sau OTP, không phải mã OTP). Token hết hạn ~1h — cần làm mới khi đặt lệnh. |
| Telegram | Nếu dùng: `TELEGRAM_ENABLED=true` + `BOT_TOKEN` + `CHAT_ID`. |
| Cron / Task Scheduler | Gọi lặp `python scripts/hmm_live.py --submit` mỗi 15 phút sau khi nến đóng (hoặc theo lịch bạn chọn). |

---

## 3. Ghi chú vận hành (quan trọng)

### 3.1 Token giao dịch DNSE (OTP → token)

- Đặt lệnh **live** cần **trading token** (chuỗi trả về sau khi đổi OTP trên API).
- Có thể đặt trong **`.env`**: `DNSE_TRADING_TOKEN=...` hoặc **`DNSE_TRADING_TOKEN_FILE`** = đường dẫn file **một dòng** chứa token (không commit file token; đã có pattern trong `.gitignore`).
- `BeeTradeClient` nạp token từ env/file khi khởi tạo — **không** cần nhập console nếu đã cấu hình.
- Nếu token trống và hết hạn, `ensure_token()` vẫn có thể hỏi OTP trên console (tùy luồng gọi).

### 3.2 Paper vs live

| `PAPER_MODE` | Hành vi |
|--------------|--------|
| `true` | Không gửi lệnh thật; lệnh giả qua `OrderManager` (paper). |
| `false` | Gọi API đặt lệnh thật (cần token hợp lệ, margin, symbol đúng). |

### 3.3 `MIN_BALANCE`

- Áp dụng khi **live** + `MIN_BALANCE` được set số.
- Số dư từ API &lt; ngưỡng → halt.

### 3.4 Timeout & halt API

- `DNSE_HTTP_TIMEOUT_SEC`, `DNSE_API_FAIL_HALT_THRESHOLD` — xem bảng biến.

### 3.5 `scripts/hmm_live.py`

- Fetch OHLC **15 phút**, tín hiệu cây **cuối**; `--symbol` mặc định = **`HMM_SYMBOL`** trong `.env`.
- **`HMM_USE_BASIS=true`** (mặc định): fetch thêm OHLC **`HMM_INDEX_SYMBOL`** (mặc định `VN30`), ghép theo `unix_ts` cùng resolution; HMM dùng 4 đặc trưng (thêm basis = `close(future) − close(index)` tại từng bar). Nếu API proxy phái sinh sang cùng chỉ số, basis ≈ 0 — nên kiểm tra log/response trong giờ giao dịch.
- **`HMM_USE_OPEN_INTEREST`** (Phase 2, mặc định `false`): thêm đặc trưng biến động OI (đổi OI giữa hai nến, z-score). OI lấy từ **OpenAPI DNSE** `GET /price/{symbol}/secdef`: trường JSON **`openInterestQuantity`** (camelCase) hoặc **`open_interest_quantity`** (snake_case) — đúng với model SDK **`SecurityDefinition.openInterestQuantity`** trong `vendor/dnse/trading_websocket/models.py` (open interest hợp đồng, không phải sổ lệnh). Lịch sử OI theo thời gian không có trong OHLC: hệ thống **ghi `data/cache/oi/{symbol}.jsonl`** (mỗi lần live đọc secdef + timestamp nến) và **forward-fill** OI lên từng bar. Backtest chỉ OHLC: nếu bật `HMM_USE_OPEN_INTEREST` trong `.env`, `scripts/backtest.py` sẽ **tắt OI** cho run đó (cảnh báo log) cho tới khi có nguồn chuỗi OI theo bar.
- `--submit` + state `data/hmm_live_state.json` chống trùng nến.

### 3.6 Telegram, đối soát, hạn chế OrderManager / OHLC proxy

- Giữ nguyên như các phiên bản trước (notify, reconcile, paper fill vs live polling, proxy VN30).

---

## 4. Bảng biến cấu hình (`Settings`)

### 4.1 Bắt buộc (không có default an toàn)

| Biến | Ý nghĩa |
|------|---------|
| `DNSE_API_KEY` | OpenAPI Key. |
| `DNSE_API_SECRET` | OpenAPI Secret. |
| `DNSE_ACCOUNT_NO` | Tiểu khoản giao dịch. |

### 4.2 DNSE bổ sung

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `DNSE_TRADING_TOKEN` | `""` | Token đặt lệnh (sau OTP). Trống = không preload. |
| `DNSE_TRADING_TOKEN_FILE` | `""` | File một dòng chứa token (nếu không dùng biến trên). |
| `DNSE_POSITIONS_FALLBACK_TO_DEALS` | `true` | OpenAPI V2: ưu tiên `GET .../positions`; nếu **404**, gọi lại legacy `.../deals`. |

### 4.3 API & WebSocket

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `DNSE_BASE_URL` | `https://openapi.dnse.com.vn` | REST. |
| `DNSE_WS_URL` | `wss://ws-openapi.dnse.com.vn/v1/stream` | WebSocket (bắt buộc `/v1/stream`; encoding `?encoding=` thêm tự động). |
| `WS_ENCODING` | `msgpack` | `msgpack` hoặc `json`. |

### 4.4 Chế độ, thuật toán, log

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `PAPER_MODE` | `true` | Paper vs live. |
| `STRATEGY_ALGO` | **`HMM`** | `HMM` = stack HMM live; `MCMC` = MCMC derivatives. |
| `LOG_LEVEL` | `INFO` | Mức log. |
| `LOG_DIR` | `logs` | Thư mục log. |
| `LOG_MAX_BYTES` | 50MB | Rotate. |
| `LOG_BACKUP_COUNT` | `7` | Số file giữ. |

### 4.5 Risk chung

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `MAX_DAILY_LOSS_PCT` | `2.0` | Lỗ ngày tối đa (% NAV). |
| `MAX_ORDERS_PER_MINUTE` | `10` | Rate limit đặt lệnh. |
| `STOPLOSS_DEFAULT_PCT` | `3.0` | Stoploss mặc định. |

### 4.6 MCMC (khi `STRATEGY_ALGO=MCMC`)

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `MCMC_DERIVATIVE_SYMBOL` | `VN30F2506` | Symbol MCMC. |
| … | … | Các `MCMC_*` khác như trong `config.py`. |

### 4.7 HMM live / kill switch

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `HMM_SYMBOL` | **`VN30F1M`** | Symbol OHLC 15m cho `hmm_live` (mặc định `--symbol`). |
| `HMM_USE_BASIS` | `true` | Bật đặc trưng basis (cần OHLC index + ghép timestamp). |
| `HMM_INDEX_SYMBOL` | `VN30` | Chỉ số cho basis (cùng resolution với `HMM_SYMBOL`). |
| `HMM_USE_OPEN_INTEREST` | `false` | Phase 2: thêm đặc trưng OI từ secdef (`openInterestQuantity`) + cache `data/cache/oi/`. |
| `MIN_BALANCE` | `None` | Sàn dư tối thiểu (VND). |
| `DNSE_HTTP_TIMEOUT_SEC` | `30` | Timeout request (giây). |
| `DNSE_API_FAIL_HALT_THRESHOLD` | `3` | Lỗi liên tiếp → halt. |
| `RECONCILE_BALANCE_EPS_VND` | `50000` | Ngưỡng đối soát. |
| `DERIVATIVE_FEE_RATE` | `0` | Phí kỳ vọng / notional. |
| `HMM_LIVE_TRAIN_WINDOW_BARS` | `1000` | Cửa sổ trượt (bar 15m); tune với `config/hmm_optimization.yaml` + `scripts/hmm_optimize_loop.py`. |
| `HMM_LIVE_REFIT_EVERY` | `5` | Refit mỗi N bar (live). |
| `HMM_LIVE_K_STATES` | `3` | Số trạng ẩn HMM (k). |
| `HMM_LIVE_CONFIDENCE` | `0.65` | Ngưỡng confidence HMM. |
| `HMM_LIVE_WARMUP_BARS` | `30` | Warm-up tối thiểu. |

### 4.8 Telegram

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `TELEGRAM_ENABLED` | `false` | Bật notify. |
| `TELEGRAM_BOT_TOKEN` | `""` | Bot token. |
| `TELEGRAM_CHAT_ID` | `""` | Chat nhận tin. |
| `TELEGRAM_ALLOWED_CHAT_IDS` | `""` | Chat được `/pause` (trống = dùng `CHAT_ID`). |

### 4.9 Runtime (không trong `.env`)

| Biến | Ý nghĩa |
|------|---------|
| `_trading_token` | Cache trong bộ nhớ (Settings). |

### 4.10 Checklist phiên giao dịch (Basis / OTP / live)

| Việc | Ghi chú |
|------|---------|
| OTP → **trading token** | Token ~1h; làm mới trước khi `--submit` live. |
| Hai chuỗi OHLC | Future + index (15m) có bar khớp `unix_ts`; nếu `basis_align_failed` → kiểm tra API/symbol. |
| `PAPER_MODE` | `true` đến khi tin tưởng tín hiệu; `false` chỉ khi sẵn sàng lệnh thật. |
| Số dư / margin | Đã nạp tiền vẫn cần `MIN_BALANCE` phù hợp và quyền phái sinh trên tài khoản. |

---

## 5. Lệnh tham khảo

```bash
# Mặc định dùng HMM_SYMBOL từ .env
python scripts/hmm_live.py

python scripts/hmm_live.py --submit
python scripts/hmm_live.py --telegram-control
```

---

*Đồng bộ với `src/config.py`. Nếu đổi field trong code, cập nhật bảng này.*
