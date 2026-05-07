# BeeTrade - Algorithmic Trading Platform

Algorithmic trading tool connecting to DNSE OpenAPI for Vietnamese stock market.

## Features

- **Pair Trading**: Atomic execution between Covered Warrants and underlying stocks
- **T0 Derivatives**: Intraday VN30F1M trading via MCMC signals (planned)
- **CW Analytics**: ATM/ITM/OTM classification and profitability tracking (planned)
- **Risk Management**: Pre-trade checks, auto-stoploss, daily loss limits
- **Paper Trading**: Simulated order matching against real market data

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure credentials
cp config/.env.example .env
# Edit .env with your DNSE API Key, Secret, and Account Number

# 3. Verify connection
python scripts/check_connection.py
```

## Project Structure

```
BeeTrade/
├── vendor/dnse/          # DNSE Official SDK (vendored)
├── src/                  # BeeTrade application code
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── config/               # Configuration templates
└── logs/                 # Runtime logs (gitignored)
```

## API Reference

Built on [DNSE OpenAPI](https://developers.dnse.com.vn/) with the official
[Python SDK](https://github.com/dnse-tech/openapi-sdk).

## Authentication

DNSE uses HMAC-SHA256 per-request signing (handled by SDK). For trading
operations, an email OTP-based Trading Token is required.

## Operations setup (production control layer)

The `src/ops/` package adds **Telegram alerts**, **JSONL audit logs**, **persistent control state**
(`runtime/control_state.json`), **order / health monitors**, and optional **OTP via Telegram** — without
changing TTM signal or scoring code. Integration is at the **order submission gate** in
`src/live/ttm_live_runner.py` only.

1. Copy variables from `config/.env.example` (Operations section) into your project-root `.env`.
2. Set `OPS_ENABLED=true` for TTM live. Use `TELEGRAM_ENABLED=true` plus `TELEGRAM_BOT_TOKEN` and either
   `TELEGRAM_CHAT_ID` or `TELEGRAM_ALLOWED_CHAT_ID` for outbound alerts and inbound control.
3. **Control state defaults**: `trade_enabled` starts **false** unless `CONTROL_TRADE_ENABLED_DEFAULT=true`
   or you `/resume` via Telegram (when `CONTROL_ENABLED=true`). `/pause` blocks **new** positions; exits
   still allowed if `trade_enabled` remains true. `/kill` + `CONFIRM` stops all orders.
4. **OTP**: With `OTP_TELEGRAM_ENABLED=true`, a missing trading token can trigger a Telegram prompt; reply
   with `OTP123456` (only from the allowed chat, while a request is pending). OTP is never written to disk
   or logged in full.
5. **Logs**: Daily JSONL under `logs/` — `control_events_YYYYMMDD.jsonl`, `order_events_*.jsonl`,
   `risk_events_*.jsonl`, `health_events_*.jsonl`.
6. **Session script**: `scripts/hmm_live_session.py` registers `stop_ops` on exit when `OPS_ENABLED` so the
   bot sends a **Bot stopped** Telegram if configured.

Failures in the ops/Telegram layer are caught and logged so the trading loop does not crash; if control
state is uncertain, use env defaults and gates (`can_send_order` / `can_open_new_position`) to stay safe.
