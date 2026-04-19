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
