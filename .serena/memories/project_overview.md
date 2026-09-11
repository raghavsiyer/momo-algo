# momo_algo — Project Overview

## Purpose
Real-time momentum trading engine for low-float small-cap US equities. Scans for price/volume breakouts, scores tickers through a two-stage evaluator, applies technical indicators and resistance mapping, then ranks them by a composite confidence score that updates live via streaming data.

## Tech Stack
- **Python 3.12+** (venv auto-activated)
- **pandas** for DataFrames (candles, indicators)
- **prompt_toolkit** for CLI REPL
- **requests** for REST API calls
- **websockets** for Tradier streaming (T&S, quotes, L2)
- **torch + transformers** for FinBERT sentiment model
- **finnhub-python** for Finnhub API
- **python-dotenv** for environment variable management
- No web framework — terminal-only CLI application

## Data Providers
| Provider | API Key Env Var | Purpose |
|---|---|---|
| Tradier | `TRADIER_ACCESS_TOKEN` | Quotes, candles (10s/1m/15m/daily), WebSocket streaming |
| Alpha Vantage | `ALPHAVANTAGE_API_KEY` | Float, profile, splits, entity-sentiment news (budget-limited) |
| Finnhub | `FINNHUB_API_KEY` | Company profile (sector, IPO), recent news |
| FMP | `FMP_API_KEY` | Float, splits, reverse splits, company profile, IPO date |
| Brave Search | `BRAVE_SEARCH_API_KEY` | News fallback (web search) |
| FinBERT | (local model) | Headline sentiment scoring |
| Momo screener | (no key) | Momentum scanner, float via livequote scraping |
| YouTube | (no key) | Live chat polling for ticker mentions and news |

## Entry Points
- `python main.py` — starts the CLI REPL with scanners
- `Makefile` target: `make bunbot`

## Key Files
- `.env` — API keys (gitignored)
- `logs/state.json` — persisted scanner state (momo interval)
- `logs/help.txt` — CLI help text
