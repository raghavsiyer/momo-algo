# Architecture

## Core Flow
```
main.py  →  scanners.py (background threads)
                ↓ ticker_q (queue.Queue)
         pipeline.py::run_pipeline()
                ↓
         Stage 1 (StockEvaluator._stage1) → Tradier quote gate
                ↓ pass
         Stage 2 (StockEvaluator._stage2) → parallel API fan-out
                ↓ pass
         Candle fetch + indicator build (technical_analysis/)
                ↓
         Resistance map (ResistanceMap)
                ↓
         Static confidence score (StaticBreakdown)
                ↓
         DynamicScorer started (Tradier WebSocket + REST polling)
                ↓
         Ticker displayed + cached (TickerCache)
```

## Module Map

| Module | Purpose |
|---|---|
| `main.py` | CLI REPL (prompt_toolkit), scanner orchestration, banner, commands |
| `pipeline.py` | `run_pipeline()` — full ticker evaluation, `EarlyStream`, `PipelineResult` |
| `scanners.py` | `_momo_scanner`, `_youtube_scanner`, `_scanner_consumer` threads |
| `config.py` | All constants, thresholds, API keys, feature flags, color helpers, NYSE calendar |
| `strategy/evaluator.py` | `StockEvaluator`, `Stage1Result`, `StaticBreakdown`, `WatchlistEntry`, 7 delta methods |
| `strategy/confidence_scorer.py` | `DynamicScorer` (stream + poll threads), `ScorerSnapshot`, edge-triggered flags |
| `technical_analysis/indicators.py` | Pure functions: EMA, MA, VWAP, Bollinger, MACD, ROC, RVOL, spread + composite builders |
| `technical_analysis/resistance.py` | `ResistanceMap` — static/structural/dynamic levels, pivot highs, topping tails |
| `market_data/tradier_rest.py` | `HistoricalClient` — quotes, candles via REST |
| `market_data/tradier_stream.py` | `RealTimeClient` — WebSocket streaming, live candle buffering |
| `market_data/momo_client.py` | `MomoClient` — momentum screener REST + Socket.IO, float scraping |
| `market_data/fmp_client.py` | `FMPClient` — float, splits, profile, IPO date |
| `market_data/alpha_vantage_client.py` | `AlphaVantageClient` — float, profile, splits, news (budget-tracked) |
| `market_data/finnhub_client.py` | `FinnhubClient` — company profile, recent news |
| `market_data/brave_client.py` | `BraveNewsClient` — web search news fallback |
| `market_data/finbert_client.py` | `FinBERTClient` — local FinBERT sentiment model |
| `market_data/youtube_client.py` | `YouTubeLiveChatClient` — live chat polling, ticker extraction |
| `cache.py` | `TickerCache` — JSON-backed with TTL |
| `display.py` | Terminal display helpers |
| `exec/` | Placeholder modules (backtest, tradier execution) — currently empty |

## Two-Stage Evaluation
- **Stage 1**: Single Tradier quote → hard filters (price range, % change, volume, RVOL 1-day, RVOL 5-min)
- **Stage 2**: Parallel API fan-out via `ThreadPoolExecutor(max_workers=5)`: float (AV → Momo fallback), Finnhub profile, AV splits, news cascade. Float gate triggers `EarlyStream`.

## Confidence Scoring
- **Static**: `CONFIDENCE_BASELINE` + 7 deltas (news, float, daily, IPO, sector, split, MA200), clamped `[FLOOR, CEILING]`
- **Dynamic**: `DynamicScorer` evaluates ROC, spread, chop, red volume, RVOL, resistance proximity. Edge-triggered via `_set_flag` (each condition fires once on transition).
- **Live score**: `WatchlistEntry.confidence_score` = `static_score + dynamic_delta`

## News Cascade Order
YouTube → Momo → Finnhub → AV → Brave (stops when fresh news found within current trading session)

## Key Patterns
- Cache-first for all inputs (manual and scanner)
- Confirmation window: momo scanner needs 2 hits within 10s for new tickers
- Volume-fail retry: removed from `queued` so scanners can re-queue later
- Hard failures stay in `queued` permanently but print short format on future hits
- `EarlyStream` starts T&S buffering as soon as float gate passes (before indicators built)
