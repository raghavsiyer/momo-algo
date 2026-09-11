# Copilot Instructions — momo_algo

## **CRITICAL: Tool Preferences (Read This First)**

### Always Use Serena First (`mcp_serena_*` MCP Server)

**For ALL code analysis, exploration, and editing tasks, use Serena semantic tools:**

1. **Start with Serena memories** — `list_memories` and `read_memory` for project context before exploring code
2. **Use semantic symbol search** — `find_symbol`, `get_symbols_overview` to understand code, not `grep_search` or `semantic_search`
3. **Trace references** — `find_referencing_symbols` to understand dependencies, not manual grep
4. **Edit at the symbol level** — `replace_symbol_body`, `insert_after_symbol`, `insert_before_symbol` for code changes, not `replace_string_in_file`
5. **Write memories** — `write_memory` to record findings for future sessions

**Instead of:** "Search the codebase for all news functions"
**Use:** `mcp_serena_find_symbol` with `name_path_pattern` for news-related symbols

**Instead of:** "How does the scorer integrate with the evaluator?"
**Use:** `mcp_serena_find_referencing_symbols` on `DynamicScorer` to trace all call sites

### Always Use Pylance for Execution

- **NEVER use `run_in_terminal`** for Python or shell execution. Always use `mcp_pylance_mcp_s_pylanceRunCodeSnippet` instead.
- The **only exception** is interactive processes or long-running servers that require async mode.

### Self-Check

If you catch yourself reaching for `run_in_terminal`, `grep_search`, `semantic_search`, or `replace_string_in_file` when Serena or Pylance can do the job — **stop and use the correct tool.**

---

## Project Overview

Real-time momentum trading engine for low-float small-cap US equities. Scanners surface price/volume breakouts; a two-stage evaluator gates and enriches candidates; quality-function scoring produces an eval confidence; a live scorer streams T&S/quotes/L2 to compute an independent live confidence, detect setups, and drive a simulated execution engine.

**Run command:** `python main.py` (venv is auto-activated)
**Test command:** `python -m pytest` (from project root)

---

## Architecture

### Core Flow

```
main.py  →  scanners.py (background threads)
                ↓ ticker_q  ("momo" | "youtube" | "seed" | manual input)
         pipeline.py::run_pipeline()
                ↓
         Stage 1 (StockEvaluator.stage1) → quick quote gate via active provider
                ↓ pass
         Stage 2 (StockEvaluator.stage2) → parallel APIs: float, profile, splits, news
                ↓ pass (float gate)
         Candle fetch + indicator build (technical_analysis/)
                ↓
         Resistance map (ResistanceMap)
                ↓
         evaluator.evaluate() → CompositeEvaluation (stock_eval → eval confidence)
                ↓
         DynamicScorer started (T&S + quotes + L2 stream; 1-min REST candle polling)
                ↓ live inputs → technical_analysis group → live confidence
                ↓ setup detection → SimulationEngine entries/exits (paper fills)
         Ticker displayed + cached (TickerCache)
```

### Market Data Providers

- **All quote/candle/stream access goes through `market_data/provider.py`** — `get_historical_client()`, `get_shared_stream()`, `stream_enabled()`. Never instantiate provider clients directly.
- **IBKR** (`IBKR_ENABLED=true`) — `IBKRHistoricalClient` + `IBKRSharedStream`, supports L2 depth. **This is the provider actually in use.**
- **Tradier** (`TRADIER_ENABLED`, config default) — `HistoricalClient` + shared WebSocket stream. Retained as fallback.

### Two-Stage Evaluation

- **Stage 1** — Hard filters on a single quote: price range, % change, volume, 1-day RVOL, 5-min RVOL. Fast and cheap. Failures are displayed inline and skipped.
- **Stage 2** — Parallel API fan-out: float data (FMP/AV/Momo), Finnhub profile (sector, IPO date), AV splits, news cascade (YouTube → Momo → Finnhub → AV → Brave). Float gate must pass before candles stream. Builds `StaticInputs` for scoring.

### Confidence Scoring — quality functions, NOT additive deltas

- `config.SCORING_GROUPS` declares variable groups (name, weight, category, cutoffs, quality functions). `strategy/scoring.py::build_groups()` materializes them as `VariableGroup`s of `ScoringVariable`s. Each variable maps a raw measurement to a quality in [0, 1]; a group score is the weighted average of its active variables.
- `WatchlistEntry.evaluation` is a `CompositeEvaluation` with two scored pillars:
  - `stock_eval` (static inputs) → `eval_confidence`
  - `technical_analysis` (live inputs, fed by `DynamicScorer`) → `live_confidence`
  - **The two are never blended into one composite score — deliberate design.**
- Criteria groups (e.g. ATTENTION) alongside the scored groups are boolean gates, not scores.
- `WatchlistEntry.watchlist_sort_score` = live confidence once live data exists, else eval confidence.

### Live Scoring & Simulated Execution

- `DynamicScorer` consumes `LiveTimesale` / `LiveQuote` / `LiveBook` events plus REST-polled 1-min candles; builds 10s/1m/2m/5m stream candles; pushes raw measurements into the `technical_analysis` group; detects setups via `technical_analysis/patterns.py` (`scan_all_patterns`).
- `exec/order_flow.py::OrderFlowTracker` — book lean, OFI (L1 + L2 depth), trade-flow imbalance via aggressor classification.
- `exec/engine.py::SimulationEngine` (`SIM_ENABLED`) — evaluates `exec/signals.py` entry/exit gates on every tick, simulates fills at live bid/ask; trades logged by `exec/sim_logger.py`, setups by `exec/setup_logger.py`, telemetry by `exec/scoring_log.py` (`SCORING_LOG_ENABLED`, CSVs written to logs/).
- `exec/backtest.py` — replays chart_cache T&S/candles through the same signal stack; results under `SIM_BACKTEST_LOG_DIR`.

### Background Scanners

- **Momo scanner** (`_momo_scanner`) — REST polls a momentum screener; confirmation-window logic requires 2 hits within 10s before queuing unevaluated tickers. First poll seeds silently.
- **YouTube scanner** (`_youtube_scanner`) — Polls a YouTube live chat for ticker mentions and news headlines from a designated author. Applies FinBERT sentiment scoring and noise filtering.
- **Consumer thread** (`_scanner_consumer`) — Dequeues tickers and runs `run_pipeline`. Handles cache-hit paths, YouTube news injection, and stale-news refresh. "seed" items (queued at startup for cached symbols) restore cached entries — refreshing Stage 1 first when the cache is from a prior ET day — and restart scorers.

All scanner threads communicate via `queue.Queue` + `threading.Lock`-protected shared sets.

---

## Module Map

| Module | Purpose |
|---|---|
| `main.py` | CLI REPL (prompt_toolkit), scanner orchestration, bottom-toolbar rendering, commands: list, cache, rank, weights, levels, pos, setup, momo/yt controls, per-ticker refresh/start/stop/float/news/split |
| `pipeline.py` | `run_pipeline()` — full ticker evaluation from Stage 1 through display + scorer start; cached-entry Stage-1 refresh helpers |
| `scanners.py` | Background scanner threads (momo, youtube, consumer) |
| `config.py` | All constants, thresholds, API keys (from env), feature flags, `SCORING_GROUPS`, `MarketClock`, color helpers, NYSE holiday calendar |
| `strategy/evaluator.py` | `StockEvaluator`, `Stage1Result`, `WatchlistEntry` (dataclass), `CompositeEvaluation` wiring |
| `strategy/scoring.py` | Quality-function engine: `ScoringVariable`, `VariableGroup`, `build_groups()`, `StaticInputs` |
| `strategy/confidence_scorer.py` | `DynamicScorer` — live scoring, stream candle building, setup detection, sim-engine hookup |
| `technical_analysis/indicators.py` | Pure functions: EMA, MA, VWAP, Bollinger (population std), MACD, ROC, RVOL. Composite builders (`build_1min`, `build_5min`, `build_daily`); in-place `_ip` variants for the scorer hot path |
| `technical_analysis/patterns.py` | `PatternSignal` detectors (pullback new-high, topping/bottoming tails, tail clusters, multi-top, wedges, H&S...), `scan_all_patterns`, move-start detection. Canonical home of single-candle tail checks |
| `technical_analysis/resistance.py` | `ResistanceMap` — STATIC (dollar levels), STRUCTURE (pivots w/ prominence + clustering, 200 EMA/MA), DYNAMIC (topping tails, intraday highs) |
| `market_data/provider.py` | Provider abstraction — **always go through this** |
| `market_data/ibkr_rest.py` / `ibkr_stream.py` | `IBKRHistoricalClient` / `IBKRSharedStream` (T&S, quotes, L2 depth) |
| `market_data/tradier_rest.py` / `tradier_stream.py` | Tradier REST + WebSocket (fallback provider) |
| `market_data/momo_client.py` | `MomoClient` — momentum screener REST + Socket.IO, float via livequote scraping |
| `market_data/fmp_client.py` / `alpha_vantage_client.py` / `finnhub_client.py` / `brave_client.py` | Float, splits, profile, IPO date, news providers (AV has per-minute budget tracking) |
| `market_data/finbert_client.py` / `youtube_client.py` | Local FinBERT sentiment; YouTube live-chat polling + ticker/news extraction |
| `exec/engine.py` | `SimulationEngine` — simulated entries/exits on live ticks (singleton per scorer) |
| `exec/signals.py` | Pure entry/exit gate evaluation — `EntrySignal`, `ExitSignal`, `TapeGates` |
| `exec/order_flow.py` | `OrderFlowTracker` — OFI, book lean, aggressor-classified trade flow |
| `exec/sim_logger.py` / `setup_logger.py` / `scoring_log.py` | CSV/JSON logs: sim trades, setup signals, scoring + tape telemetry |
| `exec/backtest.py` | Replay backtester over chart_cache data |
| `cache.py` | `TickerCache` — JSON-backed TTL cache of `(WatchlistEntry, ResistanceMap)` per ticker (evict-on-load only) |
| `chart_cache.py` | Per-session candle (10s/1m/2m/5m) + timesales cache for scorer seeding and backtest replay |
| `display.py` | `_print_stock_summary`, `print_watchlist`, `watch_status_line`, scanner-hit printing |

---

## Key Data Structures

- **`Stage1Result`** — dataclass: symbol, price, change_pct, volume, avg_volume, rvol_1day, rvol_5min, passed, fail_reason
- **`WatchlistEntry`** — dataclass: all Stage 1+2 fields, `evaluation` (`CompositeEvaluation`), `eval_confidence` / `live_confidence` / `watchlist_sort_score` properties, `setup_*` fields, `execution_snapshot`, news articles
- **`CompositeEvaluation`** — scored `groups` (`VariableGroup` of `ScoringVariable`) + gate-only `criteria_groups`
- **`DynamicScorer`** — poll thread + stream event handlers; owns `OrderFlowTracker` and optional `SimulationEngine`
- **`ResistanceMap`** — levels with type, strength, active/inactive; pivot detection with prominence/HOD/cluster filters
- **`PatternSignal`** — detected pattern with bias, strength, price level, timeframe
- **`OHLCVCandle` / `StockQuote`** — REST response models; **`LiveTimesale` / `LiveQuote` / `LiveBook`** — stream event models

---

## Coding Conventions

- **Python 3.12+**, type hints on all function signatures
- **dataclasses** for DTOs — no Pydantic
- Pure functions in `technical_analysis/indicators.py` — each returns a new DataFrame, originals are not mutated. In-place `_ip` variants exist for the scorer's hot path
- **Feature flags** in `config.py` (`IBKR_ENABLED`, `TRADIER_ENABLED`, `SIM_ENABLED`, `FMP_ENABLED`, etc.) — code must degrade gracefully when a provider is disabled
- **Logging** via stdlib `logging` — one `logger` per module, messages prefixed with `[symbol]`
- **Threading** — daemon threads for scanners/scorers. Shared state protected by `threading.Lock`. `threading.Event` for stop/pause signals
- **Watchlist concurrency** — never mutate `evaluator.watchlist` in place; build a new sorted list, then rebind (`evaluator.watchlist = new_wl`). The prompt_toolkit toolbar iterates it concurrently from the main thread. Iterate the `scorers` dict via `list(scorers.items())` snapshots
- **No web framework** — terminal-only CLI using `prompt_toolkit`
- Constants are UPPER_SNAKE_CASE in `config.py`. Color helpers (`_g`, `_y`, `_r`, `_c`, `_b`, `_dim`) for ANSI formatting
- Private methods prefixed with `_`. Public API is minimal (e.g. `stage1`, `stage2`, `evaluate`, `scan`, `start`, `stop`, `snapshot`)

## Testing Conventions

- **pytest** with `pythonpath = .` and `testpaths = tests`; test modules live in `tests/test_files/test_*.py`
- `conftest.py` forces `IBKR_ENABLED=false` and `TRADIER_STREAM_ENABLED=false` **before** config imports — tests never touch real streams/brokers
- All external APIs are mocked — no network calls. Use `unittest.mock.patch` / `MagicMock`
- Test helpers (`_make_entry`, `_make_scorer`, ...) build minimal fixtures; indicator/resistance tests use small synthetic DataFrames
- Tests are fast, isolated, and deterministic — no shared state between test methods

## API Key Management

- All API keys loaded from environment variables via `python-dotenv`
- Keys are validated with a regex check (`_key_ok`) before marking an API as active
- **Never hardcode keys.** The `.env` file is gitignored
- AV client has a budget system — `self._av.budget` tracks remaining calls per minute

## Important Patterns

- **News cascade**: YouTube → Momo → Finnhub → AV (→ Brave, disabled by default). Each source tried in order; stops when fresh news is found (published within current trading session)
- **Noise filtering**: FinBERT scores filter out neutral/irrelevant headlines below `NEWS_NOISE_THRESH`
- **Cache-first**: Manual input, scanner hits, and startup seeds all check `TickerCache` before running the pipeline; cached entries from a prior ET day get a Stage-1 quote refresh before a scorer starts (the scorer accumulates prints onto `entry.volume`)
- **Confirmation window**: Momo scanner requires 2 hits within 10 seconds for unevaluated tickers before queuing
- **Volume-fail retry**: If a ticker fails Stage 1 on volume, it's removed from `queued` so scanners can re-queue it later. Hard failures stay in `queued` permanently
- **Replay parity**: backtests replay chart_cache data through the same `OrderFlowTracker`/`signals` stack as live — keep live and replay paths behaviorally identical when editing either
