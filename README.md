# momo-algo

Real-time momentum trading engine for low-float small-cap US equities.

The system finds stocks that are breaking out (big % move on unusual volume), figures out if they're worth watching (float, news, chart quality), and then tracks them live — continuously scoring how healthy the move looks and whether the setup is actually safe to enter. Everything runs in a terminal. There is no UI, no web server, and no live order routing yet.

---

## Quick Start

```bash
# 1. Clone & create virtualenv
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys (copy .env.example or create .env)
cp .env.example .env   # then fill in your keys

# 4. Run
python main.py
# — or —
make bunbot
```

**Requirements:** Python 3.12+, API keys in `.env` for Tradier, Alpha Vantage, Finnhub, and FMP. YouTube API key is optional (live-chat scanner); Brave Search is off by default (`BRAVE_ENABLED=true` re-enables the news fallback and needs a key). IBKR TWS/Gateway is required for L2 depth, the order-flow scoring variables, and the IBKR top-gainers scanner.

---

## How It Works — The Full Flow

### Discovery: How tickers get found

Three background scanner threads run continuously, looking for candidates:

1. **Momo scanner** — polls a momentum screener (Black Box Stocks) at configurable intervals. To avoid chasing noise, a ticker must appear in **two consecutive polls within 10 seconds** before it gets queued for evaluation. The very first poll after startup seeds the "already seen" set silently so we don't flood the pipeline with stale hits.

2. **YouTube scanner** — polls a designated YouTube live chat stream, looking for ticker mentions and news headlines from a specific author. Headlines get scored through a local FinBERT model for sentiment, and anything below a noise threshold is discarded.

3. **IBKR scanner** — runs a one-shot `TOP_PERC_GAIN` market scan each interval, with price/volume/%-change floors pushed server-side. Because IBKR returns the same ranked list every poll, only a symbol **entering** the list is an event; a name that drops off and stays off for 15 minutes counts as new again if it returns. No two-hit confirmation is needed here — a seat in the ranked top 25 is itself the confirmation. The scan floors sit at **half** the pipeline's criteria levels (5% change, 50k volume), so a name is discovered while it's still building and re-queued once it clears the real gate.

All three scanners push tickers into a shared queue. A **consumer thread** picks them off, checks the cache first (so we don't re-evaluate something we already have), and runs the full pipeline on cache misses. A `queued` set guarded by a lock keeps two scanners surfacing the same name from queueing it twice. If the YouTube scanner found a news headline along with the ticker, it gets injected into the evaluation.

### Evaluation (once a ticker is found):

The evaluation pipeline runs in stages. The idea is to fail fast and cheap — don't burn API calls on a ticker that's clearly outside our criteria.

**Quote Prefilter — cheap sanity gate** (1 API call)

A single Tradier REST quote is used to avoid spending enrichment calls on symbols that are clearly outside the momentum universe. The hard prefilter is intentionally small:
- Price between $0.10 and the configured max
- % change from previous close >= 10%

The quote prefilter still records volume, average volume, RVOL, and daily high, but those values are not standalone hard rejects here.

**Stage 1 — Stock thesis / static evaluation score**

After the quote prefilter, the system builds the static thesis: how well we think the stock can move today before live execution is considered. Core thesis ingredients include:
- % change: abnormal momentum is already underway
- Float: lower supply is easier to move
- News quality: real, fresh catalyst beats stale/noisy headlines
- RVOL / volume: unusual participation, with 5x RVOL as an important reference
- Price range: small-cap momentum sweet spot
- Daily chart setup: room and structure for continuation

Low volume or RVOL should lower thesis quality or block readiness criteria, not be confused with a quote-prefilter rejection. Live attention/execution gates decide when the thesis becomes actionable.

**Deep enrichment fan-out** (4-6 API calls, parallel)

If the quote prefilter passes, we fan out in parallel to gather everything needed for the stock thesis/static score:
- **Float** — tries via Alpha Vantage first, falls back to Momo livequote scraping. Must be ≤ 50M shares (the float gate). This is a hard stop — if float is too high, the stock doesn't fit the low-float momentum thesis.
- **Company profile** — sector, industry, IPO date (via Finnhub/FMP)
- **Splits** — checks for recent reverse splits (via AV), which changes how we interpret the move
- **News** — this is a cascade: YouTube headlines → Momo → Finnhub → Alpha Vantage (→ Brave web search, off by default). Each source is tried in order and we stop as soon as we find news published during the current trading session. All headlines are FinBERT-scored for sentiment.

Once the float gate passes, candle data starts streaming immediately (even before the rest of Stage 2 finishes) so we're not waiting.

**After Stage 2 — Build the evaluation**

With all the data in hand, the pipeline:
1. Fetches historical candles (10-second, 1-minute, 15-minute, daily)
2. Builds technical indicators on each timeframe (VWAP, MACD, EMA, Bollinger, etc.)
3. Constructs a **resistance map** — price levels the stock is likely to stall at
4. Computes the initial **static evaluation** — a composite confidence score based on 6 quality groups
5. Sets up **criteria groups** — boolean pass/fail gates for execution readiness
6. Displays the result, caches it, and starts the **DynamicScorer** to track it live

---

## Scoring: How confidence is computed

There are two parallel systems working on every entry. One produces a **numeric score** (how good is this setup?), the other produces **boolean gates** (is it actually safe to enter right now?). Both must agree before entry is allowed.

### The quality-function composite (0–100)

Every input variable has two properties:
1. **Weight** (importance) — how much this variable matters relative to others. Pre-determined.
2. **Quality** (activation) — a rating between 0 and 1 for the variable's current value. Determined by a quality function.

Since we're heuristically determining impact rather than observing clean outcomes, we assess quality as a function of the raw value rather than measuring it directly.

**Quality functions** map a raw value to a [0, 1] quality rating. The function type + thresholds are set per variable:

- **LINEAR_MIN** — lower is better (1 threshold). Example: share float — the lower, the better. Max threshold 50M; quality declines linearly as float increases toward 50M.
- **LINEAR_MAX** — higher is better (1 threshold). Example: RVOL — more relative volume = better.
- **LINEAR_RANGE** — a range is good (2 thresholds). Example: trade imbalance in [-1, +1] mapped linearly.
- **BELL** — middle is best, extremes are bad (2 thresholds). Example: stock price — we want $0.50–$20, with a sweet spot in the middle.
- **BELL_SKEW** — bell curve with a shifted peak. Example: price is skewed left (cheaper is slightly better than expensive within the range).
- **SATURATING** — quality ramps up quickly then plateaus. Example: VWAP slope — once it's clearly positive we don't need it to be steeper.

Note: thresholds are cutoffs for the function domain. They don't squeeze the curve or change its range — they define where quality goes to 0.

**Variables are organized into 6 weighted groups.** Each group has its own importance weight, and within each group, each variable has a weight too. The composite score is a weighted average of group scores × 100.

| Group | Weight | What it captures |
|---|---|---|
| **CATALYST** | 1.0 | Why is it moving? Breaking news sentiment, IPO recency, reverse splits, sector |
| **SUPPLY** | 1.0 | How constrained is the float? Share float (log scale), price |
| **DEMAND** | 0.9 | How much buying interest? Volume, RVOL (1-day + 5-min), tape density (prints/sec) |
| **ORDER_FLOW** | 0.8 | Who's aggressing? Trade imbalance, big sellers, hidden seller\*, spread\*, OFI chop\* |
| **TREND** | 0.6 | Macro trajectory: 5-day trend, above 200 EMA, ATH proximity, VWAP slope |
| **PRICE_ACTION** | 0.7 | Candle health: ROC, topping tails, red candle length, leg smoothness |

\* *These three variables require Level 2 depth data, which comes from IBKR. They're marked `optional` — when `IBKR_ENABLED` is off they are excluded from the group average entirely, so they don't inflate or deflate the score.*

**PRICE_ACTION** is special — it's evaluated per timeframe (10s, 1m, 2m, 5m) and then combined using timeframe-specific weights, because a topping tail on the 5-minute chart matters more than one on a 10-second candle.

You can inspect the effective weight of every variable with the `weights` REPL command. It prints a table where all effective weights sum to ≈ 1.0, so you can see at a glance which variables dominate (e.g. `float_shares` at ~12.8%) and which are minor (`above_200ema` at ~1.6%).

### Boolean criteria gates

Some conditions don't make the stock "better" — they're just checkboxes that must be met before you should consider entry. These are separate from the scoring system and run in parallel:

**ATTENTION** (advisory — never blocks entry, just flags concerns):
- Session volume ≥ 50k
- Distance from high of day ≤ 1.5 ATRs (how far has it pulled back?)

**EXECUTION** (hard gates — ALL required criteria must pass to allow entry):
- Last close is above session VWAP (trend direction)
- MACD histogram is positive on the 1-minute chart (momentum direction)
- VWAP slope is positive (trend direction)
- Distance to the nearest major resistance level ≥ 1× recent ATR (enough headroom to run)
- Volume profile is ascending over the last 10 bars (buyers still showing up)
- No hidden seller detected — *only enforced as a hard gate when IBKR is enabled (since detection requires L2 depth); otherwise it's advisory*

### Entry decision

Both systems must agree:

```
entry_allowed = composite score ≥ 60.0
                AND every required EXECUTION criterion is passing
```

If entry is blocked, `entry_block_reasons` lists exactly which gates are failing (e.g. `"exec:close_above_vwap"`, `"confidence: 47.3 < 60.0"`).

**Note:** Entry is not yet wired to actual order routing — `entry_allowed` is computed and displayed but nothing acts on it yet.

---

## Live Tracking: The Dynamic Scorer

Once a ticker passes evaluation, a `DynamicScorer` is started for it. This runs two threads continuously:

**Stream thread** — subscribes to the Tradier WebSocket and processes every tick in real time:
- Classifies each Time & Sales print as a buy or sell (by comparing to NBBO) to compute trade imbalance
- Tracks big sell prints (large size hitting the bid) in a rolling window
- Computes live RVOL (5-min) and tape density (prints per second)
- Feeds a multi-timeframe candle buffer (builds synthetic 10s/1m/2m/5m candles from raw ticks)
- Runs pattern detection whenever a candle closes on any timeframe
- Checks for resistance level breaks

**Poll thread** — fetches fresh 1-minute candles from Tradier REST every few seconds:
- Recomputes VWAP, MACD, and ATR from the latest candle data
- Updates all EXECUTION criteria gates (close vs VWAP, MACD sign, VWAP slope, resistance headroom, volume profile slope)
- Updates hidden seller intensity (only when IBKR is enabled — skipped otherwise)
- Evaluates per-timeframe PRICE_ACTION sub-scores (ROC, topping tails, red candle length, leg smoothness) and combines them

Dynamic score changes use **edge-triggered flags** — each condition (e.g. "RVOL dropped below threshold") fires once when the transition happens, not repeatedly. This prevents the delta from accumulating on sustained bad readings.

---

## Order Flow Tracking

The `OrderFlowTracker` consumes the same L1 stream data (Tradier NBBO quotes + Time & Sales) and produces several derived signals:

- **Trade imbalance** — what fraction of T&S prints are buy-side vs sell-side, classified by NBBO. This is an **active scoring variable**.
- **Big sellers** — count of large prints hitting at or below the bid in a rolling window. Also an **active scoring variable**.
- **Book lean** — (bid_size − ask_size) / (bid_size + ask_size), EMA-smoothed. Tells you whether the visible L1 book is bid-heavy or ask-heavy. Currently computed and shown in snapshots but not yet a scoring variable.
- **OFI** (Order Flow Imbalance, Cont 2014) — tracks bid/ask price and size deltas to measure net order flow pressure.
- **Hidden seller/buyer** — detects iceberg-like behavior: lots of aggressive volume hitting one side but the size isn't actually depleting (it's being replenished/refreshed). This requires L2 depth to work properly, so it's **gated on IBKR** and inactive until that's implemented.
- **OFI chop ratio** — how often the OFI sign is flipping (directional consistency). Also **IBKR-gated**.

All thresholds (window sizes, EMA alpha, hidden liquidity detection parameters) are centralized in `config.py`.

---

## Technical Analysis

### Indicators

All indicator functions are pure — they take a DataFrame and return a new one (no mutation). For the scorer's hot path where allocation matters, `_ip` (in-place) variants exist.

Available: EMA, MA, VWAP, Bollinger Bands, MACD, ROC, RVOL (1-day and 5-min), spread, and two regression-based tools:
- `linreg_r2_slope` — returns both the R² (goodness of fit) and slope from a linear regression over a window
- `signed_smoothness` — R² × sign(slope), giving a single value in [-1, +1] that captures both how clean and how directional a price leg is. Used for the `leg_smoothness` scoring variable.

### Resistance Map

The resistance map models price levels where the stock is likely to stall or reverse. It combines three types of levels:

- **Static** — whole-dollar and half-dollar levels near the current price (psychological levels)
- **Structural** — pivot highs identified from historical candle data, plus trendlines
- **Dynamic** — levels added at runtime when the scorer detects new highs or breaks

The `nearest_above(price)` method feeds directly into the `dist_to_resistance_ok` execution gate — if the nearest resistance is too close relative to ATR, there isn't enough room for the trade to work.

Topping-tail detection (upper wicks signaling rejection) also lives here and feeds into the PRICE_ACTION scoring group.

### Pattern Detection

Multi-timeframe pattern scanning runs on 10-second, 1-minute, 2-minute, and 5-minute candles. Patterns detected on shorter timeframes can **promote** to longer ones — for example, a micro-pullback that forms on the 10-second chart promotes to a 1-minute signal when confirmed across candle boundaries. We want multi-timeframe alignment (bullish across the board), not conflict (short-term bullish but longer-term bearish).

---

## Data Sources

Every external data provider has an on/off feature flag in `config.py`. When a provider is disabled, the system degrades gracefully — it skips that data source and works with what's available.

| Provider | Flag | What it provides |
|---|---|---|
| Tradier REST | `TRADIER_ENABLED` | Quotes, candles at multiple timeframes (10s, 1m, 15m, daily) |
| Tradier WebSocket | `TRADIER_STREAM_ENABLED` | Real-time Time & Sales prints + L1 NBBO quotes |
| Alpha Vantage | `AV_ENABLED` | Float, company profile, splits, entity-sentiment news. Has a built-in budget tracker (calls/min) |
| Finnhub | `FINNHUB_ENABLED` | Company profile, recent news articles |
| FMP | `FMP_ENABLED` | Float, IPO date, company profile |
| Brave Search | `BRAVE_ENABLED` | Web search fallback for news when other sources come up empty (**off by default** — general web search returns foreign-language/off-ticker results that FinBERT then scores) |
| FinBERT | `FINBERT_ENABLED` | Local transformer model for headline sentiment classification |
| YouTube | `SCANNER_YT_ENABLED` | Live chat polling for the YouTube scanner |
| Momo | `SCANNER_MOMO_ENABLED` | Momentum screener polling + float via livequote scraping |
| IBKR | `IBKR_ENABLED` | Quotes, historical candles and ticks, real-time T&S, and L2 depth-of-book (order-flow imbalance). Takes over from Tradier as the historical/streaming provider when enabled |
| IBKR scanner | `SCANNER_IBKR_ENABLED` | Native market scanner (`TOP_PERC_GAIN`) as a third ticker-discovery feed. Only effective when `IBKR_ENABLED` is also on |

All API keys are loaded from `.env` via `python-dotenv` and validated with a regex check before marking a provider as active. Keys are never hardcoded.

---

## Caching

The `TickerCache` is JSON-backed with a configurable TTL. It stores the full `WatchlistEntry` (all evaluation data + scoring groups) alongside the `ResistanceMap` for each ticker.

Both manual ticker lookups and scanner hits check the cache before running the pipeline. If a cached entry has stale news (headline was published before the current session), the consumer thread will try to refresh just the news portion without re-running the full evaluation.

---

## REPL Commands

The system runs as a terminal REPL (using `prompt_toolkit`). Type a ticker symbol to evaluate it, or use these commands:

```
GENERAL
  list                Show current watchlist with live scores
  cache               List all cached tickers
  cache <TICKER>      Show cached evaluation for a specific ticker
  cache clear         Wipe the entire ticker cache (memory + disk)
  weights             Print effective weight breakdown sorted by weight (Σ ≈ 1.0)
  weights group       Same breakdown but grouped by scoring group
  help                Full command reference
  q / quit / exit     Shutdown everything cleanly

MOMO SCANNER
  momo stop           Pause momo polling + disable float/news data
  momo go / start     Resume
  momo poll           Force an immediate poll cycle
  momo fast           Set poll interval to 2s
  momo def            Set poll interval to 5s (default)
  momo slow           Set poll interval to 15s
  momo interval <s>   Set a custom interval in seconds

IBKR SCANNER
  top                 Toggle a live view of the ranked top-gainers list
  ibkr stop           Pause IBKR top-gainers scanning
  ibkr go / start     Resume
  ibkr poll           Force an immediate scan
  ibkr fast/def/slow  Set interval to 3s (default) / 30s / 60s
  ibkr interval <s>   Set a custom interval in seconds

YOUTUBE SCANNER
  yt stop             Pause YouTube chat polling
  yt go / start       Resume
  yt poll             Force an immediate poll
  yt fast/def/slow    Set interval to 5s / 20s / 60s
  yt auto             Use server-recommended poll interval
  yt interval <s>     Set a custom interval in seconds

TRADIER PROVIDER
  tradier             Show Tradier state (running / paused)
  tradier stop        Pause all Tradier REST polls + live streaming
  tradier go / start  Resume

TICKER COMMANDS
  <TICKER>            Run full evaluation (checks cache first)
  <TICKER> refresh    Force a fresh evaluation, skipping the cache
  <TICKER> stop/go    Mute/unmute scanner output for this ticker
  <TICKER> find       Show when the ticker was first discovered
  <TICKER> noise      Mark the current headline as noise, try to find another
  <TICKER> float <v>  Override float (e.g. "5.2M" or "123.4k"), triggers re-score
  <TICKER> news "___"   Set a manual news headline, scored through FinBERT
```

---

## Configuration

All tuneable parameters live in `config.py`. The major sections:

- **Market Clock** — session hours (RTH, pre-market, post-market), NYSE holiday calendar
- **Stage 1 filters** — price range, minimum % change, minimum volume, RVOL thresholds
- **Stage 2 gates** — maximum float, news cascade order
- **Scoring groups** (`SCORING_GROUPS`) — the full definition of all 6 groups, their weights, every variable's quality function and thresholds
- **Criteria definitions** (`CRITERIA_ATTENTION`, `CRITERIA_EXECUTION`) — the boolean gate definitions and their thresholds
- **Dynamic scorer tuning** — poll interval, ROC period, RVOL window, spread thresholds, big-seller size threshold
- **Order flow** — OFI window size, trade classification window, EMA smoothing alpha, hidden liquidity detection thresholds
- **Feature flags** — on/off toggle for every data source and scanner

---

## Module Map

| Module | What it does |
|---|---|
| `main.py` | REPL loop, scanner thread orchestration, command dispatch |
| `pipeline.py` | `run_pipeline()` — the full evaluation flow from Stage 1 through display + scorer start |
| `scanners.py` | Background scanner threads: momo poller, YouTube chat poller, IBKR market scanner, shared consumer |
| `config.py` | Every constant, threshold, feature flag, and API key reference in one place |
| `cache.py` | JSON-backed ticker cache with TTL expiration |
| `display.py` | Terminal output formatting (stock summaries, watchlist table, scanner hit banners) |
| `strategy/evaluator.py` | `StockEvaluator` (Stage 1 + 2 logic), `WatchlistEntry` dataclass, `StaticEvaluation` |
| `strategy/scoring.py` | Quality functions, `ScoringVariable`, `VariableGroup`, `compute_composite`, `LiveInputs` |
| `strategy/criteria.py` | `Criterion`, `CriteriaGroup`, `build_criteria_groups` — the boolean gate engine |
| `strategy/confidence_scorer.py` | `DynamicScorer` — live scoring via Tradier streaming + REST candle polling |
| `technical_analysis/indicators.py` | Pure indicator functions (EMA, VWAP, MACD, etc.) + composite builders |
| `technical_analysis/resistance.py` | `ResistanceMap` — static, structural, and dynamic price levels |
| `technical_analysis/patterns.py` | Multi-timeframe pattern scanning + promotion logic |
| `exec/order_flow.py` | `OrderFlowTracker` — trade imbalance, OFI, book lean, hidden liquidity |
| `exec/tradier.py` | *(placeholder — order routing, not yet implemented)* |
| `exec/backtest.py` | `BacktestRunner` — event-time replay of cached tape + L2 books through the live signal stack |
| `exec/replay_session.py` | `ReplaySessionBuilder` — builds visual replay bundles (chunked NDJSON) for the visuals site |
| `tests/visuals/charts.py` | Chart payload builders (candles, overlays, indicator panes) + CLI entry for the visuals site |
| `tests/visuals/server.py` | Stdlib HTTP server: static frontend, chart/ticker/meta APIs, replay build + streaming endpoints |
| `tests/visuals/export.py` | Self-contained static HTML export (`--save`) — bundles the ES modules + vendored charts lib |
| `tests/visuals/static/` | The web frontend: vanilla ES modules, vendored Lightweight Charts, no build step |
| `market_data/*.py` | One client per data provider: Tradier REST/stream, AV, Finnhub, FMP, Brave, FinBERT, YouTube, Momo, IBKR stubs |
| `market_data/provider.py` | Factory functions that return the correct client based on feature flags |

---

## Visuals Website & Live Replay

```bash
make visuals                          # launch the site at http://127.0.0.1:8765/
python tests/visuals/charts.py AZI    # one-off static HTML export (works offline over file://)
python -m exec.replay_session CELZ 2026-08-06 --start 06:30 --end 07:30   # build a replay bundle headlessly
```

The site has three views, deep-linkable via URL query params:

- **Leaderboard** — session ticker table with per-symbol cached-data badges (**T** tick tape / **B** L2 book).
- **Charts** — the multi-timeframe candle panels (10s/1m/2m/5m) with VWAP/EMA overlays, quarter-second
  tape indicator panes, sim-trade markers, and a green **all-gates ribbon** under the candles wherever a
  replay bundle shows every entry-chain gate passing.
- **Replay** — pick a ticker + window (suggested windows come from cached near-HOD windows and sim entries)
  and watch the cached session play back as if live: candles form tick by tick, the L2 ladder and time &
  sales stream on the right, and the criteria panel shows every ATTENTION / EXECUTION criterion and the
  full `evaluate_entry` gate chain lighting up with live values until ENTRY ALLOWED fires and the sim
  entry lands on the chart. Play/pause, 1–25x speed, ±10s step, scrub-anywhere timeline, jump to fills.

Replay is **exact**: the server re-runs the cached window through the production `DynamicScorer` +
`OrderFlowTracker` + `SimulationEngine` (reusing `exec/backtest.py`'s loaders and seeding, without its
near-HOD window gating) and captures a deterministic, chunked bundle under `logs/replay_cache/` —
repeat views are instant, and sim output is routed inside the bundle so `logs/sim/` stays clean.
Symbols outside that day's cache snapshot degrade to chart + ladder + tape replay with the criteria
panel disabled (never synthesized).

---

## Testing

```bash
python -m pytest          # 399 tests, < 1 second
python -m pytest -q       # quiet mode
python -m pytest -x       # stop on first failure
```

All external APIs are mocked for testing — no network calls, fully deterministic. Test helpers build minimal fixtures (`_make_entry`, `_make_breakdown`, `_make_scorer`) so tests are fast and isolated. Indicator and resistance tests use small synthetic DataFrames rather than real market data.

---

## Not Yet Implemented

- **Order routing** — `exec/tradier.py` is empty. The `entry_allowed` property is computed and `entry_block_reasons` lists what's failing, but nothing acts on it yet.
- **Pullback-leg detector** — `signed_smoothness` measures full-window leg quality, but a dedicated detector that isolates individual pullback legs (vs. the entire price window) is deferred. Needs the pattern detector to be refactored first.
- **Weight tuning** — all scoring weights are hand-set based on intuition. To fit them properly (e.g. ridge regression) we'd need labeled outcomes — entry signals paired with actual P&L results.
- **Aggregate analytics** — the visuals site has no cross-day P&L/win-rate dashboard yet; backtest output is analyzed per-run.
