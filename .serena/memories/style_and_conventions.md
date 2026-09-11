# Style & Conventions

## Language & Version
- Python 3.12+ required
- Type hints on all function signatures

## Data Transfer Objects
- **dataclasses** for all DTOs — no Pydantic
- Key dataclasses: `Stage1Result`, `StaticBreakdown`, `WatchlistEntry`, `ScorerSnapshot`, `OHLCVCandle`, `StockQuote`, `LiveTimesale`, `LiveQuote`, `FloatData`, `SplitEvent`, `CompanyProfile`, `ResistanceLevel`
check
## Naming
- Constants: `UPPER_SNAKE_CASE` in `config.py`
- Private methods/functions: `_leading_underscore`
- Public API is minimal: `stage1`, `stage2`, `evaluate`, `scan`, `start`, `stop`, `snapshot`
- Test classes: `Test*`, test methods: `test_*`
- Test helpers: `_make_entry`, `_make_breakdown`, `_make_scorer` (private, minimal fixtures)

## Module Patterns
- One `logger = logging.getLogger(__name__)` per module
- Log messages prefixed with `[SYMBOL]` for traceability
- Color helpers: `_g` (green), `_y` (yellow), `_r` (red), `_c` (cyan), `_b` (bold), `_dim` (dim) for ANSI formatting

## Functions & Immutability
- Pure functions in `technical_analysis/indicators.py` — each returns a **new** DataFrame, original is never mutated
- In-place `_ip` variants (e.g. `_add_ema_ip`) exist only for the scorer's hot path
- Composite builders (`build_1min`, `build_5min`, etc.) chain the pure functions

## Threading
- Daemon threads for all background work (scanners, scorers)
- Shared state protected by `threading.Lock`
- `threading.Event` for stop/pause signals
- `queue.Queue` for scanner → consumer communication

## Feature Flags
- All in `config.py`: `TRADIER_ENABLED`, `TRADIER_STREAM_ENABLED`, `FMP_ENABLED`, `AV_ENABLED`, `FINNHUB_ENABLED`, `FINBERT_ENABLED`, `BRAVE_ENABLED`, `SCANNER_MOMO_ENABLED`, `SCANNER_YT_ENABLED`
- Noise filters: `NOISE_FILTER_TITLE`, `NOISE_FILTER_YT_HEADLINE`, `NOISE_FILTER_YT_TICKERS`, `NOISE_FILTER_BRAVE`
- Code must degrade gracefully when a provider is disabled (stub values, skip logic)

## API Keys
- Loaded from environment via `python-dotenv`
- Validated with `_key_ok()` regex check (alphanumeric, contains digit, ≥8 chars)
- Never hardcoded — `.env` is gitignored
- AV client has budget system (`self._av.budget`) tracking remaining calls per minute

## Error Handling
- External API failures logged and skipped — never crash the pipeline
- `failed_apis` list tracked in Stage 2 for diagnostics
- Volume-fail allows retry; hard failures are permanent
