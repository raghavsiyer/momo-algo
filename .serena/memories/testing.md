# Testing

## Setup
- **Framework**: pytest
- **Config**: `pytest.ini` with `pythonpath = .` and `testpaths = tests`
- **Root marker**: `conftest.py` (empty, just marks project root)
- **Run**: `python -m pytest` from project root

## Test Files
| File | Tests | Covers |
|---|---|---|
| `tests/test_eval.py` | ~50 tests | `validate_ticker`, all 7 delta methods, `StaticBreakdown`, `WatchlistEntry` (apply_dynamic, resolve_weakness, confidence clamping) |
| `tests/test_scorer.py` | ~30 tests | `DynamicScorer` edge-triggered flags, ROC eval, spread eval, chop detection, red volume, L2 quote handling, candle resampling, RVOL, snapshot, cancelled prints |
| `tests/test_indicators.py` | ~40 tests | All pure indicator functions (EMA, MA, VWAP, Bollinger, MACD, ROC, RVOL, spread), composite builders, immutability checks |
| `tests/test_resistance.py` | ~30 tests | `ResistanceMap` static/structural builds, nearest levels, mark broken, add dynamic, merge logic, pivot highs, topping tail detection |
| `tests/test_brave.py` | ~16 tests | `BraveNewsClient` HTTP mocking, response parsing, error handling, timestamp parsing |

## Conventions
- **All external APIs mocked** — zero network calls. Uses `unittest.mock.patch` / `MagicMock`
- Test helpers build minimal fixtures: `_make_entry(price=5.0)`, `_make_breakdown(news_delta=0)`, `_make_scorer(entry, price=5.0)`
- Indicator/resistance tests use **small synthetic DataFrames**, not real market data
- Tests are fast, isolated, deterministic — no shared state between methods
- Classes named `Test*`, methods named `test_*`
- No parametrize — each boundary/edge case is its own method for clarity
