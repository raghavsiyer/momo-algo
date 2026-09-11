# Daily trend current-day exclusion

- Static daily trend scoring must use completed daily candles only; `pipeline.run_pipeline` calls `get_candles_daily(..., include_current_day=False)`.
- `StockEvaluator._daily_pct` is timestamp-aware: rows whose `ts` date is today in `MarketClock.ET` are filtered out, while no-`ts` test fixtures keep legacy last-row exclusion behavior.
- Focused validation: `tests/test_files/test_eval.py::TestDailyPct` and `tests/test_files/test_pipeline.py`.