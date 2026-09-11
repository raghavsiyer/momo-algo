# Backtest cache event selection

- Daily cache snapshots now include entries whose `first_seen` OR `last_updated` falls on the requested ET day.
- Backtest replay includes a cache entry if `first_seen` is inside the 07:00-10:00 ET window, otherwise if `last_updated` is inside that window.
- For prior-day cached tickers re-evaluated today (e.g. RMSG), replay starts at today's `last_updated` time, not the old `first_seen`.
- Backtest replay sanitizes copied snapshot entries before streaming: refresh current scoring thresholds, reset live TECHNICAL_ANALYSIS state, and reset `daily_high` so HOD/fade logic rebuilds from replay ticks instead of snapshot-time dynamic state.
- Focused tests: `tests/test_files/test_backtest.py`.
