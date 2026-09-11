# Backtest causal replay guards

- Cached candles are only usable once their bar is fully closed: `BacktestRunner._closed_cached_candles` filters seed candles, synthetic chart ticks, near-HOD windows, and pullback replay windows by `bar_end <= replay_end`.
- Candle-derived near-HOD windows become active at bar close and remain active through the following candle close, because the backtest only learns whether the HOD gate ended when the next candle closes. Legacy `source=candle_cache` rows and old `source=candle_cache_close` rows are migrated on load; newly derived rows use `source=candle_cache_next_close`.
- `_replay_symbol` clears cached setup-pattern state and reconstructs starting volume/rvol from pre-start seed candles/timesales instead of inheriting later cache entry values.
- `_retime_replay_static_evaluation` refreshes replay-start `price`/`rvol_1day` STOCK_EVAL variables and drops cached news articles published after the replay timestamp.
- Focused coverage lives in `tests/test_files/test_backtest.py`; full suite passed after these guards (549 tests).