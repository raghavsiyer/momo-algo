# Backtest cached live tape replay

- Backtest source priority is now: canonical `logs/chart_cache/timesales_<SYMBOL>_<YYYYMMDD>.csv`, legacy raw live timesales CSVs in `logs/`, cached chart candles (`10s` preferred, then `1m`/`2m`/`5m`) as synthetic close ticks, then Tradier tick fetch.
- Before replay starts, backtest seeds `DynamicScorer` with pre-start cached `1m`, `10s`, and raw timesales data from the current extended session, avoiding lookahead.
- If no raw cached timesales are found, `BacktestRunner` merges Tradier historical ticks with cached `logs/tape_YYYYMMDD.csv` rows where `entry_tape_gate=True`.
- Cached tape snapshots are applied directly to the scorer's `OrderFlowTracker` pressure deques before processing a synthetic at-ask tick, preserving live gate metrics (`buying_pressure`, `selling_pressure`, `buy_ratio`, `trade_speed`, acceleration) without expanding every cached print.
- Empty pullback replay windows no longer suppress valid cached near-HOD tape: `_replay_symbol` falls back to near-HOD windows when candle pullback derivation is available but returns `[]`. This fixes symbols like VIVK 2026-07-21 replaying `0` ticks despite cached near-HOD windows.
- Cached-window replay stays windowed even when the full cached tape has holes; `_iter_ticks(..., windows)` fills only holes inside selected replay windows. This avoids the KIDZ 2026-07-21 failure where a large full-range hole fetch skipped the actual replay window after Tradier body overflow.
- `_fetch_ticks` now recursively splits empty/overflowing 30-minute chunks down to 2-minute chunks instead of dropping the failed chunk.
- Focused coverage: `tests/test_files/test_backtest.py` (`test_backtest_replay_falls_back_to_near_hod_when_pullback_windows_empty`, `test_backtest_cached_window_replay_stays_windowed_when_full_tape_has_holes`, `test_backtest_fetch_ticks_recursively_splits_empty_overflow_chunks`).
