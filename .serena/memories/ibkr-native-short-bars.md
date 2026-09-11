# IBKR native short historical bars
- Verified via Pylance on 2026-07-25 against IBKR at `127.0.0.1:4001` using AAPL.
- Direct `ib.reqHistoricalData(..., barSizeSetting="10 secs", durationStr="1800 S", whatToShow="TRADES", useRTH=False)` returned 180 bars.
- Direct `barSizeSetting="5 secs"` returned 360 bars.
- Repo `IBKRHistoricalClient.get_candles_10sec()` is wired to native `barSizeSetting="10 secs"`, but returns empty if `IBKR_ENABLED=False` in config.
- `IBKRHistoricalClient.get_candles_10sec_window()` now uses native `barSizeSetting="10 secs"` with an explicit UTC `endDateTime`; Pylance live probe returned 6 AAPL bars for `2026-07-24T23:59:00Z` to `2026-07-25T00:00:00Z`.
- When `IBKR_ENABLED=True`, `BacktestRunner` uses IBKR historical trade ticks via `IBKRHistoricalClient.get_timesales_tick()` for external replay fallback and cache-hole fills; 10s candle windows remain for seeding/chart context, not synthetic tick replay.
- In IBKR mode, raw cached live timesales still win, but synthetic cached chart ticks do not override real IBKR historical ticks for setup/entry replay.
- `get_timesales_tick()` fetches IBKR `TRADES` plus historical `BID_ASK` ticks and attaches the last quote at or before each trade timestamp, avoiding future quote lookahead while feeding replay spread gates with IBKR quote data.