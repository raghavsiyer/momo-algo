# Sim Pullback Timeframe Display

- `pos` output should show pullback timeframe from `setup_timeframe` (`10sec` -> `10s`, `1min` -> `1m`) rather than formatting `setup_pullback_ts` as a time.
- Resolved `pullback_new_high` metadata must remain on `WatchlistEntry` during `_SETUP_RESOLVED_DISPLAY_SECS`; otherwise sim fills can copy only the score and display stale `setup=@50` with empty name/timeframe.
- After the resolved display window expires, clear `setup_resolved_at` with the setup fields so stale nameless setup scores cannot trigger entries.