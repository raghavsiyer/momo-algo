# Live TA Active Window Scoring

- Topping tail and jackknife live TA scoring now aggregates over the active intraday window: `max(latest_extended_session_start(reference), move_start_ts)` through current.
- Topping-tail/jackknife ATR normalization uses the rolling ATR available at each candle (`TOPPING_TAIL_ATR_PERIOD`), not the latest ATR applied backward.
- HOD gate uses running session HOD as of each candle, not final HOD and not move-window-only HOD.
- `live` detail display can still cap printed rows independently from scoring.
- `wick_quality`/`body_impulse` candle aggregation multiplies recency/materiality by relative eligible-candle volume; `leg_smoothness` uses volume-weighted signed R² over its lookback.