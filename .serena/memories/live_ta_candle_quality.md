# Live TA candle quality

- Live TECHNICAL_ANALYSIS renamed one-sided `topping_tail`/`jackknife` vars to signed `wick_quality` and `body_impulse`.
- `_update_price_action_quality` preserves near-HOD scoring gates: caller `_is_near_hod()` plus per-candle running-HOD proximity via `PATTERN_HOD_PROXIMITY`.
- `wick_quality` = lower-wick credit minus upper-wick rejection, weighted by close location and ATR-normalized.
- `body_impulse` = signed body/ATR, weighted by close location (green closes high positive, red closes low negative).
- Meaningful candle gate uses `range / ATR >= PATTERN_MIN_RAW`; recency weighting still uses `TOPPING_TAIL_DECAY`.
- `book_spread` is required, not optional; missing spread stays neutral until quote data arrives.
- TA `score_override` is the full TA live score: combined per-TF candle subscore blended with `tape_activity`, `big_sellers`, `hidden_seller`, `book_spread`, and `level_respect` by configured weights.
- `-w` renders all live-confidence TA vars under Technical Analysis: candle vars in the TF table when TF snapshots exist, all other TA vars as normal rows.
- `SETUP_PATTERN_GROUP`, `PatternSignal.topping_tail`, resistance `detect_topping_tail`, and topping-tail clusters remain separate old-name pattern/setup concepts.
