# Volume Response Live Confidence

- `DynamicScorer._update_hod_rejection` rates high-volume candle response, not penalty-only rejection.
- For each enabled stream TF (`10sec`, `1min`, `5min`; `2min` only when `PRICE_ACTION_TF_2M_ENABLED`), it scores closed candles with continuous gradients instead of a hard top-volume bucket.
- `VOLUME_RESPONSE_TOP_VOLUME_PCT` now defines the top-volume anchor for full response weight; candles below that can still matter if their volume is meaningfully above the TF median.
- Raw `hod_rejection` is blended rejection risk using per-candle rolling ATR, candle range share, close location, smooth upper-wick risk, smooth red-body risk, and `PRICE_ACTION_TF_WEIGHTS`.
- Candle size has its own significance gates: range weighting starts at `VOLUME_RESPONSE_RANGE_ATR_MIN=0.35` and reaches full range weight at `VOLUME_RESPONSE_RANGE_ATR_FULL=1.00`.
- Upper-wick rejection has its own significance gates: starts at `VOLUME_RESPONSE_WICK_ATR_MIN=0.35` and `VOLUME_RESPONSE_WICK_RANGE_MIN=0.25`; full wick-size risk at `VOLUME_RESPONSE_WICK_ATR_FULL=0.80` and full wick-shape risk at `VOLUME_RESPONSE_WICK_RANGE_FULL=0.60`.
- Red high-volume pressure has separate gates: starts at `VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN=1.50`, reaches full at `VOLUME_RESPONSE_RED_VOLUME_RATIO_FULL=2.50`, and can add up to `VOLUME_RESPONSE_RED_VOLUME_MAX_RISK=0.55` risk even when candle shape/range is otherwise mild.
- Small candles and small wicks stay near neutral unless red-volume pressure is active; large high-volume topping tails or jackknives produce high raw risk/low quality. No usable volume evidence leaves the optional variable excluded.
