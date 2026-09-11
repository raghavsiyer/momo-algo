# Visualizer Higher-Timeframe Candle Provenance

- For `tests/visuals/charts.py`, raw cached timesales (`logs/chart_cache/timesales_*`) are the best available intraday truth when present.
- Scorer-written `ohlcv_1m/2m/5m` cache rows can be `source=stream` but still differ from raw T&S because scorer candle buffers are fed after `TS_MIN_NOTIONAL` filtering and may start mid-bucket.
- Disk chart cache should persist canonical sources only: raw timesales/10s and real 1m candles. Do not persist 2m/5m chart caches; derive those from 1m or 10s on read.
- Higher timeframe visual frames should prefer tick-derived bars over cached/polled candles for matching timestamps, while cached/polled/snapshot bars remain fallback for uncovered windows.
- On a Pacific machine, ET premarket raw T&S around 04:00-07:00 ET appears around 01:00-04:00 local because the visualizer currently uses system local timezone for session windows.