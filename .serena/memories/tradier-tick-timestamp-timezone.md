# Tradier Tick Timestamp Timezone

- Tradier `interval=tick` timesales payloads may return offsetless UTC wall-clock strings.
- For tick resampling into 10s candles, parse naive tick `time` as UTC; treating it as ET shifts seeded 10s bars +4/+5 hours in live display.
- 1m/5m/15m timesales candle timestamps remain parsed as ET when offsetless.