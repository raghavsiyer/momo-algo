# Tradier Tick Volume Field Fallback

- Tradier `/markets/timesales?interval=tick` may return per-print share size as `volume` instead of `size`.
- `HistoricalClient.get_timesales_tick` normalizes `size = item.get('size') or item.get('volume')`; `_ticks_to_10sec_df` must do the same.
- ZYBT 2026-07-20 raw ticks had keys `time`, `timestamp`, `price`, `volume`; `size_sum=0`, `volume_sum>0`. Without fallback, tick-aggregated 10s candles have zero volume and volume-response scoring skips 10s.
