# Live Full Historical Fetch Behavior

- `<ticker> live full` now prefers an active `DynamicScorer` snapshot and does not force a historical Tradier rebuild when the scorer exists.
- Historical fallback is still used for tickers without an active scorer, but full premarket fallback calls `_build_historical_live_snapshot(..., include_10s=False)`.
- Reason: 10s historical candles require raw `interval=tick` Tradier timesales over many 20-minute chunks; active movers can trigger `502 TooBigBody` and retry/bisect. The display can still show useful 1m/2m/5m/1d detail without reconstructing 10s for full premarket.