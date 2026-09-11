# Visualizer Chart Cache Provenance

- Shared chart cache ownership lives in `chart_cache.py`; visualizer should import its loaders/writers instead of owning cache schema.
- Raw streamed T&S is cached under `logs/chart_cache/timesales_<SYMBOL>_<YYYYMMDD>.csv` while `DynamicScorer` streams a ticker.
- Canonical chart candle files are `logs/chart_cache/ohlcv_<TF>_<SYMBOL>_<YYYYMMDD>.csv` for `10s`, `1m`, `2m`, `5m`, with `source=stream` or `source=polled`.
- 10s cache `vwap` is session VWAP computed from tick-weighted per-bar `bar_vwap`; visualizer preserves cached 10s VWAP and only recomputes from raw ticks when available.
- Legacy `logs/chart_cache/tick_10s_ohlcv_<SYMBOL>_<YYYYMMDD>.csv` files are still read for compatibility and normalized to `source=stream`.
- Historical/API polling is opt-in globally via `allow_fetch`; polled rows must remain `source=polled`/gray and must not masquerade as streamed tick cache.