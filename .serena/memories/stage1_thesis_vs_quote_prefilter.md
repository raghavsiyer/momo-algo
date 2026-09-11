# Stage 1 terminology

- Strategy-level Stage 1 means the stock thesis/static evaluation score: percent change, float, news quality, RVOL/volume, price range, and daily chart setup estimate how well the stock can move.
- `StockEvaluator._stage1` is only the cheap Tradier quote prefilter: hard rejects are price range and percent change. It still records volume/RVOL for downstream thesis/readiness logic.
- Low volume/RVOL should lower quality or block attention/readiness criteria, not be treated as a quote-prefilter failure/retry reason.
- Scanner soft retry currently applies to low percent-change quote-prefilter failures only.
- ATTENTION `volume_min` is a live display/readiness gate: seed it from Stage 1 volume, then update it from valid T&S prints in `DynamicScorer._on_timesale`.