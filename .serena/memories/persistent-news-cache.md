# Persistent News Cache

- `TickerCache` now persists an independent `news_cache` per symbol: article, count, sentiment, source, update timestamp.
- `get_news()` only returns cached news inside the same catalyst window Stage 2 searches: last 2 trading days, with `NEWS_OLD_DAYS_MAX` fallback.
- `run_pipeline(..., cached_news=...)` applies cached news before summary printing, so refreshes no longer print `[NO NEWS]` when only live sources missed a still-valid cached catalyst.
- `TickerCache.put()` backfills news records from entries; cache hits/scanner paths hydrate entries via `apply_cached_news`.