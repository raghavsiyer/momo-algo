# YouTube cashtag bypasses stopword filter (fixed)

## Symptom
A dummy ticker (e.g. `TEST`) kept appearing in `setup_signals_*` with price frozen at
`$5.00`, `eval=50 live=50`, zero tape (`B/S=0/0`). Not manually entered.

## Root cause
`market_data/youtube_client.py` filtered `_STOPWORDS` only on the BARE-ticker branch,
NOT the CASHTAG branch, in BOTH extractors:
- `extract_tickers`: cashtag loop (`_CASHTAG_RE.finditer`) added `m.group(1)` with no check.
- `extract_news_headlines`: cashtag match set `ticker` with no check (bare branch checked).

So bare `TEST` was blocked (it's in `_STOPWORDS`, line ~74) but `$TEST` slipped through.
`pipeline.validate_ticker` only enforces 1-5 letters (no symbol-existence check), so `$TEST`
was queued (youtube-news always queues; youtube-chat needs >= SCANNER_YT_MIN_MENTIONS).
With Tradier disabled, `StockEvaluator._stage1` returns a hard-coded stub
(`price=5.00, change_pct=15.0, passed=True`) for ANY symbol -> fake $5.00 ticker -> setups.
The news author periodically posting `$TEST` (feed test) explains the recurring batches.

## Fix
Apply `_STOPWORDS` to the cashtag branches too:
- `extract_tickers`: cashtag loop now `word = m.group(1); if word not in _STOPWORDS: found.add(word)`.
- `extract_news_headlines`: cashtag branch now `if ticker in _STOPWORDS: continue`.
Tests: `tests/test_files/test_youtube.py` (`TestExtractTickersStopwordCashtags`,
`TestExtractNewsHeadlinesStopwordCashtags`).

## Note / possible follow-up
`validate_ticker` still does no stopword/existence check (offered as optional 2nd layer,
not yet applied). The `$5.00` stub is the tell that `TRADIER_ENABLED` was false.
