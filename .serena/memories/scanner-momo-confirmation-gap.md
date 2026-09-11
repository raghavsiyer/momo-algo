# Scanner tickers not auto-evaluated (momo confirmation window)

VERIFIED (not a guess): momo-sourced tickers often are NOT auto-evaluated; they only run on
manual entry. Root cause = momo confirmation window vs poll interval.

- `scanners.py::_momo_scanner` gates BOTH the hit print and queuing behind "2 distinct momentum
  snapshots within `_CONFIRM_WINDOW` = 10.0s" (scanners.py:126).
  - First appearance NEVER queues: just sets `first_seen[sym]=now` and `continue` (scanners.py:196-198).
  - Queues only if a 2nd new `updated_at` snapshot arrives while `now - first_seen <= 10s`
    (gate scanners.py:199-204, queue scanners.py:217-221).
- Default poll interval = 10s (`SCANNER_MOMO_INTERVAL=10`, config.py:925; used main.py:830).
  `now` is captured once per poll after the fetch; the 2nd poll's `now` is ~10s + fetch/loop
  overhead after `first_seen`, so `now - first_seen > _CONFIRM_WINDOW` (scanners.py:199) is
  almost always TRUE -> window resets -> ticker never confirmed -> never queued.
  Net: with default 10s (or 15s "slow") interval, momo tickers are effectively never auto-queued.
  Only works when interval is faster (2s "fast" / 5s "def") AND the ticker emits 2 fresh snapshots
  within 10s. Single-snapshot / slow-updating tickers are always skipped.
- Contributing: first poll seeds silently (scanners.py:159-168, ignores startup backlog);
  `queued`/`pipeline_done` dedup keeps hard-failed/queued syms from re-queuing.
- YouTube differs: news-author headlines ALWAYS queue (`ticker_q.put(("youtube-news",sym))`), so
  those evaluate; chat needs >= SCANNER_YT_MIN_MENTIONS (2); trusted-author needs FinBERT threshold
  + not cached/queued.
- Manual entry bypasses everything: main.py:2144 `run_pipeline(..., force=True)` — no confirmation
  window, no queued/pipeline_done check (only a separate manual-only `rejected` set at main.py:819).

Consumer (`_scanner_consumer`, scanners.py:494-748) evaluates fine once a ticker is dequeued; the
gap is purely in the momo queuing stage. Not yet fixed — reported to user.
