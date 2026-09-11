# No-news catalyst penalty (stock_eval)

Requirement (clarified): news always applies instantly when found (unchanged). But if a stock
has NO news and it's been > grace window since start-of-move, the `breaking_news` var flips from
optional/neutral (excluded) to a NEGATIVE contribution — the missing catalyst drags stock_eval down.
Note: "no news impacts stock_eval after 2 min" = the ABSENCE of news penalizes, not "news is blocked".
Nothing is ever blocked; manual overrides always apply.

Implementation:
- config.py: `NO_NEWS_GRACE_SECS = 120.0` (the time var, after start-of-move) and
  `NO_NEWS_PENALTY_QUALITY = 0.20` (breaking_news quality when penalized). Both near NEWS_* consts.
- `CompositeEvaluation.apply_no_news_penalty(quality)` (evaluator.py): sets breaking_news
  value=quality, quality=quality so it's INCLUDED (drags STOCK_EVAL down). breaking_news is
  `optional` LINEAR_RANGE lo0 hi1, so value=None = excluded/neutral (the default no-news state).
- `DynamicScorer._update_no_news_penalty(now)` (confidence_scorer.py): if `_move_start_ts` set AND
  `entry.avg_sentiment is None` (no news catalyst) AND `now - move_start > NO_NEWS_GRACE_SECS` ->
  `ev.apply_no_news_penalty(NO_NEWS_PENALTY_QUALITY)`. Gate on avg_sentiment (news paths own
  breaking_news when a catalyst exists; the penalty leaves avg_sentiment None so re-checks are
  idempotent and real news overrides it). `now` = event ts via `_coerce_utc_dt` (replay-safe).
  Called from `_on_timesale` candle-close block (`if closed_tfs:`), so the grace->penalty
  transition happens live/backtest within one candle of the 2-min mark.
- Scorer-only (no pipeline hook): a late-pickup no-news ticker shows optional briefly, then the
  scorer applies the penalty on the next candle close. Display: penalized breaking_news shows
  value 0.20 with a `✗` (quality < 0.25); within grace it shows `— opt`.
- Tests: `test_scorer.py::TestNoNewsPenalty` (within grace / after grace / news present untouched /
  no move_start). Full suite 547 passed.
