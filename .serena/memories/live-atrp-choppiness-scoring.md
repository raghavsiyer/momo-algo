# Live ATRP + candle choppiness scoring

Two live TA vars added to `config.SCORING_GROUPS` TECHNICAL_ANALYSIS group (feed
`confidence_score`), both `optional=True`, computed on candle close by
`DynamicScorer._update_atrp_and_choppiness(dfs_by_tf)` (called from
`_scan_patterns_on_close` after `_update_hod_rejection`; also runs in backtest replay).

## atrp (ATR percent)
- value = `self._last_atr / self._last_price * 100` (14-period 1-min ATR as % of price).
- `SATURATING` lo=2.0 hi=0.0, weight 0.6 → quality 0.5 at 2% ATRP, higher = better (more movement). User choice: higher=better.
- Cleared (value=None) when `_last_atr<=0` or price<=0.

## candle_choppiness
- Red/green candle alternation ratio, LOWER = cleaner = better. `LINEAR_MIN` lo=0 hi=0.6, weight 1.0.
- Pure helper `strategy.confidence_scorer._candle_choppiness_ratio(opens, highs, closes)`
  returns `(flips - excused) / (n-1)` in [0,1], or None if <2 candles.
  - flips = candle-color changes (green = close>=open).
  - A red run (pullback) is EXCUSED when a green in its recovery leg CLOSES over the
    pre-pullback peak high = the "pullback entry price" (user's rule: "breakout candle
    closes over pullback entry price"). So clean staircase trends (even RGRGRG that keeps
    making new highs) score ~0 chop; true oscillation (reds never recover) scores ~1.
- `_compute_candle_choppiness` blends per-TF ratios across ALL TFs (10s/1m/2m/5m, user
  choice) by `PRICE_ACTION_TF_WEIGHTS` (down-weights noisy 10s), over the active move
  window (`_active_price_action_start`), on CLOSED bars only (drops the forming bar via
  `_tf_buf.current_bar(tf)`), requiring >= `CANDLE_CHOP_MIN_CANDLES` (=4) per TF.
- None (excluded) when no TF has enough data.

## Display: candle_choppiness is a per-TF table row (like setup_outcome)
- `_compute_candle_choppiness` returns `(blended_ratio, {tf_score_key: per_tf_ratio})`.
- `_update_atrp_and_choppiness` stores per-TF QUALITIES in
  `self._tf_pa_var_qualities[tf]["candle_choppiness"]` via
  `compute_quality(ratio, chop_var.fn, chop_var.lo, chop_var.hi)` (imported from strategy.scoring).
- `_recompute_tf_setup_score` now MERGES into `_tf_pa_var_qualities` (setdefault) instead of
  replacing, so setup_outcome + candle_choppiness coexist per TF.
- display.py `_TA_PER_TF_VARS = ("setup_outcome", "candle_choppiness")`, label `_TA_VAR_LABELS["candle_choppiness"] = "candle chop"` (must be <= 16 chars for the table name column).
- The blended ratio still feeds the scoring var (aggregate quality via `_refresh_technical_analysis_score`); per-TF is display-only. The "combined" table row remains setup-only (`_tf_pa_scores`).
- Test: `test_display.py::test_weights_display_shows_all_live_ta_vars` asserts the LABEL "candle chop" (not the raw var name, which no longer appears since per-TF vars skip the normal-row path).

## Wiring notes (same pattern as hod_rejection)
- `LiveInputs` fields `atrp`, `candle_choppiness` (strategy/scoring.py).
- `_scored_live_input_names()` auto-includes them; `_reset_live_state` auto-migrates cached evals.
- `_make_evaluation` test helper seeds every var value=0.5, so any new optional TA var must be
  added to `test_eval.py::test_live_confidence_is_ta_only` `dynamic_vars` tuple (done for both).
- Tests: `test_scorer.py::TestCandleChoppinessRatio` (pure fn) and `::TestAtrpAndChoppiness`.

## Tooling note
- VS Code `runTests` UNDER-COUNTS test_scorer.py (reported 71 vs pytest's 81). Use pytest
  (`python -m pytest -q` via pylanceRunCodeSnippet) for authoritative counts. Full suite = 549 passed.
