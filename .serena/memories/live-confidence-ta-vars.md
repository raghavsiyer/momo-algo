# Live confidence TA vars (trimmed)

The TECHNICAL_ANALYSIS group (live `confidence_score`) now contains ONLY four vars:
`setup_outcome`, `hod_rejection`, `atrp`, `candle_choppiness`. All four are `optional`.

No-setups behavior (setup_outcome does nothing instead of a neutral 50%):
- `setup_outcome` is now `optional=True` in config, so when there are no setups (`value=None`)
  it is EXCLUDED from the group score rather than defaulting to q=0.5.
- `__init__` seeds `_tf_pa_var_qualities = {tf: {}}` (NOT `{tf: {"setup_outcome": 0.5}}`), so the
  `-w` table renders but the setup-outcome cells show `—` until real setups exist.
- `_recompute_tf_setup_score` no-data branches now `pop(tf_key)` from `_tf_pa_scores` and drop
  `setup_outcome` from `_tf_pa_var_qualities[tf]` (instead of seeding 0.5).
- `_refresh_technical_analysis_score`: when `_tf_pa_scores` is empty it sets `pa.score_override = None`
  (clears it) and returns, so `VariableGroup.score` excludes the optional setup_outcome. Live
  confidence then reflects only the real live signals (atrp/candle_choppiness), or nothing
  (`ta.score == 0.0`, `has_live_data` False -> display shows `L--`) when there is no live data.
- Display: no setups -> setup-outcome row is `—` and the "combined" row is omitted (tf_pa_scores empty).
- Tests: `test_scorer.py::TestNoSetupsLiveConfidence`.

Removed entirely (config + LiveInputs + scorer) per user request — they were pre-existing
and unwanted in the live confidence:
- `tape_activity`, `rvol_5min`, `big_sellers`, `hidden_seller`, `level_respect`.

What was removed:
- config.py: the 5 var defs from SCORING_GROUPS TECHNICAL_ANALYSIS; `BIG_SELLER_SIZE_THRESHOLD`,
  `BIG_SELLER_WINDOW_SECS`; the entire LEVEL_RESPECT_* constant block.
- strategy/scoring.py `LiveInputs`: fields `tape_activity`, `rvol_5min`, `big_sellers`,
  `hidden_seller`, `level_respect`. (Left `vwap_slope`, `above_200ema`, `ofi_chop` — they are
  NOT scored vars, just latent LiveInputs fields.)
- confidence_scorer.py: BIG_SELLER_/LEVEL_RESPECT_ imports; `_big_seller_events`,
  `_level_interactions`, `_last_level_scan_ts` init; the rvol_5min + tape_activity apply blocks
  in `_on_timesale`; the big-seller tracking + apply in `_on_quote`; the hidden_seller apply in
  `_update_criteria_and_hidden_seller`; the `_update_level_respect` call in `_scan_patterns_on_close`;
  and the methods `_update_level_respect`, `_record_level_interaction`, `_aggregate_level_respect`.
- tests: removed `TestLevelRespect` + `_seed_level`; updated `test_eval::test_live_confidence_is_ta_only`
  dynamic_vars and `test_display::test_weights_display_shows_all_live_ta_vars` to the 4 remaining.

Kept (used elsewhere, NOT for scoring):
- `_compute_rvol_5min` (feeds `ScorerSnapshot.rvol_5min` / display `rvol5m`), `entry.rvol_5min` (Stage 1).
- OFI `trade_speed` / `hidden_seller` (sim entry gating, criteria `no_hidden_seller`, EXEC_HIDDEN_SELLER_MAX).
- `_update_criteria_and_hidden_seller` still updates ATTENTION gates; its `include_hidden_seller`
  param is now DEAD (unused) but retained to avoid caller churn (backtest passes it).

Verify: `_scored_live_input_names()` == {setup_outcome, hod_rejection, atrp, candle_choppiness}.

GOTCHA (fixed): `LiveInputs` is `@dataclass(slots=True)`, so assigning a REMOVED field on an
instance raises AttributeError at runtime. When removing a LiveInputs field, grep for BOTH
`apply_live(LiveInputs(field=...))` AND attribute assignments like `poll_inputs.field = ...`.
`_refresh_candles` had a `poll_inputs.rvol_5min = rvol` that only fired live (rvol>0), not in
tests (no volume -> rvol=0 -> branch skipped) — removed it. Regression test:
`test_scorer.py::TestChartCacheIntegration::test_refresh_candles_with_volume_does_not_touch_removed_live_vars`
drives `_refresh_candles` with volume so rvol>0. Full suite 543 passed.
