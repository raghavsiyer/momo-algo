# Live HOD-rejection scoring (topping tail / jackknife)

Live confidence (`DynamicScorer` -> TECHNICAL_ANALYSIS -> `confidence_score`) now
penalizes rejection at pivots near the high of day, IN ADDITION to the pullback
`setup_outcome` channel.

- New live TA variable **`hod_rejection`** in `config.SCORING_GROUPS` TECHNICAL_ANALYSIS
  group: `LINEAR_MIN` lo=0 hi=1, `optional=True`, weight 1.5. Also a `LiveInputs` field.
  Because it is optional, when idle (`value=None`) it is EXCLUDED from the blend — it
  only affects the score when a rejection is present. Bigger rejection -> lower quality.
- Computed by `DynamicScorer._update_hod_rejection(dfs_by_tf)`, called from
  `_scan_patterns_on_close` (candle close, under lock) right after `_update_level_respect`.
  Also runs in backtest replay (ticks go through `_on_timesale`).
- Detection = "like setup scoring" (mirrors `patterns._setup_rejection_metrics`):
  for each TF (10s/1m/2m/5m) find CONFIRMED pivot highs (`patterns._find_pivot_highs`,
  per-TF N = PATTERN_PIVOT_N_10SEC/1MIN/2MIN/5MIN), keep only pivots within
  `PATTERN_HOD_PROXIMITY` of the RUNNING session HOD (`self._entry.daily_high`),
  compute ATR-normalized upper-wick ratio (topping tail) and, for red pivots, red-body
  ratio (jackknife) using `patterns._atr_from_arrays(..., TOPPING_TAIL_ATR_PERIOD)`,
  keeping only ratios >= `PATTERN_MIN_RAW`. The MAX ratio across all TFs drives the var.
  When none qualify, the var is cleared (`value=None`).
- Independent of pullback setups. Penalty direction only (confirmed with user).
- `_scored_live_input_names()` auto-includes it (field in both LiveInputs & SCORING_GROUPS).
- `cache._reset_live_state` auto-migrates cached evaluations that lack the new var
  (rebuilds TECHNICAL_ANALYSIS group when var names != SCORING_GROUPS expected).
- Display: renders in the per-TF TA table beside `setup_outcome` and
  `candle_choppiness` using per-TF qualities stored in
  `_tf_pa_var_qualities`; aggregate scoring still uses the worst raw
  rejection across TFs. When no TF qualifies, cells show empty/neutral rather
  than a scalar optional row.
- Tests: `tests/test_files/test_scorer.py::TestHodRejection`. Note `test_eval`'s
  `_make_evaluation` seeds every var value=0.5, so any new optional TA var must be added
  to the `dynamic_vars` tuple in `test_live_confidence_is_ta_only` (done).
- Distinct from the removed one-sided `wick_quality`/`body_impulse` live TA (see
  `mem:live_ta_candle_quality`) and from the separate SETUP_PATTERN topping_tail/jackknife.
