# Live ROC signal

- `roc` is not a `TECHNICAL_ANALYSIS` scoring variable.
- Period-3 T&S ROC is computed by `DynamicScorer._eval_roc()` using `ROC_TS_PERIOD=3` and updates `ScorerSnapshot.last_roc` only.
- `CRITERIA_ATTENTION` no longer defines `roc_positive`; ROC does not gate focus/entry and is not shown in criteria gates.
- Keep ROC out of `LiveInputs` and `_SCORED_LIVE_INPUT_NAMES` unless it is deliberately moved back into weighted live scoring.
- Focused coverage: `tests/test_files/test_scorer.py::TestLiveRocSnapshot`.
