# Nested 10s setup-outcome suppression

- 10s pullbacks that occur inside a detected 1-minute pullback should NOT contribute to live `setup_outcome` / TECHNICAL_ANALYSIS `10s` score.
- `DynamicScorer._on_setup_resolved(..., pullback_ts=...)` now skips starting a 10s `_ActiveLeg` when `_is_10s_pullback_nested_in_1m_pullback(pullback_ts)` is true.
- `_on_setup_breakout_rejected` also skips immediate failed 10s outcomes for nested pullbacks.
- Setup display/signals can still surface the 10s setup; only the live setup-outcome scoring leg is suppressed.
- Regression coverage: `tests/test_files/test_scorer.py::TestSetupOutcome10Sec` nested resolved/rejected cases.