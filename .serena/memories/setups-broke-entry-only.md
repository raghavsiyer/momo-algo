# Setups category = broke-the-entry pullbacks only

A pullback is only counted/badged as a "setup" once it BREAKS the entry price / new
high (`pullback_new_high` resolves). A pullback abandoned while still `pullback_pending`
(never broke) is NOT a setup.

- `DynamicScorer._update_setup_pattern_confidence`: the `pullback_pending`-abandoned
  branch now CLEARS the setup state silently (group vars → None, all `setup_*` fields
  reset, `setup_failed_at` stays None). Previously it stamped `setup_failed_at`
  (a "Failed" badge) and left the group score lingering — that made a
  failed-without-breaking pullback look like a failed setup. Removed.
- Broke-then-failed still counts as a setup:
  - Green breakout candle then leg fails → stays "Resolved"; failure tracked in
    `setup_outcome` (`_finalize_leg` success=False).
  - Red breakout candle (`_check_breakout_candle_rejection` → `_invalidate_rejected_setup`)
    → sets `setup_failed_at` (the only remaining "Failed" setup badge) + logs APPROVED
    signal + records a failed `_CompletedOutcome`.
- Already correct before this change (no edit needed): the setup SIGNAL LOG
  (`_begin_setup_signal`, only on resolved) and `setup_outcome`/`_ActiveLeg`
  (`_on_setup_resolved`, only on resolved) never include pending-only pullbacks.
- Entries are unaffected: `exec/signals.evaluate_entry` already blocks entry until
  `last_price >= setup_level` (never enters before the breakout).
- `pullback_detected` (forming, per-tick) already cleared silently when it stops.
- Forming pullbacks carry NO setup score until recorded (they broke the entry):
  - `WatchlistEntry.setup_pattern_score` returns 0.0 when
    `setup_pattern_name in ("pullback_detected","pullback_pending")`.
  - `_check_intracandle_pullback_detection` (detected) and the `pending` branch of
    `_update_setup_pattern_confidence` NO LONGER call `_fill_setup_group`; they clear the
    group vars to None. Only the `resolved` branch fills the scored group.
  - `display.py`: the forming (`pullback_detected`/`pullback_pending`) rows show `—`
    instead of a percentage; group var rows render `—` (None). Only recorded setups show a score.
- Tests: `test_scorer.py::TestSetupPatternConfidence::` `test_pending_pullback_abandoned_without_breaking_is_not_a_setup`
  and `test_forming_pullback_has_no_score_until_recorded`. Full suite = 551 passed.
