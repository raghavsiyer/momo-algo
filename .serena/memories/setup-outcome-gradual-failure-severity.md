# Setup Outcome Gradual Failure Severity

- Completed setup outcomes now store failure-candle quality and rejection risk.
- Failed outcomes score as `setup_quality * extension_credit * failure_quality` instead of hard zero.
- `extension_credit = max favorable extension / SETUP_OUTCOME_SUCCESS_PCT`, clamped to [0, 1].
- (2026-07-26) `failure_quality` now uses the shared volume-response candle scoring (`_volume_response_baseline` + `_candle_volume_response`, same machinery as `hod_rejection`): continuous wick/body/range smoothstep gradients plus red-volume pressure. Volume is a severity MODULATOR (`SETUP_FAILURE_VOLUME_SEVERITY_FLOOR = 0.40`), not an eligibility gate — low-volume red failure candles are softened (healthy pullback), high-volume rejections keep full severity. `failure_rejection` = composed volume-response risk (0-1). Tiny red failures keep most partial credit; hard rejection candles reduce it sharply.
- Successful outcomes ignore failure quality and keep full extension-based credit up to setup quality.
