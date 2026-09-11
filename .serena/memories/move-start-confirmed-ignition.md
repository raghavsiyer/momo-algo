# Move Start Confirmed Ignition

- `technical_analysis.patterns.detect_move_start` now returns the first confirmed bar after the latest sustained return, not the first price-escaped bar.
- A confirmed bar means escaped from floor and RVOL >= `rvol_mult`; this prevents low-volume spread/bid-ask bounces from becoming `move_start_ts`.
- Regression case: SMXT on 2026-06-10 previously returned 07:30 ET / 04:30 PT; fixed behavior anchors near 09:29 ET / 06:29 PT.
- Validation: `tests/test_files/test_patterns.py::TestMoveStart` and full `tests/test_files/test_patterns.py` passed after the change.