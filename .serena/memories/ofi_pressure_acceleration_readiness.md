# OFI pressure acceleration readiness

- `OrderFlowTracker.pressure_acceleration` still returns a float and uses `0.0` when not measurable for compatibility.
- Use `OrderFlowTracker.pressure_acceleration_ready` to distinguish unknown from neutral.
- `DynamicScorer._update_tick_criteria` must require readiness before passing `EXECUTION:tape_acceleration`.
- Display should show pressure acceleration as warming/unknown until both halves of the pressure window have at least 5 prints.