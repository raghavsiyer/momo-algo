# Setup signal logging

- `exec/setup_logger.py` persists setup discoveries separately from simulated fills as `logs/sim/setup_signals_YYYYMMDD.{csv,json}`.
- `DynamicScorer` starts a setup record when `pullback_new_high` resolves, updates approval/peak T&S metrics while active, and expires unapproved setups when cleared.
- `SimulationEngine` marks the newest matching setup as entered on simulated BUY; actual fills remain in `SimTradeLogger`.
- CLI command aliases `setup`, `setups`, `signals` show setup records; `pos` remains actual entries and now rolls realized P&L by symbol plus total.
- Focused validation: `tests/test_files/test_setup_logger.py`, `tests/test_files/test_sim_engine.py`, and `tests/test_files/test_scorer.py`.