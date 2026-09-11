# Visual chart trade markers

- `tests/visuals/charts.py` overlays trade fills from `logs/sim/sim_trades_YYYYMMDD.csv` (also supports `--log-dir` pointing directly at the sim dir).
- BUY fills render as green `arrowUp` markers below bars; SELL fills render as red `arrowDown` markers above bars.
- Marker text uses `side shares @ fill_price`, then `setup_name` or entry/exit reason, plus confidence when present.
- Payload frames include `markers` and `counts.fills`; renderer calls `candles.setMarkers(frame.markers || [])`.
