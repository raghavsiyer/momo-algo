# Visual chart premarket ticker table

- `tests/visuals/charts.py` server mode serves `/api/tickers` for the selected date/start/end window.
- Ticker rows are computed from local `tape_YYYYMMDD.csv` and `scoring_YYYYMMDD.csv` logs only; no network fetch.
- Initial page state shows a sortable premarket ticker table instead of placeholder text. Sorting supported by `volume` and `pctGain` (also symbol).
- Clicking a table row only fills the `#symbol` input and does not load the chart until Load is pressed.
