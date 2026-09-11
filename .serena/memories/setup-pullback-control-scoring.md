# Setup Pullback Control Scoring

- Setup detection has a separate `pullback_control` metric on `PatternSignal`: weighted blend of shallow pullback depth and short pullback length using existing cleanliness depth/length weights.
- `pullback_control` is consumed only by `SETUP_PATTERN_GROUP` / `setup_pattern_score`; it intentionally does not feed `confidence_score` or live `TECHNICAL_ANALYSIS`.
- Pending pullbacks now set strength from `pullback_control` instead of fixed 0.5; resolved pullbacks carry `pullback_control` while retaining the broader cleanliness strength formula.
- Backtest replay instantiates `DynamicScorer`, so setup scoring changes apply there too; `DynamicScorer.__init__` migrates older cached setup groups that lack `pullback_control`.
- Pullback setup leg detection now uses a 2-5 green rising-candle suffix gated by both `PATTERN_PULLBACK_MIN_LEG_PCT` (2%) and `PATTERN_PULLBACK_MIN_LEG_ATR_MULT` (1x ATR); resolved/pending detectors also allow one extra green transition candle before the break.