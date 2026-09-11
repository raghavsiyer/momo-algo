# research/ — offline measurement harnesses

Ported into the repo 2026-08-14 (previously lived in ephemeral scratch —
nothing in git reproduced the 08-10 OFI study). All scripts are standalone;
run them from this directory with the project venv.

## Data

`logs/research/ofi_tune/` (v1) and `logs/research/ofi_tune_v2/` (v2, adds
supply-side arrays: `ask_loss_r`, `ask_add_r`, `bid_loss_r`, `bid_add_r`,
`buy_fill_r`, `sell_fill_r`, `nimb5`): one npz per symbol-day — production
`OrderFlowTracker` outputs sampled on a 250ms event grid over 30 cached L2
sessions (2026-07-27 → 08-06 era). Rebuild with `ofi_tune_build.py`
(reads `logs/chart_cache/`; edit `DAYS` to widen coverage — 142 book
symbol-days exist).

## Scripts

- `ofi_tune_build.py` — phase 1: raw book+tape caches → npz grids.
  `OFI_TUNE_OUT` overrides the output dir.
- `ofi_tune_study.py` — phase 2: zigzag legs (≥4%), OFI-variant sweep
  (coverage / delay / noise / fwd60/120 / stale-ON). Reproduced the cached
  2026-08-10 results after the port (772 depth-eligible legs;
  `norm30_tracker>=1.0` → 333/772, staleON 14.93%). `OFI_TUNE_DATA`
  overrides the npz dir (defaults to v1 for reproducibility).
- `ofi_tune_verify.py` — coverage by leg-gain tier + exit metrics on big legs.
- `label_level_approaches.py` — **Stage-C labeling harness** (2026-08-14):
  triple-barrier labels (+2% / −3% / 120s) at every approach of a zigzag
  pivot high into the armed band, with instantaneous features AND the
  armed-recent evidence proxy (any-true in trailing 10s — mirrors the
  tracker latches). Output: `logs/research/level_approach_labels.csv`.
  2026-08-23: added realized outcome returns (`tb_ret`, `fwd60_close`,
  `fwd120_close`) and the production-mirroring `progress15`/
  `progress_ready` feature. Event definition unchanged — the 08-14 summary
  table reproduces exactly (18,447 events, armed split 0.542/0.432).
- `fit_entry_posterior.py` — **Upgrade-1 fit** (2026-08-23): L2 logistic
  P(up-first) over the exec/posterior.py feature encoding (single source
  with the live path), leave-one-date-out CV, calibration + EV-threshold
  study, writes `exec/entry_posterior.json` with `--write`.

## First level-approach result (2026-08-14, v2 corpus, 18,447 events)

| cohort | n | P(up-first) | P(fwdmax120 ≥ 2%) | P(down-first) |
|---|---|---|---|---|
| all approaches | 18,447 | 0.427 | 0.484 | 0.343 |
| armed_recent=1 | 8,747 | 0.469 | **0.542** | 0.379 |
| armed_recent=0 | 9,700 | 0.390 | 0.432 | 0.312 |
| armed=1 & depth | 4,681 | 0.478 | **0.557** | 0.395 |

The 08-10 study's ceiling (~48–51% from a 45–47% base) applied to
conditioning AT gate-crossing instants. Level-approach conditioning was
untested; the 43.2% → 54.2% armed split is ~3× that ceiling's spread —
evidence the armed-band redesign conditions in the right place.

Honest caveats: (a) armed approaches are livelier in BOTH directions
(down-first also rises 31% → 38%) — arming alone is not positive EV against
these barriers; selectivity must come from the trigger + tier + structure
layers, and PnL-level truth comes from the backtest A/B, not this table.
(b) Barrier geometry is asymmetric (+2/−3) and is not the production exit
policy. (c) Zigzag pivot highs approximate setup levels without the scorer's
structure filters.

## Posterior fit result (2026-08-23, Upgrade 1)

L2 logistic over all 15 gate-source features, leave-one-date-out (7 dates):

- **OOF AUC 0.560**, logloss 0.6771 vs 0.6825 base — the AND-stack's
  signals, optimally weighted, rank approach outcomes only marginally
  better than chance. This QUANTIFIES the instantaneous-conditioning
  ceiling on the level-approach population.
- Calibration tracks to ~p=0.50 then the top decile OVER-predicts
  (pred 0.558, actual 0.478): maximum-simultaneous-evidence instants
  underperform their score — the climax shape, now visible in the fit.
- **EV-threshold rule was degenerate**: mean tb_ret is negative at every
  theta with real coverage (approaches are base-negative against +2/−3);
  positive-EV thetas select <0.2% of events (tail noise). Production edge
  lives in the exits, not the entry instant. Artifact threshold fell back
  to the calibration plateau: **theta* = 0.50** (top ~21% of approaches,
  OOF P(up) 0.476 vs 0.449 deployed base).
- Largest standardized coefficients: log1p_activity +0.27,
  accel_ready +0.14, volimb +0.12, depth_active −0.08. armed_ofi and
  armed_ratio come out NEGATIVE — consistent with the pro-cyclical
  signal diagnosis.

Consumed by `SIM_ENTRY_REGIME=posterior` (exec/signals.py).

3-arm replay A/B over all 9 cached days (07-27..08-06 in-corpus,
08-11/08-12 out-of-corpus), current code, 2026-08-23:

| arm | total PnL | trades | detail |
|---|---|---|---|
| legacy | +$472.16 | 4 | YXT 08-05 +9.15%/+13.38%, CLRO 08-06 +1.99%, GLE 08-11 −$140 |
| armed_band | +$467.92 | 4 | same trades, near-identical fills |
| posterior | +$261.39 | 2 | YXT 08-05 −2.93%/+13.38% only |

Attribution of the posterior's −$211 vs legacy (all in-corpus days):
  1. **Zone cap cost the biggest winner**: legacy's YXT entry at 5.2926 was
     +2.2% over the 5.18 level — extended zone, which the posterior refuses
     by design. That extended chase WORKED (+9.15%). The cap cuts both
     tails.
  2. One wrong refusal: CLRO 08-06 scored 0.415 < 0.50, legacy took it for
     +1.99%.
  3. Its one added entry (YXT 6.03-level cross @6.013, a better price) got
     half-flushed by the ofi_negative exit for −2.93% in 6.8s.
  4. The one refusal that helped: GLE 08-11 (0.478 < 0.50, saved −$140).

Notes: (a) the model UNDERPERFORMS on its own training days at the PnL
level — the barrier label it was fit to is not the production-exit payoff;
(b) across all arms the system traded on only 3 of 9 days — Stage-4
confidence gates (incl. live_confidence, measured non-predictive) block
most setups upstream of any entry regime; (c) every arm difference hangs
on 4 trades. Verdict: do NOT flip the default; posterior stays research
infrastructure (belief state for the Stage-2 stopping layer). Revisit the
no-extended-zone choice — an extended fallback stack (as armed_band has)
would have kept the +9.15% trade.

## Exit-payoff relabel + refit (2026-08-23, same day, follow-up)

- `feed_granularity_check.py`: IBKR batched tape PRESERVES print order
  (seq monotone 100%), per-print price/size (81–99% of clumps multi-price,
  ~99% multi-size), bid/ask side context (40–71% classifiable), and 6–500
  distinct timestamps/sec. **Absorption-sequence features are feasible on
  this feed**; only sub-flush timing is lost.
- v3 corpus (`ofi_tune_build.py` + accel_raw/buy_p/sell_p) +
  `relabel_exit_payoff.py`: every event walked through the production
  flow exits. Result: median hold 2.6s, 70% accel_collapse, mean −0.24%,
  P(pnl>=0.5%)=13.9%, corr(pnl_exit, tb_ret)=0.318 — the barrier label
  was measuring a substantially different quantity.
- Refit on pnl_exit>=0.5% (`--label exit`): **OOF AUC 0.665 vs 0.560** —
  top decile doubles the base win rate (27% vs 14%). Coefficients
  reordered: ratio +0.35 and speed +0.33 dominate (sustained dominance is
  what survives the accel-collapse exit); accel_ready flipped negative.
- EV still negative everywhere (best slice ≈ −0.5% mean): indiscriminate
  approach-entry loses under production exits even with 2x selection lift.
  The entry score alone cannot make the approach population positive —
  structure + trigger + selectivity remain load-bearing.
- Artifact `exec/entry_posterior.json` now carries the exit-label fit
  (theta*=0.2883, top ~5.6% take, calibration-plateau criterion; regime
  default remains legacy).
- Replay verify (tag `posterior3`, active days 08-05/06/11/12): 1 trade,
  1W/0L, +$334 — only the best trade (YXT 6.03 cross, +13.4%/13s), the
  barrier-model's −2.9% flush dropped, GLE refusal kept. Legacy +$472 on
  4 trades over the same days; the remaining gap is the zone-cap extended
  YXT (+9.15%) and CLRO (+2%).
