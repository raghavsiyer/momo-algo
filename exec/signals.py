"""
Entry and exit signal evaluation for the simulation engine.

Signals are pure functions that evaluate the current state of a
WatchlistEntry + OrderFlowTracker and return a decision with rationale.
They do NOT execute trades — that's the engine's job.

Signal hierarchy:
  1. Setup confirmation (pullback_new_high active with score >= threshold)
  2. Buying/selling pressure (T&S trade flow imbalance)
  3. OFI (Order Flow Imbalance — enforced while the L2 depth feed is live)
    4. Separate eval/live confidence thresholds
  5. Resistance proximity + flow acknowledgment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from exec.order_flow import OrderFlowTracker
    from strategy.evaluator import WatchlistEntry
    from technical_analysis.resistance import ResistanceMap

from config import (
    EXEC_DIST_TO_RESISTANCE_MULT,
    SIM_EXIT_ACCEL_MAX,
    SIM_EXIT_ACCEL_BUY_RATIO_MAX,
    SIM_EXIT_CONFIDENCE_MIN,
    SIM_EXIT_OFI_NORM_MAX,
    SIM_EXIT_SELL_RATIO_MIN,
    SIM_EXIT_RESISTANCE_ATR,
    SIM_OFI_NORM_MIN,
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_PROGRESS_MIN,
    SIM_PRESSURE_SPEED_MIN,
    SIM_PRESSURE_VOL_IMBALANCE_MIN,
    SIM_FIRST_MACD_CYCLE_ONLY,
    SIM_MAX_SETUP_EXTENSION_PCT,
    SIM_SETUP_ANTICIPATION_PCT,
    SIM_SETUP_SCORE_MIN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal result dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class EntrySignal:
    """Result of entry signal evaluation."""
    triggered: bool = False
    reasons: list[str] = field(default_factory=list)
    net_pressure: int = 0
    ofi: int = 0
    setup_score: float = 0.0
    eval_confidence: float = 0.0
    live_confidence: float = 0.0
    ts: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass(slots=True)
class ExitSignal:
    """Result of exit signal evaluation."""
    triggered: bool = False
    reason: str = ""
    net_pressure: int = 0
    ofi: int = 0
    eval_confidence: float = 0.0
    live_confidence: float = 0.0
    ts: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ---------------------------------------------------------------------------
# Tape entry gates — single source of truth (F10)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TapeGates:
    """Snapshot of the five T&S tape entry gates.

    Produced by :func:`tape_entry_gates` — the single decision surface.
    Since the armed-band redesign the VALUE-vs-FLOOR comparisons themselves
    live on ``OrderFlowTracker`` as gate predicates (``speed_gate_ok`` …
    ``ofi_gate_ok``) so the tracker's internal latch stamps and this
    surface read one definition. Consumed by ``evaluate_entry`` (live
    decision), ``DynamicScorer._update_tick_criteria`` (EXECUTION criteria
    display), ``DynamicScorer._setup_entry_readiness_score`` (soft
    readiness) and ``ScoringLog.tape_snapshot`` (``entry_tape_gate`` CSV
    column), so the surfaces can never drift apart again (pre-F10 the CSV
    gate silently used the 10s pressure count for activity while the live
    gate read the 30s ``activity_count``).
    """
    speed: float          # speed_gate (source per TAPE_SPEED_GATE_SOURCE)
    speed_ok: bool        # >= SIM_PRESSURE_SPEED_MIN
    activity: int         # activity_count (30s activity window)
    activity_ok: bool     # >= SIM_PRESSURE_ACTIVITY_MIN
    buy_ratio: float      # ratio_gate (source per TAPE_RATIO_GATE_SOURCE)
    ratio_ok: bool        # >= SIM_PRESSURE_BUY_RATIO_MIN
    vol_imbalance: float  # trade_imbalance (30s volume flow)
    vol_flow_ok: bool     # > SIM_PRESSURE_VOL_IMBALANCE_MIN
    accel_ready: bool     # selected accel estimator has enough data
    accel_z: float        # accel_gate_z from the selected estimator
    accel_ok: bool        # accel_ready and accel_z >= SIM_PRESSURE_ACCEL_Z
    progress: Optional[float]  # price_progress (None = insufficient history)
    progress_ok: bool     # progress is not None and > SIM_PRESSURE_PROGRESS_MIN

    @property
    def all_pass(self) -> bool:
        return (
            self.speed_ok
            and self.activity_ok
            and self.ratio_ok
            and self.vol_flow_ok
            and self.accel_ok
            and self.progress_ok
        )


def tape_entry_gates(ofi_tracker: "OrderFlowTracker") -> TapeGates:
    """Evaluate the T&S tape entry gates against the SIM_PRESSURE_* floors.

    Gate semantics (all must pass for a tape-confirmed entry):
      - speed: tape active enough (gate source per TAPE_SPEED_GATE_SOURCE,
        fast-EMA rate by default)
      - activity: minimum prints over the wider activity window — ratio and
        acceleration are meaningless on thin data. Deliberately has NO
        EXECUTION display criterion; it is enforced here and surfaced by
        the backtest block-reason path.
      - buy ratio: buyers must dominate (source per TAPE_RATIO_GATE_SOURCE)
      - volume flow: count-bullish tape with 30s dollar flow heavily
        sell-dominated is distribution, not accumulation (measured -0.89%
        mean fwd60 on such fires vs +0.58% clean)
      - acceleration: buying must be accelerating and statistically
        significant (estimator per TAPE_ACCEL_ESTIMATOR: 2bin | ema)
      - progress: price must have made net trailing progress — flow with a
        pinned price is absorption/churn, not ignition (the only gate that
        reads the market's RESPONSE; fails closed on <10s of tape history)
    """
    # Floor comparisons live on the tracker (gate predicates) so the
    # armed-band latch stamps and this decision surface can never drift.
    return TapeGates(
        speed=ofi_tracker.speed_gate,
        speed_ok=ofi_tracker.speed_gate_ok(),
        activity=ofi_tracker.activity_count,
        activity_ok=ofi_tracker.activity_ok(),
        buy_ratio=ofi_tracker.ratio_gate,
        ratio_ok=ofi_tracker.ratio_gate_ok(),
        vol_imbalance=ofi_tracker.trade_imbalance,
        vol_flow_ok=ofi_tracker.vol_flow_ok(),
        accel_ready=ofi_tracker.accel_gate_ready,
        accel_z=ofi_tracker.accel_gate_z,
        accel_ok=ofi_tracker.accel_gate_ok(),
        progress=ofi_tracker.price_progress,
        progress_ok=ofi_tracker.progress_gate_ok(),
    )


# ---------------------------------------------------------------------------
# Armed-band regime: zones, arm evidence, triggers
# ---------------------------------------------------------------------------
# The armed_band regime re-times the SAME gates instead of lowering floors:
# slow evidence is latched "true recently while price sat in the band"
# (tracker stamps, Step 3), one fast trigger fires the entry, and required
# evidence rises with extension (optimal stopping: waiting cost grows as the
# move extends, so demanding maximum simultaneous evidence at maximum
# extension — the legacy AND-of-instants — selects burst tops by
# construction; measured 08-11: fills +1.5–6.2% over first-full-evidence).

ZONE_BELOW = "below"        # [band floor, level): anticipation — strictest
ZONE_CROSS = "cross"        # [level, level*(1+tol)]: the break itself
ZONE_EXTENDED = "extended"  # above tol, up to the extension cap: full stack

# Latched arm evidence. Progress is deliberately NOT in this set: it
# measures price RESPONSE, which in the armed regime is the trigger's job —
# latching it with a multi-second TTL re-blocks quiet coils at the level
# (the exact pro-cyclical failure this redesign removes). It still binds in
# the EXTENDED zone via the full legacy stack. Activity stays a hard
# instantaneous validity gate (P5), OFI applies only while depth is live.
_ARM_TAPE_EVIDENCE = ("speed", "ratio", "vol_flow")


def entry_zone(last_price: float, setup_level: float) -> str:
    """Classify where in the entry band a price sits (armed_band regime)."""
    import config as _config
    if last_price < setup_level:
        return ZONE_BELOW
    if last_price <= setup_level * (1.0 + _config.SIM_LEVEL_CROSS_TOLERANCE_PCT):
        return ZONE_CROSS
    return ZONE_EXTENDED


def arm_evidence_missing(ofi_tracker: "OrderFlowTracker", ttl: float) -> list[str]:
    """Evidence gates whose latest in-band true-stamp is older than ``ttl``.

    Empty list = armed. Reads the tracker latches that are stamped from the
    SAME gate predicates the instantaneous surface uses (F10: no fork).
    """
    missing = [
        g for g in _ARM_TAPE_EVIDENCE
        if (age := ofi_tracker.gate_last_true_age(g)) is None or age > ttl
    ]
    if ofi_tracker.depth_active:
        age = ofi_tracker.gate_last_true_age("ofi")
        if age is None or age > ttl:
            missing.append("ofi")
    return missing


def trigger_state(ofi_tracker: "OrderFlowTracker") -> tuple[bool, bool]:
    """(micro-break fresh, accel rising-edge fresh) within SIM_TRIGGER_TTL_SECS."""
    import config as _config
    ttl = _config.SIM_TRIGGER_TTL_SECS
    mb = ofi_tracker.microbreak_age()
    ac = ofi_tracker.accel_edge_age()
    return (mb is not None and mb <= ttl, ac is not None and ac <= ttl)


# ---------------------------------------------------------------------------
# Entry signal evaluation
# ---------------------------------------------------------------------------

def entry_band_floor(setup_level: float, pullback_low: Optional[float]) -> float:
    """Lower bound of the entry band for a setup level — SINGLE source.

    Shared by evaluate_entry (decision), the scorer's tracker band push
    (latch stamping zone) and the engine's Position snapshot (structural
    stop), so all three always agree on where the band starts.

    legacy: fixed SIM_SETUP_ANTICIPATION_PCT fraction under the level.
    armed_band / posterior: scaled to the setup's own structure —
    max(pullback_low, level*(1-SIM_ANTICIPATION_MAX_PCT)) — pullbacks on
    these movers run 7–13% deep, so a fixed 2% floor wasted confirmations
    that completed below it (08-11: AIHS full stack green at 3.59, floor
    3.7534, fill 3.8119). Falls back to legacy when no pullback_low is
    known (pattern predates the plumb-through, or detected-state estimate).
    The posterior regime shares this floor because its training corpus
    labels approaches over the same structural band (BAND_DEPTH=6%).
    """
    import config as _config
    if (
        _config.SIM_ENTRY_REGIME in ("armed_band", "posterior")
        and pullback_low is not None
        and 0.0 < pullback_low < setup_level
    ):
        return max(pullback_low, setup_level * (1.0 - _config.SIM_ANTICIPATION_MAX_PCT))
    return setup_level * (1.0 - SIM_SETUP_ANTICIPATION_PCT)


def resistance_search_floor(last_price: float, setup_level: Optional[float]) -> float:
    """Price from which to search for the nearest overhead resistance.

    A pullback setup's own prior high is the level the trade is designed to
    break — it must never veto its own entry, or the headroom gate blocks
    every flag re-entry (which always sits just under that high) while
    happily passing post-blow-off retests (which sit >1 ATR under a spike
    high). Any resistance at/below the top of the approved entry band
    (setup_level * (1 + SIM_MAX_SETUP_EXTENSION_PCT)) is treated as part of
    the setup's own ceiling and skipped; headroom is then measured from
    ``last_price`` to the first level above the band. With no active setup
    level the search starts at ``last_price`` (legacy behavior).
    """
    if setup_level is None or setup_level <= 0:
        return last_price
    return max(last_price, setup_level * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT))


def evaluate_entry(
    entry: "WatchlistEntry",
    ofi_tracker: "OrderFlowTracker",
    last_price: float,
    last_atr: float,
    rmap: Optional["ResistanceMap"] = None,
) -> EntrySignal:
    """
    Evaluate whether conditions are met for a simulated buy entry.

    Requirements (ALL must pass):
    1. entry_allowed is True (eval/live confidence + Stage 4 criteria)
      2. Setup pattern active with score >= SIM_SETUP_SCORE_MIN
      3. Buying pressure > selling pressure by SIM_PRESSURE_RATIO_MIN
      4. ofi_norm_gate >= SIM_OFI_NORM_MIN (only while the depth feed is active)
      5. Multi-timeframe alignment (no bearish higher-TF conflict)

    Returns EntrySignal with triggered=True if all conditions pass.
    """
    sig = EntrySignal()
    reasons: list[str] = []

    # --- Confidence & criteria gate (Stage 4) ---
    sig.eval_confidence = entry.eval_confidence
    sig.live_confidence = entry.live_confidence
    if not entry.entry_allowed:
        return sig  # Don't even evaluate further

    # --- Setup pattern confirmation ---
    sig.setup_score = entry.setup_pattern_score
    if sig.setup_score < SIM_SETUP_SCORE_MIN:
        return sig

    setup_level = getattr(entry, "setup_price_level", None)
    if setup_level is None or setup_level <= 0:
        # No active price level → no entry band → no trade. Without this a
        # cleared/failure-badge setup state passed the score gate at the
        # vacuous 50 (empty variable group scores 0.5/var) with the band
        # SKIPPED and the one-buy-per-signal guard skipped too — YXT
        # 2026-08-05 12:15:05 bought a churn burst 4% below its rejected
        # signal's band, six seconds after that setup failed.
        return sig
    # Entry band around the new-high trigger. The tape confirmation is
    # the trigger, not the price break: the lower bound sits slightly
    # UNDER the level (anticipation — buyers absorbing the dip confirm
    # before the break and enter at the best prices in the structure),
    # the upper bound caps extension so a late confirmation can't chase.
    min_entry_price = entry_band_floor(
        setup_level, getattr(entry, "setup_pullback_low", None)
    )
    if last_price < min_entry_price:
        return sig
    max_entry_price = setup_level * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT)
    if last_price > max_entry_price:
        return sig

    # --- First-MACD-cycle gate ---
    # Only the move's first 1m MACD cycle is tradeable: once the histogram
    # completes its first positive->negative flip since move start (latched
    # by the scorer on CLOSED 1m bars), every later leg is blocked — this
    # applies to ALL setup timeframes, not just the one that triggered.
    if SIM_FIRST_MACD_CYCLE_ONLY and getattr(entry, "macd_cycle_done", False):
        return sig

    # --- T&S tape conditions (single source: tape_entry_gates, F10) ---
    buy_p = ofi_tracker.buying_pressure
    sell_p = ofi_tracker.selling_pressure
    sig.net_pressure = buy_p - sell_p
    sig.ofi = ofi_tracker.ofi_rolling

    import config as _config
    gates = tape_entry_gates(ofi_tracker)
    if _config.SIM_ENTRY_REGIME == "posterior":
        # --- Fitted-posterior regime (Upgrade 1, 2026-08-23) ---
        # The tape/OFI AND-of-floors is replaced by ONE calibrated
        # probability over the same gate-source reads (exec/posterior.py):
        # graded evidence traded off by fitted weights instead of every
        # signal clearing a hand-picked floor simultaneously. Structural
        # gates above (score/band/MACD) and below (headroom) stay hard.
        # Activity stays a hard validity gate — ratio/accel are meaningless
        # on thin tape, and the fit's EV threshold was chosen on the
        # activity-passing population.
        if not ofi_tracker.activity_ok():
            return sig
        # Scored zone: the corpus labels approaches only up to
        # level*(1+SIM_LEVEL_CROSS_TOLERANCE_PCT). Above it the model
        # would extrapolate into exactly the chased-extension region the
        # climax diagnosis documented — no entry, no fallback stack.
        if last_price > setup_level * (1.0 + _config.SIM_LEVEL_CROSS_TOLERANCE_PCT):
            return sig
        # Timing trigger: a fresh micro-break (print through the trailing
        # high). The score says WHETHER this moment is worth taking; the
        # break says NOW — without it the first tick crossing theta can be
        # an arbitrary churn instant (the armed regime's lesson, kept).
        trig_break, _ = trigger_state(ofi_tracker)
        if not trig_break:
            return sig
        from exec.posterior import entry_threshold, score_entry
        posterior = score_entry(ofi_tracker, last_price, setup_level)
        theta = entry_threshold()
        if posterior < theta:
            return sig
        reasons.append(f"posterior={posterior:.3f}>={theta:.2f} trig=break")
        reasons.append(
            f"tape={gates.speed:.0f}t/s {gates.buy_ratio:.0%}buy z={gates.accel_z:.1f}"
        )
        if ofi_tracker.depth_active:
            reasons.append(f"ofi={sig.ofi} norm={ofi_tracker.ofi_norm_gate:.2f}")
    elif _config.SIM_ENTRY_REGIME == "armed_band":
        zone = entry_zone(last_price, setup_level)
        if zone == ZONE_EXTENDED:
            # Late entries pay the worst prices, so extension demands the
            # FULL legacy instantaneous stack (evidence rises with
            # extension — the legacy regime's only zone is this one).
            if not gates.all_pass:
                return sig
            if ofi_tracker.depth_active and not ofi_tracker.ofi_gate_ok():
                return sig
            reasons.append("zone=extended")
            reasons.append(
                f"tape={gates.speed:.0f}t/s {gates.buy_ratio:.0%}buy "
                f"accel={ofi_tracker.accel_gate_raw:+.2f} z={gates.accel_z:.1f}"
            )
            if ofi_tracker.depth_active:
                reasons.append(f"ofi={sig.ofi} norm={ofi_tracker.ofi_norm_gate:.2f}")
        else:
            # Armed zones (below / cross): evidence was latched while price
            # sat in the band; one fresh trigger fires the entry.
            if not ofi_tracker.activity_ok():
                return sig
            strong = zone == ZONE_BELOW
            ttl = (
                _config.SIM_ARM_STRONG_TTL_SECS
                if strong else _config.SIM_ARM_EVIDENCE_TTL_SECS
            )
            if arm_evidence_missing(ofi_tracker, ttl):
                return sig
            trig_break, trig_accel = trigger_state(ofi_tracker)
            if strong:
                # Sub-break entry (the manual method's "slightly earlier"):
                # dip absorption must be happening NOW — instantaneous
                # buyer dominance, optional wall-evaporation read, and only
                # the micro-break trigger counts (an accel edge without a
                # trailing-high break below the level is a bounce, not a
                # resumption).
                if not ofi_tracker.ratio_gate_ok():
                    return sig
                if _config.SIM_ARM_STRONG_REQUIRES_WALL and not ofi_tracker.wall_evidence_ok():
                    return sig
                if not trig_break:
                    return sig
            elif not (trig_break and trig_accel):
                # The cross entry needs the break AND a burst of buying
                # occurring WITH it (user's manual method: a lone print
                # poking the trailing high on a fading tape is not a
                # trigger). During a real break both stamps refresh
                # continuously, so the AND fires at max(break onset,
                # burst onset) — no added lateness on genuine moves.
                return sig
            # --- Book confirmation AT the trigger (both armed zones) ---
            # 1) The book must be imbalancing NOW — instantaneous
            #    depth-normalized OFI clears the floor at the trigger
            #    instant, not merely within the latch TTL. At all five
            #    08-11 live breaks this held at the break moment; it only
            #    filters quiet pokes.
            # 2) Fresh-seller veto: dip bought strongly, but a huge seller
            #    just parked on the level (wall mass >= ~2x its baseline)
            #    — do not take the break into it. Vacuous without depth or
            #    coverage; never required, only vetoing.
            if ofi_tracker.depth_active:
                if not ofi_tracker.ofi_gate_ok():
                    return sig
                wall = ofi_tracker.wall_trend
                if wall is not None and wall <= _config.SIM_WALL_REPLENISH_VETO:
                    return sig
            trig_name = (
                "break+accel" if (trig_break and trig_accel)
                else ("break" if trig_break else "accel")
            )
            reasons.append(f"zone={zone} armed<={ttl:.0f}s trig={trig_name}")
            reasons.append(
                f"tape={gates.speed:.0f}t/s {gates.buy_ratio:.0%}buy z={gates.accel_z:.1f}"
            )
            if ofi_tracker.depth_active:
                wall = ofi_tracker.wall_trend
                wall_str = f" wall={wall:+.2f}" if wall is not None else ""
                reasons.append(
                    f"ofi={sig.ofi} norm={ofi_tracker.ofi_norm_gate:.2f}{wall_str}"
                )
    else:
        if not gates.all_pass:
            return sig

        reasons.append(
            f"tape={gates.speed:.0f}t/s {gates.buy_ratio:.0%}buy "
            f"accel={ofi_tracker.accel_gate_raw:+.2f} z={gates.accel_z:.1f}"
        )

        # --- OFI confirmation (gated on a live L2 depth feed) ---
        # Gate on depth-NORMALIZED OFI: net flow as a fraction of the
        # standing book (≈ implied price impact), comparable across
        # thick/thin names. Estimator per OFI_GATE_SOURCE (EMA kernel by
        # default). Raw ofi_rolling stays on the signal for logging.
        if ofi_tracker.depth_active:
            if not ofi_tracker.ofi_gate_ok():
                return sig
            reasons.append(f"ofi={sig.ofi} norm={ofi_tracker.ofi_norm_gate:.2f}")

    # --- Multi-timeframe alignment check ---
    # Reject if there's a strong bearish signal on a higher timeframe
    if not _config.CONFIDENCE_BYPASS and hasattr(entry, "evaluation") and entry.evaluation is not None:
        pa_group = None
        for g in entry.evaluation.groups:
            if g.name == "PRICE_ACTION":
                pa_group = g
                break
        # If price action group exists and has very low score, TFs are conflicting
        if pa_group is not None and pa_group.score < 0.3:
            return sig

    # --- Resistance headroom (sanity: don't buy into a wall) ---
    # The setup's own prior high (anything inside the approved entry band)
    # is not a wall — it's the level being broken. Skip it and measure to
    # the first genuinely distinct level above the band.
    if rmap is not None and last_atr > 0:
        nearest = rmap.nearest_above(resistance_search_floor(last_price, setup_level))
        if nearest is not None:
            headroom = nearest.price - last_price
            if headroom < EXEC_DIST_TO_RESISTANCE_MULT * last_atr:
                # Too close to resistance — skip entry
                return sig

    # All conditions met
    tf_tag = f"_{entry.setup_pattern_timeframe}" if entry.setup_pattern_timeframe else ""
    reasons.append(f"setup=pullback{tf_tag}@{sig.setup_score:.0f}")
    live_str = f"{sig.live_confidence:.1f}" if entry.has_live_data else "--"
    reasons.append(f"eval={sig.eval_confidence:.1f} live={live_str}")
    sig.triggered = True
    sig.reasons = reasons
    return sig


# ---------------------------------------------------------------------------
# Entry-chain gate introspection (replay/visuals)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GateState:
    """One entry-chain gate's current state, for display surfaces.

    ``active=False`` means the gate is not applicable right now (no L2 depth,
    no setup level, no resistance overhead) and passes vacuously — exactly as
    ``evaluate_entry`` treats it.
    """
    name: str
    label: str
    passing: bool
    value: float | None
    threshold: float | None
    direction: str  # ">=", ">", "band", "bool"
    active: bool = True
    detail: str = ""


def entry_gate_states(
    entry: "WatchlistEntry",
    ofi_tracker: "OrderFlowTracker",
    last_price: float,
    last_atr: float,
    rmap: Optional["ResistanceMap"] = None,
) -> list[GateState]:
    """Report every gate ``evaluate_entry`` enforces, in its exact order.

    ``evaluate_entry`` short-circuits, so it cannot say which later gates
    would also fail; this evaluates each gate independently for display.
    Invariant (tested): ``all(g.passing) == evaluate_entry(...).triggered``.
    The first-MACD-cycle gate is reported only while
    SIM_FIRST_MACD_CYCLE_ONLY is enabled (mirroring enforcement); the dead
    PRICE_ACTION multi-TF check (no such group exists in SCORING_GROUPS)
    is excluded.
    """
    states: list[GateState] = []

    states.append(GateState(
        name="entry_allowed",
        label="Entry allowed",
        passing=bool(entry.entry_allowed),
        value=None,
        threshold=None,
        direction="bool",
    ))

    setup_score = float(entry.setup_pattern_score)
    states.append(GateState(
        name="setup_score",
        label="Setup score",
        passing=setup_score >= SIM_SETUP_SCORE_MIN,
        value=setup_score,
        threshold=SIM_SETUP_SCORE_MIN,
        direction=">=",
    ))

    setup_level = getattr(entry, "setup_price_level", None)
    if setup_level is not None and setup_level > 0:
        min_entry_price = entry_band_floor(
            setup_level, getattr(entry, "setup_pullback_low", None)
        )
        max_entry_price = setup_level * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT)
        states.append(GateState(
            name="price_band",
            label="Price in band",
            passing=min_entry_price <= last_price <= max_entry_price,
            value=last_price,
            threshold=None,
            direction="band",
            detail=f"{min_entry_price:.4g}-{max_entry_price:.4g}",
        ))
    else:
        # No active level = no band = no trade (evaluate_entry hard-fails
        # here — the failure-badge state must not read as vacuously green).
        states.append(GateState(
            name="price_band",
            label="Price in band",
            passing=False,
            value=last_price,
            threshold=None,
            direction="band",
            detail="no setup level",
        ))

    if SIM_FIRST_MACD_CYCLE_ONLY:
        cycle_done = bool(getattr(entry, "macd_cycle_done", False))
        done_ts = getattr(entry, "macd_cycle_done_ts", None)
        if cycle_done:
            detail = (
                f"1m hist flipped negative @ {done_ts:%H:%M}Z"
                if done_ts is not None else "1m hist flipped negative"
            )
        else:
            detail = ""
        states.append(GateState(
            name="macd_cycle",
            label="First MACD cycle",
            passing=not cycle_done,
            value=None,
            threshold=None,
            direction="bool",
            detail=detail,
        ))

    import config as _config
    gates = tape_entry_gates(ofi_tracker)
    armed_zone: Optional[str] = None
    if (
        _config.SIM_ENTRY_REGIME == "armed_band"
        and setup_level is not None
        and setup_level > 0
    ):
        armed_zone = entry_zone(last_price, setup_level)

    if _config.SIM_ENTRY_REGIME == "posterior":
        # Posterior chain mirrors evaluate_entry's posterior path exactly:
        # activity validity -> scored zone -> fitted probability floor.
        # The AND-stack gates are deliberately absent — they are features
        # of the score now, not gates — and the OFI floor below is skipped
        # for the same reason.
        states.append(GateState(
            name="tape_activity",
            label="Tape activity",
            passing=ofi_tracker.activity_ok(),
            value=float(ofi_tracker.activity_count),
            threshold=float(SIM_PRESSURE_ACTIVITY_MIN),
            direction=">=",
        ))
        if setup_level is not None and setup_level > 0:
            zone_top = setup_level * (1.0 + _config.SIM_LEVEL_CROSS_TOLERANCE_PCT)
            states.append(GateState(
                name="scored_zone",
                label="Scored zone",
                passing=last_price <= zone_top,
                value=last_price,
                threshold=zone_top,
                direction="<=",
                detail="model support ends at the cross tolerance",
            ))
            trig_break, _ = trigger_state(ofi_tracker)
            states.append(GateState(
                name="trigger",
                label="Entry trigger",
                passing=trig_break,
                value=None,
                threshold=_config.SIM_TRIGGER_TTL_SECS,
                direction="<=",
                detail="micro-break fresh (timing; the score is the whether)",
            ))
            from exec.posterior import entry_threshold, get_model, score_entry
            posterior = score_entry(ofi_tracker, last_price, setup_level)
            theta = entry_threshold()
            states.append(GateState(
                name="posterior",
                label="Entry posterior",
                passing=posterior >= theta,
                value=round(posterior, 4),
                threshold=theta,
                direction=">=",
                detail=str(get_model().meta.get("label", "fitted win probability")),
            ))
        else:
            states.append(GateState(
                name="posterior",
                label="Entry posterior",
                passing=False,
                value=None,
                threshold=None,
                direction=">=",
                active=False,
                detail="no setup level",
            ))
    elif armed_zone is not None and armed_zone != ZONE_EXTENDED:
        # Armed-zone chain mirrors evaluate_entry's armed path exactly.
        strong = armed_zone == ZONE_BELOW
        ttl = (
            _config.SIM_ARM_STRONG_TTL_SECS
            if strong else _config.SIM_ARM_EVIDENCE_TTL_SECS
        )
        states.append(GateState(
            name="tape_activity",
            label="Tape activity",
            passing=gates.activity_ok,
            value=float(gates.activity),
            threshold=float(SIM_PRESSURE_ACTIVITY_MIN),
            direction=">=",
        ))
        missing = arm_evidence_missing(ofi_tracker, ttl)
        states.append(GateState(
            name="arm_evidence",
            label="Armed evidence",
            passing=not missing,
            value=None,
            threshold=ttl,
            direction="<=",
            detail=(
                f"stale/never: {', '.join(missing)}" if missing
                else f"{armed_zone}: all true within {ttl:.0f}s in band"
            ),
        ))
        trig_break, trig_accel = trigger_state(ofi_tracker)
        if strong:
            states.append(GateState(
                name="ratio_now",
                label="Buy ratio (now)",
                passing=gates.ratio_ok,
                value=gates.buy_ratio,
                threshold=SIM_PRESSURE_BUY_RATIO_MIN,
                direction=">=",
                detail="sub-break entries need live dip absorption",
            ))
            if _config.SIM_ARM_STRONG_REQUIRES_WALL:
                wall_applicable = ofi_tracker.depth_active
                states.append(GateState(
                    name="wall",
                    label="Ask-wall evaporation",
                    passing=ofi_tracker.wall_evidence_ok(),
                    value=ofi_tracker.wall_trend,
                    threshold=_config.SIM_WALL_EVAP_MIN,
                    direction=">=",
                    active=wall_applicable,
                    detail="" if wall_applicable else "no L2 depth",
                ))
            trig_pass, trig_detail = trig_break, "micro-break only below the level"
        else:
            trig_pass = trig_break and trig_accel
            trig_detail = "micro-break AND accel burst together"
        fired = (
            "break+accel" if (trig_break and trig_accel)
            else ("break" if trig_break else ("accel" if trig_accel else "none"))
        )
        states.append(GateState(
            name="trigger",
            label="Entry trigger",
            passing=trig_pass,
            value=None,
            threshold=_config.SIM_TRIGGER_TTL_SECS,
            direction="<=",
            detail=f"{trig_detail} (fresh: {fired})",
        ))
        # Fresh-seller veto at the trigger — a wall that just parked on the
        # level refuses the entry (vacuous without depth/coverage).
        wall = ofi_tracker.wall_trend
        wall_applicable = ofi_tracker.depth_active and wall is not None
        states.append(GateState(
            name="wall_veto",
            label="No fresh wall",
            passing=(not wall_applicable) or wall > _config.SIM_WALL_REPLENISH_VETO,
            value=wall,
            threshold=_config.SIM_WALL_REPLENISH_VETO,
            direction=">",
            active=wall_applicable,
            detail="" if wall_applicable else "no wall read",
        ))
    else:
        states.append(GateState(
            name="tape_speed",
            label="Tape speed",
            passing=gates.speed_ok,
            value=gates.speed,
            threshold=SIM_PRESSURE_SPEED_MIN,
            direction=">=",
        ))
        states.append(GateState(
            name="tape_activity",
            label="Tape activity",
            passing=gates.activity_ok,
            value=float(gates.activity),
            threshold=float(SIM_PRESSURE_ACTIVITY_MIN),
            direction=">=",
        ))
        states.append(GateState(
            name="tape_imbalance",
            label="Buy ratio",
            passing=gates.ratio_ok,
            value=gates.buy_ratio,
            threshold=SIM_PRESSURE_BUY_RATIO_MIN,
            direction=">=",
        ))
        states.append(GateState(
            name="tape_vol_flow",
            label="Volume flow",
            passing=gates.vol_flow_ok,
            value=gates.vol_imbalance,
            threshold=SIM_PRESSURE_VOL_IMBALANCE_MIN,
            direction=">",
        ))
        states.append(GateState(
            name="tape_acceleration",
            label="Acceleration z",
            passing=gates.accel_ok,
            value=gates.accel_z,
            threshold=SIM_PRESSURE_ACCEL_Z,
            direction=">=",
            detail="" if gates.accel_ready else "warming up",
        ))
        # Progress is displayed in PERCENT (x100) — the fraction 0.005
        # renders as a misleading "0.01" under 2dp display formatting.
        # Display-only: the gate decision reads the fraction vs the config
        # constant.
        states.append(GateState(
            name="tape_progress",
            label="Price progress",
            passing=gates.progress_ok,
            value=None if gates.progress is None else round(gates.progress * 100, 2),
            threshold=round(SIM_PRESSURE_PROGRESS_MIN * 100, 3),
            direction=">",
            detail="" if gates.progress is not None else "warming up",
        ))

    # Instantaneous OFI floor: legacy stack, extended zone, AND at the
    # armed-zone trigger (the book must be imbalancing NOW — mirrors
    # evaluate_entry's book-confirmation-at-trigger exactly). NOT a gate in
    # the posterior regime — there the OFI read is a weighted feature of
    # the score, so no state is emitted at all.
    if _config.SIM_ENTRY_REGIME == "posterior":
        pass
    elif ofi_tracker.depth_active:
        states.append(GateState(
            name="ofi_norm",
            label="OFI normalized",
            passing=ofi_tracker.ofi_gate_ok(),
            value=ofi_tracker.ofi_norm_gate,
            threshold=SIM_OFI_NORM_MIN,
            direction=">=",
        ))
    else:
        states.append(GateState(
            name="ofi_norm",
            label="OFI normalized",
            passing=True,
            value=None,
            threshold=SIM_OFI_NORM_MIN,
            direction=">=",
            active=False,
            detail="no L2 depth",
        ))

    if rmap is not None and last_atr > 0:
        nearest = rmap.nearest_above(resistance_search_floor(last_price, setup_level))
        if nearest is not None:
            headroom_atrs = (nearest.price - last_price) / last_atr
            states.append(GateState(
                name="headroom",
                label="Resistance headroom",
                passing=headroom_atrs >= EXEC_DIST_TO_RESISTANCE_MULT,
                value=headroom_atrs,
                threshold=EXEC_DIST_TO_RESISTANCE_MULT,
                direction=">=",
                detail=f"next {nearest.price:.4g}",
            ))
        else:
            states.append(GateState(
                name="headroom",
                label="Resistance headroom",
                passing=True,
                value=None,
                threshold=EXEC_DIST_TO_RESISTANCE_MULT,
                direction=">=",
                active=False,
                detail="no resistance above",
            ))
    else:
        states.append(GateState(
            name="headroom",
            label="Resistance headroom",
            passing=True,
            value=None,
            threshold=EXEC_DIST_TO_RESISTANCE_MULT,
            direction=">=",
            active=False,
            detail="no ATR/levels",
        ))

    return states


# ---------------------------------------------------------------------------
# Exit signal evaluation
# ---------------------------------------------------------------------------

def evaluate_exit(
    entry: "WatchlistEntry",
    ofi_tracker: "OrderFlowTracker",
    last_price: float,
    entry_price: float,
    last_atr: float,
    rmap: Optional["ResistanceMap"] = None,
    entry_setup_level: Optional[float] = None,
) -> ExitSignal:
    """
    Evaluate whether conditions are met for a simulated sell exit.

    Exit triggers (ANY one is sufficient):
      1. Selling pressure dominates buying pressure (flow reversal)
      2. OFI strongly negative AND buy dominance lapsed (depth feed active)
    3. Eval/live confidence drops below SIM_EXIT_CONFIDENCE_MIN
      4. Approaching resistance + tuned flow reads acknowledge (confirming)

    ``entry_setup_level`` is the setup level captured at fill time (the
    engine passes it from the position): exit 4 exempts the trade's own
    breakout ceiling exactly like the entry headroom gate does. Falls back
    to the entry's CURRENT setup level when not provided (legacy callers).

    Returns ExitSignal with triggered=True and reason string.
    """
    sig = ExitSignal()

    buy_p = ofi_tracker.buying_pressure
    sell_p = ofi_tracker.selling_pressure
    sig.net_pressure = buy_p - sell_p
    sig.ofi = ofi_tracker.ofi_rolling
    sig.eval_confidence = entry.eval_confidence
    sig.live_confidence = entry.live_confidence

    # --- Exit 1: Selling pressure dominance (ratio-based) ---
    total_activity = buy_p + sell_p
    if total_activity >= SIM_PRESSURE_ACTIVITY_MIN:
        sell_ratio = sell_p / total_activity
        if sell_ratio >= SIM_EXIT_SELL_RATIO_MIN:
            sig.triggered = True
            sig.reason = (
                f"pressure_reversal: {buy_p}B/{sell_p}S ({sell_ratio:.0%} sell)"
            )
            return sig

    # --- Exit 1b: Acceleration collapse — buying momentum fading ---
    # Fires when the selected acceleration estimator drops below threshold
    # (tape is decelerating) AND buy dominance has genuinely lapsed. Reads
    # the SAME estimator as the entry ratio gate (ratio_gate), banded below
    # it (enter >= 0.60, collapse-exit <= 0.50): a true hysteresis loop, so
    # the entry and exit views can never contradict each other (F6). The
    # old boxcar <= 0.60 read shared the entry boundary on a different,
    # staler estimator — half its exits fired within 5s of entry.
    if ofi_tracker.accel_gate_ready:
        accel = ofi_tracker.accel_gate_raw
        buy_ratio = ofi_tracker.ratio_gate
        if accel <= SIM_EXIT_ACCEL_MAX and buy_ratio <= SIM_EXIT_ACCEL_BUY_RATIO_MAX:
            sig.triggered = True
            sig.reason = (
                f"accel_collapse: {accel:.2f} <= {SIM_EXIT_ACCEL_MAX} "
                f"and buy_ratio {buy_ratio:.0%} <= {SIM_EXIT_ACCEL_BUY_RATIO_MAX:.0%}"
            )
            return sig

    # --- Exit 2: OFI strongly negative (live L2 depth only) ---
    # Adverse book flow alone is NOT enough: a flow dip during an ongoing
    # squeeze is noise while the tape is still buy-dominant. Require buy
    # dominance to have lapsed too (ratio_gate <= 0.50 — the same band the
    # accel-collapse exit uses, hysteresis under the 0.60 entry floor).
    # Measured on 259 entered legs >= 6%: unconditional boxcar < -0.6 exited
    # mid-squeeze (>= 1.5% still ahead) 30.5% of the time; EMA < -0.75 with
    # the ratio-lapse condition 23.6%, median standalone capture 1.74% vs
    # 1.27%, same post-peak reaction (~23s).
    if ofi_tracker.depth_active:
        ofi_norm = ofi_tracker.ofi_norm_gate
        if (
            ofi_norm < SIM_EXIT_OFI_NORM_MAX
            and ofi_tracker.ratio_gate <= SIM_EXIT_ACCEL_BUY_RATIO_MAX
        ):
            sig.triggered = True
            sig.reason = (
                f"ofi_negative: norm {ofi_norm:.2f} < {SIM_EXIT_OFI_NORM_MAX} "
                f"and buy_ratio {ofi_tracker.ratio_gate:.0%} <= "
                f"{SIM_EXIT_ACCEL_BUY_RATIO_MAX:.0%} (raw {sig.ofi})"
            )
            return sig

    # --- Exit 3: Confidence threshold ---
    import config as _config
    if not _config.confidence_bypass_eval() and sig.eval_confidence < SIM_EXIT_CONFIDENCE_MIN:
        sig.triggered = True
        sig.reason = (
            f"eval_confidence_drop: {sig.eval_confidence:.1f} < "
            f"{SIM_EXIT_CONFIDENCE_MIN}"
        )
        return sig
    if (
        not _config.confidence_bypass_live()
        and entry.has_live_data
        and sig.live_confidence < SIM_EXIT_CONFIDENCE_MIN
    ):
        sig.triggered = True
        sig.reason = (
            f"live_confidence_drop: {sig.live_confidence:.1f} < "
            f"{SIM_EXIT_CONFIDENCE_MIN}"
        )
        return sig

    # --- Exit 4: Resistance proximity + flow acknowledgment ---
    # Measures headroom with the SAME own-ceiling exemption as the entry
    # gate (resistance_search_floor): a trade designed to break its setup
    # level must not be exited for "approaching" that level or the
    # micro-levels inside its own entry band — including the dynamic
    # EMA9/EMA20/VWAP levels a flag pulls back onto. Pre-fix this fired
    # 159ms after a YXT entry on a 2-cent "headroom" to the 1m EMA9 while
    # deep OFI read +26k shares of net buying.
    # Flow acknowledgment reads the tuned gate sources (ratio_gate /
    # ofi_norm_gate) — the old net_pressure<=0 (10s boxcar counts) and raw
    # ofi_rolling<0 terms flip on momentary sell flurries the fast-EMA and
    # depth-normalized reads correctly see through.
    if rmap is not None and last_atr > 0:
        setup_level = entry_setup_level
        if setup_level is None:
            setup_level = getattr(entry, "setup_price_level", None)
        nearest = rmap.nearest_above(resistance_search_floor(last_price, setup_level))
        if nearest is not None:
            headroom = nearest.price - last_price
            if headroom <= SIM_EXIT_RESISTANCE_ATR * last_atr:
                # Near a genuinely distinct wall — exit only if the tuned
                # flow reads acknowledge weakness.
                ratio = ofi_tracker.ratio_gate
                ofi_norm = ofi_tracker.ofi_norm_gate if ofi_tracker.depth_active else None
                flow_weakening = (
                    ratio <= SIM_EXIT_ACCEL_BUY_RATIO_MAX
                    or (ofi_norm is not None and ofi_norm < 0.0)
                )
                if flow_weakening:
                    ofi_str = f"{ofi_norm:+.2f}" if ofi_norm is not None else "n/a"
                    sig.triggered = True
                    sig.reason = (
                        f"resistance_approach: ${nearest.price:.2f} "
                        f"(headroom=${headroom:.2f}, "
                        f"{headroom/last_atr:.1f} ATR) + flow weakening "
                        f"(ratio {ratio:.0%}, ofi {ofi_str})"
                    )
                    return sig

    return sig
