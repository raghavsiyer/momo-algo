"""
tests/test_files/test_sim_engine.py

Unit tests for exec/engine.py, exec/signals.py, and exec/sim_logger.py.

All external APIs are mocked — no network calls.
Tests inject synthetic state and verify signal evaluation + fill simulation.
"""

import csv
import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")
os.environ.setdefault("SIM_ENABLED", "true")

import tempfile
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from strategy.evaluator import WatchlistEntry, CompositeEvaluation
from strategy.scoring import build_groups, VariableGroup
from strategy.criteria import build_criteria_groups
from config import (
    MarketClock,
    SCORING_GROUPS,
    CRITERIA_EXECUTION,
    SETUP_PATTERN_GROUP,
    PRESSURE_WINDOW_SECS,

    SIM_EXIT_ACCEL_MAX,
    SIM_EXIT_ACCEL_BUY_RATIO_MAX,
    SIM_MAX_SETUP_EXTENSION_PCT,
    SIM_SETUP_ANTICIPATION_PCT,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_SETUP_SCORE_MIN,
)
from exec.order_flow import OrderFlowTracker
from exec.signals import evaluate_entry, evaluate_exit, resistance_search_floor, EntrySignal, ExitSignal
from exec.engine import SimulationEngine, SimPosition
from exec.sim_logger import SimTradeLogger, SimFill
from technical_analysis.resistance import ResistanceMap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_evaluation(quality: float = 0.85) -> CompositeEvaluation:
    """Build a CompositeEvaluation with all variables at given quality."""
    groups = build_groups(SCORING_GROUPS)
    for g in groups:
        for v in g.variables:
            v.quality = quality
            v.value = quality
    gmap = {g.name: g for g in groups}
    stock_eval = gmap["STOCK_EVAL"]
    stock_eval.get("float_shares").value = 5.0
    stock_eval.get("rvol_1day").value = 10.0
    stock_eval.get("price").value = 5.0
    stock_eval.get("daily_trend").value = 25.0
    return CompositeEvaluation(
        stock_eval=stock_eval,
        technical_analysis=gmap["TECHNICAL_ANALYSIS"],
    )


def _make_entry(
    price: float = 5.00,
    quality: float = 0.85,
) -> WatchlistEntry:
    ev = _make_evaluation(quality=quality)
    entry = WatchlistEntry(
        symbol="SIM",
        evaluated_at=datetime.now(tz=timezone.utc),
        price=price,
        change_pct=20.0,
        volume=500_000,
        avg_volume=100_000,
        rvol_1day=5.0,
        rvol_5min=7.0,
        daily_high=price + 0.50,
        float_shares_m=2.0,
        free_float_pct=80.0,
        sector="Technology",
        industry="Software",
        country="US",
        ipo_date=None,
        days_since_ipo=None,
        has_recent_reverse_split=False,
        reverse_split_ratio=None,
        reverse_split_date=None,
        news_count_last_hour=3,
        news_is_fresh=True,
        news_is_breaking=True,
        avg_sentiment=0.4,
        evaluation=ev,
    )
    # Set up criteria groups (execution gates)
    entry.criteria_groups = build_criteria_groups(CRITERIA_EXECUTION)

    # Set up the setup_pattern channel
    setup_pattern = build_groups([SETUP_PATTERN_GROUP])[0]
    for v in setup_pattern.variables:
        v.reset(0.0)
    entry.setup_pattern = setup_pattern

    return entry


def _make_entry_allowed(entry: WatchlistEntry) -> None:
    """Force all execution criteria to pass."""
    for g in entry.criteria_groups:
        if g.name == "EXECUTION":
            for c in g.criteria:
                c.update(True, 1.0)


def _make_setup_active(entry: WatchlistEntry, score: float = 60.0) -> None:
    """Simulate an active setup pattern with a given score."""
    entry.setup_pattern_name = "pullback_new_high"
    entry.setup_pattern_timeframe = "1min"
    entry.setup_pullback_depth = 0.3
    # A real level is now REQUIRED for entry (no band -> no trade). Default
    # to the entry's own price so the band centers on the test's tape;
    # tests exercising band edges override it explicitly.
    entry.setup_price_level = entry.price
    # Set the quality on the setup group so .score reflects score/100
    if entry.setup_pattern is not None:
        for v in entry.setup_pattern.variables:
            v.quality = score / 100.0
            v.value = 0.5


def _make_ofi_tracker(
    buy_prints: int = 80,
    sell_prints: int = 30,
    ofi: int = 100,
    accelerating: bool = True,
) -> OrderFlowTracker:
    """Create an OrderFlowTracker with pre-loaded state.

    If accelerating=True, buys are concentrated in the recent half of the
    pressure window (simulates buyers accelerating).
    """
    tracker = OrderFlowTracker()
    now = datetime.now(tz=timezone.utc)
    half_window = PRESSURE_WINDOW_SECS / 2
    if accelerating:
        # Pack buys into the recent half of the configured pressure window.
        for i in reversed(range(buy_prints)):
            tracker._buy_prints.append(now - timedelta(seconds=i * ((half_window - 0.1) / max(buy_prints, 1))))
        # Spread sells across the older half.
        for i in reversed(range(sell_prints)):
            tracker._sell_prints.append(now - timedelta(seconds=half_window + i * ((half_window - 0.2) / max(sell_prints, 1))))
    else:
        # Uniform distribution across window
        for i in reversed(range(buy_prints)):
            tracker._buy_prints.append(now - timedelta(seconds=i * ((PRESSURE_WINDOW_SECS - 0.2) / max(buy_prints, 1))))
        for i in reversed(range(sell_prints)):
            tracker._sell_prints.append(now - timedelta(seconds=i * ((PRESSURE_WINDOW_SECS - 0.2) / max(sell_prints, 1))))
    # Mirror real ingestion: every directional print is also recorded in the
    # baseline deque, which backs activity_count.
    tracker._baseline_prints.extend(sorted(list(tracker._buy_prints) + list(tracker._sell_prints)))
    # Rebuild the EMA accumulators from the seeded prints (fixtures bypass
    # on_timesale, so the EMA estimator must be reseeded like the backtest does).
    tracker.reseed_ema_from_prints(now=now)
    # Seed the progress ring with a gently rising tape (~+2% over 20s) so the
    # price-progress gate passes by default; tests exercising the gate itself
    # overwrite _progress_samples.
    for k in range(21):
        tracker._progress_samples.append(
            (now - timedelta(seconds=20 - k), 4.90 + 0.0075 * k)
        )
    return tracker


def _make_pressure_halves(
    *,
    older_buys: int,
    older_sells: int,
    recent_buys: int,
    recent_sells: int,
) -> OrderFlowTracker:
    """Create a tracker with explicit older/recent pressure halves."""
    tracker = OrderFlowTracker()
    now = datetime.now(tz=timezone.utc)
    older_ts = now - timedelta(seconds=8)
    recent_ts = now - timedelta(seconds=2)
    for _ in range(older_buys):
        tracker._buy_prints.append(older_ts)
    for _ in range(older_sells):
        tracker._sell_prints.append(older_ts)
    for _ in range(recent_buys):
        tracker._buy_prints.append(recent_ts)
    for _ in range(recent_sells):
        tracker._sell_prints.append(recent_ts)
    tracker._baseline_prints.extend(sorted(list(tracker._buy_prints) + list(tracker._sell_prints)))
    tracker.reseed_ema_from_prints(now=now)
    return tracker


# ---------------------------------------------------------------------------
# Tests: Entry signals
# ---------------------------------------------------------------------------

class TestEntrySignals:
    def test_entry_triggered_when_all_conditions_met(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is True
        total = 500 + 120
        assert 500 / total >= SIM_PRESSURE_BUY_RATIO_MIN
        assert len(sig.reasons) > 0

    def test_entry_blocked_when_criteria_fail(self):
        entry = _make_entry()
        # Don't call _make_entry_allowed → criteria fail
        _make_setup_active(entry, score=60.0)
        ofi = _make_ofi_tracker(buy_prints=80, sell_prints=30)

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is False

    def test_entry_blocked_when_no_setup(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        # Don't activate setup
        ofi = _make_ofi_tracker(buy_prints=80, sell_prints=30)

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is False

    def test_entry_blocked_when_pressure_insufficient(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        # Equal pressure → 50% buy ratio < SIM_PRESSURE_BUY_RATIO_MIN
        ofi = _make_ofi_tracker(buy_prints=55, sell_prints=55, accelerating=False)

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is False

    def test_entry_blocked_on_volume_flow_divergence(self):
        """Count-bullish tape with 30s dollar flow heavily sell-dominated
        (distribution) must be blocked by the volume-flow guard."""
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)
        # Many small buys, few huge sells: volume imbalance -0.33 <= -0.20
        ofi._buy_vol = 50_000
        ofi._sell_vol = 100_000

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is False

    def test_entry_blocked_until_acceleration_ready(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        # Too few prints for EITHER estimator to be ready (2bin: halves < 5;
        # ema: Kish n_eff < TAPE_ACCEL_EMA_MIN_NEFF).
        ofi = _make_pressure_halves(
            older_buys=3,
            older_sells=0,
            recent_buys=3,
            recent_sells=0,
        )

        assert ofi.accel_gate_ready is False
        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is False

    def test_entry_blocked_when_setup_extension_chased(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        entry.setup_price_level = 18.80
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=entry.setup_price_level * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT) + 0.01,
            last_atr=0.10,
        )
        assert sig.triggered is False

    def test_entry_allows_price_at_setup_level(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        entry.setup_price_level = 18.80
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=entry.setup_price_level,
            last_atr=0.10,
        )
        assert sig.triggered is True

    def test_entry_allows_anticipation_below_setup_level(self):
        # Tape confirmation slightly UNDER the trigger enters pre-break:
        # buyers absorbing the dip are the confirmation, not the price cross.
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        entry.setup_price_level = 18.80
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=entry.setup_price_level * (1.0 - SIM_SETUP_ANTICIPATION_PCT) + 0.01,
            last_atr=0.10,
        )
        assert sig.triggered is True

    def test_entry_blocked_below_anticipation_floor(self):
        # Deep in the pullback — under the anticipation band — stays blocked
        # no matter how good the tape looks.
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        entry.setup_price_level = 18.80
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=entry.setup_price_level * (1.0 - SIM_SETUP_ANTICIPATION_PCT) - 0.01,
            last_atr=0.10,
        )
        assert sig.triggered is False

    def test_entry_blocked_near_resistance(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        ofi = _make_ofi_tracker(buy_prints=80, sell_prints=30)

        # Build rmap with resistance very close (within EXIT_RESISTANCE_ATR * ATR)
        rmap = ResistanceMap(symbol="SIM", current_price=5.00)
        rmap.build_static()  # Will add $5.00, $5.50 levels

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.48,  # Very close to $5.50 resistance
            last_atr=0.10,
            rmap=rmap,
        )
        assert sig.triggered is False

    def test_entry_not_vetoed_by_setup_own_prior_high(self):
        # Same tape and rmap as test_entry_blocked_near_resistance, but the
        # $5.50 "resistance" is now inside the active setup's entry band
        # (level 5.45, band top 5.45*1.02=5.559) — it is the high being
        # broken, not a wall. Headroom is measured to the next level ($6.00).
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        entry.setup_price_level = 5.45
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)

        rmap = ResistanceMap(symbol="SIM", current_price=5.00)
        rmap.build_static()

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.48,
            last_atr=0.10,   # $6.00 headroom 0.52 > 0.10 → pass
            rmap=rmap,
        )
        assert sig.triggered is True

    def test_entry_still_blocked_by_wall_above_the_band(self):
        # A genuinely distinct level above the entry band still vetoes:
        # same setup as above but ATR large enough that the $6.00 level
        # (0.52 above price) is within 1 ATR.
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        entry.setup_price_level = 5.45
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)

        rmap = ResistanceMap(symbol="SIM", current_price=5.00)
        rmap.build_static()

        sig = evaluate_entry(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.48,
            last_atr=0.60,   # $6.00 headroom 0.52 < 0.60 → blocked
            rmap=rmap,
        )
        assert sig.triggered is False

    def test_resistance_search_floor_semantics(self):
        # No active setup level → search from last price (legacy).
        assert resistance_search_floor(5.48, None) == 5.48
        assert resistance_search_floor(5.48, 0.0) == 5.48
        # Active setup → search from the top of the approved entry band.
        assert resistance_search_floor(5.48, 5.45) == pytest.approx(
            5.45 * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT)
        )
        # Price already above the band top → search from price. Derived from
        # the constant, not a literal: the band top moves when it is retuned.
        above_band = 5.45 * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT) + 0.01
        assert resistance_search_floor(above_band, 5.45) == above_band


# ---------------------------------------------------------------------------
# Tests: Exit signals
# ---------------------------------------------------------------------------

class TestExitSignals:
    def test_exit_on_pressure_reversal(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        # Selling pressure dominates (76% sell ratio, well above 60% threshold)
        ofi = _make_ofi_tracker(buy_prints=25, sell_prints=80, accelerating=False)

        sig = evaluate_exit(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.10,
            entry_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is True
        assert "pressure_reversal" in sig.reason

    def test_exit_on_confidence_drop(self):
        entry = _make_entry(quality=0.1)  # Very low → confidence well below 45
        ofi = _make_ofi_tracker(buy_prints=10, sell_prints=2)  # Buys dominate → no pressure exit

        with patch("exec.signals.SIM_EXIT_CONFIDENCE_MIN", 45.0):
            sig = evaluate_exit(
                entry=entry,
                ofi_tracker=ofi,
                last_price=5.10,
                entry_price=5.00,
                last_atr=0.10,
            )
        assert sig.triggered is True
        assert "confidence_drop" in sig.reason

    def test_no_exit_when_pressure_balanced(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        # Balanced pressure — 60% buy, not enough sell to exit
        ofi = _make_ofi_tracker(buy_prints=65, sell_prints=45)

        sig = evaluate_exit(
            entry=entry,
            ofi_tracker=ofi,
            last_price=5.10,
            entry_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is False

    def test_accel_collapse_does_not_exit_while_buyers_dominate(self):
        """Accel collapse must NOT fire while ratio_gate > threshold (buyers in control)."""
        entry = _make_entry()
        _make_entry_allowed(entry)
        tracker = OrderFlowTracker()
        now = datetime.now(tz=timezone.utc)
        # Older half (7-8.9s ago): 24 buys, 6 sells; recent (2-3.6s ago):
        # 20 buys, 10 sells — buyers clearly dominate the fast kernel too.
        for i in range(24):
            tracker._buy_prints.append(now - timedelta(seconds=7.0 + i * 0.08))
        for i in range(6):
            tracker._sell_prints.append(now - timedelta(seconds=7.0 + i * 0.08))
        for i in range(20):
            tracker._buy_prints.append(now - timedelta(seconds=2.0 + i * 0.08))
        for i in range(10):
            tracker._sell_prints.append(now - timedelta(seconds=2.0 + i * 0.08))
        tracker._baseline_prints.extend(sorted(list(tracker._buy_prints) + list(tracker._sell_prints)))
        tracker.reseed_ema_from_prints(now=now)

        # Buyers dominate on the exit-band read (F6: gate reads ratio_gate)
        assert tracker.ratio_gate > SIM_EXIT_ACCEL_BUY_RATIO_MAX
        assert tracker.pressure_acceleration_ready

        sig = evaluate_exit(
            entry=entry,
            ofi_tracker=tracker,
            last_price=5.10,
            entry_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is False or "accel_collapse" not in (sig.reason or "")

    def test_accel_collapse_holds_inside_hysteresis_band(self):
        """F6 band: decelerating tape with ratio in (0.50, 0.60] must NOT exit.

        The old boxcar <= 0.60 read shared the entry gate's boundary and
        fired here — this locks the 0.60-in / 0.50-out hysteresis loop.
        """
        entry = _make_entry()
        _make_entry_allowed(entry)
        tracker = OrderFlowTracker()
        now = datetime.now(tz=timezone.utc)
        # Steady two-sided prints until 3s ago then nothing (decelerating),
        # buy-tilted but no longer entry-grade: ratio in the band.
        for i in range(29):
            tracker._buy_prints.append(now - timedelta(seconds=3.0 + i * (7.0 / 29)))
        for i in range(23):
            tracker._sell_prints.append(now - timedelta(seconds=3.0 + i * (6.5 / 23)))
        tracker._baseline_prints.extend(sorted(list(tracker._buy_prints) + list(tracker._sell_prints)))
        tracker.reseed_ema_from_prints(now=now)

        assert tracker.accel_gate_ready
        assert tracker.accel_gate_raw <= SIM_EXIT_ACCEL_MAX
        assert SIM_EXIT_ACCEL_BUY_RATIO_MAX < tracker.ratio_gate <= 0.60
        # old knife-edge read would have fired: boxcar <= 0.60
        assert tracker.buy_ratio <= 0.60

        sig = evaluate_exit(
            entry=entry,
            ofi_tracker=tracker,
            last_price=5.10,
            entry_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is False or "accel_collapse" not in (sig.reason or "")

    def test_accel_collapse_exits_when_buy_ratio_fades(self):
        """Accel collapse fires when the tape is decaying AND dominance is lost."""
        entry = _make_entry()
        _make_entry_allowed(entry)
        tracker = OrderFlowTracker()
        now = datetime.now(tz=timezone.utc)
        # Genuine tape death: steady prints from 10s ago until 3s ago, then
        # NOTHING in the last 3s.  The fast intensity kernel decays toward
        # zero while the slow kernel still remembers the active phase — the
        # estimator reads negative acceleration.  Sellers outnumber buyers
        # (24B/28S), so the fast-kernel ratio is below the 0.50 band floor
        # while the boxcar sell share (0.538) stays under exit-1's 0.60.
        for i in range(24):
            tracker._buy_prints.append(now - timedelta(seconds=3.0 + i * (7.0 / 24)))
        for i in range(28):
            tracker._sell_prints.append(now - timedelta(seconds=3.0 + i * (6.5 / 28)))
        tracker._baseline_prints.extend(sorted(list(tracker._buy_prints) + list(tracker._sell_prints)))
        tracker.reseed_ema_from_prints(now=now)

        # Verify preconditions through the estimator-selected gates
        assert tracker.ratio_gate <= SIM_EXIT_ACCEL_BUY_RATIO_MAX
        assert tracker.accel_gate_ready
        assert tracker.accel_gate_raw <= SIM_EXIT_ACCEL_MAX

        sig = evaluate_exit(
            entry=entry,
            ofi_tracker=tracker,
            last_price=5.10,
            entry_price=5.00,
            last_atr=0.10,
        )
        assert sig.triggered is True
        assert "accel_collapse" in sig.reason


# ---------------------------------------------------------------------------
# Tests: SimulationEngine
# ---------------------------------------------------------------------------

class TestSimulationEngine:
    def setup_method(self):
        """Reset shared class state between tests."""
        SimulationEngine._positions.clear()
        SimulationEngine._trade_counts.clear()
        SimulationEngine._last_trade_ts.clear()
        # Use a fresh temp-dir-backed logger
        SimulationEngine._logger = SimTradeLogger(log_dir=tempfile.mkdtemp())
        # Disable latency for unit tests — we test signal logic, not fill timing
        self._latency_patch = patch("exec.engine.SIM_FILL_LATENCY_MS", 0)
        self._latency_patch.start()

    def teardown_method(self):
        self._latency_patch.stop()

    def _make_engine(self) -> tuple[SimulationEngine, WatchlistEntry, OrderFlowTracker]:
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)
        rmap = ResistanceMap(symbol="SIM", current_price=5.00)
        rmap.build_static()

        engine = SimulationEngine(
            symbol="SIM",
            entry=entry,
            ofi_tracker=ofi,
            rmap=rmap,
        )
        return engine, entry, ofi

    @staticmethod
    def _tick_through_dwell(engine, **kwargs) -> None:
        """Drive on_tick twice, spanning SIM_ENTRY/EXIT_DWELL_SECS.

        The signal dwell requires a triggered signal to hold across at least
        two prints, so a single on_tick can no longer fill. These tests assert
        signal LOGIC, not dwell timing (which has dedicated tests below), so
        they feed an arming print and a second print past the hold. The
        confirming print is bumped one cent so the entry dwell's up-move
        requirement (mid strictly above the arm anchor) is satisfied; the
        exit dwell is indicator-only, so the bump is inert for sell callers.
        """
        from config import SIM_ENTRY_DWELL_SECS, SIM_EXIT_DWELL_SECS

        span = max(SIM_ENTRY_DWELL_SECS, SIM_EXIT_DWELL_SECS)
        t0 = datetime.now(tz=timezone.utc)
        engine.on_tick(event_ts=t0, **kwargs)
        confirm = dict(kwargs)
        for k in ("price", "bid", "ask"):
            confirm[k] = round(confirm[k] + 0.01, 4)
        engine.on_tick(event_ts=t0 + timedelta(seconds=span), **confirm)

    def test_buy_triggered_on_tick(self):
        engine, entry, ofi = self._make_engine()
        self._tick_through_dwell(engine, price=5.05, bid=5.04, ask=5.06, last_atr=0.10)

        assert "SIM" in SimulationEngine._positions
        pos = SimulationEngine._positions["SIM"]
        assert pos.shares > 0
        assert pos.entry_price >= 5.06  # filled at ask + slippage

    def test_sell_triggered_on_pressure_reversal(self):
        engine, entry, ofi = self._make_engine()
        # First: trigger entry
        self._tick_through_dwell(engine, price=5.05, bid=5.04, ask=5.06, last_atr=0.10)
        assert "SIM" in SimulationEngine._positions

        # Now flip pressure: heavy sells, few buys (>60% sell ratio within pressure window)
        ofi._buy_prints.clear()
        ofi._sell_prints.clear()
        now = datetime.now(tz=timezone.utc)
        for i in range(80):
            ofi._sell_prints.append(now - timedelta(seconds=i * 0.02))
        for i in range(20):
            ofi._buy_prints.append(now - timedelta(seconds=i * 0.02))

        self._tick_through_dwell(engine, price=5.08, bid=5.07, ask=5.09, last_atr=0.10)
        assert "SIM" not in SimulationEngine._positions

    def test_max_concurrent_positions_enforced(self):
        """Only SIM_MAX_CONCURRENT positions can be open at once."""
        from config import SIM_MAX_CONCURRENT

        for i in range(SIM_MAX_CONCURRENT + 2):
            sym = f"T{i}"
            entry = _make_entry()
            entry.symbol = sym
            _make_entry_allowed(entry)
            _make_setup_active(entry, score=60.0)
            ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)

            engine = SimulationEngine(
                symbol=sym, entry=entry, ofi_tracker=ofi,
            )
            self._tick_through_dwell(engine, price=5.05, bid=5.04, ask=5.06, last_atr=0.10)

        assert len(SimulationEngine._positions) == SIM_MAX_CONCURRENT

    def test_cooldown_enforced(self):
        engine, entry, ofi = self._make_engine()
        # Entry 1
        self._tick_through_dwell(engine, price=5.05, bid=5.04, ask=5.06, last_atr=0.10)
        assert "SIM" in SimulationEngine._positions

        # Force exit
        engine.stop()
        assert "SIM" not in SimulationEngine._positions

        # Immediate re-entry should be blocked by cooldown
        from unittest.mock import patch
        with patch("exec.engine.SIM_COOLDOWN_SECS", 60):
            self._tick_through_dwell(engine, price=5.05, bid=5.04, ask=5.06, last_atr=0.10)
            assert "SIM" not in SimulationEngine._positions

    def test_position_size_respects_limits(self):
        engine, entry, ofi = self._make_engine()
        self._tick_through_dwell(engine, price=5.05, bid=5.04, ask=5.06, last_atr=0.10)

        pos = SimulationEngine._positions["SIM"]
        from config import SIM_ACCOUNT_SIZE, SIM_MAX_POSITION_PCT
        max_notional = SIM_ACCOUNT_SIZE * SIM_MAX_POSITION_PCT
        assert pos.shares * pos.entry_price <= max_notional * 1.01  # allow tiny rounding

    def test_depth_regime_telemetry_on_fills(self):
        """F8: fills record the OFI depth regime; depth_flips counts mid-hold
        transitions (incl. round trips invisible to entry/exit snapshots)."""
        from config import OFI_DEPTH_STALE_SECS
        from market_data.ibkr_stream import OrderBookLevel, OrderBookSnapshot

        engine, entry, ofi = self._make_engine()

        # Entry with no book — L1-only regime on the BUY row.
        self._tick_through_dwell(engine, price=5.05, bid=5.04, ask=5.06, last_atr=0.10)
        assert "SIM" in SimulationEngine._positions
        pos = SimulationEngine._positions["SIM"]
        assert pos.last_depth_active is False
        assert pos.depth_flips == 0

        # Book comes alive mid-hold (flip 1: L1 → L2).
        book_ts = datetime.now(tz=timezone.utc)
        ofi.on_book(OrderBookSnapshot(
            symbol="SIM",
            bids=[OrderBookLevel("SIM", "bid", 0, 5.04, 300, "", book_ts)],
            asks=[OrderBookLevel("SIM", "ask", 0, 5.06, 400, "", book_ts)],
            ts=book_ts,
        ))
        assert ofi.depth_active is True
        engine.on_tick(price=5.08, bid=5.07, ask=5.09, last_atr=0.10)  # tape still bullish — no exit
        assert "SIM" in SimulationEngine._positions
        assert pos.depth_flips == 1
        assert pos.last_depth_active is True

        # Book goes stale (flip 2: L2 → L1). Age the book relative to event
        # time — bumping event time forward would also age out the tape.
        ofi._last_book_ts = ofi._last_event_ts - timedelta(seconds=OFI_DEPTH_STALE_SECS + 1)
        assert ofi.depth_active is False
        engine.on_tick(price=5.08, bid=5.07, ask=5.09, last_atr=0.10)
        assert "SIM" in SimulationEngine._positions
        assert pos.depth_flips == 2

        # Flip the tape to heavy sells → exit 1 fires.
        ofi._buy_prints.clear()
        ofi._sell_prints.clear()
        now = datetime.now(tz=timezone.utc)
        for i in range(80):
            ofi._sell_prints.append(now - timedelta(seconds=i * 0.02))
        for i in range(20):
            ofi._buy_prints.append(now - timedelta(seconds=i * 0.02))
        self._tick_through_dwell(engine, price=5.08, bid=5.07, ask=5.09, last_atr=0.10)
        assert "SIM" not in SimulationEngine._positions

        fills = SimulationEngine._logger.fills
        assert [f.side for f in fills] == ["BUY", "SELL"]
        buy, sell = fills
        assert buy.depth_active is False
        assert buy.ofi_normalized == 0.0     # no depth samples at entry
        assert buy.book_depth_avg == 0.0
        assert buy.depth_flips == 0
        assert sell.depth_active is False     # stale again at exit
        assert sell.depth_flips == 2
        assert sell.book_depth_avg == 700.0   # one snapshot: 300 + 400


# ---------------------------------------------------------------------------
# Tests: signal dwell (confirmation hold)
# ---------------------------------------------------------------------------

class TestSignalDwell:
    """SIM_ENTRY_DWELL_SECS / SIM_EXIT_DWELL_SECS: a triggered signal must
    hold CONTINUOUSLY across prints before it fills."""

    DWELL = 0.25

    def setup_method(self):
        SimulationEngine._positions.clear()
        SimulationEngine._trade_counts.clear()
        SimulationEngine._last_trade_ts.clear()
        SimulationEngine._traded_signal_keys.clear()
        SimulationEngine._logger = SimTradeLogger(log_dir=tempfile.mkdtemp())
        self._patches = [
            patch("exec.engine.SIM_FILL_LATENCY_MS", 0),
            patch("exec.engine._ENTRY_DWELL", timedelta(seconds=self.DWELL)),
            patch("exec.engine._EXIT_DWELL", timedelta(seconds=self.DWELL)),
        ]
        for p in self._patches:
            p.start()

    def teardown_method(self):
        for p in self._patches:
            p.stop()

    def _make_engine(self):
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)
        engine = SimulationEngine(symbol="SIM", entry=entry, ofi_tracker=ofi)
        return engine, entry, ofi

    @staticmethod
    def _kill_entry(entry) -> None:
        """Break the entry chain without touching the tape.

        Clearing the setup level removes the entry band, which fails the chain
        outright (no level -> no trade); _make_setup_active restores it.
        """
        entry.setup_price_level = None

    def test_single_triggered_print_does_not_fill(self):
        engine, entry, ofi = self._make_engine()
        t0 = datetime.now(tz=timezone.utc)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10, event_ts=t0)
        assert "SIM" not in SimulationEngine._positions

    def test_fills_once_signal_holds_past_dwell(self):
        engine, entry, ofi = self._make_engine()
        t0 = datetime.now(tz=timezone.utc)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10, event_ts=t0)
        assert "SIM" not in SimulationEngine._positions

        # A print INSIDE the window still holds (flat mid: no reset).
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=self.DWELL / 2))
        assert "SIM" not in SimulationEngine._positions

        # First print at/past the hold with mid above the arm anchor fills.
        engine.on_tick(price=5.06, bid=5.05, ask=5.07, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=self.DWELL))
        assert "SIM" in SimulationEngine._positions

    def test_broken_signal_resets_the_hold(self):
        """The dwell measures CONTINUOUS confirmation: a failing print in the
        middle restarts the clock rather than being ignored."""
        engine, entry, ofi = self._make_engine()
        t0 = datetime.now(tz=timezone.utc)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10, event_ts=t0)

        # Signal dies mid-window.
        self._kill_entry(entry)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=self.DWELL / 2))
        assert "SIM" not in SimulationEngine._positions

        # Signal returns. Even well past the ORIGINAL arming time, this is a
        # fresh arm — it must not inherit the broken run's elapsed time.
        _make_setup_active(entry, score=60.0)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=self.DWELL * 4))
        assert "SIM" not in SimulationEngine._positions

        engine.on_tick(price=5.06, bid=5.05, ask=5.07, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=self.DWELL * 5))
        assert "SIM" in SimulationEngine._positions

    def _enter(self, engine) -> datetime:
        """Drive an entry through the dwell (arm, then up-tick confirm)."""
        t0 = datetime.now(tz=timezone.utc)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10, event_ts=t0)
        engine.on_tick(price=5.06, bid=5.05, ask=5.07, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=self.DWELL))
        assert "SIM" in SimulationEngine._positions
        return t0

    def test_exit_dwell_holds_then_sells(self):
        engine, entry, ofi = self._make_engine()
        t0 = self._enter(engine)

        # Flip the tape to seller dominance → exit 1 arms.
        ofi._buy_prints.clear()
        ofi._sell_prints.clear()
        flip = t0 + timedelta(seconds=1)
        for i in range(80):
            ofi._sell_prints.append(flip - timedelta(seconds=i * 0.02))
        for i in range(20):
            ofi._buy_prints.append(flip - timedelta(seconds=i * 0.02))

        engine.on_tick(price=5.08, bid=5.07, ask=5.09, last_atr=0.10, event_ts=flip)
        assert "SIM" in SimulationEngine._positions, "exit must not fire on first triggered print"

        engine.on_tick(price=5.08, bid=5.07, ask=5.09, last_atr=0.10,
                       event_ts=flip + timedelta(seconds=self.DWELL))
        assert "SIM" not in SimulationEngine._positions

    def test_force_exit_bypasses_dwell(self):
        """EOD / scorer-stop exits are not signal-driven and must fill at once."""
        engine, entry, ofi = self._make_engine()
        self._enter(engine)

        engine.stop()
        assert "SIM" not in SimulationEngine._positions

    def test_zero_dwell_restores_immediate_fill(self):
        """SIM_ENTRY_DWELL_SECS = 0 disables the hold entirely."""
        engine, entry, ofi = self._make_engine()
        with patch("exec.engine._ENTRY_DWELL", timedelta(0)):
            engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10,
                           event_ts=datetime.now(tz=timezone.utc))
        assert "SIM" in SimulationEngine._positions

    # -- up-move requirement (SIM_ENTRY_DWELL_UPMOVE) --

    def test_stalled_mid_extends_hold_until_uptick(self):
        """A mid pinned at the arm anchor never fills — the hold extends until
        price actually advances, however long past the dwell that is."""
        engine, entry, ofi = self._make_engine()
        t0 = datetime.now(tz=timezone.utc)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10, event_ts=t0)

        # Well past the dwell, mid unchanged → stall, no fill.
        for mult in (1, 2, 8):
            engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10,
                           event_ts=t0 + timedelta(seconds=self.DWELL * mult))
            assert "SIM" not in SimulationEngine._positions

        # First up-tick fills (hold already elapsed).
        engine.on_tick(price=5.06, bid=5.05, ask=5.07, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=self.DWELL * 9))
        assert "SIM" in SimulationEngine._positions

    def test_mid_decline_rearms_at_lower_anchor(self):
        """Mid below the anchor restarts the clock — the original schedule is
        void, and the fresh hold runs from the decline print's lower mid."""
        engine, entry, ofi = self._make_engine()
        t0 = datetime.now(tz=timezone.utc)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10, event_ts=t0)

        # Mid dips below the anchor mid-hold → re-arm at 5.04 mid.
        dip = t0 + timedelta(seconds=self.DWELL / 2)
        engine.on_tick(price=5.04, bid=5.03, ask=5.05, last_atr=0.10, event_ts=dip)
        assert "SIM" not in SimulationEngine._positions

        # The ORIGINAL dwell schedule must not fill: above the new anchor but
        # only half the fresh hold has elapsed.
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=self.DWELL))
        assert "SIM" not in SimulationEngine._positions

        # Fresh hold elapsed + mid above the re-armed anchor → fill.
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10,
                       event_ts=dip + timedelta(seconds=self.DWELL))
        assert "SIM" in SimulationEngine._positions

    def test_upmove_disabled_fills_on_flat_mid(self):
        """SIM_ENTRY_DWELL_UPMOVE = False restores the indicator-only hold."""
        engine, entry, ofi = self._make_engine()
        with patch("exec.engine.SIM_ENTRY_DWELL_UPMOVE", False):
            t0 = datetime.now(tz=timezone.utc)
            engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10, event_ts=t0)
            engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10,
                           event_ts=t0 + timedelta(seconds=self.DWELL))
        assert "SIM" in SimulationEngine._positions

    def test_exit_dwell_ignores_price_direction(self):
        """The exit hold is indicator-only: a RISING mid during the hold must
        not block a sell whose indicators keep confirming."""
        engine, entry, ofi = self._make_engine()
        t0 = self._enter(engine)

        # Seller dominance arms exit 1.
        ofi._buy_prints.clear()
        ofi._sell_prints.clear()
        flip = t0 + timedelta(seconds=1)
        for i in range(80):
            ofi._sell_prints.append(flip - timedelta(seconds=i * 0.02))
        for i in range(20):
            ofi._buy_prints.append(flip - timedelta(seconds=i * 0.02))

        engine.on_tick(price=5.08, bid=5.07, ask=5.09, last_atr=0.10, event_ts=flip)
        assert "SIM" in SimulationEngine._positions

        # Price ticks UP through the exit hold — sell still confirms.
        engine.on_tick(price=5.10, bid=5.09, ask=5.11, last_atr=0.10,
                       event_ts=flip + timedelta(seconds=self.DWELL))
        assert "SIM" not in SimulationEngine._positions


# ---------------------------------------------------------------------------
# Tests: SimTradeLogger
# ---------------------------------------------------------------------------

class TestSimTradeLogger:
    def test_log_fill_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("exec.sim_logger.SIM_LOG_DIR", tmpdir):
                lg = SimTradeLogger()
                fill = SimFill(
                    ts=datetime.now(tz=timezone.utc),
                    symbol="TEST",
                    side="BUY",
                    fill_price=5.06,
                    shares=100,
                    notional=506.0,
                    bid=5.04,
                    ask=5.06,
                    spread=0.02,
                    entry_reason="test entry",
                    eval_confidence=72.0,
                    live_confidence=70.0,
                    setup_name="pullback_new_high",
                    setup_score=65.0,
                    buying_pressure=15,
                    selling_pressure=3,
                    net_pressure=12,
                    ofi_rolling=100,
                )
                lg.log_fill(fill)
                assert lg.trade_count == 1
                lg.close()

    def test_session_pnl_calculation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("exec.sim_logger.SIM_LOG_DIR", tmpdir):
                lg = SimTradeLogger()
                now = datetime.now(tz=timezone.utc)
                # Simulate buy then sell with profit
                lg.log_fill(SimFill(
                    ts=now, symbol="X", side="BUY",
                    fill_price=5.0, shares=100, notional=500.0,
                    bid=4.99, ask=5.01, spread=0.02,
                ))
                lg.log_fill(SimFill(
                    ts=now + timedelta(seconds=30), symbol="X", side="SELL",
                    fill_price=5.20, shares=100, notional=520.0,
                    bid=5.19, ask=5.21, spread=0.02,
                    pnl_dollars=20.0, pnl_pct=0.04,
                ))
                assert lg.session_pnl == 20.0
                lg.close()

    def test_day_fills_loads_prior_process_fills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pullback_ts = datetime(2026, 6, 4, 12, 45, tzinfo=timezone.utc)
            entry_ts = datetime(2026, 6, 4, 12, 47, tzinfo=timezone.utc)
            day_str = entry_ts.astimezone(MarketClock.ET).strftime("%Y%m%d")

            lg = SimTradeLogger(log_dir=tmpdir)
            lg.log_fill(SimFill(
                ts=entry_ts,
                symbol="STI",
                side="BUY",
                fill_price=20.03,
                shares=124,
                notional=2483.72,
                bid=20.0,
                ask=20.02,
                spread=0.02,
                setup_name="pullback_new_high",
                setup_timeframe="10sec",
                setup_score=55.1,
                setup_pullback_ts=pullback_ts,
            ))
            lg.close()

            restarted = SimTradeLogger(log_dir=tmpdir)
            fills = restarted.day_fills(day_str)

            assert len(fills) == 1
            assert fills[0].symbol == "STI"
            assert fills[0].setup_timeframe == "10sec"
            assert fills[0].setup_pullback_ts == pullback_ts
            restarted.close()

    def test_depth_fields_round_trip(self):
        """F8: depth-regime telemetry survives the CSV write → day_fills read."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ts = datetime(2026, 6, 4, 12, 47, tzinfo=timezone.utc)
            day_str = ts.astimezone(MarketClock.ET).strftime("%Y%m%d")

            lg = SimTradeLogger(log_dir=tmpdir)
            lg.log_fill(SimFill(
                ts=ts, symbol="X", side="SELL",
                fill_price=5.0, shares=10, notional=50.0,
                bid=4.99, ask=5.01, spread=0.02,
                depth_active=True,
                ofi_normalized=-0.75,
                book_depth_avg=1234.5,
                depth_flips=3,
            ))
            lg.close()

            restarted = SimTradeLogger(log_dir=tmpdir)
            fills = restarted.day_fills(day_str)
            assert len(fills) == 1
            f = fills[0]
            assert f.depth_active is True
            assert f.ofi_normalized == -0.75
            assert f.book_depth_avg == 1234.5
            assert f.depth_flips == 3
            restarted.close()

    def test_legacy_csv_header_is_upgraded_before_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry_ts = datetime(2026, 6, 4, 12, 47, tzinfo=timezone.utc)
            later_ts = entry_ts + timedelta(minutes=1)
            pullback_ts = entry_ts - timedelta(minutes=2)
            day_str = entry_ts.astimezone(MarketClock.ET).strftime("%Y%m%d")
            csv_path = os.path.join(tmpdir, f"sim_trades_{day_str}.csv")
            legacy_fields = [
                "ts", "symbol", "side", "fill_price", "shares", "notional",
                "bid", "ask", "spread", "entry_reason", "exit_reason",
                "confidence", "setup_name", "setup_score", "buying_pressure",
                "selling_pressure", "net_pressure", "ofi_rolling",
                "pnl_dollars", "pnl_pct", "hold_secs",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=legacy_fields)
                writer.writeheader()
                writer.writerow({
                    "ts": entry_ts.isoformat(timespec="milliseconds"),
                    "symbol": "STI",
                    "side": "BUY",
                    "fill_price": 20.03,
                    "shares": 124,
                    "notional": 2483.72,
                    "bid": 20.0,
                    "ask": 20.02,
                    "spread": 0.02,
                    "confidence": 70.6,
                    "setup_name": "pullback_new_high",
                    "setup_score": 55.1,
                    "buying_pressure": 350,
                    "selling_pressure": 150,
                    "net_pressure": 200,
                    "ofi_rolling": 0,
                    "pnl_dollars": 0.0,
                    "pnl_pct": 0.0,
                    "hold_secs": 0.0,
                })

            lg = SimTradeLogger(log_dir=tmpdir)
            lg.log_fill(SimFill(
                ts=later_ts,
                symbol="STI",
                side="SELL",
                fill_price=20.01,
                shares=124,
                notional=2481.24,
                bid=20.02,
                ask=20.1,
                spread=0.08,
                exit_reason="test_exit",
                eval_confidence=72.0,
                live_confidence=70.6,
                setup_name="pullback_new_high",
                setup_timeframe="10sec",
                setup_score=55.1,
                setup_pullback_ts=pullback_ts,
                pnl_dollars=-2.48,
                pnl_pct=-0.001,
                hold_secs=4.3,
            ))
            lg.close()

            with open(csv_path, newline="", encoding="utf-8") as f:
                header = next(csv.reader(f))
            assert "setup_timeframe" in header
            assert "setup_pullback_ts" in header
            assert "eval_confidence" in header
            assert "live_confidence" in header

            restarted = SimTradeLogger(log_dir=tmpdir)
            fills = restarted.day_fills(day_str)
            assert len(fills) == 2
            assert fills[0].setup_score == 55.1
            assert fills[0].live_confidence == 70.6
            assert fills[1].setup_timeframe == "10sec"
            assert fills[1].setup_pullback_ts == pullback_ts
            restarted.close()


class TestThesisLifecycleExit:
    """SIM_EXIT_REGIME=thesis_lifecycle: structural stop, prove-it timeout,
    PROVEN sustained-lapse. Encodes the timing principle: an entry is a
    hypothesis about immediate follow-through."""

    def setup_method(self):
        SimulationEngine._positions.clear()
        SimulationEngine._trade_counts.clear()
        SimulationEngine._last_trade_ts.clear()
        SimulationEngine._traded_signal_keys.clear()
        SimulationEngine._logger = SimTradeLogger(log_dir=tempfile.mkdtemp())
        self._patches = [
            patch("exec.engine.SIM_FILL_LATENCY_MS", 0),
            patch("exec.engine.SIM_LATENCY_FILL", False),
            patch("exec.engine._ENTRY_DWELL", timedelta(0)),
            patch("exec.engine._EXIT_DWELL", timedelta(0)),
            patch("config.SIM_EXIT_REGIME", "thesis_lifecycle"),
        ]
        for p in self._patches:
            p.start()

    def teardown_method(self):
        for p in self._patches:
            p.stop()

    def _enter(self, level=None):
        entry = _make_entry()
        _make_entry_allowed(entry)
        _make_setup_active(entry, score=60.0)
        if level is not None:
            entry.setup_price_level = level
        ofi = _make_ofi_tracker(buy_prints=500, sell_prints=120)
        engine = SimulationEngine(symbol="SIM", entry=entry, ofi_tracker=ofi)
        t0 = datetime.now(tz=timezone.utc)
        engine.on_tick(price=5.05, bid=5.04, ask=5.06, last_atr=0.10, event_ts=t0)
        assert "SIM" in SimulationEngine._positions
        return engine, entry, ofi, t0, SimulationEngine._positions["SIM"]

    @staticmethod
    def _flip_tape_bearish(ofi, at):
        ofi._buy_prints.clear()
        ofi._sell_prints.clear()
        for i in range(80):
            ofi._sell_prints.append(at - timedelta(seconds=i * 0.02))
        for i in range(20):
            ofi._buy_prints.append(at - timedelta(seconds=i * 0.02))

    def test_at_level_fill_starts_prove_clock(self):
        engine, entry, ofi, t0, pos = self._enter()
        assert pos.break_ts is not None and not pos.proven

    def test_structural_stop_fires_instantly_unproven(self):
        engine, entry, ofi, t0, pos = self._enter()
        floor = pos.setup_band_floor
        assert floor is not None
        # Tape still bullish — only the structural stop can close this.
        engine.on_tick(price=floor * 0.99, bid=floor * 0.989, ask=floor * 0.991,
                       last_atr=0.10, event_ts=t0 + timedelta(seconds=2))
        assert "SIM" not in SimulationEngine._positions

    def test_prove_timeout_no_progress(self):
        # GLE shape: entered fine, never followed through, tape never
        # collapsed — the clock, not a meter, ends the hypothesis.
        engine, entry, ofi, t0, pos = self._enter()
        import config as _config
        mid = t0 + timedelta(seconds=_config.SIM_PROVE_PROGRESS_SECS / 2)
        engine.on_tick(price=5.04, bid=5.03, ask=5.05, last_atr=0.10, event_ts=mid)
        assert "SIM" in SimulationEngine._positions
        late = t0 + timedelta(seconds=_config.SIM_PROVE_PROGRESS_SECS + 1)
        engine.on_tick(price=5.04, bid=5.03, ask=5.05, last_atr=0.10, event_ts=late)
        assert "SIM" not in SimulationEngine._positions

    def test_proven_survives_brief_lapse_exits_sustained(self):
        engine, entry, ofi, t0, pos = self._enter()
        # Follow-through: price clears the prove target -> PROVEN.
        prove_tick = t0 + timedelta(seconds=3)
        engine.on_tick(price=5.20, bid=5.19, ask=5.21, last_atr=0.10, event_ts=prove_tick)
        assert pos.proven and "SIM" in SimulationEngine._positions

        # Flow flips: pressure_reversal triggers, but a PROVEN trade needs
        # the lapse SUSTAINED (SIM_EXIT_LAPSE_SUSTAIN_SECS), not one blip.
        t1 = prove_tick + timedelta(seconds=1)
        self._flip_tape_bearish(ofi, t1)
        engine.on_tick(price=5.20, bid=5.19, ask=5.21, last_atr=0.10, event_ts=t1)
        assert "SIM" in SimulationEngine._positions
        engine.on_tick(price=5.20, bid=5.19, ask=5.21, last_atr=0.10,
                       event_ts=t1 + timedelta(seconds=1.5))
        assert "SIM" in SimulationEngine._positions
        import config as _config
        engine.on_tick(price=5.20, bid=5.19, ask=5.21, last_atr=0.10,
                       event_ts=t1 + timedelta(seconds=_config.SIM_EXIT_LAPSE_SUSTAIN_SECS + 0.1))
        assert "SIM" not in SimulationEngine._positions

    def test_proven_trail_is_thesis_death(self):
        engine, entry, ofi, t0, pos = self._enter()
        engine.on_tick(price=5.20, bid=5.19, ask=5.21, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=3))
        assert pos.proven
        # Bullish tape, but price loses the broken level's anticipation band:
        # the structure that justified the trade is gone — instant exit.
        trail_break = pos.setup_level * 0.975
        engine.on_tick(price=trail_break, bid=trail_break - 0.01, ask=trail_break + 0.01,
                       last_atr=0.10, event_ts=t0 + timedelta(seconds=5))
        assert "SIM" not in SimulationEngine._positions

    def test_below_level_fill_owes_the_break(self):
        # Level above the tape (within the anticipation band): the fill is
        # anticipatory, so the level must TRADE within SIM_PROVE_BREAK_SECS
        # or the anticipated break never came.
        engine, entry, ofi, t0, pos = self._enter(level=5.10)
        assert pos.break_ts is None
        import config as _config
        engine.on_tick(price=5.06, bid=5.05, ask=5.07, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=_config.SIM_PROVE_BREAK_SECS + 1))
        assert "SIM" not in SimulationEngine._positions

    def test_below_level_break_starts_progress_clock(self):
        engine, entry, ofi, t0, pos = self._enter(level=5.10)
        assert pos.break_ts is None
        t_break = t0 + timedelta(seconds=5)
        engine.on_tick(price=5.11, bid=5.10, ask=5.12, last_atr=0.10, event_ts=t_break)
        assert pos.break_ts is not None and "SIM" in SimulationEngine._positions

    def test_legacy_regime_untouched(self):
        # Same shape as the sustained-lapse test, but legacy regime: the
        # first reversal print exits immediately (dwell patched to 0).
        with patch("config.SIM_EXIT_REGIME", "legacy"):
            engine, entry, ofi, t0, pos = self._enter()
            engine.on_tick(price=5.20, bid=5.19, ask=5.21, last_atr=0.10,
                           event_ts=t0 + timedelta(seconds=3))
            t1 = t0 + timedelta(seconds=4)
            self._flip_tape_bearish(ofi, t1)
            engine.on_tick(price=5.20, bid=5.19, ask=5.21, last_atr=0.10, event_ts=t1)
            assert "SIM" not in SimulationEngine._positions

    def test_fast_spike_reversal_with_flow_exits_instantly(self):
        # "What if it squeezes fast before 3 seconds": vertical to +8.5%,
        # then a violent reversal — price off the peak AND flow red at the
        # same tick are two instant facts: exit immediately, no 3s sustain.
        engine, entry, ofi, t0, pos = self._enter()
        spike = t0 + timedelta(seconds=1)
        engine.on_tick(price=5.50, bid=5.49, ask=5.51, last_atr=0.10, event_ts=spike)
        assert pos.proven and pos.peak_price == 5.50
        assert "SIM" in SimulationEngine._positions
        rev = spike + timedelta(milliseconds=400)
        self._flip_tape_bearish(ofi, rev)
        engine.on_tick(price=5.32, bid=5.31, ask=5.33, last_atr=0.10, event_ts=rev)
        assert "SIM" not in SimulationEngine._positions

    def test_fast_spike_wiggle_with_green_flow_holds(self):
        # Same giveback off the peak but flow still healthy: that is normal
        # mid-squeeze breathing, NOT a reversal — hold. (A bare price-off-
        # peak stop decapitated the +22% replay winner at +3%.)
        engine, entry, ofi, t0, pos = self._enter()
        spike = t0 + timedelta(seconds=1)
        engine.on_tick(price=5.50, bid=5.49, ask=5.51, last_atr=0.10, event_ts=spike)
        assert pos.proven
        engine.on_tick(price=5.32, bid=5.31, ask=5.33, last_atr=0.10,
                       event_ts=spike + timedelta(milliseconds=400))
        assert "SIM" in SimulationEngine._positions

    def test_peak_trail_not_armed_on_modest_move(self):
        # A proven-but-modest move (under the arm extension) keeps the wider
        # level-anchored trail — retest wiggle must not clip the position.
        engine, entry, ofi, t0, pos = self._enter()
        engine.on_tick(price=5.15, bid=5.14, ask=5.16, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=2))
        assert pos.proven
        # -2.5% off the 5.15 high but still above the level trail: hold.
        engine.on_tick(price=5.03, bid=5.02, ask=5.04, last_atr=0.10,
                       event_ts=t0 + timedelta(seconds=4))
        assert "SIM" in SimulationEngine._positions
