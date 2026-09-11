"""Armed-band entry regime: zone/tier matrix, triggers, parity, band floor.

Uses a REAL OrderFlowTracker (fed synthetic prints) so the latch stamps,
trigger edges and evaluate_entry consume production state end-to-end.
Parity invariant asserted throughout: all(entry_gate_states passing) ==
evaluate_entry(...).triggered — in BOTH regimes and every zone.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import config
from exec.order_flow import OrderFlowTracker
from exec.signals import (
    ZONE_BELOW,
    ZONE_CROSS,
    ZONE_EXTENDED,
    entry_band_floor,
    entry_gate_states,
    entry_zone,
    evaluate_entry,
)
from market_data.tradier_stream import LiveTimesale

T0 = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)

LEVEL = 10.0
PULLBACK_LOW = 9.3
PULLBACK_TS = T0 - timedelta(minutes=1)


def _print(ts, last, *, sell=False, size=100):
    bid, ask = (last, last + 0.01) if sell else (last - 0.01, last)
    return LiveTimesale(
        "timesale", "TEST", "Q", last, bid, ask, size, ts, 1, "", False, False, "normal",
    )


def _entry(**overrides):
    e = SimpleNamespace(
        entry_allowed=True,
        setup_pattern_score=60.0,
        setup_price_level=LEVEL,
        setup_pullback_low=PULLBACK_LOW,
        setup_pullback_ts=PULLBACK_TS,
        setup_pattern_timeframe="10sec",
        macd_cycle_done=False,
        macd_cycle_done_ts=None,
        eval_confidence=80.0,
        live_confidence=75.0,
        has_live_data=True,
        evaluation=None,
    )
    for k, v in overrides.items():
        setattr(e, k, v)
    return e


def _armed_tracker(prices, *, gap_ms=50, sells=0):
    """Tracker with the band set, fed a buy sequence (latches stamp in-band)."""
    t = OrderFlowTracker()
    t.set_entry_band(LEVEL, entry_band_floor(LEVEL, PULLBACK_LOW), PULLBACK_TS)
    ts = T0
    for p in prices:
        ts += timedelta(milliseconds=gap_ms)
        t.on_timesale(_print(ts, p))
    for _ in range(sells):
        ts += timedelta(milliseconds=gap_ms)
        t.on_timesale(_print(ts, prices[-1], sell=True))
    return t, ts


def _parity(entry, tracker, price):
    sig = evaluate_entry(entry, tracker, price, 0.5, None)
    states = entry_gate_states(entry, tracker, price, 0.5, None)
    assert all(s.passing for s in states) == sig.triggered, (
        f"parity broke: {[(s.name, s.passing) for s in states]} vs {sig.triggered}"
    )
    return sig, states


@pytest.fixture
def armed_regime(monkeypatch):
    monkeypatch.setattr(config, "SIM_ENTRY_REGIME", "armed_band")


class TestBandFloor:
    def test_legacy_floor_is_fixed_anticipation(self):
        assert entry_band_floor(LEVEL, PULLBACK_LOW) == pytest.approx(
            LEVEL * (1.0 - config.SIM_SETUP_ANTICIPATION_PCT)
        )

    def test_armed_floor_scales_to_pullback(self, armed_regime):
        # Pullback low (9.3) below the 6% cap floor (9.4): cap binds
        assert entry_band_floor(LEVEL, PULLBACK_LOW) == pytest.approx(
            LEVEL * (1.0 - config.SIM_ANTICIPATION_MAX_PCT)
        )
        # Shallow pullback: its low binds instead
        assert entry_band_floor(LEVEL, 9.8) == pytest.approx(9.8)
        # No pullback_low known: legacy fallback
        assert entry_band_floor(LEVEL, None) == pytest.approx(
            LEVEL * (1.0 - config.SIM_SETUP_ANTICIPATION_PCT)
        )


class TestZones:
    def test_zone_classification(self, armed_regime):
        tol = config.SIM_LEVEL_CROSS_TOLERANCE_PCT
        assert entry_zone(LEVEL - 0.01, LEVEL) == ZONE_BELOW
        assert entry_zone(LEVEL, LEVEL) == ZONE_CROSS
        assert entry_zone(LEVEL * (1 + tol), LEVEL) == ZONE_CROSS
        assert entry_zone(LEVEL * (1 + tol) + 0.01, LEVEL) == ZONE_EXTENDED


def _coil_then_burst_tracker(coil_price=9.8, coil_secs=30.0, burst_prices=None):
    """Long steady coil (kernels converge, accel ~0) then a genuine burst
    through the level: the user's manual trigger shape — the break print
    WITH a burst of buying occurring together."""
    if burst_prices is None:
        burst_prices = [round(9.82 + i * 0.005, 4) for i in range(50)]  # → 10.065
    t = OrderFlowTracker()
    t.set_entry_band(LEVEL, entry_band_floor(LEVEL, PULLBACK_LOW), PULLBACK_TS)
    ts = T0
    for _ in range(int(coil_secs * 20)):  # 20 buys/s steady
        ts += timedelta(milliseconds=50)
        t.on_timesale(_print(ts, coil_price))
    for p in burst_prices:
        ts += timedelta(milliseconds=10)  # 100/s burst, ascending through level
        t.on_timesale(_print(ts, p))
    return t, ts, burst_prices[-1]


class TestArmedEntry:
    def test_cross_zone_needs_break_AND_burst(self, armed_regime):
        # Coil arms the evidence; the entry fires when the break prints WITH
        # a burst of buying (manual method: both together, not either).
        t, end, last = _coil_then_burst_tracker()
        sig, _ = _parity(_entry(), t, last)
        assert entry_zone(last, LEVEL) == ZONE_CROSS
        assert sig.triggered
        assert any("trig=break+accel" in r for r in sig.reasons)

    def test_quiet_poke_through_level_refused(self, armed_regime):
        # A lone print poking the trailing high on converged (non-bursting)
        # tape must NOT enter — the exact failure mode the AND filters.
        t = OrderFlowTracker()
        t.set_entry_band(LEVEL, entry_band_floor(LEVEL, PULLBACK_LOW), PULLBACK_TS)
        ts = T0
        for _ in range(600):  # 30s steady coil, accel converges to ~0
            ts += timedelta(milliseconds=50)
            t.on_timesale(_print(ts, 9.8))
        t.on_timesale(_print(ts + timedelta(milliseconds=50), 10.05))
        sig, states = _parity(_entry(), t, 10.05)
        assert not sig.triggered
        trig = next(s for s in states if s.name == "trigger")
        assert not trig.passing

    def test_fresh_wall_vetoes_armed_entry(self, armed_regime, monkeypatch):
        # User scenario: dip bought strongly, break printing WITH a burst —
        # but a huge seller just parked on the level. Veto.
        t, end, last = _coil_then_burst_tracker()
        monkeypatch.setattr(OrderFlowTracker, "depth_active", property(lambda self: True))
        monkeypatch.setattr(OrderFlowTracker, "ofi_norm_gate", property(lambda self: 1.5))
        monkeypatch.setattr(OrderFlowTracker, "wall_trend", property(lambda self: -2.0))
        # ofi latch may be missing (no book fed during coil) — stamp it so
        # only the veto separates the outcomes.
        t._gate_last_true["ofi"] = end
        sig, states = _parity(_entry(), t, last)
        assert not sig.triggered
        veto = next(s for s in states if s.name == "wall_veto")
        assert not veto.passing
        # Same book state without the fresh wall: enters.
        monkeypatch.setattr(OrderFlowTracker, "wall_trend", property(lambda self: -0.4))
        sig2, _ = _parity(_entry(), t, last)
        assert sig2.triggered

    def test_book_must_imbalance_now_at_trigger(self, armed_regime, monkeypatch):
        # OFI was fine recently (latched) but is NOT clearing the floor at
        # the trigger instant — the book isn't imbalancing NOW. Refuse.
        t, end, last = _coil_then_burst_tracker()
        monkeypatch.setattr(OrderFlowTracker, "depth_active", property(lambda self: True))
        monkeypatch.setattr(OrderFlowTracker, "ofi_norm_gate", property(lambda self: 0.2))
        t._gate_last_true["ofi"] = end  # latched recently
        sig, states = _parity(_entry(), t, last)
        assert not sig.triggered
        ofi_state = next(s for s in states if s.name == "ofi_norm")
        assert not ofi_state.passing

    def test_below_zone_needs_microbreak_not_just_arm(self, armed_regime):
        # Flat in-band tape: armed, but nothing broke the trailing high —
        # a sub-break entry without a resumption print is a dip-buy, refuse.
        t, end = _armed_tracker([9.8] * 120)
        sig, _ = _parity(_entry(), t, 9.8)
        assert not sig.triggered

    def test_below_zone_break_of_trailing_high_enters(self, armed_regime):
        # Ascending in-band prints: the dip is being bought NOW (ratio ok)
        # and each new print breaks the micro trailing high — the manual
        # method's "slightly earlier" entry below the level.
        prices = [round(9.5 + i * 0.003, 4) for i in range(120)]
        t, _ = _armed_tracker(prices)
        last = prices[-1]
        sig, _ = _parity(_entry(), t, last)
        assert entry_zone(last, LEVEL) == ZONE_BELOW
        assert sig.triggered
        assert any("zone=below" in r for r in sig.reasons)

    def test_stale_evidence_refuses(self, armed_regime):
        # Buys long ago, then 8s of selling: ratio latch ages past the
        # NORMAL TTL — even a fresh break must not fire on stale evidence.
        t, end = _armed_tracker([9.8] * 120, sells=160)
        t.on_timesale(_print(end + timedelta(seconds=8), 10.05))
        sig, states = _parity(_entry(), t, 10.05)
        assert not sig.triggered
        arm = next(s for s in states if s.name == "arm_evidence")
        assert not arm.passing and "ratio" in arm.detail

    def test_extended_zone_requires_full_stack(self, armed_regime):
        # Steady tape has no acceleration significance: the extended zone
        # falls back to the legacy instantaneous stack and refuses.
        t, end = _armed_tracker([9.8] * 120)
        t.on_timesale(_print(end + timedelta(milliseconds=50), 10.3))
        sig, states = _parity(_entry(), t, 10.3)
        assert not sig.triggered
        names = {s.name for s in states}
        assert "tape_acceleration" in names and "arm_evidence" not in names

    def test_armed_chain_gate_names(self, armed_regime):
        t, end, last = _coil_then_burst_tracker()
        _, states = _parity(_entry(), t, last)
        names = [s.name for s in states]
        assert "arm_evidence" in names and "trigger" in names
        assert "wall_veto" in names and "ofi_norm" in names
        assert "tape_progress" not in names  # response is the trigger's job

    def test_legacy_regime_untouched_by_armed_state(self):
        # Same armed-favorable tracker state, default legacy regime: the
        # instantaneous stack decides alone (no armed chain nodes).
        assert config.SIM_ENTRY_REGIME == "legacy"
        t, end = _armed_tracker([9.8] * 120)
        t.on_timesale(_print(end + timedelta(milliseconds=50), 10.05))
        sig, states = _parity(_entry(), t, 10.05)
        assert not sig.triggered
        assert "arm_evidence" not in {s.name for s in states}
