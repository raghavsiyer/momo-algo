"""Posterior entry regime (Upgrade 1): encoder, artifact, regime wiring.

Same harness discipline as test_armed_band.py: a REAL OrderFlowTracker fed
synthetic prints, evaluate_entry consuming production state end-to-end, and
the parity invariant all(entry_gate_states passing) == triggered asserted
in every scenario.
"""

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import config
from exec.order_flow import OrderFlowTracker
from exec.posterior import (
    FEATURES,
    EntryPosterior,
    encode_features,
    get_model,
    score_entry,
    tracker_features,
)
from exec.signals import entry_band_floor, entry_gate_states, evaluate_entry
from market_data.tradier_stream import LiveTimesale

T0 = datetime(2026, 8, 14, 14, 30, 0, tzinfo=timezone.utc)

LEVEL = 10.0
PULLBACK_LOW = 9.3
PULLBACK_TS = T0 - timedelta(minutes=1)

RAW_BASE = {
    "speed": 20.0,
    "ratio": 0.7,
    "volimb": 0.2,
    "accelz": 1.5,
    "accel_ready": True,
    "activity": 120,
    "ofi_ema10": 0.8,
    "depth_active": True,
    "rel_to_level": -0.01,
    "progress15": 0.01,
    "progress_ready": True,
    "armed_speed": True,
    "armed_ratio": True,
    "armed_volimb": True,
    "armed_ofi": True,
}


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


def _fed_tracker(prices, *, gap_ms=50, sells=0):
    """Tracker with the band set, fed a buy sequence (production state)."""
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
def posterior_regime(monkeypatch):
    monkeypatch.setattr(config, "SIM_ENTRY_REGIME", "posterior")


class TestEncoder:
    def test_length_and_determinism(self):
        x = encode_features(RAW_BASE)
        assert len(x) == len(FEATURES)
        assert x == encode_features(dict(RAW_BASE))

    def test_no_depth_zeroes_ofi_term(self):
        raw = {**RAW_BASE, "depth_active": False}
        x = encode_features(raw)
        i_ofi = FEATURES.index("ofi_ema10")
        i_depth = FEATURES.index("depth_active")
        assert x[i_ofi] == 0.0
        assert x[i_depth] == 0.0

    def test_accel_not_ready_zeroes_z(self):
        raw = {**RAW_BASE, "accel_ready": False, "accelz": 5.0}
        x = encode_features(raw)
        assert x[FEATURES.index("accelz")] == 0.0
        assert x[FEATURES.index("accel_ready")] == 0.0

    def test_progress_not_ready_zeroes_term(self):
        raw = {**RAW_BASE, "progress_ready": False, "progress15": 0.04}
        x = encode_features(raw)
        assert x[FEATURES.index("progress_pct")] == 0.0
        assert x[FEATURES.index("progress_ready")] == 0.0

    def test_clips_bound_wild_values(self):
        raw = {**RAW_BASE, "speed": 1e9, "accelz": 500.0, "ofi_ema10": -40.0,
               "rel_to_level": 0.5, "progress15": -3.0}
        x = encode_features(raw)
        assert x[FEATURES.index("log1p_speed")] == pytest.approx(math.log1p(200.0))
        assert x[FEATURES.index("accelz")] == 6.0
        assert x[FEATURES.index("ofi_ema10")] == -3.0
        assert x[FEATURES.index("rel_level_pct")] == 2.0
        assert x[FEATURES.index("progress_pct")] == -5.0


class TestArtifact:
    def test_repo_artifact_loads_and_scores(self):
        model = get_model()
        assert model.features == FEATURES
        assert 0.0 < model.threshold < 1.0
        p = model.score(RAW_BASE)
        assert 0.0 < p < 1.0
        assert p == model.score(dict(RAW_BASE))  # deterministic

    def test_score_moves_with_coefficient_sign(self):
        model = get_model()
        i = FEATURES.index("volimb")
        lo = model.score({**RAW_BASE, "volimb": -0.5})
        hi = model.score({**RAW_BASE, "volimb": 0.5})
        if model.coef[i] > 0:
            assert hi > lo
        elif model.coef[i] < 0:
            assert hi < lo

    def test_feature_mismatch_refused(self):
        model = get_model()
        blob = {
            "features": list(reversed(FEATURES)),
            "mean": model.mean,
            "scale": model.scale,
            "coef": model.coef,
            "intercept": model.intercept,
            "threshold": model.threshold,
        }
        with pytest.raises(ValueError, match="feature mismatch"):
            EntryPosterior(blob)


class TestTrackerFeatures:
    def test_no_depth_semantics(self):
        t, ts = _fed_tracker([9.8] * 100)
        raw = tracker_features(t, 9.9, LEVEL)
        assert set(raw) == set(RAW_BASE)
        assert raw["depth_active"] is False
        assert raw["armed_ofi"] is True  # vacuous without depth
        assert raw["rel_to_level"] == pytest.approx(9.9 / LEVEL - 1.0)
        assert raw["activity"] == t.activity_count

    def test_armed_latches_follow_tracker(self):
        # In-band buys stamp speed/ratio/vol_flow latches
        t, ts = _fed_tracker([9.8] * 100)
        raw = tracker_features(t, 9.8, LEVEL)
        assert raw["armed_speed"] is True
        assert raw["armed_ratio"] is True
        # A fresh tracker with no band never stamps
        t2 = OrderFlowTracker()
        for i in range(60):
            t2.on_timesale(_print(T0 + timedelta(milliseconds=50 * i), 9.8))
        raw2 = tracker_features(t2, 9.8, LEVEL)
        assert raw2["armed_speed"] is False


class TestPosteriorRegime:
    def test_enters_when_score_clears_floor(self, posterior_regime, monkeypatch):
        monkeypatch.setattr(config, "SIM_POSTERIOR_THRESHOLD", 1e-6)
        t, _ = _fed_tracker([round(9.80 + i * 0.002, 4) for i in range(100)])
        sig, states = _parity(_entry(), t, 9.99)
        assert sig.triggered
        assert any(r.startswith("posterior=") for r in sig.reasons)

    def test_blocked_when_score_below_floor(self, posterior_regime, monkeypatch):
        monkeypatch.setattr(config, "SIM_POSTERIOR_THRESHOLD", 0.999)
        t, _ = _fed_tracker([round(9.80 + i * 0.002, 4) for i in range(100)])
        sig, states = _parity(_entry(), t, 9.99)
        assert not sig.triggered
        st = {s.name: s for s in states}
        assert not st["posterior"].passing

    def test_zone_cap_blocks_above_cross_tolerance(self, posterior_regime, monkeypatch):
        monkeypatch.setattr(config, "SIM_POSTERIOR_THRESHOLD", 1e-6)
        # The scored zone and the band both end 5% above the level in config,
        # so a price can only be in the band but past the zone when the band is
        # wider. Widen it where it's read, before the tracker fixes its top.
        import exec.order_flow
        import exec.signals
        for module in (config, exec.signals, exec.order_flow):
            monkeypatch.setattr(module, "SIM_MAX_SETUP_EXTENSION_PCT", 0.10)
        t, _ = _fed_tracker([round(9.80 + i * 0.002, 4) for i in range(100)])
        price = LEVEL * (1.0 + config.SIM_LEVEL_CROSS_TOLERANCE_PCT) + 0.02
        sig, states = _parity(_entry(), t, price)
        assert not sig.triggered
        st = {s.name: s for s in states}
        assert st["price_band"].passing      # still inside the entry band...
        assert not st["scored_zone"].passing  # ...but beyond model support

    def test_no_fresh_break_no_entry(self, posterior_regime, monkeypatch):
        # Score clears the floor but the tape hasn't broken its trailing
        # high recently: no timing trigger, no entry (churn-instant guard).
        monkeypatch.setattr(config, "SIM_POSTERIOR_THRESHOLD", 1e-6)
        t, ts = _fed_tracker([round(9.80 + i * 0.002, 4) for i in range(100)])
        for _ in range(30):  # 3s of prints below the high: break goes stale
            ts += timedelta(milliseconds=100)
            t.on_timesale(_print(ts, 9.85))
        sig, states = _parity(_entry(), t, 9.99)
        assert not sig.triggered
        st = {s.name: s for s in states}
        assert not st["trigger"].passing

    def test_thin_tape_blocked_by_activity_validity(self, posterior_regime, monkeypatch):
        monkeypatch.setattr(config, "SIM_POSTERIOR_THRESHOLD", 1e-6)
        t, _ = _fed_tracker([9.9] * 10)  # 10 prints << SIM_PRESSURE_ACTIVITY_MIN
        sig, states = _parity(_entry(), t, 9.99)
        assert not sig.triggered
        st = {s.name: s for s in states}
        assert not st["tape_activity"].passing

    def test_structural_gates_stay_hard(self, posterior_regime, monkeypatch):
        monkeypatch.setattr(config, "SIM_POSTERIOR_THRESHOLD", 1e-6)
        t, _ = _fed_tracker([round(9.80 + i * 0.002, 4) for i in range(100)])
        # No setup level -> no trade, parity holds
        sig, _ = _parity(_entry(setup_price_level=None), t, 9.99)
        assert not sig.triggered
        # Above the band extension cap -> no trade
        sig, _ = _parity(_entry(), t, LEVEL * (1 + config.SIM_MAX_SETUP_EXTENSION_PCT) + 0.05)
        assert not sig.triggered
        # MACD cycle done -> no trade
        if config.SIM_FIRST_MACD_CYCLE_ONLY:
            sig, _ = _parity(_entry(macd_cycle_done=True), t, 9.99)
            assert not sig.triggered

    def test_no_ofi_floor_state_in_posterior_chain(self, posterior_regime):
        t, _ = _fed_tracker([9.9] * 100)
        states = entry_gate_states(_entry(), t, 9.99, 0.5, None)
        names = {s.name for s in states}
        assert "ofi_norm" not in names
        assert "posterior" in names and "scored_zone" in names

    def test_band_floor_scales_like_armed(self, posterior_regime):
        assert entry_band_floor(LEVEL, PULLBACK_LOW) == pytest.approx(
            LEVEL * (1.0 - config.SIM_ANTICIPATION_MAX_PCT)
        )

    def test_legacy_chain_untouched(self):
        # Regime untouched (legacy default): posterior states absent,
        # OFI floor state present (inactive without depth, but emitted).
        t, _ = _fed_tracker([9.9] * 100)
        states = entry_gate_states(_entry(), t, 9.99, 0.5, None)
        names = {s.name for s in states}
        assert "posterior" not in names and "scored_zone" not in names
        assert "ofi_norm" in names
