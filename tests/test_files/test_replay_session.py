"""Unit tests for the visual replay backend.

Covers the invariants the replay UI depends on:
  - entry_gate_states agrees with evaluate_entry (gate parity)
  - the chunk writer produces contiguous, anchored, time-bucketed chunks
  - anchors let a client scrub without replaying from chunk zero
  - display bars close on bucket edges with EMA/VWAP continuation
"""

import gzip
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    SIM_FIRST_MACD_CYCLE_ONLY,
    SIM_OFI_NORM_MIN,
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_PROGRESS_MIN,
    SIM_PRESSURE_SPEED_MIN,
    SIM_PRESSURE_VOL_IMBALANCE_MIN,
    SIM_SETUP_SCORE_MIN,
)
from exec.order_flow import OrderFlowTracker
from exec.replay_session import CHUNK_SECONDS, _ChunkWriter, _DisplayBars
from exec.signals import entry_gate_states, evaluate_entry


class _FakeTracker(SimpleNamespace):
    """Value-stub tracker that borrows the PRODUCTION gate predicates.

    The floor comparisons live on OrderFlowTracker (armed-band Step 2);
    delegating keeps this fake incapable of drifting from the real gate
    definitions — it only fakes the values, never the logic.
    """

    speed_gate_ok = OrderFlowTracker.speed_gate_ok
    activity_ok = OrderFlowTracker.activity_ok
    ratio_gate_ok = OrderFlowTracker.ratio_gate_ok
    vol_flow_ok = OrderFlowTracker.vol_flow_ok
    accel_gate_ok = OrderFlowTracker.accel_gate_ok
    progress_gate_ok = OrderFlowTracker.progress_gate_ok
    ofi_gate_ok = OrderFlowTracker.ofi_gate_ok


def _passing_entry(**overrides):
    entry = SimpleNamespace(
        entry_allowed=True,
        setup_pattern_score=SIM_SETUP_SCORE_MIN + 10.0,
        # A real level is now REQUIRED (no band -> no entry); default matches
        # _assert_parity's default last_price=10.0 so the band passes.
        setup_price_level=10.0,
        setup_pattern_timeframe="10sec",
        macd_cycle_done=False,
        macd_cycle_done_ts=None,
        eval_confidence=80.0,
        live_confidence=75.0,
        has_live_data=True,
        evaluation=None,
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


def _passing_tracker(**overrides):
    tracker = _FakeTracker(
        speed_gate=SIM_PRESSURE_SPEED_MIN + 5.0,
        activity_count=SIM_PRESSURE_ACTIVITY_MIN + 20,
        ratio_gate=SIM_PRESSURE_BUY_RATIO_MIN + 0.1,
        trade_imbalance=SIM_PRESSURE_VOL_IMBALANCE_MIN + 0.5,
        accel_gate_ready=True,
        accel_gate_z=SIM_PRESSURE_ACCEL_Z + 1.0,
        accel_gate_raw=0.2,
        buying_pressure=100,
        selling_pressure=20,
        ofi_rolling=5000,
        depth_active=False,
        ofi_normalized=0.0,
        ofi_norm_gate=0.0,
        price_progress=SIM_PRESSURE_PROGRESS_MIN + 0.005,
    )
    for key, value in overrides.items():
        setattr(tracker, key, value)
    return tracker


class TestEntryGateParity(unittest.TestCase):
    """all(gate.passing) must equal evaluate_entry(...).triggered."""

    def _assert_parity(self, entry, tracker, last_price=10.0, last_atr=0.5, rmap=None):
        states = entry_gate_states(entry, tracker, last_price, last_atr, rmap)
        signal = evaluate_entry(entry, tracker, last_price, last_atr, rmap)
        self.assertEqual(
            all(state.passing for state in states),
            signal.triggered,
            f"gate parity broke: {[(s.name, s.passing) for s in states]}",
        )
        return states

    def test_all_pass(self):
        states = self._assert_parity(_passing_entry(), _passing_tracker())
        self.assertTrue(all(state.passing for state in states))

    def test_each_single_gate_failure(self):
        cases = [
            ({"entry_allowed": False}, {}),
            ({"setup_pattern_score": SIM_SETUP_SCORE_MIN - 1.0}, {}),

            ({}, {"speed_gate": SIM_PRESSURE_SPEED_MIN - 1.0}),
            ({}, {"activity_count": SIM_PRESSURE_ACTIVITY_MIN - 1}),
            ({}, {"ratio_gate": SIM_PRESSURE_BUY_RATIO_MIN - 0.05}),
            ({}, {"trade_imbalance": SIM_PRESSURE_VOL_IMBALANCE_MIN - 0.1}),
            ({}, {"accel_gate_z": SIM_PRESSURE_ACCEL_Z - 0.5}),
            ({}, {"accel_gate_ready": False}),
            ({}, {"price_progress": 0.0}),
            ({}, {"price_progress": None}),          # warming up -> fail closed
            ({"setup_price_level": None}, {}),       # no level -> no band -> no entry
            ({}, {"depth_active": True, "ofi_norm_gate": SIM_OFI_NORM_MIN - 0.5}),
        ]
        if SIM_FIRST_MACD_CYCLE_ONLY:
            # first MACD cycle over -> no more entries (gate reported only
            # while the flag enforces it)
            cases.append(({"macd_cycle_done": True}, {}))
        for entry_over, tracker_over in cases:
            with self.subTest(entry=entry_over, tracker=tracker_over):
                states = self._assert_parity(_passing_entry(**entry_over), _passing_tracker(**tracker_over))
                self.assertFalse(all(state.passing for state in states))

    def test_price_band(self):
        entry = _passing_entry(setup_price_level=10.0)
        self._assert_parity(entry, _passing_tracker(), last_price=10.0)
        self._assert_parity(entry, _passing_tracker(), last_price=12.0)  # above band
        self._assert_parity(entry, _passing_tracker(), last_price=9.0)   # below band

    def test_resistance_headroom(self):
        class _Level(SimpleNamespace):
            pass

        class _RMap:
            def __init__(self, price):
                self._price = price

            def nearest_above(self, price):
                return _Level(price=self._price) if self._price > price else None

        # Wall 0.1 above with ATR 0.5 → headroom 0.2 ATR < 1.0 ATR → blocked.
        self._assert_parity(_passing_entry(), _passing_tracker(), last_price=10.0, last_atr=0.5, rmap=_RMap(10.1))
        # Wall far above → passes.
        self._assert_parity(_passing_entry(), _passing_tracker(), last_price=10.0, last_atr=0.5, rmap=_RMap(20.0))

    def test_inactive_gates_pass_vacuously(self):
        states = entry_gate_states(_passing_entry(), _passing_tracker(), 10.0, 0.0, None)
        by_name = {state.name: state for state in states}
        self.assertFalse(by_name["ofi_norm"].active)
        self.assertTrue(by_name["ofi_norm"].passing)
        self.assertFalse(by_name["headroom"].active)
        self.assertTrue(by_name["headroom"].passing)
        self.assertTrue(by_name["price_band"].active)

    def test_missing_setup_level_hard_fails_band(self):
        # No setup level is a HARD fail now (no band -> no entry) — the
        # failure-badge state must not read as vacuously green.
        states = entry_gate_states(
            _passing_entry(setup_price_level=None), _passing_tracker(), 10.0, 0.0, None,
        )
        by_name = {state.name: state for state in states}
        self.assertTrue(by_name["price_band"].active)
        self.assertFalse(by_name["price_band"].passing)


class TestChunkWriter(unittest.TestCase):
    def _build(self, tmp: Path, times_ms: list[int], start_ms: int = 0):
        anchors = []

        def anchor_fn(chunk_start_ms):
            anchors.append(chunk_start_ms)
            return {"t": chunk_start_ms, "k": "anchor", "n": len(anchors)}

        writer = _ChunkWriter(tmp, start_ms, anchor_fn, on_roll=lambda n: None)
        for t in times_ms:
            writer.write({"t": t, "k": "tick"})
        end_ms = (times_ms[-1] if times_ms else start_ms) + 1
        count = writer.finalize(end_ms)
        return count, anchors

    def test_contiguous_chunks_with_anchors(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            chunk_ms = CHUNK_SECONDS * 1000
            # Records in chunks 0 and 2 — chunk 1 is quiet but must still exist.
            times = [10, 20, 2 * chunk_ms + 5]
            count, anchors = self._build(tmp, times)
            self.assertEqual(count, 3)
            files = sorted((tmp / "chunks").glob("*.ndjson.gz"))
            self.assertEqual([f.name for f in files], ["000000.ndjson.gz", "000001.ndjson.gz", "000002.ndjson.gz"])
            for index, path in enumerate(files):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle]
                self.assertEqual(records[0]["k"], "anchor")
                self.assertEqual(records[0]["t"], index * chunk_ms)
                for record in records[1:]:
                    self.assertGreaterEqual(record["t"], index * chunk_ms)
                    self.assertLess(record["t"], (index + 1) * chunk_ms)
            self.assertEqual(anchors, [0, chunk_ms, 2 * chunk_ms])

    def test_late_first_record_still_writes_leading_chunks(self):
        """Tape starting minutes into the window must not skip chunks 0..N-1
        (the client fetches sequentially from 0 — a missing leading chunk
        aborted playback for every symbol whose tape started mid-window)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            chunk_ms = CHUNK_SECONDS * 1000
            # First record lands in chunk 6.
            count, _ = self._build(tmp, [6 * chunk_ms + 100, 6 * chunk_ms + 200])
            self.assertEqual(count, 7)
            files = sorted((tmp / "chunks").glob("*.ndjson.gz"))
            self.assertEqual(len(files), 7)
            self.assertEqual(files[0].name, "000000.ndjson.gz")
            with gzip.open(files[0], "rt", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 1)  # anchor only
            self.assertEqual(records[0]["k"], "anchor")

    def test_scrub_equivalence(self):
        """anchor_n + records<=T must equal replay-from-zero<=T.

        The 'state' here is the last tick value; anchors snapshot it at every
        chunk boundary exactly like the builder snapshots criteria state.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            chunk_ms = CHUNK_SECONDS * 1000
            state = {"last": None}

            def anchor_fn(chunk_start_ms):
                return {"t": chunk_start_ms, "k": "anchor", "last": state["last"]}

            writer = _ChunkWriter(tmp, 0, anchor_fn, on_roll=lambda n: None)
            times_values = [(i * 700, i) for i in range(0, 400)]  # spans >4 chunks
            for t, value in times_values:
                state["last"] = value
                writer.write({"t": t, "k": "tick", "v": value})
            writer.finalize(times_values[-1][0] + 1)

            def read_chunk(index):
                with gzip.open(tmp / "chunks" / f"{index:06d}.ndjson.gz", "rt", encoding="utf-8") as handle:
                    return [json.loads(line) for line in handle]

            target_t = int(2.5 * chunk_ms)
            # Path A: replay everything from chunk 0.
            full = None
            for index in range(0, 3):
                for record in read_chunk(index):
                    if record["k"] == "tick" and record["t"] <= target_t:
                        full = record["v"]
            # Path B: anchor of chunk 2 + its records <= T.
            records = read_chunk(2)
            scrubbed = records[0]["last"]
            for record in records[1:]:
                if record["k"] == "tick" and record["t"] <= target_t:
                    scrubbed = record["v"]
            self.assertEqual(full, scrubbed)


class TestDisplayBars(unittest.TestCase):
    def test_bucket_close_and_continuation(self):
        bars = _DisplayBars()
        bars.seed({"10s": [
            {"t": 0, "o": 1, "h": 1, "l": 1, "c": 1.0, "v": 100, "bar_vwap": 1.0, "ema9": 1.0, "ema20": 1.0},
        ], "1m": [], "2m": [], "5m": []})
        self.assertAlmostEqual(bars.cum_pv, 100.0)
        self.assertAlmostEqual(bars.cum_v, 100.0)

        closed = bars.on_tick(10_000, 2.0, 50)
        self.assertEqual(closed, [])
        closed = bars.on_tick(15_000, 3.0, 50)   # same 10s bucket
        self.assertEqual(closed, [])
        closed = bars.on_tick(21_000, 2.5, 100)  # next 10s bucket → closes 10s bar
        self.assertEqual(len(closed), 1)
        tf, bar = closed[0]
        self.assertEqual(tf, "10s")
        self.assertEqual(bar["t"], 10_000)
        self.assertEqual((bar["o"], bar["h"], bar["l"], bar["c"], bar["v"]), (2.0, 3.0, 2.0, 3.0, 100))
        # EMA continues from the seeded value, not from scratch.
        expected_ema9 = 1.0 + (2.0 / 10.0) * (3.0 - 1.0)
        self.assertAlmostEqual(bar["ema9"], expected_ema9, places=6)
        # VWAP is cumulative: (100*1 + 2*50 + 3*50 + 2.5*100) / 300
        expected_vwap = (100 * 1.0 + 2.0 * 50 + 3.0 * 50 + 2.5 * 100) / 300.0
        self.assertAlmostEqual(bar["vwap"], expected_vwap, places=6)

    def test_forming_snapshot(self):
        bars = _DisplayBars()
        bars.on_tick(5_000, 2.0, 10)
        forming = bars.forming()
        self.assertEqual(forming["10s"]["t"], 0)
        self.assertEqual(forming["10s"]["c"], 2.0)
        self.assertEqual(forming["1m"]["v"], 10)


if __name__ == "__main__":
    unittest.main()


class TestArmedBandHashedConstants(unittest.TestCase):
    """Every armed-band constant must bust the bundle cache (stale-bundle guard)."""

    def test_new_constants_present_in_hash(self):
        from exec.replay_session import _hashed_constants

        keys = {k for k, _ in _hashed_constants()}
        for expected in (
            "entry_regime", "anticipation_max_pct", "level_cross_tol_pct",
            "arm_evidence_ttl", "arm_strong_ttl", "arm_strong_requires_wall",
            "trigger_ttl", "trigger_trail_window", "trigger_trail_min_span",
            "wall_band_pct", "wall_evap_min",
        ):
            self.assertIn(expected, keys)

    def test_hash_reflects_runtime_regime_override(self):
        import config
        from exec.replay_session import config_hash

        before = config_hash()
        original = config.SIM_ENTRY_REGIME
        try:
            config.SIM_ENTRY_REGIME = (
                "armed_band" if original != "armed_band" else "legacy"
            )
            self.assertNotEqual(before, config_hash())
        finally:
            config.SIM_ENTRY_REGIME = original
        self.assertEqual(before, config_hash())
