"""Tests for L2 depth OFI, depth_active gating, and the EMA tape-acceleration estimator."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import exec.order_flow as of_mod
from config import (
    OFI_DEPTH_STALE_SECS,
    OFI_L2_LEVEL_DECAY,
    TAPE_ACCEL_EMA_MIN_NEFF,
)
from exec.order_flow import OrderFlowTracker
from market_data.ibkr_stream import OrderBookLevel, OrderBookSnapshot
from market_data.tradier_stream import LiveQuote, LiveTimesale

T0 = datetime(2026, 7, 24, 14, 30, 0, tzinfo=timezone.utc)


def _book(ts: datetime, bids: list[tuple[float, int]], asks: list[tuple[float, int]]) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        symbol="TEST",
        bids=[OrderBookLevel("TEST", "bid", i, p, s, "", ts) for i, (p, s) in enumerate(bids)],
        asks=[OrderBookLevel("TEST", "ask", i, p, s, "", ts) for i, (p, s) in enumerate(asks)],
        ts=ts,
    )


def _quote(ts: datetime, bid: float, bid_size: int, ask: float, ask_size: int) -> LiveQuote:
    return LiveQuote("quote", "TEST", bid, bid_size, "", ts, ask, ask_size, "", ts)


def _print(ts: datetime, last: float, bid: float, ask: float, size: int = 100) -> LiveTimesale:
    return LiveTimesale(
        "timesale", "TEST", "Q", last, bid, ask, size, ts, 1, "", False, False, "normal",
    )


class TestDepthOFI:
    def test_first_snapshot_produces_no_event(self):
        t = OrderFlowTracker()
        assert t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)])) is None
        assert t.ofi_rolling == 0

    def test_inside_bid_size_increase_is_buy_flow(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        ev = t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 400)], [(5.01, 400)]))
        assert ev is not None and ev.ofi == 100

    def test_bid_price_improvement_counts_full_new_size(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.02, 400)]))
        ev = t.on_book(_book(T0 + timedelta(seconds=1), [(5.01, 250)], [(5.02, 400)]))
        assert ev.delta_bid == 250 and ev.ofi == 250

    def test_bid_price_retreat_counts_full_prev_size_consumed(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.02, 400)]))
        ev = t.on_book(_book(T0 + timedelta(seconds=1), [(4.99, 500)], [(5.02, 400)]))
        assert ev.delta_bid == -300 and ev.ofi == -300

    def test_ask_signs_inverted(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        # Ask price UP = offer consumed = buy pressure (delta_ask = -prev)
        ev = t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 300)], [(5.02, 500)]))
        assert ev.delta_ask == -400 and ev.ofi == 400

    def test_deeper_levels_geometrically_weighted(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300), (4.99, 200)], [(5.01, 400), (5.02, 100)]))
        # Level-2 bid size +100, everything else unchanged.
        ev = t.on_book(_book(
            T0 + timedelta(seconds=1),
            [(5.00, 300), (4.99, 300)],
            [(5.01, 400), (5.02, 100)],
        ))
        assert ev.delta_bid == int(round(100 * OFI_L2_LEVEL_DECAY))

    def test_level_appears_and_vanishes(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        ev = t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 300), (4.99, 200)], [(5.01, 400)]))
        assert ev.delta_bid == int(round(200 * OFI_L2_LEVEL_DECAY))
        ev = t.on_book(_book(T0 + timedelta(seconds=2), [(5.00, 300)], [(5.01, 400)]))
        assert ev.delta_bid == -int(round(200 * OFI_L2_LEVEL_DECAY))

    def test_ofi_rolling_sums_and_trims(self):
        t = OrderFlowTracker(ofi_window_secs=30)
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 400)], [(5.01, 400)]))   # +100
        t.on_book(_book(T0 + timedelta(seconds=2), [(5.00, 450)], [(5.01, 400)]))   # +50
        assert t.ofi_rolling == 150
        # Advance past the window: old events trimmed out of the running sum.
        t.on_book(_book(T0 + timedelta(seconds=29), [(5.00, 450)], [(5.01, 400)]))  # 0
        t.on_book(_book(T0 + timedelta(seconds=40), [(5.00, 460)], [(5.01, 400)]))  # +10
        assert t.ofi_rolling == 10


class TestDepthActive:
    def test_active_after_first_book(self):
        t = OrderFlowTracker()
        assert t.depth_active is False
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        assert t.depth_active is True

    def test_stale_after_gap_measured_by_event_time(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        t.on_quote(_quote(T0 + timedelta(seconds=OFI_DEPTH_STALE_SECS + 5), 5.00, 300, 5.01, 400))
        assert t.depth_active is False

    def test_l1_ofi_suppressed_while_depth_active(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        t.on_quote(_quote(T0 + timedelta(seconds=1), 5.00, 300, 5.01, 400))
        t.on_quote(_quote(T0 + timedelta(seconds=2), 5.00, 500, 5.01, 400))
        assert len(t._ofi_events) == 0  # book lean updates, but no L1 OFI event
        assert t.book_lean != 0.0

    def test_l1_ofi_resumes_when_depth_stale(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        later = T0 + timedelta(seconds=OFI_DEPTH_STALE_SECS + 10)
        t.on_quote(_quote(later, 5.00, 300, 5.01, 400))
        t.on_quote(_quote(later + timedelta(seconds=1), 5.00, 350, 5.01, 400))
        assert t.ofi_rolling == 50

    def test_resumed_depth_feed_does_not_diff_stale_ladder(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        resumed = t.on_book(_book(
            T0 + timedelta(seconds=OFI_DEPTH_STALE_SECS + 60),
            [(6.00, 9999)], [(6.01, 9999)],
        ))
        assert resumed is None  # fresh reseed, no spurious mega-delta

    def test_depth_telemetry(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 500)]))
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 100)], [(5.01, 300)]))
        assert t.book_depth_avg == (800 + 400) / 2
        assert t.ofi_normalized != 0.0 or t.ofi_rolling == 0


class TestEmaAcceleration:
    def _steady(self, t: OrderFlowTracker, start: datetime, secs: float, rate: float) -> datetime:
        ts = start
        n = int(secs * rate)
        for i in range(n):
            ts = start + timedelta(seconds=(i + 1) / rate)
            t.on_timesale(_print(ts, 5.01, 5.00, 5.01))
        return ts

    def test_not_ready_below_neff_floor(self):
        t = OrderFlowTracker()
        for i in range(int(TAPE_ACCEL_EMA_MIN_NEFF) - 5):
            t.on_timesale(_print(T0 + timedelta(seconds=i), 5.01, 5.00, 5.01))
        assert t.pressure_acceleration_ema_ready is False
        assert t.pressure_acceleration_ema == 0.0
        assert t.pressure_acceleration_ema_z == 0.0

    def test_steady_tape_reads_near_zero(self):
        t = OrderFlowTracker()
        self._steady(t, T0, secs=30, rate=2.0)
        assert t.pressure_acceleration_ema_ready
        assert abs(t.pressure_acceleration_ema) < 0.12

    def test_burst_fires_positive_and_significant(self):
        t = OrderFlowTracker()
        end = self._steady(t, T0, secs=30, rate=2.0)
        for i in range(20):  # 20 prints in 1s
            t.on_timesale(_print(end + timedelta(seconds=0.05 * (i + 1)), 5.01, 5.00, 5.01))
        assert t.pressure_acceleration_ema > 0.25
        assert t.pressure_acceleration_ema_z > 2.0

    def test_silence_decays_negative_without_new_prints(self):
        t = OrderFlowTracker()
        end = self._steady(t, T0, secs=30, rate=3.0)
        # A quote (not a print) advances event time — read decays to it.
        t.on_quote(_quote(end + timedelta(seconds=8), 5.00, 100, 5.01, 100))
        assert t.pressure_acceleration_ema < -0.2

    def test_read_is_idempotent(self):
        t = OrderFlowTracker()
        self._steady(t, T0, secs=20, rate=3.0)
        first = t.pressure_acceleration_ema
        assert t.pressure_acceleration_ema == first
        assert t.pressure_acceleration_ema == first

    def test_out_of_order_print_does_not_rewind(self):
        t = OrderFlowTracker()
        end = self._steady(t, T0, secs=20, rate=3.0)
        t.on_timesale(_print(end - timedelta(seconds=5), 5.01, 5.00, 5.01))
        assert t._ema_last_ts == end  # decay clock never rewinds
        assert t.pressure_acceleration_ema_n_eff > 0

    def test_mid_prints_do_not_feed_ema(self):
        t = OrderFlowTracker()
        n = t._ema_c_slow
        t.on_timesale(_print(T0, 5.005, 5.00, 5.01))  # inside spread → "mid"
        assert t._ema_c_slow == n


class TestAccelGateSwitch:
    def test_default_is_2bin(self, monkeypatch):
        monkeypatch.setattr(of_mod, "TAPE_ACCEL_ESTIMATOR", "2bin")
        t = OrderFlowTracker()
        for i in range(20):
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.4), 5.01, 5.00, 5.01))
        assert t.accel_gate_ready == t.pressure_acceleration_ready
        assert t.accel_gate_raw == t.pressure_acceleration
        assert t.accel_gate_z == t.pressure_acceleration_z

    def test_ema_selected(self, monkeypatch):
        monkeypatch.setattr(of_mod, "TAPE_ACCEL_ESTIMATOR", "ema")
        t = OrderFlowTracker()
        for i in range(40):
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.4), 5.01, 5.00, 5.01))
        assert t.accel_gate_ready == t.pressure_acceleration_ema_ready
        assert t.accel_gate_raw == t.pressure_acceleration_ema
        assert t.accel_gate_z == t.pressure_acceleration_ema_z


class TestSpeedGateSwitch:
    def test_ema_is_default_source(self, monkeypatch):
        monkeypatch.setattr(of_mod, "TAPE_SPEED_GATE_SOURCE", "ema")
        t = OrderFlowTracker()
        for i in range(40):
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.4), 5.01, 5.00, 5.01))
        assert t.speed_gate == t.tape_speed_ema
        assert t.speed_gate > 0.0

    def test_boxcar_fallback(self, monkeypatch):
        monkeypatch.setattr(of_mod, "TAPE_SPEED_GATE_SOURCE", "boxcar")
        t = OrderFlowTracker()
        for i in range(40):
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.4), 5.01, 5.00, 5.01))
        assert t.speed_gate == t.trade_speed

    def test_ema_arms_on_burst_before_boxcar(self, monkeypatch):
        """A 2s burst at ~20 prints/s must clear a 10/s floor on the EMA read
        while the 10s boxcar still averages it below the floor."""
        monkeypatch.setattr(of_mod, "TAPE_SPEED_GATE_SOURCE", "ema")
        t = OrderFlowTracker()
        for i in range(40):  # 40 prints over 2s → 20/s instantaneous
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.05), 5.01, 5.00, 5.01))
        assert t.speed_gate >= 10.0
        assert t.trade_speed < 10.0  # 40 prints / 10s window = 4/s


class TestRatioGateSwitch:
    def test_ema_is_default_source(self, monkeypatch):
        monkeypatch.setattr(of_mod, "TAPE_RATIO_GATE_SOURCE", "ema")
        t = OrderFlowTracker()
        for i in range(40):
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.2), 5.01, 5.00, 5.01))
        assert t.ratio_gate == t.buy_ratio_ema

    def test_boxcar_fallback(self, monkeypatch):
        monkeypatch.setattr(of_mod, "TAPE_RATIO_GATE_SOURCE", "boxcar")
        t = OrderFlowTracker()
        for i in range(40):
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.2), 5.01, 5.00, 5.01))
        assert t.ratio_gate == t.buy_ratio

    def test_empty_kernel_reads_neutral(self):
        t = OrderFlowTracker()
        assert t.buy_ratio_ema == 0.5

    def test_all_buys_reads_one(self):
        t = OrderFlowTracker()
        for i in range(20):
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.1), 5.01, 5.00, 5.01))
        assert t.buy_ratio_ema == 1.0

    def test_fast_kernel_tracks_recent_flip_before_boxcar(self):
        """8s of sells then a 2s buy burst: the fast kernel must read
        buyer-dominant while the 10s boxcar is still majority-sell."""
        t = OrderFlowTracker()
        for i in range(80):  # sells at 10/s for 8s (last <= bid)
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.1), 5.00, 5.00, 5.01))
        for i in range(40):  # buy burst at 20/s for 2s
            t.on_timesale(_print(T0 + timedelta(seconds=8.0 + i * 0.05), 5.01, 5.00, 5.01))
        assert t.buy_ratio < 0.5          # boxcar: 40 buys / 120 prints
        assert t.buy_ratio_ema >= 0.60    # fast kernel: burst dominates

    def test_reseed_rebuilds_buy_kernel(self):
        """Deque-constructed tracker + reseed must match live ingestion."""
        live = OrderFlowTracker()
        stamps: list[tuple[datetime, bool]] = []
        for i in range(60):
            ts = T0 + timedelta(seconds=i * 0.15)
            is_buy = (i % 3) != 0  # 2/3 buys
            stamps.append((ts, is_buy))
            if is_buy:
                live.on_timesale(_print(ts, 5.01, 5.00, 5.01))
            else:
                live.on_timesale(_print(ts, 5.00, 5.00, 5.01))

        seeded = OrderFlowTracker()
        for ts, is_buy in stamps:
            (seeded._buy_prints if is_buy else seeded._sell_prints).append(ts)
        seeded._baseline_prints.extend(ts for ts, _ in stamps)
        seeded.reseed_ema_from_prints(now=stamps[-1][0])

        assert abs(seeded.buy_ratio_ema - live.buy_ratio_ema) < 1e-9
        assert abs(seeded.tape_speed_ema - live.tape_speed_ema) < 1e-9


class TestOfiEmaNormalized:
    def test_zero_until_depth_samples(self):
        t = OrderFlowTracker()
        assert t.ofi_normalized_ema == 0.0
        # L1-only flow (no depth feed) contributes flow but no depth mass —
        # the normalized read stays 0, same convention as the boxcar.
        t.on_quote(_quote(T0, 5.00, 300, 5.01, 400))
        t.on_quote(_quote(T0 + timedelta(seconds=1), 5.00, 500, 5.01, 400))
        assert t.ofi_normalized_ema == 0.0

    def test_positive_flow_reads_positive(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 1000)], [(5.01, 1000)]))
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 1800)], [(5.01, 1000)]))
        assert t.ofi_normalized_ema > 0.2
        assert t.ofi_normalized > 0.2  # boxcar agrees on fresh flow

    def test_decays_on_read_while_boxcar_holds(self):
        """The gate-source fix: after flow dies, the EMA read ages out while
        the 30s boxcar keeps quoting the stale burst."""
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 1000)], [(5.01, 1000)]))
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 1800)], [(5.01, 1000)]))
        fresh = t.ofi_normalized_ema
        boxcar_fresh = t.ofi_normalized
        # A quote 20s later advances event time; no new book flow arrives.
        t.on_quote(_quote(T0 + timedelta(seconds=21), 5.00, 300, 5.01, 400))
        assert t.ofi_normalized_ema < fresh * 0.25   # e^-2 ≈ 0.135
        assert t.ofi_normalized == boxcar_fresh      # boxcar squats

    def test_read_is_idempotent(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 1000)], [(5.01, 1000)]))
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 1500)], [(5.01, 1000)]))
        first = t.ofi_normalized_ema
        assert t.ofi_normalized_ema == first
        assert t.ofi_normalized_ema == first

    def test_out_of_order_book_does_not_rewind(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 1000)], [(5.01, 1000)]))
        t.on_book(_book(T0 + timedelta(seconds=2), [(5.00, 1500)], [(5.01, 1000)]))
        clock = t._ofi_ema_last_ts
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 1600)], [(5.01, 1000)]))
        assert t._ofi_ema_last_ts == clock


class TestOfiGateSwitch:
    def _flowing(self) -> OrderFlowTracker:
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 1000)], [(5.01, 1000)]))
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 1500)], [(5.01, 1000)]))
        return t

    def test_ema_is_default_source(self, monkeypatch):
        monkeypatch.setattr(of_mod, "OFI_GATE_SOURCE", "ema")
        t = self._flowing()
        assert t.ofi_norm_gate == t.ofi_normalized_ema

    def test_boxcar_fallback(self, monkeypatch):
        monkeypatch.setattr(of_mod, "OFI_GATE_SOURCE", "boxcar")
        t = self._flowing()
        assert t.ofi_norm_gate == t.ofi_normalized


class TestOfiExitConditioning:
    """evaluate_exit exit 2: adverse OFI alone must NOT exit while the tape
    is still buy-dominant (ratio-lapse condition, hysteresis under the 0.60
    entry floor)."""

    @staticmethod
    def _entry():
        return SimpleNamespace(
            eval_confidence=80.0, live_confidence=80.0, has_live_data=True,
        )

    def _adverse_book_tracker(self) -> OrderFlowTracker:
        """Depth-active tracker with strongly negative normalized OFI
        (bid pulled + ask stacked)."""
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 1000)], [(5.01, 1000)]))
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.00, 200)], [(5.01, 2200)]))
        return t

    def test_no_exit_while_buyers_dominant(self):
        from exec.signals import evaluate_exit

        t = self._adverse_book_tracker()
        for i in range(10):  # buys: ratio_gate → 1.0
            t.on_timesale(_print(T0 + timedelta(seconds=1.1 + i * 0.05), 5.01, 5.00, 5.01))
        assert t.ofi_norm_gate < -0.75
        assert t.ratio_gate > 0.5
        sig = evaluate_exit(self._entry(), t, last_price=5.01, entry_price=5.00, last_atr=0.1)
        assert sig.triggered is False

    def test_exit_when_ratio_also_lapsed(self):
        from exec.signals import evaluate_exit

        t = self._adverse_book_tracker()
        for i in range(10):  # sells: ratio_gate → 0.0
            t.on_timesale(_print(T0 + timedelta(seconds=1.1 + i * 0.05), 5.00, 5.00, 5.01))
        assert t.ofi_norm_gate < -0.75
        assert t.ratio_gate <= 0.5
        sig = evaluate_exit(self._entry(), t, last_price=5.00, entry_price=5.00, last_atr=0.1)
        assert sig.triggered is True
        assert sig.reason.startswith("ofi_negative")


class TestPriceProgress:
    def test_not_ready_on_cold_tape(self):
        """<10s of directional history → None → entry gate fails closed."""
        t = OrderFlowTracker()
        for i in range(30):  # 3s burst from nothing
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.1), 5.01, 5.00, 5.01))
        assert t.price_progress is None

    def test_full_window_progress(self):
        t = OrderFlowTracker()
        # 20s of rising prints: 5.00 -> 5.20 linearly
        for i in range(41):
            p = 5.00 + 0.005 * i
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.5), p, p - 0.01, p))
        # reference ~15s ago: price ≈ 5.05 → progress ≈ 5.20/5.05 - 1 ≈ +3%
        prog = t.price_progress
        assert prog is not None and 0.02 < prog < 0.04

    def test_partial_window_uses_oldest(self):
        t = OrderFlowTracker()
        # 12s of history (>= 10s floor, < 15s window): oldest is the reference
        for i in range(25):
            p = 5.00 + 0.004 * i
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.5), p, p - 0.01, p))
        prog = t.price_progress
        ref = t._progress_samples[0][1]   # oldest slot (holds its second's last price)
        assert prog is not None
        assert abs(prog - (t._progress_samples[-1][1] / ref - 1.0)) < 1e-4

    def test_flat_churn_reads_zero(self):
        t = OrderFlowTracker()
        for i in range(41):  # 20s of alternating prints around 5.00
            p = 5.00 if i % 2 else 5.01
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.5), p, 4.99, 5.01))
        prog = t.price_progress
        assert prog is not None and abs(prog) < 0.005

    def test_samples_are_decimated(self):
        t = OrderFlowTracker()
        for i in range(200):  # 200 prints in 2s → ~3 slots (1s decimation)
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.01), 5.01, 5.00, 5.01))
        assert len(t._progress_samples) <= 4


class _FakeRMap:
    """Minimal stand-in: sorted ascending price levels, strict nearest-above."""

    def __init__(self, prices: list[float]) -> None:
        self._prices = sorted(prices)

    def nearest_above(self, price: float):
        for p in self._prices:
            if p > price:
                return SimpleNamespace(price=p)
        return None


class TestResistanceApproachExit:
    """evaluate_exit exit 4: own-ceiling exemption + tuned flow reads.

    The insta-sell this guards (YXT 2026-08-05 replay): entry at 5.18 under
    setup level 5.19, price dips 3.5c in 159ms onto the 1m-EMA cluster, old
    exit 4 fired on 2c of 'headroom' + net_pressure<=0 while deep OFI read
    +26k shares of net buying.
    """

    @staticmethod
    def _entry():
        return SimpleNamespace(
            eval_confidence=80.0, live_confidence=80.0, has_live_data=True,
        )

    def _lapsed_tracker(self) -> OrderFlowTracker:
        """A few sells only: ratio_gate lapsed, but too little activity or
        accel readiness for exits 1/1b, no depth for exit 2."""
        t = OrderFlowTracker()
        for i in range(8):
            t.on_timesale(_print(T0 + timedelta(seconds=i * 0.4), 5.14, 5.14, 5.15))
        return t

    def test_own_ceiling_exempted(self):
        """Levels at/under the setup's entry-band top (its own breakout
        structure — prior high, 1m EMAs, VWAP) must not trigger the exit."""
        from exec.signals import evaluate_exit

        rmap = _FakeRMap([5.15, 5.19])
        t = self._lapsed_tracker()
        # Legacy path (no captured level): the 5.15 micro-level 0.3c above
        # price fires the exit — this is the insta-sell shape.
        sig = evaluate_exit(self._entry(), t, last_price=5.147, entry_price=5.18,
                            last_atr=0.5, rmap=rmap, entry_setup_level=None)
        assert sig.triggered and sig.reason.startswith("resistance_approach")
        # With the setup level captured at fill: both levels sit inside the
        # trade's own band (top 5.19*1.02=5.294) — exempt, no exit.
        sig = evaluate_exit(self._entry(), t, last_price=5.147, entry_price=5.18,
                            last_atr=0.5, rmap=rmap, entry_setup_level=5.19)
        assert sig.triggered is False

    def test_distinct_wall_above_band_still_exits(self):
        from exec.signals import evaluate_exit

        rmap = _FakeRMap([5.15, 5.19, 5.60])
        t = self._lapsed_tracker()
        sig = evaluate_exit(self._entry(), t, last_price=5.58, entry_price=5.18,
                            last_atr=0.5, rmap=rmap, entry_setup_level=5.19)
        assert sig.triggered and sig.reason.startswith("resistance_approach")

    def test_boxcar_sell_flurry_no_longer_acknowledges(self):
        """The YXT failure: 10s boxcar net_pressure negative (old sells) but
        the fast-EMA ratio buyer-dominant — tuned reads must hold the exit."""
        from exec.signals import evaluate_exit

        t = OrderFlowTracker()
        base = T0
        for i in range(28):   # sells 8s->4s ago (dominate the boxcar counts)
            t.on_timesale(_print(base + timedelta(seconds=i * (4.0 / 28)), 5.55, 5.55, 5.56))
        for i in range(20):   # buy burst in the last 1.5s (dominates fast EMA)
            t.on_timesale(_print(base + timedelta(seconds=6.5 + i * 0.075), 5.58, 5.57, 5.58))
        assert t.buying_pressure - t.selling_pressure < 0   # old term fired here
        assert t.ratio_gate > 0.5                            # tuned term holds
        rmap = _FakeRMap([5.60])
        sig = evaluate_exit(self._entry(), t, last_price=5.58, entry_price=5.18,
                            last_atr=0.5, rmap=rmap, entry_setup_level=5.19)
        assert sig.triggered is False

    def test_adverse_book_flow_acknowledges(self):
        """Near a real wall with buy-dominant tape but negative normalized
        book flow, the OFI term confirms the exit."""
        from exec.signals import evaluate_exit

        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.55, 1000)], [(5.58, 1000)]))
        t.on_book(_book(T0 + timedelta(seconds=1), [(5.55, 300)], [(5.58, 2400)]))
        for i in range(10):  # buys: ratio_gate ~1.0, so exit 2 stays silent
            t.on_timesale(_print(T0 + timedelta(seconds=1.1 + i * 0.05), 5.58, 5.57, 5.58))
        assert t.depth_active and t.ofi_norm_gate < 0
        rmap = _FakeRMap([5.60])
        sig = evaluate_exit(self._entry(), t, last_price=5.58, entry_price=5.18,
                            last_atr=0.5, rmap=rmap, entry_setup_level=5.19)
        assert sig.triggered and sig.reason.startswith("resistance_approach")


class TestScorerOnBook:
    """DynamicScorer._on_book — density criterion + tracker feed (unbound-call, no fixture)."""

    @staticmethod
    def _stub():
        import threading
        from config import CRITERIA_ATTENTION
        from strategy.confidence_scorer import DynamicScorer, _NullChartCache
        from strategy.criteria import build_criteria_groups

        groups = build_criteria_groups(CRITERIA_ATTENTION)
        entry = SimpleNamespace(
            criteria_groups=groups,
            get_criteria_group=lambda name: next((g for g in groups if g.name == name), None),
            setup_price_level=None,
            setup_pullback_low=None,
            setup_pullback_ts=None,
        )
        stub = SimpleNamespace(
            _lock=threading.Lock(),
            _ofi_tracker=OrderFlowTracker(),
            _entry=entry,
            _chart_cache=_NullChartCache(),
        )
        # Bind the real band push so the stub exercises production code.
        stub._push_entry_band_to_tracker = (
            lambda: DynamicScorer._push_entry_band_to_tracker(stub)
        )
        return stub

    def test_thick_book_passes_density(self):
        from strategy.confidence_scorer import DynamicScorer

        stub = self._stub()
        DynamicScorer._on_book(stub, _book(T0, [(5.00, 2000)], [(5.01, 2000)]))
        crit = next(
            c for c in stub._entry.get_criteria_group("ATTENTION").criteria
            if c.name == "order_book_density_ok"
        )
        assert crit.passing is True
        assert stub._ofi_tracker.depth_active is True

    def test_thin_book_fails_density(self):
        from strategy.confidence_scorer import DynamicScorer

        stub = self._stub()
        DynamicScorer._on_book(stub, _book(T0, [(5.00, 50)], [(5.01, 40)]))
        crit = next(
            c for c in stub._entry.get_criteria_group("ATTENTION").criteria
            if c.name == "order_book_density_ok"
        )
        assert crit.passing is False

    def test_one_sided_book_fails_density(self):
        from strategy.confidence_scorer import DynamicScorer

        stub = self._stub()
        DynamicScorer._on_book(stub, _book(T0, [(5.00, 5000)], []))
        crit = next(
            c for c in stub._entry.get_criteria_group("ATTENTION").criteria
            if c.name == "order_book_density_ok"
        )
        assert crit.passing is False


class TestBacktestEmaReseed:
    def test_reseed_replays_prints_through_ema(self):
        from exec.backtest import BacktestRunner, CachedTapeSnapshot
        from pathlib import Path

        snapshot = CachedTapeSnapshot(
            symbol="TEST",
            ts=T0,
            price=5.00,
            entry_speed=12.0,
            entry_buy_ratio=0.7,
            pressure_acceleration=0.30,
            pressure_acceleration_ready=True,
            buying_pressure=84,
            selling_pressure=36,
            buy_vol=8400,
            sell_vol=3600,
            avg_print_size=100.0,
            source_path=Path("tape_test.csv"),
        )
        scorer = SimpleNamespace(_ofi_tracker=OrderFlowTracker())
        BacktestRunner._apply_cached_tape_snapshot(scorer, snapshot)
        ofi = scorer._ofi_tracker
        # 2-bin state reproduced (existing behavior)
        assert ofi.pressure_acceleration_ready
        assert ofi.pressure_acceleration > 0
        # EMA accumulators reseeded from the same reconstructed print train:
        # accelerating shape (denser recent half) must read positive too.
        assert ofi._ema_last_ts is not None
        assert ofi.pressure_acceleration_ema_ready
        assert ofi.pressure_acceleration_ema > 0


class TestBookCache:
    """book cache round-trip: snapshot → CSV row → loaded DataFrame."""

    def test_round_trip_preserves_ladders_and_order(self, tmp_path):
        from chart_cache import append_book_rows, book_row_from_snapshot, load_book_cache

        snaps = [
            _book(T0 + timedelta(seconds=1), [(5.00, 300), (4.99, 200)], [(5.01, 400)]),
            _book(T0, [(5.00, 250)], [(5.01, 350), (5.02, 100)]),  # out of order on purpose
        ]
        append_book_rows([book_row_from_snapshot(s) for s in snaps], log_dir=tmp_path)

        df = load_book_cache("TEST", T0 - timedelta(minutes=1), T0 + timedelta(minutes=1), tmp_path)
        assert len(df) == 2
        # Sorted by ts despite append order
        assert df.iloc[0]["ts"].to_pydatetime() == T0
        assert df.iloc[0]["bids"] == [[5.00, 250]]
        assert df.iloc[0]["asks"] == [[5.01, 350], [5.02, 100]]
        assert df.iloc[1]["bids"] == [[5.00, 300], [4.99, 200]]

    def test_window_and_symbol_filters(self, tmp_path):
        from chart_cache import append_book_rows, book_row_from_snapshot, load_book_cache

        snap = _book(T0, [(5.00, 300)], [(5.01, 400)])
        other = _book(T0, [(9.00, 100)], [(9.01, 100)])
        other.symbol = "OTHER"
        append_book_rows(
            [book_row_from_snapshot(snap), book_row_from_snapshot(other)], log_dir=tmp_path,
        )
        # Outside window → empty
        df = load_book_cache("TEST", T0 + timedelta(hours=1), T0 + timedelta(hours=2), tmp_path)
        assert df.empty
        # Other symbol not leaked
        df = load_book_cache("TEST", T0 - timedelta(minutes=1), T0 + timedelta(minutes=1), tmp_path)
        assert len(df) == 1 and df.iloc[0]["symbol"] == "TEST"

    def test_missing_file_returns_empty(self, tmp_path):
        from chart_cache import load_book_cache

        df = load_book_cache("NOPE", T0, T0 + timedelta(minutes=1), tmp_path)
        assert df.empty

    def test_live_chart_cache_buffers_and_flushes(self, tmp_path):
        from chart_cache import LiveChartCache, book_cache_path

        cache = LiveChartCache("TEST", log_dir=tmp_path, raw_flush_size=10, raw_flush_seconds=3600.0)
        cache.record_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        path = book_cache_path(tmp_path, "TEST", T0.strftime("%Y%m%d"))
        assert not path.exists()  # buffered, below flush size
        cache.flush_books()
        assert path.exists()
        assert path.read_text().count("\n") == 2  # header + one row


class TestBacktestBookReplay:
    """_load_book_snapshots rebuilds production OrderBookSnapshot objects."""

    def _runner(self, tmp_path):
        from exec.backtest import BacktestRunner

        return BacktestRunner(
            cache_path=tmp_path / "ticker_cache.pkl",
            eval_stats_path=tmp_path / "missing.log",
            log_dir=tmp_path,
        )

    def test_load_book_snapshots_rebuilds_levels(self, tmp_path):
        from chart_cache import append_book_rows, book_row_from_snapshot

        append_book_rows(
            [book_row_from_snapshot(_book(T0, [(5.00, 300), (4.99, 200)], [(5.01, 400)]))],
            log_dir=tmp_path,
        )
        runner = self._runner(tmp_path)
        snaps = runner._load_book_snapshots("TEST", T0 - timedelta(minutes=1), T0 + timedelta(minutes=1))
        assert len(snaps) == 1
        snap = snaps[0]
        assert snap.ts == T0
        assert [(l.price, l.size, l.position) for l in snap.bids] == [(5.00, 300, 0), (4.99, 200, 1)]
        assert [(l.price, l.size) for l in snap.asks] == [(5.01, 400)]
        # Feeding the rebuilt snapshots through the tracker works end-to-end
        trk = OrderFlowTracker()
        trk.on_book(snap)
        assert trk.depth_active is True

    def test_load_book_snapshots_empty_when_no_capture(self, tmp_path):
        runner = self._runner(tmp_path)
        assert runner._load_book_snapshots("TEST", T0, T0 + timedelta(minutes=1)) == []

    def test_replayed_ofi_matches_direct_feed(self, tmp_path):
        """Parity: cache→load→feed produces identical OFI to direct feeding."""
        from chart_cache import append_book_rows, book_row_from_snapshot

        books = [
            _book(T0, [(5.00, 300)], [(5.01, 400)]),
            _book(T0 + timedelta(seconds=1), [(5.00, 450)], [(5.01, 400)]),   # bid stacking
            _book(T0 + timedelta(seconds=2), [(5.00, 450)], [(5.01, 250)]),   # ask pulling
            _book(T0 + timedelta(seconds=3), [(5.00, 200), (4.99, 500)], [(5.02, 300)]),
        ]
        direct = OrderFlowTracker()
        for b in books:
            direct.on_book(b)

        append_book_rows([book_row_from_snapshot(b) for b in books], log_dir=tmp_path)
        runner = self._runner(tmp_path)
        replayed = OrderFlowTracker()
        for snap in runner._load_book_snapshots("TEST", T0 - timedelta(minutes=1), T0 + timedelta(minutes=1)):
            replayed.on_book(snap)

        assert replayed.ofi_rolling == direct.ofi_rolling
        assert replayed.book_depth_avg == direct.book_depth_avg
        assert replayed.depth_active is direct.depth_active


class TestGatePredicateParity:
    """Armed-band Step 2: the tracker gate predicates ARE the gate definition.

    tape_entry_gates() consumes them, and the latch stamps will too — this
    parity sweep pins the predicates to the exact inline comparisons the
    F10 surface used to make, so the refactor cannot have drifted them.
    """

    def _reference(self, t: OrderFlowTracker) -> dict[str, bool]:
        from config import (
            SIM_PRESSURE_SPEED_MIN,
            SIM_PRESSURE_ACTIVITY_MIN,
            SIM_PRESSURE_BUY_RATIO_MIN,
            SIM_PRESSURE_VOL_IMBALANCE_MIN,
            SIM_PRESSURE_ACCEL_Z,
            SIM_PRESSURE_PROGRESS_MIN,
            SIM_OFI_NORM_MIN,
        )
        progress = t.price_progress
        return {
            "speed": t.speed_gate >= SIM_PRESSURE_SPEED_MIN,
            "activity": t.activity_count >= SIM_PRESSURE_ACTIVITY_MIN,
            "ratio": t.ratio_gate >= SIM_PRESSURE_BUY_RATIO_MIN,
            "vol_flow": t.trade_imbalance > SIM_PRESSURE_VOL_IMBALANCE_MIN,
            "accel": t.accel_gate_ready and t.accel_gate_z >= SIM_PRESSURE_ACCEL_Z,
            "progress": progress is not None and progress > SIM_PRESSURE_PROGRESS_MIN,
            "ofi": (not t.depth_active) or t.ofi_norm_gate >= SIM_OFI_NORM_MIN,
        }

    def _assert_parity(self, t: OrderFlowTracker) -> None:
        from exec.signals import tape_entry_gates

        ref = self._reference(t)
        assert t.speed_gate_ok() == ref["speed"]
        assert t.activity_ok() == ref["activity"]
        assert t.ratio_gate_ok() == ref["ratio"]
        assert t.vol_flow_ok() == ref["vol_flow"]
        assert t.accel_gate_ok() == ref["accel"]
        assert t.progress_gate_ok() == ref["progress"]
        assert t.ofi_gate_ok() == ref["ofi"]

        gates = tape_entry_gates(t)
        assert gates.speed_ok == ref["speed"]
        assert gates.activity_ok == ref["activity"]
        assert gates.ratio_ok == ref["ratio"]
        assert gates.vol_flow_ok == ref["vol_flow"]
        assert gates.accel_ok == ref["accel"]
        assert gates.progress_ok == ref["progress"]

    def test_parity_empty_tracker(self):
        self._assert_parity(OrderFlowTracker())

    def test_parity_across_tape_mixes(self):
        for n_buys, n_sells in [(5, 5), (60, 10), (200, 20), (10, 200)]:
            t = OrderFlowTracker()
            ts = T0
            for _ in range(n_buys):
                ts += timedelta(milliseconds=97)
                t.on_timesale(_print(ts, 5.01, 5.00, 5.01))
            for _ in range(n_sells):
                ts += timedelta(milliseconds=83)
                t.on_timesale(_print(ts, 5.00, 5.00, 5.01))
            self._assert_parity(t)

    def test_parity_with_depth_and_progress(self):
        t = OrderFlowTracker()
        t.on_book(_book(T0, [(5.00, 300)], [(5.01, 400)]))
        ts = T0
        price = 5.01
        for i in range(120):
            ts += timedelta(milliseconds=110)
            price += 0.001
            t.on_timesale(_print(ts, price, price - 0.01, price))
            if i % 10 == 0:
                t.on_book(_book(ts, [(price - 0.01, 300 + i)], [(price, 400)]))
            if i % 25 == 0:
                self._assert_parity(t)
        self._assert_parity(t)


class TestArmedBandLatches:
    """Step-3 armed-band machinery: band push, latch stamping, trigger edges."""

    def _tracker_with_band(self, level=10.0, floor=9.0):
        t = OrderFlowTracker()
        t.set_entry_band(level, floor, T0)
        return t

    def _feed_burst(self, t, price, n=80, gap_ms=50, start=None):
        ts = start or T0
        for _ in range(n):
            ts += timedelta(milliseconds=gap_ms)
            t.on_timesale(_print(ts, price, price - 0.01, price))
        return ts

    def test_no_stamps_without_band(self):
        t = OrderFlowTracker()
        self._feed_burst(t, 9.5)
        assert t.gate_last_true_age("speed") is None
        assert t.entry_band is None

    def test_stamps_in_band(self):
        t = self._tracker_with_band()
        self._feed_burst(t, 9.5)  # 20 buys/s, in [9.0, 10.5]
        age = t.gate_last_true_age("speed")
        assert age is not None and age < 1.0
        assert t.gate_last_true_age("ratio") is not None
        assert t.gate_last_true_age("activity") is not None
        assert t.gate_last_true_age("vol_flow") is not None

    def test_floor_dip_disarms(self):
        t = self._tracker_with_band()
        end = self._feed_burst(t, 9.5)
        assert t.gate_last_true_age("speed") is not None
        t.on_timesale(_print(end + timedelta(milliseconds=50), 8.5, 8.49, 8.5))
        assert t.gate_last_true_age("speed") is None
        assert t.microbreak_age() is None

    def test_no_stamps_above_band_top(self):
        t = self._tracker_with_band()  # top = 10.0 * 1.05 = 10.5
        self._feed_burst(t, 11.0)
        assert t.gate_last_true_age("speed") is None

    def test_setup_identity_change_resets(self):
        t = self._tracker_with_band()
        self._feed_burst(t, 9.5)
        assert t.gate_last_true_age("speed") is not None
        # Same identity re-push: latches survive
        t.set_entry_band(10.0, 9.0, T0)
        assert t.gate_last_true_age("speed") is not None
        # New pullback_ts = new setup instance: reset
        t.set_entry_band(10.0, 9.0, T0 + timedelta(minutes=1))
        assert t.gate_last_true_age("speed") is None
        # Re-stamp, then level change: reset
        self._feed_burst(t, 9.5, start=T0 + timedelta(seconds=10))
        assert t.gate_last_true_age("speed") is not None
        t.set_entry_band(12.0, 10.8, T0 + timedelta(minutes=1))
        assert t.gate_last_true_age("speed") is None
        # None clears the band entirely
        t.set_entry_band(None, None, None)
        assert t.entry_band is None

    def test_ofi_stamp_requires_live_depth(self):
        t = self._tracker_with_band()
        self._feed_burst(t, 9.5)
        # No depth feed: ofi_gate_ok() is vacuously True but must NOT stamp —
        # the consumer applies the vacuous pass itself when depth is inactive.
        assert t.ofi_gate_ok() is True
        assert t.gate_last_true_age("ofi") is None

    def test_accel_rising_edge_semantics(self, monkeypatch):
        t = self._tracker_with_band()
        t.on_timesale(_print(T0, 9.5, 9.49, 9.5))  # sets last trade price in band
        monkeypatch.setattr(OrderFlowTracker, "accel_gate_ok", lambda self: True)
        t._touch_event_ts(T0 + timedelta(seconds=1))
        t._update_armed_latches(T0 + timedelta(seconds=1))
        first_edge = t._accel_edge_ts
        assert first_edge == T0 + timedelta(seconds=1)
        # Still true on the next sweep: edge does NOT re-stamp
        t._touch_event_ts(T0 + timedelta(seconds=2))
        t._update_armed_latches(T0 + timedelta(seconds=2))
        assert t._accel_edge_ts == first_edge
        # False, then true again: fresh edge
        monkeypatch.setattr(OrderFlowTracker, "accel_gate_ok", lambda self: False)
        t._update_armed_latches(T0 + timedelta(seconds=3))
        monkeypatch.setattr(OrderFlowTracker, "accel_gate_ok", lambda self: True)
        t._touch_event_ts(T0 + timedelta(seconds=4))
        t._update_armed_latches(T0 + timedelta(seconds=4))
        assert t._accel_edge_ts == T0 + timedelta(seconds=4)

    def test_microbreak_from_raw_prints(self):
        t = self._tracker_with_band()
        # 5s of 9.50 prints builds the trailing high with enough span
        end = self._feed_burst(t, 9.5, n=50, gap_ms=100)
        assert t.microbreak_age() is None
        t.on_timesale(_print(end + timedelta(milliseconds=30), 9.6, 9.59, 9.6))
        age = t.microbreak_age()
        assert age is not None and age < 0.5
        assert t.trailing_high == 9.6

    def test_microbreak_sub_second_burst_not_missed(self):
        # Two prints 30ms apart — a 1s-decimated series would fold them into
        # one slot and miss the break; the raw ratchet must not.
        t = self._tracker_with_band()
        end = self._feed_burst(t, 9.5, n=50, gap_ms=100)
        t.on_timesale(_print(end + timedelta(milliseconds=15), 9.52, 9.51, 9.52))
        t.on_timesale(_print(end + timedelta(milliseconds=45), 9.61, 9.60, 9.61))
        assert t.microbreak_age() is not None

    def test_microbreak_requires_history_span(self):
        t = self._tracker_with_band()
        t.on_timesale(_print(T0, 9.5, 9.49, 9.5))
        t.on_timesale(_print(T0 + timedelta(milliseconds=100), 9.6, 9.59, 9.6))
        assert t.microbreak_age() is None  # 100ms of history is not a level

    def test_reseed_resets_armed_state(self):
        t = self._tracker_with_band()
        self._feed_burst(t, 9.5)
        assert t.gate_last_true_age("speed") is not None
        t.reseed_ema_from_prints()
        assert t.gate_last_true_age("speed") is None
        assert t.trailing_high is None
        assert t.microbreak_age() is None


class TestAskWallFeature:
    """Step-4 level-anchored ask-wall: evaporation trend, guards, vacuous passes."""

    def _fed_tracker(self, mass=1000, n=12, level=10.0, floor=9.0):
        t = OrderFlowTracker()
        t.set_entry_band(level, floor, T0)
        ts = T0
        for _ in range(n):
            ts += timedelta(milliseconds=300)
            t.on_book(_book(ts, [(9.90, 500)], [(10.0, mass)]))
        return t, ts

    def test_vacuous_without_depth_or_band(self):
        t = OrderFlowTracker()
        assert t.wall_evidence_ok() is True   # no depth, no band
        t.set_entry_band(10.0, 9.0, T0)
        assert t.wall_evidence_ok() is True   # still no depth
        assert t.wall_trend is None

    def test_stable_wall_is_not_evaporation(self):
        t, _ = self._fed_tracker()
        assert t.wall_trend is not None
        assert abs(t.wall_trend) < 0.2
        assert t.wall_evidence_ok() is False  # real read, no evaporation

    def test_evaporation_passes(self):
        t, ts = self._fed_tracker(mass=1000)
        t.on_book(_book(ts + timedelta(milliseconds=300), [(9.90, 500)], [(10.0, 250)]))
        assert t.wall_trend is not None and t.wall_trend >= 0.5
        assert t.wall_evidence_ok() is True

    def test_replenishment_fails(self):
        t, ts = self._fed_tracker(mass=1000)
        t.on_book(_book(ts + timedelta(milliseconds=300), [(9.90, 500)], [(10.0, 2500)]))
        assert t.wall_trend is not None and t.wall_trend < 0
        assert t.wall_evidence_ok() is False

    def test_truncated_ladder_coverage_guard(self):
        t = OrderFlowTracker()
        t.set_entry_band(10.0, 9.0, T0)
        ts = T0
        # Full 10-row ladder whose deepest ask (9.09) sits below the wall
        # band top (10.1): the band scrolled off the visible view.
        asks = [(9.00 + 0.01 * i, 100) for i in range(10)]
        for _ in range(12):
            ts += timedelta(milliseconds=300)
            t.on_book(_book(ts, [(8.90, 500)], asks))
        assert t.wall_trend is None
        assert t.wall_evidence_ok() is True  # vacuous, never false evaporation

    def test_thin_history_is_vacuous(self):
        t = OrderFlowTracker()
        t.set_entry_band(10.0, 9.0, T0)
        t.on_book(_book(T0, [(9.90, 500)], [(10.0, 1000)]))
        t.on_book(_book(T0 + timedelta(milliseconds=300), [(9.90, 500)], [(10.0, 100)]))
        assert t.wall_trend is None  # baseline not established
        assert t.wall_evidence_ok() is True

    def test_level_change_resets_wall(self):
        t, ts = self._fed_tracker()
        assert t.wall_trend is not None
        t.set_entry_band(11.0, 9.9, T0)
        assert t.wall_trend is None

    def test_resumed_feed_restarts_baseline(self):
        t, ts = self._fed_tracker(mass=1000)
        # Gap beyond staleness: fresh book, baseline restarts — the first
        # post-gap sample must not read as evaporation vs the pre-gap wall.
        later = ts + timedelta(seconds=OFI_DEPTH_STALE_SECS + 10)
        t.on_book(_book(later, [(9.90, 500)], [(10.0, 100)]))
        assert t.wall_trend is None
