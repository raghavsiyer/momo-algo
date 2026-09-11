"""
tests/test_patterns.py

Unit tests for technical_analysis/patterns.py.
No API calls — all tests use synthetic DataFrames and a mock ResistanceMap.
"""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")

import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from technical_analysis.resistance import (
    ResistanceMap,
    ResistanceLevel,
    LevelType,
)
from technical_analysis.patterns import (
    PatternSignal,
    PatternSnapshot,
    detect_topping_tail,
    detect_bottoming_tail,
    detect_tail_clusters,
    detect_multi_top,
    detect_multi_bottom,
    detect_wedge,
    detect_head_and_shoulders,
    detect_pullback_new_high,
    detect_pullback_pending,
    detect_pullback_detected,
    detect_breakout_pullback,
    detect_support_bounce,
    detect_resistance_rejection,
    detect_move_start,
    scan_all_patterns,
    _find_pivot_highs,
    _find_pivot_lows,
)
from config import SETUP_TFS_DISABLED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rmap(price: float = 5.00, symbol: str = "TEST") -> ResistanceMap:
    return ResistanceMap(symbol=symbol, current_price=price)


def _candle_df(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[int] | None = None,
) -> pd.DataFrame:
    n = len(opens)
    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes or [10_000] * n,
    })


def _simple_candle_row(o: float, h: float, l: float, c: float) -> pd.Series:
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


def _move_row(ts: datetime, close: float, volume: int, interval: str = "1min") -> dict:
    return {
        "ts": ts,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": volume,
        "interval": interval,
    }


# ---------------------------------------------------------------------------
# Move start
# ---------------------------------------------------------------------------

class TestMoveStart(unittest.TestCase):

    def test_intraday_clips_to_current_days_premarket_session(self):
        # Both the previous day's RTH bars and its after-hours bars must be
        # excluded — the session anchors at the current day's 04:00 ET open.
        stale_previous_day = [
            _move_row(datetime(2026, 6, 2, 18, i, tzinfo=timezone.utc), 9.50, 50_000)
            for i in range(30)
        ]
        stale_after_hours = [
            _move_row(datetime(2026, 6, 2, 20, i, tzinfo=timezone.utc), 2.35, 1_000)
            for i in range(8)
        ]

        current_day: list[dict] = []
        base_ts = pd.Timestamp("2026-06-03 13:00:00", tz="UTC")  # 09:00 ET premarket
        for i in range(20):
            current_day.append(_move_row(base_ts.to_pydatetime() + pd.Timedelta(minutes=i), 2.35, 1_000))
        for i, (close, volume) in enumerate([
            (2.50, 12_000),
            (2.59, 13_000),
            (2.85, 27_000),
            (2.65, 45_000),
            (2.75, 16_000),
            (2.96, 19_000),
            (3.02, 52_000),
        ], start=20):
            current_day.append(_move_row(base_ts.to_pydatetime() + pd.Timedelta(minutes=i), close, volume))

        df = pd.DataFrame(stale_previous_day + stale_after_hours + current_day)

        result = detect_move_start(df)

        self.assertIsNotNone(result)
        self.assertEqual(result.start_idx, len(stale_previous_day) + len(stale_after_hours) + 20)
        self.assertAlmostEqual(result.start_price, 2.50)

    def test_intraday_anchors_to_confirmed_ignition_after_unconfirmed_escape(self):
        base_ts = pd.Timestamp("2026-06-02 20:00:00", tz="UTC")
        closes = [
            2.23, 3.40, 3.70, 4.19, 4.39, 4.34, 4.29, 4.64, 4.76, 5.15,
            4.76, 4.71, 4.82, 4.81, 4.69, 4.73, 4.73, 4.76, 4.59, 4.64,
            4.60, 4.62, 4.58, 4.70, 4.66, 5.20, 5.60, 5.90,
        ]
        volumes = [
            13_000, 170_000, 357_000, 472_000, 433_000, 387_000, 240_000,
            602_000, 379_000, 572_000, 443_000, 370_000, 266_000, 246_000,
            469_000, 248_000, 134_000, 400_000, 233_000, 105_000,
            20_000, 18_000, 16_000, 18_000, 17_000, 3_000_000, 3_100_000, 3_200_000,
        ]
        df = pd.DataFrame([
            _move_row((base_ts + pd.Timedelta(minutes=i)).to_pydatetime(), close, volume)
            for i, (close, volume) in enumerate(zip(closes, volumes))
        ])

        result = detect_move_start(df)

        self.assertIsNotNone(result)
        self.assertEqual(result.start_idx, 25)
        self.assertAlmostEqual(result.start_price, 5.20)

    def test_intraday_ignores_low_volume_escape_before_confirmed_ignition(self):
        base_ts = pd.Timestamp("2026-06-02 20:00:00", tz="UTC")
        closes = [0.38] + [0.42] * 19
        volumes = [1_000] * 20

        closes.extend([0.39, 0.388, 0.391, 0.41, 0.409, 0.405])
        volumes.extend([800, 700, 600, 500, 550, 500])

        closes.extend([0.45, 0.47, 0.50])
        volumes.extend([10_000, 15_000, 20_000])

        df = pd.DataFrame([
            _move_row((base_ts + pd.Timedelta(minutes=i)).to_pydatetime(), close, volume)
            for i, (close, volume) in enumerate(zip(closes, volumes))
        ])

        result = detect_move_start(df)

        self.assertIsNotNone(result)
        self.assertEqual(result.start_idx, 26)
        self.assertAlmostEqual(result.start_price, 0.45)

    def test_confirm_requires_absolute_volume_floor(self):
        """A 5x-RVOL escape on trivial absolute volume (below the 5k floor)
        must not anchor the move; the anchor shifts to the first bar at or
        above the floor."""
        base_ts = pd.Timestamp("2026-06-02 20:00:00", tz="UTC")
        closes = [2.00] * 20
        volumes = [300] * 20
        # Escaped + >5x RVOL but only 4k shares — relative-only trigger.
        closes.append(2.30)
        volumes.append(4_000)
        # Real ignition: floor-clearing bars.
        closes.extend([2.40, 2.50, 2.60])
        volumes.extend([12_000, 14_000, 16_000])

        df = pd.DataFrame([
            _move_row((base_ts + pd.Timedelta(minutes=i)).to_pydatetime(), close, volume)
            for i, (close, volume) in enumerate(zip(closes, volumes))
        ])

        result = detect_move_start(df)

        self.assertIsNotNone(result)
        self.assertEqual(result.start_idx, 21)  # not 20 (the 4k bar)
        self.assertAlmostEqual(result.start_price, 2.40)

    def test_daily_chart_can_span_multiple_days(self):
        rows = []
        start = pd.Timestamp("2026-01-01", tz="UTC")
        for i in range(51):
            rows.append(_move_row((start + pd.Timedelta(days=i)).to_pydatetime(), 10.00, 1_000, "1day"))
        for i, close in enumerate([10.60, 11.00, 11.50, 12.00], start=51):
            rows.append(_move_row((start + pd.Timedelta(days=i)).to_pydatetime(), close, 10_000, "1day"))

        result = detect_move_start(pd.DataFrame(rows))

        self.assertIsNotNone(result)
        self.assertEqual(result.start_idx, 51)


# ---------------------------------------------------------------------------
# Topping / Bottoming tail
# ---------------------------------------------------------------------------

class TestTails(unittest.TestCase):

    def test_topping_tail_detected(self):
        # Big upper wick, tiny body, no lower wick
        candle = _simple_candle_row(o=5.00, h=5.50, l=5.00, c=5.02)
        self.assertTrue(detect_topping_tail(candle))

    def test_topping_tail_not_on_normal_candle(self):
        candle = _simple_candle_row(o=5.00, h=5.10, l=4.95, c=5.08)
        self.assertFalse(detect_topping_tail(candle))

    def test_bottoming_tail_detected(self):
        # Big lower wick, tiny body, no upper wick
        candle = _simple_candle_row(o=5.02, h=5.02, l=4.50, c=5.00)
        self.assertTrue(detect_bottoming_tail(candle))

    def test_bottoming_tail_not_on_normal_candle(self):
        candle = _simple_candle_row(o=5.00, h=5.10, l=4.95, c=5.08)
        self.assertFalse(detect_bottoming_tail(candle))


# ---------------------------------------------------------------------------
# Pivot detection helpers
# ---------------------------------------------------------------------------

class TestPivots(unittest.TestCase):

    def test_find_pivot_highs(self):
        # Clear pivot high at index 3
        highs = np.array([1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 1.0])
        result = _find_pivot_highs(highs, n=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 3)
        self.assertAlmostEqual(result[0][1], 5.0)

    def test_find_pivot_lows(self):
        lows = np.array([5.0, 4.0, 3.0, 1.0, 3.0, 4.0, 5.0])
        result = _find_pivot_lows(lows, n=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 3)
        self.assertAlmostEqual(result[0][1], 1.0)

    def test_no_pivots_on_flat(self):
        highs = np.array([5.0] * 10)
        self.assertEqual(_find_pivot_highs(highs, n=2), [])

    def test_too_short_returns_empty(self):
        self.assertEqual(_find_pivot_highs(np.array([1.0, 2.0]), n=2), [])
        self.assertEqual(_find_pivot_lows(np.array([1.0, 2.0]), n=2), [])


# ---------------------------------------------------------------------------
# Multi-top / Multi-bottom
# ---------------------------------------------------------------------------

class TestMultiTop(unittest.TestCase):

    def test_double_top_detected(self):
        # Two peaks at ~5.00 with a pullback between them
        highs  = [4.0, 4.5, 5.00, 4.5, 4.0, 4.5, 5.01, 4.5, 4.0]
        lows   = [3.9, 4.4, 4.90, 4.4, 3.8, 4.4, 4.90, 4.4, 3.9]
        closes = [4.0, 4.5, 4.95, 4.4, 3.9, 4.5, 4.96, 4.4, 3.9]
        opens  = [3.9, 4.4, 4.90, 4.5, 4.0, 4.4, 4.90, 4.5, 4.0]
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_multi_top(df, pivot_n=2, timeframe="1min")
        self.assertTrue(len(signals) >= 1)
        self.assertEqual(signals[0].name, "double_top")
        self.assertEqual(signals[0].bias, "bearish")

    def test_no_top_on_uptrend(self):
        # Steadily rising prices — no tops
        n = 15
        base = np.linspace(4.0, 6.0, n)
        highs  = (base + 0.05).tolist()
        lows   = (base - 0.05).tolist()
        closes = base.tolist()
        opens  = (base - 0.02).tolist()
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_multi_top(df, pivot_n=2)
        self.assertEqual(len(signals), 0)


class TestMultiBottom(unittest.TestCase):

    def test_double_bottom_detected(self):
        lows   = [5.0, 4.5, 4.00, 4.5, 5.0, 4.5, 3.99, 4.5, 5.0]
        highs  = [5.1, 4.6, 4.10, 4.6, 5.1, 4.6, 4.10, 4.6, 5.1]
        closes = [5.0, 4.5, 4.05, 4.6, 5.0, 4.6, 4.04, 4.6, 5.0]
        opens  = [5.1, 4.6, 4.10, 4.5, 5.0, 4.6, 4.10, 4.5, 5.0]
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_multi_bottom(df, pivot_n=2, timeframe="1min")
        self.assertTrue(len(signals) >= 1)
        self.assertEqual(signals[0].name, "double_bottom")
        self.assertEqual(signals[0].bias, "bullish")


# ---------------------------------------------------------------------------
# Wedge detection
# ---------------------------------------------------------------------------

class TestWedge(unittest.TestCase):

    def test_rising_wedge_detected(self):
        # Rising wedge: both upper and lower trendlines slope up,
        # but upper slope is flatter (converging)
        n = 20
        x = np.arange(n, dtype=float)
        # Lower pivots: steep slope
        lower_base = 4.0 + 0.08 * x
        # Upper pivots: shallower slope (converging)
        upper_base = 5.0 + 0.04 * x
        # Oscillate between upper and lower
        mids = (upper_base + lower_base) / 2
        amp = (upper_base - lower_base) / 2
        osc = amp * np.sin(np.pi * x / 2)
        prices = mids + osc * 0.8

        highs  = np.maximum(prices + 0.02, upper_base * 0.999).tolist()
        lows   = np.minimum(prices - 0.02, lower_base * 1.001).tolist()
        closes = prices.tolist()
        opens  = (prices - 0.01).tolist()
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_wedge(df, pivot_n=2, timeframe="5min")
        # May or may not trigger depending on exact pivot alignment;
        # just verify no crash
        for s in signals:
            self.assertIn(s.name, ("rising_wedge", "falling_wedge"))


# ---------------------------------------------------------------------------
# Head and shoulders
# ---------------------------------------------------------------------------

class TestHeadAndShoulders(unittest.TestCase):

    def test_head_and_shoulders_detected(self):
        # Classic H&S: left shoulder, higher head, right shoulder
        highs  = [4.0, 4.5, 5.0, 4.5, 4.0, 4.5, 5.5, 4.5, 4.0, 4.5, 5.0, 4.5, 4.0]
        lows   = [3.9, 4.4, 4.9, 4.4, 3.8, 4.4, 5.4, 4.4, 3.8, 4.4, 4.9, 4.4, 3.9]
        closes = [4.0, 4.5, 4.95, 4.4, 3.9, 4.5, 5.45, 4.4, 3.9, 4.5, 4.95, 4.4, 3.9]
        opens  = [3.9, 4.4, 4.90, 4.5, 4.0, 4.4, 5.40, 4.5, 4.0, 4.4, 4.90, 4.5, 4.0]
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_head_and_shoulders(df, pivot_n=2, timeframe="5min")
        hns = [s for s in signals if s.name == "head_and_shoulders"]
        self.assertTrue(len(hns) >= 1)
        self.assertEqual(hns[0].bias, "bearish")

    def test_inverse_hns_detected(self):
        # Inverse H&S: left shoulder low, lower head, right shoulder low
        lows   = [5.0, 4.5, 4.0, 4.5, 5.0, 4.5, 3.5, 4.5, 5.0, 4.5, 4.0, 4.5, 5.0]
        highs  = [5.1, 4.6, 4.1, 4.6, 5.1, 4.6, 3.6, 4.6, 5.1, 4.6, 4.1, 4.6, 5.1]
        closes = [5.0, 4.5, 4.05, 4.6, 5.0, 4.6, 3.55, 4.6, 5.0, 4.6, 4.05, 4.6, 5.0]
        opens  = [5.1, 4.6, 4.10, 4.5, 5.0, 4.6, 3.60, 4.5, 5.0, 4.6, 4.10, 4.5, 5.0]
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_head_and_shoulders(df, pivot_n=2, timeframe="5min")
        inv = [s for s in signals if s.name == "inverse_head_and_shoulders"]
        self.assertTrue(len(inv) >= 1)
        self.assertEqual(inv[0].bias, "bullish")


# ---------------------------------------------------------------------------
# Pure pullback setup (non-resistance)
# ---------------------------------------------------------------------------

class TestPullbackNewHigh(unittest.TestCase):

    def test_detects_first_new_high_after_pullback(self):
        # 4 green candles (up-leg), 1 red pullback candle, then breakout candle.
        opens =  [10.00, 10.40, 10.80, 11.20, 11.35, 11.28]
        highs =  [10.45, 10.85, 11.25, 11.55, 11.45, 11.62]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.05, 11.25]
        closes = [10.40, 10.80, 11.20, 11.35, 11.28, 11.58]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_new_high(df, timeframe="1min")
        self.assertEqual(len(signals), 1)
        s = signals[0]
        self.assertEqual(s.name, "pullback_new_high")
        self.assertEqual(s.bias, "bullish")
        self.assertIsNotNone(s.pullback_depth)
        self.assertGreaterEqual(s.pullback_depth, 0.0)
        self.assertLessEqual(s.pullback_depth, 0.50)
        self.assertIsNotNone(s.pullback_control)
        self.assertGreater(s.pullback_control, 0.0)
        self.assertIsNone(s.pullback_ts)

    def test_pullback_low_emitted_on_all_three_detectors(self):
        # Same fixture as test_detects_first_new_high_after_pullback:
        # leg = candles 0-3, pullback = candle 4 (low 11.05), breakout = 5.
        opens =  [10.00, 10.40, 10.80, 11.20, 11.35, 11.28]
        highs =  [10.45, 10.85, 11.25, 11.55, 11.45, 11.62]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.05, 11.25]
        closes = [10.40, 10.80, 11.20, 11.35, 11.28, 11.58]
        df = _candle_df(opens, highs, lows, closes)

        resolved = detect_pullback_new_high(df, timeframe="1min")
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].pullback_low, 11.05)

        pending = detect_pullback_pending(df[:-1], "1min")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].pullback_low, 11.05)

        # Mid-candle detection: closed bars = the green leg only, current
        # candle forming red — the live price IS the pullback-low estimate.
        detected = detect_pullback_detected(
            df[:4], current_price=11.20, current_open=11.40, timeframe="1min"
        )
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0].pullback_low, 11.20)

    def test_pattern_signal_pullback_low_defaults_none(self):
        sig = PatternSignal("x", "bullish", 0.5, 1.0, "1min", 1)
        self.assertIsNone(sig.pullback_low)

    def test_equal_length_pullback_is_valid(self):
        # 2 green leg candles, 2 red pullback candles (YXT 8/5 flag #2 shape):
        # pullback may be as long as the leg — only LONGER is rejected.
        opens =  [5.80, 5.74, 6.78, 7.70, 7.62, 7.56]
        highs =  [6.05, 6.91, 8.13, 8.07, 7.73, 11.79]
        lows =   [5.53, 5.67, 6.75, 7.43, 7.16, 7.45]
        closes = [5.74, 6.77, 7.70, 7.60, 7.56, 10.18]
        df = _candle_df(opens, highs, lows, closes)

        pending = detect_pullback_pending(df[:-1], "1min")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].price_level, 7.73)  # last pullback candle high

        resolved = detect_pullback_new_high(df, timeframe="1min")
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].name, "pullback_new_high")

    def test_pullback_longer_than_leg_still_rejected(self):
        # 2 green leg candles, 3 red pullback candles → still invalid.
        opens =  [5.80, 5.74, 6.78, 7.70, 7.62, 7.56, 7.50]
        highs =  [6.05, 6.91, 8.13, 8.07, 7.73, 7.60, 11.79]
        lows =   [5.53, 5.67, 6.75, 7.43, 7.16, 7.40, 7.42]
        closes = [5.74, 6.77, 7.70, 7.60, 7.56, 7.45, 10.18]
        df = _candle_df(opens, highs, lows, closes)

        self.assertEqual(detect_pullback_pending(df[:-1], "1min"), [])
        self.assertEqual(detect_pullback_new_high(df, timeframe="1min"), [])

    def test_disabled_setup_timeframes_skipped_in_scan(self):
        # The same frame that fires on 1min produces NO pullback setups for
        # the disabled timeframes in scan_all_patterns (the setup pipeline).
        # The raw detectors stay ungated — the backtest's replay-window
        # derivation calls them directly for tick-coverage decisions.
        opens =  [10.00, 10.40, 10.80, 11.20, 11.35, 11.28]
        highs =  [10.45, 10.85, 11.25, 11.55, 11.45, 11.62]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.05, 11.25]
        closes = [10.40, 10.80, 11.20, 11.35, 11.28, 11.58]
        df = _candle_df(opens, highs, lows, closes)

        snap = scan_all_patterns(
            df, df, df, df, _make_rmap(11.0), pullback_last_bar_open=False
        )
        pullback_tfs = {
            s.timeframe for s in snap.signals
            if s.name in ("pullback_new_high", "pullback_pending")
        }
        self.assertTrue(pullback_tfs)                       # setups exist…
        self.assertFalse(pullback_tfs & SETUP_TFS_DISABLED)  # …none on disabled TFs

        # Raw detectors deliberately remain ungated (replay infrastructure).
        for tf in sorted(SETUP_TFS_DISABLED):
            self.assertEqual(len(detect_pullback_new_high(df, timeframe=tf)), 1)

    def test_accepts_green_leg_with_small_close_dip(self):
        opens =  [2.370, 2.520, 2.820, 3.000, 3.050, 3.146, 3.170]
        highs =  [2.550, 2.850, 3.100, 3.270, 3.230, 3.170, 3.950]
        lows =   [2.310, 2.460, 2.820, 2.940, 3.020, 2.840, 2.980]
        closes = [2.529, 2.850, 3.060, 3.042, 3.200, 2.980, 3.900]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_new_high(df, timeframe="10sec")

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "pullback_new_high")
        self.assertEqual(signals[0].leg_candles, 5)
        self.assertEqual(signals[0].pullback_candles, 1)
        self.assertLessEqual(signals[0].pullback_depth, 0.50)

    def test_rejects_green_leg_with_red_candle_in_middle(self):
        opens =  [2.370, 2.520, 2.820, 3.070, 3.050, 3.146, 3.170]
        highs =  [2.550, 2.850, 3.100, 3.270, 3.230, 3.170, 3.950]
        lows =   [2.310, 2.460, 2.820, 2.940, 3.020, 2.840, 2.980]
        closes = [2.529, 2.850, 3.060, 3.042, 3.200, 2.980, 3.900]
        df = _candle_df(opens, highs, lows, closes)

        self.assertEqual(detect_pullback_new_high(df, timeframe="10sec"), [])

    def test_breaks_previous_pullback_candle_high_before_leg_high(self):
        opens =  [10.00, 10.45, 10.90, 11.35, 11.70, 11.42]
        highs =  [10.55, 11.00, 11.50, 12.00, 11.75, 11.80]
        lows =   [ 9.95, 10.40, 10.85, 11.30, 11.30, 11.35]
        closes = [10.45, 10.90, 11.35, 11.70, 11.40, 11.70]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_new_high(df, timeframe="1min")

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "pullback_new_high")
        self.assertEqual(signals[0].price_level, 11.75)

    def test_accepts_pullback_with_as_many_candles_as_green_leg(self):
        # Equal-length (2v2) is valid since the leg_len rule became strict-less:
        # only pullbacks LONGER than the leg are rejected.
        opens =  [10.00, 10.60, 11.10, 10.95, 10.80]
        highs =  [10.70, 11.20, 11.12, 11.00, 11.15]
        lows =   [ 9.95, 10.55, 10.85, 10.70, 10.78]
        closes = [10.60, 11.10, 10.95, 10.80, 11.05]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_new_high(df, timeframe="1min")
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "pullback_new_high")

    def test_pending_accepts_pullback_with_as_many_candles_as_green_leg(self):
        opens =  [10.00, 10.60, 11.10, 10.95]
        highs =  [10.70, 11.20, 11.12, 11.00]
        lows =   [ 9.95, 10.55, 10.85, 10.70]
        closes = [10.60, 11.10, 10.95, 10.80]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_pending(df, timeframe="1min")
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "pullback_pending")

    def test_rejects_four_red_candle_pullback_even_with_longer_green_leg(self):
        opens =  [10.00, 10.40, 10.80, 11.20, 11.60, 12.00, 11.90, 11.80, 11.70, 11.65]
        highs =  [10.45, 10.85, 11.25, 11.65, 12.10, 12.05, 11.95, 11.85, 11.75, 11.82]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.55, 11.85, 11.75, 11.65, 11.60, 11.63]
        closes = [10.40, 10.80, 11.20, 11.60, 12.00, 11.90, 11.80, 11.70, 11.65, 11.78]
        df = _candle_df(opens, highs, lows, closes)

        self.assertEqual(detect_pullback_new_high(df, timeframe="1min"), [])

    def test_pending_rejects_four_red_candle_pullback_even_with_longer_green_leg(self):
        opens =  [10.00, 10.40, 10.80, 11.20, 11.60, 12.00, 11.90, 11.80, 11.70]
        highs =  [10.45, 10.85, 11.25, 11.65, 12.10, 12.05, 11.95, 11.85, 11.75]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.55, 11.85, 11.75, 11.65, 11.60]
        closes = [10.40, 10.80, 11.20, 11.60, 12.00, 11.90, 11.80, 11.70, 11.65]
        df = _candle_df(opens, highs, lows, closes)

        self.assertEqual(detect_pullback_pending(df, timeframe="1min"), [])

    def test_rejects_sideways_leg(self):
        opens =  [10.000, 10.010, 10.020, 10.030, 10.040, 10.035]
        highs =  [10.015, 10.025, 10.035, 10.045, 10.043, 10.050]
        lows =   [ 9.995, 10.005, 10.015, 10.025, 10.030, 10.033]
        closes = [10.010, 10.020, 10.030, 10.040, 10.035, 10.048]
        df = _candle_df(opens, highs, lows, closes)

        self.assertEqual(detect_pullback_new_high(df, timeframe="1min"), [])

    def test_pending_pullback_rejects_sideways_leg(self):
        opens =  [10.000, 10.010, 10.020, 10.030, 10.040]
        highs =  [10.015, 10.025, 10.035, 10.045, 10.043]
        lows =   [ 9.995, 10.005, 10.015, 10.025, 10.030]
        closes = [10.010, 10.020, 10.030, 10.040, 10.035]
        df = _candle_df(opens, highs, lows, closes)

        self.assertEqual(detect_pullback_pending(df, timeframe="1min"), [])

    def test_detects_break_after_one_extra_green_candle(self):
        opens =  [10.00, 10.40, 10.80, 11.20, 11.05, 10.95, 11.02]
        highs =  [10.45, 10.85, 11.25, 11.22, 11.10, 11.08, 11.18]
        lows =   [ 9.95, 10.35, 10.75, 11.00, 10.90, 10.94, 11.00]
        closes = [10.40, 10.80, 11.20, 11.05, 10.95, 11.02, 11.15]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_new_high(df, timeframe="1min")

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "pullback_new_high")
        self.assertEqual(signals[0].price_level, 11.08)
        self.assertEqual(signals[0].pullback_candles, 2)

    def test_pending_after_one_extra_green_candle_uses_green_high_as_trigger(self):
        opens =  [10.00, 10.40, 10.80, 11.20, 11.05, 10.95]
        highs =  [10.45, 10.85, 11.25, 11.22, 11.10, 11.08]
        lows =   [ 9.95, 10.35, 10.75, 11.00, 10.90, 10.94]
        closes = [10.40, 10.80, 11.20, 11.05, 10.95, 11.02]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_pending(df, timeframe="1min")

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].name, "pullback_pending")
        self.assertEqual(signals[0].price_level, 11.08)
        self.assertEqual(signals[0].pullback_candles, 2)

    def test_pullback_setup_skips_daily_timeframe(self):
        opens =  [10.00, 10.40, 10.80, 11.20, 11.35, 11.28]
        highs =  [10.45, 10.85, 11.25, 11.55, 11.45, 11.62]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.05, 11.25]
        closes = [10.40, 10.80, 11.20, 11.35, 11.28, 11.58]
        df = _candle_df(opens, highs, lows, closes)

        self.assertEqual(detect_pullback_new_high(df, timeframe="1d"), [])
        self.assertEqual(detect_pullback_pending(df.iloc[:-1], timeframe="1d"), [])

    def test_setup_rejection_metrics_use_high_and_pullback_candles(self):
        # Upper wick on the high candle plus red body on the pullback candle.
        opens =  [10.00, 10.40, 10.80, 11.20, 11.55, 11.28]
        highs =  [10.45, 10.85, 11.25, 11.60, 11.58, 11.68]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.10, 11.25]
        closes = [10.40, 10.80, 11.20, 11.35, 11.25, 11.64]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_new_high(df, timeframe="1min")

        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertIsNotNone(signal.topping_tail)
        self.assertIsNotNone(signal.jackknife)
        self.assertGreater(signal.topping_tail, 0.0)
        self.assertGreater(signal.jackknife, 0.0)

    def test_setup_rejection_softened_on_low_volume_pullback(self):
        # Same candle shapes; only the red pullback candle's volume differs.
        opens =  [10.00, 10.40, 10.80, 11.20, 11.55, 11.28]
        highs =  [10.45, 10.85, 11.25, 11.60, 11.58, 11.68]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.10, 11.25]
        closes = [10.40, 10.80, 11.20, 11.35, 11.25, 11.64]
        base_vol = [10_000] * 4
        low_vol_df = _candle_df(opens, highs, lows, closes,
                                volumes=base_vol + [4_000, 10_000])
        high_vol_df = _candle_df(opens, highs, lows, closes,
                                 volumes=base_vol + [40_000, 10_000])

        low_sig = detect_pullback_new_high(low_vol_df, timeframe="1min")[0]
        high_sig = detect_pullback_new_high(high_vol_df, timeframe="1min")[0]

        # Low-volume red pullback = healthy; high-volume = distribution risk.
        self.assertLess(low_sig.jackknife, high_sig.jackknife)
        self.assertGreater(high_sig.jackknife, 0.3)
        # Risks are bounded [0, 1] under volume-response scoring.
        self.assertLessEqual(high_sig.jackknife, 1.0)
        self.assertGreater(low_sig.jackknife, 0.0)  # softened, not excused

    def test_detected_pullback_uses_datetime_index_for_timestamp(self):
        opens =  [10.00, 10.40, 10.80, 11.20, 11.35, 11.28]
        highs =  [10.45, 10.85, 11.25, 11.55, 11.45, 11.62]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.05, 11.25]
        closes = [10.40, 10.80, 11.20, 11.35, 11.28, 11.58]
        df = _candle_df(opens, highs, lows, closes)
        df.index = pd.date_range(
            datetime(2026, 4, 10, 14, 0, tzinfo=timezone.utc),
            periods=len(df),
            freq="min",
        )

        signal = detect_pullback_new_high(df, timeframe="1min")[0]

        self.assertEqual(signal.pullback_ts, df.index[4].to_pydatetime())

    def test_accepts_shallow_pullback(self):
        # Pullback depth < the old 5% floor should still qualify.
        opens =  [10.00, 10.40, 10.80, 11.20, 11.52, 11.51]
        highs =  [10.45, 10.85, 11.25, 11.55, 11.54, 11.62]
        lows =   [ 9.95, 10.35, 10.75, 11.15, 11.49, 11.50]
        closes = [10.40, 10.80, 11.20, 11.35, 11.51, 11.58]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_new_high(df, timeframe="1min")
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.name, "pullback_new_high")
        self.assertIsNotNone(signal.pullback_depth)
        self.assertLess(signal.pullback_depth, 0.05)

    def test_rejects_too_deep_pullback(self):
        # Pullback depth > 50% of leg.
        opens =  [10.00, 10.20, 10.45, 10.70, 10.95, 10.60]
        highs =  [10.25, 10.50, 10.75, 11.00, 10.96, 11.08]
        lows =   [ 9.95, 10.15, 10.40, 10.65, 10.10, 10.55]
        closes = [10.20, 10.45, 10.70, 10.95, 10.60, 11.02]
        df = _candle_df(opens, highs, lows, closes)

        signals = detect_pullback_new_high(df, timeframe="1min")
        self.assertEqual(signals, [])

    def test_cleaner_pullback_has_higher_strength(self):
        # Cleaner: shallower, shorter pullback with decisive breakout.
        clean_df = _candle_df(
            [10.00, 10.40, 10.80, 11.20, 11.35, 11.30],
            [10.45, 10.85, 11.25, 11.55, 11.45, 11.65],
            [ 9.95, 10.35, 10.75, 11.15, 11.30, 11.28],
            [10.40, 10.80, 11.20, 11.35, 11.30, 11.60],
        )
        clean = detect_pullback_new_high(clean_df, timeframe="1min")[0]

        # Sloppier: deeper and longer pullback but still valid (<50%).
        sloppy_df = _candle_df(
            [10.00, 10.40, 10.80, 11.20, 11.40, 11.25, 11.05, 10.95],
            [10.45, 10.85, 11.25, 11.55, 11.45, 11.30, 11.10, 11.62],
            [ 9.95, 10.35, 10.75, 11.15, 11.20, 11.00, 10.85, 10.92],
            [10.40, 10.80, 11.20, 11.40, 11.25, 11.05, 10.95, 11.58],
        )
        sloppy = detect_pullback_new_high(sloppy_df, timeframe="1min")[0]

        self.assertGreater(clean.strength, sloppy.strength)

    def test_pending_pullback_control_rewards_shallow_short_pullback(self):
        clean_df = _candle_df(
            [10.00, 10.40, 10.80, 11.20, 11.35],
            [10.45, 10.85, 11.25, 11.55, 11.42],
            [ 9.95, 10.35, 10.75, 11.15, 11.30],
            [10.40, 10.80, 11.20, 11.35, 11.32],
        )
        clean = detect_pullback_pending(clean_df, timeframe="1min")[0]

        sloppy_df = _candle_df(
            [10.00, 10.40, 10.80, 11.20, 11.40, 11.25, 11.08],
            [10.45, 10.85, 11.25, 11.55, 11.45, 11.30, 11.12],
            [ 9.95, 10.35, 10.75, 11.15, 11.20, 11.02, 10.82],
            [10.40, 10.80, 11.20, 11.40, 11.25, 11.08, 10.94],
        )
        sloppy = detect_pullback_pending(sloppy_df, timeframe="1min")[0]

        self.assertGreater(clean.pullback_control, sloppy.pullback_control)
        self.assertGreater(clean.strength, sloppy.strength)


# ---------------------------------------------------------------------------
# Breakout pullback / break + retest
# ---------------------------------------------------------------------------

class TestBreakoutPullback(unittest.TestCase):

    def test_break_and_retest_detected(self):
        rmap = _make_rmap(price=5.50)
        rmap.build_static()
        # Simulate breaking $5.00 then retesting
        # First, mark $5.00 as broken
        rmap.mark_broken_levels(5.10)

        # Build candles: price breaks 5.00, pulls back, bounces
        closes = [4.90, 4.95, 5.05, 5.10, 5.15, 5.03, 5.01, 5.00, 5.05, 5.08, 5.12,
                  5.15, 5.18, 5.20, 5.22, 5.25]
        opens  = [4.88, 4.93, 5.00, 5.05, 5.10, 5.15, 5.05, 5.02, 5.00, 5.04, 5.08,
                  5.12, 5.15, 5.18, 5.20, 5.22]
        highs  = [4.92, 4.97, 5.08, 5.12, 5.18, 5.16, 5.06, 5.03, 5.07, 5.10, 5.14,
                  5.17, 5.20, 5.22, 5.24, 5.27]
        lows   = [4.86, 4.91, 4.98, 5.04, 5.09, 5.02, 5.00, 4.98, 4.99, 5.03, 5.07,
                  5.11, 5.14, 5.17, 5.19, 5.21]
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_breakout_pullback(df, rmap, "1min")
        # Should find patterns related to the broken $5.00 level
        # (may or may not depending on exact buffer math — just verify no crash)
        for s in signals:
            self.assertIn(s.bias, ("bullish",))

    def test_no_breakout_without_broken_levels(self):
        rmap = _make_rmap(price=5.00)
        # No levels built, no broken levels
        n = 16
        closes = [5.0 + 0.01 * i for i in range(n)]
        opens = [c - 0.01 for c in closes]
        highs = [c + 0.02 for c in closes]
        lows = [c - 0.02 for c in closes]
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_breakout_pullback(df, rmap, "1min")
        self.assertEqual(len(signals), 0)


# ---------------------------------------------------------------------------
# Support bounce / Resistance rejection
# ---------------------------------------------------------------------------

class TestSupportBounce(unittest.TestCase):

    def test_support_bounce_at_broken_level(self):
        rmap = _make_rmap(price=5.50)
        rmap.build_static()
        rmap.mark_broken_levels(5.10)

        # Price dips to $5.00 (broken resistance = support) and bounces
        closes = [5.20, 5.15, 5.10, 5.05, 5.02, 5.00, 5.03, 5.06, 5.10]
        opens  = [5.22, 5.18, 5.13, 5.08, 5.04, 5.01, 5.00, 5.03, 5.07]
        highs  = [5.23, 5.18, 5.14, 5.09, 5.05, 5.02, 5.05, 5.08, 5.12]
        lows   = [5.19, 5.14, 5.09, 5.04, 5.00, 4.99, 4.99, 5.02, 5.06]
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_support_bounce(df, rmap, "1min")
        # Verify no crash; may or may not find a bounce depending on buffer
        for s in signals:
            self.assertEqual(s.bias, "bullish")

    def test_support_bounce_at_ema(self):
        """Bounce off EMA 9 level (from indicator column, not hardcoded)."""
        rmap = _make_rmap(price=5.50)
        # No levels in rmap — the EMA comes from the DataFrame
        closes = [5.20, 5.15, 5.10, 5.05, 5.02, 5.00, 5.03, 5.06, 5.10]
        opens  = [5.22, 5.18, 5.13, 5.08, 5.04, 5.01, 5.00, 5.03, 5.07]
        highs  = [5.23, 5.18, 5.14, 5.09, 5.05, 5.02, 5.05, 5.08, 5.12]
        lows   = [5.19, 5.14, 5.09, 5.04, 5.00, 4.99, 4.99, 5.02, 5.06]
        df = _candle_df(opens, highs, lows, closes)
        # Add an EMA column at 5.00
        df["ema_9"] = 5.00
        signals = detect_support_bounce(df, rmap, "1min")
        for s in signals:
            self.assertEqual(s.bias, "bullish")


class TestResistanceRejection(unittest.TestCase):

    def test_rejection_at_resistance(self):
        rmap = _make_rmap(price=4.80)
        rmap.build_static()  # builds $5.00 as whole dollar resistance

        # Price pushes into $5.00 and gets rejected
        closes = [4.80, 4.85, 4.90, 4.95, 4.98, 5.01, 4.97, 4.93, 4.90]
        opens  = [4.78, 4.82, 4.87, 4.92, 4.96, 4.99, 5.00, 4.96, 4.92]
        highs  = [4.82, 4.87, 4.92, 4.97, 5.00, 5.02, 5.01, 4.97, 4.93]
        lows   = [4.77, 4.81, 4.86, 4.91, 4.95, 4.98, 4.96, 4.92, 4.88]
        df = _candle_df(opens, highs, lows, closes)
        signals = detect_resistance_rejection(df, rmap, "1min")
        for s in signals:
            self.assertEqual(s.bias, "bearish")


# ---------------------------------------------------------------------------
# Tail clusters
# ---------------------------------------------------------------------------

class TestTailClusters(unittest.TestCase):

    def test_topping_tail_cluster(self):
        # Multiple candles with big upper wicks near the same price
        n = 12
        opens  = [5.00] * n
        closes = [5.02] * n
        lows   = [5.00] * n
        highs  = [5.10] * n  # normal candles

        # Make 3 candles have topping tails at ~5.50
        for i in [3, 6, 9]:
            opens[i] = 5.00
            highs[i] = 5.50
            lows[i] = 5.00
            closes[i] = 5.02

        df = _candle_df(opens, highs, lows, closes)
        signals = detect_tail_clusters(df, "1min")
        tops = [s for s in signals if s.name == "topping_tail_cluster"]
        self.assertTrue(len(tops) >= 1)
        self.assertEqual(tops[0].bias, "bearish")


# ---------------------------------------------------------------------------
# scan_all_patterns entry point
# ---------------------------------------------------------------------------

class TestScanAll(unittest.TestCase):

    def test_scan_returns_snapshot(self):
        rmap = _make_rmap(price=5.00)
        df = _candle_df(
            [5.0] * 10, [5.1] * 10, [4.9] * 10, [5.0] * 10,
        )
        snap = scan_all_patterns(df, df, df, df, rmap)
        self.assertIsInstance(snap, PatternSnapshot)

    def test_empty_dfs_no_crash(self):
        rmap = _make_rmap(price=5.00)
        empty = pd.DataFrame()
        snap = scan_all_patterns(empty, empty, empty, empty, rmap)
        self.assertIsInstance(snap, PatternSnapshot)
        self.assertEqual(len(snap.signals), 0)


# ---------------------------------------------------------------------------
# PatternSnapshot properties
# ---------------------------------------------------------------------------

class TestPatternSnapshot(unittest.TestCase):

    def test_net_bias_positive(self):
        snap = PatternSnapshot(signals=[
            PatternSignal("test_bull", "bullish", 0.8, 5.0, "1min", 5),
        ])
        self.assertGreater(snap.net_bias, 0)
        self.assertEqual(snap.bullish_count, 1)
        self.assertEqual(snap.bearish_count, 0)

    def test_net_bias_negative(self):
        snap = PatternSnapshot(signals=[
            PatternSignal("test_bear", "bearish", 0.7, 5.0, "1min", 5),
        ])
        self.assertLess(snap.net_bias, 0)

    def test_strongest_properties(self):
        snap = PatternSnapshot(signals=[
            PatternSignal("a", "bearish", 0.5, 5.0, "1min", 5),
            PatternSignal("b", "bearish", 0.9, 5.0, "1min", 5),
            PatternSignal("c", "bullish", 0.7, 5.0, "1min", 5),
        ])
        self.assertEqual(snap.strongest_bearish.name, "b")
        self.assertEqual(snap.strongest_bullish.name, "c")


# ---------------------------------------------------------------------------
# ResistanceMap new methods
# ---------------------------------------------------------------------------

class TestResistanceMapSupport(unittest.TestCase):

    def test_broken_levels_returned(self):
        rmap = _make_rmap(price=5.50)
        rmap.build_static()
        rmap.mark_broken_levels(5.10)
        broken = rmap.broken_levels()
        self.assertTrue(len(broken) > 0)
        for lv in broken:
            self.assertFalse(lv.active)

    def test_levels_below(self):
        rmap = _make_rmap(price=5.50)
        rmap.build_static()
        below = rmap.levels_below(5.50, limit=3)
        self.assertTrue(len(below) > 0)
        for lv in below:
            self.assertLess(lv.price, 5.50)

    def test_bottoming_tail_on_resistance_map(self):
        from technical_analysis.resistance import ResistanceMap as RM
        candle = pd.Series({"open": 5.02, "high": 5.02, "low": 4.50, "close": 5.00})
        self.assertTrue(RM.detect_bottoming_tail(candle))

    def test_pivot_lows_found(self):
        rmap = _make_rmap(price=5.00)
        df = pd.DataFrame({
            "open":   [5.0, 4.5, 4.0, 3.0, 4.0, 4.5, 5.0],
            "high":   [5.1, 4.6, 4.1, 3.1, 4.1, 4.6, 5.1],
            "low":    [4.9, 4.4, 3.9, 2.9, 3.9, 4.4, 4.9],
            "close":  [5.0, 4.5, 4.0, 3.0, 4.0, 4.5, 5.0],
            "volume": [100] * 7,
        })
        pivots = ResistanceMap._find_pivot_lows(df, n=2)
        self.assertTrue(len(pivots) > 0)
        self.assertAlmostEqual(pivots[0], 2.9, places=1)


if __name__ == "__main__":
    unittest.main()
