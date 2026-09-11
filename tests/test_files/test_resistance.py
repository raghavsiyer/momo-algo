"""
tests/test_resistance.py

Unit tests for technical_analysis/resistance.py.
No API calls — all tests use synthetic DataFrames and plain Python data.
"""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from technical_analysis.resistance import (
    ResistanceMap,
    ResistanceLevel,
    LevelType,
    STRENGTH,
)
from display import format_levels
from config import TOPPING_TAIL_RATIO, RESISTANCE_BREAK_BUFFER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rmap(price: float = 5.00, symbol: str = "TEST") -> ResistanceMap:
    return ResistanceMap(symbol=symbol, current_price=price)


def _candle_df(highs, lows=None, closes=None) -> pd.DataFrame:
    n = len(highs)
    lows_   = lows   or [h * 0.99  for h in highs]
    closes_ = closes or [h * 0.995 for h in highs]
    opens_  = [c * 0.998 for c in closes_]
    return pd.DataFrame({
        "open":   opens_,
        "high":   highs,
        "low":    lows_,
        "close":  closes_,
        "volume": [100_000] * n,
    })


def _level(
    price: float,
    level_type: LevelType = LevelType.STATIC,
    strength: float = 0.80,
    source: str = "test",
    active: bool = True,
) -> ResistanceLevel:
    return ResistanceLevel(
        price=price,
        level_type=level_type,
        strength=strength,
        source=source,
        formed_at=datetime.now(tz=timezone.utc),
        active=active,
    )


# ---------------------------------------------------------------------------
# build_static — level generation
# ---------------------------------------------------------------------------

class TestBuildStatic(unittest.TestCase):

    def test_whole_dollar_levels_generated(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        prices = [l.price for l in rmap.all_levels()]
        self.assertIn(5.0, prices)
        self.assertIn(6.0, prices)

    def test_half_dollar_levels_generated(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        prices = [l.price for l in rmap.all_levels()]
        self.assertIn(5.5, prices)

    def test_no_quarter_dollar_above_1_dollar(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        sources = [l.source for l in rmap.all_levels()]
        self.assertNotIn("quarter dollar", sources)

    def test_quarter_dollar_for_sub_1_price(self):
        rmap = _make_rmap(price=0.60)
        rmap.build_static()
        sources = [l.source for l in rmap.all_levels()]
        self.assertIn("quarter dollar", sources)

    def test_whole_dollar_has_correct_strength(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        whole = [l for l in rmap.all_levels() if l.source == "whole dollar"]
        self.assertTrue(len(whole) > 0)
        for l in whole:
            self.assertAlmostEqual(l.strength, STRENGTH["whole_dollar"])

    def test_all_levels_active_on_creation(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        for l in rmap.all_levels():
            self.assertTrue(l.active)

    def test_levels_within_window(self):
        price = 5.00
        rmap = _make_rmap(price=price)
        rmap.build_static()
        for l in rmap.all_levels():
            self.assertGreaterEqual(l.price, price * 0.80)
            self.assertLessEqual(l.price, price * 1.40 + 0.01)  # tiny tolerance


# ---------------------------------------------------------------------------
# nearest_above / nearest_below
# ---------------------------------------------------------------------------

class TestNearest(unittest.TestCase):

    def test_nearest_above_is_5_for_485(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        nearest = rmap.nearest_above(4.85)
        self.assertIsNotNone(nearest)
        self.assertGreater(nearest.price, 4.85)
        self.assertAlmostEqual(nearest.price, 5.00)

    def test_nearest_below_exists(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        below = rmap.nearest_below(4.85)
        self.assertIsNotNone(below)
        self.assertLess(below.price, 4.85)

    def test_nearest_above_returns_none_when_empty(self):
        rmap = _make_rmap(price=4.85)
        self.assertIsNone(rmap.nearest_above())

    def test_nearest_above_skips_inactive_levels(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        # Deactivate $5.00
        for l in rmap._levels:
            if l.price == 5.00:
                l.active = False
        nearest = rmap.nearest_above(4.85)
        self.assertIsNotNone(nearest)
        self.assertNotAlmostEqual(nearest.price, 5.00)

    def test_nearest_above_uses_current_price_by_default(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        by_arg     = rmap.nearest_above(4.85)
        by_default = rmap.nearest_above()
        self.assertEqual(by_arg.price, by_default.price)


# ---------------------------------------------------------------------------
# levels_in_range
# ---------------------------------------------------------------------------

class TestLevelsInRange(unittest.TestCase):

    def test_all_returned_within_buffer(self):
        rmap = _make_rmap(price=5.0)
        rmap.build_static()
        in_range = rmap.levels_in_range(price=5.0, buffer=0.30)
        for l in in_range:
            self.assertLessEqual(abs(l.price - 5.0), 0.30)

    def test_returns_empty_when_buffer_too_tight(self):
        rmap = _make_rmap(price=5.0)
        rmap.build_static()
        # $5.15 has no level within $0.001
        result = rmap.levels_in_range(price=5.15, buffer=0.001)
        self.assertEqual(len(result), 0)

    def test_is_near_resistance_true_near_whole_dollar(self):
        rmap = _make_rmap(price=4.96)
        rmap.build_static()
        # $4.96 is within $0.05 of $5.00
        self.assertTrue(rmap.is_near_resistance(4.96, buffer=0.05))

    def test_is_near_resistance_false_midpoint(self):
        rmap = _make_rmap(price=5.25)
        rmap.build_static()
        # $5.25 is equidistant between $5.00 and $5.50 — not within $0.05
        self.assertFalse(rmap.is_near_resistance(5.25, buffer=0.05))


# ---------------------------------------------------------------------------
# mark_broken_levels
# ---------------------------------------------------------------------------

class TestMarkBrokenLevels(unittest.TestCase):

    def test_breaks_level_when_price_above_plus_buffer(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        broken = rmap.mark_broken_levels(5.00 + RESISTANCE_BREAK_BUFFER + 0.01)
        broken_prices = {l.price for l in broken}
        self.assertIn(5.00, broken_prices)

    def test_broken_level_becomes_inactive(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        rmap.mark_broken_levels(5.00 + RESISTANCE_BREAK_BUFFER + 0.01)
        level_5 = next(l for l in rmap._levels if l.price == 5.00)
        self.assertFalse(level_5.active)

    def test_level_not_broken_when_price_below(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        rmap.mark_broken_levels(4.90)
        level_5 = next((l for l in rmap._levels if l.price == 5.00), None)
        if level_5:
            self.assertTrue(level_5.active)

    def test_level_not_broken_at_exact_price_plus_buffer(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        # Exactly at 5.00 + buffer is NOT strictly > buffer
        rmap.mark_broken_levels(5.00 + RESISTANCE_BREAK_BUFFER)
        level_5 = next((l for l in rmap._levels if l.price == 5.00), None)
        if level_5:
            self.assertTrue(level_5.active)

    def test_returns_list_of_broken_levels(self):
        rmap = _make_rmap(price=3.50)
        rmap.build_static()
        broken = rmap.mark_broken_levels(5.00 + RESISTANCE_BREAK_BUFFER + 0.10)
        self.assertIsInstance(broken, list)
        self.assertGreater(len(broken), 0)


# ---------------------------------------------------------------------------
# add_dynamic
# ---------------------------------------------------------------------------

class TestAddDynamic(unittest.TestCase):

    def test_dynamic_level_added(self):
        rmap = _make_rmap(price=5.0)
        rmap.add_dynamic(5.10, source="topping tail")
        self.assertEqual(len(rmap._levels), 1)

    def test_dynamic_level_has_correct_type(self):
        rmap = _make_rmap(price=5.0)
        lvl = rmap.add_dynamic(5.10, source="topping tail")
        self.assertEqual(lvl.level_type, LevelType.DYNAMIC)

    def test_dynamic_infers_topping_tail_strength(self):
        rmap = _make_rmap(price=5.0)
        lvl = rmap.add_dynamic(5.10, source="topping tail")
        self.assertAlmostEqual(lvl.strength, STRENGTH["topping_tail"])

    def test_dynamic_infers_intraday_high_strength(self):
        rmap = _make_rmap(price=5.0)
        lvl = rmap.add_dynamic(5.10, source="intraday high")
        self.assertAlmostEqual(lvl.strength, STRENGTH["daily_high"])

    def test_dynamic_strength_override(self):
        rmap = _make_rmap(price=5.0)
        lvl = rmap.add_dynamic(5.10, source="custom", strength=0.99)
        self.assertAlmostEqual(lvl.strength, 0.99)


# ---------------------------------------------------------------------------
# _add_or_merge
# ---------------------------------------------------------------------------

class TestAddOrMerge(unittest.TestCase):

    def test_merges_same_type_nearby(self):
        rmap = _make_rmap()
        rmap._add_or_merge(_level(5.00, LevelType.STATIC))
        rmap._add_or_merge(_level(5.01, LevelType.STATIC, source="half dollar"))
        # Should merge (within 0.03 tolerance)
        self.assertEqual(len(rmap._levels), 1)

    def test_no_merge_different_type(self):
        rmap = _make_rmap()
        rmap._add_or_merge(_level(5.00, LevelType.STATIC))
        rmap._add_or_merge(_level(5.01, LevelType.STRUCTURE))
        self.assertEqual(len(rmap._levels), 2)

    def test_merge_keeps_higher_strength(self):
        rmap = _make_rmap()
        rmap._add_or_merge(_level(5.00, strength=0.60))
        rmap._add_or_merge(_level(5.01, strength=0.90))
        self.assertAlmostEqual(rmap._levels[0].strength, 0.90)

    def test_far_apart_levels_not_merged(self):
        rmap = _make_rmap()
        rmap._add_or_merge(_level(5.00))
        rmap._add_or_merge(_level(5.10))  # 0.10 > merge_tol=0.03
        self.assertEqual(len(rmap._levels), 2)

    def test_merged_source_combined(self):
        rmap = _make_rmap()
        rmap._add_or_merge(_level(5.00, source="whole dollar"))
        rmap._add_or_merge(_level(5.01, source="pivot high"))
        # Source should contain both labels
        self.assertIn("whole dollar", rmap._levels[0].source)
        self.assertIn("pivot high",   rmap._levels[0].source)


# ---------------------------------------------------------------------------
# _find_pivot_highs
# ---------------------------------------------------------------------------

class TestFindPivotHighs(unittest.TestCase):

    def test_detects_clear_pivot(self):
        # Peak at index 4 (10.0) is above its 2 neighbours on each side
        highs  = [5.0, 6.0, 7.0, 8.0, 10.0, 8.0, 7.0, 6.0, 5.0]
        closes = [4.5, 5.5, 6.5, 7.5,  9.5, 7.5, 6.5, 5.5, 4.5]
        df = _candle_df(highs, closes=closes)
        pivots = ResistanceMap._find_pivot_highs(df, n=2)
        self.assertIn(10.0, pivots)

    def test_no_pivots_in_monotone_series(self):
        highs  = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        closes = [0.9, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9]
        df = _candle_df(highs, closes=closes)
        self.assertEqual(ResistanceMap._find_pivot_highs(df, n=2), [])

    def test_empty_df_returns_empty(self):
        self.assertEqual(ResistanceMap._find_pivot_highs(pd.DataFrame(), n=2), [])

    def test_too_short_returns_empty(self):
        df = _candle_df([5.0, 6.0, 5.0])  # only 3 bars, need 2*n+1=5 for n=2
        self.assertEqual(ResistanceMap._find_pivot_highs(df, n=2), [])

    def test_pivot_below_recent_close_excluded(self):
        # Recent close is 15.0 — pivot at 10.0 should be excluded (already below)
        highs  = [5.0, 6.0, 10.0, 6.0, 5.0, 15.0, 14.0]
        closes = [4.5, 5.5,  9.5, 5.5, 4.5, 15.0, 14.0]
        df = _candle_df(highs, closes=closes)
        pivots = ResistanceMap._find_pivot_highs(df, n=2)
        self.assertNotIn(10.0, pivots)

    def test_deduplicates_nearby_pivots(self):
        # Two very close highs (5.0 and 5.02) should deduplicate
        highs  = [3.0, 4.0, 5.0, 4.0, 3.0, 4.0, 5.02, 4.0, 3.0]
        closes = [2.9, 3.9, 4.9, 3.9, 2.9, 3.9,  4.99, 3.9, 2.9]
        df = _candle_df(highs, closes=closes)
        pivots = ResistanceMap._find_pivot_highs(df, n=2)
        # Should not have two entries within $0.05 of each other
        for i in range(len(pivots)):
            for j in range(i + 1, len(pivots)):
                self.assertGreater(abs(pivots[i] - pivots[j]), 0.05)


class TestStructuralPivotConfluence(unittest.TestCase):

    def test_build_structure_refines_daily_zones_with_5min_hod_pivots(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()

        daily_highs = [
            4.70, 4.85, 4.95, 5.10, 4.95, 4.80, 4.70,
            5.55, 5.20, 5.05, 5.15, 5.30, 5.86, 5.40,
            5.20, 5.30, 6.00, 5.55, 5.30, 5.20, 6.30,
            5.80, 5.45, 5.35, 6.53, 5.90, 5.50, 5.30,
            6.90, 5.70, 5.20, 4.90, 4.70, 4.60,
        ]
        df_daily = _candle_df(
            daily_highs,
            lows=[h - 0.45 for h in daily_highs],
            closes=[min(h - 0.10, 4.80) for h in daily_highs],
        )

        min5_highs = [
            4.80, 4.90, 5.00, 5.11, 4.95, 4.85,
            5.20, 5.55, 5.90, 5.60, 5.35, 5.25,
            5.70, 6.00, 6.14, 5.85, 5.40, 5.05,
            4.90, 4.80,
        ]
        df_5min = _candle_df(
            min5_highs,
            lows=[h - 0.35 for h in min5_highs],
            closes=[min(h - 0.05, 4.75) for h in min5_highs],
        )

        rmap.build_structure(df_daily, df_5min)
        active_prices = {round(l.price, 2) for l in rmap.all_levels(active_only=True)}
        structure_sources = {
            round(l.price, 2): l.source
            for l in rmap.all_levels(active_only=True)
            if l.level_type == LevelType.STRUCTURE
        }

        self.assertIn(5.11, active_prices)
        self.assertIn(5.90, active_prices)
        self.assertIn(6.14, active_prices)
        self.assertIn("5-min HOD pivot", structure_sources[5.11])
        self.assertIn("daily pivot high", structure_sources[5.90])
        self.assertIn("5-min HOD pivot", structure_sources[5.90])
        self.assertIn("5-min HOD pivot", structure_sources[6.14])

        noisy_daily_only = {5.55, 5.86, 6.00, 6.30, 6.53, 6.90}
        structure_prices = set(structure_sources)
        self.assertTrue(noisy_daily_only.isdisjoint(structure_prices - {5.90}))

    def test_build_structure_adds_hod_double_top_with_2pct_leeway(self):
        rmap = _make_rmap(price=4.70)
        df_daily = pd.DataFrame()
        df_5min = _candle_df(
            [5.00, 4.82, 4.65, 4.74, 4.91, 4.62, 4.50, 4.45],
            lows=[4.70, 4.55, 4.40, 4.48, 4.63, 4.36, 4.28, 4.22],
            closes=[4.78, 4.60, 4.48, 4.55, 4.68, 4.40, 4.32, 4.30],
        )

        rmap.build_structure(df_daily, df_5min)
        structure_sources = {
            round(l.price, 2): l.source
            for l in rmap.all_levels(active_only=True)
            if l.level_type == LevelType.STRUCTURE
        }

        self.assertIn(5.00, structure_sources)
        self.assertIn("5-min HOD double top", structure_sources[5.00])

    def test_build_structure_skips_recent_hod_without_topping_tail(self):
        rmap = _make_rmap(price=4.90)
        df_daily = pd.DataFrame()
        df_5min = _candle_df(
            [4.80, 5.00, 4.70, 4.85, 5.10, 4.90],
            lows=[4.60, 4.80, 4.50, 4.65, 4.90, 4.70],
            closes=[4.75, 4.92, 4.60, 4.78, 5.04, 4.84],
        )

        rmap.build_structure(df_daily, df_5min)
        structure_prices = {
            round(l.price, 2)
            for l in rmap.all_levels(active_only=True)
            if l.level_type == LevelType.STRUCTURE
        }

        self.assertNotIn(5.10, structure_prices)

    def test_build_structure_keeps_recent_hod_topping_tail(self):
        rmap = _make_rmap(price=4.90)
        df_daily = pd.DataFrame()
        df_5min = _candle_df(
            [4.80, 5.00, 4.70, 4.85, 5.10, 4.90],
            lows=[4.60, 4.80, 4.50, 4.65, 4.80, 4.70],
            closes=[4.75, 4.92, 4.60, 4.78, 4.86, 4.84],
        )

        rmap.build_structure(df_daily, df_5min)
        structure_sources = {
            round(l.price, 2): l.source
            for l in rmap.all_levels(active_only=True)
            if l.level_type == LevelType.STRUCTURE
        }

        self.assertIn(5.10, structure_sources)
        self.assertIn("5-min HOD topping tail", structure_sources[5.10])


# ---------------------------------------------------------------------------
# detect_topping_tail
# ---------------------------------------------------------------------------

class TestDetectToppingTail(unittest.TestCase):

    def _candle(self, o, h, l, c):
        return pd.Series({"open": o, "high": h, "low": l, "close": c})

    def test_clear_topping_tail(self):
        # open=5.0, close=5.1, high=6.0, low=4.95
        # upper_wick = 6.0 - 5.1 = 0.90
        # body = 0.1, lower_wick = 5.0 - 4.95 = 0.05
        # ratio = 0.90 / 0.15 = 6.0 >> TOPPING_TAIL_RATIO (default 0.5)
        candle = self._candle(o=5.0, h=6.0, l=4.95, c=5.1)
        self.assertTrue(ResistanceMap.detect_topping_tail(candle))

    def test_strong_bullish_no_topping_tail(self):
        # Open=5.0, close=5.5, high=5.5, low=4.95 → no upper wick
        candle = self._candle(o=5.0, h=5.5, l=4.95, c=5.5)
        self.assertFalse(ResistanceMap.detect_topping_tail(candle))

    def test_bearish_no_topping_tail(self):
        # Open=5.5, close=5.0, high=5.6, low=4.9
        # upper_wick = 5.6 - 5.5 = 0.1; body = 0.5; lower = 5.0-4.9 = 0.1
        # ratio = 0.1 / 0.6 = 0.167 < 0.5
        candle = self._candle(o=5.5, h=5.6, l=4.9, c=5.0)
        self.assertFalse(ResistanceMap.detect_topping_tail(candle))

    def test_zero_denominator_returns_false(self):
        # All prices equal → no body, no lower wick → denominator=0
        candle = self._candle(o=5.0, h=5.0, l=5.0, c=5.0)
        self.assertFalse(ResistanceMap.detect_topping_tail(candle))

    def test_doji_with_upper_and_lower_wicks(self):
        # open=close=5.0, high=5.1, low=4.9
        # upper_wick=0.1, body=0, lower=0.1
        # ratio = 0.1/0.1 = 1.0 > 0.5 → True
        candle = self._candle(o=5.0, h=5.1, l=4.9, c=5.0)
        expected = (0.1 / 0.1) > TOPPING_TAIL_RATIO
        self.assertEqual(ResistanceMap.detect_topping_tail(candle), expected)


class TestFormatLevelsDisplay(unittest.TestCase):

    def test_static_dollar_levels_hidden_but_structure_still_shown(self):
        rmap = _make_rmap(price=4.85)
        now = datetime.now(tz=timezone.utc)
        rmap._levels.extend([
            ResistanceLevel(5.00, LevelType.STATIC, 0.80, "whole dollar", now),
            ResistanceLevel(5.50, LevelType.STATIC, 0.65, "half dollar", now),
            ResistanceLevel(5.11, LevelType.STRUCTURE, 0.85, "5-min HOD pivot", now),
        ])

        out = format_levels("TEST", SimpleNamespace(price=4.85), rmap, current_price=4.85)

        self.assertIn("5-min HOD pivot", out)
        self.assertNotIn("whole dollar", out)
        self.assertNotIn("half dollar", out)
        self.assertNotIn("[static]", out)


# ---------------------------------------------------------------------------
# ResistanceMap.summary
# ---------------------------------------------------------------------------

class TestSummary(unittest.TestCase):

    def test_summary_contains_symbol(self):
        rmap = _make_rmap(price=4.85)
        rmap.build_static()
        summary = rmap.summary()
        self.assertIn("TEST", summary)

    def test_summary_no_levels_message(self):
        rmap = _make_rmap(price=4.85)
        summary = rmap.summary()
        self.assertIn("No resistance", summary)


if __name__ == "__main__":
    unittest.main()
