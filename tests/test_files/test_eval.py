"""
tests/test_eval.py

Unit tests for strategy/evaluator.py — quality-function evaluation,
WatchlistEntry confidence logic, helper methods, and validate_ticker.

No API calls are made — all tested methods are pure static functions or
operate only on plain Python data.
"""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from strategy.evaluator import (
    Stage1Result,
    StockEvaluator,
    CompositeEvaluation,
    WatchlistEntry,
)
from market_data.finnhub_client import FinnhubNewsArticle
from market_data.fmp_client import FloatData
from strategy.scoring import (
    VariableGroup,
    ScoringVariable,
    QualityFn,
    build_groups,
)
from strategy.criteria import build_criteria_groups
from config import (
    CONFIDENCE_FLOOR,
    CONFIDENCE_CEILING,
    CRITERIA_EXECUTION,
    SCORING_GROUPS,
    SECTORS_TIER1,
    SECTORS_TIER2,
    SECTORS_TIER3,
)
from pipeline import validate_ticker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluation(**overrides) -> CompositeEvaluation:
    """Build a CompositeEvaluation from SCORING_GROUPS with optional quality overrides.

    *overrides*: ``{"STOCK_EVAL:breaking_news": 0.8, "STOCK_EVAL:float_shares": 0.9, ...}``
    Sets the specified variables to the given quality (and value=quality).
    All other variables default to 0.5 (neutral).
    """
    groups = build_groups(SCORING_GROUPS)
    gmap = {g.name: g for g in groups}
    # Set all variables to neutral first
    for g in groups:
        for v in g.variables:
            v.quality = 0.5
            v.value = 0.5
    # Apply overrides
    for key, quality in overrides.items():
        group_name, var_name = key.split(":", 1)
        g = gmap.get(group_name)
        if g:
            v = g.get(var_name)
            if v:
                v.quality = quality
                v.value = quality
    return CompositeEvaluation(
        stock_eval=gmap["STOCK_EVAL"],
        technical_analysis=gmap["TECHNICAL_ANALYSIS"],
    )


def _make_entry(**kwargs) -> WatchlistEntry:
    ev = _make_evaluation()
    defaults = dict(
        symbol="TEST",
        evaluated_at=datetime.now(tz=timezone.utc),
        price=5.00,
        change_pct=15.0,
        volume=500_000,
        avg_volume=100_000,
        rvol_1day=5.0,
        rvol_5min=7.0,
        daily_high=5.50,
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
    defaults.update(kwargs)
    return WatchlistEntry(**defaults)


# ---------------------------------------------------------------------------
# validate_ticker
# ---------------------------------------------------------------------------

class TestValidateTicker(unittest.TestCase):

    def test_uppercase_normalisation(self):
        self.assertEqual(validate_ticker("aapl"), "AAPL")
        self.assertEqual(validate_ticker("TSLA"), "TSLA")

    def test_strips_whitespace(self):
        self.assertEqual(validate_ticker("  ACMR  "), "ACMR")

    def test_single_letter(self):
        self.assertEqual(validate_ticker("A"), "A")

    def test_four_letters_max(self):
        self.assertEqual(validate_ticker("ABCD"), "ABCD")

    def test_five_letters_valid(self):
        self.assertEqual(validate_ticker("ABCDE"), "ABCDE")

    def test_numeric_invalid(self):
        self.assertIsNone(validate_ticker("12345"))

    def test_alphanumeric_invalid(self):
        self.assertIsNone(validate_ticker("A1B2"))

    def test_empty_invalid(self):
        self.assertIsNone(validate_ticker(""))

    def test_space_in_middle_invalid(self):
        self.assertIsNone(validate_ticker("A B"))


# ---------------------------------------------------------------------------
# _sector_quality (replaced _delta_sector)
# ---------------------------------------------------------------------------

class TestSectorQuality(unittest.TestCase):

    def test_tier1_defense(self):
        self.assertEqual(StockEvaluator._sector_quality("Defense", "Aerospace"), 1.0)

    def test_tier1_military(self):
        self.assertEqual(StockEvaluator._sector_quality("Military", "Government"), 1.0)

    def test_tier2_energy(self):
        self.assertAlmostEqual(StockEvaluator._sector_quality("Energy", "Oil & Gas"), 0.7)

    def test_tier2_solar(self):
        self.assertAlmostEqual(StockEvaluator._sector_quality("Utilities", "Solar Energy"), 0.7)

    def test_tier3_semiconductor(self):
        self.assertAlmostEqual(StockEvaluator._sector_quality("Technology", "Semiconductors"), 0.4)

    def test_tier3_software(self):
        self.assertAlmostEqual(StockEvaluator._sector_quality("Technology", "Software"), 0.4)

    def test_no_match(self):
        self.assertEqual(StockEvaluator._sector_quality("Healthcare", "Biotech"), 0.0)
        self.assertEqual(StockEvaluator._sector_quality("", ""), 0.0)

    def test_tier1_keyword_beats_tier2(self):
        self.assertEqual(
            StockEvaluator._sector_quality("Aerospace Energy Systems", "Defense"),
            1.0,
        )


# ---------------------------------------------------------------------------
# _daily_pct (replaced _delta_daily)
# ---------------------------------------------------------------------------

class TestDailyPct(unittest.TestCase):

    def _df(self, closes):
        """Build a DataFrame with high=close, low=close for simple cases."""
        return pd.DataFrame({"close": closes, "high": closes, "low": closes})

    def test_none_returns_zero(self):
        self.assertEqual(StockEvaluator._daily_pct(None), 0.0)

    def test_too_short_returns_zero(self):
        self.assertEqual(StockEvaluator._daily_pct(self._df([1.0])), 0.0)

    def test_uptrend(self):
        # Range = (110-100)/105 * 100 ~= 9.52, but last bar excluded so prior=[100..108]
        # max_high=108, min_low=100, avg_close=104 -> (108-100)/104*100 ~= 7.69
        df = self._df([100, 102, 104, 106, 108, 110])
        pct = StockEvaluator._daily_pct(df)
        self.assertAlmostEqual(pct, (108 - 100) / 104 * 100, places=1)

    def test_timestamped_daily_history_without_today_uses_last_completed_bar(self):
        df = self._df([100, 102, 104, 106, 108])
        df["ts"] = pd.date_range(end=pd.Timestamp.now(tz="America/New_York").date() - pd.Timedelta(days=1), periods=5, freq="D", tz="UTC")
        pct = StockEvaluator._daily_pct(df)
        self.assertAlmostEqual(pct, (108 - 100) / 104 * 100, places=1)

    def test_timestamped_daily_history_excludes_today(self):
        df = self._df([100, 102, 104, 106, 150])
        today = pd.Timestamp.now(tz="America/New_York").date()
        df["ts"] = pd.date_range(end=today, periods=5, freq="D", tz="America/New_York")
        pct = StockEvaluator._daily_pct(df)
        self.assertAlmostEqual(pct, (106 - 100) / 103 * 100, places=1)

    def test_flat(self):
        pct = StockEvaluator._daily_pct(self._df([100, 100, 100, 100, 100, 100]))
        self.assertAlmostEqual(pct, 0.0)

    def test_wide_range(self):
        # Stock went from $2 to $10 -- wide chart range
        df = pd.DataFrame({"close": [2, 4, 6, 8, 10, 12],
                           "high":  [3, 5, 7, 9, 11, 13],
                           "low":   [1, 3, 5, 7, 9, 11]})
        pct = StockEvaluator._daily_pct(df)
        # prior: high max=11, low min=1, avg close=6 -> (11-1)/6*100 ~= 166.7
        self.assertAlmostEqual(pct, (11 - 1) / 6 * 100, places=1)

    def test_selloff_is_not_tight_base(self):
        df = pd.DataFrame({
            "close": [6.80, 6.45, 6.10, 5.95, 5.72, 5.48, 5.22, 5.05, 5.30],
            "high":  [6.95, 6.55, 6.20, 6.05, 5.88, 5.62, 5.38, 5.18, 5.40],
            "low":   [6.35, 6.05, 5.80, 5.60, 5.35, 5.12, 4.92, 4.82, 5.12],
        })
        pct = StockEvaluator._daily_pct(df)
        self.assertGreater(pct, 200.0)

    def test_selloff_quality_is_below_tight_base_cutoff(self):
        df = pd.DataFrame({
            "close": [6.80, 6.45, 6.10, 5.95, 5.72, 5.48, 5.22, 5.05, 5.30],
            "high":  [6.95, 6.55, 6.20, 6.05, 5.88, 5.62, 5.38, 5.18, 5.40],
            "low":   [6.35, 6.05, 5.80, 5.60, 5.35, 5.12, 4.92, 4.82, 5.12],
        })
        pct = StockEvaluator._daily_pct(df)
        stock_eval = next(group for group in build_groups(SCORING_GROUPS) if group.name == "STOCK_EVAL")
        daily_var = stock_eval.get("daily_trend")
        self.assertIsNotNone(daily_var)
        assert daily_var is not None
        quality = daily_var.evaluate(pct)
        self.assertLess(quality, 0.5)


# ---------------------------------------------------------------------------
# _above_200 (replaced _delta_ma200)
# ---------------------------------------------------------------------------

class TestAvgDailyVolume(unittest.TestCase):

    def test_quote_average_trusted_when_present(self):
        df = pd.DataFrame({"volume": [1000, 2000, 3000]})
        self.assertEqual(StockEvaluator._avg_daily_volume(df, 500_000), 500_000)

    def test_derived_from_daily_bars_when_quote_clamped(self):
        df = pd.DataFrame({"volume": [1000, 2000, 3000]})
        self.assertEqual(StockEvaluator._avg_daily_volume(df, 1), 2000)

    def test_uses_only_last_n_bars(self):
        from config import AVG_VOLUME_DAILY_BARS
        vols = [1_000_000] * 10 + [2000] * AVG_VOLUME_DAILY_BARS
        df = pd.DataFrame({"volume": vols})
        self.assertEqual(StockEvaluator._avg_daily_volume(df, 1), 2000)

    def test_no_daily_data_falls_back_to_quote(self):
        self.assertEqual(StockEvaluator._avg_daily_volume(None, 1), 1)
        self.assertEqual(StockEvaluator._avg_daily_volume(pd.DataFrame(), 0), 0)


class TestAbove200(unittest.TestCase):

    def _df_with_ma(self, ema200=None, ma200=None):
        row = {"close": [5.0]}
        if ema200 is not None:
            row["ema_200"] = [ema200]
        if ma200 is not None:
            row["ma_200"] = [ma200]
        return pd.DataFrame(row)

    def test_none_df_returns_neutral(self):
        self.assertAlmostEqual(StockEvaluator._above_200(5.0, None), 0.5)

    def test_no_ma_columns_returns_neutral(self):
        df = pd.DataFrame({"close": [5.0]})
        self.assertAlmostEqual(StockEvaluator._above_200(5.0, df), 0.5)

    def test_above_both(self):
        df = self._df_with_ma(ema200=4.0, ma200=3.5)
        self.assertAlmostEqual(StockEvaluator._above_200(5.0, df), 1.0)

    def test_below_both(self):
        df = self._df_with_ma(ema200=6.0, ma200=7.0)
        self.assertAlmostEqual(StockEvaluator._above_200(5.0, df), 0.0)

    def test_below_one(self):
        df = self._df_with_ma(ema200=4.0, ma200=6.0)
        self.assertAlmostEqual(StockEvaluator._above_200(5.0, df), 0.0)


# ---------------------------------------------------------------------------
# _ath_ratio (replaced _delta_ipo ATH check)
# ---------------------------------------------------------------------------

class TestAthRatio(unittest.TestCase):

    def test_none_df_returns_neutral(self):
        self.assertAlmostEqual(StockEvaluator._ath_ratio(5.0, None), 0.5)

    def test_at_ath(self):
        df = pd.DataFrame({"high": [4.0, 5.0, 4.8, 5.1, 5.2]})
        ratio = StockEvaluator._ath_ratio(5.2, df)
        self.assertAlmostEqual(ratio, 1.0)

    def test_well_below_ath(self):
        df = pd.DataFrame({"high": [10.0, 9.5, 8.0, 7.5, 7.0]})
        ratio = StockEvaluator._ath_ratio(5.0, df)
        self.assertAlmostEqual(ratio, 0.5)


# ---------------------------------------------------------------------------
# _delta_split (deprecated compat)
# ---------------------------------------------------------------------------

class TestDeltaSplit(unittest.TestCase):

    def test_no_split(self):
        self.assertEqual(StockEvaluator._delta_split(False, None), 0.0)

    def test_has_split(self):
        self.assertEqual(StockEvaluator._delta_split(True, 5.0), 1.0)

    def test_has_split_no_ratio(self):
        self.assertEqual(StockEvaluator._delta_split(True, None), 1.0)


# ---------------------------------------------------------------------------
# _delta_ma200 (deprecated compat — delegates to _above_200)
# ---------------------------------------------------------------------------

class TestDeltaMa200Compat(unittest.TestCase):

    def _df_with_ma(self, ema200=None, ma200=None):
        row = {"close": [5.0]}
        if ema200 is not None:
            row["ema_200"] = [ema200]
        if ma200 is not None:
            row["ma_200"] = [ma200]
        return pd.DataFrame(row)

    def test_none_df(self):
        self.assertAlmostEqual(StockEvaluator._delta_ma200(5.0, None), 0.5)

    def test_above_both(self):
        df = self._df_with_ma(ema200=4.0, ma200=3.5)
        self.assertAlmostEqual(StockEvaluator._delta_ma200(5.0, df), 1.0)

    def test_below_both(self):
        df = self._df_with_ma(ema200=6.0, ma200=7.0)
        self.assertAlmostEqual(StockEvaluator._delta_ma200(5.0, df), 0.0)


# ---------------------------------------------------------------------------
# CompositeEvaluation container
# ---------------------------------------------------------------------------

class TestCompositeEvaluation(unittest.TestCase):
    def test_group_by_name(self):
        ev = _make_evaluation()
        self.assertIsNotNone(ev.group_by_name("STOCK_EVAL"))
        self.assertIsNotNone(ev.group_by_name("TECHNICAL_ANALYSIS"))
        self.assertIsNone(ev.group_by_name("NONEXISTENT"))

    def test_groups_returns_all(self):
        ev = _make_evaluation()
        self.assertEqual(len(ev.groups), 2)


# ---------------------------------------------------------------------------
# WatchlistEntry — confidence scoring
# ---------------------------------------------------------------------------

class TestWatchlistEntry(unittest.TestCase):

    def test_eval_and_live_confidence_ranges(self):
        entry = _make_entry()
        self.assertGreaterEqual(entry.eval_confidence, 0.0)
        self.assertLessEqual(entry.eval_confidence, 100.0)
        self.assertGreaterEqual(entry.live_confidence, 0.0)
        self.assertLessEqual(entry.live_confidence, 100.0)

    def test_eval_and_live_confidence_all_neutral_near_50(self):
        entry = _make_entry()
        self.assertAlmostEqual(entry.eval_confidence, 50.0, delta=5.0)
        self.assertAlmostEqual(entry.live_confidence, 50.0, delta=5.0)

    def test_live_confidence_is_ta_only(self):
        dynamic_vars = (
            "setup_outcome",
            "key_candles",
            "atrp",
            "candle_choppiness",
        )
        ev = _make_evaluation(**{
            f"TECHNICAL_ANALYSIS:{name}": 1.0
            for name in dynamic_vars
        })
        entry = _make_entry(evaluation=ev)

        self.assertEqual(entry.live_confidence, 100.0)

    def test_live_confidence_uses_ta_score_override(self):
        ev = _make_evaluation(**{
            "TECHNICAL_ANALYSIS:setup_outcome": 1.0,
        })
        ev.technical_analysis.score_override = 0.42
        entry = _make_entry(evaluation=ev)

        self.assertEqual(entry.live_confidence, 42.0)

    def test_book_spread_is_execution_criteria_not_live_ta_score(self):
        ev = _make_evaluation()
        execg = build_criteria_groups(CRITERIA_EXECUTION)[0]
        spread = execg.get("spread_ok")

        self.assertIsNone(ev.technical_analysis.get("book_spread"))
        self.assertIsNotNone(spread)
        self.assertTrue(spread.required)

    def test_active_weaknesses_derived(self):
        ev = _make_evaluation(**{"STOCK_EVAL:float_shares": 0.1})
        entry = _make_entry(evaluation=ev)
        self.assertTrue(any("float_shares" in w for w in entry.active_weaknesses))

    def test_active_strengths_derived(self):
        ev = _make_evaluation(**{"STOCK_EVAL:breaking_news": 0.9})
        entry = _make_entry(evaluation=ev)
        self.assertTrue(any("breaking_news" in s for s in entry.active_strengths))

    def test_eval_and_live_confidence_floored(self):
        overrides = {}
        groups = build_groups(SCORING_GROUPS)
        for g in groups:
            for v in g.variables:
                overrides[f"{g.name}:{v.name}"] = 0.0
        ev = _make_evaluation(**overrides)
        entry = _make_entry(evaluation=ev)
        self.assertEqual(entry.eval_confidence, CONFIDENCE_FLOOR)
        self.assertEqual(entry.live_confidence, CONFIDENCE_FLOOR)

    def test_eval_and_live_confidence_capped(self):
        overrides = {}
        groups = build_groups(SCORING_GROUPS)
        for g in groups:
            for v in g.variables:
                overrides[f"{g.name}:{v.name}"] = 1.0
        ev = _make_evaluation(**overrides)
        entry = _make_entry(evaluation=ev)
        self.assertEqual(entry.eval_confidence, CONFIDENCE_CEILING)
        self.assertEqual(entry.live_confidence, CONFIDENCE_CEILING)

    def test_none_evaluation_returns_zero(self):
        entry = _make_entry(evaluation=None)
        self.assertEqual(entry.eval_confidence, 0.0)
        self.assertEqual(entry.live_confidence, 0.0)

    def test_higher_quality_means_higher_eval_confidence(self):
        ev_low = _make_evaluation(**{"STOCK_EVAL:breaking_news": 0.1})
        ev_high = _make_evaluation(**{"STOCK_EVAL:breaking_news": 0.9})
        entry_low = _make_entry(evaluation=ev_low)
        entry_high = _make_entry(evaluation=ev_high)
        self.assertGreater(entry_high.eval_confidence, entry_low.eval_confidence)


# ---------------------------------------------------------------------------
# Quote prefilter boundary
# ---------------------------------------------------------------------------

class TestStage1QuotePrefilter(unittest.TestCase):

    def test_low_volume_and_rvol_do_not_fail_quote_prefilter(self):
        evaluator = StockEvaluator.__new__(StockEvaluator)
        evaluator._tradier = MagicMock()
        evaluator._tradier.get_quote.return_value = SimpleNamespace(
            last=5.0,
            change_pct=20.0,
            volume=1,
            avg_volume=100_000,
            high=5.2,
            prev_close=4.0,
            bid=4.99,
            ask=5.01,
            description="Test Corp",
            exchange="Q",
        )
        evaluator._compute_rvol_at_time = MagicMock(return_value=(0.1, 0.1))

        with patch("strategy.evaluator.MARKET_DATA_ENABLED", True), patch("strategy.evaluator.MarketClock.is_rth", return_value=True):
            result = evaluator._stage1("TEST")

        self.assertTrue(result.passed)
        self.assertEqual(result.volume, 1)
        self.assertAlmostEqual(result.rvol_1day, 0.1)

    def test_low_day_change_does_not_fail_quote_prefilter(self):
        """% change is gated by the scanner floor (pre-evaluation) and the
        live ATTENTION criterion (entry), not by the quote prefilter."""
        evaluator = StockEvaluator.__new__(StockEvaluator)
        evaluator._tradier = MagicMock()
        evaluator._tradier.get_quote.return_value = SimpleNamespace(
            last=5.0,
            change_pct=2.5,          # well under CHANGE_PCT_MIN
            volume=500_000,
            avg_volume=100_000,
            high=5.2,
            prev_close=4.88,
            bid=4.99,
            ask=5.01,
            description="Test Corp",
            exchange="Q",
        )
        evaluator._compute_rvol_at_time = MagicMock(return_value=(6.0, 6.0))

        with patch("strategy.evaluator.MARKET_DATA_ENABLED", True), patch("strategy.evaluator.MarketClock.is_rth", return_value=True):
            result = evaluator._stage1("TEST")

        self.assertTrue(result.passed)
        self.assertEqual(result.fail_reason, "")
        self.assertAlmostEqual(result.change_pct, 2.5)

    def test_price_band_still_rejects(self):
        evaluator = StockEvaluator.__new__(StockEvaluator)
        evaluator._tradier = MagicMock()
        evaluator._tradier.get_quote.return_value = SimpleNamespace(
            last=45.0,               # above PRICE_MAX
            change_pct=30.0,
            volume=500_000,
            avg_volume=100_000,
            high=46.0,
            prev_close=34.6,
            bid=44.99,
            ask=45.01,
            description="Test Corp",
            exchange="Q",
        )
        evaluator._compute_rvol_at_time = MagicMock(return_value=(6.0, 6.0))

        with patch("strategy.evaluator.MARKET_DATA_ENABLED", True), patch("strategy.evaluator.MarketClock.is_rth", return_value=True):
            result = evaluator._stage1("TEST")

        self.assertFalse(result.passed)
        self.assertIn("price", result.fail_reason)


# ---------------------------------------------------------------------------
# Regression tests — migration hardening
# ---------------------------------------------------------------------------

class TestEvaluatorRegressions(unittest.TestCase):

    def test_refresh_news_updates_catalyst_without_legacy_fields(self):
        evaluator = StockEvaluator.__new__(StockEvaluator)
        evaluator._momo = MagicMock()
        evaluator._finnhub = MagicMock()
        evaluator._brave = MagicMock()

        now = datetime.now(tz=timezone.utc)
        old_pub = now - timedelta(hours=2)
        entry = _make_entry(
            news_articles=[SimpleNamespace(published_at=old_pub, title="Old headline")],
            news_count_last_hour=1,
            avg_sentiment=None,
            news_source="",
        )

        evaluator._momo.get_news.return_value = [
            SimpleNamespace(
                symbol="TEST",
                headline="Company wins major contract",
                source="momo",
                url="",
                published_at=now,
            )
        ]

        evaluator._finnhub.get_profile.return_value = None
        evaluator._filter_and_score = MagicMock(return_value=(
            [
                FinnhubNewsArticle(
                    symbol="TEST",
                    title="Company wins major contract",
                    source="momo",
                    url="",
                    published_at=now,
                    summary="",
                )
            ],
            1,
            0.6,
        ))

        with patch("strategy.evaluator.FINNHUB_ENABLED", False), patch("strategy.evaluator.BRAVE_ENABLED", False):
            updated = evaluator.refresh_news(entry)

        self.assertTrue(updated)
        self.assertEqual(entry.news_source, "momo")
        v = entry.evaluation.stock_eval.get("breaking_news")
        self.assertIsNotNone(v)
        self.assertAlmostEqual(v.value, 0.8)

    def test_stage2_profile_future_failure_does_not_raise(self):
        evaluator = StockEvaluator.__new__(StockEvaluator)
        evaluator._eval_stats = {}
        evaluator.last_fail_reason = ""

        evaluator._momo = MagicMock()
        evaluator._momo.get_float.return_value = FloatData(
            symbol="TEST",
            float_shares=2_000_000,
            float_shares_m=2.0,
            outstanding_shares=3_000_000,
            free_float_pct=66.7,
        )
        evaluator._momo.get_news.return_value = []

        evaluator._finnhub = MagicMock()
        evaluator._finnhub.get_profile.side_effect = RuntimeError("profile down")
        evaluator._finnhub.get_recent_news.return_value = []

        evaluator._av = None
        evaluator._fmp = None
        evaluator._brave = MagicMock()
        evaluator._finbert = MagicMock()
        evaluator._filter_and_score = MagicMock(return_value=([], 0, None))

        s1 = Stage1Result(
            symbol="TEST",
            price=5.0,
            change_pct=12.0,
            volume=1_000_000,
            avg_volume=100_000,
            rvol_1day=10.0,
            rvol_5min=5.0,
            daily_high=5.2,
            passed=True,
        )

        with patch("strategy.evaluator.AV_ENABLED", False), patch("strategy.evaluator.FINNHUB_ENABLED", True), patch("strategy.evaluator.BRAVE_ENABLED", False):
            entry = evaluator._stage2(s1, df_daily=None)

        self.assertIsNotNone(entry)
        self.assertEqual(entry.symbol, "TEST")
        self.assertEqual(entry.sector, "")

    def test_stage2_float_gate_applies_when_av_disabled(self):
        evaluator = StockEvaluator.__new__(StockEvaluator)
        evaluator._eval_stats = {}
        evaluator.last_fail_reason = ""

        evaluator._momo = MagicMock()
        evaluator._momo.get_float.return_value = FloatData(
            symbol="TEST",
            float_shares=75_000_000,
            float_shares_m=75.0,
            outstanding_shares=80_000_000,
            free_float_pct=93.8,
        )
        evaluator._fmp = None
        evaluator._finnhub = MagicMock()
        evaluator._av = None
        evaluator._brave = MagicMock()
        evaluator._finbert = MagicMock()

        s1 = Stage1Result(
            symbol="TEST",
            price=5.0,
            change_pct=12.0,
            volume=1_000_000,
            avg_volume=100_000,
            rvol_1day=10.0,
            rvol_5min=5.0,
            daily_high=5.2,
            passed=True,
        )

        with patch("strategy.evaluator.AV_ENABLED", False), patch("strategy.evaluator.FINNHUB_ENABLED", False), patch("strategy.evaluator.BRAVE_ENABLED", False):
            entry = evaluator._stage2(s1, df_daily=None)

        self.assertIsNone(entry)
        self.assertIn("float", evaluator.last_fail_reason)


if __name__ == "__main__":
    unittest.main()
