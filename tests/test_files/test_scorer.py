"""
tests/test_scorer.py

Unit tests for strategy/confidence_scorer.py.

The live stream (RealTimeClient) is fully mocked — no network calls.
Tests inject synthetic timesale/quote events directly into the scorer's
handlers and verify that quality variables are updated correctly.
"""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")

import csv
import tempfile
import threading
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from chart_cache import LiveChartCache, SOURCE_POLLED, SOURCE_STREAM, load_candle_cache, load_timesales_cache, timesales_cache_path
from strategy.evaluator import WatchlistEntry, CompositeEvaluation
from strategy.scoring import build_groups, combine_tf_scores
from technical_analysis.patterns import latest_extended_session_start
from strategy.criteria import build_criteria_groups
from strategy.confidence_scorer import (
    DynamicScorer,
    _SETUP_RESOLVED_DISPLAY_SECS,
    ROC_TS_PERIOD as ROC_PERIOD,
    BOOK_ILLIQUID_MAX_SIZE as ILLIQUID_MAX_SIZE,
    BOOK_ILLIQUID_CONSEC as ILLIQUID_CONSECUTIVE,
    OFI_SELL_DOMINANT_THRESH, OFI_BUY_SURGE_THRESH,
    BOOK_LEAN_BEARISH_THRESH,
    OFI_CHOP_RATIO_THRESH, OFI_EXHAUSTION_THRESH, OFI_EXHAUSTION_MIN_QUOTES,
    _candle_choppiness_ratio,
)
from config import (
    SCORING_GROUPS,
    CONFIDENCE_FLOOR,
    CONFIDENCE_CEILING,
    MarketClock,
    ATTENTION_VOLUME_MIN,
    CRITERIA_ATTENTION,
    CRITERIA_EXECUTION,
    SETUP_PATTERN_GROUP,
)
from market_data.tradier_rest import OHLCVCandle
from market_data.tradier_stream import LiveTimesale, LiveQuote
from display import watch_status_line
from exec.order_flow import OrderFlowTracker
from exec.setup_logger import SetupSignalLogger
from technical_analysis.resistance import ResistanceMap
from technical_analysis.patterns import PatternSnapshot, PatternSignal


class TestSessionKeyBoundaries(unittest.TestCase):
    def test_four_am_belongs_to_new_premarket_session(self):
        ts = datetime(2026, 6, 3, 4, 0, tzinfo=MarketClock.ET)
        self.assertEqual(DynamicScorer._session_key_from_utc(ts), "2026-06-03-0")

    def test_one_minute_before_four_am_belongs_to_prior_postmarket_session(self):
        ts = datetime(2026, 6, 3, 3, 59, tzinfo=MarketClock.ET)
        self.assertEqual(DynamicScorer._session_key_from_utc(ts), "2026-06-02-3")

    def test_regular_open_belongs_to_rth_session(self):
        ts = datetime(2026, 6, 3, 9, 30, tzinfo=MarketClock.ET)
        self.assertEqual(DynamicScorer._session_key_from_utc(ts), "2026-06-03-1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_evaluation(**overrides) -> CompositeEvaluation:
    """Build a CompositeEvaluation with all variables at 0.5 (neutral) unless overridden."""
    groups = build_groups(SCORING_GROUPS)
    gmap = {g.name: g for g in groups}
    for g in groups:
        for v in g.variables:
            v.quality = 0.5
            v.value = 0.5
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


def _make_entry() -> WatchlistEntry:
    ev = _make_evaluation()
    return WatchlistEntry(
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


def _make_scorer(entry: WatchlistEntry = None, price: float = 5.00, **kwargs) -> DynamicScorer:
    """Build a DynamicScorer with mocked shared stream + HistoricalClient (no network)."""
    if entry is None:
        entry = _make_entry()
    avg_daily_volume = kwargs.pop("avg_daily_volume", 100_000)
    rmap = ResistanceMap(symbol="TEST", current_price=price)
    rmap.build_static()

    with patch("strategy.confidence_scorer.get_shared_stream") as mock_gs, \
         patch("strategy.confidence_scorer.get_historical_client") as MockHist:
        mock_gs.return_value = MagicMock()
        MockHist.return_value = MagicMock()
        scorer = DynamicScorer(entry, rmap, avg_daily_volume=avg_daily_volume, **kwargs)

    return scorer


def _seed_1min_frame(rows: int = 12) -> pd.DataFrame:
    base = pd.Timestamp("2026-06-02 20:00:00", tz="UTC")
    return pd.DataFrame([
        {
            "symbol": "TEST",
            "ts": base + timedelta(minutes=i),
            "open": 5.00 + i * 0.01,
            "high": 5.05 + i * 0.01,
            "low": 4.95 + i * 0.01,
            "close": 5.02 + i * 0.01,
            "volume": 10_000 + i,
            "interval": "1min",
            "vwap": None,
        }
        for i in range(rows)
    ])


def _make_quote(bid: float = 5.00, ask: float = 5.02, last: float = 0.0,
                last_size: int = 0) -> LiveQuote:
    ts = datetime.now(tz=timezone.utc)
    return LiveQuote("quote", "TEST", bid, 100, "", ts, ask, 200, "", ts,
                     last=last, last_size=last_size)


class TestQuotePriceFallback(unittest.TestCase):
    """Snapshot price falls back to the L1 quote-stream last trade when the
    tape is absent or stale — a symbol without a tape line would otherwise
    show a frozen %-change in the top-gainers display forever."""

    def test_snapshot_uses_quote_last_when_no_prints_ever(self):
        scorer = _make_scorer()
        scorer._on_quote(_make_quote(last=5.01, last_size=50))
        self.assertEqual(scorer.snapshot().last_price, 5.01)

    def test_fresh_prints_beat_the_quote_fallback(self):
        scorer = _make_scorer()
        scorer._last_price = 5.10
        scorer._last_event_ts = datetime.now(tz=timezone.utc)
        scorer._on_quote(_make_quote(last=5.01))
        self.assertEqual(scorer.snapshot().last_price, 5.10)

    def test_stale_prints_yield_to_quote_last(self):
        scorer = _make_scorer()
        scorer._last_price = 5.10
        scorer._last_event_ts = datetime.now(tz=timezone.utc) - timedelta(seconds=60)
        scorer._on_quote(_make_quote(last=5.20))
        self.assertEqual(scorer.snapshot().last_price, 5.20)

    def test_quote_last_raises_hod(self):
        scorer = _make_scorer()   # entry.daily_high starts at 5.50
        scorer._on_quote(_make_quote(last=6.00))
        self.assertEqual(scorer._entry.daily_high, 6.00)

    def test_tradier_quotes_carry_no_last_and_change_nothing(self):
        scorer = _make_scorer()
        scorer._on_quote(_make_quote())          # last defaults to 0.0
        self.assertEqual(scorer._last_quote_price, 0.0)
        self.assertEqual(scorer._entry.daily_high, 5.50)


class TestScorerPreloadedCandles(unittest.TestCase):
    def test_watch_status_line_does_not_block_on_busy_scorer(self):
        entry = _make_entry()
        scorer = _make_scorer(entry=entry)
        self.assertTrue(scorer._lock.acquire(blocking=False))
        try:
            text = watch_status_line(SimpleNamespace(watchlist=[entry]), scorers={entry.symbol: scorer})
        finally:
            scorer._lock.release()

        self.assertIn("No active tickers", text)

    def test_preloaded_1min_seed_skips_initial_full_fetch_and_preserves_move_window(self):
        entry = _make_entry()
        rmap = ResistanceMap(symbol="TEST", current_price=5.10)
        seed = _seed_1min_frame()
        move_start = pd.Timestamp(seed.iloc[4]["ts"]).to_pydatetime()

        with patch("strategy.confidence_scorer.get_shared_stream") as mock_gs, \
             patch("strategy.confidence_scorer.get_historical_client") as MockHist:
            hist = MagicMock()
            hist.get_candles_1min.side_effect = AssertionError("seed should avoid full 1m fetch")
            hist.get_candles_1min_recent.return_value = pd.DataFrame()
            mock_gs.return_value = MagicMock()
            MockHist.return_value = hist

            scorer = DynamicScorer(
                entry,
                rmap,
                avg_daily_volume=100_000,
                seed_1min_session=seed,
                move_start_ts=move_start,
            )

        df = scorer._fetch_1min_candles()

        hist.get_candles_1min.assert_not_called()
        hist.get_candles_1min_recent.assert_called_once()
        self.assertEqual(len(df), len(seed) - 4)
        self.assertEqual(pd.Timestamp(df.iloc[0]["ts"]).to_pydatetime(), move_start)
        self.assertEqual(len(scorer._df_1min_session), len(seed))

    def test_first_1min_fetch_anchors_to_session_not_bar_count(self):
        """A bar count walks back from now and can cross into yesterday,
        seeding VWAP with the prior session; the anchored window cannot.
        get_candles_1min also ignores n and pulls CANDLES_1MIN_FETCH_DAYS,
        which is the most pacing-throttled request in the system."""
        entry = _make_entry()
        rmap = ResistanceMap(symbol="TEST", current_price=5.10)

        with patch("strategy.confidence_scorer.get_shared_stream") as mock_gs,              patch("strategy.confidence_scorer.get_historical_client") as MockHist:
            hist = MagicMock()
            hist.get_candles_1min_window.return_value = pd.DataFrame()
            hist.get_candles_1min.side_effect = AssertionError(
                "first 1-min fetch must use the anchored window, not a bar count"
            )
            mock_gs.return_value = MagicMock()
            MockHist.return_value = hist
            scorer = DynamicScorer(entry, rmap, avg_daily_volume=100_000)

        scorer._df_1min_cache = pd.DataFrame()
        scorer._fetch_1min_candles()

        hist.get_candles_1min.assert_not_called()
        hist.get_candles_1min_window.assert_called_once()
        symbol_arg, start_arg = hist.get_candles_1min_window.call_args.args
        self.assertEqual(symbol_arg, "TEST")
        self.assertEqual(start_arg, latest_extended_session_start())

    def test_first_1min_fetch_falls_back_when_provider_lacks_window(self):
        """Providers without precise windows keep the old bar-count path."""
        entry = _make_entry()
        rmap = ResistanceMap(symbol="TEST", current_price=5.10)

        with patch("strategy.confidence_scorer.get_shared_stream") as mock_gs,              patch("strategy.confidence_scorer.get_historical_client") as MockHist:
            hist = MagicMock(spec=["get_candles_1min"])
            hist.get_candles_1min.return_value = pd.DataFrame()
            mock_gs.return_value = MagicMock()
            MockHist.return_value = hist
            scorer = DynamicScorer(entry, rmap, avg_daily_volume=100_000)

        scorer._df_1min_cache = pd.DataFrame()
        scorer._fetch_1min_candles()

        hist.get_candles_1min.assert_called_once()

    def test_poll_loop_releases_connection_between_polls(self):
        """A poll fetches for ~200ms once a minute. Holding the connection in
        between made the Gateway's 32-connection ceiling scale with watchlist
        size instead of with concurrent fetches."""
        entry = _make_entry()
        rmap = ResistanceMap(symbol="TEST", current_price=5.10)

        with patch("strategy.confidence_scorer.get_shared_stream") as mock_gs,              patch("strategy.confidence_scorer.get_historical_client") as MockHist:
            mock_gs.return_value = MagicMock()
            MockHist.return_value = MagicMock()
            scorer = DynamicScorer(entry, rmap, avg_daily_volume=100_000)

        scorer._stop.set()  # one pass through the loop, then exit
        polls = []
        with patch.object(scorer, "_refresh_candles", side_effect=lambda: polls.append(1)),              patch("strategy.confidence_scorer._release_ibkr_thread_conn") as mock_rel:
            scorer._poll_loop()

        # Stop is already set, so the body never runs; only the exit release.
        self.assertEqual(mock_rel.call_count, 1)

        # Now let one poll actually happen.
        scorer._stop.clear()

        def _one_then_stop():
            polls.append(1)
            scorer._stop.set()

        with patch.object(scorer, "_refresh_candles", side_effect=_one_then_stop),              patch("strategy.confidence_scorer._release_ibkr_thread_conn") as mock_rel:
            scorer._poll_loop()

        # Released after the poll AND on exit — never held across the sleep.
        self.assertEqual(mock_rel.call_count, 2)

    def test_poll_loop_releases_even_when_the_fetch_raises(self):
        """A failed poll must not leak the connection slot either."""
        entry = _make_entry()
        rmap = ResistanceMap(symbol="TEST", current_price=5.10)

        with patch("strategy.confidence_scorer.get_shared_stream") as mock_gs,              patch("strategy.confidence_scorer.get_historical_client") as MockHist:
            mock_gs.return_value = MagicMock()
            MockHist.return_value = MagicMock()
            scorer = DynamicScorer(entry, rmap, avg_daily_volume=100_000)

        def _boom():
            scorer._stop.set()
            raise RuntimeError("gateway blew up")

        with patch.object(scorer, "_refresh_candles", side_effect=_boom),              patch("strategy.confidence_scorer._release_ibkr_thread_conn") as mock_rel:
            scorer._poll_loop()

        self.assertEqual(mock_rel.call_count, 2)

    def test_poll_delay_is_jittered_within_bounds(self):
        """Startup seeds every cached symbol at once; without jitter the whole
        fleet reconnects on the same tick."""
        from config import SCORER_POLL_INTERVAL_SECS, SCORER_POLL_JITTER_SECS
        delays = {DynamicScorer._next_poll_delay() for _ in range(50)}
        self.assertGreater(len(delays), 1, "delay must not be constant")
        for d in delays:
            self.assertGreaterEqual(d, SCORER_POLL_INTERVAL_SECS)
            self.assertLessEqual(d, SCORER_POLL_INTERVAL_SECS + SCORER_POLL_JITTER_SECS)

    def test_10sec_seed_uses_move_start_window_fetch(self):
        entry = _make_entry()
        rmap = ResistanceMap(symbol="TEST", current_price=5.10)
        move_start = datetime.now(tz=timezone.utc) - timedelta(minutes=5)

        with patch("strategy.confidence_scorer.get_shared_stream") as mock_gs, \
             patch("strategy.confidence_scorer.get_historical_client") as MockHist:
            hist = MagicMock()
            hist.get_candles_10sec_window.return_value = pd.DataFrame()
            hist.get_candles_10sec.side_effect = AssertionError("10s seed should use window fetch")
            mock_gs.return_value = MagicMock()
            MockHist.return_value = hist

            scorer = DynamicScorer(
                entry,
                rmap,
                avg_daily_volume=100_000,
                move_start_ts=move_start,
            )

        scorer._seed_10s_worker()

        hist.get_candles_10sec.assert_not_called()
        hist.get_candles_10sec_window.assert_called_once()
        symbol_arg, start_arg, end_arg = hist.get_candles_10sec_window.call_args.args
        self.assertEqual(symbol_arg, "TEST")
        self.assertGreaterEqual(start_arg, move_start)
        self.assertLessEqual(start_arg, end_arg)

    def test_cached_timesales_seed_warms_order_flow_without_entry_volume_mutation(self):
        entry = _make_entry()
        original_volume = entry.volume
        rmap = ResistanceMap(symbol="TEST", current_price=5.10)
        base = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        seed_10s = pd.DataFrame([
            {
                "symbol": "TEST",
                "ts": base,
                "open": 5.00,
                "high": 5.05,
                "low": 4.98,
                "close": 5.03,
                "volume": 1000,
                "interval": "10sec",
                "vwap": 5.02,
            }
        ])
        seed_timesales = pd.DataFrame([
            {
                "symbol": "TEST",
                "ts": base + timedelta(seconds=12),
                "price": 5.04,
                "size": 200,
                "bid": 5.03,
                "ask": 5.04,
                "exchange": "Q",
                "seq": 1,
                "flag": "@",
                "session": "normal",
            },
            {
                "symbol": "TEST",
                "ts": base + timedelta(seconds=15),
                "price": 5.08,
                "size": 300,
                "bid": 5.07,
                "ask": 5.08,
                "exchange": "Q",
                "seq": 2,
                "flag": "@",
                "session": "normal",
            },
        ])

        with patch("strategy.confidence_scorer.get_shared_stream") as mock_gs, \
             patch("strategy.confidence_scorer.get_historical_client") as MockHist:
            mock_gs.return_value = MagicMock()
            MockHist.return_value = MagicMock()
            scorer = DynamicScorer(
                entry,
                rmap,
                avg_daily_volume=100_000,
                seed_10sec=seed_10s,
                seed_timesales=seed_timesales,
            )

        with scorer._lock:
            replayed = scorer._seed_timesales_from_cache_locked()

        self.assertEqual(replayed, 2)
        self.assertEqual(entry.volume, original_volume)
        # Both seed prints land in the wider activity window; the 2s signal
        # window (buying_pressure) only holds the most recent of the two.
        self.assertEqual(scorer._ofi_tracker.activity_count, 2)
        self.assertEqual(scorer._last_price, 5.08)
        self.assertEqual(scorer._vol_prints_sum, 500)
        current_10s = scorer._tf_buf.current_bar("10sec")
        self.assertIsNotNone(current_10s)
        self.assertEqual(int(current_10s.volume), 500)


def _ts_event(
    price: float,
    size: int = 100,
    ts: datetime = None,
    bid: float = 0.0,
    ask: float = 0.0,
) -> LiveTimesale:
    if ts is None:
        ts = datetime.now(tz=timezone.utc)
    event = MagicMock(spec=LiveTimesale)
    event.cancel     = False
    event.correction = False
    event.symbol = "TEST"
    event.exchange = "Q"
    event.last  = price
    event.size  = size
    event.ts    = ts
    event.bid   = bid
    event.ask   = ask
    event.seq = 0
    event.flag = "@"
    event.session = "normal"
    return event


def _quote_event(bid: float, ask: float, bid_size: int = 200, ask_size: int = 200) -> LiveQuote:
    now = datetime.now(tz=timezone.utc)
    event = MagicMock(spec=LiveQuote)
    event.bid      = bid
    event.ask      = ask
    event.bid_size = bid_size
    event.ask_size = ask_size
    event.bid_exchange = "Q"
    event.ask_exchange = "Q"
    event.bid_ts   = now
    event.ask_ts   = now
    return event


def _get_quality(scorer, group_name: str, var_name: str) -> float:
    """Utility to read a quality variable from the scorer's entry."""
    g = scorer._entry.evaluation.group_by_name(group_name)
    if g:
        v = g.get(var_name)
        if v:
            return v.quality
    return 0.5


# ---------------------------------------------------------------------------
# Spread evaluation (quality-based)
# ---------------------------------------------------------------------------

class TestSetupPatternScoring(unittest.TestCase):

    def test_dynamic_scorer_migrates_legacy_setup_pattern_group(self):
        entry = _make_entry()
        legacy_setup_def = {
            **SETUP_PATTERN_GROUP,
            "vars": [
                var_def
                for var_def in SETUP_PATTERN_GROUP["vars"]
                if var_def["name"] != "pullback_control"
            ],
        }
        entry.setup_pattern = build_groups([legacy_setup_def])[0]
        for v in entry.setup_pattern.variables:
            v.reset(0.0)

        self.assertIsNone(entry.setup_pattern.get("pullback_control"))

        _make_scorer(entry=entry)

        self.assertIsNotNone(entry.setup_pattern.get("pullback_control"))

    def test_setup_score_rewards_small_short_pullbacks_without_live_confidence_change(self):
        clean_entry = _make_entry()
        clean_scorer = _make_scorer(entry=clean_entry)
        clean_live_confidence = clean_entry.live_confidence
        clean_signal = PatternSignal(
            name="pullback_new_high",
            bias="bullish",
            strength=0.9,
            price_level=11.40,
            timeframe="1min",
            candle_count=5,
            pullback_depth=0.12,
            pullback_candles=1,
            pullback_control=0.90,
            leg_candles=4,
            topping_tail=0.0,
            jackknife=0.0,
        )
        clean_scorer._fill_setup_group(clean_entry.setup_pattern, clean_signal)
        clean_entry.setup_pattern_name = clean_signal.name

        sloppy_entry = _make_entry()
        sloppy_scorer = _make_scorer(entry=sloppy_entry)
        sloppy_live_confidence = sloppy_entry.live_confidence
        sloppy_signal = PatternSignal(
            name="pullback_new_high",
            bias="bullish",
            strength=0.6,
            price_level=11.40,
            timeframe="1min",
            candle_count=7,
            pullback_depth=0.45,
            pullback_candles=4,
            pullback_control=0.25,
            leg_candles=4,
            topping_tail=0.0,
            jackknife=0.0,
        )
        sloppy_scorer._fill_setup_group(sloppy_entry.setup_pattern, sloppy_signal)
        sloppy_entry.setup_pattern_name = sloppy_signal.name

        self.assertGreater(clean_entry.setup_pattern_score, sloppy_entry.setup_pattern_score)
        self.assertEqual(clean_entry.live_confidence, clean_live_confidence)
        self.assertEqual(sloppy_entry.live_confidence, sloppy_live_confidence)


# ---------------------------------------------------------------------------
# REST candle polling — resampling + df_to_candles
# ---------------------------------------------------------------------------

class TestResample(unittest.TestCase):

    def _make_1min_df(self, n: int = 10):
        import pandas as pd
        base = datetime(2026, 4, 6, 14, 0, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(n):
            rows.append({
                "symbol": "TEST",
                "ts": base + timedelta(minutes=i),
                "open": 5.0 + i * 0.01,
                "high": 5.0 + i * 0.01 + 0.05,
                "low": 5.0 + i * 0.01 - 0.02,
                "close": 5.0 + (i + 1) * 0.01,
                "volume": 1000,
                "interval": "1min",
            })
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def test_resample_2min_halves_bar_count(self):
        df = self._make_1min_df(10)
        sym = df["symbol"].iloc[0]
        candles = DynamicScorer._resample_indexed(df.set_index("ts"), sym, "2min")
        self.assertEqual(len(candles), 5)
        self.assertEqual(candles[0].interval, "2min")

    def test_resample_5min_fifths_bar_count(self):
        df = self._make_1min_df(10)
        sym = df["symbol"].iloc[0]
        candles = DynamicScorer._resample_indexed(df.set_index("ts"), sym, "5min")
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].interval, "5min")

    def test_resample_volume_sums(self):
        df = self._make_1min_df(10)
        sym = df["symbol"].iloc[0]
        candles = DynamicScorer._resample_indexed(df.set_index("ts"), sym, "5min")
        self.assertEqual(candles[0].volume, 5000)

    def test_resample_empty_df(self):
        import pandas as pd
        candles = DynamicScorer._resample_indexed(pd.DataFrame(), "TEST", "2min")
        self.assertEqual(candles, [])

    def test_df_to_candles_correct_count(self):
        df = self._make_1min_df(5)
        candles = DynamicScorer._df_to_candles(df, "1min")
        self.assertEqual(len(candles), 5)
        self.assertEqual(candles[0].interval, "1min")


# ---------------------------------------------------------------------------
# RVol computation
# ---------------------------------------------------------------------------

class TestComputeRvol(unittest.TestCase):

    def test_rvol_correct_ratio(self):
        scorer = _make_scorer()
        scorer._avg_vol = 78_000
        now = datetime.now(tz=timezone.utc)
        scorer._vol_prints = deque([(now, 2_000)])
        scorer._vol_prints_sum = 2_000
        result = scorer._compute_rvol_5min()
        self.assertAlmostEqual(result, 2.0)

    def test_rvol_zero_when_no_prints(self):
        scorer = _make_scorer()
        scorer._avg_vol = 100_000
        scorer._vol_prints = deque()
        scorer._vol_prints_sum = 0
        self.assertEqual(scorer._compute_rvol_5min(), 0.0)

    def test_rvol_zero_when_avg_vol_zero(self):
        scorer = _make_scorer()
        scorer._avg_vol = 0
        self.assertEqual(scorer._compute_rvol_5min(), 0.0)


# ---------------------------------------------------------------------------
# Session peak tracking (_raise_hod)
# ---------------------------------------------------------------------------

class TestRaiseHod(unittest.TestCase):
    """New HOD raises daily_high AND the persisted session peak %change."""

    @staticmethod
    def _stub(daily_high: float, prev_close: float, peak: float = 0.0):
        from types import SimpleNamespace
        return SimpleNamespace(
            _entry=SimpleNamespace(daily_high=daily_high, peak_change_pct=peak),
            _prev_close=prev_close,
        )

    def test_new_high_raises_hod_and_peak(self):
        from strategy.confidence_scorer import DynamicScorer
        stub = self._stub(daily_high=5.0, prev_close=2.0, peak=150.0)
        DynamicScorer._raise_hod(stub, 6.0)
        self.assertEqual(stub._entry.daily_high, 6.0)
        self.assertAlmostEqual(stub._entry.peak_change_pct, 200.0)  # (6-2)/2

    def test_lower_price_is_noop(self):
        from strategy.confidence_scorer import DynamicScorer
        stub = self._stub(daily_high=5.0, prev_close=2.0, peak=150.0)
        DynamicScorer._raise_hod(stub, 4.9)
        self.assertEqual(stub._entry.daily_high, 5.0)
        self.assertAlmostEqual(stub._entry.peak_change_pct, 150.0)

    def test_no_prev_close_still_raises_hod_only(self):
        from strategy.confidence_scorer import DynamicScorer
        stub = self._stub(daily_high=5.0, prev_close=0.0, peak=0.0)
        DynamicScorer._raise_hod(stub, 6.0)
        self.assertEqual(stub._entry.daily_high, 6.0)
        self.assertEqual(stub._entry.peak_change_pct, 0.0)


# ---------------------------------------------------------------------------
# Pressure acceleration readiness
# ---------------------------------------------------------------------------

class TestPressureAccelerationReadiness(unittest.TestCase):

    def _seed_pressure(
        self,
        tracker: OrderFlowTracker,
        *,
        older_buys: int = 0,
        older_sells: int = 0,
        recent_buys: int = 0,
        recent_sells: int = 0,
    ) -> None:
        now = datetime(2026, 4, 10, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(older_buys):
            tracker._buy_prints.append(now - timedelta(seconds=8.0 - i * 0.01))
        for i in range(older_sells):
            tracker._sell_prints.append(now - timedelta(seconds=8.0 - i * 0.01))
        # Recent burst sits right at the read edge (0.5s ago) — fresh enough
        # that the fast EMA kernel still carries it at full weight.
        for i in range(recent_buys):
            tracker._buy_prints.append(now - timedelta(seconds=0.5 - i * 0.01))
        for i in range(recent_sells):
            tracker._sell_prints.append(now - timedelta(seconds=0.5 - i * 0.01))
        tracker._baseline_prints.extend(
            sorted(list(tracker._buy_prints) + list(tracker._sell_prints))
        )
        tracker.reseed_ema_from_prints(now=now)

    def test_pressure_acceleration_not_ready_with_too_few_older_prints(self):
        tracker = OrderFlowTracker()
        self._seed_pressure(tracker, older_buys=4, recent_buys=10)

        self.assertFalse(tracker.pressure_acceleration_ready)
        self.assertEqual(tracker.pressure_acceleration, 0.0)

    def test_pressure_acceleration_ready_allows_neutral_zero(self):
        tracker = OrderFlowTracker()
        self._seed_pressure(
            tracker,
            older_buys=3,
            older_sells=2,
            recent_buys=3,
            recent_sells=2,
        )

        self.assertTrue(tracker.pressure_acceleration_ready)
        self.assertAlmostEqual(tracker.pressure_acceleration, 0.0)

    def test_tape_acceleration_gate_fails_until_ready(self):
        scorer = _make_scorer()
        scorer._entry.criteria_groups = build_criteria_groups(CRITERIA_EXECUTION)
        execg = scorer._entry.get_criteria_group("EXECUTION")
        criterion = execg.get("tape_acceleration")

        scorer._update_tick_criteria(5.00)
        self.assertFalse(criterion.passing)
        self.assertFalse(scorer.snapshot().pressure_acceleration_ready)

        # Significant acceleration: recent half ~2.3x the older half on enough
        # prints (accel=0.40, N=50 -> z=accel*sqrt(N)=2.83 >= SIM_PRESSURE_ACCEL_Z).
        self._seed_pressure(
            scorer._ofi_tracker,
            older_buys=15,
            recent_buys=35,
        )
        scorer._update_tick_criteria(5.00)

        self.assertTrue(scorer.snapshot().pressure_acceleration_ready)
        self.assertTrue(criterion.passing)


# ---------------------------------------------------------------------------
# Cancelled / correction prints ignored
# ---------------------------------------------------------------------------

class TestCancelledPrints(unittest.TestCase):

    def test_cancel_event_ignored(self):
        scorer = _make_scorer()
        event = _ts_event(5.0)
        event.cancel = True
        before = scorer._prints_seen
        scorer._on_timesale(event)
        self.assertEqual(scorer._prints_seen, before)

    def test_correction_event_ignored(self):
        scorer = _make_scorer()
        event = _ts_event(5.0)
        event.correction = True
        before = scorer._prints_seen
        scorer._on_timesale(event)
        self.assertEqual(scorer._prints_seen, before)


class TestLiveVolumeCriteria(unittest.TestCase):

    def test_timesales_update_attention_volume_min(self):
        entry = _make_entry()
        entry.volume = ATTENTION_VOLUME_MIN - 1_000
        entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        attn = entry.get_criteria_group("ATTENTION")
        volume_min = attn.get("volume_min")
        volume_min.update(False, float(entry.volume))

        scorer = _make_scorer(entry=entry)
        scorer._on_timesale(_ts_event(5.0, size=999))

        self.assertFalse(volume_min.passing)
        self.assertEqual(volume_min.value, float(ATTENTION_VOLUME_MIN - 1))

        scorer._on_timesale(_ts_event(5.0, size=1))

        self.assertTrue(volume_min.passing)
        self.assertEqual(volume_min.value, float(ATTENTION_VOLUME_MIN))

    def test_timesales_update_attention_change_and_rvol(self):
        entry = _make_entry()
        entry.price = 5.0
        entry.change_pct = 10.0
        entry.volume = 1_000
        entry.avg_volume = 10_000
        entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        attn = entry.get_criteria_group("ATTENTION")
        change_pct_min = attn.get("change_pct_min")
        rvol_1day_min = attn.get("rvol_1day_min")

        scorer = _make_scorer(entry=entry, avg_daily_volume=10_000)
        scorer._on_timesale(_ts_event(5.50, size=49_000))

        self.assertAlmostEqual(change_pct_min.value, 21.0, places=1)
        self.assertTrue(change_pct_min.passing)
        self.assertEqual(rvol_1day_min.value, 5.0)
        self.assertTrue(rvol_1day_min.passing)

    def test_timesales_do_not_downgrade_at_time_rvol_to_simple_rvol(self):
        entry = _make_entry()
        entry.volume = 2_730_000
        entry.avg_volume = 4_300_000
        entry.rvol_1day = 34.3
        entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        attn = entry.get_criteria_group("ATTENTION")
        rvol_1day_min = attn.get("rvol_1day_min")
        rvol_1day_min.update(True, entry.rvol_1day)

        scorer = _make_scorer(entry=entry, avg_daily_volume=4_300_000)
        scorer._on_timesale(_ts_event(0.58, size=1_000))

        self.assertAlmostEqual(entry.rvol_1day, 34.3)
        self.assertAlmostEqual(rvol_1day_min.value, 34.3)
        self.assertTrue(rvol_1day_min.passing)

    def test_poll_criteria_do_not_downgrade_at_time_rvol_to_simple_rvol(self):
        entry = _make_entry()
        entry.volume = 2_730_000
        entry.avg_volume = 4_300_000
        entry.rvol_1day = 34.3
        entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        scorer = _make_scorer(entry=entry, avg_daily_volume=4_300_000)

        base = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        df = pd.DataFrame([
            {
                "symbol": "TEST",
                "ts": base + timedelta(minutes=i),
                "open": 0.52 + i * 0.01,
                "high": 0.54 + i * 0.01,
                "low": 0.51 + i * 0.01,
                "close": 0.53 + i * 0.01,
                "volume": 10_000 + i * 1_000,
                "vwap": None,
            }
            for i in range(20)
        ])
        scorer._candles_session = scorer._df_to_candles(df, "1min")
        scorer._last_price = 0.58
        entry.daily_high = 0.59

        scorer._update_criteria_and_hidden_seller(df, include_hidden_seller=False)

        rvol_1day_min = entry.get_criteria_group("ATTENTION").get("rvol_1day_min")
        self.assertAlmostEqual(entry.rvol_1day, 34.3)
        self.assertAlmostEqual(rvol_1day_min.value, 34.3)
        self.assertTrue(rvol_1day_min.passing)


class TestFirstMacdCycleGate(unittest.TestCase):
    """entry.macd_cycle_done — first 1m MACD hist positive->negative flip
    since move start, closed bars only (SIM_FIRST_MACD_CYCLE_ONLY)."""

    _BASE = datetime(2026, 6, 10, 14, 0, tzinfo=timezone.utc)  # 10:00 ET

    def _hist_frame(self, hists, start_offset_min: int = 0):
        return pd.DataFrame([
            {
                "symbol": "TEST",
                "ts": self._BASE + timedelta(minutes=start_offset_min + i),
                "macd_hist": h,
            }
            for i, h in enumerate(hists)
        ])

    def _scorer(self):
        entry = _make_entry()
        scorer = _make_scorer(entry=entry)
        scorer._move_start_ts = self._BASE
        return scorer, entry

    def test_positive_only_not_done(self):
        scorer, entry = self._scorer()
        scorer._update_first_macd_cycle(self._hist_frame([0.05, 0.1, 0.2, 0.3, 0.4]))
        self.assertFalse(entry.macd_cycle_done)
        self.assertIsNone(entry.macd_cycle_done_ts)

    def test_flip_latches_and_repositive_stays_done(self):
        scorer, entry = self._scorer()
        # closed bars: -0.05, 0.1, 0.2, -0.1, 0.4, 0.5 (last row 0.6 forming)
        scorer._update_first_macd_cycle(
            self._hist_frame([-0.05, 0.1, 0.2, -0.1, 0.4, 0.5, 0.6])
        )
        self.assertTrue(entry.macd_cycle_done)
        self.assertEqual(entry.macd_cycle_done_ts, self._BASE + timedelta(minutes=3))

    def test_flip_only_on_forming_bar_is_ignored(self):
        scorer, entry = self._scorer()
        scorer._update_first_macd_cycle(self._hist_frame([0.1, 0.2, -0.1]))
        self.assertFalse(entry.macd_cycle_done)

    def test_never_positive_not_done(self):
        scorer, entry = self._scorer()
        scorer._update_first_macd_cycle(self._hist_frame([-0.1, -0.2, -0.3, 0.1]))
        self.assertFalse(entry.macd_cycle_done)

    def test_premove_cycle_ignored(self):
        scorer, entry = self._scorer()
        # Full pos->neg cycle BEFORE move start, only positive after it.
        scorer._update_first_macd_cycle(
            self._hist_frame([0.1, 0.2, -0.1, -0.2, 0.1, 0.2, 0.3, 0.4], start_offset_min=-5)
        )
        self.assertFalse(entry.macd_cycle_done)

    def test_late_detected_move_start_unlatches(self):
        scorer, entry = self._scorer()
        frame = self._hist_frame([0.1, 0.2, -0.1, -0.2, 0.1, 0.2, 0.3, 0.4], start_offset_min=-5)
        # Anchor initially covers the early cycle -> done.
        scorer._move_start_ts = self._BASE + timedelta(minutes=-5)
        scorer._update_first_macd_cycle(frame)
        self.assertTrue(entry.macd_cycle_done)
        # Move start detected later (real move began after the fade):
        # recompute self-corrects instead of staying stuck on the stale latch.
        scorer._move_start_ts = self._BASE
        scorer._update_first_macd_cycle(frame)
        self.assertFalse(entry.macd_cycle_done)

    def test_criteria_refresh_wires_gate_from_real_macd(self):
        entry = _make_entry()
        entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        scorer = _make_scorer(entry=entry)
        scorer._move_start_ts = self._BASE

        def _frame(closes):
            return pd.DataFrame([
                {
                    "symbol": "TEST",
                    "ts": self._BASE + timedelta(minutes=i),
                    "open": c,
                    "high": c + 0.02,
                    "low": c - 0.02,
                    "close": c,
                    "volume": 10_000,
                    "vwap": None,
                }
                for i, c in enumerate(closes)
            ])

        flat = [5.00] * 40
        rise = [5.00 + 0.10 * (i + 1) for i in range(12)]   # -> 6.20
        fall = [6.20 - 0.10 * (i + 1) for i in range(20)]   # -> 4.20

        # Mid-rise: hist positive, never flipped -> tradeable.
        scorer._update_criteria_and_hidden_seller(
            _frame(flat + rise + [6.20]), include_hidden_seller=False
        )
        self.assertFalse(entry.macd_cycle_done)

        # After the fade the 1m hist has flipped negative on closed bars ->
        # the move's first MACD cycle is over, all further entries blocked.
        scorer._update_criteria_and_hidden_seller(
            _frame(flat + rise + fall + [4.20]), include_hidden_seller=False
        )
        self.assertTrue(entry.macd_cycle_done)
        self.assertIsNotNone(entry.macd_cycle_done_ts)
        self.assertGreater(
            entry.macd_cycle_done_ts,
            self._BASE + timedelta(minutes=len(flat) + len(rise) - 1),
        )


class TestLiveRocSnapshot(unittest.TestCase):

    def test_period_three_tick_roc_updates_snapshot_without_attention_gate(self):
        self.assertEqual(ROC_PERIOD, 3)
        entry = _make_entry()
        entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        scorer = _make_scorer(entry=entry)
        self.assertIsNone(entry.get_criteria_group("ATTENTION").get("roc_positive"))

        base_ts = datetime.now(tz=timezone.utc)
        for idx, price in enumerate([5.00, 5.05, 5.10]):
            scorer._on_timesale(_ts_event(price, ts=base_ts + timedelta(milliseconds=idx)))
        self.assertEqual(scorer.snapshot().last_roc, 0.0)

        scorer._on_timesale(_ts_event(5.15, ts=base_ts + timedelta(milliseconds=3)))

        self.assertAlmostEqual(scorer.snapshot().last_roc, 3.0)

    def test_negative_tick_roc_does_not_block_attention(self):
        entry = _make_entry()
        entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        attn = entry.get_criteria_group("ATTENTION")
        for criterion in attn.criteria:
            criterion.update(True, 1.0)

        scorer = _make_scorer(entry=entry)
        base_ts = datetime.now(tz=timezone.utc)
        for idx, price in enumerate([5.15, 5.10, 5.05, 5.00]):
            scorer._on_timesale(_ts_event(price, ts=base_ts + timedelta(milliseconds=idx)))

        self.assertLess(scorer.snapshot().last_roc, 0.0)
        self.assertIsNone(attn.get("roc_positive"))
        self.assertTrue(attn.required_passing)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot(unittest.TestCase):

    def test_snapshot_returns_correct_symbol(self):
        scorer = _make_scorer()
        snap = scorer.snapshot()
        self.assertEqual(snap.symbol, "TEST")

    def test_snapshot_confidence_matches_entry(self):
        scorer = _make_scorer()
        snap = scorer.snapshot()
        self.assertAlmostEqual(snap.eval_confidence, scorer._entry.eval_confidence)
        self.assertAlmostEqual(snap.live_confidence, scorer._entry.live_confidence)

    def test_snapshot_spread_computed(self):
        scorer = _make_scorer()
        scorer._bid = 4.98
        scorer._ask = 5.02
        snap = scorer.snapshot()
        self.assertAlmostEqual(snap.spread, 0.04)

    def test_snapshot_ofi_fields_present(self):
        scorer = _make_scorer()
        snap = scorer.snapshot()
        self.assertAlmostEqual(snap.book_lean, 0.0)
        self.assertAlmostEqual(snap.buying_pressure, 0)
        self.assertAlmostEqual(snap.selling_pressure, 0)
        self.assertEqual(snap.ofi_rolling, 0)

    def test_snapshot_has_group_scores(self):
        scorer = _make_scorer()
        snap = scorer.snapshot()
        self.assertIn("STOCK_EVAL", snap.group_scores)
        self.assertIn("TECHNICAL_ANALYSIS", snap.group_scores)




class TestSetupPatternConfidence(unittest.TestCase):

    def test_pullback_signal_updates_separate_setup_score(self):
        scorer = _make_scorer()
        snap = PatternSnapshot(signals=[
            PatternSignal(
                name="pullback_new_high",
                bias="bullish",
                strength=0.8,
                price_level=5.10,
                timeframe="1min",
                candle_count=6,
                pullback_depth=0.10,
                pullback_candles=1,
                leg_candles=4,
                pullback_structure=1.0,
                breakout_extension=0.20,
            )
        ])

        scorer._update_setup_pattern_confidence(snap)

        self.assertGreater(scorer._entry.setup_pattern_score, 80.0)
        self.assertEqual(scorer._entry.setup_pattern_name, "pullback_new_high")
        self.assertEqual(scorer._entry.setup_pattern_timeframe, "1min")
        self.assertAlmostEqual(scorer._entry.setup_pullback_depth, 0.10)

    def test_pending_pullback_abandoned_without_breaking_is_not_a_setup(self):
        """A pullback that never breaks the entry/new high is not a setup: it
        clears silently rather than being recorded/badged as a failed setup."""
        scorer = _make_scorer()
        scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
            PatternSignal(
                name="pullback_pending",
                bias="bullish",
                strength=0.8,
                price_level=5.10,
                timeframe="1min",
                candle_count=6,
                pullback_depth=0.10,
                pullback_candles=1,
                leg_candles=4,
                pullback_structure=1.0,
            )
        ]))
        self.assertEqual(scorer._entry.setup_pattern_name, "pullback_pending")
        # Forming pullback → shown as a preview but carries no score yet.
        self.assertEqual(scorer._entry.setup_pattern_score, 0.0)

        # Abandoned before breaking → cleared silently, NOT a "failed" setup.
        scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[]))
        self.assertEqual(scorer._entry.setup_pattern_name, "")
        self.assertIsNone(scorer._entry.setup_failed_at)
        self.assertEqual(scorer._entry.setup_pattern_score, 0.0)

    def test_forming_pullback_has_no_score_until_recorded(self):
        """Forming pullbacks (detected/pending) carry no setup score until the
        pullback breaks the entry / new high and becomes a recorded setup."""
        scorer = _make_scorer()

        # Pending (forming) → preview only, no score, group vars empty.
        scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
            PatternSignal(
                name="pullback_pending",
                bias="bullish",
                strength=0.9,
                price_level=5.10,
                timeframe="1min",
                candle_count=6,
                pullback_depth=0.10,
                pullback_candles=1,
                leg_candles=4,
                pullback_structure=1.0,
            )
        ]))
        self.assertEqual(scorer._entry.setup_pattern_name, "pullback_pending")
        self.assertEqual(scorer._entry.setup_pattern_score, 0.0)
        self.assertTrue(all(v.value is None for v in scorer._entry.setup_pattern.variables))

        # Breakout fires (recorded setup) → now it carries a score.
        scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
            PatternSignal(
                name="pullback_new_high",
                bias="bullish",
                strength=0.9,
                price_level=5.10,
                timeframe="1min",
                candle_count=6,
                pullback_depth=0.10,
                pullback_candles=1,
                leg_candles=4,
                pullback_structure=1.0,
                breakout_extension=0.20,
            )
        ]))
        self.assertEqual(scorer._entry.setup_pattern_name, "pullback_new_high")
        self.assertGreater(scorer._entry.setup_pattern_score, 0.0)

    def test_resolved_pullback_logs_approved_setup_even_when_entry_blocked(self):
        entry = _make_entry()
        scorer = _make_scorer(entry=entry)
        scorer._last_price = 5.12
        scorer._last_event_ts = datetime(2026, 6, 4, 14, 30, tzinfo=timezone.utc)
        self.assertFalse(entry.entry_allowed)

        with tempfile.TemporaryDirectory() as tmp:
            scorer._setup_logger = SetupSignalLogger(log_dir=tmp)
            scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
                PatternSignal(
                    name="pullback_new_high",
                    bias="bullish",
                    strength=0.8,
                    price_level=5.10,
                    timeframe="1min",
                    candle_count=6,
                    pullback_depth=0.10,
                    pullback_candles=1,
                    leg_candles=4,
                    pullback_structure=1.0,
                    breakout_extension=0.20,
                )
            ]))

            signals = list(scorer._setup_logger._signals.values())
            self.assertEqual(len(signals), 1)
            self.assertTrue(signals[0].signal_approved)
            self.assertEqual(signals[0].status, "APPROVED")
            self.assertFalse(signals[0].entered)

    def test_resolved_pullback_near_hod_uses_breakout_price_not_trigger_level(self):
        entry = _make_entry()
        entry.daily_high = 3.95
        scorer = _make_scorer(entry=entry, price=3.90)
        scorer._last_price = 3.90
        scorer._last_atr = 0.2893
        scorer._last_event_ts = datetime(2026, 6, 10, 11, 45, 40, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp:
            scorer._setup_logger = SetupSignalLogger(log_dir=tmp)
            scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
                PatternSignal(
                    name="pullback_new_high",
                    bias="bullish",
                    strength=0.77,
                    price_level=3.17,
                    timeframe="10sec",
                    candle_count=7,
                    pullback_depth=0.4479,
                    pullback_candles=1,
                    leg_candles=5,
                    pullback_structure=1.0,
                    breakout_extension=0.80,
                )
            ]))

            signals = list(scorer._setup_logger._signals.values())

        self.assertEqual(len(signals), 1)
        self.assertTrue(signals[0].signal_approved)
        self.assertEqual(signals[0].setup_timeframe, "10sec")
        self.assertEqual(scorer._entry.setup_pattern_name, "pullback_new_high")

    def test_resolved_pullback_records_setup_time_failure_detail(self):
        entry = _make_entry()
        entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        for group in entry.criteria_groups:
            for criterion in group.criteria:
                criterion.update(True, 1.0)
        entry.get_criteria_group("ATTENTION").update("dist_from_hod_atr_max", False, 3.25)
        entry.get_criteria_group("EXECUTION").update("spread_ok", False, 0.041)
        entry.get_criteria_group("EXECUTION").update("dist_to_resistance_ok", False, 0.05)

        scorer = _make_scorer(entry=entry)
        scorer._last_price = 5.12
        scorer._last_atr = 0.20
        scorer._last_event_ts = datetime(2026, 6, 4, 14, 30, tzinfo=timezone.utc)

        with tempfile.TemporaryDirectory() as tmp:
            scorer._setup_logger = SetupSignalLogger(log_dir=tmp)
            scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
                PatternSignal(
                    name="pullback_new_high",
                    bias="bullish",
                    strength=0.8,
                    price_level=5.10,
                    timeframe="1min",
                    candle_count=6,
                    pullback_depth=0.10,
                    pullback_candles=1,
                    leg_candles=4,
                    pullback_structure=1.0,
                    breakout_extension=0.20,
                )
            ]))

            signals = list(scorer._setup_logger._signals.values())

        self.assertEqual(len(signals), 1)
        detail = signals[0].setup_failure_detail
        self.assertIn("dist_hod_atr=3.25 > 2.00", detail)
        self.assertIn("spread=4.10% >", detail)
        self.assertIn("res_headroom=0.05 < 0.20 (1.0 ATR)", detail)
        self.assertIn("tape_activity=0 <", detail)

    def test_non_pullback_snapshot_clears_setup_score(self):
        scorer = _make_scorer()
        setup_snap = PatternSnapshot(signals=[
            PatternSignal(
                name="pullback_new_high",
                bias="bullish",
                strength=0.8,
                price_level=5.10,
                timeframe="1min",
                candle_count=6,
                pullback_depth=0.12,
            )
        ])
        scorer._update_setup_pattern_confidence(setup_snap)
        self.assertGreater(scorer._entry.setup_pattern_score, 0.0)

        # A subsequent empty snapshot following a resolved breakout
        # keeps the score around for the "Resolved" badge window — the
        # entry retains setup_resolved_at, so the score property still
        # reports the final group value.  Pattern name and timeframe
        # are not cleared because the resolved-event name is kept so
        # the display can render the badge.
        scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[]))
        self.assertGreater(scorer._entry.setup_pattern_score, 0.0)
        self.assertIsNotNone(scorer._entry.setup_resolved_at)
        self.assertEqual(scorer._entry.setup_pattern_name, "pullback_new_high")
        self.assertEqual(scorer._entry.setup_pattern_timeframe, "1min")

        scorer._entry.setup_resolved_at -= _SETUP_RESOLVED_DISPLAY_SECS + 1
        scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[]))
        self.assertEqual(scorer._entry.setup_pattern_score, 0.0)
        self.assertEqual(scorer._entry.setup_pattern_name, "")
        self.assertEqual(scorer._entry.setup_pattern_timeframe, "")
        self.assertIsNone(scorer._entry.setup_resolved_at)

    def test_setup_rejection_metrics_lower_setup_score(self):
        clean_scorer = _make_scorer()
        rejected_scorer = _make_scorer()

        base_kwargs = dict(
            name="pullback_new_high",
            bias="bullish",
            strength=0.85,
            price_level=5.10,
            timeframe="1min",
            candle_count=6,
            pullback_depth=0.10,
            pullback_candles=1,
            leg_candles=4,
            pullback_structure=1.0,
            breakout_extension=0.20,
        )

        clean_scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
            PatternSignal(**base_kwargs)
        ]))
        rejected_scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
            PatternSignal(**base_kwargs, topping_tail=0.9, jackknife=0.8)
        ]))

        self.assertLess(
            rejected_scorer._entry.setup_pattern_score,
            clean_scorer._entry.setup_pattern_score,
        )

    def test_cleaner_setup_scores_higher_than_sloppy(self):
        scorer = _make_scorer()

        sloppy = PatternSnapshot(signals=[
            PatternSignal(
                name="pullback_new_high",
                bias="bullish",
                strength=0.55,
                price_level=5.05,
                timeframe="1min",
                candle_count=9,
                pullback_depth=0.48,
                pullback_candles=5,
                leg_candles=2,
                pullback_structure=0.2,
                breakout_extension=0.01,
            )
        ])
        scorer._update_setup_pattern_confidence(sloppy)
        sloppy_score = scorer._entry.setup_pattern_score

        clean = PatternSnapshot(signals=[
            PatternSignal(
                name="pullback_new_high",
                bias="bullish",
                strength=0.90,
                price_level=5.10,
                timeframe="1min",
                candle_count=6,
                pullback_depth=0.08,
                pullback_candles=1,
                leg_candles=4,
                pullback_structure=1.0,
                breakout_extension=0.18,
            )
        ])
        scorer._update_setup_pattern_confidence(clean)
        clean_score = scorer._entry.setup_pattern_score

        self.assertGreater(clean_score, sloppy_score)


# ---------------------------------------------------------------------------
# Breakout-candle rejection
# ---------------------------------------------------------------------------

class TestBreakoutCandleRejection(unittest.TestCase):
    """Red breakout-candle close invalidates the pullback-high setup and
    blocks a re-break at the same price point."""

    def _append_closed_bar(self, scorer, tf, ts, open_, close_):
        bar = OHLCVCandle(
            "TEST", ts, open_, max(open_, close_) + 0.05,
            min(open_, close_) - 0.05, close_, 1000, tf,
        )
        scorer._tf_buf._bars[tf].append(bar)
        return bar

    def test_red_breakout_candle_invalidates_and_remembers_level(self):
        scorer = _make_scorer()
        ts = datetime(2026, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        scorer._entry.setup_pattern_name = "pullback_new_high"
        scorer._entry.setup_price_level = 5.10
        scorer._entry.setup_pattern_timeframe = "10sec"
        # Breakout candle closes red (close < open)
        self._append_closed_bar(scorer, "10sec", ts, open_=5.11, close_=5.08)
        scorer._breakout_candle_tf = "10sec"
        scorer._breakout_candle_ts = ts
        scorer._breakout_trigger_level = 5.10

        scorer._check_breakout_candle_rejection(["10sec"], ts + timedelta(seconds=10))

        self.assertEqual(scorer._entry.setup_pattern_name, "")
        self.assertIsNone(scorer._entry.setup_price_level)
        self.assertIn(5.10, list(scorer._rejected_breakout_levels))
        self.assertIsNone(scorer._breakout_candle_tf)

    def test_green_breakout_candle_keeps_setup_valid(self):
        scorer = _make_scorer()
        ts = datetime(2026, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        scorer._entry.setup_pattern_name = "pullback_new_high"
        scorer._entry.setup_price_level = 5.10
        scorer._entry.setup_pattern_timeframe = "10sec"
        # Breakout candle closes green (close > open)
        self._append_closed_bar(scorer, "10sec", ts, open_=5.10, close_=5.18)
        scorer._breakout_candle_tf = "10sec"
        scorer._breakout_candle_ts = ts
        scorer._breakout_trigger_level = 5.10

        scorer._check_breakout_candle_rejection(["10sec"], ts + timedelta(seconds=10))

        self.assertEqual(scorer._entry.setup_pattern_name, "pullback_new_high")
        self.assertEqual(len(scorer._rejected_breakout_levels), 0)
        self.assertIsNone(scorer._breakout_candle_tf)

    def test_rejected_level_blocks_intracandle_re_break(self):
        scorer = _make_scorer()
        ts = datetime(2026, 6, 10, 14, 31, 0, tzinfo=timezone.utc)
        scorer._rejected_breakout_levels.append(5.10)
        scorer._entry.setup_pattern_name = "pullback_pending"
        scorer._entry.setup_price_level = 5.10
        scorer._entry.setup_pattern_timeframe = "10sec"

        # Price comes back up and breaks the rejected level — must not fire.
        scorer._check_intracandle_breakout(5.15, ts)

        self.assertEqual(scorer._entry.setup_pattern_name, "pullback_pending")
        self.assertIsNone(scorer._entry.setup_resolved_at)

    def test_rejected_level_blocks_candle_close_resolved_path(self):
        scorer = _make_scorer()
        scorer._last_price = 5.15
        scorer._rejected_breakout_levels.append(5.10)

        scorer._update_setup_pattern_confidence(PatternSnapshot(signals=[
            PatternSignal(
                name="pullback_new_high",
                bias="bullish",
                strength=0.85,
                price_level=5.10,
                timeframe="1min",
                candle_count=6,
                pullback_depth=0.10,
            )
        ]))

        self.assertEqual(scorer._entry.setup_pattern_name, "")


# ---------------------------------------------------------------------------
# _MultiTFCandleBuffer
# ---------------------------------------------------------------------------

class TestMultiTFCandleBuffer(unittest.TestCase):

    def _make_buf(self):
        from strategy.confidence_scorer import _MultiTFCandleBuffer
        return _MultiTFCandleBuffer("TEST", max_bars=20)

    def test_first_tick_opens_bars_on_all_timeframes(self):
        buf = self._make_buf()
        ts = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        closed = buf.on_tick(5.00, 100, ts)
        self.assertEqual(closed, [])
        for tf in ["10sec", "1min", "2min", "5min"]:
            self.assertIsNotNone(buf._current[tf])
            self.assertEqual(buf._current[tf].open, 5.00)

    def test_10sec_candle_closes_on_boundary(self):
        buf = self._make_buf()
        ts1 = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        buf.on_tick(5.00, 100, ts1)
        ts2 = datetime(2026, 4, 10, 10, 0, 17, tzinfo=timezone.utc)
        closed = buf.on_tick(5.10, 200, ts2)
        self.assertIn("10sec", closed)
        self.assertNotIn("1min", closed)

    def test_1min_close_also_closes_10sec(self):
        buf = self._make_buf()
        ts1 = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        buf.on_tick(5.00, 100, ts1)
        ts2 = datetime(2026, 4, 10, 10, 1, 3, tzinfo=timezone.utc)
        closed = buf.on_tick(5.20, 150, ts2)
        self.assertIn("10sec", closed)
        self.assertIn("1min", closed)

    def test_ohlcv_aggregation(self):
        buf = self._make_buf()
        ts = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        buf.on_tick(5.00, 100, ts)
        buf.on_tick(5.20, 200, ts.replace(second=6))
        buf.on_tick(4.80, 50, ts.replace(second=7))
        buf.on_tick(5.10, 300, ts.replace(second=8))
        cur = buf._current["10sec"]
        self.assertEqual(cur.open, 5.00)
        self.assertEqual(cur.high, 5.20)
        self.assertEqual(cur.low, 4.80)
        self.assertEqual(cur.close, 5.10)
        self.assertEqual(cur.volume, 650)

    def test_to_df_includes_current_bar(self):
        buf = self._make_buf()
        ts = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        buf.on_tick(5.00, 100, ts)
        df = buf.to_df("1min")
        self.assertEqual(len(df), 1)
        self.assertEqual(float(df.iloc[0]["open"]), 5.00)
        self.assertEqual(df.index[0], datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc))

    def test_to_df_empty_when_no_ticks(self):
        buf = self._make_buf()
        df = buf.to_df("5min")
        self.assertTrue(df.empty)

    def test_seed_adds_historical_bars(self):
        from market_data.tradier_rest import OHLCVCandle
        buf = self._make_buf()
        candles = [
            OHLCVCandle("TEST", datetime(2026, 4, 10, 9, 55, tzinfo=timezone.utc),
                        4.90, 5.00, 4.85, 4.95, 1000, "1min"),
            OHLCVCandle("TEST", datetime(2026, 4, 10, 9, 56, tzinfo=timezone.utc),
                        4.95, 5.10, 4.90, 5.05, 1500, "1min"),
        ]
        buf.seed("1min", candles)
        self.assertEqual(len(buf._bars["1min"]), 2)
        df = buf.to_df("1min")
        self.assertEqual(len(df), 2)
        self.assertEqual(df.index[0], candles[0].ts)

    def test_seed_grows_capacity_to_preserve_move_window(self):
        from market_data.tradier_rest import OHLCVCandle
        from strategy.confidence_scorer import _MultiTFCandleBuffer
        buf = _MultiTFCandleBuffer("TEST", max_bars=3)
        base = datetime(2026, 4, 10, 9, 55, tzinfo=timezone.utc)
        candles = [
            OHLCVCandle(
                "TEST", base + timedelta(seconds=i * 10),
                4.90, 5.00, 4.85, 4.95, 1000, "10sec",
            )
            for i in range(8)
        ]

        buf.seed("10sec", candles)

        self.assertEqual(buf.bar_counts["10sec"], 8)
        self.assertEqual(len(buf.to_df("10sec")), 8)

    def test_to_df_rebuilds_when_full_buffer_rotates(self):
        from market_data.tradier_rest import OHLCVCandle
        from strategy.confidence_scorer import _MultiTFCandleBuffer
        buf = _MultiTFCandleBuffer("TEST", max_bars=3)
        base = datetime(2026, 4, 10, 9, 55, tzinfo=timezone.utc)
        candles = [
            OHLCVCandle(
                "TEST", base + timedelta(seconds=i * 10),
                4.90 + i * 0.10, 5.00 + i * 0.10, 4.85 + i * 0.10,
                4.95 + i * 0.10, 1000, "10sec",
            )
            for i in range(3)
        ]
        buf.seed("10sec", candles)
        first_df = buf.to_df("10sec")
        self.assertEqual(first_df.index[-1], candles[-1].ts)

        buf.on_tick(5.30, 100, base + timedelta(seconds=30))
        buf.on_tick(5.40, 100, base + timedelta(seconds=40))
        rotated_df = buf.to_df("10sec")

        self.assertEqual(len(rotated_df), 4)
        self.assertEqual(rotated_df.index[-2], base + timedelta(seconds=30))
        self.assertEqual(float(rotated_df.iloc[-2]["close"]), 5.30)

    def test_5min_close(self):
        buf = self._make_buf()
        ts1 = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        buf.on_tick(5.00, 100, ts1)
        ts2 = datetime(2026, 4, 10, 10, 5, 1, tzinfo=timezone.utc)
        closed = buf.on_tick(5.50, 200, ts2)
        self.assertIn("5min", closed)
        self.assertIn("2min", closed)
        self.assertIn("1min", closed)
        self.assertIn("10sec", closed)

    def test_bar_counts(self):
        buf = self._make_buf()
        ts1 = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        buf.on_tick(5.00, 100, ts1)
        ts2 = datetime(2026, 4, 10, 10, 0, 15, tzinfo=timezone.utc)
        buf.on_tick(5.10, 200, ts2)
        counts = buf.bar_counts
        self.assertEqual(counts["10sec"], 1)
        self.assertEqual(counts["1min"], 0)


class TestStreamPatternTrigger(unittest.TestCase):
    """Verify that _on_timesale triggers pattern detection on candle close."""

    def test_timesale_feeds_buffer_and_triggers_patterns(self):
        scorer = _make_scorer()
        ts1 = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        ts2 = datetime(2026, 4, 10, 10, 0, 17, tzinfo=timezone.utc)
        scorer._on_timesale(_ts_event(5.00, 100, ts=ts1))
        with patch.object(scorer, "_scan_patterns_on_close") as mock_scan:
            scorer._on_timesale(_ts_event(5.05, 200, ts=ts2))
            mock_scan.assert_called_once()
            args = mock_scan.call_args[0][0]
            self.assertIn("10sec", args)


class TestChartCacheIntegration(unittest.TestCase):
    def test_load_timesales_cache_filters_window_and_normalizes(self):
        base = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            path = timesales_cache_path(log_dir, "TEST", "20260610")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "symbol", "ts", "price", "size", "bid", "ask",
                    "exchange", "seq", "flag", "session", "source",
                ])
                writer.writeheader()
                writer.writerow({
                    "symbol": "TEST", "ts": (base - timedelta(seconds=1)).isoformat(),
                    "price": "4.99", "size": "100", "bid": "4.98", "ask": "5.00",
                    "exchange": "Q", "seq": "1", "flag": "@", "session": "normal", "source": "stream",
                })
                writer.writerow({
                    "symbol": "TEST", "ts": (base + timedelta(seconds=1)).isoformat(),
                    "price": "5.01", "size": "200", "bid": "5.00", "ask": "5.01",
                    "exchange": "Q", "seq": "2", "flag": "@", "session": "normal", "source": "stream",
                })
                writer.writerow({
                    "symbol": "OTHER", "ts": (base + timedelta(seconds=2)).isoformat(),
                    "price": "6.00", "size": "300", "bid": "5.99", "ask": "6.01",
                    "exchange": "Q", "seq": "3", "flag": "@", "session": "normal", "source": "stream",
                })

            cached = load_timesales_cache(
                "TEST",
                base,
                base + timedelta(seconds=5),
                log_dir,
            )

            self.assertEqual(len(cached), 1)
            self.assertEqual(cached.iloc[0]["symbol"], "TEST")
            self.assertEqual(float(cached.iloc[0]["price"]), 5.01)
            self.assertEqual(int(cached.iloc[0]["size"]), 200)

    def test_timesales_cache_records_low_notional_stream_prints(self):
        scorer = _make_scorer()
        ts = datetime(2026, 6, 10, 10, 0, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            scorer._chart_cache = LiveChartCache("TEST", log_dir=log_dir, raw_flush_size=1000, raw_flush_seconds=999)

            scorer._on_timesale(_ts_event(1.00, 1, ts=ts, bid=0.99, ask=1.01))
            scorer._flush_chart_cache()

            path = timesales_cache_path(log_dir, "TEST", "20260610")
            self.assertTrue(path.exists())
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["symbol"], "TEST")
            self.assertEqual(rows[0]["ts"], ts.isoformat())
            self.assertEqual(float(rows[0]["price"]), 1.00)
            self.assertEqual(int(rows[0]["size"]), 1)
            self.assertEqual(float(rows[0]["bid"]), 0.99)
            self.assertEqual(float(rows[0]["ask"]), 1.01)
            self.assertEqual(rows[0]["exchange"], "Q")
            self.assertEqual(rows[0]["flag"], "@")
            self.assertEqual(rows[0]["session"], "normal")

    def test_stream_10s_cache_keeps_tick_weighted_vwap(self):
        scorer = _make_scorer()
        base = datetime(2026, 6, 10, 10, 0, 1, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            scorer._chart_cache = LiveChartCache("TEST", log_dir=log_dir, raw_flush_size=1000, raw_flush_seconds=999)

            scorer._on_timesale(_ts_event(5.00, 1000, ts=base))
            scorer._on_timesale(_ts_event(6.00, 3000, ts=base + timedelta(seconds=4)))
            scorer._on_timesale(_ts_event(7.00, 1000, ts=base + timedelta(seconds=12)))

            cached = load_candle_cache("TEST", "10s", base.replace(second=0), base + timedelta(seconds=20), log_dir)
            self.assertEqual(len(cached), 1)
            self.assertEqual(cached.iloc[0]["source"], SOURCE_STREAM)
            self.assertEqual(int(cached.iloc[0]["volume"]), 4000)
            self.assertAlmostEqual(float(cached.iloc[0]["bar_vwap"]), 5.75, places=6)
            self.assertAlmostEqual(float(cached.iloc[0]["vwap"]), 5.75, places=6)
            self.assertIn("ema_9", cached.columns)
            self.assertIn("ma_50", cached.columns)

    def test_refresh_candles_caches_polled_1m_only(self):
        scorer = _make_scorer()
        base = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        rows = [
            {
                "symbol": "TEST",
                "ts": base + timedelta(minutes=i),
                "open": 5.00 + i * 0.05,
                "high": 5.08 + i * 0.05,
                "low": 4.98 + i * 0.05,
                "close": 5.04 + i * 0.05,
                "volume": 1000 + i * 100,
            }
            for i in range(5)
        ]
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            scorer._chart_cache = LiveChartCache("TEST", log_dir=log_dir)
            scorer._df_1min_session = df.copy()
            scorer._tf_seeded = True
            scorer._fetch_1min_candles = MagicMock(return_value=df)
            scorer._is_near_hod = MagicMock(return_value=False)

            scorer._refresh_candles()

            cached_1m = load_candle_cache("TEST", "1m", base, base + timedelta(minutes=5), log_dir)
            cached_2m = load_candle_cache("TEST", "2m", base, base + timedelta(minutes=5), log_dir)
            cached_5m = load_candle_cache("TEST", "5m", base, base + timedelta(minutes=5), log_dir)
            self.assertEqual(len(cached_1m), 5)
            self.assertEqual(set(cached_1m["source"]), {SOURCE_POLLED})
            self.assertTrue(cached_2m.empty)
            self.assertTrue(cached_5m.empty)
            self.assertIn("macd_line", cached_1m.columns)
            self.assertIn("atr_14", cached_1m.columns)
            self.assertAlmostEqual(float(cached_1m.iloc[0]["atr_14"]), 0.0, places=6)
            self.assertAlmostEqual(float(cached_1m.iloc[-1]["atr_14"]), 0.10, places=6)

    def test_refresh_candles_with_volume_does_not_touch_removed_live_vars(self):
        """Regression: with live volume present (rvol > 0), _refresh_candles must
        not assign removed LiveInputs fields.  rvol_5min was dropped from
        LiveInputs (slots dataclass), so a stale ``poll_inputs.rvol_5min = rvol``
        would raise — but only once real volume made the rvol branch fire."""
        scorer = _make_scorer()  # avg_vol seeded to 100_000 by _make_scorer
        base = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        rows = [
            {
                "symbol": "TEST",
                "ts": base + timedelta(minutes=i),
                "open": 5.00 + i * 0.05,
                "high": 5.08 + i * 0.05,
                "low": 4.98 + i * 0.05,
                "close": 5.04 + i * 0.05,
                "volume": 1000 + i * 100,
            }
            for i in range(5)
        ]
        df = pd.DataFrame(rows)

        # Live-like rolling volume so _compute_rvol_5min() > 0 — the exact
        # condition that used to hit the removed poll_inputs.rvol_5min line.
        scorer._vol_prints.append((base + timedelta(minutes=4), 5000))
        scorer._vol_prints_sum = 5000
        self.assertGreater(scorer._compute_rvol_5min(), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            scorer._chart_cache = LiveChartCache("TEST", log_dir=Path(tmp))
            scorer._df_1min_session = df.copy()
            scorer._tf_seeded = True
            scorer._fetch_1min_candles = MagicMock(return_value=df)
            scorer._is_near_hod = MagicMock(return_value=False)

            scorer._refresh_candles()  # must not raise

        # rvol_5min is no longer a live-confidence scoring variable.
        ta = scorer._entry.evaluation.group_by_name("TECHNICAL_ANALYSIS")
        self.assertIsNone(ta.get("rvol_5min"))

    def test_refresh_candles_initializes_live_confidence_without_timesales(self):
        scorer = _make_scorer()
        ta = scorer._entry.evaluation.technical_analysis
        for v in ta.variables:
            v.value = None
            v.quality = 0.0

        base = datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc)
        rows = []
        for i in range(20):
            open_ = 5.00 + i * 0.05
            close = open_ + (0.08 if i % 3 else -0.02)
            rows.append({
                "symbol": "TEST",
                "ts": base + timedelta(minutes=i),
                "open": open_,
                "high": max(open_, close) + 0.04,
                "low": min(open_, close) - 0.04,
                "close": close,
                "volume": 1000 + i * 100,
            })
        df = pd.DataFrame(rows)

        scorer._last_price = float(df.iloc[-1]["close"])
        scorer._entry.daily_high = float(df["high"].max())
        scorer._df_1min_session = df.copy()
        scorer._tf_seeded = True
        scorer._fetch_1min_candles = MagicMock(return_value=df)
        scorer._is_near_hod = MagicMock(return_value=False)

        self.assertEqual(scorer._prints_seen, 0)
        self.assertFalse(scorer._entry.has_live_data)
        scorer._refresh_candles()

        self.assertEqual(scorer._prints_seen, 0)
        self.assertTrue(scorer._entry.has_live_data)
        self.assertIsNotNone(ta.get("key_candles").value)
        self.assertIsNotNone(ta.get("candle_choppiness").value)


# ---------------------------------------------------------------------------
# Level-respect tracker
# ---------------------------------------------------------------------------


def _level_df(rows):
    """Build a small OHLCV DataFrame from (ts, o, h, l, c[, v]) tuples."""
    import pandas as pd
    idx = [r[0] for r in rows]
    data = {
        "open":   [r[1] for r in rows],
        "high":   [r[2] for r in rows],
        "low":    [r[3] for r in rows],
        "close":  [r[4] for r in rows],
        "volume": [r[5] if len(r) > 5 else 1_000 for r in rows],
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))








class TestHodRejection(unittest.TestCase):
    """DynamicScorer._update_key_candles — volume-gradient candle response
    scoring across each timeframe's meaningful high-volume candles."""

    # A confirmed 1-min pivot high (pivot_n=2) sits at index 3 in each frame.
    _TOPPING_TAIL = [
        (5.20, 5.24, 5.16, 5.22),
        (5.22, 5.30, 5.20, 5.28),
        (5.28, 5.40, 5.26, 5.38),
        (5.42, 5.50, 5.40, 5.43),   # pivot high @5.50 with a long upper wick
        (5.43, 5.46, 5.40, 5.44),
        (5.44, 5.45, 5.40, 5.42),
        (5.42, 5.44, 5.40, 5.43),
    ]
    _JACKKNIFE = [
        (5.20, 5.24, 5.16, 5.22),
        (5.22, 5.30, 5.20, 5.28),
        (5.28, 5.40, 5.26, 5.38),
        (5.50, 5.50, 5.38, 5.40),   # pivot high @5.50, big red body (jackknife)
        (5.43, 5.46, 5.40, 5.44),
        (5.44, 5.45, 5.40, 5.42),
        (5.42, 5.44, 5.40, 5.43),
    ]
    _CLEAN = [
        (5.20, 5.24, 5.16, 5.22),
        (5.22, 5.30, 5.20, 5.28),
        (5.28, 5.40, 5.26, 5.38),
        (5.40, 5.50, 5.39, 5.49),   # pivot high @5.50, strong green close, tiny wick
        (5.43, 5.46, 5.40, 5.44),
        (5.44, 5.45, 5.40, 5.42),
        (5.42, 5.44, 5.40, 5.43),
    ]
    _FAR_FROM_HOD = [
        (4.90, 4.94, 4.86, 4.92),
        (4.92, 5.00, 4.90, 4.98),
        (4.98, 5.10, 4.96, 5.08),
        (5.12, 5.20, 5.10, 5.13),   # pivot high @5.20 w/ long upper wick, far below HOD
        (5.13, 5.16, 5.10, 5.14),
        (5.14, 5.15, 5.10, 5.12),
        (5.12, 5.14, 5.10, 5.13),
    ]
    _MINOR_WICK = [
        (5.20, 5.24, 5.16, 5.22),
        (5.22, 5.30, 5.20, 5.28),
        (5.28, 5.40, 5.26, 5.38),
        (5.40, 5.44, 5.39, 5.41),   # visible but not significant enough to count as rejection
        (5.41, 5.46, 5.40, 5.44),
        (5.44, 5.45, 5.40, 5.42),
        (5.42, 5.44, 5.40, 5.43),
    ]
    _SMALL_HIGH_VOLUME_RED = [
        (5.20, 5.24, 5.16, 5.22),
        (5.22, 5.30, 5.20, 5.28),
        (5.28, 5.40, 5.26, 5.38),
        (5.405, 5.405, 5.375, 5.375), # red close, but candle range is too small vs ATR to matter
        (5.37, 5.46, 5.36, 5.44),
        (5.44, 5.45, 5.40, 5.42),
        (5.42, 5.44, 5.40, 5.43),
    ]
    _SMALL_NORMAL_VOLUME_RED = [
        (5.20, 5.24, 5.16, 5.22, 1_000),
        (5.22, 5.30, 5.20, 5.28, 1_050),
        (5.28, 5.40, 5.26, 5.38, 1_100),
        (5.405, 5.405, 5.375, 5.375, 1_200),
        (5.37, 5.46, 5.36, 5.44, 1_000),
        (5.44, 5.45, 5.40, 5.42, 1_050),
        (5.42, 5.44, 5.40, 5.43, 1_100),
    ]

    @staticmethod
    def _hod_var(scorer):
        g = scorer._entry.evaluation.group_by_name("TECHNICAL_ANALYSIS")
        return g.get("key_candles")

    @staticmethod
    def _frame(rows):
        base = datetime(2026, 4, 10, 10, 0, 0, tzinfo=timezone.utc)
        return {
            "1min": _level_df([
                (
                    base + timedelta(minutes=i),
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4] if len(row) > 4 else (10_000 if i == 3 else 1_000 + i),
                )
                for i, row in enumerate(rows)
            ])
        }

    def _scorer(self):
        scorer = _make_scorer()
        scorer._entry.daily_high = 5.50   # no longer gates response scoring
        return scorer

    def test_topping_tail_near_hod_penalizes(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._TOPPING_TAIL))
        var = self._hod_var(scorer)
        self.assertIsNotNone(var.value)      # volume response recorded
        self.assertLess(var.quality, 0.95)   # penalty vs clean acceptance
        self.assertAlmostEqual(
            scorer._tf_pa_var_qualities["1m"]["key_candles"],
            var.quality,
            places=6,
        )

    def test_jackknife_near_hod_penalizes(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._JACKKNIFE))
        var = self._hod_var(scorer)
        self.assertIsNotNone(var.value)
        self.assertLess(var.quality, 0.5)

    def test_clean_high_volume_hod_acceptance_scores_well(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._CLEAN))
        var = self._hod_var(scorer)
        self.assertIsNotNone(var.value)
        self.assertGreater(var.quality, 0.9)
        self.assertAlmostEqual(
            scorer._tf_pa_var_qualities["1m"]["key_candles"],
            var.quality,
            places=6,
        )

    def test_larger_topping_tail_penalizes_more_than_small_wick(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._CLEAN))
        small_wick_quality = self._hod_var(scorer).quality

        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._TOPPING_TAIL))
        large_tail_quality = self._hod_var(scorer).quality

        self.assertGreater(small_wick_quality, 0.9)
        self.assertLess(large_tail_quality, 0.95)
        self.assertGreater(small_wick_quality, large_tail_quality)

    def test_minor_upper_wick_is_not_significant_rejection(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._MINOR_WICK))
        var = self._hod_var(scorer)

        self.assertIsNotNone(var.value)
        self.assertGreater(var.quality, 0.95)

    def test_small_normal_volume_red_candle_is_not_significant_rejection(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._SMALL_NORMAL_VOLUME_RED))
        var = self._hod_var(scorer)

        self.assertIsNotNone(var.value)
        self.assertGreater(var.quality, 0.95)

    def test_high_volume_red_candle_adds_pressure_even_when_small(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._SMALL_HIGH_VOLUME_RED))
        var = self._hod_var(scorer)

        self.assertIsNotNone(var.value)
        self.assertLess(var.quality, 0.55)
        self.assertGreater(var.quality, 0.35)

    def test_lower_high_high_volume_rejection_is_scored(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._FAR_FROM_HOD))
        var = self._hod_var(scorer)
        self.assertIsNotNone(var.value)
        self.assertLess(var.quality, 0.95)

    def test_rejection_rerates_when_clean_hod_acceptance_appears(self):
        scorer = self._scorer()
        scorer._update_key_candles(self._frame(self._TOPPING_TAIL))
        self.assertIsNotNone(self._hod_var(scorer).value)
        self.assertIn("key_candles", scorer._tf_pa_var_qualities["1m"])
        self.assertLess(self._hod_var(scorer).quality, 0.95)

        # A subsequent clean, high-volume close rates as acceptance.
        scorer._update_key_candles(self._frame(self._CLEAN))
        self.assertIsNotNone(self._hod_var(scorer).value)
        self.assertGreater(self._hod_var(scorer).quality, 0.9)
        self.assertIn("key_candles", scorer._tf_pa_var_qualities["1m"])

    def test_volume_gradient_counts_meaningful_rejection_below_top_bucket(self):
        scorer = self._scorer()
        rows = [
            (5.00, 5.10, 4.98, 5.08, 100),
            (5.08, 5.18, 5.06, 5.16, 200),
            (5.16, 5.26, 5.14, 5.24, 300),
            (5.24, 5.34, 5.22, 5.32, 400),
            (5.32, 5.42, 5.30, 5.40, 500),
            (5.40, 5.95, 5.30, 5.34, 8_000),  # ugly; below the old top-20% hard bucket
            (5.34, 5.44, 5.32, 5.42, 700),
            (5.42, 5.52, 5.40, 5.50, 900),
            (5.50, 5.62, 5.48, 5.60, 20_000),
            (5.60, 5.72, 5.58, 5.70, 18_000),
        ]

        scorer._update_key_candles(self._frame(rows))
        var = self._hod_var(scorer)

        self.assertIsNotNone(var.value)
        self.assertGreater(var.quality, 0.7)
        self.assertLess(var.quality, 0.9)


class TestCandleChoppinessRatio(unittest.TestCase):
    """Pure _candle_choppiness_ratio helper (succeeded pullbacks excused)."""

    def test_clean_staircase_is_zero(self):
        # G G G R G G — the red pullback is recovered by a new closing high.
        opens = [5.00, 5.08, 5.18, 5.28, 5.22, 5.38]
        highs = [5.10, 5.20, 5.30, 5.30, 5.40, 5.50]
        closes = [5.08, 5.18, 5.28, 5.22, 5.38, 5.48]
        self.assertEqual(_candle_choppiness_ratio(opens, highs, closes), 0.0)

    def test_alternating_but_recovered_is_zero(self):
        # G R G R G — looks alternating, but every pullback makes a new high.
        opens = [5.00, 5.08, 5.05, 5.22, 5.18]
        highs = [5.10, 5.12, 5.25, 5.26, 5.40]
        closes = [5.08, 5.05, 5.22, 5.18, 5.38]
        self.assertEqual(_candle_choppiness_ratio(opens, highs, closes), 0.0)

    def test_pure_chop_is_high(self):
        # G R G R G R — reds never recover to a new high → choppy.  The
        # trailing red run is the still-active pullback and is excused, so
        # only the two objectively failed pullbacks count (4 of 5 flips).
        opens = [5.00, 5.05, 4.98, 5.02, 4.96, 5.00]
        highs = [5.10, 5.06, 5.04, 5.03, 5.02, 5.01]
        closes = [5.05, 4.98, 5.02, 4.96, 5.00, 4.94]
        self.assertAlmostEqual(
            _candle_choppiness_ratio(opens, highs, closes), 0.8, places=6
        )

    def test_active_trailing_pullback_is_not_chop(self):
        # G G G R — the trailing red run is the current pullback; its outcome
        # is unknowable yet, so it must not count as chop.
        opens = [5.00, 5.08, 5.18, 5.28]
        highs = [5.10, 5.20, 5.30, 5.29]
        closes = [5.08, 5.18, 5.28, 5.22]
        self.assertEqual(_candle_choppiness_ratio(opens, highs, closes), 0.0)

    def test_unresolved_recovery_leg_is_not_chop(self):
        # G G R G — recovery green hasn't reclaimed the peak yet, but the
        # window ends there: still the active pullback, not chop.
        opens = [5.00, 5.08, 5.18, 5.12]
        highs = [5.10, 5.20, 5.19, 5.17]
        closes = [5.08, 5.18, 5.12, 5.16]
        self.assertEqual(_candle_choppiness_ratio(opens, highs, closes), 0.0)

    def test_failed_pullback_still_counts_once_resolved(self):
        # G R G R — the first pullback objectively failed (a second red run
        # began without the peak being reclaimed); only the trailing run is
        # excused. 2 of 3 flips count.
        opens = [5.00, 5.08, 5.00, 5.06]
        highs = [5.10, 5.09, 5.07, 5.06]
        closes = [5.08, 5.00, 5.06, 5.01]
        self.assertAlmostEqual(
            _candle_choppiness_ratio(opens, highs, closes), 2.0 / 3.0, places=6
        )

    def test_too_few_candles_returns_none(self):
        self.assertIsNone(_candle_choppiness_ratio([5.0], [5.1], [5.05]))


class TestAtrpAndChoppiness(unittest.TestCase):
    """DynamicScorer._update_atrp_and_choppiness — ATRP + candle choppiness."""

    _CLEAN = [
        (5.00, 5.10, 4.98, 5.08),
        (5.08, 5.20, 5.06, 5.18),
        (5.18, 5.30, 5.16, 5.28),
        (5.28, 5.30, 5.20, 5.22),   # red pullback
        (5.22, 5.40, 5.21, 5.38),   # recovers to a new closing high
        (5.38, 5.50, 5.36, 5.48),
    ]
    _CHOP = [
        (5.00, 5.10, 4.98, 5.05),
        (5.05, 5.06, 4.95, 4.98),
        (4.98, 5.04, 4.96, 5.02),
        (5.02, 5.03, 4.93, 4.96),
        (4.96, 5.02, 4.94, 5.00),
        (5.00, 5.01, 4.90, 4.94),
    ]

    @staticmethod
    def _ta_var(scorer, name):
        g = scorer._entry.evaluation.group_by_name("TECHNICAL_ANALYSIS")
        return g.get(name)

    @staticmethod
    def _frame_1m(rows):
        base = datetime(2026, 4, 10, 14, 0, 0, tzinfo=timezone.utc)
        return {
            "1min": _level_df([
                (base + timedelta(minutes=i), o, h, l, c)
                for i, (o, h, l, c) in enumerate(rows)
            ])
        }

    # ----- ATRP -----
    def test_atrp_value_and_midpoint_quality(self):
        scorer = _make_scorer()
        scorer._last_atr = 0.10
        scorer._last_price = 5.00
        scorer._update_atrp_and_choppiness({})
        v = self._ta_var(scorer, "atrp")
        self.assertAlmostEqual(v.value, 2.0, places=6)      # 0.10 / 5.00 * 100
        self.assertAlmostEqual(v.quality, 0.5, delta=0.02)  # SATURATING lo=2.0

    def test_atrp_higher_scores_higher(self):
        scorer = _make_scorer()
        scorer._last_atr = 0.30
        scorer._last_price = 5.00
        scorer._update_atrp_and_choppiness({})
        v = self._ta_var(scorer, "atrp")
        self.assertAlmostEqual(v.value, 6.0, places=6)
        self.assertGreater(v.quality, 0.6)

    def test_atrp_cleared_without_atr(self):
        scorer = _make_scorer()
        scorer._last_atr = 0.0
        scorer._update_atrp_and_choppiness({})
        self.assertIsNone(self._ta_var(scorer, "atrp").value)

    # ----- candle choppiness -----
    def test_choppiness_clean_frame_high_quality(self):
        scorer = _make_scorer()
        scorer._update_atrp_and_choppiness(self._frame_1m(self._CLEAN))
        v = self._ta_var(scorer, "candle_choppiness")
        self.assertIsNotNone(v.value)
        self.assertLess(v.value, 0.1)
        self.assertGreater(v.quality, 0.8)

    def test_choppiness_choppy_frame_low_quality(self):
        scorer = _make_scorer()
        scorer._update_atrp_and_choppiness(self._frame_1m(self._CHOP))
        v = self._ta_var(scorer, "candle_choppiness")
        self.assertIsNotNone(v.value)
        # 2 failed pullbacks count; the trailing (still-active) one is excused.
        self.assertGreater(v.value, 0.75)
        self.assertLess(v.quality, 0.2)

    def test_choppiness_insufficient_candles_is_none(self):
        scorer = _make_scorer()
        scorer._update_atrp_and_choppiness(self._frame_1m(self._CLEAN[:3]))
        self.assertIsNone(self._ta_var(scorer, "candle_choppiness").value)


class TestNoNewsPenalty(unittest.TestCase):
    """DynamicScorer._update_no_news_penalty — missing-catalyst penalty applied
    only after the grace window past start-of-move, and only when no news."""

    _MOVE_START = datetime(2026, 6, 10, 14, 0, 0, tzinfo=timezone.utc)

    @staticmethod
    def _bn(scorer):
        return scorer._entry.evaluation.stock_eval.get("breaking_news")

    def _no_news_scorer(self):
        entry = _make_entry()
        entry.avg_sentiment = None
        entry.news_count_last_hour = 0
        scorer = _make_scorer(entry=entry)
        bn = self._bn(scorer)
        bn.value = None          # no news → optional / excluded
        bn.quality = 0.0
        scorer._move_start_ts = self._MOVE_START
        return scorer

    def test_no_penalty_within_grace(self):
        from config import NO_NEWS_GRACE_SECS
        scorer = self._no_news_scorer()
        scorer._update_no_news_penalty(
            self._MOVE_START + timedelta(seconds=NO_NEWS_GRACE_SECS - 10)
        )
        self.assertIsNone(self._bn(scorer).value)  # still optional/neutral

    def test_penalty_applied_after_grace(self):
        from config import NO_NEWS_GRACE_SECS, NO_NEWS_PENALTY_QUALITY
        scorer = self._no_news_scorer()
        scorer._update_no_news_penalty(
            self._MOVE_START + timedelta(seconds=NO_NEWS_GRACE_SECS + 10)
        )
        bn = self._bn(scorer)
        self.assertIsNotNone(bn.value)                     # now included
        self.assertAlmostEqual(bn.quality, NO_NEWS_PENALTY_QUALITY, places=6)
        self.assertLess(bn.quality, 0.5)                   # negative impact

    def test_news_present_is_untouched_after_grace(self):
        from config import NO_NEWS_GRACE_SECS
        entry = _make_entry()
        entry.avg_sentiment = 0.6                          # news catalyst present
        scorer = _make_scorer(entry=entry)
        scorer._entry.evaluation.update_news(0.6, 1, "test")
        scorer._move_start_ts = self._MOVE_START
        before = self._bn(scorer).quality
        scorer._update_no_news_penalty(
            self._MOVE_START + timedelta(seconds=NO_NEWS_GRACE_SECS + 60)
        )
        self.assertEqual(self._bn(scorer).quality, before)  # news owns it

    def test_no_penalty_without_move_start(self):
        scorer = self._no_news_scorer()
        scorer._move_start_ts = None
        scorer._update_no_news_penalty(
            self._MOVE_START + timedelta(seconds=99999)
        )
        self.assertIsNone(self._bn(scorer).value)  # can't gate without move_start


class TestNoSetupsLiveConfidence(unittest.TestCase):
    """No setups → setup_outcome is excluded (does nothing) instead of a 50%."""

    def _clear_ta(self, scorer):
        ta = scorer._entry.evaluation.technical_analysis
        for v in ta.variables:
            v.value = None
            v.quality = 0.0
        return ta

    def test_no_setups_excludes_setup_outcome_and_score_reflects_other_signals(self):
        from strategy.scoring import LiveInputs
        scorer = _make_scorer()
        ev = scorer._entry.evaluation
        ta = self._clear_ta(scorer)
        # No setups (empty _tf_pa_scores) + one strong live signal (atrp).
        ev.apply_live(LiveInputs(atrp=6.0))
        scorer._refresh_technical_analysis_score()

        self.assertEqual(scorer._tf_pa_scores, {})           # no setups tracked
        self.assertIsNone(ta.score_override)                 # override cleared
        self.assertIsNone(ta.get("setup_outcome").value)     # excluded (—), not 0.5
        # Score reflects atrp alone, NOT pinned at 50% by setup_outcome.
        self.assertGreater(ta.score, 0.7)

    def test_no_setups_no_data_is_not_fifty_percent(self):
        scorer = _make_scorer()
        ta = self._clear_ta(scorer)
        scorer._refresh_technical_analysis_score()

        self.assertIsNone(ta.score_override)
        self.assertIsNone(ta.get("setup_outcome").value)     # excluded, not neutral 0.5
        self.assertEqual(ta.score, 0.0)                      # nothing active → does nothing
        self.assertFalse(scorer._entry.has_live_data)        # display shows "L--"


class TestSetupOutcome10Sec(unittest.TestCase):
    """10s setup outcomes populate the per-timeframe TA table."""

    @staticmethod
    def _seed_active_1m_pullback(scorer, base: datetime) -> None:
        rows = [
            (base, 5.00, 5.15, 4.98, 5.12),
            (base + timedelta(minutes=1), 5.12, 5.35, 5.10, 5.30),
            (base + timedelta(minutes=2), 5.30, 5.65, 5.28, 5.60),
            (base + timedelta(minutes=3), 5.60, 5.63, 5.38, 5.42),
        ]
        for ts, open_, high, low, close in rows:
            scorer._tf_buf._bars["1min"].append(OHLCVCandle(
                "TEST", ts, open_, high, low, close, 1000, "1min",
            ))

    @staticmethod
    def _failed_leg_score(failure_bar: OHLCVCandle):
        scorer = _make_scorer()
        ts = datetime(2026, 6, 10, 14, 30, 0, tzinfo=timezone.utc)

        # Baseline history so ATR and the relative-volume floor are realistic.
        history = [
            (5.00, 5.06, 4.96, 5.04),
            (5.04, 5.12, 5.02, 5.10),
            (5.10, 5.16, 5.06, 5.14),
            (5.14, 5.22, 5.12, 5.20),
            (5.20, 5.26, 5.16, 5.24),
        ]
        for i, (open_, high, low, close) in enumerate(history):
            scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
                "TEST", ts - timedelta(seconds=10 * (len(history) - i)),
                open_, high, low, close, 1_000, "10sec",
            ))

        scorer._on_setup_resolved("10sec", quality=0.80, breakout_price=5.00)
        scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
            "TEST", ts, 5.00, 5.30, 4.98, 5.25, 1000, "10sec",
        ))
        scorer._on_setup_leg_candle_close(["10sec"])
        scorer._tf_buf._bars["10sec"].append(failure_bar)
        scorer._on_setup_leg_candle_close(["10sec"])

        return scorer._tf_pa_var_qualities["10s"]["setup_outcome"], scorer._completed_outcomes[-1]

    def test_green_10s_leg_updates_setup_outcome_cell(self):
        scorer = _make_scorer()
        ts = datetime(2026, 6, 10, 14, 30, 0, tzinfo=timezone.utc)

        scorer._on_setup_resolved("10sec", quality=0.80, breakout_price=5.00)
        scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
            "TEST", ts, 5.00, 5.08, 4.98, 5.05, 1000, "10sec",
        ))
        scorer._on_setup_leg_candle_close(["10sec"])

        self.assertIn("10s", scorer._tf_pa_scores)
        self.assertIn("setup_outcome", scorer._tf_pa_var_qualities["10s"])
        self.assertGreater(scorer._tf_pa_var_qualities["10s"]["setup_outcome"], 0.0)

    def test_10s_setup_inside_1min_pullback_is_not_scored(self):
        scorer = _make_scorer()
        base = datetime(2026, 6, 10, 14, 26, 0, tzinfo=timezone.utc)
        self._seed_active_1m_pullback(scorer, base)

        scorer._on_setup_resolved(
            "10sec",
            quality=0.80,
            breakout_price=5.00,
            pullback_ts=base + timedelta(minutes=3, seconds=20),
        )
        scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
            "TEST", base + timedelta(minutes=3, seconds=30), 5.00, 5.20, 4.98, 5.18, 1000, "10sec",
        ))
        scorer._on_setup_leg_candle_close(["10sec"])

        self.assertNotIn("10s", scorer._tf_pa_scores)
        self.assertNotIn("setup_outcome", scorer._tf_pa_var_qualities["10s"])
        self.assertNotIn("10s", scorer._active_legs)
        self.assertEqual([o for o in scorer._completed_outcomes if o.timeframe == "10s"], [])

    def test_10s_breakout_rejection_inside_1min_pullback_is_not_scored(self):
        scorer = _make_scorer()
        base = datetime(2026, 6, 10, 14, 26, 0, tzinfo=timezone.utc)
        self._seed_active_1m_pullback(scorer, base)
        scorer._entry.setup_pullback_ts = base + timedelta(minutes=3, seconds=20)
        scorer._breakout_trigger_level = 5.00

        scorer._on_setup_breakout_rejected("10sec", OHLCVCandle(
            "TEST", base + timedelta(minutes=3, seconds=30), 5.05, 5.08, 4.90, 4.92, 1000, "10sec",
        ))

        self.assertNotIn("10s", scorer._tf_pa_scores)
        self.assertNotIn("setup_outcome", scorer._tf_pa_var_qualities["10s"])
        self.assertEqual([o for o in scorer._completed_outcomes if o.timeframe == "10s"], [])

    def test_failed_setup_severity_is_gradual(self):
        ts = datetime(2026, 6, 10, 14, 30, 10, tzinfo=timezone.utc)
        mild_score, mild_outcome = self._failed_leg_score(OHLCVCandle(
            "TEST", ts, 5.25, 5.26, 5.23, 5.24, 1000, "10sec",
        ))
        harsh_score, harsh_outcome = self._failed_leg_score(OHLCVCandle(
            "TEST", ts, 5.25, 5.65, 4.95, 5.02, 3000, "10sec",
        ))

        self.assertFalse(mild_outcome.success)
        self.assertFalse(harsh_outcome.success)
        self.assertGreater(mild_score, 0.35)
        self.assertLess(harsh_score, mild_score * 0.5)
        self.assertGreater(mild_outcome.failure_quality, harsh_outcome.failure_quality)
        self.assertGreater(harsh_outcome.failure_rejection, 0.7)

    def test_low_volume_failure_candle_is_softened(self):
        # Same ugly rejection shape; only the failure candle's volume differs.
        ts = datetime(2026, 6, 10, 14, 30, 10, tzinfo=timezone.utc)
        low_vol_score, low_vol_outcome = self._failed_leg_score(OHLCVCandle(
            "TEST", ts, 5.25, 5.65, 4.95, 5.02, 900, "10sec",
        ))
        high_vol_score, high_vol_outcome = self._failed_leg_score(OHLCVCandle(
            "TEST", ts, 5.25, 5.65, 4.95, 5.02, 3000, "10sec",
        ))

        # Low-volume red failure = healthy pullback behaviour -> keeps most credit.
        self.assertGreater(low_vol_outcome.failure_quality, 0.5)
        self.assertLess(high_vol_outcome.failure_quality, 0.1)
        self.assertGreater(
            low_vol_outcome.failure_quality,
            high_vol_outcome.failure_quality + 0.3,
        )
        self.assertGreater(low_vol_score, high_vol_score)
        # But a structurally severe candle is never scored as fully harmless.
        self.assertLess(low_vol_outcome.failure_quality, 0.9)
        self.assertGreater(low_vol_outcome.failure_rejection, 0.1)

    def test_failed_setup_recovers_after_reclaim_and_strong_close(self):
        scorer = _make_scorer()
        ts = datetime(2026, 6, 10, 14, 30, 0, tzinfo=timezone.utc)

        scorer._on_setup_resolved("10sec", quality=0.80, breakout_price=5.00)
        scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
            "TEST", ts, 5.00, 5.05, 4.85, 4.90, 1000, "10sec",
        ))
        scorer._on_setup_leg_candle_close(["10sec"])
        failed_score = scorer._tf_pa_var_qualities["10s"]["setup_outcome"]

        scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
            "TEST", ts + timedelta(seconds=10), 4.92, 5.08, 4.90, 5.06, 1000, "10sec",
        ))
        scorer._on_setup_leg_candle_close(["10sec"])
        scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
            "TEST", ts + timedelta(seconds=20), 5.06, 5.42, 5.04, 5.38, 1000, "10sec",
        ))
        scorer._on_setup_leg_candle_close(["10sec"])

        repaired_score = scorer._tf_pa_var_qualities["10s"]["setup_outcome"]
        self.assertLess(failed_score, 0.05)
        self.assertGreater(repaired_score, 0.25)
        self.assertLess(repaired_score, 0.55)


class TestSetupOutcomeVolumeSignificance(unittest.TestCase):
    """Setup legs only carry evidence while their peak candle volume clears
    the CURRENT relative-volume band (median -> top-percentile), mirroring
    key_candles's per-candle volume weighting."""

    @staticmethod
    def _scorer_with_history(volumes):
        scorer = _make_scorer()
        ts = datetime(2026, 6, 10, 14, 30, 0, tzinfo=timezone.utc)
        price = 5.00
        for i, vol in enumerate(volumes):
            open_ = price
            close = price + 0.02
            scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
                "TEST", ts - timedelta(seconds=10 * (len(volumes) - i)),
                open_, close + 0.02, open_ - 0.02, close, vol, "10sec",
            ))
            price = close
        return scorer, ts

    def test_high_volume_failure_keeps_evidence_mass(self):
        scorer, ts = self._scorer_with_history([400, 600, 800, 1_000, 1_200])
        scorer._breakout_trigger_level = 5.00

        # Failure candle prints above the window's top-volume anchor.
        scorer._on_setup_breakout_rejected("10sec", OHLCVCandle(
            "TEST", ts, 5.10, 5.12, 4.90, 4.92, 2_000, "10sec",
        ))

        self.assertIn("10s", scorer._tf_pa_scores)
        self.assertGreater(scorer._tf_pa_masses["10s"], 0.5)

    def test_micro_leg_loses_evidence_as_tape_heats_up(self):
        scorer, ts = self._scorer_with_history([400, 600, 800, 1_000, 1_200])
        scorer._breakout_trigger_level = 5.00
        scorer._on_setup_breakout_rejected("10sec", OHLCVCandle(
            "TEST", ts, 5.10, 5.12, 4.90, 4.92, 2_000, "10sec",
        ))
        self.assertGreater(scorer._tf_pa_masses["10s"], 0.5)

        # Real volume arrives: 2k was top-of-tape then, but it sits below the
        # new median once 50k candles print — the old micro-leg's evidence
        # mass collapses on the next rescore, at the current time.
        price = 5.00
        for i in range(10):
            scorer._tf_buf._bars["10sec"].append(OHLCVCandle(
                "TEST", ts + timedelta(seconds=10 * (i + 1)),
                price, price + 0.06, price - 0.02, price + 0.05, 50_000, "10sec",
            ))
            price += 0.05
        scorer._on_setup_leg_candle_close(["10sec"])

        # Score is still published for display, but carries no evidence…
        self.assertIn("10s", scorer._tf_pa_scores)
        self.assertEqual(scorer._tf_pa_masses["10s"], 0.0)
        # …so the TA blend ignores the timeframe entirely.
        self.assertEqual(
            combine_tf_scores(scorer._tf_pa_scores, scorer._tf_pa_masses), 0.5
        )

    def test_uniform_tape_falls_back_to_unweighted_score(self):
        # Undifferentiated volume: every leg is "typical", nothing clears the
        # median floor -> the score itself must not change (single outcome,
        # unweighted fallback), only the evidence mass goes to zero.
        scorer, ts = self._scorer_with_history([1_000] * 5)
        scorer._breakout_trigger_level = 5.00
        scorer._on_setup_breakout_rejected("10sec", OHLCVCandle(
            "TEST", ts, 5.10, 5.12, 4.90, 4.92, 1_000, "10sec",
        ))

        self.assertIn("10s", scorer._tf_pa_scores)
        self.assertEqual(scorer._tf_pa_masses["10s"], 0.0)


if __name__ == "__main__":
    unittest.main()
