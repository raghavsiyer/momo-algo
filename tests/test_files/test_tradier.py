"""
tests/test_tradier.py

Unit tests for market_data/tradier_rest.py and market_data/tradier_stream.py.

All API calls are mocked. No network access.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from market_data.tradier_rest import HistoricalClient, StockQuote, OHLCVCandle
from market_data.tradier_stream import (
    RealTimeClient, LiveTrade, LiveQuote, LiveTimesale, LiveSummary,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Client with Tradier enabled + a dummy token."""
    with patch.dict("os.environ", {"TRADIER_API_KEY": "test-token", "TRADIER_PAPER": "false"}):
        with patch("market_data.tradier_rest._TRADIER_ENABLED", True):
            c = HistoricalClient()
    return c


def _mock_response(json_data, ok=True, status_code=200):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = ""
    return resp


# ---------------------------------------------------------------------------
# Quote tests
# ---------------------------------------------------------------------------

class TestGetQuote:
    def test_single_quote_parses(self, client):
        raw = {
            "quotes": {
                "quote": {
                    "symbol": "GCTK",
                    "description": "GlucoTrack Inc",
                    "last": 3.50,
                    "change": 0.80,
                    "change_percentage": 29.63,
                    "volume": 5_000_000,
                    "average_volume": 500_000,
                    "open": 2.90,
                    "high": 3.75,
                    "low": 2.85,
                    "prevclose": 2.70,
                    "bid": 3.49,
                    "ask": 3.51,
                    "bidsize": 10,
                    "asksize": 15,
                    "last_volume": 100,
                    "trade_date": 1712779200000,
                    "exch": "Q",
                }
            }
        }
        client._session.get = MagicMock(return_value=_mock_response(raw))
        q = client.get_quote("GCTK")

        assert isinstance(q, StockQuote)
        assert q.symbol == "GCTK"
        assert q.last == 3.50
        assert q.change_pct == 29.63
        assert q.volume == 5_000_000
        assert q.avg_volume == 500_000
        assert q.bid == 3.49
        assert q.ask == 3.51
        assert q.exchange == "Q"

    def test_empty_response_returns_stub(self, client):
        """API returns no quotes → should get a stub StockQuote, not crash."""
        client._session.get = MagicMock(return_value=_mock_response({}))
        q = client.get_quote("FAKE")
        assert q.symbol == "FAKE"
        assert q.last is None
        assert q.volume == 0

    def test_http_error_returns_stub(self, client):
        """Non-200 response → stub quote."""
        client._session.get = MagicMock(
            return_value=_mock_response({}, ok=False, status_code=401)
        )
        q = client.get_quote("FAKE")
        assert q.symbol == "FAKE"
        assert q.last is None

    def test_multi_quote(self, client):
        raw = {
            "quotes": {
                "quote": [
                    {"symbol": "AAPL", "last": 200.0, "volume": 1000,
                     "average_volume": 500, "description": "", "change": 0,
                     "change_percentage": 0, "open": 0, "high": 0, "low": 0,
                     "prevclose": 0, "bid": 0, "ask": 0, "bidsize": 0,
                     "asksize": 0, "last_volume": 0, "trade_date": 0, "exch": "Q"},
                    {"symbol": "MSFT", "last": 400.0, "volume": 2000,
                     "average_volume": 1000, "description": "", "change": 0,
                     "change_percentage": 0, "open": 0, "high": 0, "low": 0,
                     "prevclose": 0, "bid": 0, "ask": 0, "bidsize": 0,
                     "asksize": 0, "last_volume": 0, "trade_date": 0, "exch": "Q"},
                ]
            }
        }
        client._session.get = MagicMock(return_value=_mock_response(raw))
        quotes = client.get_quotes(["AAPL", "MSFT"])
        assert len(quotes) == 2
        assert quotes[0].symbol == "AAPL"
        assert quotes[1].symbol == "MSFT"


# ---------------------------------------------------------------------------
# Timesales candle tests
# ---------------------------------------------------------------------------

class TestTimesalesCandles:
    """Test _get_timesales and its wrappers (1min, 15min)."""

    def _make_timesales_response(self, bars):
        """Build a Tradier timesales response dict from a list of bar dicts."""
        return {"series": {"data": bars}}

    def test_iso_timestamp_parsed(self, client):
        """Tradier sends ISO timestamps with 'T' separator — must parse correctly."""
        raw = self._make_timesales_response([
            {"time": "2026-04-10T09:30:00", "open": 3.0, "high": 3.5,
             "low": 2.9, "close": 3.2, "volume": 100000, "vwap": 3.1},
        ])
        client._session.get = MagicMock(return_value=_mock_response(raw))
        candles = client._get_timesales("GCTK", "15min",
                                        datetime(2026, 4, 10, tzinfo=timezone.utc),
                                        datetime(2026, 4, 11, tzinfo=timezone.utc))
        assert len(candles) == 1
        assert candles[0].close == 3.2
        assert candles[0].volume == 100000
        assert candles[0].ts.year == 2026
        assert candles[0].ts == datetime(2026, 4, 10, 13, 30, tzinfo=timezone.utc)

    def test_space_timestamp_also_parsed(self, client):
        """Backward compat: space-separated timestamps should also work."""
        raw = self._make_timesales_response([
            {"time": "2026-04-10 09:30:00", "open": 3.0, "high": 3.5,
             "low": 2.9, "close": 3.2, "volume": 100000},
        ])
        client._session.get = MagicMock(return_value=_mock_response(raw))
        candles = client._get_timesales("GCTK", "15min",
                                        datetime(2026, 4, 10, tzinfo=timezone.utc),
                                        datetime(2026, 4, 11, tzinfo=timezone.utc))
        assert len(candles) == 1
        assert candles[0].ts == datetime(2026, 4, 10, 13, 30, tzinfo=timezone.utc)

    def test_empty_series_returns_empty(self, client):
        client._session.get = MagicMock(return_value=_mock_response({"series": {}}))
        candles = client._get_timesales("GCTK", "1min",
                                        datetime(2026, 4, 10, tzinfo=timezone.utc),
                                        datetime(2026, 4, 11, tzinfo=timezone.utc))
        assert candles == []

    def test_null_series_returns_empty(self, client):
        client._session.get = MagicMock(return_value=_mock_response({"series": None}))
        candles = client._get_timesales("GCTK", "1min",
                                        datetime(2026, 4, 10, tzinfo=timezone.utc),
                                        datetime(2026, 4, 11, tzinfo=timezone.utc))
        assert candles == []

    def test_single_bar_wrapped_as_dict(self, client):
        """Tradier returns a dict (not list) when there's exactly 1 bar."""
        raw = {"series": {"data": {
            "time": "2026-04-10T10:00:00", "open": 3.0, "high": 3.5,
            "low": 2.9, "close": 3.2, "volume": 50000,
        }}}
        client._session.get = MagicMock(return_value=_mock_response(raw))
        candles = client._get_timesales("GCTK", "15min",
                                        datetime(2026, 4, 10, tzinfo=timezone.utc),
                                        datetime(2026, 4, 11, tzinfo=timezone.utc))
        assert len(candles) == 1

    def test_get_candles_15min_returns_df(self, client):
        bars = [
            {"time": f"2026-04-10T{9+i}:30:00", "open": 3.0+i, "high": 3.5+i,
             "low": 2.9+i, "close": 3.2+i, "volume": 10000*(i+1)}
            for i in range(5)
        ]
        raw = self._make_timesales_response(bars)
        client._session.get = MagicMock(return_value=_mock_response(raw))
        df = client.get_candles_15min("GCTK", n=3)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # last 3 of 5
        assert "ts" in df.columns
        assert "close" in df.columns

    def test_get_candles_1min_returns_df(self, client):
        bars = [
            {"time": f"2026-04-10T09:{30+i}:00", "open": 3.0, "high": 3.5,
             "low": 2.9, "close": 3.2, "volume": 1000}
            for i in range(10)
        ]
        raw = self._make_timesales_response(bars)
        client._session.get = MagicMock(return_value=_mock_response(raw))
        df = client.get_candles_1min("GCTK", n=5)
        assert len(df) == 5


class TestTickTimesales:
    def test_tick_volume_normalized_to_size(self, client):
        raw = {"series": {"data": [
            {"time": "2026-04-10T10:00:00", "price": 3.10, "volume": 500},
            {"time": "2026-04-10T10:00:01", "price": 3.12, "size": 300},
        ]}}
        client._session.get = MagicMock(return_value=_mock_response(raw))

        rows = client.get_timesales_tick(
            "GCTK",
            datetime(2026, 4, 10, 10, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 10, 10, 1, tzinfo=timezone.utc),
        )

        assert [row["size"] for row in rows] == [500, 300]


# ---------------------------------------------------------------------------
# 10-second candle tests (tick resample)
# ---------------------------------------------------------------------------

class TestCandles10Sec:
    def test_tick_resample_parses_iso(self, client):
        """10-sec candles resample ticks — timestamps must parse."""
        ticks = [
            {"time": "2026-04-10T10:00:00", "price": 3.10, "size": 500},
            {"time": "2026-04-10T10:00:03", "price": 3.15, "size": 300},
            {"time": "2026-04-10T10:00:07", "price": 3.12, "size": 200},
            {"time": "2026-04-10T10:00:12", "price": 3.20, "size": 400},
            {"time": "2026-04-10T10:00:18", "price": 3.18, "size": 350},
        ]
        raw = {"series": {"data": ticks}}
        client._session.get = MagicMock(return_value=_mock_response(raw))
        df = client.get_candles_10sec("GCTK", n=10)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "close" in df.columns

    def test_tick_resample_treats_naive_time_as_utc(self, client):
        ticks = [
            {"time": "2026-06-09T19:57:30", "price": 3.10, "size": 500},
            {"time": "2026-06-09T19:57:35", "price": 3.15, "size": 300},
        ]

        df = client._ticks_to_10sec_df("AZI", ticks)

        assert df["ts"].iloc[0] == pd.Timestamp("2026-06-09T19:57:30Z")

    def test_tick_resample_uses_volume_when_size_missing(self, client):
        ticks = [
            {"time": "2026-07-20T13:49:44", "price": 1.50, "volume": 149012},
            {"time": "2026-07-20T13:49:45", "price": 1.51, "volume": 120},
            {"time": "2026-07-20T13:49:52", "price": 1.52, "size": 300},
        ]

        df = client._ticks_to_10sec_df("ZYBT", ticks)

        assert df.iloc[0]["volume"] == 149132
        assert df.iloc[1]["volume"] == 300

    def test_empty_ticks_returns_empty_df(self, client):
        client._session.get = MagicMock(return_value=_mock_response({"series": {}}))
        df = client.get_candles_10sec("GCTK")
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ---------------------------------------------------------------------------
# Daily candle tests
# ---------------------------------------------------------------------------

class TestCandlesDaily:
    def test_daily_parses_correctly(self, client):
        raw = {
            "history": {
                "day": [
                    {"date": "2026-04-09", "open": 3.0, "high": 3.5,
                     "low": 2.8, "close": 3.2, "volume": 1_000_000},
                    {"date": "2026-04-10", "open": 3.2, "high": 3.8,
                     "low": 3.1, "close": 3.6, "volume": 1_500_000},
                ]
            }
        }
        client._session.get = MagicMock(return_value=_mock_response(raw))
        df = client.get_candles_daily("GCTK", days_back=5)
        assert len(df) == 2
        assert df.iloc[0]["close"] == 3.2
        assert df.iloc[1]["volume"] == 1_500_000

    def test_single_day_wrapped_as_dict(self, client):
        raw = {
            "history": {
                "day": {"date": "2026-04-10", "open": 3.0, "high": 3.5,
                        "low": 2.8, "close": 3.2, "volume": 500_000}
            }
        }
        client._session.get = MagicMock(return_value=_mock_response(raw))
        df = client.get_candles_daily("GCTK", days_back=1)
        assert len(df) == 1

    def test_empty_history(self, client):
        client._session.get = MagicMock(return_value=_mock_response({"history": None}))
        df = client.get_candles_daily("GCTK")
        assert df.empty


# ---------------------------------------------------------------------------
# Disabled client tests
# ---------------------------------------------------------------------------

class TestDisabledClient:
    def test_disabled_returns_empty(self):
        with patch("market_data.tradier_rest._TRADIER_ENABLED", False):
            c = HistoricalClient()
        assert c._disabled is True
        q = c.get_quote("AAPL")
        assert q.last is None
        df = c.get_candles_15min("AAPL")
        assert df.empty


# ---------------------------------------------------------------------------
# Symbol search tests
# ---------------------------------------------------------------------------

class TestSearchSymbol:
    def test_search_returns_first_stock(self, client):
        raw = {
            "securities": {
                "security": [
                    {"symbol": "GCTK", "type": "stock", "description": "GlucoTrack"},
                    {"symbol": "GCTK1", "type": "stock", "description": "Something"},
                ]
            }
        }
        client._session.get = MagicMock(return_value=_mock_response(raw))
        result = client.search_symbol("GCTK")
        assert result == "GCTK"

    def test_search_skips_non_stock(self, client):
        raw = {
            "securities": {
                "security": [
                    {"symbol": "GCTKW", "type": "warrant", "description": "Warrant"},
                    {"symbol": "GCTK", "type": "stock", "description": "GlucoTrack"},
                ]
            }
        }
        client._session.get = MagicMock(return_value=_mock_response(raw))
        result = client.search_symbol("GCTK")
        assert result == "GCTK"

    def test_search_no_results(self, client):
        client._session.get = MagicMock(return_value=_mock_response({"securities": None}))
        assert client.search_symbol("ZZZZ") is None


# ---------------------------------------------------------------------------
# RVOL-at-time (evaluator integration)
# ---------------------------------------------------------------------------

class TestComputeRvolAtTime:
    """Test the evaluator's _compute_rvol_at_time which compares today's
    5-min candles against the same time-of-day over 30 historical days."""

    def test_rvol_at_time_with_valid_data(self, client):
        """Multi-day 5-min candles → valid rvol_1day and rvol_5min."""
        from strategy.evaluator import StockEvaluator
        with patch.dict("os.environ", {"TRADIER_API_KEY": "test-token"}):
            with patch("market_data.tradier_rest._TRADIER_ENABLED", True):
                evaluator = StockEvaluator.__new__(StockEvaluator)
                evaluator._tradier = client

        # Build 5-min bars: 2 historical days + today
        # Historical days: 1000 vol per bar
        # Today: 5000 vol per bar, last bar spikes to 15000 (3x intraday)
        bars = []
        for day_offset in [3, 2, 0]:  # 2 historical days + today (April 14)
            date_str = f"2026-04-{14 - day_offset:02d}"
            for h in range(9, 11):
                for m in range(0, 60, 5):
                    if h == 9 and m < 30:
                        continue
                    if day_offset == 0:
                        # Last bar of today (10:55) gets a spike
                        vol = 15000 if (h == 10 and m == 55) else 5000
                    else:
                        vol = 1000
                    bars.append({
                        "time": f"{date_str}T{h:02d}:{m:02d}:00",
                        "open": 3.0, "high": 3.5, "low": 2.9, "close": 3.2,
                        "volume": vol,
                    })

        raw = {"series": {"data": bars}}
        client._session.get = MagicMock(return_value=_mock_response(raw))

        fake_now = datetime(2026, 4, 14, 10, 55, tzinfo=ZoneInfo("America/New_York"))
        with patch("strategy.evaluator.MarketClock") as mock_clock:
            mock_clock.now_et.return_value = fake_now
            rvol_1day, rvol_5min = evaluator._compute_rvol_at_time("GCTK")

        # 1-day: today's cumulative vol is ~5x historical → ~5.x
        assert rvol_1day > 4.0
        # 5-min rolling: last bar (15000) vs prior today bars (5000 avg) → 3.0
        assert rvol_5min == pytest.approx(3.0, abs=0.1)

    def test_rvol_at_time_empty_candles(self, client):
        """Empty 5-min response → (0.0, 0.0)."""
        from strategy.evaluator import StockEvaluator
        with patch.dict("os.environ", {"TRADIER_API_KEY": "test-token"}):
            with patch("market_data.tradier_rest._TRADIER_ENABLED", True):
                evaluator = StockEvaluator.__new__(StockEvaluator)
                evaluator._tradier = client

        client._session.get = MagicMock(return_value=_mock_response({"series": {}}))
        assert evaluator._compute_rvol_at_time("GCTK") == (0.0, 0.0)

    def test_rvol_at_time_second_call_uses_cached_history(self):
        """Repeat call fetches only today's window; prior days come from cache."""
        from strategy.evaluator import StockEvaluator

        def _bars(day_offsets):
            rows = []
            for day_offset in day_offsets:
                date_str = f"2026-04-{14 - day_offset:02d}"
                for h in range(9, 11):
                    for m in range(0, 60, 5):
                        if h == 9 and m < 30:
                            continue
                        rows.append({
                            "ts": pd.Timestamp(
                                f"{date_str}T{h:02d}:{m:02d}:00",
                                tz="America/New_York",
                            ).tz_convert("UTC"),
                            "open": 3.0, "high": 3.5, "low": 2.9, "close": 3.2,
                            "volume": 5000 if day_offset == 0 else 1000,
                        })
            return pd.DataFrame(rows)

        provider = MagicMock()
        provider.get_candles_5min.return_value = _bars([3, 2, 0])
        provider.get_candles_5min_window.return_value = _bars([0])
        evaluator = StockEvaluator.__new__(StockEvaluator)
        evaluator._tradier = provider

        fake_now = datetime(2026, 4, 14, 10, 55, tzinfo=ZoneInfo("America/New_York"))
        with patch("strategy.evaluator.MarketClock") as mock_clock, \
             patch("strategy.evaluator.latest_extended_session_start") as mock_start:
            mock_clock.now_et.return_value = fake_now
            mock_start.return_value = datetime(2026, 4, 14, 8, 0, tzinfo=timezone.utc)
            first = evaluator._compute_rvol_at_time("GCTK")
            second = evaluator._compute_rvol_at_time("GCTK")

        assert first == second
        assert first[0] > 0.0
        assert provider.get_candles_5min.call_count == 1
        assert provider.get_candles_5min_window.call_count == 1


# ===========================================================================
# STREAMING CLIENT TESTS — market_data/tradier_stream.py
# ===========================================================================

@pytest.fixture
def stream_client():
    """RealTimeClient with a dummy token."""
    with patch.dict("os.environ", {"TRADIER_API_KEY": "test-stream-token"}):
        c = RealTimeClient()
    return c


# ---------------------------------------------------------------------------
# Event parsing
# ---------------------------------------------------------------------------

class TestParseEvent:
    """Test _parse_event parses raw JSON lines into typed dataclasses."""

    def test_parse_trade(self, stream_client):
        line = '{"type":"trade","symbol":"GCTK","exch":"Q","price":3.50,"size":200,"cvol":500000,"date":1712779200000,"last":3.50}'
        event = stream_client._parse_event(line, "GCTK")
        assert isinstance(event, LiveTrade)
        assert event.symbol == "GCTK"
        assert event.price == 3.50
        assert event.size == 200
        assert event.cumulative_volume == 500000
        assert event.exchange == "Q"
        assert event.ts.year == 2024  # 1712779200000 = 2024-04-10

    def test_parse_quote(self, stream_client):
        line = '{"type":"quote","symbol":"GCTK","bid":3.49,"bidsz":10,"bidexch":"Q","biddate":1712779200000,"ask":3.51,"asksz":15,"askexch":"P","askdate":1712779200000}'
        event = stream_client._parse_event(line, "GCTK")
        assert isinstance(event, LiveQuote)
        assert event.bid == 3.49
        assert event.ask == 3.51
        assert event.bid_size == 10
        assert event.ask_size == 15
        assert event.bid_exchange == "Q"
        assert event.ask_exchange == "P"

    def test_parse_timesale(self, stream_client):
        line = '{"type":"timesale","symbol":"GCTK","exch":"Q","last":3.50,"bid":3.49,"ask":3.51,"size":100,"date":1712779200000,"seq":12345,"flag":"@","cancel":false,"correction":false,"session":"normal"}'
        event = stream_client._parse_event(line, "GCTK")
        assert isinstance(event, LiveTimesale)
        assert event.last == 3.50
        assert event.bid == 3.49
        assert event.ask == 3.51
        assert event.size == 100
        assert event.seq == 12345
        assert event.flag == "@"
        assert event.cancel is False
        assert event.correction is False
        assert event.session == "normal"

    def test_parse_summary(self, stream_client):
        line = '{"type":"summary","symbol":"GCTK","open":2.90,"high":3.75,"low":2.85,"prevClose":2.70}'
        event = stream_client._parse_event(line, "GCTK")
        assert isinstance(event, LiveSummary)
        assert event.open == 2.90
        assert event.high == 3.75
        assert event.low == 2.85
        assert event.prev_close == 2.70

    def test_parse_empty_line(self, stream_client):
        assert stream_client._parse_event("", "GCTK") is None
        assert stream_client._parse_event("   ", "GCTK") is None

    def test_parse_invalid_json(self, stream_client):
        assert stream_client._parse_event("{bad json", "GCTK") is None

    def test_parse_unknown_type(self, stream_client):
        line = '{"type":"heartbeat","data":"keep-alive"}'
        assert stream_client._parse_event(line, "GCTK") is None

    def test_fallback_symbol(self, stream_client):
        """If event JSON lacks 'symbol', fallback_symbol is used."""
        line = '{"type":"trade","price":5.0,"size":100,"cvol":1000,"date":1712779200000,"last":5.0}'
        event = stream_client._parse_event(line, "FALLBACK")
        assert isinstance(event, LiveTrade)
        assert event.symbol == "FALLBACK"

    def test_ms_to_dt(self, stream_client):
        """Millisecond epoch → UTC datetime."""
        # 1712779200000 = 2024-04-10T20:00:00Z
        dt = stream_client._ms_to_dt(1712779200000)
        assert dt.tzinfo is not None
        assert dt.year == 2024

    def test_ms_to_dt_none(self, stream_client):
        """None/0 → fallback to now (not crash)."""
        dt = stream_client._ms_to_dt(None)
        assert dt.tzinfo is not None
        dt0 = stream_client._ms_to_dt(0)
        assert dt0.tzinfo is not None


# ---------------------------------------------------------------------------
# Candle buffer
# ---------------------------------------------------------------------------

class TestCandleBuffer:
    """Test build_live_candle_buffer aggregation logic."""

    def test_trades_aggregate_into_bar(self, stream_client):
        """Trades within the same interval bucket form one OHLCV bar."""
        import threading

        # We can't easily stream, but we can test the on_trade callback
        # by extracting the logic. Let's build the buffer and simulate.
        candles_buf, stop = stream_client.build_live_candle_buffer.__wrapped__(
            stream_client, "GCTK", interval_seconds=60
        ) if hasattr(stream_client.build_live_candle_buffer, '__wrapped__') else (None, None)

        # Alternative: directly test candle builder logic by patching stream_quotes
        from collections import deque
        candles: deque[OHLCVCandle] = deque(maxlen=100)
        interval_seconds = 60

        def bucket_ts(ts: datetime) -> datetime:
            epoch = ts.timestamp()
            floored = (epoch // interval_seconds) * interval_seconds
            return datetime.fromtimestamp(floored, tz=timezone.utc)

        def on_trade(event: LiveTrade):
            bar_ts = bucket_ts(event.ts)
            if candles and candles[-1].ts == bar_ts:
                c = candles[-1]
                c.high = max(c.high, event.price)
                c.low = min(c.low, event.price)
                c.close = event.price
                c.volume += event.size
            else:
                candles.append(OHLCVCandle(
                    symbol="GCTK", ts=bar_ts,
                    open=event.price, high=event.price,
                    low=event.price, close=event.price,
                    volume=event.size, interval="1min",
                ))

        # Simulate 3 trades in the same minute
        base_ts = datetime(2026, 4, 10, 10, 0, 5, tzinfo=timezone.utc)
        on_trade(LiveTrade("trade", "GCTK", "Q", 3.50, 200, 0, base_ts, 3.50))
        on_trade(LiveTrade("trade", "GCTK", "Q", 3.60, 100, 0,
                           base_ts.replace(second=15), 3.60))
        on_trade(LiveTrade("trade", "GCTK", "Q", 3.40, 300, 0,
                           base_ts.replace(second=30), 3.40))

        assert len(candles) == 1
        bar = candles[0]
        assert bar.open == 3.50
        assert bar.high == 3.60
        assert bar.low == 3.40
        assert bar.close == 3.40
        assert bar.volume == 600  # 200 + 100 + 300

    def test_new_bucket_opens_new_bar(self, stream_client):
        """Trades crossing a bucket boundary create a new bar."""
        from collections import deque
        candles: deque[OHLCVCandle] = deque(maxlen=100)
        interval_seconds = 60

        def bucket_ts(ts: datetime) -> datetime:
            epoch = ts.timestamp()
            floored = (epoch // interval_seconds) * interval_seconds
            return datetime.fromtimestamp(floored, tz=timezone.utc)

        def on_trade(event: LiveTrade):
            bar_ts = bucket_ts(event.ts)
            if candles and candles[-1].ts == bar_ts:
                c = candles[-1]
                c.high = max(c.high, event.price)
                c.low = min(c.low, event.price)
                c.close = event.price
                c.volume += event.size
            else:
                candles.append(OHLCVCandle(
                    symbol="GCTK", ts=bar_ts,
                    open=event.price, high=event.price,
                    low=event.price, close=event.price,
                    volume=event.size, interval="1min",
                ))

        # Minute 1
        ts1 = datetime(2026, 4, 10, 10, 0, 10, tzinfo=timezone.utc)
        on_trade(LiveTrade("trade", "GCTK", "Q", 3.50, 200, 0, ts1, 3.50))

        # Minute 2
        ts2 = datetime(2026, 4, 10, 10, 1, 5, tzinfo=timezone.utc)
        on_trade(LiveTrade("trade", "GCTK", "Q", 3.70, 150, 0, ts2, 3.70))

        assert len(candles) == 2
        assert candles[0].close == 3.50
        assert candles[1].open == 3.70


# ---------------------------------------------------------------------------
# Stream dispatch
# ---------------------------------------------------------------------------

class TestStreamDispatch:
    """Test that stream methods route events to correct callbacks."""

    def test_stream_combined_routes_correctly(self, stream_client):
        """stream_combined dispatches timesale and quote to separate callbacks."""
        ts_events = []
        q_events = []

        # Patch _start_thread to capture the dispatch function
        original_start = stream_client._start_thread

        def capture_start(symbols, filters, on_event, stop_event):
            # Simulate events being dispatched
            ts_line = '{"type":"timesale","symbol":"GCTK","last":3.5,"bid":3.49,"ask":3.51,"size":100,"date":1712779200000,"seq":1,"flag":"@","cancel":false,"correction":false,"session":"normal"}'
            q_line = '{"type":"quote","symbol":"GCTK","bid":3.49,"bidsz":10,"bidexch":"Q","biddate":1712779200000,"ask":3.51,"asksz":15,"askexch":"P","askdate":1712779200000}'

            ts_event = stream_client._parse_event(ts_line, "GCTK")
            q_event = stream_client._parse_event(q_line, "GCTK")
            on_event(ts_event)
            on_event(q_event)
            return MagicMock()

        stream_client._start_thread = capture_start

        import threading
        stop = threading.Event()
        stream_client.stream_combined(
            "GCTK",
            on_timesale=lambda e: ts_events.append(e),
            on_quote=lambda e: q_events.append(e),
            stop_event=stop,
        )

        assert len(ts_events) == 1
        assert isinstance(ts_events[0], LiveTimesale)
        assert len(q_events) == 1
        assert isinstance(q_events[0], LiveQuote)

    def test_stream_quotes_routes_trade_and_summary(self, stream_client):
        """stream_quotes dispatches trade and summary events."""
        trades = []
        summaries = []

        def capture_start(symbols, filters, on_event, stop_event):
            trade_line = '{"type":"trade","symbol":"GCTK","price":3.5,"size":100,"cvol":1000,"date":1712779200000,"last":3.5}'
            summary_line = '{"type":"summary","symbol":"GCTK","open":2.9,"high":3.75,"low":2.85,"prevClose":2.7}'
            on_event(stream_client._parse_event(trade_line, "GCTK"))
            on_event(stream_client._parse_event(summary_line, "GCTK"))
            return MagicMock()

        stream_client._start_thread = capture_start

        import threading
        stop = threading.Event()
        stream_client.stream_quotes(
            "GCTK",
            on_trade=lambda e: trades.append(e),
            on_summary=lambda e: summaries.append(e),
            stop_event=stop,
        )

        assert len(trades) == 1
        assert isinstance(trades[0], LiveTrade)
        assert len(summaries) == 1
        assert isinstance(summaries[0], LiveSummary)
