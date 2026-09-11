"""
tests/test_ibkr.py

Unit tests for market_data/ibkr_rest.py and market_data/ibkr_stream.py.

All IB API calls are mocked. No network access or TWS/Gateway required.
"""

import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock
import math
import time

import pandas as pd
import pytest

# Stub ib_insync before any ibkr module is imported
_ib_mod = MagicMock()
sys.modules.setdefault("ib_insync", _ib_mod)

from market_data.tradier_rest import StockQuote, OHLCVCandle
from market_data.ibkr_rest import IBKRHistoricalClient, _nan_safe, _nan_int, _bars_to_candles
from market_data.ibkr_stream import (
    IBKRSharedStream, OrderBookLevel, OrderBookSnapshot, _ns, _ni,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_ib_ticker(**kwargs):
    """Build a minimal mock ib_insync.Ticker for quote tests."""
    t = MagicMock()
    t.last = kwargs.get("last", 5.25)
    t.close = kwargs.get("close", 4.50)
    t.volume = kwargs.get("volume", 500_000)
    t.avVolume = kwargs.get("avVolume", 100_000)
    t.open = kwargs.get("open", 4.60)
    t.high = kwargs.get("high", 5.50)
    t.low = kwargs.get("low", 4.40)
    t.bid = kwargs.get("bid", 5.24)
    t.ask = kwargs.get("ask", 5.26)
    t.bidSize = kwargs.get("bidSize", 200)
    t.askSize = kwargs.get("askSize", 300)
    t.lastSize = kwargs.get("lastSize", 100)
    t.time = kwargs.get("time", datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc))
    t.bidExchange = kwargs.get("bidExchange", "ARCA")
    t.askExchange = kwargs.get("askExchange", "NASDAQ")
    t.lastExchange = kwargs.get("lastExchange", "ISLAND")
    contract = MagicMock()
    contract.symbol = kwargs.get("symbol", "ACMR")
    contract.description = kwargs.get("description", "ACM Research")
    contract.exchange = kwargs.get("exchange", "SMART")
    t.contract = contract
    return t


def _make_bar_data(
    dt, open_, high, low, close, volume, average=0.0,
):
    """Build a minimal mock ib_insync.BarData."""
    bar = MagicMock()
    bar.date = dt
    bar.open = open_
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    bar.average = average
    return bar


def _make_historical_tick(dt, price, size, exchange="NASDAQ", special_conditions="T"):
    tick = MagicMock()
    tick.time = dt
    tick.price = price
    tick.size = size
    tick.exchange = exchange
    tick.specialConditions = special_conditions
    tick.tickAttribLast = MagicMock(unreported=False, pastLimit=False)
    return tick


def _make_historical_bidask(dt, bid, ask, bid_size=100, ask_size=100):
    tick = MagicMock()
    tick.time = dt
    tick.priceBid = bid
    tick.priceAsk = ask
    tick.sizeBid = bid_size
    tick.sizeAsk = ask_size
    tick.tickAttribBidAsk = MagicMock(bidPastLow=False, askPastHigh=False)
    return tick


@pytest.fixture
def ibkr_client():
    """IBKRHistoricalClient with IBKR enabled."""
    with patch("market_data.ibkr_rest.IBKR_ENABLED", True):
        c = IBKRHistoricalClient()
        c._disabled = False
    return c


# ===========================================================================
# IBKRHistoricalClient — REST-equivalent
# ===========================================================================

class TestIBKRGetQuote:
    """Snapshot quote via reqTickers."""

    @patch("market_data.ibkr_rest.get_ib")
    def test_single_quote_parses(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqTickers.return_value = [_make_ib_ticker()]

        q = ibkr_client.get_quote("ACMR")
        assert isinstance(q, StockQuote)
        assert q.symbol == "ACMR"
        assert q.last == 5.25
        assert q.volume == 500_000
        assert q.avg_volume == 100_000
        assert q.bid == 5.24
        assert q.ask == 5.26
        # change_pct = (5.25 - 4.50) / 4.50 * 100 = 16.67
        assert q.change_pct == pytest.approx(16.67, abs=0.01)
        assert q.change == pytest.approx(0.75, abs=0.001)

    @patch("market_data.ibkr_rest.get_ib")
    def test_empty_response_returns_stub(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.qualifyContracts.return_value = []
        mock_ib.reqTickers.return_value = []

        q = ibkr_client.get_quote("ACMR")
        assert q.symbol == "ACMR"
        assert q.last is None
        assert q.volume == 0

    @patch("market_data.ibkr_rest.get_ib")
    def test_connection_error_raises_unavailable(self, mock_get_ib, ibkr_client):
        """A failed fetch must NOT degrade to the same stub an unknown symbol
        yields — Stage 1 reads that stub as "symbol not found" and discards a
        valid ticker whenever the feed is down."""
        from market_data.tradier_rest import MarketDataUnavailable
        mock_get_ib.side_effect = ConnectionError("TWS not running")
        with pytest.raises(MarketDataUnavailable):
            ibkr_client.get_quote("ACMR")

    @patch("market_data.ibkr_rest.get_ib")
    def test_empty_error_message_still_reports_detail(self, mock_get_ib, ibkr_client):
        """ib_insync raises bare errors when the 32-client cap refuses a
        connection; the old handler logged an empty "failed: " line."""
        from market_data.tradier_rest import MarketDataUnavailable
        mock_get_ib.side_effect = ConnectionError("")
        with pytest.raises(MarketDataUnavailable) as excinfo:
            ibkr_client.get_quote("ACMR")
        assert "ConnectionError" in str(excinfo.value)
        assert "ACMR" in str(excinfo.value)

    @patch("market_data.ibkr_rest.get_ib")
    def test_multi_quote(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        t1 = _make_ib_ticker(symbol="ACMR", last=5.25)
        t2 = _make_ib_ticker(symbol="IONQ", last=12.50)
        mock_ib.qualifyContracts.return_value = [MagicMock(), MagicMock()]
        mock_ib.reqTickers.return_value = [t1, t2]

        quotes = ibkr_client.get_quotes(["ACMR", "IONQ"])
        assert len(quotes) == 2
        assert quotes[0].symbol == "ACMR"
        assert quotes[1].symbol == "IONQ"

    @patch("market_data.ibkr_rest.get_ib")
    def test_nan_fields_handled(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        ticker = _make_ib_ticker(last=float("nan"), close=float("nan"))
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqTickers.return_value = [ticker]

        q = ibkr_client.get_quote("ACMR")
        assert q.last is None
        assert q.prev_close is None
        assert q.change is None
        assert q.change_pct is None


class TestGetIbLoopRebinding:
    """get_ib must survive prompt_toolkit-style event-loop churn.

    PromptSession runs each prompt through a throwaway asyncio loop on the
    main thread and discards it afterwards, so the policy fabricates a fresh
    current loop while the cached connection's transport stays bound to the
    original.  ib_insync sync calls then pump the wrong loop: requests never
    hit the wire and every call — symbol lookups included — dies at the 30s
    RequestTimeout as a bare TimeoutError, turning unknown tickers into
    "market data unavailable" instead of an instant "symbol not found".
    """

    @staticmethod
    def _swap_local(rest, ib, loop, client_id=None):
        """Install fake thread-local connection state; return the old state."""
        old = (
            getattr(rest._ib_local, "ib", None),
            getattr(rest._ib_local, "loop", None),
            getattr(rest._ib_local, "client_id", None),
        )
        rest._ib_local.ib = ib
        rest._ib_local.loop = loop
        rest._ib_local.client_id = client_id
        return old

    def test_reuse_restores_recorded_loop_after_churn(self):
        import asyncio
        import market_data.ibkr_rest as rest

        fake_ib = MagicMock()
        fake_ib.isConnected.return_value = True
        loop = asyncio.new_event_loop()
        old = self._swap_local(rest, fake_ib, loop)
        try:
            async def _noop():
                pass
            asyncio.run(_noop())  # the churn PromptSession.prompt() causes

            assert rest.get_ib() is fake_ib
            # The thread's current loop must be the connection's loop again,
            # or the next sync call would pump a loop the transport isn't on.
            assert asyncio.get_event_loop_policy().get_event_loop() is loop
        finally:
            self._swap_local(rest, *old[:2], old[2])
            loop.close()
            asyncio.set_event_loop(asyncio.new_event_loop())

    def test_closed_recorded_loop_rebuilds_connection(self):
        import asyncio
        import market_data.ibkr_rest as rest

        fake_ib = MagicMock()
        fake_ib.isConnected.return_value = True
        dead_loop = asyncio.new_event_loop()
        dead_loop.close()
        old = self._swap_local(rest, fake_ib, dead_loop, client_id=999)
        try:
            with patch.object(rest, "IB") as mock_ib_cls:
                fresh = mock_ib_cls.return_value
                fresh.isConnected.return_value = True

                got = rest.get_ib()

            # The wedged connection can never pump messages again: it must be
            # torn down and a fresh one returned, not handed back.
            assert got is fresh
            fake_ib.disconnect.assert_called_once()
        finally:
            rest.release_ib()
            with rest._ib_lock:
                if 999 in rest._free_client_ids:
                    rest._free_client_ids.remove(999)
            self._swap_local(rest, *old[:2], old[2])
            asyncio.set_event_loop(asyncio.new_event_loop())


class TestIBKRHistorical:
    """Historical candle retrieval."""

    @patch("market_data.ibkr_rest.get_ib")
    def test_get_candles_1min(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        bars = [
            _make_bar_data(
                datetime(2025, 1, 15, 14, i, 0, tzinfo=timezone.utc),
                10.0 + i, 10.5 + i, 9.5 + i, 10.2 + i, 1000 * (i + 1),
                average=10.1 + i,
            )
            for i in range(5)
        ]
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqHistoricalData.return_value = bars

        df = ibkr_client.get_candles_1min("ACMR", n=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "ts" in df.columns
        assert "open" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df["interval"].iloc[0] == "1min"

    @patch("market_data.ibkr_rest.get_ib")
    def test_get_candles_daily(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        from datetime import date
        bars = [
            _make_bar_data(
                date(2025, 1, d), 10.0, 11.0, 9.0, 10.5, 100_000,
            )
            for d in range(13, 18)
        ]
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqHistoricalData.return_value = bars

        df = ibkr_client.get_candles_daily("ACMR", days_back=30)
        assert len(df) == 5
        assert df["interval"].iloc[0] == "1day"

    @patch("market_data.ibkr_rest.get_ib")
    def test_get_candles_daily_accepts_include_current_day(
        self, mock_get_ib, ibkr_client,
    ):
        """The provider is swapped behind a common interface, so this must
        take the same kwargs as the Tradier client. It previously did not:
        pipeline._fetch_daily passes include_current_day=False, the TypeError
        was swallowed by its except Exception, and every IBKR-provider run
        silently lost its daily frame."""
        from datetime import date
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        bars = [
            _make_bar_data(
                date(2025, 1, d), 10.0, 11.0, 9.0, 10.5, 100_000,
            )
            for d in range(13, 18)
        ]
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqHistoricalData.return_value = bars

        # The exact call pipeline.py:374 makes — must not raise.
        df = ibkr_client.get_candles_daily(
            "ACMR", days_back=360, include_current_day=False,
        )
        assert not df.empty
        assert df["interval"].iloc[0] == "1day"
        # Excluding the current day must actually drop today's partial bar,
        # matching Tradier's semantics rather than just ignoring the flag.
        from config import MarketClock
        from market_data.ibkr_rest import _utc_ts
        assert (df["ts"] < _utc_ts(MarketClock.today())).all()

    @patch("market_data.ibkr_rest.get_ib")
    def test_get_candles_10sec(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        bars = [
            _make_bar_data(
                datetime(2025, 1, 15, 14, 0, s * 10, tzinfo=timezone.utc),
                5.0, 5.1, 4.9, 5.05, 500,
            )
            for s in range(6)
        ]
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqHistoricalData.return_value = bars

        df = ibkr_client.get_candles_10sec("ACMR", n=6)
        assert len(df) == 6
        assert df["interval"].iloc[0] == "10sec"

    @patch("market_data.ibkr_rest.get_ib")
    def test_get_candles_10sec_window_uses_native_bar_size(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        start = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 15, 14, 1, 0, tzinfo=timezone.utc)
        bars = [
            _make_bar_data(start, 5.0, 5.1, 4.9, 5.05, 500),
            _make_bar_data(start.replace(second=10), 5.05, 5.2, 5.0, 5.15, 700),
        ]
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqHistoricalData.return_value = bars

        df = ibkr_client.get_candles_10sec_window("ACMR", start, end)

        assert len(df) == 2
        assert df["interval"].iloc[0] == "10sec"
        kwargs = mock_ib.reqHistoricalData.call_args.kwargs
        assert kwargs["barSizeSetting"] == "10 secs"
        assert kwargs["durationStr"] == "60 S"
        assert kwargs["endDateTime"] == "20250115 14:01:00 UTC"

    @patch("market_data.ibkr_rest.get_ib")
    def test_get_timesales_tick_uses_ibkr_historical_trade_ticks(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        start = datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 15, 14, 1, 0, tzinfo=timezone.utc)
        mock_ib.qualifyContracts.return_value = [MagicMock()]

        def fake_req_historical_ticks(*args, **kwargs):
            if kwargs["whatToShow"] == "BID_ASK":
                return [
                    _make_historical_bidask(start, 5.04, 5.06),
                    _make_historical_bidask(start.replace(second=5), 5.14, 5.16),
                ]
            return [
                _make_historical_tick(start, 5.05, 100, exchange="ARCA", special_conditions="T"),
                _make_historical_tick(start.replace(second=10), 5.15, 200, exchange="NASDAQ", special_conditions="I"),
            ]

        mock_ib.reqHistoricalTicks.side_effect = fake_req_historical_ticks

        rows = ibkr_client.get_timesales_tick("ACMR", start, end, session_filter="all")

        assert rows == [
            {
                "ts": start,
                "price": 5.05,
                "size": 100,
                "bid": 5.04,
                "ask": 5.06,
                "exchange": "ARCA",
                "flag": "T",
                "cancel": False,
                "correction": False,
                "session": "ibkr",
            },
            {
                "ts": start.replace(second=10),
                "price": 5.15,
                "size": 200,
                "bid": 5.14,
                "ask": 5.16,
                "exchange": "NASDAQ",
                "flag": "I",
                "cancel": False,
                "correction": False,
                "session": "ibkr",
            },
        ]
        # TRADES goes out FIRST — the tape is the result; BID_ASK only supplies
        # the quote context stitched onto each print. With quotes first, a
        # starved pacing budget spent itself on the optional pass and skipped
        # the essential one, returning zero rows.
        passes = [c.kwargs["whatToShow"] for c in mock_ib.reqHistoricalTicks.call_args_list]
        assert passes[0] == "TRADES"
        assert "BID_ASK" in passes
        kwargs = mock_ib.reqHistoricalTicks.call_args_list[0].kwargs
        assert kwargs["numberOfTicks"] == 1000
        assert kwargs["useRth"] is False
        assert kwargs["ignoreSize"] is False

    @patch("market_data.ibkr_rest.get_ib")
    def test_empty_bars_return_empty_df(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqHistoricalData.return_value = []

        df = ibkr_client.get_candles_1min("ACMR")
        assert df.empty

    @patch("market_data.ibkr_rest.get_ib")
    def test_connection_error_returns_empty_df(self, mock_get_ib, ibkr_client):
        mock_get_ib.side_effect = ConnectionError("TWS not running")
        df = ibkr_client.get_candles_1min("ACMR")
        assert df.empty


class TestIBKRDisabled:
    """When IBKR_ENABLED is false."""

    def test_disabled_returns_empty_quotes(self):
        with patch("market_data.ibkr_rest.IBKR_ENABLED", False):
            c = IBKRHistoricalClient()
        assert c.get_quotes(["ACMR"]) == []

    def test_disabled_returns_empty_df(self):
        with patch("market_data.ibkr_rest.IBKR_ENABLED", False):
            c = IBKRHistoricalClient()
        assert c.get_candles_1min("ACMR").empty


class TestScanTopGainers:
    """IBKR market scanner — the third ticker-discovery feed."""

    @staticmethod
    def _scan_row(symbol, rank, primary_exchange="NASDAQ"):
        """Build a minimal ib_insync.ScanData stand-in."""
        return SimpleNamespace(
            rank=rank,
            contractDetails=SimpleNamespace(
                contract=SimpleNamespace(
                    symbol=symbol, primaryExchange=primary_exchange,
                ),
            ),
        )

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_rows_parse_in_rank_order(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = [
            self._scan_row("IONQ", 2),
            self._scan_row("ACMR", 0),
            self._scan_row("KIDZ", 1),
        ]

        rows = ibkr_client.scan_top_gainers()
        assert [r.symbol for r in rows] == ["ACMR", "KIDZ", "IONQ"]
        assert [r.rank for r in rows] == [0, 1, 2]
        assert rows[0].primary_exchange == "NASDAQ"

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_duplicate_symbol_keeps_best_rank(self, mock_get_ib, ibkr_client):
        """The same name can list on several exchanges — one entry, best rank."""
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = [
            self._scan_row("ACMR", 4, primary_exchange="ARCA"),
            self._scan_row("ACMR", 1, primary_exchange="NASDAQ"),
            self._scan_row("IONQ", 2),
        ]

        rows = ibkr_client.scan_top_gainers()
        assert [(r.symbol, r.rank) for r in rows] == [("ACMR", 1), ("IONQ", 2)]
        assert rows[0].primary_exchange == "NASDAQ"

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_subscription_carries_config_filters(self, mock_get_ib, ibkr_client):
        from config import (
            PRICE_MIN, PRICE_MAX,
            SCANNER_IBKR_ROWS, SCANNER_IBKR_INSTRUMENT, SCANNER_IBKR_LOCATION,
            SCANNER_IBKR_SCAN_CODE_RTH, SCANNER_IBKR_MIN_CHG, SCANNER_IBKR_MIN_VOLUME,
            MarketClock,
        )
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = []

        with patch("config.SCANNER_IBKR_SCAN_CODE_PIN", ""), \
             patch.object(MarketClock, "is_rth", return_value=True):
            ibkr_client.scan_top_gainers()
        (sub, options, filters), _kwargs = mock_ib.reqScannerData.call_args
        assert sub.numberOfRows == SCANNER_IBKR_ROWS
        assert sub.instrument == SCANNER_IBKR_INSTRUMENT
        assert sub.locationCode == SCANNER_IBKR_LOCATION
        assert sub.scanCode == SCANNER_IBKR_SCAN_CODE_RTH
        assert sub.abovePrice == PRICE_MIN
        assert sub.belowPrice == PRICE_MAX
        assert sub.aboveVolume == SCANNER_IBKR_MIN_VOLUME
        assert options == []
        assert filters == [("changePercAbove", f"{SCANNER_IBKR_MIN_CHG:g}")]

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_extended_hours_uses_ah_scan_and_filter(self, mock_get_ib, ibkr_client):
        """After the close the regular scan replays the 16:00 leaderboard, so
        the poll must switch to the AH scan code AND the AH change filter —
        the regular tag on the AH scan still filters regular-session change."""
        from config import SCANNER_IBKR_SCAN_CODE_EXT, SCANNER_IBKR_MIN_CHG, MarketClock
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = []

        with patch("config.SCANNER_IBKR_SCAN_CODE_PIN", ""), \
             patch.object(MarketClock, "is_rth", return_value=False), \
             patch.object(MarketClock, "is_premarket", return_value=False):
            ibkr_client.scan_top_gainers()
        (sub, _options, filters), _kwargs = mock_ib.reqScannerData.call_args
        assert sub.scanCode == SCANNER_IBKR_SCAN_CODE_EXT
        assert filters == [("afterHoursChangePercAbove", f"{SCANNER_IBKR_MIN_CHG:g}")]

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_premarket_uses_pre_open_scan_and_filter(self, mock_get_ib, ibkr_client):
        """Premarket is its own session, not "not RTH". Measured live
        (2026-08-05): the AH scan dies when the 04:00 session opens — 0 rows
        even unfiltered — while TOP_PERC_GAIN ranks the live premarket tape
        vs the prior close. So premarket must NOT take the after-hours
        branch, and must filter on regular change, not AH change."""
        from config import (
            SCANNER_IBKR_SCAN_CODE_PRE, SCANNER_IBKR_SCAN_CODE_EXT,
            SCANNER_IBKR_MIN_CHG, MarketClock,
        )
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = []

        with patch("config.SCANNER_IBKR_SCAN_CODE_PIN", ""), \
             patch.object(MarketClock, "is_rth", return_value=False), \
             patch.object(MarketClock, "is_premarket", return_value=True):
            ibkr_client.scan_top_gainers()
        (sub, _options, filters), _kwargs = mock_ib.reqScannerData.call_args
        assert sub.scanCode == SCANNER_IBKR_SCAN_CODE_PRE
        assert sub.scanCode != SCANNER_IBKR_SCAN_CODE_EXT
        assert filters == [("changePercAbove", f"{SCANNER_IBKR_MIN_CHG:g}")]

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_pinned_scan_code_overrides_session(self, mock_get_ib, ibkr_client):
        """An explicit SCANNER_IBKR_SCAN_CODE env pin wins in every session,
        and carries the filter tag matching the pinned code's session."""
        from config import MarketClock
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = []

        with patch("config.SCANNER_IBKR_SCAN_CODE_PIN", "TOP_AFTER_HOURS_PERC_GAIN"), \
             patch.object(MarketClock, "is_rth", return_value=True):
            ibkr_client.scan_top_gainers()
        (sub, _options, filters), _kwargs = mock_ib.reqScannerData.call_args
        assert sub.scanCode == "TOP_AFTER_HOURS_PERC_GAIN"
        assert filters[0][0] == "afterHoursChangePercAbove"

    def test_scanner_floors_track_the_criteria_levels(self):
        """The change floor IS the pre-evaluation change gate (the quote
        prefilter no longer rejects on % change), so it sits at the full
        criteria level — every scan hit gets a full evaluation. The volume
        floor stays at half so a name is found while volume is building."""
        from config import (
            CHANGE_PCT_MIN, VOLUME_MIN,
            SCANNER_IBKR_MIN_CHG, SCANNER_IBKR_MIN_VOLUME,
        )
        assert SCANNER_IBKR_MIN_CHG == CHANGE_PCT_MIN
        assert SCANNER_IBKR_MIN_VOLUME == VOLUME_MIN // 2
        assert SCANNER_IBKR_MIN_VOLUME < VOLUME_MIN

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_change_filter_formats_without_trailing_zero(self, mock_get_ib, ibkr_client):
        """CHANGE_PCT_MIN is a float, so the derived threshold must not reach
        IBKR as "5.0" when "5" is what the filter expects."""
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = []

        from config import MarketClock
        with patch("market_data.ibkr_rest.SCANNER_IBKR_MIN_CHG", 5.0), \
             patch("config.SCANNER_IBKR_SCAN_CODE_PIN", ""), \
             patch.object(MarketClock, "is_rth", return_value=True):
            ibkr_client.scan_top_gainers()
        (_sub, _options, filters), _kwargs = mock_ib.reqScannerData.call_args
        assert filters == [("changePercAbove", "5")]

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_empty_result_is_quiet(self, mock_get_ib, ibkr_client):
        """Nothing clearing the filters is a valid result, not an outage."""
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = []

        assert ibkr_client.scan_top_gainers() == []

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_scan_failure_raises_unavailable(self, mock_get_ib, ibkr_client):
        """Gateway down / missing scanner permission must not look like
        "no gainers today" — the caller warns and retries instead."""
        from market_data.tradier_rest import MarketDataUnavailable
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.side_effect = TimeoutError("")

        with pytest.raises(MarketDataUnavailable) as excinfo:
            ibkr_client.scan_top_gainers()
        assert "TimeoutError" in str(excinfo.value)

    @patch("market_data.ibkr_rest.get_ib")
    def test_disabled_client_does_not_connect(self, mock_get_ib):
        with patch("market_data.ibkr_rest.IBKR_ENABLED", False):
            c = IBKRHistoricalClient()
        assert c.scan_top_gainers() == []
        mock_get_ib.assert_not_called()

    def test_snapshot_starts_empty(self):
        """The `top` display reads these before the first poll lands."""
        with patch("market_data.ibkr_rest.IBKR_ENABLED", True):
            c = IBKRHistoricalClient()
        assert c.latest_scan == []
        assert c.latest_scan_ts == 0.0

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_successful_scan_publishes_snapshot(self, mock_get_ib, ibkr_client):
        """The scan result is kept so a live view never issues its own scan —
        that would open a second IB connection against the client cap."""
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = [
            self._scan_row("IONQ", 1), self._scan_row("ACMR", 0),
        ]

        rows = ibkr_client.scan_top_gainers()
        assert ibkr_client.latest_scan == rows
        assert [r.symbol for r in ibkr_client.latest_scan] == ["ACMR", "IONQ"]
        assert ibkr_client.latest_scan_ts > 0.0

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_empty_scan_publishes_empty_snapshot(self, mock_get_ib, ibkr_client):
        """A quiet market must clear the view, not leave yesterday's names up."""
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = [self._scan_row("ACMR", 0)]
        ibkr_client.scan_top_gainers()

        mock_ib.reqScannerData.return_value = []
        ibkr_client.scan_top_gainers()
        assert ibkr_client.latest_scan == []

    @patch("market_data.ibkr_rest.TagValue", new=lambda tag, value: (tag, value))
    @patch("market_data.ibkr_rest.ScannerSubscription", new=lambda **kw: SimpleNamespace(**kw))
    @patch("market_data.ibkr_rest.get_ib")
    def test_failed_scan_leaves_last_good_snapshot(self, mock_get_ib, ibkr_client):
        """A Gateway outage must not blank the view — the snapshot's age is
        what tells the reader it has gone stale."""
        from market_data.tradier_rest import MarketDataUnavailable
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqScannerData.return_value = [self._scan_row("ACMR", 0)]
        ibkr_client.scan_top_gainers()
        ts_before = ibkr_client.latest_scan_ts

        mock_ib.reqScannerData.side_effect = TimeoutError("")
        with pytest.raises(MarketDataUnavailable):
            ibkr_client.scan_top_gainers()
        assert [r.symbol for r in ibkr_client.latest_scan] == ["ACMR"]
        assert ibkr_client.latest_scan_ts == ts_before


class TestIBKRSearchSymbol:
    """Symbol search via reqMatchingSymbols."""

    @patch("market_data.ibkr_rest.get_ib")
    def test_search_returns_first_stock(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        desc = MagicMock()
        desc.contract = MagicMock()
        desc.contract.secType = "STK"
        desc.contract.symbol = "ACMR"
        mock_ib.reqMatchingSymbols.return_value = [desc]

        result = ibkr_client.search_symbol("acm")
        assert result == "ACMR"

    @patch("market_data.ibkr_rest.get_ib")
    def test_search_no_results(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.reqMatchingSymbols.return_value = []

        result = ibkr_client.search_symbol("zzzzz")
        assert result is None


class TestQuoteDf:
    """DataFrame output for quotes."""

    @patch("market_data.ibkr_rest.get_ib")
    def test_get_quote_df(self, mock_get_ib, ibkr_client):
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqTickers.return_value = [_make_ib_ticker()]

        df = ibkr_client.get_quote_df(["ACMR"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "symbol" in df.columns


# ===========================================================================
# Order book / NaN helpers
# ===========================================================================

class TestIBKROrderBook:
    """Order book dataclass tests."""

    def test_order_book_snapshot_creation(self):
        now = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        bid = OrderBookLevel(
            symbol="ACMR", side="bid", position=0,
            price=5.24, size=200, market_maker="NSDQ", ts=now,
        )
        ask = OrderBookLevel(
            symbol="ACMR", side="ask", position=0,
            price=5.26, size=300, market_maker="ARCA", ts=now,
        )
        snap = OrderBookSnapshot(
            symbol="ACMR", bids=[bid], asks=[ask], ts=now,
        )
        assert snap.symbol == "ACMR"
        assert len(snap.bids) == 1
        assert len(snap.asks) == 1
        assert snap.bids[0].price == 5.24
        assert snap.asks[0].market_maker == "ARCA"

    def test_order_book_level_fields(self):
        now = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        lvl = OrderBookLevel(
            symbol="ACMR", side="bid", position=2,
            price=5.20, size=500, market_maker="GSCO", ts=now,
        )
        assert lvl.side == "bid"
        assert lvl.position == 2
        assert lvl.market_maker == "GSCO"


class TestIBKRNanHelpers:
    """NaN helper functions."""

    def test_ns_nan(self):
        assert _ns(float("nan")) == 0.0
        assert _ns(None) == 0.0
        assert _ns(5.25) == 5.25

    def test_ni_nan(self):
        assert _ni(float("nan")) == 0
        assert _ni(None) == 0
        assert _ni(100.0) == 100


class TestIBKRRestHelpers:
    """REST helper functions."""

    def test_nan_safe(self):
        assert _nan_safe(float("nan")) is None
        assert _nan_safe(None) is None
        assert _nan_safe(5.25) == 5.25

    def test_nan_int(self):
        assert _nan_int(float("nan")) == 0
        assert _nan_int(None) == 0
        assert _nan_int(100.0) == 100

    def test_bars_to_candles(self):
        bars = [
            _make_bar_data(
                datetime(2025, 1, 15, 14, 0, 0, tzinfo=timezone.utc),
                10.0, 10.5, 9.5, 10.2, 1000, average=10.1,
            ),
        ]
        candles = _bars_to_candles(bars, "ACMR", "1min")
        assert len(candles) == 1
        assert isinstance(candles[0], OHLCVCandle)
        assert candles[0].symbol == "ACMR"
        assert candles[0].interval == "1min"
        assert candles[0].vwap == 10.1

    def test_bars_to_candles_naive_datetime(self):
        # Naive datetime should get UTC timezone
        bars = [
            _make_bar_data(
                datetime(2025, 1, 15, 14, 0, 0),
                10.0, 10.5, 9.5, 10.2, 1000,
            ),
        ]
        candles = _bars_to_candles(bars, "ACMR", "1min")
        assert candles[0].ts.tzinfo == timezone.utc

    def test_bars_to_candles_date_object(self):
        from datetime import date
        bars = [
            _make_bar_data(date(2025, 1, 15), 10.0, 10.5, 9.5, 10.2, 1000),
        ]
        candles = _bars_to_candles(bars, "ACMR", "1day")
        assert candles[0].ts.tzinfo == timezone.utc
        assert candles[0].interval == "1day"


# ===========================================================================
# IBKRSharedStream — Multiplexed streaming
# ===========================================================================

class TestIBKRSharedStream:
    """subscribe / unsubscribe / stop + callback dispatch."""

    def _make_stream(self):
        """Build an IBKRSharedStream with the background thread patched out."""
        with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
             patch.object(IBKRSharedStream, "_run"):
            stream = IBKRSharedStream()
        return stream

    def test_subscribe_registers_callbacks(self):
        stream = self._make_stream()
        cb_ts = MagicMock()
        cb_q = MagicMock()
        stream.subscribe("ACMR", on_timesale=cb_ts, on_quote=cb_q, sub_id="scorer")
        assert "ACMR" in stream._subs
        assert "scorer" in stream._subs["ACMR"]

    def test_subscribe_enqueues_add_for_new_symbol(self):
        stream = self._make_stream()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1")
        cmd = stream._cmd_q.get_nowait()
        assert cmd == ("add", "ACMR")

    def test_second_subscribe_same_symbol_no_extra_add(self):
        stream = self._make_stream()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1")
        stream._cmd_q.get_nowait()  # consume the first "add"
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s2")
        assert stream._cmd_q.empty()  # no second "add"
        assert len(stream._subs["ACMR"]) == 2

    def test_unsubscribe_removes_callback(self):
        stream = self._make_stream()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1")
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s2")
        stream._cmd_q.get_nowait()
        stream.unsubscribe("ACMR", sub_id="s1")
        assert "s1" not in stream._subs["ACMR"]
        assert "s2" in stream._subs["ACMR"]
        assert stream._cmd_q.empty()  # still one subscriber → no remove cmd

    def test_unsubscribe_last_enqueues_remove(self):
        stream = self._make_stream()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1")
        stream._cmd_q.get_nowait()
        stream.unsubscribe("ACMR", sub_id="s1")
        cmd = stream._cmd_q.get_nowait()
        assert cmd == ("remove", "ACMR")
        assert "ACMR" not in stream._subs

    def test_unsubscribe_nonexistent_symbol_noop(self):
        stream = self._make_stream()
        stream.unsubscribe("NOPE", sub_id="s1")  # should not raise

    def test_stop_clears_subs_and_enqueues_stop(self):
        stream = self._make_stream()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1")
        stream._cmd_q.get_nowait()
        stream.stop()
        assert stream._stopped is True
        assert len(stream._subs) == 0
        cmd = stream._cmd_q.get_nowait()
        assert cmd == ("stop",)


class TestIBKRSharedStreamInternals:
    """_start_symbol / _stop_symbol called from the stream thread."""

    def _make_stream(self):
        with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
             patch.object(IBKRSharedStream, "_run"):
            stream = IBKRSharedStream()
        return stream

    @patch("market_data.ibkr_stream.get_ib")
    def test_start_symbol_creates_subscriptions(self, mock_get_ib):
        stream = self._make_stream()
        mock_ib = MagicMock()
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        ts_ticker = MagicMock()
        q_ticker = MagicMock()
        mock_ib.reqTickByTickData.return_value = ts_ticker
        mock_ib.reqMktData.return_value = q_ticker

        stream._start_symbol(mock_ib, "ACMR")

        assert "ACMR" in stream._contracts
        assert "ACMR" in stream._ts_tickers
        assert "ACMR" in stream._q_tickers
        mock_ib.reqTickByTickData.assert_called_once()
        mock_ib.reqMktData.assert_called_once()

    @patch("market_data.ibkr_stream.get_ib")
    def test_stop_symbol_cancels_and_cleans(self, mock_get_ib):
        stream = self._make_stream()
        mock_ib = MagicMock()
        contract = MagicMock()
        stream._contracts["ACMR"] = contract
        stream._ts_tickers["ACMR"] = MagicMock()
        stream._q_tickers["ACMR"] = MagicMock()
        stream._prev_bid["ACMR"] = 5.0
        stream._prev_ask["ACMR"] = 5.1
        stream._seq["ACMR"] = 0

        stream._stop_symbol(mock_ib, "ACMR")

        assert "ACMR" not in stream._contracts
        assert "ACMR" not in stream._ts_tickers
        mock_ib.cancelTickByTickData.assert_called_once_with(contract, "AllLast")
        mock_ib.cancelMktData.assert_called_once_with(contract)

    @patch("market_data.ibkr_stream.get_ib")
    def test_stop_all_cleans_everything(self, mock_get_ib):
        stream = self._make_stream()
        mock_ib = MagicMock()
        for sym in ("ACMR", "IONQ"):
            stream._contracts[sym] = MagicMock()
            stream._ts_tickers[sym] = MagicMock()
            stream._q_tickers[sym] = MagicMock()
            stream._prev_bid[sym] = 0.0
            stream._prev_ask[sym] = 0.0
            stream._seq[sym] = 0

        stream._stop_all(mock_ib)
        assert len(stream._contracts) == 0
        assert mock_ib.cancelTickByTickData.call_count == 2


class TestIBKRSharedStreamDepth:
    """L2 market-depth subscriptions: on_book wiring, slot cap + pin priority, DOM aggregation."""

    @pytest.fixture(autouse=True)
    def _no_demand_gate(self):
        """These tests exercise slot mechanics, not relevance — zero the
        floor so every symbol qualifies (gate coverage: TestSlotPriority)."""
        with patch("market_data.ibkr_stream.IBKR_DEPTH_IDLE_DOLLAR_RATE", 0.0):
            yield

    def _make_stream(self):
        with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
             patch.object(IBKRSharedStream, "_run"):
            stream = IBKRSharedStream()
        return stream

    def _mock_ib(self):
        mock_ib = MagicMock()
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        return mock_ib

    def test_subscribe_with_on_book_registers_third_callback(self):
        stream = self._make_stream()
        cb_book = MagicMock()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1", on_book=cb_book)
        assert stream._subs["ACMR"]["s1"][2] is cb_book

    def test_second_subscribe_with_on_book_enqueues_add_depth(self):
        stream = self._make_stream()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1")
        stream._cmd_q.get_nowait()  # consume "add"
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s2", on_book=MagicMock())
        assert stream._cmd_q.get_nowait() == ("add_depth", "ACMR")

    def test_start_symbol_requests_depth_when_wanted(self):
        stream = self._make_stream()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1", on_book=MagicMock())
        mock_ib = self._mock_ib()
        stream._start_symbol(mock_ib, "ACMR")
        mock_ib.reqMktDepth.assert_called_once()
        assert "ACMR" in stream._depth_tickers

    def test_start_symbol_skips_depth_when_not_wanted(self):
        stream = self._make_stream()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1")
        mock_ib = self._mock_ib()
        stream._start_symbol(mock_ib, "ACMR")
        mock_ib.reqMktDepth.assert_not_called()

    def test_depth_full_house_declines_equal_pin_challenger(self):
        """No equal-pin turnover: resubscribe churn corrupts books, so an
        unpinned newcomer waits instead of evicting an unpinned holder."""
        stream = self._make_stream()
        mock_ib = self._mock_ib()
        with patch("market_data.ibkr_stream.IBKR_L2_MAX_DEPTH_SUBS", 2):
            for sym in ("AAA", "BBB", "CCC"):
                stream._contracts[sym] = MagicMock()
                stream._start_depth(mock_ib, sym)
        assert set(stream._depth_tickers) == {"AAA", "BBB"}  # first-come keeps
        assert "CCC" not in stream._depth_tickers
        mock_ib.cancelMktDepth.assert_not_called()

    def test_depth_higher_pin_evicts_lowest_pinned_holder(self):
        stream = self._make_stream()
        mock_ib = self._mock_ib()
        with patch("market_data.ibkr_stream.IBKR_L2_MAX_DEPTH_SUBS", 2):
            for sym in ("AAA", "BBB"):
                stream._contracts[sym] = MagicMock()
                stream._start_depth(mock_ib, sym)
            stream._pins["BBB"] = 2          # confirming — protected
            stream._contracts["CCC"] = MagicMock()
            stream._pins["CCC"] = 1          # armed challenger
            stream._start_depth(mock_ib, "CCC")
        assert set(stream._depth_tickers) == {"BBB", "CCC"}  # unpinned AAA fell
        mock_ib.cancelMktDepth.assert_called_once()

    def test_depth_challenger_never_evicts_equal_or_higher_pin(self):
        stream = self._make_stream()
        mock_ib = self._mock_ib()
        with patch("market_data.ibkr_stream.IBKR_L2_MAX_DEPTH_SUBS", 2):
            for sym in ("AAA", "BBB"):
                stream._contracts[sym] = MagicMock()
                stream._start_depth(mock_ib, sym)
            stream._pins.update({"AAA": 1, "BBB": 3})
            stream._contracts["CCC"] = MagicMock()
            stream._pins["CCC"] = 1          # equal to AAA, below BBB
            stream._start_depth(mock_ib, "CCC")
        assert set(stream._depth_tickers) == {"AAA", "BBB"}
        mock_ib.cancelMktDepth.assert_not_called()

    def test_stop_symbol_cancels_depth(self):
        stream = self._make_stream()
        mock_ib = self._mock_ib()
        stream._contracts["ACMR"] = MagicMock()
        stream._start_depth(mock_ib, "ACMR")
        stream._stop_symbol(mock_ib, "ACMR")
        assert "ACMR" not in stream._depth_tickers
        assert "ACMR" not in stream._depth_lru
        mock_ib.cancelMktDepth.assert_called_once()

    def test_aggregate_dom_merges_same_price_mm_rows(self):
        from market_data.ibkr_stream import _aggregate_dom

        rows = [
            SimpleNamespace(price=5.00, size=200),
            SimpleNamespace(price=5.00, size=300),   # second MM at same price
            SimpleNamespace(price=4.99, size=100),
            SimpleNamespace(price=0.0, size=50),      # invalid → dropped
            SimpleNamespace(price=4.98, size=0),      # zero size → dropped
        ]
        bids = _aggregate_dom(rows, descending=True)
        assert bids == [(5.00, 500), (4.99, 100)]
        asks = _aggregate_dom(rows, descending=False)
        assert asks == [(4.99, 100), (5.00, 500)]

    def test_dispatch_depth_builds_snapshot_and_dedupes(self):
        stream = self._make_stream()
        cb_book = MagicMock()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1", on_book=cb_book)
        ticker = SimpleNamespace(
            domBids=[SimpleNamespace(price=5.00, size=300)],
            domAsks=[SimpleNamespace(price=5.01, size=400)],
        )
        stream._depth_tickers["ACMR"] = ticker
        stream._dispatch_depth()
        assert cb_book.call_count == 1
        snapshot = cb_book.call_args[0][0]
        assert snapshot.symbol == "ACMR"
        assert [(l.price, l.size) for l in snapshot.bids] == [(5.00, 300)]
        assert [(l.price, l.size) for l in snapshot.asks] == [(5.01, 400)]
        # Unchanged book → no duplicate dispatch
        stream._dispatch_depth()
        assert cb_book.call_count == 1
        # Changed book → dispatches again
        ticker.domBids = [SimpleNamespace(price=5.00, size=350)]
        stream._dispatch_depth()
        assert cb_book.call_count == 2

    def test_start_depth_installs_state_buffer_and_stop_removes_it(self):
        stream = self._make_stream()
        mock_ib = self._mock_ib()
        stream._contracts["ACMR"] = MagicMock()
        stream._start_depth(mock_ib, "ACMR")
        ticker = stream._depth_tickers["ACMR"]
        assert isinstance(ticker._momo_book_states, deque)
        stream._stop_depth(mock_ib, "ACMR")
        assert not hasattr(ticker, "_momo_book_states")

    def test_dispatch_depth_emits_every_buffered_intermediate_state(self):
        """The point of the buffer: two size changes captured between loop
        iterations must produce two snapshots, not one end-state sample —
        OFI measures the flow of changes, not the book's final shape."""
        stream = self._make_stream()
        cb_book = MagicMock()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1", on_book=cb_book)
        t0 = datetime(2026, 8, 3, 14, 30, 0, tzinfo=timezone.utc)
        buf = deque([
            (t0, [SimpleNamespace(price=5.00, size=300)],
             [SimpleNamespace(price=5.01, size=400)]),
            (t0, [SimpleNamespace(price=5.00, size=800)],
             [SimpleNamespace(price=5.01, size=400)]),
            # Same aggregate state again → deduped by signature
            (t0, [SimpleNamespace(price=5.00, size=800)],
             [SimpleNamespace(price=5.01, size=400)]),
        ])
        ticker = SimpleNamespace(_momo_book_states=buf, domBids=[], domAsks=[])
        stream._depth_tickers["ACMR"] = ticker
        stream._dispatch_depth()
        assert cb_book.call_count == 2
        first = cb_book.call_args_list[0][0][0]
        second = cb_book.call_args_list[1][0][0]
        assert [(l.price, l.size) for l in first.bids] == [(5.00, 300)]
        assert [(l.price, l.size) for l in second.bids] == [(5.00, 800)]
        assert first.ts == t0
        assert not buf  # drained
        # Empty buffer, no new updates → nothing further
        stream._dispatch_depth()
        assert cb_book.call_count == 2

    def test_wrapper_to_dispatch_preserves_intermediate_states(self):
        """End-to-end through the patched wrapper handler: two updates landing
        between dispatch iterations reach on_book as two snapshots (legacy
        sampling delivered only the final 900-share book)."""
        from market_data.ibkr_stream import _update_mkt_depth_l2

        stream = self._make_stream()
        cb_book = MagicMock()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1", on_book=cb_book)

        class _Ticker:
            def __init__(self):
                self.domBids, self.domAsks, self.domTicks = [], [], []

        ticker = _Ticker()
        ticker._momo_book_states = deque(maxlen=16)
        w = SimpleNamespace(reqId2Ticker={7: ticker}, lastTime=None, pendingTickers=set())
        _update_mkt_depth_l2(w, 7, 0, "NSDQ", 0, 1, 5.00, 300)
        _update_mkt_depth_l2(w, 7, 0, "NSDQ", 1, 1, 5.00, 900)
        stream._depth_tickers["ACMR"] = ticker
        stream._dispatch_depth()
        assert cb_book.call_count == 2
        sizes = [c[0][0].bids[0].size for c in cb_book.call_args_list]
        assert sizes == [300, 900]


# ===========================================================================
# Provider abstraction — market_data/provider.py
# ===========================================================================

class TestProvider:
    """Provider factory returns correct backend based on config flags."""

    @patch("market_data.provider.IBKR_ENABLED", False)
    @patch("market_data.provider.TRADIER_ENABLED", True)
    def test_historical_client_returns_tradier(self):
        from market_data.provider import get_historical_client
        client = get_historical_client()
        from market_data.tradier_rest import HistoricalClient
        assert isinstance(client, HistoricalClient)

    @patch("market_data.provider.IBKR_ENABLED", True)
    def test_historical_client_returns_ibkr(self):
        from market_data.provider import get_historical_client
        client = get_historical_client()
        assert isinstance(client, IBKRHistoricalClient)

    @patch("market_data.provider.IBKR_ENABLED", False)
    @patch("market_data.provider.TRADIER_ENABLED", True)
    @patch("market_data.provider.TRADIER_STREAM_ENABLED", True)
    def test_stream_enabled_tradier(self):
        from market_data.provider import stream_enabled
        assert stream_enabled() is True

    @patch("market_data.provider.IBKR_ENABLED", False)
    @patch("market_data.provider.TRADIER_ENABLED", False)
    @patch("market_data.provider.TRADIER_STREAM_ENABLED", False)
    def test_stream_disabled_no_provider(self):
        from market_data.provider import stream_enabled
        assert stream_enabled() is False

    @patch("market_data.provider.IBKR_ENABLED", True)
    def test_stream_enabled_ibkr(self):
        from market_data.provider import stream_enabled
        assert stream_enabled() is True

    @patch("market_data.provider.IBKR_ENABLED", False)
    @patch("market_data.provider.TRADIER_ENABLED", True)
    @patch("market_data.provider.TRADIER_STREAM_ENABLED", True)
    def test_shared_stream_returns_tradier(self):
        from market_data.provider import get_shared_stream
        from market_data.tradier_stream import SharedStreamClient
        stream = get_shared_stream()
        try:
            assert isinstance(stream, SharedStreamClient)
        finally:
            # Stop the real ws-mkt-stream thread; otherwise its manager
            # coroutine survives to interpreter shutdown and logs
            # "cannot schedule new futures after shutdown".
            stream.stop()

    @patch("market_data.provider.IBKR_ENABLED", True)
    def test_shared_stream_returns_ibkr(self):
        import market_data.provider as prov
        # Reset singleton between tests
        prov._ibkr_stream = None
        with patch.object(IBKRSharedStream, "_run"):
            stream = prov.get_shared_stream()
        assert isinstance(stream, IBKRSharedStream)
        prov._ibkr_stream = None  # clean up


# ===========================================================================
# Connection lifecycle â€” release on unwatch, disconnect on session stop
# ===========================================================================

class TestConnectionLifecycle:
    """The Gateway caps concurrent API connections at 32 and only frees a slot when
    the socket closes, so every connection must be closed explicitly."""

    def setup_method(self):
        import threading as _th
        import market_data.ibkr_rest as rest
        rest._ib_local = _th.local()
        rest._ib_clients.clear()
        rest._free_client_ids.clear()
        rest._failed_client_ids.clear()
        rest._next_client_offset = 0

    teardown_method = setup_method

    def _connected_ib(self):
        ib = MagicMock()
        ib.isConnected.return_value = True
        return ib

    def test_release_ib_disconnects_socket(self):
        import threading as _th
        import market_data.ibkr_rest as rest
        ib = self._connected_ib()
        rest._ib_local.ib = ib
        rest._ib_local.client_id = 10
        rest._ib_clients[_th.get_ident()] = ib

        rest.release_ib()

        ib.disconnect.assert_called_once()
        assert rest._ib_clients == {}
        assert getattr(rest._ib_local, "ib") is None

    def test_release_ib_returns_client_id_for_reuse(self):
        """Without id reuse, each pipeline worker burns a fresh id and we
        march into the 32-connection ceiling as the watchlist grows."""
        import market_data.ibkr_rest as rest
        rest._ib_local.ib = self._connected_ib()
        rest._ib_local.client_id = 14

        rest.release_ib()

        assert 14 in rest._free_client_ids

    def test_released_id_is_reused_by_next_connect(self):
        import market_data.ibkr_rest as rest
        rest._free_client_ids.append(17)
        made = {}

        def _fake_ib():
            ib = self._connected_ib()
            ib.connect.side_effect = lambda **kw: made.update(kw)
            return ib

        with patch.object(rest, "IB", _fake_ib), \
             patch.object(rest, "IBKR_CLIENT_ID", 10):
            rest._ib_local.ib = None
            rest._ib_local.client_id = None
            rest.get_ib()

        assert made["clientId"] == 17
        assert rest._free_client_ids == []

    def test_failed_connect_does_not_burn_client_id(self):
        """The socket never opened, so the id stays usable — just demoted."""
        import market_data.ibkr_rest as rest

        def _failing_ib():
            ib = MagicMock()
            ib.connect.side_effect = ConnectionRefusedError("refused")
            return ib

        with patch.object(rest, "IB", _failing_ib), \
             patch.object(rest, "IBKR_CLIENT_ID", 10):
            with pytest.raises(ConnectionRefusedError):
                rest.get_ib()

        assert 10 in rest._failed_client_ids
        # Not in the clean pool — see the poisoned-id test below.
        assert rest._free_client_ids == []

    def test_refused_id_is_not_retried_before_a_fresh_one(self):
        """A crashed run leaves an id reserved on the Gateway, which then
        refuses it forever. Pooling it with the clean ids would hand the same
        dead id to every subsequent caller and wedge all IBKR consumers."""
        import market_data.ibkr_rest as rest
        import threading as _th
        attempted: list[int] = []

        def _ib_refusing_10():
            ib = MagicMock()
            ib.isConnected.return_value = True

            def _connect(**kw):
                attempted.append(kw["clientId"])
                if kw["clientId"] == 10:
                    raise ConnectionRefusedError("client id already in use")

            ib.connect.side_effect = _connect
            return ib

        with patch.object(rest, "IB", _ib_refusing_10), \
             patch.object(rest, "IBKR_CLIENT_ID", 10), \
             patch.object(rest, "IBKR_MAX_CONNECTIONS", 10):
            with pytest.raises(ConnectionRefusedError):
                rest.get_ib()
            # Fresh thread-local state — as a second thread would see it.
            rest._ib_local = _th.local()
            rest.get_ib()

        assert attempted == [10, 11], "second caller must not re-try the dead id"

    def test_exhausted_pool_raises_instead_of_timing_out(self):
        """Past the concurrent-connection ceiling a connect can only fail, and
        it fails as an 8s timeout — so refuse to allocate rather than stall."""
        import market_data.ibkr_rest as rest
        from market_data.tradier_rest import MarketDataUnavailable

        with patch.object(rest, "IB", self._connected_ib), \
             patch.object(rest, "IBKR_CLIENT_ID", 10), \
             patch.object(rest, "IBKR_MAX_CONNECTIONS", 2):
            rest._next_client_offset = 2  # band 10-11 already handed out
            with pytest.raises(MarketDataUnavailable, match="pool exhausted"):
                rest.get_ib()

    def test_failed_ids_are_retried_once_the_band_is_exhausted(self):
        """Gateway genuinely down: every id fails, so cycling them is all
        that's left — the pool must not dead-end."""
        import market_data.ibkr_rest as rest
        made = {}

        def _fake_ib():
            ib = self._connected_ib()
            ib.connect.side_effect = lambda **kw: made.update(kw)
            return ib

        with patch.object(rest, "IB", _fake_ib), \
             patch.object(rest, "IBKR_CLIENT_ID", 10), \
             patch.object(rest, "IBKR_MAX_CONNECTIONS", 2):
            rest._next_client_offset = 2
            rest._failed_client_ids.extend([10, 11])
            rest.get_ib()

        assert made["clientId"] == 10

    def test_release_ib_noop_when_never_connected(self):
        import market_data.ibkr_rest as rest
        rest.release_ib()  # must not raise

    def test_release_ib_survives_disconnect_error(self):
        import market_data.ibkr_rest as rest
        ib = self._connected_ib()
        ib.disconnect.side_effect = RuntimeError("socket already gone")
        rest._ib_local.ib = ib
        rest._ib_local.client_id = 11

        rest.release_ib()  # must not propagate

        assert getattr(rest._ib_local, "ib") is None

    def test_disconnect_all_closes_every_registered_connection(self):
        import market_data.ibkr_rest as rest
        a, b, c = self._connected_ib(), self._connected_ib(), self._connected_ib()
        rest._ib_clients.update({1: a, 2: b, 3: c})

        rest.disconnect_all()

        for ib in (a, b, c):
            ib.disconnect.assert_called_once()
        assert rest._ib_clients == {}

    def test_disconnect_all_skips_already_closed(self):
        import market_data.ibkr_rest as rest
        dead = MagicMock()
        dead.isConnected.return_value = False
        rest._ib_clients[1] = dead

        rest.disconnect_all()

        dead.disconnect.assert_not_called()

    def test_disconnect_all_continues_past_failure(self):
        """One wedged socket must not strand the remaining connections."""
        import market_data.ibkr_rest as rest
        bad, good = self._connected_ib(), self._connected_ib()
        bad.disconnect.side_effect = RuntimeError("wedged")
        rest._ib_clients.update({1: bad, 2: good})

        rest.disconnect_all()

        good.disconnect.assert_called_once()
        assert rest._ib_clients == {}

    def test_disconnect_all_resets_allocator(self):
        import market_data.ibkr_rest as rest
        rest._next_client_offset = 7
        rest._free_client_ids.extend([12, 13])

        rest.disconnect_all()

        assert rest._next_client_offset == 0
        assert rest._free_client_ids == []


class TestProviderShutdown:
    """provider.shutdown_market_data() / release_thread_connection()."""

    def test_release_thread_connection_noop_for_tradier(self):
        import market_data.provider as prov
        with patch.object(prov, "IBKR_ENABLED", False), \
             patch("market_data.ibkr_rest.release_ib") as mock_release:
            prov.release_thread_connection()
        mock_release.assert_not_called()

    def test_release_thread_connection_releases_for_ibkr(self):
        import market_data.provider as prov
        with patch.object(prov, "IBKR_ENABLED", True), \
             patch("market_data.ibkr_rest.release_ib") as mock_release:
            prov.release_thread_connection()
        mock_release.assert_called_once()

    def test_shutdown_stops_stream_and_disconnects(self):
        import market_data.provider as prov
        stream = MagicMock()
        prov._ibkr_stream = stream
        with patch.object(prov, "IBKR_ENABLED", True), \
             patch("market_data.ibkr_rest.disconnect_all") as mock_disc:
            prov.shutdown_market_data()
        stream.stop.assert_called_once()
        mock_disc.assert_called_once()
        assert prov._ibkr_stream is None

    def test_shutdown_disconnects_even_if_stream_stop_fails(self):
        """A wedged stream thread must not leave sockets open."""
        import market_data.provider as prov
        stream = MagicMock()
        stream.stop.side_effect = RuntimeError("thread wedged")
        prov._ibkr_stream = stream
        with patch.object(prov, "IBKR_ENABLED", True), \
             patch("market_data.ibkr_rest.disconnect_all") as mock_disc:
            prov.shutdown_market_data()
        mock_disc.assert_called_once()

    def test_shutdown_is_idempotent(self):
        import market_data.provider as prov
        prov._ibkr_stream = None
        with patch.object(prov, "IBKR_ENABLED", True), \
             patch("market_data.ibkr_rest.disconnect_all"):
            prov.shutdown_market_data()
            prov.shutdown_market_data()  # must not raise


class TestSharedStreamThreadRelease:
    """The stream thread owns a connection for the whole session."""

    def _make_stream(self):
        with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
             patch.object(IBKRSharedStream, "_run"):
            return IBKRSharedStream()

    def test_run_releases_connection_when_loop_exits(self):
        stream = self._make_stream()
        with patch("market_data.ibkr_stream.get_ib"), \
             patch("market_data.ibkr_stream.release_ib") as mock_release, \
             patch.object(IBKRSharedStream, "_loop"):
            IBKRSharedStream._run(stream)
        mock_release.assert_called_once()

    def test_run_releases_connection_when_loop_raises(self):
        stream = self._make_stream()
        with patch("market_data.ibkr_stream.get_ib"), \
             patch("market_data.ibkr_stream.release_ib") as mock_release, \
             patch.object(IBKRSharedStream, "_loop", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                IBKRSharedStream._run(stream)
        mock_release.assert_called_once()

    def test_run_releases_when_initial_connect_fails(self):
        stream = self._make_stream()
        with patch("market_data.ibkr_stream.get_ib", side_effect=OSError("refused")), \
             patch("market_data.ibkr_stream.release_ib") as mock_release:
            IBKRSharedStream._run(stream)
        mock_release.assert_called_once()

    def test_stop_enqueues_before_setting_flag(self):
        """If _stopped is set first, the while-condition can win the race and
        _stop_all never runs, leaving subscriptions open on the connection."""
        stream = self._make_stream()
        stream.stop(wait=False)
        assert stream._cmd_q.get_nowait() == ("stop",)
        assert stream._stopped is True

    def test_loop_runs_stop_all_on_stop_command(self):
        stream = self._make_stream()
        mock_ib = MagicMock()
        stream._cmd_q.put(("stop",))
        with patch.object(stream, "_stop_all") as mock_stop_all:
            stream._loop(mock_ib)
        mock_stop_all.assert_called_once_with(mock_ib)



# ===========================================================================
# Fetch failure vs. unknown symbol â€” must stay distinguishable
# ===========================================================================

class TestQuoteFailureVsUnknownSymbol:
    """The core bug: an empty quote has two causes with opposite handling.

    Unknown symbol  -> permanent, bail.
    Fetch failed    -> transient, retry once the feed recovers.

    Collapsing both into an empty list made a Gateway hiccup look like a
    delisted ticker.
    """

    @patch("market_data.ibkr_rest.get_ib")
    def test_unqualified_symbol_returns_empty_not_raises(self, mock_get_ib, ibkr_client):
        """qualifyContracts resolving nothing is the real not-found signal."""
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.qualifyContracts.return_value = []

        assert ibkr_client.get_quotes(["NOPE"]) == []
        mock_ib.reqTickers.assert_not_called()

    @patch("market_data.ibkr_rest.get_ib")
    def test_unknown_symbol_does_not_request_tickers(self, mock_get_ib, ibkr_client):
        """reqTickers on an unqualified contract can hang until timeout."""
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.qualifyContracts.return_value = None  # ib_insync may return None

        assert ibkr_client.get_quotes(["NOPE"]) == []
        mock_ib.reqTickers.assert_not_called()

    @patch("market_data.ibkr_rest.get_ib")
    def test_req_tickers_failure_raises_unavailable(self, mock_get_ib, ibkr_client):
        from market_data.tradier_rest import MarketDataUnavailable
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.qualifyContracts.return_value = [MagicMock()]
        mock_ib.reqTickers.side_effect = TimeoutError("pacing stall")

        with pytest.raises(MarketDataUnavailable):
            ibkr_client.get_quotes(["ACMR"])

    @patch("market_data.ibkr_rest.get_ib")
    def test_qualify_failure_raises_unavailable(self, mock_get_ib, ibkr_client):
        from market_data.tradier_rest import MarketDataUnavailable
        mock_ib = MagicMock()
        mock_get_ib.return_value = mock_ib
        mock_ib.qualifyContracts.side_effect = ConnectionError("socket closed")

        with pytest.raises(MarketDataUnavailable):
            ibkr_client.get_quotes(["ACMR"])

    def test_disabled_client_returns_empty_without_raising(self):
        """IBKR switched off is not an outage â€” must stay silent."""
        with patch("market_data.ibkr_rest.IBKR_ENABLED", False):
            c = IBKRHistoricalClient()
        assert c.get_quotes(["ACMR"]) == []


class TestStage1FailureAttribution:
    """Stage 1 must not blame the ticker for an infrastructure fault."""

    def _evaluator(self):
        from strategy.evaluator import StockEvaluator
        ev = StockEvaluator.__new__(StockEvaluator)
        ev._tradier = MagicMock()
        return ev

    def test_unavailable_reports_market_data_unavailable(self):
        from market_data.tradier_rest import MarketDataUnavailable
        with patch("strategy.evaluator.MARKET_DATA_ENABLED", True):
            ev = self._evaluator()
            ev._tradier.get_quote.side_effect = MarketDataUnavailable("refused")
            s1 = ev._stage1("JZXN")

        assert s1.passed is False
        assert s1.fail_reason == "market data unavailable"
        assert s1.fail_reason != "symbol not found"

    def test_unknown_symbol_still_reports_symbol_not_found(self):
        """The genuine not-found path must be unchanged."""
        with patch("strategy.evaluator.MARKET_DATA_ENABLED", True):
            ev = self._evaluator()
            ev._tradier.get_quote.return_value = StockQuote(
                symbol="NOPE", description="", last=None, change=None,
                change_pct=None, volume=0, avg_volume=0, open=None,
                high=None, low=None, prev_close=None, bid=None, ask=None,
                bid_size=0, ask_size=0, last_volume=0, trade_ts=None,
                exchange="",
            )
            s1 = ev._stage1("NOPE")

        assert s1.passed is False
        assert s1.fail_reason == "symbol not found"

    def test_unavailable_does_not_zero_out_a_valid_ticker(self):
        """The failure must be reported, not silently scored as flat."""
        from market_data.tradier_rest import MarketDataUnavailable
        with patch("strategy.evaluator.MARKET_DATA_ENABLED", True):
            ev = self._evaluator()
            ev._tradier.get_quote.side_effect = MarketDataUnavailable("pacing")
            s1 = ev._stage1("GMEXC")

        assert s1.price == 0.0
        assert s1.passed is False
        assert s1.symbol == "GMEXC"


class TestPipelineFailureRouting:
    """run_pipeline must surface the two reasons distinctly."""

    def test_unavailable_is_not_treated_as_not_found(self):
        import pipeline as pl
        import inspect
        src = inspect.getsource(pl.run_pipeline)
        # Both reasons bail early, but they must be reported verbatim rather
        # than the message being hardcoded to "symbol not found".
        assert '"market data unavailable"' in src
        assert 'safe_print(f"  {_r(\'âœ—\')}  {symbol} â€” symbol not found")' not in src

    def test_stage1_snapshot_preserves_cache_on_unavailable(self):
        """A feed hiccup must not overwrite cached quote fields with zeros."""
        from pipeline import _apply_stage1_snapshot
        entry = SimpleNamespace(
            price=4.20, change_pct=30.0, volume=900_000, avg_volume=100_000,
            rvol_1day=9.0, rvol_5min=4.0, daily_high=4.50, evaluation=None,
        )
        s1 = SimpleNamespace(
            fail_reason="market data unavailable", price=0.0, change_pct=0.0,
            volume=0, avg_volume=0, rvol_1day=0.0, rvol_5min=0.0,
            daily_high=0.0,
        )

        assert _apply_stage1_snapshot(entry, s1) is False
        assert entry.price == 4.20        # untouched
        assert entry.volume == 900_000


class TestTradierQuoteFailureParity:
    """Both providers honour the same contract."""

    def test_5xx_raises_unavailable(self):
        from market_data.tradier_rest import HistoricalClient, MarketDataUnavailable
        with patch.dict("os.environ", {"TRADIER_API_KEY": "k", "TRADIER_PAPER": "false"}), \
             patch("market_data.tradier_rest._TRADIER_ENABLED", True):
            c = HistoricalClient()
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 503
        resp.text = "upstream unavailable"
        with patch.object(c._session, "get", return_value=resp), \
             patch("config.TRADIER_PAUSED", False):
            with pytest.raises(MarketDataUnavailable):
                c.get_quotes(["ACMR"])

    def test_network_error_raises_unavailable(self):
        import requests
        from market_data.tradier_rest import HistoricalClient, MarketDataUnavailable
        with patch.dict("os.environ", {"TRADIER_API_KEY": "k", "TRADIER_PAPER": "false"}), \
             patch("market_data.tradier_rest._TRADIER_ENABLED", True):
            c = HistoricalClient()
        with patch.object(
            c._session, "get",
            side_effect=requests.exceptions.ConnectTimeout("timed out"),
        ), patch("config.TRADIER_PAUSED", False):
            with pytest.raises(MarketDataUnavailable):
                c.get_quotes(["ACMR"])

    def test_4xx_still_returns_empty_not_raises(self):
        """A 4xx means the symbol was rejected â€” genuinely not found."""
        from market_data.tradier_rest import HistoricalClient
        with patch.dict("os.environ", {"TRADIER_API_KEY": "k", "TRADIER_PAPER": "false"}), \
             patch("market_data.tradier_rest._TRADIER_ENABLED", True):
            c = HistoricalClient()
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 404
        resp.text = "not found"
        with patch.object(c._session, "get", return_value=resp), \
             patch("config.TRADIER_PAUSED", False):
            assert c.get_quotes(["NOPE"]) == []

    def test_candle_paths_still_degrade_quietly_on_5xx(self):
        """Only the quote path opts into raising; candles must be unchanged."""
        from market_data.tradier_rest import HistoricalClient
        with patch.dict("os.environ", {"TRADIER_API_KEY": "k", "TRADIER_PAPER": "false"}), \
             patch("market_data.tradier_rest._TRADIER_ENABLED", True):
            c = HistoricalClient()
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 503
        resp.text = "upstream unavailable"
        with patch.object(c._session, "get", return_value=resp), \
             patch("config.TRADIER_PAUSED", False):
            assert c._get("/markets/history") == {}



# ===========================================================================
# Pacing gate + duration clamp â€” hit a limit, warn clearly, keep moving
# ===========================================================================




class TestWindowDurationClamp:
    """Error 321: IBKR rejects 'S'-unit durations above 86400 outright."""

    def _mock_ib(self):
        ib = MagicMock()
        ib.qualifyContracts.return_value = [MagicMock()]
        ib.reqHistoricalData.return_value = []
        return ib

    @patch("market_data.ibkr_rest.get_ib")
    def test_10s_window_over_a_day_truncates_not_rejects(self, mock_get_ib, ibkr_client):
        mock_ib = self._mock_ib()
        mock_get_ib.return_value = mock_ib

        end = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
        start = end - timedelta(days=3)
        ibkr_client.get_candles_10sec_window("QBTS", start, end)

        kwargs = mock_ib.reqHistoricalData.call_args.kwargs
        assert kwargs["durationStr"] == "86400 S"  # was "259200 S" -> Error 321

    @patch("market_data.ibkr_rest.get_ib")
    def test_minute_window_over_a_day_switches_to_day_units(self, mock_get_ib, ibkr_client):
        mock_ib = self._mock_ib()
        mock_get_ib.return_value = mock_ib

        end = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
        start = end - timedelta(days=2, hours=12)
        ibkr_client.get_candles_1min_window("ACMR", start, end)

        kwargs = mock_ib.reqHistoricalData.call_args.kwargs
        assert kwargs["durationStr"] == "3 D"

    @patch("market_data.ibkr_rest.get_ib")
    def test_window_within_a_day_is_unchanged(self, mock_get_ib, ibkr_client):
        mock_ib = self._mock_ib()
        mock_get_ib.return_value = mock_ib

        end = datetime(2026, 7, 31, 20, 0, tzinfo=timezone.utc)
        start = end - timedelta(hours=2)
        ibkr_client.get_candles_10sec_window("ACMR", start, end)

        kwargs = mock_ib.reqHistoricalData.call_args.kwargs
        assert kwargs["durationStr"] == "7200 S"



# ===========================================================================
# Log noise — ib_insync's one-shot scanner teardown
# ===========================================================================

class TestScannerTeardownLogFilter:
    """reqScannerData cancels its own subscription; the Gateway acks that as
    an Error 162, which ib_insync logs at ERROR once per poll."""

    def _filter(self):
        import logging
        from market_data import _IBKRNoiseFilter
        return _IBKRNoiseFilter(), logging

    def _record(self, logging, msg, *args):
        return logging.LogRecord("ib_insync.wrapper", logging.ERROR,
                                 __file__, 1, msg, args, None)

    def test_scanner_cancel_ack_is_dropped(self):
        f, logging = self._filter()
        rec = self._record(
            logging,
            "Error %s, reqId %s: Historical Market Data Service error "
            "message:API scanner subscription cancelled: %s", 162, 3, 3,
        )
        assert f.filter(rec) is False

    def test_pacing_violation_still_gets_through(self):
        """Same 162 code — must stay visible, it means we are over budget."""
        f, logging = self._filter()
        rec = self._record(
            logging,
            "Error %s, reqId %s: Historical Market Data Service error "
            "message:Historical data request pacing violation", 162, 4,
        )
        assert f.filter(rec) is True

    def test_unrelated_errors_still_get_through(self):
        f, logging = self._filter()
        rec = self._record(logging, "Error %s, reqId %s: %s", 326, -1,
                           "Unable to connect as the client id is already in use.")
        assert f.filter(rec) is True

    def test_data_farm_ok_chatter_is_dropped(self):
        """2104/2106/2158 fire on every connect and only ever say 'fine'."""
        f, logging = self._filter()
        for code, farm in ((2104, "usfarm.nj"), (2106, "ushmds"), (2158, "secdefil")):
            rec = self._record(
                logging, "Warning %s, reqId %s: %s data farm connection is OK:%s",
                code, -1, "Market", farm,
            )
            assert f.filter(rec) is False, code

    def test_depth_permission_warning_logs_once_then_quiets(self):
        """Invariant across symbols — the first copy is the informative one."""
        f, logging = self._filter()

        def _perm(sym):
            return self._record(
                logging,
                "Warning %s, reqId %s: Exchanges - Depth: NASDAQ; IEX; Need "
                "additional market data permissions - Depth: BATS; ARCA; , "
                "contract: Stock(symbol=%r)", 2152, 33, sym,
            )

        assert f.filter(_perm("RUBI")) is True
        assert f.filter(_perm("UPC")) is False
        assert f.filter(_perm("RUBI")) is False

    def test_quota_refusal_echoes_are_dropped(self):
        """10190/309 and the error-300 cancel echo are all translated by
        IBKRSharedStream._on_ib_error into one actionable WARNING with
        cap/backoff context — the raw ib_insync copies are duplicates that
        turned the 2026-08-05 quota incident into an error storm. (If the
        errorEvent hook ever fails to attach, that failure logs its own
        warning, so the translation loss is not silent.)"""
        f, logging = self._filter()
        for code, msg in (
            (10190, "Max number of tick-by-tick requests has been reached."),
            (309, "Max number (3) of market depth requests has been reached"),
            (300, "Can't find EId with tickerId:5719"),
        ):
            rec = self._record(logging, "Error %s, reqId %s: %s", code, 31, msg)
            assert f.filter(rec) is False, code


# ===========================================================================
# ib_insync DOM bounds patch
# ===========================================================================

class TestMktDepthL2Patch:
    """ib_insync 0.9.86 assigns dom[position] unguarded on an update, so an
    update for a level the local book never saw inserted raises IndexError —
    the decoder swallows it and the level keeps a stale size forever."""

    class _Ticker:
        """Stand-in for ib_insync.Ticker. A plain class, not SimpleNamespace —
        the handler puts tickers in pendingTickers, and SimpleNamespace defines
        __eq__ so it is unhashable."""

        def __init__(self):
            self.domBids, self.domAsks, self.domTicks = [], [], []

    def _wrapper(self):
        """A Wrapper stand-in carrying only what the handler touches."""
        ticker = self._Ticker()
        w = SimpleNamespace(
            reqId2Ticker={7: ticker}, lastTime=None, pendingTickers=set(),
        )
        return w, ticker

    def _send(self, w, **kw):
        from market_data.ibkr_stream import _update_mkt_depth_l2
        kw.setdefault("reqId", 7)
        kw.setdefault("marketMaker", "NSDQ")
        kw.setdefault("side", 1)          # bid
        kw.setdefault("size", 100)
        _update_mkt_depth_l2(w, **kw)

    def test_update_past_end_does_not_raise(self):
        """The reported crash: operation 1 at a position the book lacks."""
        w, ticker = self._wrapper()
        self._send(w, position=3, operation=1, price=4.20)
        assert ticker.domBids[3].price == 4.20

    def test_padding_keeps_later_positions_aligned(self):
        """Appending instead of padding would shift every later index."""
        w, ticker = self._wrapper()
        self._send(w, position=2, operation=1, price=4.20)
        self._send(w, position=0, operation=1, price=4.25)
        assert [lvl.price for lvl in ticker.domBids] == [4.25, 0.0, 4.20]

    def test_placeholders_are_skipped_by_the_ladder(self):
        """Padding must not invent a price level for OFI to trade off."""
        from market_data.ibkr_stream import _aggregate_dom
        w, ticker = self._wrapper()
        self._send(w, position=2, operation=1, price=4.20, size=300)
        assert _aggregate_dom(ticker.domBids, descending=True) == [(4.20, 300)]

    def test_normal_insert_update_delete_still_work(self):
        """The in-range path must behave exactly as upstream did."""
        w, ticker = self._wrapper()
        self._send(w, position=0, operation=0, price=4.10, size=100)
        self._send(w, position=1, operation=0, price=4.05, size=200)
        self._send(w, position=0, operation=1, price=4.11, size=150)
        assert [(l.price, l.size) for l in ticker.domBids] == [(4.11, 150), (4.05, 200)]
        self._send(w, position=0, operation=2, price=0.0)
        assert [(l.price, l.size) for l in ticker.domBids] == [(4.05, 200)]

    def test_ask_side_is_routed_separately(self):
        w, ticker = self._wrapper()
        self._send(w, position=1, operation=1, price=4.30, side=0)
        assert ticker.domAsks[1].price == 4.30
        assert ticker.domBids == []

    def test_unknown_reqid_is_dropped_not_a_keyerror(self):
        """A straggler crossing with our own cancelMktDepth."""
        w, _ = self._wrapper()
        self._send(w, reqId=999, position=0, operation=1, price=4.20)

    def test_negative_position_does_not_corrupt_the_tail(self):
        """dom[-1] would silently overwrite the worst level."""
        w, ticker = self._wrapper()
        self._send(w, position=0, operation=0, price=4.10)
        self._send(w, position=-1, operation=1, price=9.99)
        assert [lvl.price for lvl in ticker.domBids] == [4.10]

    def test_tick_records_a_dom_tick_and_flags_the_ticker(self):
        """_dispatch_depth only wakes on pendingTickers — a padded update that
        skipped it would land in the book but never be published."""
        w, ticker = self._wrapper()
        self._send(w, position=2, operation=1, price=4.20, size=300)
        assert len(ticker.domTicks) == 1
        assert ticker.domTicks[0].position == 2
        assert ticker in w.pendingTickers

    def test_installed_buffer_pins_each_post_update_state(self):
        """With the stream's buffer installed, every update appends a copy of
        the post-update ladder — the intermediate states OFI consumes. Copies
        must be pinned: later updates may not rewrite earlier states."""
        w, ticker = self._wrapper()
        ticker._momo_book_states = deque(maxlen=8)
        w.lastTime = datetime(2026, 8, 3, 14, 30, 0, tzinfo=timezone.utc)
        self._send(w, position=0, operation=0, price=4.10, size=100)
        self._send(w, position=0, operation=1, price=4.10, size=250)
        assert len(ticker._momo_book_states) == 2
        ts0, bids0, _ = ticker._momo_book_states[0]
        _, bids1, _ = ticker._momo_book_states[1]
        assert ts0 == w.lastTime
        assert [(l.price, l.size) for l in bids0] == [(4.10, 100)]
        assert [(l.price, l.size) for l in bids1] == [(4.10, 250)]

    def test_no_buffer_means_no_capture(self):
        """Tickers this stream does not own depth for (probes, other clients
        on the same patched Wrapper) must not accumulate states."""
        w, ticker = self._wrapper()
        self._send(w, position=0, operation=0, price=4.10)
        assert not hasattr(ticker, "_momo_book_states")


# ===========================================================================
# Tape print banking — the waitOnUpdate harvest-loss fix
# ===========================================================================

class TestTickByTickPatch:
    """ib_insync documents that a waitOnUpdate loop harvesting
    ticker.tickByTicks loses prints: when two TCP batches process inside one
    wake, tcpDataArrived for the second clears the first batch's list before
    the loop reads it. The patched handler banks each print at decode time."""

    class _Ticker:
        def __init__(self):
            self.tickByTicks = []

    def _wrapper(self):
        ticker = self._Ticker()
        w = SimpleNamespace(reqId2Ticker={9: ticker})
        return w, ticker

    @staticmethod
    def _orig(self, reqId, tickType, time, price, size, tickAttribLast,
              exchange, specialConditions):
        """Stand-in for the stock handler: appends the decoded print."""
        self.reqId2Ticker[reqId].tickByTicks.append(
            SimpleNamespace(price=price, size=size))

    def _send(self, w, price, size):
        from market_data.ibkr_stream import _tick_by_tick_all_last
        with patch("market_data.ibkr_stream._orig_tick_by_tick", self._orig):
            _tick_by_tick_all_last(w, 9, 1, 0, price, size, None, "NSDQ", "")

    def test_banks_each_print_into_the_installed_buffer(self):
        w, ticker = self._wrapper()
        ticker._momo_tape_events = deque(maxlen=8)
        self._send(w, 5.00, 100)
        self._send(w, 5.01, 200)
        assert [(t.price, t.size) for t in ticker._momo_tape_events] == [
            (5.00, 100), (5.01, 200)]
        # Delegation ran: the stock handler's own side effects survive
        assert len(ticker.tickByTicks) == 2

    def test_no_buffer_means_no_banking(self):
        """Tickers this stream does not own the tape for (probes, other
        clients on the same patched Wrapper) must not accumulate prints."""
        w, ticker = self._wrapper()
        self._send(w, 5.00, 100)
        assert not hasattr(ticker, "_momo_tape_events")
        assert len(ticker.tickByTicks) == 1    # delegation still runs

    def test_unknown_reqid_is_dropped_not_a_keyerror(self):
        from market_data.ibkr_stream import _tick_by_tick_all_last
        w, _ = self._wrapper()
        with patch("market_data.ibkr_stream._orig_tick_by_tick", None):
            _tick_by_tick_all_last(w, 999, 1, 0, 5.0, 100, None, "NSDQ", "")


class TestTapeDispatchBuffer:
    """_loop's tape dispatch drains the decode-time buffer, so prints from a
    TCP batch cleared by the next batch's tcpDataArrived still reach
    subscribers."""

    def _make_stream(self):
        with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
             patch.object(IBKRSharedStream, "_run"):
            stream = IBKRSharedStream()
        return stream

    def test_start_ts_installs_buffer_and_stop_removes_it(self):
        stream = self._make_stream()
        ib = MagicMock()
        stream._contracts["AAA"] = MagicMock()
        stream._start_ts(ib, "AAA")
        ticker = stream._ts_tickers["AAA"]
        assert isinstance(ticker._momo_tape_events, deque)
        stream._stop_ts(ib, "AAA")
        assert not hasattr(ticker, "_momo_tape_events")

    def test_buffered_prints_survive_a_batch_clear(self):
        """The loss scenario itself: batch 1's print is banked at decode
        time, then ib_insync clears tickByTicks (tcpDataArrived for batch 2)
        before the loop reads it. Both prints must still be delivered."""
        stream = self._make_stream()
        buf = deque(maxlen=8)
        ticker = SimpleNamespace(tickByTicks=[], _momo_tape_events=buf)
        t1 = SimpleNamespace(price=5.00, size=100)
        t2 = SimpleNamespace(price=5.01, size=200)
        ticker.tickByTicks.append(t1)
        buf.append(t1)
        ticker.tickByTicks = []            # tcpDataArrived wipes the list
        ticker.tickByTicks.append(t2)
        buf.append(t2)
        assert stream._pending_tape_events(ticker) == [t1, t2]
        assert not buf                     # drained
        assert stream._pending_tape_events(ticker) == []

    def test_without_buffer_falls_back_to_read_and_clear(self):
        """Legacy path (stubbed ib_insync, tickers not started by this
        stream): read tickByTicks and clear it, exactly as before."""
        stream = self._make_stream()
        t1 = SimpleNamespace(price=5.00, size=100)
        ticker = SimpleNamespace(tickByTicks=[t1])
        assert stream._pending_tape_events(ticker) == [t1]
        assert ticker.tickByTicks == []


# ===========================================================================
# Tape (tick-by-tick) quota — Error 10190
# ===========================================================================

class TestTapeQuota:
    """The tick-by-tick quota is 5 on this account — measured 2026-08-04
    (tests/test_files/tick_subs_probe.py): subs 1-5 granted, 6th refused
    with Error 10190 at request time. It is NOT derived from the market
    depth formula (3 here) as the docs suggest — the quotas are independent.
    Overrunning it is silent in-stream: reqTickByTickData answers normally
    and Error 10190 arrives later, leaving a Ticker that never emits a
    print."""

    def _make_stream(self, limit=3):
        with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
             patch.object(IBKRSharedStream, "_run"):
            stream = IBKRSharedStream()
        stream._ts_limit = limit
        return stream

    def _ib(self):
        ib = MagicMock()
        ib.qualifyContracts.return_value = [MagicMock()]
        return ib

    def _add(self, stream, ib, sym):
        stream._contracts[sym] = MagicMock()
        return stream._start_ts(ib, sym)

    def test_cap_matches_the_measured_account_quota(self):
        """Pins the measured cap (5), not the depth formula — see the class
        docstring. If IBKR resizes the account quota, re-run the probe and
        move this number with it."""
        from config import IBKR_MAX_TICK_SUBS
        assert IBKR_MAX_TICK_SUBS == 5

    def test_subscribes_up_to_the_cap(self):
        stream, ib = self._make_stream(3), self._ib()
        for sym in ("AAA", "BBB", "CCC"):
            assert self._add(stream, ib, sym) is True
        assert set(stream._ts_tickers) == {"AAA", "BBB", "CCC"}
        assert ib.reqTickByTickData.call_count == 3

    def test_over_cap_evicts_the_oldest_not_the_newcomer(self):
        stream, ib = self._make_stream(2), self._ib()
        self._add(stream, ib, "AAA")
        self._add(stream, ib, "BBB")
        self._add(stream, ib, "CCC")

        assert "AAA" not in stream._ts_tickers      # oldest dropped
        assert set(stream._ts_tickers) == {"BBB", "CCC"}
        assert stream._ts_lru == ["BBB", "CCC"]
        ib.cancelTickByTickData.assert_called_once()

    def test_eviction_keeps_l1_quotes_alive(self):
        """Losing the tape must degrade to L1, not unsubscribe the symbol."""
        stream, ib = self._make_stream(1), self._ib()
        stream._q_tickers["AAA"] = MagicMock()
        self._add(stream, ib, "AAA")
        self._add(stream, ib, "BBB")

        assert "AAA" not in stream._ts_tickers
        assert "AAA" in stream._q_tickers          # quotes untouched
        assert "AAA" in stream._contracts
        ib.cancelMktData.assert_not_called()

    def test_error_10190_lowers_the_cap_and_flags_the_symbol(self):
        stream = self._make_stream(8)
        stream._ts_tickers.update({"AAA": MagicMock(), "BBB": MagicMock(),
                                   "CCC": MagicMock(), "DDD": MagicMock()})
        contract = SimpleNamespace(symbol="DDD")

        stream._on_ib_error(31, 10190, "Max number of tick-by-tick requests has been reached.", contract)

        assert stream._ts_limit == 3     # 4 registered, the 4th was refused
        assert "DDD" in stream._ts_denied

    def test_unrelated_errors_do_not_touch_the_cap(self):
        stream = self._make_stream(8)
        stream._ts_tickers["AAA"] = MagicMock()
        stream._on_ib_error(1, 162, "Historical Market Data Service error",
                            SimpleNamespace(symbol="AAA"))
        assert stream._ts_limit == 8
        assert stream._ts_denied == set()

    def test_cap_is_never_raised_by_an_error(self):
        stream = self._make_stream(2)
        stream._ts_tickers.update({chr(65 + i) * 3: MagicMock() for i in range(9)})
        stream._on_ib_error(31, 10190, "limit", SimpleNamespace(symbol="AAA"))
        assert stream._ts_limit == 2     # learned 8, but never increase

    def test_denied_symbol_is_reaped_without_an_instant_retry(self):
        """The refused Ticker never emits — leaving it registered would hold
        a quota slot while delivering nothing. But it must NOT be re-requested
        inline either: the refusal means the quota is held outside this
        process (another session, or Gateway-leaked lines from a dead one —
        observed 2026-08-05), and instant retries turned that into an
        error-spam loop. Re-entry happens via the rebalancer after backoff."""
        stream, ib = self._make_stream(2), self._ib()
        stream._subs["CCC"] = {"s1": (MagicMock(), MagicMock(), None)}
        self._add(stream, ib, "AAA")
        self._add(stream, ib, "CCC")
        stream._ts_denied.add("CCC")
        ib.cancelTickByTickData.reset_mock()
        ib.reqTickByTickData.reset_mock()
        stream._ts_last_req.clear()      # 15s guard is not what blocks it

        stream._reap_denied_subs(ib)

        assert ib.cancelTickByTickData.called          # dead registration dropped
        ib.reqTickByTickData.assert_not_called()       # NO inline retry
        assert "CCC" not in stream._ts_tickers
        assert stream._ts_denied == set()

    def test_denial_sets_an_exponential_backoff(self):
        stream = self._make_stream(2)
        stream._ts_tickers["AAA"] = MagicMock()
        contract = SimpleNamespace(symbol="AAA")
        from config import IBKR_SUB_DENIAL_BACKOFF_SECS

        stream._on_ib_error(1, 10190, "limit", contract)
        first = stream._ts_retry_after["AAA"] - time.monotonic()
        stream._on_ib_error(2, 10190, "limit", contract)
        second = stream._ts_retry_after["AAA"] - time.monotonic()

        assert stream._ts_denials["AAA"] == 2
        assert first == pytest.approx(IBKR_SUB_DENIAL_BACKOFF_SECS, abs=1.0)
        assert second == pytest.approx(2 * IBKR_SUB_DENIAL_BACKOFF_SECS, abs=1.0)

    def test_backoff_blocks_start_ts_until_expiry(self):
        stream, ib = self._make_stream(3), self._ib()
        stream._contracts["AAA"] = MagicMock()
        stream._ts_retry_after["AAA"] = time.monotonic() + 60

        assert stream._start_ts(ib, "AAA") is False
        ib.reqTickByTickData.assert_not_called()

        stream._ts_retry_after["AAA"] = time.monotonic() - 1
        assert stream._start_ts(ib, "AAA") is True

    def test_pin_raise_bypasses_the_backoff(self):
        """A pin raise is an explicit strategy claim (breakout confirming /
        position open) — it gets one immediate attempt despite the backoff."""
        stream, ib = self._make_stream(3), self._ib()
        stream._contracts["AAA"] = MagicMock()
        stream._ts_retry_after["AAA"] = time.monotonic() + 60

        stream._set_pin(ib, "AAA", 2)

        ib.reqTickByTickData.assert_called_once()

    def test_cap_probes_back_up_after_a_quiet_period(self):
        from config import IBKR_TAPE_CAP_RECOVERY_SECS
        stream, ib = self._make_stream(2), self._ib()
        stream._ts_denials["AAA"] = 3
        stream._ts_retry_after["AAA"] = time.monotonic() + 600
        stream._ts_last_denial = time.monotonic() - IBKR_TAPE_CAP_RECOVERY_SECS - 1
        stream._last_ts_rebalance = 0.0

        stream._maybe_rebalance_ts(ib)

        assert stream._ts_limit == 3                   # one line back
        assert stream._ts_denials == {}                # fresh start
        assert stream._ts_retry_after == {}

    def test_cap_never_recovers_above_the_configured_max(self):
        from config import IBKR_MAX_TICK_SUBS, IBKR_TAPE_CAP_RECOVERY_SECS
        stream, ib = self._make_stream(IBKR_MAX_TICK_SUBS), self._ib()
        stream._ts_last_denial = time.monotonic() - IBKR_TAPE_CAP_RECOVERY_SECS - 1
        stream._last_ts_rebalance = 0.0

        stream._maybe_rebalance_ts(ib)

        assert stream._ts_limit == IBKR_MAX_TICK_SUBS

    def test_depth_309_reaps_the_dead_registration_with_backoff(self):
        """A refused reqMktDepth previously sat in _depth_tickers forever —
        a local line spent on a book that never updates (bites on a quick
        restart while a dead session's depth lines are still counted)."""
        stream, ib = self._make_stream(3), self._ib()
        stream._contracts["AAA"] = MagicMock()
        stream._depth_tickers["AAA"] = MagicMock()
        stream._depth_lru.append("AAA")

        stream._on_ib_error(7, 309, "Max number (3) of market depth requests has been reached",
                            SimpleNamespace(symbol="AAA"))
        assert "AAA" in stream._depth_denied
        assert stream._ts_limit == 3                   # tape cap untouched

        stream._reap_denied_subs(ib)
        assert "AAA" not in stream._depth_tickers
        ib.cancelMktDepth.assert_called_once()
        assert stream._depth_retry_after["AAA"] > time.monotonic()

    def test_unsubscribe_keeps_the_15s_request_guard(self):
        """The 15s per-instrument rule is server-side state — a symbol that
        leaves and re-enters the watchlist inside the window must not
        re-request immediately."""
        stream, ib = self._make_stream(3), self._ib()
        self._add(stream, ib, "AAA")
        stream._subs["AAA"] = {"s1": (MagicMock(), MagicMock(), None)}
        stream._stop_symbol(ib, "AAA")
        assert "AAA" in stream._ts_last_req

        stream._contracts["AAA"] = MagicMock()
        ib.reqTickByTickData.reset_mock()
        assert stream._start_ts(ib, "AAA") is False    # still inside 15s

    def test_resubscribe_within_15s_is_refused(self):
        """IBKR: no more than 1 tick-by-tick request per instrument per 15s."""
        stream, ib = self._make_stream(3), self._ib()
        self._add(stream, ib, "AAA")
        stream._stop_ts(ib, "AAA")
        ib.reqTickByTickData.reset_mock()

        assert self._add(stream, ib, "AAA") is False
        ib.reqTickByTickData.assert_not_called()

    def test_resubscribe_allowed_once_the_cooldown_passes(self):
        from config import IBKR_TICK_RESUB_COOLDOWN_SECS
        stream, ib = self._make_stream(3), self._ib()
        self._add(stream, ib, "AAA")
        stream._stop_ts(ib, "AAA")
        stream._ts_last_req["AAA"] -= IBKR_TICK_RESUB_COOLDOWN_SECS + 1
        ib.reqTickByTickData.reset_mock()

        assert self._add(stream, ib, "AAA") is True
        ib.reqTickByTickData.assert_called_once()


# ===========================================================================
# L1 quote dispatch — bid/ask/last change detection
# ===========================================================================

class TestQuoteDispatch:
    """LiveQuote fan-out. `last` is its own trigger: it is the only price
    feed a symbol without a tape line has, and prints inside the spread
    move it without the NBBO budging."""

    def _make_stream(self):
        with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
             patch.object(IBKRSharedStream, "_run"):
            stream = IBKRSharedStream()
        return stream

    def _ticker(self, **kw):
        return SimpleNamespace(
            bid=kw.get("bid", 5.00), ask=kw.get("ask", 5.02),
            bidSize=kw.get("bidSize", 100), askSize=kw.get("askSize", 200),
            bidExchange="ARCA", askExchange="NSDQ",
            last=kw.get("last", 5.01), lastSize=kw.get("lastSize", 50),
        )

    def test_quote_event_carries_l1_last(self):
        stream = self._make_stream()
        cb_q = MagicMock()
        stream.subscribe("ACMR", MagicMock(), cb_q, sub_id="s1")
        stream._q_tickers["ACMR"] = self._ticker()
        stream._dispatch_quotes()
        assert cb_q.call_count == 1
        evt = cb_q.call_args[0][0]
        assert evt.last == 5.01
        assert evt.last_size == 50

    def test_unchanged_quote_is_deduped(self):
        stream = self._make_stream()
        cb_q = MagicMock()
        stream.subscribe("ACMR", MagicMock(), cb_q, sub_id="s1")
        stream._q_tickers["ACMR"] = self._ticker()
        stream._dispatch_quotes()
        stream._dispatch_quotes()
        assert cb_q.call_count == 1

    def test_last_only_change_dispatches(self):
        """Print inside the spread: last moves, NBBO does not."""
        stream = self._make_stream()
        cb_q = MagicMock()
        stream.subscribe("ACMR", MagicMock(), cb_q, sub_id="s1")
        ticker = self._ticker()
        stream._q_tickers["ACMR"] = ticker
        stream._dispatch_quotes()
        ticker.last = 5.015
        stream._dispatch_quotes()
        assert cb_q.call_count == 2
        assert cb_q.call_args[0][0].last == 5.015

    def test_nan_last_is_zeroed_not_propagated(self):
        stream = self._make_stream()
        cb_q = MagicMock()
        stream.subscribe("ACMR", MagicMock(), cb_q, sub_id="s1")
        stream._q_tickers["ACMR"] = self._ticker(last=float("nan"), lastSize=float("nan"))
        stream._dispatch_quotes()
        evt = cb_q.call_args[0][0]
        assert evt.last == 0.0
        assert evt.last_size == 0


# ===========================================================================
# Slot priority — pins + L1 volume-rate ranking
# ===========================================================================

class TestSlotPriority:
    """Tape/depth lines go to pinned symbols first, then to whoever's L1
    volume rate says is hottest. Pins exist because the strategy buys
    pullbacks — the symbols that matter most have quiet tapes."""

    def _make_stream(self, limit=2):
        with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
             patch.object(IBKRSharedStream, "_run"):
            stream = IBKRSharedStream()
        stream._ts_limit = limit
        return stream

    def _ib(self):
        ib = MagicMock()
        ib.qualifyContracts.return_value = [MagicMock()]
        return ib

    def _add_ts(self, stream, ib, sym):
        stream._contracts[sym] = MagicMock()
        return stream._start_ts(ib, sym)

    def _set_rate(self, stream, sym, rate, price=10.0):
        """Fake a full-window volume history yielding *rate* shares/sec at
        an L1 *price* — so the depth gate sees rate x price dollars/sec."""
        from config import IBKR_TAPE_RANK_WINDOW_SECS as _win
        now = time.monotonic()
        stream._vol_hist[sym] = deque([(now - _win, 0), (now, int(rate * _win))])
        stream._q_tickers[sym] = SimpleNamespace(last=price, bid=price, ask=price)

    def test_set_priority_enqueues_prio_command(self):
        stream = self._make_stream()
        stream.set_priority("ACMR", 2)
        assert stream._cmd_q.get_nowait() == ("prio", "ACMR", 2)

    def test_pinned_tape_holder_survives_unpinned_challenger(self):
        stream, ib = self._make_stream(1), self._ib()
        self._add_ts(stream, ib, "AAA")
        stream._pins["AAA"] = 1
        assert self._add_ts(stream, ib, "BBB") is False
        assert set(stream._ts_tickers) == {"AAA"}
        ib.cancelTickByTickData.assert_not_called()

    def test_unpinned_tape_eviction_picks_the_slowest(self):
        stream, ib = self._make_stream(2), self._ib()
        self._add_ts(stream, ib, "AAA")
        self._add_ts(stream, ib, "BBB")
        self._set_rate(stream, "AAA", 500.0)
        self._set_rate(stream, "BBB", 50.0)
        self._add_ts(stream, ib, "CCC")
        assert set(stream._ts_tickers) == {"AAA", "CCC"}   # slow BBB evicted

    def test_higher_pin_challenger_takes_a_pinned_tape_line(self):
        stream, ib = self._make_stream(1), self._ib()
        self._add_ts(stream, ib, "AAA")
        stream._pins["AAA"] = 1                 # armed
        stream._contracts["BBB"] = MagicMock()
        stream._pins["BBB"] = 3                 # open position
        assert stream._start_ts(ib, "BBB") is True
        assert set(stream._ts_tickers) == {"BBB"}

    def test_set_pin_raise_claims_tape_and_depth(self):
        stream, ib = self._make_stream(1), self._ib()
        self._add_ts(stream, ib, "BBB")         # holds the only tape line
        stream._contracts["AAA"] = MagicMock()
        stream._subs["AAA"] = {"s1": (MagicMock(), MagicMock(), MagicMock())}
        with patch("market_data.ibkr_stream.IBKR_L2_MAX_DEPTH_SUBS", 1):
            stream._set_pin(ib, "AAA", 2)
        assert set(stream._ts_tickers) == {"AAA"}   # unpinned BBB evicted
        assert "AAA" in stream._depth_tickers
        assert stream._pins["AAA"] == 2

    def test_pin_drop_is_lazy(self):
        """A dead setup keeps its lines until a challenger wants one."""
        stream, ib = self._make_stream(2), self._ib()
        self._add_ts(stream, ib, "AAA")
        stream._pins["AAA"] = 2
        stream._set_pin(ib, "AAA", 0)
        assert "AAA" in stream._ts_tickers
        assert stream._pins["AAA"] == 0

    def test_vol_rate_from_samples_and_reset_on_decrease(self):
        stream = self._make_stream()
        now = time.monotonic()
        stream._vol_hist["AAA"] = deque([(now - 60, 10_000), (now, 40_000)])
        assert stream._vol_rate("AAA") == pytest.approx(500.0, rel=0.05)

    def test_short_history_cannot_fake_a_hot_tape(self):
        """A straggler print seconds after a symbol starts streaming must be
        averaged over the whole rank window, not its tiny sampled span —
        span-division let a dead symbol out-rank a steadily active holder."""
        from config import IBKR_TAPE_RANK_WINDOW_SECS as _win
        stream = self._make_stream()
        now = time.monotonic()
        stream._vol_hist["DEAD"] = deque([(now - 3, 100_000), (now, 100_258)])
        assert stream._vol_rate("DEAD") == pytest.approx(258 / _win, rel=0.05)
        # Cumulative volume going backwards = feed reset -> window restarts.
        stream._q_tickers["AAA"] = SimpleNamespace(volume=1_000)
        stream._last_vol_sample = 0.0
        stream._sample_volumes()
        assert len(stream._vol_hist["AAA"]) == 1

    def test_rebalance_fills_a_free_slot_with_an_outsider(self):
        stream, ib = self._make_stream(2), self._ib()
        self._add_ts(stream, ib, "AAA")
        stream._stop_ts(ib, "AAA")              # evicted earlier; line now free
        stream._ts_last_req["AAA"] -= 20        # 15s cooldown passed
        stream._subs["AAA"] = {"s1": (MagicMock(), MagicMock(), None)}
        stream._last_ts_rebalance = 0.0
        ib.reqTickByTickData.reset_mock()
        stream._maybe_rebalance_ts(ib)
        assert "AAA" in stream._ts_tickers      # won its tape back

    def test_rebalance_swaps_on_rate_margin_after_min_hold(self):
        stream, ib = self._make_stream(1), self._ib()
        self._add_ts(stream, ib, "COLD")
        stream._ts_granted_at["COLD"] -= 60     # min-hold satisfied
        stream._subs["HOT"] = {"s1": (MagicMock(), MagicMock(), None)}
        stream._contracts["HOT"] = MagicMock()
        self._set_rate(stream, "COLD", 100.0)
        self._set_rate(stream, "HOT", 300.0)    # clears the 1.5x margin
        stream._last_ts_rebalance = 0.0
        stream._maybe_rebalance_ts(ib)
        assert set(stream._ts_tickers) == {"HOT"}

    def test_rebalance_holds_below_the_margin(self):
        stream, ib = self._make_stream(1), self._ib()
        self._add_ts(stream, ib, "COLD")
        stream._ts_granted_at["COLD"] -= 60
        stream._subs["WARM"] = {"s1": (MagicMock(), MagicMock(), None)}
        stream._contracts["WARM"] = MagicMock()
        self._set_rate(stream, "COLD", 100.0)
        self._set_rate(stream, "WARM", 120.0)   # hotter, but under 1.5x
        stream._last_ts_rebalance = 0.0
        stream._maybe_rebalance_ts(ib)
        assert set(stream._ts_tickers) == {"COLD"}

    def test_rebalance_holds_inside_min_hold(self):
        stream, ib = self._make_stream(1), self._ib()
        self._add_ts(stream, ib, "FRESH")       # just granted
        stream._subs["HOT"] = {"s1": (MagicMock(), MagicMock(), None)}
        stream._contracts["HOT"] = MagicMock()
        self._set_rate(stream, "FRESH", 10.0)
        self._set_rate(stream, "HOT", 10_000.0)
        stream._last_ts_rebalance = 0.0
        stream._maybe_rebalance_ts(ib)
        assert set(stream._ts_tickers) == {"FRESH"}

    def test_rebalance_never_rate_evicts_a_pinned_holder(self):
        stream, ib = self._make_stream(1), self._ib()
        self._add_ts(stream, ib, "PINNED")
        stream._pins["PINNED"] = 1
        stream._ts_granted_at["PINNED"] -= 600
        stream._subs["HOT"] = {"s1": (MagicMock(), MagicMock(), None)}
        stream._contracts["HOT"] = MagicMock()
        self._set_rate(stream, "PINNED", 1.0)
        self._set_rate(stream, "HOT", 10_000.0)
        stream._last_ts_rebalance = 0.0
        stream._maybe_rebalance_ts(ib)
        assert set(stream._ts_tickers) == {"PINNED"}

    def test_stop_symbol_clears_pin_and_volume_history(self):
        stream, ib = self._make_stream(2), self._ib()
        self._add_ts(stream, ib, "AAA")
        stream._pins["AAA"] = 2
        self._set_rate(stream, "AAA", 10.0)
        stream._stop_symbol(ib, "AAA")
        assert "AAA" not in stream._pins
        assert "AAA" not in stream._vol_hist

    def _hold_depth(self, stream, ib, sym, *, pin=0, rate=0.0, idle_for=None):
        """Grant *sym* a depth line with the given pin/vol-rate; optionally
        backdate its idle clock by *idle_for* seconds."""
        stream._contracts[sym] = MagicMock()
        self._set_rate(stream, sym, 10_000.0)   # hot at grant time
        stream._start_depth(ib, sym)
        stream._pins[sym] = pin
        self._set_rate(stream, sym, rate)
        if idle_for is not None:
            stream._depth_idle_since[sym] = time.monotonic() - idle_for
        return sym

    def test_idle_release_drops_unpinned_quiet_depth_holder(self):
        from config import IBKR_DEPTH_IDLE_RELEASE_SECS as _idle
        stream, ib = self._make_stream(), self._ib()
        self._hold_depth(stream, ib, "DEAD", rate=1.0, idle_for=_idle + 1)
        stream._release_idle_depth(ib, time.monotonic())
        assert "DEAD" not in stream._depth_tickers
        ib.cancelMktDepth.assert_called_once()
        # cooldown armed: idle-fill may not hand the line straight back
        assert stream._depth_retry_after["DEAD"] > time.monotonic()

    def test_idle_release_waits_out_the_full_idle_window(self):
        stream, ib = self._make_stream(), self._ib()
        self._hold_depth(stream, ib, "FRESH", rate=1.0)   # quiet, but clock 0
        stream._release_idle_depth(ib, time.monotonic())
        assert "FRESH" in stream._depth_tickers
        ib.cancelMktDepth.assert_not_called()

    def test_idle_release_spares_pinned_and_active_holders(self):
        from config import IBKR_DEPTH_IDLE_RELEASE_SECS as _idle
        stream, ib = self._make_stream(), self._ib()
        self._hold_depth(stream, ib, "PINNED", pin=1, rate=0.0, idle_for=_idle + 1)
        self._hold_depth(stream, ib, "BUSY", rate=10_000.0, idle_for=_idle + 1)
        stream._release_idle_depth(ib, time.monotonic())
        assert set(stream._depth_tickers) == {"PINNED", "BUSY"}
        # standing wiped the stale idle clocks — going quiet later must
        # restart the window, not inherit the backdated start
        assert "PINNED" not in stream._depth_idle_since
        assert "BUSY" not in stream._depth_idle_since

    def test_activity_burst_resets_the_idle_clock(self):
        from config import IBKR_DEPTH_IDLE_RELEASE_SECS as _idle
        stream, ib = self._make_stream(), self._ib()
        self._hold_depth(stream, ib, "AAA", rate=1.0, idle_for=_idle - 5)
        self._set_rate(stream, "AAA", 10_000.0)           # burst mid-window
        stream._release_idle_depth(ib, time.monotonic())
        assert "AAA" in stream._depth_tickers
        assert "AAA" not in stream._depth_idle_since      # clock wiped
        self._set_rate(stream, "AAA", 1.0)                # quiet again
        stream._release_idle_depth(ib, time.monotonic())
        assert "AAA" in stream._depth_tickers             # window restarted

    def test_pin_raise_reclaims_line_through_release_cooldown(self):
        from config import IBKR_DEPTH_IDLE_RELEASE_SECS as _idle
        stream, ib = self._make_stream(), self._ib()
        self._hold_depth(stream, ib, "AAA", rate=1.0, idle_for=_idle + 1)
        stream._subs["AAA"] = {"s1": (MagicMock(), MagicMock(), MagicMock())}
        stream._release_idle_depth(ib, time.monotonic())
        assert "AAA" not in stream._depth_tickers
        stream._set_pin(ib, "AAA", 2)                     # setup armed again
        assert "AAA" in stream._depth_tickers             # cooldown bypassed

    def test_release_cooldown_blocks_plain_resubscribe(self):
        from config import IBKR_DEPTH_IDLE_RELEASE_SECS as _idle
        stream, ib = self._make_stream(), self._ib()
        self._hold_depth(stream, ib, "AAA", rate=1.0, idle_for=_idle + 1)
        stream._release_idle_depth(ib, time.monotonic())
        self._set_rate(stream, "AAA", 10_000.0)           # hot again, but early
        stream._start_depth(ib, "AAA")                    # no pin raise
        assert "AAA" not in stream._depth_tickers

    def test_demand_gate_declines_cold_unpinned_symbol(self):
        """A book nobody is trading is pure overhead — an unpinned symbol
        below the relevance floor gets no depth line even with all free."""
        stream, ib = self._make_stream(), self._ib()
        stream._contracts["COLD"] = MagicMock()
        self._set_rate(stream, "COLD", 1.0)     # $10/s — under the $500/s floor
        stream._start_depth(ib, "COLD")
        assert "COLD" not in stream._depth_tickers
        ib.reqMktDepth.assert_not_called()

    def test_demand_gate_admits_hot_or_pinned_symbols(self):
        stream, ib = self._make_stream(), self._ib()
        stream._contracts["HOT"] = MagicMock()
        self._set_rate(stream, "HOT", 10_000.0)
        stream._start_depth(ib, "HOT")
        assert "HOT" in stream._depth_tickers
        stream._contracts["ARMED"] = MagicMock()   # cold tape, but pinned
        stream._pins["ARMED"] = 1
        stream._start_depth(ib, "ARMED")
        assert "ARMED" in stream._depth_tickers

    def test_demand_gate_is_dollar_scaled_not_share_scaled(self):
        """Same share rate, different price: 100 sh/s is noise on a $2
        stock and real dollar flow on a $50 one — the floor must scale."""
        stream, ib = self._make_stream(), self._ib()
        stream._contracts["CHEAP"] = MagicMock()
        self._set_rate(stream, "CHEAP", 100.0, price=2.0)    # $200/s
        stream._start_depth(ib, "CHEAP")
        assert "CHEAP" not in stream._depth_tickers
        stream._contracts["RICH"] = MagicMock()
        self._set_rate(stream, "RICH", 100.0, price=50.0)    # $5k/s
        stream._start_depth(ib, "RICH")
        assert "RICH" in stream._depth_tickers

    def test_cold_newcomer_wins_a_line_once_its_rate_proves_out(self):
        """Subscribe-time decline is not final: the rebalance tick's
        idle-fill grants the line once the L1 rate clears the floor."""
        stream, ib = self._make_stream(), self._ib()
        stream._contracts["NEW"] = MagicMock()
        stream._subs["NEW"] = {"s1": (MagicMock(), MagicMock(), MagicMock())}
        stream._start_depth(ib, "NEW")          # no vol history yet
        assert "NEW" not in stream._depth_tickers
        self._set_rate(stream, "NEW", 10_000.0) # tape heats up
        stream._last_ts_rebalance = 0.0
        stream._maybe_rebalance_ts(ib)
        assert "NEW" in stream._depth_tickers

    def test_freed_depth_line_idle_fills_a_declined_wanter(self):
        stream, ib = self._make_stream(2), self._ib()
        with patch("market_data.ibkr_stream.IBKR_L2_MAX_DEPTH_SUBS", 1):
            stream._contracts["AAA"] = MagicMock()
            self._set_rate(stream, "AAA", 10_000.0)
            stream._start_depth(ib, "AAA")      # takes the only line
            stream._contracts["BBB"] = MagicMock()
            stream._subs["BBB"] = {"s1": (MagicMock(), MagicMock(), MagicMock())}
            self._set_rate(stream, "BBB", 10_000.0)
            stream._start_depth(ib, "BBB")      # declined: equal pin
            assert "BBB" not in stream._depth_tickers
            stream._stop_depth(ib, "AAA")       # holder releases the line
            stream._last_ts_rebalance = 0.0
            stream._maybe_rebalance_ts(ib)
            assert "BBB" in stream._depth_tickers


# ===========================================================================
# Historical tick pagination — bounded fan-out, essential data first
# ===========================================================================

class TestTickPaginationBounds:
    """reqHistoricalTicks caps at 1000 ticks, so a dense window bisects and
    re-requests. Unbounded, one call drained the whole pacing budget and
    returned nothing."""

    def _dense_ib(self):
        """Every request comes back full, so the bisection never terminates
        on its own — only the ceiling stops it."""
        ib = MagicMock()
        ib.qualifyContracts.return_value = [MagicMock()]
        ts = datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)

        def _full(*args, **kwargs):
            if kwargs["whatToShow"] == "BID_ASK":
                return [_make_historical_bidask(ts, 5.04, 5.06)] * 1000
            return [_make_historical_tick(ts, 5.05, 100)] * 1000

        ib.reqHistoricalTicks.side_effect = _full
        return ib

    @patch("market_data.ibkr_rest.get_ib")
    def test_dense_window_is_bounded_not_unbounded(self, mock_get_ib, ibkr_client):
        import market_data.ibkr_rest as rest
        mock_get_ib.return_value = self._dense_ib()
        start = datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 7, 31, 14, 30, tzinfo=timezone.utc)

        ibkr_client.get_timesales_tick("ACMR", start, end)

        passes = [c.kwargs["whatToShow"]
                  for c in mock_get_ib.return_value.reqHistoricalTicks.call_args_list]
        assert passes.count("TRADES") <= rest.IBKR_TICK_MAX_REQUESTS_TRADES
        assert passes.count("BID_ASK") <= rest.IBKR_TICK_MAX_REQUESTS_QUOTES

    @patch("market_data.ibkr_rest.get_ib")
    def test_quote_pass_cannot_starve_the_trade_pass(self, mock_get_ib, ibkr_client):
        """Separate ceilings — BID_ASK exhausting itself leaves TRADES intact."""
        mock_get_ib.return_value = self._dense_ib()
        start = datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 7, 31, 14, 30, tzinfo=timezone.utc)

        ibkr_client.get_timesales_tick("ACMR", start, end)

        passes = [c.kwargs["whatToShow"]
                  for c in mock_get_ib.return_value.reqHistoricalTicks.call_args_list]
        assert passes[0] == "TRADES"
        # Every TRADES request precedes every BID_ASK request.
        assert passes.index("BID_ASK") > max(
            i for i, p in enumerate(passes) if p == "TRADES"
        )

    @patch("market_data.ibkr_rest.get_ib")
    def test_truncation_is_logged_not_silent(self, mock_get_ib, ibkr_client, caplog):
        import logging as _logging
        mock_get_ib.return_value = self._dense_ib()
        start = datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 7, 31, 14, 30, tzinfo=timezone.utc)

        with caplog.at_level(_logging.WARNING, logger="market_data.ibkr_rest"):
            ibkr_client.get_timesales_tick("ACMR", start, end)

        assert any("ceiling" in r.message and "partial tape" in r.message
                   for r in caplog.records)

    @patch("market_data.ibkr_rest.get_ib")
    def test_no_trades_skips_the_quote_pass_entirely(self, mock_get_ib, ibkr_client):
        """Quote context is worthless with no prints to attach it to."""
        ib = MagicMock()
        ib.qualifyContracts.return_value = [MagicMock()]
        ib.reqHistoricalTicks.side_effect = lambda *a, **k: []
        mock_get_ib.return_value = ib
        start = datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)
        end = datetime(2026, 7, 31, 14, 5, tzinfo=timezone.utc)

        rows = ibkr_client.get_timesales_tick("ACMR", start, end)

        assert rows == []
        passes = [c.kwargs["whatToShow"] for c in ib.reqHistoricalTicks.call_args_list]
        assert passes == ["TRADES"]
