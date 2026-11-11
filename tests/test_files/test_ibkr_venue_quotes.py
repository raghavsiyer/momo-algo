"""
tests/test_files/test_ibkr_venue_quotes.py

Directed venue quotes (IBKR_L2_VENUE_QUOTES) merged into the L2 ladder.

IBKR's smart depth leaves ARCA, NYSE and Nasdaq BX out on this account, so each
depth line also carries one reqMktData quote per venue with the exchange set,
summed into the ladder like a depth row. All IB API calls are mocked. No
network access or TWS/Gateway required.
"""

import math
import sys
from collections import deque
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Stub ib_insync before any ibkr module is imported (same as test_ibkr.py)
sys.modules.setdefault("ib_insync", MagicMock())

from market_data.ibkr_stream import (
    IBKRSharedStream, _aggregate_dom, _directed_contract, _has_rows, _with_venue_rows,
)

_VENUES = ("ARCA", "NYSE", "BEX")


def _row(price: float, size: int, mm: str = "NSDQ") -> SimpleNamespace:
    return SimpleNamespace(price=price, size=size, marketMaker=mm)


def _venue_ticker(bid: float, bid_size: float, ask: float, ask_size: float) -> SimpleNamespace:
    return SimpleNamespace(bid=bid, bidSize=bid_size, ask=ask, askSize=ask_size)


@pytest.fixture(autouse=True)
def _fixed_venues_and_no_demand_gate():
    """Pin the venue list (independent of .env) and zero the depth demand
    floor so every symbol qualifies for a line (gate coverage lives in
    test_ibkr.py's TestSlotPriority)."""
    with patch("market_data.ibkr_stream.IBKR_L2_VENUE_QUOTES", _VENUES), \
         patch("market_data.ibkr_stream.IBKR_DEPTH_IDLE_DOLLAR_RATE", 0.0):
        yield


def _make_stream() -> IBKRSharedStream:
    with patch("market_data.ibkr_stream.IBKR_ENABLED", True), \
         patch.object(IBKRSharedStream, "_run"):
        return IBKRSharedStream()


def _contract(symbol: str = "ACMR") -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, exchange="SMART", currency="USD", conId=1234)


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

class TestVenueRowMerge:

    def test_venue_above_the_book_becomes_the_inside(self):
        rows = _with_venue_rows([_row(5.00, 300)], [_row(5.01, 400, "ARCA")])
        assert _aggregate_dom(rows, descending=True) == [(5.01, 400), (5.00, 300)]

    def test_venue_at_a_book_price_sums_into_that_level(self):
        rows = _with_venue_rows([_row(5.00, 300)], [_row(5.00, 200, "ARCA")])
        assert _aggregate_dom(rows, descending=True) == [(5.00, 500)]

    def test_venue_the_book_already_carries_is_not_added_twice(self):
        rows = _with_venue_rows([_row(5.00, 300, "ARCA")], [_row(5.01, 400, "ARCA")])
        assert _aggregate_dom(rows, descending=True) == [(5.00, 300)]

    def test_no_venue_rows_leaves_the_rows_as_they_are(self):
        rows = [_row(5.00, 300)]
        assert _with_venue_rows(rows, []) is rows

    def test_merge_does_not_modify_the_book_rows(self):
        rows = [_row(5.00, 300)]
        _with_venue_rows(rows, [_row(5.01, 400, "ARCA")])
        assert len(rows) == 1

    def test_has_rows_ignores_padding_placeholders(self):
        assert not _has_rows([])
        assert not _has_rows([_row(0.0, 0, "")])
        assert _has_rows([_row(0.0, 0, ""), _row(5.00, 100)])

    def test_directed_contract_is_a_copy_routed_to_the_venue(self):
        original = _contract()
        directed = _directed_contract(original, "ARCA")
        assert directed is not original
        assert directed.exchange == "ARCA" and directed.conId == 1234
        assert original.exchange == "SMART"


# ---------------------------------------------------------------------------
# Lifecycle: the venue quotes live exactly as long as the depth line
# ---------------------------------------------------------------------------

class TestVenueQuoteLifecycle:

    def test_depth_line_requests_one_directed_quote_per_venue(self):
        stream, ib = _make_stream(), MagicMock()
        stream._contracts["ACMR"] = _contract()
        stream._start_depth(ib, "ACMR")
        ib.reqMktDepth.assert_called_once()
        exchanges = [c.args[0].exchange for c in ib.reqMktData.call_args_list]
        assert exchanges == list(_VENUES)
        assert stream._contracts["ACMR"].exchange == "SMART"

    def test_venue_quotes_are_not_snapshots(self):
        stream, ib = _make_stream(), MagicMock()
        stream._contracts["ACMR"] = _contract()
        stream._start_depth(ib, "ACMR")
        assert all(c.kwargs.get("snapshot") is False for c in ib.reqMktData.call_args_list)

    def test_failed_depth_request_starts_no_venue_quotes(self):
        stream, ib = _make_stream(), MagicMock()
        ib.reqMktDepth.side_effect = RuntimeError("not connected")
        stream._contracts["ACMR"] = _contract()
        stream._start_depth(ib, "ACMR")
        ib.reqMktData.assert_not_called()
        assert "ACMR" not in stream._venue_quotes

    def test_stopping_depth_cancels_every_venue_quote(self):
        stream, ib = _make_stream(), MagicMock()
        stream._contracts["ACMR"] = _contract()
        stream._start_depth(ib, "ACMR")
        directed = [q[1] for q in stream._venue_quotes["ACMR"]]
        stream._stop_depth(ib, "ACMR")
        assert [c.args[0] for c in ib.cancelMktData.call_args_list] == directed
        assert "ACMR" not in stream._venue_quotes

    def test_reaped_309_refusal_takes_the_venue_quotes_down(self):
        stream, ib = _make_stream(), MagicMock()
        stream._contracts["ACMR"] = _contract()
        stream._start_depth(ib, "ACMR")
        stream._depth_denied.add("ACMR")
        stream._reap_denied_subs(ib)
        assert ib.cancelMktData.call_count == len(_VENUES)
        assert "ACMR" not in stream._venue_quotes

    def test_one_failed_venue_request_keeps_the_others(self):
        stream, ib = _make_stream(), MagicMock()
        ib.reqMktData.side_effect = [MagicMock(), RuntimeError("line limit"), MagicMock()]
        stream._contracts["ACMR"] = _contract()
        stream._start_depth(ib, "ACMR")
        assert [q[0] for q in stream._venue_quotes["ACMR"]] == ["ARCA", "BEX"]
        assert "ACMR" in stream._depth_tickers

    def test_empty_venue_list_turns_the_feature_off(self):
        stream, ib = _make_stream(), MagicMock()
        stream._contracts["ACMR"] = _contract()
        with patch("market_data.ibkr_stream.IBKR_L2_VENUE_QUOTES", ()):
            stream._start_depth(ib, "ACMR")
        ib.reqMktData.assert_not_called()
        assert "ACMR" in stream._depth_tickers


# ---------------------------------------------------------------------------
# Dispatch: venue rows in the snapshots on_book receives
# ---------------------------------------------------------------------------

class TestVenueRowsInDispatch:

    def _setup(self, depth_ticker: object) -> tuple[IBKRSharedStream, MagicMock, dict]:
        stream = _make_stream()
        cb_book = MagicMock()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1", on_book=cb_book)
        venues = {
            "ARCA": _venue_ticker(5.01, 400, 5.03, 200),
            "NYSE": _venue_ticker(math.nan, math.nan, -1.0, 0),    # no market yet / at all
            "BEX": _venue_ticker(4.90, 0, 5.20, 300),              # bid with no size
        }
        stream._depth_tickers["ACMR"] = depth_ticker
        stream._venue_quotes["ACMR"] = [(v, _contract(), t) for v, t in venues.items()]
        return stream, cb_book, venues

    @staticmethod
    def _ladder(snapshot: object) -> tuple[list, list]:
        return ([(l.price, l.size) for l in snapshot.bids],
                [(l.price, l.size) for l in snapshot.asks])

    def test_venue_tops_join_the_ladder_in_price_order(self):
        ticker = SimpleNamespace(domBids=[_row(5.00, 300)], domAsks=[_row(5.05, 400)])
        stream, cb_book, _ = self._setup(ticker)
        stream._dispatch_depth()
        bids, asks = self._ladder(cb_book.call_args[0][0])
        assert bids == [(5.01, 400), (5.00, 300)]
        assert asks == [(5.03, 200), (5.05, 400), (5.20, 300)]

    def test_a_side_with_no_market_adds_no_row(self):
        ticker = SimpleNamespace(domBids=[_row(5.00, 300)], domAsks=[_row(5.05, 400)])
        stream, cb_book, _ = self._setup(ticker)
        stream._dispatch_depth()
        bids, _ = self._ladder(cb_book.call_args[0][0])
        assert 4.90 not in [p for p, _ in bids]          # BEX bid had no size
        assert len(bids) == 2                             # NYSE's NaN bid added nothing

    def test_a_venue_change_alone_sends_the_ladder_again(self):
        t0 = datetime(2026, 11, 11, 14, 30, tzinfo=timezone.utc)
        buf = deque([(t0, [_row(5.00, 300)], [_row(5.05, 400)])])
        ticker = SimpleNamespace(_momo_book_states=buf,
                                 domBids=[_row(5.00, 300)], domAsks=[_row(5.05, 400)])
        stream, cb_book, venues = self._setup(ticker)
        stream._dispatch_depth()
        assert cb_book.call_count == 1
        venues["ARCA"].bid = 5.02                         # no depth row changed
        stream._dispatch_depth()
        assert cb_book.call_count == 2
        bids, _ = self._ladder(cb_book.call_args[0][0])
        assert bids[0] == (5.02, 400)
        stream._dispatch_depth()                          # nothing moved
        assert cb_book.call_count == 2

    def test_venue_rows_never_make_a_book_on_their_own(self):
        ticker = SimpleNamespace(domBids=[], domAsks=[])
        stream, cb_book, _ = self._setup(ticker)
        stream._dispatch_depth()
        cb_book.assert_not_called()

    def test_without_venue_quotes_the_ladder_is_the_smart_book(self):
        stream = _make_stream()
        cb_book = MagicMock()
        stream.subscribe("ACMR", MagicMock(), MagicMock(), sub_id="s1", on_book=cb_book)
        stream._depth_tickers["ACMR"] = SimpleNamespace(domBids=[_row(5.00, 300)], domAsks=[_row(5.05, 400)])
        stream._dispatch_depth()
        assert self._ladder(cb_book.call_args[0][0]) == ([(5.00, 300)], [(5.05, 400)])
