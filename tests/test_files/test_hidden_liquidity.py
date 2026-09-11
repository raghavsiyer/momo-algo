"""tests/test_hidden_liquidity.py — Hidden seller detection on OrderFlowTracker."""

from datetime import datetime, timedelta, timezone

import pytest

from exec.order_flow import OrderFlowTracker
from market_data.tradier_stream import LiveQuote, LiveTimesale


def _quote(bid: float, bid_size: int, ask: float, ask_size: int, ts: datetime) -> LiveQuote:
    return LiveQuote(
        event_type="quote", symbol="TEST",
        bid=bid, bid_size=bid_size, bid_exchange="X", bid_ts=ts,
        ask=ask, ask_size=ask_size, ask_exchange="X", ask_ts=ts,
    )


def _print(price: float, size: int, bid: float, ask: float, ts: datetime, seq: int) -> LiveTimesale:
    return LiveTimesale(
        event_type="timesale", symbol="TEST", exchange="X",
        last=price, bid=bid, ask=ask, size=size, ts=ts, seq=seq,
        flag="@", cancel=False, correction=False, session="normal",
    )


class TestHiddenSeller:
    def test_no_data_returns_zero(self):
        ofi = OrderFlowTracker()
        assert ofi.hidden_seller == 0.0

    def test_classic_iceberg_at_ask(self):
        """Ask pinned at $5.00, ask_size stays ~500, but 5,000+ shares
        of buy-side aggressor volume hits it.  Should detect hidden seller."""
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)

        # 16 quote samples over 16s with stable ask price + size
        for i in range(16):
            ofi.on_quote(_quote(4.99, 500, 5.00, 500, t0 + timedelta(seconds=i)))

        # Many buy-side aggressor prints lifting the ask
        for i in range(20):
            ofi.on_timesale(_print(5.00, 300, 4.99, 5.00,
                                   t0 + timedelta(seconds=i), seq=i))

        # 20 prints × 300 = 6,000 shares vs avg_ask_size 500 → ratio 12 → clipped to 1.0
        assert ofi.hidden_seller == pytest.approx(1.0)

    def test_no_hidden_seller_when_ask_advances(self):
        """If the ask price moves up, buyers were just chasing — no hidden seller."""
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)

        for i in range(16):
            ofi.on_quote(_quote(4.99, 500, 5.00 + i * 0.01, 500,
                                 t0 + timedelta(seconds=i)))
        for i in range(20):
            ofi.on_timesale(_print(5.00 + i * 0.01, 300, 4.99, 5.00 + i * 0.01,
                                   t0 + timedelta(seconds=i), seq=i))

        assert ofi.hidden_seller == 0.0

    def test_low_ratio_partial_intensity(self):
        """Buy volume only ~2x ask size → intensity ≈ 0.25."""
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)
        for i in range(16):
            ofi.on_quote(_quote(4.99, 1000, 5.00, 1000, t0 + timedelta(seconds=i)))
        # 4 prints × 500 = 2,000 vs avg 1000 → ratio 2 → (2-1)/4 = 0.25
        for i in range(4):
            ofi.on_timesale(_print(5.00, 500, 4.99, 5.00,
                                   t0 + timedelta(seconds=i * 2), seq=i))
        assert ofi.hidden_seller == pytest.approx(0.25, abs=0.01)

    def test_no_aggressor_volume_returns_zero(self):
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)
        for i in range(16):
            ofi.on_quote(_quote(4.99, 500, 5.00, 500, t0 + timedelta(seconds=i)))
        # No prints
        assert ofi.hidden_seller == 0.0


def _synthetic_quote(bid: float, ask: float, ts: datetime) -> LiveQuote:
    """Replay-style quote: real NBBO prices, placeholder 1-share sizes."""
    return LiveQuote(
        event_type="quote", symbol="TEST",
        bid=bid, bid_size=1, bid_exchange="X", bid_ts=ts,
        ask=ask, ask_size=1, ask_exchange="X", ask_ts=ts,
        synthetic=True,
    )


def _book(ts: datetime, bids: list, asks: list):
    from market_data.ibkr_stream import OrderBookLevel, OrderBookSnapshot
    return OrderBookSnapshot(
        symbol="TEST",
        bids=[OrderBookLevel("TEST", "bid", i, p, s, "", ts) for i, (p, s) in enumerate(bids)],
        asks=[OrderBookLevel("TEST", "ask", i, p, s, "", ts) for i, (p, s) in enumerate(asks)],
        ts=ts,
    )


class TestHiddenSellerSampleSources:
    """Replay fidelity: synthetic quotes never feed ask samples; the book's
    best ask does (decimated) while depth is active."""

    def test_synthetic_quotes_never_trip_hidden_seller(self):
        """The replay bug this guards: 1-share placeholder ask sizes made
        any non-advancing ask + a few buys read intensity 1.0, spuriously
        failing the required no_hidden_seller gate."""
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)
        for i in range(16):
            ofi.on_quote(_synthetic_quote(4.99, 5.00, t0 + timedelta(seconds=i)))
        for i in range(20):
            ofi.on_timesale(_print(5.00, 300, 4.99, 5.00,
                                   t0 + timedelta(seconds=i), seq=i))
        assert ofi.hidden_seller == 0.0

    def test_synthetic_quotes_do_not_move_book_lean(self):
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)
        ofi.on_quote(_quote(4.99, 900, 5.00, 100, t0))          # real: lean +0.8
        lean = ofi.book_lean
        ofi.on_quote(_synthetic_quote(4.99, 5.00, t0 + timedelta(seconds=1)))
        assert ofi.book_lean == lean                             # 1/1 sizes ignored

    def test_book_best_ask_drives_hidden_seller_when_depth_active(self):
        """Iceberg visible from L2 alone: best ask pinned with ~500 shown
        while buys keep printing — replay with book capture must detect it."""
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)
        for i in range(16):
            ofi.on_book(_book(t0 + timedelta(seconds=i),
                              [(4.99, 500)], [(5.00, 500), (5.01, 300)]))
        for i in range(20):
            ofi.on_timesale(_print(5.00, 300, 4.99, 5.00,
                                   t0 + timedelta(seconds=i), seq=i))
        assert ofi.hidden_seller == pytest.approx(1.0)

    def test_l1_sampling_suppressed_while_depth_active(self):
        """With the book flowing, real L1 quotes must not double-feed the
        sample deque (book supersedes — same rule as L1 OFI)."""
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)
        ofi.on_book(_book(t0, [(4.99, 500)], [(5.00, 500)]))
        n = len(ofi._ask_samples)
        ofi.on_quote(_quote(4.99, 500, 5.00, 9999, t0 + timedelta(seconds=0.1)))
        assert len(ofi._ask_samples) == n

    def test_book_samples_are_decimated(self):
        """A burst of snapshots within the decimation interval yields one
        sample — bounds hidden_seller's per-print read cost on busy books."""
        ofi = OrderFlowTracker()
        t0 = datetime(2026, 4, 27, 14, 30, tzinfo=timezone.utc)
        for i in range(10):  # 10 snapshots over 90ms
            ofi.on_book(_book(t0 + timedelta(milliseconds=i * 10),
                              [(4.99, 500)], [(5.00, 500)]))
        assert len(ofi._ask_samples) == 1
