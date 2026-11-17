"""Tests for market_data/tape_merge.py.

The TapeMerge, print-place and print-filter checks are ScalpFeedVendor's own
harness checks (~/.scalpfeed/project/ScalpFeedHarness/Program.cs: tapemerge,
printplace, printfilter), ported one for one so the merge behaves as the
Quantower tape does. The MergedTape scenarios cover the vendor's routing in
TastytradeMarketVendor.OnPrint/OnTrade/EmitPrint and IbkrBridge.OnIbkrPrint.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from market_data.tape_merge import (
    PLACE_ABOVE_ASK,
    PLACE_AT_ASK,
    PLACE_AT_BID,
    PLACE_BELOW_BID,
    PLACE_BETWEEN,
    PLACE_UNKNOWN,
    MergedTape,
    QuoteWindow,
    TapeMerge,
    TapeWatch,
    TastytradePrint,
    TastytradeTrade,
    place_of,
)
from market_data.tastytrade_stream import FEED_FIELDS, parse_feed_data

NAN = math.nan


# ---------------------------------------------------------------------------
# TapeMerge (harness "tapemerge")
# ---------------------------------------------------------------------------

def test_merge_each_feed_shows_a_shared_print_once():
    t = 1000.0
    m = TapeMerge()
    assert m.on_tastytrade_print(2.38, 100, 500, t) and not m.on_ibkr_reported_print(2.38, 100, 500, t + 1), \
        "a FINRA print tastytrade already showed doesn't show again"
    assert m.on_ibkr_reported_print(2.40, 100, 501, t) and not m.on_tastytrade_print(2.40, 100, 501, t + 1), \
        "whichever feed brings it first shows it"
    assert m.on_ibkr_reported_print(2.41, 300, 501, t) and m.on_ibkr_reported_print(2.41, 300, 501, t), \
        "a FINRA print tastytrade doesn't have shows, twice if it traded twice"


def test_merge_identical_prints_match_one_for_one():
    t = 1000.0
    m = TapeMerge()
    m.on_tastytrade_print(2.38, 100, 500, t)
    m.on_tastytrade_print(2.38, 100, 500, t)
    assert not m.on_ibkr_reported_print(2.38, 100, 500, t) and not m.on_ibkr_reported_print(2.38, 100, 500, t)
    assert m.on_ibkr_reported_print(2.38, 100, 500, t), "a third identical print is a different trade"


def test_merge_nasdaq_print_uses_up_its_partner():
    t = 1000.0
    m = TapeMerge()
    m.on_tastytrade_print(2.38, 100, 500, t)
    m.on_ibkr_carried_print(2.38, 100, 500, t)
    assert m.on_ibkr_reported_print(2.38, 100, 500, t)


def test_merge_price_and_size_rules():
    t = 1000.0
    m = TapeMerge()
    m.on_tastytrade_print(0.0591, 10731, 500, t)
    assert not m.on_ibkr_reported_print(0.059, 10731, 500, t), "sub-penny rounded differently still matches"
    m.on_tastytrade_print(0.0595, 100, 500, t)
    assert m.on_ibkr_reported_print(0.0592, 100, 500, t), "a different price doesn't match"
    m.on_tastytrade_print(5.25, 100, 500, t)
    assert m.on_ibkr_reported_print(5.25, 101, 500, t), "a different size doesn't match"
    m.on_tastytrade_print(7.25, 735.378, 500, t)
    assert not m.on_ibkr_reported_print(7.25, 735, 500, t), "a fractional size matches its whole shares"
    m.on_tastytrade_print(735.4799, 200, 500, t)
    assert not m.on_ibkr_reported_print(735.48, 200, 500, t), "a price IBKR rounded to the cent still matches"
    m.on_tastytrade_print(735.465, 89, 500, t)
    assert not m.on_ibkr_reported_print(735.47, 89, 500, t), "so does one rounded from exactly half a cent"
    m.on_tastytrade_print(0.0519, 500, 500, t)
    assert m.on_ibkr_reported_print(0.05, 500, 500, t), "under a dollar, being within a cent isn't a match"
    m.on_ibkr_carried_print(3.25, 150, 500, t)
    assert m.on_tastytrade_print(3.25, 150, 500, t) and m.on_ibkr_reported_print(3.25, 150, 500, t), \
        "a Nasdaq print IBKR brought first can't be taken for a FINRA one"
    assert m.on_ibkr_reported_print(4.25, 100, 500, t) and m.on_tastytrade_print(4.25, 100, 502, t + 2), \
        "a later trade at the same price and size on tastytrade isn't taken for an earlier FINRA print"
    m.on_tastytrade_print(735.46, 300, 500, t)
    assert m.on_ibkr_reported_print(735.47, 300, 500, t), "a different cent doesn't match"
    assert m.on_tastytrade_print(190.3988, 0.0677, 500, t), "a print under one share has no partner, and shows"


def test_merge_time_rules():
    t = 1000.0
    m = TapeMerge()
    m.on_tastytrade_print(2.38, 100, 500, t)
    assert m.on_ibkr_reported_print(2.38, 100, 504, t), "4 seconds apart is a different trade"
    assert not m.on_ibkr_reported_print(2.38, 100, 503, t), "3 seconds apart is the same trade"
    m.on_tastytrade_print(2.38, 100, 500, t)
    assert m.on_ibkr_reported_print(2.38, 100, 500, t + 7), "nothing is held on to past its time"


# ---------------------------------------------------------------------------
# Where a print hit the quote (harness "printplace")
# ---------------------------------------------------------------------------

def test_place_of():
    assert place_of(2.74, 2.70, 2.73) == PLACE_ABOVE_ASK
    assert place_of(2.73, 2.70, 2.73) == PLACE_AT_ASK
    assert place_of(2.7188, 2.70, 2.73) == PLACE_BETWEEN
    assert place_of(2.70, 2.70, 2.73) == PLACE_AT_BID
    assert place_of(2.69, 2.70, 2.73) == PLACE_BELOW_BID
    assert place_of(2.70, NAN, NAN) == PLACE_UNKNOWN
    assert place_of(2.70, -1, 2.73) == PLACE_UNKNOWN and place_of(2.70, 2.69, -1) == PLACE_UNKNOWN
    assert place_of(2.70, 2.70, 2.70) == PLACE_AT_ASK, "a locked market counts as at the ask"


def test_quote_window_judge():
    now = 1000.0
    w = QuoteWindow()
    w.add(2.70, 2.73, now - 5.0)
    assert w.judge(2.74, 2.70, 2.73, NAN, NAN, now) == PLACE_ABOVE_ASK
    assert w.judge(2.69, 2.70, 2.73, NAN, NAN, now) == PLACE_BELOW_BID
    assert w.judge(2.74, 2.70, 2.73, 2.60, 2.80, now) == PLACE_AT_ASK, "not beyond its own quote: only at the ask"
    assert w.judge(2.69, 2.70, 2.73, 2.60, 2.80, now) == PLACE_AT_BID
    assert w.judge(2.74, 2.70, 2.73, 2.60, 2.73, now) == PLACE_ABOVE_ASK, "beyond its own quote too: above"

    w = QuoteWindow()
    w.add(2.72, 2.75, now - 0.5)
    w.add(2.70, 2.73, now - 0.1)
    assert w.judge(2.74, 2.70, 2.73, NAN, NAN, now) == PLACE_AT_ASK, "an ask of the last moment covers it"
    assert w.judge(2.76, 2.70, 2.73, NAN, NAN, now) == PLACE_ABOVE_ASK
    w = QuoteWindow()
    w.add(2.68, 2.72, now - 0.5)
    w.add(2.70, 2.73, now - 0.1)
    assert w.judge(2.69, 2.70, 2.73, NAN, NAN, now) == PLACE_AT_BID
    assert w.judge(2.67, 2.70, 2.73, NAN, NAN, now) == PLACE_BELOW_BID

    w = QuoteWindow()
    w.add(2.72, 2.75, now - 2.0)
    w.add(2.70, 2.73, now - 1.5)
    assert w.judge(2.74, 2.70, 2.73, NAN, NAN, now) == PLACE_ABOVE_ASK, "an old quote no longer counts"
    w = QuoteWindow()
    w.add(-1, -1, now)
    assert w.judge(2.74, -1, -1, NAN, NAN, now) == PLACE_UNKNOWN, "a halt stays unknown"


# ---------------------------------------------------------------------------
# Print filter, through the real frame parser (harness "printfilter")
# ---------------------------------------------------------------------------

def _shows(conditions, valid, price, bid, ask, last):
    shown = []
    tape = MergedTape(shown.append, "tastytrade")
    tape.set_tastytrade_live(True)
    if not math.isnan(last):
        tape.state("X").last_price = last

    def num(v):
        return "NaN" if math.isnan(v) else v

    data = ["TimeAndSale", ["TimeAndSale", "X", 0, 1, 1000, "Q", num(price), 10, num(bid), num(ask),
                            "UNDEFINED", False, valid, "NEW", conditions]]
    parse_feed_data(data, FEED_FIELDS, tape.on_tastytrade_print, tape.on_tastytrade_trade, 1.0, 1.0)
    return len(shown) == 1


def test_print_filter():
    assert _shows("@   ", True, 10, 9.99, 10.01, NAN), "a valid print shows"
    assert _shows("@  o", False, 16.50, 16.33, 16.54, NAN), "a normal odd lot shows"
    assert _shows("@F o", False, 16.50, 16.33, 16.54, NAN), "an intermarket sweep odd lot shows"
    assert _shows("@ To", False, 16.50, 16.33, 16.54, NAN), "an extended-hours odd lot shows"
    assert _shows("@FTo", False, 16.50, 16.33, 16.54, NAN), "an extended-hours sweep odd lot shows"
    for off_market in ("@4 o", "@7 o", "@  W", "@4 W", "@7 V", "C  o", "@4 P", "@   "):
        assert not _shows(off_market, False, 16.50, 16.33, 16.54, NAN), off_market
    assert _shows("@  o", False, 16.76, 16.33, 16.54, NAN), "AEHL's real 1.3% odd lot in a fast move shows"
    assert not _shows("@  o", False, 772.525, 736.35, 736.4, NAN), "SPY's real 4.9% bad odd lot stays hidden"
    assert not _shows("@  o", False, 700, 736.35, 736.4, NAN)
    assert _shows("@  o", False, 10.1, NAN, NAN, 10), "with no quote, an odd lot near the last price shows"
    assert not _shows("@  o", False, 12, NAN, NAN, 10)
    assert not _shows("@  o", False, 10, NAN, NAN, NAN), "with nothing to compare, an odd lot stays hidden"
    assert _shows("@   ", True, 772.525, 736.35, 736.4, NAN), "valid prints aren't held to the guard"


# ---------------------------------------------------------------------------
# MergedTape routing
# ---------------------------------------------------------------------------

class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


T0 = datetime(2026, 11, 17, 13, 0, 0, tzinfo=timezone.utc)


def _tasty(symbol="ABC", price=5.00, size=100, *, at=T0, bid=4.99, ask=5.01, label="BUY",
           exchange="Q", index=0, valid=True, conditions="@   ", sale="NEW", flags=0, arrival=1000.0):
    return TastytradePrint(
        symbol=symbol, time_ms=int(at.timestamp() * 1000), price=price, size=size, bid=bid, ask=ask,
        exchange=exchange, aggressor=label, sale=sale, flags=flags, index=index, valid_tick=valid,
        conditions=conditions, arrival=arrival, arrival_wall=at.timestamp() + 0.05,
    )


def _tape(mode="merged", live=True):
    shown = []
    clock = _Clock()
    tape = MergedTape(shown.append, mode, clock=clock)
    tape.set_tastytrade_live(live)
    return tape, shown, clock


def test_merged_finra_print_shows_once_from_the_first_feed():
    tape, shown, _ = _tape()
    tape.on_ibkr_quote("ABC", 4.99, 5.00)
    tape.on_tastytrade_print(_tasty(price=5.00, size=100))
    tape.on_ibkr_print("ABC", 5.00, 100, int(T0.timestamp()), "FINRA", "", T0 + timedelta(seconds=0.4))
    assert len(shown) == 1
    ev = shown[0]
    assert ev.origin == "tastytrade" and ev.aggressor == "buy" and ev.place == PLACE_AT_ASK
    assert (ev.bid, ev.ask) == (4.99, 5.00), "the event carries IBKR's NBBO, not tastytrade's quote"
    assert ev.ts == T0, "tastytrade prints keep their exchange time"


def test_merged_finra_print_tastytrade_lacks_shows_from_ibkr():
    tape, shown, _ = _tape()
    tape.on_ibkr_quote("ABC", 4.99, 5.01)
    tape.on_ibkr_print("ABC", 4.99, 300, int(T0.timestamp()), "FINRA", "", T0)
    assert [(e.origin, e.aggressor, e.exchange) for e in shown] == [("ibkr", "sell", "FINRA")]


def test_merged_never_shows_nasdaq_from_ibkr_and_always_shows_other_venues():
    tape, shown, _ = _tape()
    tape.on_ibkr_quote("ABC", 4.99, 5.01)
    for venue in ("NASDAQ", "BEX", "PSX"):
        tape.on_ibkr_print("ABC", 5.00, 100, int(T0.timestamp()), venue, "", T0)
    assert shown == []
    arrival = T0 + timedelta(seconds=0.4)
    tape.on_ibkr_print("ABC", 5.01, 200, int(T0.timestamp()), "ARCA", "", arrival)
    assert [(e.origin, e.exchange, e.aggressor, e.size) for e in shown] == [("ibkr", "ARCA", "buy", 200)]
    assert shown[0].ts == arrival, "IBKR prints are stamped on arrival (it gives whole seconds)"


def test_ibkr_off_market_and_unbelievable_prints_stay_off():
    tape, shown, _ = _tape()
    tape.on_ibkr_quote("ABC", 4.99, 5.01)
    tape.on_ibkr_print("ABC", 5.00, 100, int(T0.timestamp()), "ARCA", "4", T0)
    tape.on_ibkr_print("ABC", 5.30, 100, int(T0.timestamp()), "ARCA", "", T0)
    assert shown == []
    tape.on_ibkr_print("ABC", 5.00, 30, int(T0.timestamp()), "ARCA", "TI", T0)
    assert len(shown) == 1, "odd lots (I) and extended hours (T) are ordinary trades"


def test_times_strictly_increase_per_symbol():
    tape, shown, _ = _tape()
    tape.on_ibkr_quote("ABC", 4.99, 5.01)
    tape.on_ibkr_print("ABC", 5.01, 100, int(T0.timestamp()), "ARCA", "", T0 + timedelta(seconds=1))
    tape.on_tastytrade_print(_tasty(price=5.00, size=100, at=T0))
    assert shown[1].ts == shown[0].ts + timedelta(microseconds=1)


def test_side_comes_from_the_nbbo_and_the_label_only_without_a_quote():
    tape, shown, _ = _tape()
    tape.on_ibkr_quote("ABC", 4.99, 5.01)
    tape.on_tastytrade_print(_tasty(price=5.00, label="BUY"))
    assert (shown[-1].place, shown[-1].aggressor) == (PLACE_BETWEEN, "mid"), "a print between is neither"
    tape2, shown2, _ = _tape()
    tape2.on_tastytrade_print(_tasty(symbol="XYZ", price=5.00, bid=NAN, ask=NAN, label="SELL"))
    assert (shown2[-1].place, shown2[-1].aggressor) == (PLACE_UNKNOWN, "sell")


def test_a_late_quote_does_not_paint_a_print_outside_the_market():
    tape, shown, clock = _tape()
    tape.on_ibkr_quote("ABC", 5.00, 5.04)
    clock.t += 0.5
    tape.on_ibkr_quote("ABC", 4.98, 5.02)
    tape.on_ibkr_print("ABC", 5.03, 100, int(T0.timestamp()), "ARCA", "", T0)
    assert shown[-1].place == PLACE_AT_ASK and shown[-1].aggressor == "buy"


def test_link_down_falls_back_to_ibkrs_whole_tape():
    tape, shown, _ = _tape(live=False)
    tape.on_ibkr_quote("ABC", 4.99, 5.01)
    tape.on_ibkr_print("ABC", 5.00, 100, int(T0.timestamp()), "NASDAQ", "", T0)
    assert [e.exchange for e in shown] == ["NASDAQ"]
    assert tape.effective_mode == "ibkr"


def test_ibkr_mode_hands_back_to_tastytrade_when_ibkrs_tape_goes_quiet():
    tape, shown, clock = _tape(mode="ibkr")
    tape.on_ibkr_quote("ABC", 4.99, 5.01)
    tape.on_ibkr_print("ABC", 5.00, 100, int(T0.timestamp()), "NASDAQ", "", T0)
    assert len(shown) == 1 and tape.state("ABC").ibkr_tape
    tape.on_tastytrade_print(_tasty(price=5.00, arrival=clock.t))
    assert len(shown) == 1, "tastytrade's print waits for IBKR's while IBKR's tape is on"
    tape.on_tastytrade_print(_tasty(price=5.00, exchange="O", index=7, arrival=clock.t))
    assert len(shown) == 2, "overnight (O) prints show whichever tape is on"
    clock.t += 3.5
    tape.tick()
    assert not tape.state("ABC").ibkr_tape
    tape.on_tastytrade_print(_tasty(price=5.00, index=9, arrival=clock.t))
    assert len(shown) == 3 and shown[-1].origin == "tastytrade"


def test_tastytrade_mode_ignores_ibkr_prints():
    tape, shown, _ = _tape(mode="tastytrade")
    tape.on_ibkr_print("ABC", 5.00, 100, int(T0.timestamp()), "ARCA", "", T0)
    tape.on_tastytrade_print(_tasty())
    assert [e.origin for e in shown] == ["tastytrade"]


def test_removed_cancelled_and_repeated_prints_are_skipped():
    tape, shown, _ = _tape(mode="tastytrade")
    tape.on_tastytrade_print(_tasty(flags=0x02))
    tape.on_tastytrade_print(_tasty(sale="CANCEL"))
    tape.on_tastytrade_print(_tasty(sale="CORRECTION"))
    tape.on_tastytrade_print(_tasty(index=42))
    tape.on_tastytrade_print(_tasty(index=42))
    assert len(shown) == 1


def test_fractional_shares():
    tape, shown, _ = _tape(mode="tastytrade")
    tape.on_tastytrade_print(_tasty(price=5.10, size=0.67))
    assert shown == [] and tape.state("ABC").last_price == 5.10, "under a share moves the last price only"
    tape.on_tastytrade_print(_tasty(price=5.11, size=2.67, index=3))
    assert shown[-1].size == 2


def test_trade_events_stand_in_when_timeandsale_is_not_served():
    tape, shown, clock = _tape(mode="tastytrade")
    for i in range(3):
        tape.on_tastytrade_trade(TastytradeTrade("ABC", int(T0.timestamp() * 1000) + i, 5.0, 100, day_volume=1000 + i))
    tape.tick()
    assert not tape.fallback_on, "not before the grace period"
    clock.t += 16
    tape.tick()
    assert tape.fallback_on
    tape.on_tastytrade_trade(TastytradeTrade("ABC", int(T0.timestamp() * 1000) + 10, 5.05, 200, day_volume=1300))
    assert [(e.last, e.size, e.origin) for e in shown] == [(5.05, 200, "tastytrade")]
    tape.on_tastytrade_print(_tasty(price=5.06, index=5))
    tape.tick()
    assert not tape.fallback_on, "TimeAndSale arriving again turns it off"


def test_tape_watch():
    w = TapeWatch()
    assert not w.take_went_quiet(100.0)
    w.on_expected_print(100.0)
    w.on_expected_print(101.0)
    assert not w.take_went_quiet(103.0)
    assert w.take_went_quiet(103.5), "the first expected print starts the clock"
    assert not w.take_went_quiet(110.0), "said once per silence"
    w.on_expected_print(120.0)
    w.on_ibkr_print()
    assert not w.take_went_quiet(130.0)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        MergedTape(lambda e: None, "both")
