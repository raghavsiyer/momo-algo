"""
market_data/tape_merge.py

One time & sales tape from tastytrade's and IBKR's, each trade once — a port
of the tape in ScalpFeedVendor, the Quantower data vendor at
``~/.scalpfeed/project`` (Feed/TapeMerge.cs, PrintFilter.cs, PrintPlace.cs,
TapeWatch.cs; TastytradeMarketVendor.OnPrint/OnTrade/EmitPrint;
IbkrBridge.OnIbkrPrint/OnIbkrTapeActive; IbkrClient.OnTickByTick). Every rule
and constant below is the vendor's, with its reasons; see its IBKR-SEAMS.md
("What each feed really is", "Tape colors") for the measurements.

What each feed is:
  * tastytrade's TimeAndSale is dxFeed's Nasdaq Last Sale: trades on Nasdaq's
    own markets plus those reported to the FINRA/Nasdaq TRF. Exchange time
    stamps to the millisecond, the first copy of a shared print 90-95% of the
    time, sub-share fractions included, no ARCA/Cboe/NYSE/IEX/MEMX.
  * IBKR's tick-by-tick AllLast is the consolidated tape: every venue, whole
    shares, whole-second time stamps, batched ~0.4 s late, 5 symbols at once.

So in the merged tape an IBKR print on NASDAQ/BEX/PSX never shows (tastytrade
has it, sooner), any other exchange always shows from IBKR, and FINRA — IBKR's
one name for every TRF — shows unless tastytrade already has the same print.

Each print's side comes from where it hit the NBBO (IBKR's quote), judged
against the last 800 ms of quotes so a late quote doesn't paint it outside
the market. tastytrade's own BUY/SELL label only decides when there is no
quote at all: it labels every print, though 66-97% hit between its bid and ask.

Modes (config ``TAPE_MODE``, the vendor's ``tape`` in ibkr.json):
  merged      both feeds, each trade once (the vendor's setting)
  tastytrade  tastytrade's tape only
  ibkr        IBKR's tape where it has a tick-by-tick line; tastytrade's for
              symbols without one, and taking over when IBKR's goes quiet
While the tastytrade link is down, every mode behaves as ``ibkr`` (the vendor
has no IBKR-only fallback because its whole connection rides on tastytrade;
momo-algo trades on, so it falls back to IBKR's tape rather than dropping
NASDAQ prints nobody else carries).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from market_data.tradier_stream import LiveTimesale

logger = logging.getLogger(__name__)

TAPE_MODES = ("merged", "tastytrade", "ibkr")

# ---------------------------------------------------------------------------
# TapeMerge — each shared print once (Feed/TapeMerge.cs)
# ---------------------------------------------------------------------------

# The two feeds can round a sub-penny price differently (0.0591 and 0.059).
PRICE_TOLERANCE = 0.00011
# Both feeds stamp the trade itself, so they rarely disagree by more than a second.
SECONDS_TOLERANCE = 3
# Longer than the slowest IBKR delivery measured, 3.2 s behind tastytrade.
KEEP_SECS = 6.0
# tastytrade is the faster feed and IBKR only ever leads it briefly. Holding
# IBKR's print any longer lets a later trade at the same price and size on
# tastytrade be taken for it, and be lost.
KEEP_IBKR_FIRST_SECS = 1.0


@dataclass(slots=True)
class _Entry:
    price: float
    size: float
    seconds: int
    added: float


class TapeMerge:
    """Matches a FINRA print across the two feeds so it shows once.

    Whichever feed brings it first shows it. tastytrade's fractional trades
    (2.67 shares arrive as 2 and 0.67, IBKR has the 2) match on whole shares;
    a print under one share has no partner and simply shows.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._from_tastytrade: list[_Entry] = []
        self._from_ibkr: list[_Entry] = []
        self._carried_first: list[_Entry] = []

    def on_tastytrade_print(self, price: float, size: float, seconds: int, now: float) -> bool:
        """Whether to show it: no, if IBKR has already shown this print."""
        with self._lock:
            _expire(self._from_tastytrade, now, KEEP_SECS)
            _expire(self._from_ibkr, now, KEEP_IBKR_FIRST_SECS)
            _expire(self._carried_first, now, KEEP_IBKR_FIRST_SECS)
            if _take_match(self._from_ibkr, price, size, seconds):
                return False
            # IBKR's copy has been and gone, so nothing further can claim this print.
            if not _take_match(self._carried_first, price, size, seconds):
                self._from_tastytrade.append(_Entry(price, size, seconds, now))
            return True

    def on_ibkr_reported_print(self, price: float, size: float, seconds: int, now: float) -> bool:
        """A FINRA print. Whether to show it: no, if tastytrade has already shown it."""
        with self._lock:
            _expire(self._from_tastytrade, now, KEEP_SECS)
            _expire(self._from_ibkr, now, KEEP_IBKR_FIRST_SECS)
            if _take_match(self._from_tastytrade, price, size, seconds):
                return False
            self._from_ibkr.append(_Entry(price, size, seconds, now))
            return True

    def on_ibkr_carried_print(self, price: float, size: float, seconds: int, now: float) -> None:
        """A print on Nasdaq's own markets, which never shows from IBKR. It uses
        up its partner, so a FINRA print can't match it by mistake."""
        with self._lock:
            _expire(self._from_tastytrade, now, KEEP_SECS)
            _expire(self._carried_first, now, KEEP_IBKR_FIRST_SECS)
            if not _take_match(self._from_tastytrade, price, size, seconds):
                self._carried_first.append(_Entry(price, size, seconds, now))


def _expire(entries: list[_Entry], now: float, keep: float) -> None:
    old = 0
    while old < len(entries) and now - entries[old].added > keep:
        old += 1
    if old:
        del entries[:old]


def _take_match(entries: list[_Entry], price: float, size: float, seconds: int) -> bool:
    best = -1
    best_score = math.inf
    for i, entry in enumerate(entries):
        if not _same_size(entry.size, size) or not _same_price(entry.price, price):
            continue
        apart = abs(entry.seconds - seconds)
        if apart > SECONDS_TOLERANCE:
            continue
        score = apart + (100 if abs(entry.price - price) > 1e-7 else 0)
        if score < best_score:
            best, best_score = i, score
            if score == 0:
                break
    if best < 0:
        return False
    del entries[best]
    return True


def _same_size(a: float, b: float) -> bool:
    return math.floor(a) == math.floor(b)


def _same_price(a: float, b: float) -> bool:
    return abs(a - b) <= PRICE_TOLERANCE or _rounds_to(a, b) or _rounds_to(b, a)


def _rounds_to(exact: float, cents: float) -> bool:
    """IBKR rounds some stocks' prices to the cent: 735.4799 and 735.465 on
    tastytrade are 735.48 and 735.47 on IBKR. Under a dollar a cent is too wide
    to say anything, and it hasn't been seen there."""
    return (
        cents >= 1
        and abs(exact - cents) <= 0.00501
        and abs(cents * 100 - round(cents * 100)) < 1e-6
    )


# ---------------------------------------------------------------------------
# Print filter (Feed/PrintFilter.cs)
# ---------------------------------------------------------------------------

# Beyond the bid and ask carried on the print. In the vendor's recordings the
# one print past 2% was a 1-share SPY odd lot 4.9% out; AEHL's furthest was
# 1.89% in a fast move, and valid prints reach past 1% too, so a tighter guard
# would hide real trades in momentum names.
MAX_BEYOND_QUOTE = 0.03

REPORTED_VENUE = "FINRA"
_CARRIED_VENUES = frozenset({"NASDAQ", "BEX", "PSX"})
_OFF_MARKET_CODES = frozenset("47WVBPCNRUZ")


def is_believable(price: float, bid: float, ask: float, last_price: float) -> bool:
    """Within 3% of the quote, or of the last price when there's no quote.
    With nothing to compare against, a print that may be bad stays off."""
    low = bid if bid > 0 else last_price
    high = ask if ask > 0 else last_price
    if not (low > 0 and high > 0):
        return False
    return low * (1 - MAX_BEYOND_QUOTE) <= price <= high * (1 + MAX_BEYOND_QUOTE)


def tastytrade_print_shows(valid_tick: bool, normal_odd_lot: bool, price: float,
                           bid: float, ask: float, last_price: float) -> bool:
    """tastytrade marks odd lots invalid, and they're most of a stock's prints
    by count, so normal odd lots stay (if believable). The other invalid prints
    (average price, contingent, derivatively priced) can be far from the market
    and don't."""
    return bool(valid_tick) or (bool(normal_odd_lot) and is_believable(price, bid, ask, last_price))


def is_normal_odd_lot(conditions: str) -> bool:
    """Settlement, reason, extended-hours and detail codes, one character each:
    '@', then ' ' or 'F', then ' ' or 'T', then 'o'."""
    return (
        isinstance(conditions, str)
        and len(conditions) == 4
        and conditions[0] == "@"
        and conditions[1] in " F"
        and conditions[2] in " T"
        and conditions[3] == "o"
    )


def is_reopening(conditions: str) -> bool:
    """Condition 5, Re-Opening Prints, the same code on both SIPs' tapes."""
    return isinstance(conditions, str) and "5" in conditions


def is_carried_by_tastytrade(venue: str) -> bool:
    """IBKR venues whose prints are all on tastytrade's tape, and reach it
    sooner: Nasdaq's own markets. FINRA isn't here because only part of it is,
    which TapeMerge sorts out print by print."""
    return venue in _CARRIED_VENUES


def is_odd_lot(conditions: str) -> bool:
    """IBKR's condition I."""
    return "I" in (conditions or "")


def is_off_market(conditions: str) -> bool:
    """IBKR's codes for what tastytrade marks invalid and the filter hides,
    plus special settlement and out-of-sequence prints. Odd lots (I), sweeps (F)
    and extended hours (T) are ordinary trades and stay."""
    return any(code in _OFF_MARKET_CODES for code in (conditions or ""))


# ---------------------------------------------------------------------------
# Where a print hit the quote (Feed/PrintPlace.cs)
# ---------------------------------------------------------------------------

PLACE_UNKNOWN = "unknown"
PLACE_ABOVE_ASK = "above_ask"
PLACE_AT_ASK = "at_ask"
PLACE_BETWEEN = "between"
PLACE_AT_BID = "at_bid"
PLACE_BELOW_BID = "below_bid"

_TOL = 1e-9
QUOTE_WINDOW_SECS = 0.8


def place_of(price: float, bid: float, ask: float) -> str:
    """Unknown if either side has no market: a halt, a one-sided book, or no quote yet."""
    if not (bid > 0) or not (ask > 0):
        return PLACE_UNKNOWN
    if price > ask + _TOL:
        return PLACE_ABOVE_ASK
    if price >= ask - _TOL:
        return PLACE_AT_ASK
    if price < bid - _TOL:
        return PLACE_BELOW_BID
    return PLACE_AT_BID if price <= bid + _TOL else PLACE_BETWEEN


class QuoteWindow:
    """Judges a print against the NBBO, allowing for that quote reaching us late.

    A print only counts as outside if it's beyond the quote it came with and
    every quote of the last 800 ms too; short of that it counts as at that side.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quotes: list[tuple[float, float, float]] = []   # (t, bid, ask)

    def add(self, bid: float, ask: float, now: float) -> None:
        if not (bid > 0) or not (ask > 0):
            return
        with self._lock:
            # Each quote stands until the next, so the one just before the window still counts.
            while len(self._quotes) >= 2 and self._quotes[1][0] < now - QUOTE_WINDOW_SECS:
                del self._quotes[0]
            self._quotes.append((now, bid, ask))

    def judge(self, price: float, bid: float, ask: float,
              own_bid: float, own_ask: float, now: float) -> str:
        place = place_of(price, bid, ask)
        if place == PLACE_ABOVE_ASK and (price <= own_ask + _TOL or self._any_ask_at_or_above(price, now)):
            return PLACE_AT_ASK
        if place == PLACE_BELOW_BID and (
            (own_bid > 0 and price >= own_bid - _TOL) or self._any_bid_at_or_below(price, now)
        ):
            return PLACE_AT_BID
        return place

    def _stood(self, i: int, now: float) -> bool:
        return i == len(self._quotes) - 1 or self._quotes[i + 1][0] >= now - QUOTE_WINDOW_SECS

    def _any_ask_at_or_above(self, price: float, now: float) -> bool:
        with self._lock:
            return any(self._stood(i, now) and q[2] >= price - _TOL for i, q in enumerate(self._quotes))

    def _any_bid_at_or_below(self, price: float, now: float) -> bool:
        with self._lock:
            return any(self._stood(i, now) and q[1] <= price + _TOL for i, q in enumerate(self._quotes))


def to_aggressor(place: str, label: str) -> str:
    """The side momo-algo's order flow counts: at/above the ask buys, at/below
    the bid sells, between is neither. tastytrade's BUY/SELL label only decides
    when there's no quote at all."""
    if place in (PLACE_ABOVE_ASK, PLACE_AT_ASK):
        return "buy"
    if place in (PLACE_BELOW_BID, PLACE_AT_BID):
        return "sell"
    if place == PLACE_BETWEEN:
        return "mid"
    if label == "BUY":
        return "buy"
    if label == "SELL":
        return "sell"
    return "mid"


# ---------------------------------------------------------------------------
# TapeWatch (Feed/TapeWatch.cs) — ibkr mode only
# ---------------------------------------------------------------------------

TAPE_WATCH_LIMIT_SECS = 3.0
# Late reports can't show IBKR's tape has stopped, so only a fresh print starts its clock.
FRESH_PRINT_SECS = 5.0


class TapeWatch:
    """Notices one symbol's IBKR tape going quiet while its quotes carry on:
    a fresh tastytrade print not followed by any IBKR print within 3 s hands
    the symbol back to tastytrade's tape until IBKR prints again."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting_since = 0.0

    def on_expected_print(self, arrival: float) -> None:
        with self._lock:
            if self._waiting_since == 0.0:
                self._waiting_since = arrival

    def on_ibkr_print(self) -> None:
        with self._lock:
            self._waiting_since = 0.0

    def take_went_quiet(self, now: float) -> bool:
        with self._lock:
            since = self._waiting_since
            if since == 0.0 or now - since <= TAPE_WATCH_LIMIT_SECS:
                return False
            self._waiting_since = 0.0
            return True


# ---------------------------------------------------------------------------
# Rows from tastytrade (filled by market_data/tastytrade_stream.py)
# ---------------------------------------------------------------------------

EVENT_FLAG_REMOVE = 0x02


@dataclass(slots=True)
class TastytradePrint:
    """One dxFeed TimeAndSale event."""
    symbol: str
    time_ms: int
    price: float
    size: float
    bid: float = math.nan            # tastytrade's (Nasdaq's) quote at the print
    ask: float = math.nan
    exchange: str = ""               # one-letter dxFeed exchange code ('Q', 'O' overnight)
    aggressor: str = "UNDEFINED"     # BUY | SELL | UNDEFINED
    sale: str = "NEW"                # NEW | CORRECTION | CANCEL
    flags: int = 0
    index: int = 0
    valid_tick: bool = True
    extended_hours: bool = False
    conditions: str = ""
    arrival: float = 0.0             # time.monotonic() at frame arrival
    arrival_wall: float = 0.0        # time.time() at frame arrival

    @property
    def normal_odd_lot(self) -> bool:
        return is_normal_odd_lot(self.conditions)

    @property
    def reopening(self) -> bool:
        return is_reopening(self.conditions)


@dataclass(slots=True)
class TastytradeTrade:
    """One dxFeed Trade event (last sale + whole-market day volume)."""
    symbol: str
    time_ms: int
    price: float
    size: float
    day_volume: float = math.nan
    arrival: float = 0.0


# ---------------------------------------------------------------------------
# MergedTape — the coordinator
# ---------------------------------------------------------------------------

# Prints at least this late against the usual delay, this many in a row, are worth a warning.
LAG_WARN_SECS = 5.0
LAG_PRINTS_TO_WARN = 5
# Volume rising on Trade events with no TimeAndSale at all, this many times,
# this long after the link came up, switches prints to Trade events.
FALLBACK_GRACE_SECS = 15.0
UNPRINTED_TRADES_FOR_FALLBACK = 3
STATS_EVERY_SECS = 60.0


@dataclass
class _SymbolTape:
    symbol: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    merge: TapeMerge = field(default_factory=TapeMerge)
    recent_quotes: QuoteWindow = field(default_factory=QuoteWindow)
    tape_watch: TapeWatch = field(default_factory=TapeWatch)
    last_price: float = math.nan
    last_print_ts: Optional[datetime] = None
    last_print_index: int = 0
    seq: int = 0
    ibkr_bid: float = math.nan
    ibkr_ask: float = math.nan
    ibkr_tape: bool = False          # ibkr mode: IBKR's tape is the one showing
    ibkr_tape_lost: bool = False
    day_volume: float = math.nan
    last_trade_time_ms: int = 0


@dataclass
class _Stats:
    from_tastytrade: int = 0
    from_ibkr: int = 0
    reported_added: int = 0
    reported_already_shown: int = 0
    carried: int = 0
    off_market: int = 0
    too_far: int = 0
    odd_lots_shown: int = 0
    odd_lots_too_far: int = 0
    fractional_dropped: int = 0


class MergedTape:
    """Builds the tape momo-algo's subscribers see from both feeds.

    ``emit`` receives each print once, as a :class:`LiveTimesale` carrying its
    side (``aggressor``), where it hit the quote (``place``) and which feed
    brought it (``origin``). It is called with the symbol's lock held, from
    whichever feed's thread brought the print, so each symbol's prints reach
    subscribers one at a time and in time order.
    """

    def __init__(
        self,
        emit: Callable[[LiveTimesale], None],
        mode: str = "merged",
        *,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        if mode not in TAPE_MODES:
            raise ValueError(f"TAPE_MODE must be one of {TAPE_MODES}, not {mode!r}")
        self._emit = emit
        self.mode = mode
        self._clock = clock
        self._wall = wall_clock
        self._states: dict[str, _SymbolTape] = {}
        self._states_lock = threading.Lock()
        self._tastytrade_live = False
        self._live_since = 0.0
        self._prints_seen = False
        self._unprinted_trades = 0
        self._fallback_on = False
        self._lag_prints = 0
        self._lagging = False
        # The vendor's expected delay is 0 on the real-time feed (and the
        # tastytrade link refuses a delayed one), so lag is measured from 0.
        self._expected_delay = 0.0
        self._last_delay = 0.0
        self._last_stats = clock()
        self.stats = _Stats()

    # -- state ------------------------------------------------------------

    def state(self, symbol: str) -> _SymbolTape:
        st = self._states.get(symbol)
        if st is None:
            with self._states_lock:
                st = self._states.setdefault(symbol, _SymbolTape(symbol))
        return st

    def forget(self, symbol: str) -> None:
        with self._states_lock:
            self._states.pop(symbol, None)

    @property
    def effective_mode(self) -> str:
        """``ibkr`` while tastytrade's link is down, whatever the setting."""
        return self.mode if self._tastytrade_live else "ibkr"

    @property
    def fallback_on(self) -> bool:
        return self._fallback_on

    # -- link state ---------------------------------------------------------

    def set_tastytrade_live(self, live: bool) -> None:
        """The tastytrade link came up (feed configured) or went down."""
        if live and not self._tastytrade_live:
            self._live_since = self._clock()
            self._prints_seen = False
            self._unprinted_trades = 0
            self._fallback_on = False
            logger.info("[tape] tastytrade tape live — %s tape", self.mode)
        elif not live and self._tastytrade_live:
            logger.warning("[tape] tastytrade tape down — IBKR's tape only until it's back")
        self._tastytrade_live = live

    def on_ibkr_down(self) -> None:
        """IBKR's connection dropped: no NBBO, no IBKR tape."""
        for st in list(self._states.values()):
            st.ibkr_tape = False
            st.ibkr_bid = math.nan
            st.ibkr_ask = math.nan

    def set_ibkr_tape(self, symbol: str, on: bool) -> None:
        """IBKR's tick-by-tick line for *symbol* was granted (on) or lost."""
        if not on:
            st = self._states.get(symbol)
            if st is not None:
                st.ibkr_tape = False

    # -- IBKR -------------------------------------------------------------------

    def on_ibkr_quote(self, symbol: str, bid: float, ask: float) -> None:
        st = self.state(symbol)
        st.ibkr_bid = bid
        st.ibkr_ask = ask
        st.recent_quotes.add(bid, ask, self._clock())

    def on_ibkr_print(
        self,
        symbol: str,
        price: float,
        size: float,
        exchange_seconds: Optional[int],
        venue: str,
        conditions: str,
        arrival_ts: datetime,
    ) -> None:
        """One IBKR tick-by-tick AllLast print (IbkrClient.OnTickByTick and
        IbkrBridge.OnIbkrPrint/OnIbkrTapeActive)."""
        if price is None or size is None or math.isnan(price) or math.isnan(size):
            return
        mode = self.effective_mode
        if mode == "tastytrade":
            return
        now = self._clock()
        seconds = int(exchange_seconds) if exchange_seconds else int(arrival_ts.timestamp())
        st = self.state(symbol)
        venue = venue or ""
        conditions = conditions or ""

        carried = mode == "merged" and is_carried_by_tastytrade(venue)
        st.tape_watch.on_ibkr_print()
        if carried:
            st.merge.on_ibkr_carried_print(price, size, seconds, now)
            self.stats.carried += 1
            return
        if is_off_market(conditions):
            self.stats.off_market += 1
            return

        if mode == "ibkr":
            st.ibkr_tape = True
            if st.ibkr_tape_lost:
                st.ibkr_tape_lost = False
                logger.info("[tape] %s IBKR's tape is printing again, back on it", symbol)

        bid, ask = st.ibkr_bid, st.ibkr_ask
        if not is_believable(price, bid, ask, st.last_price):
            self.stats.too_far += 1
            return

        if mode == "merged" and venue == REPORTED_VENUE:
            if not st.merge.on_ibkr_reported_print(price, size, seconds, now):
                self.stats.reported_already_shown += 1
                return
            self.stats.reported_added += 1

        # Stamped on arrival: IBKR gives only whole seconds.
        self.stats.from_ibkr += 1
        self._emit_print(
            st, price, size, arrival_ts, label="UNDEFINED", bid=bid, ask=ask,
            own_bid=bid, own_ask=ask, origin="ibkr", exchange=venue, flag=conditions, now=now,
        )

    # -- tastytrade ---------------------------------------------------------------

    def on_tastytrade_print(self, row: TastytradePrint) -> None:
        """One TimeAndSale event (TastytradeMarketVendor.OnPrint)."""
        if (row.flags & EVENT_FLAG_REMOVE) or row.sale != "NEW" or math.isnan(row.price):
            return
        st = self.state(row.symbol)
        if not tastytrade_print_shows(row.valid_tick, row.normal_odd_lot, row.price,
                                      row.bid, row.ask, st.last_price):
            if row.normal_odd_lot:
                self.stats.odd_lots_too_far += 1
            return

        self._prints_seen = True
        if not row.valid_tick:
            self.stats.odd_lots_shown += 1

        exchange_wall = row.time_ms / 1000.0
        delay = (row.arrival_wall or self._wall()) - exchange_wall
        self._last_delay = delay
        self._lag_prints = self._lag_prints + 1 if delay - self._expected_delay > LAG_WARN_SECS else 0

        if row.index and row.index == st.last_print_index:
            return
        st.last_print_index = row.index

        now = row.arrival or self._clock()
        exchange_ts = datetime.fromtimestamp(exchange_wall, tz=timezone.utc)
        mode = self.effective_mode
        if mode == "merged":
            if st.merge.on_tastytrade_print(row.price, row.size, row.time_ms // 1000, now):
                self._emit_tastytrade(st, row, exchange_ts, now)
        elif not st.ibkr_tape or row.exchange == "O":
            # Overnight prints (code O) aren't on IBKR's SMART tape, so they show whichever tape is on.
            self._emit_tastytrade(st, row, exchange_ts, now)
        elif delay < FRESH_PRINT_SECS:
            st.tape_watch.on_expected_print(now)

    def on_tastytrade_trade(self, row: TastytradeTrade) -> None:
        """One Trade event (TastytradeMarketVendor.OnTrade): day volume, and
        prints when TimeAndSale isn't being served."""
        st = self.state(row.symbol)
        volume_rose = (not math.isnan(row.day_volume)) and (
            math.isnan(st.day_volume) or row.day_volume > st.day_volume
        )
        if not math.isnan(row.day_volume):
            st.day_volume = row.day_volume
        if volume_rose and not self._prints_seen:
            self._unprinted_trades += 1
        if (
            self._fallback_on
            and not st.ibkr_tape
            and row.time_ms > st.last_trade_time_ms
            and not math.isnan(row.price)
        ):
            ts = datetime.fromtimestamp(row.time_ms / 1000.0, tz=timezone.utc)
            self.stats.from_tastytrade += 1
            self._emit_print(
                st, row.price, row.size, ts, label="UNDEFINED", bid=math.nan, ask=math.nan,
                own_bid=math.nan, own_ask=math.nan, origin="tastytrade", exchange="", flag="",
                now=row.arrival or self._clock(),
            )
        st.last_trade_time_ms = max(st.last_trade_time_ms, row.time_ms)

    def _emit_tastytrade(self, st: _SymbolTape, row: TastytradePrint,
                         exchange_ts: datetime, now: float) -> None:
        self.stats.from_tastytrade += 1
        self._emit_print(
            st, row.price, row.size, exchange_ts, label=row.aggressor, bid=row.bid, ask=row.ask,
            own_bid=row.bid, own_ask=row.ask, origin="tastytrade",
            exchange=row.exchange, flag=row.conditions, now=now,
            session="extended" if row.extended_hours else "normal",
        )

    # -- the tape -------------------------------------------------------------------

    def _emit_print(
        self,
        st: _SymbolTape,
        price: float,
        size: float,
        ts: datetime,
        *,
        label: str,
        bid: float,
        ask: float,
        own_bid: float,
        own_ask: float,
        origin: str,
        exchange: str,
        flag: str,
        now: float,
        session: str = "normal",
    ) -> None:
        # Two feeds write this tape, and the hand-off stays inside the lock so
        # a print's time and its place in line can't disagree.
        with st.lock:
            st.last_price = price
            shares = int(math.floor(size))
            if shares < 1:
                # A sub-share print (tastytrade only) did trade, but momo-algo
                # counts whole shares: it moves the last price and stops here.
                self.stats.fractional_dropped += 1
                return
            # Strictly increasing per symbol, as Quantower requires.
            if st.last_print_ts is not None and ts <= st.last_print_ts:
                ts = st.last_print_ts + timedelta(microseconds=1)
            st.last_print_ts = ts

            # IBKR's quote is the NBBO. tastytrade's is Nasdaq's own, for when IBKR isn't quoting.
            has_nbbo = st.ibkr_bid > 0 and st.ibkr_ask > 0
            nbbo_bid = st.ibkr_bid if has_nbbo else bid
            nbbo_ask = st.ibkr_ask if has_nbbo else ask
            place = st.recent_quotes.judge(price, nbbo_bid, nbbo_ask, own_bid, own_ask, now)
            st.seq += 1
            event = LiveTimesale(
                event_type="timesale",
                symbol=st.symbol,
                exchange=exchange,
                last=price,
                bid=nbbo_bid if nbbo_bid > 0 else 0.0,
                ask=nbbo_ask if nbbo_ask > 0 else 0.0,
                size=shares,
                ts=ts,
                seq=st.seq,
                flag=flag,
                cancel=False,
                correction=False,
                session=session,
                aggressor=to_aggressor(place, label),
                place=place,
                origin=origin,
            )
            self._emit(event)

    # -- once a second ------------------------------------------------------------------

    def tick(self) -> None:
        """Housekeeping: IBKR tape watch (ibkr mode), the Trade-event fallback,
        the lag warning and a stats line a minute."""
        now = self._clock()
        if self.effective_mode == "ibkr" and self._tastytrade_live:
            for st in list(self._states.values()):
                if st.ibkr_tape and st.tape_watch.take_went_quiet(now):
                    st.ibkr_tape = False
                    st.ibkr_tape_lost = True
                    logger.info("[tape] %s IBKR's tape went quiet while tastytrade's printed, "
                                "showing tastytrade's until it prints", st.symbol)

        if self._tastytrade_live:
            if (
                not self._fallback_on
                and not self._prints_seen
                and now - self._live_since > FALLBACK_GRACE_SECS
                and self._unprinted_trades >= UNPRINTED_TRADES_FOR_FALLBACK
            ):
                self._fallback_on = True
                logger.warning("[tape] TimeAndSale isn't arriving; using Trade events for "
                               "prints (Trade can miss extended-hours prints)")
            elif self._fallback_on and self._prints_seen:
                self._fallback_on = False
                logger.info("[tape] TimeAndSale is arriving again")

        lagging = self._lag_prints >= LAG_PRINTS_TO_WARN
        if lagging and not self._lagging:
            logger.warning("[tape] tastytrade prints are %.1fs behind exchange time "
                           "(if the PC clock is off, this can be a false alarm)", self._last_delay)
        elif not lagging and self._lagging:
            logger.info("[tape] tastytrade prints have caught up with exchange time")
        self._lagging = lagging

        if now - self._last_stats >= STATS_EVERY_SECS:
            self._last_stats = now
            s = self.stats
            if s.from_tastytrade or s.from_ibkr:
                logger.info(
                    "[tape] 60s: mode=%s from tastytrade=%d from IBKR=%d (FINRA added=%d, "
                    "already on tastytrade's=%d) Nasdaq carried=%d off-market=%d too far=%d "
                    "odd lots shown=%d / too far=%d fractional=%d",
                    self.effective_mode, s.from_tastytrade, s.from_ibkr, s.reported_added,
                    s.reported_already_shown, s.carried, s.off_market, s.too_far,
                    s.odd_lots_shown, s.odd_lots_too_far, s.fractional_dropped,
                )
            self.stats = _Stats()
