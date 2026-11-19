"""
market_data/ibkr_stream.py

Interactive Brokers streaming client for live market data via ``ib_insync``.

  SHARED STREAM  (``IBKRSharedStream``)
    - subscribe()     Per-symbol T&S + L1 quotes, plus L2 depth on request
    - unsubscribe()   Drops a callback; the symbol's subs end with the last one

One connection carries every symbol. The per-stream clients this module used
to expose (stream_timesales/quotes/level2/combined/order_book) each opened
their own connection, which burned a client id per symbol and spent the
account-wide tick-by-tick quota outside any single accounting point; they were
unused and have been removed. Reach the stream through
``provider.get_shared_stream()`` rather than constructing one directly.

Returns the same ``LiveQuote`` / ``LiveTimesale`` dataclasses used by the
Tradier streaming client, plus a new ``OrderBookLevel`` dataclass for L2
depth.

Usage:
    from market_data.provider import get_shared_stream

    stream = get_shared_stream()
    sub_id = stream.subscribe(
        "ACMR", on_timesale=lambda e: print(e.last, e.size),
    )
    ...
    stream.unsubscribe("ACMR", sub_id)
"""

import asyncio
import copy
import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

try:
    asyncio.get_event_loop_policy().get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    from ib_insync import IB, Stock, Ticker
except ImportError:  # graceful when ib_insync is not installed
    IB = Stock = Ticker = None  # type: ignore[assignment,misc]


try:
    from ib_insync.objects import DOMLevel as _DOMLevel, MktDepthData as _MktDepthData
except ImportError:  # ib_insync absent or stubbed (tests) — same shape
    from typing import NamedTuple as _NamedTuple

    class _DOMLevel(_NamedTuple):       # type: ignore[no-redef]
        price: float
        size: float
        marketMaker: str

    class _MktDepthData(_NamedTuple):   # type: ignore[no-redef]
        time: object
        position: int
        marketMaker: str
        operation: int
        side: int
        price: float
        size: float


# Backlog cap for per-update book states awaiting dispatch. Worst measured
# churn is ~130 updates/sec on a single busy book, so this holds ~30s — far
# beyond any dispatch-loop stall worth surviving. deque(maxlen) drops the
# oldest states first; the newest entry always carries the current book, so
# an overflow costs intermediate fidelity, never correctness.
_BOOK_STATE_MAXLEN = 4096


def _update_mkt_depth_l2(
    self, reqId: int, position: int, marketMaker: str, operation: int,
    side: int, price: float, size: float, isSmartDepth: bool = False,
) -> None:
    """Bounds-checked replacement for ``ib_insync.Wrapper.updateMktDepthL2``.

    ib_insync 0.9.86 assigns ``dom[position]`` with no bounds check on an
    update (operation 1) — only its delete branch is guarded — so an update
    for a level the local book never saw inserted raises IndexError.
    SMART-aggregated depth does exactly that shortly after a (re)subscribe,
    and ``_start_depth`` resubscribes whenever slot priority reassigns a
    depth line (it used to happen constantly under newest-wins LRU).

    The decoder catches that IndexError and the stream survives, which is the
    trap: the update is silently dropped and the level keeps a stale size for
    the life of the book.  ``_dispatch_depth`` feeds those levels into OFI, so
    the real cost is a corrupt book rather than a visible failure.

    Padding rather than appending keeps every later position index aligned
    with what the Gateway thinks it sent.  Placeholders carry price 0.0 /
    size 0, which ``_aggregate_dom`` already skips, so they never reach the
    ladder.  Bound onto Wrapper by :func:`_patch_ib_insync_depth`.
    """
    # operation: 0 = insert, 1 = update, 2 = delete;  side: 0 = ask, 1 = bid
    ticker = self.reqId2Ticker.get(reqId)
    if ticker is None or position < 0:
        # Upstream indexes reqId2Ticker directly, so an update crossing with
        # our own cancelMktDepth raises KeyError instead. Same class of
        # handler crash, same nothing-to-do answer.
        return

    dom = ticker.domBids if side else ticker.domAsks
    if operation == 0:
        if position > len(dom):
            dom.extend(_DOMLevel(0.0, 0, "") for _ in range(position - len(dom)))
        dom.insert(position, _DOMLevel(price, size, marketMaker))
    elif operation == 1:
        if position >= len(dom):
            dom.extend(_DOMLevel(0.0, 0, "")
                       for _ in range(position + 1 - len(dom)))
        dom[position] = _DOMLevel(price, size, marketMaker)
    elif operation == 2:
        if position < len(dom):
            level = dom.pop(position)
            price = level.price
            size = 0

    ticker.domTicks.append(_MktDepthData(
        self.lastTime, position, marketMaker, operation, side, price, size))
    self.pendingTickers.add(ticker)

    # IBKR delivers every intermediate book state (hold-and-release, not
    # conflation), but a dispatch loop that samples domBids/domAsks sees only
    # where the book ended up between iterations — measured: 67-83% of depth
    # updates discarded before OFI, which measures the flow of changes, not
    # the book's final shape. The shared stream installs this buffer on
    # tickers it owns depth for; capturing here, at decode time, is also
    # immune to ib_insync clearing domTicks on the next TCP batch. Rows are
    # immutable (_DOMLevel replaced, never mutated), so shallow copies pin
    # the exact post-update state.
    states = getattr(ticker, "_momo_book_states", None)
    if states is not None:
        states.append((self.lastTime, list(ticker.domBids), list(ticker.domAsks)))


def _patch_ib_insync_depth() -> None:
    """Bind :func:`_update_mkt_depth_l2` over ib_insync's crashing version.

    0.9.86 is the final release — the project is archived — so the IndexError
    it raises will never be fixed upstream.  A no-op when ib_insync is absent
    or stubbed: there is then no real handler to correct.
    """
    try:
        from ib_insync.wrapper import Wrapper
    except ImportError:
        return
    if getattr(Wrapper.updateMktDepthL2, "_momo_patched", False):
        return
    _update_mkt_depth_l2._momo_patched = True  # type: ignore[attr-defined]
    Wrapper.updateMktDepthL2 = _update_mkt_depth_l2


# Backlog cap for tape prints awaiting dispatch. Normal tape is ~20-30
# prints/sec; halt-resume bursts can exceed 1k/sec, which this still holds
# for several seconds of dispatch-loop stall. Overflow drops oldest-first.
_TAPE_EVENT_MAXLEN = 8192

# The unpatched Wrapper.tickByTickAllLast, captured by _patch_ib_insync_tape.
# None when ib_insync is absent or stubbed (tests).
_orig_tick_by_tick = None


class _BankedPrint:
    """A decoded tape print plus IBKR's own trade time.

    ib_insync stamps every print with the TCP batch's arrival time and drops
    the ``time`` field IBKR sends (whole Unix seconds of the trade). The merged
    tape matches FINRA prints across feeds on that trade time, so it's kept
    here. Attribute reads fall through to the print itself.
    """

    __slots__ = ("tick", "exchange_time")

    def __init__(self, tick: object, exchange_time: Optional[int]) -> None:
        self.tick = tick
        self.exchange_time = exchange_time

    def __getattr__(self, name: str):
        return getattr(self.tick, name)


def _tick_by_tick_all_last(
    self, reqId: int, tickType: int, time: int, price: float, size: float,
    tickAttribLast: object, exchange: str, specialConditions: str,
) -> None:
    """Delegating wrapper over ``Wrapper.tickByTickAllLast`` that also banks
    each print into a stream-owned buffer.

    ``waitOnUpdate``-loop harvesting of ``ticker.tickByTicks`` loses prints —
    ib_insync documents it: when two TCP batches are processed inside one
    wake, ``tcpDataArrived`` for the second clears the first batch's list
    before the loop reads it. The Gateway's tape flushes arrive as several
    back-to-back socket writes, so the loss concentrates at flush onset —
    the bursts the tape gates care most about. Banking the print here, at
    decode time, is immune by construction (same pattern as the depth-state
    buffer in :func:`_update_mkt_depth_l2`). The shared stream installs the
    buffer in ``_start_ts``; tickers without one (probes, other clients on
    this Wrapper) are untouched.
    """
    if _orig_tick_by_tick is not None:
        _orig_tick_by_tick(self, reqId, tickType, time, price, size,
                           tickAttribLast, exchange, specialConditions)
    ticker = self.reqId2Ticker.get(reqId)
    if ticker is None:
        return
    buf = getattr(ticker, "_momo_tape_events", None)
    if buf is not None and ticker.tickByTicks:
        try:
            exchange_time = int(time) if time else None
        except (TypeError, ValueError):
            exchange_time = None
        buf.append(_BankedPrint(ticker.tickByTicks[-1], exchange_time))


def _patch_ib_insync_tape() -> None:
    """Bind :func:`_tick_by_tick_all_last` over the stock tape handler."""
    global _orig_tick_by_tick
    try:
        from ib_insync.wrapper import Wrapper
    except ImportError:
        return
    if getattr(Wrapper.tickByTickAllLast, "_momo_patched", False):
        return
    _orig_tick_by_tick = Wrapper.tickByTickAllLast
    _tick_by_tick_all_last._momo_patched = True  # type: ignore[attr-defined]
    Wrapper.tickByTickAllLast = _tick_by_tick_all_last


_patch_ib_insync_depth()
_patch_ib_insync_tape()

from config import (
    IBKR_ENABLED,
    IBKR_L2_ROWS,
    IBKR_L2_MAX_DEPTH_SUBS,
    IBKR_L2_VENUE_QUOTES,
    IBKR_MAX_TICK_SUBS,
    IBKR_TICK_RESUB_COOLDOWN_SECS,
    IBKR_SUB_DENIAL_BACKOFF_SECS,
    IBKR_SUB_DENIAL_BACKOFF_MAX_SECS,
    IBKR_TAPE_CAP_RECOVERY_SECS,
    IBKR_TAPE_RANK_WINDOW_SECS,
    IBKR_TAPE_RANK_MARGIN,
    IBKR_TAPE_MIN_HOLD_SECS,
    IBKR_TAPE_REBALANCE_SECS,
    IBKR_DEPTH_IDLE_DOLLAR_RATE,
    IBKR_DEPTH_IDLE_RELEASE_SECS,
    IBKR_DEPTH_RELEASE_COOLDOWN_SECS,
    TAPE_MODE,
    TASTYTRADE_CREDENTIALS_PATH,
    TASTYTRADE_TAPE_ENABLED,
)
from market_data.ibkr_rest import get_ib, release_ib, _make_contract
from market_data.tape_merge import MergedTape
from market_data.tradier_stream import (
    LiveQuote,
    LiveTimesale,
)

logger = logging.getLogger(__name__)

# The merged tape's housekeeping (tape watch, Trade fallback, stats) runs this often.
_TAPE_TICK_SECS = 1.0

# "Max number of tick-by-tick requests has been reached" — arrives async,
# never as a raised exception.
_TICK_LIMIT_ERROR_CODE = 10190
# "Max number (N) of market depth requests has been reached" — same async
# refusal pattern for reqMktDepth.
_DEPTH_LIMIT_ERROR_CODE = 309


def _backoff_secs(denials: int) -> float:
    """Exponential backoff for quota-refused subscriptions."""
    return min(IBKR_SUB_DENIAL_BACKOFF_SECS * (2 ** max(0, denials - 1)),
               IBKR_SUB_DENIAL_BACKOFF_MAX_SECS)

# L1 volume sampling for the tape-slot ranking: one sample per second gives
# 60 points across the default 60s rank window; the deque holds a little
# more so the window is always fully covered.
_VOL_SAMPLE_SECS = 1.0
_VOL_HIST_MAXLEN = 90


# ---------------------------------------------------------------------------
# Level 2 / Order Book dataclass — new, not in Tradier
# ---------------------------------------------------------------------------

@dataclass
class OrderBookLevel:
    """Single level in the order book (bid or ask side)."""
    symbol: str
    side: str               # "bid" | "ask"
    position: int           # 0-based depth position
    price: float
    size: int
    market_maker: str       # MPID for NASDAQ; exchange code for others
    ts: datetime


@dataclass
class OrderBookSnapshot:
    """Full order book snapshot for a symbol."""
    symbol: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    ts: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


def _aggregate_dom(rows: list, descending: bool) -> list[tuple[float, int]]:
    """Aggregate raw DOM rows (per-MM) into a price ladder.

    IBKR smart depth exposes one row per market maker; multiple MMs can
    quote the same price. Sums sizes by price and returns [(price, size)]
    sorted best-first (descending for bids, ascending for asks).
    """
    by_price: dict[float, int] = {}
    for row in rows:
        price = _ns(getattr(row, "price", 0.0))
        size = _ni(getattr(row, "size", 0))
        if price <= 0 or size <= 0:
            continue
        by_price[price] = by_price.get(price, 0) + size
    return sorted(by_price.items(), key=lambda kv: kv[0], reverse=descending)


def _has_rows(rows: list) -> bool:
    """True when *rows* holds at least one real (priced, sized) DOM row."""
    return any(_ns(getattr(r, "price", 0.0)) > 0 and _ni(getattr(r, "size", 0)) > 0 for r in rows)


def _with_venue_rows(rows: list, venue_rows: list) -> list:
    """Smart-depth *rows* plus each venue's top-of-book row (IBKR_L2_VENUE_QUOTES).

    A venue the smart book already carries is skipped, so the rows can't count
    twice if the account later gains that venue's full book.
    """
    if not venue_rows:
        return rows
    in_book = {getattr(r, "marketMaker", "") for r in rows}
    return list(rows) + [v for v in venue_rows if v.marketMaker not in in_book]


def _directed_contract(contract: object, venue: str) -> object:
    """A copy of *contract* routed to one exchange, for that venue's own quote.

    A copy, never the original: ib_insync keys tickers by contract object, so
    reusing the SMART contract would hand back the symbol's L1/depth ticker.
    """
    directed = copy.copy(contract)
    directed.exchange = venue
    return directed


# ---------------------------------------------------------------------------
# NaN helpers
# ---------------------------------------------------------------------------

def _ns(val: float) -> float:
    """NaN-safe: return 0.0 for NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return float(val)


def _ni(val: float) -> int:
    """NaN-safe: return 0 for NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0
    return int(val)


# ---------------------------------------------------------------------------
# Shared / multiplexed stream  (equivalent to Tradier SharedStreamClient)
# ---------------------------------------------------------------------------

class IBKRSharedStream:
    """Single-connection multiplexed streaming client for IBKR.

    Manages per-symbol ``reqTickByTickData`` (T&S) and ``reqMktData``
    (L1 quotes) subscriptions on the shared ``IB`` connection returned
    by :func:`get_ib`.

    Thread-safe: :meth:`subscribe` / :meth:`unsubscribe` / :meth:`set_priority`
    may be called from any thread.  All IB API mutations happen on a
    dedicated daemon thread that owns the event-loop polling.

    Slot allocation: IBKR caps concurrent tape and depth lines, so both are
    allocated by pin first (``SLOT_PIN_*``, pushed by the scorer via
    :meth:`set_priority` — protects live setups and open positions whose
    tape is quiet by nature), then by L1 cumulative-volume rate, which is
    measurable for every streamed symbol regardless of slots.  A periodic
    rebalance lets slotless symbols win a tape line back when they out-rank
    the weakest holder by a margin.

    Interface mirrors ``SharedStreamClient`` from ``tradier_stream.py``
    so the pipeline and scorer can swap providers transparently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # sym -> {sub_id: (on_timesale_cb, on_quote_cb, on_book_cb|None)}
        self._subs: dict[str, dict[str, tuple[Callable, Callable, Optional[Callable]]]] = {}
        self._stopped = False

        # Managed exclusively by the stream thread
        self._contracts: dict[str, object] = {}   # symbol -> qualified Stock
        self._ts_tickers: dict[str, object] = {}  # symbol -> tick-by-tick Ticker
        self._q_tickers: dict[str, object] = {}   # symbol -> reqMktData Ticker
        self._prev_bid: dict[str, float] = {}
        self._prev_ask: dict[str, float] = {}
        self._prev_l1_last: dict[str, float] = {}
        self._seq: dict[str, int] = {}
        # L2 depth: IBKR grants a limited number of depth lines (default 3).
        # Free lines idle-fill first-come; when full, only a strictly higher
        # pin evicts. Losers degrade gracefully back to L1-only order flow.
        self._depth_tickers: dict[str, object] = {}   # symbol -> reqMktDepth Ticker
        self._depth_lru: list[str] = []               # grant order, oldest first
        self._prev_book_sig: dict[str, tuple] = {}
        # Directed top-of-book quotes for the venues smart depth leaves out
        # (IBKR_L2_VENUE_QUOTES), held exactly as long as the depth line.
        self._venue_quotes: dict[str, list[tuple[str, object, object]]] = {}  # sym -> [(venue, contract, ticker)]
        self._prev_venue_sig: dict[str, tuple] = {}
        self._depth_idle_since: dict[str, float] = {} # unpinned+quiet since (idle release)
        # Tape (tick-by-tick) subs carry their own quota, the same size as
        # the depth one. Allocated by (pin, L1 volume rate); the cap can be
        # revised down at runtime when IBKR rejects a request with 10190.
        self._ts_lru: list[str] = []                  # grant order, oldest first
        self._ts_limit: int = max(1, IBKR_MAX_TICK_SUBS)
        self._ts_denied: set[str] = set()             # hit 10190, dead registration to reap
        self._ts_last_req: dict[str, float] = {}      # 15s re-request guard
        self._ts_granted_at: dict[str, float] = {}    # min-hold before rate eviction
        # Refusal backoff: a 10190/309 usually means the quota is held outside
        # this process (another session, or lines a dead session leaked at the
        # Gateway) — refused symbols wait out an exponential backoff instead
        # of re-challenging every rebalance pass.
        self._ts_retry_after: dict[str, float] = {}   # sym -> monotonic not-before
        self._ts_denials: dict[str, int] = {}         # sym -> consecutive 10190s
        self._ts_last_denial: float = 0.0             # cap-recovery quiet timer
        self._depth_denied: set[str] = set()          # hit 309, dead registration to reap
        self._depth_retry_after: dict[str, float] = {}
        self._depth_denials: dict[str, int] = {}
        # Slot priority. Pins arrive via set_priority(); volume history is
        # sampled from the L1 quote tickers once per second.
        self._pins: dict[str, int] = {}
        self._vol_hist: dict[str, deque] = {}         # sym -> deque[(mono_ts, cum_vol)]
        self._last_vol_sample: float = 0.0
        self._last_ts_rebalance: float = 0.0
        # Merged time & sales (market_data/tape_merge.py): built on the stream
        # thread by _start_tape once IBKR is up, so tests that stub _run never
        # open a tastytrade socket. None = IBKR's prints pass through as before.
        self._tape: Optional[MergedTape] = None
        self._tasty = None
        self._last_tape_tick: float = 0.0

        self._cmd_q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="ibkr-shared-stream",
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def subscribe(
        self,
        symbol: str,
        on_timesale: Callable,
        on_quote: Callable,
        sub_id: str,
        on_book: Optional[Callable] = None,
    ) -> None:
        """Register callbacks for *symbol*. Starts market-data subscription
        on the IB connection if this is the first subscriber for the symbol.

        ``on_book`` (optional) receives :class:`OrderBookSnapshot` updates
        from an L2 market-depth subscription. Depth lines are limited
        (IBKR_L2_MAX_DEPTH_SUBS) and demand-gated: a line is only granted
        while the symbol is pinned or its L1 dollar-volume rate clears the
        relevance floor (a cold newcomer starts L1-only and wins a line at
        a rebalance tick once its rate proves out), and a full house only
        turns over to a strictly higher pin.
        """
        with self._lock:
            is_new = symbol not in self._subs
            self._subs.setdefault(symbol, {})[sub_id] = (on_timesale, on_quote, on_book)
        if is_new:
            self._cmd_q.put(("add", symbol))
        elif on_book is not None:
            # Symbol already streaming L1/T&S — ensure depth is requested too.
            self._cmd_q.put(("add_depth", symbol))

    def unsubscribe(self, symbol: str, sub_id: str) -> None:
        """Remove a subscription.  Cancels IB data if no subscribers remain."""
        remove = False
        with self._lock:
            if symbol in self._subs:
                self._subs[symbol].pop(sub_id, None)
                if not self._subs[symbol]:
                    del self._subs[symbol]
                    remove = True
        if remove:
            self._cmd_q.put(("remove", symbol))

    def set_priority(self, symbol: str, pin: int) -> None:
        """Set *symbol*'s slot pin (``SLOT_PIN_*``).

        Pins decide who holds the capped tape/depth lines: a raised pin
        immediately tries to claim a line (evicting a lower-pinned holder),
        a lowered pin releases nothing by itself — tape lines turn over only
        when a challenger wants the slot, and depth lines additionally get
        idle-released once unpinned AND quiet (see _release_idle_depth).
        """
        self._cmd_q.put(("prio", symbol, max(0, int(pin))))

    def stop(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        """Tear down all subscriptions, close the IB connection, exit the thread.

        Queues the stop command *before* setting ``_stopped`` so the loop
        drains it and runs ``_stop_all`` — setting the flag first lets the
        ``while`` condition win the race and leaves subscriptions open on a
        connection nobody will close.
        """
        with self._lock:
            self._subs.clear()
        self._cmd_q.put(("stop",))
        self._stopped = True
        if wait and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("IBKRSharedStream: stream thread did not exit within %ss", timeout)

    # ------------------------------------------------------------------
    # Stream thread internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            ib = get_ib()
        except Exception as exc:
            logger.error("IBKRSharedStream: connection failed: %s", exc)
            release_ib()
            return
        try:
            ib.errorEvent += self._on_ib_error
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("IBKRSharedStream: could not hook errorEvent: %s", exc)
        self._start_tape()
        try:
            self._loop(ib)
        finally:
            self._stop_tape()
            # Whether we exited via "stop", an unhandled error, or the
            # _stopped flag, this thread owns an IB connection that IBKR only
            # reclaims when the socket closes.
            release_ib()

    # ------------------------------------------------------------------
    # Merged tape (tastytrade + IBKR)
    # ------------------------------------------------------------------

    def _start_tape(self) -> None:
        """Bring up the merged tape and tastytrade's link, when configured.

        Any problem (setting off, no credentials, bad mode) leaves _tape None,
        so IBKR's prints pass through unchanged.
        """
        if not TASTYTRADE_TAPE_ENABLED:
            logger.info("IBKRSharedStream: tastytrade tape off (TASTYTRADE_TAPE_ENABLED) — IBKR's tape only")
            return
        try:
            from market_data.tastytrade_stream import DxLinkTapeLink, TastytradeApi
            api = TastytradeApi(TASTYTRADE_CREDENTIALS_PATH)
            tape = MergedTape(self._dispatch_print, TAPE_MODE)
        except Exception as exc:
            logger.warning("IBKRSharedStream: no tastytrade tape (%s) — IBKR's tape only", exc)
            return
        link = DxLinkTapeLink(api, tape.on_tastytrade_print, tape.on_tastytrade_trade,
                              tape.set_tastytrade_live)
        for symbol in list(self._contracts):
            link.add(symbol)
        self._tape = tape
        self._tasty = link
        link.start()
        logger.info("IBKRSharedStream: %s tape (tastytrade + IBKR)", TAPE_MODE)

    def _stop_tape(self) -> None:
        link, self._tasty = self._tasty, None
        if link is not None:
            link.stop()
        if self._tape is not None:
            self._tape.on_ibkr_down()

    def _dispatch_print(self, event: LiveTimesale) -> None:
        """Hand one merged-tape print to the symbol's subscribers.

        Runs on whichever feed's thread brought the print, with the tape's
        per-symbol lock held, so a symbol's prints arrive one at a time and
        in time order.
        """
        with self._lock:
            cbs = dict(self._subs.get(event.symbol, {}))
        for on_ts, _, _ in cbs.values():
            try:
                on_ts(event)
            except Exception:
                logger.exception("IBKRSharedStream: timesale callback failed for %s", event.symbol)

    def _loop(self, ib: object) -> None:
        while not self._stopped:
            # --- drain command queue ---
            while True:
                try:
                    cmd = self._cmd_q.get_nowait()
                except queue.Empty:
                    break
                if cmd[0] == "add":
                    self._start_symbol(ib, cmd[1])
                elif cmd[0] == "add_depth":
                    self._start_depth(ib, cmd[1])
                elif cmd[0] == "prio":
                    self._set_pin(ib, cmd[1], cmd[2])
                elif cmd[0] == "remove":
                    self._stop_symbol(ib, cmd[1])
                elif cmd[0] == "stop":
                    self._stop_all(ib)
                    return

            # --- let ib_insync process network data ---
            # Spin guard: waitOnUpdate returns immediately (without blocking)
            # when the connection is lost or the update event is already set.
            # Without a floor this loop would hot-spin a full core.
            t0 = time.monotonic()
            ib.waitOnUpdate(timeout=0.1)
            if time.monotonic() - t0 < 0.01:
                time.sleep(0.05)

            # --- reap dead registrations IBKR refused (Error 10190/309) ---
            if self._ts_denied or self._depth_denied:
                self._reap_denied_subs(ib)

            # --- dispatch tick-by-tick T&S ---
            tape = self._tape
            if tape is not None:
                # Quotes first, so the merged tape judges this pass's prints
                # against the newest NBBO (the second call below then finds
                # nothing new).
                self._dispatch_quotes()
                now_mono = time.monotonic()
                if now_mono - self._last_tape_tick >= _TAPE_TICK_SECS:
                    self._last_tape_tick = now_mono
                    tape.tick()
            for sym, ticker in list(self._ts_tickers.items()):
                for tick in self._pending_tape_events(ticker):
                    ts = tick.time or datetime.now(tz=timezone.utc)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if tape is not None:
                        # The merged tape decides what shows and dispatches it.
                        tape.on_ibkr_print(
                            sym, float(tick.price), float(tick.size),
                            getattr(tick, "exchange_time", None),
                            tick.exchange or "", tick.specialConditions or "", ts,
                        )
                        continue
                    self._seq[sym] = self._seq.get(sym, 0) + 1
                    q_tkr = self._q_tickers.get(sym)
                    bid = _ns(q_tkr.bid) if q_tkr else 0.0
                    ask = _ns(q_tkr.ask) if q_tkr else 0.0
                    event = LiveTimesale(
                        event_type="timesale",
                        symbol=sym,
                        exchange=tick.exchange or "",
                        last=float(tick.price),
                        bid=bid,
                        ask=ask,
                        size=int(tick.size),
                        ts=ts,
                        seq=self._seq[sym],
                        flag=tick.specialConditions or "",
                        cancel=False,
                        correction=False,
                        session="normal",
                    )
                    with self._lock:
                        cbs = dict(self._subs.get(sym, {}))
                    for on_ts, _, _ in cbs.values():
                        on_ts(event)

            # --- dispatch L1 quote changes ---
            self._dispatch_quotes()

            # --- dispatch L2 depth changes ---
            self._dispatch_depth()

            # --- slot upkeep: volume sampling + tape re-ranking ---
            self._sample_volumes()
            self._maybe_rebalance_ts(ib)

    def _dispatch_quotes(self) -> None:
        """Emit a LiveQuote whenever a symbol's bid, ask, or last changes.

        ``last`` matters as its own trigger: it is the only price feed a
        symbol without a tape line has, and its trades can move without
        the NBBO budging (prints inside the spread).
        """
        for sym, ticker in list(self._q_tickers.items()):
            bid = _ns(getattr(ticker, "bid", 0.0))
            ask = _ns(getattr(ticker, "ask", 0.0))
            last = _ns(getattr(ticker, "last", 0.0))
            if (
                bid == self._prev_bid.get(sym, 0.0)
                and ask == self._prev_ask.get(sym, 0.0)
                and last == self._prev_l1_last.get(sym, 0.0)
            ):
                continue
            self._prev_bid[sym] = bid
            self._prev_ask[sym] = ask
            self._prev_l1_last[sym] = last
            tape = self._tape
            if tape is not None:
                # IBKR's quote is the NBBO the merged tape sides prints by
                # (-1 / NaN on a side means no market there).
                tape.on_ibkr_quote(sym, bid, ask)
            now = datetime.now(tz=timezone.utc)
            event = LiveQuote(
                event_type="quote",
                symbol=sym,
                bid=bid,
                bid_size=_ni(getattr(ticker, "bidSize", 0)),
                bid_exchange=getattr(ticker, "bidExchange", "") or "",
                bid_ts=now,
                ask=ask,
                ask_size=_ni(getattr(ticker, "askSize", 0)),
                ask_exchange=getattr(ticker, "askExchange", "") or "",
                ask_ts=now,
                last=last,
                last_size=_ni(getattr(ticker, "lastSize", 0)),
            )
            with self._lock:
                cbs = dict(self._subs.get(sym, {}))
            for _, on_quote, _ in cbs.values():
                on_quote(event)

    def _dispatch_depth(self) -> None:
        for sym, ticker in list(self._depth_tickers.items()):
            venue_bids, venue_asks = self._venue_rows(sym)
            venue_sig = (tuple(venue_bids), tuple(venue_asks))
            states = self._pending_book_states(ticker)
            if not states and venue_sig != self._prev_venue_sig.get(sym, ((), ())):
                # A venue's quote moved with no depth update: the ladder still
                # changed, so it goes out once more against the current book.
                states = [(None,
                           getattr(ticker, "domBids", None) or [],
                           getattr(ticker, "domAsks", None) or [])]
            self._prev_venue_sig[sym] = venue_sig
            for ts, raw_bids, raw_asks in states:
                # Venue rows never stand in for a book IBKR hasn't sent yet:
                # alone they'd pass for the whole market.
                if not _has_rows(raw_bids) and not _has_rows(raw_asks):
                    continue
                # Venue quotes are as of this pass; buffered states can be up to
                # one loop iteration older. Both feeds tick far slower than that.
                bids = _aggregate_dom(_with_venue_rows(raw_bids, venue_bids), descending=True)
                asks = _aggregate_dom(_with_venue_rows(raw_asks, venue_asks), descending=False)
                if not bids and not asks:
                    continue
                sig = (tuple(bids), tuple(asks))
                if sig == self._prev_book_sig.get(sym):
                    continue
                self._prev_book_sig[sym] = sig
                now = ts or datetime.now(tz=timezone.utc)
                snapshot = OrderBookSnapshot(
                    symbol=sym,
                    bids=[
                        OrderBookLevel(symbol=sym, side="bid", position=i,
                                       price=p, size=s, market_maker="", ts=now)
                        for i, (p, s) in enumerate(bids)
                    ],
                    asks=[
                        OrderBookLevel(symbol=sym, side="ask", position=i,
                                       price=p, size=s, market_maker="", ts=now)
                        for i, (p, s) in enumerate(asks)
                    ],
                    ts=now,
                )
                with self._lock:
                    cbs = dict(self._subs.get(sym, {}))
                for _, _, on_book in cbs.values():
                    if on_book is not None:
                        on_book(snapshot)

    def _venue_rows(self, symbol: str) -> tuple[list, list]:
        """Each directed venue quote as one bid row and one ask row.

        A side with no market (IBKR's -1, NaN before the first tick, or no
        size) gives no row; a refused request (354, line limit) never ticks
        and so never gives one either.
        """
        bids: list = []
        asks: list = []
        for venue, _, ticker in self._venue_quotes.get(symbol, ()):
            bid = _ns(getattr(ticker, "bid", 0.0))
            ask = _ns(getattr(ticker, "ask", 0.0))
            bid_size = _ni(getattr(ticker, "bidSize", 0))
            ask_size = _ni(getattr(ticker, "askSize", 0))
            if bid > 0 and bid_size > 0:
                bids.append(_DOMLevel(bid, bid_size, venue))
            if ask > 0 and ask_size > 0:
                asks.append(_DOMLevel(ask, ask_size, venue))
        return bids, asks

    @staticmethod
    def _pending_book_states(ticker: object) -> list[tuple]:
        """Book states awaiting dispatch, oldest first: (ts, bids, asks).

        With the wrapper buffer installed (real ib_insync path via
        ``_start_depth``) this drains every post-update state captured since
        the last loop iteration, so OFI sees each intermediate book state
        rather than one end-of-interval sample. The timestamps are the
        wrapper's TCP-arrival times (``lastTime``); states from the same
        Gateway flush share one, which is when they genuinely arrived.

        Without the buffer (stubbed ib_insync, tests, tickers not started by
        this stream) it falls back to sampling the current aggregate state
        once per call — the legacy behavior.
        """
        buf = getattr(ticker, "_momo_book_states", None)
        if isinstance(buf, deque):
            states = []
            while buf:
                states.append(buf.popleft())
            return states
        return [(None,
                 getattr(ticker, "domBids", None) or [],
                 getattr(ticker, "domAsks", None) or [])]

    @staticmethod
    def _pending_tape_events(ticker: object) -> list:
        """Tape prints awaiting dispatch, oldest first.

        With the wrapper buffer installed (real ib_insync path via
        ``_start_ts``) this drains prints banked at decode time, so a second
        TCP batch processed inside one ``waitOnUpdate`` wake cannot clear
        prints before they are read — the documented ``waitOnUpdate`` +
        ``ticker.tickByTicks`` loss. Without it (stubbed ib_insync, tests)
        it falls back to the legacy read-and-clear of ``tickByTicks``.
        """
        buf = getattr(ticker, "_momo_tape_events", None)
        if isinstance(buf, deque):
            ticks = []
            while buf:
                ticks.append(buf.popleft())
            return ticks
        ticks = list(getattr(ticker, "tickByTicks", None) or [])
        if ticks:
            ticker.tickByTicks.clear()
        return ticks

    def _start_symbol(self, ib: object, symbol: str) -> None:
        if symbol in self._contracts:
            return
        try:
            contract = _make_contract(symbol)
            ib.qualifyContracts(contract)
            # L1 quotes first: they are cheap, uncapped, and the fallback when
            # the tape quota is full, so a symbol is never left with nothing.
            q_ticker = ib.reqMktData(contract, "", snapshot=False)
            self._contracts[symbol] = contract
            self._q_tickers[symbol] = q_ticker
            self._prev_bid[symbol] = 0.0
            self._prev_ask[symbol] = 0.0
            self._prev_l1_last[symbol] = 0.0
            self._seq[symbol] = 0
            logger.debug("IBKRSharedStream: subscribed %s", symbol)
        except Exception as exc:
            logger.warning("IBKRSharedStream: subscribe %s failed: %s", symbol, exc)
            return
        if self._tasty is not None:
            # tastytrade's tape has no slot limit: every streamed symbol gets it.
            self._tasty.add(symbol)
        self._start_ts(ib, symbol)
        with self._lock:
            wants_depth = any(
                cb[2] is not None for cb in self._subs.get(symbol, {}).values()
            )
        if wants_depth:
            self._start_depth(ib, symbol)

    def _start_ts(self, ib: object, symbol: str, *,
                  ignore_backoff: bool = False) -> bool:
        """Request the tick-by-tick tape for *symbol*, evicting the weakest
        eligible holder when the quota is full.  Returns True when the
        request went out.

        Overrunning the quota is silent: IBKR answers ``reqTickByTickData``
        normally and only later emits Error 10190, leaving a Ticker
        registered that never receives a print — so the cap has to be
        enforced here rather than discovered from a failure.

        ``ignore_backoff`` lets a pin raise (an explicit strategy claim)
        attempt a refused symbol immediately; everyone else waits out the
        denial backoff.
        """
        if symbol in self._ts_tickers:
            return False
        contract = self._contracts.get(symbol)
        if contract is None:
            return False
        now = time.monotonic()
        if not ignore_backoff and now < self._ts_retry_after.get(symbol, 0.0):
            return False
        # "No more than 1 tick-by-tick request can be made for the same
        # instrument within 15 seconds" — eviction churn could otherwise
        # re-request a symbol immediately and trip a pacing violation.
        last = self._ts_last_req.get(symbol)
        if last is not None and (now - last) < IBKR_TICK_RESUB_COOLDOWN_SECS:
            return False
        while len(self._ts_tickers) >= max(self._ts_limit, 1):
            evict = self._pick_ts_victim(symbol)
            if evict is None:
                logger.info(
                    "IBKRSharedStream: no tape slot for %s (cap %d, holders "
                    "pinned) — L1 quotes only",
                    symbol, self._ts_limit,
                )
                return False
            logger.info("IBKRSharedStream: tape slots full (%d) — evicting %s for %s",
                        self._ts_limit, evict, symbol)
            self._stop_ts(ib, evict)
        try:
            ticker = ib.reqTickByTickData(contract, "AllLast")
            # Per-print buffer for the patched wrapper (see
            # _tick_by_tick_all_last). Always fresh: ib_insync reuses Ticker
            # objects per contract, and prints queued before an
            # eviction/resubscribe gap are stale.
            ticker._momo_tape_events = deque(maxlen=_TAPE_EVENT_MAXLEN)
            self._ts_tickers[symbol] = ticker
            self._ts_lru.append(symbol)
            self._ts_last_req[symbol] = time.monotonic()
            self._ts_granted_at[symbol] = time.monotonic()
            logger.debug("IBKRSharedStream: tape subscribed %s", symbol)
            return True
        except Exception as exc:
            logger.warning("IBKRSharedStream: tape subscribe %s failed: %s", symbol, exc)
            return False

    def _reap_denied_subs(self, ib: object) -> None:
        """Drop registrations IBKR refused (Error 10190 tape / 309 depth).

        A refused request still leaves a locally registered Ticker that will
        never emit — cancelled promptly so it doesn't hold a local slot while
        delivering nothing. Deliberately NO immediate retry: a refusal means
        the quota is held outside this process (another session on the
        account, or subscriptions a dead session leaked at the Gateway, which
        stay counted for many minutes — observed 2026-08-05). The symbol
        re-enters through the rebalancer once its backoff expires; retrying
        here just re-triggers the refusal and spams the log.
        """
        denied, self._ts_denied = self._ts_denied, set()
        for symbol in denied:
            self._stop_ts(ib, symbol)          # drop the dead registration
        dead, self._depth_denied = self._depth_denied, set()
        for symbol in dead:
            self._stop_depth(ib, symbol)

    def _stop_ts(self, ib: object, symbol: str) -> None:
        """Cancel only the tape for *symbol*; L1 quotes and depth stay up."""
        ticker = self._ts_tickers.pop(symbol, None)
        self._ts_granted_at.pop(symbol, None)
        if symbol in self._ts_lru:
            self._ts_lru.remove(symbol)
        if self._tape is not None:
            # No IBKR tape line: tastytrade's tape carries the symbol alone.
            self._tape.set_ibkr_tape(symbol, False)
        if ticker is None:
            return
        try:
            del ticker._momo_tape_events
        except AttributeError:
            pass
        contract = self._contracts.get(symbol)
        if contract is not None:
            try:
                ib.cancelTickByTickData(contract, "AllLast")
            except Exception:
                pass
        logger.debug("IBKRSharedStream: tape unsubscribed %s", symbol)

    def _on_ib_error(self, reqId, errorCode, errorString, contract) -> None:
        """Catch quota refusals (10190 tape / 309 depth) and record them.

        Fires on the stream thread (ib_insync dispatches during waitOnUpdate),
        the same thread that owns the ticker dicts — so this only records the
        rejection; the loop reaps the dead registration, and the rebalancer
        re-admits the symbol after its backoff.
        """
        if errorCode not in (_TICK_LIMIT_ERROR_CODE, _DEPTH_LIMIT_ERROR_CODE):
            return
        symbol = str(getattr(contract, "symbol", "") or "").upper()
        if not symbol:
            return
        now = time.monotonic()
        if errorCode == _DEPTH_LIMIT_ERROR_CODE:
            if symbol not in self._depth_tickers:
                return
            count = self._depth_denials.get(symbol, 0) + 1
            self._depth_denials[symbol] = count
            backoff = _backoff_secs(count)
            self._depth_retry_after[symbol] = now + backoff
            self._depth_denied.add(symbol)
            logger.warning(
                "IBKRSharedStream: depth refused %s (Error 309, denial #%d) — "
                "quota held elsewhere on the account; next attempt in %.0fs",
                symbol, count, backoff,
            )
            return
        # The refused request is already counted in _ts_tickers, so at most
        # len-1 of them are actually live. Never raise the limit here, only
        # lower it — recovery probes happen in the rebalancer after a quiet
        # period.
        learned = max(1, len(self._ts_tickers) - 1)
        if learned < self._ts_limit:
            logger.warning(
                "IBKRSharedStream: tape quota rejected %s (Error 10190) — "
                "lowering cap %d -> %d", symbol, self._ts_limit, learned,
            )
            self._ts_limit = learned
        count = self._ts_denials.get(symbol, 0) + 1
        self._ts_denials[symbol] = count
        backoff = _backoff_secs(count)
        self._ts_retry_after[symbol] = now + backoff
        self._ts_last_denial = now
        logger.warning(
            "IBKRSharedStream: tape refused %s (Error 10190, denial #%d) — "
            "quota held elsewhere on the account; next attempt in %.0fs "
            "(cap %d, holding %d)",
            symbol, count, backoff, self._ts_limit, max(0, len(self._ts_tickers) - 1),
        )
        self._ts_denied.add(symbol)

    # ------------------------------------------------------------------
    # Slot priority — pins + L1 volume-rate ranking
    # ------------------------------------------------------------------

    def _set_pin(self, ib: object, symbol: str, pin: int) -> None:
        """Apply a pin pushed via :meth:`set_priority` (stream thread only).

        A raise claims lines immediately — that is the latency-critical
        path (a breakout just fired and entry confirmation needs the book).
        A drop is lazy here: the tape line stays until a challenger with
        standing wants it, and the depth line stays until either a
        higher-pinned challenger evicts it or the idle-release pass finds
        the symbol unpinned AND quiet (see _release_idle_depth).
        """
        old = self._pins.get(symbol, 0)
        if pin == old:
            return
        self._pins[symbol] = pin
        if pin < old or symbol not in self._contracts:
            return
        if symbol not in self._ts_tickers:
            self._start_ts(ib, symbol, ignore_backoff=True)
        with self._lock:
            wants_depth = any(
                cb[2] is not None for cb in self._subs.get(symbol, {}).values()
            )
        if wants_depth and symbol not in self._depth_tickers:
            self._start_depth(ib, symbol, ignore_backoff=True)

    def _sample_volumes(self) -> None:
        """Record (ts, cumulative day volume) off every L1 quote ticker.

        This is the slot-independent activity measure: reqMktData's volume
        field keeps updating for symbols that hold no tape line, so the
        ranking never starves a slotless symbol the way prints/sec would
        (no tape slot -> no prints -> no score -> no slot, forever).
        """
        now = time.monotonic()
        if now - self._last_vol_sample < _VOL_SAMPLE_SECS:
            return
        self._last_vol_sample = now
        for sym, ticker in self._q_tickers.items():
            vol = _ni(getattr(ticker, "volume", 0))
            if vol <= 0:
                continue
            hist = self._vol_hist.setdefault(sym, deque(maxlen=_VOL_HIST_MAXLEN))
            if hist and vol < hist[-1][1]:
                hist.clear()   # feed reset / session rollover — restart window
            hist.append((now, vol))

    def _vol_rate(self, symbol: str) -> float:
        """Volume per second averaged over the FULL rank window.

        The denominator is the window, never the sampled span: a symbol
        with only seconds of history (just started streaming, or history
        reset on a volume correction) gets its observed delta spread over
        the whole window. Dividing by the tiny span instead let a single
        straggler print on a dead tape read as a hot one and steal a line
        from a steadily active holder.
        """
        hist = self._vol_hist.get(symbol)
        if not hist or len(hist) < 2:
            return 0.0
        t_new, v_new = hist[-1]
        cutoff = t_new - IBKR_TAPE_RANK_WINDOW_SECS
        t_old, v_old = hist[0]
        for t, v in hist:
            if t >= cutoff:
                t_old, v_old = t, v
                break
        if t_new <= t_old:
            return 0.0
        return max(0.0, (v_new - v_old) / IBKR_TAPE_RANK_WINDOW_SECS)

    def _dollar_rate(self, symbol: str) -> float:
        """Dollar volume per second: :meth:`_vol_rate` scaled by the L1
        price. Share rate alone over/under-gates by price level (50 sh/s is
        $100/s on a $2 stock, $10k/s on a $200 one); dollars are the
        scale-free "is anything actually trading here" measure the depth
        relevance gate needs. Slot-independent like the share rate: both
        factors come off the always-on reqMktData ticker.
        """
        rate = self._vol_rate(symbol)
        if rate <= 0.0:
            return 0.0
        ticker = self._q_tickers.get(symbol)
        px = _ns(getattr(ticker, "last", 0.0)) if ticker is not None else 0.0
        if px <= 0.0 and ticker is not None:
            px = (_ns(getattr(ticker, "bid", 0.0)) + _ns(getattr(ticker, "ask", 0.0))) / 2.0
        if px <= 0.0:
            px = self._prev_l1_last.get(symbol, 0.0)
        return rate * px

    def _pick_ts_victim(self, challenger: str) -> Optional[str]:
        """Weakest tape holder *challenger* may evict, or None.

        Unpinned holders are fair game for anyone — a newly surfaced symbol
        has no volume history yet and still needs prints to prove itself,
        so recency wins against cold holders (ties break slowest-rate, then
        oldest grant). Pinned holders only fall to a strictly higher pin:
        an armed pullback's tape is quiet by design, and rate-evicting it
        hands the slot back right after the breakout it was armed for.
        """
        cp = self._pins.get(challenger, 0)
        candidates = [
            (self._pins.get(sym, 0), self._vol_rate(sym), idx, sym)
            for idx, sym in enumerate(self._ts_lru)
            if sym != challenger and self._pins.get(sym, 0) < max(cp, 1)
        ]
        if not candidates:
            return None
        return min(candidates)[3]

    def _pick_depth_victim(self, challenger: str) -> Optional[str]:
        """Lowest-pinned depth holder strictly below *challenger*'s pin.

        Strictness (no equal-pin turnover, so unpinned symbols never evict
        each other) is what kills the constant resubscribe churn depth
        suffered under newest-wins LRU."""
        cp = self._pins.get(challenger, 0)
        candidates = [
            (self._pins.get(sym, 0), idx, sym)
            for idx, sym in enumerate(self._depth_lru)
            if sym != challenger and self._pins.get(sym, 0) < cp
        ]
        if not candidates:
            return None
        return min(candidates)[2]

    def _maybe_rebalance_ts(self, ib: object) -> None:
        """Every IBKR_TAPE_REBALANCE_SECS, let the best slotless symbol
        challenge for a tape line.

        Fills a free line unconditionally (an evicted symbol wins its tape
        back once a line opens). Against a full house it takes the weakest
        holder's line when it out-pins it, or — both unpinned — when its
        L1 volume rate clears the holder's by IBKR_TAPE_RANK_MARGIN and the
        holder has had IBKR_TAPE_MIN_HOLD_SECS to prove itself. One swap
        per pass caps churn.
        """
        now = time.monotonic()
        if now - self._last_ts_rebalance < IBKR_TAPE_REBALANCE_SECS:
            return
        self._last_ts_rebalance = now
        self._prune_request_state(now)
        # Cap recovery: a 10190 lowers the cap, but the external pressure
        # behind it (another session's lines, Gateway-leaked subscriptions)
        # is usually transient — after a quiet interval, probe one line back.
        # The probing request re-lowers the cap and re-arms the timer if the
        # pressure is still there.
        if (self._ts_limit < IBKR_MAX_TICK_SUBS and self._ts_last_denial
                and now - self._ts_last_denial >= IBKR_TAPE_CAP_RECOVERY_SECS):
            self._ts_limit += 1
            self._ts_last_denial = now      # one line per quiet interval
            self._ts_denials.clear()
            self._ts_retry_after.clear()
            logger.info(
                "IBKRSharedStream: no tape denial for %.0fs — probing cap "
                "back up to %d", IBKR_TAPE_CAP_RECOVERY_SECS, self._ts_limit,
            )
        self._release_idle_depth(ib, now)
        self._fill_free_depth(ib)
        with self._lock:
            wanted = [s for s in self._subs if s in self._contracts]
        outsiders = [
            s for s in wanted
            if s not in self._ts_tickers and s not in self._ts_denied
        ]
        # Respect the 15s per-instrument re-request pacing and the denial
        # backoff — a challenger that cannot subscribe right now must not
        # evict anyone.
        ready = [
            s for s in outsiders
            if (self._ts_last_req.get(s) is None
                or (now - self._ts_last_req[s]) >= IBKR_TICK_RESUB_COOLDOWN_SECS)
            and now >= self._ts_retry_after.get(s, 0.0)
        ]
        if not ready:
            return
        best = max(ready, key=lambda s: (self._pins.get(s, 0), self._vol_rate(s)))
        if len(self._ts_tickers) < max(self._ts_limit, 1):
            self._start_ts(ib, best)
            return
        if not self._ts_lru:
            return
        holder = min(
            (self._pins.get(s, 0), self._vol_rate(s), idx, s)
            for idx, s in enumerate(self._ts_lru)
        )
        h_pin, h_rate, _, h_sym = holder
        b_pin, b_rate = self._pins.get(best, 0), self._vol_rate(best)
        if b_pin > h_pin:
            pass   # higher pin always claims the line
        elif (
            b_pin == h_pin == 0
            and b_rate > 0
            and b_rate >= h_rate * IBKR_TAPE_RANK_MARGIN
            and (now - self._ts_granted_at.get(h_sym, 0.0)) >= IBKR_TAPE_MIN_HOLD_SECS
        ):
            pass   # cold-vs-cold: clearly hotter symbol takes over
        else:
            return
        logger.info(
            "IBKRSharedStream: tape rebalance — %s (pin %d, %.0f vol/s) "
            "replaces %s (pin %d, %.0f vol/s)",
            best, b_pin, b_rate, h_sym, h_pin, h_rate,
        )
        self._stop_ts(ib, h_sym)
        self._start_ts(ib, best)

    def _prune_request_state(self, now: float) -> None:
        """Drop expired per-symbol request state for unsubscribed symbols.

        ``_ts_last_req`` and the denial backoffs survive a full unsubscribe
        on purpose — they mirror server-side state (the 15s per-instrument
        rule, a refused quota) that doesn't reset just because our interest
        did. They are only dropped here, once expired AND unwatched, so the
        dicts don't grow with a whole day's symbol churn.
        """
        with self._lock:
            live = set(self._subs)
        for sym in [s for s, t in self._ts_last_req.items()
                    if s not in live
                    and now - t > IBKR_TICK_RESUB_COOLDOWN_SECS]:
            del self._ts_last_req[sym]
        for sym in [s for s, t in self._ts_retry_after.items()
                    if s not in live and now > t]:
            del self._ts_retry_after[sym]
            self._ts_denials.pop(sym, None)
        for sym in [s for s, t in self._depth_retry_after.items()
                    if s not in live and now > t]:
            del self._depth_retry_after[sym]
            self._depth_denials.pop(sym, None)

    def _release_idle_depth(self, ib: object, now: float) -> None:
        """Proactively release depth lines whose holders stopped mattering.

        A book is a confirmation signal for symbols the strategy is working;
        an unpinned holder on a dead tape is just decode/aggregate overhead
        on every update. Once a holder has been unpinned AND below the
        dollar-rate floor continuously for IBKR_DEPTH_IDLE_RELEASE_SECS, its
        line is released without waiting for a challenger — back to L1-only
        order flow — and the freed line is idle-fillable in this same pass.

        The released symbol sits out IBKR_DEPTH_RELEASE_COOLDOWN_SECS before
        it can reclaim a line through subscribe/idle-fill (a pin raise
        bypasses the cooldown via ignore_backoff), so a symbol hovering at
        the floor cannot flap the line — resubscribe churn corrupts books
        (see _update_mkt_depth_l2).
        """
        for sym in list(self._depth_tickers):
            if (self._pins.get(sym, 0) > 0
                    or self._dollar_rate(sym) >= IBKR_DEPTH_IDLE_DOLLAR_RATE):
                self._depth_idle_since.pop(sym, None)
                continue
            idle_start = self._depth_idle_since.setdefault(sym, now)
            if now - idle_start < IBKR_DEPTH_IDLE_RELEASE_SECS:
                continue
            logger.info(
                "IBKRSharedStream: depth idle release %s (pin 0, $%.0f/s "
                "for %.0fs) — OFI degrades to L1",
                sym, self._dollar_rate(sym), now - idle_start,
            )
            self._stop_depth(ib, sym)
            self._depth_retry_after[sym] = max(
                self._depth_retry_after.get(sym, 0.0),
                now + IBKR_DEPTH_RELEASE_COOLDOWN_SECS,
            )

    def _fill_free_depth(self, ib: object) -> None:
        """Idle-fill: hand a free depth line to the best symbol wanting one.

        Grants at subscribe/pin-raise cover the common paths; this covers a
        holder unsubscribing while a previously declined symbol still wants
        the book — without it the freed line would sit dark until that
        symbol's pin next changed.
        """
        if len(self._depth_tickers) >= max(IBKR_L2_MAX_DEPTH_SUBS, 1):
            return
        now = time.monotonic()
        with self._lock:
            wanters = [
                sym for sym, cbs in self._subs.items()
                if sym in self._contracts
                and sym not in self._depth_tickers
                and sym not in self._depth_denied
                and any(cb[2] is not None for cb in cbs.values())
            ]
        wanters = [s for s in wanters if now >= self._depth_retry_after.get(s, 0.0)]
        if not wanters:
            return
        best = max(wanters, key=lambda s: (self._pins.get(s, 0), self._vol_rate(s)))
        self._start_depth(ib, best)

    def _start_depth(self, ib: object, symbol: str, *,
                     ignore_backoff: bool = False) -> None:
        """Request L2 market depth for *symbol* if it currently matters
        (pinned, or L1 dollar-volume rate at/above the relevance floor) and
        a line is free or a strictly lower-pinned holder can be evicted for
        it.

        Unlike tape, an equal pin never turns a line over: depth resubscribe
        churn corrupts books (see :func:`_update_mkt_depth_l2`) and deep OFI
        is a confirmation signal — it belongs to the symbols the strategy
        has pinned, not to whoever subscribed last. Losing or never getting
        a line degrades to L1-only order flow silently.

        ``ignore_backoff`` lets a pin raise re-attempt a 309-refused symbol
        immediately; everyone else waits out the denial backoff."""
        if symbol in self._depth_tickers:
            return
        contract = self._contracts.get(symbol)
        if contract is None:
            return
        if not ignore_backoff and time.monotonic() < self._depth_retry_after.get(symbol, 0.0):
            return
        # Demand gate — mirror image of _release_idle_depth: an unpinned
        # symbol on a quiet tape gets no book, even with lines free. Decoding
        # a book nobody is trading is pure overhead, and granting it anyway
        # just feeds the idle-release/regrant cycle. A cold newcomer (no
        # volume history yet) starts on L1 and wins a line at a rebalance
        # tick once its rate proves out; a pin raise gets one immediately.
        if (self._pins.get(symbol, 0) == 0
                and self._dollar_rate(symbol) < IBKR_DEPTH_IDLE_DOLLAR_RATE):
            return
        while len(self._depth_tickers) >= max(IBKR_L2_MAX_DEPTH_SUBS, 1):
            evict = self._pick_depth_victim(symbol)
            if evict is None:
                logger.info(
                    "IBKRSharedStream: no depth line for %s — %d in use by "
                    "equal/higher-pinned holders (OFI stays L1)",
                    symbol, len(self._depth_tickers),
                )
                return
            logger.info("IBKRSharedStream: depth slots full — evicting %s for %s", evict, symbol)
            self._stop_depth(ib, evict)
        try:
            ticker = ib.reqMktDepth(contract, numRows=IBKR_L2_ROWS, isSmartDepth=True)
            # Per-update state buffer for the patched wrapper (see
            # _update_mkt_depth_l2). Always a fresh deque: ib_insync reuses
            # Ticker objects per contract, and states queued before a
            # resubscribe gap are stale.
            ticker._momo_book_states = deque(maxlen=_BOOK_STATE_MAXLEN)
            self._depth_tickers[symbol] = ticker
            self._depth_lru.append(symbol)
            logger.debug("IBKRSharedStream: depth subscribed %s", symbol)
        except Exception as exc:
            logger.warning("IBKRSharedStream: depth subscribe %s failed: %s", symbol, exc)
            return
        self._start_venue_quotes(ib, symbol, contract)

    def _start_venue_quotes(self, ib: object, symbol: str, contract: object) -> None:
        """Directed quotes for IBKR_L2_VENUE_QUOTES, alongside *symbol*'s depth line.

        These ride on the depth line's lifetime: started with it, cancelled by
        _stop_depth, so a 309-reaped or evicted book takes them down too.
        """
        if symbol in self._venue_quotes:
            return
        quotes: list[tuple[str, object, object]] = []
        for venue in IBKR_L2_VENUE_QUOTES:
            try:
                directed = _directed_contract(contract, venue)
                quotes.append((venue, directed, ib.reqMktData(directed, "", snapshot=False)))
            except Exception as exc:
                logger.warning("IBKRSharedStream: [%s] %s quote for depth failed: %s", symbol, venue, exc)
        self._venue_quotes[symbol] = quotes

    def _stop_depth(self, ib: object, symbol: str) -> None:
        ticker = self._depth_tickers.pop(symbol, None)
        self._prev_book_sig.pop(symbol, None)
        self._prev_venue_sig.pop(symbol, None)
        self._depth_idle_since.pop(symbol, None)
        if symbol in self._depth_lru:
            self._depth_lru.remove(symbol)
        for _, directed, _ in self._venue_quotes.pop(symbol, ()):
            try:
                ib.cancelMktData(directed)
            except Exception:
                pass
        if ticker is None:
            return
        try:
            del ticker._momo_book_states
        except AttributeError:
            pass
        contract = self._contracts.get(symbol)
        if contract is not None:
            try:
                ib.cancelMktDepth(contract, isSmartDepth=True)
            except Exception:
                pass
        logger.debug("IBKRSharedStream: depth unsubscribed %s", symbol)

    def _stop_symbol(self, ib: object, symbol: str) -> None:
        self._stop_depth(ib, symbol)
        # Cancels the tape and frees its quota slot — do not repeat the
        # cancelTickByTickData below, or the slot is released twice and the
        # second cancel hits a subscription IBKR has already dropped.
        self._stop_ts(ib, symbol)
        self._ts_denied.discard(symbol)
        self._depth_denied.discard(symbol)
        # _ts_last_req and the denial backoffs survive on purpose: they
        # mirror server-side state (the 15s per-instrument rule, a refused
        # quota) that doesn't reset when our interest does. A symbol that
        # leaves and re-enters the watchlist inside 15s must not re-request
        # immediately. _prune_request_state drops them once expired.
        self._pins.pop(symbol, None)
        self._vol_hist.pop(symbol, None)
        if self._tasty is not None:
            self._tasty.remove(symbol)
        if self._tape is not None:
            self._tape.forget(symbol)
        contract = self._contracts.pop(symbol, None)
        self._q_tickers.pop(symbol, None)
        self._prev_bid.pop(symbol, None)
        self._prev_ask.pop(symbol, None)
        self._prev_l1_last.pop(symbol, None)
        self._seq.pop(symbol, None)
        if contract:
            try:
                ib.cancelMktData(contract)
            except Exception:
                pass
            logger.debug("IBKRSharedStream: unsubscribed %s", symbol)

    def _stop_all(self, ib: object) -> None:
        for sym in list(self._contracts):
            self._stop_symbol(ib, sym)
        # Cancels are queued on the asyncio transport; closing the socket
        # immediately can drop them, and the Gateway then keeps counting the
        # dead subscriptions against the account quotas for many minutes
        # (observed 2026-08-05: a dead session phantom-held all 3 depth
        # lines). One short loop turn flushes the queue before disconnect.
        try:
            ib.sleep(0.25)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Singleton accessor (mirrors tradier_stream.get_shared_stream)
# ---------------------------------------------------------------------------

_ibkr_shared_stream: Optional[IBKRSharedStream] = None
_ibkr_stream_lock = threading.Lock()


def get_ibkr_shared_stream() -> IBKRSharedStream:
    """Return the process-wide IBKRSharedStream (lazy-initialised)."""
    global _ibkr_shared_stream
    with _ibkr_stream_lock:
        if _ibkr_shared_stream is None:
            _ibkr_shared_stream = IBKRSharedStream()
    return _ibkr_shared_stream
