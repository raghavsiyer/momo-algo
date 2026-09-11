"""
market_data/tradier_stream.py

Tradier HTTP Streaming API — live market data.

  LIVE OHLCV
    - stream_quotes()         Trade + Summary events → live price feed

  LEVEL 2 / NBBO
    - stream_level2()         Quote events → real-time bid/ask updates

  TIME & SALES
    - stream_timesales()      Timesale events → sequenced T&S prints

  CANDLE BUILDER
    - build_live_candle_buffer()  Aggregates trade stream into rolling OHLCV bars

All stream methods return a daemon Thread immediately and dispatch
typed dataclass events to your callback on each tick. Pass a
threading.Event to stop_event to shut down gracefully.

Streaming endpoint: https://stream.tradier.com/v1/markets/events
Session creation:   https://api.tradier.com/v1/markets/events/session

Usage:
    from market_data.tradier_stream import RealTimeClient
    import threading, time

    client = RealTimeClient()

    # Live T&S prints
    def on_ts(event):
        print(f"{event.ts} | {event.last} x {event.size} | bid={event.bid} ask={event.ask}")

    stop = threading.Event()
    client.stream_timesales("ACMR", on_timesale=on_ts, stop_event=stop)
    time.sleep(60)
    stop.set()

    # Rolling 1-min live candle buffer
    buf, stop = client.build_live_candle_buffer("ACMR", interval_seconds=60)
    time.sleep(300)
    stop.set()
    import pandas as pd
    from dataclasses import asdict
    df = pd.DataFrame([asdict(c) for c in buf])
"""


import asyncio
import json
import logging
import os
import ssl
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import certifi
import orjson
import requests
import websockets
import websockets.exceptions

logger = logging.getLogger(__name__)

# OHLCVCandle is shared — import from tradier_rest to avoid duplication
from market_data.tradier_rest import OHLCVCandle, _ms_to_dt as _ms_to_dt_opt
import config
from config import HTTP_TIMEOUT, HTTP_STREAM_TIMEOUT, LIVE_CANDLE_MAX_BARS

_REST_BASE = "https://api.tradier.com/v1"
_STREAM_BASE = "https://stream.tradier.com/v1"


# ---------------------------------------------------------------------------
# Live stream event dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LiveTrade:
    """
    Streaming 'trade' event — single execution on an exchange.
    Real-time price / volume for the OHLCV tracker.
    """
    event_type: str                  # always "trade"
    symbol: str
    exchange: str
    price: float
    size: int
    cumulative_volume: int           # cvol: total shares traded since open
    ts: datetime                     # millisecond-precision UTC
    last: float                      # last price (same as price for equities)


@dataclass
class LiveQuote:
    """
    Streaming 'quote' event — Level 2 / NBBO bid-ask update.
    Fires whenever a new best bid or ask is posted on any exchange.
    """
    event_type: str                  # always "quote"
    symbol: str
    bid: float
    bid_size: int
    bid_exchange: str
    bid_ts: datetime
    ask: float
    ask_size: int
    ask_exchange: str
    ask_ts: datetime
    # L1 last trade — IBKR's reqMktData carries it; Tradier quote events do
    # not (their trades arrive as separate timesale events), so it defaults
    # to 0 there. Lets print-less symbols keep a moving price view.
    last: float = 0.0
    last_size: int = 0
    # True for quotes fabricated from a tick by replay/backtest
    # (BacktestRunner._quote_from_tick): prices are the real NBBO embedded in
    # the print, but the SIZES are placeholders. Size-derived reads
    # (book lean, hidden-liquidity sampling) must skip these.
    synthetic: bool = False


@dataclass
class LiveTimesale:
    """
    Streaming 'timesale' event — sequenced Time & Sales print.
    Includes bid/ask at time of execution. Events are uniquely sequenced.
    Use seq for ordering; cancel/correction flags filter bad prints.
    """
    event_type: str                  # always "timesale"
    symbol: str
    exchange: str
    last: float                      # execution price
    bid: float                       # NBBO bid at time of print
    ask: float                       # NBBO ask at time of print
    size: int                        # shares traded
    ts: datetime
    seq: int                         # monotonic sequence number
    flag: str                        # condition codes (e.g. "@" = regular)
    cancel: bool                     # true if this print was cancelled
    correction: bool                 # true if this is a corrected print
    session: str                     # "normal" | "pre" | "post"


@dataclass
class LiveSummary:
    """
    Streaming 'summary' event — session-level OHLC snapshot.
    Fired at market open, close, and session transitions.
    """
    event_type: str                  # always "summary"
    symbol: str
    open: float
    high: float
    low: float
    prev_close: float


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RealTimeClient:
    """
    Tradier streaming client for live market data.

    Reads TRADIER_API_KEY from environment.
    Session creation always hits api.tradier.com (REST).
    The stream itself hits stream.tradier.com.
    Note: paper/sandbox accounts do NOT support streaming —
    TRADIER_PAPER has no effect here.
    """

    def __init__(self) -> None:
        token = os.getenv("TRADIER_API_KEY")
        if not token:
            raise EnvironmentError(
                "TRADIER_API_KEY not set. "
                "Add to .env: TRADIER_API_KEY=your_token"
            )
        self._token = token
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        self._session = requests.Session()
        self._session.headers.update(self._headers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ms_to_dt(ms) -> datetime:
        """Tradier timestamps are milliseconds since epoch."""
        return _ms_to_dt_opt(ms) or datetime.now(tz=timezone.utc)

    def _create_stream_session(self) -> str:
        """
        POST /v1/markets/events/session → short-lived session ID.
        Must be consumed within ~5 minutes before it expires.
        """
        resp = self._session.post(
            f"{_REST_BASE}/markets/events/session",
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        session_id = resp.json().get("stream", {}).get("sessionid")
        if not session_id:
            raise RuntimeError(
                f"Failed to create Tradier stream session: {resp.text}"
            )
        return session_id

    def _parse_event(
        self, line: str, fallback_symbol: str
    ) -> Optional[LiveTrade | LiveQuote | LiveTimesale | LiveSummary]:
        """Parse a single newline-delimited JSON chunk into a typed event."""
        line = line.strip()
        if not line:
            return None
        try:
            obj = orjson.loads(line)
        except orjson.JSONDecodeError:
            return None

        t = obj.get("type", "")
        sym = obj.get("symbol", fallback_symbol)

        if t == "trade":
            return LiveTrade(
                event_type="trade",
                symbol=sym,
                exchange=obj.get("exch", ""),
                price=float(obj.get("price", 0)),
                size=int(obj.get("size", 0)),
                cumulative_volume=int(obj.get("cvol", 0)),
                ts=self._ms_to_dt(obj.get("date")),
                last=float(obj.get("last", 0)),
            )

        if t == "quote":
            return LiveQuote(
                event_type="quote",
                symbol=sym,
                bid=float(obj.get("bid", 0)),
                bid_size=int(obj.get("bidsz", 0)),
                bid_exchange=obj.get("bidexch", ""),
                bid_ts=self._ms_to_dt(obj.get("biddate")),
                ask=float(obj.get("ask", 0)),
                ask_size=int(obj.get("asksz", 0)),
                ask_exchange=obj.get("askexch", ""),
                ask_ts=self._ms_to_dt(obj.get("askdate")),
            )

        if t == "timesale":
            return LiveTimesale(
                event_type="timesale",
                symbol=sym,
                exchange=obj.get("exch", ""),
                last=float(obj.get("last", 0)),
                bid=float(obj.get("bid", 0)),
                ask=float(obj.get("ask", 0)),
                size=int(obj.get("size", 0)),
                ts=self._ms_to_dt(obj.get("date")),
                seq=int(obj.get("seq", 0)),
                flag=str(obj.get("flag", "")),
                cancel=bool(obj.get("cancel", False)),
                correction=bool(obj.get("correction", False)),
                session=obj.get("session", "normal"),
            )

        if t == "summary":
            return LiveSummary(
                event_type="summary",
                symbol=sym,
                open=float(obj.get("open", 0)),
                high=float(obj.get("high", 0)),
                low=float(obj.get("low", 0)),
                prev_close=float(obj.get("prevClose", 0)),
            )

        return None

    def _run_stream(
        self,
        symbols: list[str],
        filters: list[str],
        on_event: Callable,
        stop_event: threading.Event,
    ) -> None:
        """
        Open the HTTP chunked stream and dispatch events to on_event.
        Blocks until stop_event is set or the connection drops.
        Called inside a daemon thread — do not call directly.
        """
        if config.TRADIER_PAUSED:
            return
        session_id = self._create_stream_session()
        url = f"{_STREAM_BASE}/markets/events"
        params = {
            "sessionid": session_id,
            "symbols": ",".join(symbols),
            "filter": ",".join(filters),
            "linebreak": "true",
        }

        with self._session.get(
            url,
            params=params,
            stream=True,
            timeout=HTTP_STREAM_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if stop_event.is_set() or config.TRADIER_PAUSED:
                    break
                event = self._parse_event(raw_line, symbols[0])
                if event is not None:
                    on_event(event)

    def _start_thread(
        self,
        symbols: list[str],
        filters: list[str],
        on_event: Callable,
        stop_event: threading.Event,
    ) -> threading.Thread:
        t = threading.Thread(
            target=self._run_stream,
            args=(symbols, filters, on_event, stop_event),
            daemon=True,
        )
        t.start()
        return t

    # ------------------------------------------------------------------
    # PUBLIC STREAMING METHODS
    # ------------------------------------------------------------------

    def stream_quotes(
        self,
        symbol: str,
        on_trade: Callable[[LiveTrade], None],
        on_summary: Optional[Callable[[LiveSummary], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> threading.Thread:
        """
        Stream live OHLCV price data.
        Subscribes to 'trade' + 'summary' filter.

        on_trade(LiveTrade)       — fires on every execution print
        on_summary(LiveSummary)   — fires on session open/high/low updates (optional)
        stop_event                — set() to shut down the stream thread

        Returns the daemon Thread.
        """
        if stop_event is None:
            stop_event = threading.Event()

        def dispatch(event):
            if isinstance(event, LiveTrade):
                on_trade(event)
            elif isinstance(event, LiveSummary) and on_summary:
                on_summary(event)

        return self._start_thread([symbol], ["trade", "summary"], dispatch, stop_event)

    def stream_level2(
        self,
        symbol: str,
        on_quote: Callable[[LiveQuote], None],
        stop_event: Optional[threading.Event] = None,
    ) -> threading.Thread:
        """
        Stream Level 2 / NBBO bid-ask updates.
        Subscribes to 'quote' filter.

        on_quote(LiveQuote)  — fires on every exchange quote event
        Returns the daemon Thread.
        """
        if stop_event is None:
            stop_event = threading.Event()

        def dispatch(event):
            if isinstance(event, LiveQuote):
                on_quote(event)

        return self._start_thread([symbol], ["quote"], dispatch, stop_event)

    def stream_timesales(
        self,
        symbol: str,
        on_timesale: Callable[[LiveTimesale], None],
        stop_event: Optional[threading.Event] = None,
    ) -> threading.Thread:
        """
        Stream Time & Sales — sequenced, uniquely identified trade prints
        with bid/ask captured at the moment of execution.
        Subscribes to 'timesale' filter.

        on_timesale(LiveTimesale)  — fires on each print
        Returns the daemon Thread.
        """
        if stop_event is None:
            stop_event = threading.Event()

        def dispatch(event):
            if isinstance(event, LiveTimesale):
                on_timesale(event)

        return self._start_thread([symbol], ["timesale"], dispatch, stop_event)

    # ------------------------------------------------------------------
    # CANDLE BUILDER — aggregate trade stream into rolling OHLCV bars
    # ------------------------------------------------------------------

    def stream_combined(
        self,
        symbol: str,
        on_timesale: Callable[[LiveTimesale], None],
        on_quote: Callable[[LiveQuote], None],
        stop_event: Optional[threading.Event] = None,
    ) -> threading.Thread:
        """
        Single session streaming both timesale and quote events.
        Tradier allows multiple filters in one session — no need for two
        connections. Used by DynamicScorer to get T&S + L2 in one thread.
        """
        if stop_event is None:
            stop_event = threading.Event()

        def dispatch(event):
            if isinstance(event, LiveTimesale):
                on_timesale(event)
            elif isinstance(event, LiveQuote):
                on_quote(event)

        return self._start_thread(
            [symbol], ["timesale", "quote"], dispatch, stop_event
        )

    def build_live_candle_buffer(
        self,
        symbol: str,
        interval_seconds: int = 60,
        max_bars: int = LIVE_CANDLE_MAX_BARS,
    ) -> tuple[list[OHLCVCandle], threading.Event]:
        """
        Start a trade stream and aggregate prints into a rolling OHLCV buffer.

        Returns (candle_buffer, stop_event).
          candle_buffer[-1]  — the currently forming bar (updates in real time)
          candle_buffer[-2]  — last completed bar
          stop_event.set()   — gracefully stops the stream thread

        The buffer is thread-safe for reads between ticks.
        OHLCVCandle is imported from tradier_rest.py — same type across both modules.

        Example:
            buf, stop = client.build_live_candle_buffer("ACMR", interval_seconds=60)
            time.sleep(300)
            stop.set()
            df = pd.DataFrame([asdict(c) for c in buf])
        """
        candles: deque[OHLCVCandle] = deque(maxlen=max_bars)
        stop_event = threading.Event()
        lock = threading.Lock()
        label = f"{interval_seconds}sec" if interval_seconds < 60 else f"{interval_seconds // 60}min"

        def bucket_ts(ts: datetime) -> datetime:
            epoch = ts.timestamp()
            floored = (epoch // interval_seconds) * interval_seconds
            return datetime.fromtimestamp(floored, tz=timezone.utc)

        def on_trade(event: LiveTrade):
            bar_ts = bucket_ts(event.ts)
            with lock:
                if candles and candles[-1].ts == bar_ts:
                    # Update the forming bar
                    c = candles[-1]
                    c.high = max(c.high, event.price)
                    c.low = min(c.low, event.price)
                    c.close = event.price
                    c.volume += event.size
                else:
                    # Open a new bar (deque auto-evicts oldest when full)
                    candles.append(OHLCVCandle(
                        symbol=symbol,
                        ts=bar_ts,
                        open=event.price,
                        high=event.price,
                        low=event.price,
                        close=event.price,
                        volume=event.size,
                        interval=label,
                    ))

        self.stream_quotes(symbol, on_trade=on_trade, stop_event=stop_event)
        return candles, stop_event


# ---------------------------------------------------------------------------
# Shared stream — single session for all active tickers
# ---------------------------------------------------------------------------

class SharedStreamClient:
    """
    Single WebSocket connection multiplexed across all active DynamicScorers
    and EarlyStream instances.

    Uses wss://ws.tradier.com/v1/markets/events (Tradier WebSocket API).
    When the symbol set changes, the live payload is resent on the existing
    connection — no reconnect or new session needed. Only one market-data
    WebSocket session is permitted per Tradier token.

    Thread-safe: subscribe/unsubscribe may be called from any thread.
    """

    _WS_URL = "wss://ws.tradier.com/v1/markets/events"

    def __init__(self) -> None:
        token = os.getenv("TRADIER_API_KEY")
        if not token:
            raise EnvironmentError("TRADIER_API_KEY not set.")
        self._http = requests.Session()
        self._http.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })
        self._lock = threading.Lock()
        # sym → { sub_id → (on_timesale_cb, on_quote_cb) }
        self._subs: dict[str, dict[str, tuple[Callable, Callable]]] = {}
        self._stopped = False
        # asyncio.Event to signal symbol-set changes to the async loop.
        # Created in the main thread; safe in Python 3.10+ (no loop binding at init).
        self._change_event: asyncio.Event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="ws-mkt-stream",
        )
        self._thread.start()
        self._manager_future = asyncio.run_coroutine_threadsafe(self._manager(), self._loop)
        self._manager_future.add_done_callback(
            lambda _f: self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop.is_running() else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(
        self,
        symbol: str,
        on_timesale: Callable,
        on_quote: Callable,
        sub_id: str,
        on_book: Optional[Callable] = None,
    ) -> None:
        """Register callbacks for *symbol*. Resends payload if already connected.

        ``on_book`` is accepted for interface parity with IBKRSharedStream
        and ignored — Tradier provides no L2 depth feed.
        """
        with self._lock:
            self._subs.setdefault(symbol, {})[sub_id] = (on_timesale, on_quote)
        self._loop.call_soon_threadsafe(self._change_event.set)

    def unsubscribe(self, symbol: str, sub_id: str) -> None:
        """Remove a subscription. Resends payload to drop the symbol if no subscribers remain."""
        with self._lock:
            if symbol in self._subs:
                self._subs[symbol].pop(sub_id, None)
                if not self._subs[symbol]:
                    del self._subs[symbol]
        self._loop.call_soon_threadsafe(self._change_event.set)

    def set_priority(self, symbol: str, pin: int) -> None:
        """Interface parity with IBKRSharedStream — Tradier has no capped
        tape/depth lines to allocate, so pins are meaningless here."""
        return

    def stop(self) -> None:
        """Tear down the WebSocket entirely (e.g. at process shutdown)."""
        self._stopped = True
        with self._lock:
            self._subs.clear()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._change_event.set)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("[stream] Shared stream thread did not exit cleanly")
            elif not self._loop.is_closed():
                self._loop.close()
        global _shared_stream
        with _shared_stream_lock:
            if _shared_stream is self:
                _shared_stream = None
        logger.info("[stream] Shared stream stopped")

    # ------------------------------------------------------------------
    # Async internals — all run inside self._loop
    # ------------------------------------------------------------------

    def _get_symbols(self) -> list[str]:
        with self._lock:
            return list(self._subs.keys())

    def _make_payload(self, symbols: list[str], session_id: str) -> str:
        return json.dumps({
            "symbols": symbols,
            "filter": ["timesale", "quote"],
            "sessionid": session_id,
            "linebreak": True,
        })

    def _create_stream_session(self) -> str:
        resp = self._http.post(
            f"{_REST_BASE}/markets/events/session",
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        session_id = resp.json().get("stream", {}).get("sessionid")
        if not session_id:
            raise RuntimeError(f"Tradier session creation failed: {resp.text}")
        return session_id

    async def _manager(self) -> None:
        """Single coroutine owning the WebSocket lifecycle for the process."""
        while not self._stopped:
            # Wait until there is at least one subscriber and streaming isn't paused.
            while (not self._get_symbols() or config.TRADIER_PAUSED) and not self._stopped:
                self._change_event.clear()
                try:
                    await asyncio.wait_for(self._change_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            if self._stopped:
                break

            # Fetch session via blocking HTTP — run off the event loop.
            try:
                session_id = await self._loop.run_in_executor(
                    None, self._create_stream_session
                )
            except Exception as exc:
                logger.warning("SharedStreamClient: session creation failed: %s", exc)
                await asyncio.sleep(2.0)
                continue

            symbols = self._get_symbols()
            if not symbols:
                continue

            # Open the WebSocket and stream until disconnected or stopped.
            try:
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
                async with websockets.connect(
                    self._WS_URL, ssl=ssl_ctx, compression=None,
                ) as ws:
                    self._change_event.clear()
                    await ws.send(self._make_payload(symbols, session_id))
                    logger.debug(
                        "SharedStreamClient: WebSocket connected (%s)",
                        ", ".join(symbols),
                    )

                    while not self._stopped:
                        recv_task = asyncio.ensure_future(ws.recv())
                        change_task = asyncio.ensure_future(self._change_event.wait())
                        try:
                            done, pending = await asyncio.wait(
                                {recv_task, change_task},
                                return_when=asyncio.FIRST_COMPLETED,
                            )
                        except Exception:
                            recv_task.cancel()
                            change_task.cancel()
                            break

                        for t in pending:
                            t.cancel()
                            try:
                                await t
                            except (asyncio.CancelledError, Exception):
                                pass

                        if change_task in done:
                            self._change_event.clear()
                            if config.TRADIER_PAUSED:
                                break  # paused — close WS; outer loop waits until resumed
                            new_symbols = self._get_symbols()
                            if not new_symbols:
                                break  # no more subscribers — close
                            await ws.send(self._make_payload(new_symbols, session_id))
                            logger.debug(
                                "SharedStreamClient: symbols updated (%s)",
                                ", ".join(new_symbols),
                            )

                        if recv_task in done and not recv_task.cancelled():
                            try:
                                self._dispatch(recv_task.result())
                            except (websockets.exceptions.ConnectionClosed, Exception):
                                break

            except Exception as exc:
                if not self._stopped:
                    logger.warning("SharedStreamClient: WebSocket error: %s", exc)

            # Brief back-off before reconnecting (only when subs remain).
            if not self._stopped and self._get_symbols():
                await asyncio.sleep(1.0)

    def _dispatch(self, message: str) -> None:
        """Parse one WebSocket message and fan out to registered callbacks."""
        try:
            obj = orjson.loads(message)
        except Exception:
            return
        sym = obj.get("symbol", "")
        with self._lock:
            callbacks = dict(self._subs.get(sym, {}))
        if not callbacks:
            return
        t = obj.get("type", "")
        if t == "timesale":
            event = LiveTimesale(
                event_type="timesale",
                symbol=sym,
                exchange=obj.get("exch", ""),
                last=float(obj.get("last", 0)),
                bid=float(obj.get("bid", 0)),
                ask=float(obj.get("ask", 0)),
                size=int(obj.get("size", 0)),
                ts=_ms_to_dt_opt(obj.get("date")) or datetime.now(tz=timezone.utc),
                seq=int(obj.get("seq", 0)),
                flag=str(obj.get("flag", "")),
                cancel=bool(obj.get("cancel", False)),
                correction=bool(obj.get("correction", False)),
                session=obj.get("session", "normal"),
            )
            for sub_id, (on_ts, _) in callbacks.items():
                try:
                    on_ts(event)
                except Exception:
                    logger.exception("SharedStreamClient: callback error (timesale, sub=%s, sym=%s)", sub_id, sym)
        elif t == "quote":
            event = LiveQuote(
                event_type="quote",
                symbol=sym,
                bid=float(obj.get("bid", 0)),
                bid_size=int(obj.get("bidsz", 0)),
                bid_exchange=obj.get("bidexch", ""),
                bid_ts=_ms_to_dt_opt(obj.get("biddate")) or datetime.now(tz=timezone.utc),
                ask=float(obj.get("ask", 0)),
                ask_size=int(obj.get("asksz", 0)),
                ask_exchange=obj.get("askexch", ""),
                ask_ts=_ms_to_dt_opt(obj.get("askdate")) or datetime.now(tz=timezone.utc),
            )
            for sub_id, (_, on_quote_cb) in callbacks.items():
                try:
                    on_quote_cb(event)
                except Exception:
                    logger.exception("SharedStreamClient: callback error (quote, sub=%s, sym=%s)", sub_id, sym)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_shared_stream: Optional[SharedStreamClient] = None
_shared_stream_lock = threading.Lock()


def get_shared_stream() -> SharedStreamClient:
    """Return the process-wide SharedStreamClient (lazy-initialised)."""
    global _shared_stream
    if _shared_stream is None or _shared_stream._stopped:
        with _shared_stream_lock:
            if _shared_stream is None or _shared_stream._stopped:
                _shared_stream = SharedStreamClient()
    return _shared_stream


def wake_shared_stream() -> None:
    """Nudge the shared stream manager to re-evaluate pause/symbol state.

    Used by the "tradier stop/go" command so the WebSocket disconnects (when
    ``config.TRADIER_PAUSED`` is set) or reconnects (when cleared) promptly
    instead of waiting on the 1s manager poll. No-op if no shared stream has
    been created yet.
    """
    with _shared_stream_lock:
        stream = _shared_stream
    if stream is None:
        return
    loop = stream._loop
    if loop.is_running():
        loop.call_soon_threadsafe(stream._change_event.set)