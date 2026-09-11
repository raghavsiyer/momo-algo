"""
market_data/ibkr_rest.py

Interactive Brokers REST-equivalent client for stats + historical OHLCV data
via ``ib_insync`` (TWS/Gateway socket API).

  STATS
    - Snapshot quote: last price, % change, volume, avg volume
    - Multi-symbol batch quotes

  HISTORICAL CANDLES
    - 10-second bars    — get_candles_10sec()
    - 10-second window  — get_candles_10sec_window()
    - 1-minute bars     — get_candles_1min()
    - 1-minute window   — get_candles_1min_window()
    - 5-minute bars     — get_candles_5min()
    - 15-minute bars    — get_candles_15min()
    - Daily bars        — get_candles_daily()

    HISTORICAL TICKS
        - Trade ticks       — get_timesales_tick()

  MARKET SCANNER
    - Top percent gainers — scan_top_gainers()

Returns the same ``StockQuote`` / ``OHLCVCandle`` dataclasses and DataFrame
format used by the Tradier REST client, so the rest of the pipeline is
data-source-agnostic.

Usage:
    from market_data.ibkr_rest import IBKRHistoricalClient
    client = IBKRHistoricalClient()

    q  = client.get_quote("ACMR")
    df = client.get_candles_1min("ACMR", n=30)
"""

import asyncio
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


_ensure_event_loop()

try:
    from ib_insync import IB, ScannerSubscription, Stock, TagValue, util as ib_util
except ImportError:  # graceful when ib_insync is not installed
    IB = ScannerSubscription = Stock = TagValue = ib_util = None  # type: ignore[assignment,misc]

from config import (
    IBKR_ENABLED,
    IBKR_HOST,
    IBKR_PORT,
    IBKR_CLIENT_ID,
    IBKR_MAX_CONNECTIONS,
    IBKR_TICK_MAX_REQUESTS_TRADES,
    IBKR_TICK_MAX_REQUESTS_QUOTES,
    CANDLES_1MIN_FETCH_DAYS,
    CANDLES_5MIN_FETCH_DAYS,
    CANDLES_10SEC_FETCH_HOURS,
    CANDLES_15MIN_FETCH_DAYS,
    PRICE_MIN,
    PRICE_MAX,
    SCANNER_IBKR_INSTRUMENT,
    SCANNER_IBKR_LOCATION,
    SCANNER_IBKR_MIN_CHG,
    SCANNER_IBKR_MIN_VOLUME,
    SCANNER_IBKR_ROWS,
    MarketClock,
    scanner_ibkr_scan_spec,
)
from market_data.tradier_rest import (
    MarketDataUnavailable, StockQuote, OHLCVCandle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local IB connections — ib_insync sync APIs are event-loop bound
# ---------------------------------------------------------------------------

_ib_lock = threading.Lock()
_ib_instance: Optional[IB] = None
_ib_local = threading.local()
_ib_clients: dict[int, IB] = {}
_next_client_offset: int = 0
# Client ids returned by release_ib()/disconnect_all(), available for reuse.
# Without this, every pipeline worker thread burns a fresh id and we march
# into the Gateway's 32-connection ceiling as the watchlist grows.
_free_client_ids: list[int] = []
# Ids whose connect attempt failed. Kept apart from the free list because the
# Gateway holds an id reserved until its socket closes, so a crashed run can
# leave one permanently refused ("Error 326: client id already in use"). Were
# these pooled with the clean ids, the allocator — which prefers the pool over
# a fresh id — would hand the same dead id back on every retry and wedge every
# IBKR consumer, not just the caller that hit it. Retried only once the fresh
# band is exhausted, which is the case where the Gateway is genuinely down and
# no id would work anyway.
_failed_client_ids: list[int] = []
# Worker threads connect, release, and reconnect continuously, so logging every
# successful connect at INFO buries the session log in near-identical lines.
# The first one is worth seeing (proves the Gateway is reachable); the rest are
# DEBUG.
_logged_first_connect = False
_HISTORICAL_TICK_LIMIT = 1000
# IBKR hard-rejects durationStr values above "86400 S" (Error 321); wider
# windows must use day units or be truncated.
_IBKR_MAX_SECONDS_DURATION = 86400


def _restore_connection_loop() -> bool:
    """Make the loop this thread's IB connection was created on current again.

    ``prompt_toolkit``'s ``PromptSession`` runs every prompt through a fresh
    ``asyncio`` loop on the main thread and discards it afterwards, so the
    policy silently fabricates a *new* current loop the next time anyone asks.
    ib_insync sync calls run whatever loop is current, but the connection's
    transport stays bound to the loop it was created on — with the two out of
    sync the request is never actually sent and every call dies at
    ``RequestTimeout`` as a bare 30s ``TimeoutError`` (symbol lookups
    included, which made unknown tickers report "market data unavailable"
    instead of the instant "symbol not found").

    Returns False when the recorded loop has been closed, meaning the cached
    connection can never pump messages again and must be rebuilt.
    """
    loop = getattr(_ib_local, "loop", None)
    if loop is None:
        return True
    if loop.is_closed():
        return False
    try:
        current = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        current = None
    if current is not loop:
        asyncio.set_event_loop(loop)
    return True


def get_ib() -> IB:
    """Return this thread's ``IB`` connection, connecting on first call.

    ``ib_insync`` sync calls are bound to the current thread's asyncio event
    loop.  Sharing one ``IB`` instance across the main thread, pipeline worker
    threads, and scorer poll threads can deadlock, so each thread gets its own
    connection and client id.
    """
    global _ib_instance, _next_client_offset, _logged_first_connect
    _ensure_event_loop()

    ib = getattr(_ib_local, "ib", None)
    if ib is not None and ib.isConnected():
        if _restore_connection_loop():
            return ib
        # The connection's loop was closed under it — the socket may be open
        # but no request on it can ever complete. Rebuild from scratch.
        release_ib()

    with _ib_lock:
        ib = getattr(_ib_local, "ib", None)
        if ib is not None and ib.isConnected() and _restore_connection_loop():
            return ib

        client_id = getattr(_ib_local, "client_id", None)
        if client_id is None:
            if _free_client_ids:
                client_id = _free_client_ids.pop(0)
            elif _next_client_offset < IBKR_MAX_CONNECTIONS:
                client_id = IBKR_CLIENT_ID + _next_client_offset
                _next_client_offset += 1
            elif _failed_client_ids:
                client_id = _failed_client_ids.pop(0)
            else:
                raise MarketDataUnavailable(
                    f"IBKR client-id pool exhausted ({IBKR_MAX_CONNECTIONS} "
                    f"concurrent connections from base {IBKR_CLIENT_ID}) — "
                    "a thread is holding a connection without release_ib()"
                )
            _ib_local.client_id = client_id

        ib = IB()
        try:
            ib.connect(
                host=IBKR_HOST,
                port=IBKR_PORT,
                clientId=client_id,
                timeout=8,
                readonly=True,
            )
            ib.reqMarketDataType(1)  # live data
            # Blocking sync requests (reqHistoricalTicks etc.) default to NO
            # timeout (RequestTimeout=0) — an IBKR pacing pause would hang the
            # caller forever and look like a silent stall. Bound every request;
            # timeouts raise, get caught by per-call handlers and degrade to
            # empty results with a warning instead of hanging.
            ib.RequestTimeout = 30.0
            _ib_local.ib = ib
            # Remember which loop this connection's transport lives on, so
            # _restore_connection_loop() can re-home the thread to it after
            # prompt_toolkit (or anything else) swaps the current loop.
            _ib_local.loop = asyncio.get_event_loop_policy().get_event_loop()
            _ib_clients[threading.get_ident()] = ib
            _ib_instance = ib
            logger.log(
                logging.INFO if not _logged_first_connect else logging.DEBUG,
                "IBKR connected to %s:%s (clientId=%s)",
                IBKR_HOST, IBKR_PORT, client_id,
            )
            _logged_first_connect = True
            return ib
        except Exception as exc:
            logger.warning("IBKR connection failed (clientId=%s): %s", client_id, exc)
            # Hand the id back — the socket never opened, so burning it would
            # shrink the usable pool on every failed connect. It goes to the
            # failed list, not the free one, so the next caller tries a fresh
            # id first: if this id is the one the Gateway is refusing, retrying
            # it immediately would just re-spend an 8s connect timeout.
            _ib_local.client_id = None
            if client_id not in _failed_client_ids:
                _failed_client_ids.append(client_id)
            raise


def release_ib() -> None:
    """Disconnect this thread's IB connection and return its client id.

    Call from a worker thread that is about to exit (scorer poll/seed loops,
    stream threads).  The Gateway caps concurrent API connections at 32
    (see IBKR_MAX_CONNECTIONS), and a connection is only freed when the
    socket closes — thread death alone does not release it.  Safe to call
    when the thread never connected.
    """
    ib = getattr(_ib_local, "ib", None)
    client_id = getattr(_ib_local, "client_id", None)
    _ib_local.ib = None
    _ib_local.client_id = None
    _ib_local.loop = None
    with _ib_lock:
        _ib_clients.pop(threading.get_ident(), None)
        if client_id is not None and client_id not in _free_client_ids:
            _free_client_ids.append(client_id)
    if ib is None:
        return
    try:
        if ib.isConnected():
            ib.disconnect()
            logger.debug("IBKR disconnected (clientId=%s)", client_id)
    except Exception as exc:
        logger.debug("IBKR disconnect failed (clientId=%s): %s", client_id, exc)


def disconnect_all() -> None:
    """Close every IB connection opened by this process.

    Used at session shutdown.  Connections opened on other threads cannot be
    cleaned up by those threads once they are gone, so we close them here by
    walking the registry.
    """
    global _ib_instance, _logged_first_connect
    with _ib_lock:
        _logged_first_connect = False
        clients = list(_ib_clients.items())
        _ib_clients.clear()
        _ib_instance = None
    closed = 0
    for _tid, ib in clients:
        try:
            if ib.isConnected():
                ib.disconnect()
                closed += 1
        except Exception as exc:
            logger.debug("IBKR disconnect_all: close failed: %s", exc)
    # This thread's own thread-local slot is part of the registry above; clear
    # it so a later get_ib() on this thread reconnects instead of handing back
    # a dead socket.
    _ib_local.ib = None
    _ib_local.client_id = None
    _ib_local.loop = None
    with _ib_lock:
        _free_client_ids.clear()
        _failed_client_ids.clear()
        _next_client_offset_reset()
    if closed:
        logger.info("[shutdown] IBKR connections closed (%d)", closed)


def _next_client_offset_reset() -> None:
    """Reset the client-id allocator. Caller must hold ``_ib_lock``."""
    global _next_client_offset
    _next_client_offset = 0


# ---------------------------------------------------------------------------
# Historical-request pacing — measured, not gated
# ---------------------------------------------------------------------------
# There used to be a client-side gate here enforcing the documented "60
# historical requests per rolling 10 minutes" budget. Controlled probing
# (2026-08-04, tests/test_files/hist_pacing.py) disproved that rule: the
# Gateway absorbed 150+ small requests per 10-minute window with zero
# pushback, for every bar size and for ticks. The three numbered rules in
# IBKR's historical-limitations doc describe a 2007-era enforcement system
# that no longer exists, so the gate only threw away data (skipped calls,
# empty frames) to ration a budget nothing was spending.
#
# What the Gateway actually does is shape traffic server-side by DELAYING
# responses — never erroring, never dropping:
#
#   * Bars >= 30 secs, small payloads: unshaped. Request freely.
#   * Bars < 30 secs and ALL reqHistoricalTicks (BID_ASK costs 1x, not the
#     documented 2x): one shared queue, ~8 instant calls after idle (~25 at
#     slower pace), then quantized holds (2-6s, 12s worst) capping sustained
#     throughput around one request per 2-2.5s. Spacing sweep: 2.0s gaps drew
#     holds by call ~26; 2.4s and 2.6s gaps ran 100+ calls untouched.
#   * Large payloads (multi-thousand-bar fetches, 1000-tick pages): released
#     on a ~6s cadence regardless of bar size. Only smaller requests avoid it.
#
# Worst case is therefore a 12s-delayed response with full data, which the
# 30s RequestTimeout already bounds. The per-pass tick pagination ceilings
# below cap wall-clock per fetch; nothing else needs client-side pacing.


def _make_contract(symbol: str) -> Stock:
    """Build a US stock contract for the given ticker symbol."""
    return Stock(symbol, "SMART", "USD")


def _format_end_datetime(dt: datetime) -> str:
    """Format an IBKR historical end time in UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%d %H:%M:%S UTC")


def _utc_ts(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Market scanner
# ---------------------------------------------------------------------------

@dataclass
class ScannerRow:
    """One symbol from an IBKR market-scanner result, with its scan rank."""

    symbol: str
    rank: int
    primary_exchange: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nan_safe(val: float) -> Optional[float]:
    """Convert NaN to None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


def _nan_int(val: float) -> int:
    """Convert NaN to 0, otherwise int."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0
    return int(val)


def _bars_to_candles(
    bars, symbol: str, interval: str,
) -> list[OHLCVCandle]:
    """Convert ``ib_insync.BarData`` list to ``OHLCVCandle`` list."""
    candles: list[OHLCVCandle] = []
    for b in bars:
        ts = b.date
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        else:
            # date object from daily bars
            ts = datetime(ts.year, ts.month, ts.day, tzinfo=timezone.utc)
        candles.append(OHLCVCandle(
            symbol=symbol,
            ts=ts,
            open=float(b.open),
            high=float(b.high),
            low=float(b.low),
            close=float(b.close),
            volume=int(b.volume),
            interval=interval,
            vwap=_nan_safe(b.average) if hasattr(b, "average") else None,
        ))
    return candles


def _to_df(candles: list[OHLCVCandle]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(c) for c in candles])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.sort_values("ts", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _tick_to_row(tick) -> dict | None:
    ts = getattr(tick, "time", None)
    price = _nan_safe(getattr(tick, "price", None))
    if ts is None or price is None or price <= 0:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
    else:
        ts = pd.Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
    size = max(1, _nan_int(getattr(tick, "size", 0)))
    attrib = getattr(tick, "tickAttribLast", None)
    unreported = bool(getattr(attrib, "unreported", False)) if attrib is not None else False
    past_limit = bool(getattr(attrib, "pastLimit", False)) if attrib is not None else False
    conditions = str(getattr(tick, "specialConditions", "") or "").strip()
    flags = [part for part in (conditions, "unreported" if unreported else "", "past_limit" if past_limit else "") if part]
    return {
        "ts": ts,
        "price": price,
        "size": size,
        "bid": 0.0,
        "ask": 0.0,
        "exchange": str(getattr(tick, "exchange", "") or ""),
        "flag": "|".join(flags),
        "cancel": False,
        "correction": False,
        "session": "ibkr",
    }


def _bidask_tick_to_row(tick) -> dict | None:
    ts = getattr(tick, "time", None)
    bid = _nan_safe(getattr(tick, "priceBid", None))
    ask = _nan_safe(getattr(tick, "priceAsk", None))
    if ts is None or bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
    else:
        ts = pd.Timestamp(ts).to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
    return {
        "ts": ts,
        "bid": bid,
        "ask": ask,
        "bid_size": _nan_int(getattr(tick, "sizeBid", 0)),
        "ask_size": _nan_int(getattr(tick, "sizeAsk", 0)),
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class IBKRHistoricalClient:
    """IBKR REST-equivalent client for stats and historical OHLCV data.

    Mirrors the ``HistoricalClient`` API from ``tradier_rest.py`` so the
    rest of the pipeline can swap providers transparently.
    """

    def __init__(self) -> None:
        self._disabled = not IBKR_ENABLED
        # Most recent top-gainers snapshot, published for read-only consumers
        # (the live `top` display).  Mirrors ``MomoClient.latest_momos``: the
        # scan itself stays a one-shot, this is just the last result kept
        # around so a reader never has to issue its own scan.
        self.latest_scan: list[ScannerRow] = []
        self.latest_scan_ts: float = 0.0  # time.monotonic() of that scan; 0 = never scanned

    # ------------------------------------------------------------------
    # STATS — Snapshot quotes
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> StockQuote:
        """Fetch a snapshot quote for a single symbol.

        Returns an empty stub when the symbol is unknown; propagates
        :class:`MarketDataUnavailable` when the fetch failed, so the caller
        can tell "no such symbol" from "could not ask".
        """
        quotes = self.get_quotes([symbol])
        if not quotes:
            return StockQuote(
                symbol=symbol, description="", last=None, change=None,
                change_pct=None, volume=0, avg_volume=0, open=None,
                high=None, low=None, prev_close=None, bid=None,
                ask=None, bid_size=0, ask_size=0, last_volume=0,
                trade_ts=None, exchange="",
            )
        return quotes[0]

    def get_quotes(self, symbols: list[str]) -> list[StockQuote]:
        """Fetch snapshot quotes for one or more symbols.

        Raises :class:`MarketDataUnavailable` when the fetch itself fails
        (connection refused at the 32-client cap, pacing stall, socket
        timeout).  Returning ``[]`` there would be indistinguishable from
        "no such symbol" and gets the ticker wrongly rejected.
        """
        if self._disabled:
            return []
        try:
            ib = get_ib()
            contracts = [_make_contract(s) for s in symbols]
            # qualifyContracts returns only the contracts IB could resolve.
            # A symbol missing from the result is genuinely unknown — that,
            # not an empty quote, is the real "symbol not found" signal.
            qualified = ib.qualifyContracts(*contracts) or []
            if not qualified:
                return []
            tickers = ib.reqTickers(*qualified, regulatorySnapshot=False)
        except Exception as exc:
            # ib_insync raises bare/empty-message errors for a refused
            # connection, so log the type too — an empty "failed: " line is
            # what made this class of failure unreadable in the logs.
            logger.warning(
                "IBKR get_quotes(%s) failed: %s: %s",
                ",".join(symbols), type(exc).__name__, exc or "<no detail>",
            )
            raise MarketDataUnavailable(
                f"IBKR quote fetch failed for {','.join(symbols)}: "
                f"{type(exc).__name__}: {exc or '<no detail>'}"
            ) from exc

        result: list[StockQuote] = []
        for ticker in tickers:
            last = _nan_safe(ticker.last)
            close = _nan_safe(ticker.close)
            change: Optional[float] = None
            change_pct: Optional[float] = None
            if last is not None and close is not None and close != 0:
                change = round(last - close, 4)
                change_pct = round((change / close) * 100, 2)

            sym = ticker.contract.symbol if ticker.contract else ""
            result.append(StockQuote(
                symbol=sym,
                description=ticker.contract.description if ticker.contract else "",
                last=last,
                change=change,
                change_pct=change_pct,
                volume=_nan_int(ticker.volume),
                avg_volume=_nan_int(ticker.avVolume),
                open=_nan_safe(ticker.open),
                high=_nan_safe(ticker.high),
                low=_nan_safe(ticker.low),
                prev_close=close,
                bid=_nan_safe(ticker.bid),
                ask=_nan_safe(ticker.ask),
                bid_size=_nan_int(ticker.bidSize),
                ask_size=_nan_int(ticker.askSize),
                last_volume=_nan_int(ticker.lastSize),
                trade_ts=ticker.time,
                exchange=ticker.contract.exchange if ticker.contract else "",
            ))
        return result

    def get_quote_df(self, symbols: list[str]) -> pd.DataFrame:
        """Multi-symbol snapshot as a DataFrame."""
        return pd.DataFrame([asdict(q) for q in self.get_quotes(symbols)])

    def search_symbol(self, query: str) -> Optional[str]:
        """Search for a ticker by partial symbol or company name."""
        if self._disabled:
            return None
        try:
            ib = get_ib()
            matches = ib.reqMatchingSymbols(query)
            if matches:
                for desc in matches:
                    if desc.contract and desc.contract.secType == "STK":
                        return desc.contract.symbol
        except Exception as exc:
            logger.warning("IBKR search_symbol failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # MARKET SCANNER
    # ------------------------------------------------------------------

    def scan_top_gainers(self) -> list[ScannerRow]:
        """One-shot market scan for the session's top percent gainers.

        ``reqScannerData`` opens a scanner subscription and cancels it once
        the first result set lands, so a poll loop never accumulates
        subscriptions against IBKR's ~10-concurrent cap.  Scanner requests
        are *not* historical requests and are untouched by the historical
        traffic shaper (measured: 160-730ms, clean at 1s polling).

        Returns rank-ordered rows (best rank first), one per symbol.  An
        empty list is a valid quiet result — no name currently clears the
        filters.  Raises :class:`MarketDataUnavailable` when the scan itself
        failed, so the caller can tell that apart from "nothing qualified".
        """
        if self._disabled:
            return []
        scan_code, change_tag = scanner_ibkr_scan_spec()
        try:
            ib = get_ib()
            sub = ScannerSubscription(
                numberOfRows=SCANNER_IBKR_ROWS,
                instrument=SCANNER_IBKR_INSTRUMENT,
                locationCode=SCANNER_IBKR_LOCATION,
                scanCode=scan_code,
                abovePrice=PRICE_MIN,
                belowPrice=PRICE_MAX,
                aboveVolume=SCANNER_IBKR_MIN_VOLUME,
            )
            rows = ib.reqScannerData(
                # ":g" so a whole-number threshold goes over as "5", not "5.0".
                sub, [], [TagValue(change_tag, f"{SCANNER_IBKR_MIN_CHG:g}")],
            )
        except Exception as exc:
            # Same reasoning as get_quotes: ib_insync raises bare errors for a
            # refused connection, so name the type or the log line is empty.
            logger.warning(
                "IBKR scan_top_gainers(%s) failed: %s: %s",
                scan_code, type(exc).__name__, exc or "<no detail>",
            )
            raise MarketDataUnavailable(
                f"IBKR scanner {scan_code} failed: "
                f"{type(exc).__name__}: {exc or '<no detail>'}"
            ) from exc

        best: dict[str, ScannerRow] = {}
        for row in rows or []:
            details = getattr(row, "contractDetails", None)
            contract = getattr(details, "contract", None)
            symbol = str(getattr(contract, "symbol", "") or "").upper()
            if not symbol:
                continue
            try:
                rank = int(getattr(row, "rank", 0) or 0)
            except (TypeError, ValueError):
                continue
            prior = best.get(symbol)
            # One name can list on several exchanges — keep the best (lowest) rank.
            if prior is not None and prior.rank <= rank:
                continue
            best[symbol] = ScannerRow(
                symbol=symbol,
                rank=rank,
                primary_exchange=str(getattr(contract, "primaryExchange", "") or ""),
            )
        ranked = sorted(best.values(), key=lambda r: r.rank)
        # Publish the snapshot in one rebind so a concurrent reader sees either
        # the old list or the new one, never a half-built one.  A failed scan
        # raises above without touching it: the last good snapshot survives a
        # Gateway outage, and its age is what tells the reader it went stale.
        self.latest_scan = ranked
        self.latest_scan_ts = time.monotonic()
        return ranked

    def get_timesales_tick(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session_filter: str = "all",
    ) -> list[dict]:
        """Historical trade ticks normalized to the Tradier timesales schema."""
        if self._disabled:
            return []
        start_ts = _utc_ts(start)
        end_ts = _utc_ts(end)
        if end_ts <= start_ts:
            return []
        try:
            ib = get_ib()
            contract = _make_contract(symbol)
            ib.qualifyContracts(contract)
        except Exception as exc:
            logger.warning("IBKR get_timesales_tick(%s) setup failed: %s", symbol, exc)
            return []

        use_rth = session_filter.lower() in {"regular", "rth", "open"}
        quote_start_ts = start_ts - pd.Timedelta(seconds=60)

        # reqHistoricalTicks returns at most 1000 ticks, so a denser window has
        # to be bisected and re-requested — that is pagination, not a retry.
        # On a liquid name it recurses until the windows are ~1s wide, and
        # BID_ASK (every quote update) is dense enough that one minute of tape
        # can cost 50+ requests. Tick requests are shaped by the Gateway at
        # ~1 per 2.6s sustained (with 1000-tick pages also paying the payload
        # queue), so an unbounded fetch can burn minutes of wall-clock. Each
        # pass gets its own ceiling, and truncation is logged rather than
        # silent.
        budget = {"TRADES": IBKR_TICK_MAX_REQUESTS_TRADES,
                  "BID_ASK": IBKR_TICK_MAX_REQUESTS_QUOTES}
        truncated: set[str] = set()

        def _collect_ticks(
            what_to_show: str,
            window_start: pd.Timestamp,
            window_end: pd.Timestamp,
        ) -> list:
            if window_end <= window_start:
                return []
            if budget.get(what_to_show, 0) <= 0:
                if what_to_show not in truncated:
                    truncated.add(what_to_show)
                    logger.warning(
                        "IBKR get_timesales_tick(%s): %s hit its %d-request "
                        "ceiling — returning partial tape for %s -> %s",
                        symbol, what_to_show,
                        IBKR_TICK_MAX_REQUESTS_TRADES if what_to_show == "TRADES"
                        else IBKR_TICK_MAX_REQUESTS_QUOTES,
                        window_start.isoformat(), window_end.isoformat(),
                    )
                return []
            budget[what_to_show] -= 1
            try:
                ticks = ib.reqHistoricalTicks(
                    contract,
                    startDateTime=window_start.to_pydatetime(),
                    endDateTime=window_end.to_pydatetime(),
                    numberOfTicks=_HISTORICAL_TICK_LIMIT,
                    whatToShow=what_to_show,
                    useRth=use_rth,
                    ignoreSize=False,
                    miscOptions=[],
                )
            except Exception as exc:
                logger.warning(
                    "IBKR get_timesales_tick(%s, %s, %s -> %s) failed: %s: %s",
                    symbol, what_to_show, window_start.isoformat(),
                    window_end.isoformat(), type(exc).__name__, exc or "<no detail>",
                )
                return []

            span = window_end - window_start
            if len(ticks) >= _HISTORICAL_TICK_LIMIT and span > pd.Timedelta(seconds=1):
                mid = window_start + (span / 2)
                return _collect_ticks(what_to_show, window_start, mid) + _collect_ticks(what_to_show, mid, window_end)
            return list(ticks)

        # TRADES first, deliberately. The tape IS the result; BID_ASK only
        # supplies the quote context stitched onto each print below — if
        # anything must be cut short (ceilings above, Gateway shaping), it
        # should be the optional data, never the essential data.
        rows: list[dict] = []
        quote_idx = 0
        last_quote: dict | None = None
        trade_rows: list[dict] = []
        for tick in _collect_ticks("TRADES", start_ts, end_ts):
            row = _tick_to_row(tick)
            if row is None:
                continue
            row_ts = _utc_ts(row["ts"])
            if row_ts < start_ts or row_ts > end_ts:
                continue
            trade_rows.append(row)
        trade_rows.sort(key=lambda row: row["ts"])

        quote_rows: list[dict] = []
        if trade_rows:
            for tick in _collect_ticks("BID_ASK", quote_start_ts, end_ts):
                quote_row = _bidask_tick_to_row(tick)
                if quote_row is None:
                    continue
                quote_ts = _utc_ts(quote_row["ts"])
                if quote_ts > end_ts:
                    continue
                quote_rows.append(quote_row)
            quote_rows.sort(key=lambda row: row["ts"])

        for row in trade_rows:
            row_ts = _utc_ts(row["ts"])
            while quote_idx < len(quote_rows) and _utc_ts(quote_rows[quote_idx]["ts"]) <= row_ts:
                last_quote = quote_rows[quote_idx]
                quote_idx += 1
            if last_quote is not None:
                row["bid"] = last_quote["bid"]
                row["ask"] = last_quote["ask"]
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # HISTORICAL CANDLES
    # ------------------------------------------------------------------

    def _get_historical(
        self,
        symbol: str,
        duration: str,
        bar_size: str,
        interval_label: str,
        n: Optional[int] = None,
        end_datetime: Optional[datetime] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generic historical data fetch.

        Degrades to an empty DataFrame — with a warning naming the cause —
        when the request fails.  Never stalls the caller beyond the bounded
        RequestTimeout (Gateway shaping delays a response 12s at worst).
        """
        if self._disabled:
            return pd.DataFrame()
        try:
            ib = get_ib()
            contract = _make_contract(symbol)
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="" if end_datetime is None else _format_end_datetime(end_datetime),
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=2,
                timeout=30,
            )
        except Exception as exc:
            logger.warning(
                "IBKR _get_historical(%s, %s) failed: %s: %s",
                symbol, bar_size, type(exc).__name__, exc or "<no detail>",
            )
            return pd.DataFrame()

        candles = _bars_to_candles(bars, symbol, interval_label)
        df = _to_df(candles)
        if not df.empty and (start is not None or end is not None):
            mask = pd.Series(True, index=df.index)
            if start is not None:
                mask &= df["ts"] >= _utc_ts(start)
            if end is not None:
                mask &= df["ts"] <= _utc_ts(end)
            df = df.loc[mask].reset_index(drop=True)
        if n and len(df) > n:
            df = df.tail(n).reset_index(drop=True)
        return df

    def _get_historical_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None,
        bar_size: str,
        interval_label: str,
        n: Optional[int] = None,
    ) -> pd.DataFrame:
        end = end or datetime.now(tz=timezone.utc)
        start_ts = _utc_ts(start)
        end_ts = _utc_ts(end)
        if end_ts <= start_ts:
            return pd.DataFrame()
        duration_seconds = max(1, int(math.ceil((end_ts - start_ts).total_seconds())))
        # IBKR rejects "S"-unit durations above 86400 outright (Error 321:
        # "request for greater than 86400 seconds rejected"), so a wider
        # window must not be passed through verbatim.  Sub-minute bars are
        # capped at one day by IBKR anyway — truncate to the most recent day
        # and say so.  Minute-and-up bars switch to day units, which IBKR
        # accepts for multi-day spans; the start/end mask below still trims
        # the result to the exact window requested.
        if duration_seconds > _IBKR_MAX_SECONDS_DURATION:
            if bar_size.endswith("secs"):
                logger.warning(
                    "[%s] %s window of %ds exceeds IBKR's 86400s cap — "
                    "truncating to the most recent day",
                    symbol, bar_size, duration_seconds,
                )
                start_ts = end_ts - pd.Timedelta(seconds=_IBKR_MAX_SECONDS_DURATION)
                duration = f"{_IBKR_MAX_SECONDS_DURATION} S"
            else:
                days = max(1, int(math.ceil(duration_seconds / 86400)))
                duration = f"{days} D"
        else:
            duration = f"{duration_seconds} S"
        return self._get_historical(
            symbol,
            duration=duration,
            bar_size=bar_size,
            interval_label=interval_label,
            n=n,
            end_datetime=end_ts.to_pydatetime(),
            start=start_ts.to_pydatetime(),
            end=end_ts.to_pydatetime(),
        )

    def get_candles_10sec(self, symbol: str, n: int = 60) -> pd.DataFrame:
        """Last *n* 10-second OHLCV bars."""
        hours = max(1, CANDLES_10SEC_FETCH_HOURS)
        return self._get_historical(
            symbol,
            duration=f"{hours * 3600} S",
            bar_size="10 secs",
            interval_label="10sec",
            n=n,
        )

    def get_candles_10sec_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
        n: int | None = None,
    ) -> pd.DataFrame:
        """10-second OHLCV bars for a precise UTC-aware window."""
        return self._get_historical_window(
            symbol,
            start=start,
            end=end,
            bar_size="10 secs",
            interval_label="10sec",
            n=n,
        )

    def get_candles_1min(self, symbol: str, n: int = 30) -> pd.DataFrame:
        """Last *n* completed 1-minute candles."""
        return self._get_historical(
            symbol,
            duration=f"{CANDLES_1MIN_FETCH_DAYS} D",
            bar_size="1 min",
            interval_label="1min",
            n=n,
        )

    def get_candles_1min_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """1-minute OHLCV bars for a precise UTC-aware window."""
        return self._get_historical_window(
            symbol,
            start=start,
            end=end,
            bar_size="1 min",
            interval_label="1min",
        )

    def get_candles_1min_recent(
        self, symbol: str, n: int = 5, lookback_minutes: int = 8,
    ) -> pd.DataFrame:
        """Recent 1-minute candles for live refresh loops."""
        minutes = max(lookback_minutes, n + 1)
        return self._get_historical(
            symbol,
            duration=f"{minutes * 60} S",
            bar_size="1 min",
            interval_label="1min",
            n=n,
        )

    def get_candles_5min(
        self, symbol: str, days_back: int = CANDLES_5MIN_FETCH_DAYS,
    ) -> pd.DataFrame:
        """5-minute OHLCV candles."""
        return self._get_historical(
            symbol,
            duration=f"{days_back} D",
            bar_size="5 mins",
            interval_label="5min",
        )

    def get_candles_5min_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """5-minute OHLCV bars for a precise UTC-aware window."""
        return self._get_historical_window(
            symbol,
            start=start,
            end=end,
            bar_size="5 mins",
            interval_label="5min",
        )

    def get_candles_15min(self, symbol: str, n: int = 20) -> pd.DataFrame:
        """Last *n* 15-minute candles."""
        return self._get_historical(
            symbol,
            duration=f"{CANDLES_15MIN_FETCH_DAYS} D",
            bar_size="15 mins",
            interval_label="15min",
            n=n,
        )

    def get_candles_15min_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
        n: int | None = None,
    ) -> pd.DataFrame:
        """15-minute OHLCV bars for a precise UTC-aware window."""
        return self._get_historical_window(
            symbol,
            start=start,
            end=end,
            bar_size="15 mins",
            interval_label="15min",
            n=n,
        )

    def get_candles_daily(
        self,
        symbol: str,
        days_back: int = 30,
        include_current_day: bool = True,
    ) -> pd.DataFrame:
        """Daily OHLCV candles.

        *include_current_day* mirrors the Tradier client: dropping the current
        session is what a caller wants when today's partial bar would pollute
        a historical baseline (prior-day levels, rvol denominators).  The
        signature has to match Tradier's — the provider is swapped behind a
        common interface, and a missing kwarg here raises TypeError inside the
        caller's ``except Exception``, silently emptying the daily frame.
        """
        end = None
        if not include_current_day:
            # Cut at today's ET midnight so the in-progress session's bar is
            # excluded; IBKR stamps daily bars at the session date.
            end = MarketClock.today()
        return self._get_historical(
            symbol,
            duration=f"{days_back} D",
            bar_size="1 day",
            interval_label="1day",
            end=end,
        )

    def get_closed_dates(self, months_back: int = 2) -> frozenset:
        """Fetch closed market dates (holidays).

        IBKR doesn't expose a calendar API like Tradier, so we use
        ``reqHistoricalSchedule`` and derive closed weekdays from gaps
        in the trading schedule.  Falls back to an empty set on error.
        """
        if self._disabled:
            return frozenset()
        try:
            from datetime import date
            ib = get_ib()
            contract = _make_contract("SPY")
            ib.qualifyContracts(contract)
            sched = ib.reqHistoricalSchedule(
                contract, numDays=months_back * 30, useRTH=True,
            )
            trading_dates: set[date] = set()
            for session in sched.sessions:
                try:
                    d = datetime.strptime(
                        session.startDateTime[:8], "%Y%m%d",
                    ).date()
                    trading_dates.add(d)
                except (ValueError, IndexError):
                    continue
            # Derive closed weekdays
            closed: set[date] = set()
            today = datetime.now(tz=timezone.utc).date()
            for offset in range(months_back * 30 + 1):
                d = today - timedelta(days=offset)
                if d.weekday() < 5 and d not in trading_dates:
                    closed.add(d)
            return frozenset(closed)
        except Exception as exc:
            logger.warning("IBKR get_closed_dates failed: %s", exc)
            return frozenset()
