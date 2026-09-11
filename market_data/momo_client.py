"""
market_data/momo_client.py

Momoscreener client — fetches momentum stocks and hot news from
momoscreener.com's public API via WebSocket (live) and REST (fallback).

No API key required. Data is free and unauthenticated.

Discovered by inspecting https://momoscreener.com/static/js/main.6d3ea60e.js
(the single React bundle served by the site). All endpoints share the base URL
``https://momoscreener.com/api`` (``ja.apiUrl = "/api"`` in the bundle).


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REST ENDPOINTS  (GET, unauthenticated, no key)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GET /api/momo
    Returns the top ~10 daily percentage gainers.
    Response: {"message": [ <stock>, ... ]}

GET /api/momo?int=5&change=3
    Momentum scanner — stocks with ≥change% move in the last int minutes.
    Add &type=halted to filter to halted stocks only.
    Response: {"message": [ <stock>, ... ]}

    Each <stock> entry contains:
        symbol, quote{lastPrice, closePrice, totalVolume,
            markPercentChange, bidPrice, askPrice, highPrice, lowPrice},
        stats{floatShares, outstandingShares, ...},
        news{headline, source, url, newsTimeInLong},
        updatedAt, change5m, spread, live{lastPrice, totalVolume}

GET /api/hotnews
    Hot news headlines (separate feed from scanner news).
    Response: {"data": [ {symbol, title, source, url, publishedAt}, ... ]}

GET /api/stocklist?q=<query>
    Symbol autocomplete / search.
    Response: [ {symbol, description}, ... ]
    Example: q=ISPC → [{"symbol":"ISPC","description":"ISPECIMEN INC. COMMON STOCK"}]

GET /api/alpaca/stock?q=<symbol>
    Full daily OHLCV history for any symbol (sourced from Alpaca).
    Response: {"error": 0, "message": {"history1day": [
        {o, h, l, c, v, vw, n, t}, ...   (chronological, oldest first)
    ]}}
    Fields: o=open, h=high, l=low, c=close, v=volume, vw=VWAP,
            n=trade count, t=ISO timestamp

GET /api/p/ticker/chart?q=<symbol>&interval=<1|5>
    Intraday candles (1-min or 5-min). Only returns data for stocks
    currently tracked by the momo scanner; returns error for idle tickers.
    Response: {"error": 0, "message": {"history": [ {o,h,l,c,v,...}, ... ]}}

GET /api/earningcalendar
    Upcoming earnings calendar (no params needed).
    Response: {"error": 0, "data": [
        {symbol, date, hour, year, quarter,
         epsEstimate, epsActual, revenueEstimate, revenueActual,
         marketCap, peRatio}, ...
    ]}


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOCKET.IO v4 — TRANSPORT BEHAVIOUR  (critical quirk)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The server supports two Socket.IO transports, but they serve different roles:

  WebSocket (wss://momoscreener.com/socket.io/?EIO=4&transport=websocket)
      ┌─ PUSH-ONLY events (server → client, no ack expected):
      │     "5m3c"      — momentum scanner snapshot (primary data source)
      │     "gapper"    — gap-up scanner
      │     "halt"      — halted stocks
      │     "hotNews"   — hot news feed
      └─ Emitting "livequote" over WS causes the server to close the
         connection immediately (code 1005 — no status). Do NOT use WS
         for livequote.

  HTTP Polling (https://momoscreener.com/socket.io/?EIO=4&transport=polling)
      ┌─ REQUEST/RESPONSE events (client emits, server answers on next GET):
      │     "livequote" — full stock profile for any symbol (see below)
      │     "stop"      — stop a livequote subscription
      │     "stoplivequote" — alias for stop
      └─ Works for every symbol, not just active momo stocks.

Polling handshake sequence to get a livequote:
    1. GET  /socket.io/?EIO=4&transport=polling
           → {"sid":"...","upgrades":["websocket"],...}   (grab sid)
    2. POST /socket.io/?EIO=4&transport=polling&sid=<sid>
           body: "40"    (SIO namespace connect)
    3. GET  /socket.io/?EIO=4&transport=polling&sid=<sid>
           → "40{"sid":"..."}"  (namespace ack, drain it)
    4. POST /socket.io/?EIO=4&transport=polling&sid=<sid>
           body: '42["livequote","<SYMBOL>"]'
    5. GET  /socket.io/?EIO=4&transport=polling&sid=<sid>
           → '42["livequote", { "data": [ <full stock profile> ] }]'


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LIVEQUOTE RESPONSE SCHEMA  (the richest data source per symbol)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

42["livequote", {
  "data": [{
    "symbol": "ISPC",
    "quote": {
      "lastPrice": 0.18095,
      "closePrice": 0.1451,         # previous close
      "totalVolume": 42989930,
      "markPercentChange": 24.95,   # % change from close
      "bidPrice": 0.1805,
      "askPrice": 0.1814,
      "highPrice": 0.1518,          # day high
      "lowPrice": 0.1257            # day low
    },
    "live": {
      "lastPrice": 0.1782,          # real-time last (more current than quote)
      "totalVolume": 43094403,
      "quoteTime": 1775769225000    # ms epoch
    },
    "stats": {
      "floatShares": 5499628,       # ← SHARE FLOAT
      "outstandingShares": 5554481,
      "price": 0.1782,
      "volume": 25139491,           # session volume so far
      "avgVolume": 14165511,        # 30-day avg daily volume
      "eps": -1.07,
      "marketCap": 1367942,         # in dollars (not millions)
      "peRatio": null
    },
    "news": [ {headline, source, newsTimeInLong, id}, ... ]
  }]
}]

Float data in the scanner /api/momo response:
    Each stock entry also contains stats.floatShares (if tracked by the
    scanner). The JS prefers quote.fundamental.marketCapFloat (in millions)
    when available, falling back to stats.floatShares.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from market_data.momo_client import MomoClient

    client = MomoClient()

    # REST (synchronous, on-demand)
    momos   = client.get_momos()          # momentum scanner (5-min ≥3%)
    gainers = client.get_top_gainers()    # top daily % gainers
    news    = client.get_news()           # headlines from all momo stocks
    ispc    = client.get_news("ISPC")     # headlines for a single ticker
    hot     = client.get_hot_news()       # hot news (separate endpoint)

    # WebSocket (background thread, push-based — 5m3c / gapper / halt / hotNews)
    client.start_stream()               # start background listener
    latest = client.latest_momos        # last snapshot from WS
    client.stop_stream()                # tear down

    # Float / full profile (polling transport — works for any symbol)
    # Use get_float() which reads from the scanner cache; for arbitrary
    # symbols not on the scanner, emit "livequote" via the polling transport
    # directly (steps 1-5 above) and read stats.floatShares.
    float_data = client.get_float("ISPC")
"""

import asyncio
import certifi
import json
import logging
import ssl
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

import orjson
import requests
import websockets

from config import HTTP_TIMEOUT
from market_data import resilient_session
from market_data.fmp_client import FloatData

logger = logging.getLogger(__name__)

_BASE_URL = "https://momoscreener.com/api"
_SIO_POLL = "https://momoscreener.com/socket.io/?EIO=4&transport=polling"
_SIO_WS   = "wss://momoscreener.com/socket.io/?EIO=4&transport=websocket&sid={sid}"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Referer": "https://momoscreener.com/scanner",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MomoNews:
    """A news item from Momoscreener."""
    symbol: str
    headline: str
    source: str
    url: str
    published_at: Optional[datetime]


@dataclass
class MomoStock:
    """A single momentum stock from Momoscreener."""
    symbol: str
    last_price: float
    close_price: float
    change_pct: float
    change_5m: float
    total_volume: int
    high_price: float
    low_price: float
    bid: float
    ask: float
    spread: Optional[float]
    news: Optional[MomoNews]
    updated_at: Optional[datetime]
    float_shares: Optional[int] = None
    outstanding_shares: Optional[int] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class MomoClient:
    """
    Momoscreener public API client.
    No API key needed — all endpoints are unauthenticated.

    REST methods are synchronous and on-demand.
    WebSocket streaming runs in a background thread and pushes
    momentum updates to ``latest_momos`` (and optionally a callback).
    """

    def __init__(self) -> None:
        self._session = resilient_session()
        self._session.headers.update({**_HEADERS, "Accept": "application/json"})
        self.paused: bool = False  # when True, get_float/get_news return empty
        self._ever_received: bool = False  # True once a 5m3c event arrives this session
        self._user_initiated: bool = False  # True once user has manually started the stream

        # WebSocket state
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_stop = threading.Event()
        self._on_momo: Optional[Callable[[list[MomoStock]], None]] = None
        self.latest_momos: list[MomoStock] = []

        # Polling session for livequote / float (reused across calls)
        # connect_polling() does the 3-step SIO handshake upfront; after that
        # get_float_livequote() only needs 2 round-trips (~100 ms).
        self._poll_session: Optional[requests.Session] = None
        self._poll_url: str = ""
        self._poll_lock = threading.RLock()
        self._poll_ready: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """GET a JSON endpoint, return parsed dict or None on failure."""
        url = f"{_BASE_URL}/{path}"
        try:
            resp = self._session.get(url, params=params, timeout=HTTP_TIMEOUT)
            if not resp.ok:
                body = resp.text[:200].strip()
                logger.warning("[momo] %s %s on /%s: %s", resp.status_code, resp.reason, path, body)
                return None
            return resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("[momo] Request failed (/%s): %s", path, exc)
            return None

    @staticmethod
    def _parse_news(raw) -> Optional[MomoNews]:
        """Parse the embedded news dict from a momo stock entry."""
        if not raw or not isinstance(raw, dict) or not raw.get("headline"):
            return None
        ts = raw.get("newsTimeInLong")
        published = None
        if ts:
            try:
                published = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            except (OSError, ValueError):
                pass
        import html as _html
        return MomoNews(
            symbol=raw.get("symbol", ""),
            headline=_html.unescape(raw.get("headline", "")),
            source=raw.get("source", ""),
            url=raw.get("url", ""),
            published_at=published or _parse_iso(raw.get("publishedAt") or raw.get("date")),
        )

    @staticmethod
    def _parse_stock(raw: dict) -> MomoStock:
        """Parse a single stock entry from the API response."""
        quote = raw.get("quote", {})
        stats = raw.get("stats", {})
        return MomoStock(
            symbol=raw.get("symbol", ""),
            last_price=quote.get("lastPrice", 0.0),
            close_price=quote.get("closePrice", 0.0),
            change_pct=quote.get("markPercentChange", 0.0) or raw.get("change", 0.0),
            change_5m=raw.get("change5m", 0.0),
            total_volume=int(quote.get("totalVolume", 0)),
            high_price=quote.get("highPrice", 0.0),
            low_price=quote.get("lowPrice", 0.0),
            bid=quote.get("bidPrice", 0.0),
            ask=quote.get("askPrice", 0.0),
            spread=raw.get("spread"),
            news=MomoClient._parse_news(raw.get("news")),
            updated_at=_parse_iso(raw.get("updatedAt")),
            float_shares=stats.get("floatShares"),
            outstanding_shares=stats.get("outstandingShares"),
        )

    # ------------------------------------------------------------------
    # Public API — REST (synchronous, on-demand)
    # ------------------------------------------------------------------

    def get_momos(self, min_change_pct: int = 3) -> list[MomoStock]:
        """Momentum scanner — stocks with ≥*min_change_pct*% move in 5 min.

        Duplicates (multiple 5-min snapshots) are collapsed to the most
        recent entry per symbol.
        """
        data = self._get("momo", params={"int": 5, "change": min_change_pct})
        if not data:
            return []
        items = data.get("message", [])
        # Deduplicate: keep first occurrence (most recent) per symbol
        seen: set[str] = set()
        results: list[MomoStock] = []
        for r in items:
            stock = self._parse_stock(r)
            if stock.symbol not in seen:
                seen.add(stock.symbol)
                results.append(stock)
        return results

    def get_momos_raw(self, min_change_pct: int = 3) -> list[MomoStock]:
        """Like get_momos() but returns ALL snapshots without deduplication.

        Each entry carries its own ``updated_at`` timestamp so callers can
        measure how many times a symbol appeared within a recent time window.
        """
        data = self._get("momo", params={"int": 5, "change": min_change_pct})
        if not data:
            return []
        return [self._parse_stock(r) for r in data.get("message", []) if isinstance(r, dict)]

    def get_top_gainers(self) -> list[MomoStock]:
        """Top daily percentage gainers (default scanner view)."""
        data = self._get("momo")
        if not data:
            return []
        items = data.get("message", [])
        return [self._parse_stock(r) for r in items]

    def get_hot_news(self) -> list[MomoNews]:
        """Hot news headlines."""
        data = self._get("hotnews")
        if not data:
            return []

        items = data.get("data", [])
        results: list[MomoNews] = []
        for item in items:
            results.append(MomoNews(
                symbol=item.get("symbol", ""),
                headline=item.get("title", "") or item.get("headline", ""),
                source=item.get("source", ""),
                url=item.get("url", ""),
                published_at=_parse_iso(item.get("publishedAt") or item.get("date")),
            ))
        return results

    def get_news(self, symbol: Optional[str] = None) -> list[MomoNews]:
        """News headlines from the momentum scanner.

        Each momo stock carries one headline.  If *symbol* is given,
        only return news matching that ticker (case-insensitive).
        """
        if self.paused:
            return []
        momos = self.latest_momos or self.get_momos()
        results: list[MomoNews] = []
        for s in momos:
            if s.news is None:
                continue
            if symbol and s.symbol.upper() != symbol.upper():
                continue
            results.append(s.news)
        return results

    def get_symbols(self) -> list[str]:
        """Return just the ticker symbols from the default momo scanner."""
        return [s.symbol for s in self.get_momos()]

    def get_float(self, symbol: str) -> Optional[FloatData]:
        """Float data for *symbol*.

        Priority:
          1. WS push cache (``latest_momos``) — zero cost.
          2. REST scanner (``/api/momo``) — only active momo stocks.
          3. ``livequote`` polling — works for **any** symbol; requires
             ``connect_polling()`` to have been called first (or reconnects
             automatically on the first miss).
        """
        if self.paused:
            return None
        sym = symbol.upper()

        def _extract(momos: list) -> Optional[FloatData]:
            for stock in momos:
                if stock.symbol.upper() == sym and stock.float_shares:
                    fs = float(stock.float_shares)
                    os_shares = float(stock.outstanding_shares or 0)
                    pct = (fs / os_shares * 100) if os_shares > 0 else 0.0
                    return FloatData(
                        symbol=sym,
                        float_shares=fs,
                        float_shares_m=fs / 1_000_000,
                        outstanding_shares=os_shares,
                        free_float_pct=pct,
                    )
            return None

        # 1. Push-cached WS snapshot (zero cost)
        result = _extract(self.latest_momos)
        if result:
            return result
        # 2. REST scanner (active momo stocks only)
        result = _extract(self.get_momos())
        if result:
            return result
        # 3. Livequote polling — universal fallback
        return self.get_float_livequote(sym)

    # ------------------------------------------------------------------
    # Public API — Polling session (livequote / float for any symbol)
    # ------------------------------------------------------------------

    def connect_polling(self) -> bool:
        """Perform the Socket.IO polling handshake upfront (steps 1-3).

        Step 1 — GET  /socket.io/?EIO=4&transport=polling
                  ← {"sid":"...","pingInterval":25000,...}
        Step 2 — POST /socket.io/?EIO=4&transport=polling&sid=<sid>
                  body "40"  (SIO namespace connect packet)
        Step 3 — GET  /socket.io/?EIO=4&transport=polling&sid=<sid>
                  ← "40{...}"  (namespace ack — drain it)

        After this, ``get_float_livequote()`` only needs steps 4+5
        (~100 ms per call).  Returns True on success.
        """
        try:
            sess = requests.Session()
            sess.headers.update({**_HEADERS, "Accept": "*/*"})

            # Step 1: handshake
            r1 = sess.get(_SIO_POLL, timeout=HTTP_TIMEOUT)
            if not r1.ok:
                logger.warning(f"Momo polling handshake failed: {r1.status_code}")
                return False
            idx = r1.text.find("{")
            if idx < 0:
                return False
            sid = orjson.loads(r1.text[idx:]).get("sid")
            if not sid:
                return False
            url = f"{_SIO_POLL}&sid={sid}"

            # Step 2: namespace connect
            r2 = sess.post(url, data="40",
                           headers={"Content-Type": "text/plain"},
                           timeout=HTTP_TIMEOUT)
            if not r2.ok:
                logger.warning(f"Momo polling namespace connect failed: {r2.status_code}")
                return False

            # Step 3: drain ack
            sess.get(url, timeout=HTTP_TIMEOUT)

            with self._poll_lock:
                self._poll_session = sess
                self._poll_url = url
                self._poll_ready = True

            logger.debug("Momo polling session established")
            return True
        except Exception as exc:
            logger.warning(f"Momo polling connect error: {exc}")
            return False

    @property
    def polling_ready(self) -> bool:
        """True if the polling session has been established successfully."""
        return self._poll_ready

    def get_float_livequote(self, symbol: str) -> Optional[FloatData]:
        """Fetch float (and full stats) for *any* symbol via the Socket.IO
        polling transport's ``livequote`` event.

        Unlike ``get_float()``, this is **not** limited to stocks currently
        on the momentum scanner.

        Step 4 — POST emit '42["livequote","<SYMBOL>"]'
        Step 5 — GET  response containing stats.floatShares

        The polling session is reused across calls.  On any transport error
        the session is reset and reconnected automatically on the next call.
        """
        if self.paused:
            return None
        sym = symbol.upper()

        with self._poll_lock:
            # Reconnect if session not ready or previously expired
            if not self._poll_ready:
                if not self.connect_polling():
                    return None
            try:
                # Step 4: emit livequote
                r4 = self._poll_session.post(
                    self._poll_url,
                    data=f'42["livequote","{sym}"]',
                    headers={"Content-Type": "text/plain"},
                    timeout=HTTP_TIMEOUT,
                )
                if not r4.ok:
                    raise ValueError(f"emit failed: {r4.status_code}")

                # Step 5: read response (may contain multiple SIO packets
                # separated by \x1e; scan for the livequote event)
                r5 = self._poll_session.get(self._poll_url, timeout=HTTP_TIMEOUT)
                if not r5.ok:
                    raise ValueError(f"read failed: {r5.status_code}")

                for packet in r5.text.split("\x1e"):
                    if not packet.startswith("42"):
                        continue
                    try:
                        parsed = orjson.loads(packet[2:])
                    except (orjson.JSONDecodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(parsed, list) or parsed[0] != "livequote":
                        continue
                    payload = parsed[1] if len(parsed) > 1 else {}
                    data = payload.get("data", [])
                    if not data:
                        return None
                    stats = (data[0].get("stats") or {})
                    fs = float(stats.get("floatShares") or 0)
                    if fs <= 0:
                        return None
                    os_shares = float(stats.get("outstandingShares") or 0)
                    pct = (fs / os_shares * 100) if os_shares > 0 else 0.0
                    return FloatData(
                        symbol=sym,
                        float_shares=fs,
                        float_shares_m=fs / 1_000_000,
                        outstanding_shares=os_shares,
                        free_float_pct=pct,
                    )
                return None

            except Exception as exc:
                logger.warning(f"Momo livequote float error ({sym}): {exc}")
                # Session may be expired — force reconnect on next call
                self._poll_ready = False
                self._poll_session = None
                return None

    # ------------------------------------------------------------------
    # Public API — WebSocket (background thread, push-based)
    # ------------------------------------------------------------------

    def start_stream(
        self, on_momo: Optional[Callable[[list[MomoStock]], None]] = None,
        user_initiated: bool = False,
    ) -> None:
        """
        Start a background thread that connects to Momoscreener's
        Socket.IO WebSocket and listens for ``5m3c`` momentum events.

        Each event updates ``self.latest_momos`` and optionally calls
        *on_momo(stocks)* with the new list.
        """
        if self._ws_thread and self._ws_thread.is_alive():
            logger.warning("Momo stream already running")
            return

        if user_initiated:
            self._user_initiated = True
        self._on_momo = on_momo
        self._ws_stop.clear()
        self._ever_received = False
        self._ws_thread = threading.Thread(
            target=self._ws_run, daemon=True, name="momo-ws",
        )
        self._ws_thread.start()
        logger.debug("Momo WebSocket stream started")

    def stop_stream(self) -> None:
        """Signal the background WebSocket thread to stop."""
        self._ws_stop.set()
        if self._ws_thread:
            self._ws_thread.join(timeout=5)
        logger.debug("Momo WebSocket stream stopped")

    @property
    def is_streaming(self) -> bool:
        return self._ws_thread is not None and self._ws_thread.is_alive()

    # ------------------------------------------------------------------
    # WebSocket internals
    # ------------------------------------------------------------------

    def _ws_run(self) -> None:
        """Entry point for the background thread — runs the async loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_loop())
        except Exception as exc:
            logger.warning(f"Momo WS thread exited: {exc}")
        finally:
            loop.close()

    async def _ws_loop(self) -> None:
        """Connect, upgrade, and listen. Auto-reconnects on failure."""
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        while not self._ws_stop.is_set():
            try:
                sid = self._get_sid()
                if not sid:
                    await asyncio.sleep(5)
                    continue

                uri = _SIO_WS.format(sid=sid)
                async with websockets.connect(
                    uri, ssl=ssl_ctx,
                    additional_headers={
                        "User-Agent": _HEADERS["User-Agent"],
                        "Origin": "https://momoscreener.com",
                    },
                ) as ws:
                    # Socket.IO v4 upgrade handshake
                    await ws.send("2probe")
                    await ws.recv()  # "3probe"
                    await ws.send("5")
                    logger.debug("Momo WS connected and upgraded")

                    while not self._ws_stop.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            continue

                        if msg == "2":
                            await ws.send("3")  # pong
                            continue

                        if not msg.startswith("42"):
                            continue

                        try:
                            data = orjson.loads(msg[2:])
                        except (orjson.JSONDecodeError, json.JSONDecodeError):
                            continue

                        event = data[0] if isinstance(data, list) else ""
                        payload = data[1] if isinstance(data, list) and len(data) > 1 else {}

                        if event == "5m3c":
                            items = payload.get("message", [])
                            stocks = [self._parse_stock(r) for r in items if isinstance(r, dict)]
                            if stocks:
                                self._ever_received = True
                                self.latest_momos = stocks
                                if self._on_momo:
                                    self._on_momo(stocks)

            except Exception as exc:
                if not self._ws_stop.is_set():
                    # Code 1005 = no status received (server closed without a
                    # proper close frame). If we've never received any data this
                    # session, treat it as an unrecoverable failure and stop
                    # rather than spinning in reconnect loops.
                    is_1005 = (
                        hasattr(exc, "rcvd") and exc.rcvd is None
                        or (
                            hasattr(exc, "rcvd") and exc.rcvd is not None
                            and getattr(exc.rcvd, "code", None) == 1005
                        )
                    )
                    if is_1005 and not self._ever_received and not self._user_initiated:
                        logger.warning(
                            "\nMomo WS closed with code 1005 and no data received — "
                            "disabling for this session (use 'momo start' to retry)\n"
                        )
                        self._ws_stop.set()
                        return
                    logger.warning(f"Momo WS error: {exc} — reconnecting in 5s")
                    await asyncio.sleep(5)

    def _get_sid(self) -> Optional[str]:
        """Perform the Socket.IO polling handshake to get a session ID."""
        try:
            resp = self._session.get(_SIO_POLL, timeout=HTTP_TIMEOUT)
            if not resp.ok:
                logger.warning(f"Momo SIO handshake failed: {resp.status_code}")
                return None
            text = resp.text
            idx = text.find("{")
            if idx < 0:
                return None
            return orjson.loads(text[idx:]).get("sid")
        except Exception as exc:
            logger.warning(f"Momo SIO handshake error: {exc}")
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(val: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of an ISO timestamp string to tz-aware UTC."""
    if not val:
        return None
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None
