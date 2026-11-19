"""
market_data/tastytrade_stream.py

tastytrade's tape: dxFeed TimeAndSale (and Trade) events over the DXLink
WebSocket, for market_data/tape_merge.py. A port of the tastytrade side of
ScalpFeedVendor (``~/.scalpfeed/project``: Feed/TastytradeApi.cs,
Feed/DxLinkLink.cs, Feed/FeedFrameParser.cs), with the same credentials file,
handshake, fields and timings.

Credentials: the JSON file the vendor already uses (config
``TASTYTRADE_CREDENTIALS_PATH``, default ``~/.scalpfeed/tastytrade.json``)
with ``client_id``, ``client_secret`` and ``refresh_token``. OAuth access
tokens last 15 minutes; API quote tokens 24 hours (renewed after 23 and
re-sent on the open socket). When tastytrade rotates the refresh token, the
new one is written back to the file, and the file is re-read before every
refresh, so this process and the Quantower vendor can share it.

A token whose feed level isn't ``api`` (real time) is refused: tastytrade's
delayed feed is ~15 minutes behind, and prices that old must not reach a
trading tape. The link retries every 10 minutes in case the account changes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import requests

from market_data.tape_merge import TastytradePrint, TastytradeTrade

logger = logging.getLogger(__name__)

BASE_URL = "https://api.tastyworks.com"
USER_AGENT = "momo-algo/1.0"
DXLINK_VERSION = "0.1-momo-algo/1.0.0"
FEED_CHANNEL = 1
KEEPALIVE_SECS = 30.0
SILENCE_LIMIT_SECS = 90.0
CONNECT_TIMEOUT_SECS = 10.0
STABLE_AFTER_SECS = 60.0
BACKOFF_SECS = (0.25, 0.5, 1, 2, 4, 8, 15, 30)
QUOTE_TOKEN_MAX_AGE_SECS = 23 * 3600.0
# Too many failed logins blocks the IP for about 8 hours, so a refused refresh
# token isn't tried again for a while (a new token in the file is tried at once).
REFUSED_PAUSE_SECS = 300.0
DELAYED_RETRY_SECS = 600.0
# The vendor measured 0 against 0.1 side by side: identical prints, 0.1 only
# later and batched at ~100 ms.
AGGREGATION_PERIOD = 0.0

DEFAULT_CREDENTIALS_PATH = Path.home() / ".scalpfeed" / "tastytrade.json"

FEED_FIELDS: dict[str, list[str]] = {
    "TimeAndSale": [
        "eventType", "eventSymbol", "eventFlags", "index", "time", "exchangeCode",
        "price", "size", "bidPrice", "askPrice", "aggressorSide",
        "extendedTradingHours", "validTick", "type", "exchangeSaleConditions",
    ],
    "Trade": [
        "eventType", "eventSymbol", "time", "price", "size", "dayVolume",
        "dayTurnover", "extendedTradingHours",
    ],
}


class TastytradeConfigError(Exception):
    """The credentials file is missing or incomplete."""


class TastytradeApiError(Exception):
    """A REST call failed."""


@dataclass(frozen=True)
class QuoteToken:
    token: str
    url: str
    level: str
    fetched: float          # time.time()

    @property
    def is_real_time(self) -> bool:
        return self.level == "api"


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

_refused_token: Optional[str] = None
_refused_until = 0.0


class TastytradeApi:
    """OAuth refresh and API quote tokens. Never logs a secret."""

    def __init__(self, path: Path | str = DEFAULT_CREDENTIALS_PATH,
                 session: Optional[requests.Session] = None) -> None:
        self.path = Path(path)
        self._config = self._read_config()
        self._http = session or requests.Session()
        self._http.headers["User-Agent"] = USER_AGENT
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._access_expiry = 0.0

    def _read_config(self) -> dict[str, Any]:
        if not self.path.exists():
            raise TastytradeConfigError(f"credentials file not found: {self.path}")
        try:
            config = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise TastytradeConfigError(f"credentials file isn't valid JSON: {self.path}") from exc
        if not isinstance(config, dict):
            raise TastytradeConfigError(f"credentials file isn't a JSON object: {self.path}")
        for key in ("client_id", "client_secret", "refresh_token"):
            if not config.get(key):
                raise TastytradeConfigError(f"credentials file has no {key}: {self.path}")
        return config

    def _write_config(self) -> None:
        """Atomic: another process (the Quantower vendor) may read it at any time."""
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".tastytrade-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._config, handle, indent=2)
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def access_token(self) -> str:
        global _refused_token, _refused_until
        with self._lock:
            if self._access_token and time.time() < self._access_expiry:
                return self._access_token
            # Pick up a refresh token another process rotated since we last looked.
            try:
                self._config = self._read_config()
            except TastytradeConfigError:
                pass
            refresh = self._config["refresh_token"]
            if refresh == _refused_token and time.time() < _refused_until:
                raise TastytradeApiError(
                    "tastytrade refused these credentials; not trying again until "
                    f"{time.strftime('%H:%M:%S', time.localtime(_refused_until))} "
                    f"(update {self.path} to try sooner)"
                )
            body = {
                "grant_type": "refresh_token",
                "refresh_token": refresh,
                "client_id": self._config["client_id"],
                "client_secret": self._config["client_secret"],
            }
            try:
                response = self._http.post(f"{BASE_URL}/oauth/token", json=body, timeout=15)
            except requests.RequestException as exc:
                raise TastytradeApiError(f"OAuth token request failed: {type(exc).__name__}") from exc
            if not response.ok:
                # Server trouble says nothing about the credentials, so those keep retrying.
                if 400 <= response.status_code < 500:
                    _refused_token = refresh
                    _refused_until = time.time() + REFUSED_PAUSE_SECS
                raise TastytradeApiError(f"OAuth token request returned {response.status_code}")
            _refused_token = None
            data = response.json()
            self._access_token = data.get("access_token")
            if not self._access_token:
                raise TastytradeApiError("OAuth token response had no access_token")
            lifetime = int(data.get("expires_in") or 900)
            self._access_expiry = time.time() + max(30, lifetime - 60)
            rotated = data.get("refresh_token")
            if rotated and rotated != refresh:
                self._config["refresh_token"] = rotated
                try:
                    self._write_config()
                    logger.info("[tasty] tastytrade rotated the refresh token; saved the new one")
                except OSError as exc:
                    logger.error("[tasty] tastytrade rotated the refresh token but saving it failed: %s", exc)
            return self._access_token

    def _get(self, path: str) -> dict:
        for attempt in range(2):
            token = self.access_token()
            try:
                response = self._http.get(
                    BASE_URL + path, headers={"Authorization": f"Bearer {token}"}, timeout=15,
                )
            except requests.RequestException as exc:
                raise TastytradeApiError(f"GET {path} failed: {type(exc).__name__}") from exc
            if response.status_code == 401 and attempt == 0:
                with self._lock:
                    self._access_token = None
                continue
            if not response.ok:
                raise TastytradeApiError(f"GET {path} returned {response.status_code}")
            return response.json()
        raise TastytradeApiError(f"GET {path} unauthorized")

    def quote_token(self) -> QuoteToken:
        data = (self._get("/api-quote-tokens") or {}).get("data") or {}
        token, url = data.get("token"), data.get("dxlink-url")
        if not token or not url:
            raise TastytradeApiError("quote token response had no token or dxlink-url")
        return QuoteToken(token=token, url=url, level=str(data.get("level") or "unknown"),
                          fetched=time.time())


# ---------------------------------------------------------------------------
# COMPACT frames
# ---------------------------------------------------------------------------

def _num(value: Any) -> float:
    if isinstance(value, bool):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan   # "NaN", "Infinity", null


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _rows(values: Any, width: int) -> Iterable[list]:
    """Rows come flat (one long array) or nested (an array per event)."""
    if not isinstance(values, list) or width <= 0:
        return []
    if values and isinstance(values[0], list):
        return [row for row in values if isinstance(row, list) and len(row) == width]
    return [values[i:i + width] for i in range(0, len(values) - width + 1, width)]


def parse_feed_data(
    data: Any,
    layouts: dict[str, list[str]],
    on_print: Callable[[TastytradePrint], None],
    on_trade: Callable[[TastytradeTrade], None],
    arrival: float,
    arrival_wall: float,
) -> None:
    """Decode a COMPACT FEED_DATA ``data`` array: ["TimeAndSale", [...], "Trade", [...]]."""
    if not isinstance(data, list):
        return
    i = 0
    while i + 1 < len(data):
        name, values = data[i], data[i + 1]
        i += 2
        columns = layouts.get(name) if isinstance(name, str) else None
        if not columns:
            continue
        for raw in _rows(values, len(columns)):
            row = dict(zip(columns, raw))
            symbol = row.get("eventSymbol")
            if not isinstance(symbol, str) or not symbol:
                continue
            if name == "TimeAndSale":
                exchange = row.get("exchangeCode")
                side = row.get("aggressorSide")
                sale = row.get("type")
                conditions = row.get("exchangeSaleConditions")
                on_print(TastytradePrint(
                    symbol=symbol,
                    time_ms=_int(row.get("time")),
                    price=_num(row.get("price")),
                    size=_num(row.get("size")),
                    bid=_num(row.get("bidPrice")),
                    ask=_num(row.get("askPrice")),
                    exchange=exchange[:1] if isinstance(exchange, str) else "",
                    aggressor=side if side in ("BUY", "SELL") else "UNDEFINED",
                    sale=sale if sale in ("CANCEL", "CORRECTION") else "NEW",
                    flags=_int(row.get("eventFlags")),
                    index=_int(row.get("index")),
                    valid_tick=row.get("validTick") is not False,
                    extended_hours=row.get("extendedTradingHours") is True,
                    conditions=conditions if isinstance(conditions, str) else "",
                    arrival=arrival,
                    arrival_wall=arrival_wall,
                ))
            elif name == "Trade":
                on_trade(TastytradeTrade(
                    symbol=symbol,
                    time_ms=_int(row.get("time")),
                    price=_num(row.get("price")),
                    size=_num(row.get("size")),
                    day_volume=_num(row.get("dayVolume")),
                    arrival=arrival,
                ))


# ---------------------------------------------------------------------------
# DXLink link
# ---------------------------------------------------------------------------

class _Reconnect(Exception):
    """End this socket and start a fresh one."""


class _Delayed(Exception):
    """The token's feed level isn't real time."""


def _default_connect(url: str):
    from websockets.sync.client import connect
    return connect(url, open_timeout=CONNECT_TIMEOUT_SECS, ping_interval=None,
                   close_timeout=2, max_size=None)


class DxLinkTapeLink:
    """One DXLink socket carrying TimeAndSale and Trade for every symbol added.

    Runs on its own daemon thread; callbacks (prints, trades, live changes)
    fire there. ``add``/``remove`` may be called from any thread.
    """

    def __init__(
        self,
        api: TastytradeApi,
        on_print: Callable[[TastytradePrint], None],
        on_trade: Callable[[TastytradeTrade], None],
        on_live_change: Callable[[bool], None],
        *,
        connect: Callable[[str], Any] = _default_connect,
    ) -> None:
        self._api = api
        self._on_print = on_print
        self._on_trade = on_trade
        self._on_live_change = on_live_change
        self._connect = connect
        self._lock = threading.Lock()
        self._symbols: set[str] = set()
        self._ops: list[tuple[str, str]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._token: Optional[QuoteToken] = None
        self._ws = None
        self._layouts: dict[str, list[str]] = {k: list(v) for k, v in FEED_FIELDS.items()}
        self._ready = False
        self._live = False
        self._delayed_logged = False

    # -- public ---------------------------------------------------------------

    @property
    def live(self) -> bool:
        return self._live

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="tastytrade-tape")
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=timeout)

    def add(self, symbol: str) -> None:
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._symbols:
                return
            self._symbols.add(symbol)
            self._ops.append(("add", symbol))

    def remove(self, symbol: str) -> None:
        symbol = symbol.upper()
        with self._lock:
            if symbol not in self._symbols:
                return
            self._symbols.discard(symbol)
            self._ops.append(("remove", symbol))

    # -- thread -----------------------------------------------------------------

    def _run(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._session()
                wait = 0.0
            except _Delayed:
                wait = DELAYED_RETRY_SECS
            except _Reconnect as exc:
                logger.info("[tasty] reconnecting: %s", exc)
                wait = None
            except TastytradeConfigError as exc:
                logger.error("[tasty] %s — tastytrade tape off", exc)
                return
            except Exception as exc:
                logger.warning("[tasty] link failed: %s: %s", type(exc).__name__, exc)
                wait = None
            finally:
                self._close_socket()
                self._set_live(False)
            if self._stop.is_set():
                return
            if time.monotonic() - started > STABLE_AFTER_SECS:
                attempt = 0
            if wait is None:
                wait = BACKOFF_SECS[min(attempt, len(BACKOFF_SECS) - 1)]
                attempt += 1
            self._stop.wait(wait)

    def _fresh_token(self) -> QuoteToken:
        token = self._token
        if token is None or time.time() - token.fetched > QUOTE_TOKEN_MAX_AGE_SECS:
            token = self._api.quote_token()
            self._token = token
        if not token.is_real_time:
            if not self._delayed_logged:
                self._delayed_logged = True
                logger.error("[tasty] DELAYED DATA: tastytrade returned feed level %r, about "
                             "15 minutes behind — not putting it on the tape", token.level)
            self._token = None
            raise _Delayed()
        return token

    def _session(self) -> None:
        token = self._fresh_token()
        self._ready = False
        self._ws = self._connect(token.url)
        self._send({"type": "SETUP", "channel": 0, "version": DXLINK_VERSION,
                    "keepaliveTimeout": 60, "acceptKeepaliveTimeout": 60})
        auth_sent = False
        last_frame = time.monotonic()
        last_keepalive = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_keepalive >= KEEPALIVE_SECS:
                if now - last_frame > SILENCE_LIMIT_SECS:
                    raise _Reconnect(f"no frames for {now - last_frame:.0f}s")
                self._send({"type": "KEEPALIVE", "channel": 0})
                last_keepalive = now
            if self._ready:
                self._flush_ops()
                if time.time() - token.fetched > QUOTE_TOKEN_MAX_AGE_SECS:
                    token = self._fresh_token()
                    self._send({"type": "AUTH", "channel": 0, "token": token.token})
                    logger.info("[tasty] quote token refreshed")
            try:
                text = self._ws.recv(timeout=0.5)
            except TimeoutError:
                continue
            arrival, arrival_wall = time.monotonic(), time.time()
            last_frame = arrival
            try:
                message = json.loads(text)
            except ValueError:
                continue
            kind = message.get("type")
            if kind == "FEED_DATA":
                if message.get("channel") == FEED_CHANNEL:
                    parse_feed_data(message.get("data"), self._layouts, self._on_print,
                                    self._on_trade, arrival, arrival_wall)
            elif kind == "AUTH_STATE":
                if message.get("state") == "UNAUTHORIZED":
                    if auth_sent:
                        # Refused or expired: reconnect with a freshly minted token.
                        self._token = None
                        raise _Reconnect("quote token was refused")
                    self._send({"type": "AUTH", "channel": 0, "token": token.token})
                    auth_sent = True
                elif message.get("state") == "AUTHORIZED" and not self._ready:
                    self._send({"type": "CHANNEL_REQUEST", "channel": FEED_CHANNEL,
                                "service": "FEED", "parameters": {"contract": "AUTO"}})
            elif kind == "CHANNEL_OPENED" and message.get("channel") == FEED_CHANNEL:
                self._send({"type": "FEED_SETUP", "channel": FEED_CHANNEL,
                            "acceptAggregationPeriod": AGGREGATION_PERIOD,
                            "acceptDataFormat": "COMPACT",
                            "acceptEventFields": FEED_FIELDS})
            elif kind == "FEED_CONFIG":
                fields = message.get("eventFields")
                if isinstance(fields, dict):
                    for name, columns in fields.items():
                        if isinstance(columns, list):
                            self._layouts[name] = [str(c) for c in columns]
                if not self._ready:
                    self._ready = True
                    self._subscribe_all()
                    logger.info("[tasty] feed ready: level=%s aggregation=%s",
                                token.level, message.get("aggregationPeriod"))
                    self._set_live(True)
            elif kind in ("ERROR", "CHANNEL_CLOSED"):
                logger.warning("[tasty] %s: %s", kind, str(message)[:300])

    # -- helpers ---------------------------------------------------------------------

    def _send(self, message: dict) -> None:
        ws = self._ws
        if ws is not None:
            ws.send(json.dumps(message))

    def _subscribe_all(self) -> None:
        with self._lock:
            symbols = sorted(self._symbols)
            self._ops.clear()
        self._send({"type": "FEED_SUBSCRIPTION", "channel": FEED_CHANNEL, "reset": True,
                    "add": [{"type": t, "symbol": s} for s in symbols for t in FEED_FIELDS]})

    def _flush_ops(self) -> None:
        with self._lock:
            ops, self._ops = self._ops, []
        adds = [s for op, s in ops if op == "add"]
        removes = [s for op, s in ops if op == "remove"]
        if adds:
            self._send({"type": "FEED_SUBSCRIPTION", "channel": FEED_CHANNEL,
                        "add": [{"type": t, "symbol": s} for s in adds for t in FEED_FIELDS]})
        if removes:
            self._send({"type": "FEED_SUBSCRIPTION", "channel": FEED_CHANNEL,
                        "remove": [{"type": t, "symbol": s} for s in removes for t in FEED_FIELDS]})

    def _close_socket(self) -> None:
        ws, self._ws = self._ws, None
        self._ready = False
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _set_live(self, live: bool) -> None:
        if live != self._live:
            self._live = live
            try:
                self._on_live_change(live)
            except Exception:
                logger.exception("[tasty] live-change callback failed")
