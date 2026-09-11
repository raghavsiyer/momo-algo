"""
market_data/tradier_rest.py

Tradier REST API — Stats + Historical OHLCV data.

  STATS
    - Snapshot quote: last price, % change, volume, avg volume
    - Multi-symbol batch quotes

  HISTORICAL CANDLES (all via REST, no streaming)
    - 1-minute bars      — get_candles_1min()
    - 10-second bars     — get_candles_10sec()   (resampled from tick data)
    - 15-minute bars     — get_candles_15min()
    - Daily bars         — get_candles_daily()

Usage:
    from market_data.tradier_rest import HistoricalClient
    client = HistoricalClient()

    # Stats
    q  = client.get_quote("ACMR")
    df = client.get_quote_df(["ACMR", "IONQ"])

    # Candles
    df1   = client.get_candles_1min("ACMR", n=30)
    df10  = client.get_candles_10sec("ACMR", n=60)
    df15  = client.get_candles_15min("ACMR", n=20)
    dfd   = client.get_candles_daily("ACMR", days_back=30)
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import pandas as pd
import requests

import config
from config import MarketClock

from config import TRADIER_ENABLED as _TRADIER_ENABLED
from market_data import resilient_session
from config import HTTP_TIMEOUT as _HTTP_TIMEOUT
from config import (
    CANDLES_1MIN_FETCH_DAYS, CANDLES_5MIN_FETCH_DAYS,
    CANDLES_10SEC_FETCH_HOURS, CANDLES_15MIN_FETCH_DAYS,
    MarketClock,
)

logger = logging.getLogger(__name__)


class TradierClientError(Exception):
    """Raised for HTTP 4xx responses — the request is invalid and won't succeed on retry."""
    pass


class MarketDataUnavailable(Exception):
    """Raised when a fetch fails for reasons unrelated to the symbol itself.

    Provider-neutral: connection refused, socket timeout, IBKR pacing stall,
    5xx.  Distinct from an *empty* result, which means the provider answered
    and the symbol has no data.  Callers must not read an unavailable fetch
    as "this symbol does not exist" — that misattributes an infrastructure
    fault to the ticker and silently discards a valid one.
    """
    pass


_INTERVAL_LITERAL = Literal["tick", "1min", "5min", "15min"]

# Tick-window chunking for get_candles_10sec — proactively split the request to
# stay under Tradier's response-body cap and parallelise the round-trips.
_TICK_CHUNK_MINUTES: int = 20      # per chunk; small enough to avoid 502 TooBigBody
_TICK_CHUNK_PARALLEL: int = 4      # concurrent in-flight chunk requests


def _ms_to_dt(ms) -> Optional[datetime]:
    """Convert a millisecond-epoch timestamp to tz-aware UTC datetime."""
    if ms is None or ms == 0:
        return None
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StockQuote:
    """
    Snapshot quote from GET /v1/markets/quotes.
    Stats layer: last price, % change, volume, avg volume, bid/ask.
    """
    symbol: str
    description: str
    last: Optional[float]           # last trade price
    change: Optional[float]         # absolute $ change from prev close
    change_pct: Optional[float]     # % change from prev close
    volume: int                     # today's cumulative volume
    avg_volume: int                 # 30-day average daily volume
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    prev_close: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    bid_size: int
    ask_size: int
    last_volume: int                # size of last trade
    trade_ts: Optional[datetime]    # timestamp of last trade (UTC)
    exchange: str


@dataclass
class OHLCVCandle:
    """
    A single OHLCV bar.
    Used for 10-sec, 1-min, 15-min (timesales endpoint) and daily (history endpoint).
    """
    symbol: str
    ts: datetime                    # bar open timestamp (tz-aware UTC)
    open: float
    high: float
    low: float
    close: float
    volume: int
    interval: str                   # "10sec" | "1min" | "15min" | "1day"
    vwap: Optional[float] = None    # present in timesales response only


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class HistoricalClient:
    """
    Tradier REST client for stats and historical OHLCV data.

    Reads TRADIER_API_KEY from environment. Always uses production api.tradier.com.
    """

    def __init__(self) -> None:
        self._disabled = not _TRADIER_ENABLED
        if self._disabled:
            self._token = ""
            self._base = ""
            self._headers = {}
            self._session = resilient_session()
            return
        token = os.getenv("TRADIER_API_KEY")
        if not token:
            raise EnvironmentError(
                "TRADIER_API_KEY not set. "
                "Add to .env: TRADIER_API_KEY=your_token"
            )
        self._token = token
        self._base = "https://api.tradier.com/v1"
        self._headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }
        self._session = resilient_session()
        self._session.headers.update(self._headers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(
        self, path: str, params: dict | None = None,
        *, raise_on_unavailable: bool = False,
    ) -> dict:
        """GET *path*.  Returns ``{}`` on transport failure by default.

        ``raise_on_unavailable=True`` raises :class:`MarketDataUnavailable`
        for 5xx/network errors instead.  Opt-in because most callers treat
        an empty dict as "no data" and degrade fine; the quote path cannot,
        since there "no data" is read as "symbol does not exist".
        """
        if self._disabled or config.TRADIER_PAUSED:
            return {}
        try:
            resp = self._session.get(
                f"{self._base}{path}",
                params=params,
                timeout=_HTTP_TIMEOUT,
            )
            if not resp.ok:
                logger.warning(
                    f"Tradier {resp.status_code} on {path}: {resp.text[:200]}"
                )
                if 400 <= resp.status_code < 500:
                    raise TradierClientError(
                        f"{resp.status_code} on {path}: {resp.text[:200]}"
                    )
                if raise_on_unavailable:
                    raise MarketDataUnavailable(
                        f"Tradier {resp.status_code} on {path}"
                    )
                return {}
            return resp.json()
        except TradierClientError:
            raise
        except MarketDataUnavailable:
            raise
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Tradier request failed on {path}: {exc}")
            if raise_on_unavailable:
                raise MarketDataUnavailable(
                    f"Tradier request failed on {path}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            return {}

    @staticmethod
    def _f(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
        # Tradier sometimes returns numeric strings with a trailing '%'
        if isinstance(val, str):
            cleaned = val.strip().rstrip("%")
            try:
                return float(cleaned)
            except (TypeError, ValueError):
                pass
        logger.debug("Tradier _f: could not parse %r", val)
        return None

    @staticmethod
    def _i(val) -> int:
        try:
            return int(val) if val is not None else 0
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # STATS — Snapshot quotes
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> StockQuote:
        """Fetch a snapshot quote for a single symbol."""
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

    def search_symbol(self, query: str) -> Optional[str]:
        """Search for a ticker by partial symbol or company name.

        Returns the first matching stock symbol, or None if no match or
        Tradier is disabled.  Results are sorted by average volume by the
        API, so the first hit is the most liquid match.
        """
        try:
            data = self._get("/markets/search", params={"q": query, "indexes": "false"})
        except TradierClientError:
            return None
        securities = data.get("securities") or {}
        items = securities.get("security") or []
        if isinstance(items, dict):
            items = [items]
        for item in items:
            if item.get("type") == "stock":
                return item.get("symbol")
        return None

    def get_quotes(self, symbols: list[str]) -> list[StockQuote]:
        """
        Fetch snapshot quotes for one or more symbols.
        Returns a list of StockQuote dataclasses.

        Raises :class:`MarketDataUnavailable` if the request failed (5xx,
        network error) rather than returning an empty list, which would be
        indistinguishable from "no such symbol".
        """
        try:
            data = self._get(
                "/markets/quotes",
                params={"symbols": ",".join(symbols), "greeks": "false"},
                raise_on_unavailable=True,
            )
        except TradierClientError:
            # 4xx — the symbol itself was rejected. Genuinely not found.
            return []
        except MarketDataUnavailable:
            raise
        raw = data.get("quotes", {}).get("quote", [])
        if isinstance(raw, dict):
            raw = [raw]  # single symbol returns dict, not list

        result = []
        for q in raw:
            raw_chg = q.get("change_percentage")
            parsed_chg = self._f(raw_chg)
            if parsed_chg is None and raw_chg is not None:
                logger.warning(
                    "Tradier change_percentage unparseable for %s: %r",
                    q.get("symbol"), raw_chg,
                )
            result.append(StockQuote(
                symbol=q.get("symbol", ""),
                description=q.get("description", ""),
                last=self._f(q.get("last")),
                change=self._f(q.get("change")),
                change_pct=parsed_chg,
                volume=self._i(q.get("volume")),
                avg_volume=self._i(q.get("average_volume")),
                open=self._f(q.get("open")),
                high=self._f(q.get("high")),
                low=self._f(q.get("low")),
                prev_close=self._f(q.get("prevclose")),
                bid=self._f(q.get("bid")),
                ask=self._f(q.get("ask")),
                bid_size=self._i(q.get("bidsize")),
                ask_size=self._i(q.get("asksize")),
                last_volume=self._i(q.get("last_volume")),
                trade_ts=_ms_to_dt(q.get("trade_date")),
                exchange=q.get("exch", ""),
            ))
        return result

    def get_quote_df(self, symbols: list[str]) -> pd.DataFrame:
        """Multi-symbol snapshot as a DataFrame."""
        return pd.DataFrame([asdict(q) for q in self.get_quotes(symbols)])

    # ------------------------------------------------------------------
    # HISTORICAL — Intraday candles via timesales
    # ------------------------------------------------------------------

    def _get_timesales(
        self,
        symbol: str,
        interval: _INTERVAL_LITERAL,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVCandle]:
        """
        Pull from GET /v1/markets/timesales and return OHLCVCandle list.
        interval: "1min" | "5min" | "15min"  (not "tick" — use get_candles_10sec)
        """
        fmt = "%Y-%m-%d %H:%M"
        _et = MarketClock.ET
        data = self._get(
            "/markets/timesales",
            params={
                "symbol": symbol,
                "interval": interval,
                "start": start.astimezone(_et).strftime(fmt),
                "end": end.astimezone(_et).strftime(fmt),
                "session_filter": "all",
            },
        )
        series = data.get("series", {})
        if not series:
            return []
        raw = series.get("data", [])
        if isinstance(raw, dict):
            raw = [raw]

        candles = []
        for item in raw:
            try:
                raw_ts = datetime.fromisoformat(item["time"])
                if raw_ts.tzinfo is None:
                    ts = raw_ts.replace(tzinfo=_et).astimezone(timezone.utc)
                else:
                    ts = raw_ts.astimezone(timezone.utc)
            except (KeyError, ValueError):
                continue
            candles.append(OHLCVCandle(
                symbol=symbol,
                ts=ts,
                open=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close", 0)),
                volume=int(item.get("volume", 0)),
                interval=interval,
                vwap=self._f(item.get("vwap")),
            ))
        return candles

    def _to_df(self, candles: list[OHLCVCandle]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame([asdict(c) for c in candles])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df.sort_values("ts", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_candles_1min(self, symbol: str, n: int = 30) -> pd.DataFrame:
        """
        Last *n* completed 1-minute candles.
        Looks back 5 trading days to guarantee enough bars intraday.
        """
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=CANDLES_1MIN_FETCH_DAYS)
        candles = self._get_timesales(symbol, "1min", start, end)
        candles = candles[-n:] if len(candles) > n else candles
        return self._to_df(candles)

    def get_candles_1min_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """1-minute candles for a precise UTC-aware time window."""
        end = end or datetime.now(tz=timezone.utc)
        candles = self._get_timesales(symbol, "1min", start, end)
        return self._to_df(candles)

    def get_candles_1min_recent(
        self, symbol: str, n: int = 5, lookback_minutes: int = 8,
    ) -> pd.DataFrame:
        """Recent 1-minute candles for live refresh loops."""
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(minutes=max(lookback_minutes, n + 1))
        candles = self._get_timesales(symbol, "1min", start, end)
        candles = candles[-n:] if len(candles) > n else candles
        return self._to_df(candles)

    def _fetch_tick_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """Fetch raw tick prints for [start, end). On 502 body-overflow, bisect once."""
        _et = MarketClock.ET
        fmt = "%Y-%m-%d %H:%M"
        params = {
            "symbol": symbol,
            "interval": "tick",
            "start": start.astimezone(_et).strftime(fmt),
            "end": end.astimezone(_et).strftime(fmt),
            "session_filter": "all",
        }
        try:
            data = self._get("/markets/timesales", params=params)
        except TradierClientError:
            return []
        if data:
            series = data.get("series") or {}
            raw = series.get("data", []) if series else []
            if isinstance(raw, dict):
                raw = [raw]
            return raw
        # Empty payload (likely 502 body overflow) — bisect once.
        span = end - start
        if span <= timedelta(minutes=5):
            return []
        mid = start + span / 2
        return self._fetch_tick_window(symbol, start, mid) + \
               self._fetch_tick_window(symbol, mid, end)

    def _ticks_to_10sec_df(
        self,
        symbol: str,
        raw: list[dict],
        n: int | None = None,
    ) -> pd.DataFrame:
        rows = []
        for item in raw:
            try:
                raw_ts = datetime.fromisoformat(item["time"])
                if raw_ts.tzinfo is None:
                    ts = raw_ts.replace(tzinfo=timezone.utc)
                else:
                    ts = raw_ts.astimezone(timezone.utc)
                size = item.get("size")
                if size is None:
                    size = item.get("volume", 0)
                rows.append({
                    "ts": ts,
                    "price": float(item.get("price", 0)),
                    "volume": int(size or 0),
                })
            except (KeyError, ValueError):
                continue

        if not rows:
            return pd.DataFrame()

        ticks = pd.DataFrame(rows).set_index("ts")
        ticks.index = pd.DatetimeIndex(ticks.index, tz="UTC")
        ohlcv = ticks["price"].resample("10s").ohlc()
        ohlcv["volume"] = ticks["volume"].resample("10s").sum()
        ohlcv = ohlcv.dropna()
        if n is not None:
            ohlcv = ohlcv.tail(n)
        ohlcv = ohlcv.reset_index()
        ohlcv.columns = ["ts", "open", "high", "low", "close", "volume"]
        ohlcv["symbol"] = symbol
        ohlcv["interval"] = "10sec"
        ohlcv["vwap"] = None
        return ohlcv[["symbol", "ts", "open", "high", "low", "close", "volume", "interval", "vwap"]]

    def get_candles_10sec_window(
        self,
        symbol: str,
        start: datetime,
        end: datetime | None = None,
        n: int | None = None,
    ) -> pd.DataFrame:
        """10-second OHLCV bars for a precise tick window."""
        end = end or datetime.now(tz=timezone.utc)

        chunk_span = timedelta(minutes=_TICK_CHUNK_MINUTES)
        windows: list[tuple[datetime, datetime]] = []
        cursor = start
        while cursor < end:
            windows.append((cursor, min(cursor + chunk_span, end)))
            cursor += chunk_span

        raw: list[dict] = []
        with ThreadPoolExecutor(max_workers=_TICK_CHUNK_PARALLEL) as ex:
            for chunk in ex.map(lambda w: self._fetch_tick_window(symbol, w[0], w[1]), windows):
                raw.extend(chunk)

        return self._ticks_to_10sec_df(symbol, raw, n=n)

    def get_candles_10sec(self, symbol: str, n: int = 60) -> pd.DataFrame:
        """
        Last *n* 10-second OHLCV bars.

        Tradier REST doesn't have a native 10-sec interval — this pulls
        raw tick data and resamples into 10-second buckets using pandas.
        Only available on production tokens (not sandbox).

        Always chunks the window up-front (parallel fetch) instead of
        trying one big request and falling back on 502.  Each chunk that
        still 502s is bisected once.
        """
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(hours=CANDLES_10SEC_FETCH_HOURS)
        return self.get_candles_10sec_window(symbol, start, end, n=n)

    def get_timesales_tick(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        session_filter: str = "all",
    ) -> list[dict]:
        """
        Pull raw tick prints from GET /v1/markets/timesales.

        Returns a list of dict rows with normalized keys:
          ts, price, size, bid, ask, exchange, flag, cancel, correction, session

        Notes:
        - Tick payloads can be very large. Keep windows small (e.g., 1-5 minutes).
        - Tradier timestamp strings are parsed consistently with existing methods
          (naive parsed and tagged UTC).
        """
        _et = MarketClock.ET
        data = self._get(
            "/markets/timesales",
            params={
                "symbol": symbol,
                "interval": "tick",
                "start": start.astimezone(_et).strftime("%Y-%m-%d %H:%M"),
                "end": end.astimezone(_et).strftime("%Y-%m-%d %H:%M"),
                "session_filter": session_filter,
            },
        )

        series = data.get("series", {})
        if not series:
            return []
        raw = series.get("data", [])
        if isinstance(raw, dict):
            raw = [raw]

        rows: list[dict] = []
        for item in raw:
            try:
                ts = datetime.fromisoformat(item["time"]).replace(tzinfo=timezone.utc)
            except (KeyError, ValueError):
                continue
            size = item.get("size")
            if size is None:
                size = item.get("volume", 0)
            rows.append(
                {
                    "ts": ts,
                    "price": float(item.get("price", 0.0) or 0.0),
                    "size": int(size or 0),
                    "bid": self._f(item.get("bid")),
                    "ask": self._f(item.get("ask")),
                    "exchange": str(item.get("exchange", "") or ""),
                    "flag": str(item.get("flag", "") or ""),
                    "cancel": bool(item.get("cancel", False)),
                    "correction": bool(item.get("correction", False)),
                    "session": str(item.get("session", "") or ""),
                }
            )

        rows.sort(key=lambda r: r["ts"])
        return rows

    def get_candles_15min(self, symbol: str, n: int = 20) -> pd.DataFrame:
        """Last *n* 15-minute candles."""
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=CANDLES_15MIN_FETCH_DAYS)
        candles = self._get_timesales(symbol, "15min", start, end)
        candles = candles[-n:] if len(candles) > n else candles
        return self._to_df(candles)

    def get_candles_5min(self, symbol: str, days_back: int = CANDLES_5MIN_FETCH_DAYS) -> pd.DataFrame:
        """
        5-minute OHLCV candles via GET /v1/markets/timesales.
        Goes back *days_back* calendar days.
        """
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=days_back)
        candles = self._get_timesales(symbol, "5min", start, end)
        return self._to_df(candles)

    def get_candles_daily(
        self,
        symbol: str,
        days_back: int = 30,
        include_current_day: bool = True,
    ) -> pd.DataFrame:
        """
        Daily OHLCV candles via GET /v1/markets/history.
        Goes back *days_back* calendar days.
        """
        end = MarketClock.now_et().date()
        if not include_current_day:
            end = end - timedelta(days=1)
        start = end - timedelta(days=days_back)
        try:
            data = self._get(
                "/markets/history",
                params={
                    "symbol": symbol,
                    "interval": "daily",
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                },
            )
        except TradierClientError:
            return pd.DataFrame()
        history = data.get("history", {})
        if not history:
            return pd.DataFrame()
        raw = history.get("day", [])
        if isinstance(raw, dict):
            raw = [raw]

        candles = []
        for item in raw:
            candles.append(OHLCVCandle(
                symbol=symbol,
                ts=datetime.strptime(item["date"], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                ),
                open=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close", 0)),
                volume=int(item.get("volume", 0)),
                interval="1day",
            ))
        return self._to_df(candles)

    def get_closed_dates(self, months_back: int = 2) -> frozenset:
        """Fetch closed market dates from Tradier calendar API.

        Queries the current month and *months_back* prior months.
        Returns a frozenset of date objects for days the market is closed
        (weekends are not included — only weekday closures like holidays).
        """
        from datetime import date
        closed: set = set()
        today = datetime.now(tz=timezone.utc).date()
        for offset in range(months_back + 1):
            # Walk back month by month
            y = today.year
            m = today.month - offset
            while m < 1:
                m += 12
                y -= 1
            try:
                data = self._get(
                    "/markets/calendar",
                    params={"month": m, "year": y},
                )
            except TradierClientError:
                continue
            cal = data.get("calendar", {})
            days = cal.get("days", {})
            raw = days.get("day", [])
            if isinstance(raw, dict):
                raw = [raw]
            for item in raw:
                if item.get("status") == "closed":
                    try:
                        d = date.fromisoformat(item["date"])
                        if d.weekday() < 5:  # only track weekday closures
                            closed.add(d)
                    except (KeyError, ValueError):
                        continue
        return frozenset(closed)
