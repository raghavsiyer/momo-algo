"""
market_data/alpha_vantage_client.py

Alpha Vantage API integration — alternate data provider replacing FMP + Finnhub.

Endpoints used:
    OVERVIEW:        Company fundamentals → float, profile (sector, industry, IPO date)
    SPLITS:          Split history
    NEWS_SENTIMENT:  News with per-ticker sentiment scores

Supports both direct Alpha Vantage API and RapidAPI proxy.

Usage:
    from market_data.alpha_vantage_client import AlphaVantageClient
    client = AlphaVantageClient()

    float_data = client.get_float("ACMR")
    profile    = client.get_profile("ACMR")
    articles   = client.get_recent_news("ACMR", minutes_back=60)
    splits     = client.get_splits("ACMR")
"""

import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from config import HTTP_TIMEOUT
from market_data import resilient_session

logger = logging.getLogger(__name__)

# Re-use existing dataclasses for seamless drop-in
from market_data.fmp_client import FloatData, CompanyProfile, SplitEvent
from market_data.finnhub_client import FinnhubNewsArticle

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

_DIRECT_BASE = "https://www.alphavantage.co/query"
_RAPID_BASE = "https://alpha-vantage.p.rapidapi.com/query"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AlphaVantageClient:
    """
    Alpha Vantage REST client for company overview (float + profile),
    splits, and news with sentiment.

    Rate-limited to the free tier's 5 calls/min.
    """

    def __init__(self) -> None:
        from config import AV_API_KEY, AV_CALLS_PER_MIN
        self._key: str = AV_API_KEY
        if not self._key:
            raise EnvironmentError(
                "AV_API_KEY not set. "
                "Add to .env: AV_API_KEY=your_key"
            )
        self._rapidapi = os.getenv("ALPHAVANTAGE_RAPIDAPI", "false").lower() == "true"
        self._session = resilient_session()
        self._overview_cache: dict[str, dict] = {}
        self._call_log: deque[float] = deque()
        self._lock = threading.Lock()
        self._RATE_LIMIT = AV_CALLS_PER_MIN
        self._RATE_WINDOW = 60  # seconds

    def _redact_api_message(self, message: str) -> str:
        return message.replace(self._key, "<redacted>") if self._key else message

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    _MAX_RETRIES = 3
    _RETRY_BACKOFF = [1.5, 3.0, 6.0]   # seconds to wait after each 429

    @property
    def budget(self) -> int:
        """Number of AV API calls available in the current 60-second window."""
        with self._lock:
            now = time.monotonic()
            while self._call_log and self._call_log[0] < now - self._RATE_WINDOW:
                self._call_log.popleft()
            return max(0, self._RATE_LIMIT - len(self._call_log))

    def overview_cached(self, symbol: str) -> bool:
        """True if OVERVIEW for *symbol* is already cached (costs 0 calls)."""
        with self._lock:
            return symbol.upper() in self._overview_cache

    def _get(self, params: dict) -> dict | list:
        """Execute a GET request against AV (direct or RapidAPI) with 429 retry."""
        with self._lock:
            now = time.monotonic()
            while self._call_log and self._call_log[0] < now - self._RATE_WINDOW:
                self._call_log.popleft()
            if len(self._call_log) >= self._RATE_LIMIT:
                logger.debug("AV budget exhausted — skipping %s", params.get("function", "request"))
                return {}
            key = self._key
            self._call_log.append(now)

        if self._rapidapi:
            url = _RAPID_BASE
            headers = {
                "x-rapidapi-key": key,
                "x-rapidapi-host": "alpha-vantage.p.rapidapi.com",
            }
            req_params = {k: v for k, v in params.items() if k != "apikey"}
        else:
            url = _DIRECT_BASE
            headers = {}
            req_params = {**params, "apikey": key}

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, params=req_params, headers=headers, timeout=HTTP_TIMEOUT)
                if resp.status_code == 429:
                    if attempt < self._MAX_RETRIES:
                        wait = self._RETRY_BACKOFF[attempt]
                        logger.debug(f"AV 429 — retrying in {wait}s (attempt {attempt + 1}/{self._MAX_RETRIES})")
                        time.sleep(wait)
                        continue
                    logger.warning("AV 429 — max retries exhausted")
                    raise RuntimeError("AV 429 rate limit exhausted")
                if not resp.ok:
                    logger.warning(f"AV {resp.status_code}: {resp.text[:200]}")
                    return {}
                data = resp.json()
                if isinstance(data, dict) and (
                    "Error Message" in data or "Note" in data or "Information" in data
                ):
                    msg = data.get("Error Message") or data.get("Note") or data.get("Information", "")
                    msg = self._redact_api_message(str(msg))
                    logger.warning(f"AV API issue: {msg[:200]}")
                    return {}
                return data
            except requests.exceptions.RequestException as exc:
                logger.warning(f"AV request failed: {exc}")
                return {}
        return {}

    def _get_overview(self, symbol: str) -> dict:
        """Fetch and cache the OVERVIEW for a symbol (used by both float and profile)."""
        sym = symbol.upper()
        with self._lock:
            if sym in self._overview_cache:
                return self._overview_cache[sym]

        data = self._get({"function": "OVERVIEW", "symbol": sym})
        if not isinstance(data, dict) or not data.get("Symbol"):
            return {}

        with self._lock:
            cached = self._overview_cache.setdefault(sym, data)
            return cached

    # ------------------------------------------------------------------
    # Float  (drop-in replacement for FMPClient.get_float)
    # ------------------------------------------------------------------

    def get_float(self, symbol: str) -> Optional[FloatData]:
        """
        Fetch float data from the OVERVIEW endpoint.

        AV fields used:
            SharesFloat         → float_shares
            SharesOutstanding   → outstanding_shares
        """
        data = self._get_overview(symbol)
        if not data:
            return None

        try:
            outstanding = float(data.get("SharesOutstanding") or 0)
            float_shares = float(data.get("SharesFloat") or 0)
        except (TypeError, ValueError):
            return None

        if float_shares <= 0:
            return None

        # AV sometimes swaps the two fields (float > outstanding).
        if outstanding > 0 and float_shares > outstanding:
            float_shares, outstanding = outstanding, float_shares

        free_float_pct = (float_shares / outstanding * 100) if outstanding > 0 else 0.0

        return FloatData(
            symbol=symbol.upper(),
            float_shares=float_shares,
            float_shares_m=float_shares / 1_000_000,
            outstanding_shares=outstanding,
            free_float_pct=free_float_pct,
        )

    # ------------------------------------------------------------------
    # Profile  (drop-in replacement for FMPClient.get_profile)
    # ------------------------------------------------------------------

    def get_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """
        Fetch company profile from the OVERVIEW endpoint.

        AV fields mapped:
            Name      → name
            Sector    → sector
            Industry  → industry
            Exchange  → exchange
            (no direct IPO date — estimated from earliest split or None)
        """
        data = self._get_overview(symbol)
        if not data or not data.get("Symbol"):
            return None

        return CompanyProfile(
            symbol=symbol.upper(),
            name=data.get("Name", ""),
            sector=data.get("Sector", ""),
            industry=data.get("Industry", ""),
            exchange=data.get("Exchange", ""),
            ipo_date=None,  # AV OVERVIEW does not provide IPO date
            country=data.get("Country", ""),
        )

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def get_splits(self, symbol: str) -> list[SplitEvent]:
        """
        Fetch split history from the SPLITS endpoint.

        AV response format:
            { "symbol": "IBM", "data": [ { "effective_date": "...", "split_factor": "2.0" }, ... ] }

        split_factor > 1 → forward split (e.g. 2:1)
        split_factor < 1 → reverse split (e.g. 0.1 = 1:10)
        """
        data = self._get({"function": "SPLITS", "symbol": symbol.upper()})
        if not isinstance(data, dict):
            return []

        rows = data.get("data", [])
        if not isinstance(rows, list):
            return []

        events: list[SplitEvent] = []
        for row in rows:
            try:
                factor = float(row.get("split_factor", 1.0))
                date_str = row.get("effective_date", "")
                if factor == 1.0 or not date_str:
                    continue

                if factor >= 1.0:
                    # Forward split: e.g. factor=2.0 → 2 new shares for 1 old
                    numerator = factor
                    denominator = 1.0
                else:
                    # Reverse split: e.g. factor=0.1 → 1 new share for 10 old
                    numerator = 1.0
                    denominator = 1.0 / factor

                ratio = denominator / numerator
                events.append(SplitEvent(
                    symbol=symbol.upper(),
                    date=date_str,
                    numerator=numerator,
                    denominator=denominator,
                    ratio=ratio,
                    reverse=ratio > 1.0,
                ))
            except (TypeError, ValueError):
                continue

        events.sort(key=lambda e: e.date, reverse=True)
        return events

    def get_reverse_splits(self, symbol: str, days_back: int = 30) -> list[SplitEvent]:
        """Return only reverse splits within the last *days_back* days."""
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days_back)).date()
        return [
            e for e in self.get_splits(symbol)
            if e.reverse and e.date >= str(cutoff)
        ]

    def most_recent_reverse_split(self, symbol: str, days_back: int = 30) -> Optional[SplitEvent]:
        """Return the single most recent reverse split, or None."""
        splits = self.get_reverse_splits(symbol, days_back=days_back)
        return splits[0] if splits else None

    # ------------------------------------------------------------------
    # News + Sentiment  (replaces Finnhub)
    # ------------------------------------------------------------------

    def get_recent_news(
        self,
        symbol: str,
        minutes_back: int = 60,
        as_of: Optional[datetime] = None,
    ) -> list[FinnhubNewsArticle]:
        """
        Fetch news with per-ticker sentiment from the NEWS_SENTIMENT endpoint.

        Returns FinnhubNewsArticle objects (entity_sentiment populated from AV
        ticker_sentiment_score) for seamless drop-in with the existing pipeline.
        Sorted newest-first, filtered to articles within *minutes_back* of
        *as_of* (defaults to ``datetime.now(UTC)`` for live use).

        Passing *as_of* enables historical fetches for backtests — AV's
        NEWS_SENTIMENT endpoint supports arbitrary time_from/time_to windows.
        """
        sym = symbol.upper()

        # AV time_from/time_to format: YYYYMMDDTHHMM
        anchor = as_of if as_of is not None else datetime.now(tz=timezone.utc)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        else:
            anchor = anchor.astimezone(timezone.utc)
        since = anchor - timedelta(minutes=minutes_back)
        time_from = since.strftime("%Y%m%dT%H%M")
        time_to = anchor.strftime("%Y%m%dT%H%M")

        data = self._get({
            "function": "NEWS_SENTIMENT",
            "tickers": sym,
            "time_from": time_from,
            "time_to": time_to,
            "sort": "LATEST",
            "limit": "50",
        })

        if not isinstance(data, dict):
            return []

        feed = data.get("feed", [])
        if not isinstance(feed, list):
            return []

        articles: list[FinnhubNewsArticle] = []
        for item in feed:
            try:
                # Parse AV timestamp format: "20260407T234427"
                raw_ts = item.get("time_published", "")
                pub_dt = datetime.strptime(raw_ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

                # Only include articles within our time window
                if pub_dt < since or pub_dt > anchor:
                    continue

                # Extract ticker-specific sentiment
                entity_sentiment = None
                for ts in item.get("ticker_sentiment", []):
                    if ts.get("ticker", "").upper() == sym:
                        try:
                            entity_sentiment = float(ts["ticker_sentiment_score"])
                        except (KeyError, TypeError, ValueError):
                            pass
                        break

                articles.append(FinnhubNewsArticle(
                    symbol=sym,
                    title=item.get("title", ""),
                    source=item.get("source", ""),
                    url=item.get("url", ""),
                    published_at=pub_dt,
                    summary=item.get("summary", ""),
                    entity_sentiment=entity_sentiment,
                ))
            except (ValueError, TypeError):
                continue

        # Sort newest-first
        articles.sort(key=lambda a: a.published_at, reverse=True)
        return articles
