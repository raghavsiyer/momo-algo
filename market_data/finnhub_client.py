"""
market_data/finnhub_client.py

Finnhub REST client — company news (free) and profile (free).

Endpoints:
    GET /api/v1/company-news   — news headlines (free tier)
    GET /api/v1/stock/profile2 — company profile (free tier)

Usage:
    from market_data.finnhub_client import FinnhubClient
    client = FinnhubClient()
    articles = client.get_recent_news("ACMR", minutes_back=60)
    profile  = client.get_profile("ACMR")
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from config import HTTP_TIMEOUT
from market_data import resilient_session
from market_data.fmp_client import CompanyProfile

logger = logging.getLogger(__name__)

_BASE = "https://finnhub.io/api/v1"


# ---------------------------------------------------------------------------
# Dataclass — compatible with the news_articles field on WatchlistEntry
# ---------------------------------------------------------------------------

@dataclass
class FinnhubNewsArticle:
    """Single Finnhub news article."""
    symbol: str
    title: str
    source: str
    url: str
    published_at: datetime          # tz-aware UTC
    summary: str
    # Always None — no free-tier sentiment. Present so print code stays the same.
    entity_sentiment: Optional[float] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FinnhubClient:
    """
    Finnhub REST client for news and company profile.
    Reads FINNHUB_API_KEY from environment.
    """

    def __init__(self) -> None:
        token = os.getenv("FINNHUB_API_KEY")
        if not token:
            raise EnvironmentError(
                "FINNHUB_API_KEY not set. "
                "Add to .env: FINNHUB_API_KEY=your_token"
            )
        self._token = token
        self._session = resilient_session()
        self._profile_cache: dict[str, dict] = {}
        self._profile_fail_ts: dict[str, float] = {}  # symbol → monotonic timestamp of last failure
        self._PROFILE_FAIL_TTL = 300  # retry failed lookups after 5 minutes
        self._news_cache: dict[str, tuple[float, list]] = {}  # symbol → (timestamp, articles)
        self._NEWS_CACHE_TTL = 600  # 10 minutes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        p = {"token": self._token}
        if params:
            p.update(params)
        try:
            resp = self._session.get(f"{_BASE}{path}", params=p, timeout=HTTP_TIMEOUT)
            if not resp.ok:
                logger.warning(f"Finnhub {resp.status_code} on {path}: {resp.text[:200]}")
                return {}
            data = resp.json()
            return data
        except requests.exceptions.RequestException as exc:
            logger.warning(f"Finnhub request failed ({path}): {exc}")
            return {}

    def _get_profile_raw(self, symbol: str) -> dict:
        sym = symbol.upper()
        if sym in self._profile_cache:
            return self._profile_cache[sym]
        # If a previous lookup failed, respect the TTL before retrying
        fail_ts = self._profile_fail_ts.get(sym)
        if fail_ts is not None and (time.monotonic() - fail_ts) < self._PROFILE_FAIL_TTL:
            return {}
        data = self._get("/stock/profile2", {"symbol": sym})
        if isinstance(data, dict) and data.get("ticker"):
            self._profile_cache[sym] = data
            self._profile_fail_ts.pop(sym, None)
        else:
            self._profile_fail_ts[sym] = time.monotonic()
            data = {}
        return data

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    def get_profile(self, symbol: str) -> Optional[CompanyProfile]:
        data = self._get_profile_raw(symbol)
        if not data:
            return None
        return CompanyProfile(
            symbol=symbol.upper(),
            name=data.get("name", ""),
            sector=data.get("finnhubIndustry", ""),
            industry=data.get("finnhubIndustry", ""),
            exchange=data.get("exchange", ""),
            ipo_date=data.get("ipo"),
            country=data.get("country", ""),
        )

    # ------------------------------------------------------------------
    # News
    # ------------------------------------------------------------------

    def get_recent_news(
        self,
        symbol: str,
        minutes_back: int = 60,
        as_of: Optional[datetime] = None,
    ) -> list[FinnhubNewsArticle]:
        """
        Return articles for *symbol* published within *minutes_back* of *as_of*.

        *as_of* defaults to ``datetime.now(UTC)`` for live use; pass an
        explicit timestamp for historical fetches (backtests).  The
        Finnhub ``company-news`` endpoint supports arbitrary date ranges,
        so historical windows return real results.

        Session-level cache is skipped when *as_of* is provided so each
        historical fetch hits the API fresh (different anchors yield
        different windows).
        """
        sym = symbol.upper()
        historical = as_of is not None
        anchor = as_of if historical else datetime.now(tz=timezone.utc)
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        else:
            anchor = anchor.astimezone(timezone.utc)

        # Return cached if available and not expired (live calls only)
        if not historical and sym in self._news_cache:
            cached_ts, cached_articles = self._news_cache[sym]
            if time.time() - cached_ts < self._NEWS_CACHE_TTL:
                cutoff = anchor - timedelta(minutes=minutes_back)
                return [a for a in cached_articles if a.published_at >= cutoff]

        since = anchor - timedelta(minutes=minutes_back)
        cutoff_ts = since.timestamp()
        cap_ts = anchor.timestamp()

        # Look back far enough to cover the full minutes_back window.
        # +1 day ensures articles published around midnight aren't clipped
        # by date-boundary rounding in the Finnhub API.
        days_back = max(1, minutes_back // (24 * 60)) + 1
        date_from = (anchor - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to   = anchor.strftime("%Y-%m-%d")

        try:
            resp = self._session.get(
                f"{_BASE}/company-news",
                params={
                    "symbol": symbol.upper(),
                    "from":   date_from,
                    "to":     date_to,
                    "token":  self._token,
                },
                timeout=HTTP_TIMEOUT,
            )
            if not resp.ok:
                logger.warning(
                    f"[{symbol}] Finnhub {resp.status_code}: {resp.text[:200]} — treating as no news"
                )
                return []
            raw_list = resp.json()
            if not isinstance(raw_list, list):
                return []
        except requests.exceptions.RequestException as exc:
            logger.warning(f"[{symbol}] Finnhub request failed: {exc}")
            return []

        articles = []
        for item in raw_list:
            unix_ts = item.get("datetime", 0)
            if not unix_ts or unix_ts < cutoff_ts or unix_ts > cap_ts:
                continue
            published_at = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
            articles.append(FinnhubNewsArticle(
                symbol=symbol.upper(),
                title=item.get("headline", "").strip(),
                source=item.get("source", ""),
                url=item.get("url", ""),
                published_at=published_at,
                summary=item.get("summary", ""),
            ))

        articles.sort(key=lambda a: a.published_at, reverse=True)
        self._news_cache[sym] = (time.time(), articles)
        return articles
