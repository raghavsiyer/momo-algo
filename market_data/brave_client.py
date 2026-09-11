"""
market_data/brave_client.py

Brave Search News API — last-resort news fallback.

Endpoint:
    GET https://api.search.brave.com/res/v1/news/search
    Header: X-Subscription-Token
    Params: q, count, freshness

Searches for "NASDAQ: {TICKER}" with freshness=pd (last 24h).
Returns up to 5 results as FinnhubNewsArticle objects (no sentiment —
scored downstream by FinBERT).

Usage:
    from market_data.brave_client import BraveNewsClient
    client = BraveNewsClient()
    articles = client.search_ticker_news("MBOT", max_results=5)
"""

import html
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import requests

from config import HTTP_TIMEOUT
from market_data import resilient_session
from market_data.finnhub_client import FinnhubNewsArticle

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.search.brave.com/res/v1/news/search"

# Non-English ccTLDs commonly seen in Brave results
_NON_EN_TLDS = frozenset(
    "id co.id br com.br jp co.jp cn com.cn kr co.kr tw com.tw"
    " de fr es it nl pt pl ru cz sk hu ro bg hr rs si ua by"
    " th vn ph my sg in com.my co.th co.in com.ph com.sg"
    " ar cl co com.co mx com.mx pe com.pe"
    " tr com.tr il co.il ae sa eg ma dz tn".split()
)


class BraveNewsClient:
    """Brave Search News API client — thin wrapper for ticker news lookups."""

    def __init__(self) -> None:
        self._token = os.getenv("BRAVE_SEARCH_API_KEY", "")
        self._session = resilient_session()
        self._session.headers.update({
            "Accept": "application/json",
            "X-Subscription-Token": self._token,
        })

    # ------------------------------------------------------------------

    def search_ticker_news(
        self,
        symbol: str,
        exchange: str = "",
    ) -> list[FinnhubNewsArticle]:
        """
        Search Brave News for *symbol* and return up to *max_results* articles
        sorted newest-first.

        Query: ``{EXCHANGE}: {SYMBOL}`` (falls back to ``{SYMBOL} stock news``
        when exchange is unknown).
        Freshness: ``pd`` (past day / 24 hours).
        """
        if not self._token:
            return []

        sym = symbol.upper()

        # Build query: use exchange if known, else generic
        if exchange:
            query = f"{exchange}: {sym}"
        else:
            query = f"{sym} stock news"

        try:
            resp = self._session.get(
                _BASE_URL,
                params={
                    "q": query,
                    "freshness": "pd",
                    "search_lang": "en",
                    "country": "us",
                },
                timeout=HTTP_TIMEOUT,
            )
            if not resp.ok:
                logger.warning(
                    f"[{sym}] Brave Search {resp.status_code}: "
                    f"{resp.text[:200]} — treating as no news"
                )
                return []

            data = resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning(f"[{sym}] Brave Search request failed: {exc}")
            return []

        results = data.get("results") or []

        articles: list[FinnhubNewsArticle] = []
        for item in results:
            # Skip non-English results — Brave's search_lang param only
            # affects query interpretation, not result language filtering
            lang = (item.get("language") or "").lower()
            if lang and lang != "en":
                continue

            # Skip results from non-English domains
            hostname = ((item.get("meta_url", {}) or {}).get("hostname") or "").lower()
            if any(hostname.endswith(f".{tld}") for tld in _NON_EN_TLDS):
                continue

            # Parse page_age ISO timestamp: "2026-04-07T21:31:37+00:00"
            pub_dt = self._parse_timestamp(item)

            articles.append(FinnhubNewsArticle(
                symbol=sym,
                title=html.unescape((item.get("title") or "").strip()),
                source=(item.get("meta_url", {}) or {}).get("hostname", ""),
                url=item.get("url", ""),
                published_at=pub_dt,
                summary=html.unescape((item.get("description") or "").strip()),
            ))

        # Sort newest-first
        articles.sort(key=lambda a: a.published_at, reverse=True)
        return articles

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(item: dict) -> datetime:
        """Best-effort extraction of a tz-aware UTC datetime from a result."""
        # Try page_age first (ISO 8601), then page_fetched
        for key in ("page_age", "page_fetched"):
            raw = item.get(key)
            if not raw:
                continue
            try:
                dt = datetime.fromisoformat(raw)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (ValueError, TypeError):
                continue
        # Fallback: epoch — unknown timestamps sort as old, not fresh
        return datetime(2000, 1, 1, tzinfo=timezone.utc)
