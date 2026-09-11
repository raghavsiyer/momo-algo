"""
market_data/fmp_client.py

Financial Modeling Prep (FMP) API integration.

Endpoints used:
    Float:   GET https://financialmodelingprep.com/stable/shares-float
             → fields: symbol, floatShares, outstandingShares, freeFloat (%)
             floatShares = true tradeable float (outstanding minus restricted)
             This is the real float — not a proxy.

    Splits:  GET https://financialmodelingprep.com/stable/splits-company
             → fields: date, numerator, denominator
             Reverse split detection: denominator > numerator
             (e.g. 1-for-10: numerator=1, denominator=10)

    Profile: GET https://financialmodelingprep.com/stable/profile
             → fields: ipoDate, sector, industry, companyName, exchange, ...
             ipoDate used for IPO recency check in eval.

Free tier: 250 requests/day
Base URL: https://financialmodelingprep.com (stable endpoints)

Usage:
    from market_data.fmp_client import FMPClient
    client = FMPClient()

    float_data  = client.get_float("ACMR")
    splits      = client.get_reverse_splits("ACMR", days_back=30)
    ipo_date    = client.get_ipo_date("ACMR")
    profile     = client.get_profile("ACMR")
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests

from config import HTTP_TIMEOUT
from market_data import resilient_session

logger = logging.getLogger(__name__)

_BASE = "https://financialmodelingprep.com"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FloatData:
    """True tradeable float from FMP shares-float endpoint."""
    symbol: str
    float_shares: float          # raw share count (not in millions)
    float_shares_m: float        # float_shares / 1_000_000 — convenient for scoring
    outstanding_shares: float    # total outstanding
    free_float_pct: float        # float_shares / outstanding * 100
    source: str = ""             # provider that supplied the data ("AV", "momo", "cache", etc.)


@dataclass
class SplitEvent:
    """A single stock split event. reverse=True means shares were consolidated."""
    symbol: str
    date: str                    # "YYYY-MM-DD"
    numerator: float             # shares after  (1 in a 1-for-10 reverse)
    denominator: float           # shares before (10 in a 1-for-10 reverse)
    ratio: float                 # denominator / numerator — >1.0 means reverse split
    reverse: bool


@dataclass
class CompanyProfile:
    """Slim company profile — only fields used by the evaluator."""
    symbol: str
    name: str
    sector: str
    industry: str
    exchange: str
    ipo_date: Optional[str]      # "YYYY-MM-DD" or None
    country: str = ""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FMPClient:
    """
    FMP REST client for float and company profile.
    Reads FMP_API_KEY from environment.
    """

    def __init__(self) -> None:
        key = os.getenv("FMP_API_KEY")
        if not key:
            raise EnvironmentError(
                "FMP_API_KEY not set. "
                "Add to .env: FMP_API_KEY=your_key  "
                "(free at financialmodelingprep.com)"
            )
        self._key = key
        self._session = resilient_session()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict | None = None) -> list | dict:
        p = {"apikey": self._key}
        if params:
            p.update(params)
        try:
            resp = self._session.get(f"{_BASE}{path}", params=p, timeout=HTTP_TIMEOUT)
            if not resp.ok:
                logger.warning(
                    f"FMP {resp.status_code} on {path}: {resp.text[:200]}"
                )
                return []
            return resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning(f"FMP request failed on {path}: {exc}")
            return []

    # ------------------------------------------------------------------
    # Float
    # ------------------------------------------------------------------

    def get_float(self, symbol: str) -> Optional[FloatData]:
        """
        Fetch true tradeable float from FMP stable shares-float endpoint.
        Returns None if the symbol is not found or data is missing.

        FMP endpoint: GET /stable/shares-float?symbol=ACMR
        Response: [{ symbol, floatShares, outstandingShares, freeFloat }]
        freeFloat is returned as a percentage (e.g. 85.3 means 85.3%).
        """
        data = self._get("/stable/shares-float", params={"symbol": symbol})
        if not data or not isinstance(data, list):
            return None

        row = data[0]
        float_shares = float(row.get("floatShares") or 0)
        outstanding = float(row.get("outstandingShares") or 0)
        free_float_pct = float(row.get("freeFloat") or 0)

        if float_shares <= 0:
            return None

        return FloatData(
            symbol=symbol,
            float_shares=float_shares,
            float_shares_m=float_shares / 1_000_000,
            outstanding_shares=outstanding,
            free_float_pct=free_float_pct,
        )

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------







    # ------------------------------------------------------------------
    # Profile (IPO date + sector)
    # ------------------------------------------------------------------

    def get_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """
        Fetch company profile.
        FMP endpoint: GET /stable/profile?symbol=ACMR
        Response: [{ ipoDate, sector, industry, companyName, exchangeShortName, ... }]
        Returns None if symbol not found.
        """
        data = self._get("/stable/profile", params={"symbol": symbol})
        if not data or not isinstance(data, list):
            return None

        row = data[0]
        ipo_raw = row.get("ipoDate") or None
        # Normalize empty string to None
        ipo_date = ipo_raw if ipo_raw and ipo_raw.strip() else None

        return CompanyProfile(
            symbol=symbol,
            name=row.get("companyName", ""),
            sector=row.get("sector", ""),
            industry=row.get("industry", ""),
            exchange=row.get("exchangeShortName", ""),
            ipo_date=ipo_date,
            country=row.get("country", ""),
        )

    def get_ipo_date(self, symbol: str) -> Optional[str]:
        """Convenience method — returns just the IPO date string or None."""
        profile = self.get_profile(symbol)
        return profile.ipo_date if profile else None