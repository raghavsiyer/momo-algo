"""
market_data/firecrawl_float.py

Float from stockanalysis.com's statistics page through Firecrawl, the way the
ScalpFeed Quantower connection gets it (~/.scalpfeed/project,
ScalpFeedVendor/Feed/FloatSource.cs and FloatFacts.cs), sharing its cache so a
symbol either one has looked up costs the other nothing.

Why: on 2026-11-18 the bot's other float sources stopped answering for small
caps on this account. Alpha Vantage's OVERVIEW said "premium endpoint" for
every symbol, AAPL included; FMP's shares-float returned 402 for GDHG (large
caps only on the free plan); MomoScreener's livequote went unanswered for 25 s
and its REST API now wants a login.

The request: POST https://api.firecrawl.dev/v2/scrape with
    {"alexandria": {"provider": "stockanalysis-com",
                    "capability": "securities/stock_statistics",
                    "options": {"symbol": "<SYMBOL, BRK.B style>"}}}
Each lookup costs 5 credits. The answer's statistics have a "shares" section
(float, sharesout) and a "stockPrice" one (averageVolume, 20 days); each item's
"hover" is the exact number.

The cache, ~/.scalpfeed/float-cache.json, one entry per symbol:
    {"GDHG": {"day": "2026-11-16", "float": 29519326, "sharesOut": 47127140,
              "averageVolume": 5256126, "fetchedMs": 1794860000000}, ...}
A float found is good for a week; a symbol the site doesn't have (float null)
until the next Eastern day. Writes merge with what's on disk (newest
fetchedMs wins per symbol), so the bot and ScalpFeed don't drop each other's.

The key is the same one ScalpFeed uses, apiKey in ~/.scalpfeed/firecrawl.json,
or FIRECRAWL_API_KEY in the environment. It never goes in the log.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from zoneinfo import ZoneInfo

import requests

from market_data.fmp_client import FloatData

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.firecrawl.dev/v2/scrape"
KEEP_FOUND = timedelta(days=7)
RETRY_AFTER_FAILURE_SECS = 600.0
# A 20-day average volume this many times the float or more means the float is
# out of date (FGL, 2026-11-16: float 8,557 after a 1:100 reverse split, trading
# 1.57M a day). ScalpFeed blanks such a float; the bot doesn't use it.
DOUBTFUL_TURNOVER = 50.0

_ET = ZoneInfo("America/New_York")
_SCALPFEED_DIR = os.path.join(os.path.expanduser("~"), ".scalpfeed")
DEFAULT_CACHE_PATH = os.path.join(_SCALPFEED_DIR, "float-cache.json")
DEFAULT_KEY_PATH = os.path.join(_SCALPFEED_DIR, "firecrawl.json")

# (status, body) for one POST of this JSON with this key.
Send = Callable[[str, dict], tuple[int, str]]


def eastern_day(utc: datetime) -> str:
    return utc.astimezone(_ET).strftime("%Y-%m-%d")


def site_symbol(symbol: str) -> str:
    """stockanalysis.com writes share classes with a dot (BRK.B)."""
    return symbol.replace("/", ".")


def _number(text) -> float:
    try:
        return float(str(text).replace(",", ""))
    except (TypeError, ValueError):
        return math.nan


def facts_from_statistics(statistics: dict, day: str, fetched_ms: int) -> dict:
    """A cache entry from one answer's statistics."""
    def item(section: str, item_id: str) -> float:
        for s in (statistics or {}).get("sections") or []:
            if s.get("id") != section:
                continue
            for it in s.get("items") or []:
                if it.get("id") == item_id:
                    return _number(it.get("hover") or it.get("value"))
        return math.nan

    return {
        "day": day,
        "float": item("shares", "float"),
        "sharesOut": item("shares", "sharesout"),
        "averageVolume": item("stockPrice", "averageVolume"),
        "fetchedMs": fetched_ms,
    }


def _finite(value) -> float:
    return value if isinstance(value, (int, float)) and math.isfinite(value) else math.nan


def known(facts: dict) -> bool:
    return _finite(facts.get("float")) > 0


def doubtful(facts: dict) -> bool:
    flt = _finite(facts.get("float"))
    avg = _finite(facts.get("averageVolume"))
    return flt > 0 and avg / flt >= DOUBTFUL_TURNOVER


def fresh(facts: dict, utc: datetime) -> bool:
    """A float found is good for a week; that the site hasn't got a symbol, until the next day."""
    if known(facts):
        fetched = datetime.fromtimestamp(int(facts.get("fetchedMs") or 0) / 1000, tz=timezone.utc)
        return utc - fetched < KEEP_FOUND
    return facts.get("day") == eastern_day(utc)


def read_cache(path: str) -> dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return {}
    cache: dict[str, dict] = {}
    for symbol, entry in (raw or {}).items():
        if not isinstance(entry, dict):
            continue
        cache[symbol.upper()] = {
            "day": entry.get("day"),
            "float": _finite(entry.get("float")),
            "sharesOut": _finite(entry.get("sharesOut")),
            "averageVolume": _finite(entry.get("averageVolume")),
            "fetchedMs": int(entry.get("fetchedMs") or 0),
        }
    return cache


def write_cache(path: str, entries: dict[str, dict]) -> None:
    """Merge with what's on disk (newest per symbol), then swap the whole file in."""
    merged = read_cache(path)
    for symbol, entry in entries.items():
        old = merged.get(symbol.upper())
        if old is None or entry.get("fetchedMs", 0) >= old.get("fetchedMs", 0):
            merged[symbol.upper()] = entry

    def out(v):
        return v if isinstance(v, (int, float)) and math.isfinite(v) else None

    data = {
        s: {"day": e.get("day"), "float": out(e.get("float")), "sharesOut": out(e.get("sharesOut")),
            "averageVolume": out(e.get("averageVolume")), "fetchedMs": e.get("fetchedMs", 0)}
        for s, e in sorted(merged.items())
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temp = f"{path}.{os.getpid()}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    os.replace(temp, path)


def _read_key_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return str(json.load(handle).get("apiKey") or "")
    except (OSError, ValueError, AttributeError):
        return ""


class FirecrawlFloatClient:
    """Float for any symbol from the shared cache, else one Firecrawl lookup."""

    def __init__(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        key_path: str = DEFAULT_KEY_PATH,
        send: Optional[Send] = None,
        utc_now: Optional[Callable[[], datetime]] = None,
        timeout: float = 60.0,
    ) -> None:
        self._cache_path = cache_path
        self._key_path = key_path
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._timeout = timeout
        self._session = requests.Session() if send is None else None
        self._send = send or self._post
        self._lock = threading.Lock()  # one lookup at a time, as ScalpFeed does
        self._retry_at: dict[str, float] = {}
        self._paused_until = 0.0
        self._key_missing_logged = False

    def _key(self) -> str:
        return os.getenv("FIRECRAWL_API_KEY", "") or _read_key_file(self._key_path)

    def _post(self, key: str, body: dict) -> tuple[int, str]:
        response = self._session.post(
            ENDPOINT, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=self._timeout,
        )
        return response.status_code, response.text

    def facts(self, symbol: str) -> Optional[dict]:
        """The cache entry for *symbol*, fetched first unless fresh. None if not known yet."""
        sym = symbol.upper()
        with self._lock:
            now = self._utc_now()
            cached = read_cache(self._cache_path).get(sym)
            if cached is not None and fresh(cached, now):
                return cached
            mono = time.monotonic()
            if mono < self._paused_until or mono < self._retry_at.get(sym, 0.0):
                return cached
            fetched = self._fetch(sym, now)
            return fetched if fetched is not None else cached

    def get_float(self, symbol: str) -> Optional[FloatData]:
        """FloatData for *symbol*, or None when unknown, failed or doubtful."""
        facts = self.facts(symbol)
        if facts is None or not known(facts):
            return None
        if doubtful(facts):
            logger.warning(
                "[float] %s float %.0f from stockanalysis.com trades %.0fx a day, so likely out of date; not used",
                symbol, facts["float"], facts["averageVolume"] / facts["float"],
            )
            return None
        flt = facts["float"]
        out = _finite(facts.get("sharesOut"))
        return FloatData(
            symbol=symbol.upper(),
            float_shares=flt,
            float_shares_m=flt / 1_000_000,
            outstanding_shares=out if out > 0 else 0.0,
            free_float_pct=(flt / out * 100) if out > 0 else 0.0,
            source="firecrawl",
        )

    def _fetch(self, sym: str, now: datetime) -> Optional[dict]:
        key = self._key()
        if not key.strip():
            if not self._key_missing_logged:
                logger.info("[float] no Firecrawl key: set apiKey in %s or FIRECRAWL_API_KEY", self._key_path)
            self._key_missing_logged = True
            return None
        self._key_missing_logged = False

        body = {"alexandria": {"provider": "stockanalysis-com", "capability": "securities/stock_statistics",
                               "options": {"symbol": site_symbol(sym)}}}
        try:
            status, text = self._send(key, body)
        except requests.RequestException as exc:
            return self._failed(sym, type(exc).__name__)

        if status in (401, 403):
            return self._pause(1800, f"Firecrawl refused the key (HTTP {status})")
        if status == 402:
            return self._pause(3600, "Firecrawl says the account is out of credits (HTTP 402)")
        if status == 429:
            return self._pause(60, "Firecrawl is rate limiting (HTTP 429)")
        if not 200 <= status < 300:
            return self._failed(sym, f"HTTP {status}")

        try:
            result = ((json.loads(text) or {}).get("data") or {}).get("alexandria")[0]
        except (ValueError, TypeError, IndexError, AttributeError):
            return self._failed(sym, "unreadable answer")

        day = eastern_day(now)
        fetched_ms = int(now.timestamp() * 1000)
        error = result.get("error") or {}
        if error.get("code") == "not_found":
            facts = {"day": day, "float": math.nan, "sharesOut": math.nan, "averageVolume": math.nan,
                     "fetchedMs": fetched_ms}
        elif isinstance(result.get("data"), dict):
            facts = facts_from_statistics(result["data"], day, fetched_ms)
        else:
            return self._failed(sym, str(error.get("message") or "no statistics in the answer").replace(key, "<key>"))

        self._retry_at.pop(sym, None)
        try:
            write_cache(self._cache_path, {sym: facts})
        except OSError as exc:
            logger.warning("[float] couldn't write %s: %s", self._cache_path, type(exc).__name__)
        logger.info(
            "[float] %s %s, %s credits", sym,
            f"float {facts['float']:,.0f}" if known(facts) else "not on stockanalysis.com",
            result.get("creditsCost", "?"),
        )
        return facts

    def _failed(self, sym: str, why: str) -> None:
        self._retry_at[sym] = time.monotonic() + RETRY_AFTER_FAILURE_SECS
        logger.info("[float] %s not fetched, trying again in 10 minutes: %s", sym, why)
        return None

    def _pause(self, secs: float, why: str) -> None:
        self._paused_until = time.monotonic() + secs
        logger.warning("[float] %s. No float lookups for %.0f minutes.", why, secs / 60)
        return None
