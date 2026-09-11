"""
cache.py  —  Disk-backed ticker cache for evaluated pipeline results.

Stores WatchlistEntry + ResistanceMap data per symbol so re-queries
(from any input source) can instantly replay the execution summary
without burning API calls.

The cache persists across sessions — float, sector, IPO, and other
slow-changing fields are reused for free on subsequent runs. Volatile
fields (price, vol, news freshness) are refreshed on every cache hit
in the main loop before the summary is printed.

Usage:
    from cache import TickerCache
    cache = TickerCache()
    cache.put("ACMR", entry, rmap)
    hit = cache.get("ACMR")          # (entry, rmap) or None
    cache.has("ACMR")                # True
    cache.clear()                    # wipes memory + disk
"""

import logging
import pickle
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from config import CACHE_TTL_DAYS, MarketClock, NEWS_OLD_DAYS_MAX, news_freshness_cutoffs

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "logs" / "ticker_cache.pkl"
_SNAPSHOT_DIR = Path(__file__).parent / "logs" / "cache_snapshots"
_TTL = CACHE_TTL_DAYS * 24 * 3600   # evict on load only, never mid-session

CachedNews = tuple[list, int, Optional[float], str]


def _article_pub_utc(article) -> datetime | None:
    pub = getattr(article, "published_at", None)
    if pub is None:
        return None
    if pub.tzinfo is None:
        return pub.replace(tzinfo=timezone.utc)
    return pub.astimezone(timezone.utc)


def news_cache_cutoff() -> datetime:
    """Oldest article timestamp still usable as a cached catalyst fallback."""
    try:
        from config import _MarketCalendar
        return datetime.now(tz=timezone.utc) - _MarketCalendar.lookback(2)
    except Exception:
        return datetime.now(tz=timezone.utc) - timedelta(days=NEWS_OLD_DAYS_MAX)


def _sorted_articles(articles: list) -> list:
    return sorted(
        list(articles),
        key=lambda art: _article_pub_utc(art) or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )


def apply_cached_news(entry, cached_news: CachedNews | None) -> bool:
    """Apply cached news to an entry when it is newer than the entry's current article."""
    if not cached_news:
        return False
    articles, news_count, avg_sentiment, news_source = cached_news
    articles = _sorted_articles(articles)
    if not articles:
        return False
    cached_pub = _article_pub_utc(articles[0])
    if cached_pub is None:
        return False

    current_articles = getattr(entry, "news_articles", None) or []
    current_pub = _article_pub_utc(current_articles[0]) if current_articles else None
    if current_pub is not None and current_pub >= cached_pub:
        return False

    session_cutoff, breaking_cutoff, _ = news_freshness_cutoffs()
    entry.news_articles = articles[:1]
    entry.news_count_last_hour = news_count
    entry.avg_sentiment = avg_sentiment
    entry.news_source = news_source
    entry.news_is_fresh = cached_pub >= session_cutoff
    entry.news_is_breaking = cached_pub >= breaking_cutoff
    if getattr(entry, "evaluation", None):
        entry.evaluation.update_news(avg_sentiment, news_count, news_source)
        try:
            if hasattr(entry, "eval_confidence"):
                entry.static_confidence = entry.eval_confidence
            else:
                from config import CONFIDENCE_CEILING, CONFIDENCE_FLOOR
                raw = entry.evaluation.stock_eval.score * 100.0
                entry.static_confidence = round(max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, raw)), 1)
        except Exception:
            pass
    return True


class _CacheUnpickler(pickle.Unpickler):
    """Map legacy cache class names onto current runtime classes."""

    def find_class(self, module, name):
        if module == "strategy.evaluator" and name == "StaticEvaluation":
            from strategy.evaluator import CompositeEvaluation
            return CompositeEvaluation
        return super().find_class(module, name)


def _load_cache_pickle(file_obj):
    return _CacheUnpickler(file_obj).load()


def _has_stale_scoring_vars(ev) -> bool:
    """Detect ScoringVariable instances pickled before new slots were added."""
    groups = getattr(ev, "groups", None) or []
    required = ("cutoff", "cutoff_lo")
    for g in groups:
        for v in getattr(g, "variables", []) or []:
            if any(not hasattr(v, attr) for attr in required):
                return True
    return False


def _refresh_scoring_thresholds(ev) -> None:
    """Re-apply current config thresholds (lo/hi/cutoff/fn/weight/...) to a
    pickled evaluation's ScoringVariables.

    Cached entries keep their stored *raw values* + *qualities* but their
    threshold bounds get rewired to whatever ``SCORING_GROUPS`` currently
    says — so a config tweak (e.g. ``daily_trend hi=200 → 400``) is visible
    on cache-hit display without re-evaluating the ticker.
    """
    groups = getattr(ev, "groups", None) or []
    if not groups:
        return
    try:
        from config import SCORING_GROUPS, SETUP_PATTERN_GROUP
        from strategy.scoring import QualityFn
    except Exception:
        return

    # Flatten {var_name: var_def} from current config.
    cfg_vars: dict[str, dict] = {}
    for gd in list(SCORING_GROUPS) + [SETUP_PATTERN_GROUP]:
        for vd in gd.get("vars", []):
            cfg_vars[vd["name"]] = vd

    for g in groups:
        for v in getattr(g, "variables", []) or []:
            vd = cfg_vars.get(getattr(v, "name", ""))
            if not vd:
                continue
            try:
                fn = vd["fn"]
                v.fn = QualityFn[fn] if isinstance(fn, str) else fn
                v.weight = vd.get("weight", v.weight)
                v.lo = vd.get("lo", 0.0)
                v.hi = vd.get("hi", 0.0)
                v.skew_peak = vd.get("skew_peak")
                v.cutoff = vd.get("cutoff")
                v.cutoff_lo = vd.get("cutoff_lo")
                v.log_scale = vd.get("log_scale", False)
                v.optional = vd.get("optional", False)
                v.dampened = vd.get("dampened", False)
                # Re-score the stored raw value against the new bounds so
                # the displayed quality matches what a fresh evaluation
                # would produce.
                if getattr(v, "value", None) is not None:
                    v.evaluate(float(v.value))
            except Exception:
                # Slot-based dataclass with unknown attr — skip silently.
                continue


# Live (dynamic) groups whose variables are computed by DynamicScorer at
# runtime. On cache load these are reset so stale per-bar quality from a
# previous session doesn't bleed into the new one. STOCK_EVAL is the
# static Stage-2 layer (float, news, profile, rvol_1day, daily_trend) and
# is preserved as-is.
_LIVE_GROUP_NAMES = {"TECHNICAL_ANALYSIS"}


def _reset_live_state(ev) -> None:
    """Reset dynamic group variables to neutral and ensure new VariableGroup
    slots (e.g. ``score_override``) are initialized on unpickled instances.

    Preserves all static fields on the entry (float, news, sector, etc.)
    and the static scoring layer (STOCK_EVAL).
    """
    try:
        from config import SCORING_GROUPS
        from strategy.scoring import build_groups

        pa_def = next((g for g in SCORING_GROUPS if g["name"] == "TECHNICAL_ANALYSIS"), None)
        pa = getattr(ev, "technical_analysis", None)
        if pa_def is not None and pa is not None:
            expected = [v["name"] for v in pa_def.get("vars", [])]
            actual = [getattr(v, "name", "") for v in getattr(pa, "variables", []) or []]
            if actual != expected:
                ev.technical_analysis = build_groups([pa_def])[0]
    except Exception:
        pass

    for g in getattr(ev, "groups", None) or []:
        # Initialize newly-added slots on legacy pickles. Doing a write
        # (not just read) materializes the slot so future reads succeed.
        try:
            if getattr(g, "score_override", None) is not None or True:
                g.score_override = None
        except Exception:
            pass
        # Wipe live variables in dynamic groups so the next scorer tick
        # starts from a clean slate rather than yesterday's last quality.
        if getattr(g, "name", "") in _LIVE_GROUP_NAMES:
            for v in getattr(g, "variables", []) or []:
                v.value = None
                v.quality = 0.5


class TickerCache:
    """Thread-safe, pickle-backed {symbol: (entry, rmap)} store."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _DEFAULT_PATH
        self._lock = threading.Lock()
        self._data: dict[str, tuple] = {}   # sym -> (entry, rmap, first_seen_ts, last_updated_ts, session_found_ts)
        # sym -> (headlines, first_seen_ts)
        self._pending_yt_news: dict[str, tuple[list, float]] = {}
        # sym -> (articles, news_count, avg_sentiment, news_source, last_updated_ts)
        self._news_cache: dict[str, tuple] = {}
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, symbol: str) -> bool:
        with self._lock:
            return symbol.upper() in self._data

    def get(self, symbol: str) -> Optional[tuple]:
        """Return (entry, rmap) for *symbol*, or None if missing."""
        with self._lock:
            item = self._data.get(symbol.upper())
            if item is None:
                return None
            return (item[0], item[1])

    def get_ts(self, symbol: str) -> Optional[float]:
        """Return the Unix timestamp when *symbol* was cached, or None."""
        with self._lock:
            item = self._data.get(symbol.upper())
            if item is None or len(item) < 3:
                return None
            return item[2]

    def get_updated_ts(self, symbol: str) -> Optional[float]:
        """Return the Unix timestamp when *symbol* was last updated, or None."""
        with self._lock:
            item = self._data.get(symbol.upper())
            if item is None:
                return None
            if len(item) >= 4:
                return item[3]
            if len(item) >= 3:
                return item[2]
            return None

    def get_session_found_ts(self, symbol: str) -> Optional[float]:
        """Return the timestamp when *symbol* was first found on its most recent
        active (moving) day, or None.

        Unlike :meth:`get_ts` (which preserves the very first time the symbol was
        ever cached), this resets each trading day, so it reflects when the
        symbol was found *today* — or, if it wasn't active today, when it was
        found on the most recent day it was. Falls back to first_seen for
        legacy entries stored before this field existed.
        """
        with self._lock:
            item = self._data.get(symbol.upper())
            if item is None:
                return None
            if len(item) >= 5 and item[4] is not None:
                return item[4]
            if len(item) >= 3:
                return item[2]
            return None

    def put(self, symbol: str, entry, rmap) -> None:
        """Store an evaluation result."""
        with self._lock:
            now = time.time()
            existing = self._data.get(symbol.upper())
            first_seen_ts = existing[2] if existing is not None and len(existing) >= 3 else now
            # Session "found" timestamp: the first put on the current trading day
            # (ET). Preserve it while the most recent found day is still the
            # current trading day; otherwise reset to now so "Found" tracks the
            # most recent moving day rather than the original first-seen date.
            session_found_ts = now
            if existing is not None and len(existing) >= 5 and existing[4] is not None:
                prev_found = float(existing[4])
                prev_day = datetime.fromtimestamp(prev_found, tz=timezone.utc).astimezone(MarketClock.ET).date()
                if prev_day == datetime.now(tz=MarketClock.ET).date():
                    session_found_ts = prev_found
            self._data[symbol.upper()] = (entry, rmap, first_seen_ts, now, session_found_ts)
            self._store_entry_news_locked(symbol, entry)
            self._dirty = True

    def put_news(
        self,
        symbol: str,
        articles: list,
        news_count: int,
        avg_sentiment: Optional[float],
        news_source: str,
    ) -> None:
        """Persist a reusable news record independent of the ticker entry."""
        if not articles:
            return
        with self._lock:
            if self._store_news_locked(symbol, articles, news_count, avg_sentiment, news_source):
                self._dirty = True

    def get_news(self, symbol: str, stale_cutoff: datetime | None = None) -> CachedNews | None:
        """Return cached news if the latest article is still inside the catalyst window."""
        cutoff = stale_cutoff or news_cache_cutoff()
        with self._lock:
            sym = symbol.upper()
            item = self._news_cache.get(sym)
            if not item or len(item) < 5:
                return None
            articles, news_count, avg_sentiment, news_source, updated_ts = item
            if time.time() - updated_ts > _TTL:
                self._news_cache.pop(sym, None)
                self._dirty = True
                return None
            articles = _sorted_articles(list(articles))
            if not articles:
                return None
            pub = _article_pub_utc(articles[0])
            if pub is None or pub < cutoff:
                return None
            return (articles, int(news_count or len(articles)), avg_sentiment, str(news_source or "cache"))

    def _store_entry_news_locked(self, symbol: str, entry) -> bool:
        return self._store_news_locked(
            symbol,
            getattr(entry, "news_articles", []) or [],
            int(getattr(entry, "news_count_last_hour", 0) or 0),
            getattr(entry, "avg_sentiment", None),
            str(getattr(entry, "news_source", "") or "cache"),
        )

    def _store_news_locked(
        self,
        symbol: str,
        articles: list,
        news_count: int,
        avg_sentiment: Optional[float],
        news_source: str,
        updated_ts: float | None = None,
    ) -> bool:
        articles = _sorted_articles(articles)
        if not articles:
            return False
        sym = symbol.upper()
        new_pub = _article_pub_utc(articles[0])
        if new_pub is None:
            return False
        existing = self._news_cache.get(sym)
        if existing and len(existing) >= 1:
            old_articles = _sorted_articles(list(existing[0]))
            old_pub = _article_pub_utc(old_articles[0]) if old_articles else None
            if old_pub is not None and old_pub >= new_pub:
                return False
        self._news_cache[sym] = (
            articles[:1],
            int(news_count or len(articles)),
            avg_sentiment,
            str(news_source or "cache"),
            updated_ts or time.time(),
        )
        return True

    def add_pending_yt_news(self, symbol: str, headline) -> None:
        """Append one pending YouTube headline for *symbol*."""
        with self._lock:
            sym = symbol.upper()
            existing = self._pending_yt_news.get(sym)
            if existing is None:
                self._pending_yt_news[sym] = ([headline], time.time())
            else:
                existing[0].append(headline)
            self._dirty = True

    def get_pending_yt_news(self, symbol: str):
        """Return pending YouTube headlines for *symbol* without consuming."""
        with self._lock:
            item = self._pending_yt_news.get(symbol.upper())
            if item is None:
                return None
            return list(item[0])

    def pop_pending_yt_news(self, symbol: str):
        """Consume and return pending YouTube headlines for *symbol*."""
        with self._lock:
            item = self._pending_yt_news.pop(symbol.upper(), None)
            if item is None:
                return None
            self._dirty = True
            return list(item[0])

    def flush(self) -> None:
        """Write pending changes to disk (call periodically or on shutdown)."""
        with self._lock:
            if self._dirty:
                self._save()
                self._dirty = False
        self._maybe_snapshot()

    def _maybe_snapshot(self) -> None:
        """Auto-save a daily snapshot on each flush (overwrites to stay current)."""
        try:
            self.snapshot_day()
        except Exception as exc:
            logger.debug("Snapshot failed: %s", exc)

    def clear(self) -> None:
        """Wipe in-memory data and delete the cache file."""
        with self._lock:
            self._data.clear()
            self._pending_yt_news.clear()
            self._news_cache.clear()
            self._dirty = False
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def remove(self, symbol: str) -> bool:
        """Remove a single ticker from the cache. Returns True if it existed."""
        with self._lock:
            sym = symbol.upper()
            removed_entry = self._data.pop(sym, None) is not None
            removed_pending = self._pending_yt_news.pop(sym, None) is not None
            removed_news = self._news_cache.pop(sym, None) is not None
            removed = removed_entry or removed_pending or removed_news
            if removed:
                self._dirty = True
            return removed

    @property
    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "rb") as f:
                raw = _load_cache_pickle(f)
        except Exception as exc:
            logger.warning("Failed to load ticker cache (%s) — starting fresh: %s",
                           self._path, exc)
            return

        now = time.time()
        loaded = evicted = 0
        loaded_pending = evicted_pending = 0
        loaded_news = evicted_news = 0
        backfilled_news = 0

        if isinstance(raw, dict) and "entries" in raw:
            raw_entries = raw.get("entries", {})
            raw_pending = raw.get("pending_yt_news", {})
            raw_news = raw.get("news_cache", {})
        else:
            # Backward-compatible path for older cache files.
            raw_entries = raw
            raw_pending = {}
            raw_news = {}

        for sym, item in raw_entries.items():
            ts = item[2] if len(item) >= 3 else 0.0
            if now - ts > _TTL:
                evicted += 1
                continue
            entry = item[0]
            updated_ts = item[3] if len(item) >= 4 else ts
            self._data[sym] = (entry, item[1], ts, updated_ts)
            # Backfill any dataclass fields added after this entry was pickled
            # so legacy cache entries don't crash on attribute access (e.g.
            # execution_snapshot, setup_* fields).
            try:
                from dataclasses import fields as _dc_fields, MISSING as _DC_MISSING
                for _f in _dc_fields(entry):
                    if hasattr(entry, _f.name):
                        continue
                    if _f.default is not _DC_MISSING:
                        setattr(entry, _f.name, _f.default)
                    elif _f.default_factory is not _DC_MISSING:
                        setattr(entry, _f.name, _f.default_factory())
            except TypeError:
                pass  # not a dataclass instance
            if self._store_entry_news_locked(sym, entry):
                loaded_news += 1
                backfilled_news += 1
            # Backfill new fields added after the entry was cached
            if not hasattr(entry, "news_is_breaking"):
                entry.news_is_breaking = False
            if not hasattr(entry, "daily_high"):
                entry.daily_high = 0.0
            if not hasattr(entry, "noise_titles"):
                entry.noise_titles = set()
            # Merge persistent noise store into the entry so re-evaluated
            # entries still respect previously marked noise headlines.
            _stored_noise = NoiseStore.get().titles(sym)
            if _stored_noise:
                entry.noise_titles = entry.noise_titles | _stored_noise
            if not hasattr(entry, "criteria_groups") or not entry.criteria_groups:
                from config import CRITERIA_ATTENTION, CRITERIA_EXECUTION
                from strategy.criteria import build_criteria_groups
                entry.criteria_groups = build_criteria_groups(
                    CRITERIA_ATTENTION + CRITERIA_EXECUTION
                )
            else:
                from config import CRITERIA_ATTENTION, CRITERIA_EXECUTION
                from strategy.criteria import sync_criteria_groups
                entry.criteria_groups = sync_criteria_groups(
                    entry.criteria_groups,
                    CRITERIA_ATTENTION + CRITERIA_EXECUTION,
                )
            # Evict entries with incompatible CompositeEvaluation (pickled before refactor)
            ev = getattr(entry, "evaluation", None)
            if ev is not None and not hasattr(ev, "stock_eval"):
                evicted += 1
                continue
            # Evict entries whose ScoringVariables lack fields added after pickling
            # (slots-based dataclass → missing slot raises AttributeError at runtime).
            if ev is not None and _has_stale_scoring_vars(ev):
                evicted += 1
                continue
            # Refresh threshold bounds from current config so cache-hit
            # displays reflect live config tweaks (lo/hi/cutoff/fn/weight).
            if ev is not None:
                _refresh_scoring_thresholds(ev)
                # Reset live/dynamic scoring state so stale qualities from a
                # prior session don't show through until the scorer ticks.
                _reset_live_state(ev)
            loaded += 1

        for sym, item in raw_pending.items():
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            headlines, ts = item[0], item[1]
            if now - ts > _TTL:
                evicted_pending += 1
                continue
            if headlines:
                self._pending_yt_news[sym] = (list(headlines), ts)
                loaded_pending += 1

        for sym, item in raw_news.items():
            if not isinstance(item, tuple) or len(item) < 5:
                continue
            articles, news_count, avg_sentiment, news_source, ts = item
            if now - ts > _TTL:
                evicted_news += 1
                continue
            if self._store_news_locked(sym, articles, news_count, avg_sentiment, news_source, updated_ts=ts):
                loaded_news += 1

        if evicted:
            self._dirty = True
        if evicted_pending:
            self._dirty = True
        if evicted_news:
            self._dirty = True
        if backfilled_news:
            self._dirty = True
        logger.debug(
            "Loaded ticker cache: %d symbols, %d pending-YT, %d news from %s (%d symbols + %d pending + %d news evicted as stale)",
            loaded, loaded_pending, loaded_news, self._path, evicted, evicted_pending, evicted_news,
        )

    def _save(self) -> None:
        """Write to disk.  Caller must hold self._lock."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "wb") as f:
                pickle.dump(
                    {
                        "entries": self._data,
                        "pending_yt_news": self._pending_yt_news,
                        "news_cache": self._news_cache,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as exc:
            logger.warning("Failed to save ticker cache: %s", exc)

    def snapshot_day(self, day: Optional[date] = None) -> Path:
        """Save a dated snapshot of today's entries for backtesting.

        Includes symbols whose first_seen or last_updated falls on *day*
        (defaults to today ET).  The snapshot is a standalone pickle that
        the backtest runner loads directly.
        """
        if day is None:
            day = datetime.now(tz=MarketClock.ET).date()

        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snap_path = _SNAPSHOT_DIR / f"{day.isoformat()}.pkl"

        with self._lock:
            day_entries: dict[str, tuple] = {}
            for sym, item in self._data.items():
                if len(item) < 3:
                    continue
                first_seen = datetime.fromtimestamp(float(item[2]), tz=timezone.utc)
                updated_ts = item[3] if len(item) >= 4 else item[2]
                last_updated = datetime.fromtimestamp(float(updated_ts), tz=timezone.utc)
                if (
                    first_seen.astimezone(MarketClock.ET).date() == day
                    or last_updated.astimezone(MarketClock.ET).date() == day
                ):
                    day_entries[sym] = item

        with open(snap_path, "wb") as f:
            pickle.dump({"entries": day_entries}, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Saved cache snapshot (%d symbols)", len(day_entries))
        return snap_path

    @staticmethod
    def load_snapshot(day: date) -> dict[str, tuple]:
        """Load a dated snapshot for backtesting. Returns entries dict."""
        snap_path = _SNAPSHOT_DIR / f"{day.isoformat()}.pkl"
        if not snap_path.exists():
            return {}
        with open(snap_path, "rb") as f:
            raw = _load_cache_pickle(f)
        if isinstance(raw, dict) and "entries" in raw:
            return raw["entries"]
        if isinstance(raw, dict):
            return raw
        return {}


# ---------------------------------------------------------------------------
# Persistent noise headline store
# ---------------------------------------------------------------------------

_NOISE_PATH = Path(__file__).parent / "logs" / "noise_titles.json"


class NoiseStore:
    """Persistent {symbol: [title, ...]} store for user-marked noise headlines.

    Survives across sessions and re-evaluations. Backed by a JSON file in
    logs/noise_titles.json.
    """

    _instance: Optional["NoiseStore"] = None
    _lock = threading.Lock()

    def __init__(self, path: Path = _NOISE_PATH) -> None:
        self._path = path
        self._data: dict[str, list[str]] = {}
        self._load()

    @classmethod
    def get(cls) -> "NoiseStore":
        """Singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        try:
            import json
            with open(self._path, "r") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._data = {k.upper(): list(v) for k, v in raw.items()}
            else:
                self._data = {}
        except Exception:
            logger.debug("Failed to load noise store from %s", self._path)
            self._data = {}

    def _save(self) -> None:
        import json
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def add(self, symbol: str, title: str) -> None:
        """Mark a headline as noise for a symbol."""
        sym = symbol.upper()
        titles = self._data.setdefault(sym, [])
        if title not in titles:
            titles.append(title)
            self._save()

    def titles(self, symbol: str) -> set[str]:
        """Get all noise titles for a symbol."""
        return set(self._data.get(symbol.upper(), []))

    def remove(self, symbol: str, title: str) -> None:
        """Un-mark a headline as noise."""
        sym = symbol.upper()
        titles = self._data.get(sym, [])
        if title in titles:
            titles.remove(title)
            if not titles:
                del self._data[sym]
            self._save()

    def clear_symbol(self, symbol: str) -> None:
        """Remove all noise marks for a symbol."""
        sym = symbol.upper()
        if sym in self._data:
            del self._data[sym]
            self._save()
