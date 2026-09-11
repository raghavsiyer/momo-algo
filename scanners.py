"""
scanners.py  —  Background scanner threads for automatic ticker discovery.

Three scanner input streams run in parallel during market hours:
  1. Momo scanner  — polls momoscreener.com for 5-min momentum gainers
  2. YouTube scanner — polls a YouTube live chat for ticker mentions and
     structured news headlines from trusted authors
  3. IBKR scanner — polls IBKR's native market scanner for top percent
     gainers and queues names as they enter the ranked list

A shared consumer thread dequeues symbols and runs them through the
evaluation pipeline.
"""

import logging
import queue
import re
import threading
import time as _time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from market_data.chat_logger import ChatShadowLogger
    from market_data.tradier_rest import HistoricalClient

from strategy.evaluator import StockEvaluator
from market_data.momo_client import MomoClient
from market_data.youtube_client import (
    YouTubeLiveChatClient, ChatNewsHeadline, extract_ticker_from_text,
)
from market_data.finbert_client import FinBERTClient
# Import-time cost is config only — torch/transformers load lazily on first use.
from market_data.modernbert_client import ModernBERTNewsClassifier, alpha_word_count
from cache import TickerCache, apply_cached_news
from chart_cache import load_candle_cache, load_timesales_cache
from pipeline import run_pipeline, validate_ticker, _refresh_cached_stage1_snapshot
from display import _print_scanner_hit, _time_ago, safe_print
from technical_analysis.patterns import latest_extended_session_start
from config import (
    _dim, _c, _b, _g, _r,
    SCANNER_MOMO_MIN_CHG, SCANNER_MOMO_HITS_PER_OUTPUT,
    SCANNER_IBKR_SCAN_CODE_PIN, SCANNER_IBKR_ROWS, SCANNER_IBKR_REAPPEAR_SECS,
    SCANNER_IBKR_DROP_SECS, scanner_ibkr_scan_spec,
    SCANNER_YT_INTERVAL,
    SCANNER_YT_MIN_MENTIONS, SCANNER_YT_NEWS_AUTHOR,
    SCANNER_YT_TRUSTED_AUTHORS, SCANNER_YT_AUTHOR_SCORE_THRESHOLD,
    SCANNER_YT_HEADLINE_BUZZWORDS,
    SCANNER_YT_MODERNBERT_MODE, SCANNER_YT_MODERNBERT_MIN_WORDS,
    SCANNER_YT_SYMBOL_RESOLVE_INTERVAL_S,
    NOISE_FILTER_YT_TICKERS,
    NEWS_NOISE_THRESHOLD,
    NOISE_FILTER_YT_HEADLINE,
    news_freshness_cutoffs,
    MarketClock,
)

logger = logging.getLogger(__name__)


def _get_cached_news(cache: TickerCache, sym: str):
    getter = getattr(cache, "get_news", None)
    if getter is None:
        return None
    return getter(sym)


def _cached_entry_is_stale_day(cache: TickerCache, sym: str) -> bool:
    """True when the cached entry was last written on a previous ET day.

    The seed path uses this to decide whether a Stage-1 quote refresh is
    needed before starting a scorer: DynamicScorer accumulates print sizes
    on top of ``entry.volume``, so a prior-day cached volume/RVOL base
    would be inflated from the first print. Same-day resumes skip the
    extra quote call.
    """
    try:
        upd_ts = cache.get_updated_ts(sym)
        if upd_ts is None:
            return True
        upd_day = (
            datetime.fromtimestamp(float(upd_ts), tz=timezone.utc)
            .astimezone(MarketClock.ET)
            .date()
        )
        return upd_day != MarketClock.now_et().date()
    except Exception:
        return True  # can't tell → refresh to be safe


def _load_cached_seed_market_data(sym: str):
    end_utc = datetime.now(tz=timezone.utc)
    start_utc = latest_extended_session_start(end_utc)
    return (
        load_candle_cache(sym, "1m", start_utc, end_utc),
        load_candle_cache(sym, "10s", start_utc, end_utc),
        load_timesales_cache(sym, start_utc, end_utc),
    )


def _dedup_momos(stocks: list) -> list:
    """Return *stocks* with only the first entry per symbol kept."""
    seen: set[str] = set()
    out: list = []
    for s in stocks:
        if s.symbol not in seen:
            seen.add(s.symbol)
            out.append(s)
    return out
_SOFT_FAIL_RE = re.compile(r"^rvol_1day")



def _is_soft_failure(reason: str) -> bool:
    """True when a pipeline fail_reason is transient.

    Price and symbol failures are structural for the session. A low RVOL
    reading can improve as the session progresses, so scanners may retry
    it later. Day change no longer appears here — the prefilter stopped
    rejecting on it (the scanner's changePercAbove floor and the live
    ATTENTION criterion carry that gate). Volume is a thesis/readiness
    input, not a retry signal by itself.
    """
    return bool(_SOFT_FAIL_RE.match(reason))


def _failures_are_soft(reasons: list[str] | tuple[str, ...] | None) -> bool:
    active = [r for r in (reasons or []) if r]
    return bool(active) and all(_is_soft_failure(r) for r in active)


# ---------------------------------------------------------------------------
# Momo scanner
# ---------------------------------------------------------------------------

def _momo_scanner(
    ticker_q: queue.Queue,
    stop_event: threading.Event,
    evaluator: StockEvaluator,
    cache: TickerCache,
    queued: set[str],
    client: MomoClient,
    paused: threading.Event,
    interval: list[int],
    ready: threading.Event,
    force_poll: threading.Event,
    pipeline_done: set,
    scanner_lock: threading.Lock,
    muted_tickers: set[str] | None = None,
    scorers: dict | None = None,
) -> None:
    """REST-polling momo scanner — polls the 5-min momentum endpoint every
    *interval[0]* seconds and queues symbols that are genuinely new.

    Novelty is tracked per server-assigned ``updated_at`` timestamp — each
    unique snapshot is a distinct momentum event. Every 2 new snapshots for
    a symbol produces 1 hit-summary print, so 6 raw hits → 3 outputs.
    Unevaluated tickers require 2 hits within *_CONFIRM_WINDOW* seconds
    before being queued for the pipeline.
    """
    logger.info("[scanner:momo] Started  %s",
                _dim(f"(polling, interval={interval[0]}s, min_chg={SCANNER_MOMO_MIN_CHG}%%)"))
    ready.set()
    seen_ts: dict[str, set] = {}     # sym → set of updated_at timestamps already processed
    new_hit_counts: dict[str, int] = {}  # sym → count of unseen hits accumulated
    first_seen: dict[str, float] = {}    # sym → monotonic time of first unconfirmed hit
    _CONFIRM_WINDOW = 10.0               # seconds — must get 2nd hit within this window to queue
    first_poll = True                    # first poll seeds seen_ts silently — no output
    while not stop_event.is_set():
        if paused.is_set():
            stop_event.wait(interval[0])
            continue
        try:
            raw_stocks = client.get_momos_raw(min_change_pct=SCANNER_MOMO_MIN_CHG)

            # Staleness guard: if the freshest updated_at in the batch
            # is from before today's premarket open, the screener is
            # serving stale data (e.g. yesterday's top gainers). Skip.
            if raw_stocks:
                _freshest = max(
                    (s.updated_at for s in raw_stocks if s.updated_at),
                    default=None,
                )
                if _freshest is not None:
                    _today_et = MarketClock.now_et().date()
                    _session_start = datetime(
                        _today_et.year, _today_et.month, _today_et.day,
                        MarketClock.EXT_OPEN.hour, MarketClock.EXT_OPEN.minute,
                        tzinfo=MarketClock.ET,
                    )
                    if _freshest < _session_start:
                        logger.debug(
                            "[scanner:momo] Stale data — freshest updated_at %s < session start %s, skipping",
                            _freshest, _session_start,
                        )
                        stop_event.wait(interval[0])
                        continue

            was_first_poll = first_poll
            if first_poll:
                # Seed seen_ts with all current snapshots so the backlog is ignored
                for s in raw_stocks:
                    sym = s.symbol.upper()
                    if s.updated_at:
                        seen_ts.setdefault(sym, set()).add(s.updated_at)
                first_poll = False
            # Update latest_momos cache (deduplicated, most recent per symbol)
            client.latest_momos = _dedup_momos(raw_stocks)
            if was_first_poll:
                continue

            now = _time.monotonic()

            # Count genuinely new snapshots per symbol (keyed by updated_at)
            new_counts: dict[str, int] = {}
            sym_map: dict[str, object] = {}
            for s in raw_stocks:
                sym = s.symbol.upper()
                if validate_ticker(sym) is None:
                    continue
                if s.change_pct < SCANNER_MOMO_MIN_CHG and s.change_5m < SCANNER_MOMO_MIN_CHG:
                    continue
                ts_key = s.updated_at  # server-assigned timestamp — unique per snapshot
                if ts_key not in seen_ts.get(sym, set()):
                    seen_ts.setdefault(sym, set()).add(ts_key)
                    new_counts[sym] = new_counts.get(sym, 0) + 1
                    sym_map[sym] = s  # keep most recent for price/change display

            for sym, n_new in new_counts.items():
                s = sym_map[sym]
                emit_hit = not (muted_tickers and sym in muted_tickers) and sym not in evaluator.watchlist_symbols

                # --- Unified confirmation window for ALL tickers ---
                # Accumulate hits; only act when 2+ arrive within _CONFIRM_WINDOW.
                total = new_hit_counts.get(sym, 0) + n_new
                new_hit_counts[sym] = total
                if sym not in first_seen:
                    first_seen[sym] = now
                    continue
                if now - first_seen[sym] > _CONFIRM_WINDOW:
                    # Window expired — reset with current batch
                    first_seen[sym] = now
                    new_hit_counts[sym] = n_new
                    continue
                if total < 2:
                    continue
                # Confirmed: 2+ hits within window — reset counters
                first_seen.pop(sym)
                new_hit_counts.pop(sym, None)

                # Print hit for any confirmed ticker
                if emit_hit:
                    _print_scanner_hit(sym, cache, source="momo", price=s.last_price, change_pct=s.change_pct)

                # Queue for pipeline if not already known / not actively scored.
                # Check + mark under ONE lock so two poll iterations can't
                # both see "unknown" and double-queue the same symbol.
                with scanner_lock:
                    _has_active_scorer = scorers is not None and sym in scorers
                    _is_known = sym in queued or (cache.has(sym) and _has_active_scorer)
                    if not _is_known:
                        queued.add(sym)
                if not _is_known:
                    ticker_q.put(("momo", sym))
                    logger.debug("[scanner:momo] Queued %s  %s",
                                sym, _dim(f"({s.change_pct:.1f}% chg, {total} hits)"))
        except Exception as exc:
            logger.warning("[scanner:momo] Poll error: %s", exc)
        force_poll.wait(interval[0])
        if force_poll.is_set():
            force_poll.clear()
        if stop_event.is_set():
            break
    logger.info("[scanner:momo] Stopped")


# ---------------------------------------------------------------------------
# IBKR scanner
# ---------------------------------------------------------------------------

class _ScanNoveltyTracker:
    """Track which symbols in a ranked scan snapshot are genuinely new.

    IBKR's scanner returns the full ranked list on every poll, so the event
    worth acting on is a symbol *entering* the list — not its continued
    presence. A symbol that drops off and stays off for *reappear_secs* is
    forgotten, so a second run at the top later in the session re-emits.

    Unlike the momo scanner there is no 2-hit confirmation window: momo
    serves noisy singleton snapshots, whereas a seat in IBKR's ranked
    top-N is itself the confirmation.
    """

    def __init__(
        self, reappear_secs: float, drop_secs: float = SCANNER_IBKR_DROP_SECS,
    ) -> None:
        self.reappear_secs = float(reappear_secs)
        self.drop_secs = float(drop_secs)
        self._last_seen: dict[str, float] = {}

    def update(
        self, symbols: list[str], now: float, first_poll: bool = False,
    ) -> list[str]:
        """Age out stale entries, then return the new entrants in *symbols*.

        *symbols* is expected in rank order and the result preserves it.
        The first poll seeds silently — the standing list at startup is a
        backlog, not a set of events (mirrors the momo scanner's seed poll).
        """
        for sym, last in list(self._last_seen.items()):
            if now - last > self.reappear_secs:
                del self._last_seen[sym]
        new: list[str] = []
        for sym in symbols:
            if not first_poll and sym not in self._last_seen:
                new.append(sym)
            self._last_seen[sym] = now
        return new

    def dropped(self, watched: set[str], now: float) -> list[str]:
        """Symbols in *watched* absent from the list for ``drop_secs``.

        A name the tracker has never seen is not a drop — it may have been
        discovered by the momo or YouTube scanner and never have held a seat
        in IBKR's ranked list at all. Only a symbol this tracker actually
        saw, and has now stopped seeing, counts.
        """
        return [
            sym for sym in watched
            if (last := self._last_seen.get(sym)) is not None
            and now - last > self.drop_secs
        ]

    def forget(self, symbol: str) -> None:
        """Drop *symbol*'s last-seen mark so it re-emits as new on return.

        Called after a drop is acted on: without this the symbol would keep
        matching :meth:`dropped` every poll until ``reappear_secs`` expires,
        re-firing the fade on a name that is already faded.
        """
        self._last_seen.pop(symbol, None)

    def reset(self) -> None:
        """Forget every symbol — used when the scan code changes.

        At 09:30 / 16:00 ET the session-aware scan code swaps and the whole
        ranked list is replaced, so every incumbent would look dropped at
        once. Clearing the marks makes the post-swap poll behave like a
        first poll: nothing is a drop, and the new list seeds silently.
        """
        self._last_seen.clear()


_IBKR_SCAN_BACKOFF_CAP_SECS = 60  # ceiling for the failure backoff wait


def _ibkr_scanner(
    ticker_q: queue.Queue,
    stop_event: threading.Event,
    evaluator: StockEvaluator,
    cache: TickerCache,
    queued: set[str],
    client,
    paused: threading.Event,
    interval: list[int],
    ready: threading.Event,
    force_poll: threading.Event,
    pipeline_done: set,
    scanner_lock: threading.Lock,
    muted_tickers: set[str] | None = None,
    scorers: dict | None = None,
    on_scan: Callable | None = None,
    on_drop: Callable | None = None,
) -> None:
    """IBKR market-scanner feed — runs a one-shot top-gainers scan every
    *interval[0]* seconds and queues symbols as they enter the ranked list.

    A scan failure (Gateway down, missing scanner permission, bad scan code)
    is a warn-and-retry, never a thread death: the next poll picks up as
    soon as the Gateway is back.

    *on_scan* is called after every successful scan so a live view of the
    ranked list can repaint at the feed's own cadence.

    *on_drop* is called with the symbols that have been off the ranked list
    for ``SCANNER_IBKR_DROP_SECS`` while still being watched, so their
    scorers can be stopped. Leaving the list is the mirror of entering it:
    the name is no longer among the session's top movers, so the stream it
    holds is better spent elsewhere. Re-entry re-queues it normally.
    """
    from market_data.ibkr_rest import release_ib

    _scan_desc = f"{SCANNER_IBKR_SCAN_CODE_PIN} (pinned)" if SCANNER_IBKR_SCAN_CODE_PIN else "session-aware"
    logger.info("[scanner:ibkr] Started  %s",
                _dim(f"(scan={_scan_desc}, interval={interval[0]}s, rows={SCANNER_IBKR_ROWS})"))
    ready.set()
    tracker = _ScanNoveltyTracker(SCANNER_IBKR_REAPPEAR_SECS, SCANNER_IBKR_DROP_SECS)
    first_poll = True  # first poll seeds the tracker silently — no output
    last_scan_code = scanner_ibkr_scan_spec()[0]
    consec_failures = 0  # doubles the retry wait while the Gateway is down
    try:
        while not stop_event.is_set():
            if paused.is_set():
                stop_event.wait(interval[0])
                continue
            wait_secs = interval[0]
            try:
                # The scan code swaps at the RTH boundaries (09:30 / 16:00
                # ET) and the ranked list is replaced wholesale, so every
                # incumbent would look dropped at once. Re-seed instead.
                scan_code = scanner_ibkr_scan_spec()[0]
                if scan_code != last_scan_code:
                    logger.info(
                        "[scanner:ibkr] Scan code %s → %s — re-seeding tracker",
                        last_scan_code, scan_code,
                    )
                    tracker.reset()
                    last_scan_code = scan_code
                    first_poll = True

                rows = client.scan_top_gainers()
                if on_scan is not None:
                    # A live view is a spectator — never let one fault the feed.
                    try:
                        on_scan()
                    except Exception:
                        logger.debug("[scanner:ibkr] on_scan callback failed", exc_info=True)
                ranks = {r.symbol: r.rank for r in rows}
                symbols = [r.symbol for r in rows if validate_ticker(r.symbol)]
                # One clock read for the whole poll: re-reading it for the
                # drop check would age the marks this very update just wrote.
                poll_ts = _time.monotonic()
                new_syms = tracker.update(
                    symbols, poll_ts, first_poll=first_poll,
                )
                first_poll = False

                for sym in new_syms:
                    if not (muted_tickers and sym in muted_tickers) and sym not in evaluator.watchlist_symbols:
                        _print_scanner_hit(sym, cache, source="ibkr")

                    # Queue for pipeline if not already known / not actively
                    # scored. Check + mark under ONE lock so a concurrent momo
                    # hit can't double-queue the same symbol.
                    with scanner_lock:
                        _has_active_scorer = scorers is not None and sym in scorers
                        _is_known = sym in queued or (cache.has(sym) and _has_active_scorer)
                        if not _is_known:
                            queued.add(sym)
                    if not _is_known:
                        ticker_q.put(("ibkr", sym))
                        logger.debug("[scanner:ibkr] Queued %s  %s",
                                     sym, _dim(f"(rank {ranks.get(sym, '?')})"))

                # --- Departures: watched names that lost their seat ---
                # Only names with a live scorer are candidates — an entry
                # with no scorer is already faded or was never watched.
                if on_drop is not None and scorers:
                    watched = set(scorers) & evaluator.watchlist_symbols
                    gone = tracker.dropped(watched, poll_ts)
                    for sym in gone:
                        # Forget first: a failed callback must not leave the
                        # symbol re-firing on every subsequent poll.
                        tracker.forget(sym)
                    if gone:
                        try:
                            on_drop(gone)
                        except Exception:
                            logger.debug(
                                "[scanner:ibkr] on_drop callback failed", exc_info=True,
                            )
                consec_failures = 0
            except Exception as exc:
                # At a seconds-scale interval a down Gateway would otherwise be
                # hammered (and the log flooded) — back off exponentially and
                # only warn on the first failure and once the cap is reached.
                consec_failures += 1
                wait_secs = min(interval[0] * (2 ** consec_failures),
                                _IBKR_SCAN_BACKOFF_CAP_SECS)
                if consec_failures == 1 or wait_secs >= _IBKR_SCAN_BACKOFF_CAP_SECS:
                    logger.warning("[scanner:ibkr] Poll error (%d consecutive, next retry in %ds): %s",
                                   consec_failures, wait_secs, exc)
                else:
                    logger.debug("[scanner:ibkr] Poll error (%d consecutive, next retry in %ds): %s",
                                 consec_failures, wait_secs, exc)
            force_poll.wait(wait_secs)
            if force_poll.is_set():
                force_poll.clear()
            if stop_event.is_set():
                break
    finally:
        # This thread holds its own IB connection; the Gateway caps
        # concurrent connections at 32 and only frees one when the
        # socket closes.
        release_ib()
    logger.info("[scanner:ibkr] Stopped")


# ---------------------------------------------------------------------------
# YouTube scanner — helpers
# ---------------------------------------------------------------------------

def _first_meaningful_word(text: str) -> str:
    """First word of *text* that could name a company or symbol.

    Strips a leading ``$``, splits on the usual headline punctuation, and
    drops non-alphabetic tokens (e.g. a bare ``-``) plus
    ``SCANNER_YT_HEADLINE_BUZZWORDS`` ("BREAKING", "Alert", …).  Returns
    ``""`` when nothing survives.
    """
    words = re.split(r"[\s:,!?]+", text.lstrip("$"))
    words = [w for w in words if w.isalpha() and w.lower() not in SCANNER_YT_HEADLINE_BUZZWORDS]
    return words[0] if words else ""


class RateLimitedSymbolResolver:
    """Paced, cached wrapper around IBKR ``reqMatchingSymbols``.

    IBKR paces matching-symbol requests at roughly one per second, so calls
    are serialised behind a lock with ``min_interval_s`` spacing and every
    answer — including "no match" — is memoised in a bounded LRU.  Lookup
    failures collapse to ``None``; this is a rare best-effort path and must
    never propagate into the scanner loop.

    A ``None`` *ibkr_client* (IBKR disabled, or the scanner client failed to
    construct) makes every resolve a no-op ``None``.  There is deliberately
    no Tradier fallback: mixing resolvers would make the shadow-mode
    ticker-method statistics meaningless.
    """

    def __init__(
        self,
        ibkr_client,
        min_interval_s: float = SCANNER_YT_SYMBOL_RESOLVE_INTERVAL_S,
        cache_size: int = 512,
        *,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._client = ibkr_client
        self._min_interval = float(min_interval_s)
        self._cache_size = int(cache_size)
        self._cache: "OrderedDict[str, Optional[str]]" = OrderedDict()
        self._clock = clock or _time.monotonic
        self._sleep = sleeper or _time.sleep
        self._last_call: Optional[float] = None
        self._lock = threading.Lock()

    def resolve(self, query: str) -> Optional[str]:
        """Resolve *query* (symbol fragment or company name) to a ticker."""
        key = (query or "").strip().lower()
        if not key or self._client is None:
            return None

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            if self._last_call is not None:
                wait = self._min_interval - (self._clock() - self._last_call)
                if wait > 0:
                    self._sleep(wait)
            try:
                sym = self._client.search_symbol(query)
            except Exception as exc:
                logger.debug("[scanner:youtube] symbol resolve failed for %r: %s", query, exc)
                sym = None
            finally:
                self._last_call = self._clock()

            sym = sym.upper() if sym else None
            self._cache[key] = sym
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return sym


def _mb_headline_text(text: str, method: str, remainder: str) -> str:
    """Headline body for a ModernBERT-detected news post.

    For the structured leading forms the ticker (and any ``(note)`` /
    ``-`` delimiter) is stripped so the text matches what the Clementine
    path produces.  Anything else keeps the message verbatim.
    """
    if method in ("cashtag", "prefix"):
        rest = remainder.lstrip()
        if rest.startswith("("):
            close = rest.find(")")
            if close != -1:
                rest = rest[close + 1:].lstrip()
        rest = rest.lstrip("-").strip()
        if rest:
            return rest
    return text.strip()


def _yt_shadow_classify(
    messages: list,
    clementine_headlines: list[ChatNewsHeadline],
    news_clf: Optional[ModernBERTNewsClassifier],
    chat_log: "Optional[ChatShadowLogger]",
    symbol_resolver: Optional[RateLimitedSymbolResolver],
    first_poll: bool,
) -> list[ChatNewsHeadline]:
    """Classify a poll's chat messages and log the raw comparison rows.

    Writes one JSONL record per *raw* message (classified or not) and
    returns the headlines ModernBERT found that the Clementine matcher did
    **not** — deduped against it by message id.

    This function never touches ``ticker_q``, ``cache`` or ``queued``: in
    shadow mode the caller discards the return value, so a shadow bug can't
    reach the trading path.  It also swallows every exception — a broken
    classifier must not kill the scanner loop.
    """
    if not messages:
        return []
    try:
        # Clementine headlines carry no message id; match them back to
        # messages the same way the queueing block does.
        hl_by_key: dict[tuple[str, datetime], ChatNewsHeadline] = {}
        for hl in clementine_headlines:
            hl_by_key.setdefault((hl.author, hl.timestamp), hl)

        # Only classify messages with enough substance to be a headline.
        # The first poll can return a 100+ message backlog (≈30-60 s of CPU)
        # that is stale by definition — log it, don't score it.
        candidates: list[int] = []
        if news_clf is not None and not first_poll:
            candidates = [
                i for i, m in enumerate(messages)
                if alpha_word_count(m.text) >= SCANNER_YT_MODERNBERT_MIN_WORDS
            ]

        verdicts: list = [None] * len(messages)
        batch_ms = 0.0
        if candidates:
            t0 = _time.perf_counter()
            results = news_clf.classify_batch([messages[i].text for i in candidates])
            batch_ms = round((_time.perf_counter() - t0) * 1000.0, 1)
            for pos, verdict in zip(candidates, results):
                verdicts[pos] = verdict

        trusted = set(SCANNER_YT_TRUSTED_AUTHORS) | {SCANNER_YT_NEWS_AUTHOR}
        records: list[dict] = []
        detected: list[ChatNewsHeadline] = []
        n_news = n_agree = 0

        for i, msg in enumerate(messages):
            hl = hl_by_key.get((msg.author, msg.timestamp))
            verdict = verdicts[i]
            is_news = bool(verdict and verdict.is_news)
            n_news += int(is_news)
            n_agree += int(is_news and hl is not None)

            ticker = method = None
            remainder = ""
            if is_news:
                ticker, method, remainder = extract_ticker_from_text(msg.text)
                if ticker is None:
                    method = "none"
                    if symbol_resolver is not None:
                        word = _first_meaningful_word(msg.text.strip())
                        resolved = symbol_resolver.resolve(word) if word else None
                        # Same 3-4 char band as the regex extraction: chat-
                        # sourced symbols outside it are never accepted.
                        if resolved and 3 <= len(resolved) <= 4:
                            ticker, method = resolved, "ibkr_search"
                # A detection the Clementine matcher already produced is not
                # a new headline; a detection with no ticker can't be traded.
                if ticker and hl is None and validate_ticker(ticker):
                    detected.append(ChatNewsHeadline(
                        ticker=ticker.upper(),
                        headline=_mb_headline_text(msg.text, method, remainder),
                        author=msg.author,
                        timestamp=msg.timestamp,
                        # Clementine / trusted authors keep their bypass; a
                        # random chatter still has to clear the noise gate.
                        bypass_noise_filter=msg.author in trusted,
                    ))

            records.append({
                "ts": msg.timestamp.isoformat(),
                "message_id": msg.message_id,
                "author": msg.author,
                "text": msg.text,
                "clementine_hit": hl is not None,
                "clementine_ticker": hl.ticker if hl else None,
                "mb_evaluated": verdict is not None,
                "mb_is_news": is_news if verdict is not None else None,
                "mb_score": verdict.score if verdict is not None else None,
                "mb_ticker": ticker,
                "mb_ticker_method": method,
                "batch_ms": batch_ms,
                "model": news_clf.model if news_clf is not None else None,
                "threshold": news_clf.threshold if news_clf is not None else None,
            })

        if chat_log is not None:
            chat_log.log_records(records)

        logger.debug(
            "[scanner:youtube] shadow: %d msgs, %d news-flagged, "
            "%d agree-clementine, batch=%.0fms",
            len(messages), n_news, n_agree, batch_ms,
        )
        return detected
    except Exception as exc:
        logger.debug("[scanner:youtube] shadow classification failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# YouTube scanner
# ---------------------------------------------------------------------------

def _youtube_scanner(
    ticker_q: queue.Queue,
    stop_event: threading.Event,
    evaluator: StockEvaluator,
    cache: TickerCache,
    queued: set[str],
    paused: threading.Event,
    client: "YouTubeLiveChatClient",
    force_poll: threading.Event,
    finbert: FinBERTClient,
    tradier: "HistoricalClient",
    ready: threading.Event,
    yt_interval: list,
    scanner_lock: threading.Lock,
    muted_tickers: set[str] | None = None,
    *,
    news_clf: Optional[ModernBERTNewsClassifier] = None,
    chat_log: "Optional[ChatShadowLogger]" = None,
    symbol_resolver: Optional[RateLimitedSymbolResolver] = None,
) -> None:
    """Poll YouTube live chat for ticker mentions and news headlines.

    Regular chat messages are scanned for ticker mentions (existing logic).
    Messages from the designated news author (``SCANNER_YT_NEWS_AUTHOR``)
    are parsed for structured ``TICKER - headline`` news and stored in the
    cache-backed pending-news store for consumption by the pipeline.
    Trusted authors (``SCANNER_YT_TRUSTED_AUTHORS``) have every message
    scored by FinBERT; if the score exceeds
    ``SCANNER_YT_AUTHOR_THRESH`` the first word is looked up on
    Tradier as a symbol/company name and queued if a match is found.

    When *news_clf* is supplied, every message is also run through the
    ModernBERT zero-shot news detector.  Under
    ``SCANNER_YT_MODERNBERT_MODE == "shadow"`` (the default) its verdicts
    are logged and otherwise discarded — the trading path is byte-for-byte
    what it would be without the classifier.  Under ``"active"`` its
    detections are merged into the news headlines before queueing.
    """
    _trusted_authors = set(SCANNER_YT_TRUSTED_AUTHORS)
    _mb_active = SCANNER_YT_MODERNBERT_MODE == "active" and news_clf is not None
    logger.info("[scanner:youtube] Started  %s",
                _dim(f"(video={client.video_id}, interval={SCANNER_YT_INTERVAL}s, min_mentions={SCANNER_YT_MIN_MENTIONS})"))
    first_poll = True
    ready.set()
    while not stop_event.is_set():
        if paused.is_set():
            stop_event.wait(SCANNER_YT_INTERVAL)
            continue
        try:
            messages = client.poll()
            if messages:
                # --- News headlines (designated author only) — process FIRST
                # so pending cache headlines are populated before ticker mentions are queued.
                # This prevents the consumer from running the pipeline before
                # the headline is available (race condition).
                headlines = YouTubeLiveChatClient.extract_news_headlines(
                    messages, author=SCANNER_YT_NEWS_AUTHOR,
                )
                # Active mode: ModernBERT detections from *any* author join
                # the news path here, before anything is queued.  The
                # Clementine matcher stays in the union as a zero-cost,
                # high-precision fast path.  (Shadow mode runs the same
                # helper after the whole block and discards its result.)
                if _mb_active:
                    headlines = headlines + _yt_shadow_classify(
                        messages, headlines, news_clf, chat_log,
                        symbol_resolver, first_poll,
                    )
                news_syms: set[str] = set()
                # Track message IDs that produced headlines so trusted-author
                # scoring skips them. Build directly from author + timestamp
                # match — O(M) instead of O(H×M).
                _news_author_ts: set[tuple[str, datetime]] = {
                    (hl.author, hl.timestamp) for hl in headlines
                }
                consumed_msg_ids: set[str] = {
                    m.message_id for m in messages
                    if (m.author, m.timestamp) in _news_author_ts
                } if _news_author_ts else set()
                for hl in headlines:
                    sym = hl.ticker.upper()
                    cache.add_pending_yt_news(sym, hl)
                    logger.debug(
                        "[scanner:youtube] News headline for %s: %s",
                        sym, _dim(f'"{hl.headline}"'),
                    )
                    # Always queue — consumer handles both fresh and
                    # already-evaluated tickers differently.
                    if validate_ticker(sym):
                        ticker_q.put(("youtube-news", sym))
                        with scanner_lock:
                            if sym not in queued:
                                queued.add(sym)
                        news_syms.add(sym)
                        logger.debug("[scanner:youtube] Queued %s  %s", sym, _dim("(news headline)"))

                # --- Trusted-author FinBERT scoring ---
                # For messages from SCANNER_YT_TRUSTED_AUTHORS that were NOT
                # already caught by the structured news headline path, score
                # the text with FinBERT. If the score is above the threshold,
                # extract the first word and resolve it to a ticker via Tradier
                # search (symbol or company name). Skip @-replies and messages
                # already handled as structured headlines.
                headline_syms_lower = {s.lower() for s in news_syms}
                for msg in messages:
                    if msg.author not in _trusted_authors:
                        continue
                    if msg.message_id in consumed_msg_ids:
                        continue
                    text = msg.text.strip()
                    if not text or text.startswith("@"):
                        continue
                    # Strip leading buzzwords then extract first meaningful word.
                    first_word = _first_meaningful_word(text)
                    if not first_word or first_word.lower() in headline_syms_lower:
                        continue
                    # Quick word count without allocating a full list
                    if text.count(' ') < 8:
                        continue
                    try:
                        score = finbert.score_headline("", text)
                    except Exception as exc:
                        logger.debug("[scanner:youtube] FinBERT error: %s", exc)
                        continue
                    if score is None or score < SCANNER_YT_AUTHOR_SCORE_THRESHOLD:
                        continue
                    # Resolve first word via Tradier search
                    try:
                        sym = tradier.search_symbol(first_word)
                    except Exception as exc:
                        logger.debug("[scanner:youtube] Tradier search error for %r: %s", first_word, exc)
                        continue
                    # 3-4 char band: chat-sourced symbols outside it are never
                    # accepted (matches the extraction regexes).
                    if not sym or not (3 <= len(sym) <= 4) or validate_ticker(sym) is None:
                        continue
                    sym = sym.upper()
                    if sym in news_syms or cache.has(sym) or sym in queued:
                        continue
                    with scanner_lock:
                        if sym in queued:
                            continue
                        queued.add(sym)
                    ticker_q.put(("youtube", sym))
                    logger.debug(
                        "[scanner:youtube] Queued %s via author-sentiment  %s",
                        sym, _dim(f"(author={msg.author} score={score:.2f} query={first_word!r})"),
                    )

                # --- Chat ticker mentions (all users) ---
                # Skip on the very first poll to avoid seeding stale mentions.
                if not first_poll:
                    ticker_counts = YouTubeLiveChatClient.extract_tickers(messages)
                    for sym, count in ticker_counts.items():
                        if count < SCANNER_YT_MIN_MENTIONS:
                            continue
                        if sym in news_syms:
                            continue
                        if validate_ticker(sym) is None:
                            continue
                        with scanner_lock:
                            if sym in queued:
                                continue
                            queued.add(sym)
                        ticker_q.put(("youtube-chat", sym))
                        logger.debug(
                            "[scanner:youtube] Queued %s via chat mentions (%d)",
                            sym, count,
                        )

                # --- ModernBERT shadow pass ---
                # Deliberately last: classification runs ~250 ms/message on
                # CPU (≈1-4 s for a typical 5-15 message poll) and must never
                # sit in front of a queue put.  The return value is
                # discarded — shadow mode queues nothing.
                if not _mb_active and (news_clf is not None or chat_log is not None):
                    _yt_shadow_classify(
                        messages, headlines, news_clf, chat_log,
                        symbol_resolver, first_poll,
                    )

                first_poll = False
        except Exception as exc:
            logger.warning("[scanner:youtube] Poll error: %s", exc)
        # Use YouTube's own recommended timeout (from timedContinuationData)
        # so we poll as fast as the server allows rather than a fixed interval.
        # Fall back to SCANNER_YT_INTERVAL if no hint has been received yet.
        fixed = yt_interval[0]
        if fixed is not None:
            timeout_s = float(fixed)
        else:
            timeout_s = (client.poll_timeout_ms / 1000.0
                         if client.poll_timeout_ms > 0 else SCANNER_YT_INTERVAL)
        # Wake early if force_poll is set (yt poll command), otherwise
        # honour the server hint / fallback interval.
        force_poll.wait(timeout_s)
        if force_poll.is_set():
            force_poll.clear()
        if stop_event.is_set():
            break
    logger.info("[scanner:youtube] Stopped")


# ---------------------------------------------------------------------------
# YouTube news injection
# ---------------------------------------------------------------------------

def _inject_yt_news(
    sym: str,
    headlines: list[ChatNewsHeadline],
    evaluator: StockEvaluator,
    cache: TickerCache,
) -> None:
    """Retroactively inject YouTube news into an already-evaluated ticker.

    Scores the headline with FinBERT, applies the noise filter, then
    updates the catalyst group's breaking_news quality variable so the
    confidence score updates in place.
    """
    from market_data.finnhub_client import FinnhubNewsArticle

    entry = next((e for e in evaluator.watchlist if e.symbol == sym), None)
    if entry is None:
        return

    for hl in headlines:
        score = evaluator._finbert.score_headline(sym, headline=hl.headline)

        # Noise filter — match the threshold used in the main pipeline
        if (
            NOISE_FILTER_YT_HEADLINE
            and not hl.bypass_noise_filter
            and score is not None
            and -NEWS_NOISE_THRESHOLD < score < NEWS_NOISE_THRESHOLD
        ):
            logger.debug(
                "[%s] YouTube inject noise-filtered (score=%.2f): \"%s\"",
                sym, score, hl.headline,
            )
            continue

        art = FinnhubNewsArticle(
            symbol=sym, title=hl.headline, source="youtube-chat",
            url="", published_at=hl.timestamp, summary="",
        )
        # Update entry in place
        _session_cutoff, _breaking_cutoff, _ = news_freshness_cutoffs()
        _is_fresh = hl.timestamp >= _session_cutoff
        _is_breaking = hl.timestamp >= _breaking_cutoff
        entry.news_articles = [art]
        entry.news_count_last_hour = 1
        entry.news_is_fresh = _is_fresh
        entry.news_is_breaking = _is_breaking
        entry.avg_sentiment = score
        entry.news_source = "youtube"

        # Update the breaking_news quality variable
        old_eval = entry.eval_confidence
        if entry.evaluation:
            entry.evaluation.update_news(score, 1, "youtube")

        # Refresh cache
        cached = cache.get(sym)
        if cached:
            cache.put(sym, entry, cached[1])
            cache.flush()
        logger.debug(
            "[scanner:consumer] Injected YT news for %s: \"%s\" %s",
            sym, hl.headline,
            _dim(f"(sentiment={f'{score:.2f}' if score is not None else 'N/A'}, "
                 f"eval={entry.eval_confidence:.1f})"),
        )
        if entry.eval_confidence != old_eval:
            _ts = _dim(MarketClock.now_et().astimezone().strftime("%H:%M:%S"))
            from config import sentiment_is_bullish, sentiment_is_bearish
            _hl_color = (_g if sentiment_is_bullish(score) else _r if sentiment_is_bearish(score) else _dim) if score is not None else _dim
            safe_print(
                f"\n  {_ts}  {_c('⟐')}  {_b(sym)}  {_dim('(youtube)')}"
                f"  {_hl_color(f'\"' + hl.headline + f'\"')}"
                f"  {_dim('→ eval')} {entry.eval_confidence:.1f}"
            )
        break  # use first non-noise headline


# ---------------------------------------------------------------------------
# Scanner consumer
# ---------------------------------------------------------------------------

def _scanner_consumer(
    ticker_q: queue.Queue,
    stop_event: threading.Event,
    evaluator: StockEvaluator,
    tradier: "HistoricalClient",
    scorers: dict,
    cache: TickerCache,
    queued: set[str],
    ready: threading.Event | None = None,
    pending_momo_hits: dict | None = None,  # deprecated — kept for API compat
    pipeline_done: set | None = None,
    scanner_lock: threading.Lock | None = None,
    news_refreshed: set | None = None,
    muted_tickers: set[str] | None = None,
    faded: set | None = None,
    on_scorer_update: Callable | None = None,
) -> None:
    """Consume tickers from scanner queue and run them through the pipeline."""
    logger.info("[scanner:consumer] Started")
    if ready is not None:
        ready.set()
    while not stop_event.is_set():
        try:
            source, sym = ticker_q.get(timeout=1.0)
        except queue.Empty:
            continue

        # --- Already failed — show brief hit and skip ---
        if pipeline_done is not None and scanner_lock is not None:
            with scanner_lock:
                _already_done = sym in pipeline_done
            if _already_done:
                _quiet = muted_tickers is not None and sym in muted_tickers
                if not _quiet:
                    _print_scanner_hit(sym, cache, source=source)
                continue

        # --- Late YouTube news for an already-evaluated ticker ---
        if source == "youtube-news" and cache.has(sym):
            _quiet = muted_tickers is not None and sym in muted_tickers
            if not _quiet and sym not in evaluator.watchlist_symbols:
                _print_scanner_hit(sym, cache, source="youtube")
            headlines = cache.pop_pending_yt_news(sym)
            if headlines:
                if sym in evaluator.watchlist_symbols:
                    # Already on watchlist — inject news into existing entry
                    try:
                        _inject_yt_news(sym, headlines, evaluator, cache)
                    except Exception as exc:
                        logger.warning("[scanner:consumer] News injection error for %s: %s", sym, exc)
                else:
                    # Failed eval earlier (not on watchlist) — re-run pipeline
                    # with the headline so it gets a fair evaluation with news.
                    try:
                        result = run_pipeline(
                            sym, evaluator, tradier,
                            yt_headlines=headlines,
                            quiet=_quiet,
                            allow_manual_rating=False,
                            cached_news=_get_cached_news(cache, sym),
                            on_scorer_update=on_scorer_update,
                        )
                        if result is not None:
                            cache.put(sym, result.entry, result.rmap)
                            cache.flush()
                            if result.scorer is not None:
                                old = scorers.pop(sym, None)
                                if old:
                                    old.stop()
                                scorers[sym] = result.scorer
                            if faded is not None:
                                faded.discard(sym)
                            logger.debug("[scanner:consumer] Re-evaluated %s with YT news", sym)
                    except Exception as exc:
                        logger.warning("[scanner:consumer] Re-eval with YT news error for %s: %s", sym, exc)
            continue

        # Double-check cache (another source may have evaluated it while queued)
        if cache.has(sym):
            # For seeded tickers, load cached entry onto watchlist + start scorer
            if source == "seed":
                cached = cache.get(sym)
                if cached:
                    _entry, _rmap = cached
                    if apply_cached_news(_entry, _get_cached_news(cache, sym)):
                        cache.put(sym, _entry, _rmap)
                        cache.flush()
                    if sym not in evaluator.watchlist_symbols:
                        # Stage-1 refresh for stale-day entries (fresh
                        # price/volume/RVOL via the active provider) BEFORE
                        # the scorer reads entry.avg_volume / accumulates
                        # entry.volume, and before the sort key is computed.
                        if _cached_entry_is_stale_day(cache, sym):
                            _refresh_cached_stage1_snapshot(sym, _entry, evaluator)
                        # Build-then-rebind (never mutate in place): the
                        # toolbar render thread iterates the watchlist.
                        _new_wl = [e for e in evaluator.watchlist if e.symbol != sym]
                        _new_wl.append(_entry)
                        _new_wl.sort(key=lambda e: e.watchlist_sort_score, reverse=True)
                        evaluator.watchlist = _new_wl
                        evaluator.watchlist_symbols.add(sym)
                        from strategy.confidence_scorer import DynamicScorer
                        from market_data.provider import get_historical_client as _get_hist
                        try:
                            _df_1m_seed, _df_10s_seed, _df_timesales_seed = _load_cached_seed_market_data(sym)
                            logger.debug(
                                "[scanner:consumer] Chart-cache seed for %s: 1m=%d 10s=%d t&s=%d",
                                sym, len(_df_1m_seed), len(_df_10s_seed), len(_df_timesales_seed),
                            )
                            _hist = _get_hist()
                            _df_daily = _hist.get_candles_daily(sym, days_back=60)
                            from technical_analysis.indicators import build_daily as _build_daily
                            _df_daily = _build_daily(_df_daily) if not _df_daily.empty else _df_daily
                            _daily_candles = DynamicScorer._df_to_candles(_df_daily, "daily") if not _df_daily.empty else []
                            _scorer = DynamicScorer(_entry, _rmap, avg_daily_volume=_entry.avg_volume,
                                                   daily_candles=_daily_candles,
                                                   seed_1min_session=_df_1m_seed,
                                                   seed_10sec=_df_10s_seed,
                                                   seed_timesales=_df_timesales_seed,
                                                   on_update=on_scorer_update)
                            _scorer.start()
                            old = scorers.pop(sym, None)
                            if old:
                                old.stop()
                            scorers[sym] = _scorer
                        except Exception as exc:
                            logger.debug("[scanner:consumer] Scorer start error for cached %s: %s", sym, exc)
                        logger.debug("[scanner:consumer] Loaded cached %s onto watchlist (seed)", sym)
                continue

            # Scanner re-hit: re-run pipeline with fresh quote/news but cached
            # float. Only if not already being actively watched (scorer running).
            if source in ("momo", "ibkr") and sym not in scorers:
                _cached = cache.get(sym)
                if _cached:
                    _old_entry, _ = _cached
                    _cached_float = _old_entry.float_shares_m
                    _cached_news = _get_cached_news(cache, sym)
                    _quiet = muted_tickers is not None and sym in muted_tickers
                    headlines = cache.get_pending_yt_news(sym)
                    try:
                        result = run_pipeline(
                            sym, evaluator, tradier,
                            yt_headlines=headlines,
                            quiet=_quiet,
                            cached_float=_cached_float,
                            cached_news=_cached_news,
                            allow_manual_rating=False,
                            on_scorer_update=on_scorer_update,
                        )
                        if result is not None:
                            apply_cached_news(result.entry, _cached_news)
                            cache.pop_pending_yt_news(sym)
                            cache.put(sym, result.entry, result.rmap)
                            cache.flush()
                            if result.scorer is not None:
                                old = scorers.pop(sym, None)
                                if old:
                                    old.stop()
                                scorers[sym] = result.scorer
                            if faded is not None:
                                faded.discard(sym)
                            logger.debug("[scanner:consumer] Re-evaluated %s (%s re-hit, cached float=%.2fM)", sym, source, _cached_float)
                    except Exception as exc:
                        logger.warning("[scanner:consumer] Re-eval error for %s: %s", sym, exc)
                continue

            # Refresh stale news (1+ day old) once per session
            if news_refreshed is not None and sym not in news_refreshed:
                cached = cache.get(sym)
                if cached:
                    _entry, _rmap = cached
                    if apply_cached_news(_entry, _get_cached_news(cache, sym)):
                        cache.put(sym, _entry, _rmap)
                        cache.flush()
                    _news_stale = False
                    if _entry.news_articles:
                        _pub = _entry.news_articles[0].published_at
                        if _pub.tzinfo is None:
                            from datetime import timezone as _tz
                            _pub = _pub.replace(tzinfo=_tz.utc)
                        _, _, _stale_cut = news_freshness_cutoffs()
                        _news_stale = _pub < _stale_cut
                    else:
                        _news_stale = True
                    if _news_stale:
                        news_refreshed.add(sym)
                        try:
                            if evaluator.refresh_news(_entry):
                                cache.put(sym, _entry, _rmap)
                                cache.flush()
                                logger.debug("[scanner:consumer] Refreshed stale news for %s", sym)
                        except Exception as exc:
                            logger.debug("[scanner:consumer] News refresh error for %s: %s", sym, exc)
            # Still inject YouTube news if available
            if source in ("youtube-news", "youtube"):
                _quiet = muted_tickers is not None and sym in muted_tickers
                if not _quiet and sym not in evaluator.watchlist_symbols:
                    _print_scanner_hit(sym, cache, source="youtube")
                headlines = cache.pop_pending_yt_news(sym)
                if headlines:
                    try:
                        _inject_yt_news(sym, headlines, evaluator, cache)
                    except Exception as exc:
                        logger.warning("[scanner:consumer] News injection error for %s: %s", sym, exc)
            continue

        # Collect any YouTube news headlines for this ticker
        headlines = cache.get_pending_yt_news(sym)
        _quiet = muted_tickers is not None and sym in muted_tickers
        emit_hit = (not _quiet) and (sym not in evaluator.watchlist_symbols)
        if not _quiet:
            _src = source.replace('-news', '').replace('-chat', '')
            safe_print(f"\n  {_c('⟐')}  {_b(sym)}  {_dim(f'({_src})')}")
        try:
            result = run_pipeline(
                sym,
                evaluator,
                tradier,
                yt_headlines=headlines,
                quiet=_quiet,
                allow_manual_rating=False,
                cached_news=_get_cached_news(cache, sym),
                on_scorer_update=on_scorer_update,
            )
            if result is not None:
                cache.pop_pending_yt_news(sym)
                cache.put(sym, result.entry, result.rmap)
                cache.flush()
                if not result.passed:
                    # Cached for display but did not pass. Soft readiness
                    # failures stay retryable; hard failures are marked done.
                    if pipeline_done is not None and scanner_lock is not None:
                        with scanner_lock:
                            if _failures_are_soft(result.fail_reasons) or source == "seed":
                                queued.discard(sym)
                            else:
                                pipeline_done.add(sym)
                elif result.scorer is not None:
                    old = scorers.pop(sym, None)
                    if old:
                        old.stop()
                    scorers[sym] = result.scorer
            else:
                # Pipeline failed
                if scanner_lock is not None:
                    with scanner_lock:
                        if _is_soft_failure(evaluator.last_fail_reason) or source == "seed":
                            # Transient failure or startup seed — allow scanners to retry later
                            queued.discard(sym)
                        else:
                            # Hard failure — sym stays in queued (no retry) but future hits print short format
                            if pipeline_done is not None:
                                pipeline_done.add(sym)
                    if _is_soft_failure(evaluator.last_fail_reason) or source == "seed":
                        logger.debug("[scanner:consumer] %s soft-failed — eligible for retry", sym)
                else:
                    if _is_soft_failure(evaluator.last_fail_reason) or source == "seed":
                        queued.discard(sym)
                        logger.debug("[scanner:consumer] %s soft-failed — eligible for retry", sym)
                    elif pipeline_done is not None:
                        pipeline_done.add(sym)
        except Exception as exc:
            logger.warning("[scanner:consumer] Pipeline error for %s: %s", sym, exc)
            if pipeline_done is not None and scanner_lock is not None:
                with scanner_lock:
                    pipeline_done.add(sym)
    logger.info("[scanner:consumer] Stopped")
