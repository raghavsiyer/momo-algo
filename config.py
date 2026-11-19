"""
config.py  —  Central configuration for the algorithmic trading system.

All tuneable parameters live here. Import from anywhere in the project:
    from config import PRICE_MIN, FLOAT_MAX_M, ...

API keys are read from .env via python-dotenv and should never be
hardcoded here.
"""

import os
import re
from datetime import datetime as _dt, timedelta as _td, timezone as _tz, date as _date, time as _time
from dotenv import load_dotenv
from zoneinfo import ZoneInfo as _ZoneInfo

load_dotenv()


# =============================================================================
# MARKET CLOCK — single source of truth for time / session state
# =============================================================================

class MarketClock:
    """Lightweight helpers for market-session awareness and local display."""

    ET        = _ZoneInfo("America/New_York")
    PREMARKET_WINDOW_TZ = _ZoneInfo("America/Los_Angeles")
    RTH_OPEN  = _time(9, 30)   # regular trading hours
    RTH_CLOSE = _time(16, 0)
    EXT_OPEN  = _time(4, 0)    # pre-market open
    EXT_CLOSE = _time(20, 0)   # post-market close
    PREMARKET_WINDOW_START_LOCAL = _time(3, 45)  # user-facing analysis window
    PREMARKET_WINDOW_END_LOCAL   = _time(7, 0)

    @staticmethod
    def now_et() -> _dt:
        return _dt.now(MarketClock.ET)

    @staticmethod
    def today() -> _dt:
        """Midnight ET for today (timezone-aware)."""
        d = MarketClock.now_et().date()
        return _dt(d.year, d.month, d.day, tzinfo=MarketClock.ET)

    @staticmethod
    def is_rth() -> bool:
        """09:30–16:00 ET (regular trading hours)."""
        return MarketClock.RTH_OPEN <= MarketClock.now_et().time() < MarketClock.RTH_CLOSE

    @staticmethod
    def is_premarket() -> bool:
        """04:00–09:30 ET."""
        t = MarketClock.now_et().time()
        return MarketClock.EXT_OPEN <= t < MarketClock.RTH_OPEN

    @staticmethod
    def is_postmarket() -> bool:
        """16:00–20:00 ET."""
        t = MarketClock.now_et().time()
        return MarketClock.RTH_CLOSE <= t < MarketClock.EXT_CLOSE

    @staticmethod
    def scanners_active() -> bool:
        """04:00–20:00 ET (inclusive on both ends)."""
        t = MarketClock.now_et().time()
        return MarketClock.EXT_OPEN <= t <= MarketClock.EXT_CLOSE

    @staticmethod
    def seconds_to_session_open() -> float:
        """Seconds until the next extended-session open (04:00 ET).

        Aware-datetime subtraction goes through UTC, so the result is the
        true absolute wait even when a DST transition sits between now and
        the open.
        """
        now = MarketClock.now_et()
        target = now.replace(
            hour=MarketClock.EXT_OPEN.hour, minute=MarketClock.EXT_OPEN.minute,
            second=0, microsecond=0,
        )
        if now.time() >= MarketClock.EXT_OPEN:
            target += _td(days=1)
        return max(0.0, (target - now).total_seconds())

    @staticmethod
    def session_cutoff() -> _dt:
        """Midnight ET today — for news freshness checks."""
        return MarketClock.today()

    @staticmethod
    def fmt_local(dt: _dt) -> str:
        """Format a datetime in the system's local timezone."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def local_tz():
        """Return the system's local timezone for display formatting."""
        return _dt.now().astimezone().tzinfo

    @staticmethod
    def premarket_window_utc(day: _date | None = None, local_tz=None) -> tuple[_dt, _dt]:
        """User-facing premarket analysis window for a local calendar day."""
        tz = local_tz or MarketClock.PREMARKET_WINDOW_TZ
        if day is None:
            now_local = _dt.now(tz=tz)
            local_day = now_local.date()
            if now_local.time() < MarketClock.PREMARKET_WINDOW_START_LOCAL:
                local_day -= _td(days=1)
        else:
            local_day = day
        start_local = _dt.combine(local_day, MarketClock.PREMARKET_WINDOW_START_LOCAL, tzinfo=tz)
        end_local = _dt.combine(local_day, MarketClock.PREMARKET_WINDOW_END_LOCAL, tzinfo=tz)
        if end_local <= start_local:
            end_local += _td(days=1)
        return start_local.astimezone(_tz.utc), end_local.astimezone(_tz.utc)

    @staticmethod
    def fmt_local_ts(ts: float) -> str:
        """Format a Unix timestamp in the system's local timezone."""
        return _dt.fromtimestamp(ts, tz=_tz.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# FEATURE TOGGLES — disable data sources, scanners, or noise filters via .env
# =============================================================================

# Data source switches ("false" = disabled)
TRADIER_ENABLED:        bool = os.getenv("TRADIER_ENABLED", "true").lower() == "true"
TRADIER_PAUSED:         bool = False  # runtime-mutable pause flag for Tradier REST/WS
TRADIER_STREAM_ENABLED: bool = os.getenv("TRADIER_STREAM_ENABLED", "true").lower() == "true"
TRADIER_WS_ENABLED:     bool = os.getenv("TRADIER_WS_ENABLED", "true").lower() == "true"
# Note: Tradier has no Level-2 (depth-of-book) feed. Order-book / depth-derived
# signals (book_lean, OFI chop, hidden_seller) are sourced from IBKR when
# ``IBKR_ENABLED=true``. When IBKR is disabled, those scoring vars stay unset
# (optional) and the matching execution criteria are advisory.
IBKR_ENABLED:           bool = os.getenv("IBKR_ENABLED", "false").lower() == "true"
IBKR_HOST:              str  = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT:              int  = int(os.getenv("IBKR_PORT", "7497"))
IBKR_CLIENT_ID:         int  = int(os.getenv("IBKR_CLIENT_ID", "10"))
# Client ids are allocated as a band starting at IBKR_CLIENT_ID — one per
# thread, since ib_insync sync calls are event-loop (thread) bound. This bounds
# the band so a connection leak (a thread exiting without release_ib()) fails
# loudly instead of marching the allocator through ids forever, each new one
# costing an 8s connect timeout.
#
# 32 is the real ceiling, not a guess: measured 2026-08-03 by opening
# connections until refusal (32 succeeded, the 33rd timed out), and it matches
# IBKR's documented "up to 32 simultaneous API clients". The "10 concurrent
# connections" repeated elsewhere in this codebase is simply wrong.
#
# Do NOT lower this as a safety margin. Demand scales with the watchlist —
# ~14 before any ticker is scored (main + stream + scanner + a 6-worker
# pipeline pool + a 5-worker evaluator pool), then one more per active scorer
# poll thread. A smaller ceiling refuses connections that are actually needed.
#
# One thing it cannot do: the 32 is Gateway-wide, shared with every other
# client. If something else holds connections, the Gateway refuses us before
# this counter is reached, and that refusal costs an 8s connect timeout. This
# catches OUR leaks, nothing else's.
IBKR_MAX_CONNECTIONS:   int  = int(os.getenv("IBKR_MAX_CONNECTIONS", "32"))
IBKR_L2_ROWS:           int  = int(os.getenv("IBKR_L2_ROWS", "10"))
# Max concurrent market-depth (L2) subscriptions. IBKR grants 3 depth lines
# by default; extra symbols degrade gracefully to L1-only order flow.
IBKR_L2_MAX_DEPTH_SUBS: int  = int(os.getenv("IBKR_L2_MAX_DEPTH_SUBS", "3"))
# Venues IBKR's smart depth leaves out on this account: its Error 2152 lists
# ARCA, BEX (Nasdaq BX) and NYSE under "Need additional market data
# permissions", and ARCA's own book is refused (354). Each venue's top of book
# still comes through reqMktData with the exchange set, so every depth line also
# carries one directed quote per venue, summed into the ladder like a depth row.
# Costs one market-data line per venue per depth symbol, no depth line.
# Measured 2026-11-09 (OMH, 90s premarket, paper): the smart book's best matched
# the NBBO 54% (bid) / 34% (ask) of the time alone, 95% / 96% with these rows —
# ARCA was often the only venue at the inside. Set empty to turn off.
IBKR_L2_VENUE_QUOTES: tuple[str, ...] = tuple(
    v.strip() for v in os.getenv("IBKR_L2_VENUE_QUOTES", "ARCA,NYSE,BEX").split(",") if v.strip()
)
# ---- Time & sales: tastytrade + IBKR, as the Quantower tape builds it ----
# market_data/tape_merge.py ports ScalpFeedVendor's tape (~/.scalpfeed/project):
# tastytrade's TimeAndSale (Nasdaq Last Sale: Nasdaq's markets + the
# FINRA/Nasdaq TRF, exchange-stamped, first to arrive) merged with IBKR's
# tick-by-tick (every venue, ~0.4 s late, 5 lines), each trade once, sides
# from the NBBO. Modes, as the vendor's `tape` setting in ibkr.json:
#   merged     both feeds, each trade once (the vendor's choice, 2026-11-06)
#   tastytrade tastytrade's tape only
#   ibkr       IBKR's where it holds a tape line, tastytrade's elsewhere
# Needs IBKR_ENABLED. Without the credentials file (or with
# TASTYTRADE_TAPE_ENABLED=false) IBKR's prints pass through unchanged, as before.
TAPE_MODE: str = os.getenv("TAPE_MODE", "merged").strip().lower()
TASTYTRADE_TAPE_ENABLED: bool = os.getenv("TASTYTRADE_TAPE_ENABLED", "true").lower() == "true"
TASTYTRADE_CREDENTIALS_PATH: str = os.getenv(
    "TASTYTRADE_CREDENTIALS_PATH",
    os.path.join(os.path.expanduser("~"), ".scalpfeed", "tastytrade.json"),
)
# Max concurrent tick-by-tick (T&S tape) subscriptions. Measured at exactly 5
# on this account (2026-08-04, tests/test_files/tick_subs_probe.py) — NOT the
# depth-formula number (3) the IBKR docs suggest; the tape and depth quotas
# are independent.
#
# Exceeding it is not a raised error: IBKR refuses the surplus asynchronously
# with "Error 10190: Max number of tick-by-tick requests has been reached",
# leaving the subscription registered but permanently silent. The stream also
# trims this at runtime if a 10190 still arrives, so a wrong value self-corrects.
IBKR_MAX_TICK_SUBS:     int  = int(os.getenv("IBKR_MAX_TICK_SUBS", "5"))
# IBKR: "No more than 1 tick-by-tick request can be made for the same
# instrument within 15 seconds." Slot eviction can re-subscribe a symbol, so
# the stream must not re-request one inside this window.
IBKR_TICK_RESUB_COOLDOWN_SECS: int = 15
# A refused tape/depth request (Error 10190/309) usually means the quota is
# held OUTSIDE this process — another session on the account, or subscriptions
# a dead session leaked at the Gateway (observed 2026-08-05: a killed algo's 3
# depth lines stayed counted for ~15+ minutes; 2 phantom tick lines starved
# the live session for 2+ hours). Retrying on the rebalance cadence turns that
# into an error storm, so each refused symbol backs off exponentially:
# base * 2^(denials-1) seconds, capped at the max.
IBKR_SUB_DENIAL_BACKOFF_SECS: float = float(os.getenv("IBKR_SUB_DENIAL_BACKOFF_SECS", "30"))
IBKR_SUB_DENIAL_BACKOFF_MAX_SECS: float = float(os.getenv("IBKR_SUB_DENIAL_BACKOFF_MAX_SECS", "600"))
# A 10190 lowers the in-session tape cap, but the external pressure that
# caused it is usually transient — after this long with no denial the stream
# probes the cap back up one line per interval (never above IBKR_MAX_TICK_SUBS).
IBKR_TAPE_CAP_RECOVERY_SECS: float = float(os.getenv("IBKR_TAPE_CAP_RECOVERY_SECS", "300"))

# ---- Stream slot allocation (tape + depth lines are scarce) ----
# Tape lines are allocated by pin first (see SLOT_PIN_* below), then by each
# symbol's L1 cumulative-volume rate over this window. The rate comes from
# reqMktData's day-volume field, which keeps updating for every streamed
# symbol whether or not it holds a tape line — so challengers and holders
# are measured on the same scale. (Ranking by prints/sec would starve
# slotless symbols permanently: they emit no prints to be measured by.)
IBKR_TAPE_RANK_WINDOW_SECS: float = float(os.getenv("IBKR_TAPE_RANK_WINDOW_SECS", "60"))
# A challenger must beat the weakest holder's volume rate by this factor to
# take its tape line. Volume rates jitter; without a margin, slots flap
# between similarly active symbols and every flap costs a 15s resub cooldown.
IBKR_TAPE_RANK_MARGIN: float = float(os.getenv("IBKR_TAPE_RANK_MARGIN", "1.5"))
# Minimum time a holder keeps its tape line before rate-based eviction can
# take it — caps worst-case churn at one swap per hold period per line.
IBKR_TAPE_MIN_HOLD_SECS: float = float(os.getenv("IBKR_TAPE_MIN_HOLD_SECS", "30"))
# How often the stream re-ranks tape lines against slotless symbols.
IBKR_TAPE_REBALANCE_SECS: float = float(os.getenv("IBKR_TAPE_REBALANCE_SECS", "5"))
# ---- Depth (L2) relevance: demand-gated grant + idle release ----
# Depth lines only go to symbols that currently matter, and are taken back
# when they stop mattering. The relevance floor below gates BOTH directions:
# a grant (any path — subscribe, idle-fill, rebalance) requires the symbol
# to be pinned OR at/above the floor; a held line is released proactively —
# no challenger needed — once its holder is unpinned AND below the floor for
# the whole idle window. Decoding and aggregating a book nobody is trading
# is pure overhead, and the symbol degrades gracefully to L1-only order
# flow. The window doubles as the minimum hold: the idle clock only runs
# while the line is held, so a fresh grant always gets the full window to
# prove itself, which keeps resubscribe churn (book-corrupting, see
# _update_mkt_depth_l2) impossible by construction.
#
# The floor is DOLLAR volume per second (share rate x L1 price), not share
# rate: 50 shares/s is $100/s on a $2 stock and $10k/s on a $200 stock, so
# a share floor over- and under-gates by price level. $500/s = $30k/min.
IBKR_DEPTH_IDLE_DOLLAR_RATE: float = float(os.getenv("IBKR_DEPTH_IDLE_DOLLAR_RATE", "500"))
IBKR_DEPTH_IDLE_RELEASE_SECS: float = float(os.getenv("IBKR_DEPTH_IDLE_RELEASE_SECS", "90"))
# After an idle release the symbol may not reclaim a line (idle-fill or
# subscribe path) until this cooldown passes — a pin raise bypasses it, so a
# setup arming gets its book back immediately.
IBKR_DEPTH_RELEASE_COOLDOWN_SECS: float = float(os.getenv("IBKR_DEPTH_RELEASE_COOLDOWN_SECS", "60"))

# Slot pins — pushed by DynamicScorer via IBKRSharedStream.set_priority().
# Activity ranking alone would evict exactly the symbols that matter most:
# this strategy buys pullbacks, and a pullback is by definition a quiet
# tape. Pins protect live setups and open positions from rate-based
# eviction; among equal pins, first-come-keeps. Depth lines are granted to
# any symbol while free (idle-fill), only a strictly higher pin evicts, and
# unpinned+quiet holders are idle-released (IBKR_DEPTH_IDLE_* above).
SLOT_PIN_NONE:       int = 0   # cold / warming — rate-ranked, evictable
SLOT_PIN_ARMED:      int = 1   # pullback with trigger level, near HOD
SLOT_PIN_CONFIRMING: int = 2   # breakout fired, entry awaiting confirmation
SLOT_PIN_POSITION:   int = 3   # open (sim) position — exit logic needs data
# Historical-request pacing is handled by the Gateway itself, which shapes
# sub-30s-bar and tick traffic by delaying responses (never erroring) — the
# documented "60 requests per 10 minutes" budget is not enforced and the
# client-side gate for it was removed 2026-08-04 after controlled probing
# (tests/test_files/hist_pacing.py; details in market_data/ibkr_rest.py).
# Per-call ceiling on reqHistoricalTicks pagination. IBKR returns at most
# 1000 ticks per request, so a dense window bisects and re-requests; on a
# liquid name one minute of BID_ASK can cost 50+ requests, and the Gateway
# shapes tick requests to ~1 per 2.5s sustained — unbounded pagination
# burns minutes of wall-clock. TRADES gets the larger share because it is
# the tape itself — BID_ASK only supplies quote context for classifying
# prints.
IBKR_TICK_MAX_REQUESTS_TRADES: int = int(os.getenv("IBKR_TICK_MAX_REQUESTS_TRADES", "24"))
IBKR_TICK_MAX_REQUESTS_QUOTES: int = int(os.getenv("IBKR_TICK_MAX_REQUESTS_QUOTES", "8"))
MARKET_DATA_ENABLED:    bool = TRADIER_ENABLED or IBKR_ENABLED
FMP_ENABLED:            bool = os.getenv("FMP_ENABLED", "true").lower() == "true"
AV_ENABLED:             bool = os.getenv("AV_ENABLED", "true").lower() == "true"
AV_NEWS_ENABLED:        bool = os.getenv("AV_NEWS_ENABLED", "false").lower() == "true"
FINNHUB_ENABLED:        bool = os.getenv("FINNHUB_ENABLED", "true").lower() == "true"
FINBERT_ENABLED:        bool = os.getenv("FINBERT_ENABLED", "true").lower() == "true"
# Brave is a general web search, not a financial API: it returns foreign-
# language and off-ticker results that FinBERT (English-only) then scores
# into the catalyst variable. Off by default; set BRAVE_ENABLED=true to
# re-enable the last-resort fallback.
BRAVE_ENABLED:          bool = os.getenv("BRAVE_ENABLED", "false").lower() == "true"

# Scanner enable/disable ("false" = disabled)
SCANNER_MOMO_ENABLED:    bool = os.getenv("SCANNER_MOMO_ENABLED", "true").lower() == "true"
SCANNER_YT_ENABLED:      bool = os.getenv("SCANNER_YT_ENABLED", "true").lower() == "true"
# IBKR market-scanner feed — effective only when IBKR_ENABLED is also true
SCANNER_IBKR_ENABLED:    bool = os.getenv("SCANNER_IBKR_ENABLED", "true").lower() == "true"

# Noise-filter toggles ("false" = filter off, all headlines pass through)
NOISE_FILTER_TITLE:       bool = os.getenv("NOISE_FILTER_TITLE", "true").lower() == "true"
NOISE_FILTER_YT_HEADLINE: bool = os.getenv("NOISE_FILTER_YT_HEADLINE", "true").lower() == "true"
NOISE_FILTER_YT_TICKERS:  bool = os.getenv("NOISE_FILTER_YT_TICKERS", "true").lower() == "true"
NOISE_FILTER_BRAVE:       bool = os.getenv("NOISE_FILTER_BRAVE", "true").lower() == "true"

# =============================================================================
# DISPLAY
# =============================================================================

BOT_NAME:     str = "T H E    B U N    B O T"
BOT_SUBTITLE: str = "Momentum Trading Engine"

# ANSI colour helpers
_RST = "\033[0m"
def _style(s, code): return f"{code}{s}{_RST}"
_g   = lambda s: _style(s, "\033[32m")    # green  — good
_y   = lambda s: _style(s, "\033[33m")    # yellow — warning
_r   = lambda s: _style(s, "\033[91m")    # red    — bad
_c   = lambda s: _style(s, "\033[36m")    # cyan   — header/label/banner
_dim = lambda s: _style(s, "\033[2m")     # dim    — secondary info
_b   = lambda s: _style(s, "\033[1m")     # bright — emphasis
_w   = lambda s: _style(s, "\033[97m")    # white  — label/title

# =============================================================================
# API KEYS
# =============================================================================

# Tradier — price data, historical candles, live streaming, order execution
TRADIER_API_KEY: str = os.getenv("TRADIER_API_KEY", "")

# FMP — true float, IPO date
FMP_API_KEY:   str  = os.getenv("FMP_API_KEY", "")

# Alpha Vantage — float, splits, news with sentiment (5 calls/min)
AV_API_KEY: str = os.getenv("AV_API_KEY", "")
AV_CALLS_PER_MIN: int = 5

# Finnhub — company profile + news headlines (free tier)
FINNHUB_API_KEY:  str  = os.getenv("FINNHUB_API_KEY", "")

# Brave Search — last-resort news fallback via web search
BRAVE_SEARCH_API_KEY: str  = os.getenv("BRAVE_SEARCH_API_KEY", "")

# =============================================================================
# QUOTE PREFILTER — CHEAP HARD REJECTS BEFORE THESIS SCORING
# =============================================================================

PRICE_MIN:      float = 0.10
PRICE_MAX:      float = 25.00
# Minimum % change vs. prev close. NOT a prefilter reject: the change gate
# lives in the IBKR scanner floor (SCANNER_IBKR_MIN_CHG, pre-evaluation) and
# the live ATTENTION change_pct_min criterion (entry). The prefilter's only
# hard reject is the price band above.
CHANGE_PCT_MIN: float = 10.0

# Legacy aliases retained for compatibility with older scripts/config notes.
# The active volume/RVOL thesis-readiness thresholds are ATTENTION_VOLUME_MIN
# and ATTENTION_RVOL_1DAY_MIN below, not quote-prefilter hard rejects.
VOLUME_MIN:     int   = 100_000
RVOL_1DAY_MIN:  float = 5.0

# =============================================================================
# STAGE 2 — ENRICHMENT FILTER
# =============================================================================

FLOAT_MAX_M:       float = 50.0   # max tradeable float in millions (hard cutoff)
FLOAT_LOW_THRESH_M: float = 20.0  # low-float territory boundary (scoring reference, not a gate)
NEWS_BREAKING_MIN:    int   = 30     # articles within this window = breaking catalyst (display only)
NEWS_FRESH_MIN:       int   = 60     # minutes; threshold used by freshness checks
NEWS_NOISE_THRESHOLD: float = 0.05   # headlines with |sentiment| < this are treated as noise
NEWS_STALE_HOURS:     int   = int(os.getenv("NEWS_STALE_HOURS", "24"))  # re-query sources when cached news is older than this
NEWS_OLD_DAYS_MAX:    int   = 3      # scanner hit display: suppress news older than this many days
# No-news catalyst penalty (strategy/confidence_scorer.py::_update_no_news_penalty):
# breaking_news stays optional/neutral (excluded) only during the grace window after
# start-of-move. Once NO_NEWS_GRACE_SECS have elapsed since move_start with STILL no news
# found, breaking_news flips to NO_NEWS_PENALTY_QUALITY so the missing catalyst drags
# stock_eval down. Real news, when found, always applies instantly and overrides this.
NO_NEWS_GRACE_SECS:      float = 120.0  # seconds after start-of-move before the no-news penalty applies
NO_NEWS_PENALTY_QUALITY: float = 0.20   # breaking_news quality applied when no news past the grace window
SPLIT_WINDOW_DAYS:    int   = 30     # reverse split only flags within last N days
SPLIT_SEVERE_RATIO:   float = 5.0    # ratio > this → severe (display only)
IPO_WINDOW_DAYS:      int   = 365    # IPO bonus applies within last N days


class _MarketCalendar:
    """Lazy-loaded cache of market closed dates.

    Call ``load(closed_dates)`` once at startup with data from
    Tradier's /markets/calendar endpoint.  If never loaded (or Tradier
    is down), weekends are still skipped but holidays are treated as
    trading days — the downstream code handles fewer-than-expected days
    gracefully.
    """
    _closed: frozenset[_date] | None = None

    @classmethod
    def load(cls, closed_dates: frozenset[_date]) -> None:
        cls._closed = closed_dates

    @classmethod
    def is_trading_day(cls, d: _date) -> bool:
        return d.weekday() < 5 and d not in (cls._closed or frozenset())

    @classmethod
    def lookback(cls, n_trading_days: int) -> _td:
        """Walk back exactly *n_trading_days* trading days from now."""
        now = _dt.now()
        cursor = now
        counted = 0
        while counted < n_trading_days:
            cursor -= _td(days=1)
            if cls.is_trading_day(cursor.date()):
                counted += 1
        return now - cursor


_freshness_cache: tuple[int, tuple[_dt, _dt, _dt]] | None = None


def news_freshness_cutoffs() -> tuple[_dt, _dt, _dt]:
    """Return (session_cutoff, breaking_cutoff, stale_cutoff) for news age classification.

    session_cutoff:  midnight ET today — articles published today are "fresh".
    breaking_cutoff: now - NEWS_BREAKING_MIN — articles within this window are "breaking".
    stale_cutoff:    now - NEWS_STALE_HOURS — articles older than this need re-querying.

    Results are cached for 1 second to avoid redundant datetime math across
    multiple callers in the same render cycle.
    """
    global _freshness_cache
    now_utc = _dt.now(tz=_tz.utc)
    epoch_sec = int(now_utc.timestamp())
    if _freshness_cache is not None and _freshness_cache[0] == epoch_sec:
        return _freshness_cache[1]
    session_cutoff = MarketClock.session_cutoff()
    breaking_cutoff = now_utc - _td(minutes=NEWS_BREAKING_MIN)
    stale_cutoff = now_utc - _td(hours=NEWS_STALE_HOURS)
    result = (session_cutoff, breaking_cutoff, stale_cutoff)
    _freshness_cache = (epoch_sec, result)
    return result


# =============================================================================
# HARD PRICE ACTION DEALBREAKERS  (Stage 3 — technical gate)
# =============================================================================

# Bid-ask spread limits by price tier
# Also used by confidence_scorer.py for dynamic spread tracking
SPREAD_MAX_UNDER_1:    float = 0.02   # <$1 stocks: max 2 cents
SPREAD_MAX_1_TO_5:     float = 0.10   # $1–$5:      max 5 cents
SPREAD_MAX_5_TO_10:    float = 0.13   # $5–$10:     max 7 cents
SPREAD_MAX_10_TO_20:   float = 0.15   # $10–$20:    max 10 cents

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

# EMAs / MAs
EMA_9_PERIOD:   int = 9    # used on 10-sec and 1-min charts
EMA_20_PERIOD:  int = 20   # used on 1/2/5/15-min charts
EMA_50_PERIOD:  int = 50   # used on 1/2/5/15-min charts (as MA)
EMA_200_PERIOD: int = 200  # daily chart
MA_50_PERIOD:   int = 50   # daily chart
MA_200_PERIOD:  int = 200  # daily chart

# VWAP
VWAP_SESSION_RESET: bool = True   # reset VWAP at market open each day

# Bollinger Bands (5-min)
BB_PERIOD: int   = 20
BB_STD:    float = 2.0

# MACD-V (1-min and 5-min). Every macd_* column and MACD gate is MACD-V since
# 2026-11-18 (user request): Spiroglou's volatility-normalised MACD (2022),
# 100 * (EMA12 - EMA26) / ATR(26), so momentum is in units of the bar's own
# volatility and reads the same across prices (+-50 ranging, +-150
# overextended). Signal = EMA9 of MACD-V. ATR is Wilder's, as in the paper.
MACD_FAST:   int = 12
MACD_SLOW:   int = 26
MACD_SIGNAL: int = 9
MACD_ATR_PERIOD: int = 26

# ROC
ROC_PERIOD: int = 1   # 1-period rate of change on 10-sec candles

# =============================================================================
# DYNAMIC CONFIDENCE SCORE — WEAKNESS / STRENGTH THRESHOLDS
# =============================================================================

# Weakness indicators that reduce confidence or trigger step-away
EXTENSION_MIN_CANDLES:  int   = 5     # 5+ consecutive 1-min or 5-min candles up = extended
RVOL_5MIN_MIN:          float = 2.0   # 5-min RVol below this = weakness
TOPPING_TAIL_RATIO:     float = 0.35  # upper_wick / full_candle_range above this = topping tail warning
TOPPING_TAIL_ATR_PERIOD: int = 14     # ATR lookback for intraday topping-tail normalization

# Strength indicators
RESISTANCE_BREAK_BUFFER: float = 0.02  # price must be this far above resistance to confirm

# =============================================================================
# EXECUTION — NO-ENTRY / EXIT THRESHOLDS
# =============================================================================

# Key price levels to watch for (whole/half dollar, and sub-$1 levels)
WHOLE_DOLLAR_BUFFER:   float = 0.09   # within $0.06 of whole dollar = caution zone
HALF_DOLLAR_BUFFER:    float = 0.09   # within $0.06 of half dollar = caution zone
SUB_1_KEY_LEVELS:      list  = [0.10, 0.25, 0.50, 0.75]  # key sub-$1 levels

# MA/EMA levels treated as major resistance
MAJOR_RESISTANCE_MAS: list = [50, 200]   # period lengths flagged as major resistance

# --- Moving-average levels in the resistance/support map ---
# Every (kind, period) below is marked on every timeframe below as its own
# level (technical_analysis/moving_average_levels.py). Like every other level
# in the map they feed the entry headroom gate and the resistance-approach
# exit, so trimming a timeframe here also stops it gating trades.
# Kinds: "ema" (exponential, seeded with the SMA of its first `period`
# closes, as Pine's ta.ema does) or "sma" (simple).
MA_LEVEL_SPECS: tuple[tuple[str, int], ...] = (
    ("ema", EMA_200_PERIOD),
    ("sma", MA_200_PERIOD),
    ("sma", MA_50_PERIOD),
)
MA_LEVEL_TIMEFRAMES: tuple[str, ...] = (
    "5m", "15m", "30m", "1h", "2h", "4h", "daily", "weekly", "monthly",
)
# Calendar days of history fetched per source interval. Sized for ~3x the
# longest period on the timeframe each feeds (EMA warm-up: the seed's weight
# falls to ~e^-4 after 2 more periods) on extended-hours bars (16h/day);
# 2h/4h are built from the 1-hour fetch. Monthly asks for the most history
# the provider has; most small caps simply have fewer than 200 months, and a
# timeframe with fewer closed bars than a period leaves that MA unmarked.
MA_LEVEL_FETCH_DAYS: dict[str, int] = {
    "5min": 10,
    "15min": 21,
    "30min": 45,
    "1hour": 220,
    "1day": 1100,
    "1week": 5500,
    "1month": 11000,
}
# The pipeline waits at most this long for the MA fetch before printing the
# summary and starting the scorer (the fetch starts at the float gate, so it
# usually finishes during news/sentiment). Kept short because the scorer's
# start waits on it; a slower fetch still marks its levels when it lands.
MA_LEVEL_FETCH_WAIT_SECS: float = 3.0

# =============================================================================
# WATCHLIST
# =============================================================================

# Backtest only: replay of a symbol stops once price falls this far below
# its high for the day, so a dead name costs no further tick processing.
# The live session no longer fades on this — that check fired only while a
# ticker was being queried, so it never reached the names nobody looked at.
# Live release is the IBKR scanner's drop check (SCANNER_IBKR_DROP_SECS).
DAILY_HIGH_FADE_PCT: float = 0.25

# =========================================xwww====================================
# CONFIDENCE SCORING — QUALITY FUNCTION ENGINE
# =============================================================================
# Each group: {name, weight, vars: [{name, weight, fn, lo, hi, ...}]}
# Quality functions map raw values to [0, 1].  Composite = weighted avg of groups.
# See strategy/scoring.py for the engine.

# --- Kept for display / legacy helpers ---
CONFIDENCE_FLOOR:    float =  0.0
CONFIDENCE_CEILING:  float = 100.0

# --- Sector keyword lists (used to compute sector quality value) ---
SECTORS_TIER1: list = ["defense", "military", "aerospace", "government", "armed"]
SECTORS_TIER2: list = ["energy", "oil", "gas", "petroleum", "renewables", "solar", "nuclear"]
SECTORS_TIER3: list = ["semiconductor", "artificial intelligence", "ai ", "chip",
                       "technology", "software", "tech"]

# --- PRICE_ACTION tuning (scoring + multi-timeframe combination) ---
TOPPING_TAIL_DECAY:    float = 0.6
# Significance thresholds for pattern detection (scoring + display).
# A candle only contributes to wick/body quality if it meets both gates.
PATTERN_MIN_RAW:      float = 0.2   # min candle range / ATR for meaningful-candle scoring
PATTERN_HOD_PROXIMITY: float = 0.995 # candle high must be >= this × TF HOD
VOLUME_RESPONSE_TOP_VOLUME_PCT: float = 0.20  # top-volume percentile used as full-response volume anchor
VOLUME_RESPONSE_RANGE_ATR_MIN: float = 0.35   # candle range must be >= this ATR share before it matters
VOLUME_RESPONSE_RANGE_ATR_FULL: float = 3.00  # candle range ATR share that reaches full range weight
VOLUME_RESPONSE_WICK_ATR_MIN: float = 0.20    # upper wick must be >= this ATR share before it matters
VOLUME_RESPONSE_WICK_ATR_FULL: float = 0.80   # upper wick ATR share that reaches full wick-size risk
VOLUME_RESPONSE_WICK_RANGE_MIN: float = 0.25  # upper wick must be >= this candle-range share before it matters
VOLUME_RESPONSE_WICK_RANGE_FULL: float = 0.60 # upper wick range share that reaches full wick-shape risk
VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN: float = 1.50 # red-volume pressure starts at this × median TF candle volume
VOLUME_RESPONSE_RED_VOLUME_RATIO_FULL: float = 2.50 # red-volume pressure reaches full strength at this × median TF volume
VOLUME_RESPONSE_RED_VOLUME_MAX_RISK: float = 0.55   # max risk from red high-volume pressure alone

# --- Setup outcome scoring (live confidence per-TF) ---
# A resolved setup "works" when the continuation leg (green candles after
# breakout) extends >= this fraction of price from breakout level.
SETUP_OUTCOME_SUCCESS_PCT: float = 0.10   # 10% price extension = full success
# Failure-candle severity and the setup-segment rejection metrics
# (topping_tail / jackknife) use the same volume-response candle scoring as
# key_candles, but relative volume acts as a severity modulator rather than
# an eligibility gate: a low-volume red candle (healthy pullback behaviour)
# is softened toward this floor, while a high-volume rejection keeps full
# structural severity.
SETUP_VOLUME_SEVERITY_FLOOR: float = 0.40
# Exponential decay for historical outcomes: most recent outcome has weight 1.0,
# each prior outcome is multiplied by this factor.
SETUP_OUTCOME_DECAY:       float = 0.5
# Max partial credit a failed setup can earn after price repairs the failure by
# reclaiming the breakout level and holding near the upper session range.
SETUP_OUTCOME_RECOVERY_MAX_CREDIT: float = 0.65
# Volume significance gate (like key_candles's per-candle weighting): each
# leg's evidence weight is smoothstep(peak closed-candle volume vs the CURRENT
# median->top-VOLUME_RESPONSE_TOP_VOLUME_PCT anchors of its timeframe),
# recomputed on every candle close. Legs whose biggest candle sits at/below
# the current median carry no evidence; the per-TF evidence mass scales
# setup_outcome's weight in the TA blend (strategy.scoring.combine_tf_scores /
# MIN_TF_EVIDENCE_MASS), so stale low-volume micro-legs stop dragging live
# confidence once real volume shows what participation looks like.

# Minimum candles (per timeframe, within the active move window) required
# before candle choppiness is scored into live confidence.
CANDLE_CHOP_MIN_CANDLES: int = 4

# Multi-timeframe blend in strategy.scoring.combine_tf_scores().
PRICE_ACTION_TF_WEIGHTS: dict[str, float] = {
    "5m":  0.40,
    "2m":  0.25,
    "1m":  0.20,
    "10s": 0.15,
}
PRICE_ACTION_TF_ALIGNMENT_BONUS: float = 1.15
PRICE_ACTION_TF_CONFLICT_PENALTY: float = 0.70
PRICE_ACTION_TF_2M_ENABLED:      bool  = False  # gate 2m timeframe in multi-TF scoring

# --- Live variable dampening (EMA with decaying alpha) ---
# As more observations accumulate, each new reading has less marginal impact.
# α(n) = max(LIVE_DAMP_FLOOR, LIVE_DAMP_START / (1 + n / LIVE_DAMP_HALFLIFE))
LIVE_DAMP_START:    float = 0.40   # initial alpha (first reading gets 40% weight vs neutral)
LIVE_DAMP_FLOOR:    float = 0.05   # minimum alpha (never fully frozen; genuine reversals still penetrate)
LIVE_DAMP_HALFLIFE: float = 50.0   # observations until alpha halves from start

# --- Variable group definitions ---
SCORING_GROUPS: list[dict] = [
    # ── 1. STOCK EVALUATION — Is this stock worth trading? ────────────
    # Static pillars that determine fundamental tradability of a low-float
    # momentum ticker.  Set at evaluation time (Stage 2).
    {
        "name": "STOCK_EVAL", "weight": 1.0, "category": "QUALITY",
        "vars": [
            # float in millions.  Lower = better.  log_scale for power-law gradient.
            # Curve extends to 50M for gradual taper, hard cutoff at 20M.
            {"name": "float_shares",   "weight": 1.5, "fn": "LINEAR_MIN",
             "lo": 0.0, "hi": 50.0, "log_scale": True, "cutoff": 20.0},
            # Input: avg_sentiment remapped to (sent+1)/2 → [0, 1].
            # No news → value=None → excluded (optional).
            # Good news is a bonus; bad news actively hurts.
            {"name": "breaking_news",  "weight": 1.3, "fn": "LINEAR_RANGE",
             "lo": 0.0, "hi": 1.0, "optional": True},
            # 1-day RVOL.  Threshold=5 → quality 0.5 at 5×.
            # Hard cutoff at RVOL_1DAY_MIN — below 5× scores zero.
            {"name": "rvol_1day",      "weight": 1.5, "fn": "SATURATING",
             "lo": 5.0, "hi": 0.0, "cutoff_lo": 5.0},
            # price: symmetric linear bell, peak $3.00, extends ±17 → lo=-14/hi=20
            # hard cutoffs at PRICE_MIN=$0.10 (left) and PRICE_MAX=$20 (right)
            {"name": "price",          "weight": 0.7, "fn": "BELL_SKEW",
             "lo": -14.0, "hi": 20.0, "skew_peak": 3.0, "cutoff_lo": 0.10},
            # Total chart range (%) = (max_high - min_low) / avg_close over prior 20 daily bars.
            # Observed (logs): median ~19%, p75 ~40%, p90 ~115%, p95 ~340%, p99 ~600%.
            # hi=400 keeps q≈0.5 near p93 so only genuinely wild charts (>p95) score near 0.
            {"name": "daily_trend",    "weight": 0.6, "fn": "LINEAR_MIN",
             "lo": 0.0, "hi": 400.0},
        ],
    },
    # ── 2. TECHNICAL ANALYSIS — Cleanliness of the move ───────────────
    # Evaluated per-timeframe (10s/1m/2m/5m) and combined via
    # combine_tf_scores() in the scorer.  All variables are live/dynamic.
    # Measures decisiveness: clean legs vs choppy / rejected high-volume candles.
    {
        "name": "TECHNICAL_ANALYSIS", "weight": 1.0, "category": "STRENGTH_BUY",
        "vars": [
            # setup_outcome: per-TF outcome quality of resolved pullback setups.
            # Setup quality is the ceiling; favorable extension earns partial
            # credit toward SETUP_OUTCOME_SUCCESS_PCT, and failed outcomes are
            # multiplied by the failure candle's volume-response quality (same
            # candle scoring as key_candles; low-volume red failure candles
            # are softened via SETUP_VOLUME_SEVERITY_FLOOR).
            # Each leg is weighed by its peak candle volume against the
            # CURRENT volume baseline (median -> top-percentile, like
            # key_candles): volume-insignificant legs carry ~no evidence
            # and this weight (3.0) scales down with the evidence mass.
            # Score set directly via score_override from combine_tf_scores.
            {"name": "setup_outcome", "weight": 3.0, "fn": "LINEAR_RANGE",
             "lo": 0.0, "hi": 1.0, "optional": True},
            # key_candles: high-volume candle response across stream timeframes.
            # Set on candle close by DynamicScorer._update_key_candles. Raw
            # value = blended rejection risk across timeframes: each TF applies
            # a continuous relative-volume gradient, then measures rolling-ATR
            # and range-normalized upper-wick / red-body rejection risk. Candles
            # below VOLUME_RESPONSE_RANGE_* size gates and small wicks below
            # VOLUME_RESPONSE_WICK_* significance gates stay near neutral, but
            # high-volume red closes add pressure risk above VOLUME_RESPONSE_RED_*
            # thresholds. Large high-volume topping tails / jackknives produce
            # high raw values (low quality). optional → excluded when there is no
            # usable volume evidence.
            {"name": "key_candles",  "weight": 1.5, "fn": "LINEAR_MIN",
             "lo": 0.0, "hi": 1.0, "optional": True},
            # atrp: Average True Range as a percent of price (14-period 1-min
            # ATR / price * 100), set on candle close by
            # DynamicScorer._update_atrp_and_choppiness.  Higher = more intraday
            # movement/opportunity.  SATURATING: quality 0.5 at 2% ATRP,
            # asymptotes to 1.0.  optional -> excluded until ATR is available.
            {"name": "atrp",           "weight": 0.6, "fn": "SATURATING",
             "lo": 2.0, "hi": 0.0, "optional": True},
            # candle_choppiness: fraction of candle-color flips (red<->green)
            # that are NOT part of a succeeded or still-active pullback,
            # blended across all timeframes (10s/1m/2m/5m) by
            # PRICE_ACTION_TF_WEIGHTS.  A succeeded pullback (breakout candle
            # closes over the pullback entry price) is excused so clean
            # staircase trends don't read as choppy; the trailing, not-yet-
            # resolved pullback is excused too so an in-progress pullback
            # can't drag live confidence during its own entry window — a
            # pullback only counts as chop once it has objectively failed.
            # Lower = cleaner = better.  LINEAR_MIN: ratio 0 -> quality 1.0,
            # ratio >= 0.6 -> quality 0.  optional -> excluded until enough
            # candles exist.
            {"name": "candle_choppiness", "weight": 1.0, "fn": "LINEAR_MIN",
             "lo": 0.0, "hi": 0.6, "optional": True},
        ],
    },
]


# =============================================================================
# SUMMARY DISPLAY
# Thresholds used in display.py for colour coding and sentiment labels.
# =============================================================================

# Confidence score colour thresholds (Quality × Strength model)
CONFIDENCE_GREEN_COLOR:  float = 70.0   # score >= this → green
CONFIDENCE_YELLOW_COLOR: float = 30.0   # score >= this → yellow, else red

# Sentiment label boundaries for FinBERT-style scores where
# score = P(positive) - P(negative). A wider neutral band avoids treating
# weak polarity as directional sentiment.
SENT_VERY_BULLISH_MIN: float =  0.75
SENT_BULLISH_MIN:      float =  0.45
SENT_NEUTRAL_ABS_MAX:  float =  0.15   # -0.15 to +0.15 → neutral
SENT_MILDLY_BEAR_MIN:  float = -0.45   # above this (and < -neutral band) → mildly bearish
SENT_BEARISH_MIN:      float = -0.75   # above this → bearish


def sentiment_label(score: float) -> str:
    """Map a sentiment score in [-1, 1] to a human-readable label."""
    if score >= SENT_VERY_BULLISH_MIN:
        return "Very Bullish"
    if score >= SENT_BULLISH_MIN:
        return "Bullish"
    if score > SENT_NEUTRAL_ABS_MAX:
        return "Mildly Bullish"
    if score >= -SENT_NEUTRAL_ABS_MAX:
        return "Neutral"
    if score >= SENT_MILDLY_BEAR_MIN:
        return "Mildly Bearish"
    if score >= SENT_BEARISH_MIN:
        return "Bearish"
    return "Very Bearish"


def sentiment_is_bullish(score: float) -> bool:
    """True when the score is Bullish or Very Bullish."""
    return score >= SENT_BULLISH_MIN


def sentiment_is_bearish(score: float) -> bool:
    """True when the score is Bearish or Very Bearish."""
    return score < SENT_MILDLY_BEAR_MIN

# Resistance distance threshold — within this many dollars = "approaching" not "room to run"
RESISTANCE_APPROACH_DIST: float = 0.20

# Float display labels (for the summary printer — separate from the scoring FLOAT_TIER_* above)
FLOAT_DISPLAY_ULTRA_LOW: float = 1.0   # < 1M  → "ultra-low float"
FLOAT_DISPLAY_LOW:       float = 5.0   # < 5M  → "low float"

# =============================================================================
# DYNAMIC CONFIDENCE SCORER — TUNING  (confidence_scorer.py)
# Controls which live market conditions trigger the weakness/strength deltas.
# =============================================================================

# ROC — rate of change computed over T&S print prices
# (distinct from the technical-indicator ROC_PERIOD=1 defined above)
ROC_TS_PERIOD:      int   = 3      # number of T&S ticks in the ROC window

# Order book — L2 quote size thresholds
BOOK_ILLIQUID_MAX_SIZE: int =  50    # both sides <= this = thin / illiquid book
BOOK_ILLIQUID_CONSEC:   int =   4    # consecutive thin-book events before flagging

# Order flow imbalance — OFI thresholds (exec/order_flow.py → confidence_scorer.py)
#
# Tuned for low-float small-caps ($0.50–$20, <10M float, high RVOL).
# On these names even a modest imbalance is meaningful because total
# depth is thin and prints carry proportionally more information.
#
# Trade flow (T&S aggressor classification)
# buying_pressure / selling_pressure are order counts in PRESSURE_WINDOW_SECS (execution-only).
OFI_SELL_DOMINANT_THRESH:  float = -0.30   # 65%+ sell-initiated → sellers aggressing
OFI_BUY_SURGE_THRESH:      float =  0.35   # 67%+ buy-initiated  → buyers aggressing
OFI_HYSTERESIS:            float =  0.10   # deactivation band to prevent rapid toggling
# Book lean (L1 volume imbalance, EMA-smoothed)
# ρ ∈ [-1, +1]; -0.40 means ~70% of L1 size on the ask side.
BOOK_LEAN_BEARISH_THRESH:  float = -0.40   # persistent ask-heavy lean
# OFI quality (chop + exhaustion)
# chop_ratio = fraction of consecutive OFI samples with a sign change.
OFI_CHOP_RATIO_THRESH:     float =  0.55   # 55%+ alternation = directionless
# ofi_intensity = avg |OFI| per quote event (shares). On low-float stocks
# with avg print 100-300 shares, 25 shares/event is near-zero net movement.
OFI_EXHAUSTION_THRESH:     float = 25.0    # near-zero aggression
OFI_EXHAUSTION_MIN_QUOTES: int   = 20      # need ~3-5s of quotes before calling exhaustion

# RVol — 5-min rolling window
RVOL_WINDOW_SECONDS:  int = 300   # rolling window length in seconds
RVOL_BARS_PER_SESSION: int =  78  # 5-min bars in a full 6.5-hour session

# Scorer background thread — max seconds to wait for clean shutdown
SCORER_STOP_TIMEOUT_SECS: int = 3

# Minimum notional value (price × shares) for a T&S print to be processed.
# Filters out sub-$x noise prints that add no signal.
TS_MIN_NOTIONAL: float = 100.0

# Scorer REST poll interval — how often to fetch fresh 1-min candles (seconds)
SCORER_POLL_INTERVAL_SECS: int = 60
# Random spread added to each poll wait. Startup seeds every cached symbol at
# once, so without it every scorer polls on the same tick — and since the poll
# thread now connects per poll rather than holding a connection, that would
# spike the Gateway's 32-connection ceiling all at once. Jitter staggers them.
SCORER_POLL_JITTER_SECS:   int = 10
SCORER_CANDLE_MAX_BARS:    int = 20   # 1-min candle buffer cap

# When the newest tape print is older than this (or no print ever arrived —
# e.g. the symbol holds no tape line), ScorerSnapshot.last_price falls back
# to the L1 quote stream's last-trade price so the display %-change and HOD
# distance keep moving. Execution stays print-gated: entry/tape gates never
# read the fallback.
SCORER_QUOTE_PRICE_FALLBACK_SECS: float = 10.0

# Maximum cached T&S prints to replay through full setup reconstruction on scorer start.
# Larger tapes fall back to lightweight rolling-state seeding so interactive refresh stays fast.
SCORER_MAX_RECONSTRUCT_TICKS: int = 25000

# Scoring telemetry — log every variable's raw value + quality each poll cycle
SCORING_LOG_ENABLED:       bool = True
SCORING_LOG_DIR:           str  = os.path.join(os.path.dirname(__file__), "logs")

# =============================================================================
# PATTERN DETECTION  (technical_analysis/patterns.py)
# =============================================================================

# --- Pivot detection ---
PATTERN_PIVOT_N_10SEC: int = 3     # pivot window for 10-sec candles (noisier → wider)
PATTERN_PIVOT_N_1MIN:  int = 2     # pivot window for 1-min candles
PATTERN_PIVOT_N_2MIN:  int = 2     # pivot window for 2-min candles
PATTERN_PIVOT_N_5MIN:  int = 2     # pivot window for 5-min candles

# --- Multi-top / multi-bottom ---
PATTERN_TOP_PRICE_TOL_PCT:  float = 0.005   # peaks within 0.5% = same cluster
PATTERN_TOP_MIN_PULLBACK:   float = 0.003   # minimum pullback between peaks (% of price)

# --- Wedge ---
PATTERN_WEDGE_MIN_TOUCHES: int    = 3       # min pivot points per trendline
PATTERN_WEDGE_MIN_BARS:    int    = 10      # min candles the wedge must span
PATTERN_WEDGE_CONVERGE_MAX: float = 0.95    # slope ratio must be < this to confirm convergence

# --- Head and shoulders ---
PATTERN_HNS_SHOULDER_TOL_PCT: float = 0.015  # shoulders within 1.5% of each other
PATTERN_HNS_HEAD_MIN_EXCEED:  float = 0.005  # head must exceed shoulders by at least 0.5%

# --- Breakout pullback / break + retest ---
PATTERN_RETEST_BUFFER_PCT:  float = 0.005    # price must come within 0.5% of level
PATTERN_RETEST_MAX_BARS:    int   = 15       # max bars after break to still qualify as retest
PATTERN_RETEST_BOUNCE_BARS: int   = 3        # bars of hold/bounce after retest to confirm

# --- Support bounce / resistance rejection ---
PATTERN_BOUNCE_PROXIMITY_PCT: float = 0.005  # within 0.5% of level = "touching"
PATTERN_BOUNCE_CONFIRM_BARS:  int   = 2      # bars closing away from level = confirmed

# --- Tail cluster ---
PATTERN_TAIL_CLUSTER_BARS:    int   = 10     # window to look for repeated tails
PATTERN_TAIL_CLUSTER_MIN:     int   = 2      # min tails in window to flag a cluster
PATTERN_TAIL_CLUSTER_TOL_PCT: float = 0.005  # tails within 0.5% price = same cluster

# --- Pure pullback setup (new-high trigger, non-resistance) ---
# Pullback depth is measured against the most recent up-leg:
# depth = (leg_high - pullback_low) / (leg_high - leg_open)
PATTERN_PULLBACK_MAX_DEPTH: float = 0.50   # maximum valid pullback depth = 50%
PATTERN_PULLBACK_MIN_LEG_PCT: float = 0.05  # minimum leg move from open to high = 5%
PATTERN_PULLBACK_MIN_LEG_ATR_MULT: float = 1.0  # leg range must be at least 1x recent ATR
PATTERN_PULLBACK_MIN_LEG_CANDLES: int = 2  # minimum consecutive green candles before pullback
PATTERN_PULLBACK_MAX_LEG_CANDLES: int = 5  # maximum setup leg window before pullback
PATTERN_PULLBACK_MAX_CANDLES: int = 3  # maximum consecutive red candles in pullback
# When True the setup leg high must reach session HOD (within
# PATTERN_HOD_PROXIMITY) or the setup is rejected. Turn off to allow
# continuation pullbacks whose leg peaked below the day's high.
PATTERN_PULLBACK_REQUIRE_HOD: bool = False
# Timeframes on which pullback SETUP detection (and therefore setup tracking,
# setup_outcome legs, and entries) is disabled. Enforced in the setup pipeline
# (scan_all_patterns + the scorer's forming-pullback loop) — NOT inside the
# detectors: the backtest's replay-window derivation calls the detectors
# directly and must keep seeing all timeframes, because it decides which tick
# spans get replayed (coverage infrastructure, not strategy).
#   "2min": disabled outright — 2m is also excluded from the live-confidence
#           blends via PRICE_ACTION_TF_2M_ENABLED=False, so the timeframe is
#           fully dead (2m setups repeatedly anchored stale levels that
#           squatted on the single active-setup slot, e.g. YXT 2026-08-05).
#   "5min": disabled as a SETUP timeframe only — 5m candles still feed live
#           confidence (key_candles, candle_choppiness, multi-TF weights),
#           so higher-TF context scores the move while entries trade the
#           10s/1m structure.
SETUP_TFS_DISABLED: frozenset[str] = frozenset({"2min", "5min"})

# Separate setup/pattern confidence group (does not feed live confidence).
# Inputs are pullback cleanliness features from the detector. This remains
# fully separate from the live confidence score path.
# topping_tail / jackknife carry volume-response rejection risks in [0, 1]
# (see SETUP_VOLUME_SEVERITY_FLOOR) rather than raw ATR ratios.
# pullback_depth is evaluated in raw space:
#   0.00 -> best (shallow pullback), 0.50 -> worst (50% pullback)
SETUP_PATTERN_GROUP: dict = {
    "name": "SETUP_PATTERN",
    "weight": 1.0,
    "category": "SETUP",
    "vars": [
        {
            "name": "pullback_depth",
            "weight": 1.4,
            "fn": "LINEAR_MIN",
            "lo": 0.0,
            "hi": PATTERN_PULLBACK_MAX_DEPTH,
        },
        {
            "name": "pullback_control",
            "weight": 1.0,
            "fn": "LINEAR_RANGE",
            "lo": 0.0,
            "hi": 1.0,
        },
        {
            "name": "topping_tail",
            "weight": 1.0,
            "fn": "LINEAR_MIN",
            "lo": 0.0,
            "hi": 1.0,
        },
        {
            "name": "jackknife",
            "weight": 1.0,
            "fn": "LINEAR_MIN",
            "lo": 0.0,
            "hi": 1.0,
        },
        {
            "name": "pullback_candles",
            "weight": 0.7,
            "fn": "LINEAR_MIN",
            "lo": 0.0,
            "hi": 5.0,
        },
        {
            "name": "pullback_structure",
            "weight": 0.8,
            "fn": "LINEAR_MAX",
            "lo": 0.65,
            "hi": 1.0,
        },
        {
            "name": "breakout_extension",
            "weight": 0.8,
            "fn": "SATURATING",
            "lo": 0.12,
            "hi": 0.0,
        },
        {
            "name": "leg_candles",
            "weight": 0.4,
            "fn": "SATURATING",
            "lo": 3.0,
            "hi": 0.0,
        },
    ],
}

# =============================================================================
# CRITERIA GROUPS — boolean gates (no scoring contribution)
# =============================================================================
# Two parallel structures alongside SCORING_GROUPS:
#   - CRITERIA_ATTENTION: "Is this stock worth watching?"  Level-triggered state
#     checks — must ALL be true for the ticker to remain FOCUSED.
#   - CRITERIA_EXECUTION: "Is NOW the right moment to enter?"  Edge-triggered
#     T&S event detectors for speed, imbalance, and acceleration.
# =============================================================================

CRITERIA_ATTENTION: list[dict] = [
    {
        "name": "ATTENTION",
        "criteria": [
            {"name": "change_pct_min",
             "description": f"% change >= {CHANGE_PCT_MIN}%",
             "required": True},
            {"name": "volume_min",
             "description": "volume >= 50k",
             "required": True},
            {"name": "rvol_1day_min",
             "description": "1-day RVol >= threshold",
             "required": True},
            {"name": "dist_from_hod_atr_max",
             "description": "within 2 ATRs or 10% of HOD",
             "required": True},
            # L2-dependent: only enforced when IBKR (depth feed) is enabled.
            {"name": "order_book_density_ok",
             "description": "order book density within range (L2)",
             "required": IBKR_ENABLED},
            {"name": "vwap_slope_positive",
             "description": "VWAP slope >= -0.1 $/min (10s-sampled)",
             "required": True},
            {"name": "close_above_vwap",
             "description": "last close > session VWAP",
             "required": True},
            {"name": "macd_positive",
             "description": "MACD-V histogram > 0 (1-min)",
             "required": True},
        ],
    },
]

CRITERIA_EXECUTION: list[dict] = [
    {
        "name": "EXECUTION",
        "criteria": [
            # ── T&S event detectors (edge-triggered) ──
            {"name": "tape_speed",
             "description": "speed gate (source per TAPE_SPEED_GATE_SOURCE; fast-EMA prints/sec by default) >= SIM_PRESSURE_SPEED_MIN",
             "required": True},
            {"name": "tape_imbalance",
             "description": "buy ratio (source per TAPE_RATIO_GATE_SOURCE; fast-EMA count ratio by default) >= SIM_PRESSURE_BUY_RATIO_MIN",
             "required": True},
            {"name": "tape_vol_flow",
             "description": "30s volume flow not sell-dominated: trade_imbalance > SIM_PRESSURE_VOL_IMBALANCE_MIN (guards count-bullish/volume-bearish tape)",
             "required": True},
            {"name": "tape_acceleration",
             "description": "enough pressure data and accel significance z=accel*sqrt(N) >= SIM_PRESSURE_ACCEL_Z",
             "required": True},
            # ── Structural safety gates ──
            {"name": "spread_ok",
             "description": "bid-ask spread <= EXEC_SPREAD_PCT_MAX of mid price",
             "required": True},
            {"name": "dist_to_resistance_ok",
             "description": "headroom to nearest major resistance > 1 ATR "
                            "(the active setup's own prior high / entry band "
                            "is skipped — it is the level being broken, not "
                            "a wall; see exec.signals.resistance_search_floor)",
             "required": True},
            # L2-dependent: only enforced when IBKR (depth feed) is enabled.
            {"name": "no_hidden_seller",
             "description": "hidden_seller intensity below threshold (L2)",
             "required": IBKR_ENABLED},
        ],
    },
]

# Stage-1 attention thresholds
ATTENTION_VOLUME_MIN: int = 50_000
ATTENTION_RVOL_1DAY_MIN: float = 5.0          # 1-day relative volume floor
ATTENTION_DIST_FROM_HOD_ATR_MAX: float = 2.0  # must be within 2 ATRs of HOD
ATTENTION_VWAP_SLOPE_MIN: float = -0.1        # VWAP slope floor ($/min); slight downtrend tolerated
# HOD-proximity widens to whichever is bigger: N ATRs OR pct of HOD.
HOD_PROXIMITY_PCT_MAX: float = 0.10

# Toolbar "MOVING" classification
MOVING_DIST_HOD_ATR_MAX: float = 2.0    # must be within 2 ATRs of daily high
MOVING_TAPE_PRESSURE_MIN: int = 10      # buying+selling pressure sum to qualify


def hod_proximity_max_atrs(hod: float, atr: float,
                           base_atr_max: float,
                           pct_max: float = HOD_PROXIMITY_PCT_MAX) -> float:
    """Effective max ATR distance from HOD that counts as 'near'.

    Returns max(base_atr_max, pct_max*hod/atr) so a simple
    `dist_atrs <= threshold` check honors "within N ATRs OR P% of HOD".
    """
    if atr <= 0 or hod <= 0:
        return base_atr_max
    return max(base_atr_max, (pct_max * hod) / atr)

# Stage-4 execution thresholds
EXEC_SPREAD_PCT_MAX: float = 0.02            # gate trips when NBBO spread > 2% of mid
EXEC_DIST_TO_RESISTANCE_MULT: float = 1.0     # headroom must be >= 1.0× recent ATR
EXEC_HIDDEN_SELLER_MAX: float = 0.5           # gate trips when hidden_seller > this

# T&S telemetry baseline windows
TAPE_BASELINE_WINDOW_SECS: int = 60           # baseline computed from this many seconds
TAPE_BASELINE_MIN_PRINTS: int = 20            # min prints before baseline telemetry is valid

# Static confidence floor — tickers below this after Stage 2 are rejected entirely
STATIC_CONFIDENCE_MIN: float = 60.0

# Eval and live confidence required separately for entry.
ENTRY_CONFIDENCE_MIN: float = 50.0

# When True, confidence thresholds are ignored for entry/exit decisions.
# Eval/live confidence is still calculated and displayed, but entry_allowed and
# exit signals skip eval/live confidence gates. Toggle via "conf stop/go".
# The scoped flags bypass only one of the two gates ("conf eval off" /
# "conf live off"); CONFIDENCE_BYPASS=True overrides both regardless.
CONFIDENCE_BYPASS: bool = False
CONFIDENCE_BYPASS_EVAL: bool = os.getenv("CONFIDENCE_BYPASS_EVAL", "").lower() in ("1", "true")
CONFIDENCE_BYPASS_LIVE: bool = os.getenv("CONFIDENCE_BYPASS_LIVE", "").lower() in ("1", "true")


def confidence_bypass_eval() -> bool:
    """Effective bypass state for the eval_confidence gate (read at call time)."""
    return CONFIDENCE_BYPASS or CONFIDENCE_BYPASS_EVAL


def confidence_bypass_live() -> bool:
    """Effective bypass state for the live_confidence gate (read at call time)."""
    return CONFIDENCE_BYPASS or CONFIDENCE_BYPASS_LIVE

# =============================================================================
# ORDER FLOW TRACKER — windows + tuning
# =============================================================================
# All knobs for exec.order_flow.OrderFlowTracker live here so tuning
# is single-file.  Window sizes are seconds; ratios/tolerances are
# in their natural units.

OFI_WINDOW_SECS:     int   = 30      # rolling window for OFI summation
PRESSURE_WINDOW_SECS: int  = 10       # rolling window for T&S pressure + trade flow (signal window)
PRESSURE_ACTIVITY_WINDOW_SECS: int = 30  # wider window for directional activity count
BOOK_LEAN_EMA_ALPHA: float = 0.15    # EMA smoothing for book lean

# --- Level-2 depth OFI (IBKR market depth) ---
# Multi-level OFI per Cont/Kukanov/Stoikov extended to the top M book levels.
# Each level's flow contribution is weighted by OFI_L2_LEVEL_DECAY^(level-1)
# so the inside quote dominates while queue-building deeper in the book still
# registers. Depth events feed the same rolling OFI window as L1 events.
OFI_L2_LEVEL_DECAY:  float = 0.7     # geometric weight per book level below the inside
OFI_DEPTH_STALE_SECS: float = 30.0   # depth feed considered stale/inactive after this gap
ORDER_BOOK_MIN_DEPTH_SHARES: int = 500  # min total displayed shares (both sides) for order_book_density_ok

# --- OFI gate estimator selection ---
# "boxcar" = ofi_normalized: 30s rolling sum / 30s mean displayed depth (legacy).
# "ema"    = ofi_normalized_ema: exponentially forgotten flow sum / EMA mean
#            depth at OFI_EMA_TAU_SECS — per-event updates, no window cliff.
# Default "ema" after replay calibration on 30 cached L2 sessions (07/27-08/06,
# 772 zigzag squeeze legs >= 4%): at matched noise-episode counts vs the
# boxcar>=1.0 gate, coverage 347 vs 333 legs, added-delay p90 36.6s vs 44.3s,
# stale-ON (gate still passing on dead tape, speed < 3/s) 6.8% vs 14.9% —
# the boxcar's failure mode is a 30s-old burst squatting in the window after
# flow dies. Logged telemetry (tape CSV ofi_normalized) stays boxcar; the
# EMA read is logged alongside as ofi_norm_ema.
OFI_GATE_SOURCE: str = os.getenv("OFI_GATE_SOURCE", "ema").lower()
OFI_EMA_TAU_SECS: float = 10.0   # ~10s memory: tau=5 measured +40 noise episodes (twitchy);
                                 # tau=10 keeps the boxcar's noise rejection with EMA reactivity

# --- Tape acceleration estimator selection ---
# "2bin" = balanced two-half ratio with z = accel*sqrt(N) significance (legacy).
# "ema"  = dual exponential kernel intensity ratio (MACD-style) with Kish
#          effective-N significance — smoother, per-print updates, no bin edges.
# Default "ema" after replay calibration on 20 top-mover tapes (07/06-07/23):
# surge detection 95% vs 85%, p90 onset latency 0.96s vs 3.40s, ~1/3 the
# false fires of 2bin at matched thresholds. "2bin" stays available for A/B.
TAPE_ACCEL_ESTIMATOR: str  = os.getenv("TAPE_ACCEL_ESTIMATOR", "ema").lower()
# --- Entry speed-gate source ---
# "ema"    = tape_speed_ema: instantaneous directional prints/sec from the fast
#            exponential kernel (tau=2s). 40-tape replay vs the boxcar at the
#            same 10/s floor: surge arming latency med 0.43s/p90 0.90s vs
#            2.26s/7.95s, never-armed-during-surge 1.6% vs 56.9% (surges often
#            end before the 10s window fills), stale passes on dead tape
#            halved (8.0% vs 15.2%), pooled pass rate ~equal (0.838 vs 0.818).
# "boxcar" = legacy trade_speed: directional prints / fixed 10s window
#            (remains the logged/backtest-parity telemetry quantity either way).
TAPE_SPEED_GATE_SOURCE: str = os.getenv("TAPE_SPEED_GATE_SOURCE", "ema").lower()
# --- Entry buy-ratio gate source ---
# "ema"    = buy_ratio_ema: fast-kernel (tau=2s) count ratio. 40-tape replay
#            vs the 10s boxcar at the same 0.60 floor (pass rates 0.409 vs
#            0.396): noise-passes (gate ON while centered +/-3s truth <= 0.50)
#            0.031 vs 0.067, flip-stale (truth <= 0.40) 0.006 vs 0.020, surge
#            coverage 0.840 vs 0.724, full-gate surge hits 332 vs 314.
#            (Wilson-LB significance gating was also measured and LOST: phi
#            spikes during bursts crush n_eff exactly at onset - cov 0.345.)
# "boxcar" = legacy buy_ratio (counts / fixed 10s window), which remains the
#            logged/backtest-parity telemetry quantity either way.
TAPE_RATIO_GATE_SOURCE: str = os.getenv("TAPE_RATIO_GATE_SOURCE", "ema").lower()
TAPE_EMA_FAST_TAU_SECS:  float = 2.0   # fast kernel: ignition bursts ramp over ~1-2s (CETX trace; sweep confirms)
TAPE_EMA_SLOW_TAU_SECS:  float = 10.0  # slow kernel: matches the legacy 10s pressure window baseline
TAPE_ACCEL_EMA_MIN_NEFF: float = 10.0  # Kish effective-N floor before EMA accel is trusted
# Dispersion correction (phi-hat): decayed CV^2 of directional inter-arrival
# gaps at the slow tau. Kish n_eff assumes independent arrivals; real tape is
# batched/clustered (40-tape replay: median phi ~4.9, range 2.3 to ~100 on
# burst-pathological tape like VSME 06/10), inflating z by ~sqrt(phi) by
# unequal amounts per tape. Dividing z by sqrt(phi-hat) equalizes the null:
# cross-tape CV of the z pass rate drops 29% (0.36 -> 0.26) at matched
# overall strictness, with no material detection/latency/false-fire change
# through the full entry gate.
TAPE_PHI_CLIP_HI:        float = 8.0   # cap on phi-hat — burst-pathological tape would otherwise fully desensitize the gate
TAPE_PHI_MIN_WEIGHT:     float = 8.0   # min decayed gap mass before cv2 is trusted (carries last estimate below; 1.0 if never warm)

# --- Hidden seller / buyer detection (iceberg refill at the L1 quote) ---
HIDDEN_WINDOW_SECS:    int   = 15    # window over which we evaluate "is the price still pinned?"
HIDDEN_KEEP_SECS:      int   = 30    # how long to retain one-sided book samples
HIDDEN_PRICE_TOL:      float = 0.005 # tolerated micro-flicker on the pinned side ($)
HIDDEN_RATIO_FLOOR:    float = 1.0   # ratio at/below which intensity = 0
HIDDEN_RATIO_CEILING:  float = 5.0   # ratio at/above which intensity = 1
HIDDEN_MIN_SAMPLES:    int   = 3     # need this many quote samples in window
HIDDEN_MIN_HALF_WINDOW_SECS: float = 7.5  # need at least this much window span
HIDDEN_BOOK_SAMPLE_SECS: float = 0.25  # decimation for book-driven ask samples — matches the ~4/s
                                       # L1 quote cadence the detector was tuned on, and bounds the
                                       # per-read cost (hidden_seller walks the window every print)


# =============================================================================
# SIMULATION ENGINE — execution simulation on live feed
# =============================================================================

SIM_ENABLED:                bool = os.getenv("SIM_ENABLED", "true").lower() == "true"
SIM_LOG_DIR:                str  = os.path.join(os.path.dirname(__file__), "logs", "sim")
SIM_BACKTEST_LOG_DIR:       str  = os.path.join(SIM_LOG_DIR, "backtest")

# --- Backtest replay windowing (exec/backtest.py) ---
# A derived pullback replay window is padded by this many bars of the
# detecting timeframe when it closes, so the resumption/breakout bars right
# after the pullback — where entry evaluation actually happens — are always
# inside replay coverage (2026-08-05 YXT: the 05:19 resumption bar and the
# 05:23:56 live entry moment both fell into one-bar gaps between windows).
BACKTEST_PULLBACK_WINDOW_TAIL_BARS: int = 2
# Consecutive replay windows separated by at most this many seconds are
# coalesced into one, so a single bar marginally past the near-HOD ATR
# threshold can't punch a sub-minute hole in coverage (YXT 12:22 closed
# 2.03 ATRs off HOD vs the 2.0 limit → 05:23:00-05:23:58 uncovered).
BACKTEST_REPLAY_WINDOW_BRIDGE_SECS: int = 60

# Setup rejection
SETUP_REJECTED_LEVEL_TOL_PCT: float = 0.005  # 0.5% tolerance for rejected-level matching

# Position sizing (simulated)
SIM_ACCOUNT_SIZE:           float = 2500.0    # simulated buying power
SIM_MAX_POSITION_PCT:       float = 1.00        # max 100% of account per trade
SIM_MAX_CONCURRENT:         int   = 3           # max simultaneous positions

# Entry signal thresholds (all T&S-derived)
SIM_PRESSURE_BUY_RATIO_MIN: float = 0.60  # buy/(buy+sell) ratio required (60% = buyer dominance)
SIM_PRESSURE_VOL_IMBALANCE_MIN: float = -0.20  # entry blocked when 30s VOLUME flow (trade_imbalance)
                                               # <= this (buy share <= 0.40): guards count-bullish /
                                               # volume-bearish divergence (distribution tape). 40-tape
                                               # replay: such fires had -0.89% mean fwd60 vs +0.58%
                                               # clean; guard cost only 14/333 surge hits (16/20 tapes
                                               # directionally consistent, bootstrap CI excl. zero)
# Accel significance floor, estimator-conditional:
#   2bin: z = accel*sqrt(N) >= 2.0 (318-break calibration: continued ~2.2 vs failed ~1.3)
#   ema:  z = accel*sqrt(Kish n_eff / phi-hat) >= 1.07 — dispersion-corrected
#         significance (see TAPE_PHI_CLIP_HI). Percentile-matched on the
#         40-tape replay to the former uncorrected 2.5 gate (t* = 1.067):
#         equal overall strictness, cross-tape CV of the pass rate -29%
#         (0.36 -> 0.26). The old 2.5 was 2.5 Kish-sigmas ~= 1.07 true sigmas
#         on median-dispersion tape; the correction makes that explicit and
#         stops the z gate double-filtering thin (low-clustering) tape that
#         the speed/activity gates already own.
SIM_PRESSURE_ACCEL_Z:       float = 1.07 if TAPE_ACCEL_ESTIMATOR == "ema" else 2.0
SIM_PRESSURE_SPEED_MIN:     float = 10.0  # minimum directional prints/sec (active tape); read from the source
                                          # selected by TAPE_SPEED_GATE_SOURCE (fast-EMA rate by default — same
                                          # units/steady-state meaning as the boxcar, ~5x faster to arm/disarm)
SIM_PRESSURE_ACTIVITY_MIN:  int   = 50   # absolute floor on total prints — ratio/accel meaningless below this
# --- Trailing price-progress qualifier (churn veto) ---
# The tape/OFI gates measure FLOW; this is the only gate that asks whether
# PRICE responded. Entry requires net trade-price progress over the trailing
# window: blocks fires where heavy tape churns against a pinned price
# (absorption), the failure shape of 2026-08-05/06 entries. Swept on 30 cached
# L2 sessions x 1311 chain-fire starts (progress_sweep): thresholds trade
# win-rate for squeeze coverage smoothly — 15s/0.5% keeps 85% of fires and
# 277/292 covered legs for P(fwd90>=2%) 48.3% vs 47.0% and P(>=4%) 25.5% vs
# 24.3%. Steeper floors (1%) reach ~51% but drop 10-17% of covered legs —
# rejected: PnL here is dominated by catching big legs, not per-fire win rate.
# Honest limit: at the instant the tape gates first align, NO instantaneous
# conditioner moved P(win) past ~48-51% on this dataset (supply-withdrawal
# reads and an ARM/TRIGGER consolidation-breakout restructure were both
# measured and REFUTED) — this qualifier trims churn entries, it does not
# solve burst-top timing.
SIM_PRESSURE_PROGRESS_MIN:  float = 0.005  # trailing net progress must EXCEED this
PROGRESS_WINDOW_SECS:       int   = 15     # lookback for the progress read
PROGRESS_MIN_SPAN_SECS:     float = 10.0   # need >= this much directional-tape history,
                                           # else the read is not ready and the gate fails
                                           # closed (a seconds-old awakening is exactly the
                                           # cold burst this gate exists to not chase)
SIM_OFI_NORM_MIN:           float = 0.70 if OFI_GATE_SOURCE == "ema" else 1.0
                                          # Entry floor on ofi_norm_gate (depth-normalized net flow, estimator per
                                          # OFI_GATE_SOURCE). EMA value tuned on 30 cached L2 sessions / 772 squeeze
                                          # legs (zigzag low->high >= 4%, >= 2% still ahead, depth-active): the sweep
                                          # 0.2..0.8 trades coverage for noise monotonically; 0.70 matches the old
                                          # boxcar>=1.0 gate's noise-episode count (221 vs 225) while covering +14
                                          # legs and halving stale-ON on dead tape (6.8% vs 14.9%). Boxcar floor 1.0
                                          # retained for OFI_GATE_SOURCE=boxcar (original Cont et al. impact prior:
                                          # ~1.8 ticks of implied book pressure per 30s).
SIM_SETUP_SCORE_MIN:        float = 40.0  # minimum setup_pattern_score for entry
SIM_MAX_SETUP_EXTENSION_PCT: float = 0.05 # max price extension above setup level before sim entry
# Anticipation band BELOW the setup level: entry may fire while price is still
# up to this fraction under the new-high trigger, when the full tape gate
# already confirms (buyers absorbing the dip before the break). Confirmation
# is the trigger — the entry band is [level*(1-anticipation), level*(1+ext)];
# the first moment all gates pass inside it enters. Below-level entries get
# the best prices in the structure but add one failure class (confirmed
# pullbacks that never break — the tape exits own those), so keep this SMALL:
# it is anticipation at the level, not dip-buying inside the flag.
SIM_SETUP_ANTICIPATION_PCT: float = 0.02  # max dip below setup level for entry, as a FRACTION of the
                                          # level like SIM_MAX_SETUP_EXTENSION_PCT above (0.02 = 2%).
                                          # NOT whole percent: 2.0 would mean 200% and disable the floor.

# --- Armed-band entry regime (2026-08-14 redesign) --------------------------
# Measured on the 08-11 live-sim session: the AND-of-instants gate form plus
# the fixed 2% anticipation floor made entries pay a +1.5–6.2% premium over
# the first moment full evidence existed (pullbacks run 7–13% deep, so
# confirmations completing below level*0.98 were wasted; the six-EMA
# conjunction only aligns mid-vertical-burst — a burst-apex selector).
# The armed_band regime re-times the SAME gates (floors unchanged, per the
# 2026-08-05 directive): slow evidence is latched "true recently while price
# is in the band", one fast trigger fires the entry, and required evidence
# scales with extension (strictest BELOW the level, loosest at the cross,
# full legacy stack when extended).
SIM_ENTRY_REGIME: str = os.getenv("SIM_ENTRY_REGIME", "legacy").lower()
                                          # "legacy" | "armed_band" | "posterior". Default stays legacy
                                          # until the replay A/B guardrails pass (flip is its own commit).
                                          # "posterior" (2026-08-23, Upgrade 1): the tape/OFI AND-stack
                                          # is replaced by one fitted probability (exec/posterior.py,
                                          # artifact exec/entry_posterior.json); structural gates stay.
SIM_POSTERIOR_THRESHOLD:    float = float(os.getenv("SIM_POSTERIOR_THRESHOLD", "0") or 0)
                                          # posterior regime entry floor override. 0 = use the fitted
                                          # threshold stored in the model artifact (label, criterion and
                                          # provenance live in its meta — research/fit_entry_posterior.py).
SIM_ANTICIPATION_MAX_PCT:   float = 0.06  # armed_band floor: max(pullback_low, level*(1-this)).
                                          # 6% ≈ where the 08-11 AIHS full-stack confirmation lived
                                          # (green @3.59 vs level 3.83 = 6.3% under); the pullback low
                                          # still binds when shallower.
SIM_LEVEL_CROSS_TOLERANCE_PCT: float = 0.05  # "at the level" zone = [level, level*(1+this)] — the
                                          # break itself. Above it counts as extended.
SIM_ARM_EVIDENCE_TTL_SECS:  float = 10.0  # NORMAL arm: every evidence gate must have been true within
                                          # this many seconds while price was in the band.
SIM_ARM_STRONG_TTL_SECS:    float = 5.0   # STRONG arm (required below the level): tighter recency.
SIM_ARM_STRONG_REQUIRES_WALL: bool = False  # STRONG arm additionally requires ask-wall evaporation at
                                          # the level (only while depth is active). Ships False —
                                          # wall trend is logged as evidence and this flips on A/B
                                          # data, not intuition.
SIM_TRIGGER_TTL_SECS:       float = 2.0   # a trigger edge (accel rising-edge / micro-break) is
                                          # consumable for this long after it stamps.
SIM_TRIGGER_TRAIL_WINDOW_SECS: float = 10.0  # micro-break trigger: print price exceeding the trailing
                                          # high of this window (the "break of the previous candle
                                          # high" of the manual method, at 10s-candle scale). Ratchet
                                          # runs on RAW prints — a 1s-decimated series misses
                                          # verticals that clear the high between samples.
SIM_TRIGGER_TRAIL_MIN_SPAN_SECS: float = 3.0  # trailing-high window must span at least this much
                                          # history before a break can stamp (a two-print-old "high"
                                          # is not a level).
SIM_WALL_BAND_PCT:          float = 0.01  # ask-wall read: sum displayed ask shares within ±this
                                          # fraction of the setup level. The book is IMPEDANCE — this
                                          # measures the specific supply defending the level under
                                          # attack (a pure book read anchored at the level; no
                                          # cancel-vs-fill attribution needed — consumed and cancelled
                                          # walls are both bullish, which sidesteps the ~500ms
                                          # tape-vs-book flush skew that broke the 08-10 withdrawal
                                          # instrument).
SIM_WALL_EVAP_MIN:          float = 0.5   # wall evaporation trend = 1 - current_mass/baseline_EMA.
                                          # >= this (mass at least halved vs its recent baseline)
                                          # counts as "sellers at the level are evaporating".
                                          # Consulted only when SIM_ARM_STRONG_REQUIRES_WALL — the
                                          # wall is ARM EVIDENCE, never a hard gate (most replay
                                          # sessions carry no book data; depth-inactive is vacuous).
SIM_WALL_REPLENISH_VETO:    float = -1.0  # fresh-seller veto (user 2026-08-14: "dip could get bought
                                          # strongly but then suddenly a huge seller pops up on the
                                          # level 2"). trend <= this (current ask mass >= 2x its
                                          # recent baseline) REFUSES an armed-zone entry at the
                                          # trigger — do not take the break into a wall that just
                                          # parked. Asymmetric to the evidence read: never REQUIRED
                                          # (vacuous without depth/coverage), but a strongly
                                          # replenishing wall vetoes even a perfect tape.
SIM_FIRST_MACD_CYCLE_ONLY:  bool  = True   # only trade the FIRST 1m MACD-V cycle of the move: once the 1m
                                          # MACD-V histogram completes its first positive->negative flip on
                                          # CLOSED 1m bars since move start, ALL new entries (every setup
                                          # timeframe) are blocked — later legs are untouchable even if the
                                          # histogram turns positive again. Latched by DynamicScorer
                                          # (entry.macd_cycle_done), recomputed each candle refresh from the
                                          # session 1m frame + move anchor so a stale cached flag or a
                                          # late-detected move start self-corrects. Replaces the old
                                          # SIM_10S_EARLY_PHASE_ONLY 10s-only first-1m-pullback gate.

# --- Thesis-lifecycle exit regime (2026-08-14, pairs with armed_band entry) --
# User principle: "if the price doesn't move up quickly soon after you buy,
# you timed it wrong." An entry is a hypothesis about immediate follow-through,
# so the exit is a lifecycle, not a grace patch:
#   UNPROVEN (from fill): legacy instant flow exits still apply (bail fast on
#     failed bursts), plus a structural stop at the snapshotted band floor and
#     a prove-it timeout — level must break, then progress must appear.
#   PROVEN (target reached): flow-lapse exits must SUSTAIN (no single-blip
#     shakeouts mid-squeeze); the structural stop trails up to the broken
#     level's anticipation band (level no longer holding = thesis dead).
SIM_EXIT_REGIME: str = os.getenv("SIM_EXIT_REGIME", "legacy").lower()
                                          # "legacy" | "thesis_lifecycle". Flips with the entry regime
                                          # after the replay A/B passes.
SIM_STRUCT_STOP_BUFFER_PCT: float = 0.005 # structural stop fires when price breaks the stop reference
                                          # by this fraction (noise margin under the floor/trail).
SIM_PROVE_BREAK_SECS:       float = 20.0  # below-level fills: the level must TRADE within this many
                                          # seconds of fill, else the anticipated break never came.
SIM_PROVE_PROGRESS_SECS:    float = 30.0  # once the level has traded (at fill, for at/above-level
                                          # fills): price must reach the prove target within this.
SIM_PROVE_PROGRESS_PCT:     float = 0.01  # prove target = max(fill, level) * (1 + this) — the break
                                          # must EXTEND, not just touch.
SIM_EXIT_LAPSE_SUSTAIN_SECS: float = 3.0  # PROVEN phase: pressure_reversal / accel_collapse /
                                          # ofi_negative must hold continuously this long before the
                                          # exit fires (GLE 08-11: a 1-tick accel breath 21s into a
                                          # proven-shaped hold exited -3.9% before a +9.5% move).
# Peak reversal (user 2026-08-14: "what if it squeezes fast before 3
# seconds"): on a fast vertical the sustained-lapse rule would wait out its
# 3s while the level-anchored trail sits far below the spike high — the
# whole spike could round-trip on a violent blow-off reversal. The exit is
# the conjunction of two INSTANT facts: price has given back
# SIM_TRAIL_PEAK_GIVEBACK_PCT off the peak AND a flow-lapse condition is
# true at that very tick (single-instant, not the 3s sustained form). A real
# blow-off shows both within a tick or two -> exits near the top; a healthy
# mid-squeeze wiggle (flow still green) holds. Price-off-peak ALONE is
# deliberately NOT a stop: measured on the 9-day replay it decapitated the
# +22% winner at +3% (3% wiggles are normal breathing on these names).
SIM_TRAIL_PEAK_ARM_PCT:     float = 0.03  # peak must exceed max(fill, level) by this to arm
SIM_TRAIL_PEAK_GIVEBACK_PCT: float = 0.03 # peak-reversal giveback fraction off the armed peak

# Exit signal thresholds
SIM_EXIT_SELL_RATIO_MIN:    float = 0.60  # sell/(buy+sell) ratio for exit (60% sells = seller dominance)
# Accel-collapse exit floor, estimator-conditional (percentile-matched on the same
# 20-tape replay: P(2bin accel <= -0.11 | ready) = 28.5% -> EMA quantile = -0.066):
SIM_EXIT_ACCEL_MAX:         float = -0.07 if TAPE_ACCEL_ESTIMATOR == "ema" else -0.11
SIM_EXIT_ACCEL_BUY_RATIO_MAX: float = 0.50 # accel exit only once buy DOMINANCE IS LOST (F6:
# read from ratio_gate — same estimator as the entry gate — banded below the 0.60 entry
# floor (in >= 0.60, collapse-out <= 0.50). Measured on 2212 paired one-shot trades across
# 40 tapes vs the old boxcar <= 0.60 knife edge: <=5s instant exits 23.6% -> 18.3%,
# post-exit 30s give-up halved (0.118% -> 0.047%), left tail unchanged (p10 -1.61 both).
# Boxcar-band (0.50) and Wilson-LB exits were rejected: their pooled-mean gains were
# hold-time/sample-drift confounds paid for with a deeper left tail (p10 -2.0 to -2.6)
# — unacceptable with no stop-loss. Revisit toward confirmation semantics per depth_active
# once OFI-L2 sits in front of the tape exits (F8).
SIM_EXIT_OFI_NORM_MAX:      float = -0.75 if OFI_GATE_SOURCE == "ema" else -0.6
                                          # ofi_norm_gate below this triggers the adverse-flow exit — ONLY once buy
                                          # dominance has also lapsed (ratio_gate <= SIM_EXIT_ACCEL_BUY_RATIO_MAX,
                                          # same band as the accel-collapse exit). Tuned on 259 entered legs >= 6%:
                                          # the old unconditional boxcar<-0.6 exit fired mid-squeeze (>=1.5% still
                                          # ahead) on 30.5% of entered legs — a book-flow dip during an ongoing
                                          # squeeze is noise when the tape is still buy-dominant. EMA<-0.75 with the
                                          # ratio-lapse condition: premature 23.6%, median standalone capture 1.74%
                                          # vs 1.27%, post-peak reaction 23s vs 24s at the same miss rate.
SIM_EXIT_CONFIDENCE_MIN:    float = 40.0  # eval/live confidence (0-100 scale) below this → exit.
                                          # 10-pt hysteresis under ENTRY_CONFIDENCE_MIN (50) so a marginal
                                          # confidence wobble can't churn entry→exit. (Was 0.4 — a 0-1 scale
                                          # remnant that could never fire against 0-100 confidence values.)
SIM_EXIT_RESISTANCE_ATR:    float = 0.1   # exit when within 0.1 ATR of resistance + pressure confirms

# Signal dwell (confirmation hold)
# A triggered signal must stay CONTINUOUSLY triggered for this many seconds
# before it fills. The engine re-evaluates on every T&S print; the first
# print at which the signal goes from not-triggered to triggered starts the
# hold, any print that fails the signal resets it, and the first print at or
# past the hold executes. Fills therefore land on a print, never on a timer:
# a signal that arms and dies inside the window with no further tape never
# fires (correct — the tape it was reading has stopped).
# Rationale: the entry gates read fast estimators (buy_ratio_ema tau=2s,
# tape_speed_ema tau=2s, OFI EMA tau=10s) that can satisfy the full chain on
# a single-print spike. The dwell demands the condition persist rather than
# flicker. 0.0 disables the hold entirely (fill on the first triggered print,
# the pre-dwell behaviour).
# NOTE: this is confirmation, not latency — it delays the DECISION, whereas
# SIM_FILL_LATENCY_MS models the round trip after the decision is made. They
# compose: total time from first-arming print to fill is dwell + latency.
SIM_ENTRY_DWELL_SECS:       float = 0.25  # entry signal must hold this long before buying
SIM_EXIT_DWELL_SECS:        float = 0.25  # exit signal must hold this long before selling
SIM_ENTRY_DWELL_UPMOVE:     bool  = True  # entry dwell also requires mid-price progress over
                                          # the hold: mid dropping below the arm anchor re-arms
                                          # (fresh clock, new anchor at the lower mid); the fill
                                          # requires mid STRICTLY above the anchor, so a stalled
                                          # mid extends the hold and never fills. Mid-quote, not
                                          # last print — prints alternate bid/ask, so a last-
                                          # price anchor would read ordinary sell prints as
                                          # "came back down" and reset on every spread bounce.
                                          # The 15s tape_progress gate can't do this job: it
                                          # trails too long to see a spike stalling inside a
                                          # sub-second hold (the climax-top entry shape).
                                          # Exit dwell is deliberately indicator-only (no
                                          # stop-loss; don't block protective exits on a stall).

# Fill simulation
SIM_LATENCY_FILL:           bool  = False # enable latency-based fill simulation
SIM_FILL_LATENCY_MS:        int   = 150   # simulated fill latency in milliseconds
SIM_FILL_OFFSET_CENTS:      int   = 1     # price offset in cents for latency fills
SIM_FILL_SLIP_BPS:          int   = 5     # simulated slippage in bps (applied to mid)
SIM_USE_ASK_FOR_BUY:        bool  = True  # True = fill at ask; False = fill at mid
SIM_USE_BID_FOR_SELL:       bool  = True  # True = fill at bid; False = fill at mid

# Cooldown
SIM_COOLDOWN_SECS:          int   = 0    # min seconds between trades on same symbol
SIM_MAX_TRADES_PER_SYMBOL:  int   = 0     # 0 = unlimited entries per symbol per session


# =============================================================================
# API — HTTP TIMEOUTS
# =============================================================================

HTTP_TIMEOUT:        int = 10   # default REST request timeout in seconds (all clients)
HTTP_STREAM_TIMEOUT: int = 30   # streaming GET connection timeout (Tradier live stream)

# =============================================================================
# DATA LOOKBACK WINDOWS
# =============================================================================

CANDLES_1MIN_LOOKBACK:   int = 30    # number of 1-min bars to pull
CANDLES_10SEC_LOOKBACK:  int = 30    # number of 10-sec bars to pull
CANDLES_15MIN_LOOKBACK:  int = 64    # number of 15-min bars to pull - Caution, this should go through the night to request rest of candles from previous day
CANDLES_DAILY_LOOKBACK:  int = 360    # number of daily bars to pull (for 50/200 MA)
AVG_VOLUME_DAILY_BARS:   int = 50    # daily bars averaged for avg_volume (IBKR quotes lack avVolume)

# Historical REST lookback windows — how far back to request from Tradier
# to guarantee enough bars for the above counts
CANDLES_1MIN_FETCH_DAYS:   int = 5   # 5 trading days of 1-min data
CANDLES_5MIN_FETCH_DAYS:   int = 18  # Tradier max for 5-min with session_filter=all
CANDLES_10SEC_FETCH_HOURS: int = 2   # 2 hours of tick data (resampled to 10-sec)
CANDLES_15MIN_FETCH_DAYS:  int = 10  # 10 days of 15-min data

# RVOL-at-time — number of historical trading periods to average
RVOL_AT_TIME_PERIODS: int = 10

# Session bar counts — used to normalise volume when computing RVol per bar
BARS_PER_SESSION_1MIN: int = 390   # 6.5 hours * 60 min = 390 bars
BARS_PER_SESSION_5MIN: int = 78    # 6.5 hours * 12 bars/hr = 78 bars

# Live candle buffer sizes
LIVE_CANDLE_MAX_BARS:    int = 50   # tradier_stream.py build_live_candle_buffer() rolling cap

# =============================================================================
# SCANNERS — BACKGROUND INPUT STREAMS
# =============================================================================

# Momoscreener scanner
SCANNER_MOMO_INTERVAL: int      = 10   # seconds between polls (default = def)
SCANNER_MOMO_INTERVAL_FAST: int = 2    # momo fast
SCANNER_MOMO_INTERVAL_DEF: int  = 5   # momo def
SCANNER_MOMO_INTERVAL_SLOW: int = 15   # momo slow
SCANNER_MOMO_MIN_CHG:  int   = 3    # minimum % change for momentum scanner
SCANNER_MOMO_HITS_PER_OUTPUT: int = 2  # new snapshots required per hit-summary print

# IBKR top-gainers scanner (ib.reqScannerData — one-shot subscription per poll).
# Each poll re-runs the scan server-side, so fast polling returns fresher
# results than reqScannerSubscription, whose push updates arrive on a fixed
# ~30s cadence. Scans are not historical requests and are untouched by the
# Gateway's historical traffic shaping, so the interval is the only throttle.
SCANNER_IBKR_INTERVAL_FAST: int = int(os.getenv("SCANNER_IBKR_INTERVAL_FAST", "2"))  # ibkr fast
SCANNER_IBKR_INTERVAL_DEF: int  = 30   # ibkr def
SCANNER_IBKR_INTERVAL_SLOW: int = 60   # ibkr slow
SCANNER_IBKR_INTERVAL: int = SCANNER_IBKR_INTERVAL_FAST  # seconds between scans (default = fast)
# The change floor IS the pre-evaluation change gate now — the quote
# prefilter no longer rejects on % change, so anything the scan surfaces
# gets a full evaluation. Full CHANGE_PCT_MIN, not half, keeps that spend
# bounded; the volume floor stays at half so a name is discovered while
# volume is still building (rvol fails are soft, hence retryable). Derived,
# not literal, so retuning CHANGE_PCT_MIN / VOLUME_MIN moves the scanner.
SCANNER_IBKR_MIN_CHG:    float = CHANGE_PCT_MIN       # 10% — the real gate
SCANNER_IBKR_MIN_VOLUME: int   = VOLUME_MIN // 2      # 50k vs. VOLUME_MIN=100k
SCANNER_IBKR_ROWS:     int = 25   # rows requested per scan (IBKR caps at 50)
SCANNER_IBKR_INSTRUMENT: str = "STK"
# The scan code is chosen per poll, by session: TOP_PERC_GAIN ranks by
# regular-session change only, so outside RTH it just replays the 16:00
# close leaderboard (verified live 2026-08-04: the reigning AH runner was
# absent from TOP_PERC_GAIN's top 25 but rank 0 on the AH variant). Setting
# SCANNER_IBKR_SCAN_CODE pins one code for the whole run. Discover valid
# codes via ib.reqScannerParameters().
SCANNER_IBKR_SCAN_CODE_PIN: str = os.getenv("SCANNER_IBKR_SCAN_CODE", "")
SCANNER_IBKR_SCAN_CODE_RTH: str = "TOP_PERC_GAIN"
# Premarket reuses TOP_PERC_GAIN, and that is measured, not assumed: there is
# no pre-open scan code in the Gateway's reqScannerParameters() list, and run
# live in premarket (2026-08-05 07:1x ET) TOP_AFTER_HOURS_PERC_GAIN returned
# 0 rows even UNFILTERED — the AH scan dies when the 04:00 session opens —
# while TOP_PERC_GAIN returned the live premarket leaderboard (change vs
# prior close on premarket trades). The "replays the frozen 16:00
# leaderboard" behavior above is a post-close artifact only.
SCANNER_IBKR_SCAN_CODE_PRE: str = "TOP_PERC_GAIN"
SCANNER_IBKR_SCAN_CODE_EXT: str = "TOP_AFTER_HOURS_PERC_GAIN"
SCANNER_IBKR_LOCATION:  str = os.getenv("SCANNER_IBKR_LOCATION", "STK.US.MAJOR")


def scanner_ibkr_scan_spec() -> tuple[str, str]:
    """(scan code, change-filter tag) for the session this poll runs in.

    Three sessions, two codes: RTH and premarket both rank change vs the
    prior close via TOP_PERC_GAIN (kept as separate constants because the
    premarket choice is its own measured decision — see above), while
    post-close switches to the after-hours scan.  Collapsing premarket into
    the after-hours branch scans a session that has already rolled and comes
    back empty.

    The tag must travel with the code: ``changePercAbove`` on the after-hours
    scan still filters by regular-session change, which would hide a name
    that only started moving after the close.
    """
    if SCANNER_IBKR_SCAN_CODE_PIN:
        code = SCANNER_IBKR_SCAN_CODE_PIN
    elif MarketClock.is_rth():
        code = SCANNER_IBKR_SCAN_CODE_RTH
    elif MarketClock.is_premarket():
        code = SCANNER_IBKR_SCAN_CODE_PRE
    else:
        code = SCANNER_IBKR_SCAN_CODE_EXT
    tag = "afterHoursChangePercAbove" if "AFTER_HOURS" in code else "changePercAbove"
    return code, tag
SCANNER_IBKR_REAPPEAR_SECS: int = 900  # a name off the list this long counts as new again
# A watched name absent from the ranked list this long stops being watched:
# scorer stopped, entry moved to FADED, and re-queueable if it climbs back.
# Above the poll interval so ordinary churn around the last rank slot (a
# name flickering in and out at rank 24-26) does not trip it, and well
# below REAPPEAR_SECS so the scorer is released long before the novelty
# tracker forgets the symbol.
SCANNER_IBKR_DROP_SECS: int = 30

# YouTube live-chat scanner
SCANNER_YT_CHANNEL: str = os.getenv("SCANNER_YT_CHANNEL", "Zendoo")  # YouTube channel handle (without @) — live stream URL is auto-resolved at startup
SCANNER_YT_URL:     str = os.getenv("SCANNER_YT_URL", "")  # manual override — if set, skips auto-resolve
SCANNER_YT_INTERVAL: int      = 5   # seconds between chat polls (fallback when auto has no hint yet)
SCANNER_YT_RESOLVE_RETRY_S: int = 30  # retry cadence while waiting for the stream to go live (opens ~06:29 ET, well after the 04:00 session start)
SCANNER_YT_INTERVAL_FAST: int = 2    # yt fast
SCANNER_YT_INTERVAL_DEF: int  = 5   # yt def
SCANNER_YT_INTERVAL_SLOW: int = 10   # yt slow
SCANNER_YT_MIN_MENTIONS: int = 2  # minimum mention count to submit a ticker
SCANNER_YT_NEWS_AUTHOR:  str = "@Clementine\u30fc"  # only this user posts news headlines

# Trusted authors: messages scoring above SCANNER_YT_AUTHOR_SCORE_THRESHOLD via
# FinBERT trigger a Tradier company-name / symbol lookup to resolve a ticker.
SCANNER_YT_TRUSTED_AUTHORS: list[str] = ["@Clementine\u30fc", "@someoneanother3901"]
SCANNER_YT_AUTHOR_SCORE_THRESHOLD: float = 0.25

# Words stripped from the start of a trusted-author message before the first
# meaningful word is extracted for Tradier company/symbol lookup.
SCANNER_YT_HEADLINE_BUZZWORDS: frozenset = frozenset({
    "exclusive", "breaking", "news", "alert", "just", "update", "flash",
    "developing", "report", "reported", "reports", "urgent", "watch",
    "new", "latest", "live", "now",
})

# --- ModernBERT news detection -------------------------------------------
# Zero-shot classifier that flags news posts from *any* chat author, not just
# SCANNER_YT_NEWS_AUTHOR.  Runs in shadow mode by default: every message is
# classified and logged, but nothing is queued — see market_data/chat_logger.py
# and yt_shadow_report.py.  Flip MODE to "active" to merge its detections into
# the news path.  The classifier loads only when
# SCANNER_YT_ENABLED and SCANNER_YT_MODERNBERT_ENABLED.
SCANNER_YT_MODERNBERT_ENABLED: bool = os.getenv("SCANNER_YT_MODERNBERT_ENABLED", "true").lower() == "true"
SCANNER_YT_MODERNBERT_MODE:    str = os.getenv("SCANNER_YT_MODERNBERT_MODE", "shadow").strip().lower()  # "shadow" | "active"
SCANNER_YT_MODERNBERT_MODEL:   str = os.getenv("SCANNER_YT_MODERNBERT_MODEL", "MoritzLaurer/ModernBERT-base-zeroshot-v2.0")
# Precision-biased: shadow logs the raw scores, so retune offline from the
# yt_shadow_report.py threshold table rather than guessing here.
SCANNER_YT_MODERNBERT_THRESHOLD: float = float(os.getenv("SCANNER_YT_MODERNBERT_THRESHOLD", "0.85"))
SCANNER_YT_MODERNBERT_HYPOTHESIS: str = os.getenv("SCANNER_YT_MODERNBERT_HYPOTHESIS", "This live chat message is {}.")
# Two competing hypotheses, softmaxed against each other.  Raw single-label
# entailment is not comparable across sentences (real headlines measured
# anywhere from 0.06 to 0.94); the explicit chit-chat alternative turns it
# into the relative question and separates news 0.90-0.99 from chatter
# 0.01-0.42.  Set the contrast label to "" for the single-hypothesis form.
SCANNER_YT_MODERNBERT_LABEL: str = os.getenv(
    "SCANNER_YT_MODERNBERT_LABEL",
    "a news headline reporting a factual event about a company",
)
SCANNER_YT_MODERNBERT_CONTRAST_LABEL: str = os.getenv(
    "SCANNER_YT_MODERNBERT_CONTRAST_LABEL",
    "a personal opinion, reaction, or chit-chat",
)
SCANNER_YT_MODERNBERT_MIN_WORDS: int = 3   # fewer alphabetic words than this → not worth a forward pass

# Persist raw chat + shadow verdicts to logs/yt_chat/chat_YYYYMMDD.jsonl
SCANNER_YT_CHAT_LOG_ENABLED: bool = os.getenv("SCANNER_YT_CHAT_LOG_ENABLED", "true").lower() == "true"

# IBKR reqMatchingSymbols is paced at ~1 req/s — keep a margin.
SCANNER_YT_SYMBOL_RESOLVE_INTERVAL_S: float = 1.1


# =============================================================================
# CACHE
# =============================================================================

CACHE_TTL_DAYS: int = 7   # ticker cache entries older than this are evicted on next load
