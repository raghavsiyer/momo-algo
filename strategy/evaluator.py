"""
strategy/evaluator.py

Stock evaluation pipeline — two-stage screening with separate eval/live confidence layers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1 — Tradier hard filter (price / volume / rvol)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Price:        $0.10 - $10.00
    % Change:     > 10%
    Volume:       > 100,000
    RVol 1-day:   > 5x
    RVol 5-min:   > 5x

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2 — Enrichment filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Float:     < FLOAT_MAX_M shares (hard cutoff — true float from FMP/AV)
    News:      last 2 trading days (fresh = within 60 min)
    Splits:    reverse split window 30 days
    IPO:       recent = within 365 days

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Two confidence values live on a 0-100 scale and are kept separate:

    1. Eval Confidence: one-time STOCK_EVAL Stage 2 variables such as float, catalyst,
      1-day RVOL, price, and daily trend.
    2. Live Confidence: TECHNICAL_ANALYSIS variables updated by DynamicScorer through
      LiveInputs, such as wick quality, body impulse, tape activity,
      big sellers, spread, hidden seller, and level respect.

These are never aggregated into one composite confidence.  static_confidence is
the initial eval-only snapshot before live data arrives.

Boolean criteria groups (ATTENTION and EXECUTION) sit alongside scoring.
They gate focus/entry but do not add to the confidence score.  For example,
T&S ROC updates the live snapshot, not ATTENTION or the scored
TECHNICAL_ANALYSIS group.
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import pandas as pd

from market_data.provider import get_historical_client
from market_data.tradier_rest import MarketDataUnavailable
from market_data.fmp_client import FloatData, FMPClient
from market_data.alpha_vantage_client import AlphaVantageClient
from market_data.finnhub_client import FinnhubClient, FinnhubNewsArticle
from market_data.finbert_client import FinBERTClient
from market_data.brave_client import BraveNewsClient
from market_data.momo_client import MomoClient
from config import (
    MARKET_DATA_ENABLED, AV_ENABLED, FINNHUB_ENABLED, FINBERT_ENABLED,
    BRAVE_ENABLED, FMP_ENABLED,
    _dim,
    # Stage 1 thresholds
    PRICE_MIN, PRICE_MAX, CHANGE_PCT_MIN,
    # Stage 2 thresholds
    FLOAT_MAX_M, SPLIT_WINDOW_DAYS, IPO_WINDOW_DAYS,
    # Confidence display
    CONFIDENCE_FLOOR, CONFIDENCE_CEILING,
    # Sector keywords (for quality value computation)
    SECTORS_TIER1, SECTORS_TIER2, SECTORS_TIER3,
    # Scoring group definitions
    SCORING_GROUPS,
    # Sentiment helpers
    sentiment_label, sentiment_is_bullish, sentiment_is_bearish,
    news_freshness_cutoffs,
    NEWS_NOISE_THRESHOLD,
    NOISE_FILTER_TITLE,
    NOISE_FILTER_YT_HEADLINE,
    NOISE_FILTER_BRAVE,
    MarketClock,
    RVOL_AT_TIME_PERIODS,
    AVG_VOLUME_DAILY_BARS,
    _MarketCalendar,
    SETUP_PATTERN_GROUP,
    CRITERIA_ATTENTION,
    CRITERIA_EXECUTION,
    ATTENTION_VOLUME_MIN,
    ATTENTION_RVOL_1DAY_MIN,
)
from strategy.scoring import (
    VariableGroup, ScoringVariable, StaticInputs, LiveInputs,
    build_groups,
)
from strategy.criteria import build_criteria_groups
from technical_analysis.patterns import latest_extended_session_start

logger = logging.getLogger(__name__)


def _scored_live_input_names() -> set[str]:
    """LiveInputs fields that are also active scoring variables."""
    scoring_var_names = {
        var_def["name"]
        for group_def in SCORING_GROUPS
        for var_def in group_def.get("vars", [])
    }
    return set(LiveInputs.__dataclass_fields__) & scoring_var_names


_SCORED_LIVE_INPUT_NAMES: set[str] = _scored_live_input_names()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Stage1Result:
    symbol: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int
    rvol_1day: float
    rvol_5min: float
    daily_high: float
    passed: bool
    fail_reason: str = ""


@dataclass
class CompositeEvaluation:
    """Quality-function-based evaluation with static and live scored layers.

    Layer 1 (STOCK_EVAL): static pillars — float, news, rvol, price, daily trend.
    Layer 2 (TECHNICAL_ANALYSIS): live per-TF candle cleanliness — wick quality,
        body impulse, leg smoothness.  Updated by the scorer.

    The legacy class name is kept for cache compatibility; it is a container
    for separate layers, not a source of aggregate confidence.
    """
    stock_eval:          VariableGroup
    technical_analysis:  VariableGroup

    @property
    def groups(self) -> list[VariableGroup]:
        return [self.stock_eval, self.technical_analysis]

    def group_by_name(self, name: str) -> VariableGroup | None:
        for g in self.groups:
            if g.name == name:
                return g
        return None

    # ------------------------------------------------------------------
    # Typed input application
    # ------------------------------------------------------------------

    def apply_static(self, inputs: StaticInputs) -> None:
        """Evaluate all static variables from a typed input struct.

        Fields set to ``None`` are skipped (the variable keeps its
        current quality — useful for optional vars like ``breaking_news``).
        Live variables (technical_analysis) are left unevaluated until the
        scorer provides data.
        """
        _apply_inputs(self.groups, inputs)

    def apply_live(self, inputs: LiveInputs) -> None:
        """Evaluate live variables from a typed input struct.

        Fields set to ``None`` are skipped so callers can do partial
        updates (e.g. only demand vars on a timesale tick, only spread
        on a quote event).
        """
        _apply_inputs(self.groups, inputs)

    def update_news(self, avg_sentiment: float | None, news_count: int = 1,
                    news_source: str = "", noise_gate: bool = False) -> None:
        """Update the ``breaking_news`` catalyst variable.

        Centralises the sentiment → quality mapping and noise-gate logic
        that was previously duplicated across main.py, scanners.py, pipeline.py,
        and evaluator.refresh_news.

        *noise_gate* — if True, the value is silently skipped (variable stays
        excluded as optional with value=None).
        """
        v = self.stock_eval.get("breaking_news")
        if v is None:
            return
        if noise_gate or news_count <= 0 or avg_sentiment is None:
            v.value = None
            return
        v.evaluate((avg_sentiment + 1.0) / 2.0)

    def apply_no_news_penalty(self, quality: float) -> None:
        """Penalize the ``breaking_news`` catalyst when a stock has no news.

        Unlike the default optional/excluded state (used during the grace window
        after start-of-move), this *includes* ``breaking_news`` with a low quality
        so the missing catalyst drags the STOCK_EVAL score down.  Called by the
        scorer once the no-news grace window has elapsed; real news via
        ``update_news`` always overrides it.
        """
        v = self.stock_eval.get("breaking_news")
        if v is None:
            return
        v.value = quality
        v.quality = quality


_fields_cache: dict[type, tuple] = {}


def _apply_inputs(groups: list[VariableGroup], inputs: StaticInputs | LiveInputs) -> None:
    """Shared loop: match dataclass field names to ScoringVariable names."""
    from dataclasses import fields as dc_fields
    cls = type(inputs)
    if cls not in _fields_cache:
        _fields_cache[cls] = dc_fields(cls)
    field_map = {f.name: getattr(inputs, f.name) for f in _fields_cache[cls]}
    for g in groups:
        for v in g.variables:
            if v.name in field_map:
                val = field_map[v.name]
                if val is not None:
                    v.evaluate(val)



def _static_only_score(ev) -> float:
    """Composite score using only static (non-dynamic) vars.

    Groups where all vars are dynamic are excluded entirely so
    uninitialized live vars don't drag the score down.
    """
    import math

    quality_groups: list[tuple[float, float]] = []
    strength_groups: list[tuple[float, float]] = []

    for g in ev.groups:
        w_sum = 0.0
        wq_sum = 0.0
        for v in g.variables:
            if v.name in _SCORED_LIVE_INPUT_NAMES:
                continue
            if v.optional and v.value is None:
                continue
            q = v.quality if v.value is not None else 0.5
            w_sum += v.weight
            wq_sum += v.weight * q
        if w_sum <= 0:
            continue
        group_score = wq_sum / w_sum
        if g.category in ("STRENGTH_BUY", "STRENGTH_SELL"):
            strength_groups.append((g.weight, group_score))
        else:
            quality_groups.append((g.weight, group_score))

    if not quality_groups and not strength_groups:
        return 0.0

    if quality_groups:
        qw = sum(w for w, _ in quality_groups)
        quality = sum(w * s for w, s in quality_groups) / qw if qw > 0 else 0.5
    else:
        quality = 0.5

    if strength_groups:
        sw = sum(w for w, _ in strength_groups)
        if sw > 0:
            log_sum = sum(w * math.log(max(s, 1e-6)) for w, s in strength_groups)
            strength = math.exp(log_sum / sw)
        else:
            strength = 0.5
        return quality * strength
    return quality


@dataclass
class WatchlistEntry:
    symbol: str
    evaluated_at: datetime

    # Stage 1
    price: float
    change_pct: float
    volume: int
    avg_volume: int
    rvol_1day: float
    rvol_5min: float
    daily_high: float

    # Stage 2 — FMP
    float_shares_m: float
    free_float_pct: float
    sector: str
    industry: str
    country: str
    ipo_date: Optional[str]
    days_since_ipo: Optional[int]
    has_recent_reverse_split: bool
    reverse_split_ratio: Optional[float]
    reverse_split_date: Optional[str]

    # Stage 2 — News
    news_count_last_hour: int
    news_is_fresh: bool
    news_is_breaking: bool
    avg_sentiment: Optional[float]

    # Quality-function evaluation (replaces StaticBreakdown)
    evaluation: CompositeEvaluation = field(default=None)  # type: ignore[assignment]
    float_source: str = ""
    static_confidence: float = 0.0  # snapshot at evaluation time (before scorer)
    setup_pattern: Optional[VariableGroup] = field(default=None)
    setup_pattern_name: str = ""
    setup_pattern_timeframe: str = ""
    setup_pullback_depth: Optional[float] = None
    setup_price_level: Optional[float] = None  # leg high the pullback needs to reclaim
    setup_pullback_low: Optional[float] = None  # absolute pullback low — armed-band floor anchor
    setup_pullback_ts: Optional[datetime] = None
    setup_resolved_at: Optional[float] = None  # time.monotonic() when breakout fired
    setup_breakout_ts: Optional[datetime] = None  # real datetime when breakout fired
    setup_breakout_price: Optional[float] = None  # price at which breakout triggered
    setup_failed_at: Optional[float] = None    # time.monotonic() when pending pullback was abandoned
    macd_cycle_done: bool = False              # 1m MACD hist completed its first positive->negative flip since
                                               # move start (closed bars). Blocks ALL new entries under
                                               # SIM_FIRST_MACD_CYCLE_ONLY. Recomputed by DynamicScorer.
    macd_cycle_done_ts: Optional[datetime] = None  # ts of the closed 1m bar whose negative hist completed the cycle
    news_articles: list = field(default_factory=list)
    news_source: str = ""

    # Fields that could not be fetched from APIs
    missing_data: list[str] = field(default_factory=list)

    # Session peak tracking — prev_close fixed at evaluation; peak_change_pct
    # raised live by DynamicScorer on every new HOD and persisted via the
    # TickerCache day snapshot so backtest ordering can use the true observed
    # peak without re-deriving it from candles.
    prev_close: float = 0.0
    peak_change_pct: float = 0.0

    # Boolean gate groups (parallel to scoring groups, do NOT contribute to score).
    # Populated by build_criteria_groups() in _finalize_entry.
    criteria_groups: list = field(default_factory=list)

    # Live execution / order-flow snapshot — updated by DynamicScorer,
    # persisted via TickerCache pickle so backtest and charts can reference it.
    execution_snapshot: dict = field(default_factory=dict)

    # Legacy compat: noise_titles for user-muted headlines
    noise_titles: set = field(default_factory=set)

    @property
    def eval_confidence(self) -> float:
        """STOCK_EVAL-only score (0-100).

        The static evaluation pillar (float, news, rvol, price, daily trend)
        on its own — *not* multiplied by the live TECHNICAL_ANALYSIS strength
        factor.  This is what the 'Eval Confidence' display reports so the
        number matches the STOCK_EVAL group panel directly.
        """
        if self.evaluation is None:
            return 0.0
        try:
            raw = self.evaluation.stock_eval.score * 100.0
        except AttributeError:
            return 0.0
        return round(max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, raw)), 1)

    @property
    def live_confidence(self) -> float:
        """TECHNICAL_ANALYSIS-only live score (0-100)."""
        if self.evaluation is None:
            return 0.0
        try:
            raw = self.evaluation.technical_analysis.score * 100.0
        except AttributeError:
            return 0.0
        return round(max(CONFIDENCE_FLOOR, min(CONFIDENCE_CEILING, raw)), 1)

    @property
    def watchlist_sort_score(self) -> float:
        """Ranking helper: live confidence once available, otherwise eval confidence."""
        return self.live_confidence if self.has_live_data else self.eval_confidence

    @property
    def has_live_data(self) -> bool:
        """True if at least one scored live input has received a real measurement."""
        if self.evaluation is None:
            return False
        for g in self.evaluation.groups:
            for v in g.variables:
                if v.name in _SCORED_LIVE_INPUT_NAMES and v.value is not None:
                    return True
        return False

    def _hard_cutoff_failures(self) -> list[str]:
        """Scored variables whose configured hard cutoffs are currently broken."""
        if self.evaluation is None:
            return []
        failures: list[str] = []
        for g in self.evaluation.groups:
            for v in g.variables:
                if v.value is None:
                    continue
                cutoff = getattr(v, "cutoff", None)
                if cutoff is not None and v.value > cutoff:
                    failures.append(f"{g.name.lower()}:{v.name} {v.value:g} > {cutoff:g}")
                    continue
                cutoff_lo = getattr(v, "cutoff_lo", None)
                if cutoff_lo is not None and v.value < cutoff_lo:
                    failures.append(f"{g.name.lower()}:{v.name} {v.value:g} < {cutoff_lo:g}")
        return failures

    @property
    def entry_allowed(self) -> bool:
        """True iff eval/live confidence gates and every required criterion pass.

        This is the Stage-4 gate that the execution engine MUST consult
        before placing any buy order. ATTENTION (relevance) and EXECUTION
        (trade-setup) criteria groups must have their required gates met.
        """
        import config as _config
        from config import ENTRY_CONFIDENCE_MIN
        if not _config.confidence_bypass_eval() and self.eval_confidence < ENTRY_CONFIDENCE_MIN:
            return False
        if (
            not _config.confidence_bypass_live()
            and self.has_live_data
            and self.live_confidence < ENTRY_CONFIDENCE_MIN
        ):
            return False
        if self._hard_cutoff_failures():
            return False
        # Criteria gates
        for g in self.criteria_groups:
            if not g.required_passing:
                return False
        return True

    @property
    def entry_block_reasons(self) -> list[str]:
        """List of failing required criteria across all groups (for display/logging)."""
        import config as _config
        from config import ENTRY_CONFIDENCE_MIN
        reasons: list[str] = []
        if not _config.confidence_bypass_eval() and self.eval_confidence < ENTRY_CONFIDENCE_MIN:
            reasons.append(f"eval_confidence {self.eval_confidence:.1f} < {ENTRY_CONFIDENCE_MIN}")
        if (
            not _config.confidence_bypass_live()
            and self.has_live_data
            and self.live_confidence < ENTRY_CONFIDENCE_MIN
        ):
            reasons.append(f"live_confidence {self.live_confidence:.1f} < {ENTRY_CONFIDENCE_MIN}")
        reasons.extend(self._hard_cutoff_failures())
        for g in self.criteria_groups:
            for c in g.criteria:
                if c.required and not c.passing:
                    reasons.append(f"{g.name.lower()}:{c.name}")
        return reasons

    def get_criteria_group(self, name: str):
        for g in self.criteria_groups:
            if g.name == name:
                return g
        return None

    @property
    def setup_pattern_score(self) -> float:
        """Separate setup/pattern confidence score (0-100).

        This is intentionally isolated from the main live confidence score.
        Returns 0 only when there is no active or recently-active setup —
        a resolved/failed badge keeps the score visible briefly.
        """
        if self.setup_pattern is None:
            return 0.0
        # Forming pullbacks (detected/pending) are previews only — they carry
        # NO setup score until the pullback breaks the entry / new high and
        # becomes a recorded setup (which still counts even if it fails after).
        if self.setup_pattern_name in ("pullback_detected", "pullback_pending"):
            return 0.0
        if not (self.setup_pattern_name or self.setup_resolved_at or self.setup_failed_at):
            return 0.0
        raw = self.setup_pattern.score * 100.0
        return round(max(0.0, min(100.0, raw)), 1)

    @property
    def active_weaknesses(self) -> list[str]:
        """Variables with quality < 0.2 (derived, not stored)."""
        out: list[str] = []
        if self.evaluation is None:
            return out
        for g in self.evaluation.groups:
            for v in g.variables:
                if v.value is not None and v.quality < 0.2:
                    out.append(f"{g.name.lower()}:{v.name}")
        return out

    @property
    def active_strengths(self) -> list[str]:
        """Variables with quality > 0.7 (derived, not stored)."""
        out: list[str] = []
        if self.evaluation is None:
            return out
        for g in self.evaluation.groups:
            for v in g.variables:
                if v.value is not None and v.quality > 0.7:
                    out.append(f"{g.name.lower()}:{v.name}")
        return out


# ---------------------------------------------------------------------------
# Evaluation stats / telemetry
# ---------------------------------------------------------------------------

_STATS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
_STATS_FILE = os.path.join(_STATS_DIR, "eval_stats.log")

_STATS_FIELDS = [
    "timestamp", "symbol", "passed", "total_ms",
    "stage1_ms", "stage2_parallel_ms",
    "news_momo_ms", "news_finnhub_ms", "news_brave_ms", "news_av_ms",
    "finbert_ms", "splits_ms", "daily_candles_ms",
    "news_source", "failed_apis",
]

_STATS_COL_WIDTHS = {
    "timestamp":           19,
    "symbol":               6,
    "passed":               6,
    "total_ms":             8,
    "stage1_ms":            8,
    "stage2_parallel_ms":   8,
    "news_momo_ms":         8,
    "news_finnhub_ms":     14,
    "news_brave_ms":        8,
    "news_av_ms":           8,

    "finbert_ms":           8,
    "splits_ms":           14,
    "daily_candles_ms":     8,
    "news_source":         14,
    "failed_apis":         20,
}

_STATS_HEADERS = {
    "timestamp":           "Timestamp",
    "symbol":              "Sym",
    "passed":              "Pass",
    "total_ms":            "Total",
    "stage1_ms":           "Stg1",
    "stage2_parallel_ms":  "Stg2Par",
    "news_momo_ms":        "Momo",
    "news_finnhub_ms":     "Finnhub(net)",

    "news_av_ms":          "AV News",
    "news_brave_ms":       "Brave",
    "finbert_ms":          "FinBERT",
    "splits_ms":           "Splits(net)",
    "daily_candles_ms":    "Candles",
    "news_source":         "News Source",
    "failed_apis":         "Failed APIs",
}


def _write_stats(row: dict) -> None:
    """Append a human-readable aligned stats row to the log."""
    os.makedirs(_STATS_DIR, exist_ok=True)
    # A zero-byte file (left behind by the pre-fix encoding crash below) still
    # needs its header.
    write_header = (
        not os.path.exists(_STATS_FILE) or os.path.getsize(_STATS_FILE) == 0
    )

    def _fmt(key: str, val: str) -> str:
        w = _STATS_COL_WIDTHS[key]
        return val.rjust(w) if key.endswith("_ms") else val.ljust(w)

    clean = {}
    for k in _STATS_FIELDS:
        v = row.get(k, "")
        if k.endswith("_ms") and not v:
            v = "—"
        clean[k] = str(v)

    # Combine Finnhub actual time with net parallel wait: "300 (0)"
    fh = clean["news_finnhub_ms"]
    fh_net = row.get("news_finnhub_net_ms", "0")
    clean["news_finnhub_ms"] = f"{fh} ({fh_net})" if fh != "—" else "—"

    # Combine Splits actual time with net parallel wait: "150 (0)"
    sp = clean["splits_ms"]
    sp_net = row.get("splits_net_ms", "0")
    clean["splits_ms"] = f"{sp} ({sp_net})" if sp != "—" else "—"

    # Explicit UTF-8: the box-drawing separators below are not encodable in
    # the Windows default (cp1252). Without this every write raised
    # UnicodeEncodeError, was swallowed at DEBUG by the caller, and the stats
    # log stayed empty — the pipeline had no latency telemetry at all.
    with open(_STATS_FILE, "a", encoding="utf-8") as f:
        if write_header:
            hdr = " │ ".join(_fmt(k, _STATS_HEADERS[k]) for k in _STATS_FIELDS)
            sep = "─┼─".join("─" * _STATS_COL_WIDTHS[k] for k in _STATS_FIELDS)
            f.write(f"{hdr}\n{sep}\n")
        line = " │ ".join(_fmt(k, clean[k]) for k in _STATS_FIELDS)
        f.write(f"{line}\n")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class StockEvaluator:
    """
    Two-stage stock scanner with static confidence scoring.
    Dynamic confidence is updated externally via entry.apply_dynamic().
    """

    def __init__(self) -> None:
        self._tradier   = get_historical_client()
        self._av        = None if not AV_ENABLED else AlphaVantageClient()
        self._fmp       = None if not FMP_ENABLED else FMPClient()
        self._finnhub   = FinnhubClient()
        self._finbert   = FinBERTClient()
        self._brave     = BraveNewsClient()
        self._momo      = MomoClient()
        self._momo.connect_polling()
        self.watchlist: list[WatchlistEntry] = []
        self.watchlist_symbols: set[str] = set()
        self.last_fail_reason: str = ""
        self._load_market_calendar()

    def _load_market_calendar(self) -> None:
        """Fetch closed dates from the market-data provider and cache in _MarketCalendar."""
        if not MARKET_DATA_ENABLED:
            return
        try:
            closed = self._tradier.get_closed_dates(months_back=2)
            if closed:
                _MarketCalendar.load(closed)
                logger.info("Market calendar loaded: %d closed dates cached", len(closed))
        except Exception as exc:
            logger.debug("Failed to load Tradier market calendar: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, symbol: str, df_daily: Optional[pd.DataFrame] = None) -> Optional[WatchlistEntry]:
        """
        Run Stage 1 and Stage 2 for a single symbol.

        df_daily: optional pre-built daily candle DataFrame (with ema_200,
                  ma_200 columns). If None, daily trend and MA checks are
                  skipped (scored as neutral 0).

        Returns WatchlistEntry on success, None if either stage fails.
        """
        symbol = symbol.upper().strip()

        t1 = time.perf_counter()
        s1 = self.stage1(symbol)
        self._stage1_ms = (time.perf_counter() - t1) * 1000
        if not s1.passed:
            logger.debug(f"[{symbol}] Stage 1 FAIL -- {s1.fail_reason}")
            self.last_fail_reason = s1.fail_reason
            return None

        return self.stage2(s1, df_daily=df_daily)

    def stage1(self, symbol: str) -> Stage1Result:
        """Run Stage 1 screening (price band; measures vol/rvol/change). Public wrapper."""
        return self._stage1(symbol.upper().strip())

    def stage2(self, s1: Stage1Result, df_daily: Optional[pd.DataFrame] = None,
               on_float_pass: Optional[Callable] = None,
               yt_headlines: Optional[list] = None,
               skip_float_gate: bool = False,
               cached_float: Optional[float] = None) -> Optional[WatchlistEntry]:
        """
        Run Stage 2 enrichment given a passing Stage1Result.
        Adds the entry to the watchlist on success.
        Returns WatchlistEntry or None if float gate fails.

        *on_float_pass* — optional no-arg callback fired immediately after
        the float gate passes (before news/splits). Use to start streaming
        data in parallel with the rest of stage 2.

        *yt_headlines* — optional list of ``ChatNewsHeadline`` from the
        YouTube live chat news author. Scored with FinBERT; headlines with
        sentiment between -NOISE and +NOISE are discarded as noise.

        *skip_float_gate* — if True, continue evaluation even when the
        float exceeds FLOAT_MAX_M (used for forced manual queries so the
        full summary is always displayed).
        """
        symbol = s1.symbol

        logger.debug(
            f"[{symbol}] Stage 1 PASS | ${s1.price} {s1.change_pct:+.1f}% "
            f"{_dim(f'vol={s1.volume:,} rvol_1d={s1.rvol_1day:.1f}x rvol_5m={s1.rvol_5min:.1f}x')}"
        )

        # Init stats dict — _stage2 / _stage2_av populate the timings
        self._eval_stats: dict = {
            "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol,
            "stage1_ms": f"{getattr(self, '_stage1_ms', 0):.0f}",
            "daily_candles_ms": f"{getattr(self, '_daily_candles_ms', 0):.0f}",
        }

        t0 = time.perf_counter()
        entry = self._stage2(s1, df_daily=df_daily, on_float_pass=on_float_pass,
                             yt_headlines=yt_headlines, skip_float_gate=skip_float_gate,
                             cached_float=cached_float)
        total_ms = (time.perf_counter() - t0) * 1000

        self._eval_stats["total_ms"] = f"{total_ms:.0f}"
        self._eval_stats["passed"] = "yes" if entry is not None else "no"

        # Write stats row
        try:
            _write_stats(self._eval_stats)
        except Exception as exc:
            logger.debug(f"[{symbol}] Failed to write eval stats: {exc}")

        if entry is None:
            if not self.last_fail_reason:
                self.last_fail_reason = f"float > {FLOAT_MAX_M:.0f}M shares"
            return None

        # Build-then-rebind (never mutate in place): evaluate() runs on
        # scanner/consumer threads while the main thread's toolbar iterates
        # the watchlist — an in-place sort would expose a half-sorted list.
        new_wl = [e for e in self.watchlist if e.symbol != symbol]
        new_wl.append(entry)
        new_wl.sort(key=lambda e: e.watchlist_sort_score, reverse=True)
        self.watchlist = new_wl
        self.watchlist_symbols.add(symbol)

        _fl = f"{entry.float_shares_m:.2f}M" if entry.float_shares_m > 0 else "—"
        logger.debug(
            f"[{symbol}] -> WATCHLIST | eval={entry.eval_confidence:.1f} live={entry.live_confidence:.1f} "
            f"{_dim(f'float={_fl} sector={entry.sector or chr(8212)} '
            f'news={entry.news_count_last_hour} sentiment={entry.avg_sentiment}')}"
        )
        return entry

    def refresh_news(self, entry: WatchlistEntry, exclude_titles: set[str] | None = None, *, allow_older: bool = False) -> bool:
        """Re-query news sources for an existing entry and update it in place.

        Runs Momo → Finnhub → Brave cascade. If newer news is found the
        entry's news fields and catalyst quality are updated.

        *exclude_titles* — set of headline strings to skip (marked as noise).
        *allow_older* — if True, accept replacement news even if it's older
            than the current headline (used when the current headline was
            explicitly marked as noise).

        Returns True if the entry was updated with fresher news.
        """
        symbol = entry.symbol
        # Merge caller-provided exclusions with the persistent noise store.
        from cache import NoiseStore
        _stored_noise = NoiseStore.get().titles(symbol)
        exclude_titles = (exclude_titles or set()) | _stored_noise

        _news_lookback_min = int(_MarketCalendar.lookback(2).total_seconds() / 60)
        _fetch_since = datetime.now(tz=timezone.utc) - timedelta(minutes=_news_lookback_min)
        _co_name = ""
        _us_company = True

        # Try to get company name from Finnhub profile for relevance filtering
        try:
            if FINNHUB_ENABLED:
                profile = self._finnhub.get_profile(symbol)
                if profile:
                    _co_name = profile.name or ""
                    _us_company = (profile.country or "").upper() in ("US", "USA", "UNITED STATES")
        except Exception:
            pass

        articles: list = []
        news_count = 0
        avg_sentiment = None
        _news_source = ""
        _session_cutoff, _, _ = news_freshness_cutoffs()

        def _is_fresh(arts: list) -> bool:
            return bool(arts and arts[0].published_at >= _session_cutoff)

        # 1) Momo
        try:
            momo_news = self._momo.get_news(symbol)
            if momo_news:
                from market_data.finnhub_client import FinnhubNewsArticle
                raw = [
                    FinnhubNewsArticle(
                        symbol=n.symbol, title=n.headline, source=n.source,
                        url=n.url, published_at=n.published_at or datetime.now(tz=timezone.utc),
                        summary="",
                    )
                    for n in momo_news
                    if n.published_at and n.published_at >= _fetch_since
                ]
                arts, cnt, sent = self._filter_and_score(raw, symbol, _co_name, us_company=_us_company, skip_relevance=True, exclude_titles=exclude_titles)
                if cnt > 0:
                    articles, news_count, avg_sentiment, _news_source = arts, cnt, sent, "momo"
        except Exception as exc:
            logger.debug(f"[{symbol}] News refresh momo error: {exc}")

        # 2) Finnhub
        if not _is_fresh(articles) and FINNHUB_ENABLED:
            try:
                raw = self._finnhub.get_recent_news(symbol, minutes_back=_news_lookback_min)
                arts, cnt, sent = self._filter_and_score(raw, symbol, _co_name, us_company=_us_company, exclude_titles=exclude_titles)
                if cnt > 0:
                    articles, news_count, avg_sentiment, _news_source = arts, cnt, sent, "finnhub"
            except Exception as exc:
                logger.debug(f"[{symbol}] News refresh finnhub error: {exc}")

        # 3) Brave
        if not _is_fresh(articles) and BRAVE_ENABLED:
            try:
                raw = self._brave.search_ticker_news(symbol)
                arts, cnt, sent = self._filter_and_score(raw, symbol, _co_name, us_company=_us_company, exclude_titles=exclude_titles)
                if cnt > 0 and (not articles or arts[0].published_at > articles[0].published_at):
                    articles, news_count, avg_sentiment, _news_source = arts, cnt, sent, "brave"
            except Exception as exc:
                logger.debug(f"[{symbol}] News refresh brave error: {exc}")

        if not articles:
            return False

        # Only update if the new news is actually newer than what we had
        # (unless allow_older=True, which is used when the current headline
        # was explicitly marked as noise and we accept any replacement).
        old_pub = entry.news_articles[0].published_at if entry.news_articles else None
        new_pub = articles[0].published_at
        if not allow_older and old_pub and new_pub <= old_pub:
            return False

        session_cutoff, breaking_cutoff, _ = news_freshness_cutoffs()
        entry.news_articles = articles[:1]
        entry.news_count_last_hour = news_count
        entry.avg_sentiment = avg_sentiment
        entry.news_source = _news_source
        entry.news_is_fresh = new_pub >= session_cutoff
        entry.news_is_breaking = new_pub >= breaking_cutoff

        # Keep catalyst quality in sync with refreshed news.
        if entry.evaluation:
            noise_gate = (
                avg_sentiment is not None
                and -NEWS_NOISE_THRESHOLD <= avg_sentiment <= NEWS_NOISE_THRESHOLD
            )
            entry.evaluation.update_news(avg_sentiment, news_count, _news_source, noise_gate)

        logger.debug(
            f"[{symbol}] News refreshed from cache: source={_news_source} "
            f"sentiment={f'{avg_sentiment:.2f}' if avg_sentiment is not None else 'N/A'}"
        )
        return True

    def scan(self, symbols: list, df_daily_map: Optional[dict] = None) -> list:
        """
        Batch evaluate. df_daily_map: optional {symbol: df_daily} dict.
        Skips errors without raising.
        """
        passed = []
        for sym in symbols:
            try:
                df_d = (df_daily_map or {}).get(sym.upper())
                entry = self.evaluate(sym, df_daily=df_d)
                if entry:
                    passed.append(entry)
            except Exception as exc:
                logger.warning(f"[{sym}] Evaluation error: {exc}")
        return passed

    def get_watchlist_df(self) -> pd.DataFrame:
        """Flat DataFrame for the current watchlist state."""
        if not self.watchlist:
            return pd.DataFrame()
        rows = []
        for e in self.watchlist:
            row = asdict(e)
            row["eval_confidence"] = e.eval_confidence
            row["live_confidence"] = e.live_confidence
            row["setup_pattern_score"] = e.setup_pattern_score
            row.pop("evaluation", None)
            row.pop("setup_pattern", None)
            rows.append(row)
        df = pd.DataFrame(rows)
        df["evaluated_at"] = pd.to_datetime(df["evaluated_at"], utc=True)
        return df

    def clear_watchlist(self) -> None:
        self.watchlist = []
        self.watchlist_symbols.clear()
        logger.info("Watchlist cleared.")

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def _stage1(self, symbol: str) -> Stage1Result:
        if not MARKET_DATA_ENABLED:
            logger.debug(f"[{symbol}] Market data disabled — Stage 1 bypassed with stub values")
            return Stage1Result(
                symbol=symbol, price=5.00, change_pct=15.0,
                volume=1_000_000, avg_volume=200_000,
                rvol_1day=10.0, rvol_5min=10.0, daily_high=5.00,
                passed=True,
            )
        try:
            quote = self._tradier.get_quote(symbol)
        except MarketDataUnavailable as exc:
            # The provider could not answer — connection refused at IBKR's
            # 32-client cap, a pacing stall, a socket timeout.  This says
            # nothing about the symbol, so it must NOT be reported as
            # "symbol not found": that reason is treated as permanent and
            # bails even under force=True, silently discarding a valid
            # ticker whenever the data feed hiccups.
            logger.warning("[%s] Stage 1 quote unavailable: %s", symbol, exc)
            return Stage1Result(
                symbol=symbol, price=0.0, change_pct=0.0,
                volume=0, avg_volume=0,
                rvol_1day=0.0, rvol_5min=0.0, daily_high=0.0,
                passed=False, fail_reason="market data unavailable",
            )

        # Providers return an all-None stub (last=None, no volume, empty
        # description/exchange) for unmatched symbols.  Detect both the
        # original "all numeric fields zero" shape and the secondary
        # signal of missing description/exchange so we bail fast instead
        # of letting parallel candle/Momo calls fire against a bogus sym.
        no_quote = (
            quote.last is None and quote.volume == 0 and quote.avg_volume == 0
        )
        no_metadata = not quote.description and not quote.exchange
        if no_quote or no_metadata:
            return Stage1Result(
                symbol=symbol, price=0.0, change_pct=0.0,
                volume=0, avg_volume=0,
                rvol_1day=0.0, rvol_5min=0.0, daily_high=0.0,
                passed=False, fail_reason="symbol not found",
            )

        prev_close = quote.prev_close

        # During RTH, `last` is live. Outside RTH (pre/post-market),
        # `last` is stale (stuck at prev close) — use bid/ask midpoint.
        if MarketClock.is_rth():
            price = quote.last or 0.0
            change_pct = quote.change_pct or 0.0
        else:
            bid, ask = quote.bid, quote.ask
            if bid and ask:
                price = (bid + ask) / 2.0
            else:
                price = quote.last or 0.0
            if price and prev_close:
                change_pct = (price - prev_close) / prev_close * 100.0
            else:
                change_pct = quote.change_pct or 0.0

        logger.debug(
            "[%s] Stage 1 quote: price=%.2f change_pct=%.2f prevclose=%s vol=%s avg_vol=%s",
            symbol, price, change_pct, prev_close,
            quote.volume, quote.avg_volume,
        )

        volume     = quote.volume
        avg_volume = max(quote.avg_volume or 1, 1)
        daily_high = quote.high or price
        rvol_1day = rvol_5min = 0.0

        def fail(reason: str) -> Stage1Result:
            return Stage1Result(
                symbol=symbol, price=price, change_pct=change_pct,
                volume=volume, avg_volume=avg_volume,
                rvol_1day=rvol_1day, rvol_5min=rvol_5min,
                daily_high=daily_high,
                passed=False, fail_reason=reason,
            )

        # Quote prefilter hard rejects only structural price range. Day
        # change, volume, and RVOL are thesis/readiness inputs handled by
        # static quality and the live ATTENTION criteria (change is also
        # pre-gated by the IBKR scanner's changePercAbove floor), not
        # cheap-quote reject reasons.
        #
        # The price gate runs BEFORE the RVOL computation on purpose: the
        # RVOL history fetch (~14 days of 5-min bars on first sight) is the
        # single most expensive call in Stage 1 — multi-second under IBKR's
        # payload shaping, 30s when it times out — and the consumer runs
        # tickers serially. Paying it for an $80 name the price band is about
        # to reject just delays every ticker queued behind it.
        if not (PRICE_MIN <= price <= PRICE_MAX):
            return fail(f"price ${price:.2f} not in [{PRICE_MIN}-{PRICE_MAX}]")

        rvol_1day, rvol_5min = self._compute_rvol_at_time(symbol)

        return Stage1Result(
            symbol=symbol, price=price, change_pct=change_pct,
            volume=volume, avg_volume=avg_volume,
            rvol_1day=rvol_1day, rvol_5min=rvol_5min,
            daily_high=daily_high,
            passed=True,
        )

    @staticmethod
    def _tag_session_times(df: pd.DataFrame, market_tz) -> pd.DataFrame:
        """Add ET-session ``date`` / ``tod`` columns used by the RVOL math."""
        df = df.copy()
        ts_et = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(market_tz)
        df["date"] = ts_et.dt.date
        df["tod"] = ts_et.dt.time
        return df

    def _rvol_frames(self, symbol: str, today, market_tz) -> tuple[pd.DataFrame, pd.DataFrame, list]:
        """Return (df_today, df_hist, hist_dates) of tagged 5-min bars.

        Prior-day bars never change intraday, so they are fetched once per
        symbol per session day and cached; repeat Stage-1 runs re-fetch only
        today's session (a 1-day payload instead of the full lookback window,
        which also keeps IBKR pacing pressure down).
        """
        # Lazy init: tests build the evaluator via __new__, skipping __init__.
        cache = getattr(self, "_rvol_hist_cache", None)
        if cache is None:
            cache = self._rvol_hist_cache = {}

        cached = cache.get(symbol)
        fetch_window = getattr(self._tradier, "get_candles_5min_window", None)
        if cached is not None and cached[0] == today and fetch_window is not None:
            _, df_hist, hist_dates = cached
            df_today = fetch_window(symbol, latest_extended_session_start())
            if not df_today.empty:
                df_today = self._tag_session_times(df_today, market_tz)
                df_today = df_today[df_today["date"] == today]
            return df_today, df_hist, hist_dates

        cal_days = _MarketCalendar.lookback(RVOL_AT_TIME_PERIODS).days
        df = self._tradier.get_candles_5min(symbol, days_back=cal_days)
        if df.empty:
            return df, df, []

        df = self._tag_session_times(df, market_tz)
        df_today = df[df["date"] == today]
        df_hist = df[df["date"] < today]

        # Keep only the most recent RVOL_AT_TIME_PERIODS trading days.
        hist_dates = sorted(df_hist["date"].unique())[-RVOL_AT_TIME_PERIODS:]
        df_hist = df_hist[df_hist["date"].isin(set(hist_dates))]

        # Cache even an empty history (e.g. a recent IPO) — no prior days will
        # appear mid-session, so refetching the full window would be wasted.
        cache[symbol] = (today, df_hist, hist_dates)
        return df_today, df_hist, hist_dates

    def _compute_rvol_at_time(self, symbol: str) -> tuple[float, float]:
        """
        Stage 1 RVol computation:
          - 1-day RVOL (at-time): today's cumulative volume up to now vs
            avg cumulative volume at the same time-of-day over N prior days.
            Falls back to full-day average when no historical data exists at
            the current time-of-day (e.g. premarket evaluation).
          - 5-min RVOL (rolling): current 5-min bar vs mean of prior
            today bars — detects real-time acceleration.  Falls back to
            historical average bar volume when only 1 bar exists today.

        Prior-day bars come from a per-session cache (see _rvol_frames);
        only the first evaluation of a symbol pays for the full lookback fetch.
        """
        try:
            now_et = MarketClock.now_et()
            current_time = now_et.time()
            today = now_et.date()
            market_tz = now_et.tzinfo or MarketClock.ET

            df_today, df_hist, hist_dates = self._rvol_frames(symbol, today, market_tz)
            if df_today.empty or df_hist.empty:
                return 0.0, 0.0

            # --- 1-day RVOL at time ---
            today_cum_vol = float(
                df_today.loc[df_today["tod"] <= current_time, "volume"].sum()
            )
            hist_cum_vols = []
            for d in hist_dates:
                day_df = df_hist[df_hist["date"] == d]
                cum = float(day_df.loc[day_df["tod"] <= current_time, "volume"].sum())
                hist_cum_vols.append(cum)
            avg_cum = sum(hist_cum_vols) / len(hist_cum_vols) if hist_cum_vols else 0

            if avg_cum > 0:
                rvol_1day = today_cum_vol / avg_cum
            else:
                # Premarket fallback: no historical candles at this time-of-day.
                # Compare today's cumulative volume against the average full-day
                # historical volume to produce a meaningful RVOL signal.
                hist_daily_vols = []
                for d in hist_dates:
                    day_df = df_hist[df_hist["date"] == d]
                    hist_daily_vols.append(float(day_df["volume"].sum()))
                avg_daily = (sum(hist_daily_vols) / len(hist_daily_vols)
                             if hist_daily_vols else 0)
                rvol_1day = today_cum_vol / avg_daily if avg_daily > 0 else 0.0

            # --- 5-min RVOL rolling (today only) ---
            today_vols = df_today["volume"].values
            if len(today_vols) < 2:
                # Only 1 bar today (common in early premarket). Compare against
                # the historical average bar volume for a baseline signal.
                hist_bar_avg = float(df_hist["volume"].mean()) if len(df_hist) > 0 else 0
                single_vol = float(today_vols[0]) if len(today_vols) == 1 else 0
                rvol_5min = single_vol / hist_bar_avg if hist_bar_avg > 0 else 0.0
            else:
                last_vol = float(today_vols[-1])
                prior_avg = float(today_vols[:-1].mean())
                rvol_5min = last_vol / prior_avg if prior_avg > 0 else 0.0

            return rvol_1day, rvol_5min
        except Exception as exc:
            logger.debug(f"[{symbol}] RVOL-at-time error: {exc}")
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # News relevance filter
    # ------------------------------------------------------------------

    _NOISE_RE = re.compile(
        r"\b(pre?|hours?|gainers?|gappers?|stocks|movers|market-moving|session|dow|"
        r"monday'?s?|tuesday'?s?|wednesday'?s?|thursday'?s?|friday'?s?|"
        r"saturday'?s?|sunday'?s?)\b"
        r"|news\s+for\s+(january|february|march|april|may|june|july|august|"
        r"september|october|november|december|\w+day|\d)"
        r"|\?",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_relevant_article(title: str, symbol: str, company_name: str = "", *, us_company: bool = True, summary: str = "") -> bool:
        """Return True if the headline is about this specific stock.

        Must contain the ticker OR company name, AND must not match
        the noise blacklist (generic roundup / day-of-week headlines).
        Non-ASCII titles (non-English) are rejected.

        Name matching: US companies use first two words (to avoid
        collisions like "Advanced" matching both ADVB and AMD).
        Non-US companies use the first word (often distinctive).
        """
        if not title.isascii():
            return False
        t = title.lower()
        sym_lower = symbol.lower()
        has_ticker = sym_lower in t and bool(re.search(rf"\b{re.escape(symbol)}\b", t, re.IGNORECASE))
        has_name = False
        if company_name:
            _corp_suffixes = {"inc", "corp", "ltd", "llc", "plc", "co", "sa", "nv", "ag", "se", "group", "holdings"}
            words = [w.strip(",.") for w in company_name.split()
                     if len(w.strip(",.")) >= 2 and w.strip(",.").lower() not in _corp_suffixes]
            n = 2 if us_company else 1
            key = " ".join(words[:n]).lower()
            if key and key in t:
                has_name = True
        if not has_ticker and not has_name:
            return False
        # Blacklist generic roundup headlines even if the ticker appears
        if NOISE_FILTER_TITLE and StockEvaluator._NOISE_RE.search(title):
            return False
        return True

    # ------------------------------------------------------------------
    # News scoring helpers
    # ------------------------------------------------------------------

    def _finbert_score(self, symbol: str, article) -> Optional[float]:
        """Score a single article's headline with FinBERT. Returns None on failure or if disabled."""
        if not FINBERT_ENABLED:
            return None
        t_fb = time.perf_counter()
        score = self._finbert.score_headline(
            symbol,
            headline=article.title,
            summary=getattr(article, "summary", ""),
        )
        elapsed = (time.perf_counter() - t_fb) * 1000
        if hasattr(self, "_eval_stats"):
            prev = float(self._eval_stats.get("finbert_ms", 0))
            self._eval_stats["finbert_ms"] = f"{prev + elapsed:.0f}"
        if score is not None:
            logger.debug(f"[{symbol}] FinBERT sentiment={score:.2f} {_dim(f'for {chr(39)}{article.title}{chr(39)}')}")
        return score

    def _filter_and_score(
        self, raw_articles: list, symbol: str, co_name: str, *,
        use_entity: bool = False, us_company: bool = True,
        exclude_titles: set[str] | None = None,
        skip_relevance: bool = False,
    ) -> tuple[list, int, Optional[float]]:
        """Filter articles for relevance, count them, and compute sentiment.

        When *use_entity* is True (AV), average the built-in
        entity_sentiment scores; fall back to FinBERT for the top article
        if no entity scores exist. Otherwise score with FinBERT directly.

        *exclude_titles* — headlines to skip (user-marked noise).
        *skip_relevance* — skip _is_relevant_article (Momo already scopes by ticker).

        Returns (filtered_articles, count, avg_sentiment).
        """
        _exclude = exclude_titles or set()
        _noise_re = StockEvaluator._NOISE_RE if (skip_relevance and NOISE_FILTER_TITLE) else None
        articles = [
            a for a in raw_articles
            if (skip_relevance or self._is_relevant_article(
                    a.title, symbol, co_name, us_company=us_company,
                    summary=getattr(a, 'summary', '')))
            and a.title not in _exclude
            and (_noise_re is None or not _noise_re.search(a.title))
        ]
        count = len(articles)
        sentiment: Optional[float] = None
        if not articles:
            return articles, count, sentiment

        if use_entity:
            scores = [a.entity_sentiment for a in articles if a.entity_sentiment is not None]
            if scores:
                sentiment = sum(scores) / len(scores)
            else:
                sentiment = self._finbert_score(symbol, articles[0])
        else:
            sentiment = self._finbert_score(symbol, articles[0])

        return articles, count, sentiment

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def _stage2(self, s1: Stage1Result, df_daily: Optional[pd.DataFrame] = None,
                on_float_pass: Optional[Callable] = None,
                yt_headlines: Optional[list] = None,
                skip_float_gate: bool = False,
                cached_float: Optional[float] = None) -> Optional[WatchlistEntry]:
        symbol = s1.symbol
        stats = self._eval_stats
        failed_apis: list[str] = []

        _news_lookback_min = int(_MarketCalendar.lookback(2).total_seconds() / 60)

        # ------------------------------------------------------------------
        # AV budget check — decide what AV handles vs Momo fallback
        # ------------------------------------------------------------------
        if not AV_ENABLED or self._av is None:
            use_av_float = use_av_splits = use_av_news = False
            logger.debug(f"[{symbol}] AV disabled — float=momo splits=skip news=cascade")
        else:
            from config import AV_NEWS_ENABLED, AV_CALLS_PER_MIN
            overview_cost = 0 if self._av.overview_cached(symbol) else 1
            budget = self._av.budget
            # Float only needs the OVERVIEW call (1 or 0 if cached)
            use_av_float = budget >= overview_cost
            # Splits costs 1 additional call beyond overview
            use_av_splits = budget >= overview_cost + 1
            # News costs 1 more on top of splits
            use_av_news = AV_NEWS_ENABLED and budget >= overview_cost + 2
            logger.debug(
                f"[{symbol}] AV budget={budget}/{AV_CALLS_PER_MIN} — "
                f"float={'av' if use_av_float else 'momo'} "
                f"splits={'av' if use_av_splits else 'skip'} "
                f"news={'av' if use_av_news else 'cascade'}"
            )

        # ------------------------------------------------------------------
        # Fire all independent API calls in parallel:
        #   AV/Momo float, Finnhub profile, AV splits, Finnhub news
        # ------------------------------------------------------------------

        def _fetch_float():
            # Use cached float if provided (scanner re-hit — float doesn't change)
            if cached_float is not None:
                from market_data.fmp_client import FloatData
                return FloatData(
                    symbol=symbol, float_shares=int(cached_float * 1_000_000),
                    float_shares_m=cached_float,
                    outstanding_shares=int(cached_float * 1_000_000),
                    free_float_pct=100.0,
                    source="cache",
                )
            if use_av_float:
                result = self._av.get_float(symbol)
                if result:
                    result.source = "AV"
                    return result
                # AV returned nothing — try Momo
            momo_float = self._momo.get_float(symbol)
            if momo_float and momo_float.float_shares_m > 0:
                momo_float.source = "momo"
                return momo_float
            # FMP fallback (when AV budget exhausted / Momo failed)
            if self._fmp is not None:
                fmp_float = self._fmp.get_float(symbol)
                if fmp_float and fmp_float.float_shares_m > 0:
                    fmp_float.source = "fmp"
                    return fmp_float
            if not AV_ENABLED:
                return FloatData(
                    symbol=symbol, float_shares=2_000_000,
                    float_shares_m=2.0, outstanding_shares=2_000_000,
                    free_float_pct=100.0,
                    source="default",
                )
            return None

        def _fetch_profile():
            if not FINNHUB_ENABLED:
                return None
            return self._finnhub.get_profile(symbol)

        def _fetch_splits():
            _t = time.perf_counter()
            if use_av_splits:
                result = self._av.most_recent_reverse_split(symbol, days_back=SPLIT_WINDOW_DAYS)
                stats["splits_ms"] = f"{(time.perf_counter() - _t) * 1000:.0f}"
                if result:
                    return result
            return None

        def _fetch_finnhub():
            if not FINNHUB_ENABLED:
                return []
            _t = time.perf_counter()
            result = self._finnhub.get_recent_news(symbol, minutes_back=_news_lookback_min)
            stats["news_finnhub_ms"] = f"{(time.perf_counter() - _t) * 1000:.0f}"
            return result

        def _fetch_av_news():
            _t = time.perf_counter()
            result = self._av.get_recent_news(symbol, minutes_back=_news_lookback_min)
            stats["news_av_ms"] = f"{(time.perf_counter() - _t) * 1000:.0f}"
            return result

        t_par = time.perf_counter()
        futures_map = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures_map["float"]   = pool.submit(_fetch_float)
            futures_map["profile"] = pool.submit(_fetch_profile)
            futures_map["splits"]  = pool.submit(_fetch_splits)
            futures_map["finnhub"] = pool.submit(_fetch_finnhub)
            if use_av_news:
                futures_map["av_news"] = pool.submit(_fetch_av_news)
        stats["stage2_parallel_ms"] = f"{(time.perf_counter() - t_par) * 1000:.0f}"

        # ------------------------------------------------------------------
        # Collect results
        # ------------------------------------------------------------------

        # --- float ---
        try:
            float_data = futures_map["float"].result()
        except Exception as exc:
            logger.warning(f"[{symbol}] Float fetch error: {exc}")
            failed_apis.append("av_float")
            stats["failed_apis"] = ",".join(failed_apis)
            if not skip_float_gate:
                return None
            float_data = None

        if float_data is None:
            if not skip_float_gate:
                logger.warning(f"[{symbol}] No float data returned")
                failed_apis.append("av_float")
                stats["failed_apis"] = ",".join(failed_apis)
                # Name the real cause: without this the caller's fallback
                # reported "float > 50M shares" for a lookup that never
                # returned any float at all.
                self.last_fail_reason = "no float data (all float sources failed)"
                return None
            # Stub float so evaluation can continue for display
            from market_data.fmp_client import FloatData
            float_data = FloatData(
                symbol=symbol, float_shares=0, float_shares_m=0.0,
                outstanding_shares=0, free_float_pct=0.0,
                source="none",
            )
        if float_data.float_shares_m > FLOAT_MAX_M:
            logger.debug(f"[{symbol}] FAIL float {float_data.float_shares_m:.2f}M > {FLOAT_MAX_M}M")
            self.last_fail_reason = f"float: {float_data.float_shares_m:.2f}M (> {FLOAT_MAX_M:.0f}M)"
            if not skip_float_gate:
                return None

        # Float gate passed — fire early-start callback (e.g. streaming)
        if on_float_pass is not None:
            on_float_pass()

        # --- profile ---
        sector = industry = ""
        ipo_date_str = None
        days_since_ipo = None
        profile = None
        try:
            profile = futures_map["profile"].result()
            if profile:
                sector       = profile.sector
                industry     = profile.industry
                ipo_date_str = profile.ipo_date
        except Exception as exc:
            logger.warning(f"[{symbol}] Finnhub profile error: {exc}")
            failed_apis.append("finnhub_profile")

        # --- company name + country for relevance filter ---
        _co_name = profile.name if profile else ""
        _us_company = (profile.country or "").upper() in ("US", "USA", "UNITED STATES") if profile else True

        # --- IPO recency ---
        def _days_since_ipo(date_str: Optional[str]) -> Optional[int]:
            if not date_str:
                return None
            try:
                ipo_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                return (datetime.now(tz=timezone.utc) - ipo_dt).days
            except ValueError as exc:
                logger.warning(f"[{symbol}] bad IPO date '{date_str}': {exc}")
                return None

        days_since_ipo = _days_since_ipo(ipo_date_str)

        # FMP fallback. Finnhub often reports a company's foreign primary listing
        # (e.g. PLSM → Tel Aviv 2021) rather than the recent US IPO/uplisting that
        # is the real catalyst — or omits the date entirely. When Finnhub shows no
        # recent-IPO signal (missing date, or an old date on a foreign-domiciled
        # name), consult FMP (which tracks the US IPO) and prefer a more-recent date.
        _stale_ipo = days_since_ipo is None or days_since_ipo > IPO_WINDOW_DAYS
        if self._fmp is not None and (not ipo_date_str or (not _us_company and _stale_ipo)):
            try:
                fmp_ipo = self._fmp.get_ipo_date(symbol)
            except Exception as exc:
                logger.warning(f"[{symbol}] FMP IPO date error: {exc}")
                fmp_ipo = None
            fmp_days = _days_since_ipo(fmp_ipo)
            if fmp_days is not None and (days_since_ipo is None or fmp_days < days_since_ipo):
                ipo_date_str, days_since_ipo = fmp_ipo, fmp_days

        # --- splits ---
        has_split = False
        split_ratio = None
        split_date  = None
        try:
            rev_split = futures_map["splits"].result()
            if rev_split:
                has_split   = True
                split_ratio = rev_split.ratio
                split_date  = rev_split.date
        except Exception as exc:
            logger.warning(f"[{symbol}] AV splits error: {exc}")
            failed_apis.append("av_splits")

        # --- news: YouTube → Momo → Finnhub → AV → Brave ---
        # Keep falling through sources until we find fresh news (within
        # NEWS_FRESH_MIN). Each source that returns articles replaces the
        # previous best. At the end, use whatever we have.
        news_count    = 0
        avg_sentiment = None
        articles: list = []
        _news_source = ""
        _session_cutoff, _, _ = news_freshness_cutoffs()
        _fetch_since = datetime.now(tz=timezone.utc) - timedelta(minutes=_news_lookback_min)

        def _is_fresh(arts: list) -> bool:
            return bool(arts and arts[0].published_at >= _session_cutoff)

        # Load persistent noise store titles for this symbol so previously
        # marked headlines are excluded from all news sources.
        from cache import NoiseStore
        _noise_exclude = NoiseStore.get().titles(symbol)

        # 0) YouTube live chat news (from designated news author)
        if yt_headlines:
            t_n = time.perf_counter()
            # Batch-score all headlines in one forward pass (3-5x faster)
            hl_texts = [hl.headline for hl in yt_headlines]
            hl_scores = self._finbert.score_headlines_batch(hl_texts) if FINBERT_ENABLED else [None] * len(hl_texts)
            for hl, score in zip(yt_headlines, hl_scores):
                if hl.headline in _noise_exclude:
                    continue
                if (
                    NOISE_FILTER_YT_HEADLINE
                    and not hl.bypass_noise_filter
                    and score is not None
                    and -NEWS_NOISE_THRESHOLD < score < NEWS_NOISE_THRESHOLD
                ):
                    logger.debug(
                        "[%s] YouTube headline noise-filtered (score=%.2f): \"%s\"",
                        symbol, score, hl.headline,
                    )
                    continue
                art = FinnhubNewsArticle(
                    symbol=symbol, title=hl.headline, source="youtube-chat",
                    url="", published_at=hl.timestamp, summary="",
                )
                articles.append(art)
                news_count += 1
                avg_sentiment = score
                _news_source = "youtube"
                logger.debug(
                    "[%s] YouTube headline accepted (score=%s): \"%s\"",
                    symbol, f"{score:.2f}" if score is not None else "N/A",
                    hl.headline,
                )
                break  # use the first non-noise headline
            stats["news_youtube_ms"] = f"{(time.perf_counter() - t_n) * 1000:.0f}"

        # 1) Momoscreener (fastest — already in memory, no API key)
        if not _is_fresh(articles):
            t_n = time.perf_counter()
            try:
                momo_news = self._momo.get_news(symbol)
                if momo_news:
                    raw = [
                        FinnhubNewsArticle(
                            symbol=n.symbol, title=n.headline, source=n.source,
                            url=n.url, published_at=n.published_at or datetime.now(tz=timezone.utc),
                            summary="",
                        )
                        for n in momo_news
                        if n.published_at and n.published_at >= _fetch_since
                    ]
                    arts, cnt, sent = self._filter_and_score(raw, symbol, _co_name, us_company=_us_company, skip_relevance=True, exclude_titles=_noise_exclude)
                    if cnt > 0:
                        articles, news_count, avg_sentiment, _news_source = arts, cnt, sent, "momo"
            except Exception as exc:
                logger.debug(f"[{symbol}] Momo news error: {exc}")
            stats["news_momo_ms"] = f"{(time.perf_counter() - t_n) * 1000:.0f}"

        # 2) Finnhub (already timed inside the thread)
        if not _is_fresh(articles):
            t_n = time.perf_counter()
            try:
                raw = futures_map["finnhub"].result()
                arts, cnt, sent = self._filter_and_score(raw, symbol, _co_name, us_company=_us_company, exclude_titles=_noise_exclude)
                if cnt > 0:
                    articles, news_count, avg_sentiment, _news_source = arts, cnt, sent, "finnhub"
            except Exception as exc:
                if FINNHUB_ENABLED:
                    logger.warning(f"[{symbol}] Finnhub unavailable ({exc}) — trying AV")
                    failed_apis.append("finnhub")
            stats["news_finnhub_net_ms"] = f"{(time.perf_counter() - t_n) * 1000:.0f}"

        # 3) Alpha Vantage (entity sentiment, already timed in thread)
        if not _is_fresh(articles) and use_av_news and "av_news" in futures_map:
            t_n = time.perf_counter()
            try:
                raw = futures_map["av_news"].result()
                arts, cnt, sent = self._filter_and_score(
                    raw, symbol, _co_name, use_entity=True, us_company=_us_company, exclude_titles=_noise_exclude,
                )
                if cnt > 0:
                    articles, news_count, avg_sentiment, _news_source = arts, cnt, sent, "alpha_vantage"
            except Exception as exc:
                logger.warning(f"[{symbol}] AV news error: {exc}")
                failed_apis.append("av_news")
            stats["news_av_ms"] = stats.get("news_av_ms") or f"{(time.perf_counter() - t_n) * 1000:.0f}"

        # 4) Brave Search
        if not _is_fresh(articles) and BRAVE_ENABLED:
            t_n = time.perf_counter()
            try:
                _exchange = profile.exchange if profile else ""
                raw = self._brave.search_ticker_news(symbol, exchange=_exchange)
                arts, cnt, sent = self._filter_and_score(raw, symbol, _co_name, us_company=_us_company, exclude_titles=_noise_exclude)
                if cnt > 0 and (not articles or arts[0].published_at > articles[0].published_at):
                    articles, news_count, avg_sentiment, _news_source = arts, cnt, sent, "brave"
            except Exception as exc:
                logger.warning(f"[{symbol}] Brave Search fallback error: {exc}")
                failed_apis.append("brave")
            stats["news_brave_ms"] = f"{(time.perf_counter() - t_n) * 1000:.0f}"

        stats["news_source"] = _news_source or "none"
        stats["failed_apis"] = ",".join(failed_apis) if failed_apis else ""

        _country = profile.country if profile and profile.country else ""

        return self._finalize_entry(
            s1=s1, df_daily=df_daily, float_data=float_data,
            sector=sector, industry=industry, country=_country,
            ipo_date_str=ipo_date_str, days_since_ipo=days_since_ipo,
            has_split=has_split, split_ratio=split_ratio, split_date=split_date,
            articles=articles, news_count=news_count,
            avg_sentiment=avg_sentiment, news_source=_news_source,
        )

    # ------------------------------------------------------------------
    # Shared entry construction
    # ------------------------------------------------------------------

    def _finalize_entry(
        self,
        s1: Stage1Result,
        df_daily: Optional[pd.DataFrame],
        float_data,
        sector: str,
        industry: str,
        country: str,
        ipo_date_str: Optional[str],
        days_since_ipo: Optional[int],
        has_split: bool,
        split_ratio: Optional[float],
        split_date: Optional[str],
        articles: list,
        news_count: int,
        avg_sentiment: Optional[float],
        news_source: str,
    ) -> WatchlistEntry:
        """Freshness tagging (display only) and WatchlistEntry assembly."""
        now_utc = datetime.now(tz=timezone.utc)
        session_cutoff, breaking_cutoff, _ = news_freshness_cutoffs()
        news_is_fresh = bool(articles and articles[0].published_at >= session_cutoff)
        news_is_breaking = bool(articles and articles[0].published_at >= breaking_cutoff)
        articles = articles[:1]

        # Noise gate: if sentiment is within the noise band, discard the article
        # entirely so it doesn't display or score.
        if (
            avg_sentiment is not None
            and -NEWS_NOISE_THRESHOLD <= avg_sentiment <= NEWS_NOISE_THRESHOLD
        ):
            articles = []
            news_count = 0
            avg_sentiment = None

        evaluation = self._build_evaluation(
            price=s1.price,
            volume=s1.volume,
            rvol_1day=s1.rvol_1day,
            news_count=news_count,
            avg_sentiment=avg_sentiment,
            float_m=float_data.float_shares_m,
            df_daily=df_daily,
            news_source=news_source,
        )

        setup_pattern = build_groups([SETUP_PATTERN_GROUP])[0]
        for v in setup_pattern.variables:
            v.reset(0.0)

        # Build criteria groups and seed Stage-1 attention values.
        # CRITERIA_EXECUTION starts all-failing (gates trip closed until the
        # live scorer confirms speed, imbalance, and acceleration).
        criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
        for cg in criteria_groups:
            if cg.name == "ATTENTION":
                cg.update("change_pct_min", s1.change_pct >= CHANGE_PCT_MIN, s1.change_pct)
                cg.update("volume_min", s1.volume >= ATTENTION_VOLUME_MIN, float(s1.volume))
                cg.update("rvol_1day_min", s1.rvol_1day >= ATTENTION_RVOL_1DAY_MIN, s1.rvol_1day)
                # dist_from_hod_atr_max, vwap_slope_positive, close_above_vwap,
                # macd_positive, order_book_density_ok — updated live by scorer

        # Session peak seed: prev_close is exact at eval time (price and
        # change_pct come from the same quote); the quote's daily high may
        # already exceed the eval-moment change.
        _prev_close = (
            s1.price / (1.0 + s1.change_pct / 100.0)
            if s1.price > 0 and s1.change_pct > -99.0 else 0.0
        )
        _peak_change = s1.change_pct
        if _prev_close > 0 and s1.daily_high > 0:
            _peak_change = max(_peak_change, (s1.daily_high - _prev_close) / _prev_close * 100.0)

        avg_volume = self._avg_daily_volume(df_daily, s1.avg_volume)

        entry = WatchlistEntry(
            symbol=s1.symbol,
            evaluated_at=now_utc,
            price=s1.price,
            change_pct=s1.change_pct,
            volume=s1.volume,
            avg_volume=avg_volume,
            rvol_1day=s1.rvol_1day,
            rvol_5min=s1.rvol_5min,
            daily_high=s1.daily_high,
            prev_close=_prev_close,
            peak_change_pct=_peak_change,
            float_shares_m=float_data.float_shares_m,
            free_float_pct=float_data.free_float_pct,
            float_source=float_data.source,
            sector=sector,
            industry=industry,
            country=country,
            ipo_date=ipo_date_str,
            days_since_ipo=days_since_ipo,
            has_recent_reverse_split=has_split,
            reverse_split_ratio=split_ratio,
            reverse_split_date=split_date,
            news_count_last_hour=news_count,
            news_is_fresh=news_is_fresh,
            news_is_breaking=news_is_breaking,
            avg_sentiment=avg_sentiment,
            news_articles=articles,
            news_source=news_source,
            evaluation=evaluation,
            setup_pattern=setup_pattern,
            criteria_groups=criteria_groups,
        )

        # Track which key data fields were unavailable from APIs
        _missing: list[str] = []
        if float_data.float_shares_m <= 0:
            _missing.append("float")
        if not sector:
            _missing.append("sector")
        if avg_volume <= 1:
            _missing.append("avg volume")
        entry.missing_data = _missing

        entry.static_confidence = entry.eval_confidence
        return entry

    @staticmethod
    def _avg_daily_volume(df_daily: Optional[pd.DataFrame], quote_avg: int) -> int:
        """Average daily volume over the last AVG_VOLUME_DAILY_BARS daily bars.

        IBKR snapshot quotes never carry avVolume (generic tick 165 is not
        requested), so the quote-supplied average arrives clamped to 1 —
        derive the real figure from the daily candles already fetched for
        the MAs. A quote average > 1 (Tradier) is trusted as-is.
        """
        if quote_avg > 1:
            return quote_avg
        if df_daily is None or df_daily.empty or "volume" not in df_daily.columns:
            return quote_avg
        tail = pd.to_numeric(df_daily["volume"], errors="coerce").dropna().tail(AVG_VOLUME_DAILY_BARS)
        if tail.empty:
            return quote_avg
        return max(int(tail.mean()), 1)

    # ------------------------------------------------------------------
    # Static confidence breakdown
    # ------------------------------------------------------------------

    def _build_evaluation(
        self,
        price: float,
        volume: int,
        rvol_1day: float,
        news_count: int,
        avg_sentiment: Optional[float] = None,
        float_m: float = 0.0,
        df_daily: Optional[pd.DataFrame] = None,
        news_source: str = "",
    ) -> CompositeEvaluation:
        """Build the 2-group quality evaluation from raw Stage 2 data."""

        groups = build_groups(SCORING_GROUPS)
        gmap: dict[str, VariableGroup] = {g.name: g for g in groups}

        # ── Determine noise gate ─────────────────────────────────────
        noise_gate = (
            avg_sentiment is not None
            and -NEWS_NOISE_THRESHOLD <= avg_sentiment <= NEWS_NOISE_THRESHOLD
        )

        # ── Build typed inputs ────────────────────────────────────────
        inputs = StaticInputs(
            float_shares=float_m if float_m > 0 else None,
            breaking_news=None if (news_count <= 0 or avg_sentiment is None or noise_gate)
                          else (avg_sentiment + 1.0) / 2.0,
            rvol_1day=rvol_1day if rvol_1day > 0 else None,
            price=price,
            daily_trend=self._daily_pct(df_daily),
        )

        ev = CompositeEvaluation(
            stock_eval=gmap["STOCK_EVAL"],
            technical_analysis=gmap["TECHNICAL_ANALYSIS"],
        )
        ev.apply_static(inputs)
        return ev

    # --- Individual static delta methods ---

    @staticmethod
    def _sector_quality(sector: str, industry: str) -> float:
        """Map sector/industry keywords to a 0-1 quality value."""
        combined = f"{sector} {industry}".lower()

        def matches(keywords: list) -> bool:
            for kw in keywords:
                kw_clean = kw.strip()
                if re.search(r'\b' + re.escape(kw_clean) + r'\b', combined):
                    return True
            return False

        if matches(SECTORS_TIER1): return 1.0
        if matches(SECTORS_TIER2): return 0.7
        if matches(SECTORS_TIER3): return 0.4
        return 0.0

    @staticmethod
    def _daily_pct(df_daily: Optional[pd.DataFrame]) -> float:
        """Daily base risk (%) over recent bars.

        Uses up to 20 completed daily bars before today.  The base risk starts with
        total chart range, then adds a selloff penalty when the window is
        clearly trending down and closing near its lows.  Lower remains better
        for the LINEAR_MIN scorer, but a controlled selloff no longer looks
        like a tight consolidation just because the high/low range compressed.
        """
        if df_daily is None or len(df_daily) < 2:
            return 0.0
        try:
            if "ts" in df_daily.columns:
                daily_ts = pd.to_datetime(df_daily["ts"], errors="coerce", utc=True)
                today = MarketClock.now_et().date()
                prior = df_daily.loc[daily_ts.dt.tz_convert(MarketClock.ET).dt.date < today].iloc[-20:]
            else:
                prior = df_daily.iloc[:-1].iloc[-20:]
            if prior.empty:
                return 0.0

            closes = pd.to_numeric(prior["close"], errors="coerce").dropna()
            highs = pd.to_numeric(prior["high"], errors="coerce").dropna()
            lows = pd.to_numeric(prior["low"], errors="coerce").dropna()
            if closes.empty or highs.empty or lows.empty:
                return 0.0

            max_high = float(highs.max())
            min_low = float(lows.min())
            avg_close = float(closes.mean())
            start_close = float(closes.iloc[0])
            end_close = float(closes.iloc[-1])
            if avg_close <= 0 or min_low <= 0 or start_close <= 0 or max_high <= 0:
                return 0.0

            range_pct = (max_high - min_low) / avg_close * 100.0
            window_return_pct = (end_close - start_close) / start_close * 100.0
            high_to_close_drawdown_pct = (max_high - end_close) / max_high * 100.0
            chart_range = max_high - min_low
            close_location = (end_close - min_low) / chart_range if chart_range > 0 else 0.5

            selloff_penalty = 0.0
            if window_return_pct < -8.0:
                selloff_penalty += (abs(window_return_pct) - 8.0) * 6.0
            if high_to_close_drawdown_pct > 15.0 and close_location < 0.35:
                selloff_penalty += (high_to_close_drawdown_pct - 15.0) * 5.0
            if len(closes) >= 5:
                lower_close_ratio = float((closes.diff().dropna() < 0).mean())
                if lower_close_ratio >= 0.6 and window_return_pct < -5.0:
                    selloff_penalty += 40.0

            return range_pct + selloff_penalty
        except Exception:
            return 0.0

    @staticmethod
    def _above_200(price: float, df_daily: Optional[pd.DataFrame]) -> float:
        """1.0 if price above both 200 EMA and 200 MA.

        0.0 only if below AND distance to the nearest 200-level is < 100%.
        If far below (>= 100% away), return neutral 0.5 — no penalty.
        """
        if df_daily is None or df_daily.empty:
            return 0.5  # unknown → neutral
        try:
            last = df_daily.iloc[-1]
            ema200 = last.get("ema_200") if "ema_200" in df_daily.columns else None
            ma200  = last.get("ma_200")  if "ma_200"  in df_daily.columns else None
            if ema200 is None and ma200 is None:
                return 0.5
            above_ema = (price > float(ema200)) if ema200 is not None and pd.notna(ema200) else True
            above_ma  = (price > float(ma200))  if ma200  is not None and pd.notna(ma200)  else True
            if above_ema and above_ma:
                return 1.0
            # Below at least one — check distance
            levels = []
            if ema200 is not None and pd.notna(ema200):
                levels.append(float(ema200))
            if ma200 is not None and pd.notna(ma200):
                levels.append(float(ma200))
            nearest = min(levels, key=lambda lv: abs(lv - price))
            dist_pct = (nearest - price) / price if price > 0 else 0
            if dist_pct >= 1.0:
                return 0.5  # too far away — neutral, no penalty
            return 0.0
        except Exception:
            return 0.5

    @staticmethod
    def _ath_ratio(price: float, df_daily: Optional[pd.DataFrame]) -> float:
        """Ratio of current price to all-time high (0-1+).  0.5 if unknown."""
        if df_daily is None or df_daily.empty:
            return 0.5
        try:
            all_time_high = float(df_daily["high"].max())
            if all_time_high <= 0:
                return 0.5
            return price / all_time_high
        except Exception:
            return 0.5

    @staticmethod
    def _delta_sector(sector: str, industry: str) -> float:
        """DEPRECATED — use _sector_quality() instead. Kept for test compat."""
        return StockEvaluator._sector_quality(sector, industry)

    @staticmethod
    def _delta_split(has_split: bool, ratio: Optional[float]) -> float:
        """DEPRECATED — quality evaluation replaces delta system."""
        return 1.0 if has_split else 0.0

    @staticmethod
    def _delta_ma200(price: float, df_daily: Optional[pd.DataFrame]) -> float:
        """DEPRECATED — quality evaluation replaces delta system."""
        return StockEvaluator._above_200(price, df_daily)