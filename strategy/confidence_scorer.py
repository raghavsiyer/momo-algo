"""
strategy/confidence_scorer.py

Dynamic confidence scorer — runs background threads once a stock passes all
three stages. Subscribes to one combined Tradier stream session (timesale +
quote filters) and polls REST 1-min candles for accurate OHLCV data.

DATA SOURCES AND WHAT EACH FEEDS:
    LiveTimesale events  ->  ROC attention, tape activity, order flow,
        stream-built candles, pattern/setup state, and level-respect scoring.

    LiveQuote events  ->  big-seller pressure, NBBO spread, and IBKR-gated
        order-flow inputs when depth data is available.

    REST-polled 1-min candles  ->  candle refresh, attention criteria, hidden
        seller updates, and multi-timeframe technical-analysis context.

LIVE SCORING MODEL:
    DynamicScorer does not apply additive score deltas. It pushes raw
    measurements into ``WatchlistEntry.evaluation`` through ``LiveInputs``;
    the TECHNICAL_ANALYSIS group owns the weights/quality functions, and
    ``WatchlistEntry.live_confidence`` reports that layer independently.

    Some signals are intentionally non-scoring: ROC(p=3) is stored in
    ``_last_roc`` / ``ScorerSnapshot.last_roc``, but it does not update an
    attention criterion or a scored live variable.

USAGE:
    from strategy.confidence_scorer import DynamicScorer

    scorer = DynamicScorer(entry, rmap, avg_daily_volume=entry.avg_volume)
    scorer.start()

    # entry.live_confidence now updates live
    # Read scorer.snapshot() or scorer.format_summary() at any time

    scorer.stop()
"""

import logging
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from chart_cache import LiveChartCache, SOURCE_POLLED, SOURCE_STREAM, get_chart_cache_writer
from market_data.tradier_rest import OHLCVCandle, TradierClientError
from market_data.tradier_stream import LiveTimesale, LiveQuote
from market_data.provider import (
    get_historical_client,
    get_shared_stream,
    release_thread_connection as _release_ibkr_thread_conn,
)
from strategy.evaluator import WatchlistEntry
from strategy.scoring import (
    VariableGroup,
    LiveInputs,
    combine_tf_scores,
    build_groups,
    compute_quality,
    MIN_TF_EVIDENCE_MASS,
)
from technical_analysis.resistance import ResistanceMap, ResistanceLevel
from technical_analysis.patterns import (
    PatternSignal,
    PatternSnapshot,
    scan_all_patterns,
    detect_pullback_new_high,
    detect_pullback_detected,
    detect_pullback_pending,
    detect_move_start,
    _TIMEFRAME_RANK,
    _atr_from_arrays,
    _candle_volume_response,
    _smoothstep,
    _volume_response_baseline,
    latest_extended_session_start,
)
from exec.order_flow import ClassifiedTrade, OrderFlowTracker
from exec.engine import SimulationEngine
from exec.signals import (
    TapeGates,
    entry_band_floor,
    evaluate_entry,
    resistance_search_floor,
    tape_entry_gates,
)
from exec.setup_logger import SetupSignal, SetupSignalLogger, get_setup_signal_logger
from config import (
    ENTRY_CONFIDENCE_MIN,
    SIM_ENABLED,
    SETUP_TFS_DISABLED,
    SIM_FIRST_MACD_CYCLE_ONLY,
    SIM_MAX_SETUP_EXTENSION_PCT,
    SIM_SETUP_SCORE_MIN,
    SIM_PRESSURE_SPEED_MIN,
    SIM_PRESSURE_VOL_IMBALANCE_MIN,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_ACCEL_Z,
    SIM_OFI_NORM_MIN,
    SIM_PRESSURE_PROGRESS_MIN,
    SCORING_GROUPS,
    SETUP_PATTERN_GROUP,
    SETUP_REJECTED_LEVEL_TOL_PCT,
    IBKR_ENABLED,
    SLOT_PIN_NONE,
    SLOT_PIN_ARMED,
    SLOT_PIN_CONFIRMING,
    SLOT_PIN_POSITION,
    ORDER_BOOK_MIN_DEPTH_SHARES,
    SPREAD_MAX_UNDER_1, SPREAD_MAX_1_TO_5, SPREAD_MAX_10_TO_20,
    EXEC_SPREAD_PCT_MAX,
    EXEC_DIST_TO_RESISTANCE_MULT,
    EXEC_HIDDEN_SELLER_MAX,
    RESISTANCE_BREAK_BUFFER,
    _dim,
    # Dynamic scorer tuning
    ROC_TS_PERIOD,
    OFI_WINDOW_SECS,
    PRESSURE_WINDOW_SECS,
    TAPE_BASELINE_WINDOW_SECS,
    BOOK_ILLIQUID_MAX_SIZE, BOOK_ILLIQUID_CONSEC,
    RVOL_WINDOW_SECONDS, RVOL_BARS_PER_SESSION,
    SPREAD_MAX_5_TO_10,
    SCORER_STOP_TIMEOUT_SECS,
    SCORER_CANDLE_MAX_BARS,
    SCORER_MAX_RECONSTRUCT_TICKS,
    SCORER_POLL_INTERVAL_SECS,
    SCORER_POLL_JITTER_SECS,
    SCORER_QUOTE_PRICE_FALLBACK_SECS,
    TS_MIN_NOTIONAL,
    NO_NEWS_GRACE_SECS,
    NO_NEWS_PENALTY_QUALITY,
    # OFI thresholds
    OFI_SELL_DOMINANT_THRESH, OFI_BUY_SURGE_THRESH, OFI_HYSTERESIS,
    BOOK_LEAN_BEARISH_THRESH,
    OFI_CHOP_RATIO_THRESH, OFI_EXHAUSTION_THRESH, OFI_EXHAUSTION_MIN_QUOTES,
    TOPPING_TAIL_ATR_PERIOD,
    TOPPING_TAIL_DECAY,
    CANDLE_CHOP_MIN_CANDLES,
    CHANGE_PCT_MIN,
    ATTENTION_VOLUME_MIN,
    ATTENTION_RVOL_1DAY_MIN,
    ATTENTION_DIST_FROM_HOD_ATR_MAX,
    ATTENTION_VWAP_SLOPE_MIN,
    PRICE_ACTION_TF_WEIGHTS,
    PRICE_ACTION_TF_2M_ENABLED,
    SCORING_LOG_ENABLED,
    SETUP_OUTCOME_SUCCESS_PCT,
    SETUP_OUTCOME_DECAY,
    SETUP_OUTCOME_RECOVERY_MAX_CREDIT,
    SETUP_VOLUME_SEVERITY_FLOOR,
)

logger = logging.getLogger(__name__)

# Lazy import to avoid circular — only used when SCORING_LOG_ENABLED
_scoring_logger = None
def _get_scoring_logger():
    global _scoring_logger
    if _scoring_logger is None and SCORING_LOG_ENABLED:
        from exec.scoring_log import ScoringLogger
        _scoring_logger = ScoringLogger.get()
    return _scoring_logger

_LOG_EVERY_N_PRINTS = 50  # log live vars every N timesale prints (between polls)
# Cap on stream events parked while a scorer replays cached ticks (see
# DynamicScorer._event_hold). A 25k-tick reconstruction runs ~15s at most;
# a halt-resume burst tops out ~1k prints/s, so this is far beyond need.
_HELD_EVENTS_MAX = 200_000

# How long to keep a just-resolved setup pattern visible in the display
# before clearing it.  Gives the user a brief "Resolved" badge so it
# doesn't blink out the moment price moves away.
_SETUP_RESOLVED_DISPLAY_SECS: float = 10.0


class _NullChartCache:
    def drain(self, timeout=None) -> bool:
        return True

    def record_timesale(self, event) -> None:
        return

    def flush_timesales(self) -> None:
        return

    def record_book(self, snapshot) -> None:
        return

    def flush_books(self) -> None:
        return

    def record_candles(self, timeframe: str, candles, source: str) -> None:
        return


# ---------------------------------------------------------------------------
# Setup outcome tracking
# ---------------------------------------------------------------------------

@dataclass
class _ActiveLeg:
    """An ongoing continuation leg from a resolved setup breakout."""
    timeframe: str
    quality: float            # PatternSignal.strength at resolution time (0-1)
    breakout_price: float     # trigger/entry price
    highest_close: float      # highest close seen so far in the leg
    pullback_ts: datetime | None = None  # first pullback candle, if known
    candle_count: int = 0     # green candles in the leg so far
    peak_volume: float = 0.0  # largest closed-candle volume in the leg so far
                              # (breakout candle onward — pullback candles are
                              # deliberately excluded: healthy pullbacks are
                              # low-volume, so leg significance is carried by
                              # its breakout/continuation/failure candles)

    @property
    def extension_pct(self) -> float:
        if self.breakout_price <= 0:
            return 0.0
        return (self.highest_close - self.breakout_price) / self.breakout_price


@dataclass
class _CompletedOutcome:
    """A finalized setup outcome (leg ended by red candle close)."""
    timeframe: str
    quality: float            # setup quality at breakout
    breakout_price: float     # failed setup trigger/entry price
    pullback_ts: datetime | None
    extension_pct: float      # max extension achieved
    success: bool             # extension_pct >= SETUP_OUTCOME_SUCCESS_PCT
    failure_ts: datetime      # candle timestamp when the setup failed
    failure_close: float      # close of the failure candle
    failure_quality: float     # failure candle quality: 1.0 harmless, 0.0 severe rejection
    failure_rejection: float   # volume-response rejection risk of the failure candle (0-1)
    mono_ts: float            # time.monotonic() when completed
    peak_volume: float = 0.0  # largest closed-candle volume the leg printed
                              # (breakout candle through failure candle) —
                              # weighed against the CURRENT volume baseline at
                              # every rescore, like key_candles


# ---------------------------------------------------------------------------
# Snapshot (readable from outside)
# ---------------------------------------------------------------------------

@dataclass
class ScorerSnapshot:
    symbol: str
    eval_confidence: float
    live_confidence: float
    last_price: float
    last_roc: float
    spread: float
    rvol_5min: float
    prints_seen: int
    candles_built: int
    captured_at: datetime
    patterns: PatternSnapshot = None  # type: ignore[assignment]
    # Order flow fields
    book_lean: float = 0.0
    buying_pressure: int = 0
    selling_pressure: int = 0
    ofi_rolling: int = 0
    trade_imbalance: float = 0.0
    buy_volume: int = 0
    sell_volume: int = 0
    trade_speed: float = 0.0
    pressure_acceleration: float = 0.0
    pressure_acceleration_ready: bool = False
    baseline_speed: float = 0.0
    speed_ratio: float = 0.0
    # Per-group scores for display
    group_scores: dict = field(default_factory=dict)
    setup_pattern_score: float = 0.0
    tf_pa_scores: dict = field(default_factory=dict)
    # Per-TF per-variable qualities for TA table display:
    #   { tf_label: { var_name: quality_0_to_1, ... }, ... }
    tf_pa_var_qualities: dict = field(default_factory=dict)
    # Toolbar classification helpers
    dist_from_hod_atrs: float = 0.0  # (HOD - price) / ATR
    move_start_ts: datetime | None = None

    def __post_init__(self) -> None:
        if self.patterns is None:
            self.patterns = PatternSnapshot()


# ---------------------------------------------------------------------------
# Order-flow views — what the `tape` and `ofi` REPL displays render
# ---------------------------------------------------------------------------
# Both are plain-data copies assembled under the scorer lock, so the prompt
# event loop can format them without holding it.  Every field is read from
# state the tracker already maintains; nothing here fetches or subscribes.


@dataclass(frozen=True, slots=True)
class TapeView:
    """The live T&S tape plus every tape var the algo derives from it."""
    symbol: str
    prints: tuple[ClassifiedTrade, ...]   # newest first
    retained: int          # directional prints held in the rolling window
    window_secs: int       # span of that window
    prints_seen: int       # lifetime prints ingested (includes mid/neutral)
    last_price: float
    bid: float
    ask: float
    last_event_ts: datetime | None
    captured_at: datetime
    # Entry gates — the five the algo actually gates entries on.
    gates: TapeGates
    # Raw reads behind (and alongside) the gates.
    trade_speed: float          # boxcar prints/sec over the pressure window
    tape_speed_ema: float       # fast-kernel instantaneous prints/sec
    baseline_speed: float
    speed_ratio: float
    buy_ratio: float            # boxcar
    buy_ratio_ema: float        # fast kernel
    buy_volume: int
    sell_volume: int
    trade_imbalance: float
    buying_pressure: int
    selling_pressure: int
    avg_print_size: float
    tape_prints_per_sec: float
    density_imbalance: float
    accel_raw: float
    accel_z: float
    accel_ready: bool
    accel_n_eff: float
    dispersion_phi: float
    hidden_seller: float


@dataclass(frozen=True, slots=True)
class BookView:
    """The live order book plus the OFI values computed from it."""
    symbol: str
    bids: tuple[tuple[float, int], ...]   # best first
    asks: tuple[tuple[float, int], ...]
    depth_active: bool     # L2 feed fresh — OFI is level-weighted deep OFI
    l1_bid: float
    l1_bid_size: int
    l1_ask: float
    l1_ask_size: int
    last_price: float
    last_book_ts: datetime | None
    last_event_ts: datetime | None
    captured_at: datetime
    # OFI + book-shape reads
    ofi_rolling: int
    ofi_rate: float
    ofi_normalized: float
    ofi_norm_gate: float   # gate-source read (EMA by default) the entry/exit enforce
    ofi_samples: int
    ofi_window_secs: int
    ofi_chop: float
    ofi_intensity: float
    book_lean: float
    book_lean_raw: float
    book_depth_avg: float
    quote_count: int


# ---------------------------------------------------------------------------
# DynamicScorer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-timeframe candle buffer (stream-driven)
# ---------------------------------------------------------------------------

# Ordered from fastest to slowest
_TF_INTERVALS: list[tuple[str, int]] = [
    ("10sec", 10),
    ("1min",  60),
    ("2min",  120),
    ("5min",  300),
]
_CHART_CACHE_TF = {
    "10sec": "10s",
    "1min": "1m",
    "2min": "2m",
    "5min": "5m",
}

_HOD_RESPONSE_TFS = ("10sec", "1min", "2min", "5min")


class _MultiTFCandleBuffer:
    """
    Aggregates T&S ticks into OHLCV candles across four timeframes.

    On each tick, updates the current (open) bar for every timeframe.
    When a tick crosses a bucket boundary the previous bar is "closed"
    and the set of closed timeframes is returned so the caller can
    trigger pattern detection at the appropriate frequency.

    Thread-safety: caller must hold a lock around ``on_tick``.
    """

    def __init__(self, symbol: str, max_bars: int = SCORER_CANDLE_MAX_BARS) -> None:
        self._sym = symbol
        self._bars: dict[str, deque[OHLCVCandle]] = {
            tf: deque(maxlen=max_bars) for tf, _ in _TF_INTERVALS
        }
        # Current (incomplete) bar per timeframe — None until first tick
        self._current: dict[str, Optional[OHLCVCandle]] = {
            tf: None for tf, _ in _TF_INTERVALS
        }
        # Cached DataFrames: {timeframe: (DataFrame, n_closed_bars, last_closed_ts)}
        self._df_cache: dict[str, tuple[pd.DataFrame, int, datetime | None]] = {}
        # Oldest timestamp the buffer must retain (start of the active move).
        # Closed bars whose ts is at/after this point are never evicted — the
        # deque grows instead — so choppiness/pattern context spans the whole
        # move rather than sliding into a partial trailing window.
        self._retention_start: datetime | None = None
        # Backtest-only: keep closed bars chronological + ts-deduped even when a
        # source feeds out-of-order/overlapping bars (the replay re-seeds whole-
        # session candle context on every near-HOD window entry, which would
        # otherwise drop stale bars into the middle of the live sequence and
        # silently break pattern detection).  Off in live — zero behaviour change.
        self._reconcile: bool = False

    def ensure_capacity(self, min_bars: int) -> None:
        """Grow closed-bar buffers so a historical seed is kept intact."""
        if min_bars <= 0:
            return
        for tf, _ in _TF_INTERVALS:
            buf = self._bars[tf]
            if buf.maxlen is not None and buf.maxlen >= min_bars:
                continue
            self._bars[tf] = deque(buf, maxlen=min_bars)
        self._df_cache.clear()

    def set_retention_start(self, ts: datetime | None) -> None:
        """Set the oldest timestamp to retain (start of the active move)."""
        self._retention_start = ts

    def _rebuild_sorted(self, tf: str, incoming: list[OHLCVCandle], incoming_wins: bool) -> None:
        """Rebuild the *tf* deque as a chronological, timestamp-deduped set.

        ``incoming_wins`` sets precedence at a shared timestamp: True for a
        freshly-closed tick bar (authoritative), False for a batch seed (existing
        tick-built bars keep precedence).  Used only in reconcile (backtest) mode.
        """
        buf = self._bars[tf]
        merged: dict[float, OHLCVCandle] = {}
        first, second = (list(buf), incoming) if incoming_wins else (incoming, list(buf))
        for c in first:
            merged[c.ts.timestamp()] = c
        for c in second:
            merged[c.ts.timestamp()] = c
        ordered = [merged[k] for k in sorted(merged)]
        maxlen = buf.maxlen
        if maxlen is not None and maxlen < len(ordered):
            maxlen = len(ordered)
        self._bars[tf] = deque(ordered, maxlen=maxlen)
        self._df_cache.pop(tf, None)

    def _append_closed(self, tf: str, bar: OHLCVCandle) -> None:
        """Append a closed bar, growing the deque instead of evicting a bar
        that is still within the retention window (start of move → now)."""
        buf = self._bars[tf]
        if self._reconcile and buf and bar.ts <= buf[-1].ts:
            # Out-of-order / duplicate close (backtest seed+tick overlap): upsert
            # by timestamp so the buffer stays chronological; the tick bar wins.
            self._rebuild_sorted(tf, [bar], incoming_wins=True)
            return
        if (
            buf.maxlen is not None
            and len(buf) >= buf.maxlen
            and buf
            and self._retention_start is not None
            and buf[0].ts >= self._retention_start
        ):
            # Evicting the oldest bar would drop move-window context — grow.
            buf = deque(buf, maxlen=buf.maxlen * 2)
            self._bars[tf] = buf
        buf.append(bar)

    @staticmethod
    def _bucket_ts(ts: datetime, interval_secs: int) -> datetime:
        epoch = ts.timestamp()
        floored = (epoch // interval_secs) * interval_secs
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    def on_tick(self, price: float, size: int, ts: datetime) -> list[str]:
        """
        Process a single T&S print.

        Returns a list of timeframe labels whose candle just closed
        (e.g. ``["10sec"]`` or ``["10sec", "1min"]``).  Empty list
        if no candle closed.
        """
        closed: list[str] = []
        for tf, secs in _TF_INTERVALS:
            bar_ts = self._bucket_ts(ts, secs)
            cur = self._current[tf]
            if cur is not None and cur.ts != bar_ts:
                # Bucket boundary crossed — close previous bar
                self._append_closed(tf, cur)
                closed.append(tf)
                cur = None  # will open new bar below
            if cur is None:
                self._current[tf] = OHLCVCandle(
                    symbol=self._sym, ts=bar_ts,
                    open=price, high=price, low=price, close=price,
                    volume=size, interval=tf, vwap=price,
                )
            else:
                old_vol = int(cur.volume)
                new_vol = old_vol + size
                old_vwap = float(cur.vwap) if cur.vwap is not None else float(cur.close)
                cur.high = max(cur.high, price)
                cur.low = min(cur.low, price)
                cur.close = price
                cur.volume = new_vol
                if new_vol > 0:
                    cur.vwap = ((old_vwap * old_vol) + (price * size)) / new_vol
        return closed

    def _closed_frame(self, timeframe: str) -> pd.DataFrame:
        """Cached DataFrame of CLOSED bars only (shared object — read-only).

        Rebuilt only when a bar is appended/rotated. A full deque can rotate
        while its length stays unchanged, so the newest closed timestamp is
        part of the cache key.
        """
        cache_key = timeframe
        bars = self._bars.get(timeframe, deque())
        n_closed = len(bars)
        last_closed_ts = bars[-1].ts if bars else None
        cached, cached_n, cached_last_ts = self._df_cache.get(cache_key, (None, -1, None))
        if cached is None or cached_n != n_closed or cached_last_ts != last_closed_ts:
            if not bars:
                base = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
                base.index.name = "ts"
            else:
                base = pd.DataFrame(
                    [(b.ts, b.open, b.high, b.low, b.close, b.volume, b.vwap) for b in bars],
                    columns=["ts", "open", "high", "low", "close", "volume", "vwap"],
                ).set_index("ts")
            self._df_cache[cache_key] = (base, n_closed, last_closed_ts)
            cached = base
        return cached

    def closed_df(self, timeframe: str) -> pd.DataFrame:
        """Closed bars for *timeframe* as a DataFrame, WITHOUT the forming bar.

        Returns the buffer's cached frame itself — callers must treat it as
        read-only. This is the per-print accessor: unlike :meth:`to_df` it
        performs no concat and no copy, so on a hot tape it costs a dict
        lookup instead of ~0.8 ms of DataFrame construction.
        """
        return self._closed_frame(timeframe)

    def to_df(self, timeframe: str) -> pd.DataFrame:
        """
        Build an OHLCV DataFrame from closed bars for the given timeframe.
        Includes the current (incomplete) bar at the end.

        Uses a cached DataFrame that is only rebuilt when bars are
        appended/rotated, avoiding full reconstruction on every call.
        """
        cur = self._current.get(timeframe)
        cached = self._closed_frame(timeframe)

        if cur is not None:
            # Append current bar as a single-row DF (cheap concat)
            cur_row = pd.DataFrame(
                [(cur.open, cur.high, cur.low, cur.close, cur.volume, cur.vwap)],
                columns=["open", "high", "low", "close", "volume", "vwap"],
                index=pd.Index([cur.ts], name="ts"),
            )
            return pd.concat([cached, cur_row])
        return cached.copy() if not cached.empty else pd.DataFrame()

    def current_bar(self, timeframe: str) -> Optional[OHLCVCandle]:
        """Return the current (incomplete) bar for *timeframe*."""
        return self._current.get(timeframe)

    def last_closed_bar(self, timeframe: str) -> Optional[OHLCVCandle]:
        """Return the most recent closed bar for *timeframe*."""
        bars = self._bars.get(timeframe)
        if not bars:
            return None
        return bars[-1]

    def seed(self, timeframe: str, candles: list[OHLCVCandle]) -> None:
        """
        Backfill historical candles from REST into the buffer.
        Called once at startup so pattern detection has context
        before the first candle closes from the stream.
        """
        buf = self._bars.get(timeframe)
        if buf is None or not candles:
            return
        if self._reconcile:
            # Merge chronologically, deduped by ts; existing (tick-built) bars
            # win so a whole-session re-seed can't reorder the live sequence.
            self._rebuild_sorted(timeframe, candles, incoming_wins=False)
            return
        self.ensure_capacity(len(candles))
        buf = self._bars[timeframe]
        for c in candles:
            buf.append(c)
        # Invalidate cached DF for this timeframe
        self._df_cache.pop(timeframe, None)

    @property
    def bar_counts(self) -> dict[str, int]:
        """Number of closed bars per timeframe (for snapshot)."""
        return {tf: len(self._bars[tf]) for tf, _ in _TF_INTERVALS}


class DynamicScorer:
    """
    Subscribes to live market data and pushes real-time measurements into the
    WatchlistEntry's composite evaluation.
    """

    @staticmethod
    def _coerce_utc_dt(value: datetime | pd.Timestamp | None) -> datetime | None:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()

    @staticmethod
    def _normalize_candle_df(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        normalized = df.copy()
        if "ts" in normalized.columns:
            normalized["ts"] = pd.to_datetime(normalized["ts"], utc=True)
            normalized = normalized.sort_values("ts")
            normalized = normalized.drop_duplicates(subset=["ts"], keep="last")
        return normalized.reset_index(drop=True)

    @staticmethod
    def _normalize_timesales_df(df: pd.DataFrame | None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        required = {"ts", "price", "size"}
        if not required.issubset(df.columns):
            return pd.DataFrame()
        normalized = df.copy()
        normalized["ts"] = pd.to_datetime(normalized["ts"], utc=True, errors="coerce")
        normalized["price"] = pd.to_numeric(normalized["price"], errors="coerce")
        normalized["size"] = pd.to_numeric(normalized["size"], errors="coerce")
        for column in ("bid", "ask", "seq"):
            if column not in normalized.columns:
                normalized[column] = 0
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0)
        for column in ("symbol", "exchange", "flag", "session"):
            if column not in normalized.columns:
                normalized[column] = ""
            normalized[column] = normalized[column].fillna("").astype(str)
        normalized = normalized.dropna(subset=["ts", "price", "size"]).copy()
        if normalized.empty:
            return pd.DataFrame()
        normalized["size"] = normalized["size"].astype(int)
        normalized["seq"] = normalized["seq"].astype(int)
        normalized = normalized.loc[(normalized["price"] > 0) & (normalized["size"] > 0)].copy()
        if normalized.empty:
            return pd.DataFrame()
        return normalized.sort_values(["ts", "seq"]).reset_index(drop=True)

    def _timesale_seed_tf_start(self) -> datetime | None:
        latest_10s = self._latest_df_ts(self._df_10sec_seed)
        if latest_10s is not None:
            return latest_10s + timedelta(seconds=10)
        latest_1m = self._latest_df_ts(self._df_1min_cache)
        if latest_1m is not None:
            return latest_1m + timedelta(minutes=1)
        return None

    def _timesale_from_seed_row(self, row) -> LiveTimesale | None:
        try:
            ts = pd.Timestamp(row["ts"])
            if ts.tzinfo is None:
                ts = ts.tz_localize(timezone.utc)
            else:
                ts = ts.tz_convert("UTC")
            return LiveTimesale(
                event_type="timesale",
                symbol=str(row.get("symbol") or self._sym).upper(),
                exchange=str(row.get("exchange") or ""),
                last=float(row["price"]),
                bid=float(row.get("bid", 0.0) or 0.0),
                ask=float(row.get("ask", 0.0) or 0.0),
                size=int(row["size"]),
                ts=ts.to_pydatetime(),
                seq=int(row.get("seq", 0) or 0),
                flag=str(row.get("flag") or ""),
                cancel=False,
                correction=False,
                session=str(row.get("session") or ""),
            )
        except Exception:
            return None

    def _seed_timesales_from_cache_locked(self) -> int:
        if self._df_timesales_seed.empty:
            return 0

        seed = self._normalize_timesales_df(self._df_timesales_seed)
        self._df_timesales_seed = pd.DataFrame()
        if seed.empty:
            return 0

        latest_ts = pd.to_datetime(seed["ts"], utc=True).max()
        if pd.isna(latest_ts):
            return 0
        lookback_seconds = max(
            RVOL_WINDOW_SECONDS,
            OFI_WINDOW_SECS,
            PRESSURE_WINDOW_SECS + TAPE_BASELINE_WINDOW_SECS,
        )
        rolling_start = latest_ts.to_pydatetime() - timedelta(seconds=lookback_seconds)
        tf_start = self._timesale_seed_tf_start()

        ts_series = pd.to_datetime(seed["ts"], utc=True)
        rolling_mask = ts_series >= pd.Timestamp(rolling_start)
        if tf_start is not None:
            tf_mask = ts_series >= pd.Timestamp(tf_start)
            seed = seed.loc[rolling_mask | tf_mask].copy()
        else:
            seed = seed.loc[rolling_mask].copy()
        if seed.empty:
            return 0

        replayed = 0
        for _, row in seed.iterrows():
            event = self._timesale_from_seed_row(row)
            if event is None or event.cancel or event.correction:
                continue

            price = event.last
            size = event.size
            now = event.ts
            if price * size < TS_MIN_NOTIONAL:
                continue

            self._prints_seen += 1
            self._last_event_ts = now
            self._last_price = price
            self._tick_prices.append(price)
            self._raise_hod(price)
            if event.bid > 0:
                self._bid = event.bid
            if event.ask > 0:
                self._ask = event.ask

            self._vol_prints.append((now, size))
            self._vol_prints_sum += size
            cutoff = now - timedelta(seconds=RVOL_WINDOW_SECONDS)
            while self._vol_prints and self._vol_prints[0][0] < cutoff:
                self._vol_prints_sum -= self._vol_prints[0][1]
                self._vol_prints.popleft()
            self._last_trim_epoch = now.timestamp()

            self._eval_roc()
            self._eval_resistance(price)
            feed_tf = tf_start is None or now >= tf_start
            if feed_tf:
                closed_tfs = self._tf_buf.on_tick(price, size, now)
                self._update_stream_indicators(now, closed_tfs)

            self._push_entry_band_to_tracker()
            self._ofi_tracker.on_timesale(event)
            self._update_tick_criteria(price)
            replayed += 1

        if replayed and self._on_update is not None:
            self._on_update()
        return replayed

    def _reconstruct_pattern_state_from_ticks_locked(self) -> int:
        """Rebuild the pattern buffer, setup-outcome history and candle
        choppiness by replaying the cached tape over the *entire* active move
        window through the setup/candle pipeline.

        This is what lets ``setup_outcome`` and ``candle_choppiness`` survive a
        mid/post-move restart (e.g. ``TICKER -r``).  The plain candle seed only
        fills the buffer; it never runs the tick-driven setup detection, because
        intracandle breakouts need the forming-bar mechanics of ``on_tick``.
        Replaying the tape reconstructs the exact live state the scorer would
        have had if it had streamed continuously from the start of the move.

        Runs in warm-up mode (``self._warmup``) so the setup-signal log, sim
        engine, display callbacks and chart cache are left untouched.  VWAP/EMA
        are deliberately NOT recomputed here (the session seed + live stream own
        them) so move-window volume is not double-counted.

        Caller must hold ``self._lock``.
        """
        if self._df_timesales_seed is None or self._df_timesales_seed.empty:
            return 0
        seed = self._normalize_timesales_df(self._df_timesales_seed)
        self._df_timesales_seed = pd.DataFrame()
        if seed.empty:
            return 0
        ts_series = pd.to_datetime(seed["ts"], utc=True)
        latest_ts = ts_series.max()
        if pd.isna(latest_ts):
            return 0
        active_start = self._active_price_action_start(latest_ts.to_pydatetime())
        seed = seed.loc[ts_series >= pd.Timestamp(active_start)].copy()
        if seed.empty:
            return 0

        # Retain the whole move window in the pattern buffer (no eviction) so
        # candle choppiness spans start-of-move → now.
        self._tf_buf.set_retention_start(active_start)

        self._warmup = True
        replayed = 0
        try:
            for _, row in seed.iterrows():
                event = self._timesale_from_seed_row(row)
                if event is None or event.cancel or event.correction:
                    continue
                price = event.last
                size = event.size
                now = event.ts
                if price * size < TS_MIN_NOTIONAL:
                    continue

                self._prints_seen += 1
                self._last_event_ts = now
                self._last_price = price
                self._tick_prices.append(price)
                self._raise_hod(price)
                if event.bid > 0:
                    self._bid = event.bid
                if event.ask > 0:
                    self._ask = event.ask

                self._vol_prints.append((now, size))
                self._vol_prints_sum += size
                cutoff = now - timedelta(seconds=RVOL_WINDOW_SECONDS)
                while self._vol_prints and self._vol_prints[0][0] < cutoff:
                    self._vol_prints_sum -= self._vol_prints[0][1]
                    self._vol_prints.popleft()
                self._last_trim_epoch = now.timestamp()

                self._eval_roc()

                # Build candles + run the tick-driven setup/choppiness pipeline.
                closed_tfs = self._tf_buf.on_tick(price, size, now)
                if closed_tfs:
                    self._scan_patterns_on_close(closed_tfs)
                    self._on_setup_leg_candle_close(closed_tfs)
                    self._check_breakout_candle_rejection(closed_tfs, now)
                self._check_intracandle_pullback_detection(price)
                self._check_intracandle_breakout(price, now)

                self._push_entry_band_to_tracker()
                self._ofi_tracker.on_timesale(event)
                self._update_tick_criteria(price)
                replayed += 1
        finally:
            self._warmup = False
        return replayed

    @staticmethod
    def _clip_df_since(df: pd.DataFrame, start_ts: datetime | None) -> pd.DataFrame:
        if df.empty or start_ts is None or "ts" not in df.columns:
            return df.copy()
        clipped = df[pd.to_datetime(df["ts"], utc=True) >= pd.Timestamp(start_ts)]
        return clipped.reset_index(drop=True)

    @staticmethod
    def _latest_df_ts(df: pd.DataFrame) -> datetime | None:
        if df.empty or "ts" not in df.columns:
            return None
        latest = pd.to_datetime(df["ts"], utc=True).max()
        if pd.isna(latest):
            return None
        return latest.to_pydatetime()

    def _active_price_action_start(self, reference_ts: datetime | pd.Timestamp | None = None) -> datetime:
        ref = self._coerce_utc_dt(reference_ts) or datetime.now(tz=timezone.utc)
        session_start = latest_extended_session_start(ref)
        if self._move_start_ts is None:
            return session_start
        return max(session_start, self._move_start_ts)

    def _price_action_candle_windows(
        self,
        tf_label: str,
        candles: list[OHLCVCandle],
    ) -> tuple[list[OHLCVCandle], list[OHLCVCandle]]:
        if tf_label == "1d" or not candles:
            return candles, candles
        reference_ts = self._coerce_utc_dt(candles[-1].ts)
        session_start = latest_extended_session_start(reference_ts)
        active_start = self._active_price_action_start(reference_ts)
        session_candles: list[OHLCVCandle] = []
        active_candles: list[OHLCVCandle] = []
        for candle in candles:
            candle_ts = self._coerce_utc_dt(candle.ts)
            if candle_ts is None or candle_ts < session_start:
                continue
            session_candles.append(candle)
            if candle_ts >= active_start:
                active_candles.append(candle)
        return active_candles, session_candles

    def __init__(
        self,
        entry: WatchlistEntry,
        rmap: ResistanceMap,
        avg_daily_volume: int = 0,
        daily_candles: list[OHLCVCandle] | None = None,
        seed_1min_session: pd.DataFrame | None = None,
        seed_10sec: pd.DataFrame | None = None,
        seed_timesales: pd.DataFrame | None = None,
        move_start_ts: datetime | pd.Timestamp | None = None,
        cache_live_data: bool = True,
        on_update: Optional[callable] = None,
    ) -> None:
        self._entry  = entry
        self._rmap   = rmap
        self._sym    = entry.symbol
        self._avg_vol = avg_daily_volume or entry.avg_volume
        self._stop   = threading.Event()
        self._lock   = threading.Lock()
        # True only while replaying cached ticks at seed time so live-only side
        # effects (sim engine, display refresh, setup-signal log, chart cache)
        # are suppressed during reconstruction.
        self._warmup = False
        # Event hold (see _event_hold): while the poll thread replays cached
        # ticks under _lock, live stream events are parked here instead of
        # letting the shared stream thread block on that lock — a blocked
        # stream thread stalls EVERY symbol, not just this one. Drained in
        # arrival order once the replay finishes.
        self._hold_events: bool = False
        self._held_events: deque = deque(maxlen=_HELD_EVENTS_MAX)
        self._held_lock = threading.Lock()
        self._hold_release_tid: int | None = None

        # Ensure separate setup-pattern channel exists even for tests/legacy fixtures.
        # Cached entries may have an older setup schema; rebuild from config while
        # preserving any still-known values.
        setup_pattern = self._entry.setup_pattern
        if setup_pattern is None or setup_pattern.get("pullback_control") is None:
            migrated_setup = build_groups([SETUP_PATTERN_GROUP])[0]
            old_vars = {v.name: v for v in setup_pattern.variables} if setup_pattern is not None else {}
            for v in migrated_setup.variables:
                old = old_vars.get(v.name)
                if old is not None:
                    v.value = old.value
                    v.quality = old.quality
                    v._n_obs = getattr(old, "_n_obs", 0)
                else:
                    v.reset(0.0)
            self._entry.setup_pattern = migrated_setup

        # --- T&S state ---
        self._tick_prices: deque = deque(maxlen=ROC_TS_PERIOD + 2)
        self._last_price: float  = entry.price
        self._prev_close: float = 0.0
        if entry.price > 0 and entry.change_pct > -99.0:
            self._prev_close = entry.price / (1.0 + entry.change_pct / 100.0)
        # Prefer the eval-time stored prev_close: entry.change_pct is mutated
        # live (entry.price is not), so re-deriving from a re-tracked cached
        # entry would give a corrupt base.
        stored_prev_close = float(getattr(entry, "prev_close", 0.0) or 0.0)
        if stored_prev_close > 0:
            self._prev_close = stored_prev_close
        elif self._prev_close > 0:
            entry.prev_close = self._prev_close
        # Seed the session peak from what's already known (HOD at eval).
        if self._prev_close > 0 and entry.daily_high > 0:
            hod_change = (entry.daily_high - self._prev_close) / self._prev_close * 100.0
            if hod_change > float(getattr(entry, "peak_change_pct", 0.0) or 0.0):
                entry.peak_change_pct = hod_change
        self._last_roc:   float  = 0.0
        self._prints_seen: int   = 0

        # --- 1-min candle buffer (REST-polled, used for chop/vol metrics) ---
        self._candles: list[OHLCVCandle] = []
        self._candles_session: list[OHLCVCandle] = []  # full session (not move-clipped) for ATR/MACD
        self._candles_2min: list[OHLCVCandle] = []
        self._candles_5min: list[OHLCVCandle] = []
        self._candles_daily: list[OHLCVCandle] = daily_candles or []
        self._move_start_ts: datetime | None = self._coerce_utc_dt(move_start_ts)
        self._df_1min_session: pd.DataFrame = self._normalize_candle_df(seed_1min_session)
        seed_1min_move = self._clip_df_since(
            self._df_1min_session,
            self._active_price_action_start(self._latest_df_ts(self._df_1min_session)),
        )
        self._df_1min_cache: pd.DataFrame = seed_1min_move.reset_index(drop=True)
        self._df_10sec_seed: pd.DataFrame = self._normalize_candle_df(seed_10sec)
        self._df_timesales_seed: pd.DataFrame = self._normalize_timesales_df(seed_timesales)
        self._hist = get_historical_client()

        # --- Multi-timeframe stream candle buffer (pattern detection) ---
        seed_capacity = max(
            SCORER_CANDLE_MAX_BARS,
            len(self._df_1min_cache),
            len(self._df_10sec_seed),
        )
        self._tf_buf = _MultiTFCandleBuffer(self._sym, max_bars=seed_capacity)
        self._tf_seeded: bool = False
        self._first_stream_ts: Optional[datetime] = None
        # Live caches hand their disk writes to the process-wide writer
        # thread so a bar close never costs the stream thread a file rewrite.
        self._chart_cache = (
            LiveChartCache(self._sym, writer=get_chart_cache_writer())
            if cache_live_data else _NullChartCache()
        )

        # --- L2 state ---
        self._bid: float = 0.0
        self._ask: float = 0.0
        self._thin_count:       int   = 0
        # L1 quote-stream last trade — display/HOD fallback when the tape is
        # absent or stale (see _effective_price_locked). Never read by
        # entry/tape gates.
        self._last_quote_price: float = 0.0

        # --- Order flow tracker ---
        self._ofi_tracker = OrderFlowTracker()

        # --- Rolling volume for RVol ---
        self._vol_prints:  deque = deque()  # (datetime, int size)
        self._vol_prints_sum: int = 0       # running total of sizes in deque



        # --- Trim throttle: only trim deques once per second ---
        self._last_trim_epoch: float = 0.0

        # --- Rmap dynamic level update throttle (separate from trim) ---
        self._rmap_last_epoch: float = 0.0

        # --- Per-TF price action scores for combine_tf_scores ---
        self._tf_pa_scores: dict[str, float] = {}
        # Per-TF evidence mass (0-1) for setup_outcome: how much of the score
        # is backed by legs whose peak candle volume clears the current
        # relative-volume band (median -> top-percentile, like key_candles).
        # Mass scales the setup_outcome weight in the TA blend; ~0 mass means
        # the legs were volume-insignificant and setup_outcome drops out.
        self._tf_pa_masses: dict[str, float] = {}
        # Per-TF per-variable qualities (for TA table in weights display).
        # Seed all TF keys so the table renders, but leave the per-var cells
        # empty (—) until real data exists — setup_outcome does NOT show a
        # neutral 50% when there are no setups.
        self._tf_pa_var_qualities: dict[str, dict[str, float]] = {
            tf: {} for tf in ("10s", "1m", "2m", "5m")
        }

        # --- Setup outcome tracking (replaces candle quality scoring) ---
        self._active_legs: dict[str, _ActiveLeg] = {}      # tf -> active leg
        self._completed_outcomes: list[_CompletedOutcome] = []

        # --- Last-known VWAP + ATR for tick-level criteria checks ---
        # _last_vwap: authoritative session VWAP from REST poll (full session).
        # _stream_vwap: incrementally computed from stream bars only (partial).
        self._last_vwap: float = 0.0
        self._stream_vwap: float = 0.0
        self._last_atr: float = 0.0

        # --- Stream-derived live indicators (resource-conservative) ---
        self._vwap_session_key: str = ""
        self._vwap_cum_pv_closed: float = 0.0
        self._vwap_cum_vol_closed: int = 0
        self._vwap_closed_history: deque = deque(maxlen=32)
        # Session VWAP sampled at each closed 10s bar; feeds the slope gate at
        # 10s cadence (falls back to _vwap_closed_history until warmed up).
        self._vwap_10s_history: deque = deque(maxlen=32)
        self._ema9_base: Optional[float] = None
        self._ema20_base: Optional[float] = None
        self._last_ema9: float = 0.0
        self._last_ema20: float = 0.0

        # --- Pattern detection state ---
        self._patterns: PatternSnapshot = PatternSnapshot()
        self._last_event_ts: datetime | None = None

        # --- Setup signal logging (separate from simulated fills) ---
        self._setup_logger: SetupSignalLogger = get_setup_signal_logger()
        self._active_setup_signal: SetupSignal | None = None
        self._active_setup_signature: tuple[str, str, str, float] | None = None
        self._active_setup_dirty: bool = False
        self._last_setup_log_write_epoch: float = 0.0

        # --- Breakout-candle rejection tracking ---
        # When a pullback-high breakout fires intracandle we remember the
        # breaking candle so we can invalidate the setup if it closes red.
        self._breakout_candle_tf: str | None = None
        self._breakout_candle_ts: datetime | None = None
        self._breakout_trigger_level: float | None = None
        # Levels whose breakout candle closed red — re-breaks here are blocked.
        self._rejected_breakout_levels: deque[float] = deque(maxlen=32)

        # Last slot pin pushed to the shared stream (see _push_slot_priority).
        self._last_pushed_pin: int = SLOT_PIN_NONE

        self._on_update = on_update
        self._poll_thread: Optional[threading.Thread] = None
        self._seed_thread: Optional[threading.Thread] = None

        # --- Simulation engine (evaluates entry/exit on each tick) ---
        self._sim_engine: Optional[SimulationEngine] = None
        if SIM_ENABLED:
            self._sim_engine = SimulationEngine(
                symbol=self._sym,
                entry=self._entry,
                ofi_tracker=self._ofi_tracker,
                rmap=self._rmap,
                setup_logger=self._setup_logger,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Subscribe to the shared stream + start REST candle poller. Returns immediately."""
        # Fail-open init for the L2 density gate: depth lines are limited
        # (IBKR_L2_MAX_DEPTH_SUBS) so a symbol without a slot must not be
        # blocked by a never-updated required criterion. Real depth
        # measurements from _on_book override this immediately.
        attn = self._entry.get_criteria_group("ATTENTION")
        if attn is not None:
            attn.update("order_book_density_ok", True, None)
        get_shared_stream().subscribe(
            self._sym,
            on_timesale=self._on_timesale,
            on_quote=self._on_quote,
            sub_id=f"scorer:{self._sym}",
            on_book=self._on_book if IBKR_ENABLED else None,
        )
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name=f"poll-{self._sym}",
        )
        self._poll_thread.start()
        logger.info(f"[{self._sym}] DynamicScorer started")

    def prime_from_seed_candles(self) -> None:
        """Populate live-confidence inputs from the startup 1-min seed without REST polling."""
        if self._df_1min_cache.empty:
            return
        self._refresh_candles(use_seed_only=True, record_cache=False, notify=False)

    def stop(self, *, wait: bool = True) -> None:
        """Unsubscribe from shared stream + stop poller. Waits for clean shutdown."""
        get_shared_stream().unsubscribe(self._sym, sub_id=f"scorer:{self._sym}")
        self._stop.set()
        if self._sim_engine is not None:
            self._sim_engine.stop()
        if not wait:
            self._flush_chart_cache()
            logger.debug(f"[{self._sym}] DynamicScorer stop requested")
            return
        if self._poll_thread:
            self._poll_thread.join(timeout=SCORER_STOP_TIMEOUT_SECS)
            if self._poll_thread.is_alive():
                logger.warning(f"[{self._sym}] DynamicScorer poll thread did not exit within {SCORER_STOP_TIMEOUT_SECS}s")
        if self._seed_thread:
            self._seed_thread.join(timeout=SCORER_STOP_TIMEOUT_SECS)
            if self._seed_thread.is_alive():
                logger.warning(f"[{self._sym}] 10s seed worker did not exit within {SCORER_STOP_TIMEOUT_SECS}s")
        self._flush_chart_cache()
        logger.debug(f"[{self._sym}] DynamicScorer stopped")

    def _raise_hod(self, price: float) -> None:
        """Raise daily_high and the persisted session peak on a new high.

        peak_change_pct rides along in the TickerCache day snapshot so
        backtest ordering can use the true observed peak without candles.
        """
        if price <= self._entry.daily_high:
            return
        self._entry.daily_high = price
        if self._prev_close > 0:
            change = (price - self._prev_close) / self._prev_close * 100.0
            if change > float(getattr(self._entry, "peak_change_pct", 0.0) or 0.0):
                self._entry.peak_change_pct = change

    def _effective_price_locked(self) -> float:
        """Print price while the tape is fresh; L1 quote last otherwise.

        A symbol without a tape line stops printing entirely — without this
        fallback its snapshot price (display %-change, HOD distance) freezes
        at the last print, or at evaluation time if it never held a line.
        Execution paths read ``_last_price`` directly and stay print-gated.
        """
        if self._last_quote_price <= 0:
            return self._last_price
        if self._last_price <= 0 or self._last_event_ts is None:
            return self._last_quote_price
        ts = self._last_event_ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(tz=timezone.utc) - ts).total_seconds()
        if age > SCORER_QUOTE_PRICE_FALLBACK_SECS:
            return self._last_quote_price
        return self._last_price

    def _snapshot_locked(self) -> ScorerSnapshot:
        spread = round(self._ask - self._bid, 4) if self._ask > 0 else 0.0
        ev = self._entry.evaluation
        group_scores = {g.name: round(g.score, 3) for g in ev.groups} if ev else {}
        hod = self._entry.daily_high
        last_price = self._effective_price_locked()
        dist_hod_atrs = 0.0
        if self._last_atr > 0 and hod > 0 and last_price > 0:
            dist_hod_atrs = (hod - last_price) / self._last_atr
        return ScorerSnapshot(
            symbol=self._sym,
            eval_confidence=self._entry.eval_confidence,
            live_confidence=self._entry.live_confidence,
            last_price=last_price,
            last_roc=self._last_roc,
            spread=spread,
            rvol_5min=self._compute_rvol_5min(),
            prints_seen=self._prints_seen,
            candles_built=len(self._candles),
            captured_at=datetime.now(tz=timezone.utc),
            patterns=self._patterns,
            book_lean=self._ofi_tracker.book_lean,
            buying_pressure=self._ofi_tracker.buying_pressure,
            selling_pressure=self._ofi_tracker.selling_pressure,
            ofi_rolling=self._ofi_tracker.ofi_rolling,
            trade_imbalance=self._ofi_tracker.trade_imbalance,
            buy_volume=self._ofi_tracker.buy_volume,
            sell_volume=self._ofi_tracker.sell_volume,
            trade_speed=self._ofi_tracker.trade_speed,
            pressure_acceleration=self._ofi_tracker.pressure_acceleration,
            pressure_acceleration_ready=self._ofi_tracker.pressure_acceleration_ready,
            baseline_speed=self._ofi_tracker.baseline_speed,
            speed_ratio=self._ofi_tracker.speed_ratio,
            group_scores=group_scores,
            setup_pattern_score=self._entry.setup_pattern_score,
            tf_pa_scores=dict(self._tf_pa_scores),
            tf_pa_var_qualities={tf: dict(d) for tf, d in self._tf_pa_var_qualities.items()},
            dist_from_hod_atrs=dist_hod_atrs,
            move_start_ts=self._move_start_ts,
        )

    def snapshot(self) -> ScorerSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def try_snapshot(self) -> ScorerSnapshot | None:
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._snapshot_locked()
        finally:
            self._lock.release()

    # ------------------------------------------------------------------
    # Order-flow views (`tape` / `ofi` displays)
    # ------------------------------------------------------------------
    # Non-blocking by design: these render on the prompt event loop, and the
    # stream thread holds _lock on every print.  Returning None on contention
    # drops one frame instead of stalling the REPL — same contract as
    # try_snapshot().

    def try_tape_view(self, limit: int) -> TapeView | None:
        """Tape + tape vars, or None if the scorer thread holds the lock."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._tape_view_locked(limit)
        finally:
            self._lock.release()

    def try_book_view(self) -> BookView | None:
        """Order book + OFI reads, or None if the scorer thread holds the lock."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            return self._book_view_locked()
        finally:
            self._lock.release()

    def _tape_view_locked(self, limit: int) -> TapeView:
        ofi = self._ofi_tracker
        # limit <= 0 means "everything streamed": the full rolling window,
        # exactly what the tape vars are computed from.
        return TapeView(
            symbol=self._sym,
            prints=tuple(ofi.recent_prints(limit if limit > 0 else ofi.retained_prints)),
            retained=ofi.retained_prints,
            window_secs=ofi.trade_window_secs,
            prints_seen=self._prints_seen,
            last_price=self._last_price,
            bid=self._bid,
            ask=self._ask,
            last_event_ts=ofi.last_event_ts,
            captured_at=datetime.now(tz=timezone.utc),
            gates=tape_entry_gates(ofi),
            trade_speed=ofi.trade_speed,
            tape_speed_ema=ofi.tape_speed_ema,
            baseline_speed=ofi.baseline_speed,
            speed_ratio=ofi.speed_ratio,
            buy_ratio=ofi.buy_ratio,
            buy_ratio_ema=ofi.buy_ratio_ema,
            buy_volume=ofi.buy_volume,
            sell_volume=ofi.sell_volume,
            trade_imbalance=ofi.trade_imbalance,
            buying_pressure=ofi.buying_pressure,
            selling_pressure=ofi.selling_pressure,
            avg_print_size=ofi.avg_print_size,
            tape_prints_per_sec=ofi.tape_prints_per_sec,
            density_imbalance=ofi.density_imbalance,
            accel_raw=ofi.accel_gate_raw,
            accel_z=ofi.accel_gate_z,
            accel_ready=ofi.accel_gate_ready,
            accel_n_eff=ofi.pressure_acceleration_ema_n_eff,
            dispersion_phi=ofi.tape_dispersion_phi,
            hidden_seller=ofi.hidden_seller,
        )

    def _book_view_locked(self) -> BookView:
        ofi = self._ofi_tracker
        bids, asks = ofi.book_ladder
        cur_bid, cur_ask = ofi.l1_quote
        return BookView(
            symbol=self._sym,
            bids=tuple(bids),
            asks=tuple(asks),
            depth_active=ofi.depth_active,
            l1_bid=cur_bid.price if cur_bid is not None else 0.0,
            l1_bid_size=cur_bid.size if cur_bid is not None else 0,
            l1_ask=cur_ask.price if cur_ask is not None else 0.0,
            l1_ask_size=cur_ask.size if cur_ask is not None else 0,
            last_price=self._last_price,
            last_book_ts=ofi.last_book_ts,
            last_event_ts=ofi.last_event_ts,
            captured_at=datetime.now(tz=timezone.utc),
            ofi_rolling=ofi.ofi_rolling,
            ofi_rate=ofi.ofi_rate,
            ofi_normalized=ofi.ofi_normalized,
            ofi_norm_gate=ofi.ofi_norm_gate,
            ofi_samples=ofi.ofi_sample_count,
            ofi_window_secs=ofi.ofi_window_secs,
            ofi_chop=ofi.ofi_chop_ratio,
            ofi_intensity=ofi.ofi_intensity,
            book_lean=ofi.book_lean,
            book_lean_raw=ofi.book_lean_raw,
            book_depth_avg=ofi.book_depth_avg,
            quote_count=ofi.quote_count,
        )

    def format_summary(self) -> str:
        """Compact one-line summary for display."""
        s = self.snapshot()
        gs = "  ".join(f"{k}={v:.0%}" for k, v in s.group_scores.items())
        return (
            f"  [{s.symbol}] Eval confidence: {s.eval_confidence:.1f}  "
            f"Live confidence: {s.live_confidence:.1f}\n"
            f"  setup confidence: {s.setup_pattern_score:.1f}\n"
            f"  price=${s.last_price:.2f}  roc={s.last_roc:+.2f}  "
            f"spread=${s.spread:.3f}  rvol5m={s.rvol_5min:.1f}x\n"
            f"  groups: {gs}"
        )

    def get_candle_snapshot(self) -> dict[str, list[OHLCVCandle]]:
        """Return a copy of current candle buffers per timeframe label.

        Used by the ``live`` command to inspect per-candle quality details.
        Keys match the TF labels used in _tf_pa_scores: "10s", "1m", "2m", "5m".
        Daily ("1d") is included for reference but not scored.
        """
        with self._lock:
            result: dict[str, list[OHLCVCandle]] = {}
            # Stream-built 10s from the multi-TF buffer
            bars_10s = list(self._tf_buf._bars.get("10sec", []))
            cur_10s = self._tf_buf._current.get("10sec")
            if cur_10s is not None:
                bars_10s.append(cur_10s)
            if bars_10s:
                result["10s"] = bars_10s
            # REST-polled timeframes
            if self._candles:
                result["1m"] = list(self._candles)
            if self._candles_2min:
                result["2m"] = list(self._candles_2min)
            if self._candles_5min:
                result["5m"] = list(self._candles_5min)
            if self._candles_daily:
                result["1d"] = list(self._candles_daily)
            return result

    # ------------------------------------------------------------------
    # HOD proximity gate — tech analysis only meaningful near highs
    # ------------------------------------------------------------------

    def _is_near_hod(self, price: float | None = None) -> bool:
        """True if price is within N ATRs OR 10% of HOD (whichever is bigger)."""
        from config import hod_proximity_max_atrs
        p = price if price is not None else self._last_price
        hod = self._entry.daily_high
        atr = self._last_atr
        if atr <= 0 or hod <= 0 or p <= 0:
            return True  # no data yet → allow updates (seed phase)
        max_atrs = hod_proximity_max_atrs(hod, atr, ATTENTION_DIST_FROM_HOD_ATR_MAX)
        return (hod - p) / atr <= max_atrs

    # ------------------------------------------------------------------
    # Stream slot priority
    # ------------------------------------------------------------------

    def _push_slot_priority(self) -> None:
        """Recompute this symbol's slot pin and push it when it changes.

        The shared stream allocates its capped tape/depth lines by pin
        first, then by live activity. Activity alone would evict exactly
        the symbols that matter most — this strategy buys pullbacks, and a
        pullback is a quiet tape — so the funnel states that must keep
        their feeds are pinned: armed pullback, breakout awaiting entry
        confirmation, open position.
        """
        if not IBKR_ENABLED:
            return
        name = self._entry.setup_pattern_name
        if SIM_ENABLED and self._sym in SimulationEngine.active_positions():
            pin = SLOT_PIN_POSITION
        elif name == "pullback_new_high":
            pin = SLOT_PIN_CONFIRMING
        elif (
            name in ("pullback_pending", "pullback_detected")
            and (self._entry.setup_price_level or 0) > 0
            and self._is_near_hod()
        ):
            pin = SLOT_PIN_ARMED
        else:
            pin = SLOT_PIN_NONE
        if pin == self._last_pushed_pin:
            return
        self._last_pushed_pin = pin
        try:
            set_prio = getattr(get_shared_stream(), "set_priority", None)
            if set_prio is not None:
                set_prio(self._sym, pin)
        except Exception:
            logger.debug(f"[{self._sym}] slot priority push failed", exc_info=True)

    # ------------------------------------------------------------------
    # T&S handler
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Event hold — keeps the stream thread off _lock during tick replay
    # ------------------------------------------------------------------

    def _hold_event(self, kind: str, event) -> bool:
        """Park *event* if a hold is active. Returns True when parked.

        The releasing thread bypasses the hold so the drain can dispatch
        through the normal handlers without re-parking its own events.
        """
        with self._held_lock:
            if not self._hold_events or threading.get_ident() == self._hold_release_tid:
                return False
            self._held_events.append((kind, event))
            return True

    def _begin_event_hold(self) -> None:
        with self._held_lock:
            self._hold_events = True

    def _release_held_events(self) -> None:
        """Dispatch parked events in arrival order, then lift the hold.

        The hold is lifted under _held_lock only once the deque is empty,
        so an event the stream thread appends during the drain is either
        picked up by the next pass or, after the lift, handled directly —
        never lost and never reordered (the stream thread is the sole
        producer and dispatches sequentially).
        """
        dispatch = {"t": self._on_timesale, "q": self._on_quote, "b": self._on_book}
        with self._held_lock:
            self._hold_release_tid = threading.get_ident()
        try:
            while True:
                with self._held_lock:
                    batch = list(self._held_events)
                    self._held_events.clear()
                    if not batch:
                        self._hold_events = False
                        return
                for kind, ev in batch:
                    try:
                        dispatch[kind](ev)
                    except Exception:
                        logger.debug(f"[{self._sym}] held {kind} event dispatch failed", exc_info=True)
        finally:
            with self._held_lock:
                self._hold_release_tid = None
                self._hold_events = False

    def _event_hold(self, active: bool) -> "_EventHold":
        """Context manager: hold stream events for the duration when *active*."""
        return _EventHold(self, active)

    # ------------------------------------------------------------------
    # T&S handler
    # ------------------------------------------------------------------

    def _on_timesale(self, event: LiveTimesale) -> None:
        if event.cancel or event.correction:
            return
        # getattr: tests drive these handlers on bare stubs without __init__.
        if getattr(self, "_hold_events", False) and self._hold_event("t", event):
            return

        self._chart_cache.record_timesale(event)
        closed_cache_bars: list[tuple[str, OHLCVCandle]] = []

        with self._lock:
            price = event.last
            size  = event.size
            now   = event.ts
            self._last_event_ts = now

            if size > 0:
                self._entry.volume += size
                if self._avg_vol > 0:
                    simple_rvol = self._entry.volume / self._avg_vol
                    if simple_rvol > self._entry.rvol_1day:
                        self._entry.rvol_1day = simple_rvol
                attn = self._entry.get_criteria_group("ATTENTION")
                if attn is not None:
                    if self._prev_close > 0 and price > 0:
                        self._entry.change_pct = (price - self._prev_close) / self._prev_close * 100.0
                        attn.update(
                            "change_pct_min",
                            self._entry.change_pct >= CHANGE_PCT_MIN,
                            self._entry.change_pct,
                        )
                    attn.update(
                        "volume_min",
                        self._entry.volume >= ATTENTION_VOLUME_MIN,
                        float(self._entry.volume),
                    )
                    if self._avg_vol > 0:
                        attn.update(
                            "rvol_1day_min",
                            self._entry.rvol_1day >= ATTENTION_RVOL_1DAY_MIN,
                            self._entry.rvol_1day,
                        )

            if price * size < TS_MIN_NOTIONAL:
                if self._on_update is not None:
                    self._on_update()
                return

            self._prints_seen += 1

            # Record first accepted stream print so REST seed bars can be
            # clipped to pre-stream history only (avoid overlap skew).
            if self._first_stream_ts is None:
                self._first_stream_ts = now

            self._last_price = price
            self._tick_prices.append(price)

            # Keep daily_high current from live prints
            self._raise_hod(price)

            # Update bid/ask from print (backup if L2 lags)
            if event.bid > 0:
                self._bid = event.bid
            if event.ask > 0:
                self._ask = event.ask

            # Track volume for RVol
            self._vol_prints.append((now, size))
            self._vol_prints_sum += size

            # Trim deques every 250ms (keep memory tight during volume spikes)
            now_epoch = now.timestamp()
            if now_epoch - self._last_trim_epoch >= 0.25:
                self._last_trim_epoch = now_epoch
                cutoff = now - timedelta(seconds=RVOL_WINDOW_SECONDS)
                while self._vol_prints and self._vol_prints[0][0] < cutoff:
                    self._vol_prints_sum -= self._vol_prints[0][1]
                    self._vol_prints.popleft()


            # ROC from T&S ticks (always per-tick — lightweight)
            self._eval_roc()

            # Resistance breaks
            self._eval_resistance(price)

            # Feed multi-timeframe candle buffer
            closed_tfs = self._tf_buf.on_tick(price, size, now)
            if closed_tfs:
                self._tf_buf.set_retention_start(self._active_price_action_start(now))
            self._update_stream_indicators(now, closed_tfs)
            closed_cache_bars = self._closed_stream_cache_bars(closed_tfs)
            if closed_tfs:
                self._scan_patterns_on_close(closed_tfs)
                self._on_setup_leg_candle_close(closed_tfs)
                self._update_no_news_penalty(now)

            # --- Breakout-candle rejection: invalidate if the breaking
            #     candle closed red (and remember the level) ---
            if closed_tfs:
                self._check_breakout_candle_rejection(closed_tfs, now)

            # --- Mid-candle pullback detection (early awareness) ---
            self._check_intracandle_pullback_detection(price)

            # --- Intracandle breakout: promote pending/detected→resolved ---
            self._check_intracandle_breakout(price, now)

            # --- Order flow: classify trade + evaluate ---
            self._push_entry_band_to_tracker()
            self._ofi_tracker.on_timesale(event)

            # --- Live scoring update (every tick) ---
            self._update_tick_criteria(price)

            if self._on_update is not None:
                self._on_update()

            # --- Setup signal log: approval + peak T&S snapshot ---
            self._update_active_setup_signal(price, now)

            # --- Simulation engine: evaluate entry/exit on each tick ---
            if self._sim_engine is not None:
                self._sim_engine.on_tick(
                    price=price,
                    bid=self._bid,
                    ask=self._ask,
                    last_atr=self._last_atr,
                    event_ts=now,
                )

            # --- Throttled scoring telemetry ---
            if SCORING_LOG_ENABLED and self._prints_seen % _LOG_EVERY_N_PRINTS == 0:
                sl = _get_scoring_logger()
                if sl:
                    sl.snapshot(self._entry)
                    sl.tape_snapshot(self._entry, self._ofi_tracker, price, self._tf_pa_scores)

        for chart_tf, bar in closed_cache_bars:
            self._chart_cache.record_candles(chart_tf, [bar], SOURCE_STREAM)

    # ------------------------------------------------------------------
    # L2 handler
    # ------------------------------------------------------------------

    def _on_quote(self, event: LiveQuote) -> None:
        if getattr(self, "_hold_events", False) and self._hold_event("q", event):
            return
        with self._lock:
            self._bid = event.bid
            self._ask = event.ask

            # L1 last trade (IBKR feed only; 0.0 from Tradier). This is a
            # real print, just sampled ~4x/sec — safe for HOD and for the
            # snapshot's price fallback, and the only moving price a symbol
            # without a tape line has.
            if getattr(event, "last", 0.0) > 0:
                self._last_quote_price = event.last
                self._raise_hod(event.last)

            if event.bid > 0 and event.ask > 0:
                spread = event.ask - event.bid
                mid = (event.ask + event.bid) / 2.0
                spread_pct = spread / mid if mid > 0 else float("inf")
                execg = self._entry.get_criteria_group("EXECUTION")
                if execg is not None:
                    execg.update(
                        "spread_ok",
                        spread >= 0.0 and spread_pct <= EXEC_SPREAD_PCT_MAX,
                        spread_pct,
                    )

            # Tradier provides L1 NBBO only; true depth (L2) comes from IBKR.
            # All order-book / OFI scoring vars stay unset (None) when IBKR is
            # disabled so they're excluded from the ORDER_FLOW group score
            # rather than silently maxing out.
            if not IBKR_ENABLED:
                return

            inputs = LiveInputs()

            # --- Order flow: L1 OFI + book lean ---
            self._ofi_tracker.on_quote(event)
            inputs.ofi_chop = self._ofi_tracker.ofi_chop_ratio

            self._entry.evaluation.apply_live(inputs)

            # --- Simulation engine: update quotes ---
            if self._sim_engine is not None:
                self._sim_engine.on_quote(event.bid, event.ask)

    def _push_entry_band_to_tracker(self) -> None:
        """Anchor the OFI tracker to the active setup's entry band.

        Armed-band latches stamp only while price sits inside the band, and
        a setup-identity change resets them — so this is pushed immediately
        before EVERY tracker ingestion (live tick, book, and both replay
        paths) to keep the tracker at most one event stale. Cheap no-op when
        nothing changed.
        """
        level = self._entry.setup_price_level
        if level is not None and level > 0:
            self._ofi_tracker.set_entry_band(
                level,
                entry_band_floor(level, self._entry.setup_pullback_low),
                self._entry.setup_pullback_ts,
            )
        else:
            self._ofi_tracker.set_entry_band(None, None, None)

    def _on_book(self, snapshot) -> None:
        """L2 depth callback (IBKR only): multi-level OFI + density criterion."""
        if getattr(self, "_hold_events", False) and self._hold_event("b", snapshot):
            return
        with self._lock:
            self._push_entry_band_to_tracker()
            self._ofi_tracker.on_book(snapshot)
            # Persist the raw ladder — IBKR has no historical depth, so live
            # capture is the only source for backtest replay + calibration.
            self._chart_cache.record_book(snapshot)

            # ATTENTION: order_book_density_ok — both sides populated with
            # enough displayed shares for OFI/hidden-liquidity reads to mean
            # anything. Uses the rolling average so one flickering snapshot
            # doesn't strobe the gate.
            attn = self._entry.get_criteria_group("ATTENTION")
            if attn is not None:
                has_both_sides = bool(snapshot.bids) and bool(snapshot.asks)
                avg_depth = self._ofi_tracker.book_depth_avg
                attn.update(
                    "order_book_density_ok",
                    has_both_sides and avg_depth >= ORDER_BOOK_MIN_DEPTH_SHARES,
                    avg_depth,
                )

    # ------------------------------------------------------------------
    # Condition evaluators
    # ------------------------------------------------------------------

    def _eval_roc(self) -> None:
        """Compute ROC(period=ROC_TS_PERIOD) from raw T&S ticks.

        Stores the result in ``_last_roc`` for the snapshot only. ROC does
        not update attention gates or scored live variables.
        """
        if len(self._tick_prices) < ROC_TS_PERIOD + 1:
            return

        p_now  = self._tick_prices[-1]
        p_prev = self._tick_prices[-(ROC_TS_PERIOD + 1)]
        if p_prev <= 0:
            return

        self._last_roc = (p_now - p_prev) / p_prev * 100

    def _eval_spread(self, bid: float, ask: float) -> None:
        """Legacy no-op; spread is enforced as an execution criterion."""
        return

    @staticmethod
    def _compute_vwap_slope(df: pd.DataFrame) -> Optional[float]:
        """Compute VWAP slope ($/candle) from 1-min candle DataFrame.

        Returns None if insufficient data.
        """
        if "vwap" not in df.columns or len(df) < 5:
            return None
        vwaps = df["vwap"].dropna()
        if len(vwaps) < 5:
            return None
        recent = vwaps.iloc[-5:]
        return (float(recent.iloc[-1]) - float(recent.iloc[0])) / (len(recent) - 1)

    def _compute_tape_density(self) -> Optional[float]:
        """Compute prints/sec from rolling volume deque. None if insufficient data."""
        if not self._vol_prints or len(self._vol_prints) < 2:
            return None
        first_ts = self._vol_prints[0][0]
        last_ts = self._vol_prints[-1][0]
        elapsed = (last_ts - first_ts).total_seconds()
        if elapsed <= 0:
            return None
        return len(self._vol_prints) / elapsed

    @staticmethod
    def _session_key_from_utc(ts: datetime) -> str:
        """ET session key using reset windows 04:00, 09:30, 16:00, 20:00."""
        # Use configured ET zone from MarketClock to respect DST.
        from config import MarketClock
        et = ts.astimezone(MarketClock.ET)
        minutes = et.hour * 60 + et.minute
        resets = [240, 570, 960, 1200]
        if minutes < resets[0]:
            idx = len(resets) - 1
            et_day = (et - timedelta(days=1)).date()
        else:
            idx = 0
            for i, val in enumerate(resets):
                if minutes >= val:
                    idx = i
            et_day = et.date()
        return f"{et_day.isoformat()}-{idx}"

    @staticmethod
    def _bar_typical_price(bar: OHLCVCandle) -> float:
        return (float(bar.high) + float(bar.low) + float(bar.close)) / 3.0

    @staticmethod
    def _bar_vwap_price(bar: OHLCVCandle) -> float:
        if bar.vwap is not None and not pd.isna(bar.vwap) and float(bar.vwap) > 0:
            return float(bar.vwap)
        return DynamicScorer._bar_typical_price(bar)

    def _compute_live_vwap_slope(self) -> Optional[float]:
        """Slope over recent session VWAP points including current provisional value."""
        if self._stream_vwap <= 0:
            return None
        vals = list(self._vwap_closed_history)
        vals.append(self._stream_vwap)
        vals = vals[-5:]
        if len(vals) < 5:
            return None
        return (vals[-1] - vals[0]) / (len(vals) - 1)

    def _update_stream_indicators(self, now: datetime, closed_tfs: list[str]) -> None:
        """Incrementally update stream VWAP/EMA and sync dynamic reference levels."""
        session_key = self._session_key_from_utc(now)
        if session_key != self._vwap_session_key:
            self._vwap_session_key = session_key
            self._vwap_cum_pv_closed = 0.0
            self._vwap_cum_vol_closed = 0
            self._vwap_closed_history.clear()
            self._vwap_10s_history.clear()

        if "1min" in closed_tfs:
            closed_1m = self._tf_buf.last_closed_bar("1min")
            if closed_1m is not None:
                c = float(closed_1m.close)
                alpha9 = 2.0 / (9 + 1)
                alpha20 = 2.0 / (20 + 1)
                self._ema9_base = c if self._ema9_base is None else self._ema9_base + alpha9 * (c - self._ema9_base)
                self._ema20_base = c if self._ema20_base is None else self._ema20_base + alpha20 * (c - self._ema20_base)

                tp = self._bar_vwap_price(closed_1m)
                vol = int(closed_1m.volume)
                self._vwap_cum_pv_closed += tp * vol
                self._vwap_cum_vol_closed += vol
                if self._vwap_cum_vol_closed > 0:
                    self._vwap_closed_history.append(self._vwap_cum_pv_closed / self._vwap_cum_vol_closed)

        cur_1m = self._tf_buf.current_bar("1min")
        if cur_1m is None:
            return

        close_now = float(cur_1m.close)
        alpha9 = 2.0 / (9 + 1)
        alpha20 = 2.0 / (20 + 1)
        self._last_ema9 = close_now if self._ema9_base is None else self._ema9_base + alpha9 * (close_now - self._ema9_base)
        self._last_ema20 = close_now if self._ema20_base is None else self._ema20_base + alpha20 * (close_now - self._ema20_base)

        cur_tp = self._bar_vwap_price(cur_1m)
        cur_vol = int(cur_1m.volume)
        total_pv = self._vwap_cum_pv_closed + cur_tp * cur_vol
        total_vol = self._vwap_cum_vol_closed + cur_vol
        if total_vol > 0:
            self._stream_vwap = total_pv / total_vol
        if "10sec" in closed_tfs and self._stream_vwap > 0:
            self._vwap_10s_history.append(self._stream_vwap)

        # Refresh moving reference levels no more than ~1Hz to keep overhead low.
        now_epoch = now.timestamp()
        if now_epoch - self._rmap_last_epoch >= 1.0:
            self._rmap_last_epoch = now_epoch
            if self._stream_vwap > 0:
                self._rmap.set_dynamic(self._stream_vwap, "VWAP (session live)")
            if self._last_ema9 > 0:
                self._rmap.set_dynamic(self._last_ema9, "EMA 9 (1m live)")
            if self._last_ema20 > 0:
                self._rmap.set_dynamic(self._last_ema20, "EMA 20 (1m live)")

    def _closed_stream_cache_bars(self, closed_tfs: list[str]) -> list[tuple[str, OHLCVCandle]]:
        cache_bars: list[tuple[str, OHLCVCandle]] = []
        for buf_tf in closed_tfs:
            chart_tf = _CHART_CACHE_TF.get(buf_tf)
            if chart_tf is None:
                continue
            bar = self._tf_buf.last_closed_bar(buf_tf)
            if bar is not None:
                cache_bars.append((chart_tf, bar))
        return cache_bars

    def _flush_chart_cache(self) -> None:
        self._chart_cache.flush_timesales()
        self._chart_cache.flush_books()
        with self._lock:
            for buf_tf, chart_tf in _CHART_CACHE_TF.items():
                df = self._tf_buf.to_df(buf_tf)
                if not df.empty:
                    self._chart_cache.record_candles(chart_tf, df, SOURCE_STREAM)
        # Writes go through the background writer; wait for them so a
        # re-track that reads the cache right after stop() sees everything.
        if not self._chart_cache.drain(timeout=SCORER_STOP_TIMEOUT_SECS):
            logger.debug(f"[{self._sym}] chart cache still flushing after {SCORER_STOP_TIMEOUT_SECS}s")

    def _eval_resistance(self, price: float) -> None:
        """Check resistance breaks (informational — no quality variable)."""
        broken = self._rmap.mark_broken_levels(price, RESISTANCE_BREAK_BUFFER)
        if broken:
            logger.debug(f"[{self._sym}] resistance level broken at ${price:.2f}")

    def _eval_patterns(self) -> None:
        """Log pattern detection results (informational — patterns feed into
        price action quality via _scan_patterns_on_close)."""
        snap = self._patterns
        if snap.strongest_bearish:
            b = snap.strongest_bearish
            logger.debug(
                f"[{self._sym}] pattern: {b.name} @ ${b.price_level:.2f} "
                f"(str={b.strength:.0%})"
            )
        if snap.strongest_bullish:
            b = snap.strongest_bullish
            logger.debug(
                f"[{self._sym}] pattern: {b.name} @ ${b.price_level:.2f} "
                f"(str={b.strength:.0%})"
            )

    def _check_intracandle_breakout(self, price: float, ts: datetime) -> None:
        """Promote pullback_pending/pullback_detected → pullback_new_high on trigger break.

        Called on every T&S tick.  When the entry is in ``pullback_pending``
        or ``pullback_detected`` state with a known trigger level
        (``setup_price_level``), we check if the current trade price exceeds
        that level.  If so, we immediately transition to ``pullback_new_high``
        without waiting for the candle to close.

        Gates:
        - Lower-TF setups (10sec, 2min) are blocked from entry while the
          parent 1-min candle is currently forming red. This prevents a
          fleeting 10s pullback break from triggering while the 1m is mid-
          pullback.
        - ``pullback_detected`` (V-recovery path): a candle that dipped red
          but recovered green and breaks the trigger is valid. The 1m-red
          gate does NOT apply to 1min-timeframe setups themselves.
        """
        name = self._entry.setup_pattern_name
        if name not in ("pullback_pending", "pullback_detected"):
            return
        trigger = self._entry.setup_price_level
        if trigger is None or trigger <= 0:
            return
        if price <= trigger:
            return
        if not self._is_near_hod(price):
            return

        # --- Rejected-level gate ---
        # If this trigger was already broken by a candle that closed red
        # (rejection), block the re-break so the setup can't re-fire here.
        if self._is_rejected_level(trigger):
            return

        # --- 1m-red gate for lower timeframes ---
        setup_tf = self._entry.setup_pattern_timeframe or ""
        if setup_tf != "1min":
            cur_1m = self._tf_buf.current_bar("1min")
            if cur_1m is not None and cur_1m.close < cur_1m.open:
                # Parent 1m candle is forming red — block lower-TF entry
                return

        # --- Intracandle breakout confirmed ---
        now_mono = time.monotonic()
        self._entry.setup_pattern_name = "pullback_new_high"
        self._entry.setup_resolved_at = now_mono
        self._entry.setup_breakout_ts = ts
        self._entry.setup_breakout_price = price
        self._entry.setup_failed_at = None

        # Start tracking setup outcome leg
        setup_quality = self._entry.setup_pattern.score if self._entry.setup_pattern else 0.5
        self._on_setup_resolved(
            self._entry.setup_pattern_timeframe or "10sec",
            setup_quality,
            price,
            pullback_ts=self._entry.setup_pullback_ts,
        )

        # Remember the breaking candle so we can invalidate the setup if it
        # closes red (rejection). The breakout fired inside the currently
        # forming bar on the setup timeframe.
        bo_tf = self._entry.setup_pattern_timeframe or "10sec"
        cur_bo = self._tf_buf.current_bar(bo_tf)
        self._breakout_candle_tf = bo_tf
        self._breakout_candle_ts = cur_bo.ts if cur_bo is not None else None
        self._breakout_trigger_level = trigger

        detail = f"intracandle break of ${trigger:.2f}"
        if name == "pullback_detected":
            detail = f"V-recovery break of ${trigger:.2f} (no closed red candle)"

        # Construct a synthetic PatternSignal for _begin_setup_signal
        signal = PatternSignal(
            name="pullback_new_high",
            bias="bullish",
            strength=self._entry.setup_pattern.score if self._entry.setup_pattern else 0.5,
            price_level=trigger,
            timeframe=self._entry.setup_pattern_timeframe or "10sec",
            candle_count=0,
            detail=detail,
            pullback_depth=self._entry.setup_pullback_depth,
            pullback_candles=None,
            pullback_control=None,
            leg_candles=None,
            pullback_ts=self._entry.setup_pullback_ts,
            pullback_low=self._entry.setup_pullback_low,
        )
        self._begin_setup_signal(signal, ts)

    # ------------------------------------------------------------------
    # Breakout-candle rejection
    # ------------------------------------------------------------------
    def _is_rejected_level(self, price: float) -> bool:
        """True if *price* sits within tolerance of a previously rejected level."""
        if price <= 0 or not self._rejected_breakout_levels:
            return False
        tol = max(price, 0.01) * SETUP_REJECTED_LEVEL_TOL_PCT
        return any(abs(price - lvl) <= tol for lvl in self._rejected_breakout_levels)

    def _remember_rejected_level(self, level: float) -> None:
        """Record a breakout level whose candle closed red (rejection)."""
        if level <= 0:
            return
        tol = max(level, 0.01) * SETUP_REJECTED_LEVEL_TOL_PCT
        if any(abs(level - lvl) <= tol for lvl in self._rejected_breakout_levels):
            return
        self._rejected_breakout_levels.append(level)

    def _clear_breakout_candle_tracking(self) -> None:
        self._breakout_candle_tf = None
        self._breakout_candle_ts = None
        self._breakout_trigger_level = None

    def _check_breakout_candle_rejection(self, closed_tfs: list[str], ts: datetime) -> None:
        """Invalidate a fired pullback breakout if the breaking candle closed red.

        When an intracandle breakout promotes the setup to
        ``pullback_new_high`` we record the breaking candle.  Once that
        candle closes, a red close (close < open) is treated as a rejection:
        the setup signal is ended and the broken level is remembered so a
        subsequent break at the same price point is blocked.
        """
        bo_tf = self._breakout_candle_tf
        if bo_tf is None or bo_tf not in closed_tfs:
            return

        closed_bar = self._tf_buf.last_closed_bar(bo_tf)
        if closed_bar is None:
            self._clear_breakout_candle_tracking()
            return

        # Only act on the actual breaking candle (matched by its open ts).
        if self._breakout_candle_ts is not None and closed_bar.ts != self._breakout_candle_ts:
            self._clear_breakout_candle_tracking()
            return

        # If the setup already moved on, stop tracking.
        if self._entry.setup_pattern_name != "pullback_new_high":
            self._clear_breakout_candle_tracking()
            return

        if closed_bar.close < closed_bar.open:
            # Rejection — breakout candle closed red. Invalidate + remember.
            if self._breakout_trigger_level:
                self._remember_rejected_level(self._breakout_trigger_level)
            self._invalidate_rejected_setup(ts, closed_bar)

        self._clear_breakout_candle_tracking()

    def _invalidate_rejected_setup(self, ts: datetime, failure_bar: OHLCVCandle) -> None:
        """End the active pullback setup after a red breakout-candle close."""
        # Record setup outcome failure before clearing state
        rejected_tf = self._entry.setup_pattern_timeframe or "10sec"
        self._on_setup_breakout_rejected(rejected_tf, failure_bar)

        self._expire_active_setup_signal(
            "breakout candle closed red — pullback high rejected", ts
        )
        group = self._entry.setup_pattern
        if group is not None:
            for v in group.variables:
                v.value = None
                v.quality = 0.0
        self._entry.setup_pattern_name = ""
        self._entry.setup_pattern_timeframe = ""
        self._entry.setup_pullback_depth = None
        self._entry.setup_price_level = None
        self._entry.setup_pullback_low = None
        self._entry.setup_pullback_ts = None
        self._entry.setup_resolved_at = None
        self._entry.setup_breakout_ts = None
        self._entry.setup_breakout_price = None
        self._entry.setup_failed_at = time.monotonic()

    def _check_intracandle_pullback_detection(self, price: float) -> None:
        """Detect a pullback forming mid-candle (before the red candle closes).

        Called on every T&S tick.  Surfaces ``pullback_detected`` state when
        the current candle is forming red and there's a valid green leg behind
        it.  This provides early awareness without affecting entry logic — entry
        still requires the trigger break via ``_check_intracandle_breakout``.

        Does NOT fire if a pullback_pending or pullback_new_high already exists
        (those are higher-confidence states that should not be overwritten).
        """
        # Don't overwrite higher-confidence states
        current_name = self._entry.setup_pattern_name
        if current_name in ("pullback_pending", "pullback_new_high"):
            return
        # Skip if we just resolved/failed recently (avoid flicker)
        if self._entry.setup_resolved_at is not None:
            return

        if not self._is_near_hod(price):
            # Clear detected state if we drifted from HOD
            if current_name == "pullback_detected":
                self._entry.setup_pattern_name = ""
                self._entry.setup_price_level = None
            return

        # Check each timeframe for a forming pullback (highest rank wins)
        best_signal: PatternSignal | None = None
        best_rank: int = -1

        for tf in ("1min", "5min", "2min", "10sec"):
            if tf in SETUP_TFS_DISABLED:
                continue
            cur = self._tf_buf.current_bar(tf)
            if cur is None:
                continue
            # Current candle must be forming red
            if cur.close >= cur.open:
                continue
            # Get closed bars for this TF
            bars = self._tf_buf._bars.get(tf)
            if not bars or len(bars) < 3:
                continue
            # Closed bars only, straight from the buffer's cache. This runs
            # on every print: to_df's per-call concat of the forming bar
            # (only to strip it again with iloc[:-1]) was measured
            # 2026-09-03 at ~0.8 ms per call — about a quarter of all
            # per-print CPU on the shared stream thread.
            df_closed = self._tf_buf.closed_df(tf)
            if df_closed.empty:
                continue

            signals = detect_pullback_detected(
                df_closed,
                current_price=price,
                current_open=float(cur.open),
                timeframe=tf,
            )
            if not signals:
                continue

            rank = _TIMEFRAME_RANK.get(tf, 0)
            sig = signals[0]
            if rank > best_rank or (rank == best_rank and sig.strength > (best_signal.strength if best_signal else 0)):
                best_signal = sig
                best_rank = rank

        if best_signal is not None:
            # Surface detected state (forming — carries no setup score until it
            # breaks the entry / new high and becomes a recorded setup).
            group = self._entry.setup_pattern
            if group is not None:
                for v in group.variables:
                    v.value = None
                    v.quality = 0.0
            self._entry.setup_pattern_name = "pullback_detected"
            self._entry.setup_pattern_timeframe = best_signal.timeframe
            self._entry.setup_pullback_depth = float(best_signal.pullback_depth) if best_signal.pullback_depth is not None else None
            self._entry.setup_price_level = float(best_signal.price_level) if best_signal.price_level else None
            self._entry.setup_pullback_low = float(best_signal.pullback_low) if best_signal.pullback_low else None
            self._entry.setup_pullback_ts = best_signal.pullback_ts
            self._entry.setup_resolved_at = None
            self._entry.setup_failed_at = None
        elif current_name == "pullback_detected":
            # Forming candle is no longer red on any TF — clear detected state
            self._entry.setup_pattern_name = ""
            self._entry.setup_pattern_timeframe = ""
            self._entry.setup_pullback_depth = None
            self._entry.setup_price_level = None
            self._entry.setup_pullback_low = None
            self._entry.setup_pullback_ts = None


    def _update_setup_pattern_confidence(self, snapshot: PatternSnapshot) -> None:
        """Update separate setup-pattern confidence from pullback setup signals.

        Three real detector states are surfaced onto the entry:

        - **pending**  : a valid leg+pullback exists, no breakout yet
                         (``pullback_pending`` signal present)
        - **resolved** : breakout candle just closed
                         (``pullback_new_high`` signal — one-shot)
        - **failed**   : entry was pending on the prior tick, neither
                         signal is present now (pullback abandoned)

        Resolved/failed states are stamped onto the entry with a
        ``time.monotonic()`` timestamp so the display layer can show a
        short-lived badge before clearing it.  This intentionally does
        not modify the main live confidence score.
        """
        group = self._entry.setup_pattern
        if group is None:
            return
        pullback_var = group.get("pullback_depth")
        if pullback_var is None:
            return

        resolved = [
            s for s in snapshot.signals
            if s.name == "pullback_new_high" and s.pullback_depth is not None
            and self._is_near_hod(self._last_price or s.price_level)
            and not self._is_rejected_level(s.price_level)
        ]
        pending = [
            s for s in snapshot.signals
            if s.name == "pullback_pending" and s.pullback_depth is not None
            and self._is_near_hod(s.price_level)
        ]

        prev_name = self._entry.setup_pattern_name
        now = time.monotonic()

        # Pending setups are gated on the trigger level. Resolved setups are gated
        # on the current breakout price, because an explosive breakout candle can
        # push HOD well above the original trigger before the candle closes.
        if resolved:
            # Breakout fired — surface the strongest as the "resolved"
            # event, then clear active name so it doesn't linger.
            best = max(
                resolved,
                key=lambda s: (
                    s.strength,
                    _TIMEFRAME_RANK.get(s.timeframe, 0),
                    -(float(s.pullback_depth) if s.pullback_depth is not None else 1.0),
                ),
            )
            self._fill_setup_group(group, best)
            self._entry.setup_pattern_name = best.name
            self._entry.setup_pattern_timeframe = best.timeframe
            self._entry.setup_pullback_depth = float(best.pullback_depth if best.pullback_depth is not None else 0.0)
            self._entry.setup_price_level = float(best.price_level) if best.price_level else None
            self._entry.setup_pullback_low = float(best.pullback_low) if best.pullback_low else None
            self._entry.setup_pullback_ts = best.pullback_ts
            self._entry.setup_resolved_at = now
            self._entry.setup_breakout_ts = self._last_event_ts or datetime.now(tz=timezone.utc)
            self._entry.setup_breakout_price = float(best.price_level) if best.price_level else self._last_price
            self._entry.setup_failed_at = None
            self._on_setup_resolved(
                best.timeframe,
                best.strength,
                float(best.price_level) if best.price_level else self._last_price,
                pullback_ts=best.pullback_ts,
            )
            self._begin_setup_signal(best, self._last_event_ts or datetime.now(tz=timezone.utc))
            return

        if pending:
            # Don't regress an intracandle breakout that already resolved
            if prev_name == "pullback_new_high" and self._entry.setup_resolved_at is not None:
                return
            best = max(
                pending,
                key=lambda s: (
                    s.strength,
                    _TIMEFRAME_RANK.get(s.timeframe, 0),
                    -(float(s.pullback_depth) if s.pullback_depth is not None else 1.0),
                ),
            )
            # Forming pullback — surface the preview but carry NO setup score
            # until it breaks the entry / new high (becomes a recorded setup).
            for v in group.variables:
                v.value = None
                v.quality = 0.0
            self._entry.setup_pattern_name = best.name
            self._entry.setup_pattern_timeframe = best.timeframe
            self._entry.setup_pullback_depth = float(best.pullback_depth if best.pullback_depth is not None else 0.0)
            self._entry.setup_price_level = float(best.price_level) if best.price_level else None
            self._entry.setup_pullback_low = float(best.pullback_low) if best.pullback_low else None
            self._entry.setup_pullback_ts = best.pullback_ts
            # Entering a fresh pending state clears stale resolved/failed
            # badges so the display doesn't double-render.
            self._entry.setup_resolved_at = None
            self._entry.setup_failed_at = None
            return

        # No signal of either kind.  If we were pending last tick, the
        # pullback was abandoned WITHOUT ever breaking the entry price / new
        # high.  A pullback that never broke is not a setup, so it is cleared
        # silently — it is NOT recorded or badged as a "failed" setup.  Only
        # pullbacks that actually broke belong in the setups category (they are
        # surfaced via the resolved path, and a broke-then-failed setup sets
        # setup_failed_at through the breakout-candle rejection instead).
        if prev_name == "pullback_pending":
            self._expire_active_setup_signal(
                "pending pullback abandoned before breakout",
                self._last_event_ts or datetime.now(tz=timezone.utc),
            )
            for v in group.variables:
                v.value = None
                v.quality = 0.0
            self._entry.setup_pattern_name = ""
            self._entry.setup_pattern_timeframe = ""
            self._entry.setup_pullback_depth = None
            self._entry.setup_price_level = None
            self._entry.setup_pullback_low = None
            self._entry.setup_pullback_ts = None
            self._entry.setup_resolved_at = None
            self._entry.setup_breakout_ts = None
            self._entry.setup_breakout_price = None
            self._entry.setup_failed_at = None
            return

        # pullback_detected is managed per-tick — don't clear it from
        # the candle-close path (it will be cleared naturally when the
        # forming candle goes green or the tick detector stops finding it).
        if prev_name == "pullback_detected":
            return

        if prev_name == "pullback_new_high" and self._entry.setup_resolved_at is not None:
            if now - self._entry.setup_resolved_at <= _SETUP_RESOLVED_DISPLAY_SECS:
                self._expire_active_setup_signal(
                    "setup cleared before execution entry",
                    self._last_event_ts or datetime.now(tz=timezone.utc),
                )
                return

        self._expire_active_setup_signal(
            "setup cleared before execution entry",
            self._last_event_ts or datetime.now(tz=timezone.utc),
        )
        for v in group.variables:
            v.value = None
            v.quality = 0.0
        self._entry.setup_pattern_name = ""
        self._entry.setup_pattern_timeframe = ""
        self._entry.setup_pullback_depth = None
        self._entry.setup_price_level = None
        self._entry.setup_pullback_low = None
        self._entry.setup_pullback_ts = None
        self._entry.setup_resolved_at = None
        self._entry.setup_breakout_ts = None
        self._entry.setup_breakout_price = None
        self._entry.setup_failed_at = None

    def _fill_setup_group(self, group: VariableGroup, best: PatternSignal) -> None:
        """Populate the SETUP_PATTERN variable group from a chosen signal."""
        pullback_var = group.get("pullback_depth")
        for v in group.variables:
            v.reset(0.5)

        raw_depth = float(best.pullback_depth) if best.pullback_depth is not None else 0.0
        if pullback_var is not None:
            pullback_var.evaluate(max(0.0, raw_depth))

        v = group.get("pullback_control")
        if v is not None and best.pullback_control is not None:
            v.evaluate(max(0.0, min(1.0, float(best.pullback_control))))

        v = group.get("topping_tail")
        if v is not None and best.topping_tail is not None:
            v.evaluate(max(0.0, float(best.topping_tail)))

        v = group.get("jackknife")
        if v is not None and best.jackknife is not None:
            v.evaluate(max(0.0, float(best.jackknife)))

        v = group.get("pullback_candles")
        if v is not None and best.pullback_candles is not None:
            v.evaluate(float(best.pullback_candles))

        v = group.get("pullback_structure")
        if v is not None and best.pullback_structure is not None:
            v.evaluate(float(best.pullback_structure))

        v = group.get("breakout_extension")
        if v is not None and best.breakout_extension is not None:
            v.evaluate(float(best.breakout_extension))

        v = group.get("leg_candles")
        if v is not None and best.leg_candles is not None:
            v.evaluate(float(best.leg_candles))

    def _setup_signature(self, signal: PatternSignal) -> tuple[str, str, str, float]:
        pullback_ts = signal.pullback_ts.isoformat(timespec="milliseconds") if signal.pullback_ts else ""
        return (
            signal.name,
            signal.timeframe,
            pullback_ts,
            round(float(signal.price_level or 0.0), 4),
        )

    @staticmethod
    def _fmt_setup_number(value: float | None, places: int = 2) -> str:
        if value is None:
            return "n/a"
        return f"{value:.{places}f}"

    def _format_setup_criterion_failure(self, group_name: str, criterion) -> str:
        name = getattr(criterion, "name", "")
        value = getattr(criterion, "value", None)
        numeric = float(value) if value is not None else None
        if name == "change_pct_min":
            return f"change={self._fmt_setup_number(numeric, 1)}% < {CHANGE_PCT_MIN:.1f}%"
        if name == "volume_min":
            return f"volume={int(numeric or 0):,} < {ATTENTION_VOLUME_MIN:,}"
        if name == "rvol_1day_min":
            return f"rvol_1d={self._fmt_setup_number(numeric, 2)} < {ATTENTION_RVOL_1DAY_MIN:.2f}"
        if name == "dist_from_hod_atr_max":
            return f"dist_hod_atr={self._fmt_setup_number(numeric, 2)} > {ATTENTION_DIST_FROM_HOD_ATR_MAX:.2f}"
        if name == "order_book_density_ok":
            return "order_book_density failed"
        if name == "vwap_slope_positive":
            return f"vwap_slope={self._fmt_setup_number(numeric, 4)} < {ATTENTION_VWAP_SLOPE_MIN:g}"
        if name == "close_above_vwap":
            return f"close_minus_vwap={self._fmt_setup_number(numeric, 4)} <= 0"
        if name == "macd_positive":
            return f"macd_hist={self._fmt_setup_number(numeric, 4)} <= 0"
        if name == "tape_speed":
            return f"tape_speed={self._fmt_setup_number(numeric, 1)}/s < {SIM_PRESSURE_SPEED_MIN:.1f}/s"
        if name == "tape_imbalance":
            return f"buy_ratio={(numeric or 0.0):.0%} < {SIM_PRESSURE_BUY_RATIO_MIN:.0%}"
        if name == "tape_vol_flow":
            return f"vol_imbalance={self._fmt_setup_number(numeric, 2)} <= {SIM_PRESSURE_VOL_IMBALANCE_MIN:.2f}"
        if name == "tape_acceleration":
            return f"tape_accel_z={self._fmt_setup_number(numeric, 2)} < {SIM_PRESSURE_ACCEL_Z:.2f}"
        if name == "spread_ok":
            return f"spread={(numeric or 0.0):.2%} > {EXEC_SPREAD_PCT_MAX:.2%}"
        if name == "dist_to_resistance_ok":
            threshold = EXEC_DIST_TO_RESISTANCE_MULT * self._last_atr if self._last_atr > 0 else None
            if threshold is not None:
                return (
                    f"res_headroom={self._fmt_setup_number(numeric, 2)} < "
                    f"{self._fmt_setup_number(threshold, 2)} ({EXEC_DIST_TO_RESISTANCE_MULT:.1f} ATR)"
                )
            return f"res_headroom={self._fmt_setup_number(numeric, 2)} < {EXEC_DIST_TO_RESISTANCE_MULT:.1f} ATR"
        if name == "no_hidden_seller":
            return f"hidden_seller={self._fmt_setup_number(numeric, 2)} > {EXEC_HIDDEN_SELLER_MAX:.2f}"
        return f"{group_name.lower()}:{name} failed"

    def _setup_failure_detail(self, price: float, setup_level: float, approved_min: float, approved_max: float) -> str:
        failures: list[str] = []
        seen: set[str] = set()

        import config as _config
        if not _config.confidence_bypass_eval() and self._entry.eval_confidence < ENTRY_CONFIDENCE_MIN:
            failures.append(f"eval_confidence={self._entry.eval_confidence:.1f} < {ENTRY_CONFIDENCE_MIN:.1f}")
        if (
            not _config.confidence_bypass_live()
            and self._entry.has_live_data
            and self._entry.live_confidence < ENTRY_CONFIDENCE_MIN
        ):
            failures.append(f"live_confidence={self._entry.live_confidence:.1f} < {ENTRY_CONFIDENCE_MIN:.1f}")

        for group in self._entry.criteria_groups:
            for criterion in group.criteria:
                if criterion.required and not criterion.passing:
                    failures.append(self._format_setup_criterion_failure(group.name, criterion))
                    seen.add(str(getattr(criterion, "name", "")))

        setup_score = self._entry.setup_pattern_score
        if setup_score < SIM_SETUP_SCORE_MIN:
            failures.append(f"setup_score={setup_score:.0f} < {SIM_SETUP_SCORE_MIN:.0f}")

        if setup_level > 0 and price > 0:
            if price < approved_min:
                failures.append(f"price={price:.2f} < setup_min={approved_min:.2f}")
            elif price > approved_max:
                failures.append(f"price={price:.2f} > setup_max={approved_max:.2f}")

        if setup_level <= 0:
            failures.append("no setup level (band required for entry)")

        if SIM_FIRST_MACD_CYCLE_ONLY and getattr(self._entry, "macd_cycle_done", False):
            done_ts = getattr(self._entry, "macd_cycle_done_ts", None)
            when = f" @ {done_ts:%H:%M}Z" if done_ts is not None else ""
            failures.append(f"first_macd_cycle_done (1m hist flipped negative{when})")

        ofi = self._ofi_tracker
        total_activity = ofi.activity_count
        if total_activity < SIM_PRESSURE_ACTIVITY_MIN:
            failures.append(f"tape_activity={total_activity} < {SIM_PRESSURE_ACTIVITY_MIN}")
        progress = ofi.price_progress
        if progress is None:
            failures.append("price_progress warming up")
        elif progress <= SIM_PRESSURE_PROGRESS_MIN:
            failures.append(f"price_progress={progress:.3%} <= {SIM_PRESSURE_PROGRESS_MIN:.1%}")
        if ofi.depth_active and ofi.ofi_norm_gate < SIM_OFI_NORM_MIN:
            failures.append(f"ofi_norm={ofi.ofi_norm_gate:.2f} < {SIM_OFI_NORM_MIN}")

        if self._rmap is not None and self._last_atr > 0 and "dist_to_resistance_ok" not in seen:
            try:
                nearest = self._rmap.nearest_above(
                    resistance_search_floor(price, setup_level if setup_level > 0 else None)
                )
            except Exception:
                nearest = None
            if nearest is not None:
                headroom = nearest.price - price
                threshold = EXEC_DIST_TO_RESISTANCE_MULT * self._last_atr
                if headroom < threshold:
                    failures.append(
                        f"res_headroom={headroom:.2f} < {threshold:.2f} "
                        f"({EXEC_DIST_TO_RESISTANCE_MULT:.1f} ATR)"
                    )

        return "; ".join(dict.fromkeys(failures))

    def _begin_setup_signal(self, best: PatternSignal, ts: datetime) -> None:
        signature = self._setup_signature(best)
        if self._active_setup_signature == signature and self._active_setup_signal is not None:
            return

        self._expire_active_setup_signal("superseded by newer setup", ts)

        setup_level = float(best.price_level or self._last_price or 0.0)
        # Entry band mirrors evaluate_entry: floor via the shared
        # entry_band_floor (regime-aware), extension cap above the trigger.
        approved_min = (
            entry_band_floor(setup_level, self._entry.setup_pullback_low)
            if setup_level > 0 else 0.0
        )
        approved_max = setup_level * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT) if setup_level > 0 else 0.0
        signal_price = float(self._last_price or setup_level or 0.0)
        setup_failure_detail = self._setup_failure_detail(signal_price, setup_level, approved_min, approved_max)
        pullback_key = signature[2] or ts.isoformat(timespec="milliseconds")
        setup_id = f"{self._sym}:{best.name}:{best.timeframe}:{signature[3]:.4f}:{pullback_key}"
        signal = SetupSignal(
            setup_id=setup_id,
            ts=ts,
            last_updated_ts=ts,
            symbol=self._sym,
            setup_name=best.name,
            setup_timeframe=best.timeframe,
            setup_score=self._entry.setup_pattern_score,
            eval_confidence=self._entry.eval_confidence,
            live_confidence=self._entry.live_confidence,
            signal_price=signal_price,
            setup_level=setup_level,
            approved_min_price=approved_min,
            approved_max_price=approved_max,
            signal_approved=True,
            approval_ts=ts,
            approval_price=self._last_price or setup_level,
            entry_allowed=self._entry.entry_allowed,
            block_reasons=", ".join(self._entry.entry_block_reasons),
            setup_failure_detail=setup_failure_detail,
            best_ts=ts,
            best_price=signal_price,
            best_distance_to_range=self._price_distance_to_approved_range(signal_price, approved_min, approved_max),
            status="APPROVED",
        )
        self._active_setup_signal = signal
        self._active_setup_signature = signature
        self._active_setup_dirty = False
        if not self._warmup:
            self._setup_logger.upsert_signal(signal)
        self._last_setup_log_write_epoch = ts.timestamp()
        # The latency-critical pin raise: a breakout just fired and entry
        # confirmation wants the tape + book warm right now. Everything
        # else rides the poll-loop refresh.
        self._push_slot_priority()

    def _price_distance_to_approved_range(self, price: float, low: float, high: float) -> float:
        if low <= 0 or high <= 0:
            return 0.0
        if low <= price <= high:
            return 0.0
        return min(abs(price - low), abs(price - high))

    def _setup_entry_readiness_score(self, price: float, triggered: bool) -> float:
        signal = self._active_setup_signal
        ofi = self._ofi_tracker
        score = 10.0 if triggered else 0.0
        score += 3.0 if self._entry.entry_allowed else 0.0
        score += min(self._entry.setup_pattern_score / max(SIM_SETUP_SCORE_MIN, 1.0), 1.0) * 2.0
        if signal is not None:
            width = max(signal.approved_max_price - signal.approved_min_price, 0.01)
            distance = self._price_distance_to_approved_range(
                price, signal.approved_min_price, signal.approved_max_price
            )
            score += max(0.0, 1.0 - distance / width)
        # Same gate values the live entry reads (F10 single source)
        gates = tape_entry_gates(ofi)
        score += min(gates.speed / max(SIM_PRESSURE_SPEED_MIN, 1.0), 1.0)
        score += min(gates.activity / max(SIM_PRESSURE_ACTIVITY_MIN, 1), 1.0)
        score += min(gates.buy_ratio / max(SIM_PRESSURE_BUY_RATIO_MIN, 0.01), 1.0)
        if gates.accel_ready:
            score += min(max(gates.accel_z, 0.0) / SIM_PRESSURE_ACCEL_Z, 1.0)
        if gates.progress is not None:
            score += min(max(gates.progress, 0.0) / max(SIM_PRESSURE_PROGRESS_MIN, 1e-6), 1.0)
        if ofi.depth_active and SIM_OFI_NORM_MIN > 0:
            score += min(max(ofi.ofi_norm_gate, 0.0) / SIM_OFI_NORM_MIN, 1.0)
        return score

    def _update_active_setup_signal(self, price: float, ts: datetime) -> None:
        signal = self._active_setup_signal
        if signal is None:
            return

        entry_sig = evaluate_entry(
            entry=self._entry,
            ofi_tracker=self._ofi_tracker,
            last_price=price,
            last_atr=self._last_atr,
            rmap=self._rmap,
        )
        ofi = self._ofi_tracker
        distance = self._price_distance_to_approved_range(
            price, signal.approved_min_price, signal.approved_max_price
        )
        readiness = self._setup_entry_readiness_score(price, entry_sig.triggered)
        changed = False
        force_flush = False

        if (
            signal.best_ts is None
            or readiness > signal.best_score
            or (readiness == signal.best_score and distance < signal.best_distance_to_range)
        ):
            signal.best_ts = ts
            signal.best_price = price
            signal.best_score = readiness
            signal.best_distance_to_range = distance
            changed = True

        updates = {
            "highest_trade_speed": ofi.trade_speed,
            "highest_buy_ratio": ofi.buy_ratio,
            "highest_pressure_acceleration": ofi.pressure_acceleration,
            "highest_trade_imbalance": ofi.trade_imbalance,
            "highest_ofi_rolling": ofi.ofi_rolling,
            "highest_net_pressure": ofi.buying_pressure - ofi.selling_pressure,
            "highest_buying_pressure": ofi.buying_pressure,
            "highest_selling_pressure": ofi.selling_pressure,
        }
        for attr, value in updates.items():
            if value > getattr(signal, attr):
                setattr(signal, attr, value)
                changed = True

        arm_state = self._arm_state_summary(price)
        if signal.arm_state != arm_state:
            signal.arm_state = arm_state
            changed = True

        block_reasons = ", ".join(self._entry.entry_block_reasons)
        status_reason = " | ".join(entry_sig.reasons) if entry_sig.triggered else (block_reasons or "waiting for tape confirmation")
        if (
            signal.entry_allowed != self._entry.entry_allowed
            or signal.block_reasons != block_reasons
            or signal.eval_confidence != self._entry.eval_confidence
            or signal.live_confidence != self._entry.live_confidence
            or signal.setup_score != self._entry.setup_pattern_score
        ):
            signal.entry_allowed = self._entry.entry_allowed
            signal.block_reasons = block_reasons
            signal.eval_confidence = self._entry.eval_confidence
            signal.live_confidence = self._entry.live_confidence
            signal.setup_score = self._entry.setup_pattern_score
            changed = True

        if entry_sig.triggered:
            if not signal.entered:
                if signal.status != "APPROVED":
                    signal.status = "APPROVED"
                    force_flush = True
                    changed = True
            if signal.status_reason != status_reason:
                signal.status_reason = status_reason
                changed = True
        elif not signal.entered and signal.status_reason != status_reason:
            signal.status_reason = status_reason
            changed = True

        if changed:
            signal.last_updated_ts = ts
            self._active_setup_dirty = True
        self._flush_active_setup_signal(ts, force=force_flush)

    def _arm_state_summary(self, price: float) -> str:
        """Compact armed-band telemetry for the setup-signal log.

        Empty in the legacy regime. Format: "<zone> armed trig=<break|accel|none>"
        or "<zone> stale:<gates>" — last evaluated state wins (block_reasons
        pattern), enough for the A/B report to attribute non-entries.
        Posterior regime: "p=<score><cmp><threshold>" or "out-of-zone".
        """
        import config as _config
        if _config.SIM_ENTRY_REGIME == "posterior":
            level = self._entry.setup_price_level
            if not level or level <= 0:
                return ""
            if price > level * (1.0 + _config.SIM_LEVEL_CROSS_TOLERANCE_PCT):
                return "out-of-zone"
            from exec.posterior import entry_threshold, score_entry
            from exec.signals import trigger_state
            p = score_entry(self._ofi_tracker, price, level)
            th = entry_threshold()
            trig = "break" if trigger_state(self._ofi_tracker)[0] else "none"
            return f"p={p:.3f}{'>=' if p >= th else '<'}{th:.2f} trig={trig}"
        if _config.SIM_ENTRY_REGIME != "armed_band":
            return ""
        level = self._entry.setup_price_level
        if not level or level <= 0:
            return ""
        from exec.signals import ZONE_BELOW, ZONE_EXTENDED, arm_evidence_missing, entry_zone, trigger_state
        zone = entry_zone(price, level)
        if zone == ZONE_EXTENDED:
            return "extended"
        ttl = (
            _config.SIM_ARM_STRONG_TTL_SECS
            if zone == ZONE_BELOW else _config.SIM_ARM_EVIDENCE_TTL_SECS
        )
        missing = arm_evidence_missing(self._ofi_tracker, ttl)
        if missing:
            return f"{zone} stale:{','.join(missing)}"
        trig_break, trig_accel = trigger_state(self._ofi_tracker)
        fired = (
            "break+accel" if (trig_break and trig_accel)
            else ("break" if trig_break else ("accel" if trig_accel else "none"))
        )
        wall = self._ofi_tracker.wall_trend
        wall_str = f" wall={wall:+.2f}" if wall is not None else ""
        veto = (
            " WALL-VETO"
            if (
                self._ofi_tracker.depth_active
                and wall is not None
                and wall <= _config.SIM_WALL_REPLENISH_VETO
            ) else ""
        )
        return f"{zone} armed trig={fired}{wall_str}{veto}"

    def _expire_active_setup_signal(self, reason: str, ts: datetime) -> None:
        signal = self._active_setup_signal
        if signal is None:
            return
        if not signal.entered:
            if signal.signal_approved:
                signal.status = "APPROVED"
                if signal.status_reason:
                    signal.status_reason = f"{signal.status_reason}; {reason}"
                else:
                    signal.status_reason = reason
            else:
                signal.status = "NOT_APPROVED"
                signal.status_reason = reason
                signal.block_reasons = ", ".join(self._entry.entry_block_reasons)
        signal.last_updated_ts = ts
        self._active_setup_dirty = True
        self._flush_active_setup_signal(ts, force=True)
        self._active_setup_signal = None
        self._active_setup_signature = None

    def _flush_active_setup_signal(self, ts: datetime, force: bool = False) -> None:
        if self._warmup:
            return
        signal = self._active_setup_signal
        if signal is None or not self._active_setup_dirty:
            return
        if not force and ts.timestamp() - self._last_setup_log_write_epoch < 1.0:
            return
        self._setup_logger.upsert_signal(signal)
        self._active_setup_dirty = False
        self._last_setup_log_write_epoch = ts.timestamp()

    # ------------------------------------------------------------------
    # Level-respect tracker
    # ------------------------------------------------------------------
    

    def _update_key_candles(self, dfs_by_tf: dict[str, pd.DataFrame]) -> None:
        """Rate the high-volume candle response on each stream timeframe.

        ``key_candles`` remains a LINEAR_MIN raw input, but it behaves as a
        high-volume response score instead of a punishment-only warning.  Each
        timeframe scores closed candles with continuous volume, wick, and body
        gradients: larger relative volume makes a candle matter more, larger
        topping tails / red bodies increase rejection risk, and small wicks stay
        near neutral.  If no timeframe has usable volume evidence, the optional
        variable is cleared and excluded from the live blend.
        """
        ev = self._entry.evaluation
        if ev is None:
            return
        ta = ev.technical_analysis
        var = ta.get("key_candles") if ta is not None else None
        if var is None:
            return

        risk_num = 0.0
        weight_sum = 0.0
        for tf_values in self._tf_pa_var_qualities.values():
            tf_values.pop("key_candles", None)
        for tf_label in _HOD_RESPONSE_TFS:
            score_key = _CHART_CACHE_TF.get(tf_label, "")
            if not score_key:
                continue
            if score_key == "2m" and not PRICE_ACTION_TF_2M_ENABLED:
                continue
            df = dfs_by_tf.get(tf_label)
            if df is None or df.empty:
                continue
            if self._tf_buf.current_bar(tf_label) is not None and len(df) >= 1:
                df = df.iloc[:-1]
            if len(df) < 2 or "volume" not in df:
                continue
            highs = df["high"].to_numpy(dtype=float)
            opens = df["open"].to_numpy(dtype=float)
            lows = df["low"].to_numpy(dtype=float)
            closes = df["close"].to_numpy(dtype=float)
            volumes = df["volume"].to_numpy(dtype=float)

            baseline = _volume_response_baseline(volumes)
            if baseline is None:
                continue
            volume_floor, volume_full = baseline

            atr_values: list[float] = []
            tr_window: deque[float] = deque(maxlen=max(1, TOPPING_TAIL_ATR_PERIOD))
            tr_sum = 0.0
            prev_close = float(closes[0])
            for idx in range(len(closes)):
                high = float(highs[idx])
                low = float(lows[idx])
                close = float(closes[idx])
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                if len(tr_window) == tr_window.maxlen:
                    tr_sum -= tr_window[0]
                tr_window.append(tr)
                tr_sum += tr
                atr_values.append(tr_sum / len(tr_window) if tr_window else 0.0)
                prev_close = close

            tf_num = 0.0
            tf_weight = 0.0
            for idx in range(len(closes)):
                metrics = _candle_volume_response(
                    float(opens[idx]),
                    float(highs[idx]),
                    float(lows[idx]),
                    float(closes[idx]),
                    float(volumes[idx]),
                    atr_values[idx],
                    volume_floor,
                    volume_full,
                )
                if metrics is None:
                    continue
                volume_weight, structural_risk, red_volume_risk = metrics
                if volume_weight <= 0.0:
                    continue

                rejection = min(
                    1.0,
                    1.0 - ((1.0 - structural_risk) * (1.0 - red_volume_risk)),
                )
                tf_num += volume_weight * rejection
                tf_weight += volume_weight
            if tf_weight <= 0.0:
                continue

            tf_risk = tf_num / tf_weight
            if score_key:
                q = compute_quality(tf_risk, var.fn, var.lo, var.hi)
                self._tf_pa_var_qualities.setdefault(score_key, {})["key_candles"] = q
            weight = PRICE_ACTION_TF_WEIGHTS.get(score_key, 0.0)
            if weight <= 0.0:
                continue
            risk_num += weight * tf_risk
            weight_sum += weight

        if weight_sum > 0.0:
            ev.apply_live(LiveInputs(key_candles=risk_num / weight_sum))
            self._refresh_technical_analysis_score()
        elif var.value is not None:
            # No current usable volume evidence.
            var.value = None
            var.quality = 0.0
            self._refresh_technical_analysis_score()

    def _compute_candle_choppiness(
        self, dfs_by_tf: dict[str, pd.DataFrame]
    ) -> tuple[float | None, dict[str, float]]:
        """Per-timeframe candle choppiness.

        For each timeframe, choppiness is measured over the active move window
        on CLOSED candles only (the forming bar is dropped).  Returns a tuple
        ``(blended_ratio, {tf_score_key: per_tf_ratio})``: the blended ratio
        combines per-TF ratios by ``PRICE_ACTION_TF_WEIGHTS`` (which down-weights
        the noisy 10-second series) and feeds the scoring var, while the per-TF
        ratios drive the weights-display table.  ``blended_ratio`` is ``None``
        when no timeframe has enough data.
        """
        num = 0.0
        w_sum = 0.0
        per_tf: dict[str, float] = {}
        for tf_label in ("10sec", "1min", "2min", "5min"):
            if tf_label == "2min" and not PRICE_ACTION_TF_2M_ENABLED:
                # 2m is gated out of every live-confidence blend (same check
                # as key_candles / combine_tf_scores).
                continue
            df = dfs_by_tf.get(tf_label)
            if df is None or df.empty:
                continue
            # Drop the still-forming bar so a 1-tick candle can't skew colors.
            if self._tf_buf.current_bar(tf_label) is not None and len(df) >= 1:
                df = df.iloc[:-1]
            if len(df) < CANDLE_CHOP_MIN_CANDLES:
                continue
            active_start = self._active_price_action_start(df.index[-1])
            df = df[df.index >= active_start]
            if len(df) < CANDLE_CHOP_MIN_CANDLES:
                continue
            ratio = _candle_choppiness_ratio(
                df["open"].to_numpy(dtype=float),
                df["high"].to_numpy(dtype=float),
                df["close"].to_numpy(dtype=float),
            )
            if ratio is None:
                continue
            score_key = _CHART_CACHE_TF.get(tf_label, "")
            if score_key:
                per_tf[score_key] = ratio
            weight = PRICE_ACTION_TF_WEIGHTS.get(score_key, 0.0)
            if weight <= 0.0:
                continue
            num += weight * ratio
            w_sum += weight

        blended = num / w_sum if w_sum > 0.0 else None
        return blended, per_tf

    def _update_atrp_and_choppiness(self, dfs_by_tf: dict[str, pd.DataFrame]) -> None:
        """Refresh the ATRP and candle-choppiness live TECHNICAL_ANALYSIS vars.

        - ``atrp`` = current 14-period 1-min ATR as a percent of price
          (higher = more movement; SATURATING).
        - ``candle_choppiness`` = red/green alternation ratio, succeeded and
          still-active pullbacks excused (lower = cleaner; LINEAR_MIN).  The
          blended ratio across
          timeframes feeds the scoring var; per-TF qualities feed the weights
          table (rendered per-TF like setup_outcome).

        Both are optional, so when their input is unavailable they are cleared
        (``value=None``) and excluded from the blend rather than diluting it.
        """
        ev = self._entry.evaluation
        if ev is None:
            return
        ta = ev.technical_analysis
        if ta is None:
            return

        atrp_var = ta.get("atrp")
        if atrp_var is not None:
            if self._last_atr > 0 and self._last_price > 0:
                ev.apply_live(LiveInputs(atrp=self._last_atr / self._last_price * 100.0))
            elif atrp_var.value is not None:
                atrp_var.value = None
                atrp_var.quality = 0.0

        chop_var = ta.get("candle_choppiness")
        if chop_var is not None:
            chop, per_tf_ratios = self._compute_candle_choppiness(dfs_by_tf)
            # Per-TF qualities drive the weights-display table (like setup_outcome).
            for score_key, ratio in per_tf_ratios.items():
                q = compute_quality(ratio, chop_var.fn, chop_var.lo, chop_var.hi)
                self._tf_pa_var_qualities.setdefault(score_key, {})["candle_choppiness"] = q
            if chop is not None:
                ev.apply_live(LiveInputs(candle_choppiness=chop))
            elif chop_var.value is not None:
                chop_var.value = None
                chop_var.quality = 0.0

        self._refresh_technical_analysis_score()

    def _update_no_news_penalty(self, now: datetime) -> None:
        """Flip ``breaking_news`` from optional/neutral to a penalty once the
        no-news grace window has elapsed since start-of-move with STILL no news
        found.

        Real news, when found, always owns ``breaking_news`` (via
        ``update_news``) and is left untouched here.  The reference clock is the
        latest event time, so this is replay-safe in the backtest.
        """
        ev = self._entry.evaluation
        if ev is None or self._move_start_ts is None:
            return
        if self._entry.avg_sentiment is not None:
            return  # news catalyst present — news paths own breaking_news
        ref = self._coerce_utc_dt(now)
        if ref is None:
            return
        if (ref - self._move_start_ts).total_seconds() <= NO_NEWS_GRACE_SECS:
            return  # still within the grace window — leave breaking_news neutral
        ev.apply_no_news_penalty(NO_NEWS_PENALTY_QUALITY)

    def _update_price_action_quality(self, tf_label: str, candles) -> None:  # noqa: ARG002
        """Legacy no-op — retained for backtest compatibility."""
        pass

    def _refresh_technical_analysis_score(self) -> None:
        """Refresh TECHNICAL_ANALYSIS override from TF setup outcome score plus live rows."""
        ev = self._entry.evaluation
        if ev is None:
            return
        pa = ev.technical_analysis
        if not self._tf_pa_scores:
            # No setups — clear the override so setup_outcome (optional) simply
            # drops out of the group score instead of pinning it at a neutral
            # 50%.  The live confidence then reflects only the real live signals
            # (or nothing / "L--" when there is no live data yet).
            pa.score_override = None
            return

        setup_vars = {"setup_outcome"}
        candle_score = combine_tf_scores(self._tf_pa_scores, self._tf_pa_masses)
        # Evidence mass gates how much setup_outcome weighs in the blend: one
        # fully-evidenced TF is complete evidence (max, not mean — quiet TFs
        # must not dilute a hot one). Below the negligible-evidence floor the
        # setup vars drop out of the blend entirely, exactly like the
        # no-setups branch above.
        candle_mass = max(
            (
                max(0.0, min(1.0, self._tf_pa_masses.get(tf, 1.0)))
                for tf in self._tf_pa_scores
            ),
            default=0.0,
        )
        if candle_mass <= MIN_TF_EVIDENCE_MASS:
            candle_mass = 0.0
        candle_weight = sum(v.weight for v in pa.variables if v.name in setup_vars) * candle_mass

        w_sum = candle_weight
        wq_sum = candle_weight * candle_score
        for v in pa.variables:
            if v.name in setup_vars:
                continue
            if v.value is None and (v.optional or v.dampened):
                continue
            q = v.quality if v.value is not None else 0.5
            w_sum += v.weight
            wq_sum += v.weight * q

        pa.score_override = wq_sum / w_sum if w_sum > 0.0 else candle_score

    

    

    def _scan_patterns_on_close(self, closed_tfs: list[str]) -> None:
        """
        Run pattern detection when stream-built candles close.

        Called from _on_timesale (under lock).  Full flow:

        1. Build OHLCV DataFrames from the multi-timeframe buffer.
        2. Run scan_all_patterns() — stateless detection across all 4 TFs.
        3. Feed the raw snapshot into the pullback tracker:
           - New patterns start being tracked.
           - Existing patterns that persist beyond their candle width get
             promoted to the next higher timeframe (10s→1m→2m→5m).
           - Patterns whose price level drifted away are expired.
        4. Merge promoted signals back into the snapshot.
        5. Run _eval_patterns() to set/clear dynamic delta flags.
        """
        df_10s = self._tf_buf.to_df("10sec")
        df_1m  = self._tf_buf.to_df("1min")
        df_2m  = self._tf_buf.to_df("2min")
        df_5m  = self._tf_buf.to_df("5min")

        # Step 2: Run stateless pattern detectors
        snap = scan_all_patterns(df_10s, df_1m, df_2m, df_5m, self._rmap)

        self._patterns = snap

        # Keep setup confidence in a separate channel from live confidence.
        self._update_setup_pattern_confidence(snap)

        # Step 3: Set/clear bullish/bearish flags
        self._eval_patterns()

        dfs_by_tf = {"10sec": df_10s, "1min": df_1m, "2min": df_2m, "5min": df_5m}

        # High-volume candle response (topping tails / jackknives / red-volume
        # pressure) across stream timeframes.  Independent of the pullback
        # setup channel.
        self._update_key_candles(dfs_by_tf)

        # Step 6: ATRP (volatility) + candle choppiness live-quality vars.
        self._update_atrp_and_choppiness(dfs_by_tf)

        # Log which timeframes triggered
        highest = max(closed_tfs, key=lambda t: _TIMEFRAME_RANK.get(t, 0))
        if self._patterns.signals:
            logger.debug(
                f"[{self._sym}] pattern scan triggered by {highest} close: "
                f"{len(self._patterns.signals)} signals"
            )

    # ------------------------------------------------------------------
    # REST candle poller
    # ------------------------------------------------------------------

    @staticmethod
    def _next_poll_delay() -> float:
        """Poll interval plus jitter, so scorers do not all wake together.

        Startup seeds every cached symbol at once, which would otherwise line
        every scorer up on the same tick — and each one now reconnects on its
        poll, so an aligned fleet would spike the connection ceiling.
        """
        return SCORER_POLL_INTERVAL_SECS + random.uniform(0, SCORER_POLL_JITTER_SECS)

    def _poll_loop(self) -> None:
        """Fetch 1-min candles via REST every SCORER_POLL_INTERVAL_SECS."""
        # Initial fetch immediately, then loop.
        try:
            while not self._stop.is_set():
                try:
                    self._refresh_candles()
                except TradierClientError:
                    # Transient API rejection (common for thin pre-market names).
                    # Disregard and keep polling — never blacklist a silent ticker;
                    # streaming/candles resume automatically when it comes back.
                    logger.debug(f"[{self._sym}] candle poll skipped — symbol rejected by API (will retry)")
                except Exception:
                    logger.exception(f"[{self._sym}] candle poll error")
                finally:
                    # Release between polls, not just at thread exit. The fetch
                    # is ~200ms once a minute — a ~0.3% duty cycle — so holding
                    # the connection idle in between made the Gateway's
                    # 32-connection ceiling scale with watchlist size. Now it
                    # scales with concurrent fetches instead. Measured cost of
                    # reconnecting: ~200ms per poll against a 60s interval, so
                    # ~0.3% overhead. Safe when the poll never connected, and a
                    # no-op when the provider is Tradier.
                    _release_ibkr_thread_conn()
                # Catch-all pin refresh: arming on candle close, position
                # open/close, and lazy downgrades all land within one poll.
                self._push_slot_priority()
                self._stop.wait(self._next_poll_delay())
        finally:
            # Belt and braces: the loop above already released after the last
            # poll, but an exit between the release and the wait must not leak.
            _release_ibkr_thread_conn()

    def _start_10s_seed_worker(self) -> None:
        """Dispatch the 10-sec tick-window seed off the poll thread.

        The REST fetch is chunked + parallelised internally but can still
        take several seconds end-to-end; running it on the poll thread
        delays every other live update.  This worker fires once, merges
        the result into ``_tf_buf`` under ``_lock``, then exits.
        """
        if self._seed_thread is not None:
            return
        self._seed_thread = threading.Thread(
            target=self._seed_10s_worker, daemon=True, name=f"seed10s-{self._sym}",
        )
        self._seed_thread.start()

    def _seed_10s_worker(self) -> None:
        try:
            self._seed_10s_worker_inner()
        finally:
            # Fires once then exits — release its connection slot on every
            # exit path, including the early returns below.
            _release_ibkr_thread_conn()

    def _seed_10s_worker_inner(self) -> None:
        source = "rest"
        try:
            if not self._df_10sec_seed.empty:
                source = "cache"
                df_10s = self._df_10sec_seed.copy()
                self._df_10sec_seed = pd.DataFrame()
            else:
                end = datetime.now(tz=timezone.utc)
                start = self._active_price_action_start(end)
                fetch_window = getattr(self._hist, "get_candles_10sec_window", None)
                if fetch_window is None:
                    df_10s = self._hist.get_candles_10sec(self._sym, n=SCORER_CANDLE_MAX_BARS)
                else:
                    df_10s = fetch_window(self._sym, start, end)
        except TradierClientError:
            logger.warning(f"[{self._sym}] 10s seed skipped — symbol rejected by API")
            return
        except Exception:
            logger.exception(f"[{self._sym}] 10s seed fetch failed")
            return
        if self._stop.is_set():
            return
        candles_10s = self._df_to_candles(df_10s, "10sec") if not df_10s.empty else []
        if candles_10s and source != "cache":
            self._chart_cache.record_candles("10s", candles_10s, SOURCE_POLLED)
        with self._lock:
            if self._stop.is_set():
                return
            self._tf_buf.seed("10sec", candles_10s)
        logger.info(f"[{self._sym}] 10s {source} seed merged into pattern buffer: {len(candles_10s)} bars")

    def _fetch_1min_candles(self) -> pd.DataFrame:
        if self._df_1min_cache.empty:
            # First fetch: get the full session for accurate VWAP seeding,
            # but keep move-window bars for scoring/pattern context.
            #
            # Anchored to the session open rather than a bar count. A count
            # walks backwards from *now*, so early in the day it reaches into
            # the prior session and seeds VWAP with yesterday's bars; the
            # anchor cannot. It is also the single most pacing-throttled call
            # in the system — `get_candles_1min` asks for CANDLES_1MIN_FETCH_DAYS
            # regardless of n, ~4200 bars to keep ~390, and IBKR queues
            # requests that large behind multi-second waits. Nothing reads a
            # bar older than the session open: detect_move_start masks to it,
            # and _active_price_action_start clamps to it.
            fetch_window = getattr(self._hist, "get_candles_1min_window", None)
            if fetch_window is not None:
                df = fetch_window(self._sym, latest_extended_session_start())
            else:
                # Providers without precise windows: bar count, as before.
                from config import BARS_PER_SESSION_1MIN
                df = self._hist.get_candles_1min(self._sym, n=BARS_PER_SESSION_1MIN)
            if not df.empty:
                self._df_1min_session = self._normalize_candle_df(df)
                self._df_1min_cache = self._clip_df_since(
                    self._df_1min_session,
                    self._active_price_action_start(self._latest_df_ts(self._df_1min_session)),
                )
            return self._df_1min_cache

        fetch_recent = getattr(self._hist, "get_candles_1min_recent", None)
        if fetch_recent is None:
            df_recent = self._hist.get_candles_1min(self._sym, n=SCORER_CANDLE_MAX_BARS)
        else:
            df_recent = fetch_recent(self._sym, n=min(5, SCORER_CANDLE_MAX_BARS), lookback_minutes=8)
        if df_recent.empty:
            return self._df_1min_cache

        # Append new bars to the full-session frame (for VWAP accuracy)
        if not self._df_1min_session.empty:
            session_merged = pd.concat([self._df_1min_session, df_recent], ignore_index=True)
            session_merged["ts"] = pd.to_datetime(session_merged["ts"], utc=True)
            self._df_1min_session = (
                session_merged.drop_duplicates(subset=["ts"], keep="last")
                .sort_values("ts")
                .reset_index(drop=True)
            )

        merged = pd.concat([self._df_1min_cache, df_recent], ignore_index=True)
        merged["ts"] = pd.to_datetime(merged["ts"], utc=True)
        merged = (
            merged.drop_duplicates(subset=["ts"], keep="last")
            .sort_values("ts")
            .reset_index(drop=True)
        )
        merged = self._clip_df_since(
            merged,
            self._active_price_action_start(self._latest_df_ts(merged)),
        )
        self._df_1min_cache = merged
        return merged

    def _refresh_candles(
        self,
        *,
        use_seed_only: bool = False,
        record_cache: bool = True,
        notify: bool = True,
    ) -> None:
        """Pull 1-min candles from REST, resample to 2/5-min, evaluate per-TF price action."""
        df = self._df_1min_cache if use_seed_only else self._fetch_1min_candles()
        if df.empty:
            return
        updated = False

        # Recompute move_start_ts once polled candles confirm the move.
        if self._move_start_ts is None:
            df_session = self._df_1min_session if not self._df_1min_session.empty else df
            move = detect_move_start(df_session)
            if move is not None:
                try:
                    ts = pd.Timestamp(df_session.iloc[move.start_idx]["ts"]).to_pydatetime()
                    self._move_start_ts = self._coerce_utc_dt(ts)
                except (IndexError, KeyError):
                    pass

        candles_1m = self._df_to_candles(df, "1min")
        df_session = self._df_1min_session if not self._df_1min_session.empty else df
        candles_1m_session = self._df_to_candles(df_session, "1min")

        # Index once for both resample operations
        sym = df["symbol"].iloc[0]
        df_ts = df.set_index("ts")
        candles_2m = self._resample_indexed(df_ts, sym, "2min")
        candles_5m = self._resample_indexed(df_ts, sym, "5min")
        df_session_ts = df_session.set_index("ts")
        candles_2m_session = self._resample_indexed(df_session_ts, sym, "2min")
        candles_5m_session = self._resample_indexed(df_session_ts, sym, "5min")
        polled_dfs_by_tf = {
            "1min": df_ts,
            "2min": self._candles_to_df(candles_2m),
            "5min": self._candles_to_df(candles_5m),
        }

        if record_cache:
            self._chart_cache.record_candles("1m", candles_1m_session, SOURCE_POLLED)

        # First seed with a cached tape replays up to SCORER_MAX_RECONSTRUCT_TICKS
        # ticks under _lock (measured ~0.6 ms/tick -> up to ~15 s). Hold live
        # stream events for that window so the shared stream thread never
        # blocks on this lock — it dispatches every symbol, so blocking it
        # here froze the whole watchlist. The hold context exits AFTER the
        # lock is released and drains the parked events in order.
        hold_for_replay = (not self._tf_seeded) and not self._df_timesales_seed.empty
        with self._event_hold(hold_for_replay), self._lock:
            self._candles = candles_1m
            self._candles_session = candles_1m_session
            self._candles_2min = candles_2m
            self._candles_5min = candles_5m
            updated = True

            # vwap_slope / above_200ema always update (not TA-gated)
            if len(self._candles) >= 3:
                poll_inputs = LiveInputs()
                slope = self._compute_live_vwap_slope()
                if slope is None:
                    slope = self._compute_vwap_slope(df)
                if slope is not None:
                    poll_inputs.vwap_slope = slope

                # Update above_200ema if price crossed over
                ma200_val = self._check_above_200()
                if ma200_val is not None:
                    poll_inputs.above_200ema = ma200_val

                self._entry.evaluation.apply_live(poll_inputs)

            # Update execution criteria + hidden seller (best-effort)
            _near = self._is_near_hod()
            try:
                self._update_criteria_and_hidden_seller(df, include_hidden_seller=_near)
            except Exception:
                logger.exception(f"[{self._sym}] criteria update failed")

            try:
                self._update_key_candles(polled_dfs_by_tf)
                self._update_atrp_and_choppiness(polled_dfs_by_tf)
            except Exception:
                logger.exception(f"[{self._sym}] poll-derived live score update failed")

            # Seed the multi-TF buffer once.  The 10-sec seed is dispatched
            # to a worker thread because it does a heavy tick-window REST
            # fetch (chunked → multiple round-trips) that can otherwise
            # stall the 1-min refresh cycle for several seconds.
            if not self._tf_seeded:
                self._tf_seeded = True
                seeds = {
                    "1min": self._clip_seed_candles_pre_stream("1min", candles_1m),
                    "2min": self._clip_seed_candles_pre_stream("2min", candles_2m),
                    "5min": self._clip_seed_candles_pre_stream("5min", candles_5m),
                }

                # Seed stream VWAP cumulatives from full-session 1-min candles
                # so _stream_vwap is accurately anchored from session open.
                session_candles = (
                    self._df_to_candles(self._df_1min_session, "1min")
                    if not self._df_1min_session.empty
                    else candles_1m
                )
                now_utc = datetime.now(tz=timezone.utc)
                current_session = self._session_key_from_utc(now_utc)
                self._vwap_session_key = current_session
                cum_pv = 0.0
                cum_vol = 0
                history: list[float] = []
                for c in session_candles:
                    if self._session_key_from_utc(c.ts) == current_session:
                        tp = self._bar_vwap_price(c)
                        vol = int(c.volume)
                        cum_pv += tp * vol
                        cum_vol += vol
                        if cum_vol > 0:
                            history.append(cum_pv / cum_vol)
                self._vwap_cum_pv_closed = cum_pv
                self._vwap_cum_vol_closed = cum_vol
                self._vwap_closed_history = deque(history[-32:], maxlen=32)
                if cum_vol > 0:
                    self._stream_vwap = cum_pv / cum_vol

                # Seed EMA 9/20 from full-session closes so they start
                # converged rather than initializing flat on first stream bar.
                alpha9 = 2.0 / (9 + 1)
                alpha20 = 2.0 / (20 + 1)
                ema9: float | None = None
                ema20: float | None = None
                for c in session_candles:
                    if self._session_key_from_utc(c.ts) == current_session:
                        close = float(c.close)
                        ema9 = close if ema9 is None else ema9 + alpha9 * (close - ema9)
                        ema20 = close if ema20 is None else ema20 + alpha20 * (close - ema20)
                if ema9 is not None:
                    self._ema9_base = ema9
                    self._last_ema9 = ema9
                if ema20 is not None:
                    self._ema20_base = ema20
                    self._last_ema20 = ema20

                # Retain the whole active move in the pattern buffer so candle
                # choppiness spans start-of-move → now instead of a trailing
                # window that slides forward as new bars close.
                self._tf_buf.set_retention_start(
                    self._active_price_action_start(now_utc)
                )

                if not self._df_timesales_seed.empty and len(self._df_timesales_seed) <= SCORER_MAX_RECONSTRUCT_TICKS:
                    # Cached tape available → rebuild the pattern buffer, setup
                    # outcome history and candle choppiness from the start of the
                    # move by replaying the tape.  This is what makes those
                    # metrics survive a mid/post-move restart (``TICKER -r``).
                    n = self._reconstruct_pattern_state_from_ticks_locked()
                    logger.info(
                        f"[{self._sym}] reconstructed pattern state from {n} "
                        f"cached ticks (1d={len(self._candles_daily)})"
                    )
                elif not self._df_timesales_seed.empty:
                    seeded_timesales = self._seed_timesales_from_cache_locked()
                    if seeded_timesales:
                        logger.info(
                            f"[{self._sym}] T&S seed replayed: {seeded_timesales} prints "
                            f"(full reconstruction skipped; cached tape too large)"
                        )
                else:
                    # No cached tape → seed the pattern buffer from REST candles
                    # and fetch the 10-sec window off-thread.
                    self._tf_buf.seed("1min", seeds["1min"])
                    self._tf_buf.seed("2min", seeds["2min"])
                    self._tf_buf.seed("5min", seeds["5min"])
                    logger.info(
                        f"[{self._sym}] seeded pattern buffer: "
                        f"1m={len(seeds['1min'])} 2m={len(seeds['2min'])} "
                        f"5m={len(seeds['5min'])} 1d={len(self._candles_daily)} "
                        f"(10s seed dispatched async)"
                    )
                    self._start_10s_seed_worker()

            # --- Scoring telemetry snapshot (every poll) ---
            if SCORING_LOG_ENABLED:
                sl = _get_scoring_logger()
                if sl:
                    sl.snapshot(self._entry)
                    sl.tape_snapshot(self._entry, self._ofi_tracker, self._last_price, self._tf_pa_scores)
        if notify and updated and self._on_update is not None:
            self._on_update()

    @staticmethod
    def _candles_to_df(candles: list[OHLCVCandle]) -> pd.DataFrame:
        if not candles:
            df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])
            df.index.name = "ts"
            return df
        return pd.DataFrame(
            [(c.ts, c.open, c.high, c.low, c.close, c.volume, c.vwap) for c in candles],
            columns=["ts", "open", "high", "low", "close", "volume", "vwap"],
        ).set_index("ts")

    @staticmethod
    def _df_to_candles(df: pd.DataFrame, interval: str) -> list[OHLCVCandle]:
        """Convert a candle DataFrame to a list of OHLCVCandle."""
        if df.empty:
            return []
        candles: list[OHLCVCandle] = []
        for row in df.itertuples(index=False):
            raw_vwap = getattr(row, "vwap", None)
            candles.append(
                OHLCVCandle(
                    symbol=row.symbol, ts=row.ts,
                    open=float(row.open), high=float(row.high), low=float(row.low), close=float(row.close),
                    volume=int(row.volume), interval=interval,
                    vwap=None if raw_vwap is None or pd.isna(raw_vwap) else float(raw_vwap),
                )
            )
        return candles

    def _clip_seed_candles_pre_stream(
        self,
        timeframe: str,
        candles: list[OHLCVCandle],
    ) -> list[OHLCVCandle]:
        """Keep only seed candles that are strictly older than stream coverage.

        The scorer subscribes to stream before the poller starts, so stream bars
        may already exist by the time REST seed data arrives. Seeding overlapping
        bars can skew setup pattern timeframe selection.
        """
        if not candles or self._first_stream_ts is None:
            return candles
        secs = dict(_TF_INTERVALS).get(timeframe)
        if secs is None:
            return candles
        cutoff = _MultiTFCandleBuffer._bucket_ts(self._first_stream_ts, secs)
        return [c for c in candles if c.ts < cutoff]

    @staticmethod
    def _resample_indexed(df_ts: pd.DataFrame, sym: str, rule: str) -> list[OHLCVCandle]:
        """Resample an already-indexed DataFrame to OHLCVCandle list (avoids duplicate set_index)."""
        if df_ts.empty:
            return []
        ohlcv = (
            df_ts.resample(rule)
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna(subset=["open"])
        )
        if "vwap" in df_ts.columns:
            vwap_source = df_ts[["vwap", "volume"]].copy()
            vwap_source["vwap"] = pd.to_numeric(vwap_source["vwap"], errors="coerce")
            vwap_source["volume"] = pd.to_numeric(vwap_source["volume"], errors="coerce").fillna(0)
            vwap_source["_pv"] = vwap_source["vwap"] * vwap_source["volume"]
            grouped = vwap_source.resample(rule).agg({"_pv": "sum", "volume": "sum"})
            ohlcv["vwap"] = grouped["_pv"] / grouped["volume"].where(grouped["volume"] > 0)
        return [
            OHLCVCandle(
                symbol=sym, ts=ts,
                open=float(o), high=float(h), low=float(lo), close=float(c),
                volume=int(v), interval=rule,
                vwap=None if "vwap" not in ohlcv.columns or pd.isna(vw) else float(vw),
            )
            for values in zip(
                ohlcv.index, ohlcv["open"], ohlcv["high"],
                ohlcv["low"], ohlcv["close"], ohlcv["volume"],
                ohlcv["vwap"] if "vwap" in ohlcv.columns else [None] * len(ohlcv),
            )
            for ts, o, h, lo, c, v, vw in (values,)
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_above_200(self) -> float | None:
        """Return 1.0 if price is above both 200 EMA and 200 MA levels.

        Returns 0.0 if below and distance < 100%, 0.5 (neutral) if >= 100% away.
        Returns None if no 200 EMA/MA levels are found in the resistance map.
        """
        levels_200 = [
            l for l in self._rmap.all_levels()
            if "200 EMA" in l.source or "200 MA" in l.source
        ]
        if not levels_200:
            return None
        price = self._last_price
        if price <= 0:
            return None
        above_all = all(price > l.price for l in levels_200)
        if above_all:
            return 1.0
        # Below at least one — check distance to nearest
        nearest = min(levels_200, key=lambda l: abs(l.price - price))
        dist_pct = (nearest.price - price) / price
        if dist_pct >= 1.0:
            return 0.5  # too far — neutral
        return 0.0

    def _compute_rvol_5min(self) -> float:
        """
        Live 5-min RVol: sum T&S tick volume in rolling window vs
        avg_daily / RVOL_BARS_PER_SESSION (30-day benchmark).

        See also: evaluator.py:_compute_5min_rvol() (relative-to-recent-self),
                  indicators.py:compute_rvol_5min_bars() (candle-based).
        """
        if self._avg_vol <= 0 or not self._vol_prints:
            return 0.0
        window_vol = self._vol_prints_sum
        expected   = self._avg_vol / RVOL_BARS_PER_SESSION
        return round(window_vol / expected, 2) if expected > 0 else 0.0

    def _update_tick_criteria(self, price: float) -> None:
        """Re-evaluate execution criteria on every timesale tick.

        Execution criteria are edge-triggered T&S event detectors:
                    - tape_speed: speed gate (fast-EMA rate by default) >= execution speed floor
                    - tape_imbalance: buy_ratio >= execution imbalance floor
                    - tape_acceleration: enough data and acceleration >= floor
                    - spread_ok: NBBO spread <= execution spread floor
          - dist_to_resistance_ok: headroom to major resistance
          - no_hidden_seller: L2 iceberg detection (IBKR-gated)

        Also updates ATTENTION:dist_from_hod_atr_max per-tick.
        """
        from config import (
            EXEC_DIST_TO_RESISTANCE_MULT,
            ATTENTION_DIST_FROM_HOD_ATR_MAX,
            EXEC_HIDDEN_SELLER_MAX,
        )

        execg = self._entry.get_criteria_group("EXECUTION")
        if execg is None:
            return

        # EXECUTION: spread_ok — L1 NBBO spread as a percentage of midpoint.
        if self._bid > 0 and self._ask > 0:
            spread = self._ask - self._bid
            mid = (self._ask + self._bid) / 2.0
            spread_pct = spread / mid if mid > 0 else float("inf")
            execg.update(
                "spread_ok",
                spread >= 0.0 and spread_pct <= EXEC_SPREAD_PCT_MAX,
                spread_pct,
            )

        # ── T&S event detectors ──────────────────────────────────────────
        # Single source of truth: exec.signals.tape_entry_gates — the exact
        # values/thresholds the live entry gate reads (F10). The activity
        # gate deliberately has no display criterion (enforced by
        # evaluate_entry and the backtest block-reason path).
        gates = tape_entry_gates(self._ofi_tracker)
        execg.update("tape_speed", gates.speed_ok, gates.speed)
        execg.update("tape_imbalance", gates.ratio_ok, gates.buy_ratio)
        execg.update("tape_vol_flow", gates.vol_flow_ok, gates.vol_imbalance)
        execg.update("tape_acceleration", gates.accel_ok, gates.accel_z)

        # ── Structural safety gates ──────────────────────────────────────

        # EXECUTION: Distance to nearest major resistance. The active setup's
        # own prior high (anything inside the approved entry band) is the
        # level being broken, not a wall — skipped via resistance_search_floor
        # so this criterion can never veto the setup's own re-entry (F10
        # parity with evaluate_entry).
        if self._rmap is not None and self._last_atr > 0:
            try:
                search_floor = resistance_search_floor(
                    price, self._entry.setup_price_level
                )
                nearest = self._rmap.nearest_above(search_floor)
                if nearest is not None:
                    headroom = nearest.price - price
                    ok = headroom >= EXEC_DIST_TO_RESISTANCE_MULT * self._last_atr
                    execg.update("dist_to_resistance_ok", ok, headroom)
                else:
                    execg.update("dist_to_resistance_ok", True, 0.0)
            except Exception:
                pass

        # EXECUTION: no_hidden_seller (L2 — only meaningful with IBKR)
        if IBKR_ENABLED:
            hs = self._ofi_tracker.hidden_seller
            execg.update("no_hidden_seller", hs <= EXEC_HIDDEN_SELLER_MAX, hs)

        # ── Persist execution snapshot on entry for cache serialisation ───
        ofi = self._ofi_tracker
        snap = self._entry.execution_snapshot
        snap["trade_speed"] = ofi.trade_speed
        snap["buy_ratio"] = ofi.buy_ratio
        snap["pressure_acceleration"] = ofi.pressure_acceleration
        snap["pressure_acceleration_ready"] = ofi.pressure_acceleration_ready
        snap["trade_imbalance"] = ofi.trade_imbalance
        snap["ofi_rolling"] = ofi.ofi_rolling
        snap["ofi_normalized"] = ofi.ofi_normalized
        snap["book_depth_avg"] = ofi.book_depth_avg
        snap["buying_pressure"] = ofi.buying_pressure
        snap["selling_pressure"] = ofi.selling_pressure
        snap["buy_volume"] = ofi.buy_volume
        snap["sell_volume"] = ofi.sell_volume
        snap["baseline_speed"] = ofi.baseline_speed
        snap["speed_ratio"] = ofi.speed_ratio
        snap["book_lean"] = ofi.book_lean

        # ── ATTENTION: dist from HOD (per-tick for responsiveness) ───────
        # Strict ATR gate — no 10% widening for execution criteria.
        if self._last_atr > 0 and self._entry.daily_high > 0:
            hod = self._entry.daily_high
            dist_atrs = (hod - price) / self._last_atr
            attn = self._entry.get_criteria_group("ATTENTION")
            if attn is not None:
                attn.update("dist_from_hod_atr_max",
                            dist_atrs <= ATTENTION_DIST_FROM_HOD_ATR_MAX,
                            dist_atrs)

    def _update_criteria_and_hidden_seller(
        self,
        df_1m: pd.DataFrame,
        *,
        include_hidden_seller: bool = True,
    ) -> None:
        """Update attention criteria + hidden_seller live var from REST poll.

        Called from _refresh_candles inside _lock with the freshly fetched
        1-min dataframe.  All updates are idempotent and side-effect-free
        on errors (best-effort).

        Updates:
          * ATTENTION:dist_from_hod_atr_max       — required gate
          * ATTENTION:vwap_slope_positive         — required gate
          * ATTENTION:rvol_1day_min               — required gate
          * ATTENTION:close_above_vwap            — required gate
          * ATTENTION:macd_positive               — required gate
        """
        from technical_analysis.indicators import (
            _add_macd_ip, linreg_r2_slope,
        )
        from config import (
            ATTENTION_DIST_FROM_HOD_ATR_MAX,
            ATTENTION_RVOL_1DAY_MIN,
        )

        entry = self._entry
        ev = entry.evaluation
        if ev is None:
            return

        attn = entry.get_criteria_group("ATTENTION")
        if attn is None:
            return

        # ── 2. ATTENTION gates ───────────────────────────────────────────
        # Use full-session candles for ATR so it's available immediately
        # (not gated on move_start_ts requiring 2 closed bars).
        atr = _atr_from_candles(self._candles_session or self._candles, 14)
        if atr > 0:
            self._last_atr = atr

        # Distance from HOD (attention — is it still in play?)
        # Strict ATR gate — no 10% widening for execution criteria.
        if atr > 0 and entry.daily_high > 0 and self._last_price > 0:
            dist_atrs = (entry.daily_high - self._last_price) / atr
            attn.update("dist_from_hod_atr_max",
                        dist_atrs <= ATTENTION_DIST_FROM_HOD_ATR_MAX,
                        dist_atrs)

        # RVol 1-day (re-evaluate from running volume vs avg)
        if self._avg_vol > 0 and entry.volume > 0:
            live_rvol = entry.volume / self._avg_vol
            if live_rvol > entry.rvol_1day:
                entry.rvol_1day = live_rvol
            attn.update("rvol_1day_min", entry.rvol_1day >= ATTENTION_RVOL_1DAY_MIN, entry.rvol_1day)

        # ── 3. ATTENTION indicators (VWAP, MACD) ────────────────────────
        # Use full-session df for MACD so it's not gated on move_start_ts.
        macd_df = self._df_1min_session if not self._df_1min_session.empty else df_1m
        try:
            macd_df = macd_df.copy()
            _add_macd_ip(macd_df)
        except Exception:
            return

        if macd_df.empty or len(macd_df) < 2:
            return
        last = macd_df.iloc[-1]
        last_close = float(last.get("close", 0.0)) or self._last_price
        macd_hist = float(last.get("macd_hist", 0.0))

        # Use the stream VWAP (full-session-seeded) as authoritative.
        vwap_now = self._stream_vwap if self._stream_vwap > 0 else self._last_vwap

        # ATTENTION: VWAP slope >= ATTENTION_VWAP_SLOPE_MIN ($/min).
        # 10s-sampled session VWAP for fast refresh; falls back to closed 1-min
        # history until the 10s series has warmed up (~30s of stream).
        try:
            vwap_series = list(self._vwap_10s_history)[-6:]
            candles_per_min = 6.0
            if len(vwap_series) < 3:
                vwap_series = list(self._vwap_closed_history)[-6:]
                candles_per_min = 1.0
            if len(vwap_series) >= 3:
                _, slope = linreg_r2_slope(vwap_series)
                slope_per_min = slope * candles_per_min
                attn.update("vwap_slope_positive", slope_per_min >= ATTENTION_VWAP_SLOPE_MIN, slope_per_min)
        except Exception:
            pass

        # ATTENTION: close > VWAP
        if vwap_now > 0:
            self._last_vwap = vwap_now
            attn.update("close_above_vwap", last_close > vwap_now,
                        last_close - vwap_now)

        # ATTENTION: MACD histogram positive
        attn.update("macd_positive", macd_hist > 0, macd_hist)

        # First-MACD-cycle gate state (SIM_FIRST_MACD_CYCLE_ONLY) — reuses
        # the macd_df computed above so live and replay stay in lockstep.
        self._update_first_macd_cycle(macd_df)

    def _update_first_macd_cycle(self, macd_df: pd.DataFrame) -> None:
        """Recompute the first-MACD-cycle entry block from closed 1m bars.

        SIM_FIRST_MACD_CYCLE_ONLY only trades the move's FIRST 1m MACD
        cycle: once the histogram completes its first positive->negative
        flip since move start, ``entry.macd_cycle_done`` marks every later
        leg untradeable (``evaluate_entry`` blocks ALL setup timeframes).

        Derived fresh on every candle refresh as a pure function of
        (session 1m bars, move anchor) rather than latched statefully, so
        a stale cached flag from a pickled entry or a late-detected move
        start self-corrects. The last row is the forming minute under
        live REST polls and replay tick-resamples alike and is excluded:
        a permanent no-more-entries state must not fire off an
        intracandle wiggle — candle close confirms the flip, matching the
        macd_positive gate's closed-bar semantics elsewhere.
        """
        entry = self._entry
        if macd_df.empty or "macd_hist" not in macd_df.columns or "ts" not in macd_df.columns:
            return
        closed = macd_df.iloc[:-1]
        if closed.empty:
            entry.macd_cycle_done = False
            entry.macd_cycle_done_ts = None
            return

        # Live, _refresh_candles re-detects a missing move start every poll;
        # the backtest replay reaches here without that loop, so mirror the
        # lazy recompute for live/replay parity (a session-open anchor would
        # let a premarket pop-and-fade cycle block the real move's first leg).
        if self._move_start_ts is None:
            try:
                move = detect_move_start(macd_df)
                if move is not None:
                    ts = pd.Timestamp(macd_df.iloc[move.start_idx]["ts"]).to_pydatetime()
                    self._move_start_ts = self._coerce_utc_dt(ts)
            except Exception:
                pass

        anchor = self._active_price_action_start(self._latest_df_ts(macd_df))
        ts_vals = pd.to_datetime(closed["ts"], utc=True)
        window = closed.loc[ts_vals >= anchor]
        hist = window["macd_hist"].to_numpy(dtype=float)

        done = False
        done_ts = None
        pos_mask = hist > 0
        if pos_mask.any():
            first_pos = int(pos_mask.argmax())
            neg_after = hist[first_pos + 1:] < 0
            if neg_after.any():
                done = True
                flip_idx = first_pos + 1 + int(neg_after.argmax())
                done_ts = self._coerce_utc_dt(window.iloc[flip_idx]["ts"])
        entry.macd_cycle_done = done
        entry.macd_cycle_done_ts = done_ts

    def _is_10s_pullback_nested_in_1m_pullback(
        self,
        pullback_ts: datetime | pd.Timestamp | None,
    ) -> bool:
        """True when a 10s setup's pullback belongs to a detected 1m pullback."""
        pullback_dt = self._coerce_utc_dt(pullback_ts)
        if pullback_dt is None:
            return False
        df_1m = self._tf_buf.to_df("1min")
        if df_1m is None or df_1m.empty or len(df_1m) < 4:
            return False
        try:
            df_1m = df_1m.copy().sort_index()
            df_1m.index = pd.to_datetime(df_1m.index, utc=True)
            candidates = detect_pullback_pending(df_1m, "1min")
            candidates.extend(detect_pullback_new_high(df_1m, "1min"))
        except Exception:
            return False

        for signal in candidates:
            start = self._coerce_utc_dt(signal.pullback_ts)
            if start is None:
                continue
            candle_count = max(int(signal.pullback_candles or 1), 1)
            end = start + timedelta(minutes=candle_count)
            if start <= pullback_dt < end:
                return True
        return False

    def _on_setup_resolved(
        self,
        timeframe: str,
        quality: float,
        breakout_price: float,
        *,
        pullback_ts: datetime | pd.Timestamp | None = None,
    ) -> None:
        """Called when a pullback breakout fires. Start tracking the continuation leg."""
        # Map pattern timeframe to TF score key
        tf_key = _pattern_tf_to_score_key(timeframe)
        normalized_pullback_ts = self._coerce_utc_dt(pullback_ts)
        if tf_key == "10s" and self._is_10s_pullback_nested_in_1m_pullback(normalized_pullback_ts):
            self._active_legs.pop(tf_key, None)
            self._recompute_tf_setup_score(tf_key)
            logger.debug(
                f"[{self._sym}] setup outcome: skipped nested 10s pullback "
                f"inside 1m pullback (breakout=${breakout_price:.2f})"
            )
            return
        self._active_legs[tf_key] = _ActiveLeg(
            timeframe=tf_key,
            quality=quality,
            breakout_price=breakout_price,
            highest_close=breakout_price,
            pullback_ts=normalized_pullback_ts,
            candle_count=0,
        )
        logger.debug(
            f"[{self._sym}] setup outcome: tracking leg on {tf_key} "
            f"(quality={quality:.2f}, breakout=${breakout_price:.2f})"
        )

    def _on_setup_leg_candle_close(self, closed_tfs: list[str]) -> None:
        """Called on every candle close. Track active legs and finalize on red close."""
        for tf_key in list(self._active_legs):
            buf_tf = _score_key_to_buf_tf(tf_key)
            if buf_tf not in closed_tfs:
                continue
            leg = self._active_legs[tf_key]
            bar = self._tf_buf.last_closed_bar(buf_tf)
            if bar is None:
                continue

            close = float(bar.close)
            open_ = float(bar.open)

            if close >= open_:
                # Green candle — extends the leg
                leg.candle_count += 1
                leg.peak_volume = max(leg.peak_volume, float(bar.volume or 0.0))
                if close > leg.highest_close:
                    leg.highest_close = close
                # Update live per-TF score with current extension
                self._recompute_tf_setup_score(tf_key)
            else:
                # Red candle — leg over, finalize outcome
                self._finalize_leg(tf_key, leg, bar)

        completed_tfs = {o.timeframe for o in self._completed_outcomes}
        for tf_key in completed_tfs:
            if tf_key in self._active_legs:
                continue
            if _score_key_to_buf_tf(tf_key) in closed_tfs:
                self._recompute_tf_setup_score(tf_key)

    def _on_setup_breakout_rejected(self, timeframe: str, failure_bar: OHLCVCandle) -> None:
        """Called when a breakout candle closes red (immediate failure)."""
        tf_key = _pattern_tf_to_score_key(timeframe)
        leg = self._active_legs.pop(tf_key, None)
        pullback_ts = leg.pullback_ts if leg else self._entry.setup_pullback_ts
        if tf_key == "10s" and self._is_10s_pullback_nested_in_1m_pullback(pullback_ts):
            self._recompute_tf_setup_score(tf_key)
            logger.debug(f"[{self._sym}] setup outcome: skipped nested 10s breakout rejection")
            return
        # Record as failure regardless of whether we were tracking a leg
        quality = leg.quality if leg else 0.5
        breakout_price = leg.breakout_price if leg else float(self._breakout_trigger_level or failure_bar.open or failure_bar.close)
        failure_quality, failure_rejection = self._setup_failure_candle_quality(tf_key, failure_bar)
        peak_volume = max(leg.peak_volume if leg else 0.0, float(failure_bar.volume or 0.0))
        self._completed_outcomes.append(_CompletedOutcome(
            timeframe=tf_key,
            quality=quality,
            breakout_price=breakout_price,
            pullback_ts=pullback_ts,
            extension_pct=0.0,
            success=False,
            failure_ts=failure_bar.ts,
            failure_close=float(failure_bar.close),
            failure_quality=failure_quality,
            failure_rejection=failure_rejection,
            mono_ts=time.monotonic(),
            peak_volume=peak_volume,
        ))
        self._recompute_tf_setup_score(tf_key)
        logger.debug(f"[{self._sym}] setup outcome: breakout rejected on {tf_key}")

    def _setup_failure_candle_quality(
        self,
        tf_key: str,
        bar: OHLCVCandle,
    ) -> tuple[float, float]:
        """Return (quality, rejection_risk) for the red candle ending a setup leg.

        Uses the same volume-response candle scoring as ``key_candles``
        (continuous wick/body/range gradients plus red-volume pressure), but
        relative volume acts as a severity modulator instead of an eligibility
        gate: a low-volume red failure candle (healthy pullback behaviour) is
        softened toward SETUP_VOLUME_SEVERITY_FLOOR, while a
        high-volume rejection keeps full structural severity.
        """
        buf_tf = _score_key_to_buf_tf(tf_key)
        df = self._tf_buf.to_df(buf_tf)
        if df is None or df.empty:
            return 1.0, 0.0
        if self._tf_buf.current_bar(buf_tf) is not None and len(df) >= 1:
            df = df.iloc[:-1]
        if len(df) < 2 or "volume" not in df:
            return 1.0, 0.0

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        atr = _atr_from_arrays(highs, lows, closes, len(highs) - 1, TOPPING_TAIL_ATR_PERIOD)
        if atr <= 0:
            return 1.0, 0.0
        baseline = _volume_response_baseline(df["volume"].to_numpy(dtype=float))
        if baseline is None:
            return 1.0, 0.0
        volume_floor, volume_full = baseline

        metrics = _candle_volume_response(
            float(bar.open),
            float(bar.high),
            float(bar.low),
            float(bar.close),
            float(bar.volume or 0.0),
            atr,
            volume_floor,
            volume_full,
        )
        if metrics is None:
            return 1.0, 0.0
        volume_weight, structural_risk, red_volume_risk = metrics

        floor = max(0.0, min(1.0, SETUP_VOLUME_SEVERITY_FLOOR))
        severity_scale = floor + ((1.0 - floor) * volume_weight)
        risk = min(
            1.0,
            1.0 - ((1.0 - (structural_risk * severity_scale)) * (1.0 - red_volume_risk)),
        )
        return max(0.0, 1.0 - risk), risk

    def _finalize_leg(self, tf_key: str, leg: _ActiveLeg, failure_bar: OHLCVCandle) -> None:
        """End an active leg and record the outcome."""
        del self._active_legs[tf_key]
        ext = leg.extension_pct
        success = ext >= SETUP_OUTCOME_SUCCESS_PCT
        failure_quality, failure_rejection = self._setup_failure_candle_quality(tf_key, failure_bar)
        self._completed_outcomes.append(_CompletedOutcome(
            timeframe=tf_key,
            quality=leg.quality,
            breakout_price=leg.breakout_price,
            pullback_ts=leg.pullback_ts,
            extension_pct=ext,
            success=success,
            failure_ts=failure_bar.ts,
            failure_close=float(failure_bar.close),
            failure_quality=failure_quality,
            failure_rejection=failure_rejection,
            mono_ts=time.monotonic(),
            peak_volume=max(leg.peak_volume, float(failure_bar.volume or 0.0)),
        ))
        self._recompute_tf_setup_score(tf_key)
        logger.debug(
            f"[{self._sym}] setup outcome: leg ended on {tf_key} "
            f"ext={ext:.1%} {'SUCCESS' if success else 'FAIL'}"
        )

    def _setup_recovery_credit(self, outcome: _CompletedOutcome) -> float:
        """Partial repair credit after a failed setup reclaims and holds."""
        if outcome.success or outcome.breakout_price <= 0:
            return 0.0

        buf_tf = _score_key_to_buf_tf(outcome.timeframe)
        df = self._tf_buf.to_df(buf_tf)
        if df is None or df.empty:
            return 0.0
        if self._tf_buf.current_bar(buf_tf) is not None and len(df) >= 1:
            df = df.iloc[:-1]
        if df.empty:
            return 0.0

        if "ts" in df.columns:
            ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(df.index, utc=True, errors="coerce")
        failure_ts = pd.Timestamp(outcome.failure_ts)
        if failure_ts.tzinfo is None:
            failure_ts = failure_ts.tz_localize(timezone.utc)
        else:
            failure_ts = failure_ts.tz_convert(timezone.utc)
        post = df.loc[ts > failure_ts]
        if post.empty:
            return 0.0

        breakout = float(outcome.breakout_price)
        closes = post["close"].astype(float)
        highest_close = float(closes.max())
        last_close = float(closes.iloc[-1])
        if highest_close <= breakout:
            return 0.0

        reclaim_progress = min(
            max((highest_close - breakout) / (breakout * SETUP_OUTCOME_SUCCESS_PCT), 0.0),
            1.0,
        )
        hold_quality = 1.0 if last_close >= breakout else 0.0
        if hold_quality == 0.0 and breakout > outcome.failure_close:
            hold_quality = min(
                max((last_close - outcome.failure_close) / (breakout - outcome.failure_close), 0.0),
                1.0,
            )

        session_high = float(df["high"].astype(float).max())
        session_low = float(df["low"].astype(float).min())
        if session_high > session_low:
            range_pos = min(max((last_close - session_low) / (session_high - session_low), 0.0), 1.0)
        else:
            range_pos = 0.5

        repair_quality = (0.35 * reclaim_progress) + (0.35 * hold_quality) + (0.30 * range_pos)
        return SETUP_OUTCOME_RECOVERY_MAX_CREDIT * min(max(repair_quality, 0.0), 1.0)

    def _setup_volume_anchors(self, tf_key: str) -> tuple[float, float] | None:
        """Current-window relative-volume anchors for leg significance.

        Same (median floor, top-percentile full) anchors key_candles uses,
        recomputed from the CURRENT closed candles of the timeframe on every
        rescore — so a leg's significance is always judged against what volume
        looks like now, not what it looked like when the leg printed.
        """
        buf_tf = _score_key_to_buf_tf(tf_key)
        df = self._tf_buf.to_df(buf_tf)
        if df is None or df.empty or "volume" not in df:
            return None
        if self._tf_buf.current_bar(buf_tf) is not None and len(df) >= 1:
            df = df.iloc[:-1]
        if df.empty:
            return None
        return _volume_response_baseline(df["volume"].to_numpy(dtype=float))

    @staticmethod
    def _leg_volume_weight(peak_volume: float, anchors: tuple[float, float] | None) -> float:
        """Evidence weight (0-1) of a leg from its peak closed-candle volume.

        The same smoothstep key_candles applies per candle: at/below the
        current median a leg carries no evidence; at/above the top-percentile
        anchor it carries full evidence. ``anchors=None`` (no positive-volume
        candles in the window) leaves the leg ungated — absence of volume data
        must not erase outcomes.
        """
        if anchors is None:
            return 1.0
        if not math.isfinite(peak_volume) or peak_volume <= 0.0:
            return 0.0
        volume_floor, volume_full = anchors
        return _smoothstep(
            (peak_volume - volume_floor) / max(1e-9, volume_full - volume_floor)
        )

    def _recompute_tf_setup_score(self, tf_key: str) -> None:
        """Recompute the per-TF setup outcome score and refresh TA override."""
        outcomes = [o for o in self._completed_outcomes if o.timeframe == tf_key]
        active = self._active_legs.get(tf_key)

        if not outcomes and active is None:
            # No setups for this TF — exclude it entirely (do nothing) rather
            # than contributing a neutral 0.5.
            self._tf_pa_scores.pop(tf_key, None)
            self._tf_pa_masses.pop(tf_key, None)
            self._tf_pa_var_qualities.setdefault(tf_key, {}).pop("setup_outcome", None)
            self._refresh_technical_analysis_score()
            return

        anchors = self._setup_volume_anchors(tf_key)
        # (recency_weight, volume_weight, score) per leg
        contributions: list[tuple[float, float, float]] = []

        # Completed outcomes with exponential decay (most recent = weight 1.0)
        for i, o in enumerate(reversed(outcomes)):
            weight = SETUP_OUTCOME_DECAY ** i
            extension_credit = min(max(o.extension_pct, 0.0) / SETUP_OUTCOME_SUCCESS_PCT, 1.0)
            failure_quality = 1.0 if o.success else max(0.0, min(1.0, o.failure_quality))
            if o.success:
                score_factor = extension_credit
            else:
                severity_adjusted_recovery = self._setup_recovery_credit(o) * (0.5 + (0.5 * failure_quality))
                score_factor = max(extension_credit * failure_quality, severity_adjusted_recovery)
            score = o.quality * score_factor
            contributions.append((weight, self._leg_volume_weight(o.peak_volume, anchors), score))

        # Active leg contributes at full weight (most recent event)
        if active is not None and active.candle_count > 0:
            ext = active.extension_pct
            if ext > 0:
                score = active.quality * min(ext / SETUP_OUTCOME_SUCCESS_PCT, 1.0)
                contributions.append((1.0, self._leg_volume_weight(active.peak_volume, anchors), score))

        if not contributions:
            # An active leg with no extension yet — nothing to score, exclude it.
            self._tf_pa_scores.pop(tf_key, None)
            self._tf_pa_masses.pop(tf_key, None)
            self._tf_pa_var_qualities.setdefault(tf_key, {}).pop("setup_outcome", None)
            self._refresh_technical_analysis_score()
            return

        # Volume-significant blend: like key_candles, only legs whose peak
        # candle volume clears the current median->top-percentile band carry
        # evidence. The evidence mass scales setup_outcome's weight in the TA
        # blend; when every leg is volume-insignificant the score is still
        # published for display (unweighted), but with ~0 mass it drops out of
        # the live confidence.
        vw_sum = sum(rw * vw for rw, vw, _ in contributions)
        if vw_sum > 0.0:
            tf_score = sum(rw * vw * s for rw, vw, s in contributions) / vw_sum
        else:
            w_sum = sum(rw for rw, _, _ in contributions)
            tf_score = (
                sum(rw * s for rw, _, s in contributions) / w_sum if w_sum > 0 else 0.5
            )

        self._tf_pa_scores[tf_key] = tf_score
        self._tf_pa_masses[tf_key] = min(1.0, vw_sum)
        self._tf_pa_var_qualities.setdefault(tf_key, {})["setup_outcome"] = tf_score
        self._refresh_technical_analysis_score()


# ---------------------------------------------------------------------------
# Standalone helper — spread limit by price tier
# ---------------------------------------------------------------------------

class _EventHold:
    """Context manager behind DynamicScorer._event_hold.

    Used as ``with scorer._event_hold(active), scorer._lock:`` — Python
    enters the hold first and exits it LAST, so the drain in ``__exit__``
    runs after the scorer lock is released and the replayed handlers can
    take the lock normally. Exceptions inside the block still lift the
    hold, so a failed replay can never leave a symbol's stream parked.
    """

    __slots__ = ("_scorer", "_active")

    def __init__(self, scorer: "DynamicScorer", active: bool) -> None:
        self._scorer = scorer
        self._active = active

    def __enter__(self) -> "_EventHold":
        if self._active:
            self._scorer._begin_event_hold()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._active:
            self._scorer._release_held_events()
        return False


def _spread_limit(price: float) -> float:
    if price < 1.0:   return SPREAD_MAX_UNDER_1
    if price < 5.0:   return SPREAD_MAX_1_TO_5
    if price < 10.0:  return SPREAD_MAX_5_TO_10
    return SPREAD_MAX_10_TO_20


# ---------------------------------------------------------------------------
# TF key mapping: pattern timeframe names ↔ score dict keys ↔ buffer TF names
# ---------------------------------------------------------------------------

_PATTERN_TF_TO_SCORE_KEY: dict[str, str] = {
    "10sec": "10s", "1min": "1m", "2min": "2m", "5min": "5m",
    "10s": "10s", "1m": "1m", "2m": "2m", "5m": "5m",
}
_SCORE_KEY_TO_BUF_TF: dict[str, str] = {
    "10s": "10sec", "1m": "1min", "2m": "2min", "5m": "5min",
}


def _pattern_tf_to_score_key(tf: str) -> str:
    return _PATTERN_TF_TO_SCORE_KEY.get(tf, "1m")


def _score_key_to_buf_tf(key: str) -> str:
    return _SCORE_KEY_TO_BUF_TF.get(key, "1min")


def _atr_from_candles(candles: list[OHLCVCandle], period: int) -> float:
    """Compute ATR from OHLCV candles using true range and a simple mean."""
    if len(candles) < 2:
        return 0.0

    tr_values: list[float] = []
    prev_close = float(candles[0].close)
    for c in candles:
        high = float(c.high)
        low = float(c.low)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        tr_values.append(tr)
        prev_close = float(c.close)

    lookback = min(max(1, period), len(tr_values))
    if lookback <= 0:
        return 0.0
    return sum(tr_values[-lookback:]) / lookback


def _rolling_atr_values(candles: list[OHLCVCandle], period: int) -> list[float]:
    """Compute the ATR available at each candle using only prior/current candles."""
    if not candles:
        return []

    lookback = max(1, period)
    tr_window: deque[float] = deque(maxlen=lookback)
    tr_sum = 0.0
    out: list[float] = []
    prev_close = float(candles[0].close)
    for c in candles:
        high = float(c.high)
        low = float(c.low)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        if len(tr_window) == lookback:
            tr_sum -= tr_window[0]
        tr_window.append(tr)
        tr_sum += tr
        out.append(tr_sum / len(tr_window))
        prev_close = float(c.close)
    return out


def _candle_choppiness_ratio(opens, highs, closes) -> float | None:
    """Fraction of candle-color flips (red<->green) that are NOT part of a
    succeeded OR still-unresolved pullback.

    A pullback is a maximal run of red candles.  It is "succeeded" when a green
    candle in the following recovery leg CLOSES above the pre-pullback peak high
    (the pullback entry price) before the next pullback begins — in that case
    its ``G->R`` and ``R->G`` flips are excused, so a clean staircase uptrend
    with healthy pullbacks does not read as choppy.

    A pullback only COUNTS as chop once it has objectively failed: the next red
    run began without the peak being reclaimed.  The trailing pullback of the
    window — the currently active one, whose red run or recovery leg reaches the
    last closed candle without resolution — is excused too: judging it as chop
    would penalize live confidence during the exact window where a pullback
    entry is being evaluated, before the outcome is knowable.

    Returns ``choppy_flips / (n - 1)`` in ``[0, 1]`` (0 = clean, 1 = fully
    alternating), or ``None`` when there are fewer than 2 candles.
    """
    n = len(closes)
    if n < 2:
        return None
    greens = [float(closes[i]) >= float(opens[i]) for i in range(n)]
    highs = [float(h) for h in highs]
    closes = [float(c) for c in closes]

    flips = 0
    excused = 0
    running_peak = float("-inf")
    neg_inf = float("-inf")
    i = 0
    while i < n:
        if not greens[i]:
            # Start of a maximal red run at index a.
            a = i
            peak_before = running_peak  # highest high of candles [0 .. a-1]
            b = a
            while b < n and not greens[b]:
                if highs[b] > running_peak:
                    running_peak = highs[b]
                b += 1
            down = 1 if a > 0 else 0        # G->R (only if a green preceded)
            up = 1 if b < n else 0          # R->G (only if a green follows)
            flips += down + up
            # Succeeded pullback: a green in the recovery leg closes over the
            # pullback entry price (the pre-pullback peak high).
            succeeded = False
            c = b
            while c < n and greens[c]:
                if peak_before != neg_inf and closes[c] > peak_before:
                    succeeded = True
                if highs[c] > running_peak:
                    running_peak = highs[c]
                c += 1
            # Unresolved at the window's edge = the currently active pullback:
            # its red run (b == n) or recovery leg (c == n) reached the last
            # closed candle without the outcome being decided.
            pending = b == n or (c == n and not succeeded)
            if succeeded or pending:
                excused += down + up
            i = c
        else:
            if highs[i] > running_peak:
                running_peak = highs[i]
            i += 1

    denom = n - 1
    if denom <= 0:
        return 0.0
    return (flips - excused) / denom
