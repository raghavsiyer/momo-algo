"""
technical_analysis/patterns.py

Live pattern detection for intraday momentum trading.

PATTERNS DETECTED:
    - Pullback new-high setup (non-resistance continuation)
    - Topping tail / Bottoming tail (single candle + cluster detection)
    - Multi-top (double, triple, N-top) / Multi-bottom
    - Breakout pullback (post-break retest confirmed as support)
    - Resistance pullback-then-break (pre-break: reject → pullback → break)
    - Break + retest (break level → pullback to level → bounce)
    - Rising wedge (bearish) / Falling wedge (bullish)
    - Head and shoulders / Inverse head and shoulders
    - Support bounce / Resistance rejection (from ResistanceMap levels +
      dynamic indicators: EMA 9/20, VWAP, MA 50)

DESIGN:
    All detection functions are pure — they take DataFrames and/or a
    ResistanceMap and return PatternSignal lists.  No mutation of inputs.

    The single entry point ``scan_all_patterns()`` runs all detectors
    across four timeframes (10sec, 1min, 2min, 5min) and deduplicates,
    always preferring higher-timeframe detections.

    Detection is event-driven: called from DynamicScorer._on_timesale
    whenever a stream-built candle closes (every 10s at minimum).

INTEGRATION:
    ResistanceMap provides all support/resistance levels (static, structural,
    dynamic, broken-as-support).  Indicator DataFrames provide EMA, VWAP, etc.
    Nothing is hardcoded — all levels come from the map or computed columns.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    PATTERN_PIVOT_N_10SEC,
    PATTERN_PIVOT_N_1MIN,
    PATTERN_PIVOT_N_2MIN,
    PATTERN_PIVOT_N_5MIN,
    PATTERN_TOP_PRICE_TOL_PCT,
    PATTERN_TOP_MIN_PULLBACK,
    PATTERN_WEDGE_MIN_TOUCHES,
    PATTERN_WEDGE_MIN_BARS,
    PATTERN_WEDGE_CONVERGE_MAX,
    PATTERN_HNS_SHOULDER_TOL_PCT,
    PATTERN_HNS_HEAD_MIN_EXCEED,
    PATTERN_RETEST_BUFFER_PCT,
    PATTERN_RETEST_MAX_BARS,
    PATTERN_RETEST_BOUNCE_BARS,
    PATTERN_BOUNCE_PROXIMITY_PCT,
    PATTERN_BOUNCE_CONFIRM_BARS,
    PATTERN_TAIL_CLUSTER_BARS,
    PATTERN_TAIL_CLUSTER_MIN,
    PATTERN_TAIL_CLUSTER_TOL_PCT,
    PATTERN_PULLBACK_MAX_DEPTH,
    PATTERN_PULLBACK_MIN_LEG_PCT,
    PATTERN_PULLBACK_MIN_LEG_ATR_MULT,
    PATTERN_PULLBACK_MIN_LEG_CANDLES,
    PATTERN_PULLBACK_MAX_LEG_CANDLES,
    PATTERN_PULLBACK_MAX_CANDLES,
    PATTERN_PULLBACK_REQUIRE_HOD,
    SETUP_TFS_DISABLED,
    TOPPING_TAIL_ATR_PERIOD,
    PATTERN_MIN_RAW,
    PATTERN_HOD_PROXIMITY,
    TOPPING_TAIL_RATIO,
    VOLUME_RESPONSE_TOP_VOLUME_PCT,
    VOLUME_RESPONSE_RANGE_ATR_MIN,
    VOLUME_RESPONSE_RANGE_ATR_FULL,
    VOLUME_RESPONSE_WICK_ATR_MIN,
    VOLUME_RESPONSE_WICK_ATR_FULL,
    VOLUME_RESPONSE_WICK_RANGE_MIN,
    VOLUME_RESPONSE_WICK_RANGE_FULL,
    VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN,
    VOLUME_RESPONSE_RED_VOLUME_RATIO_FULL,
    VOLUME_RESPONSE_RED_VOLUME_MAX_RISK,
    SETUP_VOLUME_SEVERITY_FLOOR,
    MarketClock,
    _MarketCalendar,
)

logger = logging.getLogger(__name__)

# Higher rank = higher timeframe = preferred in deduplication
_TIMEFRAME_RANK: dict[str, int] = {
    "10sec": 0,
    "1min": 3,
    "2min": 1,
    "5min": 2,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatternSignal:
    """A single detected pattern."""
    name: str               # e.g. "double_top", "rising_wedge"
    bias: str               # "bullish" | "bearish"
    strength: float         # 0.0 – 1.0 (how textbook / reliable)
    price_level: float      # key price the pattern orbits
    timeframe: str          # "1min" | "5min"
    candle_count: int       # how many candles the pattern spans
    detail: str = ""        # human-readable description
    pullback_depth: Optional[float] = None  # raw depth ratio [0, 1], if applicable
    pullback_candles: Optional[int] = None
    pullback_control: Optional[float] = None  # 0-1, higher = shallow/short pullback
    leg_candles: Optional[int] = None
    pullback_structure: Optional[float] = None  # 0-1, higher = cleaner
    breakout_extension: Optional[float] = None  # fraction of leg size
    topping_tail: Optional[float] = 0.0          # volume-response topping-tail risk (0-1) in setup segment
    jackknife: Optional[float] = 0.0             # volume-response red-body risk (0-1) in setup segment
    pullback_ts: Optional[datetime] = None      # timestamp of first pullback candle
    pullback_low: Optional[float] = None        # absolute low of the pullback window
                                                # (armed-band entry floor anchor)

    def __repr__(self) -> str:
        arrow = "▲" if self.bias == "bullish" else "▼"
        return (
            f"{arrow} {self.name} @ ${self.price_level:.2f} "
            f"({self.timeframe}, {self.candle_count} bars, "
            f"str={self.strength:.0%})"
        )


@dataclass
class PatternSnapshot:
    """Collection of all active patterns at a point in time."""
    signals: list[PatternSignal] = field(default_factory=list)

    @property
    def strongest_bearish(self) -> Optional[PatternSignal]:
        bears = [s for s in self.signals if s.bias == "bearish"]
        return max(bears, key=lambda s: s.strength) if bears else None

    @property
    def strongest_bullish(self) -> Optional[PatternSignal]:
        bulls = [s for s in self.signals if s.bias == "bullish"]
        return max(bulls, key=lambda s: s.strength) if bulls else None

    @property
    def bearish_count(self) -> int:
        return sum(1 for s in self.signals if s.bias == "bearish")

    @property
    def bullish_count(self) -> int:
        return sum(1 for s in self.signals if s.bias == "bullish")

    @property
    def net_bias(self) -> float:
        """Positive = net bullish, negative = net bearish."""
        return sum(
            s.strength if s.bias == "bullish" else -s.strength
            for s in self.signals
        )


@dataclass
class MoveStart:
    """Location of the bar where the current price move began.

    A "move" is defined as the most recent contiguous run of
    elevated-volume bars ending at (or within ``allow_gaps`` of) the
    latest bar.  Pure structural info — no bias / no strength score.
    """
    start_idx: int             # absolute bar index in the source DataFrame
    bars_since_start: int      # how many bars between start and latest
    start_price: float         # close at start_idx
    current_price: float       # close at last bar
    price_change_pct: float    # (current - start) / start * 100
    direction: str             # "up" | "down" | "flat"
    volume_ratio: float        # peak volume in the run / baseline volume
    baseline_volume: float     # median pre-move volume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_pivot_highs(
    highs: np.ndarray,
    n: int,
) -> list[tuple[int, float]]:
    """Return list of (index, price) for pivot highs."""
    length = len(highs)
    if length < 2 * n + 1:
        return []
    is_pivot = np.ones(length, dtype=bool)
    is_pivot[:n] = False
    is_pivot[length - n:] = False
    for offset in range(1, n + 1):
        is_pivot[n:length - n] &= highs[n:length - n] > highs[n - offset:length - n - offset]
        is_pivot[n:length - n] &= highs[n:length - n] > highs[n + offset:length - n + offset]
    indices = np.where(is_pivot)[0]
    return [(int(i), float(highs[i])) for i in indices]


def _find_pivot_lows(
    lows: np.ndarray,
    n: int,
) -> list[tuple[int, float]]:
    """Return list of (index, price) for pivot lows."""
    length = len(lows)
    if length < 2 * n + 1:
        return []
    is_pivot = np.ones(length, dtype=bool)
    is_pivot[:n] = False
    is_pivot[length - n:] = False
    for offset in range(1, n + 1):
        is_pivot[n:length - n] &= lows[n:length - n] < lows[n - offset:length - n - offset]
        is_pivot[n:length - n] &= lows[n:length - n] < lows[n + offset:length - n + offset]
    indices = np.where(is_pivot)[0]
    return [(int(i), float(lows[i])) for i in indices]


def _cluster_prices(
    prices: list[tuple[int, float]],
    tol_pct: float,
) -> list[list[tuple[int, float]]]:
    """
    Group (index, price) tuples into clusters where prices are within
    tol_pct of the cluster's first price.
    """
    if not prices:
        return []
    sorted_pts = sorted(prices, key=lambda x: x[1])
    clusters: list[list[tuple[int, float]]] = [[sorted_pts[0]]]
    for pt in sorted_pts[1:]:
        ref = clusters[-1][0][1]
        if abs(pt[1] - ref) / ref <= tol_pct:
            clusters[-1].append(pt)
        else:
            clusters.append([pt])
    return clusters


def _get_support_levels(
    rmap,
    df: pd.DataFrame,
    current_price: float,
) -> list[tuple[float, str]]:
    """
    Collect all support levels from ResistanceMap + indicator columns.
    Returns list of (price, source_label).
    Does NOT hardcode any levels — everything comes from the map or
    computed indicator columns in the DataFrame.
    """
    levels: list[tuple[float, str]] = []

    # Broken resistance levels → now support
    for lv in rmap.broken_levels():
        if lv.price < current_price:
            levels.append((lv.price, f"broken {lv.source}"))

    # Active levels below price (structural support: pivot lows, prev day low,
    # static dollar levels, etc.)
    for lv in rmap.levels_below(current_price, limit=10):
        levels.append((lv.price, lv.source))

    # Dynamic indicator levels from the last candle
    if not df.empty:
        last = df.iloc[-1]
        for col, label in [
            ("ema_9",  "EMA 9"),
            ("ema_20", "EMA 20"),
            ("vwap",   "VWAP"),
            ("ma_50",  "MA 50"),
            ("ema_200", "EMA 200"),
            ("ma_200",  "MA 200"),
            ("bb_lower", "BB lower"),
        ]:
            if col in df.columns:
                val = last.get(col)
                if pd.notna(val):
                    fval = float(val)
                    if fval < current_price:
                        levels.append((fval, label))

    return levels


def _get_resistance_levels(
    rmap,
    df: pd.DataFrame,
    current_price: float,
) -> list[tuple[float, str]]:
    """
    Collect all resistance levels from ResistanceMap + indicator columns.
    Returns list of (price, source_label).
    """
    levels: list[tuple[float, str]] = []

    for lv in rmap.levels_above(current_price, limit=10):
        levels.append((lv.price, lv.source))

    if not df.empty:
        last = df.iloc[-1]
        for col, label in [
            ("ema_9",  "EMA 9"),
            ("ema_20", "EMA 20"),
            ("vwap",   "VWAP"),
            ("ma_50",  "MA 50"),
            ("ema_200", "EMA 200"),
            ("ma_200",  "MA 200"),
            ("bb_upper", "BB upper"),
        ]:
            if col in df.columns:
                val = last.get(col)
                if pd.notna(val):
                    fval = float(val)
                    if fval > current_price:
                        levels.append((fval, label))

    return levels


def _row_timestamp(df: pd.DataFrame, row_idx: int) -> Optional[datetime]:
    """Return a real timestamp for a row, never a positional RangeIndex value."""
    if row_idx < 0 or row_idx >= len(df):
        return None
    raw = df.iloc[row_idx]["ts"] if "ts" in df.columns else df.index[row_idx]
    if raw is None or pd.isna(raw):
        return None
    if hasattr(raw, "to_pydatetime"):
        return raw.to_pydatetime()
    if isinstance(raw, datetime):
        return raw
    return None


def _pullback_control_components(pullback_depth: float, pullback_len: int) -> tuple[float, float, float]:
    depth_span = PATTERN_PULLBACK_MAX_DEPTH
    depth_pos = 0.0 if depth_span <= 0 else pullback_depth / depth_span
    depth_quality = 1.0 - max(0.0, min(1.0, depth_pos))
    max_bars = float(PATTERN_PULLBACK_MAX_CANDLES)
    pullback_len_quality = max(
        0.0,
        1.0 - min((pullback_len - 1) / max(max_bars, 1.0), 1.0),
    )
    control = (depth_quality + pullback_len_quality) / 2.0
    return depth_quality, pullback_len_quality, max(0.0, min(1.0, control))


def _atr_from_arrays(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    end_idx: int,
    period: int,
) -> float:
    """Compute ATR through ``end_idx`` using true range and a simple mean."""
    if end_idx <= 0:
        return 0.0
    start_idx = max(0, end_idx - max(1, period) + 1)
    tr_values: list[float] = []
    for idx in range(start_idx, end_idx + 1):
        prev_close = float(closes[idx - 1]) if idx > 0 else float(closes[idx])
        high = float(highs[idx])
        low = float(lows[idx])
        tr_values.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return sum(tr_values) / len(tr_values) if tr_values else 0.0


# ---------------------------------------------------------------------------
# Volume-response candle scoring (shared by key_candles, setup-outcome
# failure scoring, and the setup-segment rejection metrics)
# ---------------------------------------------------------------------------


def _smoothstep(raw: float) -> float:
    x = max(0.0, min(1.0, raw))
    return x * x * (3.0 - (2.0 * x))


def _volume_response_baseline(volumes) -> tuple[float, float] | None:
    """Return (volume_floor, volume_full) anchors for relative-volume weighting.

    ``volume_floor`` is the median positive candle volume; ``volume_full`` is
    the top-percentile anchor (at least 2x the floor).  Returns None when the
    window has no positive volume evidence.
    """
    positive_volumes: list[float] = []
    for volume_raw in volumes:
        volume = float(volume_raw)
        if math.isfinite(volume) and volume > 0.0:
            positive_volumes.append(volume)
    if not positive_volumes:
        return None

    positive_volumes.sort()
    mid = len(positive_volumes) // 2
    if len(positive_volumes) % 2:
        median_volume = positive_volumes[mid]
    else:
        median_volume = (positive_volumes[mid - 1] + positive_volumes[mid]) / 2.0
    pct = min(1.0, max(0.0, VOLUME_RESPONSE_TOP_VOLUME_PCT))
    top_count = max(1, math.ceil(len(positive_volumes) * pct)) if pct > 0.0 else 1
    volume_anchor = positive_volumes[-top_count]
    volume_floor = max(median_volume, 1.0)
    volume_full = max(volume_anchor, volume_floor * 2.0)
    if volume_full <= volume_floor:
        volume_full = volume_floor * 2.0
    return volume_floor, volume_full


def _red_volume_pressure(volume_ratio: float) -> float:
    """Pressure risk added by a red close on elevated relative volume."""
    intensity = _smoothstep(
        (volume_ratio - VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN)
        / max(
            1e-9,
            VOLUME_RESPONSE_RED_VOLUME_RATIO_FULL - VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN,
        )
    )
    return intensity * max(0.0, min(1.0, VOLUME_RESPONSE_RED_VOLUME_MAX_RISK))


def _candle_rejection_components(
    open_: float,
    high: float,
    low: float,
    close: float,
    atr: float,
) -> tuple[float, float, float] | None:
    """Structural rejection gradients for one closed candle.

    Returns ``(tail_risk, body_risk, range_weight)`` or None when the candle
    has no usable ATR/range evidence.
    """
    candle_range = max(0.0, high - low)
    if atr <= 0.0 or candle_range <= 0.0:
        return None

    range_atr = candle_range / atr
    range_weight = _smoothstep(
        (range_atr - VOLUME_RESPONSE_RANGE_ATR_MIN)
        / max(1e-9, VOLUME_RESPONSE_RANGE_ATR_FULL - VOLUME_RESPONSE_RANGE_ATR_MIN)
    )

    upper_wick = max(0.0, high - max(open_, close))
    upper_wick_atr = upper_wick / atr
    upper_wick_share = upper_wick / candle_range
    tail_size = _smoothstep(
        (upper_wick_atr - VOLUME_RESPONSE_WICK_ATR_MIN)
        / max(1e-9, VOLUME_RESPONSE_WICK_ATR_FULL - VOLUME_RESPONSE_WICK_ATR_MIN)
    )
    tail_shape = _smoothstep(
        (upper_wick_share - VOLUME_RESPONSE_WICK_RANGE_MIN)
        / max(1e-9, VOLUME_RESPONSE_WICK_RANGE_FULL - VOLUME_RESPONSE_WICK_RANGE_MIN)
    )
    tail_risk = tail_size * tail_shape

    red_body = max(0.0, open_ - close)
    red_body_atr = red_body / atr
    red_body_share = red_body / candle_range
    close_location = (close - low) / candle_range
    body_size = _smoothstep(
        (red_body_atr - PATTERN_MIN_RAW) / max(1e-9, 0.85 - PATTERN_MIN_RAW)
    )
    body_shape = _smoothstep((red_body_share - 0.25) / max(1e-9, 0.70 - 0.25))
    close_low = _smoothstep(((1.0 - close_location) - 0.45) / max(1e-9, 0.90 - 0.45))
    body_risk = body_size * max(body_shape, close_low)

    return tail_risk, body_risk, range_weight


def _candle_volume_response(
    open_: float,
    high: float,
    low: float,
    close: float,
    vol: float,
    atr: float,
    volume_floor: float,
    volume_full: float,
) -> tuple[float, float, float] | None:
    """Volume-response metrics for a single closed candle.

    Returns ``(volume_weight, structural_risk, red_volume_risk)`` or None when
    the candle has no usable volume/ATR/range evidence.
    """
    if not math.isfinite(vol) or vol <= 0.0:
        return None
    components = _candle_rejection_components(open_, high, low, close, atr)
    if components is None:
        return None
    tail_risk, body_risk, range_weight = components

    volume_weight = _smoothstep(
        (vol - volume_floor) / max(1e-9, volume_full - volume_floor)
    )
    volume_ratio = vol / volume_floor if volume_floor > 0.0 else 0.0
    structural_risk = min(1.0, max(tail_risk, body_risk) * range_weight)
    red_volume_risk = _red_volume_pressure(volume_ratio) if close < open_ else 0.0
    return volume_weight, structural_risk, red_volume_risk


def _setup_timeframe_is_daily(df: pd.DataFrame, timeframe: str) -> bool:
    tf = timeframe.lower().replace("_", "").replace("-", "")
    if tf in _DAILY_INTERVALS:
        return True
    if tf in _INTRADAY_INTERVALS:
        # The caller already knows the bar size — never fall back to
        # inspecting the frame for a recognised intraday label. This matters
        # because detect_pullback_detected runs on EVERY print in the live
        # scorer, and the frame inspection (to_datetime + sort + diff over
        # the whole frame) was ~1 ms per call: measured 2026-09-03 as the
        # single largest consumer of stream-thread CPU, about a third of all
        # per-print work. Unknown labels still fall through to the
        # inspection so offline callers with odd labels keep working.
        return False
    return _is_daily_chart(df)


def _setup_pullback_start(
    opens: np.ndarray,
    closes: np.ndarray,
    pullback_end: int,
) -> int | None:
    if pullback_end < 0 or closes[pullback_end] >= opens[pullback_end]:
        return None

    pullback_start = pullback_end
    while pullback_start - 1 >= 0 and closes[pullback_start - 1] < opens[pullback_start - 1]:
        pullback_start -= 1
    return pullback_start


def _setup_green_leg(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    leg_end: int,
) -> tuple[int, int, float, float, float] | None:
    max_len = max(PATTERN_PULLBACK_MIN_LEG_CANDLES, PATTERN_PULLBACK_MAX_LEG_CANDLES)
    longest_len = min(max_len, leg_end + 1)

    for leg_len in range(longest_len, PATTERN_PULLBACK_MIN_LEG_CANDLES - 1, -1):
        leg_start = leg_end - leg_len + 1
        if leg_start < 0:
            continue
        if any(closes[idx] <= opens[idx] for idx in range(leg_start, leg_end + 1)):
            continue
        leg_open = float(opens[leg_start])
        if leg_open <= 0:
            continue
        leg_high_slice = highs[leg_start:leg_end + 1]
        leg_high_idx = int(leg_start + np.argmax(leg_high_slice))
        leg_high = float(leg_high_slice[leg_high_idx - leg_start])
        leg_size = leg_high - leg_open
        if leg_size <= 0:
            continue
        if leg_size / leg_open < PATTERN_PULLBACK_MIN_LEG_PCT:
            continue

        atr = _atr_from_arrays(highs, lows, closes, leg_end, TOPPING_TAIL_ATR_PERIOD)
        if atr > 0 and leg_size < PATTERN_PULLBACK_MIN_LEG_ATR_MULT * atr:
            continue

        # Leg must have reached session HOD — reject sub-HOD setups.
        # Gated by PATTERN_PULLBACK_REQUIRE_HOD so sub-HOD continuation
        # pullbacks can be taken when the requirement is turned off.
        if PATTERN_PULLBACK_REQUIRE_HOD:
            session_hod = float(np.max(highs[:leg_end + 1]))
            if session_hod > 0 and leg_high < session_hod * PATTERN_HOD_PROXIMITY:
                continue

        return leg_start, leg_high_idx, leg_open, leg_high, leg_size

    return None


def _setup_extra_green_transition(
    opens: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    transition_idx: int,
    pullback_end: int,
) -> bool:
    return (
        transition_idx > pullback_end
        and closes[transition_idx] > opens[transition_idx]
        and highs[transition_idx] <= highs[pullback_end]
    )


def _resolved_pullback_window(
    opens: np.ndarray,
    highs: np.ndarray,
    closes: np.ndarray,
    breakout_idx: int,
) -> tuple[int, int, float] | None:
    pullback_end = breakout_idx - 1
    pullback_start = _setup_pullback_start(opens, closes, pullback_end)
    if pullback_start is not None:
        breakout_level = float(highs[pullback_end])
        if highs[breakout_idx] > breakout_level:
            return pullback_start, pullback_end, breakout_level

    transition_idx = breakout_idx - 1
    pullback_end = breakout_idx - 2
    pullback_start = _setup_pullback_start(opens, closes, pullback_end)
    if pullback_start is None:
        return None
    if not _setup_extra_green_transition(opens, highs, closes, transition_idx, pullback_end):
        return None

    breakout_level = float(highs[transition_idx])
    if highs[breakout_idx] <= breakout_level:
        return None
    return pullback_start, pullback_end, breakout_level


def _setup_rejection_metrics(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    leg_high_idx: int,
    pullback_start: int,
    pullback_end: int,
    end_idx: int,
    leg_high: float,
    volumes: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return max setup topping-tail and jackknife rejection risks (0-1).

    Uses the shared volume-response candle scoring (continuous wick / body /
    range gradients).  Relative volume modulates severity toward
    SETUP_VOLUME_SEVERITY_FLOOR: a low-volume red pullback candle is healthy
    behaviour and is softened, while a high-volume rejection keeps full
    structural severity and adds red-volume pressure to the jackknife
    channel.  Without volume data the structural risks are used as-is.

    The scanned segment is intentionally limited to the leg-high candle plus
    the pullback candles. The breakout candle is not included.
    """
    atr = _atr_from_arrays(highs, lows, closes, end_idx, TOPPING_TAIL_ATR_PERIOD)
    if atr <= 0:
        return 0.0, 0.0

    baseline = None
    if volumes is not None and len(volumes) >= end_idx + 1:
        baseline = _volume_response_baseline(volumes[: end_idx + 1])
    floor = max(0.0, min(1.0, SETUP_VOLUME_SEVERITY_FLOOR))

    min_high = leg_high * PATTERN_HOD_PROXIMITY
    indexes = [leg_high_idx, *range(pullback_start, pullback_end + 1)]
    topping_tail = 0.0
    jackknife = 0.0
    for idx in indexes:
        if idx < 0 or idx >= len(highs):
            continue
        high = float(highs[idx])
        if high < min_high:
            continue
        open_ = float(opens[idx])
        close = float(closes[idx])
        components = _candle_rejection_components(
            open_, high, float(lows[idx]), close, atr
        )
        if components is None:
            continue
        tail_risk, body_risk, range_weight = components

        severity_scale = 1.0
        red_volume_risk = 0.0
        if baseline is not None:
            volume_floor, volume_full = baseline
            vol = float(volumes[idx])
            if math.isfinite(vol) and vol > 0.0:
                volume_weight = _smoothstep(
                    (vol - volume_floor) / max(1e-9, volume_full - volume_floor)
                )
                severity_scale = floor + ((1.0 - floor) * volume_weight)
                if close < open_ and volume_floor > 0.0:
                    red_volume_risk = _red_volume_pressure(vol / volume_floor)

        tail = min(1.0, tail_risk * range_weight) * severity_scale
        body = min(1.0, body_risk * range_weight) * severity_scale
        if red_volume_risk > 0.0:
            body = min(1.0, 1.0 - ((1.0 - body) * (1.0 - red_volume_risk)))
        topping_tail = max(topping_tail, tail)
        jackknife = max(jackknife, body)
    return topping_tail, jackknife


# ---------------------------------------------------------------------------
# 0. Pure pullback setup (non-resistance)
# ---------------------------------------------------------------------------

def detect_pullback_new_high(
    df: pd.DataFrame,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """
    Detect the classic momentum continuation setup:

    1) two to five consecutive green, rising candles
    2) one or more red pullback candles
    3) first candle that breaks the previous candle high, either directly
       after the pullback or after one extra green transition candle

    Pullback depth is measured against the up-leg:
        depth = (leg_high - pullback_low) / (leg_high - leg_open)

    Valid leg expansion must meet both ROC and ATR significance gates, and
    valid depth must not exceed PATTERN_PULLBACK_MAX_DEPTH.
    """
    signals: list[PatternSignal] = []
    if len(df) < 5 or _setup_timeframe_is_daily(df, timeframe):
        return signals

    opens = df["open"].astype(float).values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values

    # We only fire on the latest closed candle so the signal is event-like.
    breakout_idx = len(df) - 1
    if breakout_idx < 3:
        return signals

    window = _resolved_pullback_window(opens, highs, closes, breakout_idx)
    if window is None:
        return signals
    pullback_start, pullback_end, breakout_level = window

    leg_end = pullback_start - 1
    if leg_end < 0:
        return signals
    leg = _setup_green_leg(opens, highs, lows, closes, leg_end)
    if leg is None:
        return signals
    leg_start, leg_high_idx, leg_open, leg_high, leg_size = leg
    leg_len = leg_end - leg_start + 1
    pullback_len = pullback_end - pullback_start + 1
    if pullback_len > PATTERN_PULLBACK_MAX_CANDLES:
        return signals
    # Pullback may be at most as long as the leg (equal allowed): violent legs
    # are often only 2 bars, and a 2-bar pullback after them is a valid flag —
    # depth (PATTERN_PULLBACK_MAX_DEPTH) bounds the price proportion.
    if leg_len < pullback_len:
        return signals

    pullback_low = float(np.min(lows[pullback_start:pullback_end + 1]))
    pullback_depth = (leg_high - pullback_low) / leg_size

    if pullback_depth > PATTERN_PULLBACK_MAX_DEPTH:
        return signals

    setup_topping_tail, setup_jackknife = _setup_rejection_metrics(
        opens,
        highs,
        lows,
        closes,
        leg_high_idx=leg_high_idx,
        pullback_start=pullback_start,
        pullback_end=pullback_end,
        end_idx=breakout_idx,
        leg_high=leg_high,
        volumes=df["volume"].astype(float).values if "volume" in df.columns else None,
    )

    pullback_highs = highs[pullback_start:pullback_end + 1]
    pullback_lows = lows[pullback_start:pullback_end + 1]
    pullback_ts = _row_timestamp(df, pullback_start)

    # Clean pullbacks usually stair-step down in highs/lows.
    if pullback_len <= 1:
        structure_quality = 1.0
    else:
        ordered = 0
        for i in range(1, pullback_len):
            if (
                pullback_highs[i] <= pullback_highs[i - 1]
                and pullback_lows[i] <= pullback_lows[i - 1]
            ):
                ordered += 1
        structure_quality = ordered / (pullback_len - 1)

    breakout_ext = max(0.0, (float(closes[breakout_idx]) - breakout_level) / leg_size)

    depth_quality, pullback_len_quality, pullback_control = _pullback_control_components(
        pullback_depth,
        pullback_len,
    )

    # Strength: shallow/short pullback = stronger signal.
    strength = round(max(0.0, min(1.0, 0.50 + 0.45 * pullback_control)), 2)

    signals.append(PatternSignal(
        name="pullback_new_high",
        bias="bullish",
        strength=strength,
        price_level=round(breakout_level, 2),
        timeframe=timeframe,
        candle_count=max(1, breakout_idx - leg_start + 1),
        detail=(
            f"break of prior pullback high: trigger=${breakout_level:.2f}, "
            f"leg_high=${leg_high:.2f}, depth={pullback_depth:.1%}"
        ),
        pullback_depth=round(pullback_depth, 4),
        pullback_candles=pullback_len,
        pullback_control=round(pullback_control, 4),
        leg_candles=leg_len,
        pullback_structure=round(structure_quality, 4),
        breakout_extension=round(breakout_ext, 4),
        topping_tail=round(setup_topping_tail, 4),
        jackknife=round(setup_jackknife, 4),
        pullback_ts=pullback_ts,
        pullback_low=round(pullback_low, 4),
    ))

    return signals


def detect_pullback_pending(
    df: pd.DataFrame,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """Detect a pullback that is still in progress (no breakout yet).

    Mirrors ``detect_pullback_new_high`` but treats the latest candle as
    part of the pullback rather than the breakout.  Emits one signal when:

     1) consecutive green up-leg of >= PATTERN_PULLBACK_MIN_LEG_CANDLES
         through PATTERN_PULLBACK_MAX_LEG_CANDLES with ROC + ATR significance
     2) one or more consecutive red candles immediately after, ending on
         the latest closed candle, or one extra green transition candle after it
    3) pullback depth is at or below PATTERN_PULLBACK_MAX_DEPTH
    4) latest candle high becomes the trigger level

    ``price_level`` carries the latest candle high — the price
    the setup needs to break to resolve.
    """
    signals: list[PatternSignal] = []
    if len(df) < 4 or _setup_timeframe_is_daily(df, timeframe):
        return signals

    opens = df["open"].astype(float).values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values

    pullback_end = len(df) - 1
    end_idx = pullback_end

    pullback_start = _setup_pullback_start(opens, closes, pullback_end)
    if pullback_start is None:
        transition_idx = len(df) - 1
        pullback_end = transition_idx - 1
        pullback_start = _setup_pullback_start(opens, closes, pullback_end)
        if pullback_start is None:
            return signals
        if not _setup_extra_green_transition(opens, highs, closes, transition_idx, pullback_end):
            return signals
        end_idx = transition_idx

    leg_end = pullback_start - 1
    if leg_end < 0:
        return signals
    leg = _setup_green_leg(opens, highs, lows, closes, leg_end)
    if leg is None:
        return signals
    leg_start, leg_high_idx, leg_open, leg_high, leg_size = leg
    leg_len = leg_end - leg_start + 1
    pullback_len = pullback_end - pullback_start + 1
    if pullback_len > PATTERN_PULLBACK_MAX_CANDLES:
        return signals
    # Pullback may be at most as long as the leg (equal allowed): violent legs
    # are often only 2 bars, and a 2-bar pullback after them is a valid flag —
    # depth (PATTERN_PULLBACK_MAX_DEPTH) bounds the price proportion.
    if leg_len < pullback_len:
        return signals

    pullback_low = float(np.min(lows[pullback_start:pullback_end + 1]))
    pullback_depth = (leg_high - pullback_low) / leg_size
    if pullback_depth > PATTERN_PULLBACK_MAX_DEPTH:
        return signals

    breakout_level = float(highs[end_idx])

    setup_topping_tail, setup_jackknife = _setup_rejection_metrics(
        opens,
        highs,
        lows,
        closes,
        leg_high_idx=leg_high_idx,
        pullback_start=pullback_start,
        pullback_end=pullback_end,
        end_idx=end_idx,
        leg_high=leg_high,
        volumes=df["volume"].astype(float).values if "volume" in df.columns else None,
    )

    _, _, pullback_control = _pullback_control_components(pullback_depth, pullback_len)
    pullback_ts = _row_timestamp(df, pullback_start)
    strength = round(max(0.0, min(1.0, 0.50 + 0.45 * pullback_control)), 2)
    signals.append(PatternSignal(
        name="pullback_pending",
        bias="bullish",
        strength=strength,
        price_level=round(breakout_level, 2),
        timeframe=timeframe,
        candle_count=max(1, end_idx - leg_start + 1),
        detail=(
            f"pullback in progress: leg_open=${leg_open:.2f}, "
            f"trigger=${breakout_level:.2f}, leg_high=${leg_high:.2f}, "
            f"depth={pullback_depth:.1%}"
        ),
        pullback_depth=round(pullback_depth, 4),
        pullback_candles=pullback_len,
        pullback_control=round(pullback_control, 4),
        leg_candles=leg_len,
        topping_tail=round(setup_topping_tail, 4),
        jackknife=round(setup_jackknife, 4),
        pullback_ts=pullback_ts,
        pullback_low=round(pullback_low, 4),
    ))
    return signals


def detect_pullback_detected(
    df: pd.DataFrame,
    current_price: float,
    current_open: float,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """Detect a pullback forming mid-candle (before the red candle closes).

    This mirrors ``detect_pullback_pending`` but uses the in-progress candle's
    live state rather than waiting for it to close.  Fires when:

    1) The current (unclosed) candle is forming red (current_price < current_open)
    2) A valid green up-leg exists immediately before this candle
    3) Pullback depth from the in-progress low is within limits

    The signal name is ``pullback_detected`` — an awareness state that
    precedes ``pullback_pending`` (confirmed on close).

    ``df`` should contain only CLOSED bars (the in-progress bar is NOT in df).
    ``current_price`` and ``current_open`` represent the forming candle.
    """
    signals: list[PatternSignal] = []
    if current_price >= current_open:
        return signals  # candle is green — no pullback forming
    if len(df) < 3 or _setup_timeframe_is_daily(df, timeframe):
        return signals

    opens = df["open"].astype(float).values
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values

    # The last closed candle must be green (end of the leg)
    leg_end = len(df) - 1
    if closes[leg_end] <= opens[leg_end]:
        return signals

    leg = _setup_green_leg(opens, highs, lows, closes, leg_end)
    if leg is None:
        return signals
    leg_start, leg_high_idx, leg_open, leg_high, leg_size = leg

    # Pullback depth from the in-progress candle
    pullback_low = current_price  # current price is the live low estimate
    pullback_depth = (leg_high - pullback_low) / leg_size
    if pullback_depth > PATTERN_PULLBACK_MAX_DEPTH or pullback_depth < 0:
        return signals

    # Trigger level = leg high (the price that must break for entry)
    breakout_level = leg_high

    _, _, pullback_control = _pullback_control_components(pullback_depth, 1)
    pullback_ts = None  # no closed pullback candle yet
    strength = round(max(0.0, min(1.0, 0.50 + 0.45 * pullback_control)), 2)
    leg_len = leg_end - leg_start + 1

    signals.append(PatternSignal(
        name="pullback_detected",
        bias="bullish",
        strength=strength,
        price_level=round(breakout_level, 2),
        timeframe=timeframe,
        candle_count=max(1, leg_len + 1),
        detail=(
            f"pullback forming: leg_open=${leg_open:.2f}, "
            f"trigger=${breakout_level:.2f}, "
            f"depth={pullback_depth:.1%} (mid-candle)"
        ),
        pullback_depth=round(pullback_depth, 4),
        pullback_candles=1,
        pullback_control=round(pullback_control, 4),
        leg_candles=leg_len,
        topping_tail=0.0,
        jackknife=0.0,
        pullback_ts=pullback_ts,
        pullback_low=round(pullback_low, 4),
    ))
    return signals


# ---------------------------------------------------------------------------
# 1. Topping tail / Bottoming tail — single candle
# ---------------------------------------------------------------------------

def detect_topping_tail(candle: pd.Series) -> bool:
    """Return True if the candle has a significant upper wick rejection."""
    o, h, l, c = (
        float(candle["open"]),
        float(candle["high"]),
        float(candle["low"]),
        float(candle["close"]),
    )
    upper_wick = h - max(o, c)
    full_range = h - l
    if full_range <= 0:
        return False
    return (upper_wick / full_range) > TOPPING_TAIL_RATIO


def detect_bottoming_tail(candle: pd.Series) -> bool:
    """Return True if the candle has a significant lower wick rejection."""
    o, h, l, c = (
        float(candle["open"]),
        float(candle["high"]),
        float(candle["low"]),
        float(candle["close"]),
    )
    lower_wick = min(o, c) - l
    full_range = h - l
    if full_range <= 0:
        return False
    return (lower_wick / full_range) > TOPPING_TAIL_RATIO


# ---------------------------------------------------------------------------
# 2. Tail cluster detection
# ---------------------------------------------------------------------------

def detect_tail_clusters(
    df: pd.DataFrame,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """
    Detect clusters of topping or bottoming tails at similar price levels
    within a rolling window.
    """
    signals: list[PatternSignal] = []
    if len(df) < PATTERN_TAIL_CLUSTER_BARS:
        return signals

    window = df.iloc[-PATTERN_TAIL_CLUSTER_BARS:]
    top_tails: list[tuple[int, float]] = []
    bot_tails: list[tuple[int, float]] = []

    for i, (idx, row) in enumerate(window.iterrows()):
        if detect_topping_tail(row):
            top_tails.append((i, float(row["high"])))
        if detect_bottoming_tail(row):
            bot_tails.append((i, float(row["low"])))

    # Cluster topping tails
    for cluster in _cluster_prices(top_tails, PATTERN_TAIL_CLUSTER_TOL_PCT):
        if len(cluster) >= PATTERN_TAIL_CLUSTER_MIN:
            avg_price = sum(p for _, p in cluster) / len(cluster)
            signals.append(PatternSignal(
                name="topping_tail_cluster",
                bias="bearish",
                strength=min(0.4 + 0.15 * len(cluster), 1.0),
                price_level=round(avg_price, 2),
                timeframe=timeframe,
                candle_count=PATTERN_TAIL_CLUSTER_BARS,
                detail=f"{len(cluster)} topping tails near ${avg_price:.2f}",
            ))

    # Cluster bottoming tails
    for cluster in _cluster_prices(bot_tails, PATTERN_TAIL_CLUSTER_TOL_PCT):
        if len(cluster) >= PATTERN_TAIL_CLUSTER_MIN:
            avg_price = sum(p for _, p in cluster) / len(cluster)
            signals.append(PatternSignal(
                name="bottoming_tail_cluster",
                bias="bullish",
                strength=min(0.4 + 0.15 * len(cluster), 1.0),
                price_level=round(avg_price, 2),
                timeframe=timeframe,
                candle_count=PATTERN_TAIL_CLUSTER_BARS,
                detail=f"{len(cluster)} bottoming tails near ${avg_price:.2f}",
            ))

    return signals


# ---------------------------------------------------------------------------
# 3. Multi-top / Multi-bottom
# ---------------------------------------------------------------------------

def detect_multi_top(
    df: pd.DataFrame,
    pivot_n: int,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """
    Detect double-top, triple-top, etc. from pivot high clusters.
    Requires meaningful pullbacks between peaks.
    """
    signals: list[PatternSignal] = []
    if len(df) < 2 * pivot_n + 1:
        return signals

    highs = df["high"].values
    lows = df["low"].values
    pivots = _find_pivot_highs(highs, pivot_n)
    if len(pivots) < 2:
        return signals

    clusters = _cluster_prices(pivots, PATTERN_TOP_PRICE_TOL_PCT)

    for cluster in clusters:
        if len(cluster) < 2:
            continue
        # Sort by index to check pullbacks between consecutive peaks
        cluster_sorted = sorted(cluster, key=lambda x: x[0])
        valid_peaks: list[tuple[int, float]] = [cluster_sorted[0]]

        for prev, curr in zip(cluster_sorted, cluster_sorted[1:]):
            # Check for meaningful pullback between peaks
            between = lows[prev[0]:curr[0] + 1]
            if len(between) == 0:
                continue
            trough = float(np.min(between))
            peak_avg = (prev[1] + curr[1]) / 2
            pullback_pct = (peak_avg - trough) / peak_avg
            if pullback_pct >= PATTERN_TOP_MIN_PULLBACK:
                valid_peaks.append(curr)

        if len(valid_peaks) < 2:
            continue

        n_peaks = len(valid_peaks)
        names = {2: "double_top", 3: "triple_top"}
        name = names.get(n_peaks, f"{n_peaks}x_top")
        avg_price = sum(p for _, p in valid_peaks) / n_peaks
        span = valid_peaks[-1][0] - valid_peaks[0][0]

        # Volume divergence: declining volume on later peaks = stronger
        vol_strength = 0.0
        if "volume" in df.columns and n_peaks >= 2:
            vols = [float(df["volume"].iloc[idx]) for idx, _ in valid_peaks]
            if all(v > 0 for v in vols) and vols[0] > 0:
                if vols[-1] < vols[0]:
                    vol_strength = 0.15  # volume declining = more bearish

        base_strength = min(0.5 + 0.12 * n_peaks + vol_strength, 1.0)

        signals.append(PatternSignal(
            name=name,
            bias="bearish",
            strength=round(base_strength, 2),
            price_level=round(avg_price, 2),
            timeframe=timeframe,
            candle_count=span,
            detail=f"{n_peaks} peaks near ${avg_price:.2f} over {span} bars",
        ))

    return signals


def detect_multi_bottom(
    df: pd.DataFrame,
    pivot_n: int,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """
    Detect double-bottom, triple-bottom, etc. from pivot low clusters.
    """
    signals: list[PatternSignal] = []
    if len(df) < 2 * pivot_n + 1:
        return signals

    highs = df["high"].values
    lows = df["low"].values
    pivots = _find_pivot_lows(lows, pivot_n)
    if len(pivots) < 2:
        return signals

    clusters = _cluster_prices(pivots, PATTERN_TOP_PRICE_TOL_PCT)

    for cluster in clusters:
        if len(cluster) < 2:
            continue
        cluster_sorted = sorted(cluster, key=lambda x: x[0])
        valid_troughs: list[tuple[int, float]] = [cluster_sorted[0]]

        for prev, curr in zip(cluster_sorted, cluster_sorted[1:]):
            between = highs[prev[0]:curr[0] + 1]
            if len(between) == 0:
                continue
            peak = float(np.max(between))
            trough_avg = (prev[1] + curr[1]) / 2
            rally_pct = (peak - trough_avg) / trough_avg if trough_avg > 0 else 0
            if rally_pct >= PATTERN_TOP_MIN_PULLBACK:
                valid_troughs.append(curr)

        if len(valid_troughs) < 2:
            continue

        n_troughs = len(valid_troughs)
        names = {2: "double_bottom", 3: "triple_bottom"}
        name = names.get(n_troughs, f"{n_troughs}x_bottom")
        avg_price = sum(p for _, p in valid_troughs) / n_troughs
        span = valid_troughs[-1][0] - valid_troughs[0][0]

        vol_strength = 0.0
        if "volume" in df.columns and n_troughs >= 2:
            vols = [float(df["volume"].iloc[idx]) for idx, _ in valid_troughs]
            if all(v > 0 for v in vols) and vols[0] > 0:
                if vols[-1] > vols[0]:
                    vol_strength = 0.15

        base_strength = min(0.5 + 0.12 * n_troughs + vol_strength, 1.0)

        signals.append(PatternSignal(
            name=name,
            bias="bullish",
            strength=round(base_strength, 2),
            price_level=round(avg_price, 2),
            timeframe=timeframe,
            candle_count=span,
            detail=f"{n_troughs} troughs near ${avg_price:.2f} over {span} bars",
        ))

    return signals


# ---------------------------------------------------------------------------
# 4. Wedge detection (rising / falling)
# ---------------------------------------------------------------------------

def _fit_line(points: list[tuple[int, float]]) -> Optional[tuple[float, float]]:
    """Fit y = slope*x + intercept via least squares.  Returns (slope, intercept)."""
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    A = np.vstack([xs, np.ones(len(xs))]).T
    result = np.linalg.lstsq(A, ys, rcond=None)
    if len(result[0]) < 2:
        return None
    return float(result[0][0]), float(result[0][1])


def detect_wedge(
    df: pd.DataFrame,
    pivot_n: int,
    timeframe: str = "5min",
) -> list[PatternSignal]:
    """
    Detect rising wedge (bearish) and falling wedge (bullish).
    Uses pivot highs for upper trendline and pivot lows for lower trendline.
    """
    signals: list[PatternSignal] = []
    if len(df) < PATTERN_WEDGE_MIN_BARS:
        return signals

    highs = df["high"].values
    lows = df["low"].values
    ph = _find_pivot_highs(highs, pivot_n)
    pl = _find_pivot_lows(lows, pivot_n)

    if len(ph) < PATTERN_WEDGE_MIN_TOUCHES or len(pl) < PATTERN_WEDGE_MIN_TOUCHES:
        return signals

    upper = _fit_line(ph)
    lower = _fit_line(pl)
    if upper is None or lower is None:
        return signals

    upper_slope, _ = upper
    lower_slope, _ = lower

    # Check convergence — slopes must be moving toward each other
    if upper_slope == 0 and lower_slope == 0:
        return signals

    span = max(ph[-1][0], pl[-1][0]) - min(ph[0][0], pl[0][0])
    if span < PATTERN_WEDGE_MIN_BARS:
        return signals

    mid_price = (ph[-1][1] + pl[-1][1]) / 2
    touches = len(ph) + len(pl)
    touch_strength = min(0.1 * (touches - 4), 0.2) if touches > 4 else 0.0

    # Rising wedge: both slopes positive, upper slope < lower slope (converging)
    if upper_slope > 0 and lower_slope > 0:
        if upper_slope < lower_slope * PATTERN_WEDGE_CONVERGE_MAX:
            signals.append(PatternSignal(
                name="rising_wedge",
                bias="bearish",
                strength=round(min(0.6 + touch_strength, 1.0), 2),
                price_level=round(mid_price, 2),
                timeframe=timeframe,
                candle_count=span,
                detail=(
                    f"upper slope={upper_slope:.4f}, "
                    f"lower slope={lower_slope:.4f}, "
                    f"{touches} touches"
                ),
            ))

    # Falling wedge: both slopes negative, lower slope more neg than upper
    if upper_slope < 0 and lower_slope < 0:
        if lower_slope < upper_slope * PATTERN_WEDGE_CONVERGE_MAX:
            signals.append(PatternSignal(
                name="falling_wedge",
                bias="bullish",
                strength=round(min(0.6 + touch_strength, 1.0), 2),
                price_level=round(mid_price, 2),
                timeframe=timeframe,
                candle_count=span,
                detail=(
                    f"upper slope={upper_slope:.4f}, "
                    f"lower slope={lower_slope:.4f}, "
                    f"{touches} touches"
                ),
            ))

    return signals


# ---------------------------------------------------------------------------
# 5. Head and shoulders / Inverse
# ---------------------------------------------------------------------------

def detect_head_and_shoulders(
    df: pd.DataFrame,
    pivot_n: int,
    timeframe: str = "5min",
) -> list[PatternSignal]:
    """
    Detect head-and-shoulders (bearish) and inverse H&S (bullish).
    Requires 3 pivot highs (H&S) or 3 pivot lows (inverse).
    """
    signals: list[PatternSignal] = []
    if len(df) < 2 * pivot_n + 5:
        return signals

    highs = df["high"].values
    lows = df["low"].values

    # --- Standard H&S (bearish) ---
    ph = _find_pivot_highs(highs, pivot_n)
    if len(ph) >= 3:
        # Check all consecutive triplets
        for i in range(len(ph) - 2):
            left_idx, left_p = ph[i]
            head_idx, head_p = ph[i + 1]
            right_idx, right_p = ph[i + 2]

            # Head must be highest
            if head_p <= left_p or head_p <= right_p:
                continue
            # Head must meaningfully exceed shoulders
            shoulder_avg = (left_p + right_p) / 2
            if (head_p - shoulder_avg) / shoulder_avg < PATTERN_HNS_HEAD_MIN_EXCEED:
                continue
            # Shoulders must be roughly equal
            if abs(left_p - right_p) / max(left_p, right_p) > PATTERN_HNS_SHOULDER_TOL_PCT:
                continue

            # Neckline from the two troughs between peaks
            trough_1 = float(np.min(lows[left_idx:head_idx + 1]))
            trough_2 = float(np.min(lows[head_idx:right_idx + 1]))
            neckline = (trough_1 + trough_2) / 2

            # Shoulder symmetry bonus
            sym = 1.0 - abs(left_p - right_p) / max(left_p, right_p)
            strength = round(min(0.55 + 0.25 * sym, 1.0), 2)

            signals.append(PatternSignal(
                name="head_and_shoulders",
                bias="bearish",
                strength=strength,
                price_level=round(neckline, 2),
                timeframe=timeframe,
                candle_count=right_idx - left_idx,
                detail=(
                    f"L=${left_p:.2f} H=${head_p:.2f} R=${right_p:.2f} "
                    f"neckline=${neckline:.2f}"
                ),
            ))

    # --- Inverse H&S (bullish) ---
    pl = _find_pivot_lows(lows, pivot_n)
    if len(pl) >= 3:
        for i in range(len(pl) - 2):
            left_idx, left_p = pl[i]
            head_idx, head_p = pl[i + 1]
            right_idx, right_p = pl[i + 2]

            # Head must be lowest
            if head_p >= left_p or head_p >= right_p:
                continue
            shoulder_avg = (left_p + right_p) / 2
            if shoulder_avg == 0:
                continue
            if (shoulder_avg - head_p) / shoulder_avg < PATTERN_HNS_HEAD_MIN_EXCEED:
                continue
            if abs(left_p - right_p) / max(left_p, right_p) > PATTERN_HNS_SHOULDER_TOL_PCT:
                continue

            peak_1 = float(np.max(highs[left_idx:head_idx + 1]))
            peak_2 = float(np.max(highs[head_idx:right_idx + 1]))
            neckline = (peak_1 + peak_2) / 2

            sym = 1.0 - abs(left_p - right_p) / max(left_p, right_p)
            strength = round(min(0.55 + 0.25 * sym, 1.0), 2)

            signals.append(PatternSignal(
                name="inverse_head_and_shoulders",
                bias="bullish",
                strength=strength,
                price_level=round(neckline, 2),
                timeframe=timeframe,
                candle_count=right_idx - left_idx,
                detail=(
                    f"L=${left_p:.2f} H=${head_p:.2f} R=${right_p:.2f} "
                    f"neckline=${neckline:.2f}"
                ),
            ))

    return signals


# ---------------------------------------------------------------------------
# 6. Breakout pullback / Break + retest
# ---------------------------------------------------------------------------

def detect_breakout_pullback(
    df: pd.DataFrame,
    rmap,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """
    Detect two resistance-interaction patterns:

    (a) Break + retest (post-break):
        Price breaks above a level, pulls back to within tolerance of that
        level, then bounces (closes above for N bars).  Bullish confirmation.

    (b) Resistance pullback → break (pre-break):
        Price approaches resistance, pulls back (fails first attempt),
        then breaks through on the second push.  Bullish accumulation.

    All levels come from rmap (broken + active) — nothing hardcoded.
    """
    signals: list[PatternSignal] = []
    if len(df) < PATTERN_RETEST_MAX_BARS:
        return signals

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    current_price = float(closes[-1])

    # --- (a) Break + retest: use broken levels ---
    for lv in rmap.broken_levels():
        level_price = lv.price
        if level_price >= current_price:
            continue  # only care about broken levels now below us

        # Find the break point: first bar where close > level + buffer
        buf = level_price * PATTERN_RETEST_BUFFER_PCT
        break_idx = None
        for i in range(len(df) - PATTERN_RETEST_MAX_BARS, len(df)):
            if i < 0:
                continue
            if closes[i] > level_price + buf:
                break_idx = i
                break

        if break_idx is None:
            continue

        # Look for a pullback to the level after the break
        retest_idx = None
        for i in range(break_idx + 1, min(break_idx + PATTERN_RETEST_MAX_BARS, len(df))):
            if lows[i] <= level_price + buf:
                retest_idx = i
                break

        if retest_idx is None:
            continue

        # Confirm bounce: N bars closing above the level after retest
        bounce_count = 0
        for i in range(retest_idx + 1, min(retest_idx + PATTERN_RETEST_BOUNCE_BARS + 1, len(df))):
            if closes[i] > level_price:
                bounce_count += 1

        if bounce_count >= PATTERN_RETEST_BOUNCE_BARS:
            span = retest_idx - break_idx + bounce_count
            signals.append(PatternSignal(
                name="break_and_retest",
                bias="bullish",
                strength=round(min(0.6 + lv.strength * 0.2, 1.0), 2),
                price_level=round(level_price, 2),
                timeframe=timeframe,
                candle_count=span,
                detail=f"broke ${level_price:.2f} ({lv.source}), retested, bounced",
            ))

    # --- (b) Resistance pullback → break ---
    # Check recent active-then-broken levels: price rejected, pulled back,
    # then broke through
    for lv in rmap.broken_levels():
        level_price = lv.price
        if level_price >= current_price:
            continue

        buf = level_price * PATTERN_RETEST_BUFFER_PCT
        search_start = max(0, len(df) - PATTERN_RETEST_MAX_BARS * 2)

        # Find a rejection: high touched level but close stayed below
        reject_idx = None
        for i in range(search_start, len(df)):
            if highs[i] >= level_price - buf and closes[i] < level_price - buf:
                reject_idx = i
                break

        if reject_idx is None:
            continue

        # Find pullback: price moved away from level
        pullback_idx = None
        for i in range(reject_idx + 1, min(reject_idx + PATTERN_RETEST_MAX_BARS, len(df))):
            if highs[i] < level_price - buf * 2:
                pullback_idx = i
                break

        if pullback_idx is None:
            continue

        # Find the break: price then broke above
        broke_idx = None
        for i in range(pullback_idx + 1, len(df)):
            if closes[i] > level_price + buf:
                broke_idx = i
                break

        if broke_idx is not None:
            span = broke_idx - reject_idx
            signals.append(PatternSignal(
                name="resistance_pullback_break",
                bias="bullish",
                strength=round(min(0.65 + lv.strength * 0.2, 1.0), 2),
                price_level=round(level_price, 2),
                timeframe=timeframe,
                candle_count=span,
                detail=(
                    f"rejected at ${level_price:.2f} ({lv.source}), "
                    f"pulled back, then broke through"
                ),
            ))

    return signals


# ---------------------------------------------------------------------------
# 7. Support bounce / Resistance rejection (dip to level + bounce)
# ---------------------------------------------------------------------------

def detect_support_bounce(
    df: pd.DataFrame,
    rmap,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """
    Detect price dipping to a support level and bouncing.
    Support levels include: broken resistance, structural pivot lows,
    static dollar levels, EMA 9/20, VWAP, MA 50, etc.
    All pulled dynamically from ResistanceMap + indicator columns.
    """
    signals: list[PatternSignal] = []
    if len(df) < PATTERN_BOUNCE_CONFIRM_BARS + 2:
        return signals

    closes = df["close"].values
    lows = df["low"].values
    current_price = float(closes[-1])
    support_levels = _get_support_levels(rmap, df, current_price)

    for level_price, source in support_levels:
        if level_price <= 0:
            continue
        buf = level_price * PATTERN_BOUNCE_PROXIMITY_PCT

        # Look for a dip to this level in recent bars
        search_window = min(PATTERN_RETEST_MAX_BARS, len(df))
        touch_idx = None
        for i in range(len(df) - search_window, len(df)):
            if i < 0:
                continue
            if lows[i] <= level_price + buf and closes[i] >= level_price - buf:
                touch_idx = i
                break

        if touch_idx is None:
            continue

        # Confirm bounce: N bars closing above the level
        bounce_count = 0
        for i in range(touch_idx + 1, min(touch_idx + PATTERN_BOUNCE_CONFIRM_BARS + 1, len(df))):
            if closes[i] > level_price:
                bounce_count += 1

        if bounce_count >= PATTERN_BOUNCE_CONFIRM_BARS:
            span = (len(df) - 1) - touch_idx
            signals.append(PatternSignal(
                name="support_bounce",
                bias="bullish",
                strength=round(min(0.5 + 0.1 * bounce_count, 0.85), 2),
                price_level=round(level_price, 2),
                timeframe=timeframe,
                candle_count=span,
                detail=f"bounced off {source} at ${level_price:.2f}",
            ))

    return signals


def detect_resistance_rejection(
    df: pd.DataFrame,
    rmap,
    timeframe: str = "1min",
) -> list[PatternSignal]:
    """
    Detect price pushing into a resistance level and getting rejected.
    Resistance pulled dynamically from ResistanceMap + indicator columns.
    """
    signals: list[PatternSignal] = []
    if len(df) < PATTERN_BOUNCE_CONFIRM_BARS + 2:
        return signals

    closes = df["close"].values
    highs = df["high"].values
    current_price = float(closes[-1])
    resist_levels = _get_resistance_levels(rmap, df, current_price)

    for level_price, source in resist_levels:
        if level_price <= 0:
            continue
        buf = level_price * PATTERN_BOUNCE_PROXIMITY_PCT

        search_window = min(PATTERN_RETEST_MAX_BARS, len(df))
        touch_idx = None
        for i in range(len(df) - search_window, len(df)):
            if i < 0:
                continue
            if highs[i] >= level_price - buf and closes[i] <= level_price + buf:
                touch_idx = i
                break

        if touch_idx is None:
            continue

        # Confirm rejection: N bars closing below the level
        reject_count = 0
        for i in range(touch_idx + 1, min(touch_idx + PATTERN_BOUNCE_CONFIRM_BARS + 1, len(df))):
            if closes[i] < level_price:
                reject_count += 1

        if reject_count >= PATTERN_BOUNCE_CONFIRM_BARS:
            span = (len(df) - 1) - touch_idx
            signals.append(PatternSignal(
                name="resistance_rejection",
                bias="bearish",
                strength=round(min(0.5 + 0.1 * reject_count, 0.85), 2),
                price_level=round(level_price, 2),
                timeframe=timeframe,
                candle_count=span,
                detail=f"rejected at {source} at ${level_price:.2f}",
            ))

    return signals


# ---------------------------------------------------------------------------
# Move-start detector (timeframe-agnostic)
# ---------------------------------------------------------------------------

_DAILY_INTERVALS = {"1day", "1d", "day", "daily"}
# Bar-size labels that are unambiguously intraday. _setup_timeframe_is_daily
# short-circuits on these instead of inspecting the frame (see there).
_INTRADAY_INTERVALS = {
    "1sec", "5sec", "10sec", "10s", "15sec", "30sec",
    "1min", "1m", "2min", "2m", "3min", "3m", "5min", "5m",
    "10min", "10m", "15min", "15m", "30min", "30m",
    "1hour", "1h", "60min",
}
# Bars at least this far apart are read as a daily chart (20h in ns).
_DAILY_MIN_GAP_NS = 20 * 3_600 * 1_000_000_000


def _is_daily_chart(df: pd.DataFrame) -> bool:
    if "interval" in df.columns:
        intervals = {str(v).lower() for v in df["interval"].dropna().unique()}
        if intervals:
            return intervals.issubset(_DAILY_INTERVALS)

    # Work on int64 nanoseconds via numpy rather than Series.sort_values /
    # .diff — same answer, an order of magnitude cheaper on the frames the
    # live scorer passes here (a DatetimeIndex of closed bars).
    if "ts" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["ts"], errors="coerce", utc=True).dropna())
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index[~df.index.isna()]
    else:
        return False
    if len(idx) < 2:
        return False
    # pandas 3 infers microsecond resolution from Python datetimes; pin the
    # unit so the gap constant below is always compared in nanoseconds.
    ns = np.sort(idx.as_unit("ns").asi8)
    diffs = np.diff(ns)
    return bool(diffs.size and diffs.min() >= _DAILY_MIN_GAP_NS)


def _previous_trading_day(d) -> object:
    cursor = d
    while True:
        cursor = cursor - pd.Timedelta(days=1)
        if _MarketCalendar.is_trading_day(cursor):
            return cursor


def latest_extended_session_start(reference: datetime | pd.Timestamp | None = None) -> datetime:
    """Return the current trading day's premarket open (04:00 ET) as UTC.

    Before 04:00 ET, or on a non-trading day, falls back to the most recent
    trading day's premarket open so there is always a session at or before
    *reference*.
    """
    if reference is None:
        reference_ts = pd.Timestamp.now(tz=MarketClock.ET)
    else:
        reference_ts = pd.Timestamp(reference)
        if reference_ts.tzinfo is None:
            reference_ts = reference_ts.tz_localize(timezone.utc)
        reference_ts = reference_ts.tz_convert(MarketClock.ET)
    session_day = reference_ts.date()
    if (
        not _MarketCalendar.is_trading_day(session_day)
        or reference_ts.time() < MarketClock.EXT_OPEN
    ):
        session_day = _previous_trading_day(session_day)
    return (
        pd.Timestamp.combine(session_day, MarketClock.EXT_OPEN)
        .tz_localize(MarketClock.ET)
        .tz_convert("UTC")
        .to_pydatetime()
    )


def _latest_extended_session_mask(df: pd.DataFrame) -> Optional[np.ndarray]:
    ts_source = df["ts"] if "ts" in df.columns else None
    if ts_source is None and isinstance(df.index, pd.DatetimeIndex):
        ts_source = pd.Series(df.index, index=df.index)
    if ts_source is None:
        return None

    timestamps = pd.to_datetime(ts_source, errors="coerce", utc=True)
    valid = timestamps.notna()
    if not valid.any():
        return None

    et_times = timestamps.dt.tz_convert(MarketClock.ET)
    latest_et = et_times[valid].iloc[-1]
    session_start = pd.Timestamp(latest_extended_session_start(latest_et)).tz_convert(MarketClock.ET)
    return (et_times >= session_start).to_numpy(dtype=bool)


def detect_move_start(
    df: pd.DataFrame,
    *,
    rvol_window: int = 50,
    rvol_mult: float = 5.0,
    price_escape_pct: float = 5.0,
    floor_bars: int = 20,
    min_return_streak: int = 3,
    min_move_bars: int = 3,
    min_confirm_volume: float = 5_000.0,
) -> Optional[MoveStart]:
    """Find when the *whole* current move began.

    Works on any OHLCV DataFrame regardless of timeframe.  Intraday charts
    are clipped to the current trading day's extended session, which opens
    at the 04:00 ET premarket; daily charts are allowed to span multiple
    days.

    Algorithm
    ---------
     1. ``price_floor`` is the pre-move reference level.  Usually this is
         the median close over the first ``floor_bars`` of the DataFrame;
         when the move has already begun inside that window, use the edge of
         that early range so the detector can trace back to the actual lift.
     2. Rolling RVOL: each bar's volume divided by the average of up to the
         prior ``rvol_window`` bars from the same extended session.  Intraday
         charts can ignite once there are at least 5 prior bars; daily charts
         still require the full ``rvol_window`` history.
     3. A bar "escaped" when ``|close - price_floor| / price_floor`` is
       at least ``price_escape_pct`` percent.
     4. A bar is "confirmed" when it escaped AND its rolling RVOL is at
         least ``rvol_mult`` (default 5×) AND its absolute volume is at
         least ``min_confirm_volume`` shares (default 5k) — relative RVOL
         alone can trigger on trivial size off a dead-quiet baseline.
         Confirmation proves the move is real and anchors the returned start.
     5. Walk forward from the first escape.  The move is in force as
       long as price hasn't *sustained* a return to the floor.  A
       sustained return = ``min_return_streak`` consecutive non-escaped
       bars.  Single-bar dips through the floor are absorbed.
         6. The current move starts at the first confirmed bar after the
             most recent sustained return.  If price has never sustainedly
                 returned, the move starts at the first confirmed bar.

     This returns the start of the whole active move, not the first later
     continuation spike.

    Returns ``None`` when there is no clear move, the DataFrame is
    too short, or the baseline is zero.
    """
    if df is None:
        return None
    if "volume" not in df.columns or "close" not in df.columns:
        return None

    source_positions = np.arange(len(df), dtype=int)
    is_daily_chart = _is_daily_chart(df)
    if not is_daily_chart:
        session_mask = _latest_extended_session_mask(df)
        if session_mask is not None and session_mask.any():
            source_positions = source_positions[session_mask]
            df = df.loc[session_mask].reset_index(drop=True)

    min_rvol_history = rvol_window if is_daily_chart else min(5, rvol_window)
    if len(df) < max(floor_bars + min_move_bars, min_rvol_history + 1, 10):
        return None

    closes = df["close"].to_numpy(dtype=float)
    vols   = df["volume"].to_numpy(dtype=float)
    n = len(df)

    initial_count = min(floor_bars, n)
    initial_closes = closes[:initial_count]
    initial_median = float(np.median(initial_closes))
    initial_low = float(np.min(initial_closes))
    initial_high = float(np.max(initial_closes))
    escape_threshold = price_escape_pct / 100.0

    price_floor = initial_median
    if closes[-1] >= initial_median:
        if initial_low > 0 and (initial_median - initial_low) / initial_low >= escape_threshold:
            price_floor = initial_low
    elif initial_median > 0 and (initial_high - initial_median) / initial_median >= escape_threshold:
        price_floor = initial_high
    if price_floor <= 0:
        return None

    # Rolling RVOL — bar volume / mean of prior bars in the same extended session.
    # Daily charts require the full rvol_window; intraday charts can ignite
    # earlier once a small same-day baseline exists.
    rvol = np.zeros(n, dtype=float)
    for i in range(1, n):
        history_len = min(rvol_window, i)
        if history_len < min_rvol_history:
            continue
        avg = float(vols[i - history_len:i].mean())
        if avg > 0:
            rvol[i] = vols[i] / avg

    escape_frac = np.abs(closes - price_floor) / price_floor
    escaped  = escape_frac >= escape_threshold
    elevated = rvol >= rvol_mult
    voluminous = vols >= min_confirm_volume
    confirmed = escaped & elevated & voluminous

    if confirmed.sum() < min_move_bars:
        return None

    first_escape = int(np.where(escaped)[0][0])

    # Sustained return = >= min_return_streak consecutive non-escaped
    # bars after the first escape.  Find the END index of the most
    # recent such streak.
    sustained_return_end: int = -1
    run = 0
    for i in range(first_escape, n):
        if not escaped[i]:
            run += 1
            if run >= min_return_streak:
                sustained_return_end = i
        else:
            run = 0

    search_from = sustained_return_end + 1 if sustained_return_end >= 0 else first_escape
    after = np.where(confirmed[search_from:])[0]
    if after.size == 0:
        return None
    start_idx = search_from + int(after[0])

    confirmed_in_move = int(confirmed[start_idx:].sum())
    if confirmed_in_move < min_move_bars:
        return None

    start_price = float(closes[start_idx])
    current_price = float(closes[-1])
    if start_price <= 0:
        return None

    pct = (current_price - start_price) / start_price * 100.0
    if abs(pct) < 0.1:
        direction = "flat"
    elif pct > 0:
        direction = "up"
    else:
        direction = "down"

    # Baseline reported back to caller is the prior-window avg at the
    # start bar (the same reference the RVOL ratio used).
    base_history = min(rvol_window, start_idx)
    base_at_start = float(vols[start_idx - base_history:start_idx].mean()) if base_history > 0 else 0.0
    return MoveStart(
        start_idx=int(source_positions[start_idx]),
        bars_since_start=n - 1 - start_idx,
        start_price=start_price,
        current_price=current_price,
        price_change_pct=pct,
        direction=direction,
        volume_ratio=float(rvol[start_idx:].max()),
        baseline_volume=base_at_start,
    )


# ---------------------------------------------------------------------------
# Entry point — scan all patterns
# ---------------------------------------------------------------------------

def scan_all_patterns(
    df_10s: pd.DataFrame,
    df_1m: pd.DataFrame,
    df_2m: pd.DataFrame,
    df_5m: pd.DataFrame,
    rmap,
    *,
    pullback_last_bar_open: bool = True,
) -> PatternSnapshot:
    """
    Run all pattern detectors on the provided candle DataFrames and
    ResistanceMap.  Called from DynamicScorer._refresh_candles.

    Accepts four timeframes: 10sec, 1min, 2min, 5min.
    When the same pattern (name + price_level) appears on multiple
    timeframes, the higher timeframe is always preferred.

    ``pullback_last_bar_open``: when True (default — matches the live
    scanner where ``to_df`` always appends the in-progress current bar),
    the pullback detectors run against ``df.iloc[:-1]`` so both the
    pullback candle AND the breakout candle must be CLOSED before they
    can fire.  Set to False for offline analysis on pre-closed DataFrames.

    Returns a PatternSnapshot with all detected signals.
    """
    signals: list[PatternSignal] = []

    _timeframes: list[tuple[pd.DataFrame, str, int]] = [
        (df_10s, "10sec", PATTERN_PIVOT_N_10SEC),
        (df_1m,  "1min",  PATTERN_PIVOT_N_1MIN),
        (df_2m,  "2min",  PATTERN_PIVOT_N_2MIN),
        (df_5m,  "5min",  PATTERN_PIVOT_N_5MIN),
    ]

    for df, tf, pivot_n in _timeframes:
        if df.empty or len(df) < 5:
            continue
        # Pullback detectors only consider CLOSED candles — strip the
        # in-progress last bar so transient red/green wicks during
        # formation can't trigger a false setup.
        # Setup-disabled timeframes (SETUP_TFS_DISABLED) are skipped HERE,
        # not inside the detectors: the backtest's replay-window derivation
        # calls the detectors directly and must keep seeing all timeframes
        # (it decides which tick spans to replay — infrastructure, not
        # strategy).
        pb_df = df.iloc[:-1] if pullback_last_bar_open and len(df) > 1 else df
        if len(pb_df) >= 4 and tf not in SETUP_TFS_DISABLED:
            signals.extend(detect_pullback_new_high(pb_df, tf))
            signals.extend(detect_pullback_pending(pb_df, tf))
        signals.extend(detect_tail_clusters(df, tf))
        signals.extend(detect_multi_top(df, pivot_n, tf))
        signals.extend(detect_multi_bottom(df, pivot_n, tf))
        signals.extend(detect_wedge(df, pivot_n, tf))
        signals.extend(detect_head_and_shoulders(df, pivot_n, tf))
        signals.extend(detect_breakout_pullback(df, rmap, tf))
        signals.extend(detect_support_bounce(df, rmap, tf))
        signals.extend(detect_resistance_rejection(df, rmap, tf))

    # Deduplicate: if same pattern name + same price level on multiple
    # timeframes, always keep the higher timeframe version
    seen: dict[tuple[str, float], PatternSignal] = {}
    for s in signals:
        key = (s.name, s.price_level)
        existing = seen.get(key)
        if existing is None:
            seen[key] = s
        elif _TIMEFRAME_RANK.get(s.timeframe, 0) > _TIMEFRAME_RANK.get(existing.timeframe, 0):
            seen[key] = s  # higher timeframe always wins

    return PatternSnapshot(signals=list(seen.values()))
