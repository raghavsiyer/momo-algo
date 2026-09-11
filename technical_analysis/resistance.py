"""
technical_analysis/resistance.py

Resistance level detection and management.

LEVEL TYPES (enum LevelType):
    STATIC      — whole/half/quarter dollars, always present, computed from price
    STRUCTURE   — historical pivots, 200 EMA, 200 MA, daily highs from candle data
    DYNAMIC     — topping tails and intraday highs added live as they form
    TRENDLINE   — descending resistance line through 2+ pivot highs (STUB — future)

STRENGTH WEIGHTS:
    200 EMA / 200 MA          0.95
    Daily pivot high          0.85
    Whole dollar              0.80
    5-min HOD pivot high      0.70
    Half dollar               0.65
    Topping tail              0.60
    Quarter dollar (sub-$1)   0.55

PIVOT DETECTION:
    A pivot high is a candle whose high exceeds the N candles on both sides.
    N=3 for daily, N=2 for 5-min HOD pivots (smaller timeframe = tighter window).

TOPPING TAIL DETECTION:
    Upper wick / (body + lower wick) > TOPPING_TAIL_RATIO from config.
    Only flagged as resistance after price has retraced below the tail high.

USAGE:
    from technical_analysis.resistance import ResistanceMap

    # Build at eval time (after Stage 3 passes)
    rmap = ResistanceMap(symbol="ACMR", current_price=4.85)
    rmap.build_static()
    rmap.build_structure(df_daily, df_5min)

    # Query
    nearest = rmap.nearest_above(4.85)
    nearby  = rmap.levels_in_range(4.85, buffer=0.15)

    # Add live as price action develops
    rmap.add_dynamic(price=5.10, source="topping tail", formed_at=datetime.now(utc))

    # Check if price has cleanly broken a level
    rmap.mark_broken_levels(current_price=5.25)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

import pandas as pd

from config import (
    RESISTANCE_BREAK_BUFFER,
    WHOLE_DOLLAR_BUFFER,
)
from technical_analysis.patterns import (
    detect_topping_tail as _patterns_detect_topping_tail,
    detect_bottoming_tail as _patterns_detect_bottoming_tail,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LevelType(Enum):
    STATIC    = auto()    # whole/half/quarter dollar
    STRUCTURE = auto()    # pivots, 200 EMA/MA, daily high from candle history
    DYNAMIC   = auto()    # topping tails, intraday highs added from live stream
    TRENDLINE = auto()    # descending trendline through 2+ highs (future)


# ---------------------------------------------------------------------------
# Strength constants
# ---------------------------------------------------------------------------

STRENGTH = {
    "200_ema":          0.95,
    "200_ma":           0.95,
    "daily_pivot":      0.85,
    "whole_dollar":     0.80,
    "5min_pivot":       0.70,
    "15min_pivot":      0.70,  # backwards-compatible alias
    "half_dollar":      0.65,
    "topping_tail":     0.60,
    "bottoming_tail":   0.60,
    "daily_high":       0.60,
    "daily_low":        0.60,
    "daily_pivot_low":  0.80,
    "5min_pivot_low":   0.65,
    "15min_pivot_low":  0.65,  # backwards-compatible alias
    "quarter_dollar":   0.55,   # sub-$1 only: 0.25 and 0.75
    "trendline":        0.75,   # future — placeholder
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ResistanceLevel:
    """A single resistance level."""
    price: float
    level_type: LevelType
    strength: float                    # 0.0 – 1.0
    source: str                        # human-readable origin label
    formed_at: datetime                # when this level was identified
    active: bool = True                # False once cleanly broken above

    # Trendline extension (STUB — only used for TRENDLINE type)
    # slope is in dollars-per-second; intercept anchors the line
    _slope: Optional[float] = field(default=None, repr=False)
    _intercept: Optional[float] = field(default=None, repr=False)
    _anchor_ts: Optional[datetime] = field(default=None, repr=False)

    def projected_price(self, at: datetime) -> float:
        """
        Return the projected price of a trendline level at a given time.
        For non-trendline levels this just returns self.price.

        STUB: trendline projection not yet implemented — always returns
        self.price until slope/intercept are populated by the descending
        resistance detector (future Step).
        """
        if self.level_type != LevelType.TRENDLINE:
            return self.price
        if self._slope is None or self._intercept is None or self._anchor_ts is None:
            return self.price
        seconds_elapsed = (at - self._anchor_ts).total_seconds()
        return self._intercept + self._slope * seconds_elapsed

    def __repr__(self) -> str:
        status = "active" if self.active else "BROKEN"
        return (
            f"ResistanceLevel(${self.price:.2f} | {self.source} | "
            f"strength={self.strength:.2f} | {status})"
        )


# ---------------------------------------------------------------------------
# ResistanceMap
# ---------------------------------------------------------------------------

class ResistanceMap:
    """
    Holds and manages all resistance levels for a single symbol
    for the duration of a trading session.

    Instantiate after Stage 3 passes. Call build_static() and
    build_structure() once. Then call add_dynamic() from the live
    stream as new levels form.
    """

    def __init__(self, symbol: str, current_price: float) -> None:
        self.symbol        = symbol.upper()
        self.current_price = current_price
        self._levels: list[ResistanceLevel] = []

    # ------------------------------------------------------------------
    # TYPE 1 — Static price levels
    # ------------------------------------------------------------------

    def build_static(self, price: Optional[float] = None) -> None:
        """
        Generate whole-dollar, half-dollar, and (if price < $1)
        quarter-dollar levels in a window above and below current price.

        Window: current_price ± 20% to avoid cluttering with distant levels.
        Only levels ABOVE current price are resistance; levels below are
        support — but we store both so the map is useful for support queries
        too (active flag and queries handle the distinction).
        """
        p = price or self.current_price
        now = datetime.now(tz=timezone.utc)

        # Build search range: 20% below to 40% above current price
        low  = p * 0.80
        high = p * 1.40

        # --- Whole dollars ---
        first_whole = int(low) + (1 if low % 1 > 0 else 0)
        for dollar in range(first_whole, int(high) + 2):
            level_price = float(dollar)
            if low <= level_price <= high:
                self._add_or_merge(ResistanceLevel(
                    price=round(level_price, 2),
                    level_type=LevelType.STATIC,
                    strength=STRENGTH["whole_dollar"],
                    source="whole dollar",
                    formed_at=now,
                ))

        # --- Half dollars ---
        first_half = (int(low * 2) + 1) / 2
        val = first_half
        while val <= high:
            if val % 1 != 0:  # skip whole dollars already added
                self._add_or_merge(ResistanceLevel(
                    price=round(val, 2),
                    level_type=LevelType.STATIC,
                    strength=STRENGTH["half_dollar"],
                    source="half dollar",
                    formed_at=now,
                ))
            val = round(val + 0.5, 2)

        # --- Quarter dollars (only for sub-$1 stocks) ---
        if p < 1.0:
            for quarter in [0.25, 0.75]:
                for base in range(0, 2):
                    level_price = round(base + quarter, 2)
                    if low <= level_price <= high:
                        self._add_or_merge(ResistanceLevel(
                            price=level_price,
                            level_type=LevelType.STATIC,
                            strength=STRENGTH["quarter_dollar"],
                            source="quarter dollar",
                            formed_at=now,
                        ))

        logger.debug(
            f"[{self.symbol}] Static levels built: "
            f"{len([l for l in self._levels if l.level_type == LevelType.STATIC])}"
        )

    # ------------------------------------------------------------------
    # TYPE 2 — Structure levels from candle history
    # ------------------------------------------------------------------

    def build_structure(
        self,
        df_daily: pd.DataFrame,
        df_5min: pd.DataFrame,
    ) -> None:
        """
        Extract structural support/resistance from candle DataFrames.

        Daily pivots are treated as broad zones. 5-minute high-of-day pivots
        are treated as the intraday confirmation/refinement of those zones:
        when the two match, the level is kept at the exact 5-minute pivot
        price and the sources are merged. This keeps the ladder focused on
        tradable confluence instead of every historical local high.

        Fresh HOD candles are only treated as resistance after two later
        5-minute candles confirm them, unless the HOD candle is itself a
        topping-tail rejection.
        """
        now = datetime.now(tz=timezone.utc)

        min5_pivots: list[float] = []
        min5_sources: dict[float, str] = {}
        intraday_hod: Optional[float] = None
        if not df_5min.empty:
            if "high" in df_5min.columns:
                intraday_hod = float(pd.to_numeric(df_5min["high"], errors="coerce").max())
            min5_pivots = self._find_pivot_highs(
                df_5min,
                n=2,
                hod_only=True,
                min_prominence_pct=0.015,
                max_pivots=6,
            )
            min5_sources = {
                pivot_price: "5-min HOD pivot"
                for pivot_price in min5_pivots
            }
            hod_double_top = self._find_hod_double_top(df_5min)
            if hod_double_top is not None:
                matching_pivot = self._nearest_matching_pivot(
                    hod_double_top,
                    min5_pivots,
                )
                if matching_pivot is not None:
                    source = min5_sources.pop(matching_pivot, "5-min HOD pivot")
                    if "5-min HOD double top" not in source:
                        source = f"{source} / 5-min HOD double top"
                    min5_pivots = [
                        pivot_price
                        for pivot_price in min5_pivots
                        if pivot_price != matching_pivot
                    ]
                    min5_pivots.append(hod_double_top)
                    min5_sources[hod_double_top] = source
                else:
                    min5_pivots.append(hod_double_top)
                    min5_sources[hod_double_top] = "5-min HOD double top"
                min5_pivots.sort()

            hod_topping_tail = self._find_recent_hod_topping_tail(df_5min)
            if hod_topping_tail is not None:
                matching_pivot = self._nearest_matching_pivot(
                    hod_topping_tail,
                    min5_pivots,
                )
                if matching_pivot is not None:
                    source = min5_sources.pop(matching_pivot, "5-min HOD pivot")
                    if "5-min HOD topping tail" not in source:
                        source = f"{source} / 5-min HOD topping tail"
                    min5_pivots = [
                        pivot_price
                        for pivot_price in min5_pivots
                        if pivot_price != matching_pivot
                    ]
                    min5_pivots.append(hod_topping_tail)
                    min5_sources[hod_topping_tail] = source
                else:
                    min5_pivots.append(hod_topping_tail)
                    min5_sources[hod_topping_tail] = "5-min HOD topping tail"
                min5_pivots.sort()

            eligible_min5_pivots: list[float] = []
            for pivot_price in min5_pivots:
                if self._hod_level_is_mature_or_tail(df_5min, pivot_price):
                    eligible_min5_pivots.append(pivot_price)
                else:
                    min5_sources.pop(pivot_price, None)
            min5_pivots = eligible_min5_pivots

        # --- Daily: 200 EMA and 200 MA ---
        if not df_daily.empty:
            last_daily = df_daily.iloc[-1]

            if "ema_200" in df_daily.columns:
                ema_200_val = last_daily.get("ema_200")
                if pd.notna(ema_200_val):
                    self._add_or_merge(ResistanceLevel(
                        price=round(float(ema_200_val), 2),
                        level_type=LevelType.STRUCTURE,
                        strength=STRENGTH["200_ema"],
                        source="200 EMA (daily)",
                        formed_at=now,
                    ))

            if "ma_200" in df_daily.columns:
                ma_200_val = last_daily.get("ma_200")
                if pd.notna(ma_200_val):
                    self._add_or_merge(ResistanceLevel(
                        price=round(float(ma_200_val), 2),
                        level_type=LevelType.STRUCTURE,
                        strength=STRENGTH["200_ma"],
                        source="200 MA (daily)",
                        formed_at=now,
                    ))

            daily_pivots = self._find_pivot_highs(
                df_daily,
                n=3,
                min_prominence_pct=0.03,
                cluster_tol_pct=0.025,
                max_pivots=12,
            )
            matched_min5: set[float] = set()
            daily_levels: list[tuple[float, str]] = []
            for pivot_price in daily_pivots:
                if intraday_hod is not None and pivot_price > intraday_hod * 1.02:
                    continue
                match = self._nearest_matching_pivot(pivot_price, min5_pivots)
                if match is not None:
                    matched_min5.add(match)
                    daily_levels.append((
                        match,
                        f"daily pivot high / {min5_sources.get(match, '5-min HOD pivot')}",
                    ))
                elif self._near_whole_dollar_zone(pivot_price):
                    daily_levels.append((pivot_price, "daily pivot high"))

            for pivot_price, source in daily_levels:
                self._add_or_merge(ResistanceLevel(
                    price=round(pivot_price, 2),
                    level_type=LevelType.STRUCTURE,
                    strength=STRENGTH["daily_pivot"],
                    source=source,
                    formed_at=now,
                ))

            for pivot_price in min5_pivots:
                if any(self._levels_match(pivot_price, matched) for matched in matched_min5):
                    continue
                self._add_or_merge(ResistanceLevel(
                    price=round(pivot_price, 2),
                    level_type=LevelType.STRUCTURE,
                    strength=STRENGTH["5min_pivot"],
                    source=min5_sources.get(pivot_price, "5-min HOD pivot"),
                    formed_at=now,
                ))

            # Most recent completed day's session high
            if len(df_daily) >= 2:
                prev_day_high = float(df_daily.iloc[-2]["high"])
                self._add_or_merge(ResistanceLevel(
                    price=round(prev_day_high, 2),
                    level_type=LevelType.STRUCTURE,
                    strength=STRENGTH["daily_high"],
                    source="prev day high",
                    formed_at=now,
                ))
        else:
            for pivot_price in min5_pivots:
                self._add_or_merge(ResistanceLevel(
                    price=round(pivot_price, 2),
                    level_type=LevelType.STRUCTURE,
                    strength=STRENGTH["5min_pivot"],
                    source=min5_sources.get(pivot_price, "5-min HOD pivot"),
                    formed_at=now,
                ))

        # --- Support: daily pivot lows ---
        if not df_daily.empty:
            daily_pivot_lows = self._find_pivot_lows(df_daily, n=3)
            for pivot_price in daily_pivot_lows:
                self._add_or_merge(ResistanceLevel(
                    price=round(pivot_price, 2),
                    level_type=LevelType.STRUCTURE,
                    strength=STRENGTH["daily_pivot_low"],
                    source="daily pivot low",
                    formed_at=now,
                ))

            # Previous day's session low
            if len(df_daily) >= 2:
                prev_day_low = float(df_daily.iloc[-2]["low"])
                self._add_or_merge(ResistanceLevel(
                    price=round(prev_day_low, 2),
                    level_type=LevelType.STRUCTURE,
                    strength=STRENGTH["daily_low"],
                    source="prev day low",
                    formed_at=now,
                ))

        # --- Support: 5-min pivot lows ---
        if not df_5min.empty:
            min5_pivot_lows = self._find_pivot_lows(df_5min, n=2)
            for pivot_price in min5_pivot_lows:
                self._add_or_merge(ResistanceLevel(
                    price=round(pivot_price, 2),
                    level_type=LevelType.STRUCTURE,
                    strength=STRENGTH["5min_pivot_low"],
                    source="5-min pivot low",
                    formed_at=now,
                ))

        structure_count = len([
            l for l in self._levels if l.level_type == LevelType.STRUCTURE
        ])
        logger.debug(
            f"[{self.symbol}] Structure levels built: {structure_count}"
        )

    # ------------------------------------------------------------------
    # TYPE 3 — Dynamic levels (live, called from stream)
    # ------------------------------------------------------------------

    def add_dynamic(
        self,
        price: float,
        source: str,
        formed_at: Optional[datetime] = None,
        strength: Optional[float] = None,
    ) -> ResistanceLevel:
        """
        Add a dynamic resistance level formed by live price action.
        Called when a topping tail or new intraday high is detected
        in the live candle stream.

        source: descriptive string, e.g. "topping tail", "intraday high"
        strength: override default; if None, infers from source string.
        """
        if formed_at is None:
            formed_at = datetime.now(tz=timezone.utc)

        if strength is None:
            if "topping" in source.lower():
                strength = STRENGTH["topping_tail"]
            elif "high" in source.lower():
                strength = STRENGTH["daily_high"]
            else:
                strength = 0.55  # conservative default

        level = ResistanceLevel(
            price=round(price, 2),
            level_type=LevelType.DYNAMIC,
            strength=strength,
            source=source,
            formed_at=formed_at,
        )
        self._add_or_merge(level)
        logger.debug(
            f"[{self.symbol}] Dynamic level added: ${price:.2f} ({source})"
        )
        return level

    def set_dynamic(
        self,
        price: float,
        source: str,
        formed_at: Optional[datetime] = None,
        strength: Optional[float] = None,
    ) -> ResistanceLevel:
        """Create or update a dynamic level by source.

        This is intended for live-moving references (e.g. session VWAP,
        EMA 9/20) where the level should be refreshed in place rather than
        repeatedly appended and logged.
        """
        if formed_at is None:
            formed_at = datetime.now(tz=timezone.utc)

        if strength is None:
            if "ema" in source.lower():
                strength = STRENGTH["200_ema"]
            elif "vwap" in source.lower():
                strength = STRENGTH["daily_pivot"]
            else:
                strength = 0.60

        rounded_price = round(price, 2)
        for existing in self._levels:
            if existing.level_type == LevelType.DYNAMIC and existing.source == source:
                existing.price = rounded_price
                existing.formed_at = formed_at
                existing.strength = strength
                existing.active = True
                return existing

        level = ResistanceLevel(
            price=rounded_price,
            level_type=LevelType.DYNAMIC,
            strength=strength,
            source=source,
            formed_at=formed_at,
        )
        self._levels.append(level)
        return level

    # ------------------------------------------------------------------
    # TYPE 4 — Descending trendline (STUB)
    # ------------------------------------------------------------------

    def build_descending_trendline(self) -> Optional[ResistanceLevel]:
        """
        STUB — Descending resistance trendline through 2+ pivot highs.

        Implementation plan (future Step):
            1. Collect all DYNAMIC + STRUCTURE levels formed in sequence.
            2. Filter to levels where each successive high is lower than
               the previous (lower highs = descending resistance).
            3. Require at least 2 confirmed lower highs.
            4. Fit a line: price = slope * seconds_from_anchor + intercept
               using the two most recent lower highs as anchor points.
            5. Create a ResistanceLevel with level_type=TRENDLINE and
               populate _slope, _intercept, _anchor_ts.
            6. Call projected_price(datetime.now()) to get the current
               value of the line.

        Preconditions not yet met:
            - Requires live candle stream to have run long enough to form
              2+ lower highs (minimum ~10-15 minutes of price action).
            - Needs the dynamic confidence scorer (Step 3c) to be running
              so topping tails are being added via add_dynamic().

        Returns None until implemented.
        """
        logger.debug(
            f"[{self.symbol}] build_descending_trendline() called — "
            "not yet implemented, returning None"
        )
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def nearest_above(self, price: Optional[float] = None) -> Optional[ResistanceLevel]:
        """
        Return the closest active resistance level strictly above *price*.
        If price is None, uses self.current_price.
        """
        p = price if price is not None else self.current_price
        candidates = [
            l for l in self._levels
            if l.active and l.price > p
        ]
        return min(candidates, key=lambda l: l.price) if candidates else None

    def nearest_below(self, price: Optional[float] = None) -> Optional[ResistanceLevel]:
        """
        Return the closest active level strictly below *price* (support).
        """
        p = price if price is not None else self.current_price
        candidates = [
            l for l in self._levels
            if l.active and l.price < p
        ]
        return max(candidates, key=lambda l: l.price) if candidates else None

    def levels_in_range(
        self,
        price: Optional[float] = None,
        buffer: float = 0.15,
        active_only: bool = True,
    ) -> list[ResistanceLevel]:
        """
        Return all levels within *buffer* dollars of *price*, sorted by price.
        Useful for checking if price is approaching a cluster of resistance.
        """
        p = price if price is not None else self.current_price
        results = [
            l for l in self._levels
            if abs(l.price - p) <= buffer
            and (not active_only or l.active)
        ]
        return sorted(results, key=lambda l: l.price)

    def levels_above(
        self,
        price: Optional[float] = None,
        limit: int = 5,
        active_only: bool = True,
    ) -> list[ResistanceLevel]:
        """
        Return the *limit* closest active levels above *price*, sorted ascending.
        Used to display the resistance ladder above current price.
        """
        p = price if price is not None else self.current_price
        candidates = [
            l for l in self._levels
            if l.price > p and (not active_only or l.active)
        ]
        return sorted(candidates, key=lambda l: l.price)[:limit]

    def strongest_in_range(
        self,
        price: Optional[float] = None,
        buffer: float = 0.20,
    ) -> Optional[ResistanceLevel]:
        """
        Return the highest-strength level within *buffer* of *price*.
        Useful for deciding how much caution to apply near a level.
        """
        candidates = self.levels_in_range(price, buffer)
        return max(candidates, key=lambda l: l.strength) if candidates else None

    def levels_below(
        self,
        price: Optional[float] = None,
        limit: int = 5,
        include_broken: bool = True,
    ) -> list[ResistanceLevel]:
        """
        Return the *limit* closest levels below *price*, sorted descending
        (nearest first).  Includes both active levels below price and broken
        levels that now act as support.
        """
        p = price if price is not None else self.current_price
        candidates = [
            l for l in self._levels
            if l.price < p and (l.active or include_broken)
        ]
        return sorted(candidates, key=lambda l: l.price, reverse=True)[:limit]

    def broken_levels(self) -> list[ResistanceLevel]:
        """Return all broken (inactive) levels — these act as support."""
        return sorted(
            [l for l in self._levels if not l.active],
            key=lambda l: l.price,
        )

    def is_near_resistance(
        self,
        price: Optional[float] = None,
        buffer: Optional[float] = None,
    ) -> bool:
        """
        Return True if *price* is within *buffer* of any active resistance.
        Buffer defaults to WHOLE_DOLLAR_BUFFER from config.
        """
        buf = buffer if buffer is not None else WHOLE_DOLLAR_BUFFER
        return bool(self.levels_in_range(price, buffer=buf))

    def mark_broken_levels(
        self,
        current_price: float,
        break_buffer: float = RESISTANCE_BREAK_BUFFER,
    ) -> list[ResistanceLevel]:
        """
        Deactivate any levels that current_price has cleanly broken above.
        A level is considered broken when price > level.price + break_buffer.
        Returns the list of newly broken levels.

        Call this on each price tick in the live stream to keep the map current.
        """
        self.current_price = current_price
        broken = []
        for level in self._levels:
            if level.active and current_price > level.price + break_buffer:
                level.active = False
                broken.append(level)
                logger.debug(
                    f"[{self.symbol}] Level broken: ${level.price:.2f} "
                    f"({level.source})"
                )
        return broken

    def all_levels(self, active_only: bool = False) -> list[ResistanceLevel]:
        """Return all levels, sorted by price ascending."""
        levels = self._levels if not active_only else [
            l for l in self._levels if l.active
        ]
        return sorted(levels, key=lambda l: l.price)

    def summary(self, price: Optional[float] = None) -> str:
        """
        Return a formatted string of the resistance ladder above current price.
        Useful for logging and the watchlist display.
        """
        p = price or self.current_price
        above = self.levels_above(p, limit=6)
        if not above:
            return f"  [{self.symbol}] No resistance levels above ${p:.2f}"

        lines = [f"  [{self.symbol}] Resistance above ${p:.2f}:"]
        for l in above:
            dist = l.price - p
            lines.append(
                f"    ${l.price:.2f}  (+${dist:.2f})  "
                f"{l.source:<22}  strength={l.strength:.2f}  "
                f"{'ACTIVE' if l.active else 'broken'}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _levels_match(
        price_a: float,
        price_b: float,
        *,
        tol_pct: float = 0.025,
        min_abs: float = 0.06,
        max_abs: float = 0.15,
    ) -> bool:
        ref = max(abs(price_a), abs(price_b), 1.0)
        tolerance = min(max(ref * tol_pct, min_abs), max_abs)
        return abs(price_a - price_b) <= tolerance

    @staticmethod
    def _near_whole_dollar_zone(price: float, max_abs: float = 0.12) -> bool:
        nearest_whole = round(price)
        if nearest_whole <= 0:
            return False
        return abs(price - nearest_whole) <= max_abs

    @classmethod
    def _nearest_matching_pivot(cls, pivot_price: float, candidates: list[float]) -> Optional[float]:
        matches = [p for p in candidates if cls._levels_match(pivot_price, p)]
        if not matches:
            return None
        return min(matches, key=lambda p: abs(p - pivot_price))

    @classmethod
    def _is_static_structure_confluence(
        cls,
        existing: ResistanceLevel,
        new_level: ResistanceLevel,
    ) -> bool:
        if {existing.level_type, new_level.level_type} != {LevelType.STATIC, LevelType.STRUCTURE}:
            return False
        static_level = existing if existing.level_type == LevelType.STATIC else new_level
        structure_level = existing if existing.level_type == LevelType.STRUCTURE else new_level
        if static_level.source != "whole dollar":
            return False
        if "pivot" not in structure_level.source.lower():
            return False
        return abs(static_level.price - structure_level.price) <= 0.12

    def _add_or_merge(self, new_level: ResistanceLevel, merge_tol: float = 0.03) -> None:
        """
        Add a level, or merge with an existing level if it represents the
        same zone.

        Same-type levels merge by a tight dollar tolerance. Whole-dollar
        static levels may also align to nearby structural pivot highs so a
        confirmed zone like $5.11 is displayed as one confluence level rather
        than a separate $5.00 static row plus a $5.11 pivot row.
        """
        for existing in self._levels:
            if not existing.active:
                continue

            if (
                abs(existing.price - new_level.price) <= merge_tol
                and existing.level_type == new_level.level_type
            ):
                if new_level.strength > existing.strength:
                    existing.strength = new_level.strength
                existing.price = round(
                    (existing.price + new_level.price) / 2, 2
                )
                if new_level.source not in existing.source:
                    existing.source = f"{existing.source} / {new_level.source}"
                return

            if self._is_static_structure_confluence(existing, new_level):
                if existing.level_type == LevelType.STATIC:
                    existing.price = round(new_level.price, 2)
                    self._levels.append(new_level)
                    return
                new_level.price = round(existing.price, 2)
                self._levels.append(new_level)
                return

        self._levels.append(new_level)

    @staticmethod
    def _find_hod_double_top(
        df: pd.DataFrame,
        *,
        tol_pct: float = 0.02,
        n: int = 2,
    ) -> Optional[float]:
        """
        Return the session HOD when at least two distinct local tops form
        within ``tol_pct`` of that HOD.

        This catches opening-spike HODs that later retest near the high, a
        shape strict pivot detection can miss because the first touch is too
        close to the left edge and the retest is not a new running high.
        """
        if df.empty or "high" not in df.columns:
            return None

        highs = pd.to_numeric(df["high"], errors="coerce").dropna().to_list()
        if len(highs) < 2:
            return None

        session_hod = max(float(price) for price in highs)
        if session_hod <= 0:
            return None

        threshold = session_hod * (1.0 - tol_pct)
        candidates: list[tuple[int, float]] = []
        for idx, raw_price in enumerate(highs):
            price = float(raw_price)
            if price < threshold:
                continue

            left = [float(p) for p in highs[max(0, idx - n):idx]]
            right = [float(p) for p in highs[idx + 1:idx + n + 1]]
            if not right:
                continue
            if left and price < max(left):
                continue
            if price < max(right):
                continue
            candidates.append((idx, price))

        if len(candidates) < 2:
            return None

        distinct: list[tuple[int, float]] = []
        min_gap = max(1, n)
        for idx, price in candidates:
            if distinct and idx - distinct[-1][0] <= min_gap:
                if price > distinct[-1][1]:
                    distinct[-1] = (idx, price)
                continue
            distinct.append((idx, price))

        if len(distinct) < 2:
            return None
        return round(session_hod, 4)

    @classmethod
    def _find_recent_hod_topping_tail(
        cls,
        df: pd.DataFrame,
        *,
        recent_bars: int = 2,
    ) -> Optional[float]:
        if df.empty or "high" not in df.columns:
            return None

        highs = pd.to_numeric(df["high"], errors="coerce").reset_index(drop=True)
        if highs.empty or highs.isna().all():
            return None

        session_hod = float(highs.max())
        if session_hod <= 0:
            return None

        tolerance = max(abs(session_hod) * 0.0005, 0.0001)
        start_idx = max(0, len(highs) - recent_bars)
        for idx in range(start_idx, len(highs)):
            high = highs.iloc[idx]
            if pd.isna(high) or abs(float(high) - session_hod) > tolerance:
                continue
            try:
                if cls.detect_topping_tail(df.iloc[idx]):
                    return round(session_hod, 4)
            except (KeyError, TypeError, ValueError):
                continue
        return None

    @classmethod
    def _hod_level_is_mature_or_tail(
        cls,
        df: pd.DataFrame,
        price: float,
        *,
        recent_bars: int = 2,
    ) -> bool:
        if df.empty or "high" not in df.columns:
            return True

        highs = pd.to_numeric(df["high"], errors="coerce").reset_index(drop=True)
        if highs.empty or highs.isna().all():
            return True

        tolerance = max(abs(price) * 0.0005, 0.0001)
        matching_idxs = [
            int(idx)
            for idx, high in highs.items()
            if pd.notna(high) and abs(float(high) - price) <= tolerance
        ]
        if not matching_idxs:
            return True

        mature_cutoff = max(0, len(highs) - recent_bars)
        if any(idx < mature_cutoff for idx in matching_idxs):
            return True

        for idx in matching_idxs:
            try:
                if cls.detect_topping_tail(df.iloc[idx]):
                    return True
            except (KeyError, TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _find_pivot_highs(
        df: pd.DataFrame,
        n: int = 2,
        *,
        hod_only: bool = False,
        min_prominence_pct: float = 0.0,
        cluster_tol_pct: float = 0.02,
        max_pivots: Optional[int] = None,
    ) -> list[float]:
        """
        Identify swing pivot highs in a candle DataFrame.

        A pivot high at index i has a high strictly greater than the N
        candles on each side. Optional filters let callers ask for only
        high-of-day pivots, require a minimum pullback prominence, and
        cluster nearby prices into one zone.
        """
        if df.empty or len(df) < (2 * n + 1):
            return []

        import numpy as np
        highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
        lows = pd.to_numeric(df.get("low", df["high"]), errors="coerce").to_numpy(dtype=float)
        closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
        if len(highs) == 0 or np.isnan(highs).all() or np.isnan(closes).all():
            return []
        recent_close = float(closes[~np.isnan(closes)][-1])

        length = len(highs)
        is_pivot = np.ones(length, dtype=bool)
        is_pivot[:n] = False
        is_pivot[length - n:] = False
        is_pivot &= ~np.isnan(highs)

        for offset in range(1, n + 1):
            is_pivot[n:length - n] &= highs[n:length - n] > highs[n - offset:length - n - offset]
            is_pivot[n:length - n] &= highs[n:length - n] > highs[n + offset:length - n + offset]

        is_pivot &= highs > recent_close

        if hod_only:
            running_high = np.maximum.accumulate(np.nan_to_num(highs, nan=-np.inf))
            is_pivot &= highs >= running_high

        candidate_idxs = np.where(is_pivot)[0]
        if min_prominence_pct > 0:
            prominent: list[int] = []
            look = max(n * 3, n + 1)
            for idx in candidate_idxs:
                right_lows = lows[idx + 1:min(length, idx + look + 1)]
                left_lows = lows[max(0, idx - look):idx]
                right_lows = right_lows[~np.isnan(right_lows)]
                left_lows = left_lows[~np.isnan(left_lows)]
                if len(right_lows) == 0 or len(left_lows) == 0:
                    continue
                pullback_low = min(float(right_lows.min()), float(left_lows.min()))
                if highs[idx] <= 0:
                    continue
                prominence = (float(highs[idx]) - pullback_low) / float(highs[idx])
                if prominence >= min_prominence_pct:
                    prominent.append(int(idx))
            candidate_idxs = np.array(prominent, dtype=int)

        if candidate_idxs.size == 0:
            return []

        candidates = sorted(
            (float(highs[idx]), int(idx)) for idx in candidate_idxs
        )
        clusters: list[list[tuple[float, int]]] = []
        for price, idx in candidates:
            if not clusters:
                clusters.append([(price, idx)])
                continue
            anchor = clusters[-1][0][0]
            if anchor > 0 and abs(price - anchor) / anchor <= cluster_tol_pct:
                clusters[-1].append((price, idx))
            else:
                clusters.append([(price, idx)])

        ranked: list[tuple[float, int, int]] = []
        for cluster in clusters:
            representative = max(price for price, _ in cluster)
            latest_idx = max(idx for _, idx in cluster)
            touch_count = len(cluster)
            ranked.append((representative, touch_count, latest_idx))

        if max_pivots is not None and len(ranked) > max_pivots:
            ranked.sort(key=lambda item: (item[1], item[2]), reverse=True)
            ranked = ranked[:max_pivots]

        return sorted(round(price, 4) for price, _, _ in ranked)

    @staticmethod
    def _find_pivot_lows(df: pd.DataFrame, n: int = 2) -> list[float]:
        """
        Identify swing pivot lows in a candle DataFrame.
        A pivot low at index i: df['low'][i] is strictly less than
        the N candles on each side.

        Only returns pivots that are below the most recent close
        (so we skip historical pivots that have already been eclipsed).

        Returns a deduplicated list of pivot low prices.
        """
        if df.empty or len(df) < (2 * n + 1):
            return []

        import numpy as np
        lows = df["low"].values
        recent_close = float(df["close"].iloc[-1])

        length = len(lows)
        is_pivot = np.ones(length, dtype=bool)
        is_pivot[:n] = False
        is_pivot[length - n:] = False

        for offset in range(1, n + 1):
            is_pivot[n:length - n] &= lows[n:length - n] < lows[n - offset:length - n - offset]
            is_pivot[n:length - n] &= lows[n:length - n] < lows[n + offset:length - n + offset]

        # Filter to only pivots below recent close
        is_pivot &= lows < recent_close
        pivots = lows[is_pivot].tolist()

        if not pivots:
            return []
        pivots.sort()
        deduped = [pivots[0]]
        for p in pivots[1:]:
            if deduped[-1] == 0 or (p - deduped[-1]) / deduped[-1] > 0.02:
                deduped.append(p)
        return deduped

    # Single-candle wick-rejection checks — the one canonical implementation
    # lives in technical_analysis.patterns (TOPPING_TAIL_RATIO threshold).
    # Static aliases preserve the long-standing ResistanceMap API used by
    # the live stream and dynamic-level call sites.
    detect_topping_tail = staticmethod(_patterns_detect_topping_tail)
    detect_bottoming_tail = staticmethod(_patterns_detect_bottoming_tail)

