"""
technical_analysis/moving_average_levels.py

Moving-average levels for the resistance/support map: every (kind, period)
in config.MA_LEVEL_SPECS (200 EMA, 200 MA, 50 MA by default) on every
timeframe in config.MA_LEVEL_TIMEFRAMES (5m through monthly).

Conventions, chosen to match an extended-hours chart:
  * Intraday bars include pre/post market — IBKR useRTH=False, Tradier
    session_filter=all — the same session the rest of the system charts.
  * 5m/15m/30m/1h buckets align to the clock. 2h/4h buckets align to ET
    midnight (4h: 04:00, 08:00, 12:00, 16:00 ET) and are built from 1h bars,
    so their alignment never depends on how a provider anchors multi-hour
    bars. Daily/weekly/monthly buckets are the ET session date, the ISO week
    (keyed by its Monday) and the calendar month.
  * EMA is seeded with the SMA of its first ``period`` closed bars, as Pine's
    ta.ema does. Neither MA is marked until ``period`` closed bars exist.
  * The marked value includes the forming bar, as a chart's MA line does at
    the right edge: SMA = mean(last period-1 closed closes + forming close),
    EMA = ema_closed + alpha * (forming close - ema_closed).
  * Live: the scorer feeds every closed 1m bar. Each timeframe updates its
    forming bar's close and, at its bucket boundary, rolls that bar into the
    closed series. Bars at or before the last closed bucket are ignored, so a
    replay that starts earlier than the snapshot cannot rewind it.

Fetching (``fetch_ma_frames``) asks the historical client for each source
interval once, via ``get_candles_interval`` (an empty answer gets one retry
at half the lookback): IBKR serves every interval natively; Tradier has no
30-minute or 1-hour bars, so those derive from its 15-minute bars (and are
therefore limited to its ~18 days of history). The same derivation covers
an IBKR 30-minute or 1-hour request that comes back empty.
Daily/weekly/monthly results are memoised per symbol per ET date — a closed
bar cannot change within the session.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd

from config import (
    MA_LEVEL_FETCH_DAYS,
    MA_LEVEL_SPECS,
    MA_LEVEL_TIMEFRAMES,
    MarketClock,
)

logger = logging.getLogger(__name__)

_ET = MarketClock.ET

# Clock-aligned intraday buckets (seconds). ET's UTC offset is a whole number
# of hours, so flooring in UTC gives the same buckets as flooring in ET.
_CLOCK_SECS: dict[str, int] = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
# Multi-hour buckets anchored at ET midnight (hours per bucket).
_ET_HOUR_BLOCKS: dict[str, int] = {"2h": 2, "4h": 4}
_SESSION_TFS = ("daily", "weekly", "monthly")

# Source interval each timeframe is computed from.
SOURCE_INTERVAL: dict[str, str] = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "2h": "1hour",
    "4h": "1hour",
    "daily": "1day",
    "weekly": "1week",
    "monthly": "1month",
}
# Where an interval comes from when the provider has no native bars for it.
_DERIVE_FROM: dict[str, str] = {"30min": "15min", "1hour": "15min"}
# Fetch order: a derivation source always precedes what derives from it.
_INTERVAL_ORDER = ("5min", "15min", "30min", "1hour", "1day", "1week", "1month")
# Bars stamped with their session date (midnight UTC) rather than a time.
_SESSION_STAMPED = frozenset({"1day", "1week", "1month"})
# An empty answer gets one retry with half the lookback: the longest duration
# a provider accepts per bar size is provider-specific (and IBKR's is not
# documented consistently), and a rejected request comes back empty.
_RETRY_SHORTER = frozenset(_INTERVAL_ORDER)

_TF_LABEL: dict[str, str] = {tf: tf for tf in ("5m", "15m", "30m", "1h", "2h", "4h")}
_TF_LABEL.update({"daily": "daily", "weekly": "weekly", "monthly": "monthly"})


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

def bucket_key(timeframe: str, ts: datetime, *, session_date: Optional[date] = None):
    """Bucket a timestamp falls in on *timeframe*.

    Intraday timeframes return the bucket's start as an aware UTC datetime;
    daily/weekly/monthly return a ``date`` (session date, ISO-week Monday,
    first of month). *session_date* overrides the ET date for bars that are
    stamped with their session date at midnight UTC.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    secs = _CLOCK_SECS.get(timeframe)
    if secs is not None:
        epoch = ts.timestamp()
        return datetime.fromtimestamp(epoch - (epoch % secs), tz=timezone.utc)
    hours = _ET_HOUR_BLOCKS.get(timeframe)
    if hours is not None:
        et = ts.astimezone(_ET)
        start = et.replace(hour=et.hour - et.hour % hours, minute=0, second=0, microsecond=0)
        return start.astimezone(timezone.utc)
    day = session_date if session_date is not None else ts.astimezone(_ET).date()
    if timeframe == "daily":
        return day
    if timeframe == "weekly":
        return day - timedelta(days=day.weekday())
    if timeframe == "monthly":
        return day.replace(day=1)
    raise ValueError(f"unknown timeframe {timeframe!r}")


def bucket_bars(df: pd.DataFrame, timeframe: str, *, session_stamped: bool = False) -> pd.DataFrame:
    """Collapse OHLCV rows into one close per *timeframe* bucket.

    Returns a DataFrame with ``bucket`` and ``close`` columns, oldest first.
    Bars at the same or a finer resolution both work: native bars map one to
    one, finer bars keep the last close in each bucket.
    """
    if df is None or df.empty or "ts" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(columns=["bucket", "close"])
    frame = pd.DataFrame({
        "ts": pd.to_datetime(df["ts"], utc=True, errors="coerce"),
        "close": pd.to_numeric(df["close"], errors="coerce"),
    }).dropna()
    frame = frame[frame["close"] > 0].sort_values("ts", kind="stable")
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "close"])
    keys = []
    for ts in frame["ts"]:
        py = ts.to_pydatetime()
        keys.append(bucket_key(
            timeframe, py, session_date=py.date() if session_stamped else None,
        ))
    frame["bucket"] = keys
    out = frame.groupby("bucket", sort=True)["close"].last().reset_index()
    return out[["bucket", "close"]]


# ---------------------------------------------------------------------------
# Per-timeframe MA state
# ---------------------------------------------------------------------------

class TimeframeMA:
    """Closed-bar moving-average state for one timeframe.

    Holds the last ``max_period`` closed closes (for SMAs), one running EMA
    per EMA period, and the forming bar. Pure Python and picklable: it lives
    on the ResistanceMap, which is cached to disk and deep-copied by replays.
    """

    def __init__(self, timeframe: str, specs: Iterable[tuple[str, int]]) -> None:
        self.timeframe = timeframe
        self.specs: tuple[tuple[str, int], ...] = tuple((k.lower(), int(p)) for k, p in specs)
        self.max_period = max((p for _, p in self.specs), default=1)
        self.closes: deque[float] = deque(maxlen=self.max_period)
        self.n_closed = 0
        self.ema: dict[int, Optional[float]] = {p: None for k, p in self.specs if k == "ema"}
        self.last_closed_key = None
        self.forming_key = None
        self.forming_close: Optional[float] = None

    # -- construction -----------------------------------------------------

    @classmethod
    def from_bars(
        cls,
        timeframe: str,
        specs: Iterable[tuple[str, int]],
        bars: pd.DataFrame,
        now: datetime,
        last_price: Optional[float] = None,
    ) -> "TimeframeMA":
        """Build from ``bucket_bars`` output as of *now*.

        Buckets before the current one are closed. A bar in the current
        bucket becomes the forming bar, priced at *last_price* when given
        (fresher than the fetch). No current-bucket bar means no forming bar:
        nothing has traded in it yet, so none is invented.
        """
        ma = cls(timeframe, specs)
        now_key = bucket_key(timeframe, now)
        forming_close: Optional[float] = None
        for key, close in zip(bars["bucket"], bars["close"]):
            key = key.to_pydatetime() if hasattr(key, "to_pydatetime") else key
            if key < now_key:
                ma.push_closed(key, float(close))
            elif key == now_key:
                forming_close = float(close)
        if forming_close is not None:
            ma.forming_key = now_key
            ma.forming_close = float(last_price) if last_price and last_price > 0 else forming_close
        return ma

    # -- updates ----------------------------------------------------------

    def push_closed(self, key, close: float) -> None:
        """Append one closed bar and advance every EMA."""
        self.closes.append(close)
        self.n_closed += 1
        self.last_closed_key = key
        for period, value in self.ema.items():
            if value is None:
                if self.n_closed >= period:
                    recent = list(self.closes)[-period:]
                    self.ema[period] = sum(recent) / period
            else:
                self.ema[period] = value + (2.0 / (period + 1)) * (close - value)

    def on_bar(self, key, close: float) -> bool:
        """Fold in a finer closed bar whose bucket on this timeframe is *key*.

        Returns True when the state changed.
        """
        if close <= 0:
            return False
        if self.last_closed_key is not None and key <= self.last_closed_key:
            return False
        if self.forming_key is None:
            self.forming_key, self.forming_close = key, close
            return True
        if key == self.forming_key:
            self.forming_close = close
            return True
        if key < self.forming_key:
            return False
        self.push_closed(self.forming_key, float(self.forming_close))
        self.forming_key, self.forming_close = key, close
        return True

    # -- reads --------------------------------------------------------------

    def value(self, kind: str, period: int) -> Optional[float]:
        """Current MA value including the forming bar; None until warmed."""
        if self.n_closed < period:
            return None
        kind = kind.lower()
        forming = self.forming_close
        if kind == "sma":
            recent = list(self.closes)[-period:]
            if forming is not None:
                return (sum(recent[1:]) + forming) / period
            return sum(recent) / period
        if kind == "ema":
            ema = self.ema.get(period)
            if ema is None:
                return None
            if forming is not None:
                return ema + (2.0 / (period + 1)) * (forming - ema)
            return ema
        return None


def ma_level_source(kind: str, period: int, timeframe: str) -> str:
    """Display/source label, e.g. ``200 EMA (1h)`` or ``50 MA (daily)``."""
    name = "EMA" if kind.lower() == "ema" else "MA"
    return f"{period} {name} ({_TF_LABEL.get(timeframe, timeframe)})"


def build_trackers(
    frames: dict[str, pd.DataFrame],
    now: datetime,
    last_price: Optional[float] = None,
    specs: Iterable[tuple[str, int]] = MA_LEVEL_SPECS,
) -> dict[str, TimeframeMA]:
    """One TimeframeMA per timeframe present in *frames* (``bucket_bars`` output)."""
    specs = tuple(specs)
    trackers: dict[str, TimeframeMA] = {}
    for tf, bars in frames.items():
        if bars is None or bars.empty:
            continue
        trackers[tf] = TimeframeMA.from_bars(tf, specs, bars, now, last_price)
    return trackers


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

_memo_lock = threading.Lock()
_memo_day: Optional[date] = None
_memo: dict[tuple[str, str], pd.DataFrame] = {}


def _memo_get(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    global _memo_day
    today = MarketClock.now_et().date()
    with _memo_lock:
        if _memo_day != today:
            _memo.clear()
            _memo_day = today
        return _memo.get((symbol, interval))


def _memo_put(symbol: str, interval: str, df: pd.DataFrame) -> None:
    with _memo_lock:
        _memo[(symbol, interval)] = df


def _fetch_interval(client, symbol: str, interval: str) -> pd.DataFrame:
    fetch = getattr(client, "get_candles_interval", None)
    if fetch is None:
        return pd.DataFrame()
    memoised = interval in _SESSION_STAMPED
    if memoised:
        cached = _memo_get(symbol, interval)
        if cached is not None:
            return cached
    days = int(MA_LEVEL_FETCH_DAYS.get(interval, 30))
    df = pd.DataFrame()
    for attempt_days in ((days, max(1, days // 2)) if interval in _RETRY_SHORTER else (days,)):
        try:
            df = fetch(symbol, interval, attempt_days)
        except Exception as exc:
            logger.debug("[%s] MA fetch %s (%dd) failed: %s", symbol, interval, attempt_days, exc)
            df = pd.DataFrame()
        if df is not None and not df.empty:
            break
    df = df if df is not None else pd.DataFrame()
    if memoised and not df.empty:
        _memo_put(symbol, interval, df)
    return df


def fetch_ma_frames(
    client,
    symbol: str,
    timeframes: Iterable[str] = MA_LEVEL_TIMEFRAMES,
) -> dict[str, pd.DataFrame]:
    """Fetch and bucket the bars every MA timeframe needs.

    Returns ``{timeframe: bucket_bars(...)}`` for each timeframe that got
    data. A client without ``get_candles_interval`` yields ``{}``.
    """
    timeframes = [tf for tf in timeframes if tf in SOURCE_INTERVAL]
    if not timeframes or getattr(client, "get_candles_interval", None) is None:
        return {}
    needed = {SOURCE_INTERVAL[tf] for tf in timeframes}
    for interval in list(needed):
        base = _DERIVE_FROM.get(interval)
        if base is not None:
            needed.add(base)  # fetched first so a missing native interval can derive
    raw: dict[str, pd.DataFrame] = {}
    for interval in _INTERVAL_ORDER:
        if interval not in needed:
            continue
        df = _fetch_interval(client, symbol, interval)
        if df.empty and interval in _DERIVE_FROM:
            df = raw.get(_DERIVE_FROM[interval], pd.DataFrame())
        raw[interval] = df
    frames: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        interval = SOURCE_INTERVAL[tf]
        bars = bucket_bars(raw.get(interval), tf, session_stamped=interval in _SESSION_STAMPED)
        if not bars.empty:
            frames[tf] = bars
    logger.debug(
        "[%s] MA frames: %s", symbol,
        ", ".join(f"{tf}={len(df)}" for tf, df in frames.items()) or "none",
    )
    return frames
