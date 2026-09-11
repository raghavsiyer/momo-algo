"""
technical_analysis/indicators.py

All technical indicator calculations. Pure functions — take a DataFrame in,
return a DataFrame out with new columns appended. No API calls here.

Indicators computed per timeframe:

    10-sec / 1-min:
        ema_9

    1 / 2 / 5 / 15-min:
        ema_20, vwap, ma_50

    5-min:
        bb_upper, bb_middle, bb_lower  (Bollinger Bands)

    Daily:
        ema_200, ma_50, ma_200

    Any timeframe:
        macd_line, macd_signal, macd_hist  (1-min and 5-min per workflow)
        roc                                (10-sec per workflow)
        rvol                               (5-min and daily)

All functions expect a DataFrame with at minimum:
    ts      — datetime index or column (tz-aware UTC)
    open, high, low, close, volume  — float/int columns

VWAP resets at each ET session boundary (04:00, 09:30, 16:00, 20:00).
"""

import numpy as np
import pandas as pd

from config import (
    EMA_9_PERIOD, EMA_20_PERIOD, EMA_50_PERIOD,
    EMA_200_PERIOD, MA_50_PERIOD, MA_200_PERIOD,
    BB_PERIOD, BB_STD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    ROC_PERIOD,
    BARS_PER_SESSION_1MIN, BARS_PER_SESSION_5MIN,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Set ts as DatetimeIndex if it isn't already."""
    df = df.copy()
    if "ts" in df.columns:
        df = df.set_index("ts")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def _restore_ts_column(df: pd.DataFrame) -> pd.DataFrame:
    """Move the index back to a ts column."""
    df = df.reset_index()
    if "index" in df.columns:
        df = df.rename(columns={"index": "ts"})
    return df


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def add_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    """
    Append an EMA column named f'ema_{period}'.
    Works on any timeframe — pass the correct period for the chart.
    Uses pandas ewm with adjust=False (standard recursive EMA).
    """
    df = df.copy()
    df[f"ema_{period}"] = (
        df[col].ewm(span=period, adjust=False).mean()
    )
    return df


def add_ema_9(df: pd.DataFrame) -> pd.DataFrame:
    """9 EMA — for 10-sec and 1-min charts."""
    return add_ema(df, EMA_9_PERIOD)


def add_ema_20(df: pd.DataFrame) -> pd.DataFrame:
    """20 EMA — for 1/2/5/15-min charts."""
    return add_ema(df, EMA_20_PERIOD)


def add_ema_200(df: pd.DataFrame) -> pd.DataFrame:
    """200 EMA — for daily chart."""
    return add_ema(df, EMA_200_PERIOD)


# ---------------------------------------------------------------------------
# Simple Moving Average
# ---------------------------------------------------------------------------

def add_ma(df: pd.DataFrame, period: int, col: str = "close") -> pd.DataFrame:
    """Append a simple MA column named f'ma_{period}'."""
    df = df.copy()
    df[f"ma_{period}"] = df[col].rolling(window=period).mean()
    return df


def add_ma_50(df: pd.DataFrame) -> pd.DataFrame:
    """50 MA — for 1/2/5/15-min and daily charts."""
    return add_ma(df, MA_50_PERIOD)


def add_ma_200(df: pd.DataFrame) -> pd.DataFrame:
    """200 MA — for daily chart."""
    return add_ma(df, MA_200_PERIOD)


# ---------------------------------------------------------------------------
# VWAP  (session-anchored, ET session windows)
# ---------------------------------------------------------------------------

def _session_vwap_groups(index: pd.DatetimeIndex) -> pd.Series:
    """Return group keys for session VWAP resets in ET.

    Session reset windows follow:
      04:00, 09:30, 16:00, 20:00 ET

    Bars from 00:00-04:00 ET are grouped into the prior day's 20:00 session.
    """
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    et_idx = idx.tz_convert("America/New_York")

    et_minutes = et_idx.hour * 60 + et_idx.minute
    session_idx = np.select(
        [
            et_minutes <= 240,
            et_minutes <= 570,
            et_minutes <= 960,
            et_minutes <= 1200,
        ],
        [3, 0, 1, 2],
        default=3,
    ).astype(int)

    session_day = et_idx.normalize()
    overnight_mask = et_minutes <= 240
    session_day = session_day - pd.to_timedelta(overnight_mask.astype(int), unit="D")

    day_key = pd.Series(session_day.strftime("%Y-%m-%d"), index=index)
    idx_key = pd.Series(session_idx.astype(str), index=index)
    return day_key + "-" + idx_key

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a 'vwap' column.

    Anchored VWAP: resets at ET session boundaries (04:00, 09:30, 16:00, 20:00).
    Formula: cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3

    For intraday use. Pass daily bars only if you want a daily VWAP
    (less useful — VWAP is an intraday concept).
    """
    df = df.copy()
    df = _ensure_ts_index(df)

    tp      = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol  = tp * df["volume"]

    session_s = _session_vwap_groups(pd.DatetimeIndex(df.index))
    df["vwap"] = tp_vol.groupby(session_s).cumsum() / df["volume"].groupby(session_s).cumsum()

    df = _restore_ts_column(df)
    return df


# ---------------------------------------------------------------------------
# Bollinger Bands  (5-min)
# ---------------------------------------------------------------------------

def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = BB_PERIOD,
    std_dev: float = BB_STD,
) -> pd.DataFrame:
    """
    Append bb_middle, bb_upper, bb_lower columns.
    bb_middle = SMA(close, period)
    bb_upper  = bb_middle + std_dev * rolling_std
    bb_lower  = bb_middle - std_dev * rolling_std
    Uses population std (ddof=0) — the standard Bollinger definition.
    """
    df = df.copy()
    rolling = df["close"].rolling(window=period)
    df["bb_middle"] = rolling.mean()
    df["bb_std"]    = rolling.std(ddof=0)
    df["bb_upper"]  = df["bb_middle"] + std_dev * df["bb_std"]
    df["bb_lower"]  = df["bb_middle"] - std_dev * df["bb_std"]
    df = df.drop(columns=["bb_std"])
    return df


# ---------------------------------------------------------------------------
# MACD  (1-min and 5-min)
# ---------------------------------------------------------------------------

def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Append macd_line, macd_signal, macd_hist columns.
    macd_line   = EMA(fast) - EMA(slow)
    macd_signal = EMA(macd_line, signal)
    macd_hist   = macd_line - macd_signal
    Positive hist = bullish momentum. Negative = bearish.
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd_line"]   = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"]   = df["macd_line"] - df["macd_signal"]
    return df


# ---------------------------------------------------------------------------
# ROC — Rate of Change  (10-sec)
# ---------------------------------------------------------------------------

def add_roc(df: pd.DataFrame, period: int = ROC_PERIOD) -> pd.DataFrame:
    """
    Append 'roc' column: percentage change over *period* bars.
    roc = (close - close[n]) / close[n] * 100

    On 10-sec candles with period=1 this is the bar-over-bar % change,
    used to detect momentum (ROC < 5 for 6 consecutive bars = weak).
    """
    df = df.copy()
    df["roc"] = df["close"].pct_change(periods=period) * 100
    return df


# ---------------------------------------------------------------------------
# RVol — Relative Volume
# ---------------------------------------------------------------------------

def add_rvol(df: pd.DataFrame, avg_volume: int) -> pd.DataFrame:
    """
    Append 'rvol' column: each bar's volume / avg_volume.
    avg_volume comes from the Tradier quote (30-day average daily volume).

    For 5-min bars, avg_volume should be the avg daily volume divided by
    the number of 5-min bars in a session (BARS_PER_SESSION_5MIN for a 6.5-hr session):
        avg_5min_volume = avg_daily_volume / BARS_PER_SESSION_5MIN

    Pass the appropriate avg for the timeframe you're working with.
    """
    df = df.copy()
    df["rvol"] = df["volume"] / avg_volume if avg_volume > 0 else 0.0
    return df


def compute_rvol_current(df: pd.DataFrame, avg_volume: int) -> float:
    """
    Return the RVol of the most recent bar only.
    Convenience function for the technical filter gate.
    """
    if df.empty or avg_volume <= 0:
        return 0.0
    return float(df["volume"].iloc[-1] / avg_volume)


def compute_rvol_5min_bars(df_5min: pd.DataFrame, avg_daily_volume: int) -> float:
    """
    Most recent 5-min bar's volume vs avg_daily / BARS_PER_SESSION_5MIN
    (30-day benchmark). Used after indicator DataFrames are built.

    See also: evaluator.py:_compute_5min_rvol() (relative-to-recent-self),
              confidence_scorer.py:_compute_rvol_5min() (live T&S ticks).
    """
    if df_5min.empty or avg_daily_volume <= 0:
        return 0.0
    avg_per_bar = avg_daily_volume / BARS_PER_SESSION_5MIN
    return float(df_5min["volume"].iloc[-1] / avg_per_bar) if avg_per_bar > 0 else 0.0


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------

def compute_spread(bid: float, ask: float) -> float:
    """Absolute bid-ask spread."""
    return round(ask - bid, 4)


# ---------------------------------------------------------------------------
# Linear regression — leg smoothness (R²) and slope
# ---------------------------------------------------------------------------

def linreg_r2_slope(values: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """Least-squares fit of *values* against x = [0, 1, …, n-1].

    Returns ``(r2, slope)``.

    * ``r2`` ∈ [0, 1] — coefficient of determination (1.0 = perfectly straight line).
    * ``slope`` — change in *value* per unit x.

    Returns ``(0.0, 0.0)`` if fewer than 2 points or all values identical.
    Used for "leg smoothness" — how cleanly a price leg ramps in one direction.
    """
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    y = np.asarray(values, dtype=np.float64)
    mean_y = y.mean()
    ss_xx = n * (n * n - 1) / 12.0  # sum((i - mean_x)^2) for i=0..n-1
    if ss_xx <= 0:
        return 0.0, 0.0
    mean_x = (n - 1) / 2.0
    x = np.arange(n, dtype=np.float64) - mean_x
    dy = y - mean_y
    ss_xy = np.dot(x, dy)
    ss_tot = np.dot(dy, dy)
    if ss_tot <= 0:
        return 0.0, 0.0
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    predicted = intercept + slope * np.arange(n, dtype=np.float64)
    residuals = y - predicted
    ss_res = np.dot(residuals, residuals)
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    return r2, slope


def signed_volume_weighted_smoothness(
    values: list[float] | tuple[float, ...],
    volumes: list[float] | tuple[float, ...],
) -> float:
    """Volume-weighted R² signed by slope direction.  Range [-1, +1]."""
    n = len(values)
    if n != len(volumes) or n < 2:
        return signed_smoothness(values)

    y_all = np.asarray(values, dtype=np.float64)
    w_all = np.asarray(volumes, dtype=np.float64)
    mask = np.isfinite(y_all) & np.isfinite(w_all) & (w_all > 0.0)
    if int(mask.sum()) < 2:
        return signed_smoothness(values)

    x = np.arange(n, dtype=np.float64)[mask]
    y = y_all[mask]
    w = w_all[mask]
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return signed_smoothness(values)

    mean_x = float(np.dot(w, x) / w_sum)
    mean_y = float(np.dot(w, y) / w_sum)
    dx = x - mean_x
    dy = y - mean_y
    ss_xx = float(np.dot(w, dx * dx))
    ss_tot = float(np.dot(w, dy * dy))
    if ss_xx <= 0.0 or ss_tot <= 0.0:
        return 0.0

    slope = float(np.dot(w, dx * dy) / ss_xx)
    intercept = mean_y - slope * mean_x
    predicted = intercept + slope * x
    residuals = y - predicted
    ss_res = float(np.dot(w, residuals * residuals))
    r2 = min(1.0, max(0.0, 1.0 - ss_res / ss_tot))
    if slope > 0.0:
        return r2
    if slope < 0.0:
        return -r2
    return 0.0


def signed_smoothness(values: list[float] | tuple[float, ...]) -> float:
    """R² signed by slope direction.  Range [-1, +1].

    +R² → clean upward leg.  -R² → clean downward leg.  0 → noise / flat.
    Use with ``QualityFn.LINEAR_RANGE`` (lo=-0.5, hi=0.85) to give clean
    up-legs high quality and clean down-legs low quality.
    """
    r2, slope = linreg_r2_slope(values)
    if slope > 0:
        return r2
    if slope < 0:
        return -r2
    return 0.0


# ---------------------------------------------------------------------------
# In-place helpers (used by build_* composites to avoid redundant copies)
# ---------------------------------------------------------------------------

def _add_ema_ip(df: pd.DataFrame, period: int, col: str = "close") -> None:
    df[f"ema_{period}"] = df[col].ewm(span=period, adjust=False).mean()

def _add_ma_ip(df: pd.DataFrame, period: int, col: str = "close") -> None:
    df[f"ma_{period}"] = df[col].rolling(window=period).mean()

def _add_vwap_ip(df: pd.DataFrame) -> None:
    needs_index = "ts" in df.columns
    if needs_index:
        df.set_index("ts", inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = tp * df["volume"]
    session_s = _session_vwap_groups(pd.DatetimeIndex(df.index))
    cum_tp_vol = tp_vol.groupby(session_s).cumsum()
    cum_vol = df["volume"].groupby(session_s).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol
    if needs_index:
        df.reset_index(inplace=True)
        if "index" in df.columns:
            df.rename(columns={"index": "ts"}, inplace=True)

def _add_bb_ip(df: pd.DataFrame, period: int = BB_PERIOD, std_dev: float = BB_STD) -> None:
    rolling = df["close"].rolling(window=period)
    mid = rolling.mean()
    std = rolling.std(ddof=0)  # population std — standard Bollinger definition
    df["bb_middle"] = mid
    df["bb_upper"] = mid + std_dev * std
    df["bb_lower"] = mid - std_dev * std

def _add_macd_ip(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> None:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_line.ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

def _add_roc_ip(df: pd.DataFrame, period: int = ROC_PERIOD) -> None:
    df["roc"] = df["close"].pct_change(periods=period) * 100

def _add_rvol_ip(df: pd.DataFrame, avg_volume: int) -> None:
    df["rvol"] = df["volume"] / avg_volume if avg_volume > 0 else 0.0


# ---------------------------------------------------------------------------
# Composite: build full indicator set per timeframe
# ---------------------------------------------------------------------------

def build_10sec(df: pd.DataFrame) -> pd.DataFrame:
    """10-sec candles: ema_9, roc."""
    df = df.copy()
    _add_ema_ip(df, EMA_9_PERIOD)
    _add_roc_ip(df)
    return df


def build_1min(df: pd.DataFrame, avg_daily_volume: int = 0) -> pd.DataFrame:
    """1-min candles: ema_9, ema_20, vwap, ma_50, macd, rvol."""
    df = df.copy()
    _add_ema_ip(df, EMA_9_PERIOD)
    _add_ema_ip(df, EMA_20_PERIOD)
    _add_vwap_ip(df)
    _add_ma_ip(df, MA_50_PERIOD)
    _add_macd_ip(df)
    if avg_daily_volume > 0:
        _add_rvol_ip(df, avg_daily_volume // BARS_PER_SESSION_1MIN)
    return df


def build_5min(df: pd.DataFrame, avg_daily_volume: int = 0) -> pd.DataFrame:
    """5-min candles: ema_20, vwap, ma_50, bb, macd, rvol."""
    df = df.copy()
    _add_ema_ip(df, EMA_20_PERIOD)
    _add_vwap_ip(df)
    _add_ma_ip(df, MA_50_PERIOD)
    _add_bb_ip(df)
    _add_macd_ip(df)
    if avg_daily_volume > 0:
        _add_rvol_ip(df, avg_daily_volume // BARS_PER_SESSION_5MIN)
    return df


def build_15min(df: pd.DataFrame) -> pd.DataFrame:
    """15-min candles: ema_20, vwap, ma_50."""
    df = df.copy()
    _add_ema_ip(df, EMA_20_PERIOD)
    _add_vwap_ip(df)
    _add_ma_ip(df, MA_50_PERIOD)
    return df


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily candles: ema_200, ma_50, ma_200."""
    df = df.copy()
    _add_ema_ip(df, EMA_200_PERIOD)
    _add_ma_ip(df, MA_50_PERIOD)
    _add_ma_ip(df, MA_200_PERIOD)
    return df