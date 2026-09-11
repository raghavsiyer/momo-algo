"""
_squeeze_analysis_scratch.py — research scratch (not part of the live engine)

Scientific analysis of tick-by-tick T&S data for ASTC (8:30-9:29 ET)
and AMSS (7:03-7:59 ET) on 2026-05-27.

Goal: Identify what T&S variable levels/combinations uniquely signal
the beginning of squeeze moves. Build 10-second and 1-minute candles,
compute order flow metrics, and compare pre-squeeze vs squeeze vs
non-squeeze regimes.

Run:  python _squeeze_analysis_scratch.py
"""

from __future__ import annotations

import logging
import math
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_data.provider import get_historical_client
from config import MarketClock, TS_MIN_NOTIONAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ET = MarketClock.ET
LOCAL_TZ = MarketClock.local_tz()


# ─── DATA FETCH ───────────────────────────────────────────────────────────────

def fetch_ticks(symbol: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """Fetch tick data exactly like backtest.py: full range first, 30-min chunked fallback on overflow.

    Normalizes bid/ask/size the same way BacktestRunner._iter_ticks does:
      - Synthesizes bid/ask from price (0.2% spread) when missing/zero
      - Sets size=max(1, raw_size) so zero-size ticks count as 1 share
      - Filters out-of-order timestamps
    """
    client = get_historical_client()

    # Try full range first (same as backtest._fetch_ticks)
    rows = client.get_timesales_tick(symbol, start_utc, end_utc, session_filter="all")
    if not rows:
        # 502 body overflow — split into 30-minute chunks (same as backtest)
        logger.info(f"[{symbol}] Chunked fallback (body overflow)")
        rows = []
        cursor = start_utc
        while cursor < end_utc:
            chunk_end = min(cursor + timedelta(minutes=30), end_utc)
            chunk = client.get_timesales_tick(symbol, cursor, chunk_end, session_filter="all")
            rows.extend(chunk)
            cursor = chunk_end

    if not rows:
        logger.warning(f"[{symbol}] No ticks fetched!")
        return pd.DataFrame()

    # Normalize rows exactly like BacktestRunner._iter_ticks
    normalized = []
    last_ts = None
    for row in rows:
        ts = row["ts"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        # Filter to window
        if ts < start_utc or ts > end_utc:
            continue
        # Skip out-of-order timestamps
        if last_ts is not None and ts < last_ts:
            continue
        last_ts = ts

        px = float(row.get("price", 0.0) or 0.0)
        if px <= 0:
            continue

        # Skip cancel/correction prints (same as DynamicScorer._on_timesale)
        if row.get("cancel", False) or row.get("correction", False):
            continue

        # Synthesize bid/ask when missing (backtest pattern)
        bid = float(row.get("bid") or 0.0)
        ask = float(row.get("ask") or 0.0)
        if bid <= 0 or ask <= 0:
            spread = max(px * 0.002, 0.0001)
            bid = px - spread / 2.0
            ask = px + spread / 2.0

        # Size = max(1, raw) so every print counts
        raw_size = int(row.get("size", 0) or 0)
        size = max(1, raw_size)

        # Filter sub-notional noise prints (same as DynamicScorer._on_timesale)
        # Only apply when we have real size data — pre-market historical ticks
        # report size=0 but are still real executions.
        if raw_size > 0 and px * size < TS_MIN_NOTIONAL:
            continue

        normalized.append({
            "ts": ts,
            "price": px,
            "size": size,
            "bid": bid,
            "ask": ask,
            "exchange": str(row.get("exchange", "") or ""),
            "session": str(row.get("session", "") or ""),
        })

    if not normalized:
        logger.warning(f"[{symbol}] No valid ticks after normalization!")
        return pd.DataFrame()

    df = pd.DataFrame(normalized)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    logger.info(f"[{symbol}] Fetched {len(df)} ticks from {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")
    return df


# ─── CANDLE CONSTRUCTION ──────────────────────────────────────────────────────

def build_candles(df: pd.DataFrame, freq: str = "10s") -> pd.DataFrame:
    """Resample ticks into OHLCV candles at the given frequency."""
    if df.empty:
        return pd.DataFrame()

    ohlc = df.set_index("ts")["price"].resample(freq).ohlc().dropna()
    vol = df.set_index("ts")["size"].resample(freq).sum()
    count = df.set_index("ts")["price"].resample(freq).count()

    candles = ohlc.join(vol.rename("volume")).join(count.rename("tick_count")).dropna()
    candles["range"] = candles["high"] - candles["low"]
    candles["body"] = candles["close"] - candles["open"]
    candles["body_pct"] = candles["body"] / candles["open"] * 100
    candles["upper_wick"] = candles["high"] - candles[["open", "close"]].max(axis=1)
    candles["lower_wick"] = candles[["open", "close"]].min(axis=1) - candles["low"]
    candles["pct_change"] = candles["close"].pct_change() * 100
    return candles


# ─── T&S METRICS (rolling window) ────────────────────────────────────────────

def compute_ts_metrics(
    df: pd.DataFrame,
    window_secs: int = 10,
    ema_speed_tau: float = 3.0,
    ema_accel_fast_tau: float = 3.0,
    ema_accel_slow_tau: float = 10.0,
    ema_imbalance_tau: float = 10.0,
) -> pd.DataFrame:
    """Compute rolling T&S metrics for each tick.

    Uses the same bid/ask classification as DynamicScorer (price >= ask = buy,
    price <= bid = sell) since fetch_ticks now guarantees valid bid/ask.

    Metrics (rolling over window_secs):
      - speed: prints/sec
      - volume_speed: shares/sec
      - avg_size: avg print size
      - buy_ratio: fraction of buy-initiated prints (price >= ask)
      - sell_ratio: fraction of sell-initiated prints (price <= bid)
      - imbalance: (buy_vol - sell_vol) / (buy_vol + sell_vol)
      - tick_imbalance: (uptick_count - downtick_count) / total in window
      - momentum: net price change / window range
      - price_roc_3, roc_10, roc_30: rate of change over N prints (%)
      - acceleration: (recent_half - older_half)/(recent_half + older_half) directional prints, [-1, 1] (matches live pressure_acceleration)
      - volatility: stdev of tick-to-tick returns in window
      - range_pct: (high - low) / low in window as %
      - streak: consecutive uptick/downtick counter

    EMA (time-decayed) research variants — offline only; the live
    OrderFlowTracker is unchanged. Each print decays the running accumulators
    by exp(-Δt/τ) then adds its own contribution; for a steady Poisson tape
    E[c] = λ·τ so c/τ estimates the instantaneous directional-print rate:
      - speed_ema: directional prints/sec via an exponential kernel @ ema_speed_tau
      - acceleration_ema: MACD-style balanced ratio of a fast vs slow decayed
        rate, (r_fast - r_slow) / (r_fast + r_slow), bounded [-1, 1]
      - acceleration_ema_z: acceleration_ema * sqrt(N_eff) significance, where
        N_eff = (Σw)² / Σw² (the exponential-kernel analog of accel * sqrt(N))
      - imbalance_ema: decayed buy/sell volume ratio @ ema_imbalance_tau, [-1, 1]
    """
    if df.empty:
        return df

    df = df.copy()
    n_total = len(df)

    # Trade classification: tick rule (Lee & Ready 1991).
    # When historical NBBO is unavailable (pre-market), uptick = buy, downtick = sell.
    # When real bid/ask exist, use NBBO (price >= ask = buy, price <= bid = sell).
    df["price_diff"] = df["price"].diff().fillna(0)
    df["tick_dir"] = 0
    df.loc[df["price_diff"] > 0, "tick_dir"] = 1
    df.loc[df["price_diff"] < 0, "tick_dir"] = -1

    # Check if we have real NBBO (not synthesized from price ± spread)
    # Real NBBO: bid and ask differ meaningfully from price and from each other
    has_real_nbbo = (
        df["bid"].notna() & df["ask"].notna() &
        (df["ask"] - df["bid"] > df["price"] * 0.003)  # spread > 0.3% = likely real
    )

    # Default to tick rule
    df["side"] = "neutral"
    df.loc[df["tick_dir"] > 0, "side"] = "buy"
    df.loc[df["tick_dir"] < 0, "side"] = "sell"

    # Override with NBBO where available (RTH data)
    if has_real_nbbo.any():
        df.loc[has_real_nbbo & (df["price"] >= df["ask"]), "side"] = "buy"
        df.loc[has_real_nbbo & (df["price"] <= df["bid"]), "side"] = "sell"
        logger.info(f"  Using NBBO classification for {has_real_nbbo.sum()}/{n_total} ticks")

    # Consecutive uptick/downtick streak
    streaks = []
    streak = 0
    for d in df["tick_dir"].values:
        if d > 0:
            streak = max(streak, 0) + 1
        elif d < 0:
            streak = min(streak, 0) - 1
        else:
            streak = 0
        streaks.append(streak)

    # Arrays for vectorized computation
    ts_arr = df["ts"].values.astype("datetime64[ns]")
    price_arr = df["price"].values.astype(np.float64)
    size_arr = df["size"].values.astype(np.float64)
    tick_dir_arr = df["tick_dir"].values.astype(np.float64)
    side_arr = df["side"].values
    is_buy = (side_arr == "buy").astype(np.float64)
    is_sell = (side_arr == "sell").astype(np.float64)
    buy_vol = is_buy * size_arr
    sell_vol = is_sell * size_arr

    window_ns = np.timedelta64(window_secs, "s")

    # Cumulative arrays for efficient window sums
    cum_size = np.cumsum(size_arr)
    cum_upticks = np.cumsum((tick_dir_arr > 0).astype(np.float64))
    cum_downticks = np.cumsum((tick_dir_arr < 0).astype(np.float64))
    cum_buy_count = np.cumsum(is_buy)
    cum_sell_count = np.cumsum(is_sell)
    cum_buy_vol = np.cumsum(buy_vol)
    cum_sell_vol = np.cumsum(sell_vol)

    # Output arrays
    speed_out = np.zeros(n_total)
    vol_speed_out = np.zeros(n_total)
    avg_size_out = np.zeros(n_total)
    buy_ratio_out = np.zeros(n_total)
    sell_ratio_out = np.zeros(n_total)
    imbalance_out = np.zeros(n_total)
    tick_imbal_out = np.zeros(n_total)
    momentum_out = np.zeros(n_total)
    volatility_out = np.zeros(n_total)
    range_pct_out = np.zeros(n_total)

    # Window start indices
    window_starts = np.searchsorted(ts_arr, ts_arr - window_ns, side="left")

    def _cum_window(cum_arr, j, i):
        return cum_arr[i] - (cum_arr[j - 1] if j > 0 else 0)

    for i in range(n_total):
        j = window_starts[i]
        n = i + 1 - j
        if n <= 0:
            continue

        elapsed_ns = (ts_arr[i] - ts_arr[j]).astype(np.float64)
        elapsed_secs = elapsed_ns / 1e9
        if elapsed_secs <= 0:
            elapsed_secs = 0.01

        # Speed
        speed_out[i] = n / elapsed_secs

        # Volume
        vol_sum = _cum_window(cum_size, j, i)
        vol_speed_out[i] = vol_sum / elapsed_secs
        avg_size_out[i] = vol_sum / n

        # Buy/sell ratios (NBBO-based, same as DynamicScorer)
        buys = _cum_window(cum_buy_count, j, i)
        sells = _cum_window(cum_sell_count, j, i)
        buy_ratio_out[i] = buys / n
        sell_ratio_out[i] = sells / n

        # Volume imbalance
        bv = _cum_window(cum_buy_vol, j, i)
        sv = _cum_window(cum_sell_vol, j, i)
        total_v = bv + sv
        imbalance_out[i] = (bv - sv) / total_v if total_v > 0 else 0.0

        # Tick imbalance
        upticks = _cum_window(cum_upticks, j, i)
        downticks = _cum_window(cum_downticks, j, i)
        tick_imbal_out[i] = (upticks - downticks) / n

        # Price momentum & range
        win_prices = price_arr[j:i + 1]
        p_high = win_prices.max()
        p_low = win_prices.min()
        p_range = p_high - p_low
        if p_range > 0:
            momentum_out[i] = (price_arr[i] - price_arr[j]) / p_range
            range_pct_out[i] = p_range / p_low * 100
        # Volatility
        if n >= 3:
            returns = np.diff(win_prices) / win_prices[:-1]
            volatility_out[i] = returns.std() * 100

    # ROC computations
    roc_3 = np.zeros(n_total)
    roc_10 = np.zeros(n_total)
    roc_30 = np.zeros(n_total)
    for i in range(3, n_total):
        roc_3[i] = (price_arr[i] / price_arr[i - 3] - 1) * 100
    for i in range(10, n_total):
        roc_10[i] = (price_arr[i] / price_arr[i - 10] - 1) * 100
    for i in range(30, n_total):
        roc_30[i] = (price_arr[i] / price_arr[i - 30] - 1) * 100

    result_df = pd.DataFrame({
        "ts": df["ts"].values,
        "price": price_arr,
        "size": size_arr,
        "side": side_arr,
        "tick_dir": tick_dir_arr,
        "streak": np.array(streaks, dtype=np.float64),
        "speed": speed_out,
        "volume_speed": vol_speed_out,
        "avg_size": avg_size_out,
        "buy_ratio": buy_ratio_out,
        "sell_ratio": sell_ratio_out,
        "imbalance": imbalance_out,
        "tick_imbalance": tick_imbal_out,
        "momentum": momentum_out,
        "volatility": volatility_out,
        "range_pct": range_pct_out,
        "roc_3": roc_3,
        "roc_10": roc_10,
        "roc_30": roc_30,
    })

    # Acceleration — matches the live OrderFlowTracker definition: the balanced
    # ratio between the recent and older half of the window on directional
    # prints, (recent - older) / (recent + older), bounded to [-1, 1]. Zero
    # unless both halves have >= 5 directional prints (live readiness gate).
    is_dir = is_buy + is_sell
    cum_dir = np.cumsum(is_dir)
    half_ns = np.timedelta64(int(window_secs * 1e9 / 2), "ns")
    mid_starts = np.searchsorted(ts_arr, ts_arr - half_ns, side="left")
    accel_out = np.zeros(n_total)
    for i in range(n_total):
        j = window_starts[i]
        m = mid_starts[i]
        older_cum = cum_dir[m - 1] if m > 0 else 0.0
        recent = cum_dir[i] - older_cum
        older = older_cum - (cum_dir[j - 1] if j > 0 else 0.0)
        if recent >= 5 and older >= 5:
            accel_out[i] = (recent - older) / (recent + older)

    result_df["acceleration"] = accel_out

    # ── EMA (time-decayed) variants of speed / acceleration / imbalance ────────
    # Smooth analogs of the hard-window metrics above: an exponential kernel
    # replaces the fixed window (speed) and the 2-bin recent/older split
    # (acceleration). Single forward O(n) recurrence with vectorised decay
    # factors — decay each accumulator by exp(-Δt/τ), then add the new print.
    # Neutral prints (is_dir == 0) do not contribute, matching the live tracker
    # which only counts directional prints. Inter-print Δt comes from numpy
    # datetime64 differencing (safe); avoid tz-aware Series.astype(int64).
    dt_arr = np.zeros(n_total)
    if n_total > 1:
        dt_arr[1:] = np.diff(ts_arr).astype(np.float64) / 1e9
    dt_arr[dt_arr < 0.0] = 0.0
    decay_speed = np.exp(-dt_arr / ema_speed_tau)
    decay_fast = np.exp(-dt_arr / ema_accel_fast_tau)
    decay_slow = np.exp(-dt_arr / ema_accel_slow_tau)
    decay_slow_sq = np.exp(-2.0 * dt_arr / ema_accel_slow_tau)
    decay_imb = np.exp(-dt_arr / ema_imbalance_tau)

    speed_ema_out = np.zeros(n_total)
    accel_ema_out = np.zeros(n_total)
    accel_ema_z_out = np.zeros(n_total)
    imbalance_ema_out = np.zeros(n_total)

    c_speed = c_fast = c_slow = c_slow_sq = 0.0
    buy_ema = sell_ema = 0.0
    for i in range(n_total):
        d = is_dir[i]
        c_speed = c_speed * decay_speed[i] + d
        c_fast = c_fast * decay_fast[i] + d
        c_slow = c_slow * decay_slow[i] + d
        c_slow_sq = c_slow_sq * decay_slow_sq[i] + d
        buy_ema = buy_ema * decay_imb[i] + buy_vol[i]
        sell_ema = sell_ema * decay_imb[i] + sell_vol[i]

        speed_ema_out[i] = c_speed / ema_speed_tau

        r_fast = c_fast / ema_accel_fast_tau
        r_slow = c_slow / ema_accel_slow_tau
        denom = r_fast + r_slow
        if denom > 0.0:
            accel = (r_fast - r_slow) / denom
            accel_ema_out[i] = accel
            if c_slow_sq > 0.0:
                n_eff = (c_slow * c_slow) / c_slow_sq
                accel_ema_z_out[i] = accel * np.sqrt(n_eff)

        tot = buy_ema + sell_ema
        if tot > 0.0:
            imbalance_ema_out[i] = (buy_ema - sell_ema) / tot

    result_df["speed_ema"] = speed_ema_out
    result_df["acceleration_ema"] = accel_ema_out
    result_df["acceleration_ema_z"] = accel_ema_z_out
    result_df["imbalance_ema"] = imbalance_ema_out

    return result_df


# ─── SQUEEZE DETECTION ────────────────────────────────────────────────────────

def detect_squeeze_candles(candles: pd.DataFrame, min_body_pct: float = 1.5,
                           min_volume_mult: float = 2.0) -> pd.DataFrame:
    """Identify squeeze candles: large bullish body + above-average volume.

    A 'squeeze candle' is one where:
      - body_pct > min_body_pct (strong upward move)
      - volume > median * min_volume_mult (surge in activity)
      - close > open (bullish)
    """
    if candles.empty:
        return candles

    candles = candles.copy()
    median_vol = candles["volume"].median()
    candles["is_squeeze"] = (
        (candles["body_pct"] > min_body_pct) &
        (candles["volume"] > median_vol * min_volume_mult) &
        (candles["close"] > candles["open"])
    )
    return candles


def find_squeeze_starts(candles: pd.DataFrame) -> pd.DatetimeIndex:
    """Find the first candle of each squeeze sequence (transition to squeeze)."""
    squeeze_mask = candles["is_squeeze"].astype(int)
    # First squeeze candle = where mask transitions from 0 to 1
    starts = squeeze_mask.diff().fillna(0) == 1
    return candles.index[starts]


# ─── REGIME SEGMENTATION ─────────────────────────────────────────────────────

def segment_regimes(ts_metrics: pd.DataFrame, squeeze_starts: list,
                    pre_window_secs: int = 30, squeeze_duration_secs: int = 30) -> pd.DataFrame:
    """Label each tick as 'pre_squeeze', 'squeeze', or 'baseline'.

    - pre_squeeze: 30s window before each squeeze start
    - squeeze: 30s window starting at squeeze
    - baseline: everything else
    """
    ts_metrics = ts_metrics.copy()
    ts_arr = pd.to_datetime(ts_metrics["ts"], utc=True)
    labels = pd.Series("baseline", index=ts_metrics.index)

    for sq_start in squeeze_starts:
        sq_start_ts = pd.Timestamp(sq_start).tz_localize("UTC") if pd.Timestamp(sq_start).tzinfo is None else pd.Timestamp(sq_start).tz_convert("UTC")
        pre_start = sq_start_ts - pd.Timedelta(seconds=pre_window_secs)
        sq_end = sq_start_ts + pd.Timedelta(seconds=squeeze_duration_secs)

        pre_mask = (ts_arr >= pre_start) & (ts_arr < sq_start_ts)
        sq_mask = (ts_arr >= sq_start_ts) & (ts_arr < sq_end)

        labels[pre_mask] = "pre_squeeze"
        labels[sq_mask] = "squeeze"

    ts_metrics["regime"] = labels
    return ts_metrics


# ─── STATISTICAL COMPARISON ──────────────────────────────────────────────────

def compare_regimes(ts_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare mean T&S metrics across regimes."""
    numeric_cols = ["speed", "volume_speed", "avg_size", "buy_ratio",
                    "sell_ratio", "imbalance", "tick_imbalance", "momentum",
                    "volatility", "range_pct", "roc_3", "roc_10", "roc_30",
                    "acceleration"]

    stats = ts_metrics.groupby("regime")[numeric_cols].agg(["mean", "std", "median"])
    return stats


def compute_discriminators(ts_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute effect sizes (Cohen's d) between pre_squeeze and baseline."""
    numeric_cols = ["speed", "volume_speed", "avg_size", "buy_ratio",
                    "imbalance", "tick_imbalance", "momentum",
                    "volatility", "range_pct", "roc_3", "roc_10", "roc_30",
                    "acceleration"]

    baseline = ts_metrics[ts_metrics["regime"] == "baseline"]
    pre_sq = ts_metrics[ts_metrics["regime"] == "pre_squeeze"]
    squeeze = ts_metrics[ts_metrics["regime"] == "squeeze"]

    rows = []
    for col in numeric_cols:
        b_mean = baseline[col].mean()
        b_std = baseline[col].std()
        p_mean = pre_sq[col].mean() if len(pre_sq) > 0 else 0.0
        s_mean = squeeze[col].mean() if len(squeeze) > 0 else 0.0

        # Cohen's d: how many baseline-stdevs the pre_squeeze/squeeze mean is away
        d_pre = (p_mean - b_mean) / b_std if b_std > 0 else 0.0
        d_sq = (s_mean - b_mean) / b_std if b_std > 0 else 0.0

        rows.append({
            "metric": col,
            "baseline_mean": round(b_mean, 4),
            "baseline_std": round(b_std, 4),
            "pre_squeeze_mean": round(p_mean, 4),
            "squeeze_mean": round(s_mean, 4),
            "cohens_d_pre": round(d_pre, 3),
            "cohens_d_squeeze": round(d_sq, 3),
        })

    return pd.DataFrame(rows).sort_values("cohens_d_pre", ascending=False, key=abs)


# ─── COMPOSITE SIGNAL DISCOVERY ──────────────────────────────────────────────

def compute_squeeze_signal(ts_metrics: pd.DataFrame) -> pd.DataFrame:
    """Derive a composite 'squeeze_signal' score from the most discriminating vars.

    signal = z(speed) + z(imbalance) + z(roc_3) + z(acceleration) + z(buy_ratio) + z(momentum)
    where z() = (value - baseline_mean) / baseline_std
    """
    baseline = ts_metrics[ts_metrics["regime"] == "baseline"]
    ts_metrics = ts_metrics.copy()

    signal_vars = ["speed", "imbalance", "roc_3", "acceleration", "volume_speed", "buy_ratio", "momentum"]
    for col in signal_vars:
        b_mean = baseline[col].mean()
        b_std = baseline[col].std()
        if b_std > 0:
            ts_metrics[f"z_{col}"] = (ts_metrics[col] - b_mean) / b_std
        else:
            ts_metrics[f"z_{col}"] = 0.0

    z_cols = [f"z_{c}" for c in signal_vars]
    ts_metrics["squeeze_signal"] = ts_metrics[z_cols].sum(axis=1)
    return ts_metrics


# ─── TOPPING TAIL CROSS-REFERENCE ────────────────────────────────────────────

def compute_topping_tails(candles: pd.DataFrame, atr_period: int = 5) -> pd.DataFrame:
    """Compute topping tail ratio for each candle (upper_wick / ATR)."""
    candles = candles.copy()
    candles["atr"] = candles["range"].rolling(atr_period, min_periods=1).mean()
    candles["topping_tail_ratio"] = candles["upper_wick"] / candles["atr"].replace(0, np.nan)
    candles["topping_tail_ratio"] = candles["topping_tail_ratio"].fillna(0)
    return candles


# ─── MAIN ANALYSIS ───────────────────────────────────────────────────────────

def analyze_ticker(symbol: str, start_et_str: str, end_et_str: str, day: str = "2026-05-27"):
    """Full analysis pipeline for one ticker."""
    # Parse times
    start_et = datetime.strptime(f"{day} {start_et_str}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    end_et = datetime.strptime(f"{day} {end_et_str}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    start_utc = start_et.astimezone(timezone.utc)
    end_utc = end_et.astimezone(timezone.utc)

    print(f"\n{'='*80}")
    print(f"  SQUEEZE ANALYSIS: {symbol}")
    print(f"  Window: {start_et_str} - {end_et_str} ET ({day})")
    print(f"{'='*80}\n")

    # 1. Fetch tick data
    df = fetch_ticks(symbol, start_utc, end_utc)
    if df.empty:
        print(f"  NO DATA for {symbol}")
        return None

    print(f"  Total ticks: {len(df)}")
    print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"  Total volume: {df['size'].sum():,.0f} shares")
    print(f"  Duration: {(df['ts'].iloc[-1] - df['ts'].iloc[0]).total_seconds():.0f}s")

    # 2. Build candles (10s and 1m)
    candles_10s = build_candles(df, freq="10s")
    candles_1m = build_candles(df, freq="1min")

    print(f"\n  10-sec candles: {len(candles_10s)}")
    print(f"  1-min candles: {len(candles_1m)}")

    # 3. Detect squeeze candles
    candles_10s = detect_squeeze_candles(candles_10s, min_body_pct=1.0, min_volume_mult=1.5)
    candles_1m = detect_squeeze_candles(candles_1m, min_body_pct=2.0, min_volume_mult=1.5)
    candles_10s = compute_topping_tails(candles_10s)
    candles_1m = compute_topping_tails(candles_1m)

    squeeze_starts_10s = find_squeeze_starts(candles_10s)
    squeeze_starts_1m = find_squeeze_starts(candles_1m)

    print(f"\n  Squeeze candles (10s): {candles_10s['is_squeeze'].sum()} / {len(candles_10s)}")
    print(f"  Squeeze starts (10s): {len(squeeze_starts_10s)}")
    print(f"  Squeeze candles (1m): {candles_1m['is_squeeze'].sum()} / {len(candles_1m)}")
    print(f"  Squeeze starts (1m): {len(squeeze_starts_1m)}")

    # Show squeeze candle details
    if not candles_1m.empty and candles_1m["is_squeeze"].any():
        print(f"\n  {'─'*60}")
        print(f"  1-MIN SQUEEZE CANDLES:")
        print(f"  {'─'*60}")
        sq_1m = candles_1m[candles_1m["is_squeeze"]]
        for idx, row in sq_1m.iterrows():
            ts_str = idx.astimezone(LOCAL_TZ).strftime("%H:%M:%S")
            print(f"    {ts_str}  O={row['open']:.2f} H={row['high']:.2f} "
                  f"L={row['low']:.2f} C={row['close']:.2f}  "
                  f"body={row['body_pct']:+.1f}%  vol={row['volume']:,.0f}  "
                  f"ticks={row['tick_count']:.0f}  "
                  f"topping_tail={row['topping_tail_ratio']:.2f}")

    # 4. Compute T&S metrics
    print(f"\n  Computing T&S rolling metrics...")
    ts_metrics = compute_ts_metrics(df, window_secs=10)

    # 5. Segment regimes using 1-min squeeze starts (more reliable)
    all_squeeze_starts = list(squeeze_starts_1m) + list(squeeze_starts_10s)
    ts_metrics = segment_regimes(ts_metrics, all_squeeze_starts,
                                  pre_window_secs=30, squeeze_duration_secs=30)

    regime_counts = ts_metrics["regime"].value_counts()
    print(f"\n  Regime distribution:")
    for regime, count in regime_counts.items():
        print(f"    {regime}: {count} ticks ({count/len(ts_metrics)*100:.1f}%)")

    # 6. Statistical comparison
    print(f"\n  {'─'*60}")
    print(f"  REGIME COMPARISON (mean values):")
    print(f"  {'─'*60}")
    stats = compare_regimes(ts_metrics)
    # Print means only for readability
    for regime in ["baseline", "pre_squeeze", "squeeze"]:
        if regime in stats.index:
            print(f"\n  [{regime.upper()}]")
            for col in ["speed", "volume_speed", "avg_size", "buy_ratio",
                        "imbalance", "tick_imbalance", "momentum",
                        "roc_3", "roc_10", "acceleration"]:
                mean_val = stats.loc[regime, (col, "mean")]
                std_val = stats.loc[regime, (col, "std")]
                print(f"    {col:<20s}  mean={mean_val:>8.3f}  std={std_val:>8.3f}")

    # 7. Effect sizes (discriminators)
    print(f"\n  {'─'*60}")
    print(f"  DISCRIMINATING POWER (Cohen's d vs baseline):")
    print(f"  {'─'*60}")
    disc = compute_discriminators(ts_metrics)
    print(f"  {'metric':<20s} {'baseline':>10s} {'pre_squeeze':>12s} {'squeeze':>10s} {'d(pre)':>8s} {'d(sq)':>8s}")
    print(f"  {'-'*70}")
    for _, row in disc.iterrows():
        print(f"  {row['metric']:<20s} {row['baseline_mean']:>10.3f} "
              f"{row['pre_squeeze_mean']:>12.3f} {row['squeeze_mean']:>10.3f} "
              f"{row['cohens_d_pre']:>+8.2f} {row['cohens_d_squeeze']:>+8.2f}")

    # 8. Composite signal
    ts_metrics = compute_squeeze_signal(ts_metrics)

    # Signal threshold analysis: what signal level would have caught squeeze starts?
    print(f"\n  {'─'*60}")
    print(f"  COMPOSITE SIGNAL ANALYSIS:")
    print(f"  {'─'*60}")
    for regime in ["baseline", "pre_squeeze", "squeeze"]:
        subset = ts_metrics[ts_metrics["regime"] == regime]
        if len(subset) > 0:
            p50 = subset["squeeze_signal"].quantile(0.50)
            p75 = subset["squeeze_signal"].quantile(0.75)
            p90 = subset["squeeze_signal"].quantile(0.90)
            p95 = subset["squeeze_signal"].quantile(0.95)
            print(f"  [{regime.upper()}]  p50={p50:.2f}  p75={p75:.2f}  p90={p90:.2f}  p95={p95:.2f}")

    # Find optimal threshold
    baseline_p95 = ts_metrics[ts_metrics["regime"] == "baseline"]["squeeze_signal"].quantile(0.95)
    pre_sq = ts_metrics[ts_metrics["regime"] == "pre_squeeze"]
    if len(pre_sq) > 0:
        recall_at_p95 = (pre_sq["squeeze_signal"] >= baseline_p95).mean()
        print(f"\n  At baseline p95 threshold ({baseline_p95:.2f}):")
        print(f"    Pre-squeeze recall: {recall_at_p95:.1%}")
        print(f"    (= fraction of pre-squeeze ticks that exceed baseline 95th percentile)")

    # 9. Time-series of signal leading into each squeeze
    print(f"\n  {'─'*60}")
    print(f"  SIGNAL RAMP-UP BEFORE EACH SQUEEZE START:")
    print(f"  {'─'*60}")
    for sq_start in squeeze_starts_1m:
        sq_ts = pd.Timestamp(sq_start).tz_localize("UTC") if pd.Timestamp(sq_start).tzinfo is None else pd.Timestamp(sq_start).tz_convert("UTC")
        window = ts_metrics[
            (pd.to_datetime(ts_metrics["ts"], utc=True) >= sq_ts - pd.Timedelta(seconds=60)) &
            (pd.to_datetime(ts_metrics["ts"], utc=True) < sq_ts + pd.Timedelta(seconds=10))
        ]
        if window.empty:
            continue
        et_str = sq_ts.astimezone(LOCAL_TZ).strftime("%H:%M:%S")
        print(f"\n  Squeeze start: {et_str}")
        print(f"  {'sec_before':<12s} {'signal':>8s} {'speed':>7s} {'imbal':>7s} {'roc3':>7s} {'accel':>7s} {'buy%':>7s} {'momtm':>7s}")
        for _, tick in window.iterrows():
            tick_ts = pd.Timestamp(tick["ts"], tz="UTC")
            secs_before = (sq_ts - tick_ts).total_seconds()
            sig = tick.get("squeeze_signal", 0)
            print(f"  {secs_before:>+8.1f}s     {sig:>8.2f} {tick['speed']:>7.1f} "
                  f"{tick['imbalance']:>+7.3f} {tick['roc_3']:>+7.3f} "
                  f"{tick['acceleration']:>+7.2f} {tick['buy_ratio']:>7.1%} "
                  f"{tick['momentum']:>+7.3f}")

    # 10. Topping tail analysis near squeeze tops
    print(f"\n  {'─'*60}")
    print(f"  TOPPING TAIL ANALYSIS (1-min):")
    print(f"  {'─'*60}")
    if not candles_1m.empty:
        # Find candles after squeeze candles — potential tops
        high_tail = candles_1m[candles_1m["topping_tail_ratio"] > 0.5]
        if not high_tail.empty:
            print(f"  Candles with topping_tail_ratio > 0.5:")
            for idx, row in high_tail.iterrows():
                ts_str = idx.astimezone(LOCAL_TZ).strftime("%H:%M:%S")
                was_squeeze = "SQUEEZE" if row.get("is_squeeze", False) else "       "
                print(f"    {ts_str}  {was_squeeze}  O={row['open']:.2f} H={row['high']:.2f} "
                      f"C={row['close']:.2f}  tail_ratio={row['topping_tail_ratio']:.2f}")
        else:
            print(f"  No significant topping tails detected")

    return {
        "symbol": symbol,
        "ticks": df,
        "candles_10s": candles_10s,
        "candles_1m": candles_1m,
        "ts_metrics": ts_metrics,
        "discriminators": disc,
    }


# ─── CROSS-TICKER COMPARISON ─────────────────────────────────────────────────

def cross_compare(results: list[dict]):
    """Compare squeeze signatures across tickers to find universal patterns."""
    print(f"\n{'='*80}")
    print(f"  CROSS-TICKER COMPARISON")
    print(f"{'='*80}")

    all_disc = []
    for r in results:
        if r is None:
            continue
        d = r["discriminators"].copy()
        d["symbol"] = r["symbol"]
        all_disc.append(d)

    if len(all_disc) < 2:
        print("  Need at least 2 tickers for comparison")
        return

    combined = pd.concat(all_disc, ignore_index=True)

    # Find metrics that are consistently discriminating across both
    print(f"\n  CONSISTENT DISCRIMINATORS (both tickers):")
    print(f"  {'metric':<20s}  {'ASTC d(pre)':>12s}  {'AMSS d(pre)':>12s}  {'avg |d|':>8s}  {'same_sign':>10s}")
    print(f"  {'-'*70}")

    metrics = combined["metric"].unique()
    comparison_rows = []
    for m in metrics:
        subset = combined[combined["metric"] == m]
        if len(subset) < 2:
            continue
        d_values = subset["cohens_d_pre"].values
        avg_abs = np.abs(d_values).mean()
        same_sign = all(d > 0 for d in d_values) or all(d < 0 for d in d_values)
        comparison_rows.append((m, d_values[0], d_values[1], avg_abs, same_sign))

    comparison_rows.sort(key=lambda x: x[3], reverse=True)
    for m, d1, d2, avg, same in comparison_rows:
        sign_str = "✓ YES" if same else "✗ NO"
        print(f"  {m:<20s}  {d1:>+12.3f}  {d2:>+12.3f}  {avg:>8.3f}  {sign_str:>10s}")

    # Summary insights
    print(f"\n  {'─'*60}")
    print(f"  KEY FINDINGS:")
    print(f"  {'─'*60}")
    consistent = [(m, d1, d2, avg) for m, d1, d2, avg, same in comparison_rows if same and avg > 0.3]
    if consistent:
        print(f"\n  Metrics that CONSISTENTLY distinguish pre-squeeze from baseline")
        print(f"  (same direction in both tickers, |Cohen's d| > 0.3):\n")
        for m, d1, d2, avg in consistent:
            direction = "↑ higher" if d1 > 0 else "↓ lower"
            print(f"    • {m}: {direction} before squeezes (avg effect: {avg:.2f}σ)")
    else:
        print(f"\n  No strongly consistent discriminators found — squeezes may have")
        print(f"  different signatures. Check individual ticker results above.")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

def main():
    results = []

    # ASTC: 8:30-9:29 AM ET
    r1 = analyze_ticker("ASTC", "08:30", "09:29")
    results.append(r1)

    # AMSS: 7:03-7:59 AM ET
    r2 = analyze_ticker("AMSS", "07:03", "07:59")
    results.append(r2)

    # Cross-comparison
    cross_compare([r for r in results if r is not None])

    print(f"\n{'='*80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
