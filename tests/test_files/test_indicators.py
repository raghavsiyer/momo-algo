"""
tests/test_indicators.py

Unit tests for technical_analysis/indicators.py — pure function tests.
All functions take DataFrames in and return DataFrames out; no API calls.
"""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from technical_analysis.indicators import (
    add_ema, add_ema_9, add_ema_20, add_ema_200,
    add_ma, add_ma_50, add_ma_200,
    add_vwap,
    add_bollinger_bands,
    add_macd,
    _add_macd_ip,
    add_roc,
    add_rvol,
    compute_spread,
    compute_rvol_current,
    compute_rvol_5min_bars,
    build_10sec, build_1min, build_5min, build_15min, build_daily,
)
from config import (
    EMA_9_PERIOD, EMA_20_PERIOD, EMA_200_PERIOD,
    BB_PERIOD, BB_STD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, MACD_ATR_PERIOD,
    ROC_PERIOD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 60, start_price: float = 10.0, step: float = 0.05) -> pd.DataFrame:
    """Synthetic rising OHLCV DataFrame with UTC timestamps, n bars, 1-min spacing."""
    now = datetime(2026, 4, 6, 14, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        c = start_price + i * step
        rows.append({
            "symbol":   "TEST",
            "ts":       now + timedelta(minutes=i),
            "open":     c - 0.02,
            "high":     c + 0.05,
            "low":      c - 0.05,
            "close":    c,
            "volume":   10_000 + i * 100,
            "interval": "1min",
            "vwap":     None,
        })
    return pd.DataFrame(rows)


def _flat(n: int = 60, price: float = 5.0) -> pd.DataFrame:
    """Flat OHLCV — all bars at the same price."""
    return _ohlcv(n, start_price=price, step=0.0)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA(unittest.TestCase):

    def test_ema_column_added(self):
        df = add_ema(_ohlcv(), period=9)
        self.assertIn("ema_9", df.columns)

    def test_ema_length_preserved(self):
        df = _ohlcv(50)
        result = add_ema(df, period=9)
        self.assertEqual(len(result), len(df))

    def test_ema_no_leading_nans_adjust_false(self):
        # adjust=False means ewm initialises on first value — no NaN
        df = add_ema(_ohlcv(30), period=9)
        self.assertEqual(df["ema_9"].isna().sum(), 0)

    def test_ema_flat_equals_price(self):
        df = add_ema(_flat(60), period=9)
        # For a constant price series all EMA values should equal the price
        self.assertTrue((df["ema_9"] == 5.0).all())

    def test_ema_9_shorthand(self):
        df = add_ema_9(_ohlcv())
        self.assertIn(f"ema_{EMA_9_PERIOD}", df.columns)

    def test_ema_20_shorthand(self):
        df = add_ema_20(_ohlcv())
        self.assertIn(f"ema_{EMA_20_PERIOD}", df.columns)

    def test_ema_200_shorthand(self):
        df = add_ema_200(_ohlcv(250))
        self.assertIn(f"ema_{EMA_200_PERIOD}", df.columns)

    def test_ema_does_not_mutate_input(self):
        df = _ohlcv()
        original_cols = set(df.columns)
        add_ema(df, period=9)
        self.assertEqual(set(df.columns), original_cols)


# ---------------------------------------------------------------------------
# Moving Average
# ---------------------------------------------------------------------------

class TestMA(unittest.TestCase):

    def test_ma_column_added(self):
        df = add_ma(_ohlcv(60), period=50)
        self.assertIn("ma_50", df.columns)

    def test_ma_has_correct_nan_count(self):
        # Simple MA with window=50 on 60 bars → first 49 are NaN
        df = add_ma(_ohlcv(60), period=50)
        self.assertEqual(df["ma_50"].isna().sum(), 49)

    def test_ma_flat_equals_price(self):
        df = add_ma(_flat(60), period=10)
        self.assertTrue((df["ma_10"].dropna() == 5.0).all())

    def test_ma_50_shorthand(self):
        df = add_ma_50(_ohlcv(60))
        self.assertIn("ma_50", df.columns)

    def test_ma_200_shorthand(self):
        df = add_ma_200(_ohlcv(250))
        self.assertIn("ma_200", df.columns)

    def test_ma_does_not_mutate_input(self):
        df = _ohlcv()
        original_cols = set(df.columns)
        add_ma(df, period=50)
        self.assertEqual(set(df.columns), original_cols)


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------

class TestVWAP(unittest.TestCase):

    def test_vwap_column_added(self):
        df = add_vwap(_ohlcv())
        self.assertIn("vwap", df.columns)

    def test_vwap_no_nans(self):
        df = add_vwap(_ohlcv(30))
        self.assertEqual(df["vwap"].isna().sum(), 0)

    def test_vwap_flat_price_equals_price(self):
        df = add_vwap(_flat(30))
        # VWAP of a flat series equals the constant price
        self.assertTrue(np.allclose(df["vwap"].values, 5.0, atol=1e-6))

    def test_vwap_resets_on_new_day(self):
        # Build two days of 10 bars each
        day1 = _ohlcv(10, start_price=5.0)
        day2 = _ohlcv(10, start_price=5.0)
        # Shift day2 timestamps to the next day
        day2["ts"] = day2["ts"] + timedelta(days=1)
        df = pd.concat([day1, day2], ignore_index=True)
        result = add_vwap(df)
        # At bar index 10 (first bar of day 2), VWAP should reset
        # meaning it equals the typical price of that first bar
        bar_10 = result.iloc[10]
        expected_tp = (bar_10["high"] + bar_10["low"] + bar_10["close"]) / 3
        self.assertAlmostEqual(bar_10["vwap"], expected_tp, places=4)

    def test_vwap_does_not_mutate_input(self):
        df = _ohlcv()
        original_cols = set(df.columns)
        # "vwap" may already be in columns due to _ohlcv() adding it, so check temp cols
        add_vwap(df)
        self.assertNotIn("_date", df.columns)
        self.assertNotIn("_tp",   df.columns)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands(unittest.TestCase):

    def test_columns_added(self):
        df = add_bollinger_bands(_ohlcv(40))
        for col in ("bb_middle", "bb_upper", "bb_lower"):
            self.assertIn(col, df.columns)

    def test_no_bb_std_column_left_behind(self):
        df = add_bollinger_bands(_ohlcv(40))
        self.assertNotIn("bb_std", df.columns)

    def test_upper_above_middle_above_lower(self):
        df = add_bollinger_bands(_ohlcv(40)).dropna()
        self.assertTrue((df["bb_upper"] >= df["bb_middle"]).all())
        self.assertTrue((df["bb_middle"] >= df["bb_lower"]).all())

    def test_flat_series_bands_equidistant(self):
        # On a flat series, std=0 → all bands collapse to the price
        df = add_bollinger_bands(_flat(40))
        valid = df.dropna()
        self.assertTrue(np.allclose(valid["bb_upper"], valid["bb_middle"], atol=1e-10))
        self.assertTrue(np.allclose(valid["bb_lower"], valid["bb_middle"], atol=1e-10))

    def test_nan_count_matches_period(self):
        df = add_bollinger_bands(_ohlcv(40), period=BB_PERIOD)
        self.assertEqual(df["bb_middle"].isna().sum(), BB_PERIOD - 1)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD(unittest.TestCase):

    def test_columns_added(self):
        df = add_macd(_ohlcv(60))
        for col in ("macd_line", "macd_signal", "macd_hist"):
            self.assertIn(col, df.columns)

    def test_hist_equals_line_minus_signal(self):
        df = add_macd(_ohlcv(60)).dropna()
        diff = (df["macd_line"] - df["macd_signal"] - df["macd_hist"]).abs()
        self.assertTrue((diff < 1e-10).all())

    def test_rising_series_positive_macd_eventually(self):
        # In a steadily rising series, fast EMA > slow EMA after warmup
        df = add_macd(_ohlcv(100, step=0.10))
        warmed = df.iloc[MACD_SLOW:]
        self.assertTrue((warmed["macd_line"] > 0).all())

    def test_flat_series_macd_zero(self):
        df = add_macd(_flat(60))
        self.assertTrue(np.allclose(df["macd_line"].values, 0.0, atol=1e-10))

    # --- MACD-V (Spiroglou 2022): 100 * (EMA12 - EMA26) / Wilder ATR(26) ---

    @staticmethod
    def _walk(n: int = 200, scale: float = 1.0, seed: int = 7) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 5.0 + np.cumsum(rng.normal(0.0, 0.03, n))
        spread = np.abs(rng.normal(0.0, 0.02, n)) + 0.005
        df = _ohlcv(n)
        df["close"] = close * scale
        df["open"] = np.r_[close[0], close[:-1]] * scale
        df["high"] = (np.maximum(close, np.r_[close[0], close[:-1]]) + spread) * scale
        df["low"] = (np.minimum(close, np.r_[close[0], close[:-1]]) - spread) * scale
        return df

    def test_macd_v_matches_the_published_formula(self):
        df = self._walk()
        out = add_macd(df)
        close, high, low = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
        # Wilder's ATR written out bar by bar, independent of the pandas version.
        atr = np.empty(len(df))
        for i in range(len(df)):
            tr = high[i] - low[i] if i == 0 else max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            atr[i] = tr if i == 0 else atr[i - 1] + (tr - atr[i - 1]) / MACD_ATR_PERIOD
        ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean().to_numpy()
        ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean().to_numpy()
        expected = 100.0 * (ema_fast - ema_slow) / atr
        self.assertTrue(np.allclose(out["macd_line"].to_numpy(), expected, atol=1e-9))
        expected_signal = pd.Series(expected).ewm(span=MACD_SIGNAL, adjust=False).mean().to_numpy()
        self.assertTrue(np.allclose(out["macd_signal"].to_numpy(), expected_signal, atol=1e-9))

    def test_macd_v_is_the_same_at_any_price_level(self):
        # The point of MACD-V: a $0.50 and a $50 stock with the same shape read the same.
        cheap = add_macd(self._walk(scale=0.1))
        dear = add_macd(self._walk(scale=10.0))
        self.assertTrue(np.allclose(cheap["macd_line"].to_numpy(), dear["macd_line"].to_numpy(), atol=1e-9))
        self.assertTrue(np.allclose(cheap["macd_hist"].to_numpy(), dear["macd_hist"].to_numpy(), atol=1e-9))

    def test_macd_v_shrinks_when_the_bars_are_wilder(self):
        calm = self._walk()
        wild = calm.copy()
        wild["high"] = wild["high"] + 0.10
        wild["low"] = wild["low"] - 0.10
        calm_v = add_macd(calm)["macd_line"].abs().iloc[MACD_SLOW:]
        wild_v = add_macd(wild)["macd_line"].abs().iloc[MACD_SLOW:]
        self.assertTrue((wild_v < calm_v).all())

    def test_macd_v_with_no_range_at_all_is_zero_not_nan(self):
        df = _flat(40)
        df["high"] = df["close"]
        df["low"] = df["close"]
        out = add_macd(df)
        self.assertTrue(np.isfinite(out[["macd_line", "macd_signal", "macd_hist"]].to_numpy()).all())
        self.assertTrue(np.allclose(out["macd_line"].to_numpy(), 0.0))

    def test_macd_v_from_closes_alone_uses_close_to_close_range(self):
        df = self._walk()[["ts", "close"]].copy()
        out = add_macd(df)
        self.assertTrue(np.isfinite(out["macd_line"].to_numpy()).all())
        self.assertGreater(out["macd_line"].abs().max(), 0.0)

    def test_in_place_and_copying_versions_agree(self):
        df = self._walk()
        copied = add_macd(df)
        in_place = df.copy()
        _add_macd_ip(in_place)
        for col in ("macd_line", "macd_signal", "macd_hist"):
            self.assertTrue(np.allclose(copied[col].to_numpy(), in_place[col].to_numpy()))


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

class TestROC(unittest.TestCase):

    def test_roc_column_added(self):
        df = add_roc(_ohlcv())
        self.assertIn("roc", df.columns)

    def test_flat_roc_is_zero(self):
        df = add_roc(_flat(30))
        # First ROC_PERIOD rows are NaN; rest should be 0
        valid = df["roc"].dropna()
        self.assertTrue(np.allclose(valid.values, 0.0, atol=1e-10))

    def test_roc_positive_on_rising(self):
        df = add_roc(_ohlcv(30, step=0.10))
        valid = df["roc"].dropna()
        self.assertTrue((valid > 0).all())

    def test_roc_nan_count(self):
        df = add_roc(_ohlcv(30), period=1)
        self.assertEqual(df["roc"].isna().sum(), 1)


# ---------------------------------------------------------------------------
# RVol
# ---------------------------------------------------------------------------

class TestRVol(unittest.TestCase):

    def test_rvol_column_added(self):
        df = add_rvol(_ohlcv(), avg_volume=10_000)
        self.assertIn("rvol", df.columns)

    def test_rvol_ratio_correct(self):
        df = _flat(5)
        df["volume"] = 20_000
        result = add_rvol(df, avg_volume=10_000)
        self.assertTrue((result["rvol"] == 2.0).all())

    def test_rvol_zero_avg_volume(self):
        result = add_rvol(_ohlcv(5), avg_volume=0)
        self.assertTrue((result["rvol"] == 0.0).all())

    def test_compute_rvol_current(self):
        df = _flat(5)
        df["volume"] = 15_000
        self.assertAlmostEqual(compute_rvol_current(df, avg_volume=5_000), 3.0)

    def test_compute_rvol_current_empty(self):
        self.assertEqual(compute_rvol_current(pd.DataFrame(), avg_volume=1000), 0.0)

    def test_compute_rvol_5min_bars(self):
        df = _flat(1)
        df["volume"] = 10   # avg_daily=780 → per bar=780/78=10 → rvol=1.0
        self.assertAlmostEqual(compute_rvol_5min_bars(df, avg_daily_volume=780), 1.0)


# ---------------------------------------------------------------------------
# compute_spread
# ---------------------------------------------------------------------------

class TestComputeSpread(unittest.TestCase):

    def test_basic_spread(self):
        self.assertAlmostEqual(compute_spread(4.98, 5.02), 0.04)

    def test_zero_spread(self):
        self.assertAlmostEqual(compute_spread(5.00, 5.00), 0.0)

    def test_rounded_to_4_decimal(self):
        result = compute_spread(4.9999, 5.0001)
        self.assertEqual(result, round(5.0001 - 4.9999, 4))


# ---------------------------------------------------------------------------
# Composite builders
# ---------------------------------------------------------------------------

class TestCompositeBuilders(unittest.TestCase):

    def test_build_10sec_has_ema9_and_roc(self):
        df = build_10sec(_ohlcv(60))
        self.assertIn("ema_9",  df.columns)
        self.assertIn("roc",    df.columns)

    def test_build_1min_has_all_columns(self):
        df = build_1min(_ohlcv(60), avg_daily_volume=500_000)
        for col in ("ema_9", "ema_20", "vwap", "ma_50", "macd_line", "rvol"):
            self.assertIn(col, df.columns)

    def test_build_5min_has_all_columns(self):
        df = build_5min(_ohlcv(60), avg_daily_volume=500_000)
        for col in ("ema_20", "vwap", "ma_50", "bb_middle", "macd_line", "rvol"):
            self.assertIn(col, df.columns)

    def test_build_15min_has_all_columns(self):
        df = build_15min(_ohlcv(30))
        for col in ("ema_20", "vwap", "ma_50"):
            self.assertIn(col, df.columns)

    def test_build_daily_has_all_columns(self):
        df = build_daily(_ohlcv(250))
        for col in ("ema_200", "ma_50", "ma_200"):
            self.assertIn(col, df.columns)

    def test_build_1min_no_rvol_when_avg_vol_zero(self):
        df = build_1min(_ohlcv(60), avg_daily_volume=0)
        self.assertNotIn("rvol", df.columns)

    def test_builders_preserve_row_count(self):
        n = 60
        df = _ohlcv(n)
        self.assertEqual(len(build_10sec(df)), n)
        self.assertEqual(len(build_1min(df)),  n)
        self.assertEqual(len(build_15min(df)), n)

    def test_builders_do_not_mutate_input(self):
        df = _ohlcv(60)
        cols_before = set(df.columns)
        build_1min(df, avg_daily_volume=100_000)
        self.assertEqual(set(df.columns), cols_before)


if __name__ == "__main__":
    unittest.main()
