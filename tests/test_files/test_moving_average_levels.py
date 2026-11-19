"""Tests for technical_analysis/moving_average_levels.py and its wiring into
the resistance map, the pipeline and the scorer. No network."""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")

import copy
import pickle
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import technical_analysis.moving_average_levels as mal
from technical_analysis.moving_average_levels import (
    TimeframeMA,
    bucket_bars,
    bucket_key,
    fetch_ma_frames,
    ma_level_source,
)
from technical_analysis.resistance import LevelType, ResistanceMap

UTC = timezone.utc
SPECS = (("ema", 200), ("sma", 200), ("sma", 50))


def _frame(start: datetime, step: timedelta, closes) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": [start + step * i for i in range(len(closes))],
        "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000] * len(closes),
    })


def _walk(n: int, seed: int = 7, start: float = 5.0) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(start * np.exp(np.cumsum(rng.normal(0, 0.01, n))))


def _ref_ema(closes: list[float], period: int) -> float:
    ema = sum(closes[:period]) / period
    alpha = 2.0 / (period + 1)
    for c in closes[period:]:
        ema += alpha * (c - ema)
    return ema


@pytest.fixture(autouse=True)
def _clear_memo():
    mal._memo.clear()
    mal._memo_day = None
    yield
    mal._memo.clear()
    mal._memo_day = None


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------

def test_clock_buckets_floor_to_the_interval():
    ts = datetime(2026, 11, 16, 14, 37, 12, tzinfo=UTC)
    assert bucket_key("5m", ts) == datetime(2026, 11, 16, 14, 35, tzinfo=UTC)
    assert bucket_key("15m", ts) == datetime(2026, 11, 16, 14, 30, tzinfo=UTC)
    assert bucket_key("1h", ts) == datetime(2026, 11, 16, 14, 0, tzinfo=UTC)


def test_multi_hour_buckets_anchor_at_et_midnight_across_dst():
    # 08:30 EST -> 08:00 EST block = 13:00 UTC
    assert bucket_key("4h", datetime(2026, 11, 16, 13, 30, tzinfo=UTC)) == datetime(2026, 11, 16, 13, 0, tzinfo=UTC)
    # 09:30 EDT -> 08:00 EDT block = 12:00 UTC
    assert bucket_key("4h", datetime(2026, 7, 16, 13, 30, tzinfo=UTC)) == datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
    # 10:10 EST -> 10:00 EST block = 15:00 UTC
    assert bucket_key("2h", datetime(2026, 11, 16, 15, 10, tzinfo=UTC)) == datetime(2026, 11, 16, 15, 0, tzinfo=UTC)


def test_session_buckets_use_et_date_week_monday_and_month():
    # 02:00 UTC on the 17th is 21:00 ET on the 16th
    assert bucket_key("daily", datetime(2026, 11, 17, 2, 0, tzinfo=UTC)) == date(2026, 11, 16)
    thursday = datetime(2026, 11, 19, 0, 0, tzinfo=UTC)
    assert bucket_key("weekly", thursday, session_date=date(2026, 11, 19)) == date(2026, 11, 16)
    assert bucket_key("monthly", thursday, session_date=date(2026, 11, 19)) == date(2026, 11, 1)


def test_bucket_bars_keeps_last_close_per_bucket():
    start = datetime(2026, 11, 16, 14, 0, tzinfo=UTC)
    df = _frame(start, timedelta(minutes=15), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = bucket_bars(df, "30m")
    assert list(out["close"]) == [2.0, 4.0, 6.0]


# ---------------------------------------------------------------------------
# MA math
# ---------------------------------------------------------------------------

def test_values_match_reference_on_closed_bars():
    closes = _walk(600)
    start = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    bars = bucket_bars(_frame(start, timedelta(hours=1), closes), "1h")
    now = start + timedelta(hours=len(closes) + 2)   # every bar closed
    ma = TimeframeMA.from_bars("1h", SPECS, bars, now)
    assert ma.forming_close is None
    assert ma.value("sma", 200) == pytest.approx(np.mean(closes[-200:]), rel=1e-12)
    assert ma.value("sma", 50) == pytest.approx(np.mean(closes[-50:]), rel=1e-12)
    assert ma.value("ema", 200) == pytest.approx(_ref_ema(closes, 200), rel=1e-12)


def test_forming_bar_is_included_like_a_chart():
    closes = _walk(400)
    start = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    bars = bucket_bars(_frame(start, timedelta(hours=1), closes), "1h")
    now = start + timedelta(hours=len(closes) - 1, minutes=20)   # inside the last bar
    live = 9.99
    ma = TimeframeMA.from_bars("1h", SPECS, bars, now, last_price=live)
    closed = closes[:-1]
    assert ma.forming_close == live
    assert ma.value("sma", 200) == pytest.approx((sum(closed[-199:]) + live) / 200, rel=1e-12)
    ema_closed = _ref_ema(closed, 200)
    assert ma.value("ema", 200) == pytest.approx(ema_closed + (2 / 201) * (live - ema_closed), rel=1e-12)


def test_too_few_bars_leaves_long_mas_unmarked():
    closes = _walk(120)
    start = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    bars = bucket_bars(_frame(start, timedelta(hours=1), closes), "1h")
    ma = TimeframeMA.from_bars("1h", SPECS, bars, start + timedelta(days=30))
    assert ma.value("sma", 200) is None
    assert ma.value("ema", 200) is None
    assert ma.value("sma", 50) == pytest.approx(np.mean(closes[-50:]))


def test_live_bars_roll_the_forming_bar_into_the_closed_series():
    closes = _walk(260)
    start = datetime(2026, 11, 16, 9, 0, tzinfo=UTC)
    bars = bucket_bars(_frame(start, timedelta(minutes=5), closes), "5m")
    last_closed = start + timedelta(minutes=5 * (len(closes) - 1))
    ma = TimeframeMA.from_bars("5m", SPECS, bars, last_closed + timedelta(minutes=5))
    assert ma.n_closed == 260

    nxt = last_closed + timedelta(minutes=5)
    # a stale bar from the snapshot's past is ignored
    assert ma.on_bar(bucket_key("5m", last_closed), 1.0) is False
    for minute, px in enumerate((6.0, 6.1, 6.2)):
        ts = nxt + timedelta(minutes=minute)
        assert ma.on_bar(bucket_key("5m", ts), px)
    assert ma.n_closed == 260 and ma.forming_close == 6.2
    # first 1m bar of the following 5m bucket closes the forming bar at 6.2
    ma.on_bar(bucket_key("5m", nxt + timedelta(minutes=5)), 6.3)
    assert ma.n_closed == 261
    assert ma.last_closed_key == nxt
    expected = closes + [6.2]
    assert ma.value("sma", 50) == pytest.approx((sum(expected[-49:]) + 6.3) / 50, rel=1e-12)


# ---------------------------------------------------------------------------
# Resistance map
# ---------------------------------------------------------------------------

def _map_with_mas(price: float = 5.0):
    closes = _walk(300)
    start = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    hourly = bucket_bars(_frame(start, timedelta(hours=1), closes), "1h")
    daily = bucket_bars(
        _frame(datetime(2025, 1, 2, tzinfo=UTC), timedelta(days=1), closes),
        "daily", session_stamped=True,
    )
    now = start + timedelta(hours=len(closes) + 1)
    rmap = ResistanceMap("TEST", current_price=price)
    n = rmap.build_moving_averages({"1h": hourly, "daily": daily}, now=now)
    return rmap, n, now


def test_map_marks_one_level_per_timeframe_and_spec():
    rmap, n, _ = _map_with_mas()
    levels = rmap.moving_average_levels()
    assert n == 6 == len(levels)
    assert {l.source for l in levels} == {
        ma_level_source(k, p, tf) for tf in ("1h", "daily") for k, p in SPECS
    }
    assert "200 EMA (1h)" in {l.source for l in levels}
    assert "50 MA (daily)" in {l.source for l in levels}
    for l in levels:
        assert l.level_type == LevelType.MOVING_AVERAGE
        assert l.timeframe in ("1h", "daily") and l.ma_period in (50, 200)


def test_live_update_moves_levels_in_place_without_duplicates():
    rmap, _, now = _map_with_mas()
    before = {l.source: l.price for l in rmap.moving_average_levels()}
    for minute in range(70):   # crosses an hour boundary
        rmap.update_moving_averages(now + timedelta(minutes=minute), 12.0)
    after = {l.source: l.price for l in rmap.moving_average_levels()}
    assert len(rmap.moving_average_levels()) == 6
    assert after["50 MA (1h)"] > before["50 MA (1h)"]


def test_level_reactivates_when_price_falls_back_under():
    flat = [5.0] * 300
    start = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
    bars = bucket_bars(_frame(start, timedelta(hours=1), flat), "1h")
    now = start + timedelta(hours=301)
    rmap = ResistanceMap("TEST", current_price=6.0)
    rmap.build_moving_averages({"1h": bars}, now=now)
    assert all(not l.active for l in rmap.moving_average_levels())   # price is above them
    rmap.mark_broken_levels(4.0)                                    # price drops under
    rmap.update_moving_averages(now + timedelta(minutes=1), 5.0)
    assert all(l.active for l in rmap.moving_average_levels())
    assert rmap.nearest_above(4.0).level_type == LevelType.MOVING_AVERAGE


def test_map_pickles_deep_copies_and_keeps_updating():
    rmap, _, now = _map_with_mas()
    for clone in (pickle.loads(pickle.dumps(rmap)), copy.deepcopy(rmap)):
        assert [l.source for l in clone.moving_average_levels()] == [l.source for l in rmap.moving_average_levels()]
        clone.update_moving_averages(now + timedelta(hours=2), 20.0)
        assert len(clone.moving_average_levels()) == 6


def test_map_cached_before_ma_levels_is_a_no_op():
    rmap = ResistanceMap("OLD", current_price=5.0)
    rmap.build_static()
    del rmap._ma_trackers            # what an old pickle looks like
    rmap.update_moving_averages(datetime(2026, 11, 16, 15, 0, tzinfo=UTC), 5.0)
    assert rmap.moving_average_levels() == []


def test_build_structure_no_longer_adds_daily_200s():
    daily = pd.DataFrame({
        "ts": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
        "open": [5.0] * 5, "high": [5.2] * 5, "low": [4.8] * 5, "close": [5.0] * 5,
        "ema_200": [7.0] * 5, "ma_200": [8.0] * 5,
    })
    rmap = ResistanceMap("TEST", current_price=5.0)
    rmap.build_structure(daily, pd.DataFrame())
    assert not any("200" in l.source for l in rmap.all_levels())


def test_above_200_flag_reads_daily_200s_only():
    from strategy.confidence_scorer import DynamicScorer
    rmap, _, _ = _map_with_mas()
    daily_200 = [l for l in rmap.moving_average_levels() if l.timeframe == "daily" and l.ma_period == 200]
    hourly_200 = [l for l in rmap.moving_average_levels() if l.timeframe == "1h" and l.ma_period == 200]
    for l in daily_200:
        l.price = 3.0
    for l in hourly_200:
        l.price = 9.0
    fake = SimpleNamespace(_rmap=rmap, _last_price=5.0)
    assert DynamicScorer._check_above_200(fake) == 1.0


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

class _FakeClient:
    """Serves synthetic bars per interval; records every request."""

    def __init__(self, native=("5min", "15min", "30min", "1hour", "1day", "1week", "1month"),
                 month_max_days: int | None = None) -> None:
        self.native = set(native)
        self.month_max_days = month_max_days
        self.calls: list[tuple[str, int]] = []

    def get_candles_interval(self, symbol, interval, lookback_days):
        self.calls.append((interval, lookback_days))
        if interval not in self.native:
            return pd.DataFrame()
        if interval == "1month" and self.month_max_days and lookback_days > self.month_max_days:
            return pd.DataFrame()
        now = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
        step = {
            "5min": timedelta(minutes=5), "15min": timedelta(minutes=15),
            "30min": timedelta(minutes=30), "1hour": timedelta(hours=1),
            "1day": timedelta(days=1), "1week": timedelta(weeks=1), "1month": timedelta(days=31),
        }[interval]
        n = 300
        if interval in ("1day", "1week", "1month"):
            start = datetime(now.year, now.month, now.day, tzinfo=UTC) - step * n
        else:
            start = now - step * n
        return _frame(start, step, _walk(n, seed=len(interval)))


def test_fetch_covers_every_timeframe_and_memoises_session_bars():
    client = _FakeClient()
    frames = fetch_ma_frames(client, "TEST")
    assert set(frames) == {"5m", "15m", "30m", "1h", "2h", "4h", "daily", "weekly", "monthly"}
    assert sorted(i for i, _ in client.calls) == sorted(
        ["5min", "15min", "30min", "1hour", "1day", "1week", "1month"]
    )
    # 4h buckets are built from 1h bars and sit on ET 00/04/08/12/16/20
    hours_et = {k.astimezone(mal._ET).hour for k in frames["4h"]["bucket"]}
    assert hours_et <= {0, 4, 8, 12, 16, 20}
    client.calls.clear()
    fetch_ma_frames(client, "TEST")
    assert {i for i, _ in client.calls} == {"5min", "15min", "30min", "1hour"}


def test_tradier_style_client_derives_30m_and_hourly_from_15m():
    client = _FakeClient(native=("5min", "15min", "1day", "1week", "1month"))
    frames = fetch_ma_frames(client, "TEST")
    for tf in ("30m", "1h", "2h", "4h"):
        assert tf in frames and not frames[tf].empty
    minutes = {k.minute for k in frames["1h"]["bucket"]}
    assert minutes == {0}


def test_monthly_retries_with_a_shorter_lookback():
    client = _FakeClient(month_max_days=6000)
    frames = fetch_ma_frames(client, "TEST", timeframes=("monthly",))
    month_calls = [d for i, d in client.calls if i == "1month"]
    assert len(month_calls) == 2 and month_calls[1] == month_calls[0] // 2
    assert "monthly" in frames


def test_client_without_interval_method_yields_nothing():
    assert fetch_ma_frames(object(), "TEST") == {}


def test_rejected_intraday_request_retries_shorter_then_derives():
    class _Rejecting(_FakeClient):
        def get_candles_interval(self, symbol, interval, lookback_days):
            if interval == "1hour":           # e.g. duration too long for the bar size
                self.calls.append((interval, lookback_days))
                return pd.DataFrame()
            return super().get_candles_interval(symbol, interval, lookback_days)

    client = _Rejecting()
    frames = fetch_ma_frames(client, "TEST", timeframes=("1h", "4h"))
    hour_calls = [d for i, d in client.calls if i == "1hour"]
    assert len(hour_calls) == 2 and hour_calls[1] == hour_calls[0] // 2
    assert {"1h", "4h"} <= set(frames)        # derived from the 15-minute bars


def test_rebuild_with_less_history_drops_levels_that_lost_their_value():
    rmap, _, now = _map_with_mas()
    short = bucket_bars(
        _frame(datetime(2026, 1, 5, 9, 0, tzinfo=UTC), timedelta(hours=1), _walk(100)), "1h",
    )
    rmap.build_moving_averages({"1h": short}, now=now)
    hourly = {l.source for l in rmap.moving_average_levels() if l.timeframe == "1h"}
    assert hourly == {"50 MA (1h)"}
    assert len([l for l in rmap.moving_average_levels() if l.timeframe == "daily"]) == 3


# ---------------------------------------------------------------------------
# Pipeline + scorer wiring
# ---------------------------------------------------------------------------

def test_pipeline_marks_ma_levels_on_the_map():
    from tests.test_files.test_pipeline import _DummyEvaluator, _DummyScorer, _DummyTradier, _make_entry
    from pipeline import run_pipeline

    class _Client(_DummyTradier, _FakeClient):
        def __init__(self) -> None:
            _DummyTradier.__init__(self)
            _FakeClient.__init__(self)

    entry = _make_entry()
    with patch("pipeline.MARKET_DATA_ENABLED", True), \
         patch("pipeline.stream_enabled", return_value=True), \
         patch("pipeline.DynamicScorer", _DummyScorer), \
         patch("pipeline.detect_move_start", return_value=SimpleNamespace(start_idx=5)), \
         patch("pipeline.load_candle_cache", return_value=pd.DataFrame()), \
         patch("pipeline._print_stock_summary", lambda *a, **k: None), \
         patch("config.STATIC_CONFIDENCE_MIN", 0), \
         patch("builtins.input", side_effect=AssertionError("input should not be called")):
        result = run_pipeline("TEST", _DummyEvaluator(entry), _Client(), quiet=True, allow_manual_rating=False)

    assert result is not None and result.ma_levels is not None
    assert result.ma_levels.result(timeout=10) > 0
    timeframes = {l.timeframe for l in result.rmap.moving_average_levels()}
    assert {"5m", "1h", "4h", "daily", "weekly", "monthly"} <= timeframes


def test_scorer_advances_ma_levels_on_each_closed_1m_bar():
    from tests.test_files.test_scorer import _make_scorer
    scorer = _make_scorer()
    calls = []
    scorer._rmap.update_moving_averages = lambda ts, close: calls.append((ts, close))
    t0 = datetime(2026, 11, 16, 15, 0, 5, tzinfo=UTC)
    with scorer._lock:
        for ts, px in ((t0, 5.0), (t0 + timedelta(seconds=30), 5.1), (t0 + timedelta(minutes=1), 5.2)):
            closed = scorer._tf_buf.on_tick(px, 100, ts)
            scorer._update_stream_indicators(ts, closed)
    assert calls == [(datetime(2026, 11, 16, 15, 0, tzinfo=UTC), 5.1)]
