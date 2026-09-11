"""
tests/test_stream_hotpath.py

Regression tests for the 2026-09-03 stream-thread stall fixes:

* pattern helpers short-circuit the frame inspection for known intraday labels
* the candle buffer exposes a copy-free closed-bars frame for per-print checks
* live stream events are parked, not blocked, while a scorer replays cached ticks
* chart-cache disk writes run on a background writer that coalesces and drains
"""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")

import threading
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from test_scorer import _make_scorer, _ts_event
from chart_cache import (
    ChartCacheWriter,
    LiveChartCache,
    SOURCE_STREAM,
    candle_cache_path,
    load_candle_cache,
)
from market_data.tradier_rest import OHLCVCandle
import technical_analysis.patterns as patterns
from technical_analysis.patterns import _is_daily_chart, _setup_timeframe_is_daily


def _bar(ts, o, h, l, c, v=100, interval="10s"):
    return OHLCVCandle(symbol="TEST", ts=ts, open=o, high=h, low=l, close=c,
                       volume=v, interval=interval, vwap=(h + l) / 2)


class TestIntradayShortCircuit:
    def test_known_intraday_label_skips_frame_inspection(self, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(patterns, "_is_daily_chart", lambda df: calls.append(1) or True)
        df = pd.DataFrame({"open": [1, 2], "high": [1, 2], "low": [1, 2], "close": [1, 2]})
        for tf in ("10sec", "1min", "2min", "5min"):
            assert _setup_timeframe_is_daily(df, tf) is False
        assert calls == []                                  # never inspected
        assert _setup_timeframe_is_daily(df, "1d") is True
        assert _setup_timeframe_is_daily(df, "weird") is True   # unknown label -> inspected
        assert calls == [1]

    def test_is_daily_chart_on_datetime_index(self):
        base = datetime(2026, 6, 1, tzinfo=timezone.utc)
        daily = pd.DataFrame(
            {"close": [1.0] * 4},
            index=pd.DatetimeIndex([base + timedelta(days=i) for i in range(4)], name="ts"),
        )
        intraday = pd.DataFrame(
            {"close": [1.0] * 4},
            index=pd.DatetimeIndex([base + timedelta(seconds=10 * i) for i in range(4)], name="ts"),
        )
        assert _is_daily_chart(daily) is True
        assert _is_daily_chart(intraday) is False
        assert _is_daily_chart(pd.DataFrame({"close": [1.0]})) is False

    def test_is_daily_chart_on_ts_column(self):
        base = datetime(2026, 6, 1, tzinfo=timezone.utc)
        df = pd.DataFrame({"ts": [base + timedelta(days=i) for i in range(3)], "close": [1.0] * 3})
        assert _is_daily_chart(df) is True
        df["ts"] = [base + timedelta(minutes=i) for i in range(3)]
        assert _is_daily_chart(df) is False


class TestClosedFrameAccessor:
    def test_closed_df_excludes_forming_bar_and_is_cached(self):
        scorer = _make_scorer(cache_live_data=False)
        base = datetime(2026, 6, 10, 10, 0, 0, tzinfo=timezone.utc)
        buf = scorer._tf_buf
        for i in range(4):
            buf.on_tick(5.0 + i * 0.01, 100, base + timedelta(seconds=10 * i))
        closed = buf.closed_df("10sec")
        assert len(closed) == 3                         # 3 closed + 1 forming
        assert closed is buf.closed_df("10sec")         # cached object, no rebuild
        full = buf.to_df("10sec")
        assert len(full) == 4
        pd.testing.assert_frame_equal(full.iloc[:-1], closed)
        buf.on_tick(5.10, 100, base + timedelta(seconds=40))
        assert len(buf.closed_df("10sec")) == 4         # rebuilt after a close


class TestEventHold:
    def test_replay_does_not_block_stream_thread_and_loses_nothing(self):
        base = datetime(2026, 6, 10, 14, 0, 0, tzinfo=timezone.utc)
        n_seed = 3000
        seed = pd.DataFrame([
            {
                "symbol": "TEST", "ts": base + timedelta(milliseconds=100 * i),
                "price": 5.0 + (i % 50) * 0.01, "size": 200, "bid": 4.99, "ask": 5.01,
                "exchange": "Q", "seq": i, "flag": "@", "session": "normal", "source": "stream",
            }
            for i in range(n_seed)
        ])
        seed_1m = pd.DataFrame([
            {
                "symbol": "TEST", "ts": base - timedelta(minutes=10) + timedelta(minutes=i),
                "open": 5.0, "high": 5.1, "low": 4.9, "close": 5.05, "volume": 1000,
                "interval": "1min", "vwap": None,
            }
            for i in range(10)
        ])
        scorer = _make_scorer(seed_1min_session=seed_1m, seed_timesales=seed, cache_live_data=False)
        assert not scorer._tf_seeded

        live_start = base + timedelta(minutes=6)
        n_live = 400
        latencies: list[float] = []

        def stream_thread() -> None:
            for i in range(n_live):
                t0 = time.perf_counter()
                scorer._on_timesale(_ts_event(
                    5.20 + (i % 10) * 0.01, 300, ts=live_start + timedelta(milliseconds=50 * i),
                ))
                latencies.append(time.perf_counter() - t0)
                time.sleep(0.002)

        t = threading.Thread(target=stream_thread)
        t.start()
        time.sleep(0.05)                                # let live prints hit the hold
        scorer._refresh_candles(use_seed_only=True, record_cache=False, notify=False)
        t.join(timeout=60)

        assert not t.is_alive()
        assert scorer._tf_seeded
        assert scorer._hold_events is False
        assert len(scorer._held_events) == 0
        # every print (seed replay + live) was processed exactly once
        assert scorer._prints_seen == n_seed + n_live
        # the stream thread never waited out the replay on the scorer lock
        assert max(latencies) < 0.5
        # order preserved: the last live print is the last price seen
        assert scorer._last_price == pytest.approx(5.20 + ((n_live - 1) % 10) * 0.01)

    def test_hold_is_lifted_when_replay_raises(self):
        scorer = _make_scorer(cache_live_data=False)
        with pytest.raises(RuntimeError):
            with scorer._event_hold(True):
                assert scorer._hold_events is True
                raise RuntimeError("boom")
        assert scorer._hold_events is False

    def test_inactive_hold_is_a_no_op(self):
        scorer = _make_scorer(cache_live_data=False)
        with scorer._event_hold(False):
            assert scorer._hold_events is False


class TestChartCacheWriter:
    def test_coalesces_and_drains(self, tmp_path):
        w = ChartCacheWriter(name="test-writer")
        cache = LiveChartCache("TEST", log_dir=tmp_path, writer=w)
        base = datetime(2026, 6, 10, 10, 0, 0, tzinfo=timezone.utc)
        for i in range(6):
            cache.record_candles(
                "10s", [_bar(base + timedelta(seconds=10 * i), 1, 1.1, 0.9, 1.05)], SOURCE_STREAM,
            )
        # a newer bar for an existing timestamp must win
        cache.record_candles("10s", [_bar(base, 1, 1.5, 0.9, 1.4)], SOURCE_STREAM)
        assert w.drain(timeout=30)
        out = load_candle_cache("TEST", "10s", base, base + timedelta(minutes=1), tmp_path)
        assert len(out) == 6
        assert float(out.iloc[0]["high"]) == pytest.approx(1.5)
        assert w.pending == 0

    def test_row_appends_go_through_writer(self, tmp_path):
        w = ChartCacheWriter(name="test-writer-rows")
        cache = LiveChartCache("TEST", log_dir=tmp_path, raw_flush_size=1000, raw_flush_seconds=999, writer=w)
        ts = datetime(2026, 6, 10, 10, 0, 1, tzinfo=timezone.utc)
        cache.record_timesale(_ts_event(1.00, 1, ts=ts, bid=0.99, ask=1.01))
        cache.flush_timesales()
        assert w.drain(timeout=30)
        from chart_cache import timesales_cache_path
        assert timesales_cache_path(tmp_path, "TEST", "20260610").exists()

    def test_sync_mode_without_writer_writes_inline(self, tmp_path):
        cache = LiveChartCache("TEST", log_dir=tmp_path)
        base = datetime(2026, 6, 10, 10, 0, 0, tzinfo=timezone.utc)
        cache.record_candles("1m", [_bar(base, 1, 1.1, 0.9, 1.05, interval="1m")], SOURCE_STREAM)
        assert candle_cache_path(tmp_path, "TEST", "1m", "20260610").exists()
        assert cache.drain() is True
