from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from main import _build_historical_live_snapshot, _parse_live_time_range


def test_parse_live_time_range_accepts_dash_and_space_forms():
    specs = [
        ("12-16", "12:00-16:00 PT"),
        ("12:10-16:05", "12:10-16:05 PT"),
        ("12:10:05-16:10:05", "12:10:05-16:10:05 PT"),
        ("12:10:05 16:10:05", "12:10:05-16:10:05 PT"),
        ("12 - 16", "12:00-16:00 PT"),
    ]
    pt = ZoneInfo("America/Los_Angeles")

    for spec, expected_label in specs:
        parsed = _parse_live_time_range(spec)
        assert parsed is not None
        start_utc, end_utc, label = parsed
        assert label == expected_label
        assert start_utc < end_utc
        assert start_utc.astimezone(pt).hour == 12


def test_parse_live_time_range_rejects_invalid_specs():
    for spec in ("25-16", "12:", "abc", "12-99"):
        assert _parse_live_time_range(spec) is None


class _DummyHistoricalClient:
    def __init__(self, start: datetime):
        self.start = start
        self.one_min_window = None
        self.ten_sec_window = None

    def _df(self, interval: timedelta, count: int) -> pd.DataFrame:
        rows = []
        for offset in range(count):
            ts = self.start + interval * offset
            rows.append({
                "symbol": "AZI",
                "ts": ts,
                "open": 10.0 + offset,
                "high": 10.5 + offset,
                "low": 9.8 + offset,
                "close": 10.2 + offset,
                "volume": 1000 + offset,
            })
        return pd.DataFrame(rows)

    def get_candles_1min_window(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.one_min_window = (symbol, start, end)
        return self._df(timedelta(minutes=1), 6)

    def get_candles_10sec_window(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        self.ten_sec_window = (symbol, start, end)
        return self._df(timedelta(seconds=10), 3)

    def get_candles_daily(self, symbol: str, days_back: int = 60) -> pd.DataFrame:
        return pd.DataFrame()


def test_historical_live_snapshot_uses_window_and_resamples():
    start = datetime(2026, 6, 9, 19, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=6)
    hist = _DummyHistoricalClient(start)

    snapshot = _build_historical_live_snapshot("AZI", hist, time_range=(start, end))

    assert snapshot is not None
    candle_map = snapshot.get_candle_snapshot()
    assert hist.one_min_window == ("AZI", start, end)
    assert hist.ten_sec_window == ("AZI", start, end)
    assert len(candle_map["1m"]) == 6
    assert len(candle_map["2m"]) == 3
    assert len(candle_map["5m"]) == 2
    assert len(candle_map["10s"]) == 3
    assert snapshot.get_candle_snapshot()["1m"] is not candle_map["1m"]


def test_historical_live_snapshot_full_premarket_clips_to_move_start(monkeypatch):
    start = datetime(2026, 6, 9, 8, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=6)
    hist = _DummyHistoricalClient(start)
    monkeypatch.setattr("technical_analysis.patterns.detect_move_start", lambda df: SimpleNamespace(start_idx=2))

    snapshot = _build_historical_live_snapshot("AZI", hist, time_range=(start, end), full_premarket=True)

    assert snapshot is not None
    candle_map = snapshot.get_candle_snapshot()
    move_start = start + timedelta(minutes=2)
    assert hist.one_min_window == ("AZI", start, end)
    assert hist.ten_sec_window == ("AZI", move_start, end)
    assert candle_map["1m"][0].ts == move_start
    assert all(candle.ts >= move_start for candle in candle_map["1m"])
