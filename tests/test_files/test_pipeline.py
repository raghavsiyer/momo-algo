"""Regression tests for pipeline interaction behavior."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from config import SCORING_GROUPS
from market_data.finnhub_client import FinnhubNewsArticle
from pipeline import run_pipeline
from strategy.evaluator import Stage1Result, CompositeEvaluation, WatchlistEntry
from strategy.scoring import build_groups


def _make_evaluation() -> CompositeEvaluation:
    groups = build_groups(SCORING_GROUPS)
    gmap = {g.name: g for g in groups}
    for g in groups:
        for v in g.variables:
            v.quality = 0.5
            v.value = 0.5
    return CompositeEvaluation(
        stock_eval=gmap["STOCK_EVAL"],
        technical_analysis=gmap["TECHNICAL_ANALYSIS"],
    )


def _make_entry() -> WatchlistEntry:
    return WatchlistEntry(
        symbol="TEST",
        evaluated_at=datetime.now(tz=timezone.utc),
        price=5.0,
        change_pct=12.0,
        volume=500_000,
        avg_volume=100_000,
        rvol_1day=5.0,
        rvol_5min=2.5,
        daily_high=5.2,
        float_shares_m=2.0,
        free_float_pct=80.0,
        country="US",
        sector="Technology",
        industry="Software",
        ipo_date=None,
        days_since_ipo=None,
        has_recent_reverse_split=False,
        reverse_split_ratio=None,
        reverse_split_date=None,
        news_count_last_hour=1,
        news_is_fresh=True,
        news_is_breaking=True,
        avg_sentiment=None,
        evaluation=_make_evaluation(),
    )


class _DummyEvaluator:
    def __init__(self, entry: WatchlistEntry) -> None:
        self._entry = entry
        self.last_fail_reason = ""

    def stage1(self, symbol: str) -> Stage1Result:
        return Stage1Result(
            symbol=symbol,
            price=5.0,
            change_pct=12.0,
            volume=500_000,
            avg_volume=100_000,
            rvol_1day=5.0,
            rvol_5min=2.5,
            daily_high=5.2,
            passed=True,
        )

    def stage2(self, s1: Stage1Result, **kwargs) -> WatchlistEntry:
        return self._entry


class _DummyTradier:
    def __init__(self) -> None:
        self.one_min_window_calls = 0
        self.one_min_calls = 0
        self.ten_sec_calls = 0
        self.fifteen_min_calls = 0

    def get_candles_daily(self, symbol: str, days_back: int) -> pd.DataFrame:
        return pd.DataFrame()

    def get_candles_1min_window(self, symbol: str, start, end=None) -> pd.DataFrame:
        self.one_min_window_calls += 1
        base = pd.Timestamp("2026-06-02 20:00:00", tz="UTC")
        return pd.DataFrame([
            {
                "symbol": symbol,
                "ts": base + pd.Timedelta(minutes=i),
                "open": 4.0 + i * 0.01,
                "high": 4.05 + i * 0.01,
                "low": 3.95 + i * 0.01,
                "close": 4.02 + i * 0.01,
                "volume": 10_000 + i,
                "interval": "1min",
                "vwap": None,
            }
            for i in range(20)
        ])

    def get_candles_1min(self, symbol: str, n: int = 20) -> pd.DataFrame:
        self.one_min_calls += 1
        raise AssertionError("pipeline should use get_candles_1min_window once")

    def get_candles_10sec(self, symbol: str, n: int = 20) -> pd.DataFrame:
        self.ten_sec_calls += 1
        raise AssertionError("pipeline should not fetch 10s candles")

    def get_candles_15min(self, symbol: str, n: int = 20) -> pd.DataFrame:
        self.fifteen_min_calls += 1
        raise AssertionError("pipeline should derive 15m candles from the 1m seed")


class _DummyScorer:
    init_kwargs = None
    events = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).init_kwargs = kwargs
        type(self).events = []
        self.entry = args[0]
        self.started = False

    @staticmethod
    def _df_to_candles(df: pd.DataFrame, interval: str):
        return []

    def prime_from_seed_candles(self) -> None:
        type(self).events.append("prime")
        var = self.entry.evaluation.technical_analysis.get("key_candles")
        var.value = 0.42
        var.quality = 0.42

    def start(self) -> None:
        type(self).events.append("start")
        self.started = True


def test_manual_rating_can_be_disabled_for_noninteractive_paths() -> None:
    entry = _make_entry()
    evaluator = _DummyEvaluator(entry)
    tradier = _DummyTradier()

    with patch("pipeline.MARKET_DATA_ENABLED", False), patch("pipeline._print_stock_summary", lambda *a, **k: None), patch("config.STATIC_CONFIDENCE_MIN", 0), patch("builtins.input", side_effect=AssertionError("input should not be called")):
        result = run_pipeline(
            "TEST",
            evaluator,
            tradier,
            quiet=False,
            allow_manual_rating=False,
        )

    assert result is not None
    assert entry.avg_sentiment is None


def test_manual_rating_still_works_for_interactive_manual_runs() -> None:
    entry = _make_entry()
    evaluator = _DummyEvaluator(entry)
    tradier = _DummyTradier()

    with patch("pipeline.MARKET_DATA_ENABLED", False), patch("pipeline._print_stock_summary", lambda *a, **k: None), patch("config.STATIC_CONFIDENCE_MIN", 0), patch("builtins.input", return_value="4") as input_mock:
        result = run_pipeline("TEST", evaluator, tradier, quiet=False)

    assert result is not None
    input_mock.assert_called_once()
    assert entry.avg_sentiment == 0.4


def test_pipeline_applies_cached_news_before_summary() -> None:
    entry = _make_entry()
    entry.news_count_last_hour = 0
    evaluator = _DummyEvaluator(entry)
    tradier = _DummyTradier()
    article = FinnhubNewsArticle(
        symbol="TEST",
        title="TEST announces new catalyst",
        source="momo",
        url="",
        published_at=datetime.now(tz=timezone.utc),
        summary="",
    )
    captured = {}

    def _capture_summary(summary_entry, *args, **kwargs):
        captured["news_articles"] = list(summary_entry.news_articles)
        captured["news_source"] = summary_entry.news_source

    with patch("pipeline.MARKET_DATA_ENABLED", False), patch("pipeline._print_stock_summary", _capture_summary), patch("config.STATIC_CONFIDENCE_MIN", 0):
        result = run_pipeline(
            "TEST",
            evaluator,
            tradier,
            quiet=False,
            allow_manual_rating=False,
            cached_news=([article], 1, 0.6, "momo"),
        )

    assert result is not None
    assert captured["news_articles"][0].title == "TEST announces new catalyst"
    assert captured["news_source"] == "momo"


def test_pipeline_reuses_single_intraday_seed_for_scorer_and_derived_frames() -> None:
    entry = _make_entry()
    evaluator = _DummyEvaluator(entry)
    tradier = _DummyTradier()
    move_start_idx = 5
    cached_10s = pd.DataFrame([
        {
            "symbol": "TEST",
            "timeframe": "10s",
            "ts": pd.Timestamp("2026-06-02 20:05:00", tz="UTC"),
            "open": 5.0,
            "high": 5.1,
            "low": 4.9,
            "close": 5.05,
            "volume": 10_000,
            "vwap": 5.0,
            "source": "polled",
        }
    ])

    with patch("pipeline.MARKET_DATA_ENABLED", True), \
         patch("pipeline.stream_enabled", return_value=True), \
         patch("pipeline.DynamicScorer", _DummyScorer), \
         patch("pipeline.detect_move_start", return_value=SimpleNamespace(start_idx=move_start_idx)), \
         patch("pipeline.load_candle_cache", return_value=cached_10s), \
         patch("pipeline._print_stock_summary", lambda *a, **k: None), \
         patch("config.STATIC_CONFIDENCE_MIN", 0), \
         patch("builtins.input", side_effect=AssertionError("input should not be called")):
        result = run_pipeline(
            "TEST",
            evaluator,
            tradier,
            quiet=True,
            allow_manual_rating=False,
        )

    assert result is not None
    assert tradier.one_min_window_calls == 1
    assert tradier.one_min_calls == 0
    assert tradier.ten_sec_calls == 0
    assert tradier.fifteen_min_calls == 0

    scorer_kwargs = _DummyScorer.init_kwargs
    assert scorer_kwargs is not None
    assert _DummyScorer.events == ["prime", "start"]
    seed = scorer_kwargs["seed_1min_session"]
    assert len(seed) == 20
    assert scorer_kwargs["seed_10sec"].equals(cached_10s)
    assert scorer_kwargs["move_start_ts"] == pd.Timestamp(seed.iloc[move_start_idx]["ts"]).to_pydatetime()


def test_pipeline_primes_live_confidence_before_first_summary() -> None:
    entry = _make_entry()
    for variable in entry.evaluation.technical_analysis.variables:
        variable.value = None
        variable.quality = 0.0
    evaluator = _DummyEvaluator(entry)
    tradier = _DummyTradier()
    captured = {}

    def _capture_summary(summary_entry, *args, **kwargs):
        captured["has_live_data"] = summary_entry.has_live_data
        captured["live_confidence"] = summary_entry.live_confidence
        captured["events"] = list(_DummyScorer.events)

    with patch("pipeline.MARKET_DATA_ENABLED", True), \
         patch("pipeline.stream_enabled", return_value=True), \
         patch("pipeline.DynamicScorer", _DummyScorer), \
         patch("pipeline.detect_move_start", return_value=SimpleNamespace(start_idx=5)), \
         patch("pipeline.load_candle_cache", return_value=pd.DataFrame()), \
         patch("pipeline._print_stock_summary", _capture_summary), \
         patch("config.STATIC_CONFIDENCE_MIN", 0), \
         patch("builtins.input", side_effect=AssertionError("input should not be called")):
        result = run_pipeline(
            "TEST",
            evaluator,
            tradier,
            quiet=False,
            allow_manual_rating=False,
        )

    assert result is not None
    assert captured["has_live_data"] is True
    assert captured["live_confidence"] > 0
    assert captured["events"] == ["prime", "start"]
