import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")

from types import SimpleNamespace

import pandas as pd

from main import (
    _apply_stage1_snapshot,
    _current_cached_fail_reasons,
    _ensure_cached_scorer,
    _refresh_cached_stage1_snapshot,
)
from strategy.criteria import build_criteria_groups
from strategy.evaluator import Stage1Result
from config import (
    ATTENTION_RVOL_1DAY_MIN,
    ATTENTION_VOLUME_MIN,
    CRITERIA_ATTENTION,
    CRITERIA_EXECUTION,
    RVOL_1DAY_MIN,
)
from tests.test_files.test_eval import _make_entry


def test_ensure_cached_scorer_starts_dynamic_scorer(monkeypatch):
    entry = _make_entry(symbol="CLWT", price=3.73, avg_volume=100_000)
    evaluator = SimpleNamespace(watchlist=[], watchlist_symbols=set())
    rmap = SimpleNamespace()
    scorers = {}
    events = []

    class DummyScorer:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def prime_from_seed_candles(self):
            events.append("prime")

        def start(self):
            events.append("start")

    monkeypatch.setattr("main.stream_enabled", lambda: True)
    monkeypatch.setattr("main.DynamicScorer", DummyScorer)
    monkeypatch.setattr("main._load_cached_seed_market_data", lambda _symbol: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr("main.build_daily", lambda df: df)
    monkeypatch.setattr(DummyScorer, "_df_to_candles", staticmethod(lambda _df, _interval: []), raising=False)

    tradier = SimpleNamespace(get_candles_daily=lambda _symbol, days_back=60: pd.DataFrame())

    assert _ensure_cached_scorer("CLWT", entry, rmap, evaluator, tradier, scorers)
    assert "CLWT" in scorers
    assert events == ["prime", "start"]
    assert evaluator.watchlist == [entry]
    assert evaluator.watchlist_symbols == {"CLWT"}


def test_current_cached_fail_reasons_include_low_rvol_until_refreshed_above_gate():
    entry = _make_entry(
        symbol="TNMG",
        price=0.39,
        volume=253_000,
        avg_volume=72_000,
        rvol_1day=3.52,
        rvol_5min=0.61,
    )

    reasons = _current_cached_fail_reasons(entry)
    assert reasons == [f"rvol_1day: 3.5x (< {RVOL_1DAY_MIN:.0f}x)"]

    snapshot = Stage1Result(
        symbol="TNMG",
        price=0.37,
        change_pct=15.2,
        volume=464_000,
        avg_volume=72_700,
        rvol_1day=6.38,
        rvol_5min=1.1,
        daily_high=0.40,
        passed=True,
    )

    assert _apply_stage1_snapshot(entry, snapshot)
    assert _current_cached_fail_reasons(entry, snapshot) == []


def test_refresh_cached_stage1_snapshot_can_clear_delayed_rvol_reading():
    entry = _make_entry(
        symbol="CHAI",
        price=0.56,
        volume=303_000,
        avg_volume=4_300_000,
        rvol_1day=0.07,
        rvol_5min=0.07,
    )

    snapshots = iter([
        Stage1Result(
            symbol="CHAI",
            price=0.57,
            change_pct=26.7,
            volume=473_000,
            avg_volume=4_300_000,
            rvol_1day=0.11,
            rvol_5min=0.11,
            daily_high=0.58,
            passed=True,
        ),
        Stage1Result(
            symbol="CHAI",
            price=0.58,
            change_pct=28.3,
            volume=488_000,
            avg_volume=4_300_000,
            rvol_1day=34.3,
            rvol_5min=0.11,
            daily_high=0.59,
            passed=True,
        ),
    ])
    evaluator = SimpleNamespace(stage1=lambda _ticker: next(snapshots))

    _, reasons = _refresh_cached_stage1_snapshot("CHAI", entry, evaluator)
    assert reasons == [f"rvol_1day: 0.1x (< {RVOL_1DAY_MIN:.0f}x)"]

    _, reasons = _refresh_cached_stage1_snapshot("CHAI", entry, evaluator)
    assert reasons == []
    assert entry.rvol_1day == 34.3


def test_apply_stage1_snapshot_refreshes_cached_price_change_and_rvol():
    entry = _make_entry(
        price=1.10,
        change_pct=58.9,
        volume=1_000_000,
        avg_volume=500_000,
        rvol_1day=2.0,
        rvol_5min=2.0,
        daily_high=1.20,
    )
    entry.criteria_groups = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)

    snapshot = Stage1Result(
        symbol="RMSG",
        price=1.38,
        change_pct=98.2,
        volume=max(ATTENTION_VOLUME_MIN + 1_000, 63_920_000),
        avg_volume=10_000_000,
        rvol_1day=max(ATTENTION_RVOL_1DAY_MIN + 1.0, 6.39),
        rvol_5min=8.7,
        daily_high=1.45,
        passed=False,
        fail_reason="change 98.2% < 100.0%",
    )

    assert _apply_stage1_snapshot(entry, snapshot)

    assert entry.price == 1.38
    assert entry.change_pct == 98.2
    assert entry.volume == snapshot.volume
    assert entry.avg_volume == 10_000_000
    assert entry.rvol_1day == snapshot.rvol_1day
    assert entry.rvol_5min == 8.7
    assert entry.daily_high == 1.45

    price_var = entry.evaluation.stock_eval.get("price")
    rvol_var = entry.evaluation.stock_eval.get("rvol_1day")
    assert price_var is not None
    assert rvol_var is not None
    assert price_var.value == 1.38
    assert rvol_var.value == snapshot.rvol_1day

    attention = entry.get_criteria_group("ATTENTION")
    assert attention is not None
    change_pct_min = attention.get("change_pct_min")
    volume_min = attention.get("volume_min")
    rvol_1day_min = attention.get("rvol_1day_min")
    assert change_pct_min is not None
    assert volume_min is not None
    assert rvol_1day_min is not None
    assert change_pct_min.value == snapshot.change_pct
    assert change_pct_min.passing
    assert volume_min.passing
    assert rvol_1day_min.passing
