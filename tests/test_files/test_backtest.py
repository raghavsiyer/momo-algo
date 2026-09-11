import pickle
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

import cache as cache_mod
import exec.backtest as backtest_mod
from cache import TickerCache
from chart_cache import (
    load_near_hod_window_cache,
    timesales_cache_path,
    write_candle_cache,
    write_near_hod_window_cache,
)
from config import MarketClock, SCORING_GROUPS, SETUP_PATTERN_GROUP
from exec.sim_logger import SimFill, SimTradeLogger
from exec.backtest import BacktestRunner
from exec.order_flow import OrderFlowTracker
from market_data.tradier_rest import OHLCVCandle
from strategy.criteria import CriteriaGroup, Criterion
from strategy.evaluator import CompositeEvaluation
from strategy.scoring import build_groups


def _et_ts(day: date, hour: int, minute: int = 0) -> float:
    return datetime.combine(
        day,
        datetime.min.time().replace(hour=hour, minute=minute),
        tzinfo=MarketClock.ET,
    ).astimezone(timezone.utc).timestamp()


def test_backtest_default_window_uses_configured_premarket_window(tmp_path):
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log")
    start_utc, end_utc = runner._window_bounds_utc(date(2026, 6, 10))
    local_tz = MarketClock.local_tz()
    start_local = start_utc.astimezone(local_tz)
    end_local = end_utc.astimezone(local_tz)

    assert (start_local.hour, start_local.minute) == (
        MarketClock.PREMARKET_WINDOW_START_LOCAL.hour,
        MarketClock.PREMARKET_WINDOW_START_LOCAL.minute,
    )
    assert (end_local.hour, end_local.minute) == (
        MarketClock.PREMARKET_WINDOW_END_LOCAL.hour,
        MarketClock.PREMARKET_WINDOW_END_LOCAL.minute,
    )


def test_backtest_summary_prints_sim_trade_results(capsys):
    start = datetime(2026, 6, 10, 11, 0, tzinfo=timezone.utc)
    result = backtest_mod.SymbolBacktestResult(
        symbol="VSME",
        stream_start_utc=start,
        stream_end_utc=start + timedelta(minutes=45),
        premarket_change_pct=212.4,
        ticks_processed=1234,
        tradier_calls=0,
        approved_setups=3,
        failure_detail="",
        faded_from_hod=False,
        final_eval=80.4,
        final_live=86.6,
        sim_trades=2,
        sim_pnl_dollars=1144.70,
        sim_realized_pct=0.2291,
        sim_winners=2,
        sim_losers=0,
    )

    backtest_mod._print_backtest_summary([result])

    out = capsys.readouterr().out
    assert "Sim trades: 2" in out
    assert "Sim P&L: $+1144.70" in out
    assert "TRADES" in out
    assert "P&L%" in out
    assert "VSME" in out
    assert "2/0" in out


def test_backtest_trade_results_prints_entry_exit_and_pullback_timeframe(capsys):
    buy_ts = datetime(2026, 6, 10, 11, 1, 5, tzinfo=timezone.utc)
    sell_ts = buy_ts + timedelta(minutes=3)
    pullback_ts = buy_ts - timedelta(minutes=2)
    breakout_ts = buy_ts - timedelta(seconds=10)
    breakout_price = 2.48
    fills = [
        SimFill(
            ts=buy_ts,
            symbol="VSME",
            side="BUY",
            fill_price=2.50,
            shares=1000,
            notional=2500.00,
            bid=2.49,
            ask=2.50,
            spread=0.01,
            entry_reason="entry gate approved",
            setup_name="pullback_new_high",
            setup_timeframe="1m",
            setup_pullback_ts=pullback_ts,
            setup_breakout_ts=breakout_ts,
            setup_breakout_price=breakout_price,
        ),
        SimFill(
            ts=sell_ts,
            symbol="VSME",
            side="SELL",
            fill_price=3.25,
            shares=1000,
            notional=3250.00,
            bid=3.25,
            ask=3.26,
            spread=0.01,
            exit_reason="sell_surge",
            setup_name="pullback_new_high",
            setup_timeframe="1m",
            setup_pullback_ts=pullback_ts,
            setup_breakout_ts=breakout_ts,
            setup_breakout_price=breakout_price,
            pnl_dollars=750.00,
            pnl_pct=0.30,
            hold_secs=180.0,
        ),
    ]

    backtest_mod._print_trade_results(fills)

    out = capsys.readouterr().out
    fmt_short = lambda ts: ts.astimezone(MarketClock.local_tz()).strftime("%H:%M:%S")
    assert "BUY" in out
    assert "SELL" in out
    assert "TF" in out
    assert "BREAKOUT" in out
    assert fmt_short(breakout_ts) in out
    assert "2.48" in out
    assert fmt_short(buy_ts) in out
    assert fmt_short(sell_ts) in out
    assert "1m" in out
    assert "sell_surge" in out


def test_backtest_replay_result_counts_sim_fills_logged_during_symbol(monkeypatch, tmp_path):
    start = datetime(2026, 6, 10, 11, 0, tzinfo=timezone.utc)
    sim_logger = SimTradeLogger(log_dir=str(tmp_path))
    monkeypatch.setattr(backtest_mod.SimulationEngine, "_logger", sim_logger)

    class Entry(SimpleNamespace):
        @property
        def eval_confidence(self):
            return 80.0

        @property
        def live_confidence(self):
            return 80.0

        @property
        def has_live_data(self):
            return True

    class FakeScorer:
        def __init__(self, entry, _rmap, **_kwargs):
            self._entry = entry

        def stop(self):
            logger = backtest_mod.SimulationEngine.get_logger()
            assert logger is not None
            logger.log_fill(SimFill(
                ts=start,
                symbol="VSME",
                side="BUY",
                fill_price=3.00,
                shares=100,
                notional=300.00,
                bid=2.99,
                ask=3.00,
                spread=0.01,
            ))
            logger.log_fill(SimFill(
                ts=start + timedelta(minutes=1),
                symbol="VSME",
                side="SELL",
                fill_price=3.50,
                shares=100,
                notional=350.00,
                bid=3.50,
                ask=3.51,
                spread=0.01,
                pnl_dollars=50.00,
                pnl_pct=0.1667,
                hold_secs=60.0,
            ))

    entry = Entry(
        symbol="VSME",
        avg_volume=100_000,
        price=3.25,
        daily_high=3.50,
        evaluation=None,
        setup_pattern_score=80.0,
        setup_price_level=3.00,
    )
    event = backtest_mod.BacktestEvent(
        symbol="VSME",
        stream_start_utc=start,
        first_seen_utc=start,
        updated_utc=None,
        entry=entry,
        rmap=SimpleNamespace(),
    )

    monkeypatch.setattr(backtest_mod, "DynamicScorer", FakeScorer)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    monkeypatch.setattr(runner, "_iter_ticks", lambda symbol, start_utc, end_utc: iter(()))

    result = runner._replay_symbol(event, start + timedelta(minutes=2))

    assert result.sim_trades == 1
    assert result.sim_pnl_dollars == 50.00
    assert result.sim_realized_pct == 50.00 / 300.00
    assert result.sim_winners == 1
    assert result.sim_losers == 0


def test_backtest_resets_cached_replay_criteria_state():
    entry = SimpleNamespace(criteria_groups=[
        CriteriaGroup(
            name="ATTENTION",
            criteria=[Criterion("vwap_slope_positive", required=True, passing=False, value=-0.0065)],
        ),
    ])

    backtest_mod._reset_replay_criteria(entry)

    criterion = entry.criteria_groups[0].criteria[0]
    assert criterion.passing is False
    assert criterion.value is None
    assert criterion.updated_at is None


def test_backtest_seeds_replay_stream_vwap_from_cached_session_candles():
    class FakeScorer:
        pass

    scorer = FakeScorer()
    start = datetime(2026, 6, 10, 11, 0, tzinfo=timezone.utc)
    candles = [
        OHLCVCandle("VSME", start, 5.00, 5.10, 4.90, 5.00, 1000, "1min", vwap=5.00),
        OHLCVCandle("VSME", start + timedelta(minutes=1), 5.00, 5.20, 4.95, 5.15, 1200, "1min", vwap=5.10),
        OHLCVCandle("VSME", start + timedelta(minutes=2), 5.15, 5.40, 5.10, 5.35, 1400, "1min", vwap=5.25),
    ]

    backtest_mod.BacktestRunner._seed_replay_stream_vwap(scorer, candles)

    expected = ((5.00 * 1000) + (5.10 * 1200) + (5.25 * 1400)) / 3600
    assert scorer._vwap_session_key == "2026-06-10-0"
    assert scorer._vwap_cum_vol_closed == 3600
    assert scorer._stream_vwap == expected
    assert scorer._last_vwap == expected
    assert list(scorer._vwap_closed_history)[-1] == expected


def test_backtest_quote_from_tick_does_not_treat_trade_size_as_ask_size():
    ts = datetime(2026, 6, 10, 11, 0, tzinfo=timezone.utc)
    tick = backtest_mod.LiveTimesale(
        event_type="timesale",
        symbol="VSME",
        exchange="Q",
        seq=1,
        bid=4.99,
        ask=5.00,
        last=5.00,
        size=50_000,
        flag="",
        cancel=False,
        correction=False,
        session="pre",
        ts=ts,
    )

    quote = BacktestRunner._quote_from_tick(tick)

    assert quote.bid == 4.99
    assert quote.ask == 5.00
    assert quote.bid_size == 1
    assert quote.ask_size == 1


def test_near_hod_window_cache_roundtrip(tmp_path):
    replay_start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    replay_end = replay_start + timedelta(minutes=10)
    window_start = replay_start + timedelta(minutes=3)
    window_end = replay_start + timedelta(minutes=5)

    write_near_hod_window_cache(
        "RMSG",
        [{
            "symbol": "RMSG",
            "replay_start_ts": replay_start,
            "replay_end_ts": replay_end,
            "start_ts": window_start,
            "end_ts": window_end,
            "start_price": 5.20,
            "start_hod": 5.50,
            "start_atr_14": 0.18,
            "threshold_atrs": 2.0,
            "source": "replay",
        }],
        log_dir=tmp_path,
    )

    cached = load_near_hod_window_cache("RMSG", replay_start, replay_end, tmp_path)

    assert len(cached) == 1
    assert cached.iloc[0]["start_ts"] == pd.Timestamp(window_start)
    assert cached.iloc[0]["end_ts"] == pd.Timestamp(window_end)
    assert float(cached.iloc[0]["start_atr_14"]) == 0.18


def test_snapshot_day_includes_entries_updated_on_day(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "_SNAPSHOT_DIR", tmp_path)
    ticker_cache = TickerCache(path=tmp_path / "ticker_cache.pkl")
    day = date(2026, 6, 5)
    prior_day = date(2026, 6, 4)

    ticker_cache._data["RMSG"] = (
        SimpleNamespace(symbol="RMSG"),
        SimpleNamespace(),
        _et_ts(prior_day, 17, 5),
        _et_ts(day, 8, 46),
    )
    ticker_cache._data["OLD"] = (
        SimpleNamespace(symbol="OLD"),
        SimpleNamespace(),
        _et_ts(prior_day, 9, 30),
        _et_ts(prior_day, 9, 35),
    )

    path = ticker_cache.snapshot_day(day)

    with path.open("rb") as handle:
        entries = pickle.load(handle)["entries"]
    assert "RMSG" in entries
    assert "OLD" not in entries


def test_backtest_uses_updated_time_for_prior_day_cache_entry(monkeypatch, tmp_path):
    day = date(2026, 6, 5)
    prior_day = date(2026, 6, 4)
    first_seen = _et_ts(prior_day, 17, 5)
    updated = _et_ts(day, 8, 46)
    entries = {
        "RMSG": (
            SimpleNamespace(symbol="RMSG", eval_confidence=78.3, live_confidence=0.0, price=3.10),
            SimpleNamespace(),
            first_seen,
            updated,
        )
    }

    monkeypatch.setattr(TickerCache, "load_snapshot", staticmethod(lambda _day: entries))
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())

    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log")
    events = runner._build_day_events(day)

    assert len(events) == 1
    assert events[0].symbol == "RMSG"
    assert events[0].first_seen_utc.astimezone(MarketClock.ET).date() == prior_day
    assert events[0].stream_start_utc == datetime.fromtimestamp(updated, tz=timezone.utc)


def test_backtest_replays_same_day_cache_entry_from_window_start(monkeypatch, tmp_path):
    day = date(2026, 6, 5)
    entries = {
        "EARLY": (
            SimpleNamespace(symbol="EARLY", eval_confidence=78.3, live_confidence=0.0, price=3.10),
            SimpleNamespace(),
            _et_ts(day, 4, 50),
            _et_ts(day, 11, 50),
        )
    }

    monkeypatch.setattr(TickerCache, "load_snapshot", staticmethod(lambda _day: entries))
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())

    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log")
    events = runner._build_day_events(day)

    assert len(events) == 1
    assert events[0].symbol == "EARLY"
    assert events[0].stream_start_utc == runner._window_bounds_utc(day)[0]


def test_today_backtest_refreshes_snapshot_from_live_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_mod, "_SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())
    day = MarketClock.now_et().date()
    prior_day = day - timedelta(days=1)
    live_cache_path = tmp_path / "ticker_cache.pkl"

    stale_snapshot = tmp_path / f"{day.isoformat()}.pkl"
    with stale_snapshot.open("wb") as handle:
        pickle.dump({"entries": {}}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    live_entries = {
        "RMSG": (
            SimpleNamespace(symbol="RMSG", eval_confidence=78.3, live_confidence=0.0, price=3.10),
            SimpleNamespace(),
            _et_ts(prior_day, 17, 5),
            _et_ts(day, 8, 46),
        )
    }
    with live_cache_path.open("wb") as handle:
        pickle.dump(
            {"entries": live_entries, "pending_yt_news": {}},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    runner = BacktestRunner(cache_path=live_cache_path, eval_stats_path=tmp_path / "missing.log")
    entries = runner._load_cache_entries(day)

    assert "RMSG" in entries
    with stale_snapshot.open("rb") as handle:
        refreshed = pickle.load(handle)["entries"]
    assert "RMSG" in refreshed


def test_backtest_replay_resets_cached_live_ta_state(monkeypatch, tmp_path):
    groups = build_groups(SCORING_GROUPS)
    gmap = {group.name: group for group in groups}
    ta_var = gmap["TECHNICAL_ANALYSIS"].get("setup_outcome")
    assert ta_var is not None
    ta_var.value = 0.75
    ta_var.quality = 0.25
    rvol_var = gmap["STOCK_EVAL"].get("rvol_1day")
    assert rvol_var is not None
    rvol_var.value = 9.99
    rvol_var.quality = 1.0
    news_var = gmap["STOCK_EVAL"].get("breaking_news")
    assert news_var is not None
    news_var.value = 0.95
    news_var.quality = 0.95
    setup_pattern = build_groups([SETUP_PATTERN_GROUP])[0]
    setup_var = setup_pattern.get("pullback_depth")
    assert setup_var is not None
    setup_var.value = 0.18
    setup_var.quality = 0.95

    evaluation = CompositeEvaluation(
        stock_eval=gmap["STOCK_EVAL"],
        technical_analysis=gmap["TECHNICAL_ANALYSIS"],
    )

    class Entry(SimpleNamespace):
        @property
        def eval_confidence(self):
            return 50.0

        @property
        def live_confidence(self):
            return 50.0

        @property
        def has_live_data(self):
            return True

    setup_ts = datetime(2026, 6, 5, 11, 58, tzinfo=timezone.utc)
    entry = Entry(
        symbol="RMSG",
        avg_volume=100_000,
        price=3.10,
        daily_high=99.00,
        volume=999_999,
        rvol_1day=9.99,
        evaluation=evaluation,
        news_articles=[SimpleNamespace(published_at=datetime(2026, 6, 5, 12, 30, tzinfo=timezone.utc), entity_sentiment=0.9)],
        news_count_last_hour=1,
        news_is_fresh=True,
        news_is_breaking=True,
        avg_sentiment=0.9,
        setup_pattern=setup_pattern,
        setup_pattern_name="pullback_pending",
        setup_pattern_timeframe="1min",
        setup_pullback_depth=0.18,
        setup_price_level=3.25,
        setup_pullback_ts=setup_ts,
        setup_resolved_at=123.0,
        setup_failed_at=456.0,
    )
    captured = {}

    class FakeScorer:
        def __init__(self, entry, rmap, **kwargs):
            captured["entry"] = entry
            captured["cache_live_data"] = kwargs.get("cache_live_data")

        def stop(self):
            pass

    monkeypatch.setattr(backtest_mod, "DynamicScorer", FakeScorer)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())

    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log")
    monkeypatch.setattr(runner, "_iter_ticks", lambda symbol, start_utc, end_utc: iter(()))

    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    event = backtest_mod.BacktestEvent(
        symbol="RMSG",
        stream_start_utc=start,
        first_seen_utc=start,
        updated_utc=None,
        entry=entry,
        rmap=SimpleNamespace(),
    )

    runner._replay_symbol(event, start + timedelta(minutes=1))

    replay_ta_var = captured["entry"].evaluation.technical_analysis.get("setup_outcome")
    assert replay_ta_var is not None
    assert replay_ta_var.value is None
    assert replay_ta_var.quality == 0.5
    assert captured["entry"].daily_high == 0.0
    assert captured["entry"].volume == 0
    assert captured["entry"].rvol_1day == 0.0
    replay_rvol_var = captured["entry"].evaluation.stock_eval.get("rvol_1day")
    assert replay_rvol_var is not None
    assert replay_rvol_var.value is None
    replay_news_var = captured["entry"].evaluation.stock_eval.get("breaking_news")
    assert replay_news_var is not None
    assert replay_news_var.value is None
    assert captured["entry"].news_articles == []
    assert captured["entry"].avg_sentiment is None
    assert captured["entry"].setup_pattern_name == ""
    replay_setup_var = captured["entry"].setup_pattern.get("pullback_depth")
    assert replay_setup_var is not None
    assert replay_setup_var.value is None
    assert replay_setup_var.quality == 0.0
    assert captured["cache_live_data"] is False


def test_backtest_seeds_scorer_from_cached_market_data(monkeypatch, tmp_path):
    groups = build_groups(SCORING_GROUPS)
    gmap = {group.name: group for group in groups}
    evaluation = CompositeEvaluation(
        stock_eval=gmap["STOCK_EVAL"],
        technical_analysis=gmap["TECHNICAL_ANALYSIS"],
    )

    class Entry(SimpleNamespace):
        @property
        def eval_confidence(self):
            return 50.0

        @property
        def live_confidence(self):
            return 50.0

        @property
        def has_live_data(self):
            return True

    stream_start = datetime(2026, 6, 5, 12, 5, tzinfo=timezone.utc)
    seed_ts = stream_start - timedelta(minutes=1)
    write_candle_cache(
        "RMSG",
        "1m",
        pd.DataFrame([{"ts": seed_ts, "open": 3.00, "high": 3.10, "low": 2.95, "close": 3.05, "volume": 1000}]),
        log_dir=tmp_path,
    )
    write_candle_cache(
        "RMSG",
        "10s",
        pd.DataFrame([{"ts": seed_ts, "open": 3.00, "high": 3.08, "low": 2.98, "close": 3.04, "volume": 400}]),
        log_dir=tmp_path,
    )
    path = timesales_cache_path(tmp_path, "RMSG", "20260605")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "symbol,ts,price,size,bid,ask,exchange,seq,flag,session,source\n"
        "RMSG,2026-06-05T12:04:30+00:00,3.04,200,3.03,3.04,Q,7,,pre,stream\n"
    )

    entry = Entry(symbol="RMSG", avg_volume=100_000, price=3.10, daily_high=99.00, evaluation=evaluation)
    captured = {}

    class FakeScorer:
        def __init__(self, entry, rmap, **kwargs):
            captured["kwargs"] = kwargs

        def stop(self):
            pass

    monkeypatch.setattr(backtest_mod, "DynamicScorer", FakeScorer)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    monkeypatch.setattr(runner, "_iter_ticks", lambda symbol, start_utc, end_utc: iter(()))
    event = backtest_mod.BacktestEvent(
        symbol="RMSG",
        stream_start_utc=stream_start,
        first_seen_utc=stream_start,
        updated_utc=None,
        entry=entry,
        rmap=SimpleNamespace(),
    )

    runner._replay_symbol(event, stream_start + timedelta(minutes=1))

    assert len(captured["kwargs"]["seed_1min_session"]) == 1
    assert len(captured["kwargs"]["seed_10sec"]) == 1
    assert len(captured["kwargs"]["seed_timesales"]) == 1


def test_backtest_ta_refresh_skips_when_far_from_rolling_hod():
    class Entry(SimpleNamespace):
        pass

    class FakeScorer:
        def __init__(self):
            self._sym = "RMSG"
            self._entry = Entry(daily_high=0.0)
            self._last_price = 0.0
            self._last_atr = 0.0
            self.calls: list[str] = []

        def _update_price_action_quality(self, tf_label, candles):
            self.calls.append(tf_label)

    scorer = FakeScorer()
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    ticks = [
        (start + timedelta(minutes=0), 10.00, 100),
        (start + timedelta(minutes=1), 9.80, 100),
        (start + timedelta(minutes=2), 9.50, 100),
    ]

    backtest_mod.BacktestRunner._refresh_ta_from_ticks(scorer, ticks)

    assert scorer._entry.daily_high == 10.00
    assert scorer._last_atr > 0
    assert scorer.calls == []


def test_backtest_ta_refresh_updates_when_near_rolling_hod():
    class Entry(SimpleNamespace):
        pass

    class FakeScorer:
        def __init__(self):
            self._sym = "RMSG"
            self._entry = Entry(daily_high=0.0)
            self._last_price = 0.0
            self._last_atr = 0.0
            self.calls: list[str] = []

        def _update_price_action_quality(self, tf_label, candles):
            self.calls.append(tf_label)

    scorer = FakeScorer()
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    ticks = [
        (start + timedelta(minutes=0), 10.00, 100),
        (start + timedelta(minutes=0, seconds=20), 9.80, 100),
        (start + timedelta(minutes=1), 9.95, 100),
        (start + timedelta(minutes=1, seconds=20), 9.88, 100),
        (start + timedelta(minutes=2), 9.96, 100),
        (start + timedelta(minutes=2, seconds=20), 9.94, 100),
    ]

    backtest_mod.BacktestRunner._refresh_ta_from_ticks(scorer, ticks)

    assert scorer._entry.daily_high == 10.00
    assert scorer._last_atr > 0
    assert "1m" in scorer.calls


def test_backtest_ta_refresh_keeps_session_1min_frame_current():
    """Replay must extend _df_1min_session with tick-built bars.

    The attention criteria (MACD, close-vs-VWAP) read _df_1min_session in
    preference to the df they are handed; live it is refreshed by the REST
    poll loop, but the replay has no poll loop — without the merge the frame
    stays frozen at its pre-stream seed and every criteria snapshot in the
    diagnostics reads stale closes/MACD (observed on the 2026-08-05 YXT
    replay: macd_hist identical across the whole session)."""
    class Entry(SimpleNamespace):
        pass

    class FakeScorer:
        def __init__(self, seed_df):
            self._sym = "RMSG"
            self._entry = Entry(daily_high=0.0)
            self._last_price = 0.0
            self._last_atr = 0.0
            self._df_1min_session = seed_df
            self.calls: list[str] = []

        def _update_price_action_quality(self, tf_label, candles):
            self.calls.append(tf_label)

    seed_start = datetime(2026, 6, 5, 11, 0, tzinfo=timezone.utc)
    seed = pd.DataFrame([
        {
            "symbol": "RMSG",
            "ts": seed_start + timedelta(minutes=i),
            "open": 3.70, "high": 3.80, "low": 3.60, "close": 3.74,
            "volume": 1_000, "vwap": None,
        }
        for i in range(3)
    ])
    scorer = FakeScorer(seed.copy())
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    ticks = [
        (start + timedelta(minutes=0), 10.00, 100),
        (start + timedelta(minutes=1), 9.80, 100),
        (start + timedelta(minutes=2), 9.50, 100),
    ]

    backtest_mod.BacktestRunner._refresh_ta_from_ticks(scorer, ticks)

    session = scorer._df_1min_session
    # Seed rows retained, tick-built bars appended — the frame is current.
    assert len(session) == 6
    last_ts = pd.to_datetime(session["ts"], utc=True).max()
    assert last_ts == pd.Timestamp(start + timedelta(minutes=2))
    assert float(session.iloc[-1]["close"]) == 9.50
    # Seed rows are still the oldest bars (merge, not replace).
    assert float(session.iloc[0]["close"]) == 3.74


def test_backtest_seed_uses_cached_atr_column():
    class Entry(SimpleNamespace):
        pass

    class TfBuffer:
        def seed(self, _timeframe, _candles):
            pass

    class FakeScorer:
        def __init__(self):
            self._sym = "RMSG"
            self._entry = Entry(daily_high=0.0)
            self._last_price = 0.0
            self._last_atr = 0.0
            self._tf_buf = TfBuffer()
            self.calls: list[str] = []

        def _update_price_action_quality(self, tf_label, candles):
            self.calls.append(tf_label)

    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    df_1m_seed = pd.DataFrame([
        {"symbol": "RMSG", "ts": start, "open": 5.00, "high": 5.10, "low": 4.90, "close": 5.00, "volume": 1000, "vwap": 5.00, "atr_14": 0.00},
        {"symbol": "RMSG", "ts": start + timedelta(minutes=1), "open": 5.00, "high": 5.20, "low": 4.95, "close": 5.15, "volume": 1200, "vwap": 5.10, "atr_14": 0.20},
        {"symbol": "RMSG", "ts": start + timedelta(minutes=2), "open": 5.15, "high": 5.40, "low": 5.10, "close": 5.35, "volume": 1400, "vwap": 5.25, "atr_14": 9.99},
    ])
    scorer = FakeScorer()

    backtest_mod.BacktestRunner._seed_scorer_from_cached_market_data(
        scorer,
        df_1m_seed,
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert scorer._entry.daily_high == 5.40
    assert scorer._last_price == 5.35
    assert scorer._last_atr == 9.99


def test_backtest_seed_excludes_unclosed_cached_candles(tmp_path):
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    stream_start = start + timedelta(seconds=30)
    write_candle_cache(
        "RMSG",
        "1m",
        pd.DataFrame([
            {"ts": start - timedelta(minutes=1), "open": 4.90, "high": 5.05, "low": 4.85, "close": 5.00, "volume": 100},
            {"ts": start, "open": 5.00, "high": 9.99, "low": 4.95, "close": 9.50, "volume": 10_000},
        ]),
        log_dir=tmp_path,
    )
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)

    df_1m, _, _ = runner._load_cached_seed_market_data("RMSG", stream_start)

    assert len(df_1m) == 1
    assert pd.Timestamp(df_1m.iloc[0]["ts"]) == pd.Timestamp(start - timedelta(minutes=1))


def test_backtest_replay_volume_starts_from_seed_data(monkeypatch, tmp_path):
    stream_start = datetime(2026, 6, 5, 12, 5, tzinfo=timezone.utc)
    seed_ts = stream_start - timedelta(minutes=1)
    write_candle_cache(
        "RMSG",
        "1m",
        pd.DataFrame([
            {"ts": seed_ts, "open": 3.00, "high": 3.10, "low": 2.95, "close": 3.05, "volume": 1200},
        ]),
        log_dir=tmp_path,
    )
    path = timesales_cache_path(tmp_path, "RMSG", "20260605")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "symbol,ts,price,size,bid,ask,exchange,seq,flag,session,source\n"
        "RMSG,2026-06-05T12:04:30+00:00,3.04,5000,3.03,3.04,Q,7,,pre,stream\n"
    )

    class Entry(SimpleNamespace):
        @property
        def eval_confidence(self):
            return 50.0

        @property
        def live_confidence(self):
            return 50.0

        @property
        def has_live_data(self):
            return False

    entry = Entry(
        symbol="RMSG",
        avg_volume=100_000,
        price=3.10,
        daily_high=0.0,
        volume=999_999,
        rvol_1day=9.99,
        evaluation=None,
    )
    captured = {}

    class FakeScorer:
        def __init__(self, entry, _rmap, **_kwargs):
            captured["entry"] = entry

        def stop(self):
            pass

    monkeypatch.setattr(backtest_mod, "DynamicScorer", FakeScorer)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)
    monkeypatch.setattr(runner, "_iter_ticks", lambda symbol, start_utc, end_utc: iter(()))

    runner._replay_symbol(
        backtest_mod.BacktestEvent("RMSG", stream_start, stream_start, None, entry, SimpleNamespace()),
        stream_start + timedelta(minutes=1),
    )

    assert captured["entry"].volume == 5000
    assert captured["entry"].rvol_1day == 0.05


def test_backtest_derives_near_hod_windows_from_cached_atr_candles(tmp_path):
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=4)
    write_candle_cache(
        "RMSG",
        "1m",
        pd.DataFrame([
            {"ts": start, "open": 10.00, "high": 10.00, "low": 10.00, "close": 10.00, "volume": 100},
            {"ts": start + timedelta(minutes=1), "open": 9.40, "high": 9.41, "low": 9.39, "close": 9.40, "volume": 200},
            {"ts": start + timedelta(minutes=2), "open": 9.00, "high": 9.01, "low": 8.99, "close": 9.00, "volume": 300},
            {"ts": start + timedelta(minutes=3), "open": 9.70, "high": 9.71, "low": 9.69, "close": 9.70, "volume": 400},
        ]),
        log_dir=tmp_path,
    )
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)

    windows = runner._derive_near_hod_windows_from_candle_cache("RMSG", start, end)

    assert len(windows) == 2
    assert windows[0].start_ts == start + timedelta(minutes=2) - timedelta(microseconds=1)
    assert windows[0].end_ts == start + timedelta(minutes=3) - timedelta(microseconds=1)
    assert windows[0].start_volume == 300
    assert runner._tick_in_near_hod_windows(start + timedelta(minutes=2), windows, 0) == (True, 0)
    assert windows[1].start_ts == start + timedelta(minutes=4) - timedelta(microseconds=1)
    assert windows[1].start_volume == 1000
    cached = runner._load_cached_near_hod_windows("RMSG", start, end)
    assert len(cached) == 2


def test_backtest_loads_legacy_candle_cache_windows_at_bar_close(tmp_path):
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=4)
    legacy_bar_start = start + timedelta(minutes=1)
    legacy_bar_end = legacy_bar_start + timedelta(minutes=1) - timedelta(microseconds=1)
    write_near_hod_window_cache(
        "RMSG",
        [{
            "symbol": "RMSG",
            "replay_start_ts": start,
            "replay_end_ts": end,
            "start_ts": legacy_bar_start,
            "end_ts": legacy_bar_end,
            "start_price": 9.40,
            "start_hod": 10.00,
            "start_atr_14": 0.10,
            "start_volume": 300,
            "threshold_atrs": 2.0,
            "source": "candle_cache",
        }],
        log_dir=tmp_path,
    )
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)

    windows = runner._load_cached_near_hod_windows("RMSG", start, end)

    assert len(windows) == 1
    assert windows[0].start_ts == legacy_bar_end
    assert windows[0].end_ts == legacy_bar_end + timedelta(minutes=1)
    assert runner._tick_in_near_hod_windows(start + timedelta(minutes=2), windows, 0) == (True, 0)


def test_backtest_loads_old_close_based_hod_windows_through_next_close(tmp_path):
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=4)
    bar_end = start + timedelta(minutes=2) - timedelta(microseconds=1)
    write_near_hod_window_cache(
        "RMSG",
        [{
            "symbol": "RMSG",
            "replay_start_ts": start,
            "replay_end_ts": end,
            "start_ts": bar_end,
            "end_ts": bar_end,
            "start_price": 9.40,
            "start_hod": 10.00,
            "start_atr_14": 0.10,
            "start_volume": 300,
            "threshold_atrs": 2.0,
            "source": "candle_cache_close",
        }],
        log_dir=tmp_path,
    )
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)

    windows = runner._load_cached_near_hod_windows("RMSG", start, end)

    assert len(windows) == 1
    assert windows[0].start_ts == bar_end
    assert windows[0].end_ts == bar_end + timedelta(minutes=1)
    assert runner._tick_in_near_hod_windows(start + timedelta(minutes=2), windows, 0) == (True, 0)


def test_backtest_derives_pullback_replay_windows_from_cached_candles(tmp_path):
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=7)
    # Green leg bars 0-3 must exceed PATTERN_PULLBACK_MIN_LEG_PCT (10%).
    # Leg open=9.50 → leg high=10.72 → move=12.8%.
    write_candle_cache(
        "RMSG",
        "1m",
        pd.DataFrame([
            {"ts": start, "open": 9.50, "high": 9.80, "low": 9.48, "close": 9.75, "volume": 100},
            {"ts": start + timedelta(minutes=1), "open": 9.75, "high": 10.10, "low": 9.73, "close": 10.05, "volume": 100},
            {"ts": start + timedelta(minutes=2), "open": 10.05, "high": 10.40, "low": 10.03, "close": 10.35, "volume": 100},
            {"ts": start + timedelta(minutes=3), "open": 10.35, "high": 10.72, "low": 10.33, "close": 10.62, "volume": 100},
            {"ts": start + timedelta(minutes=4), "open": 10.62, "high": 10.64, "low": 10.38, "close": 10.45, "volume": 100},
            {"ts": start + timedelta(minutes=5), "open": 10.45, "high": 10.82, "low": 10.43, "close": 10.76, "volume": 100},
        ]),
        log_dir=tmp_path,
    )
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)
    runner._tradier = SimpleNamespace()
    near_hod_windows = [backtest_mod.CachedNearHodWindow(start + timedelta(minutes=4), end, 9.50, 10.72, 0.20, 0)]

    windows, available = runner._derive_pullback_replay_windows_from_candle_cache("RMSG", start, end, near_hod_windows)

    assert available is True
    assert len(windows) == 1
    # Window starts at the pullback bar's OPEN (min4, the red candle) so the
    # scorer rebuilds that candle from replayed ticks — a close-anchored start
    # dropped it between the candle seed and the tick stream.
    assert windows[0].start_ts == start + timedelta(minutes=4)
    # Resolution bar end (min6-1µs) + BACKTEST_PULLBACK_WINDOW_TAIL_BARS bars,
    # clamped to end_utc — the resumption bars stay inside replay coverage.
    assert windows[0].end_ts == start + timedelta(minutes=7)


def test_backtest_window_merge_bridges_sub_minute_gaps():
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)

    def _w(offset_secs, end_offset_secs):
        return backtest_mod.CachedNearHodWindow(
            start_ts=start + timedelta(seconds=offset_secs),
            end_ts=start + timedelta(seconds=end_offset_secs),
            start_price=9.50, start_hod=10.00, start_atr_14=0.10, start_volume=0,
        )

    # Gap of exactly BACKTEST_REPLAY_WINDOW_BRIDGE_SECS (60s) → coalesced:
    # one marginal bar outside the near-HOD gate must not split coverage.
    merged = backtest_mod.BacktestRunner._merge_near_hod_windows([_w(0, 120), _w(180, 240)])
    assert len(merged) == 1
    assert merged[0].start_ts == start
    assert merged[0].end_ts == start + timedelta(seconds=240)

    # A materially larger gap stays split.
    apart = backtest_mod.BacktestRunner._merge_near_hod_windows([_w(0, 120), _w(300, 360)])
    assert len(apart) == 2


def test_backtest_empty_cached_replay_windows_skip_tick_loading(monkeypatch, tmp_path):
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)
    monkeypatch.setattr(runner, "_load_cached_timesales", lambda *_args: (_ for _ in ()).throw(AssertionError("loaded timesales")))

    rows = list(runner._iter_ticks(
        "RMSG",
        datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 5, 12, 5, tzinfo=timezone.utc),
        [],
    ))

    assert rows == []


def test_backtest_replay_skips_handlers_outside_cached_near_hod_windows(monkeypatch, tmp_path):
    class Entry(SimpleNamespace):
        @property
        def eval_confidence(self):
            return 80.0

        @property
        def live_confidence(self):
            return 80.0

        @property
        def has_live_data(self):
            return False

        def get_criteria_group(self, _name):
            return None

    class FakeScorer:
        def __init__(self, entry, _rmap, **_kwargs):
            self._entry = entry
            self._sym = entry.symbol
            self._avg_vol = entry.avg_volume
            self._last_price = entry.price
            self._last_atr = 0.10
            self._prev_close = 9.50
            self._bid = 0.0
            self._ask = 0.0
            self._setup_logger = None
            self._ofi_tracker = SimpleNamespace(buying_pressure=999, selling_pressure=0, activity_count=999)
            self.quote_count = 0
            self.timesale_count = 0
            captured["scorer"] = self

        def _on_quote(self, _quote):
            self.quote_count += 1

        def _on_timesale(self, tick):
            self.timesale_count += 1
            self._last_price = tick.last
            self._entry.volume = int(getattr(self._entry, "volume", 0) or 0) + tick.size
            if tick.last > self._entry.daily_high:
                self._entry.daily_high = tick.last

        def stop(self):
            pass

    captured: dict[str, FakeScorer] = {}
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=3)
    window_ts = start + timedelta(minutes=1)
    write_near_hod_window_cache(
        "RMSG",
        [{
            "symbol": "RMSG",
            "replay_start_ts": start,
            "replay_end_ts": end,
            "start_ts": window_ts,
            "end_ts": window_ts,
            "start_price": 9.90,
            "start_hod": 10.00,
            "start_atr_14": 0.10,
            "threshold_atrs": 2.0,
            "source": "replay",
        }],
        log_dir=tmp_path,
    )
    ticks = [
        backtest_mod.LiveTimesale("timesale", "RMSG", "Q", 10.00, 9.99, 10.01, 100, start, 1, "", False, False, "pre"),
        backtest_mod.LiveTimesale("timesale", "RMSG", "Q", 9.90, 9.89, 9.91, 200, window_ts, 2, "", False, False, "pre"),
        backtest_mod.LiveTimesale("timesale", "RMSG", "Q", 9.80, 9.79, 9.81, 300, start + timedelta(minutes=2), 3, "", False, False, "pre"),
    ]
    entry = Entry(
        symbol="RMSG",
        avg_volume=100_000,
        price=10.00,
        daily_high=0.0,
        volume=0,
        rvol_1day=0.0,
        change_pct=0.0,
        setup_pattern_score=100.0,
        evaluation=None,
    )
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)
    runner._tradier = SimpleNamespace()

    monkeypatch.setattr(backtest_mod, "DynamicScorer", FakeScorer)
    monkeypatch.setattr(backtest_mod.BacktestRunner, "_refresh_ta_from_ticks", staticmethod(lambda _scorer, _ticks: None))
    monkeypatch.setattr(runner, "_iter_ticks", lambda _symbol, _start_utc, _end_utc, _windows=None: ((0, tick) for tick in ticks))

    result = runner._replay_symbol(
        backtest_mod.BacktestEvent("RMSG", start, start, None, entry, SimpleNamespace()),
        end,
    )

    scorer = captured["scorer"]
    assert result.ticks_processed == 3
    assert scorer.quote_count == 1
    assert scorer.timesale_count == 1
    assert scorer._entry.volume == 600
    assert scorer._entry.daily_high == 10.00


def test_backtest_replay_falls_back_to_near_hod_when_pullback_windows_empty(monkeypatch, tmp_path):
    class Entry(SimpleNamespace):
        @property
        def eval_confidence(self):
            return 80.0

        @property
        def live_confidence(self):
            return 80.0

        @property
        def has_live_data(self):
            return False

        def get_criteria_group(self, _name):
            return None

    class FakeScorer:
        def __init__(self, entry, _rmap, **_kwargs):
            self._entry = entry
            self._sym = entry.symbol
            self._avg_vol = entry.avg_volume
            self._last_price = entry.price
            self._last_atr = 0.10
            self._prev_close = 9.50
            self._bid = 0.0
            self._ask = 0.0
            self._setup_logger = None
            self._ofi_tracker = SimpleNamespace(buying_pressure=999, selling_pressure=0, activity_count=999)
            self.quote_count = 0
            self.timesale_count = 0
            captured["scorer"] = self

        def _on_quote(self, _quote):
            self.quote_count += 1

        def _on_timesale(self, tick):
            self.timesale_count += 1
            self._last_price = tick.last

        def stop(self):
            pass

    captured: dict[str, object] = {}
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    window = backtest_mod.CachedNearHodWindow(
        start + timedelta(minutes=1), start + timedelta(minutes=2), 9.90, 10.00, 0.10, 0
    )
    tick = backtest_mod.LiveTimesale(
        "timesale", "RMSG", "Q", 9.95, 9.94, 9.96, 100,
        start + timedelta(minutes=1, seconds=30), 1, "", False, False, "pre",
    )
    entry = Entry(
        symbol="RMSG",
        avg_volume=100_000,
        price=10.00,
        daily_high=0.0,
        volume=0,
        rvol_1day=0.0,
        change_pct=0.0,
        setup_pattern_score=100.0,
        evaluation=None,
    )
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)
    runner._tradier = SimpleNamespace()

    monkeypatch.setattr(backtest_mod, "DynamicScorer", FakeScorer)
    monkeypatch.setattr(runner, "_load_cached_seed_market_data", lambda *_args: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(runner, "_seed_scorer_from_cached_market_data", lambda *_args: None)
    monkeypatch.setattr(runner, "_load_cached_near_hod_windows", lambda *_args: [window])
    monkeypatch.setattr(runner, "_derive_pullback_replay_windows_from_candle_cache", lambda *_args: ([], True))
    monkeypatch.setattr(runner, "_seed_replay_window_candle_context", lambda *_args: None)
    monkeypatch.setattr(backtest_mod.BacktestRunner, "_refresh_ta_from_ticks", staticmethod(lambda _scorer, _ticks: None))

    def fake_iter_ticks(_symbol, _start_utc, _end_utc, windows=None):
        captured["windows"] = windows
        if windows:
            yield 0, tick

    monkeypatch.setattr(runner, "_iter_ticks", fake_iter_ticks)

    result = runner._replay_symbol(
        backtest_mod.BacktestEvent("RMSG", start, start, None, entry, SimpleNamespace()),
        end,
    )

    assert captured["windows"] == [window]
    assert result.ticks_processed == 1
    assert captured["scorer"].quote_count == 1
    assert captured["scorer"].timesale_count == 1


def test_backtest_cached_window_replay_stays_windowed_when_full_tape_has_holes(monkeypatch, tmp_path):
    class Entry(SimpleNamespace):
        @property
        def eval_confidence(self):
            return 80.0

        @property
        def live_confidence(self):
            return 80.0

        @property
        def has_live_data(self):
            return False

        def get_criteria_group(self, _name):
            return None

    class FakeScorer:
        def __init__(self, entry, _rmap, **_kwargs):
            self._entry = entry
            self._sym = entry.symbol
            self._avg_vol = entry.avg_volume
            self._last_price = entry.price
            self._last_atr = 0.10
            self._prev_close = 9.50
            self._bid = 0.0
            self._ask = 0.0
            self._setup_logger = None
            self._ofi_tracker = SimpleNamespace(buying_pressure=999, selling_pressure=0, activity_count=999)

        def stop(self):
            pass

    captured: dict[str, object] = {}
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=10)
    window = backtest_mod.CachedNearHodWindow(start + timedelta(minutes=1), start + timedelta(minutes=2), 9.90, 10.00, 0.10, 0)
    entry = Entry(
        symbol="RMSG",
        avg_volume=100_000,
        price=10.00,
        daily_high=0.0,
        volume=0,
        rvol_1day=0.0,
        change_pct=0.0,
        setup_pattern_score=100.0,
        evaluation=None,
    )
    runner = BacktestRunner(cache_path=tmp_path / "ticker_cache.pkl", eval_stats_path=tmp_path / "missing.log", log_dir=tmp_path)
    runner._tradier = SimpleNamespace()

    monkeypatch.setattr(backtest_mod, "DynamicScorer", FakeScorer)
    monkeypatch.setattr(runner, "_load_cached_seed_market_data", lambda *_args: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(runner, "_seed_scorer_from_cached_market_data", lambda *_args: None)
    monkeypatch.setattr(runner, "_load_cached_near_hod_windows", lambda *_args: [window])
    monkeypatch.setattr(runner, "_derive_pullback_replay_windows_from_candle_cache", lambda *_args: ([window], True))
    monkeypatch.setattr(runner, "_cached_timesale_frame_holes", lambda *_args: [(start, end)])
    monkeypatch.setattr(backtest_mod.BacktestRunner, "_refresh_ta_from_ticks", staticmethod(lambda _scorer, _ticks: None))

    def fake_iter_ticks(_symbol, _start_utc, _end_utc, windows=None):
        captured["windows"] = windows
        return iter(())

    monkeypatch.setattr(runner, "_iter_ticks", fake_iter_ticks)

    runner._replay_symbol(
        backtest_mod.BacktestEvent("RMSG", start, start, None, entry, SimpleNamespace()),
        end,
    )

    assert captured["windows"] == [window]


def test_backtest_prefers_cached_live_timesales(monkeypatch, tmp_path):
    day = date(2026, 6, 5)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())
    path = tmp_path / "live_timesales_20260605.csv"
    path.write_text(
        "ts,symbol,last,bid,ask,size,seq\n"
        "2026-06-05T12:01:00+00:00,RMSG,3.10,3.09,3.10,250,42\n"
        "2026-06-05T12:01:01+00:00,OTHER,1.00,0.99,1.00,100,43\n"
    )

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 5, 12, 5, tzinfo=timezone.utc)

    rows = list(runner._iter_ticks("RMSG", start, end))

    assert len(rows) == 1
    call_count, event = rows[0]
    assert call_count == 0
    assert event.last == 3.10
    assert event.bid == 3.09
    assert event.ask == 3.10
    assert event.seq == 42


def test_backtest_prefers_canonical_cached_timesales_before_tradier(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace(
        get_timesales_tick=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tradier called")),
    ))
    path = timesales_cache_path(tmp_path, "RMSG", "20260605")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "symbol,ts,price,size,bid,ask,exchange,seq,flag,session,source\n"
        "RMSG,2026-06-05T12:01:00+00:00,3.10,250,3.09,3.10,Q,42,,pre,stream\n"
    )

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 5, 12, 5, tzinfo=timezone.utc)

    rows = list(runner._iter_ticks("RMSG", start, end))

    assert len(rows) == 1
    call_count, event = rows[0]
    assert call_count == 0
    assert event.last == 3.10
    assert event.bid == 3.09
    assert event.ask == 3.10
    assert event.seq == 42


def test_backtest_filters_cached_timesales_across_windows_with_one_cache_load(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace(
        get_timesales_tick=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tradier called")),
    ))
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=4)
    events = [
        backtest_mod.LiveTimesale("timesale", "RMSG", "Q", 3.00, 2.99, 3.01, 100, start + timedelta(seconds=10), 1, "", False, False, "pre"),
        backtest_mod.LiveTimesale("timesale", "RMSG", "Q", 3.10, 3.09, 3.11, 100, start + timedelta(seconds=31), 2, "", False, False, "pre"),
        backtest_mod.LiveTimesale("timesale", "RMSG", "Q", 3.20, 3.19, 3.21, 100, start + timedelta(minutes=2), 3, "", False, False, "pre"),
    ]
    calls = []

    def fake_load_cached_timesales(symbol, start_utc, end_utc):
        calls.append((symbol, start_utc, end_utc))
        return events

    monkeypatch.setattr(runner, "_load_cached_timesales", fake_load_cached_timesales)
    windows = [
        backtest_mod.CachedNearHodWindow(start, start + timedelta(seconds=30), 3.00, 3.10, 0.05),
        backtest_mod.CachedNearHodWindow(start + timedelta(seconds=31), start + timedelta(seconds=45), 3.10, 3.10, 0.05),
    ]

    rows = list(runner._iter_ticks("RMSG", start, end, windows))

    assert len(calls) == 1
    assert calls[0] == ("RMSG", start, end)
    assert [event.last for _, event in rows] == [3.00, 3.10]


def test_backtest_fills_cached_timesale_window_holes_from_tradier(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_mod, "IBKR_ENABLED", False)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=4)
    cached_event = backtest_mod.LiveTimesale(
        "timesale", "RMSG", "Q", 3.00, 2.99, 3.01, 100,
        start + timedelta(seconds=10), 1, "", False, False, "pre",
    )
    windows = [
        backtest_mod.CachedNearHodWindow(start, start + timedelta(seconds=30), 3.00, 3.10, 0.05),
        backtest_mod.CachedNearHodWindow(start + timedelta(minutes=2), start + timedelta(minutes=3), 3.10, 3.10, 0.05),
    ]
    fetch_calls = []

    def fake_load_cached_timesales(symbol, start_utc, end_utc, windows=None):
        return [cached_event]

    def fake_fetch_ticks(symbol, start_utc, end_utc):
        fetch_calls.append((symbol, start_utc, end_utc))
        return [
            {"ts": start + timedelta(minutes=2, seconds=15), "price": 3.20, "bid": 3.19, "ask": 3.21, "size": 100},
        ]

    monkeypatch.setattr(runner, "_load_cached_timesales", fake_load_cached_timesales)
    monkeypatch.setattr(runner, "_fetch_ticks", fake_fetch_ticks)

    rows = list(runner._iter_ticks("RMSG", start, end, windows))

    assert fetch_calls == [("RMSG", start + timedelta(minutes=2), start + timedelta(minutes=3))]
    assert [(call_count, event.last) for call_count, event in rows] == [(0, 3.00), (1, 3.20)]


def test_backtest_fetch_ticks_recursively_splits_empty_overflow_chunks(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_mod, "IBKR_ENABLED", False)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)
    target_start = start + timedelta(minutes=7, seconds=30)
    target_end = start + timedelta(minutes=15)
    calls: list[tuple[datetime, datetime]] = []

    def fake_get_timesales_tick(_symbol, start_utc, end_utc, session_filter="all"):
        calls.append((start_utc, end_utc))
        duration = end_utc - start_utc
        if start_utc == target_start and end_utc == target_end:
            return [{"ts": start_utc + timedelta(minutes=1), "price": 3.20, "size": 100}]
        if duration > timedelta(minutes=2):
            return []
        return []

    runner._tradier = SimpleNamespace(get_timesales_tick=fake_get_timesales_tick)

    rows = runner._fetch_ticks("RMSG", start, end)

    assert rows == [{"ts": target_start + timedelta(minutes=1), "price": 3.20, "size": 100}]
    assert (target_start, target_end) in calls


def test_replay_change_pct_uses_first_and_last_replay_prices():
    assert backtest_mod._replay_change_pct(2.00, 3.00) == 50.0
    assert backtest_mod._replay_change_pct(0.0, 3.00) == 0.0


def test_backtest_uses_cached_chart_ticks_before_tradier(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_mod, "IBKR_ENABLED", False)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace(
        get_timesales_tick=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("tradier called")),
    ))
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    write_candle_cache(
        "RMSG",
        "10s",
        pd.DataFrame([
            {"ts": start + timedelta(seconds=10), "open": 3.00, "high": 3.08, "low": 2.98, "close": 3.05, "volume": 500},
            {"ts": start + timedelta(seconds=20), "open": 3.05, "high": 3.15, "low": 3.04, "close": 3.12, "volume": 700},
        ]),
        log_dir=tmp_path,
    )

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    rows = list(runner._iter_ticks("RMSG", start, start + timedelta(minutes=1)))

    assert len(rows) == 2
    assert all(call_count == 0 for call_count, _ in rows)
    assert rows[0][1].exchange == "chart_cache"
    assert rows[0][1].flag == "chart_cache:10s"
    assert rows[0][1].last == 3.05
    assert rows[1][1].last == 3.12


def test_backtest_uses_ibkr_ticks_instead_of_tradier_or_candle_events(monkeypatch, tmp_path):
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    write_candle_cache(
        "RMSG",
        "10s",
        pd.DataFrame([
            {"ts": start + timedelta(seconds=10), "open": 2.90, "high": 2.95, "low": 2.85, "close": 2.92, "volume": 100},
        ]),
        log_dir=tmp_path,
    )

    class FakeIBKR:
        def get_timesales_tick(self, symbol, start_utc, end_utc, session_filter="all"):
            assert symbol == "RMSG"
            assert start_utc == start
            assert end_utc == start + timedelta(minutes=1)
            assert session_filter == "all"
            return [
                {"ts": start + timedelta(seconds=1), "price": 3.05, "size": 500, "exchange": "IBKR", "flag": "T"},
                {"ts": start + timedelta(seconds=11), "price": 3.12, "size": 700, "exchange": "IBKR", "flag": "T"},
            ]

        def get_candles_10sec_window(self, symbol, start_utc, end_utc):
            raise AssertionError("candle fallback should not be called for external replay")

    monkeypatch.setattr(backtest_mod, "IBKR_ENABLED", True)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: FakeIBKR())
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )

    rows = list(runner._iter_ticks("RMSG", start, start + timedelta(minutes=1)))

    assert [(call_count, event.last, event.exchange, event.flag) for call_count, event in rows] == [
        (1, 3.05, "IBKR", "T"),
        (1, 3.12, "IBKR", "T"),
    ]


def test_backtest_does_not_replay_unclosed_cached_chart_candle(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace(
        get_timesales_tick=lambda *args, **kwargs: [],
    ))
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    write_candle_cache(
        "RMSG",
        "1m",
        pd.DataFrame([
            {"ts": start, "open": 3.00, "high": 9.99, "low": 2.98, "close": 9.50, "volume": 500},
        ]),
        log_dir=tmp_path,
    )
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )

    rows = list(runner._iter_ticks("RMSG", start, start + timedelta(seconds=30)))

    assert rows == []


def test_backtest_merges_ticks_and_cached_tape_snapshots_in_replay_order(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_mod, "IBKR_ENABLED", False)
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())
    tape_path = tmp_path / "tape_20260605.csv"
    tape_path.write_text(
        "ts,symbol,price,avg_print_size,entry_speed,entry_buy_ratio,pressure_acceleration,"
        "pressure_acceleration_ready,entry_tape_gate,buying_pressure,selling_pressure,buy_vol,sell_vol\n"
        "2026-06-05T12:01:01+00:00,RMSG,3.11,100,10.0,0.60,0.10,True,True,60,40,6000,4000\n"
        "2026-06-05T12:06:00+00:00,RMSG,3.30,100,10.0,0.60,0.10,True,True,60,40,6000,4000\n"
    )

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 5, 12, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(
        runner,
        "_fetch_ticks",
        lambda symbol, start_utc, end_utc: [
            {"ts": start - timedelta(seconds=1), "price": 2.99, "bid": 2.98, "ask": 3.00, "size": 100},
            {"ts": start + timedelta(minutes=1), "price": 3.10, "bid": 3.09, "ask": 3.10, "size": 100},
            {"ts": start + timedelta(minutes=2), "price": 3.12, "bid": 3.11, "ask": 3.12, "size": 100},
            {"ts": end + timedelta(seconds=1), "price": 3.40, "bid": 3.39, "ask": 3.40, "size": 100},
        ],
    )

    rows = list(runner._iter_ticks("RMSG", start, end))

    assert [(call_count, event.ts, event.last) for call_count, event in rows] == [
        (1, start + timedelta(minutes=1), 3.10),
        (0, start + timedelta(minutes=1, seconds=1), 3.11),
        (1, start + timedelta(minutes=2), 3.12),
    ]
    assert getattr(rows[1][1], "_cached_tape_snapshot").source_path == tape_path


def test_backtest_replays_cached_tape_gate_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(backtest_mod, "get_historical_client", lambda: SimpleNamespace())
    path = tmp_path / "tape_20260605.csv"
    path.write_text(
        "ts,symbol,price,avg_print_size,entry_speed,entry_buy_ratio,pressure_acceleration,"
        "pressure_acceleration_ready,entry_tape_gate,buying_pressure,selling_pressure,buy_vol,sell_vol\n"
        "2026-06-05T12:01:00+00:00,RMSG,3.10,100,10.0,0.60,0.10,True,True,60,40,6000,4000\n"
        "2026-06-05T12:01:01+00:00,RMSG,3.11,100,1.0,0.50,0.00,False,False,5,5,500,500\n"
    )

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 5, 12, 5, tzinfo=timezone.utc)

    snapshots = runner._load_cached_tape_snapshots("RMSG", start, end)

    assert len(snapshots) == 1
    event = runner._synthetic_tick_from_snapshot(snapshots[0], seq=7)
    assert getattr(event, "_cached_tape_snapshot") is snapshots[0]
    assert event.ask == event.last

    scorer = SimpleNamespace(_ofi_tracker=OrderFlowTracker())
    runner._apply_cached_tape_snapshot(scorer, snapshots[0])

    ofi = scorer._ofi_tracker
    # Pressure (signal) window is rebuilt from the logged rate (entry_speed=10
    # over a 10s window = 100 prints, split 0.60 buy), so trade_speed stays
    # faithful to the logged rate regardless of the configured signal window.
    assert ofi.buying_pressure == 60
    assert ofi.selling_pressure == 40
    assert ofi.trade_speed == 10.0
    assert ofi.buy_ratio == 0.60
    assert ofi.pressure_acceleration_ready is True
    # Activity floor reads the wider window and preserves the logged print total.
    assert ofi.activity_count == 300


def test_backtest_summary_sorts_by_premarket_change_and_shows_new_columns(capsys):
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    results = [
        backtest_mod.SymbolBacktestResult(
            symbol=f"S{i:02d}",
            stream_start_utc=start,
            stream_end_utc=start + timedelta(minutes=1),
            premarket_change_pct=float(i),
            ticks_processed=10 + i,
            tradier_calls=0,
            approved_setups=i % 3,
            failure_detail=f"fail-{i}",
            faded_from_hod=False,
            final_eval=70.0,
            final_live=71.0,
        )
        for i in range(11)
    ]

    backtest_mod._print_backtest_summary(results)

    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line[:3] in {f"S{i:02d}" for i in range(11)}]
    assert "START" in out
    assert "END" in out
    assert "CHG%" in out
    assert "SETUPS" in out
    assert "Setup Diagnostics" in out
    assert rows[0].startswith("S10")
    assert " +10.0%" in rows[0]
    assert "fail-10" in out
    assert rows[-1].startswith("S00")
    assert "fail-0" not in out


def _peak_event(symbol: str, start: datetime, price: float, change_pct: float) -> "backtest_mod.BacktestEvent":
    return backtest_mod.BacktestEvent(
        symbol=symbol,
        stream_start_utc=start,
        first_seen_utc=start,
        updated_utc=None,
        entry=SimpleNamespace(price=price, change_pct=change_pct),
        rmap=None,
    )


def test_peak_change_pct_uses_cached_1m_candles(tmp_path):
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
        no_tradier=True,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    highs = [3.0, 4.5, 9.0, 5.0, 4.0, 3.5]
    write_candle_cache(
        "PEAK",
        "1m",
        pd.DataFrame([
            {"ts": start + timedelta(minutes=i), "open": 3.0, "high": h, "low": 2.9, "close": 3.0, "volume": 1000}
            for i, h in enumerate(highs)
        ]),
        log_dir=tmp_path,
    )

    # price=3.0 at +50% → prev_close=2.0; peak high 9.0 → (9−2)/2 = +350%
    event = _peak_event("PEAK", start, price=3.0, change_pct=50.0)
    assert abs(runner._peak_change_pct(event, end) - 350.0) < 1e-6


def test_peak_change_pct_bounded_by_cached_near_hod_windows(tmp_path):
    """Algo stopped watching at 12:03 (last cached near-HOD window end) —
    the 20.0 spike at 12:08 must not count toward the sort key."""
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
        no_tradier=True,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=10)
    watch_end = start + timedelta(minutes=3)
    highs = {0: 3.0, 1: 6.0, 2: 5.0, 3: 4.0, 8: 20.0, 10: 4.0}
    write_candle_cache(
        "PEAK",
        "1m",
        pd.DataFrame([
            {"ts": start + timedelta(minutes=i), "open": 3.0, "high": h, "low": 2.9, "close": 3.0, "volume": 1000}
            for i, h in highs.items()
        ]),
        log_dir=tmp_path,
    )
    write_near_hod_window_cache(
        "PEAK",
        [{
            "symbol": "PEAK",
            "replay_start_ts": start - timedelta(minutes=5),
            "replay_end_ts": end + timedelta(minutes=5),
            "start_ts": start,
            "end_ts": watch_end,
            "start_price": 3.0,
            "start_hod": 3.0,
            "start_atr_14": 0.1,
            "start_volume": 1000,
            "threshold_atrs": 2.0,
            "source": "replay",
        }],
        tmp_path,
    )

    # prev_close=2.0; max high while watched = 6.0 → +200%, NOT +900%
    event = _peak_event("PEAK", start, price=3.0, change_pct=50.0)
    assert abs(runner._peak_change_pct(event, end) - 200.0) < 1e-6


def test_peak_change_pct_partial_cache_never_fetches(tmp_path, monkeypatch):
    """Partial candle coverage keys on whatever max it has — no provider
    fetch from the sort path; the replay's own backfills fill it in later."""
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
        no_tradier=False,  # fetch would be allowed — prove it isn't attempted
    )

    def _boom(*args, **kwargs):
        raise AssertionError("sort path must not fetch candles")

    monkeypatch.setattr(runner, "_fetch_candle_frame_from_provider", _boom)

    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)
    # Only 2 of 30 minutes cached
    write_candle_cache(
        "PART",
        "1m",
        pd.DataFrame([
            {"ts": start, "open": 3.0, "high": 5.0, "low": 2.9, "close": 3.0, "volume": 1000},
            {"ts": start + timedelta(minutes=1), "open": 3.0, "high": 4.0, "low": 2.9, "close": 3.0, "volume": 500},
        ]),
        log_dir=tmp_path,
    )

    # prev_close=2.0; partial max high 5.0 → +150% (beats eval-time +50%)
    event = _peak_event("PART", start, price=3.0, change_pct=50.0)
    assert abs(runner._peak_change_pct(event, end) - 150.0) < 1e-6

    # No candles at all → eval-time change_pct is the key
    event_none = _peak_event("NONE", start, price=3.0, change_pct=50.0)
    assert runner._peak_change_pct(event_none, end) == 50.0


def test_peak_change_pct_prefers_live_cached_peak_and_stored_prev_close(tmp_path):
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
        no_tradier=True,
    )
    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)

    # Live-tracked snapshot: scorer persisted peak_change_pct=300 via
    # _raise_hod; no candles cached at all → stored peak wins outright.
    event = backtest_mod.BacktestEvent(
        symbol="LIVE",
        stream_start_utc=start,
        first_seen_utc=start,
        updated_utc=None,
        entry=SimpleNamespace(price=3.0, change_pct=50.0, prev_close=2.0,
                              peak_change_pct=300.0, daily_high=0.0),
        rmap=None,
    )
    assert runner._peak_change_pct(event, end) == 300.0

    # Snapshot HOD candidate: daily_high live-updated to 9.0 with stored
    # prev_close 2.0 → (9−2)/2 = +350% even without candles or stored peak.
    event_hod = backtest_mod.BacktestEvent(
        symbol="HODX",
        stream_start_utc=start,
        first_seen_utc=start,
        updated_utc=None,
        entry=SimpleNamespace(price=3.0, change_pct=50.0, prev_close=2.0,
                              peak_change_pct=0.0, daily_high=9.0),
        rmap=None,
    )
    assert abs(runner._peak_change_pct(event_hod, end) - 350.0) < 1e-6


def test_run_range_replays_events_by_peak_desc_across_days(tmp_path, monkeypatch):
    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
        no_tradier=True,
    )
    day_a, day_b = date(2020, 1, 6), date(2020, 1, 7)
    start_a = datetime(2020, 1, 6, 12, 0, tzinfo=timezone.utc)
    start_b = datetime(2020, 1, 7, 12, 0, tzinfo=timezone.utc)
    events_by_day = {
        day_a: [_peak_event("AAA", start_a, 3.0, 50.0), _peak_event("BBB", start_a, 3.0, 50.0)],
        day_b: [_peak_event("CCC", start_b, 3.0, 50.0)],
    }
    peaks = {"AAA": 50.0, "BBB": 250.0, "CCC": 120.0}
    monkeypatch.setattr(runner, "_build_day_events", lambda day: list(events_by_day.get(day, [])))
    monkeypatch.setattr(runner, "_peak_change_pct", lambda event, end_utc: peaks[event.symbol])

    replayed: list[str] = []

    def _fake_replay(event, end_utc):
        replayed.append(event.symbol)
        return backtest_mod.SymbolBacktestResult(
            symbol=event.symbol,
            stream_start_utc=event.stream_start_utc,
            stream_end_utc=end_utc,
            premarket_change_pct=0.0,
            ticks_processed=0,
            tradier_calls=0,
            approved_setups=0,
            failure_detail="",
            faded_from_hod=False,
            final_eval=0.0,
            final_live=0.0,
        )

    monkeypatch.setattr(runner, "_replay_symbol", _fake_replay)

    results = runner.run_range(day_a, day_b)

    assert replayed == ["BBB", "CCC", "AAA"]  # peak desc across BOTH days
    assert [r.symbol for r in results] == ["BBB", "CCC", "AAA"]


def test_no_api_implies_no_tradier_and_skips_client_construction(tmp_path, monkeypatch):
    """--no-api is the superset: it blocks provider fetches AND never builds a
    provider client (which alone can fail on a box with no credentials)."""
    def _boom():
        raise AssertionError("--no-api must not construct a provider client")

    monkeypatch.setattr(backtest_mod, "get_historical_client", _boom)

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
        no_api=True,
    )

    assert runner._no_api is True
    assert runner._no_tradier is True  # fetch gates stay closed
    assert runner._tradier is None

    start = datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=30)
    assert runner._fetch_ticks("NOAPI", start, end) == []
    assert runner._fetch_tick_chunk("NOAPI", start, end) == []
    assert runner._fetch_candle_frame_from_provider("NOAPI", "1m", 60, start, end).empty

    with pytest.raises(RuntimeError, match="no_api"):
        runner.add_tickers_to_day(["NOAPI"], date(2026, 6, 5))


def test_no_api_stubs_shared_stream_during_replay(tmp_path, monkeypatch):
    """The scorer dials the Gateway via get_shared_stream() on the slot-priority
    push and in stop()'s unsubscribe — under --no-api both must be absorbed."""
    import strategy.confidence_scorer as scorer_module

    real_getter = scorer_module.get_shared_stream

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
        no_api=True,
    )
    day = date(2020, 1, 6)
    start = datetime(2020, 1, 6, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(runner, "_build_day_events", lambda d: [_peak_event("STUB", start, 3.0, 50.0)])
    monkeypatch.setattr(runner, "_peak_change_pct", lambda event, end_utc: 50.0)

    def _fake_replay(event, end_utc):
        assert scorer_module.get_shared_stream is not real_getter  # displaced
        stream = scorer_module.get_shared_stream()
        # Every scorer→stream call the replay makes is absorbed, not dialed.
        assert stream.subscribe("STUB", on_timesale=None) is None
        assert stream.unsubscribe("STUB", sub_id="scorer:STUB") is None
        assert getattr(stream, "set_priority", None)("STUB", 3) is None
        return backtest_mod.SymbolBacktestResult(
            symbol=event.symbol,
            stream_start_utc=event.stream_start_utc,
            stream_end_utc=end_utc,
            premarket_change_pct=0.0,
            ticks_processed=0,
            tradier_calls=0,
            approved_setups=0,
            failure_detail="",
            faded_from_hod=False,
            final_eval=0.0,
            final_live=0.0,
        )

    monkeypatch.setattr(runner, "_replay_symbol", _fake_replay)

    assert [r.symbol for r in runner.run_day(day)] == ["STUB"]
    # Restored afterwards so a later live run isn't left with a dead stream.
    assert scorer_module.get_shared_stream is real_getter


def test_replay_leaves_shared_stream_alone_without_no_api(tmp_path, monkeypatch):
    """--no-tradier alone keeps the real stream getter in place (replay_session
    installs its own stub), so the guard must not fire unasked."""
    import strategy.confidence_scorer as scorer_module

    real_getter = scorer_module.get_shared_stream

    runner = BacktestRunner(
        cache_path=tmp_path / "ticker_cache.pkl",
        eval_stats_path=tmp_path / "missing.log",
        log_dir=tmp_path,
        no_tradier=True,
    )
    day = date(2020, 1, 6)
    start = datetime(2020, 1, 6, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(runner, "_build_day_events", lambda d: [_peak_event("REAL", start, 3.0, 50.0)])
    monkeypatch.setattr(runner, "_peak_change_pct", lambda event, end_utc: 50.0)

    seen: list[object] = []

    def _fake_replay(event, end_utc):
        seen.append(scorer_module.get_shared_stream)
        return backtest_mod.SymbolBacktestResult(
            symbol=event.symbol,
            stream_start_utc=event.stream_start_utc,
            stream_end_utc=end_utc,
            premarket_change_pct=0.0,
            ticks_processed=0,
            tradier_calls=0,
            approved_setups=0,
            failure_detail="",
            faded_from_hod=False,
            final_eval=0.0,
            final_live=0.0,
        )

    monkeypatch.setattr(runner, "_replay_symbol", _fake_replay)
    runner.run_day(day)

    assert seen == [real_getter]
    assert runner._no_api is False


def test_backtest_failure_detail_stops_at_price_stage():
    entry = SimpleNamespace(
        price=backtest_mod.PRICE_MAX + 10.0,
        eval_confidence=100.0,
        live_confidence=100.0,
        has_live_data=True,
        evaluation=SimpleNamespace(groups=[]),
        setup_pattern_score=100.0,
        setup_price_level=None,
    )
    exec_group = CriteriaGroup("EXECUTION", [Criterion("tape_speed", passing=False, value=3.0)])
    entry.get_criteria_group = lambda name: exec_group if name == "EXECUTION" else None

    detail = backtest_mod._backtest_failure_detail(entry)

    assert detail == f"price={backtest_mod.PRICE_MAX + 10.0:.2f} > {backtest_mod.PRICE_MAX:.2f}"
    assert "tape_speed" not in detail


def test_backtest_failure_detail_reports_execution_stage_literal():
    exec_group = CriteriaGroup("EXECUTION", [Criterion("tape_speed", passing=False, value=3.0)])

    entry = SimpleNamespace(
        price=3.50,
        eval_confidence=90.0,
        live_confidence=90.0,
        has_live_data=True,
        evaluation=SimpleNamespace(groups=[SimpleNamespace(name="STOCK_EVAL", category="QUALITY", score=1.0)]),
        setup_pattern_score=100.0,
        setup_price_level=3.50,
    )
    entry.get_criteria_group = lambda name: exec_group if name == "EXECUTION" else None

    detail = backtest_mod._backtest_failure_detail(entry)

    assert detail == f"tape_speed=3.0/s < {backtest_mod.SIM_PRESSURE_SPEED_MIN:.1f}/s"


def test_setup_failure_detail_lists_all_setup_snapshots_oldest_first():
    first = SimpleNamespace(
        ts=datetime(2026, 6, 4, 14, 30, tzinfo=timezone.utc),
        setup_score=45.0,
        best_score=3.0,
        setup_failure_detail="setup_score=45 < 50",
        block_reasons="",
    )
    second = SimpleNamespace(
        ts=datetime(2026, 6, 4, 14, 41, 30, tzinfo=timezone.utc),
        setup_score=88.7,
        best_score=4.0,
        setup_failure_detail="tape_speed=3.0/s < 10.0/s; res_headroom=0.04 < 0.20 (1.0 ATR)",
        block_reasons="",
    )
    third = SimpleNamespace(
        ts=datetime(2026, 6, 4, 14, 55, tzinfo=timezone.utc),
        setup_score=70.0,
        best_score=5.0,
        setup_failure_detail="",
        block_reasons="all setup-time hard gates passed",
    )

    detail = backtest_mod._setup_failure_detail([third, second, first])

    assert detail.index("setup@") == 0
    assert detail.index("setup_score=45 < 50") < detail.index("tape_speed=3.0/s < 10.0/s")
    assert detail.index("tape_speed=3.0/s < 10.0/s") < detail.index("all setup-time hard gates passed")
    assert "tape_speed=3.0/s < 10.0/s" in detail
    assert "res_headroom=0.04 < 0.20" in detail