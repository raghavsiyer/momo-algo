"""Tests for market_data/tastytrade_stream.py and the merged tape's wiring
into the IBKR stream, the order-flow tracker and the tape cache. No network."""

import json
import math
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import market_data.tastytrade_stream as ts_mod
from market_data.tape_merge import MergedTape, TastytradePrint, TastytradeTrade
from market_data.tastytrade_stream import (
    FEED_FIELDS,
    DxLinkTapeLink,
    QuoteToken,
    TastytradeApi,
    TastytradeApiError,
    TastytradeConfigError,
    _Delayed,
    _Reconnect,
    parse_feed_data,
)


# ---------------------------------------------------------------------------
# COMPACT parsing
# ---------------------------------------------------------------------------

def _collect(data, layouts=FEED_FIELDS):
    prints, trades = [], []
    parse_feed_data(data, layouts, prints.append, trades.append, 5.0, 1700000000.0)
    return prints, trades


def test_parses_flat_and_nested_rows_with_both_event_types():
    sale = ["TimeAndSale", "ABC", 0, 77, 1700000000123, "Q", 2.5, 150.0, 2.49, 2.51,
            "BUY", True, False, "NEW", "@ To"]
    trade = ["Trade", "ABC", 1700000000123, 2.5, 150.0, 1234567.0, 3000000.0, True]
    prints, trades = _collect(["TimeAndSale", sale + sale, "Trade", [trade]])
    assert len(prints) == 2 and len(trades) == 1
    p = prints[0]
    assert (p.symbol, p.time_ms, p.price, p.size, p.bid, p.ask) == ("ABC", 1700000000123, 2.5, 150.0, 2.49, 2.51)
    assert (p.exchange, p.aggressor, p.sale, p.index, p.valid_tick, p.extended_hours) == ("Q", "BUY", "NEW", 77, False, True)
    assert p.normal_odd_lot and not p.reopening and p.arrival == 5.0
    assert trades[0].day_volume == 1234567.0


def test_nan_strings_and_missing_values_read_as_nan_and_defaults():
    sale = ["TimeAndSale", "ABC", 0, 0, 1700000000000, None, 2.5, 10.0, "NaN", "NaN",
            "UNDEFINED", None, None, "CANCEL", None]
    prints, _ = _collect(["TimeAndSale", sale])
    p = prints[0]
    assert math.isnan(p.bid) and math.isnan(p.ask)
    assert p.valid_tick, "validTick reads true unless it's false"
    assert p.sale == "CANCEL" and p.exchange == "" and p.conditions == ""


def test_server_column_order_from_feed_config_is_honoured():
    layouts = {"TimeAndSale": ["eventType", "eventSymbol", "price", "size", "time"]}
    prints, _ = _collect(["TimeAndSale", ["TimeAndSale", "ABC", 3.0, 100.0, 1700000000000]], layouts)
    assert (prints[0].price, prints[0].size, prints[0].time_ms) == (3.0, 100.0, 1700000000000)


# ---------------------------------------------------------------------------
# REST: OAuth refresh, rotation, quote token
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class _Session:
    def __init__(self, posts, gets):
        self.headers = {}
        self.posts = list(posts)
        self.gets = list(gets)
        self.post_bodies = []
        self.get_headers = []

    def post(self, url, json=None, timeout=None):
        self.post_bodies.append(json)
        return self.posts.pop(0)

    def get(self, url, headers=None, timeout=None):
        self.get_headers.append(headers)
        return self.gets.pop(0)


@pytest.fixture(autouse=True)
def _reset_refusal():
    ts_mod._refused_token = None
    ts_mod._refused_until = 0.0
    yield
    ts_mod._refused_token = None
    ts_mod._refused_until = 0.0


def _creds(tmp_path, **extra):
    path = tmp_path / "tastytrade.json"
    path.write_text(json.dumps({"refresh_token": "r1", "client_id": "cid", "client_secret": "sec", **extra}))
    return path


def test_missing_or_incomplete_credentials_are_refused(tmp_path):
    with pytest.raises(TastytradeConfigError):
        TastytradeApi(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"client_id": "x"}))
    with pytest.raises(TastytradeConfigError):
        TastytradeApi(bad)


def test_refresh_saves_a_rotated_token_and_mints_a_quote_token(tmp_path):
    path = _creds(tmp_path, note="kept")
    session = _Session(
        posts=[_Response(200, {"access_token": "a1", "expires_in": 900, "refresh_token": "r2"})],
        gets=[_Response(200, {"data": {"token": "q1", "dxlink-url": "wss://x/realtime", "level": "api"}})],
    )
    api = TastytradeApi(path, session=session)
    token = api.quote_token()
    assert (token.token, token.url, token.is_real_time) == ("q1", "wss://x/realtime", True)
    assert session.post_bodies[0] == {"grant_type": "refresh_token", "refresh_token": "r1",
                                      "client_id": "cid", "client_secret": "sec"}
    assert session.get_headers[0] == {"Authorization": "Bearer a1"}
    saved = json.loads(path.read_text())
    assert saved["refresh_token"] == "r2" and saved["note"] == "kept"


def test_a_token_rotated_by_another_process_is_picked_up_before_refreshing(tmp_path):
    path = _creds(tmp_path)
    session = _Session(posts=[_Response(200, {"access_token": "a1", "expires_in": 900})], gets=[])
    api = TastytradeApi(path, session=session)
    path.write_text(json.dumps({"refresh_token": "r9", "client_id": "cid", "client_secret": "sec"}))
    api.access_token()
    assert session.post_bodies[0]["refresh_token"] == "r9"


def test_refused_credentials_pause_retries(tmp_path):
    path = _creds(tmp_path)
    session = _Session(posts=[_Response(401, {})], gets=[])
    api = TastytradeApi(path, session=session)
    with pytest.raises(TastytradeApiError):
        api.access_token()
    with pytest.raises(TastytradeApiError, match="not trying again"):
        api.access_token()
    assert len(session.post_bodies) == 1, "no second login attempt inside the pause"


# ---------------------------------------------------------------------------
# DXLink handshake against a fake socket
# ---------------------------------------------------------------------------

class _FakeSocket:
    def __init__(self, link, messages):
        self.link = link
        self.inbox = deque(json.dumps(m) for m in messages)
        self.sent = []
        self.closed = False

    def send(self, text):
        self.sent.append(json.loads(text))

    def recv(self, timeout=None):
        if self.inbox:
            return self.inbox.popleft()
        self.link._stop.set()
        raise TimeoutError

    def close(self):
        self.closed = True


class _FakeApi:
    def __init__(self, level="api"):
        self.level = level
        self.calls = 0

    def quote_token(self):
        self.calls += 1
        return QuoteToken(token=f"q{self.calls}", url="wss://fake", level=self.level, fetched=time.time())


def _link(messages, level="api"):
    prints, trades, live = [], [], []
    holder = {}

    def connect(url):
        holder["ws"] = _FakeSocket(link, messages)
        return holder["ws"]

    link = DxLinkTapeLink(_FakeApi(level), prints.append, trades.append, live.append, connect=connect)
    return link, holder, prints, trades, live


def test_handshake_subscribes_every_symbol_and_parses_prints():
    sale = ["TimeAndSale", "ABC", 0, 5, 1700000000000, "Q", 2.5, 100.0, 2.49, 2.51,
            "SELL", False, True, "NEW", "@   "]
    messages = [
        {"type": "SETUP", "channel": 0, "version": "x", "keepaliveTimeout": 60},
        {"type": "AUTH_STATE", "channel": 0, "state": "UNAUTHORIZED"},
        {"type": "AUTH_STATE", "channel": 0, "state": "AUTHORIZED"},
        {"type": "CHANNEL_OPENED", "channel": 1, "service": "FEED"},
        {"type": "FEED_CONFIG", "channel": 1, "aggregationPeriod": 0, "dataFormat": "COMPACT",
         "eventFields": FEED_FIELDS},
        {"type": "FEED_DATA", "channel": 1, "data": ["TimeAndSale", sale]},
    ]
    link, holder, prints, _, live = _link(messages)
    link.add("abc")
    link.add("XYZ")
    link._session()
    sent = holder["ws"].sent
    kinds = [m["type"] for m in sent]
    assert kinds[:5] == ["SETUP", "AUTH", "CHANNEL_REQUEST", "FEED_SETUP", "FEED_SUBSCRIPTION"]
    assert sent[1]["token"] == "q1"
    assert sent[2]["parameters"] == {"contract": "AUTO"} and sent[2]["service"] == "FEED"
    assert sent[3]["acceptDataFormat"] == "COMPACT" and sent[3]["acceptAggregationPeriod"] == 0.0
    assert set(sent[3]["acceptEventFields"]) == {"TimeAndSale", "Trade"}
    sub = sent[4]
    assert sub["reset"] is True
    assert {(a["type"], a["symbol"]) for a in sub["add"]} == {
        ("TimeAndSale", "ABC"), ("Trade", "ABC"), ("TimeAndSale", "XYZ"), ("Trade", "XYZ")}
    assert live == [True]
    assert [(p.symbol, p.price, p.aggressor) for p in prints] == [("ABC", 2.5, "SELL")]


def test_symbols_added_and_removed_after_ready_are_sent_incrementally():
    messages = [
        {"type": "AUTH_STATE", "state": "UNAUTHORIZED"},
        {"type": "AUTH_STATE", "state": "AUTHORIZED"},
        {"type": "CHANNEL_OPENED", "channel": 1},
        {"type": "FEED_CONFIG", "channel": 1, "eventFields": FEED_FIELDS},
    ]
    link, holder, *_ = _link(messages)
    link.add("ABC")
    original_recv = _FakeSocket.recv
    state = {"step": 0}

    def recv(self, timeout=None):
        if self.inbox:
            return self.inbox.popleft()
        state["step"] += 1
        if state["step"] == 1:
            link.add("NEW")
            link.remove("ABC")
            raise TimeoutError
        return original_recv(self, timeout)

    with patch.object(_FakeSocket, "recv", recv):
        link._session()
    subs = [m for m in holder["ws"].sent if m["type"] == "FEED_SUBSCRIPTION"]
    assert {(a["type"], a["symbol"]) for a in subs[1]["add"]} == {("TimeAndSale", "NEW"), ("Trade", "NEW")}
    assert {(a["type"], a["symbol"]) for a in subs[2]["remove"]} == {("TimeAndSale", "ABC"), ("Trade", "ABC")}


def test_a_refused_quote_token_reconnects_with_a_new_one():
    messages = [{"type": "AUTH_STATE", "state": "UNAUTHORIZED"},
                {"type": "AUTH_STATE", "state": "UNAUTHORIZED"}]
    link, *_ = _link(messages)
    with pytest.raises(_Reconnect):
        link._session()
    assert link._token is None


def test_a_delayed_feed_is_never_used():
    link, holder, *_ = _link([], level="demo")
    with pytest.raises(_Delayed):
        link._session()
    assert "ws" not in holder, "no socket is opened for delayed data"


# ---------------------------------------------------------------------------
# Wiring: IBKR stream, order-flow tracker, tape cache
# ---------------------------------------------------------------------------

def test_the_ibkr_tape_patch_keeps_ibkrs_trade_time():
    from market_data import ibkr_stream
    ticker = SimpleNamespace(tickByTicks=[], _momo_tape_events=deque(maxlen=4))
    wrapper = SimpleNamespace(reqId2Ticker={9: ticker})

    def orig(self, reqId, tickType, t, price, size, attrib, exchange, conditions):
        self.reqId2Ticker[reqId].tickByTicks.append(SimpleNamespace(price=price, size=size, exchange=exchange))

    with patch.object(ibkr_stream, "_orig_tick_by_tick", orig):
        ibkr_stream._tick_by_tick_all_last(wrapper, 9, 1, 1763384400, 5.0, 100, None, "ARCA", "")
    banked = ticker._momo_tape_events[0]
    assert banked.exchange_time == 1763384400 and banked.price == 5.0 and banked.exchange == "ARCA"


def _stream_with_tape(mode="merged"):
    from market_data.ibkr_stream import IBKRSharedStream
    with patch("market_data.ibkr_stream.IBKR_ENABLED", True), patch.object(IBKRSharedStream, "_run"):
        stream = IBKRSharedStream()
    stream._tape = MergedTape(stream._dispatch_print, mode)
    stream._tape.set_tastytrade_live(True)
    return stream


def _run_one_pass(stream):
    ib = MagicMock()
    ib.waitOnUpdate.side_effect = lambda timeout=None: setattr(stream, "_stopped", True)
    stream._loop(ib)


def test_stream_routes_ibkr_prints_through_the_merged_tape():
    stream = _stream_with_tape()
    got = []
    stream._contracts["ABC"] = MagicMock()  # already started: the queued add is a no-op
    stream.subscribe("ABC", got.append, lambda q: None, sub_id="t")
    stream._q_tickers["ABC"] = SimpleNamespace(bid=4.99, ask=5.01, last=5.0, bidSize=100, askSize=100)
    arrival = datetime(2026, 11, 17, 14, 0, tzinfo=timezone.utc)
    ticker = SimpleNamespace(tickByTicks=[], _momo_tape_events=deque())
    for exchange, price in (("NASDAQ", 5.00), ("ARCA", 5.01)):
        tick = SimpleNamespace(time=arrival, price=price, size=100, exchange=exchange, specialConditions="")
        ticker._momo_tape_events.append(SimpleNamespace(tick=tick, exchange_time=int(arrival.timestamp()),
                                                        **vars(tick)))
    stream._ts_tickers["ABC"] = ticker
    _run_one_pass(stream)
    assert [(e.exchange, e.origin, e.aggressor) for e in got] == [("ARCA", "ibkr", "buy")]


def test_stream_without_a_tape_passes_ibkr_prints_through_as_before():
    from market_data.ibkr_stream import IBKRSharedStream
    with patch("market_data.ibkr_stream.IBKR_ENABLED", True), patch.object(IBKRSharedStream, "_run"):
        stream = IBKRSharedStream()
    got = []
    stream._contracts["ABC"] = MagicMock()  # already started: the queued add is a no-op
    stream.subscribe("ABC", got.append, lambda q: None, sub_id="t")
    stream._q_tickers["ABC"] = SimpleNamespace(bid=4.99, ask=5.01, last=5.0, bidSize=100, askSize=100)
    tick = SimpleNamespace(time=datetime(2026, 11, 17, 14, 0, tzinfo=timezone.utc), price=5.0, size=100,
                           exchange="NASDAQ", specialConditions="")
    stream._ts_tickers["ABC"] = SimpleNamespace(tickByTicks=[], _momo_tape_events=deque([tick]))
    _run_one_pass(stream)
    assert [(e.exchange, e.aggressor, e.origin) for e in got] == [("NASDAQ", None, "")]


def test_order_flow_uses_the_tapes_side():
    from exec.order_flow import OrderFlowTracker
    from market_data.tradier_stream import LiveTimesale
    ofi = OrderFlowTracker()
    base = dict(event_type="timesale", symbol="ABC", exchange="ARCA", size=100, seq=1, flag="",
                cancel=False, correction=False, session="normal")
    now = datetime(2026, 11, 17, 14, 0, tzinfo=timezone.utc)
    sided = LiveTimesale(last=5.00, bid=5.00, ask=5.02, ts=now, aggressor="buy", **base)
    assert ofi.on_timesale(sided).side == "buy", "the tape's side wins over the quote rule"
    plain = LiveTimesale(last=5.00, bid=5.00, ask=5.02, ts=now + timedelta(seconds=1), **base)
    assert ofi.on_timesale(plain).side == "sell", "without one, the quote rule decides as before"


def test_tape_cache_records_side_and_keeps_old_files_aligned(tmp_path):
    import csv
    from chart_cache import RAW_TIMESALES_COLUMNS, append_timesales, load_timesales_cache, timesale_row_from_event
    from market_data.tradier_stream import LiveTimesale
    ts = datetime(2026, 11, 17, 14, 0, tzinfo=timezone.utc)
    ev = LiveTimesale(event_type="timesale", symbol="ABC", exchange="Q", last=5.0, bid=4.99, ask=5.01,
                      size=100, ts=ts, seq=1, flag="", cancel=False, correction=False, session="normal",
                      aggressor="sell", origin="tastytrade")
    row = timesale_row_from_event(ev)
    assert row["side"] == "sell" and row["origin"] == "tastytrade"
    append_timesales([row], tmp_path)
    loaded = load_timesales_cache("ABC", ts - timedelta(minutes=1), ts + timedelta(minutes=1), tmp_path)
    assert list(loaded["side"]) == ["sell"] and list(loaded["origin"]) == ["tastytrade"]

    old_cols = [c for c in RAW_TIMESALES_COLUMNS if c not in ("side", "origin")]
    old = tmp_path / "chart_cache" / "timesales_OLD_20261117.csv"
    with old.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_cols)
        writer.writeheader()
        writer.writerow({**{c: "" for c in old_cols}, "symbol": "OLD", "ts": ts.isoformat(), "price": 1, "size": 1})
    append_timesales([{**row, "symbol": "OLD"}], tmp_path)
    with old.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert all(len(r) == len(old_cols) for r in rows), "an old day file keeps its own columns"


def test_backtest_replay_reads_the_recorded_side():
    from exec.backtest import BacktestRunner
    import pandas as pd
    runner = BacktestRunner.__new__(BacktestRunner)
    row = pd.Series({"ts": "2026-11-17T14:00:00+00:00", "price": 5.0, "size": 100, "bid": 5.0, "ask": 5.02,
                     "exchange": "Q", "seq": 1, "flag": "", "session": "", "side": "buy", "origin": "tastytrade"})
    ev = runner._timesale_event_from_cache_row("ABC", row, 1)
    assert ev.aggressor == "buy" and ev.origin == "tastytrade"
    ev2 = runner._timesale_event_from_cache_row("ABC", row.drop(["side", "origin"]), 2)
    assert ev2.aggressor is None
