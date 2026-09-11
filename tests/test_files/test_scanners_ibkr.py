"""
tests/test_files/test_scanners_ibkr.py

Unit tests for the IBKR top-gainers scanner feed: the novelty tracker, the
scanner thread's queueing behaviour, and the consumer's ``ibkr`` re-hit
branch.  No IB connection is required — the client is a stub.
"""

from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import scanners
from market_data.ibkr_rest import ScannerRow
from pipeline import PipelineResult


# ---------------------------------------------------------------------------
# _ScanNoveltyTracker — pure, no threads
# ---------------------------------------------------------------------------

class TestScanNoveltyTracker:
    """A ranked snapshot repeats every poll; only entries are events."""

    def test_first_poll_seeds_silently(self):
        t = scanners._ScanNoveltyTracker(900)
        assert t.update(["ACMR", "IONQ"], now=100.0, first_poll=True) == []

    def test_new_entrants_returned_in_rank_order(self):
        t = scanners._ScanNoveltyTracker(900)
        t.update(["ACMR"], now=100.0, first_poll=True)
        assert t.update(["ACMR", "KIDZ", "IONQ"], now=110.0) == ["KIDZ", "IONQ"]

    def test_steady_state_is_silent(self):
        t = scanners._ScanNoveltyTracker(900)
        t.update(["ACMR", "IONQ"], now=100.0, first_poll=True)
        assert t.update(["ACMR", "IONQ"], now=110.0) == []
        assert t.update(["IONQ", "ACMR"], now=120.0) == []

    def test_reappear_before_aging_stays_silent(self):
        """Dropping off for one poll is scanner churn, not a new event."""
        t = scanners._ScanNoveltyTracker(900)
        t.update(["ACMR"], now=100.0, first_poll=True)
        assert t.update([], now=130.0) == []
        assert t.update(["ACMR"], now=160.0) == []

    def test_reappear_after_aging_re_emits(self):
        """A second run at the top later in the session is a real event."""
        t = scanners._ScanNoveltyTracker(900)
        t.update(["ACMR"], now=100.0, first_poll=True)
        assert t.update([], now=500.0) == []
        assert t.update(["ACMR"], now=1100.0) == ["ACMR"]

    def test_duplicate_symbol_in_one_snapshot_emits_once(self):
        t = scanners._ScanNoveltyTracker(900)
        t.update([], now=100.0, first_poll=True)
        assert t.update(["ACMR", "ACMR"], now=110.0) == ["ACMR"]


class TestScanNoveltyTrackerDrops:
    """Losing a seat on the ranked list is the mirror of gaining one."""

    def test_still_listed_is_not_a_drop(self):
        t = scanners._ScanNoveltyTracker(900, drop_secs=30)
        t.update(["ACMR"], now=100.0, first_poll=True)
        t.update(["ACMR"], now=200.0)
        assert t.dropped({"ACMR"}, now=210.0) == []

    def test_absent_within_grace_is_not_a_drop(self):
        """Rank churn around the last slot must not stop a scorer."""
        t = scanners._ScanNoveltyTracker(900, drop_secs=30)
        t.update(["ACMR"], now=100.0, first_poll=True)
        assert t.dropped({"ACMR"}, now=125.0) == []

    def test_absent_beyond_grace_is_a_drop(self):
        t = scanners._ScanNoveltyTracker(900, drop_secs=30)
        t.update(["ACMR"], now=100.0, first_poll=True)
        assert t.dropped({"ACMR"}, now=140.0) == ["ACMR"]

    def test_unseen_symbol_is_never_a_drop(self):
        """A momo/YouTube discovery never held an IBKR seat to lose."""
        t = scanners._ScanNoveltyTracker(900, drop_secs=30)
        t.update(["ACMR"], now=100.0, first_poll=True)
        assert t.dropped({"KIDZ"}, now=999.0) == []

    def test_forget_stops_repeat_drops_and_rearms_entry(self):
        t = scanners._ScanNoveltyTracker(900, drop_secs=30)
        t.update(["ACMR"], now=100.0, first_poll=True)
        assert t.dropped({"ACMR"}, now=140.0) == ["ACMR"]
        t.forget("ACMR")
        # No longer tracked, so it cannot drop again...
        assert t.dropped({"ACMR"}, now=200.0) == []
        # ...and a genuine return is a fresh entry event.
        assert t.update(["ACMR"], now=260.0) == ["ACMR"]

    def test_reset_clears_all_marks(self):
        """The RTH scan-code swap replaces the list wholesale."""
        t = scanners._ScanNoveltyTracker(900, drop_secs=30)
        t.update(["ACMR", "IONQ"], now=100.0, first_poll=True)
        t.reset()
        assert t.dropped({"ACMR", "IONQ"}, now=999.0) == []


# ---------------------------------------------------------------------------
# _ibkr_scanner thread
# ---------------------------------------------------------------------------

def _run_ibkr_scanner_once(scan_results: list, *, queued=None, scorers=None,
                           cache=None, interval=None, force_poll=None,
                           on_scan=None, on_drop=None, watchlist_symbols=None):
    """Drive one poll of ``_ibkr_scanner`` and return the queued tuples.

    *scan_results* holds one entry per poll — either a list of ``ScannerRow``
    or an exception to raise.  The thread is stopped once they run out.
    """
    ticker_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    force_poll = force_poll if force_poll is not None else threading.Event()
    interval = interval if interval is not None else [0]
    calls = list(scan_results)

    def _scan():
        if not calls:
            stop_event.set()
            return []
        nxt = calls.pop(0)
        if not calls:
            # Last scripted poll — let the loop finish its tail and exit.
            stop_event.set()
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    client = SimpleNamespace(scan_top_gainers=_scan)
    cache = cache if cache is not None else SimpleNamespace(has=MagicMock(return_value=False))
    evaluator = SimpleNamespace(
        watchlist_symbols=set() if watchlist_symbols is None else watchlist_symbols,
    )
    queued = queued if queued is not None else set()

    with patch.object(scanners, "_print_scanner_hit") as mock_hit, \
         patch("market_data.ibkr_rest.release_ib") as mock_release:
        t = threading.Thread(
            target=scanners._ibkr_scanner,
            args=(ticker_q, stop_event, evaluator, cache, queued, client,
                  threading.Event(), interval, threading.Event(), force_poll,
                  set(), threading.Lock(), None, scorers),
            kwargs={"on_scan": on_scan, "on_drop": on_drop},
            daemon=True,
        )
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive()

    out = []
    while not ticker_q.empty():
        out.append(ticker_q.get_nowait())
    return out, queued, mock_hit, mock_release


class TestIBKRScannerThread:

    def test_first_poll_seeds_without_queueing(self):
        out, queued, _hit, _rel = _run_ibkr_scanner_once(
            [[ScannerRow("ACMR", 0), ScannerRow("IONQ", 1)]],
        )
        assert out == []
        assert queued == set()

    def test_new_entrant_is_queued_with_ibkr_source(self):
        out, queued, mock_hit, _rel = _run_ibkr_scanner_once([
            [ScannerRow("ACMR", 0)],
            [ScannerRow("ACMR", 0), ScannerRow("KIDZ", 1)],
        ])
        assert out == [("ibkr", "KIDZ")]
        assert queued == {"KIDZ"}
        assert mock_hit.call_args.kwargs["source"] == "ibkr"

    def test_already_queued_symbol_is_not_requeued(self):
        out, _queued, mock_hit, _rel = _run_ibkr_scanner_once(
            [[], [ScannerRow("ACMR", 0)]], queued={"ACMR"},
        )
        assert out == []
        # Still printed — the operator wants to see the recurring hit.
        assert mock_hit.called

    def test_invalid_symbols_are_dropped(self):
        out, _queued, _hit, _rel = _run_ibkr_scanner_once(
            [[], [ScannerRow("BRK B", 0), ScannerRow("KIDZ", 1)]],
        )
        assert out == [("ibkr", "KIDZ")]

    def test_scan_failure_does_not_kill_the_thread(self):
        """Gateway down must degrade to a warning and a retry next poll."""
        from market_data.tradier_rest import MarketDataUnavailable
        out, queued, _hit, mock_release = _run_ibkr_scanner_once([
            [],
            MarketDataUnavailable("gateway down"),
            [ScannerRow("KIDZ", 0)],
        ])
        assert out == [("ibkr", "KIDZ")]
        assert queued == {"KIDZ"}

    def test_connection_is_released_on_exit(self):
        """The thread holds its own IB connection against a 32-client cap."""
        _out, _queued, _hit, mock_release = _run_ibkr_scanner_once([[]])
        mock_release.assert_called_once()

    def test_scan_failure_backoff_grows_caps_and_resets(self):
        """Consecutive failures double the retry wait (capped), success resets it."""
        from market_data.tradier_rest import MarketDataUnavailable

        class _RecordingEvent(threading.Event):
            def __init__(self):
                super().__init__()
                self.waits: list = []

            def wait(self, timeout=None):
                self.waits.append(timeout)
                return False

        force_poll = _RecordingEvent()
        out, _queued, _hit, _rel = _run_ibkr_scanner_once(
            [[]] + [MarketDataUnavailable("gateway down")] * 6
                 + [[ScannerRow("KIDZ", 0)]],
            interval=[2], force_poll=force_poll,
        )
        assert force_poll.waits == [2, 4, 8, 16, 32, 60, 60, 2]
        assert out == [("ibkr", "KIDZ")]


class TestIBKRScannerOnScan:
    """The `top` display repaints at the feed's own cadence via on_scan."""

    def test_fires_once_per_successful_poll_including_the_seed(self):
        """The seed poll queues nothing but does produce the first snapshot,
        so the view must be told about it."""
        calls = []
        _out, _queued, _hit, _rel = _run_ibkr_scanner_once(
            [[ScannerRow("ACMR", 0)], [ScannerRow("ACMR", 0), ScannerRow("KIDZ", 1)]],
            on_scan=lambda: calls.append(1),
        )
        assert len(calls) == 2

    def test_does_not_fire_on_a_failed_poll(self):
        """Nothing new was published — a repaint would only reset the age."""
        from market_data.tradier_rest import MarketDataUnavailable
        calls = []
        _out, _queued, _hit, _rel = _run_ibkr_scanner_once(
            [[], MarketDataUnavailable("gateway down"), [ScannerRow("KIDZ", 0)]],
            on_scan=lambda: calls.append(1),
        )
        assert len(calls) == 2

    def test_raising_callback_does_not_kill_the_feed(self):
        """A live view is a spectator — a render bug must not stop scanning."""
        def _boom():
            raise RuntimeError("toolbar exploded")

        out, queued, _hit, mock_release = _run_ibkr_scanner_once(
            [[ScannerRow("ACMR", 0)], [ScannerRow("ACMR", 0), ScannerRow("KIDZ", 1)]],
            on_scan=_boom,
        )
        assert out == [("ibkr", "KIDZ")]
        assert queued == {"KIDZ"}
        mock_release.assert_called_once()

    def test_absent_callback_is_the_default(self):
        _out, _queued, _hit, _rel = _run_ibkr_scanner_once([[ScannerRow("ACMR", 0)]])


# ---------------------------------------------------------------------------
# Consumer — ibkr re-hit branch
# ---------------------------------------------------------------------------

def _run_consumer_rehit(source: str, sym: str, float_shares_m: float):
    """Feed one (source, sym) for a cached symbol and capture the re-eval."""
    ticker_q: queue.Queue = queue.Queue()
    ticker_q.put((source, sym))
    stop_event = threading.Event()

    old_entry = SimpleNamespace(symbol=sym, float_shares_m=float_shares_m)
    cache = SimpleNamespace(
        has=MagicMock(return_value=True),
        get=MagicMock(return_value=(old_entry, object())),
        get_news=MagicMock(return_value=None),
        get_pending_yt_news=MagicMock(return_value=None),
        pop_pending_yt_news=MagicMock(return_value=None),
        put=MagicMock(),
        flush=MagicMock(),
    )
    evaluator = SimpleNamespace(watchlist=[], watchlist_symbols=set(), last_fail_reason="")
    seen: dict = {}
    result = PipelineResult(
        entry=SimpleNamespace(symbol=sym), rmap=object(),
        scorer=None, early_stream=None, passed=True,
    )

    def _run_pipeline(sym_arg, evaluator_arg, tradier_arg, **kwargs):
        seen["sym"] = sym_arg
        seen.update(kwargs)
        stop_event.set()
        return result

    original = scanners.run_pipeline
    scanners.run_pipeline = _run_pipeline
    try:
        t = threading.Thread(
            target=scanners._scanner_consumer,
            kwargs={
                "ticker_q": ticker_q,
                "stop_event": stop_event,
                "evaluator": evaluator,
                "tradier": MagicMock(),
                "scorers": {},
                "cache": cache,
                "queued": {sym},
                "pipeline_done": set(),
                "scanner_lock": threading.Lock(),
            },
            daemon=True,
        )
        t.start()
        t.join(timeout=5.0)
        stop_event.set()
        t.join(timeout=2.0)
        assert not t.is_alive()
    finally:
        scanners.run_pipeline = original
    return seen, cache


class TestConsumerIBKRRehit:
    """A cached name re-hit by the IBKR scanner must re-evaluate, not skip."""

    def test_ibkr_rehit_reuses_cached_float(self):
        seen, cache = _run_consumer_rehit("ibkr", "ACMR", 4.25)
        assert seen["sym"] == "ACMR"
        assert seen["cached_float"] == 4.25
        cache.put.assert_called_once()
        cache.flush.assert_called_once()

    def test_momo_rehit_still_works(self):
        seen, _cache = _run_consumer_rehit("momo", "ACMR", 8.5)
        assert seen["cached_float"] == 8.5


class TestIBKRScannerDropCallback:
    """on_drop fires for watched names that lose their seat."""

    def test_drop_fires_for_watched_symbol_off_the_list(self):
        dropped: list[list[str]] = []
        # drop_secs=0 → absence on the very next poll counts, so the test
        # needs no clock control.
        with patch.object(scanners, "SCANNER_IBKR_DROP_SECS", 0):
            _run_ibkr_scanner_once(
                [[ScannerRow("ACMR", 0)], []],
                scorers={"ACMR": MagicMock()},
                watchlist_symbols={"ACMR"},
                on_drop=dropped.append,
            )
        assert dropped == [["ACMR"]]

    def test_no_drop_while_symbol_holds_its_seat(self):
        dropped: list[list[str]] = []
        with patch.object(scanners, "SCANNER_IBKR_DROP_SECS", 0):
            _run_ibkr_scanner_once(
                [[ScannerRow("ACMR", 0)], [ScannerRow("ACMR", 0)]],
                scorers={"ACMR": MagicMock()},
                watchlist_symbols={"ACMR"},
                on_drop=dropped.append,
            )
        assert dropped == []

    def test_no_drop_for_symbol_without_a_scorer(self):
        """Already faded, or never watched — nothing to stop."""
        dropped: list[list[str]] = []
        with patch.object(scanners, "SCANNER_IBKR_DROP_SECS", 0):
            _run_ibkr_scanner_once(
                [[ScannerRow("ACMR", 0)], []],
                scorers={},
                watchlist_symbols={"ACMR"},
                on_drop=dropped.append,
            )
        assert dropped == []

    def test_drop_fires_once_not_every_poll(self):
        dropped: list[list[str]] = []
        with patch.object(scanners, "SCANNER_IBKR_DROP_SECS", 0):
            _run_ibkr_scanner_once(
                [[ScannerRow("ACMR", 0)], [], [], []],
                scorers={"ACMR": MagicMock()},
                watchlist_symbols={"ACMR"},
                on_drop=dropped.append,
            )
        assert dropped == [["ACMR"]]

    def test_failing_on_drop_does_not_kill_the_feed(self):
        def _boom(_syms):
            raise RuntimeError("callback exploded")

        with patch.object(scanners, "SCANNER_IBKR_DROP_SECS", 0):
            out, _queued, _hit, _rel = _run_ibkr_scanner_once(
                [[ScannerRow("ACMR", 0)], [], [ScannerRow("KIDZ", 1)]],
                scorers={"ACMR": MagicMock()},
                watchlist_symbols={"ACMR"},
                on_drop=_boom,
            )
        # The feed survived and kept queueing new entrants.
        assert ("ibkr", "KIDZ") in out
