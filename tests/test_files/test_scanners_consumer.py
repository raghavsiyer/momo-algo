from __future__ import annotations

import queue
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import scanners
from market_data.youtube_client import ChatNewsHeadline
from pipeline import PipelineResult


def _headline(sym: str) -> ChatNewsHeadline:
    return ChatNewsHeadline(
        ticker=sym,
        headline="Sample catalyst headline",
        author="yt-author",
        timestamp=datetime.now(timezone.utc),
    )


def _start_consumer_once(
    *,
    source: str,
    sym: str,
    pipeline_result,
    pending_news: dict[str, list[ChatNewsHeadline]],
) -> tuple[SimpleNamespace, threading.Thread]:
    ticker_q: queue.Queue = queue.Queue()
    ticker_q.put((source, sym))
    queued = {sym}
    pipeline_done: set[str] = set()

    stop_event = threading.Event()
    cache = SimpleNamespace(
        _pending=pending_news,
        has=MagicMock(return_value=False),
        get_pending_yt_news=MagicMock(side_effect=lambda s: cache._pending.get(s)),
        pop_pending_yt_news=MagicMock(side_effect=lambda s: cache._pending.pop(s, None)),
        put=MagicMock(),
        flush=MagicMock(),
    )

    evaluator = SimpleNamespace(
        watchlist=[],
        watchlist_symbols=set(),
        last_fail_reason="rvol_1day: 3.5x (< 5x)",
    )

    def _run_pipeline(sym_arg, evaluator_arg, tradier_arg, yt_headlines=None, quiet=False, allow_manual_rating=False, **kwargs):
        assert sym_arg == sym
        assert yt_headlines is pending_news[sym]
        stop_event.set()
        return pipeline_result

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
                "queued": queued,
                "pipeline_done": pipeline_done,
                "scanner_lock": threading.Lock(),
            },
            daemon=True,
        )
        t.start()
        t.join(timeout=2.0)
        if t.is_alive():
            stop_event.set()
            t.join(timeout=2.0)
        return cache, t, queued, pipeline_done
    finally:
        scanners.run_pipeline = original


def test_scanner_consumer_retains_pending_yt_news_on_pipeline_none() -> None:
    sym = "KIDZ"
    pending_news = {sym: [_headline(sym)]}

    _, t, _, _ = _start_consumer_once(
        source="youtube-news",
        sym=sym,
        pipeline_result=None,
        pending_news=pending_news,
    )

    assert not t.is_alive()
    assert sym in pending_news
    assert len(pending_news[sym]) == 1


def test_scanner_consumer_consumes_pending_yt_news_after_success() -> None:
    sym = "KIDZ"
    pending_news = {sym: [_headline(sym)]}

    cache, t, _, _ = _start_consumer_once(
        source="youtube-news",
        sym=sym,
        pipeline_result=PipelineResult(entry=object(), rmap=object(), scorer=None, early_stream=None, passed=True),
        pending_news=pending_news,
    )

    assert not t.is_alive()
    assert sym not in pending_news
    cache.pop_pending_yt_news.assert_called_once_with(sym)
    cache.put.assert_called_once()
    cache.flush.assert_called_once()



def test_only_rvol_failures_are_retryable() -> None:
    # "change ..." reasons no longer exist — the prefilter stopped rejecting
    # on % change (scanner floor + ATTENTION criterion carry that gate).
    assert scanners._is_soft_failure("change 8.8% < 10.0%") is False
    assert scanners._is_soft_failure("rvol_1day: 3.5x (< 5x)") is True
    assert scanners._failures_are_soft(["rvol_1day: 3.5x (< 5x)"]) is True
    assert scanners._is_soft_failure("volume 1000 < 50000") is False
    assert scanners._is_soft_failure("price $25.00 not in [0.1-20.0]") is False
    assert scanners._failures_are_soft(["rvol_1day: 3.5x (< 5x)", "float: 77.60M (> 50M)"]) is False


def test_scanner_consumer_retries_soft_pipeline_result_failure() -> None:
    sym = "TNMG"
    pending_news = {sym: [_headline(sym)]}

    _, t, queued, pipeline_done = _start_consumer_once(
        source="momo",
        sym=sym,
        pipeline_result=PipelineResult(
            entry=object(),
            rmap=object(),
            scorer=None,
            early_stream=None,
            passed=False,
            fail_reasons=["rvol_1day: 3.5x (< 5x)"],
        ),
        pending_news=pending_news,
    )

    assert not t.is_alive()
    assert sym not in queued
    assert sym not in pipeline_done
