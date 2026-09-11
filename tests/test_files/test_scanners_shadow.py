"""Tests for ModernBERT shadow-mode wiring in the YouTube scanner.

Two things are being protected here:

  1. **The invariant.** In shadow mode the classifier must never queue a
     ticker, write the cache, or mutate ``queued`` — the trading path has
     to be identical to a run with the classifier switched off.
  2. **The corpus.** Every raw message gets a JSONL row, with the
     Clementine matcher's verdict alongside ModernBERT's, so the offline
     report can compare them.

No model and no network: the classifier is a stub and the chat log writes
into a temp directory.
"""
import json
import queue
import shutil
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import scanners
from market_data.chat_logger import ChatShadowLogger
from market_data.modernbert_client import NewsVerdict
from market_data.youtube_client import ChatMessage, ChatNewsHeadline

NEWS_AUTHOR = "@Clementineー"
_T0 = datetime(2026, 8, 3, 14, 30, tzinfo=timezone.utc)


def _msg(text: str, author: str = "@viewer", mid: str = "m1", secs: int = 0) -> ChatMessage:
    return ChatMessage(
        message_id=mid, author=author, text=text,
        timestamp=_T0 + timedelta(seconds=secs),
    )


class _StubClassifier:
    """Returns a preset score per text; records what it was asked to score."""

    model = "stub-model"
    threshold = 0.85

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[str] = []

    def classify_batch(self, texts):
        self.seen.extend(texts)
        out = []
        for t in texts:
            score = self.scores.get(t)
            out.append(None if score is None
                       else NewsVerdict(is_news=score >= self.threshold, score=score))
        return out


class _ShadowTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="yt-shadow-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.log = ChatShadowLogger(self.tmp)

    def rows(self) -> list[dict]:
        files = sorted(self.tmp.glob("chat_*.jsonl"))
        rows: list[dict] = []
        for f in files:
            with f.open("r", encoding="utf-8") as fh:
                rows.extend(json.loads(line) for line in fh if line.strip())
        return rows


class TestYtShadowClassify(_ShadowTestCase):
    def test_row_written_for_every_message(self):
        msgs = [_msg("lol", mid="a"),
                _msg("MSFT beats on earnings and guides higher", mid="b"),
                _msg("nothing to see here folks", mid="c")]
        clf = _StubClassifier({
            "MSFT beats on earnings and guides higher": 0.93,
            "nothing to see here folks": 0.10,
        })
        scanners._yt_shadow_classify(msgs, [], clf, self.log, None, first_poll=False)

        rows = self.rows()
        self.assertEqual([r["message_id"] for r in rows], ["a", "b", "c"])
        # "lol" is below the min-words guard — logged, never scored.
        self.assertFalse(rows[0]["mb_evaluated"])
        self.assertIsNone(rows[0]["mb_score"])
        self.assertNotIn("lol", clf.seen)
        self.assertTrue(rows[1]["mb_evaluated"])
        self.assertTrue(rows[1]["mb_is_news"])
        self.assertEqual(rows[1]["mb_ticker"], "MSFT")
        self.assertEqual(rows[1]["mb_ticker_method"], "bare")
        self.assertFalse(rows[2]["mb_is_news"])
        self.assertEqual(rows[2]["mb_ticker"], None)
        for r in rows:
            self.assertEqual(r["model"], "stub-model")
            self.assertEqual(r["threshold"], 0.85)

    def test_clementine_hit_tagged(self):
        msgs = [_msg("$AAPL - beats earnings", author=NEWS_AUTHOR, mid="a"),
                _msg("some random chatter here", mid="b", secs=1)]
        headlines = [ChatNewsHeadline(
            ticker="AAPL", headline="beats earnings",
            author=NEWS_AUTHOR, timestamp=_T0, bypass_noise_filter=True,
        )]
        clf = _StubClassifier({"$AAPL - beats earnings": 0.97,
                               "some random chatter here": 0.05})
        scanners._yt_shadow_classify(msgs, headlines, clf, self.log, None, first_poll=False)

        rows = self.rows()
        self.assertTrue(rows[0]["clementine_hit"])
        self.assertEqual(rows[0]["clementine_ticker"], "AAPL")
        self.assertFalse(rows[1]["clementine_hit"])
        self.assertIsNone(rows[1]["clementine_ticker"])

    def test_first_poll_logs_but_does_not_classify(self):
        msgs = [_msg("MSFT beats on earnings and guides higher", mid="a")]
        clf = _StubClassifier({"MSFT beats on earnings and guides higher": 0.99})
        out = scanners._yt_shadow_classify(msgs, [], clf, self.log, None, first_poll=True)

        self.assertEqual(out, [])
        self.assertEqual(clf.seen, [])
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["mb_evaluated"])

    def test_symbol_resolver_fallback(self):
        text = "Rocket Lab wins a major defense contract today"
        msgs = [_msg(text, mid="a")]
        clf = _StubClassifier({text: 0.95})
        resolver = MagicMock(resolve=MagicMock(return_value="RKLB"))
        out = scanners._yt_shadow_classify(msgs, [], clf, self.log, resolver, first_poll=False)

        resolver.resolve.assert_called_once_with("Rocket")
        rows = self.rows()
        self.assertEqual(rows[0]["mb_ticker"], "RKLB")
        self.assertEqual(rows[0]["mb_ticker_method"], "ibkr_search")
        self.assertEqual([h.ticker for h in out], ["RKLB"])

    def test_resolver_result_outside_3_to_4_chars_rejected(self):
        for resolved in ("GM", "GOOGL"):
            text = "General Motors beats on earnings and guides higher"
            msgs = [_msg(text, mid=f"a-{resolved}")]
            clf = _StubClassifier({text: 0.95})
            resolver = MagicMock(resolve=MagicMock(return_value=resolved))
            out = scanners._yt_shadow_classify(msgs, [], clf, self.log, resolver, first_poll=False)

            rows = self.rows()
            self.assertIsNone(rows[-1]["mb_ticker"], resolved)
            self.assertEqual(rows[-1]["mb_ticker_method"], "none", resolved)
            self.assertEqual(out, [], resolved)

    def test_news_with_no_resolvable_ticker(self):
        text = "the whole market is ripping right now"
        msgs = [_msg(text, mid="a")]
        clf = _StubClassifier({text: 0.95})
        resolver = MagicMock(resolve=MagicMock(return_value=None))
        out = scanners._yt_shadow_classify(msgs, [], clf, self.log, resolver, first_poll=False)

        rows = self.rows()
        self.assertEqual(rows[0]["mb_ticker_method"], "none")
        self.assertIsNone(rows[0]["mb_ticker"])
        self.assertEqual(out, [])

    def test_detections_dedup_against_clementine(self):
        msgs = [_msg("$AAPL - beats earnings", author=NEWS_AUTHOR, mid="a")]
        headlines = [ChatNewsHeadline(
            ticker="AAPL", headline="beats earnings",
            author=NEWS_AUTHOR, timestamp=_T0, bypass_noise_filter=True,
        )]
        clf = _StubClassifier({"$AAPL - beats earnings": 0.99})
        out = scanners._yt_shadow_classify(msgs, headlines, clf, self.log, None, first_poll=False)
        self.assertEqual(out, [])   # already covered by the matcher

    def test_bypass_noise_filter_only_for_trusted_authors(self):
        generic = "$RKLB - wins a major defense contract"
        trusted = "$NVDA - guides above consensus"
        msgs = [_msg(generic, mid="a"),
                _msg(trusted, author=NEWS_AUTHOR, mid="b", secs=1)]
        clf = _StubClassifier({generic: 0.95, trusted: 0.95})
        out = scanners._yt_shadow_classify(msgs, [], clf, self.log, None, first_poll=False)

        by_ticker = {h.ticker: h for h in out}
        self.assertFalse(by_ticker["RKLB"].bypass_noise_filter)
        self.assertTrue(by_ticker["NVDA"].bypass_noise_filter)
        # Structured prefix → ticker stripped from the headline body.
        self.assertEqual(by_ticker["RKLB"].headline, "wins a major defense contract")

    def test_classifier_failure_is_swallowed(self):
        msgs = [_msg("MSFT beats on earnings and guides higher", mid="a")]
        clf = MagicMock(model="m", threshold=0.85)
        clf.classify_batch.side_effect = RuntimeError("model exploded")
        out = scanners._yt_shadow_classify(msgs, [], clf, self.log, None, first_poll=False)
        self.assertEqual(out, [])

    def test_empty_messages(self):
        self.assertEqual(
            scanners._yt_shadow_classify([], [], None, self.log, None, first_poll=False), [],
        )
        self.assertEqual(self.rows(), [])


# ---------------------------------------------------------------------------
# Scanner-level invariant
# ---------------------------------------------------------------------------

class _StubChatClient:
    """Yields canned polls, then stops the scanner."""

    video_id = "vid"
    poll_timeout_ms = 0

    def __init__(self, polls, stop_event):
        self._polls = list(polls)
        self._stop = stop_event

    def poll(self):
        if self._polls:
            return self._polls.pop(0)
        self._stop.set()
        return []


def _run_scanner(polls, clf, chat_log, *, resolver=None):
    """Drive ``_youtube_scanner`` over *polls* and return the observed effects."""
    stop_event = threading.Event()
    ticker_q = MagicMock(spec=queue.Queue)
    cache = MagicMock()
    cache.has.return_value = False
    queued: set[str] = set()
    ready = threading.Event()

    t = threading.Thread(
        target=scanners._youtube_scanner,
        args=(ticker_q, stop_event, MagicMock(), cache, queued,
              threading.Event(), _StubChatClient(polls, stop_event),
              threading.Event(), MagicMock(), MagicMock(), ready, [0],
              threading.Lock(), set()),
        kwargs={"news_clf": clf, "chat_log": chat_log, "symbol_resolver": resolver},
        daemon=True,
    )
    t.start()
    t.join(timeout=10.0)
    stop_event.set()
    t.join(timeout=5.0)
    assert not t.is_alive(), "scanner thread did not stop"
    return ticker_q, cache, queued


class TestScannerShadowInvariant(_ShadowTestCase):
    def test_shadow_mode_never_queues(self):
        # A confident, well-formed news post from a *generic* author. In
        # shadow mode nothing about it may reach the trading path.
        text = "$RKLB - wins a major defense contract"
        polls = [[_msg("warmup message from the backlog", mid="w")],
                 [_msg(text, mid="a", secs=1)]]
        clf = _StubClassifier({text: 0.99,
                               "warmup message from the backlog": 0.01})
        ticker_q, cache, queued = _run_scanner(polls, clf, self.log)

        ticker_q.put.assert_not_called()
        cache.add_pending_yt_news.assert_not_called()
        self.assertEqual(queued, set())
        # …but it was still classified and logged.
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[1]["mb_is_news"])
        self.assertEqual(rows[1]["mb_ticker"], "RKLB")

    def test_clementine_path_unaffected_by_shadow(self):
        text = "$AAPL - beats earnings"
        polls = [[_msg("warmup message from the backlog", mid="w")],
                 [_msg(text, author=NEWS_AUTHOR, mid="a", secs=1)]]
        clf = _StubClassifier({text: 0.02,     # classifier disagrees — irrelevant
                               "warmup message from the backlog": 0.01})
        ticker_q, cache, queued = _run_scanner(polls, clf, self.log)

        ticker_q.put.assert_called_with(("youtube-news", "AAPL"))
        self.assertEqual(cache.add_pending_yt_news.call_args[0][0], "AAPL")
        self.assertIn("AAPL", queued)

    def test_active_mode_queues_generic_author_news(self):
        text = "$RKLB - wins a major defense contract"
        polls = [[_msg("warmup message from the backlog", mid="w")],
                 [_msg(text, mid="a", secs=1)]]
        clf = _StubClassifier({text: 0.99,
                               "warmup message from the backlog": 0.01})
        original = scanners.SCANNER_YT_MODERNBERT_MODE
        scanners.SCANNER_YT_MODERNBERT_MODE = "active"
        try:
            ticker_q, cache, queued = _run_scanner(polls, clf, self.log)
        finally:
            scanners.SCANNER_YT_MODERNBERT_MODE = original

        ticker_q.put.assert_called_with(("youtube-news", "RKLB"))
        self.assertIn("RKLB", queued)
        headline = cache.add_pending_yt_news.call_args[0][1]
        self.assertEqual(headline.ticker, "RKLB")
        self.assertEqual(headline.headline, "wins a major defense contract")
        # Generic author → the FinBERT noise gate still has to screen it.
        self.assertFalse(headline.bypass_noise_filter)


# ---------------------------------------------------------------------------
# Chat logger
# ---------------------------------------------------------------------------

class TestChatShadowLogger(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="yt-chatlog-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_dated_file_and_utf8_roundtrip(self):
        log = ChatShadowLogger(self.tmp / "nested")
        log.log_records([
            {"author": NEWS_AUTHOR, "text": "テスト — em dash & 🚀"},
            {"author": "@viewer", "text": "plain"},
        ])
        files = list((self.tmp / "nested").glob("chat_*.jsonl"))
        self.assertEqual(len(files), 1)
        self.assertRegex(files[0].name, r"^chat_\d{8}\.jsonl$")

        with files[0].open("r", encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        self.assertEqual(rows[0]["author"], NEWS_AUTHOR)
        self.assertEqual(rows[0]["text"], "テスト — em dash & 🚀")
        # Non-ASCII stored raw, not \uXXXX-escaped
        self.assertIn("テスト", files[0].read_text(encoding="utf-8"))

    def test_appends_across_batches(self):
        log = ChatShadowLogger(self.tmp)
        log.log_records([{"n": 1}])
        log.log_records([{"n": 2}, {"n": 3}])
        rows = [json.loads(l) for l in log.path_for_today().read_text(
            encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual([r["n"] for r in rows], [1, 2, 3])

    def test_empty_batch_creates_nothing(self):
        log = ChatShadowLogger(self.tmp / "unused")
        log.log_records([])
        self.assertFalse((self.tmp / "unused").exists())

    def test_exotic_values_stringified_not_raised(self):
        log = ChatShadowLogger(self.tmp)
        log.log_records([{"ts": _T0, "obj": object()}])
        rows = [json.loads(l) for l in log.path_for_today().read_text(
            encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(rows[0]["ts"], str(_T0))
        self.assertIsInstance(rows[0]["obj"], str)


if __name__ == "__main__":
    unittest.main()
