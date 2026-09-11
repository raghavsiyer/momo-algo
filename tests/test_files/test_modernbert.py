"""Tests for the ModernBERT zero-shot news classifier and the paced
IBKR symbol resolver that backs it.

No model is ever loaded: ``_pipe`` is replaced with a stub, which makes
``_ensure_model()`` a no-op.  No network, no IBKR connection.
"""
import unittest
from unittest.mock import MagicMock

from market_data.modernbert_client import (
    ModernBERTNewsClassifier,
    NewsVerdict,
    alpha_word_count,
)
from scanners import RateLimitedSymbolResolver


class _FakePipe:
    """Stands in for a transformers zero-shot pipeline.

    Mirrors the real one's contract: with several candidate labels the
    result is sorted by score descending, so the news label is *not*
    necessarily first.
    """

    def __init__(self, scores, *, single_dict=False, raises=False):
        self.scores = list(scores)
        self.single_dict = single_dict
        self.raises = raises
        self.calls: list[list[str]] = []
        self.label_calls: list[list[str]] = []
        self.multi_label_calls: list[bool] = []

    def __call__(self, texts, candidate_labels=None, hypothesis_template=None,
                 multi_label=None):
        if self.raises:
            raise RuntimeError("boom")
        self.calls.append(list(texts))
        self.label_calls.append(list(candidate_labels))
        self.multi_label_calls.append(multi_label)
        out = []
        for t in texts:
            score = self.scores.pop(0)
            pairs = [(candidate_labels[0], score)]
            for other in candidate_labels[1:]:
                pairs.append((other, 1.0 - score))
            pairs.sort(key=lambda p: p[1], reverse=True)
            out.append({
                "sequence": t,
                "labels": [p[0] for p in pairs],
                "scores": [p[1] for p in pairs],
            })
        return out[0] if (self.single_dict and len(out) == 1) else out


def _clf(scores, threshold=0.85, **kw) -> ModernBERTNewsClassifier:
    clf = ModernBERTNewsClassifier(threshold=threshold)
    clf._pipe = _FakePipe(scores, **kw)
    return clf


class TestAlphaWordCount(unittest.TestCase):
    def test_counts_only_words(self):
        self.assertEqual(alpha_word_count("MSFT beats on earnings"), 4)
        self.assertEqual(alpha_word_count("$AAPL 🚀🚀 12 3"), 1)
        self.assertEqual(alpha_word_count(""), 0)


class TestClassifyBatch(unittest.TestCase):
    def test_threshold_flip(self):
        clf = _clf([0.91, 0.84], threshold=0.85)
        out = clf.classify_batch([
            "MSFT beats on earnings and guides higher",
            "this stock is going to the moon boys",
        ])
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].is_news)
        self.assertAlmostEqual(out[0].score, 0.91)
        self.assertFalse(out[1].is_news)
        self.assertAlmostEqual(out[1].score, 0.84)

    def test_threshold_boundary_is_inclusive(self):
        clf = _clf([0.85], threshold=0.85)
        self.assertTrue(clf.classify_batch(["MSFT beats on earnings"])[0].is_news)

    def test_order_preserved_with_skipped_texts(self):
        # "lol" is below the min-words guard: it gets None and must not
        # consume a score or shift the others.
        clf = _clf([0.90, 0.10], threshold=0.85)
        out = clf.classify_batch([
            "MSFT beats on earnings",
            "lol",
            "random chatter about nothing",
        ])
        self.assertTrue(out[0].is_news)
        self.assertIsNone(out[1])
        self.assertFalse(out[2].is_news)
        self.assertEqual(clf._pipe.calls, [[
            "MSFT beats on earnings", "random chatter about nothing",
        ]])

    def test_all_short_texts_skip_the_model(self):
        clf = _clf([])
        self.assertEqual(clf.classify_batch(["lol", "gm", ""]), [None, None, None])
        self.assertEqual(clf._pipe.calls, [])

    def test_empty_input(self):
        self.assertEqual(_clf([]).classify_batch([]), [])

    def test_pipeline_exception_yields_all_none(self):
        clf = _clf([], raises=True)
        out = clf.classify_batch(["MSFT beats on earnings", "NVDA guides higher"])
        self.assertEqual(out, [None, None])

    def test_single_dict_result_normalised(self):
        clf = _clf([0.99], single_dict=True)
        out = clf.classify_batch(["MSFT beats on earnings"])
        self.assertTrue(out[0].is_news)

    def test_length_mismatch_drops_batch(self):
        clf = _clf([0.99])
        clf._pipe = MagicMock(return_value=[{"labels": ["x"], "scores": [0.9]}])
        out = clf.classify_batch(["MSFT beats on earnings", "NVDA guides higher"])
        self.assertEqual(out, [None, None])

    def test_chunks_above_max_batch(self):
        clf = _clf([0.9] * 70)
        out = clf.classify_batch([f"headline number {i} today" for i in range(70)])
        self.assertEqual(len(out), 70)
        self.assertTrue(all(isinstance(v, NewsVerdict) for v in out))
        self.assertEqual([len(c) for c in clf._pipe.calls], [64, 6])

    def test_classify_single(self):
        clf = _clf([0.95])
        self.assertTrue(clf.classify("MSFT beats on earnings").is_news)
        self.assertIsNone(clf.classify(""))

    def test_latency_recorded(self):
        clf = _clf([0.9])
        self.assertGreaterEqual(clf.classify_batch(["MSFT beats on earnings"])[0].latency_ms, 0.0)

    def test_contrast_label_softmaxed_and_matched_by_name(self):
        # Chatter: the contrast label wins, so it sorts first.  Reading
        # scores[0] would report 0.85 news instead of 0.15.
        clf = _clf([0.15])
        out = clf.classify_batch(["anyone else watching this level right now"])
        self.assertAlmostEqual(out[0].score, 0.15)
        self.assertFalse(out[0].is_news)
        self.assertEqual(clf._pipe.label_calls[0],
                         [clf.label, clf.contrast_label])
        self.assertEqual(clf._pipe.multi_label_calls, [False])

    def test_no_contrast_label_falls_back_to_single_hypothesis(self):
        clf = ModernBERTNewsClassifier(threshold=0.85, contrast_label="")
        clf._pipe = _FakePipe([0.9])
        out = clf.classify_batch(["MSFT beats on earnings"])
        self.assertTrue(out[0].is_news)
        self.assertEqual(clf._pipe.label_calls[0], [clf.label])
        self.assertEqual(clf._pipe.multi_label_calls, [True])


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self.slept.append(secs)
        self.t += secs


class TestRateLimitedSymbolResolver(unittest.TestCase):
    def _resolver(self, client, **kw):
        clock = _FakeClock()
        return RateLimitedSymbolResolver(
            client, clock=clock, sleeper=clock.sleep, **kw,
        ), clock

    def test_resolves_and_uppercases(self):
        client = MagicMock(search_symbol=MagicMock(return_value="rklb"))
        r, _ = self._resolver(client)
        self.assertEqual(r.resolve("Rocket Lab"), "RKLB")

    def test_lru_hit_skips_the_call(self):
        client = MagicMock(search_symbol=MagicMock(return_value="RKLB"))
        r, _ = self._resolver(client)
        self.assertEqual(r.resolve("Rocket"), "RKLB")
        self.assertEqual(r.resolve("rocket"), "RKLB")   # case-insensitive key
        self.assertEqual(r.resolve("  Rocket "), "RKLB")
        client.search_symbol.assert_called_once()

    def test_negative_result_is_cached(self):
        client = MagicMock(search_symbol=MagicMock(return_value=None))
        r, _ = self._resolver(client)
        self.assertIsNone(r.resolve("notacompany"))
        self.assertIsNone(r.resolve("notacompany"))
        client.search_symbol.assert_called_once()

    def test_exception_becomes_none_and_is_cached(self):
        client = MagicMock(search_symbol=MagicMock(side_effect=RuntimeError("ib down")))
        r, _ = self._resolver(client)
        self.assertIsNone(r.resolve("anything"))
        self.assertIsNone(r.resolve("anything"))
        client.search_symbol.assert_called_once()

    def test_pacing(self):
        client = MagicMock(search_symbol=MagicMock(return_value="AAA"))
        r, clock = self._resolver(client, min_interval_s=1.1)
        r.resolve("one")                 # first call — no wait
        self.assertEqual(clock.slept, [])
        r.resolve("two")                 # immediately after — full spacing
        self.assertEqual([round(s, 4) for s in clock.slept], [1.1])
        clock.t += 5.0                   # long gap — no wait needed
        r.resolve("three")
        self.assertEqual(len(clock.slept), 1)

    def test_cache_is_bounded(self):
        client = MagicMock(search_symbol=MagicMock(return_value="AAA"))
        r, _ = self._resolver(client, cache_size=3)
        for q in ("a", "b", "c", "d"):
            r.resolve(q)
        self.assertEqual(len(r._cache), 3)
        self.assertNotIn("a", r._cache)   # oldest evicted

    def test_no_client_is_a_noop(self):
        r, clock = self._resolver(None)
        self.assertIsNone(r.resolve("Rocket Lab"))
        self.assertEqual(clock.slept, [])

    def test_blank_query(self):
        client = MagicMock(search_symbol=MagicMock(return_value="AAA"))
        r, _ = self._resolver(client)
        self.assertIsNone(r.resolve(""))
        self.assertIsNone(r.resolve("   "))
        client.search_symbol.assert_not_called()


if __name__ == "__main__":
    unittest.main()
