"""Tests for YouTube live-chat ticker extraction.

Focus: cashtag'd stopwords (e.g. ``$TEST``) must be filtered exactly like
bare stopwords so junk symbols never reach the scanner queue. No network
calls — all inputs are synthetic ``ChatMessage`` objects.
"""
import unittest
from datetime import datetime, timezone

from market_data.youtube_client import (
    ChatMessage,
    YouTubeLiveChatClient,
    NEWS_AUTHOR,
    extract_ticker_from_text,
)


def _msg(text: str, author: str = "viewer", mid: str = "1") -> ChatMessage:
    return ChatMessage(
        message_id=mid,
        author=author,
        text=text,
        timestamp=datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc),
    )


class TestExtractTickersStopwordCashtags(unittest.TestCase):
    def test_cashtag_stopword_filtered(self):
        # $TEST is a cashtag'd stopword — must NOT be treated as a ticker.
        counts = YouTubeLiveChatClient.extract_tickers([_msg("$TEST breaking out")])
        self.assertNotIn("TEST", counts)

    def test_real_cashtag_still_extracted(self):
        counts = YouTubeLiveChatClient.extract_tickers([_msg("$AAPL to the moon")])
        self.assertIn("AAPL", counts)

    def test_bare_stopword_still_filtered(self):
        counts = YouTubeLiveChatClient.extract_tickers([_msg("THIS IS A TEST")])
        self.assertNotIn("TEST", counts)
        self.assertNotIn("THIS", counts)


class TestExtractNewsHeadlinesStopwordCashtags(unittest.TestCase):
    def test_cashtag_stopword_headline_skipped(self):
        # News author posts "$TEST - ..." (feed test) — must be skipped.
        msgs = [_msg("$TEST - ignore me", author=NEWS_AUTHOR)]
        heads = YouTubeLiveChatClient.extract_news_headlines(msgs)
        self.assertEqual(heads, [])

    def test_real_cashtag_headline_extracted(self):
        msgs = [_msg("$AAPL - beats earnings", author=NEWS_AUTHOR)]
        heads = YouTubeLiveChatClient.extract_news_headlines(msgs)
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads[0].ticker, "AAPL")
        self.assertEqual(heads[0].headline, "beats earnings")

    def test_bare_stopword_headline_still_skipped(self):
        msgs = [_msg("TEST - ignore me", author=NEWS_AUTHOR)]
        heads = YouTubeLiveChatClient.extract_news_headlines(msgs)
        self.assertEqual(heads, [])


class TestExtractTickerFromText(unittest.TestCase):
    """The parsing head shared by the Clementine matcher and ModernBERT.

    The Clementine tests above are the regression proof for the refactor;
    these cover the function's own contract.
    """

    def test_leading_cashtag(self):
        ticker, method, rest = extract_ticker_from_text("$AAPL - beats earnings")
        self.assertEqual((ticker, method), ("AAPL", "cashtag"))
        self.assertEqual(rest, " - beats earnings")

    def test_leading_cashtag_lowercase(self):
        ticker, method, rest = extract_ticker_from_text("$rklb - wins contract")
        self.assertEqual((ticker, method), ("RKLB", "cashtag"))
        self.assertEqual(rest, " - wins contract")

    def test_prefix_with_parenthetical(self):
        ticker, method, rest = extract_ticker_from_text("AAPL (weak open) - beats earnings")
        self.assertEqual((ticker, method), ("AAPL", "prefix"))
        self.assertEqual(rest, " (weak open) - beats earnings")

    def test_prefix_lowercase(self):
        ticker, method, _ = extract_ticker_from_text("rklb - wins contract")
        self.assertEqual((ticker, method), ("RKLB", "prefix"))

    def test_prefix_requires_dash(self):
        # No dash → not a structured prefix; falls through to the bare scan,
        # which finds nothing tradeable in an all-lowercase sentence.
        ticker, method, _ = extract_ticker_from_text("rklb wins a contract today")
        self.assertIsNone(ticker)
        self.assertEqual(method, "none")

    def test_bare_ticker_midsentence(self):
        text = "big move on RKLB after the contract news"
        ticker, method, rest = extract_ticker_from_text(text)
        self.assertEqual((ticker, method), ("RKLB", "bare"))
        self.assertEqual(rest, text)

    def test_bare_cashtag_midsentence(self):
        ticker, method, _ = extract_ticker_from_text("watching $NVDA here")
        self.assertEqual((ticker, method), ("NVDA", "bare"))

    def test_stopword_cashtag_not_a_ticker(self):
        ticker, method, _ = extract_ticker_from_text("$TEST - ignore me")
        self.assertIsNone(ticker)
        self.assertEqual(method, "none")

    def test_stopword_prefix_not_a_ticker(self):
        ticker, method, _ = extract_ticker_from_text("TEST - ignore me")
        self.assertIsNone(ticker)
        self.assertEqual(method, "none")

    def test_no_ticker(self):
        ticker, method, rest = extract_ticker_from_text("lmao this chat is wild")
        self.assertIsNone(ticker)
        self.assertEqual(method, "none")
        self.assertEqual(rest, "lmao this chat is wild")

    def test_empty(self):
        self.assertEqual(extract_ticker_from_text(""), (None, "none", ""))
        self.assertEqual(extract_ticker_from_text("   "), (None, "none", ""))

    def test_symbols_outside_3_to_4_chars_never_matched(self):
        # Only 3-4 letter symbols are extractable from chat: 1-2 letters are
        # ordinary words / dollar amounts, 5 letters are outside the traded
        # universe.
        for text in ("$F - recalls trucks", "$GM - beats earnings",
                     "F - recalls trucks", "GM (strong) - beats earnings",
                     "watching $T here",
                     "$GOOGL - antitrust ruling", "GOOGL - antitrust ruling",
                     "googl (weak) - antitrust ruling", "watching $GOOGL here"):
            ticker, method, _ = extract_ticker_from_text(text)
            self.assertIsNone(ticker, text)
            self.assertEqual(method, "none", text)

    def test_out_of_band_cashtags_not_counted_as_mentions(self):
        counts = YouTubeLiveChatClient.extract_tickers(
            [_msg("$GM and $F and $GOOGL ripping")])
        self.assertEqual(counts, {})

    def test_out_of_band_symbol_headline_skipped(self):
        for text in ("$GM - beats earnings", "$GOOGL - antitrust ruling"):
            msgs = [_msg(text, author=NEWS_AUTHOR)]
            self.assertEqual(YouTubeLiveChatClient.extract_news_headlines(msgs), [], text)

    def test_bare_method_rejected_by_news_matcher(self):
        # A ticker merely mentioned mid-sentence must not become a headline,
        # even from the news author.
        msgs = [_msg("big move on RKLB today folks", author=NEWS_AUTHOR)]
        self.assertEqual(YouTubeLiveChatClient.extract_news_headlines(msgs), [])


if __name__ == "__main__":
    unittest.main()
