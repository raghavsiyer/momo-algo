"""
tests/test_brave.py

Unit tests for market_data/brave_client.py.

All tests mock HTTP calls — no real API requests.
"""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")
os.environ.setdefault("BRAVE_SEARCH_API_KEY", "_test_brave_key_")

import unittest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from market_data.brave_client import BraveNewsClient


# ---------------------------------------------------------------------------
# Sample Brave API response
# ---------------------------------------------------------------------------

_BRAVE_RESP = {
    "type": "news",
    "query": {"original": "NASDAQ: MBOT"},
    "results": [
        {
            "title": "Microbot Medical (MBOT) price target raised to $7.65",
            "url": "https://example.com/mbot-1",
            "description": "Analyst raises PT on MBOT after strong Q1",
            "page_age": "2026-04-07T18:30:00+00:00",
            "meta_url": {"hostname": "seekingalpha.com"},
        },
        {
            "title": "Top Small Cap Stocks to Watch Monday",
            "url": "https://example.com/roundup",
            "description": "A roundup of movers",
            "page_age": "2026-04-07T16:00:00+00:00",
            "meta_url": {"hostname": "stocknews.com"},
        },
        {
            "title": "MBOT FDA breakthrough — shares surge",
            "url": "https://example.com/mbot-2",
            "description": "FDA grants breakthrough designation",
            "page_age": "2026-04-07T14:00:00+00:00",
            "meta_url": {"hostname": "benzinga.com"},
        },
    ],
}

_BRAVE_EMPTY = {
    "type": "news",
    "query": {"original": "NASDAQ: ZZZZZ"},
    "results": [],
}

_BRAVE_NO_RESULTS_KEY = {
    "type": "news",
    "query": {"original": "NASDAQ: ZZZZZ"},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBraveNewsClient(unittest.TestCase):

    # --- basic search ---

    @patch("requests.Session.get")
    def test_search_returns_articles(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = _BRAVE_RESP
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("MBOT")

        self.assertEqual(len(articles), 3)
        # Sorted newest-first
        self.assertGreaterEqual(articles[0].published_at, articles[1].published_at)
        self.assertGreaterEqual(articles[1].published_at, articles[2].published_at)

    @patch("requests.Session.get")
    def test_article_fields(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = _BRAVE_RESP
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("MBOT")

        top = articles[0]
        self.assertEqual(top.symbol, "MBOT")
        self.assertIn("price target", top.title.lower())
        self.assertEqual(top.source, "seekingalpha.com")
        self.assertTrue(top.url.startswith("https://"))
        self.assertIsNotNone(top.published_at)
        self.assertIsNone(top.entity_sentiment)

    @patch("requests.Session.get")
    def test_query_format_with_exchange(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = _BRAVE_EMPTY
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        client.search_ticker_news("tsla", exchange="NASDAQ")

        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["q"], "NASDAQ: TSLA")
        self.assertEqual(kwargs["params"]["freshness"], "pd")

    @patch("requests.Session.get")
    def test_query_format_no_exchange(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = _BRAVE_EMPTY
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        client.search_ticker_news("F")

        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["q"], "F stock news")

    @patch("requests.Session.get")
    def test_query_format_nyse(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = _BRAVE_EMPTY
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        client.search_ticker_news("F", exchange="NYSE")

        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["q"], "NYSE: F")

    @patch("requests.Session.get")
    def test_auth_header_sent(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = _BRAVE_EMPTY
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        client.search_ticker_news("AAPL")

        # Headers are set on the session
        self.assertIn("X-Subscription-Token", client._session.headers)
        self.assertTrue(len(client._session.headers["X-Subscription-Token"]) > 0)

    # --- empty / missing results ---

    @patch("requests.Session.get")
    def test_empty_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = _BRAVE_EMPTY
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("ZZZZZ")
        self.assertEqual(articles, [])

    @patch("requests.Session.get")
    def test_no_results_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = _BRAVE_NO_RESULTS_KEY
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("ZZZZZ")
        self.assertEqual(articles, [])

    # --- error handling ---

    @patch("requests.Session.get")
    def test_http_error_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.text = "Rate limited"
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("AAPL")
        self.assertEqual(articles, [])

    @patch("requests.Session.get")
    def test_network_exception_returns_empty(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")

        client = BraveNewsClient()
        articles = client.search_ticker_news("AAPL")
        self.assertEqual(articles, [])

    def test_no_api_key_returns_empty(self):
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}):
            client = BraveNewsClient()
            articles = client.search_ticker_news("AAPL")
            self.assertEqual(articles, [])

    # --- timestamp parsing ---

    @patch("requests.Session.get")
    def test_page_age_parsed(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "results": [{
                "title": "Test",
                "url": "https://example.com",
                "page_age": "2026-04-07T18:30:00+00:00",
                "meta_url": {"hostname": "example.com"},
            }]
        }
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("TEST")
        self.assertEqual(articles[0].published_at.year, 2026)
        self.assertEqual(articles[0].published_at.month, 4)
        self.assertEqual(articles[0].published_at.hour, 18)

    @patch("requests.Session.get")
    def test_fallback_to_page_fetched(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "results": [{
                "title": "Test",
                "url": "https://example.com",
                "page_fetched": "2026-04-06T10:00:00Z",
                "meta_url": {"hostname": "example.com"},
            }]
        }
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("TEST")
        self.assertEqual(articles[0].published_at.day, 6)

    @patch("requests.Session.get")
    def test_no_timestamp_uses_now(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "results": [{
                "title": "No date",
                "url": "https://example.com",
                "meta_url": {"hostname": "example.com"},
            }]
        }
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("TEST")
        self.assertEqual(articles[0].published_at.year, 2000)

    # --- meta_url edge cases ---

    @patch("requests.Session.get")
    def test_missing_meta_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "results": [{
                "title": "No meta",
                "url": "https://example.com",
                "page_age": "2026-04-07T12:00:00Z",
            }]
        }
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("TEST")
        self.assertEqual(articles[0].source, "")

    @patch("requests.Session.get")
    def test_null_meta_url(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "results": [{
                "title": "Null meta",
                "url": "https://example.com",
                "page_age": "2026-04-07T12:00:00Z",
                "meta_url": None,
            }]
        }
        mock_get.return_value = mock_resp

        client = BraveNewsClient()
        articles = client.search_ticker_news("TEST")
        self.assertEqual(articles[0].source, "")


if __name__ == "__main__":
    unittest.main()
