"""
tests/test_alpha_vantage.py

Unit tests for market_data/alpha_vantage_client.py.

All tests mock HTTP calls -- no real API requests.
"""

import os
os.environ.setdefault("TRADIER_API_KEY", "_test_")
os.environ.setdefault("FMP_API_KEY", "_test_")
os.environ.setdefault("FINNHUB_API_KEY", "_test_")
os.environ.setdefault("AV_API_KEY", "_test_av_key_")

import unittest
from unittest.mock import MagicMock, patch

import config
from market_data.alpha_vantage_client import AlphaVantageClient


class TestAlphaVantageClient(unittest.TestCase):
    def _client(self) -> AlphaVantageClient:
        with (
            patch.object(config, "AV_API_KEY", "_test_av_key_"),
            patch.object(config, "AV_CALLS_PER_MIN", 5),
        ):
            return AlphaVantageClient()

    def test_information_response_returns_empty_and_logs_redacted_message(self):
        client = self._client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "Information": (
                "We have detected your API key as _test_av_key_ and our standard "
                "API rate limit is 25 requests per day."
            )
        }
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        with self.assertLogs("market_data.alpha_vantage_client", level="WARNING") as logs:
            result = client._get({"function": "OVERVIEW", "symbol": "IBM"})

        log_output = "\n".join(logs.output)
        self.assertEqual(result, {})
        self.assertIn("AV API issue", log_output)
        self.assertIn("<redacted>", log_output)
        self.assertNotIn("_test_av_key_", log_output)


if __name__ == "__main__":
    unittest.main()