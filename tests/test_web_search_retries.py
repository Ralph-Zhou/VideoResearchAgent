"""Offline tests for Serper web-search retry behavior."""

import unittest
from unittest.mock import Mock, patch

import httpx

from video_agent.tools.web_search import WebSearchTool


class WebSearchRetryTests(unittest.TestCase):
    def _tool(self, max_retries: int = 3) -> WebSearchTool:
        return WebSearchTool(
            provider="serper",
            serper_api_key="test-key",
            serper_max_retries=max_retries,
            serper_retry_backoff_sec=0,
        )

    @patch("video_agent.tools.web_search.httpx.Client")
    def test_retries_transient_errors_then_returns_results(self, client_cls):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "organic": [
                {
                    "title": "result",
                    "link": "https://example.com",
                    "snippet": "snippet",
                }
            ]
        }
        post = client_cls.return_value.__enter__.return_value.post
        post.side_effect = [
            httpx.ReadTimeout("timed out"),
            httpx.ConnectError("connection failed"),
            response,
        ]

        results = self._tool().search("test query", max_results=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["url"], "https://example.com")
        self.assertEqual(post.call_count, 3)

    @patch("video_agent.tools.web_search.httpx.Client")
    def test_stops_after_configured_number_of_attempts(self, client_cls):
        post = client_cls.return_value.__enter__.return_value.post
        post.side_effect = httpx.ReadTimeout("timed out")

        results = self._tool(max_retries=5).search("test query")

        self.assertEqual(results, [])
        self.assertEqual(post.call_count, 5)


if __name__ == "__main__":
    unittest.main()
