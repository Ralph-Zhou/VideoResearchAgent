"""Web search using the public Serper API."""

import os
import time
import logging
from typing import List, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

SERPER_API_URL_DEFAULT = "https://google.serper.dev/search"


class WebSearchTool:
    """Web search supporting the public Serper API."""

    def __init__(
        self,
        provider: str = "serper",
        serper_api_key: Optional[str] = None,
        serper_api_url: Optional[str] = None,
        serper_max_retries: int = 10,
        serper_timeout: int = 60,
        serper_retry_backoff_sec: float = 1.0,
    ):
        if provider != "serper":
            raise ValueError("Only the Serper search provider is supported")
        self.provider = provider
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.serper_api_url = (
            serper_api_url
            or os.getenv("SERPER_API_URL", SERPER_API_URL_DEFAULT)
        ).rstrip("/")
        self.serper_max_retries = max(1, serper_max_retries)
        self.serper_timeout = serper_timeout
        self.serper_retry_backoff_sec = max(0.0, serper_retry_backoff_sec)

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        if self.provider == "serper" and self.serper_api_key:
            return self._serper_search(query, max_results)
        return []


    def _serper_search(self, query: str, max_results: int) -> List[Dict]:
        url = self.serper_api_url
        headers = {"X-API-KEY": self.serper_api_key, "Content-Type": "application/json"}
        data = {"q": query, "num": max_results}

        last_error = None
        for attempt in range(1, self.serper_max_retries + 1):
            try:
                with httpx.Client() as client:
                    resp = client.post(
                        url, headers=headers, json=data, timeout=self.serper_timeout,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    if "not enough credits" in str(result).lower():
                        last_error = RuntimeError("not enough credits")
                        logger.warning(
                            "Serper: not enough credits (attempt %d/%d)",
                            attempt, self.serper_max_retries,
                        )
                        if attempt < self.serper_max_retries:
                            time.sleep(self._retry_delay(attempt, minimum=5.0))
                        continue

                    results = []
                    for r in result.get("organic", [])[:max_results]:
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("link", ""),
                            "snippet": r.get("snippet", ""),
                        })
                    logger.info("Serper search '%s' returned %d results", query, len(results))
                    return results

            except Exception as e:
                last_error = e
                if attempt < self.serper_max_retries:
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "Serper search error (attempt %d/%d): %s; "
                        "retrying in %.1fs",
                        attempt, self.serper_max_retries, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "Serper search error (attempt %d/%d): %s",
                        attempt, self.serper_max_retries, e,
                    )

        logger.error("Serper search failed for '%s': %s", query, last_error)
        return []

    def _retry_delay(self, attempt: int, minimum: float = 0.0) -> float:
        """Return a capped exponential delay after a failed attempt."""
        delay = self.serper_retry_backoff_sec * (2 ** (attempt - 1))
        return max(minimum, min(delay, 10.0))

