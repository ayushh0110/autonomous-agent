"""Wikipedia tool — fetch article summaries from the free Wikipedia API.

No API key required. Uses the MediaWiki REST API for fast, clean summaries.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
_TIMEOUT = 10.0
_HEADERS = {
    "User-Agent": "AutonomousAgent/1.0 (https://github.com/ayushh0110/autonomous-agent; educational project)",
}


class WikipediaTool(BaseTool):
    """Look up article summaries from Wikipedia."""

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return (
            "Search Wikipedia and get a summary of an article. "
            "Use for factual questions about people, places, events, concepts, "
            "science, history, etc. Input a topic or search query."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The topic or search term to look up on Wikipedia.",
            }
        }

    def run(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "").strip()
        if not query:
            return "Error: No query provided."

        logger.info("[WIKIPEDIA] Looking up: %s", query)

        try:
            # First try direct title lookup (faster)
            result = self._fetch_summary(query)
            if result:
                return result

            # Fall back to search
            title = self._search(query)
            if title:
                result = self._fetch_summary(title)
                if result:
                    return result

            return f"No Wikipedia article found for '{query}'."

        except Exception as exc:
            logger.warning("[WIKIPEDIA] Error: %s", exc)
            return f"Error looking up Wikipedia: {exc}"

    def _fetch_summary(self, title: str) -> str | None:
        """Fetch summary for a specific article title."""
        url = _SUMMARY_URL.format(title=quote(title.replace(" ", "_")))
        try:
            resp = httpx.get(url, timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True)
            if resp.status_code != 200:
                return None
            data = resp.json()
            extract = data.get("extract", "")
            page_title = data.get("title", title)
            if not extract:
                return None

            # Truncate to ~800 chars for context efficiency
            if len(extract) > 800:
                extract = extract[:800].rsplit(".", 1)[0] + "."

            return f"**{page_title}** (Wikipedia)\n\n{extract}"
        except Exception:
            return None

    def _search(self, query: str) -> str | None:
        """Search Wikipedia and return the best-matching title."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        }
        try:
            resp = httpx.get(_SEARCH_URL, params=params, timeout=_TIMEOUT, headers=_HEADERS)
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            if results:
                return results[0]["title"]
            return None
        except Exception:
            return None
