"""Web search tool using the ddgs library for real results.

Uses an adaptive search strategy:
- Recency queries ("latest", "recent", etc.) → news search first, then text
- All queries → merge and deduplicate both sources for better coverage
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ddgs import DDGS

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5

# Keywords that indicate the user wants recent/current information
_RECENCY_KEYWORDS = re.compile(
    r"\b(latest|recent|new|current|today|trending|breaking|"
    r"this week|this month|this year|2024|2025|2026)\b",
    re.IGNORECASE,
)


class SearchTool(BaseTool):
    """Perform a web search via DuckDuckGo and return a summary of results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current, real-time information. "
            "Use this when the user asks about recent events, news, "
            "live data, or anything you don't have reliable knowledge about."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web",
            }
        }

    def run(self, **kwargs: Any) -> str:
        """Query DuckDuckGo and format the response.

        Uses adaptive strategy:
        - Recency queries → news first, supplement with text
        - Other queries → text first, supplement with news

        Args:
            **kwargs: Must include 'query' (str).

        Returns:
            A formatted string of search results, or an error message.
        """
        query = kwargs.get("query", "")
        if not query:
            return "Error: 'query' parameter is required."

        logger.info("SearchTool running query: %s", query)
        is_recency = bool(_RECENCY_KEYWORDS.search(query))

        try:
            if is_recency:
                # Recency queries: news first (actual articles), text as supplement
                logger.info("Recency query detected — prioritizing news search")
                primary = self._news_search(query)
                secondary = self._text_search(query)
                results = self._merge_results(
                    primary, secondary,
                    primary_label="news", secondary_label="web",
                    max_total=_MAX_RESULTS,
                )
            else:
                # Standard queries: text first, news as supplement
                primary = self._text_search(query)
                secondary = self._news_search(query)
                results = self._merge_results(
                    primary, secondary,
                    primary_label="web", secondary_label="news",
                    max_total=_MAX_RESULTS,
                )

            if not results:
                logger.warning("No results returned for query: %s", query)
                return f"No search results found for: '{query}'"

            return self._format_results(results)

        except Exception as exc:
            logger.error("Search failed: %s", exc, exc_info=True)
            return f"Search tool error: {exc}"

    # ----- search helpers -----

    @staticmethod
    def _text_search(query: str) -> list[dict]:
        """Run a DuckDuckGo text search."""
        ddgs = DDGS()
        try:
            return list(ddgs.text(query, max_results=_MAX_RESULTS))
        except Exception as exc:
            logger.warning("Text search error: %s", exc)
            return []

    @staticmethod
    def _news_search(query: str) -> list[dict]:
        """Run a DuckDuckGo news search."""
        ddgs = DDGS()
        try:
            return list(ddgs.news(query, max_results=_MAX_RESULTS))
        except Exception as exc:
            logger.warning("News search error: %s", exc)
            return []

    @staticmethod
    def _merge_results(
        primary: list[dict],
        secondary: list[dict],
        primary_label: str = "web",
        secondary_label: str = "web",
        max_total: int = 5,
    ) -> list[dict]:
        """Merge and deduplicate results from two sources.

        Primary results come first, then secondary results that have
        unique URLs. Each result is tagged with its source type
        (_search_type) so the LLM can distinguish quality.
        """
        merged: list[dict] = []
        seen_urls: set[str] = set()

        for result, label in (
            *((r, primary_label) for r in primary),
            *((r, secondary_label) for r in secondary),
        ):
            if len(merged) >= max_total:
                break
            url = result.get("href", result.get("url", ""))
            # Normalize URL for dedup
            url_key = url.strip().rstrip("/").lower()
            if url_key and url_key in seen_urls:
                continue
            if url_key:
                seen_urls.add(url_key)
            result["_search_type"] = label
            merged.append(result)

        return merged

    # ----- formatting -----

    @staticmethod
    def _format_results(results: list[dict]) -> str:
        """Format search results into a readable string."""
        parts: list[str] = []

        for idx, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            body = result.get("body", result.get("excerpt", "No description"))
            href = result.get("href", result.get("url", ""))
            source = result.get("source", "")
            search_type = result.get("_search_type", "web").upper()

            header = f"[Result {idx}] [{search_type}] {title}"
            if source:
                header += f" ({source})"

            parts.append(f"{header}\n{body}\nSource: {href}")

        return "\n\n".join(parts)
