"""Web reader tool — fetches and extracts clean content from web pages.

Uses a lightweight extraction pipeline:
1. HTTP fetch via httpx (fast, no browser needed)
2. Content extraction via readability-lxml (Mozilla's Readability algorithm)
3. Fallback to BeautifulSoup heuristic extraction
4. Text cleaning: dedup, noise removal, fact extraction

Returns concise bullet-point facts extracted from the page content.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_TIMEOUT = 10.0               # seconds
_MAX_CONTENT_BYTES = 500_000  # reject very large pages (500 KB)
_MAX_TEXT_CHARS = 15_000      # truncate extracted text before processing
_MAX_FACTS = 10               # max bullet-point facts to return
_MIN_PARAGRAPH_LENGTH = 40    # chars — skip short/noisy lines
_MIN_FACT_WORDS = 5           # words — minimum for a meaningful fact

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

_BLOCKED_EXTENSIONS = frozenset({
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".mp4", ".mp3", ".wav", ".zip", ".tar", ".gz", ".rar",
    ".exe", ".dmg", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
})


# ── Tool class ─────────────────────────────────────────────────────────────

class WebReaderTool(BaseTool):
    """Fetch a web page and extract its main content as clean text."""

    @property
    def name(self) -> str:
        return "web_reader"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page URL and extract its main article content as clean text. "
            "Use this AFTER web_search to read the full content of a promising result. "
            "Only use for 1–2 URLs per query — do not over-fetch. "
            "Input must be a valid HTTP/HTTPS URL."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "url": {
                "type": "string",
                "description": (
                    "The full URL of the web page to read "
                    "(must start with http:// or https://)"
                ),
            }
        }

    def run(self, **kwargs: Any) -> str:
        """Fetch URL → extract content → return bullet-point facts."""

        url = kwargs.get("url", "")
        if not url:
            return "Error: 'url' parameter is required."

        # ── Validate URL ──
        if not url.startswith(("http://", "https://")):
            return f"Error: Invalid URL — must start with http:// or https://. Got: {url}"

        # Block non-HTML resources
        lower_url = url.lower().split("?")[0]
        if any(lower_url.endswith(ext) for ext in _BLOCKED_EXTENSIONS):
            return f"Error: URL points to a non-HTML resource: {url}"

        logger.info("[WebReader] Fetching: %s", url)

        # ── Fetch ──
        try:
            html = self._fetch(url)
        except Exception as exc:
            logger.error("[WebReader] Fetch failed for %s: %s", url, exc)
            return f"Error: Failed to fetch page — {exc}"

        if not html or len(html.strip()) < 100:
            return "Error: Page returned empty or minimal content."

        # ── Extract main content ──
        text = self._extract_content(html, url)
        if not text or len(text.strip()) < 50:
            return f"Error: Could not extract meaningful content from {url}"

        # ── Clean and extract facts ──
        facts = self._extract_facts(text)
        if not facts:
            return f"Error: No meaningful facts could be extracted from {url}"

        # ── Format output ──
        header = f"Content extracted from: {url}\n"
        body = "\n".join(f"- {fact}" for fact in facts)
        result = header + body

        logger.info(
            "[WebReader] Extracted %d facts (%d chars) from %s",
            len(facts), len(result), url,
        )
        return result

    # ── Fetch ──────────────────────────────────────────────────────────

    @staticmethod
    def _fetch(url: str) -> str:
        """Fetch page HTML via httpx with timeout and size limits."""
        headers = {
            "User-Agent": _USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        with httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()

            # Verify content-type is HTML
            content_type = response.headers.get("content-type", "")
            if (
                "text/html" not in content_type
                and "application/xhtml" not in content_type
            ):
                raise ValueError(f"Non-HTML content type: {content_type}")

            # Reject oversized pages
            if len(response.content) > _MAX_CONTENT_BYTES:
                raise ValueError(
                    f"Page too large: {len(response.content):,} bytes "
                    f"(limit {_MAX_CONTENT_BYTES:,})"
                )

            return response.text

    # ── Content extraction ─────────────────────────────────────────────

    @staticmethod
    def _extract_content(html: str, url: str) -> str:
        """Extract main article text.

        Strategy 1: readability-lxml (Mozilla Readability algorithm)
        Strategy 2: BeautifulSoup heuristic extraction
        """
        # Strategy 1: readability-lxml
        text = WebReaderTool._try_readability(html, url)
        if text and len(text.strip()) > 100:
            logger.info("[WebReader] Extracted via readability (%d chars)", len(text))
            return text[:_MAX_TEXT_CHARS]

        # Strategy 2: BeautifulSoup heuristics
        text = WebReaderTool._try_beautifulsoup(html)
        if text and len(text.strip()) > 100:
            logger.info("[WebReader] Extracted via BeautifulSoup (%d chars)", len(text))
            return text[:_MAX_TEXT_CHARS]

        logger.warning(
            "[WebReader] Both extraction methods returned insufficient content"
        )
        return ""

    @staticmethod
    def _try_readability(html: str, url: str) -> str:
        """Extract using readability-lxml (Mozilla Readability algorithm)."""
        try:
            from readability import Document  # type: ignore[import-untyped]
            from bs4 import BeautifulSoup

            doc = Document(html, url=url)
            summary_html = doc.summary()

            # Convert the summary HTML fragment to plain text
            soup = BeautifulSoup(summary_html, "lxml")
            return soup.get_text(separator="\n", strip=True)

        except ImportError:
            logger.debug("[WebReader] readability-lxml not installed — skipping")
            return ""
        except Exception as exc:
            logger.warning("[WebReader] Readability extraction failed: %s", exc)
            return ""

    @staticmethod
    def _try_beautifulsoup(html: str) -> str:
        """Extract using BeautifulSoup heuristics — find largest text block."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")

            # Remove non-content elements
            for tag_name in (
                "script", "style", "nav", "header", "footer",
                "aside", "iframe", "noscript", "form",
                "button", "input", "select", "textarea",
            ):
                for tag in soup.find_all(tag_name):
                    tag.decompose()

            # Remove elements with common non-content classes / IDs
            _noise_re = re.compile(
                r"(nav|menu|sidebar|footer|header|banner|ad[sv]?|social|"
                r"share|comment|cookie|popup|modal|newsletter|subscribe|"
                r"related|widget|breadcrumb)",
                re.IGNORECASE,
            )
            for tag in soup.find_all(True):
                classes = " ".join(tag.get("class", []))
                tag_id = tag.get("id", "")
                if _noise_re.search(classes) or _noise_re.search(tag_id):
                    tag.decompose()

            # Try common article containers first
            article = (
                soup.find("article")
                or soup.find("main")
                or soup.find(attrs={"role": "main"})
                or soup.find(
                    "div",
                    class_=re.compile(
                        r"(article|content|post|entry|story)", re.I
                    ),
                )
            )
            if article:
                text = article.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text

            # Fallback: div/section with the most paragraph text
            best_text = ""
            for container in soup.find_all(["div", "section"]):
                paragraphs = container.find_all("p")
                if len(paragraphs) >= 3:
                    text = "\n".join(
                        p.get_text(strip=True) for p in paragraphs
                    )
                    if len(text) > len(best_text):
                        best_text = text

            if best_text:
                return best_text

            # Last resort: all <p> elements
            all_p = soup.find_all("p")
            if all_p:
                return "\n".join(
                    p.get_text(strip=True)
                    for p in all_p
                    if len(p.get_text(strip=True)) > 20
                )

            return soup.get_text(separator="\n", strip=True)

        except ImportError:
            logger.error("[WebReader] beautifulsoup4 not installed")
            return ""
        except Exception as exc:
            logger.warning("[WebReader] BeautifulSoup extraction failed: %s", exc)
            return ""

    # ── Fact extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_facts(text: str) -> list[str]:
        """Clean text and extract concise bullet-point facts."""
        lines = text.split("\n")

        # Phase 1: Clean lines — dedup, remove noise
        cleaned: list[str] = []
        seen: set[str] = set()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(line) < _MIN_PARAGRAPH_LENGTH:
                continue
            if len(line.split()) < _MIN_FACT_WORDS:
                continue

            # Skip duplicate / near-duplicate lines
            normalised = re.sub(r"\s+", " ", line.lower())
            if normalised in seen:
                continue
            seen.add(normalised)

            # Skip obvious noise
            if WebReaderTool._is_noise(line):
                continue

            cleaned.append(line)

        # Phase 2: Score and select top facts
        scored: list[tuple[int, str]] = []
        for line in cleaned:
            score = 0
            # Bonus for numbers / dates
            if re.search(r"\d", line):
                score += 2
            # Bonus for proper nouns
            if re.search(r"[A-Z][a-z]{2,}", line):
                score += 1
            # Bonus for factual language
            if re.search(
                r"(?:according|reported|announced|published|"
                r"launched|released|percent|%|\$)",
                line,
                re.I,
            ):
                score += 2
            # Bonus for longer substantive lines
            if len(line) > 100:
                score += 1
            scored.append((score, line))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top facts, truncate overly long ones
        facts: list[str] = []
        for _, line in scored[:_MAX_FACTS]:
            if len(line) > 300:
                # Truncate at the nearest sentence boundary
                sentences = re.split(r"(?<=[.!?])\s+", line)
                truncated = ""
                for s in sentences:
                    if len(truncated) + len(s) > 280:
                        break
                    truncated += s + " "
                line = truncated.strip()
                if not line.endswith((".", "!", "?")):
                    line += "…"
            facts.append(line)

        return facts

    # ── Noise detection ────────────────────────────────────────────────

    @staticmethod
    def _is_noise(line: str) -> bool:
        """Return True if the line is likely navigation/UI noise."""
        lower = line.lower()

        noise_starters = (
            "cookie", "subscribe", "sign up", "log in", "sign in",
            "follow us", "share this", "read more", "click here",
            "advertisement", "sponsored", "all rights reserved",
            "privacy policy", "terms of", "copyright",
            "skip to", "toggle", "menu", "search for",
            "accept all", "reject all", "manage preferences",
        )
        if any(lower.startswith(n) for n in noise_starters):
            return True

        # Too many special characters → likely UI elements
        special_ratio = (
            sum(1 for c in line if not c.isalnum() and c != " ")
            / max(len(line), 1)
        )
        if special_ratio > 0.3:
            return True

        return False
