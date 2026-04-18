"""Dictionary tool — word definitions via the Free Dictionary API.

No API key required. Returns definitions, part of speech, and examples.
API: https://dictionaryapi.dev/
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_API_URL = "https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
_TIMEOUT = 10.0


class DictionaryTool(BaseTool):
    """Look up word definitions, pronunciation, and examples."""

    @property
    def name(self) -> str:
        return "dictionary"

    @property
    def description(self) -> str:
        return (
            "Look up the definition of an English word. Returns meanings, "
            "part of speech, examples, and synonyms. "
            "Use for: 'define X', 'what does X mean', 'synonym of X'."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "word": {
                "type": "string",
                "description": "The English word to look up.",
            }
        }

    def run(self, **kwargs: Any) -> str:
        word = kwargs.get("word", "").strip().lower()
        if not word:
            return "Error: No word provided."

        logger.info("[DICTIONARY] Looking up: %s", word)

        try:
            resp = httpx.get(
                _API_URL.format(word=word),
                timeout=_TIMEOUT,
                follow_redirects=True,
            )

            if resp.status_code == 404:
                return f"No definition found for '{word}'."
            if resp.status_code != 200:
                return f"Dictionary API error (HTTP {resp.status_code})."

            data = resp.json()
            if not data or not isinstance(data, list):
                return f"No definition found for '{word}'."

            return self._format(word, data[0])

        except Exception as exc:
            logger.warning("[DICTIONARY] Error: %s", exc)
            return f"Error looking up '{word}': {exc}"

    @staticmethod
    def _format(word: str, entry: dict) -> str:
        """Format the API response into a readable string."""
        parts: list[str] = [f"**{word}**"]

        # Phonetic
        phonetic = entry.get("phonetic", "")
        if phonetic:
            parts[0] += f"  ({phonetic})"

        meanings = entry.get("meanings", [])
        for meaning in meanings[:3]:  # max 3 parts of speech
            pos = meaning.get("partOfSpeech", "")
            definitions = meaning.get("definitions", [])

            if pos:
                parts.append(f"\n*{pos}*")

            for i, defn in enumerate(definitions[:2], 1):  # max 2 defs per POS
                text = defn.get("definition", "")
                example = defn.get("example", "")
                parts.append(f"  {i}. {text}")
                if example:
                    parts.append(f"     Example: \"{example}\"")

            synonyms = meaning.get("synonyms", [])[:5]
            if synonyms:
                parts.append(f"  Synonyms: {', '.join(synonyms)}")

        return "\n".join(parts)
