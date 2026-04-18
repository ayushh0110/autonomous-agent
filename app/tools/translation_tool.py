"""Translation tool — translate text using the free MyMemory API.

No API key required. Supports 200+ languages.
Free tier: 1000 words/day (anonymous), 10000/day (with email).
API: https://mymemory.translated.net/doc/spec.php
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_API_URL = "https://api.mymemory.translated.net/get"
_TIMEOUT = 10.0

# Common language name → code mapping
_LANG_CODES: dict[str, str] = {
    "english": "en", "spanish": "es", "french": "fr",
    "german": "de", "italian": "it", "portuguese": "pt",
    "dutch": "nl", "russian": "ru", "chinese": "zh-CN",
    "japanese": "ja", "korean": "ko", "arabic": "ar",
    "hindi": "hi", "turkish": "tr", "polish": "pl",
    "swedish": "sv", "danish": "da", "norwegian": "no",
    "finnish": "fi", "greek": "el", "czech": "cs",
    "romanian": "ro", "hungarian": "hu", "thai": "th",
    "vietnamese": "vi", "indonesian": "id", "malay": "ms",
    "tagalog": "tl", "filipino": "tl", "swahili": "sw",
    "hebrew": "he", "persian": "fa", "farsi": "fa",
    "urdu": "ur", "bengali": "bn", "tamil": "ta",
    "telugu": "te", "marathi": "mr", "gujarati": "gu",
    "punjabi": "pa", "kannada": "kn", "malayalam": "ml",
    "ukrainian": "uk", "croatian": "hr", "serbian": "sr",
    "bulgarian": "bg", "slovak": "sk", "slovenian": "sl",
    "estonian": "et", "latvian": "lv", "lithuanian": "lt",
    "catalan": "ca", "basque": "eu", "galician": "gl",
    "icelandic": "is", "irish": "ga", "welsh": "cy",
    "scots gaelic": "gd", "afrikaans": "af",
    "mandarin": "zh-CN", "cantonese": "zh-TW",
    "simplified chinese": "zh-CN", "traditional chinese": "zh-TW",
    "brazilian portuguese": "pt-BR",
    "latin american spanish": "es-419",
}


class TranslationTool(BaseTool):
    """Translate text between languages using MyMemory API."""

    @property
    def name(self) -> str:
        return "translate"

    @property
    def description(self) -> str:
        return (
            "Translate text between languages. Supports 200+ languages. "
            "Example: text='Hello, how are you?', from_lang='english', "
            "to_lang='spanish'. You can use language names or codes "
            "(en, es, fr, de, ja, ko, hi, etc.)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "text": {
                "type": "string",
                "description": "The text to translate.",
            },
            "from_lang": {
                "type": "string",
                "description": "Source language (name or code, e.g. 'english' or 'en'). Default: 'en'.",
            },
            "to_lang": {
                "type": "string",
                "description": "Target language (name or code, e.g. 'spanish' or 'es').",
            },
        }

    def run(self, **kwargs: Any) -> str:
        text = str(kwargs.get("text", "")).strip()
        from_lang = str(kwargs.get("from_lang", "en")).strip().lower()
        to_lang = str(kwargs.get("to_lang", "")).strip().lower()

        if not text:
            return "Error: No text provided to translate."
        if not to_lang:
            return "Error: 'to_lang' is required (e.g. 'spanish', 'fr', 'japanese')."

        # Resolve language names to codes
        from_code = _LANG_CODES.get(from_lang, from_lang)
        to_code = _LANG_CODES.get(to_lang, to_lang)
        lang_pair = f"{from_code}|{to_code}"

        logger.info("[TRANSLATE] %s → %s: %s", from_code, to_code, text[:50])

        try:
            resp = httpx.get(
                _API_URL,
                params={"q": text, "langpair": lang_pair},
                timeout=_TIMEOUT,
            )

            if resp.status_code != 200:
                return f"Translation API error (HTTP {resp.status_code})."

            data = resp.json()
            response_data = data.get("responseData", {})
            translated = response_data.get("translatedText", "")
            match_quality = response_data.get("match", 0)

            if not translated or translated == text:
                return f"Could not translate from {from_lang} to {to_lang}."

            # Check for error responses
            status = data.get("responseStatus", 200)
            if status != 200:
                error_msg = data.get("responseDetails", "Unknown error")
                return f"Translation error: {error_msg}"

            result = f"**{from_lang} → {to_lang}**\n\n"
            result += f"Original: {text}\n"
            result += f"Translation: {translated}"

            if match_quality and match_quality < 0.5:
                result += "\n\n(Note: translation confidence is low)"

            return result

        except Exception as exc:
            logger.warning("[TRANSLATE] Error: %s", exc)
            return f"Error translating: {exc}"
