"""Groq LLM client — wrapper over the Groq chat completions API.

Supports both single-shot generation and multi-turn chat (for the agent loop).
Includes:
- Proactive self-throttling to prevent 429 rate-limit errors
- Automatic retry with backoff when rate-limited
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

_TIMEOUT = 30.0  # seconds
_MAX_RETRIES = 5  # max retry attempts for 429 rate-limit errors
_DEFAULT_RETRY_DELAY = 1.0  # fallback delay (seconds) if no Retry-After header
_MIN_REQUEST_GAP = 1.0  # minimum seconds between API calls (self-throttle)
# With 2 keys, each key gets called every 2s which is safe for Groq free tier


class GroqClient:
    """Synchronous client for the Groq chat completions endpoint.

    Includes a proactive self-throttle that enforces a minimum gap
    between consecutive API calls, preventing most 429 errors before
    they happen. This is transparent to all callers.
    """

    def __init__(self, settings: Settings) -> None:
        self._api_keys: list[str] = list(settings.groq_api_keys) or [settings.groq_api_key]
        self._key_index: int = 0
        self._model: str = settings.groq_model
        self._url: str = settings.groq_api_url
        self._last_request_time: float = 0.0
        logger.info("GroqClient initialized with %d API key(s)", len(self._api_keys))

    # ----- public API -----

    def generate_response(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Single-shot: send one prompt and get a response.

        Convenience wrapper around chat() for simple cases.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        """Multi-turn chat: send a full message history and get a response.

        This is the core method used by the agent loop. Each iteration
        appends to the message history and calls this again.

        Includes:
        - Proactive self-throttle to space out requests
        - Automatic retry with backoff for 429 rate-limit errors

        Args:
            messages: List of {"role": ..., "content": ...} message dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            json_mode: If True, force the model to output valid JSON.

        Returns:
            The assistant's text response.

        Raises:
            RuntimeError: On network or API errors (after retries exhausted).
        """
        # ── Proactive throttle: wait if calling too fast ──
        self._throttle()

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._next_key()}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                self._last_request_time = time.monotonic()
                response = httpx.post(
                    self._url,
                    json=payload,
                    headers=headers,
                    timeout=_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                return self._extract_content(data)

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                body = exc.response.text

                if status == 429 and attempt < _MAX_RETRIES:
                    # Rotate to next key immediately on rate limit
                    if len(self._api_keys) > 1:
                        headers["Authorization"] = f"Bearer {self._next_key()}"
                        logger.warning(
                            "Rate limited (429) on attempt %d/%d — "
                            "rotating to next API key",
                            attempt, _MAX_RETRIES,
                        )
                        # Brief pause then retry with new key
                        time.sleep(0.5)
                    else:
                        delay = self._parse_retry_delay(body, attempt)
                        logger.warning(
                            "Rate limited (429) on attempt %d/%d — "
                            "retrying in %.1fs",
                            attempt, _MAX_RETRIES, delay,
                        )
                        time.sleep(delay)
                    last_exc = exc
                    continue

                logger.error(
                    "Groq API returned %s: %s", status, body,
                )
                raise RuntimeError(
                    f"Groq API error (HTTP {status})"
                ) from exc

            except httpx.RequestError as exc:
                logger.error("Network error contacting Groq: %s", exc)
                raise RuntimeError("Failed to reach Groq API") from exc

        # Should not reach here, but just in case
        raise RuntimeError(
            f"Groq API rate limit exceeded after {_MAX_RETRIES} retries"
        ) from last_exc

    # ----- helpers -----

    def _next_key(self) -> str:
        """Return the next API key in round-robin order."""
        key = self._api_keys[self._key_index % len(self._api_keys)]
        self._key_index += 1
        return key

    def _throttle(self) -> None:
        """Enforce minimum gap between API calls to prevent rate limiting.

        If the last request was made less than _MIN_REQUEST_GAP seconds ago,
        sleep for the remaining time. This is proactive — it prevents 429s
        rather than reacting to them.
        """
        if self._last_request_time > 0:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < _MIN_REQUEST_GAP:
                wait = _MIN_REQUEST_GAP - elapsed
                logger.debug("Throttling: waiting %.1fs before next request", wait)
                time.sleep(wait)

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        """Pull the assistant message text from the API response."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected Groq response shape: %s", data)
            raise RuntimeError("Unexpected response from Groq API") from exc

    @staticmethod
    def _parse_retry_delay(body: str, attempt: int) -> float:
        """Extract retry delay from Groq's 429 error message.

        Groq includes text like "Please try again in 470ms" or
        "Please try again in 1.2s" in the error body.
        Falls back to exponential backoff if parsing fails.
        """
        # Try to extract "in Xms" or "in Xs" from the error message
        match = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", body, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            delay = value / 1000.0 if unit == "ms" else value
            # Add a small buffer to avoid hitting the limit again
            return delay + 0.5

        # Fallback: exponential backoff
        return _DEFAULT_RETRY_DELAY * (2 ** (attempt - 1))
