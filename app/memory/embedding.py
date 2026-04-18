"""Embedding generation for semantic memory retrieval.

Uses sentence-transformers to produce dense vector embeddings.
The model is loaded lazily on first call to avoid startup overhead.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # output dimension of all-MiniLM-L6-v2

# ── Lazy model loading ────────────────────────────────────────────────────

_model: Any = None
_model_lock = threading.Lock()


def is_available() -> bool:
    """Check whether sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model() -> Any:
    """Load the sentence-transformer model (thread-safe, lazy)."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None:
            return _model

        from sentence_transformers import SentenceTransformer

        logger.info("[EMBEDDING] Loading model: %s …", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("[EMBEDDING] Model loaded (dim=%d)", EMBEDDING_DIM)
        return _model


def get_embedding(text: str) -> list[float]:
    """Generate a dense vector embedding for the given text.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats (length == EMBEDDING_DIM).

    Raises:
        ValueError: If the input text is empty.
        RuntimeError: If encoding fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate embedding for empty text")

    try:
        model = _load_model()
        vector = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return vector.tolist()
    except Exception as exc:
        logger.error("[EMBEDDING] Encoding failed: %s", exc)
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc
