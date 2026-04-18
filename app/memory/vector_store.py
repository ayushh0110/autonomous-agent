"""Vector store — semantic similarity index for memory retrieval.

Provides fast nearest-neighbour search over embedding vectors.

Backend strategy:
    1. FAISS (preferred) — fast, production-grade
    2. NumPy cosine similarity (fallback) — zero extra deps
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# ── Backend detection ──────────────────────────────────────────────────────

try:
    import faiss  # type: ignore[import-untyped]
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


# ── VectorStore ────────────────────────────────────────────────────────────

class VectorStore:
    """Thread-safe vector index with add / search / remove support.

    Uses FAISS when available; falls back to brute-force NumPy cosine
    similarity for environments where FAISS cannot be installed.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self._lock = threading.Lock()

        if _HAS_FAISS:
            base = faiss.IndexFlatIP(dimension)
            self._index: faiss.IndexIDMap = faiss.IndexIDMap(base)
            self._backend = "faiss"
        else:
            self._vectors: dict[int, np.ndarray] = {}
            self._backend = "numpy"

        logger.info(
            "[VECTOR] Initialised (dim=%d, backend=%s)",
            dimension, self._backend,
        )

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Return the active backend name."""
        return self._backend

    @property
    def size(self) -> int:
        """Current number of indexed vectors."""
        with self._lock:
            if self._backend == "faiss":
                return self._index.ntotal
            return len(self._vectors)

    # ── Public API ─────────────────────────────────────────────────────

    def add(self, embedding: list[float], entry_id: int) -> None:
        """Index an embedding under the given *entry_id*."""
        vec = self._to_unit_vec(embedding)

        with self._lock:
            if self._backend == "faiss":
                ids = np.array([entry_id], dtype=np.int64)
                self._index.add_with_ids(vec, ids)
            else:
                self._vectors[entry_id] = vec.flatten()

        logger.debug("[VECTOR] Added entry_id=%d (total=%d)", entry_id, self.size)

    def remove(self, entry_id: int) -> None:
        """Remove an embedding by *entry_id*."""
        with self._lock:
            if self._backend == "faiss":
                ids = np.array([entry_id], dtype=np.int64)
                try:
                    self._index.remove_ids(ids)
                except Exception as exc:
                    logger.warning(
                        "[VECTOR] FAISS remove failed for id=%d: %s",
                        entry_id, exc,
                    )
            else:
                self._vectors.pop(entry_id, None)

        logger.debug("[VECTOR] Removed entry_id=%d", entry_id)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return the *top_k* most similar entries.

        Returns:
            List of ``(entry_id, similarity_score)`` sorted descending.
        """
        vec = self._to_unit_vec(query_embedding)

        with self._lock:
            if self._backend == "faiss":
                return self._search_faiss(vec, top_k)
            return self._search_numpy(vec, top_k)

    def clear(self) -> None:
        """Drop all indexed vectors."""
        with self._lock:
            if self._backend == "faiss":
                base = faiss.IndexFlatIP(self._dimension)
                self._index = faiss.IndexIDMap(base)
            else:
                self._vectors.clear()
        logger.info("[VECTOR] Index cleared")

    # ── Internals ──────────────────────────────────────────────────────

    def _to_unit_vec(self, embedding: list[float]) -> np.ndarray:
        """Convert to float32 row vector and L2-normalise."""
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _search_faiss(
        self, query_vec: np.ndarray, top_k: int,
    ) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        scores, ids = self._index.search(query_vec, k)
        return [
            (int(eid), float(score))
            for score, eid in zip(scores[0], ids[0])
            if eid >= 0
        ]

    def _search_numpy(
        self, query_vec: np.ndarray, top_k: int,
    ) -> list[tuple[int, float]]:
        if not self._vectors:
            return []
        q = query_vec.flatten()
        scored = [
            (eid, float(np.dot(q, vec)))
            for eid, vec in self._vectors.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
