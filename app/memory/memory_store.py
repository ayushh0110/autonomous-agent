"""Memory store — lightweight in-memory RAG for the agent.

Stores high-quality past interactions and retrieves relevant context
for new queries using hybrid retrieval: semantic vector search (first
stage) followed by keyword-based re-ranking (second stage).

Design decisions:
- In-memory storage (no external DB) with FIFO eviction at max_entries.
- Two-stage retrieval: vector search → keyword re-rank → top_k.
- Enhanced keyword matching: Jaccard + synonym expansion + substring
  boost + summary weighting + time-decay.
- Deduplication prevents redundant entries from near-identical queries.
- Relevance threshold to prevent injecting noise into the pipeline.
- Graceful fallback to keyword-only if embeddings unavailable.
- Thread-safe for concurrent FastAPI requests.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.memory.embedding import (
    get_embedding,
    is_available as _embedding_is_available,
    EMBEDDING_DIM,
)
from app.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_DEFAULT_MAX_ENTRIES = 100
_DEFAULT_MAX_RETRIEVAL = 3
_RELEVANCE_THRESHOLD = 0.10  # minimum final score to consider relevant

# ── Similarity tuning ─────────────────────────────────────────────────────

_FACT_WEIGHT = 0.8       # weight for Jaccard(query, entry_query + facts)
_SUMMARY_WEIGHT = 0.2    # weight for Jaccard(query, summary)
_SUBSTRING_BOOST = 0.15  # bonus when query is substring of stored query or vice versa
_MIN_TOKENS_FOR_SUBSTRING = 2  # query must have at least this many tokens for substring boost
_DECAY_HALF_LIFE_H = 48.0  # hours until recency factor halves
_DECAY_FLOOR = 0.7       # minimum recency multiplier (old entries never fully killed)
_DECAY_RANGE = 1.0 - _DECAY_FLOOR  # 0.3: the variable range above the floor
_MAX_SYNONYM_EXPANSIONS = 10  # cap on total synonym tokens added per tokenization
_DEDUP_THRESHOLD = 0.80  # token overlap ratio to consider a query a duplicate
_CONFIDENCE_MULTIPLIERS = {"high": 1.0, "medium": 0.9}  # score adjustment by confidence
_VECTOR_TOP_K = 5        # max candidates returned by vector search (first stage)

# ── Hybrid scoring tuning ──────────────────────────────────────────────────

_KEYWORD_SCORE_WEIGHT = 0.6   # weight for keyword score in hybrid blend
_VECTOR_SCORE_WEIGHT = 0.4    # weight for vector score in hybrid blend
_VECTOR_KEYWORD_GATE = 0.05   # min keyword overlap to keep a vector candidate
_EMBED_MAX_FACTS = 5          # max facts included in embedding text
_HYBRID_DEDUP_THRESHOLD = 0.80  # token overlap to treat two results as duplicates

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "about", "against", "over",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "also", "now", "then", "here", "there", "when",
    "where", "why", "how", "what", "which", "who", "whom", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "our", "your", "my", "me", "we", "you",
    "i", "if", "up", "out", "off",
})

# Conservative synonym map — only domain-relevant terms, no generic
# words like "new" that would cause false positives everywhere.
# Each key maps to a frozenset of synonyms (bidirectional expansion).
_SYNONYM_GROUPS: list[frozenset[str]] = [
    frozenset({"latest", "recent", "current"}),
    frozenset({"trends", "developments", "advances", "breakthroughs"}),
    frozenset({"compare", "comparison", "versus", "vs"}),
    frozenset({"explain", "explanation", "overview"}),
    frozenset({"best", "top", "leading"}),
    frozenset({"price", "cost", "pricing"}),
    frozenset({"fast", "quick", "rapid", "speed"}),
    frozenset({"build", "create", "develop", "make"}),
    frozenset({"issue", "problem", "bug", "error"}),
    frozenset({"guide", "tutorial", "howto"}),
    frozenset({"performance", "benchmark", "speed"}),
    frozenset({"ai", "artificial intelligence"}),
    frozenset({"ml", "machine learning"}),
    frozenset({"llm", "large language model"}),
]

# Pre-build a lookup: word → set of synonyms (excludes the word itself)
_SYNONYM_MAP: dict[str, frozenset[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        # Merge with any existing synonyms (handles overlapping groups)
        existing = _SYNONYM_MAP.get(_word, frozenset())
        _SYNONYM_MAP[_word] = (existing | _group) - {_word}


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single stored memory entry."""
    query: str
    facts: list[str]
    summary: str
    timestamp: float
    confidence: str
    entry_id: int = 0
    embedding: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "facts": self.facts,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }


# ── MemoryStore ────────────────────────────────────────────────────────────

class MemoryStore:
    """Thread-safe in-memory store with hybrid retrieval.

    Storage rules:
        - Only stores entries where confidence != "low".
        - Deduplicates: skips if a near-identical query already exists.
        - FIFO eviction when max_entries is reached.

    Retrieval — two-stage hybrid pipeline:
        Stage 1: Vector (semantic) search → top 5 candidates
        Stage 2: Keyword scoring + filters → re-rank → top 3
        Fallback: keyword-only if vector search unavailable
    """

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_retrieval: int = _DEFAULT_MAX_RETRIEVAL,
    ) -> None:
        self._entries: list[MemoryEntry] = []
        self._max_entries = max_entries
        self._max_retrieval = max_retrieval
        self._lock = threading.Lock()
        self._next_id: int = 0

        # ── Vector search initialisation ──
        self._embedding_enabled: bool = False
        self._vector_store: VectorStore | None = None
        if _embedding_is_available():
            try:
                self._vector_store = VectorStore(dimension=EMBEDDING_DIM)
                self._embedding_enabled = True
                logger.info("[MEMORY] Vector search enabled (semantic retrieval)")
            except Exception as exc:
                logger.warning("[MEMORY] Vector store init failed: %s", exc)
        else:
            logger.info("[MEMORY] Vector search disabled (sentence-transformers not installed)")

        logger.info(
            "MemoryStore initialised | max_entries=%d | max_retrieval=%d | vector=%s",
            max_entries, max_retrieval, self._embedding_enabled,
        )

    # ── Public API ─────────────────────────────────────────────────────

    def add_entry(
        self,
        query: str,
        answer: str,
        facts: list[str],
        confidence: str,
    ) -> bool:
        """Store a memory entry if it meets quality and uniqueness criteria.

        Storage rules:
            - confidence == "high"  → always store (if not duplicate)
            - confidence == "medium" → store (if not duplicate)
            - confidence == "low"   → reject

        Deduplication:
            - If an existing entry has >80% token overlap with the new
              query, the new entry is skipped to prevent redundancy.

        Args:
            query:      The original user query.
            answer:     The final synthesized answer (truncated for storage).
            facts:      List of extracted facts from tool steps.
            confidence: Critic confidence level ("high", "medium", "low").

        Returns:
            True if the entry was stored, False if rejected.
        """
        if not query or not query.strip():
            return False

        if confidence == "low":
            logger.info("[MEMORY] Skipped storage — confidence too low (%s)", confidence)
            return False

        # ── Deduplication check ──
        query_tokens = self._tokenize(query)
        with self._lock:
            for existing in self._entries:
                existing_tokens = self._tokenize(existing.query)
                if self._token_overlap_ratio(query_tokens, existing_tokens) >= _DEDUP_THRESHOLD:
                    logger.info(
                        "[MEMORY] ⏭️ Skipped storage — duplicate of existing entry: %r",
                        existing.query[:60],
                    )
                    return False

        # ── Assign unique ID ──
        with self._lock:
            entry_id = self._next_id
            self._next_id += 1

        # ── Generate embedding (outside lock — may be slow on first call) ──
        # Use query + top facts ONLY (no summary — LLM text introduces noise)
        embedding: list[float] = []
        if self._embedding_enabled:
            try:
                top_facts = (facts or [])[:_EMBED_MAX_FACTS]
                embed_text = f"{query} {' '.join(top_facts)}"
                embedding = get_embedding(embed_text)
            except Exception as exc:
                logger.warning("[MEMORY] Embedding generation failed: %s", exc)

        entry = MemoryEntry(
            query=query.strip(),
            facts=facts or [],
            summary=answer[:500] if answer else "",
            timestamp=time.time(),
            confidence=confidence,
            entry_id=entry_id,
            embedding=embedding,
        )

        with self._lock:
            # Evict oldest if at capacity
            if len(self._entries) >= self._max_entries:
                evicted = self._entries.pop(0)
                logger.info("[MEMORY] Evicted oldest entry: %r", evicted.query[:50])
                # Remove evicted entry from vector index
                if self._vector_store:
                    self._vector_store.remove(evicted.entry_id)

            self._entries.append(entry)

        # ── Index in vector store (outside main lock) ──
        if self._vector_store and embedding:
            try:
                self._vector_store.add(embedding, entry_id)
            except Exception as exc:
                logger.warning("[MEMORY] Vector index add failed: %s", exc)

        logger.info(
            "[MEMORY] ✅ Stored entry (confidence=%s, facts=%d, total=%d, vector=%s): %r",
            confidence, len(facts or []), self.size, bool(embedding), query[:60],
        )
        return True

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryEntry]:
        """Retrieve the most relevant past entries for a query.

        Two-stage hybrid pipeline (when vector search is available):
            Stage 1: Vector (semantic) search → top 5 candidates
            Stage 2: Keyword scoring + filters → re-rank → top 3

        Falls back to keyword-only retrieval if vector search is
        unavailable or returns no results after re-ranking.
        """
        top_k = top_k or self._max_retrieval

        if not query or not query.strip():
            return []

        with self._lock:
            if not self._entries:
                logger.info("[MEMORY] No entries in store — nothing to retrieve")
                return []

        # ── Stage 1: Hybrid retrieval (vector + keyword re-rank) ──
        if self._embedding_enabled and self._vector_store and self._vector_store.size > 0:
            try:
                hybrid_results = self._hybrid_retrieve(query, top_k)
                if hybrid_results:
                    return hybrid_results
                logger.info(
                    "[MEMORY] Hybrid retrieval returned no results — "
                    "falling back to keyword",
                )
            except Exception as exc:
                logger.warning(
                    "[MEMORY] Hybrid retrieval failed: %s — "
                    "falling back to keyword",
                    exc,
                )

        # ── Stage 2: Fallback keyword-only retrieval ──
        return self._keyword_retrieve(query, top_k)

    # ── Hybrid retrieval ───────────────────────────────────────────────

    def _hybrid_retrieve(
        self, query: str, top_k: int,
    ) -> list[MemoryEntry]:
        """Two-stage hybrid: vector search → filter → blend → deduplicate.

        Step 1: Semantic search retrieves top candidates by embedding sim.
        Step 2: Keyword overlap gate discards cross-domain false positives.
        Step 3: Blended scoring (0.6 keyword + 0.4 vector) for final rank.
        Step 4: Semantic deduplication ensures result diversity.
        """
        # Step 1: Vector search → top candidates
        query_embedding = get_embedding(query)
        vector_results = self._vector_store.search(
            query_embedding, top_k=_VECTOR_TOP_K,
        )

        if not vector_results:
            return []

        candidate_ids = {eid for eid, _ in vector_results}
        vector_scores = {eid: score for eid, score in vector_results}

        # Look up entries
        with self._lock:
            candidates = [
                e for e in self._entries if e.entry_id in candidate_ids
            ]

        if not candidates:
            return []

        query_tokens = self._tokenize(query)
        now = time.time()

        # Step 2: Keyword overlap gate + scoring
        scored: list[tuple[float, float, MemoryEntry]] = []  # (blended, keyword, entry)

        for entry in candidates:
            # Gate: require minimum keyword overlap to prevent false positives
            entry_tokens = self._tokenize_base(entry.query + " " + " ".join(entry.facts))
            overlap = len(query_tokens & entry_tokens) / len(query_tokens) if query_tokens else 0
            if overlap < _VECTOR_KEYWORD_GATE:
                logger.debug(
                    "[MEMORY] Hybrid gate rejected entry %d (overlap=%.3f): %r",
                    entry.entry_id, overlap, entry.query[:50],
                )
                continue

            keyword_score = self._compute_similarity(query, query_tokens, entry, now)
            v_score = vector_scores.get(entry.entry_id, 0.0)

            # Step 3: Blended score
            blended = (_KEYWORD_SCORE_WEIGHT * keyword_score) + (_VECTOR_SCORE_WEIGHT * v_score)

            if blended >= _RELEVANCE_THRESHOLD:
                scored.append((blended, keyword_score, entry))

        scored.sort(key=lambda x: (x[0], x[2].timestamp), reverse=True)

        # Step 4: Semantic deduplication — keep diverse results
        deduplicated = self._deduplicate_results(
            [(blended, entry) for blended, _, entry in scored]
        )

        results = [entry for _, entry in deduplicated[:top_k]]

        if results:
            score_summary = ", ".join(
                f"v={vector_scores.get(e.entry_id, 0):.3f}|k={ks:.3f}|b={bs:.3f}"
                for bs, ks, e in scored[:top_k]
            )
            logger.info(
                "[MEMORY] Hybrid retrieved %d entries for: %r (scores: %s)",
                len(results), query[:60], score_summary,
            )

        return results

    @staticmethod
    def _deduplicate_results(
        scored: list[tuple[float, MemoryEntry]],
    ) -> list[tuple[float, MemoryEntry]]:
        """Remove near-duplicate results to ensure diversity.

        Keeps the higher-scoring (or longer) entry when two entries
        have > 80% token overlap.
        """
        if len(scored) <= 1:
            return scored

        kept: list[tuple[float, MemoryEntry]] = []
        for score, entry in scored:
            entry_tokens = set(
                (entry.query + " " + " ".join(entry.facts)).lower().split()
            )
            is_dup = False
            for _, kept_entry in kept:
                kept_tokens = set(
                    (kept_entry.query + " " + " ".join(kept_entry.facts)).lower().split()
                )
                smaller = min(len(entry_tokens), len(kept_tokens))
                if smaller == 0:
                    continue
                overlap = len(entry_tokens & kept_tokens) / smaller
                if overlap >= _HYBRID_DEDUP_THRESHOLD:
                    is_dup = True
                    break
            if not is_dup:
                kept.append((score, entry))

        return kept

    # ── Keyword-only retrieval ─────────────────────────────────────────

    def _keyword_retrieve(
        self, query: str, top_k: int,
    ) -> list[MemoryEntry]:
        """Keyword-only retrieval — the original scoring system."""
        with self._lock:
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            now = time.time()
            scored: list[tuple[float, MemoryEntry]] = []

            for entry in self._entries:
                score = self._compute_similarity(query, query_tokens, entry, now)
                if score >= _RELEVANCE_THRESHOLD:
                    scored.append((score, entry))
                    logger.debug(
                        "[MEMORY] Score %.4f for entry: %r",
                        score, entry.query[:60],
                    )

            scored.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)

        results = [entry for _, entry in scored[:top_k]]

        if results:
            score_summary = ", ".join(
                f"{s:.3f}" for s, _ in scored[:top_k]
            )
            logger.info(
                "[MEMORY] Retrieved %d relevant entries for: %r (scores: %s)",
                len(results), query[:60], score_summary,
            )
        else:
            logger.info(
                "[MEMORY] No relevant entries found for: %r "
                "(checked %d entries, all below threshold %.2f)",
                query[:60], len(self._entries), _RELEVANCE_THRESHOLD,
            )

        return results

    @property
    def size(self) -> int:
        """Current number of stored entries."""
        with self._lock:
            return len(self._entries)

    # ── Internals ──────────────────────────────────────────────────────

    def _compute_similarity(
        self,
        query_raw: str,
        query_tokens: set[str],
        entry: MemoryEntry,
        now: float,
    ) -> float:
        """Compute multi-signal similarity between query and entry.

        Combines:
        1. Weighted Jaccard: 0.8 × (query+facts) + 0.2 × (summary)
           - Summary score capped at fact_score to prevent noise dominance
        2. Substring boost: +0.15 (only if query has ≥ 2 tokens)
        3. Time decay: score × (0.7 + 0.3 × recency_factor)
        4. Confidence multiplier: high → 1.0, medium → 0.9
        """
        # ── Signal 1: Jaccard on query + facts (primary) ──
        # Entry tokens use base tokenizer (NO synonym expansion)
        # to keep entry token sets precise
        main_text = entry.query + " " + " ".join(entry.facts)
        main_tokens = self._tokenize_base(main_text)

        fact_score = self._jaccard(query_tokens, main_tokens)

        # ── Signal 2: Jaccard on summary (secondary) ──
        summary_score = 0.0
        if entry.summary:
            summary_tokens = self._tokenize_base(entry.summary)
            # Only use tokens unique to summary (avoid double-counting)
            summary_only = summary_tokens - main_tokens
            if summary_only:
                summary_score = self._jaccard(query_tokens, summary_only)

        # Cap summary score — summary must never exceed fact relevance
        summary_score = min(summary_score, fact_score)

        # ── Weighted blend ──
        base_score = (_FACT_WEIGHT * fact_score) + (_SUMMARY_WEIGHT * summary_score)

        # ── Signal 3: Substring boost for near-duplicate queries ──
        # Only apply if query has enough meaningful tokens to avoid
        # weak matches like "ai" matching "ai trends"
        if len(query_tokens) >= _MIN_TOKENS_FOR_SUBSTRING:
            query_lower = query_raw.strip().lower()
            entry_lower = entry.query.strip().lower()
            if query_lower in entry_lower or entry_lower in query_lower:
                base_score += _SUBSTRING_BOOST

        # ── Signal 4: Time-decay multiplier ──
        age_hours = max(0.0, (now - entry.timestamp) / 3600.0)
        recency_factor = 1.0 / (1.0 + age_hours / _DECAY_HALF_LIFE_H)
        decay_multiplier = _DECAY_FLOOR + _DECAY_RANGE * recency_factor

        # ── Signal 5: Confidence multiplier ──
        confidence_mult = _CONFIDENCE_MULTIPLIERS.get(entry.confidence, 0.9)

        return base_score * decay_multiplier * confidence_mult

    @staticmethod
    def _jaccard(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two token sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def _tokenize_base(text: str) -> set[str]:
        """Tokenize text into base tokens (NO synonym expansion).

        Used for entry/memory tokens to keep them precise and prevent
        inflating the Jaccard denominator.
        """
        words = re.findall(r'\w+', text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 1}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text with synonym expansion.

        Used for QUERY tokens only. Synonym expansion improves recall
        ("latest" matches "recent") but is deliberately NOT applied
        to entry tokens to preserve precision.
        """
        words = re.findall(r'\w+', text.lower())
        base_tokens = {w for w in words if w not in _STOPWORDS and len(w) > 1}

        # Synonym expansion (capped to avoid inflating the token set)
        expanded: set[str] = set()
        expansion_count = 0
        for token in base_tokens:
            synonyms = _SYNONYM_MAP.get(token)
            if synonyms:
                for syn in synonyms:
                    if syn not in base_tokens and expansion_count < _MAX_SYNONYM_EXPANSIONS:
                        expanded.add(syn)
                        expansion_count += 1

        return base_tokens | expanded

    @staticmethod
    def _token_overlap_ratio(tokens_a: set[str], tokens_b: set[str]) -> float:
        """Compute the overlap ratio for deduplication.

        Returns the fraction of the smaller set that overlaps with the larger.
        This is more suitable than Jaccard for dedup because it focuses on
        whether one query is a near-subset of the other.
        """
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        smaller = min(len(tokens_a), len(tokens_b))
        return len(intersection) / smaller if smaller else 0.0
