"""
Embedding providers (section 13-15).

The rest of the pipeline only talks to the :class:`EmbeddingProvider`
interface, so the underlying model can be swapped (config ``embedding_model``)
— or replaced entirely, e.g. with the offline hashing provider used in tests —
without touching any other module.

Embeddings are used ONLY for semantic relevance ranking (RULE 4).  They never
decide experience eligibility, overqualification or hard requirements
(RULE 2).
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Interface every embedding backend must implement."""

    name: str = "abstract"

    @abstractmethod
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Return an L2-normalised (n, dim) float32 matrix."""

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for normalised vectors == dot product."""
        return float(np.dot(a, b))

    def similarity_matrix(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        return candidates @ query


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence-transformers backend (default: BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed; run "
                "`pip install sentence-transformers torch` or choose another "
                "embedding provider."
            ) from exc
        self.name = model_name
        logger.info("Loading embedding model '%s' (once, reused for the whole run)", model_name)
        self._model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingProvider(EmbeddingProvider):
    """
    Deterministic character-n-gram hashing embeddings (no downloads, no torch).

    Good enough for offline tests/smoke runs; NOT for production matching.
    """

    def __init__(self, model_name: str = "hashing-dummy-256", dim: int = 256) -> None:
        self.name = model_name
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        normalized = " ".join((text or "").lower().split())
        for i in range(len(normalized) - 2):
            gram = normalized[i : i + 3]
            bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16) % self.dim
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return np.vstack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


def get_provider(model_name: str, allow_fallback: bool = False) -> EmbeddingProvider:
    """
    Factory.  With ``allow_fallback=True`` a missing sentence-transformers
    installation degrades to the hashing provider (logged loudly) instead of
    crashing — used by tests and the offline demo.
    """
    if model_name.startswith("hashing"):
        return HashingProvider(model_name=model_name)
    try:
        return SentenceTransformerProvider(model_name)
    except Exception as exc:
        if allow_fallback:
            logger.warning(
                "Falling back to HashingProvider ('%s' unavailable: %s) — "
                "semantic quality will be degraded",
                model_name, exc,
            )
            return HashingProvider()
        raise


def rescale_similarity(
    similarity: float, floor: float, ceiling: float
) -> float:
    """Map a raw cosine similarity to a 0-100 score (linear, clamped)."""
    if ceiling <= floor:
        return 50.0
    score = 100.0 * (similarity - floor) / (ceiling - floor)
    return round(max(0.0, min(100.0, score)), 2)
