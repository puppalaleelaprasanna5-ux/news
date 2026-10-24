"""
Text embeddings (Phase 7).

`SentenceTransformerEmbedder` wraps a local Sentence-Transformers model. The same embedder
instance (or at least the same model name) must be used for indexing and for queries; the
saved index records the model name and refuses to load for a different one.

All vectors returned are float32 and L2-normalized, so the FAISS inner product between two
of them equals their cosine similarity.

Model download: the first use of a model name downloads its files from the Hugging Face Hub
(`all-MiniLM-L6-v2` is roughly 90 MB) into the local Hugging Face cache; afterwards it runs
fully offline (set `local_files_only=True` or env `HF_HUB_OFFLINE=1` to forbid network access).
No API key or paid service is involved.

`HashingEmbedder` is a tiny *lexical* (word-overlap) embedder with no dependencies. It exists
ONLY for unit tests and offline demos. It has no semantic understanding and must not be used
as evidence retrieval in real analysis.
"""

import hashlib
import logging
import re
from typing import List, Optional, Sequence

import numpy as np

from src.rag.config import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when text cannot be embedded (bad input, missing model, model failure)."""


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return float32 L2-normalized rows; raises EmbeddingError for NaN/inf or zero vectors."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise EmbeddingError(f"Expected a 2-D embedding array, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise EmbeddingError("Embedding contains NaN or infinite values.")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingError("Embedding contains an all-zero vector and cannot be normalized.")
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


def _validate_texts(texts: Sequence[str]) -> List[str]:
    if isinstance(texts, str) or not hasattr(texts, "__iter__"):
        raise EmbeddingError("embed_texts expects a list of strings.")
    cleaned = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise EmbeddingError(f"Text #{i} is not a string ({type(text).__name__}).")
        if not text.strip():
            raise EmbeddingError(f"Text #{i} is empty.")
        cleaned.append(text)
    return cleaned


class SentenceTransformerEmbedder:
    """Lazy-loading wrapper around a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.local_files_only = local_files_only
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise EmbeddingError(
                "The 'sentence-transformers' package is not installed. Run: pip install -r requirements.txt"
            ) from err
        try:
            self._model = SentenceTransformer(
                self.model_name, device=self.device, local_files_only=self.local_files_only
            )
        except Exception as err:  # download failure, missing offline cache, bad model name ...
            raise EmbeddingError(
                f"Could not load embedding model '{self.model_name}': {err}. The first run needs internet "
                f"access to download the model once; afterwards it works offline."
            ) from err
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a list of non-empty strings -> (n, dim) float32 normalized array."""
        cleaned = _validate_texts(texts)
        if not cleaned:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._load()
        try:
            raw = model.encode(
                cleaned,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as err:
            raise EmbeddingError(f"Embedding failed: {err}") from err
        return l2_normalize(raw)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one non-empty query string -> (dim,) float32 normalized vector."""
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingError("Query is empty.")
        return self.embed_texts([query])[0]


_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the this to was were will with".split()
)


class HashingEmbedder:
    """Dependency-free lexical embedder for tests/demos ONLY (word overlap, not meaning)."""

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.model_name = f"hashing-embedder-{dim}"
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        tokens = [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS] or _TOKEN.findall(text.lower())
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in tokens or ["__empty__"]:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self._dim
            vec[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
        if not vec.any():  # signs cancelled out exactly
            vec[0] = 1.0
        return vec

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        cleaned = _validate_texts(texts)
        if not cleaned:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(t) for t in cleaned]))

    def embed_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingError("Query is empty.")
        return self.embed_texts([query])[0]
