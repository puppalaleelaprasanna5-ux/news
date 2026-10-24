"""
Evidence retrieval (Phase 7): find reference passages relevant to a claim.

Output for each claim (JSON-friendly):

    {
      "claim": "...",
      "retrieved_evidence": [
          {"text": "...", "source": "file.txt", "source_id": "file.txt", "page": None,
           "chunk_id": 3, "rank": 1, "score": <cosine similarity>, "score_type": "cosine_similarity"}
      ],
      "retrieval_status": "evidence_retrieved" | "no_relevant_evidence"
                          | "retrieval_unavailable" | "invalid_claim",
      "limitations": [...],
      "parameters": {"top_k": ..., "min_score": ..., "score_type": "cosine_similarity"}
    }

IMPORTANT: `score` is the cosine similarity between the claim embedding and the passage
embedding. It measures topical/semantic closeness only. It is NOT a probability, and a
high score does NOT mean the passage supports the claim (a passage can contradict a claim
and still be highly similar to it). Phase 7 never labels a claim true or false.

"source", "page" and "chunk_id" come straight from the stored metadata; nothing (URLs,
dates, page numbers) is ever invented. `page` is None for sources without pages (.txt).
"""

import logging
import re
from typing import Dict, Iterable, List, Optional

from src.rag.config import DEFAULT_DEDUP_JACCARD, DEFAULT_INDEX_DIR, DEFAULT_MIN_SCORE, DEFAULT_TOP_K
from src.rag.embeddings import EmbeddingError
from src.rag.faiss_index import EvidenceIndex, EvidenceIndexError

logger = logging.getLogger(__name__)

STATUS_RETRIEVED = "evidence_retrieved"
STATUS_NONE = "no_relevant_evidence"
STATUS_UNAVAILABLE = "retrieval_unavailable"
STATUS_INVALID = "invalid_claim"

SCORE_TYPE = "cosine_similarity"

BASE_LIMITATIONS = [
    "Retrieval relevance does not establish factual correctness.",
    "Scores are cosine similarities between text embeddings, not probabilities that a claim is true or false.",
    "Passages come only from the supplied reference collection, which may be incomplete, outdated or unreliable.",
]

_WORD = re.compile(r"\w+")


def _word_set(text: str) -> frozenset:
    return frozenset(_WORD.findall(text.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class EvidenceRetriever:
    """Embeds claims and searches an EvidenceIndex. Never raises for ordinary retrieval failures."""

    def __init__(
        self,
        index: Optional[EvidenceIndex],
        embedder,
        top_k: int = DEFAULT_TOP_K,
        min_score: Optional[float] = DEFAULT_MIN_SCORE,
        dedup_jaccard: float = DEFAULT_DEDUP_JACCARD,
        unavailable_reason: Optional[str] = None,
    ):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1.")
        if min_score is not None and not (-1.0 <= min_score <= 1.0):
            raise ValueError("min_score must be between -1 and 1 (or None to disable the threshold).")
        if not (0.0 < dedup_jaccard <= 1.0):
            raise ValueError("dedup_jaccard must be in (0, 1].")
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.dedup_jaccard = dedup_jaccard
        self.unavailable_reason = unavailable_reason or (None if index is not None else "No index loaded.")

    @classmethod
    def from_directory(cls, index_dir=DEFAULT_INDEX_DIR, embedder=None, **kwargs) -> "EvidenceRetriever":
        """
        Load a saved index. A missing/corrupted/mismatched index does not raise: the returned
        retriever reports `retrieval_unavailable` (with the reason) for every claim.
        """
        if embedder is None:
            from src.rag.embeddings import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        try:
            index = EvidenceIndex.load(index_dir, expected_model_name=embedder.model_name)
            return cls(index, embedder, **kwargs)
        except EvidenceIndexError as err:
            logger.warning("Evidence index unavailable: %s", err)
            return cls(None, embedder, unavailable_reason=str(err), **kwargs)

    # --------------------------------------------------------------- helpers
    def _parameters(self, top_k: int, min_score: Optional[float]) -> dict:
        return {"top_k": top_k, "min_score": min_score, "score_type": SCORE_TYPE}

    def _result(self, claim, status, evidence, extra_limitations, top_k, min_score, error=None) -> dict:
        result = {
            "claim": claim,
            "retrieved_evidence": evidence,
            "retrieval_status": status,
            "limitations": BASE_LIMITATIONS + list(extra_limitations),
            "parameters": self._parameters(top_k, min_score),
        }
        if error:
            result["error"] = error
        return result

    # -------------------------------------------------------------- retrieval
    def retrieve(self, claim, top_k: Optional[int] = None, min_score="default") -> dict:
        """Retrieve evidence passages for ONE claim. `min_score=None` disables the threshold."""
        top_k = self.top_k if top_k is None else top_k
        threshold = self.min_score if min_score == "default" else min_score
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be an integer >= 1.")

        claim_text = claim if isinstance(claim, str) else ""
        if not claim_text.strip():
            return self._result(
                claim_text, STATUS_INVALID, [], ["The claim was empty or not a string, so nothing was searched."], top_k, threshold,
                error="Claim must be a non-empty string.",
            )

        if self.index is None:
            return self._result(
                claim_text, STATUS_UNAVAILABLE, [], ["Retrieval was unavailable, so no evidence was searched."], top_k, threshold,
                error=self.unavailable_reason,
            )

        if self.index.is_empty:
            return self._result(
                claim_text, STATUS_NONE, [],
                ["The reference collection is empty, so no passages could be retrieved."], top_k, threshold,
            )

        try:
            query_vector = self.embedder.embed_query(claim_text)
            # Over-fetch so that threshold filtering and de-duplication can still fill top_k.
            candidates = self.index.search(query_vector, max(top_k * 4, top_k + 10))
        except (EmbeddingError, EvidenceIndexError) as err:
            return self._result(
                claim_text, STATUS_UNAVAILABLE, [], ["Retrieval failed, so no evidence was searched."], top_k, threshold, error=str(err)
            )
        except Exception as err:  # unexpected backend failure must not crash the caller
            logger.exception("Unexpected retrieval failure")
            return self._result(
                claim_text, STATUS_UNAVAILABLE, [], ["Retrieval failed, so no evidence was searched."], top_k, threshold,
                error=f"{type(err).__name__}: {err}",
            )

        evidence: List[dict] = []
        kept_words: List[frozenset] = []
        for chunk, score in candidates:
            if threshold is not None and score < threshold:
                continue
            words = _word_set(chunk.text)
            if any(_jaccard(words, seen) >= self.dedup_jaccard for seen in kept_words):
                continue  # near-identical to a higher-ranked passage
            kept_words.append(words)
            evidence.append(
                {
                    "text": chunk.text,
                    "source": chunk.filename,
                    "source_id": chunk.source_id,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "rank": len(evidence) + 1,
                    "score": round(score, 4),
                    "score_type": SCORE_TYPE,
                }
            )
            if len(evidence) == top_k:
                break

        if evidence:
            return self._result(claim_text, STATUS_RETRIEVED, evidence, [], top_k, threshold)
        note = (
            f"No passage reached the minimum similarity threshold ({threshold}). This does not mean the claim is "
            f"true or false; the reference collection may simply not cover the topic."
            if threshold is not None
            else "No passages were returned. This does not mean the claim is true or false."
        )
        return self._result(claim_text, STATUS_NONE, [], [note], top_k, threshold)

    def retrieve_many(self, claims: Iterable, top_k: Optional[int] = None, min_score="default") -> List[dict]:
        """Retrieve for several claims independently; one claim failing never affects the others."""
        return [self.retrieve(c, top_k=top_k, min_score=min_score) for c in claims]


def retrieve_evidence(claim: str, index_dir=DEFAULT_INDEX_DIR, embedder=None, top_k: int = DEFAULT_TOP_K,
                      min_score: Optional[float] = DEFAULT_MIN_SCORE) -> dict:
    """Convenience one-shot: load the saved index and retrieve evidence for a single claim."""
    retriever = EvidenceRetriever.from_directory(index_dir, embedder, top_k=top_k, min_score=min_score)
    return retriever.retrieve(claim)
