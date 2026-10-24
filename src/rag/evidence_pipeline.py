"""
Retrieve evidence for the claims in a news article.

Flow:
    article text
      |- (optional) FakeNewsPredictor.predict()          -> "ml_prediction"
      |- manual claims, or rule-based sentence selection  -> claims
      '- EvidenceRetriever.retrieve_many(claims)          -> "claims"[*]

The outputs are returned side by side and are NEVER merged into one prediction. No verdict,
credibility score or true/false label is produced here.

Claims come from `manual_claims` when supplied, otherwise from a simple rule-based sentence
selector (src.rag.claim_selector). If neither yields anything, the result simply has no claims.
"""

import logging
from typing import Any, Dict, List, Optional

from src.rag.claim_selector import select_claims

logger = logging.getLogger(__name__)

DEFAULT_MAX_CLAIMS = 10

PIPELINE_LIMITATIONS = [
    "Evidence retrieval finds similar passages only; it does not verify claims or label them true or false.",
    "The ML prediction and the retrieved evidence are independent outputs and are not combined.",
]


def _clean_claims(claims, max_claims: int) -> List[str]:
    seen, cleaned = set(), []
    for claim in claims or []:
        text = claim.strip() if isinstance(claim, str) else ""
        key = " ".join(text.lower().split())
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned[:max_claims]


def gather_evidence_for_article(
    article_text: str,
    retriever,
    predictor=None,
    manual_claims: Optional[List[str]] = None,
    article_metadata: Optional[Dict[str, Any]] = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Select claims and retrieve evidence for one article.

    Args:
        article_text:      the article body (preserved unchanged in the result).
        retriever:         an EvidenceRetriever.
        predictor:         optional FakeNewsPredictor; its output is passed through as-is.
        manual_claims:     claims supplied by the user. When given they are used as-is; otherwise
                           candidate sentences are chosen by the rule-based selector.
        article_metadata:  optional dict (title, source name, URL, date ...) returned unchanged.
                           Only what the caller supplies is ever included.
    """
    errors: List[str] = []
    result: Dict[str, Any] = {
        "article": {"text": article_text, "metadata": dict(article_metadata or {})},
        "ml_prediction": None,
        "claim_source": "none",
        "claim_extraction_status": "not_attempted",
        "claims": [],
        "errors": errors,
        "limitations": list(PIPELINE_LIMITATIONS),
    }

    if predictor is not None:
        try:
            result["ml_prediction"] = predictor.predict(article_text)
        except Exception as err:
            logger.warning("ML prediction failed: %s", err)
            errors.append(f"ml_prediction_failed: {type(err).__name__}: {err}")

    claims = _clean_claims(manual_claims, max_claims)
    if claims:
        result["claim_source"] = "manual"
        result["claim_extraction_status"] = "manual_claims_supplied"
    else:
        claims = _clean_claims(select_claims(article_text, max_claims), max_claims)
        if claims:
            result["claim_source"] = "heuristic"
            result["claim_extraction_status"] = "extracted"
        else:
            result["claim_extraction_status"] = "no_claims_found"

    if retriever is not None and claims:
        result["claims"] = retriever.retrieve_many(claims, top_k=top_k)
    elif retriever is None and claims:
        errors.append("retrieval_unavailable: no retriever was provided.")
    return result
