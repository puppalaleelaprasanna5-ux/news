"""
Tests: claims -> retrieved evidence (the article-level evidence pipeline).

The ML predictor is mocked; no saved model is needed.

    python -m pytest tests/test_rag_pipeline.py -v
"""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embeddings import HashingEmbedder
from src.rag.evidence_pipeline import gather_evidence_for_article
from src.rag.retriever import STATUS_NONE, STATUS_RETRIEVED, STATUS_UNAVAILABLE, EvidenceRetriever

faiss = pytest.importorskip("faiss")
from src.rag.faiss_index import build_or_update_index  # noqa: E402

ARTICLE = "The council said the Alder River levee was built in 1990. Officials claim tram tickets cost one dollar."

CLAIMS = [
    "The Alder River levee was built in 1990.",
    "Tram tickets cost one dollar.",
    "Tram tickets cost one dollar.",   # duplicate
    "   ",                             # blank
    "Dragons guard the city hall.",
]


@pytest.fixture
def retriever(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rivers.txt").write_text("The Alder River levee was built in 1990 by engineers.", encoding="utf-8")
    (docs / "transit.txt").write_text("Tram tickets cost one dollar and trams run every ten minutes.", encoding="utf-8")
    emb = HashingEmbedder()
    result = build_or_update_index(docs, tmp_path / "index", emb, 200, 20)
    return EvidenceRetriever(result.index, emb, min_score=0.3)


def test_supplied_claims_get_independent_evidence(retriever):
    out = gather_evidence_for_article(ARTICLE, retriever, manual_claims=CLAIMS, article_metadata={"title": "T", "source_name": "Demo"})
    assert out["claim_source"] == "manual" and out["claim_extraction_status"] == "manual_claims_supplied"
    claims = [c["claim"] for c in out["claims"]]
    assert claims == ["The Alder River levee was built in 1990.", "Tram tickets cost one dollar.", "Dragons guard the city hall."]
    statuses = [c["retrieval_status"] for c in out["claims"]]
    assert statuses == [STATUS_RETRIEVED, STATUS_RETRIEVED, STATUS_NONE]
    assert out["claims"][0]["retrieved_evidence"][0]["source"] == "rivers.txt"
    assert out["claims"][1]["retrieved_evidence"][0]["source"] == "transit.txt"


def test_rule_based_selection_is_used_when_no_claims_are_supplied(retriever):
    out = gather_evidence_for_article(ARTICLE, retriever)
    assert out["claim_source"] == "heuristic" and out["claim_extraction_status"] == "extracted"
    assert [c["claim"] for c in out["claims"]] == ["The council said the Alder River levee was built in 1990."]
    assert out["claims"][0]["retrieval_status"] == STATUS_RETRIEVED
    assert out["claims"][0]["retrieved_evidence"][0]["source"] == "rivers.txt"


def test_no_checkable_sentences_gives_no_claims_and_does_not_invent_any(retriever):
    out = gather_evidence_for_article("nothing here looks checkable at all really, just plain words in a row", retriever)
    assert out["claims"] == [] and out["claim_source"] == "none" and out["claim_extraction_status"] == "no_claims_found"


def test_article_and_metadata_are_preserved_unchanged(retriever):
    meta = {"title": "My title", "source_name": "Example Daily"}
    out = gather_evidence_for_article(ARTICLE, retriever, manual_claims=["x claim"], article_metadata=meta)
    assert out["article"] == {"text": ARTICLE, "metadata": meta}
    out2 = gather_evidence_for_article(ARTICLE, retriever, manual_claims=["x claim"])
    assert out2["article"]["metadata"] == {}  # nothing invented


def test_outputs_stay_separate_and_unmerged(retriever):
    predictor = MagicMock()
    predictor.predict.return_value = {"label": "Real", "fake_probability": 0.12}
    out = gather_evidence_for_article(ARTICLE, retriever, predictor=predictor, manual_claims=["The Alder River levee was built in 1990."])
    assert out["ml_prediction"] == {"label": "Real", "fake_probability": 0.12}
    assert "llama_analysis" not in out
    assert isinstance(out["claims"], list)
    forbidden = {"verdict", "credibility_score", "final_label", "is_true", "verified", "combined_score"}
    assert not (forbidden & set(out))
    for claim in out["claims"]:
        assert not (forbidden & set(claim))
    assert any("does not verify claims" in lim for lim in out["limitations"])


def test_retrieval_works_without_any_other_component(retriever):
    out = gather_evidence_for_article(ARTICLE, retriever, manual_claims=["The Alder River levee was built in 1990."])
    assert out["claim_source"] == "manual" and out["claim_extraction_status"] == "manual_claims_supplied"
    assert out["errors"] == []
    assert retriever.retrieve("The Alder River levee was built in 1990.")["retrieved_evidence"] == out["claims"][0]["retrieved_evidence"]


def test_predictor_failure_is_recorded_not_raised(retriever):
    predictor = MagicMock()
    predictor.predict.side_effect = ValueError("empty after cleaning")
    out = gather_evidence_for_article(ARTICLE, retriever, predictor=predictor, manual_claims=["a claim"])
    assert out["ml_prediction"] is None and any("ml_prediction_failed" in e for e in out["errors"])


def test_unavailable_retriever_is_reported_per_claim(tmp_path):
    retriever = EvidenceRetriever.from_directory(tmp_path / "missing", HashingEmbedder())
    out = gather_evidence_for_article(ARTICLE, retriever, manual_claims=["The Alder River levee was built in 1990."])
    assert out["claims"] and all(c["retrieval_status"] == STATUS_UNAVAILABLE for c in out["claims"])


def test_max_claims_limit(retriever):
    out = gather_evidence_for_article(ARTICLE, retriever, manual_claims=[f"claim number {i}" for i in range(30)], max_claims=4)
    assert len(out["claims"]) == 4


def test_importing_rag_package_does_not_load_heavy_libraries():
    code = "import sys, src.rag; print('faiss' in sys.modules, 'sentence_transformers' in sys.modules, 'torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT_ROOT,
                         env={**__import__('os').environ, "PYTHONPATH": str(PROJECT_ROOT) + ":" + __import__('os').environ.get("PYTHONPATH", "")})
    assert out.stdout.strip() == "False False False", out.stderr
