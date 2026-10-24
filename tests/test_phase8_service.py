"""
Tests: AnalysisService (orchestration), input validation, partial results, health.

The ML model tests use the REAL saved artifacts in models/ (skipped if absent). The embedding
model is replaced by a fake, so no model download is needed.
RAG tests that build a real index use the dependency-free HashingEmbedder and need faiss-cpu.

    python -m pytest tests/test_phase8_service.py -v -rs
"""

import sys
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.credibility.service import (
    MAX_INPUT_CHARS, AnalysisOptions, AnalysisService, InputValidationError, validate_input_text,
)
from src.models.predict import DEFAULT_MODELS_DIR, FakeNewsPredictor
from src.rag.embeddings import HashingEmbedder
from src.rag.faiss_index import (
    EvidenceIndex, EvidenceIndexError, FaissNotInstalledError, IndexCorruptedError, IndexMismatchError,
    IndexNotFoundError,
)

ARTIFACTS_PRESENT = all((DEFAULT_MODELS_DIR / n).exists() for n in ("best_model.joblib", "tfidf_vectorizer.joblib", "model_metadata.json"))

ARTICLE = ("WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a spending bill that funds the government through "
           "September, officials said, after weeks of negotiations between the two parties over defense and health programs. "
           "The measure now goes to the House of Representatives, which is expected to vote on it later this week.")

def fake_predictor(label="Real", confidence_level="high", n_words=60):
    p = MagicMock()
    p.predict.return_value = {"label": label, "fake_probability": 0.1 if label == "Real" else 0.9, "confidence_level": confidence_level,
                              "n_words": n_words, "warnings": [], "model_name": "fake"}
    p.model_info.return_value = {"model_name": "fake", "load_warnings": []}
    return p


def make_service(tmp_path=None, predictor=None, index_dir=None, **kw):
    return AnalysisService(
        models_dir=tmp_path or DEFAULT_MODELS_DIR,
        index_dir=index_dir or (Path(tmp_path) / "no_index" if tmp_path else PROJECT_ROOT / "no_such_index"),
        predictor_factory=(lambda: predictor) if predictor is not None else None,
        embedder_factory=lambda: HashingEmbedder(),
        **kw,
    )


# ------------------------------------------------------------------- validation
def test_validate_input_text_rules():
    assert validate_input_text("  hello world  ") == "hello world"
    for bad in ("", "   \n\t", None, 12, ["x"]):
        with pytest.raises(InputValidationError):
            validate_input_text(bad)
    with pytest.raises(InputValidationError, match="too long"):
        validate_input_text("a" * (MAX_INPUT_CHARS + 1))
    assert len(validate_input_text("a" * MAX_INPUT_CHARS)) == MAX_INPUT_CHARS
    assert validate_input_text("ab\x00cd") == "abcd"


def test_analyze_raises_only_for_bad_input(tmp_path):
    svc = make_service(tmp_path, fake_predictor())
    with pytest.raises(InputValidationError):
        svc.analyze("   ")
    with pytest.raises(InputValidationError):
        svc.analyze("x" * (MAX_INPUT_CHARS + 5))


# ---------------------------------------------------- real saved ML model
@unittest.skipUnless(ARTIFACTS_PRESENT, "model artifacts not present")
def test_real_ml_model_prediction_through_service(tmp_path):
    svc = AnalysisService(index_dir=tmp_path / "none",
                          embedder_factory=lambda: HashingEmbedder())
    out = svc.analyze(ARTICLE, AnalysisOptions(run_rag=False))
    ml = out["components"]["ml_prediction"]
    assert ml["status"] == "ok"
    assert ml["result"]["label"] in ("Fake", "Real")
    assert 0.0 <= ml["result"]["fake_probability"] <= 1.0
    assert "not proof of factual correctness" in ml["notice"]
    assert "top_terms" in ml["result"]


@unittest.skipUnless(ARTIFACTS_PRESENT, "model artifacts not present")
def test_ml_model_is_loaded_once_and_reused():
    calls = []

    def factory():
        calls.append(1)
        return FakeNewsPredictor()

    svc = AnalysisService(index_dir=PROJECT_ROOT / "none", predictor_factory=factory, embedder_factory=lambda: HashingEmbedder())
    for _ in range(3):
        svc.analyze(ARTICLE, AnalysisOptions(run_rag=False))
    assert len(calls) == 1


def test_missing_model_files_give_unavailable_ml_but_other_components_run(tmp_path):
    def boom():
        raise FileNotFoundError("best_model.joblib")

    svc = AnalysisService(models_dir=tmp_path, index_dir=tmp_path / "none", predictor_factory=boom, embedder_factory=lambda: HashingEmbedder())
    out = svc.analyze(ARTICLE)
    assert out["components"]["ml_prediction"]["status"] == "unavailable"
    assert "notebooks/03_ml_models.ipynb" in out["components"]["ml_prediction"]["error"]
    assert "Traceback" not in str(out) and "best_model.joblib" not in out["components"]["ml_prediction"]["error"]
    assert out["partial"] is True


def test_ml_model_load_is_retried_after_failure(tmp_path):
    state = {"fail": True}

    def factory():
        if state["fail"]:
            raise FileNotFoundError("x")
        return fake_predictor()

    svc = AnalysisService(index_dir=tmp_path / "none", predictor_factory=factory, embedder_factory=lambda: HashingEmbedder())
    assert svc.analyze(ARTICLE, AnalysisOptions(run_rag=False))["components"]["ml_prediction"]["status"] == "unavailable"
    state["fail"] = False
    assert svc.analyze(ARTICLE, AnalysisOptions(run_rag=False))["components"]["ml_prediction"]["status"] == "ok"


def test_corrupted_model_gives_generic_message(tmp_path):
    def boom():
        raise RuntimeError("secret internal path C:/Users/x/model.joblib")

    svc = AnalysisService(index_dir=tmp_path / "none", predictor_factory=boom, embedder_factory=lambda: HashingEmbedder())
    ml = svc.analyze(ARTICLE, AnalysisOptions(run_rag=False))["components"]["ml_prediction"]
    assert ml["status"] == "unavailable" and "C:/Users" not in str(ml)


def test_ml_value_error_is_reported_not_raised(tmp_path):
    predictor = fake_predictor()
    predictor.predict.side_effect = ValueError("Input #0 is empty after cleaning")
    out = make_service(tmp_path, predictor).analyze("!!! ???")
    assert out["components"]["ml_prediction"]["status"] == "error"


# --------------------------------------------------------------------------- RAG
def build_demo_index(tmp_path):
    pytest.importorskip("faiss")
    from src.rag.config import SAMPLE_DOCS_DIR
    from src.rag.faiss_index import build_or_update_index
    build_or_update_index(SAMPLE_DOCS_DIR, tmp_path / "idx", HashingEmbedder(), 300, 60)
    return tmp_path / "idx"


def test_missing_rag_index_gives_unavailable_evidence_with_other_components_ok(tmp_path):
    # "No index" can only be reported once FAISS has imported: the library is checked first (see
    # FaissNotInstalledError), so this exercises the real missing-index path only where faiss-cpu is installed.
    # The FAISS-missing case and every error->message mapping are covered below WITHOUT needing faiss.
    pytest.importorskip("faiss")
    out = make_service(tmp_path, fake_predictor()).analyze(ARTICLE)
    ev = out["components"]["evidence"]
    assert ev["status"] == "unavailable" and "scripts/rag_cli.py build" in ev["error"]
    assert str(tmp_path) not in str(ev)  # no filesystem paths leak
    assert out["components"]["ml_prediction"]["status"] == "ok"
    claim_items = out["credibility_assessment"]["article_claims"]["items"]
    assert claim_items and all(item["retrieval_status"] == "retrieval_unavailable" for item in claim_items)
    assert all(item["evidence_interpretation"]["classification"] == "inconclusive" for item in claim_items)
    assert out["partial"] is True


def test_retrieval_exception_preserves_selected_claims_as_inconclusive(tmp_path):
    svc = make_service(tmp_path, fake_predictor())
    broken_retriever = MagicMock()
    broken_retriever.retrieve_many.side_effect = RuntimeError("private retrieval detail")
    svc._get_retriever = lambda: (broken_retriever, None)

    out = svc.analyze(ARTICLE)

    evidence = out["components"]["evidence"]
    assert evidence["status"] == "error"
    assert "private retrieval detail" not in str(out)
    claims = out["credibility_assessment"]["article_claims"]["items"]
    assert claims and all(item["evidence_interpretation"]["classification"] == "inconclusive" for item in claims)
    assert out["components"]["ml_prediction"]["status"] == "ok"


def _without_faiss():
    """Make `import faiss` fail inside the block (also when faiss IS installed): a None entry in sys.modules."""
    return mock.patch.dict(sys.modules, {"faiss": None})


def test_missing_faiss_is_reported_distinctly_from_a_missing_index(tmp_path):
    with _without_faiss():
        out = make_service(tmp_path, fake_predictor()).analyze(ARTICLE)
    ev = out["components"]["evidence"]
    assert ev["status"] == "unavailable"
    assert "FAISS" in ev["error"] and "pip install -r requirements.txt" in ev["error"]
    assert "rag_cli.py build" not in ev["error"]  # building cannot work without FAISS, so do not suggest it
    assert str(tmp_path) not in str(ev)
    assert out["components"]["ml_prediction"]["status"] == "ok"
    assert out["partial"] is True


def test_faiss_is_checked_before_the_index_directory(tmp_path):
    """Documents the ordering: no FAISS -> FaissNotInstalledError even for a directory that does not exist."""
    with _without_faiss():
        with pytest.raises(FaissNotInstalledError) as caught:
            EvidenceIndex.load(tmp_path / "does_not_exist")
    assert isinstance(caught.value, EvidenceIndexError)          # existing `except EvidenceIndexError` code still works
    assert not isinstance(caught.value, IndexNotFoundError)
    assert "faiss-cpu is not installed" in str(caught.value)     # original message preserved


def test_health_reports_missing_faiss_for_the_rag_component(tmp_path):
    with _without_faiss():
        health = make_service(tmp_path, fake_predictor()).health()
    rag = health["components"]["rag"]
    assert rag["status"] == "unavailable" and "FAISS" in rag["error"] and str(tmp_path) not in str(rag)


def test_each_index_failure_gets_its_own_actionable_message(tmp_path):
    """Maps every index exception to a distinct, user-safe message. Needs no faiss: EvidenceIndex.load is patched."""
    cases = [
        (FaissNotInstalledError("faiss-cpu is not installed."), ["FAISS", "pip install"], ["rag_cli.py build"]),
        (IndexNotFoundError("Index directory not found: /secret/path"), ["scripts/rag_cli.py build"], ["FAISS", "/secret/path"]),
        (IndexCorruptedError("checksum mismatch in /secret/path"), ["damaged", "--force"], ["/secret/path"]),
        (IndexMismatchError("built with other-model"), ["different embedding model"], ["other-model"]),
        (EvidenceIndexError("boom at /secret/path"), ["could not be loaded", "--force"], ["/secret/path", "not installed"]),
        (RuntimeError("surprise at /secret/path"), ["could not be prepared"], ["/secret/path", "surprise"]),
    ]
    messages = []
    for exc, must_have, must_not_have in cases:
        with mock.patch.object(EvidenceIndex, "load", side_effect=exc):
            out = make_service(tmp_path, fake_predictor()).analyze(ARTICLE)
        ev = out["components"]["evidence"]
        assert ev["status"] == "unavailable", type(exc).__name__
        for text in must_have:
            assert text in ev["error"], (type(exc).__name__, text, ev["error"])
        for text in must_not_have:
            assert text not in ev["error"], (type(exc).__name__, text, ev["error"])
        assert out["components"]["ml_prediction"]["status"] == "ok" and out["partial"] is True
        messages.append(ev["error"])
    assert len(set(messages)) == len(cases)  # six conditions, six different messages


def test_rag_evidence_for_supplied_claims_and_synthetic_labelling(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = make_service(tmp_path, fake_predictor(), index_dir=idx, min_score=0.2)
    out = svc.analyze(ARTICLE, AnalysisOptions(manual_claims=["The Alder River levee was built in 1990.", "Library membership is free."]))
    ev = out["components"]["evidence"]
    assert ev["status"] == "ok" and ev["claim_source"] == "manual"
    assert [c["claim"] for c in ev["claims"]] == ["The Alder River levee was built in 1990.", "Library membership is free."]
    first = ev["claims"][0]
    assert first["retrieval_status"] == "evidence_retrieved"
    top = first["retrieved_evidence"][0]
    assert top["source"] == "brightwater_flood_report.txt" and top["is_synthetic_demo"] is True
    assert top["score_type"] == "cosine_similarity"
    assert ev["uses_synthetic_demo_documents"] is True
    assert any("FICTIONAL" in n for n in ev["notes"])
    assert ev["claims"][1]["retrieved_evidence"][0]["source"] == "brightwater_library_notes.txt"
    # evidence is separate from, and does not alter, the other components
    assert out["components"]["ml_prediction"]["result"]["label"] == "Real"
    fa = out["final_assessment"]
    assert fa["is_verdict"] is False and fa["overall_signal"] in (
        "insufficient_information", "mixed_signals", "warning_signs_detected", "few_warning_signs_detected")
    assert not any(k in fa for k in ("verdict", "is_true", "verified", "credibility_score"))


def test_rag_no_relevant_evidence_is_reported_honestly(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = make_service(tmp_path, fake_predictor(), index_dir=idx, min_score=0.35)
    ev = svc.analyze(ARTICLE, AnalysisOptions(manual_claims=["Quantum entanglement of distant photons."]))["components"]["evidence"]
    assert ev["status"] == "ok"
    assert ev["claims"][0]["retrieval_status"] == "no_relevant_evidence" and ev["claims"][0]["retrieved_evidence"] == []


def test_rag_uses_manual_claims(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = make_service(tmp_path, fake_predictor(), index_dir=idx, min_score=0.2)
    out = svc.analyze(ARTICLE, AnalysisOptions(manual_claims=["Tram tickets cost two dollars."]))
    ev = out["components"]["evidence"]
    assert ev["status"] == "ok" and ev["claim_source"] == "manual"
    assert ev["claims"][0]["retrieved_evidence"][0]["source"] == "northfield_tram_bulletin.txt"


def test_short_headline_is_used_as_query_when_no_claims(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = make_service(tmp_path, fake_predictor(), index_dir=idx, min_score=0.2)
    ev = svc.analyze("Library membership is free for all residents")["components"]["evidence"]
    assert ev["claim_source"] == "input_text" and ev["claims"][0]["claim"] == "Library membership is free for all residents"
    assert any("input text itself" in n for n in ev["notes"])


def test_long_article_uses_rule_based_sentence_selection(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = make_service(tmp_path, fake_predictor(), index_dir=idx, min_score=0.0)
    text = ARTICLE + " " + ARTICLE
    ev = svc.analyze(text)["components"]["evidence"]
    assert ev["status"] == "ok" and ev["claim_source"] == "heuristic" and ev["claims"]
    sentences = [c["claim"] for c in ev["claims"]]
    assert all(c in text for c in sentences)                       # claims are real sentences from the text
    assert len(sentences) <= 5 and len(set(sentences)) == len(sentences)
    assert any("rule-based" in n for n in ev["notes"])


def test_manual_claims_take_priority_over_rule_based_selection(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = make_service(tmp_path, fake_predictor(), index_dir=idx, min_score=0.0)
    ev = svc.analyze(ARTICLE + " " + ARTICLE, AnalysisOptions(manual_claims=["Library membership is free."]))["components"]["evidence"]
    assert ev["claim_source"] == "manual" and [c["claim"] for c in ev["claims"]] == ["Library membership is free."]


def test_long_text_with_no_checkable_sentences_skips_retrieval_instead_of_inventing_claims(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = make_service(tmp_path, fake_predictor(), index_dir=idx)
    flat = "word " * 50 + ". " + "thing " * 50 + "."          # no figures, names or assertion verbs
    ev = svc.analyze(flat)["components"]["evidence"]
    assert ev["status"] == "skipped" and ev["claims"] == [] and ev["claim_source"] == "none"


def test_duplicate_and_blank_claims_are_cleaned(tmp_path):
    idx = build_demo_index(tmp_path)
    claims = ["Library membership is free.", "library  membership is FREE.", "   "]
    ev = make_service(tmp_path, fake_predictor(), index_dir=idx).analyze(
        ARTICLE, AnalysisOptions(manual_claims=claims))["components"]["evidence"]
    assert len(ev["claims"]) == 1


def test_corrupted_index_is_reported_without_crashing(tmp_path):
    idx = build_demo_index(tmp_path)
    (idx / "index.faiss").write_bytes(b"garbage")
    ev = make_service(tmp_path, fake_predictor(), index_dir=idx).analyze(ARTICLE)["components"]["evidence"]
    assert ev["status"] == "unavailable" and "Rebuild" in ev["error"] and str(tmp_path) not in ev["error"]


def test_index_built_with_other_model_is_reported(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = AnalysisService(index_dir=idx, predictor_factory=lambda: fake_predictor(),
                          embedder_factory=lambda: HashingEmbedder(dim=32))
    ev = svc.analyze(ARTICLE)["components"]["evidence"]
    assert ev["status"] == "unavailable" and "different embedding model" in ev["error"]


def test_embedding_failure_is_reported_as_error_status(tmp_path):
    idx = build_demo_index(tmp_path)

    class Broken(HashingEmbedder):
        def embed_query(self, q):
            from src.rag.embeddings import EmbeddingError
            raise EmbeddingError("Could not load embedding model 'x': no internet (C:/secret/cache)")

    svc = AnalysisService(index_dir=idx, predictor_factory=lambda: fake_predictor(),
                          embedder_factory=lambda: Broken())
    # building the index above used HashingEmbedder with the same model_name, so the index loads fine
    out = svc.analyze(ARTICLE)
    ev = out["components"]["evidence"]
    assert ev["status"] == "error" and "secret" not in str(ev)
    assert "embedding model could not be loaded" in ev["claims"][0]["error"]
    assert out["components"]["ml_prediction"]["status"] == "ok"


def test_index_reloads_after_rebuild_without_restart(tmp_path):
    pytest.importorskip("faiss")
    from src.rag.faiss_index import build_or_update_index
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Glaciers are retreating quickly in the north.", encoding="utf-8")
    svc = make_service(tmp_path, fake_predictor(), index_dir=tmp_path / "idx", min_score=0.1)
    assert svc.analyze("Glaciers are retreating")["components"]["evidence"]["status"] == "unavailable"
    build_or_update_index(docs, tmp_path / "idx", HashingEmbedder(), 200, 20)
    ev = svc.analyze("Glaciers are retreating")["components"]["evidence"]
    assert ev["status"] == "ok" and ev["claims"][0]["retrieval_status"] == "evidence_retrieved"
    assert ev["uses_synthetic_demo_documents"] is False  # a user's own documents are not labelled synthetic


def test_rag_can_be_disabled(tmp_path):
    out = make_service(tmp_path, fake_predictor()).analyze(ARTICLE, AnalysisOptions(run_rag=False))
    assert out["components"]["evidence"]["status"] == "skipped"


# ----------------------------------------------------------- output separation
def test_outputs_are_separate_and_no_percentage_or_verdict(tmp_path):
    idx = build_demo_index(tmp_path)
    out = make_service(tmp_path, fake_predictor(), index_dir=idx).analyze(ARTICLE)
    assert set(out["components"]) == {"ml_prediction", "evidence"}
    forbidden = {"credibility_score", "credibility_percentage", "verdict", "is_true", "verified"}
    for blob in (out, out["final_assessment"], *out["components"].values()):
        assert not (forbidden & set(blob))
    assert out["final_assessment"]["is_verdict"] is False
    assert "not a fact-check" in out["disclaimer"]


def test_all_components_failing_still_returns_a_structured_result(tmp_path):
    def boom():
        raise FileNotFoundError("x")

    svc = AnalysisService(index_dir=tmp_path / "none", predictor_factory=boom, embedder_factory=lambda: HashingEmbedder())
    out = svc.analyze(ARTICLE)
    assert out["component_status"]["ml_prediction"] == "unavailable"
    assert out["final_assessment"]["overall_signal"] == "insufficient_information"
    assert out["partial"] is True and out["warnings"]


# ------------------------------------------------------------------------ health
def test_health_all_ok_degraded_and_unavailable(tmp_path):
    idx = build_demo_index(tmp_path)
    ok = make_service(tmp_path, fake_predictor(), index_dir=idx).health()
    assert ok["status"] == "ok" and ok["api"] == "ok"
    assert set(ok["components"]) == {"ml_model", "rag"}          # no Ollama/Llama component any more
    assert ok["components"]["rag"]["num_passages"] > 0 and ok["components"]["rag"]["uses_synthetic_demo_documents"] is True

    def boom():
        raise FileNotFoundError("x")

    degraded = AnalysisService(index_dir=idx, predictor_factory=boom, embedder_factory=lambda: HashingEmbedder()).health()
    assert degraded["status"] == "degraded" and degraded["components"]["ml_model"]["status"] == "unavailable"
    assert degraded["components"]["rag"]["status"] == "ok"

    none_ok = AnalysisService(index_dir=tmp_path / "none", predictor_factory=boom, embedder_factory=lambda: HashingEmbedder()).health()
    assert none_ok["status"] == "unavailable" and none_ok["api"] == "ok"


def test_health_never_raises_and_does_not_leak_internal_errors(tmp_path):
    predictor = fake_predictor()
    predictor.model_info.side_effect = RuntimeError("kaboom")
    h = make_service(tmp_path, predictor).health()
    assert h["api"] == "ok" and h["components"]["ml_model"]["status"] == "ok" and "kaboom" not in str(h)


def test_no_llama_or_ollama_in_service_output(tmp_path):
    idx = build_demo_index(tmp_path)
    svc = make_service(tmp_path, fake_predictor(), index_dir=idx, min_score=0.0)
    blob = (str(svc.analyze(ARTICLE)) + str(svc.health())).lower()
    assert "llama" not in blob and "ollama" not in blob
