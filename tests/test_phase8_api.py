"""
Tests: FastAPI endpoints (/health, /analyze) via Starlette's TestClient.

Requires fastapi and httpx (both in requirements.txt). If they are not installed these tests are
SKIPPED - check the pytest summary (-rs) so skipped API tests are not mistaken for passing ones.
Components are replaced by fakes: no model download.

    python -m pytest tests/test_phase8_api.py -v -rs
"""

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import create_app  # noqa: E402
from src.credibility.service import MAX_CLAIMS, MAX_INPUT_CHARS  # noqa: E402
from tests.test_phase8_service import ARTICLE, fake_predictor, make_service  # noqa: E402


def client_for(tmp_path, predictor=None):
    service = make_service(tmp_path, predictor or fake_predictor())
    return TestClient(create_app(service), raise_server_exceptions=False), service


def test_health_endpoint_reports_components(tmp_path):
    client, _ = client_for(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["api"] == "ok" and body["status"] in ("ok", "degraded", "unavailable")
    assert set(body["components"]) == {"ml_model", "rag"}
    assert body["components"]["ml_model"]["status"] == "ok"
    assert body["components"]["rag"]["status"] == "unavailable"  # no index in tmp_path


def test_health_still_answers_when_everything_is_down(tmp_path):
    def boom():
        raise FileNotFoundError("x")

    from src.credibility.service import AnalysisService
    from src.rag.embeddings import HashingEmbedder
    svc = AnalysisService(index_dir=tmp_path / "none", predictor_factory=boom,
                          embedder_factory=lambda: HashingEmbedder())
    resp = TestClient(create_app(svc), raise_server_exceptions=False).get("/health")
    assert resp.status_code == 200 and resp.json()["status"] == "unavailable"


def test_analyze_valid_input_returns_structured_json(tmp_path):
    client, _ = client_for(tmp_path)
    resp = client.post("/analyze", json={"text": ARTICLE})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["components"]) == {"ml_prediction", "evidence"}
    assert body["components"]["ml_prediction"]["status"] == "ok"
    assert "llama" not in resp.text.lower() and "ollama" not in resp.text.lower()
    assert body["final_assessment"]["is_verdict"] is False
    structured = body["credibility_assessment"]
    assert set(structured) == {"machine_learning", "article_claims", "overall"}
    assert structured["machine_learning"]["probabilities_calibrated"] is False
    assert body["components"]["ml_prediction"]["result"]["probabilities_calibrated"] is False
    assert structured["article_claims"]["status"] == body["components"]["evidence"]["status"]
    assert structured["overall"]["is_verdict"] is False
    assert "disclaimer" in body and body["input"]["word_count"] > 10


def test_analyze_response_model_is_published_in_openapi(tmp_path):
    client, _ = client_for(tmp_path)

    schema = client.get("/openapi.json").json()

    assert schema["paths"]["/analyze"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert "AnalyzeResponse" in schema["components"]["schemas"]
    assert "ValidationErrorResponse" in schema["components"]["schemas"]


def test_analyze_response_model_preserves_evidence_metadata(tmp_path):
    client, service = client_for(tmp_path)
    passage = {
        "text": "A fictional source passage.",
        "source": "demo.txt",
        "source_id": "demo.txt",
        "page": 2,
        "chunk_id": 4,
        "rank": 1,
        "score": 0.82,
        "score_type": "cosine_similarity",
    }
    retriever = SimpleNamespace(
        index=SimpleNamespace(size=1, model_name="test-embedder"),
        min_score=0.35,
        retrieve_many=lambda claims, top_k: [{
            "claim": claims[0],
            "retrieval_status": "evidence_retrieved",
            "retrieved_evidence": [passage.copy()],
            "limitations": ["Similarity is not support."],
            "parameters": {"top_k": top_k, "min_score": 0.35, "score_type": "cosine_similarity"},
        }],
    )
    service._synthetic_ids = {"demo.txt"}
    service._get_retriever = lambda: (retriever, None)

    response = client.post("/analyze", json={"text": "The Alder River levee was completed in 1990."})

    assert response.status_code == 200
    body = response.json()
    claim = body["components"]["evidence"]["claims"][0]
    assert claim["retrieved_evidence"][0]["source"] == "demo.txt"
    assert claim["retrieved_evidence"][0]["is_synthetic_demo"] is True
    structured_claim = body["credibility_assessment"]["article_claims"]["items"][0]
    assert structured_claim["passages"][0]["metadata"]["page"] == 2
    assert structured_claim["evidence_interpretation"]["classification"] == "unclassified"


def test_analyze_rejects_empty_and_whitespace_input(tmp_path):
    client, _ = client_for(tmp_path)
    for text in ("", "   \n\t "):
        resp = client.post("/analyze", json={"text": text})
        body = resp.json()
        assert resp.status_code == 422
        assert body == {
            "error": "invalid_input",
            "message": "The request was not valid.",
            "problems": [{"field": "text", "message": "Please provide a headline or article text; the input is empty."}],
        }


def test_analyze_rejects_too_long_input_without_echoing_it(tmp_path):
    client, _ = client_for(tmp_path)
    resp = client.post("/analyze", json={"text": "a" * (MAX_INPUT_CHARS + 1)})
    assert resp.status_code == 422
    assert resp.json()["error"] == "invalid_input"
    assert resp.json()["problems"][0]["field"] == "text"
    assert "too long" in resp.json()["problems"][0]["message"]
    huge = client.post("/analyze", json={"text": "zq" * MAX_INPUT_CHARS * 2})
    assert huge.status_code == 422
    assert huge.json()["error"] == "invalid_input"
    assert "zqzq" not in huge.text  # input is never echoed back


def test_analyze_rejects_malformed_requests(tmp_path):
    client, _ = client_for(tmp_path)
    assert client.post("/analyze", json={}).status_code == 422
    assert client.post("/analyze", json={"text": 123}).status_code == 422
    assert client.post("/analyze", json={"text": "ok text", "top_k": 99}).status_code == 422
    assert client.post("/analyze", content="not json", headers={"content-type": "application/json"}).status_code == 422
    body = client.post("/analyze", json={}).json()
    assert set(body) == {"error", "message", "problems"}
    assert body["error"] == "invalid_input" and body["problems"]
    assert all(set(problem) == {"field", "message"} for problem in body["problems"])


def test_service_and_pydantic_validation_use_the_same_safe_error_shape(tmp_path):
    client, _ = client_for(tmp_path)
    service_error = client.post("/analyze", json={"text": ""})
    submitted_secret = "private-article-" + ("x" * (MAX_INPUT_CHARS * 2))
    pydantic_error = client.post("/analyze", json={"text": submitted_secret})

    assert service_error.status_code == pydantic_error.status_code == 422
    assert set(service_error.json()) == set(pydantic_error.json()) == {
        "error", "message", "problems"
    }
    assert service_error.json()["error"] == pydantic_error.json()["error"] == "invalid_input"
    assert all(
        set(problem) == {"field", "message"}
        for response in (service_error, pydantic_error)
        for problem in response.json()["problems"]
    )
    assert "private-article" not in pydantic_error.text


def test_analyze_bounds_manual_claim_count_and_length(tmp_path):
    client, _ = client_for(tmp_path)
    accepted = client.post(
        "/analyze",
        json={
            "text": ARTICLE,
            "claims": ["claim " + str(i) for i in range(MAX_CLAIMS)],
            "run_rag": False,
        },
    )
    assert accepted.status_code == 200

    too_many = client.post(
        "/analyze",
        json={
            "text": ARTICLE,
            "claims": ["claim " + str(i) for i in range(MAX_CLAIMS + 1)],
        },
    )
    assert too_many.status_code == 422
    assert too_many.json()["error"] == "invalid_input"

    long_claim = "private-claim " + ("x" * MAX_INPUT_CHARS)
    too_long = client.post(
        "/analyze",
        json={"text": ARTICLE, "claims": [long_claim]},
    )
    assert too_long.status_code == 422
    assert too_long.json()["error"] == "invalid_input"
    assert "private-claim" not in too_long.text


def test_analyze_partial_result_when_the_ml_model_is_down(tmp_path):
    from src.credibility.service import AnalysisService
    from src.rag.embeddings import HashingEmbedder

    def boom():
        raise FileNotFoundError("best_model.joblib")

    svc = AnalysisService(models_dir=tmp_path, index_dir=tmp_path / "none", predictor_factory=boom,
                          embedder_factory=lambda: HashingEmbedder())
    resp = TestClient(create_app(svc), raise_server_exceptions=False).post("/analyze", json={"text": ARTICLE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["partial"] is True and body["components"]["ml_prediction"]["status"] == "unavailable"
    assert body["credibility_assessment"]["machine_learning"]["status"] == "unavailable"
    assert body["components"]["ml_prediction"]["result"] is None
    assert "Traceback" not in resp.text


def test_analyze_response_model_accepts_unscorable_prediction(tmp_path):
    predictor = fake_predictor()
    predictor.predict.return_value = {
        "label": "Undetermined",
        "label_id": None,
        "fake_probability": None,
        "real_probability": None,
        "probabilities_calibrated": None,
        "confidence": None,
        "confidence_level": None,
        "n_words": 4,
        "n_known_terms": 0,
        "warnings": ["No known vocabulary."],
        "model_name": "fake",
        "disclaimer": "Not verified truth.",
    }
    client, _ = client_for(tmp_path, predictor)

    response = client.post("/analyze", json={"text": "zzqx qwxz vvkk xxjq"})

    assert response.status_code == 200
    body = response.json()
    result = body["components"]["ml_prediction"]["result"]
    assert result["label"] == "Undetermined"
    assert result["fake_probability"] is None
    assert result["probabilities_calibrated"] is None
    assert body["credibility_assessment"]["machine_learning"]["predicted_label"] is None
    assert body["credibility_assessment"]["machine_learning"]["probabilities_calibrated"] is None


def test_analyze_options_are_passed_through(tmp_path):
    client, _ = client_for(tmp_path)
    body = client.post("/analyze", json={"text": ARTICLE, "run_rag": False}).json()
    assert body["components"]["evidence"]["status"] == "skipped"
    assert body["components"]["ml_prediction"]["status"] == "ok"


def test_analyze_ignores_the_removed_run_llama_field(tmp_path):
    """Old clients that still send run_llama must not crash; the field is simply ignored."""
    client, _ = client_for(tmp_path)
    resp = client.post("/analyze", json={"text": ARTICLE, "run_llama": False, "run_rag": False})
    assert resp.status_code == 200 and "llama_analysis" not in resp.json()["components"]


def test_unexpected_server_error_is_generic_and_has_no_stack_trace(tmp_path):
    client, service = client_for(tmp_path)

    def explode(*a, **k):
        raise RuntimeError("secret internal detail C:/Users/me/file.py")

    service.analyze = explode
    resp = client.post("/analyze", json={"text": ARTICLE})
    assert resp.status_code == 500
    assert resp.json()["error"] == "internal_error"
    assert "secret" not in resp.text and "Traceback" not in resp.text and "C:/Users" not in resp.text
