"""
Tests: FastAPI endpoints (/health, /analyze) via Starlette's TestClient.

Requires fastapi and httpx (both in requirements.txt). If they are not installed these tests are
SKIPPED - check the pytest summary (-rs) so skipped API tests are not mistaken for passing ones.
Components are replaced by fakes: no model download.

    python -m pytest tests/test_phase8_api.py -v -rs
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.backend.main import create_app  # noqa: E402
from src.credibility.service import MAX_INPUT_CHARS  # noqa: E402
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
    assert "disclaimer" in body and body["input"]["word_count"] > 10


def test_analyze_rejects_empty_and_whitespace_input(tmp_path):
    client, _ = client_for(tmp_path)
    for text in ("", "   \n\t "):
        resp = client.post("/analyze", json={"text": text})
        assert resp.status_code == 422 and resp.json()["error"] == "invalid_input"


def test_analyze_rejects_too_long_input_without_echoing_it(tmp_path):
    client, _ = client_for(tmp_path)
    resp = client.post("/analyze", json={"text": "a" * (MAX_INPUT_CHARS + 1)})
    assert resp.status_code == 422 and "too long" in resp.json()["message"]
    huge = client.post("/analyze", json={"text": "zq" * MAX_INPUT_CHARS * 2})
    assert huge.status_code == 422 and "zqzq" not in huge.text  # input is never echoed back


def test_analyze_rejects_malformed_requests(tmp_path):
    client, _ = client_for(tmp_path)
    assert client.post("/analyze", json={}).status_code == 422
    assert client.post("/analyze", json={"text": 123}).status_code == 422
    assert client.post("/analyze", json={"text": "ok text", "top_k": 99}).status_code == 422
    assert client.post("/analyze", content="not json", headers={"content-type": "application/json"}).status_code == 422
    body = client.post("/analyze", json={}).json()
    assert body["error"] == "invalid_request" and body["problems"]


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
    assert "Traceback" not in resp.text


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
