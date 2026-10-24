"""
NewsCred dashboard tests.

* render.py: HTML builders (escaping, honest handling of every API status, no verdict wording).
* helpers.call_api_ex: failure kinds.
* app.py: executed with Streamlit's AppTest. API calls are routed to the REAL FastAPI app (via
  TestClient) running on fake components, so the page is checked against the actual endpoints and
  response schema. No model download.

    python -m pytest tests/test_newscred_frontend.py -v -rs
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.frontend import render  # noqa: E402
from app.frontend.helpers import KIND_HTTP_ERROR, KIND_TIMEOUT, KIND_UNREACHABLE, call_api_ex  # noqa: E402

APP_FILE = str(PROJECT_ROOT / "app" / "frontend" / "app.py")

ML_OK = {"status": "ok", "error": None, "notice": "This is a model prediction, not proof.", "result": {
    "label": "Real", "fake_probability": 0.1234, "real_probability": 0.8766, "confidence_level": "high",
    "warnings": ["Very short input"], "top_terms": {"toward_fake": [{"term": "shock", "contribution": 0.2}],
                                                    "toward_real": [{"term": "reuters", "contribution": -0.5}]}}}


def has_no_raw_newlines(markup: str) -> bool:
    return "\n" not in markup


# ----------------------------------------------------------------------------- render.py
def test_esc_neutralises_html_and_none():
    assert render.esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"
    assert render.esc(None) == "" and render.esc(5) == "5"


def test_ml_card_shows_only_returned_scores_and_the_disclaimer():
    out = render.ml_card_html(ML_OK)
    assert "Real" in out and "12.3%" in out and "87.7%" in out and "High" in out
    assert "not proof" in out and "Very short input" in out and "reuters" in out
    assert has_no_raw_newlines(out)


def test_ml_card_with_missing_scores_does_not_invent_numbers():
    comp = {"status": "ok", "result": {"label": "Undetermined", "fake_probability": None, "real_probability": None,
                                       "confidence_level": None}}
    out = render.ml_card_html(comp)
    assert out.count("n/a") >= 2 and "Undetermined" in out


def test_ml_card_unavailable_shows_backend_message():
    out = render.ml_card_html({"status": "unavailable", "error": "Model files not found."})
    assert "Model files not found." in out and "Unavailable" in out and "Predicted label" not in out


def test_llama_card_has_been_removed_from_the_renderer():
    assert not hasattr(render, "llama_card_html")
    page = render.how_it_works_html() + render.about_html()
    assert "llama" not in page.lower() and "ollama" not in page.lower()


def evidence_comp(status, **extra):
    base = {"status": "ok", "notice": "n", "claim_source": "heuristic", "notes": [], "claims": [], "uses_synthetic_demo_documents": False}
    base.update(extra)
    base["claims"] = [{"claim": "The levee was built in 1990.", "retrieval_status": status, **base.get("claim_extra", {})}]
    return base


def test_evidence_card_with_passages_shows_source_page_and_labelled_similarity():
    comp = evidence_comp("evidence_retrieved", claim_extra={"retrieved_evidence": [
        {"text": "The levee was completed in 1990.", "source": "report.pdf", "page": 3, "chunk_id": 7, "rank": 1,
         "score": 0.4567, "is_synthetic_demo": True}]}, uses_synthetic_demo_documents=True)
    out = render.evidence_card_html(comp)
    assert "The levee was completed in 1990." in out and "report.pdf" in out and "page 3" in out
    assert "cosine similarity 0.457" in out and "fictional demo document" in out
    assert "not real-world evidence" in out and "not probabilities of truth" in out


def test_evidence_card_omits_page_when_unknown():
    comp = evidence_comp("evidence_retrieved", claim_extra={"retrieved_evidence": [
        {"text": "t", "source": "a.txt", "page": None, "chunk_id": 0, "rank": 1, "score": 0.5}]})
    out = render.evidence_card_html(comp)
    assert "page" not in out.split("source:")[1].split("passage")[0]


def test_evidence_card_no_relevant_evidence_message_is_readable_and_not_a_verdict():
    out = render.evidence_card_html(evidence_comp("no_relevant_evidence", claim_extra={"retrieved_evidence": []}))
    assert "No sufficiently relevant passage" in out and "does not mean the claim is true or false" in out


def test_evidence_card_unavailable_and_skipped():
    down = render.evidence_card_html({"status": "unavailable", "error": "No evidence index was found.", "claims": []})
    assert "No evidence index was found." in down
    assert "Disabled" in render.evidence_card_html({"status": "skipped", "detail": "Disabled for this request."})


def test_assessment_card_is_not_a_verdict():
    fa = {"overall_signal": "few_warning_signs_detected", "headline": "Fewer automated warning signs.",
          "observations": ["ML said Real"], "notes": [], "cannot_conclude": ["Whether claims are true"],
          "suggested_next_steps": ["Check the source"], "basis": "Based on two components.", "is_verdict": False}
    out = render.assessment_card_html(fa)
    assert "Fewer warning signs detected" in out and "not a verdict" in out and "Check the source" in out
    lowered = out.lower()
    for forbidden in ("verified", "is true", "is false", "credibility score:"):
        assert forbidden not in lowered.replace("not a verdict and not a credibility score", "")


def test_static_pages_do_not_overclaim():
    for page in (render.how_it_works_html(), render.about_html(), render.header_html()):
        low = page.lower()
        assert "fact-check" in low or "not proof" in low or "does not" in low
        assert "100% accurate" not in low and "guarantee" not in low
    assert "NewsCred" in render.header_html().replace("News<span>Cred</span>", "NewsCred")
    assert "AI-assisted news analysis and evidence exploration" in render.header_html()


# ----------------------------------------------------------------------------- helpers.call_api_ex
def test_call_api_ex_failure_kinds():
    with patch("app.frontend.helpers.requests.request", side_effect=requests.exceptions.ConnectionError("x")):
        data, msg, kind = call_api_ex("GET", "/health")
    assert data is None and kind == KIND_UNREACHABLE and "uvicorn app.backend.main:app" in msg
    with patch("app.frontend.helpers.requests.request", side_effect=requests.exceptions.Timeout("x")):
        assert call_api_ex("GET", "/health", timeout=3)[2] == KIND_TIMEOUT
    bad = MagicMock(status_code=422)
    bad.json.return_value = {"message": "empty"}
    with patch("app.frontend.helpers.requests.request", return_value=bad):
        assert call_api_ex("POST", "/analyze", {})[1:] == ("empty", KIND_HTTP_ERROR)
    good = MagicMock(status_code=200)
    good.json.return_value = {"a": 1}
    with patch("app.frontend.helpers.requests.request", return_value=good):
        assert call_api_ex("GET", "/health") == ({"a": 1}, None, None)


# ----------------------------------------------------------------------------- app.py via AppTest
pytest.importorskip("streamlit")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from app.backend.main import create_app  # noqa: E402
from tests.test_phase8_service import ARTICLE, fake_predictor, make_service  # noqa: E402


class Router:
    """Stands in for call_api_ex: sends the page's requests to the real FastAPI app and records them."""

    def __init__(self, client):
        self.client, self.calls = client, []

    def __call__(self, method, path, payload=None, timeout=30.0, base_url=None):
        self.calls.append((method, path, payload))
        resp = self.client.request(method, path, json=payload)
        data = resp.json()
        if resp.status_code >= 400:
            return None, data.get("message"), KIND_HTTP_ERROR
        return data, None, None


def real_api_router(tmp_path, with_index=False, **kwargs):
    index_dir = None
    if with_index:
        pytest.importorskip("faiss")
        from src.rag.config import SAMPLE_DOCS_DIR
        from src.rag.embeddings import HashingEmbedder
        from src.rag.faiss_index import build_or_update_index
        build_or_update_index(SAMPLE_DOCS_DIR, tmp_path / "idx", HashingEmbedder(), 300, 60)
        index_dir = tmp_path / "idx"
        kwargs.setdefault("min_score", 0.2)
    service = make_service(tmp_path, kwargs.pop("predictor", None) or fake_predictor(), index_dir=index_dir, **kwargs)
    return Router(TestClient(create_app(service), raise_server_exceptions=False))


def unreachable(*args, **kwargs):
    return None, "Cannot reach the analysis API at http://127.0.0.1:8000. Start it ...", KIND_UNREACHABLE


def run_app(router):
    with patch("app.frontend.helpers.call_api_ex", router):
        at = AppTest.from_file(APP_FILE, default_timeout=60)
        at.run()
        return at, router


def texts(at):
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning] + [c.value for c in at.caption]
    return "\n".join(parts)


def click_analyze(router, at, text):
    with patch("app.frontend.helpers.call_api_ex", router):
        at.text_area(key="input_text").set_value(text)
        next(b for b in at.button if b.label == "Analyze News").click()
        at.run()
    return at


def test_page_loads_with_header_tabs_and_primary_button(tmp_path):
    at, router = run_app(real_api_router(tmp_path))
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Analyze News", "How It Works", "Evaluation", "About"]
    assert any(b.label == "Analyze News" for b in at.button)
    assert "AI-assisted news analysis and evidence exploration" in texts(at)
    assert ("GET", "/health", None) in router.calls
    assert "Analysis service connected" in texts(at)


def test_advanced_options_are_inside_an_expander_with_safe_defaults(tmp_path):
    at, _ = run_app(real_api_router(tmp_path))
    assert any(e.label == "Advanced options" for e in at.expander)
    assert [c.value for c in at.checkbox] == [True]          # only 'Retrieve evidence passages' remains
    assert at.slider[0].value == 3


def test_empty_input_shows_a_message_and_makes_no_analyze_call(tmp_path):
    at, router = run_app(real_api_router(tmp_path))
    for blank in ("", "   \n  "):
        click_analyze(router, at, blank)
        assert any("paste a headline or article" in e.value.lower() for e in at.error)
    assert not any(c[:2] == ("POST", "/analyze") for c in router.calls)
    assert not at.exception


def test_backend_unavailable_state_is_friendly_and_never_fakes_results(tmp_path):
    at, _ = run_app(unreachable)
    assert not at.exception
    page = texts(at)
    assert "analysis service is not running" in page
    assert any(b.label == "Retry connection" for b in at.button)
    assert any(e.label == "Troubleshooting details" for e in at.expander)
    assert "Analysis service connected" not in page
    click_analyze(unreachable, at, ARTICLE)
    assert any("not reachable" in e.value for e in at.error)
    assert "Machine learning prediction" not in texts(at)  # no results, no fake demo results


def test_retry_connection_recovers_when_the_service_comes_up(tmp_path):
    at, _ = run_app(unreachable)
    router = real_api_router(tmp_path)
    with patch("app.frontend.helpers.call_api_ex", router):
        next(b for b in at.button if b.label == "Retry connection").click()
        at.run()
    assert "Analysis service connected" in texts(at)
    assert not any(b.label == "Retry connection" for b in at.button)


def test_example_selection_fills_the_text_area(tmp_path):
    at, _ = run_app(real_api_router(tmp_path))
    with patch("app.frontend.helpers.call_api_ex", real_api_router(tmp_path)):
        at.selectbox(key="example_choice").select("Short headline (fictional topic)")
        at.run()
    assert "Alder River levee" in at.text_area(key="input_text").value


def test_results_are_rendered_from_the_real_api_response(tmp_path):
    router = real_api_router(tmp_path, with_index=True)
    at, _ = run_app(router)
    at.text_area[1].set_value("The Alder River levee was built in 1990.")   # claims to search for (Advanced options)
    click_analyze(router, at, ARTICLE)
    assert not at.exception
    post = [c for c in router.calls if c[:2] == ("POST", "/analyze")]
    assert len(post) == 1 and post[0][2]["text"] == ARTICLE
    assert post[0][2]["run_rag"] is True and post[0][2]["top_k"] == 3 and "run_llama" not in post[0][2]
    assert post[0][2]["claims"] == ["The Alder River levee was built in 1990."]

    page = texts(at)
    for expected in ("Machine learning prediction", "Retrieved evidence", "Overall assessment",
                     "The Alder River levee was built in 1990.", "10.0%"):
        assert expected in page
    assert "llama" not in page.lower() and "ollama" not in page.lower()
    # the fake predictor returns no real_probability, so the page must show n/a rather than derive 90%
    assert "90.0%" not in page and "n/a" in page
    assert "cosine similarity" in page and ".txt" in page          # real retrieved passage metadata
    assert "not a verdict" in page


def test_advanced_options_are_sent_to_the_api(tmp_path):
    router = real_api_router(tmp_path)
    at, _ = run_app(router)
    at.checkbox[0].set_value(False)
    at.slider[0].set_value(5)
    click_analyze(router, at, ARTICLE)
    payload = [c for c in router.calls if c[:2] == ("POST", "/analyze")][0][2]
    assert payload["run_rag"] is False and payload["top_k"] == 5
    assert "Disabled for this request" in texts(at)


def test_partial_result_when_the_ml_model_fails_shows_reason_and_other_cards(tmp_path):
    broken = fake_predictor()
    broken.predict.side_effect = RuntimeError("boom")
    router = real_api_router(tmp_path, predictor=broken)
    at, _ = run_app(router)
    click_analyze(router, at, ARTICLE)
    page = texts(at)
    assert not at.exception
    assert "partial result" in page and "The ML prediction failed unexpectedly." in page
    assert "Retrieved evidence" in page and "Overall assessment" in page and "boom" not in page


def test_api_validation_error_is_shown_not_a_stack_trace(tmp_path):
    router = real_api_router(tmp_path)
    at, _ = run_app(router)
    click_analyze(router, at, "x" * 5)  # valid; then force a backend error through an oversized payload
    with patch("app.frontend.helpers.call_api_ex", lambda *a, **k: (None, "The text is too long (20,001 characters).", KIND_HTTP_ERROR)):
        at.text_area(key="input_text").set_value("y" * 50)
        next(b for b in at.button if b.label == "Analyze News").click()
        at.run()
    assert any("too long" in e.value for e in at.error) and not at.exception
