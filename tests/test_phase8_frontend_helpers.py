"""
Phase 8 tests: Streamlit-free dashboard helpers (API client error handling, formatting).
The Streamlit page itself (app/frontend/app.py) is not covered by automated tests.

    python -m pytest tests/test_phase8_frontend_helpers.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.frontend.helpers import (
    call_api, format_percent, md_escape, parse_manual_claims, signal_style, status_style,
)


def response(status=200, body=None, json_error=False):
    r = MagicMock()
    r.status_code = status
    if json_error:
        r.json.side_effect = ValueError("no json")
    else:
        r.json.return_value = body
    return r


def test_call_api_success():
    with patch("app.frontend.helpers.requests.request", return_value=response(200, {"status": "ok"})) as req:
        data, err = call_api("GET", "/health", base_url="http://x:1")
    assert data == {"status": "ok"} and err is None
    assert req.call_args.args[:2] == ("GET", "http://x:1/health")


def test_call_api_connection_error_has_start_command():
    with patch("app.frontend.helpers.requests.request", side_effect=requests.exceptions.ConnectionError("refused")):
        data, err = call_api("GET", "/health")
    assert data is None and "uvicorn app.backend.main:app" in err


def test_call_api_timeout_and_other_request_errors():
    with patch("app.frontend.helpers.requests.request", side_effect=requests.exceptions.Timeout("slow")):
        assert "did not answer within 5 seconds" in call_api("POST", "/analyze", {}, timeout=5)[1]
    with patch("app.frontend.helpers.requests.request", side_effect=requests.exceptions.RequestException("odd")):
        assert call_api("GET", "/health")[0] is None


def test_call_api_http_errors_use_the_api_message_without_stack_traces():
    body = {"error": "invalid_input", "message": "Please provide a headline or article text; the input is empty."}
    with patch("app.frontend.helpers.requests.request", return_value=response(422, body)):
        data, err = call_api("POST", "/analyze", {"text": ""})
    assert data is None and err == body["message"]
    with patch("app.frontend.helpers.requests.request", return_value=response(500, {"error": "internal_error", "message": "Something went wrong"})):
        assert call_api("POST", "/analyze", {})[1] == "Something went wrong"
    with patch("app.frontend.helpers.requests.request", return_value=response(502, json_error=True)):
        assert "HTTP 502" in call_api("GET", "/health")[1]
    with patch("app.frontend.helpers.requests.request", return_value=response(200, ["not", "a", "dict"])):
        assert call_api("GET", "/health")[0] is None


def test_call_api_validation_problems_are_listed():
    body = {"message": "The request was not valid.", "problems": [{"field": "top_k", "message": "must be <= 10"}]}
    with patch("app.frontend.helpers.requests.request", return_value=response(422, body)):
        assert "top_k: must be <= 10" in call_api("POST", "/analyze", {})[1]


def test_md_escape_neutralises_markdown_and_latex():
    out = md_escape("**bold** [link](http://x) $x$ <b>")
    assert "**" not in out.replace("\\*", "") and "$" not in out.replace("\\$", "")
    assert md_escape(None) == "" and md_escape(5) == "5"


def test_formatting_helpers():
    assert format_percent(0.9312) == "93.1%" and format_percent(None) == "n/a"
    assert status_style("ok")[0] == "Available" and status_style("weird")[0] == "Weird"
    assert signal_style("few_warning_signs_detected")[0] == "Fewer warning signs detected"
    assert parse_manual_claims(" a \n\n b \n c", limit=2) == ["a", "b"] and parse_manual_claims(None) == []
