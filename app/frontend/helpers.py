"""
Streamlit-free helpers for the dashboard (so they can be unit-tested without Streamlit).

* call_api(): talks to the FastAPI backend and turns every failure into a short message.
* small formatting helpers used by app/frontend/app.py.
"""

import os
import re
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()  # lets BACKEND_URL / BACKEND_HOST / BACKEND_PORT be set in a local .env file (never overrides real env vars)

BACKEND_URL = os.getenv("BACKEND_URL") or f"http://{os.getenv('BACKEND_HOST', '127.0.0.1')}:{os.getenv('BACKEND_PORT', '8000')}"
# Must be longer than the embedding-model / index start-up time on the first request.
ANALYZE_TIMEOUT = float(os.getenv("ANALYZE_TIMEOUT", "300"))
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "30"))

STATUS_STYLES = {
    "ok": ("Available", "#1b7f3b"),
    "skipped": ("Skipped", "#6b7280"),
    "unavailable": ("Unavailable", "#b45309"),
    "timeout": ("Timed out", "#b45309"),
    "error": ("Error", "#b91c1c"),
}

SIGNAL_STYLES = {
    "insufficient_information": ("Not enough information", "#6b7280"),
    "mixed_signals": ("Mixed signals", "#b45309"),
    "warning_signs_detected": ("Warning signs detected", "#b91c1c"),
    "few_warning_signs_detected": ("Fewer warning signs detected", "#1b7f3b"),
}

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>$~])")


def md_escape(text: Any) -> str:
    """Escape Markdown/LaTeX special characters so untrusted text (LLM output, passages) renders literally."""
    return _MD_SPECIAL.sub(r"\\\1", str(text if text is not None else ""))


def status_style(status: str) -> Tuple[str, str]:
    return STATUS_STYLES.get(status, (str(status).title(), "#6b7280"))


def signal_style(signal: str) -> Tuple[str, str]:
    return SIGNAL_STYLES.get(signal, (str(signal).replace("_", " ").title(), "#6b7280"))


def format_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# Reasons returned by call_api_ex() so the page can tell "service not running" apart from other problems.
KIND_UNREACHABLE = "unreachable"
KIND_TIMEOUT = "timeout"
KIND_HTTP_ERROR = "http_error"
KIND_BAD_RESPONSE = "bad_response"

API_START_COMMAND = "python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000"


def call_api_ex(method: str, path: str, payload: Optional[dict] = None, timeout: float = 30.0,
                base_url: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Call the backend. Returns (json_dict, None, None) on success or (None, message, kind) on any failure,
    where `kind` is one of the KIND_* constants. Messages never contain stack traces.
    """
    url = (base_url or BACKEND_URL).rstrip("/") + path
    try:
        resp = requests.request(method, url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return None, (f"Cannot reach the analysis API at {base_url or BACKEND_URL}. Start it in another terminal with: "
                      f"{API_START_COMMAND}"), KIND_UNREACHABLE
    except requests.exceptions.Timeout:
        return None, f"The analysis API did not answer within {int(timeout)} seconds.", KIND_TIMEOUT
    except requests.exceptions.RequestException:
        return None, "The request to the analysis API failed.", KIND_UNREACHABLE

    try:
        data = resp.json()
    except ValueError:
        return None, f"The API returned an unexpected response (HTTP {resp.status_code}).", KIND_BAD_RESPONSE

    if resp.status_code >= 400:
        message = data.get("message") if isinstance(data, dict) else None
        problems = data.get("problems") if isinstance(data, dict) else None
        if problems:
            message = (message or "Invalid request.") + " " + "; ".join(
                f"{p.get('field') or 'input'}: {p.get('message')}" for p in problems if isinstance(p, dict))
        return None, message or f"The API returned an error (HTTP {resp.status_code}).", KIND_HTTP_ERROR
    if not isinstance(data, dict):
        return None, "The API returned an unexpected response.", KIND_BAD_RESPONSE
    return data, None, None


def call_api(method: str, path: str, payload: Optional[dict] = None, timeout: float = 30.0,
             base_url: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Call the backend. Returns (json_dict, None) on success or (None, user_friendly_message) on any failure.
    """
    data, message, _kind = call_api_ex(method, path, payload, timeout, base_url)
    return data, message


def parse_manual_claims(raw: str, limit: int = 5):
    """One claim per line; blank lines ignored."""
    return [line.strip() for line in (raw or "").splitlines() if line.strip()][:limit]
