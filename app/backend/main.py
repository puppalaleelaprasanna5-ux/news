"""
FastAPI backend.

Endpoints
    GET  /health   which components (ML model, RAG index) are available
    POST /analyze  run ML prediction and evidence retrieval for a headline/article

Run from the project root (see README):
    python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000

All the logic lives in src/credibility/service.py; this file is a thin HTTP layer.
Errors never include stack traces: unexpected problems are logged server-side and the client
receives a generic message.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.credibility.service import (
    DEFAULT_RAG_TOP_K,
    MAX_INPUT_CHARS,
    AnalysisOptions,
    AnalysisService,
    InputValidationError,
)

logger = logging.getLogger("fake_news_api")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_INPUT_CHARS * 2, description="Headline or article text.")
    claims: Optional[List[str]] = Field(
        default=None, description="Optional claims to search evidence for; if omitted, claims are selected from the text by rules."
    )
    run_rag: bool = True
    top_k: int = Field(default=DEFAULT_RAG_TOP_K, ge=1, le=10)


def create_app(service: Optional[AnalysisService] = None) -> FastAPI:
    """App factory (tests pass a service with fake components)."""
    svc = service or AnalysisService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the (cheap) ML model once at start-up. Failures are not fatal: the API still starts
        # and reports the problem in /health and in each /analyze result.
        try:
            svc._get_predictor()
        except Exception:  # pragma: no cover - _get_predictor already handles errors
            logger.exception("Warm-up failed")
        yield

    app = FastAPI(
        title="Fake News Detection and Credibility Analysis API",
        description="Local, experimental analysis. Not a fact-checking service.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.service = svc

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Do not echo the submitted input back (it can be very large); report only where/what failed.
        problems = [
            {"field": ".".join(str(p) for p in err.get("loc", []) if p != "body"), "message": err.get("msg", "invalid value")}
            for err in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"error": "invalid_request", "message": "The request was not valid.", "problems": problems})

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error while serving %s", request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal_error", "message": "Something went wrong on the server. Please try again."})

    @app.get("/health")
    def health():
        # Plain `def` -> run in FastAPI's thread pool, so slow checks never block the event loop.
        return svc.health()

    @app.post("/analyze")
    def analyze(request: AnalyzeRequest):
        try:
            return svc.analyze(
                request.text,
                AnalysisOptions(run_rag=request.run_rag, manual_claims=request.claims, top_k=request.top_k),
            )
        except InputValidationError as err:
            return JSONResponse(status_code=422, content={"error": "invalid_input", "message": str(err)})

    return app


app = create_app()
