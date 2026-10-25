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
from typing import Annotated, Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from fastapi.middleware.cors import CORSMiddleware


from src.credibility.service import (
    DEFAULT_RAG_TOP_K,
    MAX_CLAIMS,
    MAX_INPUT_CHARS,
    AnalysisOptions,
    AnalysisService,
    InputValidationError,
)

logger = logging.getLogger("fake_news_api")


class AnalyzeRequest(BaseModel):
    text: str = Field(..., max_length=MAX_INPUT_CHARS * 2, description="Headline or article text.")
    claims: Optional[List[Annotated[str, Field(max_length=MAX_INPUT_CHARS)]]] = Field(
        default=None,
        max_length=MAX_CLAIMS,
        description=(
            f"Optional claims to search for (at most {MAX_CLAIMS}, each at most {MAX_INPUT_CHARS} characters); "
            "if omitted, claims are selected from the text by rules."
        ),
    )
    run_rag: bool = True
    top_k: int = Field(default=DEFAULT_RAG_TOP_K, ge=1, le=10)


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class PredictionResult(_ResponseModel):
    label: Optional[str] = None
    label_id: Optional[int] = None
    fake_probability: Optional[float] = None
    real_probability: Optional[float] = None
    probabilities_calibrated: Optional[bool] = None
    confidence: Optional[float] = None
    confidence_level: Optional[str] = None
    n_words: Optional[int] = None
    n_known_terms: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)
    model_name: Optional[str] = None
    disclaimer: Optional[str] = None


class EvidencePassage(_ResponseModel):
    text: Optional[str] = None
    source: Optional[str] = None
    source_id: Optional[str] = None
    page: Optional[Any] = None
    chunk_id: Optional[int] = None
    rank: Optional[int] = None
    score: Optional[float] = None
    score_type: Optional[str] = None
    is_synthetic_demo: Optional[bool] = None


class EvidenceInterpretation(_ResponseModel):
    classification: str
    reason: str


class RetrievedClaim(_ResponseModel):
    claim: Optional[str] = None
    retrieval_status: str
    retrieved_evidence: List[EvidencePassage] = Field(default_factory=list)
    evidence_interpretation: Optional[EvidenceInterpretation] = None
    limitations: List[str] = Field(default_factory=list)
    parameters: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AnalysisComponent(_ResponseModel):
    status: str
    error: Optional[str] = None
    notice: Optional[str] = None
    result: Optional[PredictionResult] = None
    model_name: Optional[str] = None
    detail: Optional[str] = None
    claim_source: Optional[str] = None
    claims: List[RetrievedClaim] = Field(default_factory=list)
    uses_synthetic_demo_documents: Optional[bool] = None
    notes: List[str] = Field(default_factory=list)
    reference_collection: Optional[Dict[str, Any]] = None


class InputSummary(_ResponseModel):
    char_count: int
    word_count: int
    input_kind: str


class FinalAssessment(_ResponseModel):
    overall_signal: Optional[str] = None
    headline: Optional[str] = None
    observations: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    cannot_conclude: List[str] = Field(default_factory=list)
    suggested_next_steps: List[str] = Field(default_factory=list)
    is_verdict: bool
    basis: Optional[str] = None


class MachineLearningAssessment(_ResponseModel):
    status: str
    predicted_label: Optional[str] = None
    fake_probability: Optional[float] = None
    real_probability: Optional[float] = None
    probabilities_calibrated: Optional[bool] = None
    explanation: str


class ArticleClaimAssessment(_ResponseModel):
    claim: Optional[str] = None
    retrieval_status: str
    passages: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_interpretation: EvidenceInterpretation


class ArticleClaimsAssessment(_ResponseModel):
    status: str
    claim_source: str
    items: List[ArticleClaimAssessment] = Field(default_factory=list)
    empty_state: Optional[str] = None


class OverallAssessment(_ResponseModel):
    summary: Optional[str] = None
    model_observation: str
    evidence_observation: str
    limitations: List[str] = Field(default_factory=list)
    limitation_statement: str
    is_verdict: bool


class CredibilityAssessment(_ResponseModel):
    machine_learning: MachineLearningAssessment
    article_claims: ArticleClaimsAssessment
    overall: OverallAssessment


class AnalyzeResponse(_ResponseModel):
    input: InputSummary
    components: Dict[str, AnalysisComponent]
    component_status: Dict[str, str]
    partial: bool
    warnings: List[str]
    final_assessment: FinalAssessment
    credibility_assessment: CredibilityAssessment
    limitations: List[str]
    disclaimer: str


class ValidationProblem(BaseModel):
    field: str
    message: str


class ValidationErrorResponse(BaseModel):
    error: str = "invalid_input"
    message: str = "The request was not valid."
    problems: List[ValidationProblem]


def _validation_response(problems: List[dict]) -> JSONResponse:
    body = ValidationErrorResponse(
        problems=[ValidationProblem(**problem) for problem in problems]
    )
    return JSONResponse(status_code=422, content=body.model_dump())


def _safe_validation_message(error: dict) -> str:
    error_type = error.get("type")
    if error_type == "missing":
        return "This field is required."
    if error_type == "string_type":
        return "This value must be a string."
    if error_type == "string_too_long":
        return "This value exceeds the maximum allowed length."
    if error_type == "list_too_long":
        return "This list exceeds the maximum number of items."
    if error_type == "json_invalid":
        return "The request body is not valid JSON."
    return "This value is invalid."


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

    app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Do not echo the submitted input back (it can be very large); report only where/what failed.
        problems = [
            {
                "field": ".".join(str(p) for p in err.get("loc", []) if p != "body") or "body",
                "message": _safe_validation_message(err),
            }
            for err in exc.errors()
        ]
        return _validation_response(problems)

    @app.exception_handler(Exception)
    async def unexpected_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error while serving %s", request.url.path)
        return JSONResponse(status_code=500, content={"error": "internal_error", "message": "Something went wrong on the server. Please try again."})

    @app.get("/health")
    def health():
        # Plain `def` -> run in FastAPI's thread pool, so slow checks never block the event loop.
        return svc.health()

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        response_model_exclude_unset=True,
        responses={422: {"model": ValidationErrorResponse}},
    )
    def analyze(request: AnalyzeRequest):
        try:
            return svc.analyze(
                request.text,
                AnalysisOptions(run_rag=request.run_rag, manual_claims=request.claims, top_k=request.top_k),
            )
        except InputValidationError as err:
            return _validation_response([{"field": "text", "message": str(err)}])

    return app


app = create_app()
