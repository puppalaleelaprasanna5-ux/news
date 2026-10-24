"""Phase 8: final integration (analysis service and cautious assessment)."""

from src.credibility.assessment import build_final_assessment
from src.credibility.service import (
    AnalysisOptions,
    AnalysisService,
    InputValidationError,
    MAX_INPUT_CHARS,
    validate_input_text,
)

__all__ = ["AnalysisOptions", "AnalysisService", "InputValidationError", "MAX_INPUT_CHARS",
           "build_final_assessment", "validate_input_text"]
