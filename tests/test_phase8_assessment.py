"""
Tests for the cautious final assessment (pure functions, no models needed).

    python -m pytest tests/test_phase8_assessment.py -v
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.credibility.assessment import (
    SIGNAL_FEW_WARNINGS, SIGNAL_INSUFFICIENT, SIGNAL_MIXED, SIGNAL_WARNING,
    build_final_assessment, build_structured_assessment,
)


def ml(label="Real", level="high", n_words=200, status="ok"):
    if status != "ok":
        return {"status": status, "result": None}
    return {"status": "ok", "result": {"label": label, "confidence_level": level, "n_words": n_words}}


def evidence(status="ok", claims=None, synthetic=False, detail=None):
    return {"status": status, "claims": claims or [], "uses_synthetic_demo_documents": synthetic, "detail": detail}


def assess(m, e=None, words=200):
    return build_final_assessment({"ml_prediction": m, "evidence": e or evidence("skipped")}, words)


def test_ml_fake_prediction_gives_warning_signs():
    out = assess(ml("Fake"))
    assert out["overall_signal"] == SIGNAL_WARNING
    assert any("weaker" in n for n in out["notes"])


def test_ml_real_prediction_is_never_reassuring_on_its_own():
    """Only one component feeds the signal, so 'Real' must not produce a reassuring signal."""
    for level in ("high", "moderate", "low"):
        out = assess(ml("Real", level))
        assert out["overall_signal"] == SIGNAL_INSUFFICIENT
        assert out["overall_signal"] not in (SIGNAL_FEW_WARNINGS, SIGNAL_MIXED)
        assert any("no reassuring signal" in n for n in out["notes"])
    assert any("predicted Real" in o for o in assess(ml("Real"))["observations"])


def test_short_real_headline_adds_a_short_input_note():
    out = assess(ml("Real", "high", n_words=8), words=8)
    assert out["overall_signal"] == SIGNAL_INSUFFICIENT and any("very short" in n for n in out["notes"])


def test_low_confidence_fake_is_flagged_in_notes():
    out = assess(ml("Fake", "low"))
    assert out["overall_signal"] == SIGNAL_WARNING and any("low uncalibrated score band" in n for n in out["notes"])


def test_nothing_usable_is_insufficient_and_says_so():
    out = assess(ml(status="unavailable"), evidence("unavailable", detail="Evidence index unavailable."))
    assert out["overall_signal"] == SIGNAL_INSUFFICIENT
    assert any("not available" in o for o in out["observations"])


def test_undetermined_ml_label_is_not_treated_as_a_prediction():
    out = assess(ml("Undetermined"))
    assert out["overall_signal"] == SIGNAL_INSUFFICIENT
    assert any("could not produce a prediction" in o for o in out["observations"])


def test_no_llama_wording_remains_in_the_assessment():
    out = assess(ml("Fake"), evidence("ok", [{"retrieval_status": "evidence_retrieved"}]))
    text = " ".join(out["observations"] + out["notes"] + [out["basis"], out["headline"]]).lower()
    assert "llama" not in text and "ollama" not in text


def test_evidence_never_changes_the_signal_but_is_described_carefully():
    claims = [{"retrieval_status": "evidence_retrieved"}, {"retrieval_status": "no_relevant_evidence"}]
    for label in ("Real", "Fake"):
        base = assess(ml(label), evidence("skipped"))
        with_ev = assess(ml(label), evidence("ok", claims, synthetic=True))
        assert base["overall_signal"] == with_ev["overall_signal"]
    text = " ".join(with_ev["observations"])
    assert "1 of 2" in text and "similarity, not agreement" in text and "FICTIONAL" in text
    assert any("no retrieved passage" in c for c in with_ev["cannot_conclude"])


def test_assessment_has_no_numeric_score_and_is_not_a_verdict():
    out = assess(ml("Fake"), evidence("ok", [{"retrieval_status": "evidence_retrieved"}]))
    assert out["is_verdict"] is False
    assert not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in out.values())
    assert "No percentage score" in out["basis"]
    assert len(out["suggested_next_steps"]) >= 2


def test_structured_assessment_keeps_prediction_claims_and_observations_distinct():
    components = {
        "ml_prediction": {"status": "ok", "result": {
            "label": "Fake", "fake_probability": 0.82, "real_probability": 0.18,
        }},
        "evidence": {"status": "ok", "claim_source": "manual", "claims": [{
            "claim": "The library opened in 1990.",
            "retrieval_status": "evidence_retrieved",
            "retrieved_evidence": [{
                "text": "The library opened in 1991.", "source": "annual_report.txt",
                "page": 4, "source_id": "annual_report.txt", "chunk_id": 2, "score": 0.77,
            }],
        }]},
    }
    final = build_final_assessment(components, 12)

    assessment = build_structured_assessment(components, final)

    assert assessment["machine_learning"]["predicted_label"] == "Fake"
    assert assessment["machine_learning"]["fake_probability"] == 0.82
    assert assessment["machine_learning"]["probabilities_calibrated"] is False
    claim = assessment["article_claims"]["items"][0]
    assert claim["claim"] == "The library opened in 1990."
    assert claim["evidence_interpretation"]["classification"] == "unclassified"
    assert claim["passages"][0]["source_name"] == "annual_report.txt"
    assert claim["passages"][0]["text"] == "The library opened in 1991."
    assert claim["passages"][0]["metadata"]["page"] == 4
    assert assessment["overall"]["is_verdict"] is False
    assert "uncalibrated score band" in assessment["overall"]["model_observation"]
    assert "score" not in assessment["overall"]


def test_structured_assessment_has_predictable_empty_states_and_validates_probabilities():
    components = {
        "ml_prediction": {"status": "ok", "result": {
            "label": "Undetermined", "fake_probability": float("nan"), "real_probability": 1.2,
        }},
        "evidence": {"status": "unavailable", "claim_source": "none", "claims": [],
                     "error": "Evidence index unavailable."},
    }
    assessment = build_structured_assessment(components, build_final_assessment(components, 3))

    assert assessment["machine_learning"]["predicted_label"] is None
    assert assessment["machine_learning"]["fake_probability"] is None
    assert assessment["machine_learning"]["real_probability"] is None
    assert assessment["machine_learning"]["probabilities_calibrated"] is None
    assert assessment["article_claims"]["items"] == []
    assert assessment["article_claims"]["empty_state"] == "Evidence index unavailable."
    assert assessment["overall"]["is_verdict"] is False
