"""
Final cautious assessment.

Turns the component outputs into a short, plain-language synthesis WITHOUT producing a
credibility percentage and WITHOUT claiming anything is verified.

Rules (deliberately simple and transparent):

* The overall signal is based on the ML prediction ONLY. Retrieved evidence is described but
  NEVER changes the overall signal: similarity is not support, and this tool does not judge
  whether a passage supports or contradicts a claim.
* Because only one component feeds the signal, the strongest possible outcomes are
  "warning_signs_detected" (the ML model predicted Fake) or "insufficient_information".
  A "Real" prediction never produces a reassuring signal on its own: the model was trained on
  one dataset and loses accuracy on others (see the Evaluation tab).
* Nothing usable -> "insufficient_information".

overall_signal values:
    insufficient_information | mixed_signals | warning_signs_detected | few_warning_signs_detected
    (mixed_signals and few_warning_signs_detected are kept for compatibility but are not produced
    while the ML model is the only component feeding the signal.)
"""

import math
from numbers import Real
from typing import Any, Dict, List, Optional

SIGNAL_INSUFFICIENT = "insufficient_information"
SIGNAL_MIXED = "mixed_signals"
SIGNAL_WARNING = "warning_signs_detected"
SIGNAL_FEW_WARNINGS = "few_warning_signs_detected"

# Below this many words the ML model (trained on full-length articles) is not a reliable basis
# for a "few warning signs" statement.
MIN_WORDS_FOR_REASSURING_SIGNAL = 30

SIGNAL_HEADLINES = {
    SIGNAL_INSUFFICIENT: "Not enough usable information to say anything cautious about this text.",
    SIGNAL_MIXED: "The components point in different directions, so no clear overall signal.",
    SIGNAL_WARNING: "One or more components flagged warning signs. Treat this text with extra caution and check it independently.",
    SIGNAL_FEW_WARNINGS: "Fewer automated warning signs were detected, but this does NOT mean the text is accurate.",
}

STANDARD_CANNOT_CONCLUDE = [
    "Whether the claims in the text are true or false. Nothing here has been fact-checked.",
    "Whether the source is trustworthy, or whether the text is accurate, complete or current.",
]

NEXT_STEPS = [
    "Find the original source and check who published it and when.",
    "Look for the same claims in independent, reputable outlets or primary sources (official records, data, direct statements).",
    "Treat this tool's output as one input to your own judgement, not as a verdict.",
]


def _usable(component: Optional[dict]) -> bool:
    return bool(component) and component.get("status") == "ok" and component.get("result") is not None


def _ml_summary(ml_component: dict) -> Dict[str, Any]:
    """Returns the usable prediction label, uncalibrated score band, and input length."""
    if not _usable(ml_component):
        return {"usable": False}
    res = ml_component["result"]
    label = res.get("label")
    if label not in ("Fake", "Real"):  # e.g. "Undetermined" (no known vocabulary)
        return {"usable": False, "undetermined": True}
    return {
        "usable": True,
        "label": label,
        "confidence_level": res.get("confidence_level"),
        "warns": label == "Fake",
        "n_words": res.get("n_words") or 0,
    }


def build_final_assessment(components: Dict[str, dict], input_word_count: int) -> Dict[str, Any]:
    """
    Build the cautious synthesis from the `components` dict produced by AnalysisService
    (keys: ml_prediction, evidence).
    """
    ml = _ml_summary(components.get("ml_prediction") or {})
    evidence = components.get("evidence") or {}

    observations: List[str] = []
    cannot_conclude: List[str] = list(STANDARD_CANNOT_CONCLUDE)

    # --- observations, one per component, stated as what that component said ---
    if ml["usable"]:
        observations.append(
            f"ML model: predicted {ml['label']} (uncalibrated score band: {ml['confidence_level']}). "
            "This is a statistical pattern match against past articles, not proof about this text."
        )
    elif ml.get("undetermined"):
        observations.append("ML model: could not produce a prediction (the text has no words the model knows).")
    else:
        observations.append("ML model: not available for this request.")

    ev_status = evidence.get("status")
    if ev_status == "ok":
        claims = evidence.get("claims", [])
        n_with = sum(1 for c in claims if c.get("retrieval_status") == "evidence_retrieved")
        text = (f"Evidence retrieval: passages were retrieved for {n_with} of {len(claims)} claim(s). "
                "Retrieval measures similarity, not agreement: a retrieved passage may support, contradict or merely mention a claim, "
                "and this tool does not judge which.")
        if evidence.get("uses_synthetic_demo_documents"):
            text += " The reference collection is FICTIONAL demonstration data, not real fact-checking sources."
        observations.append(text)
        if n_with < len(claims):
            cannot_conclude.append("Claims with no retrieved passage: this only means the reference collection did not cover them.")
    elif ev_status == "skipped":
        observations.append("Evidence retrieval: not run for this request.")
    else:
        observations.append("Evidence retrieval: not available or no claims to search (" + str(evidence.get("detail") or "see component status") + ").")
        cannot_conclude.append("Anything about supporting evidence: none was retrieved.")

    # --- overall signal from the ML prediction only ---
    notes: List[str] = []
    if not ml["usable"]:
        signal = SIGNAL_INSUFFICIENT
    elif ml["warns"]:
        signal = SIGNAL_WARNING
        notes.append("Only the ML model feeds the overall signal, so it is weaker than a multi-component signal.")
    else:
        signal = SIGNAL_INSUFFICIENT
        notes.append(
            "The ML model predicted Real, but it is the only component feeding the signal and it loses accuracy on "
            "unfamiliar news sources, so no reassuring signal is given."
        )
        if input_word_count < MIN_WORDS_FOR_REASSURING_SIGNAL:
            notes.append(f"The input is also very short (under {MIN_WORDS_FOR_REASSURING_SIGNAL} words); the model was trained on full articles.")

    if ml["usable"] and ml["warns"] and ml.get("confidence_level") == "low":
        notes.append("The ML 'Fake' prediction has a low uncalibrated score band.")

    return {
        "overall_signal": signal,
        "headline": SIGNAL_HEADLINES[signal],
        "observations": observations,
        "notes": notes,
        "cannot_conclude": cannot_conclude,
        "suggested_next_steps": list(NEXT_STEPS),
        "is_verdict": False,
        "basis": "Rule-based summary of the ML prediction only. No percentage score. Retrieved evidence does not change the signal.",
    }


def interpret_retrieved_evidence(claim_result: Dict[str, Any]) -> Dict[str, str]:
    """Describe retrieval status without inferring whether a passage supports a claim."""
    passages = claim_result.get("retrieved_evidence") or []
    if passages:
        return {
            "classification": "unclassified",
            "reason": "Passages were retrieved by similarity, but the system does not determine whether they support or contradict the claim.",
        }
    status = claim_result.get("retrieval_status")
    if status == "no_relevant_evidence":
        reason = "No sufficiently relevant passage was found in the reference collection; this is not evidence for or against the claim."
    elif status == "retrieval_unavailable":
        reason = "Evidence retrieval was unavailable, so no passages could be assessed."
    elif status == "invalid_claim":
        reason = "The claim was empty or invalid, so no passages were searched."
    else:
        reason = "No passages were retrieved, so the claim cannot be assessed from this reference collection."
    return {"classification": "inconclusive", "reason": reason}


def _valid_probability(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


def build_structured_assessment(
    components: Dict[str, dict],
    final_assessment: Dict[str, Any],
) -> Dict[str, Any]:
    """Build additive, predictable API sections while keeping prediction and retrieval separate."""
    ml = components.get("ml_prediction") or {}
    ml_result = ml.get("result") or {}
    label = ml_result.get("label")
    fake_probability = _valid_probability(ml_result.get("fake_probability"))
    real_probability = _valid_probability(ml_result.get("real_probability"))

    evidence = components.get("evidence") or {}
    claims = []
    for result in evidence.get("claims") or []:
        passages = []
        for passage in result.get("retrieved_evidence") or []:
            metadata = {key: value for key, value in passage.items() if key not in ("text", "source")}
            passages.append({
                "source_name": passage.get("source"),
                "text": passage.get("text"),
                "metadata": metadata,
            })
        claims.append({
            "claim": result.get("claim"),
            "retrieval_status": result.get("retrieval_status", "retrieval_unavailable"),
            "passages": passages,
            "evidence_interpretation": result.get("evidence_interpretation") or interpret_retrieved_evidence(result),
        })

    observations = final_assessment.get("observations") or []
    return {
        "machine_learning": {
            "status": ml.get("status", "unavailable"),
            "predicted_label": label if label in ("Fake", "Real") else None,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "probabilities_calibrated": False if fake_probability is not None or real_probability is not None else None,
            "explanation": "The label is a model prediction based on learned text patterns, not verified truth.",
        },
        "article_claims": {
            "status": evidence.get("status", "unavailable"),
            "claim_source": evidence.get("claim_source", "none"),
            "items": claims,
            "empty_state": (
                evidence.get("detail")
                or evidence.get("error")
                or ("No claims were selected." if not claims else None)
            ) if not claims else None,
        },
        "overall": {
            "summary": final_assessment.get("headline"),
            "model_observation": observations[0] if observations else "The model prediction is unavailable.",
            "evidence_observation": observations[1] if len(observations) > 1 else "Evidence retrieval was not run.",
            "limitations": list(final_assessment.get("cannot_conclude") or []),
            "limitation_statement": (
                "The classifier learns text patterns and does not verify truth; retrieval similarity does not establish "
                "whether a passage supports or contradicts a claim."
            ),
            "is_verdict": False,
        },
    }
