import { describe, expect, it } from "vitest";

import { ApiError, MAX_ARTICLE_LENGTH, normalizeAnalysisResponse } from "@/lib/api";

// The normalizer must only surface values the backend actually returns —
// never invented labels, confidence scores, or evidence.
describe("normalizeAnalysisResponse", () => {
  it("reads a straightforward { label, confidence } response", () => {
    const result = normalizeAnalysisResponse({ label: "REAL", confidence: 0.87 });
    expect(result.prediction.label).toBe("REAL");
    expect(result.prediction.confidence).toBe(0.87);
    expect(result.evidence).toHaveLength(0);
  });

  it("reads alternate field names, including nested prediction objects", () => {
    const result = normalizeAnalysisResponse({
      prediction: { predicted_class: "FAKE", probability: 0.92 },
    });
    expect(result.prediction.label).toBe("FAKE");
    expect(result.prediction.confidence).toBe(0.92);
  });

  it("keeps confidence null when the backend does not report one", () => {
    const result = normalizeAnalysisResponse({ label: "REAL" });
    expect(result.prediction.confidence).toBeNull();
  });

  it("normalizes evidence passages with source metadata", () => {
    const result = normalizeAnalysisResponse({
      label: "FAKE",
      passages: [{ text: "Passage about the event.", source: "Reuters", score: 0.814 }],
    });
    const passage = result.evidence[0];
    expect(result.evidence).toHaveLength(1);
    expect(passage?.text).toBe("Passage about the event.");
    expect(passage?.source).toBe("Reuters");
    expect(passage?.score).toBe(0.814);
    expect(passage?.document).toBeNull();
  });

  it("reads the actual FastAPI response and flattens claim evidence with metadata", () => {
    const result = normalizeAnalysisResponse({
      components: {
        ml_prediction: {
          status: "ok",
          warnings: ["A model notice."],
          result: {
            label: "Fake",
            confidence: 0.91,
            fake_probability: 0.91,
            real_probability: 0.09,
            probabilities_calibrated: false,
          },
        },
        evidence: {
          status: "ok",
          uses_synthetic_demo_documents: true,
          claims: [
            {
              claim: "The Alder River levee is 4.2 kilometres long.",
              retrieval_status: "evidence_retrieved",
              evidence_interpretation: {
                classification: "unclassified",
                reason: "Similarity does not determine support.",
              },
              retrieved_evidence: [
                {
                  text: "A demonstration passage about the levee.",
                  source: "brightwater_flood_report.txt",
                  source_id: "brightwater_flood_report.txt",
                  score: 0.6355,
                  score_type: "cosine_similarity",
                  chunk_id: 3,
                  is_synthetic_demo: true,
                },
              ],
            },
          ],
        },
      },
      credibility_assessment: { overall: { is_verdict: false } },
      final_assessment: { is_verdict: false },
      warnings: ["A top-level notice."],
    });

    expect(result.prediction).toEqual({
      label: "Fake",
      confidence: 0.91,
      fakeProbability: 0.91,
      realProbability: 0.09,
      probabilitiesCalibrated: false,
    });
    expect(result.evidence).toHaveLength(1);
    expect(result.evidence[0]).toMatchObject({
      text: "A demonstration passage about the levee.",
      source: "brightwater_flood_report.txt",
      score: 0.6355,
      claim: "The Alder River levee is 4.2 kilometres long.",
      isSyntheticDemo: true,
      metadata: {
        source_id: "brightwater_flood_report.txt",
        score_type: "cosine_similarity",
        chunk_id: 3,
        retrieval_status: "evidence_retrieved",
        evidence_interpretation: { classification: "unclassified" },
      },
    });
    expect(result.usesSyntheticDemoDocuments).toBe(true);
    expect(result.claims).toEqual([
      {
        claim: "The Alder River levee is 4.2 kilometres long.",
        retrievalStatus: "evidence_retrieved",
        interpretation: {
          classification: "unclassified",
          reason: "Similarity does not determine support.",
        },
        limitations: [],
      },
    ]);
    expect(result.warnings).toEqual(["A top-level notice.", "A model notice."]);
    expect(result.finalAssessment).toEqual({ is_verdict: false });
    expect(result.credibilityAssessment).toEqual({ overall: { is_verdict: false } });
  });

  it("accepts no_relevant_evidence as a valid result with no passages", () => {
    const result = normalizeAnalysisResponse({
      components: {
        ml_prediction: {
          status: "ok",
          result: {
            label: "Real",
            confidence: 0.74,
            fake_probability: 0.26,
            real_probability: 0.74,
            probabilities_calibrated: false,
          },
        },
        evidence: {
          status: "ok",
          claims: [
            {
              claim: "No related article was found.",
              retrieval_status: "no_relevant_evidence",
              retrieved_evidence: [],
              evidence_interpretation: {
                classification: "inconclusive",
                reason: "No relevant passage was found.",
              },
            },
          ],
        },
      },
      credibility_assessment: { article_claims: { items: [] } },
      final_assessment: { is_verdict: false },
    });

    expect(result.prediction.label).toBe("Real");
    expect(result.evidence).toEqual([]);
    expect(result.evidenceStatus).toBe("no_relevant_evidence");
  });

  it("accepts a partial response when the classifier is unavailable", () => {
    const result = normalizeAnalysisResponse({
      components: {
        ml_prediction: { status: "unavailable", result: null },
        evidence: { status: "unavailable", claims: [] },
      },
      partial: true,
      credibility_assessment: { overall: { is_verdict: false } },
    });

    expect(result.prediction.label).toBeNull();
    expect(result.prediction.confidence).toBeNull();
    expect(result.evidence).toEqual([]);
  });

  it("accepts plain-string evidence items", () => {
    const result = normalizeAnalysisResponse({
      prediction: "REAL",
      evidence: ["A relevant passage."],
    });
    expect(result.evidence[0]?.text).toBe("A relevant passage.");
    expect(result.evidence[0]?.source).toBeNull();
  });

  it("treats a bare label string as the prediction", () => {
    const result = normalizeAnalysisResponse("FAKE");
    expect(result.prediction.label).toBe("FAKE");
    expect(result.prediction.confidence).toBeNull();
  });

  it("does not invent evidence when none is returned", () => {
    const result = normalizeAnalysisResponse({ label: "REAL", evidence: [] });
    expect(result.evidence).toHaveLength(0);
  });

  it("rejects payloads with no recognizable prediction or evidence", () => {
    expect(() => normalizeAnalysisResponse({ unrelated: 42 })).toThrow(ApiError);
    try {
      normalizeAnalysisResponse({ unrelated: 42 });
    } catch (error) {
      expect((error as ApiError).kind).toBe("format");
    }
  });

  it("keeps the default maximum article length at 20000 characters", () => {
    expect(MAX_ARTICLE_LENGTH).toBe(20000);
  });
});
