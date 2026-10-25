import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ResultsSection } from "@/components/AnalysisResults";
import { normalizeAnalysisResponse } from "@/lib/api";

describe("Analysis results", () => {
  it("shows the uncalibrated disclaimer and fictional-source warning", () => {
    const result = normalizeAnalysisResponse({
      components: {
        ml_prediction: {
          status: "ok",
          warnings: ["The classifier loaded with a notice."],
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
              claim: "A demonstration claim.",
              retrieval_status: "evidence_retrieved",
              evidence_interpretation: {
                classification: "unclassified",
                reason: "Similarity does not establish support.",
              },
              retrieved_evidence: [
                {
                  text: "A fictional reference passage.",
                  source: "demo.txt",
                  score: 0.8,
                  is_synthetic_demo: true,
                },
              ],
            },
          ],
        },
      },
      warnings: ["An API warning."],
      credibility_assessment: {
        overall: {
          model_observation: "The classifier predicted Fake.",
          evidence_observation: "Evidence was inconclusive.",
          limitation_statement: "This is not a fact-check.",
        },
      },
      final_assessment: {
        headline: "A cautious assessment.",
        observations: ["The classifier predicted Fake.", "Evidence was inconclusive."],
        cannot_conclude: ["The source was not verified."],
      },
    });

    render(<ResultsSection state="done" result={result} error={null} onRetry={() => {}} />);

    expect(
      screen.getByText(/probability\/confidence values are uncalibrated/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/fictional demonstration documents/i)).toBeInTheDocument();
    expect(screen.getByText("A fictional reference passage.")).toBeInTheDocument();
    expect(screen.getByText("Raw Fake probability")).toBeInTheDocument();
    expect(screen.getByText("Raw Real probability")).toBeInTheDocument();
    expect(screen.getByText("An API warning.")).toBeInTheDocument();
    expect(screen.getByText("The classifier loaded with a notice.")).toBeInTheDocument();
    expect(screen.getByText("A cautious assessment.")).toBeInTheDocument();
    expect(screen.getByText("This is not a fact-check.")).toBeInTheDocument();
    expect(screen.getAllByText("The classifier predicted Fake.")).toHaveLength(1);
    expect(screen.getByText("The source was not verified.")).toBeInTheDocument();
  });

  it("presents an empty no-relevant-evidence response without inventing passages", () => {
    const result = normalizeAnalysisResponse({
      components: {
        ml_prediction: { status: "ok", result: { label: "Real", confidence: 0.7 } },
        evidence: {
          status: "ok",
          claims: [
            {
              claim: "No matching passage.",
              retrieval_status: "no_relevant_evidence",
              retrieved_evidence: [],
              evidence_interpretation: {
                classification: "inconclusive",
                reason: "No sufficiently relevant passage was found for this claim.",
              },
            },
          ],
        },
      },
    });

    render(<ResultsSection state="done" result={result} error={null} onRetry={() => {}} />);

    expect(
      screen.getByText(/no sufficiently relevant passage was found in the reference collection/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText("No sufficiently relevant passage was found for this claim."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/fictional demonstration documents/i)).not.toBeInTheDocument();
  });
});
