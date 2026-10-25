import { FileSearch, Loader2 } from "lucide-react";
import { RetryButton } from "@/components/ArticleInput";
import type { AnalysisResult } from "@/lib/api";

function formatConfidence(confidence: number): string {
  return `${formatPercentage(confidence)} confidence`;
}

function formatPercentage(value: number): string {
  return value <= 1 ? `${(value * 100).toFixed(1)}%` : `${value.toFixed(1)}%`;
}

export function PredictionCard({
  label,
  confidence,
  fakeProbability,
  realProbability,
  probabilitiesCalibrated,
}: {
  label: string | null;
  confidence: number | null;
  fakeProbability: number | null;
  realProbability: number | null;
  probabilitiesCalibrated: boolean | null;
}) {
  return (
    <article className="rounded-xl border border-border bg-card p-5 sm:p-6">
      <h3 className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        Classifier prediction
      </h3>
      <div className="mt-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        {label !== null ? (
          <p className="font-display text-3xl font-semibold text-foreground">{label}</p>
        ) : (
          <p className="text-lg text-muted-foreground">No label was returned for this article.</p>
        )}
        {confidence !== null && (
          <p className="text-sm font-medium tabular-nums text-primary">
            {formatConfidence(confidence)}
          </p>
        )}
      </div>
      <p className="mt-3 text-sm leading-relaxed text-muted-foreground">
        This is a model prediction based on similarities to the examples it was trained on — not a
        verified truth rating.
      </p>
      {confidence !== null && (
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
          {probabilitiesCalibrated === false
            ? "The model’s probability/confidence values are uncalibrated and are not verified confidence."
            : probabilitiesCalibrated === true
              ? "The API reports these probabilities as calibrated; calibration does not verify the article’s claims."
              : "Probability calibration status was not reported; these values are not verified confidence."}
        </p>
      )}
      {(fakeProbability !== null || realProbability !== null) && (
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
          {fakeProbability !== null && (
            <div>
              <dt className="text-muted-foreground">Raw Fake probability</dt>
              <dd className="mt-1 font-medium tabular-nums text-foreground">
                {formatPercentage(fakeProbability)}
              </dd>
            </div>
          )}
          {realProbability !== null && (
            <div>
              <dt className="text-muted-foreground">Raw Real probability</dt>
              <dd className="mt-1 font-medium tabular-nums text-foreground">
                {formatPercentage(realProbability)}
              </dd>
            </div>
          )}
        </dl>
      )}
    </article>
  );
}

export function EvidenceCard({
  passages,
  evidenceStatus,
  usesSyntheticDemoDocuments,
  claims,
}: {
  passages: AnalysisResult["evidence"];
  evidenceStatus: AnalysisResult["evidenceStatus"];
  usesSyntheticDemoDocuments: AnalysisResult["usesSyntheticDemoDocuments"];
  claims: AnalysisResult["claims"];
}) {
  return (
    <article className="rounded-xl border border-border bg-card p-5 sm:p-6">
      <h3 className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        Retrieved evidence
      </h3>
      {passages.length === 0 ? (
        <div className="mt-3 flex items-start gap-3">
          <FileSearch
            className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground/70"
            aria-hidden="true"
          />
          <p className="text-sm leading-relaxed text-muted-foreground">
            {evidenceStatus === "no_relevant_evidence"
              ? "No sufficiently relevant passage was found in the reference collection. This is not evidence for or against the claim."
              : evidenceStatus === "unavailable" ||
                  claims.some((claim) => claim.retrievalStatus === "retrieval_unavailable")
                ? "Evidence retrieval was unavailable, so no evidence could be assessed. This is not evidence for or against the claim."
                : "No evidence passages were returned for this article. This is not evidence for or against its claims."}
          </p>
        </div>
      ) : (
        <ul className="mt-4 space-y-4">
          {passages.map((passage, index) => (
            <li key={index} className="border-l-2 border-primary/30 pl-4">
              <blockquote className="text-[15px] leading-relaxed text-foreground">
                {passage.text}
              </blockquote>
              {(passage.source || passage.document || passage.score !== null) && (
                <p className="mt-1.5 text-xs uppercase tracking-wide text-muted-foreground">
                  {passage.source && <span>{passage.source}</span>}
                  {passage.source && passage.document && <span className="px-1.5">·</span>}
                  {passage.document && <span>{passage.document}</span>}
                  {passage.score !== null && (
                    <span className="px-1.5">· similarity {passage.score.toFixed(3)}</span>
                  )}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
      {passages.length === 0 && claims.length > 0 && (
        <ul className="mt-4 space-y-3">
          {claims.map((claim, index) => (
            <li
              key={`${claim.claim ?? "claim"}-${index}`}
              className="border-l-2 border-border pl-4"
            >
              {claim.claim && <p className="text-sm font-medium text-foreground">{claim.claim}</p>}
              {claim.interpretation?.reason && (
                <p className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  {claim.interpretation.reason}
                </p>
              )}
              {claim.limitations.map((limitation) => (
                <p key={limitation} className="mt-1 text-sm leading-relaxed text-muted-foreground">
                  {limitation}
                </p>
              ))}
            </li>
          ))}
        </ul>
      )}
      {passages.length > 0 && (
        <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
          Passages are the closest matches found in your reference documents. Similarity alone does
          not establish whether a claim is true or whether a passage supports or contradicts it.
        </p>
      )}
      {(usesSyntheticDemoDocuments || passages.some((passage) => passage.isSyntheticDemo)) && (
        <p className="mt-4 text-sm font-medium leading-relaxed text-amber-700">
          Warning: these passages come from fictional demonstration documents, not real-world
          sources.
        </p>
      )}
    </article>
  );
}

export function InterpretationCard({ result }: { result: AnalysisResult }) {
  const { label, confidence } = result.prediction;
  const count = result.evidence.length;

  const predictionLine =
    label !== null
      ? `The classifier predicts “${label}”${
          confidence !== null
            ? ` (shown above with its confidence value)`
            : "; the model did not report a confidence value"
        }.`
      : "The classifier did not return a label for this article.";
  const evidenceLine =
    count === 0
      ? result.evidenceStatus === "no_relevant_evidence"
        ? "The retrieval system found no sufficiently relevant passages in the available reference documents."
        : result.evidenceStatus === "unavailable" ||
            result.claims.some((claim) => claim.retrievalStatus === "retrieval_unavailable")
          ? "Evidence retrieval was unavailable, so there are no evidence observations."
          : "The response contained no evidence passages."
      : count === 1
        ? "The retrieval system returned 1 passage from your reference documents."
        : `The retrieval system returned ${count} passages from your reference documents.`;
  const overall = result.credibilityAssessment?.["overall"];
  const modelObservation =
    isRecord(overall) && typeof overall["model_observation"] === "string"
      ? overall["model_observation"]
      : null;
  const evidenceObservation =
    isRecord(overall) && typeof overall["evidence_observation"] === "string"
      ? overall["evidence_observation"]
      : null;
  const limitationStatement =
    isRecord(overall) && typeof overall["limitation_statement"] === "string"
      ? overall["limitation_statement"]
      : null;
  const assessmentHeadline = result.finalAssessment?.["headline"];
  const finalObservations = readStringList(result.finalAssessment?.["observations"]);
  const limitations = isRecord(overall) ? readStringList(overall["limitations"]) : [];
  const cannotConclude = readStringList(result.finalAssessment?.["cannot_conclude"]);
  const alreadyDisplayed = [modelObservation, evidenceObservation, limitationStatement].filter(
    (value): value is string => value !== null,
  );
  const additionalAssessmentNotes = [
    ...new Set([...finalObservations, ...limitations, ...cannotConclude]),
  ].filter((item) => !alreadyDisplayed.includes(item));

  return (
    <article className="rounded-xl border border-border bg-secondary p-5 sm:p-6">
      <h3 className="text-xs font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        Overall interpretation
      </h3>
      <p className="mt-3 text-[15px] leading-relaxed text-foreground">{predictionLine}</p>
      <p className="mt-2 text-[15px] leading-relaxed text-foreground">{evidenceLine}</p>
      {typeof assessmentHeadline === "string" && (
        <p className="mt-3 text-[15px] font-medium leading-relaxed text-foreground">
          {assessmentHeadline}
        </p>
      )}
      {modelObservation && (
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{modelObservation}</p>
      )}
      {evidenceObservation && (
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{evidenceObservation}</p>
      )}
      {additionalAssessmentNotes.map((item, index) => (
        <p key={`${index}-${item}`} className="mt-2 text-sm leading-relaxed text-muted-foreground">
          {item}
        </p>
      ))}
      {limitationStatement && (
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{limitationStatement}</p>
      )}
      <p className="mt-2 text-[15px] leading-relaxed text-muted-foreground">
        Treat both outputs as signals to investigate further, not as verified conclusions.
      </p>
    </article>
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

export function AnalysisLoading() {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-3 rounded-xl border border-border bg-card p-5 text-sm text-muted-foreground sm:p-6"
    >
      <Loader2 className="h-5 w-5 animate-spin text-primary" aria-hidden="true" />
      Analyzing your article…
    </div>
  );
}

export function AnalysisError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div role="alert" className="rounded-xl border border-border bg-card p-5 sm:p-6">
      <h3 className="text-sm font-semibold text-foreground">Analysis could not complete</h3>
      <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{message}</p>
      <div className="mt-4">
        <RetryButton onRetry={onRetry} />
      </div>
    </div>
  );
}

export function ResultsSection({
  state,
  result,
  error,
  onRetry,
}: {
  state: "idle" | "loading" | "error" | "done";
  result: AnalysisResult | null;
  error: string | null;
  onRetry: () => void;
}) {
  if (state === "idle") return null;

  return (
    <section
      aria-label="Analysis results"
      aria-busy={state === "loading"}
      className="mt-8 space-y-4"
    >
      <h2 className="font-display text-xl font-semibold text-foreground">Analysis</h2>
      {state === "loading" && <AnalysisLoading />}
      {state === "error" && error && <AnalysisError message={error} onRetry={onRetry} />}
      {state === "done" && result && (
        <>
          {result.warnings.length > 0 && (
            <div
              role="status"
              className="rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm leading-relaxed text-amber-900"
            >
              <h3 className="font-semibold">Service notices</h3>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                {result.warnings.map((warning, index) => (
                  <li key={`${index}-${warning}`}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
          <PredictionCard
            label={result.prediction.label}
            confidence={result.prediction.confidence}
            fakeProbability={result.prediction.fakeProbability}
            realProbability={result.prediction.realProbability}
            probabilitiesCalibrated={result.prediction.probabilitiesCalibrated}
          />
          <EvidenceCard
            passages={result.evidence}
            evidenceStatus={result.evidenceStatus}
            usesSyntheticDemoDocuments={result.usesSyntheticDemoDocuments}
            claims={result.claims}
          />
          <InterpretationCard result={result} />
        </>
      )}
    </section>
  );
}
