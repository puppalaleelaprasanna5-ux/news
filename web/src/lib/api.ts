/**
 * NewsCred API service module.
 *
 * All requests to the existing Python/FastAPI backend live here — no component
 * talks to the network directly. The backend base URL and endpoint paths are
 * configurable through Vite environment variables (see .env.example).
 *
 * The response normalizer only surfaces values the backend actually returns.
 * It never invents labels, confidence scores, or evidence.
 */

export interface PredictionResult {
  /** Raw label string returned by the classifier, or null when absent. */
  label: string | null;
  /** Confidence/probability when the backend provides one, else null. */
  confidence: number | null;
  fakeProbability: number | null;
  realProbability: number | null;
  probabilitiesCalibrated: boolean | null;
}

export interface EvidencePassage {
  text: string;
  source: string | null;
  document: string | null;
  score: number | null;
  claim: string | null;
  isSyntheticDemo: boolean;
  /** Any extra metadata fields the backend includes, preserved as-is. */
  metadata: Record<string, unknown>;
}

export interface ClaimStatus {
  claim: string | null;
  retrievalStatus: string;
  interpretation: { classification: string; reason: string } | null;
  limitations: string[];
}

export interface AnalysisResult {
  prediction: PredictionResult;
  evidence: EvidencePassage[];
  evidenceStatus: string | null;
  claims: ClaimStatus[];
  warnings: string[];
  usesSyntheticDemoDocuments: boolean;
  finalAssessment: Record<string, unknown> | null;
  credibilityAssessment: Record<string, unknown> | null;
}

export type ApiErrorKind = "network" | "http" | "format";

export class ApiError extends Error {
  kind: ApiErrorKind;
  status?: number;

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    if (status !== undefined) this.status = status;
  }
}

const RAW_BASE_URL = (import.meta.env["VITE_API_BASE_URL"] as string | undefined) ?? "";
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, "");

const ANALYZE_PATH = (import.meta.env["VITE_API_ANALYZE_PATH"] as string | undefined) ?? "/analyze";
const HEALTH_PATH = (import.meta.env["VITE_API_HEALTH_PATH"] as string | undefined) ?? "/health";

/** Maximum article length accepted by the input UI; match it to the backend. */
export const MAX_ARTICLE_LENGTH = Number(
  (import.meta.env["VITE_MAX_ARTICLE_LENGTH"] as string | undefined) ?? 20000,
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function firstString(records: Array<Record<string, unknown>>, keys: string[]): string | null {
  for (const record of records) {
    if (!record) continue;
    for (const key of keys) {
      const value = record[key];
      if (typeof value === "string" && value.trim().length > 0) return value;
      if (typeof value === "number" || typeof value === "boolean") return String(value);
    }
  }
  return null;
}

function firstNumber(records: Array<Record<string, unknown>>, keys: string[]): number | null {
  for (const record of records) {
    if (!record) continue;
    for (const key of keys) {
      const value = record[key];
      if (typeof value === "number" && Number.isFinite(value)) return value;
    }
  }
  return null;
}

function firstArray(value: unknown, keys: string[]): unknown[] | null {
  if (Array.isArray(value)) return value;
  if (isRecord(value)) {
    for (const key of keys) {
      if (Array.isArray(value[key])) return value[key] as unknown[];
    }
  }
  return null;
}

const LABEL_KEYS = [
  "label",
  "prediction",
  "predicted_label",
  "predicted_class",
  "class",
  "result",
  "verdict",
  "category",
];

const CONFIDENCE_KEYS = ["confidence", "probability", "prob", "proba", "confidence_score", "score"];

const EVIDENCE_ARRAY_KEYS = [
  "evidence",
  "passages",
  "retrieved_passages",
  "retrieved",
  "matches",
  "neighbors",
  "results",
  "sources",
  "documents",
];

const PASSAGE_TEXT_KEYS = ["text", "passage", "content", "chunk", "snippet", "sentence", "body"];
const PASSAGE_SOURCE_KEYS = ["source", "title", "name", "publication", "outlet"];
const PASSAGE_DOCUMENT_KEYS = ["document", "doc", "file", "filename", "article_id", "doc_id"];
const PASSAGE_SCORE_KEYS = ["score", "similarity", "relevance", "distance"];
const NESTED_RESULT_KEYS = ["result", "analysis", "prediction", "data", "output", "response"];

function emptyAnalysisResult(
  label: string | null = null,
  confidence: number | null = null,
): AnalysisResult {
  return {
    prediction: {
      label,
      confidence,
      fakeProbability: null,
      realProbability: null,
      probabilitiesCalibrated: null,
    },
    evidence: [],
    evidenceStatus: null,
    claims: [],
    warnings: [],
    usesSyntheticDemoDocuments: false,
    finalAssessment: null,
    credibilityAssessment: null,
  };
}

/**
 * Normalizes an analysis response into {@link AnalysisResult}.
 *
 * Accepts the common field names used by classifier + retrieval backends and
 * ignores everything else (extra fields are not dropped from evidence items:
 * they are preserved under `metadata`). Throws an {@link ApiError} of kind
 * "format" when neither a prediction nor any evidence can be recognized, so
 * the UI never shows fabricated results.
 */
export function normalizeAnalysisResponse(payload: unknown): AnalysisResult {
  if (typeof payload === "string") {
    // Some backends return the bare label.
    return emptyAnalysisResult(payload);
  }

  if (Array.isArray(payload)) {
    // Bare array: treat as a list of evidence passages.
    const evidence = normalizeEvidence(payload);
    if (evidence.length > 0) return { ...emptyAnalysisResult(), evidence };
    throw new ApiError("format", "The analysis service returned an unrecognized response format.");
  }

  if (!isRecord(payload)) {
    throw new ApiError("format", "The analysis service returned an unrecognized response format.");
  }

  const nested = NESTED_RESULT_KEYS.flatMap((key) =>
    isRecord(payload[key]) ? [payload[key] as Record<string, unknown>] : [],
  );
  const records: Array<Record<string, unknown>> = [payload, ...nested];

  const components = isRecord(payload["components"]) ? payload["components"] : null;
  const mlComponent =
    components && isRecord(components["ml_prediction"]) ? components["ml_prediction"] : null;
  const mlResult = mlComponent && isRecord(mlComponent["result"]) ? mlComponent["result"] : null;
  const evidenceComponent =
    components && isRecord(components["evidence"]) ? components["evidence"] : null;

  if (mlResult || evidenceComponent) {
    const label = mlResult ? firstString([mlResult], LABEL_KEYS) : null;
    const confidence = mlResult
      ? firstNumber([mlResult], ["confidence", ...CONFIDENCE_KEYS])
      : null;
    const claims =
      evidenceComponent && Array.isArray(evidenceComponent["claims"])
        ? evidenceComponent["claims"]
        : [];
    const evidence = normalizeClaimEvidence(claims);
    const claimStatuses = normalizeClaimStatuses(claims);
    const calibration = mlResult?.["probabilities_calibrated"];
    const fakeProbability = mlResult ? firstNumber([mlResult], ["fake_probability"]) : null;
    const realProbability = mlResult ? firstNumber([mlResult], ["real_probability"]) : null;
    const status = evidenceComponent?.["status"];
    const hasNoRelevantEvidence = claims.some(
      (claim) => isRecord(claim) && claim["retrieval_status"] === "no_relevant_evidence",
    );
    const evidenceStatus =
      hasNoRelevantEvidence && evidence.length === 0
        ? "no_relevant_evidence"
        : typeof status === "string"
          ? status
          : null;
    const componentRecords = components ? Object.values(components).filter(isRecord) : [];
    const warnings = [
      ...readStringList(payload["warnings"]),
      ...readStringList(mlResult?.["warnings"]),
      ...componentRecords.flatMap((component) => [
        ...readStringList(component["warnings"]),
        ...(typeof component["error"] === "string" ? [component["error"]] : []),
      ]),
    ];

    return {
      prediction: {
        label,
        confidence,
        fakeProbability,
        realProbability,
        probabilitiesCalibrated: typeof calibration === "boolean" ? calibration : null,
      },
      evidence,
      evidenceStatus,
      claims: claimStatuses,
      warnings: [...new Set(warnings)],
      usesSyntheticDemoDocuments:
        evidenceComponent?.["uses_synthetic_demo_documents"] === true ||
        evidence.some((passage) => passage.isSyntheticDemo),
      finalAssessment: isRecord(payload["final_assessment"]) ? payload["final_assessment"] : null,
      credibilityAssessment: isRecord(payload["credibility_assessment"])
        ? payload["credibility_assessment"]
        : null,
    };
  }

  const label = firstString(records, LABEL_KEYS);
  const confidence = firstNumber(records, CONFIDENCE_KEYS);
  const evidence = normalizeEvidence(firstArray(payload, EVIDENCE_ARRAY_KEYS));

  if (label === null && confidence === null && evidence.length === 0) {
    throw new ApiError("format", "The analysis service returned an unrecognized response format.");
  }

  return { ...emptyAnalysisResult(label, confidence), evidence };
}

function normalizeClaimEvidence(claims: unknown[]): EvidencePassage[] {
  const passages: EvidencePassage[] = [];
  for (const claimItem of claims) {
    if (!isRecord(claimItem)) continue;
    const claim = typeof claimItem["claim"] === "string" ? claimItem["claim"] : null;
    const interpretation = claimItem["evidence_interpretation"];
    const evidence = normalizeEvidence(
      Array.isArray(claimItem["retrieved_evidence"]) ? claimItem["retrieved_evidence"] : [],
    );
    for (const passage of evidence) {
      passages.push({
        ...passage,
        claim,
        metadata: {
          ...passage.metadata,
          ...(isRecord(interpretation) ? { evidence_interpretation: interpretation } : {}),
          retrieval_status: claimItem["retrieval_status"],
        },
      });
    }
  }
  return passages;
}

function normalizeClaimStatuses(claims: unknown[]): ClaimStatus[] {
  return claims.flatMap((claimItem) => {
    if (!isRecord(claimItem)) return [];
    const interpretation = claimItem["evidence_interpretation"];
    return [
      {
        claim: typeof claimItem["claim"] === "string" ? claimItem["claim"] : null,
        retrievalStatus:
          typeof claimItem["retrieval_status"] === "string"
            ? claimItem["retrieval_status"]
            : "unknown",
        interpretation:
          isRecord(interpretation) &&
          typeof interpretation["classification"] === "string" &&
          typeof interpretation["reason"] === "string"
            ? {
                classification: interpretation["classification"],
                reason: interpretation["reason"],
              }
            : null,
        limitations: readStringList(claimItem["limitations"]),
      },
    ];
  });
}

function readStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
}

function normalizeEvidence(list: unknown[] | null): EvidencePassage[] {
  if (!list) return [];
  const passages: EvidencePassage[] = [];
  for (const item of list) {
    if (typeof item === "string") {
      if (item.trim().length > 0) {
        passages.push({
          text: item,
          source: null,
          document: null,
          score: null,
          claim: null,
          isSyntheticDemo: false,
          metadata: {},
        });
      }
      continue;
    }
    if (!isRecord(item)) continue;
    const text = firstString([item], PASSAGE_TEXT_KEYS);
    if (!text) continue;
    passages.push({
      text,
      source: firstString([item], PASSAGE_SOURCE_KEYS),
      document: firstString([item], PASSAGE_DOCUMENT_KEYS),
      score: firstNumber([item], PASSAGE_SCORE_KEYS),
      claim: null,
      isSyntheticDemo: item["is_synthetic_demo"] === true,
      metadata: Object.fromEntries(
        Object.entries(item).filter(
          ([key]) =>
            !PASSAGE_TEXT_KEYS.includes(key) &&
            !PASSAGE_SOURCE_KEYS.includes(key) &&
            !PASSAGE_DOCUMENT_KEYS.includes(key) &&
            !PASSAGE_SCORE_KEYS.includes(key),
        ),
      ),
    });
  }
  return passages;
}

/**
 * POSTs the article text to the analysis endpoint and normalizes the response.
 *
 * Assumption (documented in README): the endpoint accepts
 * `POST {ANALYZE_PATH}` with JSON `{ "text": "<article>" }`. If your backend
 * expects a different field name, adjust it here — this is the only place
 * that needs to change.
 */
export async function analyzeArticle(text: string, signal?: AbortSignal): Promise<AnalysisResult> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${ANALYZE_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: signal ?? null,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new ApiError(
      "network",
      "Could not reach the analysis service. Make sure the backend is running, then try again.",
    );
  }

  if (!response.ok) {
    throw new ApiError(
      "http",
      `The analysis service returned an error (HTTP ${response.status}).`,
      response.status,
    );
  }

  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new ApiError(
      "format",
      "The analysis service returned a response that is not valid JSON.",
    );
  }

  // Helpful while aligning the normalizer with the real backend contract.
  if (import.meta.env.DEV) {
    console.info("[NewsCred] Raw analysis response:", payload);
  }

  return normalizeAnalysisResponse(payload);
}

/**
 * GETs the backend health endpoint. Used only to give a clearer error message
 * when an analysis request fails — never shown as a permanent status badge.
 */
export async function checkBackendUp(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}${HEALTH_PATH}`, { method: "GET" });
    return response.ok;
  } catch {
    return false;
  }
}
