"""
Orchestration: one framework-independent service that runs the two components
(ML prediction, RAG evidence retrieval) and assembles a structured result.

Design rules
* The outputs stay SEPARATE under `components`; nothing is merged into one score.
* Every component can fail on its own. A failure becomes `status != "ok"` with a short,
  user-safe message; the other components still run (partial result). Raw exception text and
  stack traces are only written to the log, never returned.
* Expensive objects (ML model, embedding model, FAISS index) are loaded once and reused.
  A failed load is retried on the next request, so fixing the problem (training the model,
  building the index) does not require restarting the API.
* Retrieval is evidence *retrieval*, not verification: no claim is labelled true/false.

Component status values: ok | unavailable | timeout | error | skipped
"""

import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.credibility.assessment import build_final_assessment, build_structured_assessment, interpret_retrieved_evidence
from src.models.predict import DEFAULT_MODELS_DIR, FakeNewsPredictor
from src.rag.config import DEFAULT_INDEX_DIR, DEFAULT_MIN_SCORE, PROJECT_ROOT, SAMPLE_DOCS_DIR
from src.rag.claim_selector import select_claims
from src.rag.embeddings import SentenceTransformerEmbedder
from src.rag.evidence_pipeline import PIPELINE_LIMITATIONS
from src.rag.faiss_index import (
    MANIFEST_FILENAME,
    EvidenceIndex,
    EvidenceIndexError,
    FaissNotInstalledError,
    IndexCorruptedError,
    IndexMismatchError,
    IndexNotFoundError,
)
from src.rag.retriever import STATUS_RETRIEVED, EvidenceRetriever

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 20000  # maximum accepted input length, in characters
HEADLINE_WORD_LIMIT = 40  # at or below this many words the input is labelled "headline / short text"
MAX_CLAIMS = 5
DEFAULT_RAG_TOP_K = 3

DISCLAIMER = (
    "This is an automated, experimental analysis for educational use. It is not a fact-check and does not verify "
    "anything. The ML model and the retrieval step can both be wrong, and neither has checked "
    "this text against the real world. Always verify important claims with trustworthy independent sources."
)
ML_NOTICE = "This is a model prediction (a statistical pattern match), not proof of factual correctness."
EVIDENCE_NOTICE = ("Passages are retrieved by text similarity from the reference collection. A similarity score is not a "
                   "probability of truth, and a retrieved passage may support, contradict or merely mention a claim.")
SYNTHETIC_NOTICE = ("These passages come from FICTIONAL demonstration documents shipped with the project. "
                    "They are not real-world evidence or real fact-checking sources.")


class InputValidationError(ValueError):
    """The request text is empty, not a string, or too long. The message is safe to show to users."""


def validate_input_text(text: Any, max_chars: int = MAX_INPUT_CHARS) -> str:
    if not isinstance(text, str):
        raise InputValidationError("The text must be a string.")
    cleaned = text.replace("\x00", "").strip()
    if not cleaned:
        raise InputValidationError("Please provide a headline or article text; the input is empty.")
    if len(cleaned) > max_chars:
        raise InputValidationError(
            f"The text is too long ({len(cleaned):,} characters). The maximum is {max_chars:,} characters."
        )
    return cleaned


def _component(status: str, **fields) -> Dict[str, Any]:
    base = {"status": status, "error": None, "notice": None, "result": None}
    base.update(fields)
    return base


def _unavailable_claim_results(claims: List[str]) -> List[dict]:
    return [
        {
            "claim": claim,
            "retrieval_status": "retrieval_unavailable",
            "retrieved_evidence": [],
            "evidence_interpretation": interpret_retrieved_evidence(
                {"retrieval_status": "retrieval_unavailable", "retrieved_evidence": []}
            ),
        }
        for claim in claims
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_index_dir() -> Path:
    """
    Where the FAISS index lives. Uses RAG_INDEX_DIR if it is set (relative paths are relative to
    the project root). If it is NOT set and `data/rag/index` does not exist but the demo index
    `data/rag/demo_index` does, the demo index is used; this is reported in /health.
    """
    env = os.getenv("RAG_INDEX_DIR")
    if env:
        p = Path(env)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    default = Path(DEFAULT_INDEX_DIR)
    demo = PROJECT_ROOT / "data" / "rag" / "demo_index"
    if not default.exists() and demo.exists():
        return demo
    return default


@dataclass
class AnalysisOptions:
    run_rag: bool = True
    manual_claims: Optional[List[str]] = None
    top_k: int = DEFAULT_RAG_TOP_K


class AnalysisService:
    """Loads components lazily, caches successful loads, and runs one analysis per call."""

    def __init__(
        self,
        models_dir: Path = DEFAULT_MODELS_DIR,
        index_dir: Optional[Path] = None,
        predictor_factory: Optional[Callable[[], Any]] = None,
        embedder_factory: Optional[Callable[[], Any]] = None,
        min_score: Optional[float] = DEFAULT_MIN_SCORE,
    ):
        self.models_dir = Path(models_dir)
        self._index_dir_override = Path(index_dir) if index_dir is not None else None
        self._predictor_factory = predictor_factory or (lambda: FakeNewsPredictor(self.models_dir))
        self._embedder_factory = embedder_factory or SentenceTransformerEmbedder
        self.min_score = min_score
        self._lock = threading.RLock()
        self._predictor = None
        self._embedder = None
        self._retriever: Optional[EvidenceRetriever] = None
        self._retriever_key = None
        self._synthetic_ids: set = set()

    @property
    def index_dir(self) -> Path:
        return self._index_dir_override if self._index_dir_override is not None else resolve_index_dir()

    # ------------------------------------------------------------------ ML model
    def _get_predictor(self):
        """Returns (predictor, None) or (None, user_safe_error)."""
        with self._lock:
            if self._predictor is not None:
                return self._predictor, None
            try:
                self._predictor = self._predictor_factory()
                return self._predictor, None
            except FileNotFoundError:
                logger.warning("ML model artifacts missing in %s", self.models_dir, exc_info=True)
                return None, ("The trained model files were not found in the 'models/' folder. "
                              "Run notebooks/03_ml_models.ipynb to create them.")
            except Exception:
                logger.exception("ML model failed to load")
                return None, "The trained model could not be loaded (the files may be corrupted or from an incompatible version)."

    def _run_ml(self, text: str) -> Dict[str, Any]:
        predictor, error = self._get_predictor()
        if predictor is None:
            return _component("unavailable", error=error, notice=ML_NOTICE)
        try:
            result = predictor.predict(text, explain=True, top_n=5)
        except ValueError as err:  # e.g. empty after cleaning
            return _component("error", error=f"The ML model could not analyze this text: {err}", notice=ML_NOTICE)
        except Exception:
            logger.exception("ML prediction failed")
            return _component("error", error="The ML prediction failed unexpectedly.", notice=ML_NOTICE)
        result["probabilities_calibrated"] = (
            False
            if result.get("fake_probability") is not None or result.get("real_probability") is not None
            else None
        )
        return _component("ok", result=result, notice=ML_NOTICE, model_name=result.get("model_name"))

    # ----------------------------------------------------------------------- RAG
    def _get_retriever(self):
        """Returns (retriever, None) or (None, user_safe_error). Reloads the index if it changed on disk."""
        index_dir = self.index_dir
        manifest = index_dir / MANIFEST_FILENAME
        try:
            key = (str(index_dir), manifest.stat().st_mtime_ns)
        except OSError:
            key = None
        with self._lock:
            if self._retriever is not None and key is not None and key == self._retriever_key:
                return self._retriever, None
            try:
                if self._embedder is None:
                    self._embedder = self._embedder_factory()
                index = EvidenceIndex.load(index_dir, expected_model_name=self._embedder.model_name)
            except FaissNotInstalledError:
                # Distinct from a missing index: without FAISS the index can be neither loaded nor built.
                logger.warning("FAISS could not be imported; evidence retrieval is unavailable")
                self._retriever = None
                return None, ("FAISS (the faiss-cpu package) is not installed or could not be imported, so the "
                              "evidence index cannot be used. Install it with: pip install -r requirements.txt "
                              "(see the README, 'Troubleshooting'), then restart the backend.")
            except IndexNotFoundError:
                self._retriever = None
                return None, ("No evidence index was found. Build one with: "
                              "python scripts/rag_cli.py build (see the README, 'RAG index creation').")
            except IndexMismatchError:
                self._retriever = None
                return None, "The evidence index was built with a different embedding model. Rebuild it."
            except IndexCorruptedError:
                logger.warning("Evidence index corrupted", exc_info=True)
                self._retriever = None
                return None, "The evidence index is damaged or unreadable. Rebuild it with scripts/rag_cli.py build --force."
            except EvidenceIndexError:
                logger.warning("Evidence index error", exc_info=True)
                self._retriever = None
                return None, ("The evidence index could not be loaded (details are in the backend log). "
                              "Try rebuilding it with: python scripts/rag_cli.py build --force.")
            except Exception:
                logger.exception("Unexpected error preparing the retriever")
                self._retriever = None
                return None, "The evidence retriever could not be prepared."
            self._retriever = EvidenceRetriever(index, self._embedder, top_k=DEFAULT_RAG_TOP_K, min_score=self.min_score)
            self._retriever_key = key
            self._synthetic_ids = self._detect_synthetic_sources(index)
            return self._retriever, None

    @staticmethod
    def _detect_synthetic_sources(index: EvidenceIndex) -> set:
        """source_ids whose indexed content hash equals a shipped synthetic sample document."""
        synthetic = set()
        try:
            sample_hashes = {p.relative_to(SAMPLE_DOCS_DIR).as_posix(): _sha256(p)
                             for p in SAMPLE_DOCS_DIR.rglob("*.txt")}
        except OSError:
            return synthetic
        for source_id, digest in (index.manifest.get("sources") or {}).items():
            if sample_hashes.get(source_id) == digest:
                synthetic.add(source_id)
        return synthetic

    @staticmethod
    def _retrieval_error_message(raw: str) -> str:
        """Map a raw retrieval error to a short user-safe message (the raw text is only logged)."""
        low = str(raw).lower()
        if "could not load embedding model" in low or "sentence-transformers" in low:
            return ("The embedding model could not be loaded. The first run needs internet access to download it once; "
                    "see the README ('Embedding model setup').")
        if "claim must be" in low:
            return "The claim was empty."
        return "Retrieval failed for this claim."

    @staticmethod
    def _dedupe_claims(claims, limit: int) -> List[str]:
        if limit <= 0:
            return []
        seen, out = set(), []
        for claim in claims or []:
            text = claim.strip() if isinstance(claim, str) else ""
            key = " ".join(text.lower().split())
            if text and key not in seen:
                seen.add(key)
                out.append(text)
                if len(out) == limit:
                    break
        return out

    def _run_rag(self, text: str, options: AnalysisOptions) -> Dict[str, Any]:
        base = {"claim_source": "none", "claims": [], "notice": EVIDENCE_NOTICE}
        # --- which claims do we search for? (never invented) ---
        claims: List[str] = []
        manual = self._dedupe_claims(options.manual_claims, MAX_CLAIMS)
        if manual:
            claims, base["claim_source"] = manual, "manual"
        elif len(text.split()) <= HEADLINE_WORD_LIMIT:
            claims, base["claim_source"] = [text], "input_text"  # short input: search with the text itself
        else:
            claims = self._dedupe_claims(select_claims(text, MAX_CLAIMS), MAX_CLAIMS)
            if claims:
                base["claim_source"] = "heuristic"
        if not claims:
            return _component(
                "skipped", **{**base, "detail": "No sentences that look like checkable claims were found in the text."})

        retriever, error = self._get_retriever()
        if retriever is None:
            base["claims"] = _unavailable_claim_results(claims)
            return _component("unavailable", error=error, detail="Evidence index unavailable.", **base)

        try:
            results = retriever.retrieve_many(claims, top_k=options.top_k)
        except Exception:
            logger.exception("Retrieval failed")
            base["claims"] = _unavailable_claim_results(claims)
            return _component("error", error="Evidence retrieval failed unexpectedly.", detail="Retrieval failed.", **base)

        any_failed = False
        for r in results:
            if r.get("error"):
                logger.warning("Retrieval problem for a claim: %s", r["error"])
                r["error"] = self._retrieval_error_message(r["error"])
            r["evidence_interpretation"] = interpret_retrieved_evidence(r)
            if r["retrieval_status"] in ("retrieval_unavailable",):
                any_failed = True
            for item in r["retrieved_evidence"]:
                item["is_synthetic_demo"] = item["source_id"] in self._synthetic_ids
        n_failed = sum(1 for r in results if r["retrieval_status"] == "retrieval_unavailable")
        uses_synthetic = any(i["is_synthetic_demo"] for r in results for i in r["retrieved_evidence"]) or bool(self._synthetic_ids)
        status = "error" if n_failed == len(results) else "ok"
        extra = {}
        if status == "error":
            extra["error"] = "Evidence retrieval failed for every claim."
        elif any_failed:
            extra["error"] = "Evidence retrieval failed for some claims."
        notes = []
        if uses_synthetic:
            notes.append(SYNTHETIC_NOTICE)
        if base["claim_source"] == "input_text":
            notes.append("The (short) input text itself was used as the search query.")
        elif base["claim_source"] == "heuristic":
            notes.append("Search queries were chosen by a simple rule-based sentence selector (figures, names, "
                         "assertion verbs). It can pick poor sentences and does not understand the text.")
        elif base["claim_source"] == "manual":
            notes.append("Claims were supplied manually in the request.")
        base.update({"claims": results, "uses_synthetic_demo_documents": uses_synthetic, "notes": notes,
                     "reference_collection": {"num_passages": retriever.index.size, "embedding_model": retriever.index.model_name,
                                              "min_similarity": retriever.min_score, "score_type": "cosine_similarity"}})
        return _component(status, detail=None, **{**base, **extra})

    # ------------------------------------------------------------------- analyze
    def analyze(self, text: Any, options: Optional[AnalysisOptions] = None) -> Dict[str, Any]:
        """
        Run all enabled components. Raises InputValidationError only for bad input; every
        component failure is reported inside the returned dict instead.
        """
        options = options or AnalysisOptions()
        clean = validate_input_text(text)
        words = len(clean.split())

        ml = self._run_ml(clean)
        if options.run_rag:
            evidence = self._run_rag(clean, options)
        else:
            evidence = _component("skipped", detail="Disabled for this request.", claims=[], claim_source="none", notice=EVIDENCE_NOTICE)

        components = {"ml_prediction": ml, "evidence": evidence}
        statuses = {name: c["status"] for name, c in components.items()}
        partial = any(s not in ("ok", "skipped") for s in statuses.values())
        warnings = []
        for name, c in components.items():
            if c["status"] not in ("ok", "skipped") and c.get("error"):
                warnings.append(f"{name}: {c['error']}")
        if ml["status"] == "ok":
            warnings.extend(ml["result"].get("warnings", []) or [])

        final_assessment = build_final_assessment(components, words)
        return {
            "input": {
                "char_count": len(clean),
                "word_count": words,
                "input_kind": "headline_or_short_text" if words <= HEADLINE_WORD_LIMIT else "article",
            },
            "components": components,
            "component_status": statuses,
            "partial": partial,
            "warnings": warnings,
            "final_assessment": final_assessment,
            "credibility_assessment": build_structured_assessment(components, final_assessment),
            "limitations": PIPELINE_LIMITATIONS + [
                "The ML model was trained on a news dataset with known formatting artifacts; high accuracy there does not "
                "mean it generalises to all news."
            ],
            "disclaimer": DISCLAIMER,
        }

    # -------------------------------------------------------------------- health
    def health(self) -> Dict[str, Any]:
        """Component availability for GET /health. Never raises."""
        comps: Dict[str, Any] = {}

        predictor, error = self._get_predictor()
        if predictor is not None:
            info = {}
            try:
                info = predictor.model_info()
            except Exception:
                logger.exception("model_info failed")
            comps["ml_model"] = {"status": "ok", "model_name": info.get("model_name"), "load_warnings": info.get("load_warnings") or []}
        else:
            comps["ml_model"] = {"status": "unavailable", "error": error}

        retriever, error = self._get_retriever()
        index_dir = self.index_dir
        try:
            shown = index_dir.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            shown = index_dir.name
        if retriever is not None:
            comps["rag"] = {"status": "ok", "index": shown, "num_passages": retriever.index.size,
                            "embedding_model": retriever.index.model_name,
                            "uses_synthetic_demo_documents": bool(self._synthetic_ids)}
        else:
            comps["rag"] = {"status": "unavailable", "index": shown, "error": error}

        n_ok = sum(1 for c in comps.values() if c["status"] == "ok")
        overall = "ok" if n_ok == len(comps) else ("unavailable" if n_ok == 0 else "degraded")
        return {"api": "ok", "status": overall, "components": comps}
