"""
RAG configuration (Phase 7): default paths and tunable retrieval parameters.

Every value can be overridden with an environment variable (see `.env.example`) or by
passing an argument explicitly to the relevant function. Invalid environment values fall
back to the documented default instead of crashing at import time.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Small (~90 MB), CPU-friendly, 384-dimensional, Apache-2.0 sentence embedding model.
# It is downloaded once from the Hugging Face Hub on first use and cached locally.
DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Where the user places their reference documents (.txt / .pdf) and where the index is stored.
DEFAULT_DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", PROJECT_ROOT / "data" / "rag" / "reference_docs"))
DEFAULT_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", PROJECT_ROOT / "data" / "rag" / "index"))

# Synthetic demonstration documents shipped with the project (NOT real-world evidence).
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "rag" / "sample_reference_docs"

# Chunking, in characters.
DEFAULT_CHUNK_SIZE = _env_int("RAG_CHUNK_SIZE", 500)
DEFAULT_CHUNK_OVERLAP = _env_int("RAG_CHUNK_OVERLAP", 100)

# Retrieval.
DEFAULT_TOP_K = _env_int("RAG_TOP_K", 5)
# Minimum cosine similarity for a passage to be returned. This is a heuristic starting point
# for all-MiniLM-L6-v2, NOT a calibrated value: tune it on your own reference collection.
DEFAULT_MIN_SCORE = _env_float("RAG_MIN_SCORE", 0.35)
# Two retrieved passages whose word sets overlap at least this much (Jaccard) count as
# near-duplicates and only the higher-ranked one is returned.
DEFAULT_DEDUP_JACCARD = 0.90
