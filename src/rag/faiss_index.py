"""
FAISS index + persistent chunk metadata (Phase 7).

On-disk layout of an index directory (all plain, inspectable formats - no pickle):

    index.faiss    serialized FAISS IndexFlatIP (exact inner-product search)
    chunks.jsonl   one JSON object per chunk; line N describes vector N
    manifest.json  embedding model, dimension, chunking settings, per-document content hashes,
                   and SHA-256 checksums of the two files above

Vector ID == chunk_id == line number in chunks.jsonl. `load()` validates this mapping, the
checksums, the vector count and the dimension, and raises a typed error instead of returning
a half-valid index.

Similarity: vectors are L2-normalized, so the inner product returned by FAISS is the cosine
similarity (range [-1, 1], higher = more similar). It is NOT a probability and says nothing
about whether a passage is true.

Safety: the checksums detect accidental corruption and partial writes. They do not protect
against someone who can edit both the files and the manifest. FAISS' reader is not hardened
against malicious files, so only load indexes you built yourself.

Rebuilds write to a temporary sibling directory and swap it in, so a failed rebuild leaves
the previous index untouched. Embeddings of chunks whose text is unchanged are reused from
the previous index (same embedding model required) instead of being recomputed.
"""

import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.rag.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.rag.document_loader import LoadReport, load_documents
from src.rag.text_chunker import Chunk, chunk_documents, text_hash

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1
METRIC_DESCRIPTION = "cosine similarity (inner product of L2-normalized vectors)"


class EvidenceIndexError(Exception):
    """Base class for index problems."""


class IndexNotFoundError(EvidenceIndexError):
    """The index directory or one of its files does not exist."""


class IndexCorruptedError(EvidenceIndexError):
    """Index files are unreadable, inconsistent, or fail their checksums."""


class IndexMismatchError(EvidenceIndexError):
    """The index was built with a different embedding model than the one in use."""


class FaissNotInstalledError(EvidenceIndexError):
    """The `faiss` package could not be imported (not installed, or its native library failed to load).

    A subclass of EvidenceIndexError, so existing `except EvidenceIndexError` handlers keep working, while callers
    that need to tell "FAISS is missing" apart from "the index is missing/damaged" can catch this type instead of
    matching on message text. FAISS is checked BEFORE the index directory: without FAISS nothing can be loaded or
    built, so reporting "build the index" first would send the user down a path that fails.
    """


def _require_faiss():
    try:
        import faiss
    except ImportError as err:
        raise FaissNotInstalledError("faiss-cpu is not installed. Run: pip install -r requirements.txt") from err
    return faiss


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_chunk_index": chunk.doc_chunk_index,
        "text": chunk.text,
        "source_id": chunk.source_id,
        "filename": chunk.filename,
        "page": chunk.page,
        "text_hash": chunk.text_hash,
    }


def _chunk_from_dict(data: dict, expected_id: int) -> Chunk:
    try:
        chunk = Chunk(
            chunk_id=data["chunk_id"],
            doc_chunk_index=data["doc_chunk_index"],
            text=data["text"],
            source_id=data["source_id"],
            filename=data["filename"],
            page=data["page"],
            text_hash=data["text_hash"],
        )
    except (KeyError, TypeError) as err:
        raise IndexCorruptedError(f"Chunk record #{expected_id} is missing fields: {err}") from err
    valid = (
        isinstance(chunk.chunk_id, int)
        and isinstance(chunk.text, str)
        and chunk.text.strip() != ""
        and isinstance(chunk.source_id, str)
        and isinstance(chunk.filename, str)
        and (chunk.page is None or isinstance(chunk.page, int))
    )
    if not valid:
        raise IndexCorruptedError(f"Chunk record #{expected_id} has invalid field values.")
    if chunk.chunk_id != expected_id:
        raise IndexCorruptedError(f"Chunk id {chunk.chunk_id} found at position {expected_id}; vector IDs would not match.")
    if text_hash(chunk.text) != chunk.text_hash:
        raise IndexCorruptedError(f"Chunk #{expected_id} text does not match its stored hash.")
    return chunk


class EvidenceIndex:
    """A FAISS flat inner-product index plus the chunk metadata for each vector."""

    def __init__(self, index, chunks: List[Chunk], manifest: dict):
        self.index = index
        self.chunks = chunks
        self.manifest = manifest
        self.build_stats: Dict[str, int] = {}

    # ------------------------------------------------------------- properties
    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def dimension(self) -> int:
        return int(self.manifest["dimension"])

    @property
    def model_name(self) -> str:
        return self.manifest["embedding_model"]

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        chunks: List[Chunk],
        embedder,
        source_fingerprints: Optional[Dict[str, str]] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        previous: Optional["EvidenceIndex"] = None,
    ) -> "EvidenceIndex":
        """Embed `chunks` (reusing unchanged ones from `previous`) and build a new index."""
        faiss = _require_faiss()
        chunks = list(chunks)
        for position, chunk in enumerate(chunks):
            chunk.chunk_id = position  # vector ID == position

        dim = int(embedder.dimension)
        vectors = np.zeros((len(chunks), dim), dtype=np.float32)

        reusable: Dict[str, np.ndarray] = {}
        if previous is not None and not previous.is_empty and previous.model_name == embedder.model_name and previous.dimension == dim:
            old_vectors = previous.index.reconstruct_n(0, previous.size)
            for old_chunk, vec in zip(previous.chunks, old_vectors):
                reusable.setdefault(old_chunk.text_hash, vec)

        to_embed: List[int] = []
        for i, chunk in enumerate(chunks):
            if chunk.text_hash in reusable:
                vectors[i] = reusable[chunk.text_hash]
            else:
                to_embed.append(i)

        if to_embed:
            new_vectors = np.asarray(embedder.embed_texts([chunks[i].text for i in to_embed]), dtype=np.float32)
            if new_vectors.shape != (len(to_embed), dim):
                raise EvidenceIndexError(
                    f"Embedder returned shape {new_vectors.shape}, expected {(len(to_embed), dim)}."
                )
            if not np.all(np.isfinite(new_vectors)):
                raise EvidenceIndexError("Embedder returned NaN or infinite values.")
            vectors[to_embed] = new_vectors

        index = faiss.IndexFlatIP(dim)
        if len(chunks):
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": embedder.model_name,
            "dimension": dim,
            "metric": METRIC_DESCRIPTION,
            "index_type": "IndexFlatIP",
            "num_chunks": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "sources": dict(source_fingerprints or {}),
        }
        built = cls(index, chunks, manifest)
        built.build_stats = {"reused_embeddings": len(chunks) - len(to_embed), "new_embeddings": len(to_embed)}
        return built

    # ------------------------------------------------------------------- save
    def save(self, directory) -> Path:
        """Write the index atomically (temp directory, then swap). Returns the directory."""
        faiss = _require_faiss()
        target = Path(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex[:8]}"
        backup = target.parent / f".{target.name}.old-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            index_bytes = faiss.serialize_index(self.index).tobytes()
            chunks_bytes = ("\n".join(json.dumps(_chunk_to_dict(c), ensure_ascii=False) for c in self.chunks)).encode("utf-8")
            (tmp / INDEX_FILENAME).write_bytes(index_bytes)
            (tmp / CHUNKS_FILENAME).write_bytes(chunks_bytes)
            manifest = dict(self.manifest)
            manifest["num_chunks"] = len(self.chunks)
            manifest["index_sha256"] = _sha256_bytes(index_bytes)
            manifest["chunks_sha256"] = _sha256_bytes(chunks_bytes)
            (tmp / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

            had_old = target.exists()
            if had_old:
                os.replace(target, backup)
            try:
                os.replace(tmp, target)
            except Exception:
                if had_old:
                    os.replace(backup, target)  # restore the previous index
                raise
            if had_old:
                shutil.rmtree(backup, ignore_errors=True)
            self.manifest = manifest
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return target

    # ------------------------------------------------------------------- load
    @classmethod
    def load(cls, directory, expected_model_name: Optional[str] = None) -> "EvidenceIndex":
        """Load and fully validate an index directory."""
        faiss = _require_faiss()
        root = Path(directory)
        if not root.is_dir():
            raise IndexNotFoundError(f"Index directory not found: {root}")
        paths = {name: root / name for name in (INDEX_FILENAME, CHUNKS_FILENAME, MANIFEST_FILENAME)}
        missing = [name for name, p in paths.items() if not p.is_file()]
        if missing:
            raise IndexNotFoundError(f"Index directory {root} is missing: {', '.join(missing)}")

        try:
            manifest = json.loads(paths[MANIFEST_FILENAME].read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            raise IndexCorruptedError(f"Manifest is unreadable: {err}") from err
        if not isinstance(manifest, dict):
            raise IndexCorruptedError("Manifest is not a JSON object.")
        required = ("schema_version", "embedding_model", "dimension", "num_chunks", "index_sha256", "chunks_sha256")
        absent = [k for k in required if k not in manifest]
        if absent:
            raise IndexCorruptedError(f"Manifest is missing keys: {', '.join(absent)}")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise IndexCorruptedError(f"Unsupported index schema version {manifest['schema_version']!r}.")
        if not isinstance(manifest["dimension"], int) or manifest["dimension"] <= 0:
            raise IndexCorruptedError("Manifest has an invalid dimension.")

        if expected_model_name is not None and manifest["embedding_model"] != expected_model_name:
            raise IndexMismatchError(
                f"Index was built with '{manifest['embedding_model']}' but '{expected_model_name}' is in use. "
                f"Rebuild the index with the current model."
            )

        try:
            index_bytes = paths[INDEX_FILENAME].read_bytes()
            chunks_bytes = paths[CHUNKS_FILENAME].read_bytes()
        except OSError as err:
            raise IndexCorruptedError(f"Index files are unreadable: {err}") from err
        if _sha256_bytes(index_bytes) != manifest["index_sha256"]:
            raise IndexCorruptedError("index.faiss does not match its recorded checksum (corrupted or modified).")
        if _sha256_bytes(chunks_bytes) != manifest["chunks_sha256"]:
            raise IndexCorruptedError("chunks.jsonl does not match its recorded checksum (corrupted or modified).")

        chunks: List[Chunk] = []
        try:
            lines = [ln for ln in chunks_bytes.decode("utf-8").split("\n") if ln.strip()]
            for position, line in enumerate(lines):
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise IndexCorruptedError(f"Chunk record #{position} is not a JSON object.")
                chunks.append(_chunk_from_dict(record, position))
        except (UnicodeDecodeError, ValueError) as err:
            raise IndexCorruptedError(f"chunks.jsonl is unreadable: {err}") from err

        try:
            index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8).copy())
        except Exception as err:
            raise IndexCorruptedError(f"FAISS could not read index.faiss: {err}") from err

        if index.d != manifest["dimension"]:
            raise IndexCorruptedError(f"Index dimension {index.d} != manifest dimension {manifest['dimension']}.")
        if index.ntotal != len(chunks) or manifest["num_chunks"] != len(chunks):
            raise IndexCorruptedError(
                f"Vector count ({index.ntotal}), chunk records ({len(chunks)}) and manifest "
                f"({manifest['num_chunks']}) disagree."
            )
        return cls(index, chunks, manifest)

    # ----------------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        """Return up to k (chunk, cosine similarity) pairs, best first. Empty index -> []."""
        if self.is_empty or k <= 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.shape != (1, self.dimension):
            raise EvidenceIndexError(
                f"Query vector has shape {query.shape}; the index expects dimension {self.dimension}."
            )
        if not np.all(np.isfinite(query)):
            raise EvidenceIndexError("Query vector contains NaN or infinite values.")
        scores, ids = self.index.search(np.ascontiguousarray(query), min(k, self.size))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append((self.chunks[int(idx)], float(max(-1.0, min(1.0, score)))))
        return results


# ---------------------------------------------------------------- build / update
@dataclass
class BuildResult:
    index: EvidenceIndex
    status: str  # "created" | "rebuilt" | "unchanged"
    load_report: LoadReport
    reused_embeddings: int = 0
    new_embeddings: int = 0
    warnings: List[str] = field(default_factory=list)


def _is_current(manifest: dict, fingerprints: Dict[str, str], chunk_size: int, overlap: int, model_name: str) -> bool:
    return (
        manifest.get("sources") == fingerprints
        and manifest.get("chunk_size") == chunk_size
        and manifest.get("chunk_overlap") == overlap
        and manifest.get("embedding_model") == model_name
    )


def build_or_update_index(
    docs_dir,
    index_dir,
    embedder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    force: bool = False,
    recursive: bool = True,
) -> BuildResult:
    """
    Make sure `index_dir` holds an index for the documents in `docs_dir`.

    * No index yet                      -> build and save ("created")
    * Documents/settings/model changed   -> rebuild, reusing unchanged embeddings ("rebuilt")
    * Nothing changed and not `force`    -> load the existing index ("unchanged")
    * Existing index corrupted           -> rebuild from scratch, with a warning

    Raises DocumentLoadError if `docs_dir` is missing; in that case the existing index is
    left untouched. If the directory has no usable documents an EMPTY index is saved.
    """
    report = load_documents(docs_dir, recursive=recursive)  # may raise before anything is changed
    fingerprints = {d.source_id: d.content_hash for d in report.documents}
    warnings: List[str] = []

    existing: Optional[EvidenceIndex] = None
    try:
        existing = EvidenceIndex.load(index_dir)
    except IndexNotFoundError:
        existing = None
    except EvidenceIndexError as err:
        warnings.append(f"Existing index could not be used and will be rebuilt: {err}")
        existing = None

    if existing is not None and not force and _is_current(existing.manifest, fingerprints, chunk_size, chunk_overlap, embedder.model_name):
        return BuildResult(existing, "unchanged", report, warnings=warnings)

    chunks = chunk_documents(report.documents, chunk_size, chunk_overlap)
    if not chunks:
        warnings.append("No usable text found in the reference directory; the index is empty.")
    new_index = EvidenceIndex.build(
        chunks, embedder, fingerprints, chunk_size, chunk_overlap, previous=None if force else existing
    )
    new_index.save(index_dir)
    return BuildResult(
        new_index,
        "rebuilt" if existing is not None else "created",
        report,
        reused_embeddings=new_index.build_stats["reused_embeddings"],
        new_embeddings=new_index.build_stats["new_embeddings"],
        warnings=warnings,
    )
