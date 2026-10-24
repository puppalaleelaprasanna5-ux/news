"""
Phase 7 tests: embeddings, FAISS index, persistence, retrieval.

Uses the dependency-free HashingEmbedder (lexical, deterministic) so no model download is
needed; a mocked sentence-transformers module covers the real embedder wrapper. Tests that
need FAISS skip if faiss-cpu is not installed (it is listed in requirements.txt).

    python -m pytest tests/test_rag_retrieval.py -v
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embeddings import EmbeddingError, HashingEmbedder, SentenceTransformerEmbedder
from src.rag.retriever import (
    STATUS_INVALID, STATUS_NONE, STATUS_RETRIEVED, STATUS_UNAVAILABLE, EvidenceRetriever,
)

faiss = pytest.importorskip("faiss")

from src.rag.faiss_index import (  # noqa: E402  (needs faiss importable first)
    CHUNKS_FILENAME, INDEX_FILENAME, MANIFEST_FILENAME, EvidenceIndex, IndexCorruptedError,
    IndexMismatchError, IndexNotFoundError, build_or_update_index,
)
from src.rag.text_chunker import chunk_documents  # noqa: E402
from src.rag.document_loader import load_documents  # noqa: E402

DOCS = {
    "rivers.txt": "The Alder River flows through the town of Brightwater. The river floods every spring. "
                  "Engineers built a levee along the Alder River in the year 1990.",
    "library.txt": "The Brightwater public library opened a new reading room. The library hosts a weekly chess club. "
                   "Library membership is free for all residents.",
    "transit.txt": "The city tram network adds two new stations next month. Tram tickets will cost one dollar. "
                   "Trams run every ten minutes during rush hour.",
}


def write_docs(directory, docs=DOCS):
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def build(tmp_path, docs=DOCS, embedder=None, chunk_size=140, overlap=30):
    embedder = embedder or HashingEmbedder()
    docs_dir = write_docs(tmp_path / "docs", docs)
    result = build_or_update_index(docs_dir, tmp_path / "index", embedder, chunk_size, overlap)
    return result, embedder, docs_dir


# ------------------------------------------------------------------ embeddings
def test_embedding_dimensions_and_normalization():
    emb = HashingEmbedder(dim=64)
    vectors = emb.embed_texts(["first text", "second text here", "third"])
    assert vectors.shape == (3, 64) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert emb.embed_query("a query").shape == (64,)
    assert emb.dimension == 64


def test_embedding_empty_inputs_raise_embedding_error():
    emb = HashingEmbedder()
    for bad in ("", "   ", None, 5):
        with pytest.raises(EmbeddingError):
            emb.embed_query(bad)
    with pytest.raises(EmbeddingError):
        emb.embed_texts(["ok", ""])
    with pytest.raises(EmbeddingError):
        emb.embed_texts("just a string")
    assert emb.embed_texts([]).shape == (0, 256)


def test_sentence_transformer_wrapper_with_mocked_model():
    class FakeModel:
        def __init__(self, name, device=None, local_files_only=False):
            self.name, self.calls = name, 0

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            self.calls += 1
            return np.array([[3.0, 4.0, 0.0]] * len(texts))  # deliberately NOT normalized

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    saved = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = module
    try:
        emb = SentenceTransformerEmbedder("fake-model")
        vecs = emb.embed_texts(["a", "b"])
        assert vecs.shape == (2, 3)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)  # wrapper normalizes
        assert emb.dimension == 3
        with pytest.raises(EmbeddingError):
            emb.embed_query("   ")
    finally:
        if saved is None:
            del sys.modules["sentence_transformers"]
        else:
            sys.modules["sentence_transformers"] = saved


def test_sentence_transformer_load_failure_is_embedding_error():
    class Boom:
        def __init__(self, *a, **k):
            raise OSError("no internet and model not cached")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = Boom
    saved = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = module
    try:
        with pytest.raises(EmbeddingError, match="Could not load embedding model"):
            SentenceTransformerEmbedder("missing-model").embed_query("hello")
    finally:
        if saved is None:
            del sys.modules["sentence_transformers"]
        else:
            sys.modules["sentence_transformers"] = saved


# ----------------------------------------------------------------- FAISS index
def test_index_creation_counts_and_dimension(tmp_path):
    result, emb, _ = build(tmp_path)
    assert result.status == "created"
    index = result.index
    assert index.size == index.index.ntotal > 0
    assert index.dimension == index.index.d == emb.dimension
    assert result.new_embeddings == index.size and result.reused_embeddings == 0
    assert {c.filename for c in index.chunks} == set(DOCS)


def test_save_and_load_roundtrip_preserves_everything(tmp_path):
    result, emb, _ = build(tmp_path)
    for name in (INDEX_FILENAME, CHUNKS_FILENAME, MANIFEST_FILENAME):
        assert (tmp_path / "index" / name).is_file()
    loaded = EvidenceIndex.load(tmp_path / "index", expected_model_name=emb.model_name)
    assert loaded.size == result.index.size
    assert [c.text for c in loaded.chunks] == [c.text for c in result.index.chunks]
    assert [(c.chunk_id, c.filename, c.page) for c in loaded.chunks] == [
        (c.chunk_id, c.filename, c.page) for c in result.index.chunks
    ]
    q = emb.embed_query("Alder River flood levee")
    assert [(c.chunk_id, round(s, 5)) for c, s in loaded.search(q, 3)] == [
        (c.chunk_id, round(s, 5)) for c, s in result.index.search(q, 3)
    ]


def test_search_results_map_to_the_right_source_passage(tmp_path):
    result, emb, _ = build(tmp_path)
    q = emb.embed_query("The library hosts a weekly chess club")
    chunk, score = result.index.search(q, 1)[0]
    assert chunk.filename == "library.txt" and "chess" in chunk.text
    # the stored vector for that chunk id really is the embedding of that chunk's text
    stored = result.index.index.reconstruct_n(chunk.chunk_id, 1)[0]
    assert np.allclose(stored, emb.embed_texts([chunk.text])[0], atol=1e-5)
    assert score == pytest.approx(float(stored @ q), abs=1e-5)


def test_empty_collection_builds_an_empty_index_without_crashing(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "blank.txt").write_text("   ", encoding="utf-8")
    result = build_or_update_index(docs_dir, tmp_path / "index", HashingEmbedder(), 200, 20)
    assert result.index.is_empty and result.warnings
    reloaded = EvidenceIndex.load(tmp_path / "index")
    assert reloaded.is_empty and reloaded.search(np.ones(256, dtype=np.float32), 5) == []


def test_missing_docs_directory_leaves_existing_index_untouched(tmp_path):
    result, emb, docs_dir = build(tmp_path)
    before = (tmp_path / "index" / MANIFEST_FILENAME).read_text()
    from src.rag.document_loader import DocumentLoadError
    with pytest.raises(DocumentLoadError):
        build_or_update_index(tmp_path / "gone", tmp_path / "index", emb)
    assert (tmp_path / "index" / MANIFEST_FILENAME).read_text() == before


def test_unchanged_documents_are_not_re_embedded(tmp_path):
    result, emb, docs_dir = build(tmp_path)
    again = build_or_update_index(docs_dir, tmp_path / "index", emb, 140, 30)
    assert again.status == "unchanged" and again.new_embeddings == 0


def test_changed_collection_rebuilds_and_reuses_unchanged_embeddings(tmp_path):
    result, emb, docs_dir = build(tmp_path)
    calls = []
    original = emb.embed_texts
    emb.embed_texts = lambda texts: (calls.append(len(texts)), original(texts))[1]
    (docs_dir / "transit.txt").write_text("Trams now run every five minutes. Night buses were added.", encoding="utf-8")
    rebuilt = build_or_update_index(docs_dir, tmp_path / "index", emb, 140, 30)
    assert rebuilt.status == "rebuilt"
    assert rebuilt.reused_embeddings > 0 and rebuilt.new_embeddings > 0
    assert sum(calls) == rebuilt.new_embeddings  # only new chunks were sent to the embedder
    texts = " ".join(c.text for c in rebuilt.index.chunks)
    assert "five minutes" in texts and "one dollar" not in texts  # stale passages are gone
    assert [c.chunk_id for c in rebuilt.index.chunks] == list(range(rebuilt.index.size))


def test_changed_chunk_settings_or_force_triggers_rebuild(tmp_path):
    result, emb, docs_dir = build(tmp_path)
    assert build_or_update_index(docs_dir, tmp_path / "index", emb, 200, 30).status == "rebuilt"
    forced = build_or_update_index(docs_dir, tmp_path / "index", emb, 200, 30, force=True)
    assert forced.status == "rebuilt" and forced.reused_embeddings == 0


def test_removed_document_disappears_from_index(tmp_path):
    result, emb, docs_dir = build(tmp_path)
    (docs_dir / "library.txt").unlink()
    rebuilt = build_or_update_index(docs_dir, tmp_path / "index", emb, 140, 30)
    assert "library.txt" not in {c.filename for c in rebuilt.index.chunks}


# ------------------------------------------------------ invalid / corrupted files
def test_load_missing_index_raises_not_found(tmp_path):
    with pytest.raises(IndexNotFoundError):
        EvidenceIndex.load(tmp_path / "nothing_here")
    (tmp_path / "partial").mkdir()
    with pytest.raises(IndexNotFoundError):
        EvidenceIndex.load(tmp_path / "partial")


def test_corrupted_index_file_is_detected(tmp_path):
    build(tmp_path)
    path = tmp_path / "index" / INDEX_FILENAME
    path.write_bytes(path.read_bytes()[:-5] + b"XXXXX")
    with pytest.raises(IndexCorruptedError, match="checksum"):
        EvidenceIndex.load(tmp_path / "index")


def test_tampered_metadata_is_detected(tmp_path):
    build(tmp_path)
    path = tmp_path / "index" / CHUNKS_FILENAME
    path.write_text(path.read_text().replace("Alder", "Elder", 1), encoding="utf-8")
    with pytest.raises(IndexCorruptedError):
        EvidenceIndex.load(tmp_path / "index")


def test_garbage_manifest_is_detected(tmp_path):
    build(tmp_path)
    manifest = tmp_path / "index" / MANIFEST_FILENAME
    manifest.write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexCorruptedError):
        EvidenceIndex.load(tmp_path / "index")
    manifest.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(IndexCorruptedError, match="missing keys"):
        EvidenceIndex.load(tmp_path / "index")


def test_chunk_count_mismatch_and_wrong_ids_are_detected(tmp_path):
    result, emb, _ = build(tmp_path)
    from src.rag import faiss_index as fi
    # chunk records reordered but with *valid* checksums -> still caught via id/position check
    lines = (tmp_path / "index" / CHUNKS_FILENAME).read_text().split("\n")
    lines[0], lines[1] = lines[1], lines[0]
    data = "\n".join(lines).encode("utf-8")
    (tmp_path / "index" / CHUNKS_FILENAME).write_bytes(data)
    manifest_path = tmp_path / "index" / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["chunks_sha256"] = fi._sha256_bytes(data)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(IndexCorruptedError, match="vector IDs"):
        EvidenceIndex.load(tmp_path / "index")


def test_index_built_with_other_model_is_refused(tmp_path):
    build(tmp_path)
    with pytest.raises(IndexMismatchError):
        EvidenceIndex.load(tmp_path / "index", expected_model_name="some-other-model")


def test_corrupted_existing_index_is_rebuilt_with_warning(tmp_path):
    result, emb, docs_dir = build(tmp_path)
    (tmp_path / "index" / INDEX_FILENAME).write_bytes(b"junk")
    rebuilt = build_or_update_index(docs_dir, tmp_path / "index", emb, 140, 30)
    assert rebuilt.warnings and rebuilt.index.size > 0
    EvidenceIndex.load(tmp_path / "index")  # now valid again


def test_search_rejects_wrong_dimension_query(tmp_path):
    result, _, _ = build(tmp_path)
    from src.rag.faiss_index import EvidenceIndexError
    with pytest.raises(EvidenceIndexError):
        result.index.search(np.ones(7, dtype=np.float32), 3)


# ------------------------------------------------------------------- retrieval
def make_retriever(tmp_path, **kwargs):
    result, emb, _ = build(tmp_path)
    return EvidenceRetriever(result.index, emb, **kwargs), result


def test_retrieval_returns_documented_schema_with_real_scores(tmp_path):
    retriever, result = make_retriever(tmp_path, top_k=3, min_score=0.1)
    out = retriever.retrieve("The Alder River floods every spring")
    assert out["retrieval_status"] == STATUS_RETRIEVED
    assert out["claim"] == "The Alder River floods every spring"
    top = out["retrieved_evidence"][0]
    assert set(top) == {"text", "source", "source_id", "page", "chunk_id", "rank", "score", "score_type"}
    assert top["source"] == "rivers.txt" and top["page"] is None and top["rank"] == 1
    assert top["score_type"] == "cosine_similarity" and -1.0 <= top["score"] <= 1.0
    # text/source/chunk_id are exactly what is stored for that chunk
    stored = result.index.chunks[top["chunk_id"]]
    assert top["text"] == stored.text and top["source_id"] == stored.source_id
    assert any("does not establish factual correctness" in lim for lim in out["limitations"])
    scores = [e["score"] for e in out["retrieved_evidence"]]
    assert scores == sorted(scores, reverse=True)
    assert [e["rank"] for e in out["retrieved_evidence"]] == list(range(1, len(scores) + 1))
    assert "verdict" not in out and "verified" not in json.dumps(out).lower().replace("not establish", "")


def test_top_k_is_respected(tmp_path):
    for k in (1, 2):
        retriever, _ = make_retriever(tmp_path / f"k{k}", top_k=k, min_score=None)
        assert len(retriever.retrieve("Brightwater river library tram")["retrieved_evidence"]) <= k
    retriever, _ = make_retriever(tmp_path / "override", top_k=5, min_score=None)
    assert len(retriever.retrieve("Brightwater river library tram", top_k=1)["retrieved_evidence"]) == 1
    with pytest.raises(ValueError):
        EvidenceRetriever(None, HashingEmbedder(), top_k=0)


def test_empty_and_invalid_claims_are_reported_not_raised(tmp_path):
    retriever, _ = make_retriever(tmp_path)
    for bad in ("", "   ", None, 42):
        out = retriever.retrieve(bad)
        assert out["retrieval_status"] == STATUS_INVALID and out["retrieved_evidence"] == []


def test_no_relevant_evidence_status_when_nothing_passes_threshold(tmp_path):
    retriever, _ = make_retriever(tmp_path, min_score=0.35)
    out = retriever.retrieve("Quantum computers factor large prime numbers")
    assert out["retrieval_status"] == STATUS_NONE and out["retrieved_evidence"] == []
    assert any("does not mean the claim is true or false" in lim for lim in out["limitations"])


def test_threshold_can_be_overridden_per_call(tmp_path):
    retriever, _ = make_retriever(tmp_path, min_score=0.99)
    claim = "Alder River levee"
    assert retriever.retrieve(claim)["retrieval_status"] == STATUS_NONE
    assert retriever.retrieve(claim, min_score=None)["retrieval_status"] == STATUS_RETRIEVED


def test_missing_index_gives_retrieval_unavailable(tmp_path):
    retriever = EvidenceRetriever.from_directory(tmp_path / "no_index", HashingEmbedder())
    out = retriever.retrieve("Any claim at all")
    assert out["retrieval_status"] == STATUS_UNAVAILABLE and out["retrieved_evidence"] == []
    assert "not found" in out["error"].lower()


def test_corrupted_index_gives_retrieval_unavailable(tmp_path):
    build(tmp_path)
    (tmp_path / "index" / INDEX_FILENAME).write_bytes(b"garbage")
    out = EvidenceRetriever.from_directory(tmp_path / "index", HashingEmbedder()).retrieve("claim text")
    assert out["retrieval_status"] == STATUS_UNAVAILABLE and out["error"]


def test_model_mismatch_gives_retrieval_unavailable(tmp_path):
    build(tmp_path)
    out = EvidenceRetriever.from_directory(tmp_path / "index", HashingEmbedder(dim=32)).retrieve("claim text")
    assert out["retrieval_status"] == STATUS_UNAVAILABLE and "model" in out["error"].lower()


def test_empty_index_gives_no_relevant_evidence(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    result = build_or_update_index(docs_dir, tmp_path / "index", HashingEmbedder())
    out = EvidenceRetriever(result.index, HashingEmbedder()).retrieve("Something to look up")
    assert out["retrieval_status"] == STATUS_NONE


def test_embedding_failure_during_retrieval_is_reported(tmp_path):
    retriever, _ = make_retriever(tmp_path)

    class Broken:
        model_name = retriever.embedder.model_name
        dimension = retriever.embedder.dimension

        def embed_query(self, q):
            raise EmbeddingError("model exploded")

    retriever.embedder = Broken()
    out = retriever.retrieve("A valid claim")
    assert out["retrieval_status"] == STATUS_UNAVAILABLE and "exploded" in out["error"]


def test_multiple_claims_are_handled_independently(tmp_path):
    retriever, _ = make_retriever(tmp_path, min_score=0.2)
    claims = ["The Alder River floods every spring", "", "Tram tickets will cost one dollar", "Quantum entanglement of photons"]
    outs = retriever.retrieve_many(claims)
    assert [o["claim"] for o in outs] == claims
    assert [o["retrieval_status"] for o in outs] == [STATUS_RETRIEVED, STATUS_INVALID, STATUS_RETRIEVED, STATUS_NONE]
    assert outs[0]["retrieved_evidence"][0]["source"] == "rivers.txt"
    assert outs[2]["retrieved_evidence"][0]["source"] == "transit.txt"


def test_near_duplicate_passages_are_returned_once(tmp_path):
    same = "The harbour bridge reopened on Monday after repairs were completed by the council."
    result, emb, _ = build(tmp_path, docs={"a.txt": same, "b.txt": same, "c.txt": same + " Traffic is flowing normally."})
    retriever = EvidenceRetriever(result.index, emb, top_k=5, min_score=0.1, dedup_jaccard=0.9)
    out = retriever.retrieve("harbour bridge reopened after repairs")
    texts = [e["text"] for e in out["retrieved_evidence"]]
    assert len(texts) == len(set(texts))
    assert len(texts) < 3  # the identical copies collapsed to one
    assert [e["rank"] for e in out["retrieved_evidence"]] == list(range(1, len(texts) + 1))


def test_pdf_page_numbers_flow_through_to_results(tmp_path):
    pytest.importorskip("pypdf")
    from tests.test_rag_ingestion import make_pdf
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "paper.pdf").write_bytes(make_pdf(["Glaciers are retreating quickly", "Volcanoes erupted twice this decade"]))
    emb = HashingEmbedder()
    result = build_or_update_index(docs_dir, tmp_path / "index", emb, 200, 20)
    out = EvidenceRetriever(result.index, emb, min_score=0.1).retrieve("Volcanoes erupted")
    assert out["retrieved_evidence"][0]["page"] == 2 and out["retrieved_evidence"][0]["source"] == "paper.pdf"
