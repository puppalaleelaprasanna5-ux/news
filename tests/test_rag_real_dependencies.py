"""
Real-library verification for Phase 7 (added in the Phase 7 verification pass).

Every test here uses the REAL `faiss-cpu` library and the REAL Sentence-Transformers model
(`all-MiniLM-L6-v2`, or whatever RAG_EMBEDDING_MODEL is set to). No stand-in embedder, no mocked model and
no NumPy replacement for FAISS is used anywhere in this file.

If `faiss` or `sentence-transformers` is not installed, or the embedding model cannot be loaded (for example
the very first run without internet access), each test is SKIPPED with the reason. A skip means UNVERIFIED -
it does not mean the check passed. Check that the summary shows 0 skips in this file before treating Phase 7 as
verified on your machine.

The embedding model (~90 MB) is downloaded from the Hugging Face Hub once, on the first run; afterwards it is
used from the local cache. These tests take roughly a minute because they start the model several times
(including through the command-line tool).

    python -m pytest tests/test_rag_real_dependencies.py -v -rs
    python -m unittest tests.test_rag_real_dependencies -v

What is checked (items 3-5 of the Phase 7 verification list):
    embedding model: loads, 384 dims, float32, unit length, deterministic, query/document compatible,
                     semantically sensible ordering on the synthetic sample documents
    real FAISS:      IndexFlatIP creation, vector count/dimension, vector-ID <-> chunk-text mapping,
                     save + reload, top-k ranking, score == cosine similarity, empty index,
                     missing/corrupted/mismatched index, rebuild when documents change
    end to end:      build -> save -> reload -> search; an unrelated claim returns no evidence;
                     the documented CLI commands, including `--offline` use of the cached model
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, SAMPLE_DOCS_DIR
from src.rag.document_loader import load_documents
from src.rag.text_chunker import chunk_documents

HAS_FAISS = importlib.util.find_spec("faiss") is not None
HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None

# claim -> the sample document that should rank first for it
CLAIMS_AND_SOURCES = [
    ("The levee along the Alder River is 4.2 kilometres long.", "brightwater_flood_report.txt"),
    ("Library membership is free for all residents of Brightwater.", "brightwater_library_notes.txt"),
    ("A single tram ticket costs two dollars.", "northfield_tram_bulletin.txt"),
]
UNRELATED_CLAIM = "Quantum computers use qubits to run Shor's algorithm for integer factorisation."

_cache = {}


def _require_faiss(test):
    if not HAS_FAISS:
        test.skipTest("faiss-cpu is not installed - real FAISS behaviour is UNVERIFIED")


def _real_embedder(test, need_faiss=True):
    """Return the real SentenceTransformerEmbedder (loaded once), or skip the test with the reason."""
    if need_faiss:
        _require_faiss(test)
    if not HAS_SENTENCE_TRANSFORMERS:
        test.skipTest("sentence-transformers is not installed - the real embedding model is UNVERIFIED")
    if "embedder" not in _cache:
        from src.rag.embeddings import EmbeddingError, SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder()
        try:
            embedder.embed_query("warm-up")  # forces the model to load now
            _cache["embedder"], _cache["why"] = embedder, None
        except EmbeddingError as err:
            _cache["embedder"], _cache["why"] = None, str(err)
    if _cache["embedder"] is None:
        test.skipTest(f"embedding model could not be loaded ({_cache['why'][:150]}) - UNVERIFIED")
    return _cache["embedder"]


def _sample_chunks():
    return chunk_documents(load_documents(SAMPLE_DOCS_DIR).documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)


class TestRealEmbeddingModel(unittest.TestCase):

    def test_model_loads_and_produces_expected_shape(self):
        emb = _real_embedder(self, need_faiss=False)
        vectors = emb.embed_texts(["The levee is 4.2 kilometres long.", "Trams run every ten minutes."])
        self.assertEqual(vectors.dtype, np.float32)
        self.assertEqual(vectors.shape, (2, emb.dimension))
        if emb.model_name == "sentence-transformers/all-MiniLM-L6-v2":
            self.assertEqual(emb.dimension, 384)

    def test_embeddings_are_unit_length(self):
        emb = _real_embedder(self, need_faiss=False)
        norms = np.linalg.norm(emb.embed_texts([c.text for c in _sample_chunks()]), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_query_and_document_embeddings_are_compatible(self):
        emb = _real_embedder(self, need_faiss=False)
        text = "The Brightwater Works Department completed a stone levee."
        query_vector, doc_vector = emb.embed_query(text), emb.embed_texts([text])[0]
        self.assertEqual(query_vector.shape, doc_vector.shape)
        self.assertGreater(float(np.dot(query_vector, doc_vector)), 0.9999)  # same model, same text

    def test_embedding_is_deterministic(self):
        emb = _real_embedder(self, need_faiss=False)
        a, b = emb.embed_texts(["Trams run every ten minutes."]), emb.embed_texts(["Trams run every ten minutes."])
        np.testing.assert_allclose(a, b, atol=1e-4)

    def test_inner_product_is_cosine_similarity(self):
        emb = _real_embedder(self, need_faiss=False)
        x, y = emb.embed_texts(["The levee is long.", "Trams run often."])
        cosine = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
        self.assertAlmostEqual(float(np.dot(x, y)), cosine, places=4)

    def test_related_claims_rank_their_own_source_first(self):
        emb = _real_embedder(self, need_faiss=False)
        chunks = _sample_chunks()
        chunk_vectors = emb.embed_texts([c.text for c in chunks])
        for claim, expected_source in CLAIMS_AND_SOURCES:
            with self.subTest(claim=claim):
                scores = chunk_vectors @ emb.embed_query(claim)
                self.assertEqual(chunks[int(np.argmax(scores))].filename, expected_source,
                                 f"top score {scores.max():.3f}")


class TestRealFaissEndToEnd(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _build(self, emb, docs=SAMPLE_DOCS_DIR, index_dir=None, **kwargs):
        from src.rag.faiss_index import build_or_update_index

        return build_or_update_index(docs, index_dir or self.tmp / "index", emb, **kwargs)

    # ---- build / save / reload / mapping ------------------------------------------------------------
    def test_build_save_reload_and_vector_to_chunk_mapping(self):
        emb = _real_embedder(self)  # skips first if FAISS / the model are unavailable
        import faiss
        from src.rag.faiss_index import EvidenceIndex

        expected_chunks = _sample_chunks()
        result = self._build(emb)

        self.assertEqual(result.status, "created")
        self.assertGreater(result.index.size, 0)
        self.assertEqual(result.index.size, len(expected_chunks))
        self.assertIsInstance(result.index.index, faiss.IndexFlatIP)          # real FAISS object
        self.assertEqual(result.index.index.ntotal, len(expected_chunks))       # every vector was added
        self.assertEqual(result.index.index.d, emb.dimension)                   # correct dimension
        for name in ("index.faiss", "chunks.jsonl", "manifest.json"):
            self.assertTrue((self.tmp / "index" / name).is_file(), name)

        reloaded = EvidenceIndex.load(self.tmp / "index", expected_model_name=emb.model_name)
        self.assertEqual(reloaded.size, result.index.size)
        self.assertEqual(reloaded.index.ntotal, reloaded.size)
        self.assertEqual(reloaded.model_name, emb.model_name)
        # vector ID i must belong to chunk i: re-embed the chunk text and compare with the stored vector
        for i, chunk in enumerate(reloaded.chunks):
            self.assertEqual(chunk.chunk_id, i)
            np.testing.assert_allclose(reloaded.index.reconstruct(i), emb.embed_texts([chunk.text])[0], atol=1e-4)

    # ---- retrieval ---------------------------------------------------------------------------------
    def test_top_k_retrieval_fields_ranks_and_score_meaning(self):
        from src.rag.faiss_index import EvidenceIndex
        from src.rag.retriever import EvidenceRetriever

        emb = _real_embedder(self)
        self._build(emb)
        index = EvidenceIndex.load(self.tmp / "index", expected_model_name=emb.model_name)  # reload from disk
        retriever = EvidenceRetriever(index, emb, top_k=3)

        for claim, expected_source in CLAIMS_AND_SOURCES:
            with self.subTest(claim=claim):
                out = retriever.retrieve(claim)
                self.assertEqual(out["retrieval_status"], "evidence_retrieved", out)
                evidence = out["retrieved_evidence"]
                self.assertLessEqual(len(evidence), 3)
                self.assertEqual(evidence[0]["source"], expected_source)
                self.assertEqual([e["rank"] for e in evidence], list(range(1, len(evidence) + 1)))
                scores = [e["score"] for e in evidence]
                self.assertEqual(scores, sorted(scores, reverse=True))
                query_vector = emb.embed_query(claim)
                for e in evidence:
                    chunk = index.chunks[e["chunk_id"]]
                    self.assertEqual(e["text"], chunk.text)
                    self.assertEqual(e["source"], chunk.filename)
                    self.assertIsNone(e["page"])                       # .txt files have no pages
                    self.assertEqual(e["score_type"], "cosine_similarity")
                    self.assertTrue(-1.0 <= e["score"] <= 1.0)
                    # the reported score really is the cosine similarity of claim and stored vector
                    self.assertAlmostEqual(e["score"], float(np.dot(query_vector, index.index.reconstruct(e["chunk_id"]))), places=3)

    def test_unrelated_claim_returns_no_relevant_evidence(self):
        from src.rag.faiss_index import EvidenceIndex
        from src.rag.retriever import EvidenceRetriever

        emb = _real_embedder(self)
        self._build(emb)
        retriever = EvidenceRetriever(EvidenceIndex.load(self.tmp / "index", expected_model_name=emb.model_name), emb)
        best = retriever.retrieve(UNRELATED_CLAIM, min_score=None)["retrieved_evidence"][0]["score"]
        out = retriever.retrieve(UNRELATED_CLAIM)
        self.assertEqual(
            out["retrieval_status"], "no_relevant_evidence",
            f"best unrelated score was {best:.3f} vs threshold {retriever.min_score}; if this fails, "
            f"RAG_MIN_SCORE needs tuning for your documents",
        )
        self.assertEqual(out["retrieved_evidence"], [])

    # ---- empty / missing / corrupted / mismatched -----------------------------------------------------
    def test_empty_index_behaviour(self):
        from src.rag.faiss_index import EvidenceIndex
        from src.rag.retriever import EvidenceRetriever

        emb = _real_embedder(self)
        empty = EvidenceIndex.build([], emb)
        self.assertEqual(empty.search(emb.embed_query("anything at all"), 3), [])
        empty.save(self.tmp / "empty")
        reloaded = EvidenceIndex.load(self.tmp / "empty", expected_model_name=emb.model_name)
        self.assertTrue(reloaded.is_empty)
        self.assertEqual(EvidenceRetriever(reloaded, emb).retrieve("A claim.")["retrieval_status"], "no_relevant_evidence")

    def test_missing_corrupted_and_mismatched_index_are_reported_not_crashed(self):
        from src.rag.embeddings import SentenceTransformerEmbedder
        from src.rag.faiss_index import EvidenceIndex, IndexCorruptedError, IndexMismatchError, IndexNotFoundError
        from src.rag.retriever import EvidenceRetriever

        emb = _real_embedder(self)
        self._build(emb)
        index_dir = self.tmp / "index"

        with self.assertRaises(IndexNotFoundError):
            EvidenceIndex.load(self.tmp / "does_not_exist")
        with self.assertRaises(IndexMismatchError):
            EvidenceIndex.load(index_dir, expected_model_name="some/other-model")

        retriever = EvidenceRetriever.from_directory(index_dir, SentenceTransformerEmbedder("some/other-model"))
        out = retriever.retrieve("A claim.")
        self.assertEqual(out["retrieval_status"], "retrieval_unavailable")  # model mismatch -> status, not exception

        faiss_file = index_dir / "index.faiss"
        data = faiss_file.read_bytes()
        faiss_file.write_bytes(data[: len(data) // 2])                      # truncated file
        with self.assertRaises(IndexCorruptedError):
            EvidenceIndex.load(index_dir)
        faiss_file.unlink()                                                  # deleted file
        with self.assertRaises(IndexNotFoundError):
            EvidenceIndex.load(index_dir)

    # ---- rebuild when documents change ------------------------------------------------------------------
    def test_index_is_rebuilt_when_documents_change(self):
        from src.rag.retriever import EvidenceRetriever

        emb = _real_embedder(self)
        docs = self.tmp / "docs"
        shutil.copytree(SAMPLE_DOCS_DIR, docs)
        self.assertEqual(self._build(emb, docs).status, "created")
        self.assertEqual(self._build(emb, docs).status, "unchanged")

        flood = docs / "brightwater_flood_report.txt"
        flood.write_text(
            flood.read_text(encoding="utf-8")
            + "\n\nA bronze statue of a heron was unveiled in Brightwater Square in the fictional spring.\n",
            encoding="utf-8",
        )
        changed = self._build(emb, docs)
        self.assertEqual(changed.status, "rebuilt")
        self.assertGreater(changed.new_embeddings, 0)
        self.assertGreater(changed.reused_embeddings, 0)        # unchanged chunks were not re-embedded
        # threshold disabled: this test is about the rebuilt index containing the new text, not about min_score
        out = EvidenceRetriever(changed.index, emb, top_k=1).retrieve(
            "A bronze heron statue was unveiled in Brightwater Square.", min_score=None
        )
        self.assertEqual(out["retrieved_evidence"][0]["source"], "brightwater_flood_report.txt")
        self.assertIn("heron", out["retrieved_evidence"][0]["text"])

        (docs / "brightwater_library_notes.txt").unlink()
        removed = self._build(emb, docs)
        self.assertEqual(removed.status, "rebuilt")
        self.assertNotIn("brightwater_library_notes.txt", {c.filename for c in removed.index.chunks})


class TestDocumentedCommandLine(unittest.TestCase):
    """Runs the README's `scripts/rag_cli.py` commands as real subprocesses with the real model and FAISS."""

    def _run(self, *args):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        return subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "rag_cli.py"), *args],
                              cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", env=env, timeout=900)

    def test_build_then_search_then_offline_search(self):
        _real_embedder(self)  # skips if FAISS / sentence-transformers / the model are unavailable
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = str(Path(tmp) / "demo_index")
            build = self._run("build", "--docs", str(SAMPLE_DOCS_DIR), "--index", index_dir)
            self.assertEqual(build.returncode, 0, build.stderr)
            self.assertIn("Index created", build.stdout)

            again = self._run("--offline", "build", "--docs", str(SAMPLE_DOCS_DIR), "--index", index_dir)
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertIn("Index unchanged", again.stdout)

            claim = "The Alder River levee was built in 1990."
            found = self._run("search", claim, "--index", index_dir, "--top-k", "3", "--json")
            self.assertEqual(found.returncode, 0, found.stderr)
            result = json.loads(found.stdout)
            self.assertEqual(result["retrieval_status"], "evidence_retrieved")
            self.assertEqual(result["retrieved_evidence"][0]["source"], "brightwater_flood_report.txt")

            # the model must work from the local cache without any network access
            offline = self._run("--offline", "search", claim, "--index", index_dir, "--top-k", "3")
            self.assertEqual(offline.returncode, 0, offline.stderr)
            self.assertIn("brightwater_flood_report.txt", offline.stdout)
            self.assertIn("cosine similarity", offline.stdout)


if __name__ == "__main__":
    unittest.main()
