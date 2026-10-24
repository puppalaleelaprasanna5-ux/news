"""
Phase 7: FAISS-based evidence retrieval (RAG retrieval half).

Importing this package is cheap: FAISS and sentence-transformers are only imported when an
index is built/loaded or an embedding model is first used.
"""

from src.rag.document_loader import DocumentLoadError, LoadReport, ReferenceDocument, load_documents
from src.rag.embeddings import EmbeddingError, HashingEmbedder, SentenceTransformerEmbedder
from src.rag.evidence_pipeline import gather_evidence_for_article
from src.rag.faiss_index import (
    EvidenceIndex,
    EvidenceIndexError,
    FaissNotInstalledError,
    IndexCorruptedError,
    IndexMismatchError,
    IndexNotFoundError,
    build_or_update_index,
)
from src.rag.retriever import EvidenceRetriever, retrieve_evidence
from src.rag.text_chunker import Chunk, chunk_documents, split_text

__all__ = [
    "Chunk", "DocumentLoadError", "EmbeddingError", "EvidenceIndex", "EvidenceIndexError", "EvidenceRetriever",
    "FaissNotInstalledError", "HashingEmbedder", "IndexCorruptedError", "IndexMismatchError", "IndexNotFoundError", "LoadReport",
    "ReferenceDocument", "SentenceTransformerEmbedder", "build_or_update_index", "chunk_documents",
    "gather_evidence_for_article", "load_documents", "retrieve_evidence", "split_text",
]
