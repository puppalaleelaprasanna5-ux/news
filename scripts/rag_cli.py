"""
Command-line tool for the Phase 7 evidence index (build it, then search it).

Run from the project root:

    # build or update the index from data/rag/reference_docs/ (your own .txt / .pdf files)
    python scripts/rag_cli.py build

    # try it on the synthetic demo documents instead (fictional content, not real evidence)
    python scripts/rag_cli.py build --docs data/rag/sample_reference_docs --index data/rag/demo_index

    # search the index for a claim
    python scripts/rag_cli.py search "The Alder River levee was built in 1990." --top-k 3

Retrieval scores are cosine similarities, not probabilities, and a retrieved passage is not
proof that a claim is true or false.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.config import (
    DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_DOCS_DIR, DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_DIR, DEFAULT_MIN_SCORE, DEFAULT_TOP_K,
)
from src.rag.document_loader import DocumentLoadError
from src.rag.embeddings import EmbeddingError, SentenceTransformerEmbedder
from src.rag.faiss_index import EvidenceIndexError, build_or_update_index
from src.rag.retriever import EvidenceRetriever


def cmd_build(args) -> int:
    embedder = SentenceTransformerEmbedder(args.model, local_files_only=args.offline)
    try:
        result = build_or_update_index(
            args.docs, args.index, embedder, args.chunk_size, args.chunk_overlap, force=args.force
        )
    except (DocumentLoadError, EmbeddingError, EvidenceIndexError) as err:
        print(f"Build failed: {err}", file=sys.stderr)
        return 1
    print(f"Index {result.status}: {result.index.size} chunks from {len(result.load_report.documents)} documents "
          f"(new embeddings: {result.new_embeddings}, reused: {result.reused_embeddings}) -> {args.index}")
    for skipped in result.load_report.skipped:
        print(f"  skipped {skipped.source_id}: {skipped.reason}")
    for warning in result.warnings:
        print(f"  warning: {warning}")
    return 0


def cmd_search(args) -> int:
    embedder = SentenceTransformerEmbedder(args.model, local_files_only=args.offline)
    retriever = EvidenceRetriever.from_directory(args.index, embedder, top_k=args.top_k, min_score=args.min_score)
    result = retriever.retrieve(args.claim)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    print(f"Claim: {result['claim']}\nStatus: {result['retrieval_status']}")
    if result.get("error"):
        print(f"Error: {result['error']}")
    for item in result["retrieved_evidence"]:
        page = f", page {item['page']}" if item["page"] is not None else ""
        print(f"\n#{item['rank']}  {item['source']}{page}  (chunk {item['chunk_id']}, cosine similarity {item['score']:.3f})")
        print(f"   {item['text']}")
    print("\nNote: " + " ".join(result["limitations"][:2]))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7 evidence index tool")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="sentence-transformers model name")
    parser.add_argument("--offline", action="store_true", help="never contact the Hugging Face Hub (model must be cached)")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build or update the FAISS index")
    build.add_argument("--docs", default=str(DEFAULT_DOCS_DIR))
    build.add_argument("--index", default=str(DEFAULT_INDEX_DIR))
    build.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    build.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    build.add_argument("--force", action="store_true", help="re-embed everything")
    build.set_defaults(func=cmd_build)

    search = sub.add_parser("search", help="retrieve passages for a claim")
    search.add_argument("claim")
    search.add_argument("--index", default=str(DEFAULT_INDEX_DIR))
    search.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    search.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
