"""
Text chunking (Phase 7): split reference documents into overlapping passages.

Strategy (sizes are in characters, both configurable):
  1. Split the text into sentence-like units (paragraph breaks and sentence-ending
     punctuation; common abbreviations such as "U.S." and "Mr." do not end a unit).
     A unit longer than `chunk_size` is split on word boundaries.
  2. Pack whole units into a chunk until adding the next one would exceed `chunk_size`.
  3. Start the next chunk with the tail of the previous one (up to `overlap` characters):
     whole trailing sentences if they fit, otherwise a word-boundary fragment. This keeps
     context across boundaries, at the cost of a chunk sometimes starting mid-sentence.

Chunks never cross page boundaries, so every chunk keeps an accurate page number. Empty
chunks are never produced and every chunk keeps its source metadata.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional

from src.rag.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.rag.document_loader import ReferenceDocument

_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])"
    r"(?<![A-Z]\.)(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!St\.)(?<!Jr\.)(?<!Sr\.)"
    r"\s+(?=[\"'(\[A-Z0-9])"
)


@dataclass
class Chunk:
    chunk_id: int  # position in the index; -1 until assigned by chunk_documents()
    doc_chunk_index: int  # running number of this chunk within its source document
    text: str
    source_id: str
    filename: str
    page: Optional[int]
    text_hash: str  # sha256 of `text`, used to reuse embeddings of unchanged chunks


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_long_unit(unit: str, chunk_size: int) -> List[str]:
    pieces, current = [], ""
    for word in unit.split():
        while len(word) > chunk_size:  # an unbroken token (e.g. a long URL)
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:chunk_size])
            word = word[chunk_size:]
        candidate = f"{current} {word}".strip()
        if len(candidate) > chunk_size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, chunk_size: int) -> List[str]:
    units: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(_split_long_unit(sentence, chunk_size))
    return units


def _overlap_tail(units: List[str], overlap: int) -> List[str]:
    """Trailing part of the just-emitted chunk to repeat at the start of the next one."""
    if overlap <= 0 or not units:
        return []
    tail: List[str] = []
    length = 0
    for unit in reversed(units):
        new_length = length + len(unit) + (1 if tail else 0)
        if new_length > overlap:
            break
        tail.insert(0, unit)
        length = new_length
    if tail:
        return tail
    # No whole sentence fits: fall back to a word-boundary fragment of the last unit.
    words = units[-1].split()
    fragment: List[str] = []
    length = 0
    for word in reversed(words):
        new_length = length + len(word) + (1 if fragment else 0)
        if new_length > overlap:
            break
        fragment.insert(0, word)
        length = new_length
    return [" ".join(fragment)] if fragment else []


def split_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split `text` into overlapping chunk strings. Returns [] for empty/whitespace text."""
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be an integer with 0 <= overlap < chunk_size.")
    if not isinstance(text, str) or not text.strip():
        return []

    chunks: List[str] = []
    current: List[str] = []
    for unit in _units(text, chunk_size):
        if current and len(" ".join(current + [unit])) > chunk_size:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)
            while current and len(" ".join(current + [unit])) > chunk_size:
                current.pop(0)  # make room for the new unit
        current.append(unit)
    if current:
        last = " ".join(current)
        if not chunks or last != chunks[-1]:
            chunks.append(last)
    return [c for c in chunks if c.strip()]


def chunk_documents(
    documents: List[ReferenceDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """Chunk every page of every document; `chunk_id` is assigned sequentially from 0."""
    chunks: List[Chunk] = []
    for doc in documents:
        doc_index = 0
        for page in doc.pages:
            for piece in split_text(page.text, chunk_size, overlap):
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        doc_chunk_index=doc_index,
                        text=piece,
                        source_id=doc.source_id,
                        filename=doc.filename,
                        page=page.page,
                        text_hash=text_hash(piece),
                    )
                )
                doc_index += 1
    return chunks
