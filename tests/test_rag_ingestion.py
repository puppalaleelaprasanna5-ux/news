"""
Phase 7 tests: document loading and text chunking (no FAISS, no model downloads needed).

    python -m pytest tests/test_rag_ingestion.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.document_loader import DocumentLoadError, load_documents
from src.rag.text_chunker import chunk_documents, split_text, text_hash


def make_pdf(pages):
    """Build a minimal valid text PDF (Helvetica) with one string per page, without extra libraries."""
    objects = []
    n_pages = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")
    for i, text in enumerate(pages):
        page_obj, content_obj = 3 + 2 * i, 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_obj} 0 R "
            f"/Resources << /Font << /F1 {3 + 2 * n_pages} 0 R >> >> >>"
        )
        stream = f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


# --------------------------------------------------------------------- chunking
def test_chunks_respect_size_and_are_never_empty():
    text = " ".join(f"Sentence number {i} talks about a topic." for i in range(40))
    chunks = split_text(text, chunk_size=120, overlap=30)
    assert len(chunks) > 3
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 120 for c in chunks)


def test_chunk_overlap_repeats_text_between_neighbours():
    text = " ".join(f"Item {i} is described here." for i in range(30))
    chunks = split_text(text, chunk_size=100, overlap=40)
    assert len(chunks) > 2
    for prev, nxt in zip(chunks, chunks[1:]):
        # the start of the next chunk must already appear at the end of the previous one
        first_words = nxt.split()[:2]
        assert " ".join(first_words) in prev


def test_zero_overlap_gives_disjoint_chunks():
    text = " ".join(f"Item {i} is described here." for i in range(20))
    chunks = split_text(text, chunk_size=100, overlap=0)
    assert " ".join(chunks) == text


def test_every_sentence_is_covered_by_some_chunk():
    sentences = [f"Fact {i} is about subject {i}." for i in range(25)]
    chunks = split_text(" ".join(sentences), chunk_size=90, overlap=20)
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_abbreviations_do_not_split_sentences():
    chunks = split_text("The U.S. Senate met. Dr. Smith spoke.", chunk_size=1000, overlap=0)
    assert chunks == ["The U.S. Senate met. Dr. Smith spoke."]


def test_very_long_sentence_and_long_token_are_split():
    long_sentence = "word " * 200
    chunks = split_text(long_sentence, chunk_size=100, overlap=20)
    assert all(0 < len(c) <= 100 for c in chunks)
    url = "x" * 350
    assert all(len(c) <= 100 for c in split_text(url, chunk_size=100, overlap=10))


def test_chunker_handles_empty_and_invalid_input():
    assert split_text("") == []
    assert split_text("   \n\n  ") == []
    assert split_text(None) == []
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=-1)


def test_short_text_gives_single_chunk():
    assert split_text("Just one short sentence.", chunk_size=500, overlap=100) == ["Just one short sentence."]


# ---------------------------------------------------------------------- loading
def test_load_txt_documents_with_metadata(tmp_path):
    (tmp_path / "a.txt").write_text("Alpha document text.", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("Bravo document text.", encoding="utf-8")
    report = load_documents(tmp_path)
    ids = {d.source_id for d in report.documents}
    assert ids == {"a.txt", "sub/b.txt"}
    doc = next(d for d in report.documents if d.source_id == "sub/b.txt")
    assert doc.filename == "b.txt"
    assert doc.pages[0].page is None  # plain text has no page numbers
    assert len(doc.content_hash) == 64


def test_empty_unsupported_and_dotfiles_are_handled(tmp_path):
    (tmp_path / "empty.txt").write_text("   \n  ", encoding="utf-8")
    (tmp_path / "notes.docx").write_bytes(b"binary")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / "good.txt").write_text("Real content.", encoding="utf-8")
    report = load_documents(tmp_path)
    assert [d.source_id for d in report.documents] == ["good.txt"]
    reasons = {s.source_id: s.reason for s in report.skipped}
    assert "empty" in reasons["empty.txt"]
    assert "unsupported" in reasons["notes.docx"]
    assert ".gitkeep" not in reasons


def test_non_utf8_text_file_is_still_loaded(tmp_path):
    (tmp_path / "latin.txt").write_bytes("caf\xe9 society".encode("latin-1"))
    report = load_documents(tmp_path)
    assert "caf" in report.documents[0].pages[0].text


def test_missing_directory_raises_clear_error(tmp_path):
    with pytest.raises(DocumentLoadError):
        load_documents(tmp_path / "does_not_exist")


def test_pdf_pages_and_page_numbers(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "report.pdf").write_bytes(make_pdf(["First page about rivers", "Second page about mountains"]))
    report = load_documents(tmp_path)
    assert len(report.documents) == 1
    pages = report.documents[0].pages
    assert [p.page for p in pages] == [1, 2]
    assert "rivers" in pages[0].text and "mountains" in pages[1].text


def test_corrupted_pdf_is_skipped_not_fatal(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "broken.pdf").write_bytes(b"this is not a pdf at all")
    (tmp_path / "ok.txt").write_text("Fine.", encoding="utf-8")
    report = load_documents(tmp_path)
    assert [d.source_id for d in report.documents] == ["ok.txt"]
    assert any(s.source_id == "broken.pdf" and "unreadable" in s.reason for s in report.skipped)


def test_pdf_without_text_is_skipped(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(tmp_path / "blank.pdf", "wb") as handle:
        writer.write(handle)
    report = load_documents(tmp_path)
    assert report.documents == []
    assert "no extractable text" in report.skipped[0].reason


# ------------------------------------------------- multi-document chunk metadata
def test_multiple_documents_keep_metadata_and_get_sequential_ids(tmp_path):
    (tmp_path / "one.txt").write_text(" ".join(f"Alpha fact {i} stated clearly." for i in range(20)), encoding="utf-8")
    (tmp_path / "two.txt").write_text(" ".join(f"Bravo fact {i} stated clearly." for i in range(20)), encoding="utf-8")
    docs = load_documents(tmp_path).documents
    chunks = chunk_documents(docs, chunk_size=120, overlap=20)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert {c.filename for c in chunks} == {"one.txt", "two.txt"}
    for c in chunks:
        assert c.page is None
        assert c.text_hash == text_hash(c.text)
        assert ("Alpha" in c.text) == (c.filename == "one.txt")
    per_doc = [c.doc_chunk_index for c in chunks if c.filename == "two.txt"]
    assert per_doc == list(range(len(per_doc)))


def test_pdf_chunks_never_cross_pages(tmp_path):
    pytest.importorskip("pypdf")
    (tmp_path / "r.pdf").write_bytes(make_pdf(["Page one text about alpha", "Page two text about bravo"]))
    chunks = chunk_documents(load_documents(tmp_path).documents, chunk_size=500, overlap=50)
    assert [(c.page, "alpha" in c.text, "bravo" in c.text) for c in chunks] == [(1, True, False), (2, False, True)]


def test_empty_document_list_gives_no_chunks():
    assert chunk_documents([], 100, 10) == []
