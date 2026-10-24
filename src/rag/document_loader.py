"""
Reference-document ingestion (Phase 7).

Reads plain-text (.txt) and PDF (.pdf) files from a local directory into
`ReferenceDocument` objects that keep the metadata needed later for citation:
filename, a stable source identifier (path relative to the reference directory),
page numbers (PDF only) and a content hash (used to detect changed documents).

Empty, unreadable, encrypted, oversized or unsupported files are NOT fatal: they are
listed in `LoadReport.skipped` with a reason. Nothing is ever invented - a `.txt` file
has no page numbers, so its pages carry `page=None`.

Page numbers for PDFs are the 1-based physical page position in the file, not any
printed page label inside the document.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".pdf")
MAX_FILE_BYTES = 50 * 1024 * 1024  # skip anything bigger than 50 MB


class DocumentLoadError(Exception):
    """Raised when the reference directory itself cannot be used."""


@dataclass
class PageText:
    page: Optional[int]  # 1-based for PDFs, None for plain text
    text: str


@dataclass
class ReferenceDocument:
    source_id: str  # path relative to the reference directory, POSIX separators
    filename: str
    pages: List[PageText]
    content_hash: str  # sha256 of the raw file bytes
    warnings: List[str] = field(default_factory=list)


@dataclass
class SkippedDocument:
    source_id: str
    reason: str


@dataclass
class LoadReport:
    documents: List[ReferenceDocument] = field(default_factory=list)
    skipped: List[SkippedDocument] = field(default_factory=list)


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def _decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        logger.warning("File is not valid UTF-8; decoding as latin-1.")
        return raw.decode("latin-1")


def _read_txt(raw: bytes) -> tuple:
    text = _normalize(_decode_text(raw))
    return ([PageText(None, text)] if text else []), []


def _read_pdf(path: Path) -> tuple:
    try:
        from pypdf import PdfReader
    except ImportError as err:
        raise RuntimeError("PDF support requires the 'pypdf' package (pip install pypdf).") from err

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception as err:  # pypdf raises several error types here
            raise RuntimeError(f"PDF is encrypted and could not be opened: {err}") from err
        if not ok:
            raise RuntimeError("PDF is password-protected.")

    pages, warnings = [], []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = _normalize(page.extract_text() or "")
        except Exception as err:
            warnings.append(f"Page {number}: text extraction failed ({type(err).__name__}).")
            continue
        if text:
            pages.append(PageText(number, text))
    return pages, warnings


def load_documents(directory, recursive: bool = True) -> LoadReport:
    """
    Load every supported document under `directory`.

    Raises DocumentLoadError only if the directory is missing or not a directory.
    Problem files are reported in `LoadReport.skipped` instead of raising.
    """
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise DocumentLoadError(f"Reference directory not found: {root}")
    root_resolved = root.resolve()

    report = LoadReport()
    candidates = sorted(root.rglob("*") if recursive else root.glob("*"))

    for path in candidates:
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            resolved = path.resolve()
            source_id = resolved.relative_to(root_resolved).as_posix()
        except ValueError:
            # A symlink pointing outside the reference directory.
            report.skipped.append(SkippedDocument(path.name, "symlink points outside the reference directory"))
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            report.skipped.append(SkippedDocument(source_id, f"unsupported file type '{suffix or 'none'}'"))
            continue

        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                report.skipped.append(SkippedDocument(source_id, f"file larger than {MAX_FILE_BYTES // (1024 * 1024)} MB"))
                continue
            raw = path.read_bytes()
            if suffix == ".txt":
                pages, warnings = _read_txt(raw)
            else:
                pages, warnings = _read_pdf(path)
        except Exception as err:  # unreadable / corrupted / missing optional dependency
            logger.warning("Skipping %s: %s", source_id, err)
            report.skipped.append(SkippedDocument(source_id, f"unreadable: {err}"))
            continue

        if not pages:
            report.skipped.append(SkippedDocument(source_id, "empty or no extractable text"))
            continue

        report.documents.append(
            ReferenceDocument(
                source_id=source_id,
                filename=path.name,
                pages=pages,
                content_hash=hashlib.sha256(raw).hexdigest(),
                warnings=warnings,
            )
        )
    return report
