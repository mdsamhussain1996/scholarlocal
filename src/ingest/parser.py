from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Tuple

import pymupdf

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for change tracking."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


class PDFDocument:
    """Parsed representation of a PDF document."""

    def __init__(self, file_path: str, pages: List[Tuple[int, str]], metadata: Dict[str, Any]):
        self.file_path = file_path
        self.pages = pages  # (1-indexed page_num, text)
        self.metadata = metadata
        self.file_hash = compute_file_hash(file_path)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"[Page {p}]\n{txt}" for p, txt in self.pages)


def parse_pdf(file_path: str) -> PDFDocument:
    """Parse a PDF document using PyMuPDF page by page, preserving page numbers."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = pymupdf.open(file_path)
    pages: List[Tuple[int, str]] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text").strip()
        pages.append((page_idx + 1, text))

    meta = doc.metadata or {}
    doc.close()

    return PDFDocument(file_path=file_path, pages=pages, metadata=meta)
