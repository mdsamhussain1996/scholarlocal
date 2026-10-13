from __future__ import annotations

import logging
import os
from typing import Optional

import pymupdf

logger = logging.getLogger(__name__)


def render_pdf_page_as_png(pdf_path: str, page_number: int = 1, dpi: int = 150) -> Optional[bytes]:
    """Render a specific 1-indexed page of a PDF as PNG image bytes."""
    if not os.path.exists(pdf_path):
        return None
    try:
        doc = pymupdf.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            page_number = max(1, min(page_number, len(doc)))
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception as e:
        logger.error("Error rendering PDF page %d of %s: %s", page_number, pdf_path, e)
        return None


def get_pdf_page_count(pdf_path: str) -> int:
    """Return total page count of a PDF."""
    if not os.path.exists(pdf_path):
        return 0
    try:
        doc = pymupdf.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0
