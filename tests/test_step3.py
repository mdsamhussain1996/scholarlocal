import os
import pytest
from src.ingest.viewer_utils import render_pdf_page_as_png, get_pdf_page_count
from src.agent.qa import ScholarQA
from src.config import load_config
from src.storage import Database
from src.index.vector_store import VectorStore
from src.index.bm25_store import BM25Store
from src.index.hybrid import HybridRetriever


def test_pdf_rendering():
    pdf_path = "library/ansari2024prabhakar.pdf"
    assert os.path.exists(pdf_path)

    # Check page count
    total_pages = get_pdf_page_count(pdf_path)
    assert total_pages == 3

    # Render page 1
    png_bytes = render_pdf_page_as_png(pdf_path, page_number=1, dpi=100)
    assert png_bytes is not None
    assert len(png_bytes) > 1000
    # PNG signature check: first 8 bytes must match standard PNG magic header
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    # Render page 3
    png_page3 = render_pdf_page_as_png(pdf_path, page_number=3, dpi=100)
    assert png_page3 is not None
    assert len(png_page3) > 1000


def test_qa_retrieval_and_citation_formatting():
    cfg = load_config("config.yaml")
    db = Database(os.path.join(cfg.data_dir, "scholar.db"))
    vstore = VectorStore(persist_dir=os.path.join(cfg.data_dir, "chroma"))
    bm25 = BM25Store(persist_path=os.path.join(cfg.data_dir, "bm25_index.pkl"))
    retriever = HybridRetriever(vector_store=vstore, bm25_store=bm25, db=db, config=cfg)
    qa = ScholarQA(retriever=retriever, client=None, config=cfg)

    # Search passages
    passages = retriever.search("Prabhakar Mittag-Leffler controllability", top_k=2)
    assert len(passages) > 0
    assert "citation" in passages[0]
    assert "Ansari" in passages[0]["citation"]
