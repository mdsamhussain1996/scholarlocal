import os
import pytest
from src.ingest.chunker import chunk_document
from src.ingest.metadata import match_bibtex, parse_bib_files
from src.ingest.theorems import extract_theorems_from_pages, generate_theorem_summary
from src.llm import OllamaClient


def test_bibtex_matching():
    entries = parse_bib_files(["library/references.bib"])
    assert len(entries) >= 5

    # Test exact DOI match
    matched = match_bibtex(
        title="Some Random Title",
        doi="10.1007/s12346-023-00919-4",
        bib_entries=entries,
    )
    assert matched is not None
    assert matched["cite_key"] == "ansari2024prabhakar"

    # Test fuzzy title match
    matched_fuzzy = match_bibtex(
        title="Controllability of Prabhakar Fractional Dynamical Systems",
        doi="",
        bib_entries=entries,
    )
    assert matched_fuzzy is not None
    assert matched_fuzzy["cite_key"] == "ansari2024prabhakar"


def test_chunking_preserves_theorems():
    pages = [
        (
            1,
            "1. Introduction\n"
            "Here is the introduction to fractional systems.\n\n"
            "Theorem 3.1 (Stability Criterion). If the matrix A is Hurwitz, then the fractional "
            "dynamical system with Caputo derivative is asymptotically stable.\n\n"
            "2. Numerical Results\n"
            "Here are the numerical results.",
        )
    ]
    chunks = chunk_document(paper_id="test_paper", pages=pages, target_tokens=100)
    assert len(chunks) >= 1

    # Verify that Theorem 3.1 appears completely within a chunk
    found_theorem = any("Theorem 3.1 (Stability Criterion)" in c["text"] and "asymptotically stable" in c["text"] for c in chunks)
    assert found_theorem, "Theorem was split or lost across chunk boundaries"


def test_theorem_structured_summary():
    statement = (
        "Theorem 3.2 (Finite-Time Synchronization via Sliding Mode). "
        "Consider a fractional-order quaternion-valued neural network with Caputo derivative of order alpha in (0, 1). "
        "Under sliding mode control with parameter bounds, the error dynamics achieves finite-time synchronization."
    )
    summary = generate_theorem_summary(statement, client=None)
    assert summary["fractional_derivative"] == "Caputo"
    assert summary["system_type"] == "quaternion-valued neural network"
    assert summary["control_type"] == "sliding mode"
    assert summary["result_type"] == "finite-time"


def test_ollama_client_embed():
    client = OllamaClient()
    embs = client.embed(["Fractional calculus and controllability"])
    assert len(embs) == 1
    assert len(embs[0]) == 768
