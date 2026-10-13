import pytest
from src.index.bm25_store import BM25Store, math_tokenize
from src.index.hybrid import format_citation, extract_first_author_surname, HybridRetriever


def test_math_tokenization():
    text = "The Mittag-Leffler kernel and Lyapunov-Krasovskii functional for phi-Caputo derivative."
    tokens = math_tokenize(text)
    assert "mittag-leffler" in tokens
    assert "lyapunov-krasovskii" in tokens
    assert "phi-caputo" in tokens


def test_bm25_exact_matching():
    chunks = [
        {"id": "c1", "text": "This paper investigates Mittag-Leffler stability of fractional networks.", "year": 2024},
        {"id": "c2", "text": "This work studies ordinary differential equations with boundary values.", "year": 2021},
        {"id": "c3", "text": "Neural networks and deep learning for computer vision classification.", "year": 2023},
    ]
    bm25 = BM25Store(persist_path="data/test_bm25.pkl")
    bm25.build_index(chunks)

    results = bm25.search("Mittag-Leffler")
    assert len(results) > 0
    assert results[0]["id"] == "c1"
    assert results[0]["bm25_score"] > 0


def test_citation_formatting():
    cite1 = format_citation(
        cite_key="ansari2024prabhakar",
        authors="Md Samshad Hussain Ansari, Muslim Malik, Dumitru Baleanu",
        year=2024,
        start_page=3,
        end_page=3,
    )
    assert cite1 == "[Ansari2024, p. 3]"

    cite2 = format_citation(
        cite_key="ansari2024competitive",
        authors="Ansari, Md Samshad Hussain and Malik, Muslim",
        year=2024,
        start_page=2,
        end_page=4,
    )
    assert cite2 == "[Ansari2024, pp. 2–4]"


def test_extract_first_author_surname():
    assert extract_first_author_surname("**Md Samshad Hussain Ansari**, Muslim Malik") == "Ansari"
    assert extract_first_author_surname("Ansari, Md Samshad Hussain and Malik, Muslim") == "Ansari"
