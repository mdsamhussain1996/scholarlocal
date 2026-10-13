from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.config import ScholarConfig, load_config
from src.index.bm25_store import BM25Store
from src.index.vector_store import VectorStore
from src.storage import Database

logger = logging.getLogger(__name__)


def extract_first_author_surname(authors_str: str) -> str:
    """Extract primary author's surname for standard academic citations (e.g., Ansari2024)."""
    if not authors_str:
        return "Unknown"
    # Clean markdown formatting
    cleaned = authors_str.replace("**", "").replace("*", "").strip()
    first_author = cleaned.split(",")[0].strip()
    first_author = first_author.split(" and ")[0].strip()
    parts = first_author.split()
    return parts[-1].capitalize() if parts else "Author"


def format_citation(
    cite_key: str,
    authors: str,
    year: Optional[int],
    start_page: int,
    end_page: int,
) -> str:
    """Generate exact page-accurate academic citation: [AuthorYear, p. 7] or [AuthorYear, pp. 7–8]."""
    surname = extract_first_author_surname(authors)
    year_str = str(year) if year else ""
    author_year = f"{surname}{year_str}" if (surname and year_str) else (cite_key or "Anon")

    if start_page == end_page:
        page_str = f"p. {start_page}"
    else:
        page_str = f"pp. {start_page}–{end_page}"

    return f"[{author_year}, {page_str}]"


class HybridRetriever:
    """Hybrid search combining dense vector embeddings and BM25 exact matching via Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        db: Database,
        config: Optional[ScholarConfig] = None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.db = db
        self.config = config or load_config()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "hybrid",  # "hybrid", "vector", "bm25"
    ) -> List[Dict[str, Any]]:
        """Retrieve and rank documents using requested mode."""
        k_final = top_k or self.config.retrieval.top_k_final
        k_vec = self.config.retrieval.top_k_vector
        k_bm25 = self.config.retrieval.top_k_bm25
        rrf_constant = self.config.retrieval.hybrid_rrf_k

        # 1. Vector Search
        vector_results: List[Dict[str, Any]] = []
        if mode in ("hybrid", "vector"):
            # Build vector filters if applicable
            vec_filter = None
            if filters and "paper_id" in filters:
                vec_filter = {"paper_id": filters["paper_id"]}
            vector_results = self.vector_store.similarity_search(query, top_k=k_vec, filters=vec_filter)

        # 2. BM25 Search
        bm25_results: List[Dict[str, Any]] = []
        if mode in ("hybrid", "bm25"):
            bm25_results = self.bm25_store.search(query, top_k=k_bm25, filters=filters)

        # Shortcut for single mode
        if mode == "vector":
            return self._enrich_and_finalize(vector_results[:k_final])
        if mode == "bm25":
            return self._enrich_and_finalize(bm25_results[:k_final])

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (k + rank_vector) + 1 / (k + rank_bm25)
        combined: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vector_results, start=1):
            chunk_id = item["id"]
            if chunk_id not in combined:
                combined[chunk_id] = {
                    "item": item,
                    "vector_rank": rank,
                    "vector_score": item.get("score", 0.0),
                    "bm25_rank": None,
                    "bm25_score": 0.0,
                    "rrf_score": 0.0,
                }
            else:
                combined[chunk_id]["vector_rank"] = rank
                combined[chunk_id]["vector_score"] = item.get("score", 0.0)

            combined[chunk_id]["rrf_score"] += 1.0 / (rrf_constant + rank)

        for rank, item in enumerate(bm25_results, start=1):
            chunk_id = item["id"]
            if chunk_id not in combined:
                combined[chunk_id] = {
                    "item": item,
                    "vector_rank": None,
                    "vector_score": 0.0,
                    "bm25_rank": rank,
                    "bm25_score": item.get("bm25_score", 0.0),
                    "rrf_score": 0.0,
                }
            else:
                combined[chunk_id]["bm25_rank"] = rank
                combined[chunk_id]["bm25_score"] = item.get("bm25_score", 0.0)

            combined[chunk_id]["rrf_score"] += 1.0 / (rrf_constant + rank)

        # Sort by RRF score descending
        ranked_list = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)

        final_items: List[Dict[str, Any]] = []
        for entry in ranked_list[:k_final]:
            doc = entry["item"]
            doc["rrf_score"] = round(entry["rrf_score"], 6)
            doc["vector_rank"] = entry["vector_rank"]
            doc["bm25_rank"] = entry["bm25_rank"]
            doc["vector_score"] = entry["vector_score"]
            doc["bm25_score"] = entry["bm25_score"]
            final_items.append(doc)

        return self._enrich_and_finalize(final_items)

    def _enrich_and_finalize(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add paper metadata, exact citation string, and validation tags to retrieved chunks."""
        enriched: List[Dict[str, Any]] = []
        papers_cache: Dict[str, dict] = {}

        for item in items:
            paper_id = item.get("paper_id", "")
            if paper_id and paper_id not in papers_cache:
                paper = self.db.get_paper(paper_id)
                if paper:
                    papers_cache[paper_id] = paper

            paper = papers_cache.get(paper_id, {})
            title = paper.get("title") or item.get("title") or paper_id
            authors = paper.get("authors") or ""
            year = paper.get("year")
            cite_key = paper.get("cite_key") or item.get("cite_key") or ""
            doi = paper.get("doi") or ""
            start_p = item.get("start_page", 1)
            end_p = item.get("end_page", 1)

            citation = format_citation(
                cite_key=cite_key,
                authors=authors,
                year=year,
                start_page=start_p,
                end_page=end_p,
            )

            res = dict(item)
            res["title"] = title
            res["authors"] = authors
            res["year"] = year
            res["cite_key"] = cite_key
            res["doi"] = doi
            res["citation"] = citation
            enriched.append(res)

        return enriched
