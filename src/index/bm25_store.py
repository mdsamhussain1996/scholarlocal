from __future__ import annotations

import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def math_tokenize(text: str) -> List[str]:
    """Tokenize mathematical and scientific text preserving hyphenated terms and formulas."""
    # Normalize unicode hyphens/dashes to standard hyphen
    normalized = re.sub(r"[–—−]", "-", text.lower())
    # Keep hyphenated technical terms (e.g., mittag-leffler, lyapunov-krasovskii, phi-caputo)
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized)
    return tokens


class BM25Store:
    """In-memory and disk-persisted BM25 keyword search index for math papers."""

    def __init__(self, persist_path: str = "data/bm25_index.pkl"):
        self.persist_path = persist_path
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.load()

    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build BM25 index from chunk dictionaries with text and metadata."""
        self.corpus_chunks = chunks
        self.corpus_tokens = [math_tokenize(c["text"]) for c in chunks]
        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None
        self.save()

    def search(
        self,
        query: str,
        top_k: int = 15,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search chunks using BM25 with optional metadata filtering."""
        if not self.bm25 or not self.corpus_chunks:
            return []

        query_tokens = math_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Pair scores with chunk objects
        scored_chunks: List[Tuple[float, Dict[str, Any]]] = []
        for idx, (score, chunk) in enumerate(zip(scores, self.corpus_chunks)):
            if score <= 0.0 and idx < len(self.corpus_tokens):
                if any(t in self.corpus_tokens[idx] for t in query_tokens):
                    score = 0.01

            if score <= 0.0:
                continue

            # Apply filters if provided
            if filters:
                match = True
                if "paper_id" in filters and chunk.get("paper_id") != filters["paper_id"]:
                    match = False
                if "cite_key" in filters and chunk.get("cite_key") != filters["cite_key"]:
                    match = False
                if "year_min" in filters and chunk.get("year") and chunk["year"] < filters["year_min"]:
                    match = False
                if "year_max" in filters and chunk.get("year") and chunk["year"] > filters["year_max"]:
                    match = False
                if not match:
                    continue

            scored_chunks.append((float(score), chunk))

        # Sort descending by BM25 score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (score, chunk) in enumerate(scored_chunks[:top_k], start=1):
            item = dict(chunk)
            item["bm25_score"] = round(score, 4)
            item["bm25_rank"] = rank
            results.append(item)

        return results

    def save(self) -> None:
        """Persist BM25 index and corpus chunks to disk."""
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        try:
            with open(self.persist_path, "wb") as f:
                pickle.dump({"chunks": self.corpus_chunks, "tokens": self.corpus_tokens}, f)
        except Exception as e:
            logger.error("Failed to save BM25 index: %s", e)

    def load(self) -> bool:
        """Load BM25 index from disk if present."""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, "rb") as f:
                    data = pickle.load(f)
                    self.corpus_chunks = data.get("chunks", [])
                    self.corpus_tokens = data.get("tokens", [])
                    if self.corpus_tokens:
                        self.bm25 = BM25Okapi(self.corpus_tokens)
                return True
            except Exception as e:
                logger.warning("Failed to load BM25 index: %s", e)
        return False
