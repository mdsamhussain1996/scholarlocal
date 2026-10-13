from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from src.llm import OllamaClient

logger = logging.getLogger(__name__)


class VectorStore:
    """Persistent vector store powered by ChromaDB and local Ollama embeddings."""

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "scholarlocal_chunks",
        ollama_client: Optional[OllamaClient] = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.llm = ollama_client or OllamaClient()

        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], paper_metadata: Dict[str, Any]) -> None:
        """Embed and add chunks with bibliographic metadata into the vector index."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.llm.embed(texts)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in chunks:
            ids.append(c["id"])
            documents.append(c["text"])
            metadatas.append({
                "paper_id": c["paper_id"],
                "cite_key": paper_metadata.get("cite_key", ""),
                "title": paper_metadata.get("title", ""),
                "start_page": int(c["start_page"]),
                "end_page": int(c["end_page"]),
                "section_title": c.get("section_title", "General"),
                "chunk_index": int(c.get("chunk_index", 0)),
            })

        # Add or update in ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def delete_paper_chunks(self, paper_id: str) -> None:
        """Remove all indexed chunks belonging to a paper."""
        try:
            self.collection.delete(where={"paper_id": paper_id})
        except Exception as e:
            logger.warning("Error deleting chunks for paper %s: %s", paper_id, e)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector similarity search."""
        if self.collection.count() == 0:
            return []

        query_emb = self.llm.embed([query])[0]
        where_clause = filters if filters else None

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, self.collection.count()),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        matches: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return matches

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids)):
            dist = distances[i] if i < len(distances) else 1.0
            # Cosine distance: similarity score = 1 - distance
            score = max(0.0, min(1.0, 1.0 - dist))
            meta = metas[i] if i < len(metas) else {}
            matches.append({
                "id": ids[i],
                "score": round(score, 4),
                "text": docs[i] if i < len(docs) else "",
                "metadata": meta,
                "paper_id": meta.get("paper_id", ""),
                "cite_key": meta.get("cite_key", ""),
                "title": meta.get("title", ""),
                "start_page": meta.get("start_page", 1),
                "end_page": meta.get("end_page", 1),
                "section_title": meta.get("section_title", ""),
            })

        return matches

    def count(self) -> int:
        return self.collection.count()
