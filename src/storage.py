from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    """SQLite persistence layer for ScholarLocal."""

    def __init__(self, db_path: str = "data/scholar.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Papers table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    file_path TEXT UNIQUE,
                    file_hash TEXT,
                    title TEXT,
                    authors TEXT,
                    year INTEGER,
                    doi TEXT,
                    journal TEXT,
                    cite_key TEXT,
                    abstract TEXT,
                    page_count INTEGER,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
                """
            )

            # Chunks table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    start_page INTEGER,
                    end_page INTEGER,
                    section_title TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    token_count INTEGER,
                    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
                )
                """
            )

            # Theorems and Definitions structured records
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS theorems (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    type TEXT,
                    number TEXT,
                    page INTEGER,
                    full_text TEXT,
                    summary_json TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(id) ON DELETE CASCADE
                )
                """
            )

            # Chat history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    citations_json TEXT,
                    created_at TIMESTAMP
                )
                """
            )

            conn.commit()

    def save_paper(self, paper_dict: Dict[str, Any]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO papers (
                    id, file_path, file_hash, title, authors, year, doi, journal,
                    cite_key, abstract, page_count, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_dict["id"],
                    paper_dict["file_path"],
                    paper_dict["file_hash"],
                    paper_dict.get("title", ""),
                    paper_dict.get("authors", ""),
                    paper_dict.get("year", None),
                    paper_dict.get("doi", ""),
                    paper_dict.get("journal", ""),
                    paper_dict.get("cite_key", ""),
                    paper_dict.get("abstract", ""),
                    paper_dict.get("page_count", 0),
                    paper_dict.get("created_at", utc_now()),
                    utc_now(),
                ),
            )
            conn.commit()

    def get_paper_by_hash(self, file_hash: str) -> Optional[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_paper(self, paper_id: str) -> Optional[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_papers(self) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM papers ORDER BY year DESC, title ASC")
            return [dict(r) for r in cursor.fetchall()]

    def delete_paper(self, paper_id: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
            cursor.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            cursor.execute("DELETE FROM theorems WHERE paper_id = ?", (paper_id,))
            conn.commit()

    def save_chunks(self, paper_id: str, chunks: List[Dict[str, Any]]) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Delete old chunks for this paper
            cursor.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
            for c in chunks:
                cursor.execute(
                    """
                    INSERT INTO chunks (
                        id, paper_id, start_page, end_page, section_title,
                        chunk_index, text, token_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        c["id"],
                        paper_id,
                        c["start_page"],
                        c["end_page"],
                        c.get("section_title", ""),
                        c.get("chunk_index", 0),
                        c["text"],
                        c.get("token_count", len(c["text"].split())),
                    ),
                )
            conn.commit()

    def get_all_chunks(self) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks")
            return [dict(r) for r in cursor.fetchall()]

    def get_chunks_by_paper(self, paper_id: str) -> List[dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE paper_id = ? ORDER BY chunk_index", (paper_id,))
            return [dict(r) for r in cursor.fetchall()]
