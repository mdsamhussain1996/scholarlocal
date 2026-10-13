from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import bibtexparser
from rapidfuzz import fuzz

from src.llm import OllamaClient, extract_json_block

logger = logging.getLogger(__name__)


def normalize_doi(doi: str) -> str:
    """Normalize DOI by stripping URLs and lowercase."""
    if not doi:
        return ""
    doi = doi.strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    return doi


def parse_bib_files(bib_paths: List[str]) -> List[Dict[str, Any]]:
    """Parse one or more .bib files into normalized dictionaries."""
    entries: List[Dict[str, Any]] = []
    for path in bib_paths:
        if not os.path.exists(path):
            continue
        try:
            library = bibtexparser.parse_file(path)
            for entry in library.entries:
                fields = {k.lower(): v.value for k, v in entry.fields_dict.items()}
                entries.append({
                    "cite_key": entry.key,
                    "title": fields.get("title", "").replace("{", "").replace("}", ""),
                    "author": fields.get("author", ""),
                    "year": int(fields.get("year", 0)) if fields.get("year", "").isdigit() else None,
                    "doi": normalize_doi(fields.get("doi", "")),
                    "journal": fields.get("journal", ""),
                })
        except Exception as e:
            logger.error("Error parsing bib file %s: %s", path, e)
    return entries


def match_bibtex(
    title: str,
    doi: str,
    bib_entries: List[Dict[str, Any]],
    threshold: float = 80.0,
) -> Optional[Dict[str, Any]]:
    """Match paper against bib entries by DOI first, then fuzzy title."""
    norm_doi = normalize_doi(doi)

    # 1. Exact DOI match
    if norm_doi:
        for entry in bib_entries:
            if entry["doi"] and entry["doi"] == norm_doi:
                return entry

    # 2. Fuzzy title match
    clean_title = re.sub(r"[^\w\s]", "", title.lower()).strip()
    if not clean_title:
        return None

    best_match = None
    best_score = 0.0

    for entry in bib_entries:
        cand_title = re.sub(r"[^\w\s]", "", entry["title"].lower()).strip()
        score = fuzz.token_set_ratio(clean_title, cand_title)
        if score > best_score:
            best_score = score
            best_match = entry

    if best_score >= threshold and best_match:
        return best_match

    return None


def extract_metadata_from_text(
    first_page_text: str,
    client: Optional[OllamaClient] = None,
    prompt_template_path: str = "prompts/metadata_extraction.txt",
) -> Dict[str, Any]:
    """Extract metadata (title, authors, year, doi, abstract) using heuristic regex or local LLM."""
    metadata: Dict[str, Any] = {
        "title": "",
        "authors": "",
        "year": None,
        "doi": "",
        "journal": "",
        "abstract": "",
    }

    # Heuristic regex extraction
    lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
    if lines:
        # Title is often first non-empty line
        metadata["title"] = lines[0]

    # DOI search
    doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", first_page_text)
    if doi_match:
        metadata["doi"] = normalize_doi(doi_match.group(0))

    # Year search
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", first_page_text)
    if year_match:
        metadata["year"] = int(year_match.group(0))

    # Abstract search
    abstract_match = re.search(
        r"Abstract[\s:–—.-]+(.*?)(?=\n\s*(?:1\.?|Introduction|Keywords|$))",
        first_page_text,
        re.DOTALL | re.IGNORECASE,
    )
    if abstract_match:
        metadata["abstract"] = " ".join(abstract_match.group(1).split())

    # Fall back to LLM if title is very short or missing and client is online
    if (len(metadata["title"]) < 10 or not metadata["doi"]) and client and client.is_online():
        try:
            if os.path.exists(prompt_template_path):
                with open(prompt_template_path, "r", encoding="utf-8") as f:
                    tpl = f.read()
                prompt = tpl.replace("{text}", first_page_text[:3000])
                resp = client.chat([{"role": "user", "content": prompt}], json_mode=True)
                json_data = extract_json_block(resp)
                if json_data:
                    for k in ("title", "authors", "year", "doi", "journal", "abstract"):
                        if json_data.get(k) and not metadata.get(k):
                            metadata[k] = json_data[k]
        except Exception as e:
            logger.warning("LLM metadata extraction failed, falling back to heuristics: %s", e)

    return metadata
