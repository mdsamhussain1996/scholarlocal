from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.config import ScholarConfig, load_config
from src.index.hybrid import HybridRetriever
from src.llm import OllamaClient

logger = logging.getLogger(__name__)


class ScholarQA:
    """Rigorous grounded Q&A engine with citation extraction and validation."""

    def __init__(
        self,
        retriever: HybridRetriever,
        client: Optional[OllamaClient] = None,
        config: Optional[ScholarConfig] = None,
        prompt_path: str = "prompts/ask_qa.txt",
    ):
        self.retriever = retriever
        self.config = config or load_config()
        self.client = client or OllamaClient(
            host=self.config.models.ollama_host,
            chat_model=self.config.models.chat_model,
            embedding_model=self.config.models.embedding_model,
        )
        self.prompt_path = prompt_path

    def ask(
        self,
        question: str,
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retrieve passages and generate answer with exact citations."""
        # 1. Retrieve passages
        passages = self.retriever.search(query=question, top_k=top_k, filters=filters, mode="hybrid")

        if not passages:
            return {
                "answer": "No relevant passages were found in your library for this query. Have you indexed your PDFs?",
                "passages": [],
                "citations": [],
            }

        # 2. Format context
        context_blocks = []
        for idx, p in enumerate(passages, start=1):
            citation = p.get("citation", f"[Ref{idx}]")
            title = p.get("title", "")
            sec = p.get("section_title", "")
            text = p.get("text", "")
            block = f"Passage {idx} {citation} — \"{title}\" (Section: {sec}):\n{text}"
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        # 3. Read prompt template
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                prompt_tpl = f.read()
        else:
            prompt_tpl = (
                "Answer using only the retrieved passages. Use exact citations [AuthorYear, p. X].\n"
                "Passages:\n{context}\n\nQuestion:\n{question}"
            )

        prompt = prompt_tpl.replace("{context}", context_str).replace("{question}", question)

        # 4. Generate answer via Ollama
        messages = [
            {"role": "system", "content": "You are a precise mathematical assistant adhering strictly to grounded citations."},
            {"role": "user", "content": prompt},
        ]

        try:
            answer = self.client.chat(messages=messages, temperature=0.1, max_tokens=900)
        except Exception as e:
            logger.error("LLM generation error: %s", e)
            answer = f"Error generating answer via local Ollama: {e}\n\nRetrieved passages are available in the inspector."

        # 5. Normalize any "Passage X" references into exact citations
        citation_map = {idx: p.get("citation", f"[Ref{idx}]") for idx, p in enumerate(passages, start=1)}
        for idx, tag in citation_map.items():
            answer = re.sub(rf"\bPassage\s+{idx}\b", tag, answer, flags=re.IGNORECASE)

        # Extract citations found in answer
        citation_matches = re.findall(r"\[([A-Za-z]+(?:\d{4})?),\s*pp?\.?\s*([0-9–\-]+)\]", answer)
        citations_found = [f"[{auth}, p. {pg}]" for auth, pg in citation_matches]
        if not citations_found:
            citations_found = [p.get("citation", "") for p in passages if p.get("citation")]

        return {
            "answer": answer,
            "passages": passages,
            "citations_found": list(dict.fromkeys(citations_found)),
            "prompt_used": prompt,
        }
