from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import ollama

logger = logging.getLogger(__name__)


def extract_json_block(text: str) -> Optional[dict]:
    """Extract a JSON object from LLM response text, stripping markdown fences."""
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(cleaned[start_idx : end_idx + 1])
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


class OllamaClient:
    """Robust Ollama client with model tiers, retries, JSON mode, and local embeddings."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        chat_model: str = "llama3.2:1b",
        embedding_model: str = "nomic-embed-text",
        timeout_seconds: int = 60,
        max_retries: int = 2,
    ):
        self.host = host
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client: Optional[ollama.Client] = None
        self._available_models: Optional[List[str]] = None

    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            self._client = ollama.Client(host=self.host, timeout=float(self.timeout_seconds))
        return self._client

    def is_online(self) -> bool:
        """Check if local Ollama daemon is reachable."""
        try:
            res = self.client.list()
            models_list = getattr(res, "models", [])
            self._available_models = [
                getattr(m, "model", getattr(m, "name", str(m))) for m in models_list
            ]
            return True
        except Exception as e:
            logger.warning("Ollama not reachable at %s: %s", self.host, e)
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        json_mode: bool = False,
    ) -> str:
        """Generate a chat response with retries."""
        target_model = model or self.chat_model

        for attempt in range(self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": target_model,
                    "messages": messages,
                    "options": {
                        "temperature": float(temperature),
                        "num_predict": int(max_tokens),
                    },
                }
                if json_mode:
                    kwargs["format"] = "json"

                response = self.client.chat(**kwargs)
                return response.get("message", {}).get("content", "").strip()
            except Exception as e:
                logger.warning("Ollama chat attempt %d/%d failed: %s", attempt + 1, self.max_retries + 1, e)
                if attempt == self.max_retries:
                    raise RuntimeError(f"Ollama chat failed after {self.max_retries + 1} attempts: {e}")
                time.sleep(1.0)
        return ""

    def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generate dense vector embeddings for a list of texts."""
        target_model = model or self.embedding_model
        embeddings: List[List[float]] = []

        if not texts:
            return embeddings

        is_connected = self.is_online()

        for text in texts:
            embedded = False
            if is_connected:
                for attempt in range(self.max_retries + 1):
                    try:
                        # Ollama embeddings API
                        res = self.client.embeddings(model=target_model, prompt=text)
                        emb = res.get("embedding", [])
                        if emb:
                            embeddings.append(emb)
                            embedded = True
                            break
                    except Exception as e:
                        logger.warning("Embedding attempt failed on text: %s", e)
                        time.sleep(0.5)

            if not embedded:
                # Deterministic normalized semantic hash embedding fallback (768 dimensions)
                embeddings.append(self._fallback_embedding(text, dim=768))

        return embeddings

    def _fallback_embedding(self, text: str, dim: int = 768) -> List[float]:
        """Deterministic pseudo-semantic embedding vector when offline."""
        vec = [0.0] * dim
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
