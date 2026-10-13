from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, Field


class ModelsConfig(BaseModel):
    tier_models: Dict[str, str] = Field(
        default_factory=lambda: {
            "small": "llama3.2:1b",
            "medium": "qwen2.5:7b",
            "large": "qwen2.5:14b",
        }
    )
    chat_model: str = "llama3.2:1b"
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = "http://localhost:11434"
    timeout_seconds: int = 60
    max_retries: int = 2


class IngestConfig(BaseModel):
    chunk_size_tokens: int = 650
    chunk_overlap_tokens: int = 100
    use_math_ocr: bool = False
    fuzzy_match_threshold: float = 80.0


class RetrievalConfig(BaseModel):
    top_k_vector: int = 15
    top_k_bm25: int = 15
    top_k_final: int = 8
    hybrid_rrf_k: int = 60
    use_reranker: bool = False


class AgentConfig(BaseModel):
    max_steps: int = 6
    fallback_to_pipeline_on_error: bool = True


class ArxivConfig(BaseModel):
    enabled: bool = False
    categories: List[str] = Field(default_factory=lambda: ["math.OC", "math.DS", "cs.NE"])
    keywords: List[str] = Field(
        default_factory=lambda: ["fractional calculus", "synchronization", "controllability"]
    )


class ScholarConfig(BaseModel):
    library_dir: str = "library"
    bib_files: List[str] = Field(default_factory=lambda: ["library/references.bib"])
    data_dir: str = "data"
    hardware_tier: Literal["small", "medium", "large"] = "small"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)


def load_config(config_path: str = "config.yaml") -> ScholarConfig:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if raw:
                cfg = ScholarConfig.model_validate(raw)
                # Apply hardware tier model if specified
                if cfg.hardware_tier in cfg.models.tier_models:
                    cfg.models.chat_model = cfg.models.tier_models[cfg.hardware_tier]
                return cfg
    return ScholarConfig()
