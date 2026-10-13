from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.llm import OllamaClient, extract_json_block

logger = logging.getLogger(__name__)

THEOREM_PATTERN = re.compile(
    r"\b(Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Remark)\s*(\d+(?:\.\d+)?)?[\.\:]?\s*(.*?)(?=\n\s*(?:Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Remark|\d+\.?\s+[A-Z]|Proof|\Z))",
    re.DOTALL | re.IGNORECASE,
)


def extract_theorems_from_pages(
    paper_id: str,
    pages: List[Tuple[int, str]],
    client: Optional[OllamaClient] = None,
    summarize_with_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Extract structured mathematical theorem and definition records."""
    records: List[Dict[str, Any]] = []
    item_counter = 0

    for page_num, page_text in pages:
        matches = THEOREM_PATTERN.finditer(page_text)
        for match in matches:
            t_type = match.group(1).capitalize()
            t_num = (match.group(2) or "").strip()
            content = match.group(3).strip()

            full_statement = f"{t_type} {t_num}: {content}" if t_num else f"{t_type}: {content}"
            if len(content) < 15:
                continue

            item_counter += 1
            record_id = f"{paper_id}_thm_{item_counter}"

            # Extract structured summary
            summary = generate_theorem_summary(full_statement, client if summarize_with_llm else None)

            records.append({
                "id": record_id,
                "paper_id": paper_id,
                "type": t_type,
                "number": t_num,
                "page": page_num,
                "full_text": full_statement,
                "summary_json": json.dumps(summary, indent=2),
            })

    return records


def generate_theorem_summary(statement: str, client: Optional[OllamaClient] = None) -> Dict[str, Any]:
    """Generate structured JSON summary using rule-based parsing + optional LLM."""
    # 1. Deterministic rule-based extraction
    summary: Dict[str, Any] = {
        "system_type": None,
        "fractional_derivative": None,
        "order_range": None,
        "control_type": None,
        "delays": None,
        "result_type": None,
        "key_conditions": [],
        "method": None,
    }

    lower = statement.lower()

    # System type
    if "quaternion" in lower:
        summary["system_type"] = "quaternion-valued neural network"
    elif "impulsive" in lower:
        summary["system_type"] = "impulsive dynamical system"
    elif "competitive" in lower:
        summary["system_type"] = "competitive neural network"
    elif "semilinear" in lower or "nonlinear" in lower:
        summary["system_type"] = "nonlinear dynamical system"
    elif "linear" in lower:
        summary["system_type"] = "linear dynamical system"

    # Fractional derivative
    if "prabhakar" in lower:
        summary["fractional_derivative"] = "Prabhakar"
    elif "phi-caputo" in lower or "φ-caputo" in lower:
        summary["fractional_derivative"] = "phi-Caputo"
    elif "caputo" in lower:
        summary["fractional_derivative"] = "Caputo"
    elif "riemann-liouville" in lower:
        summary["fractional_derivative"] = "Riemann-Liouville"
    elif "atangana-baleanu" in lower:
        summary["fractional_derivative"] = "Atangana-Baleanu"

    # Order range
    order_match = re.search(r"(?:order|alpha|beta)\s*(?:in|=|\b)\s*(\([0-9\.,\s\)]+|[0-9\.]+)", statement)
    if order_match:
        summary["order_range"] = order_match.group(1).strip()
    elif "(0, 1)" in statement or "0 < alpha < 1" in lower:
        summary["order_range"] = "(0, 1)"

    # Control type
    if "sliding mode" in lower:
        summary["control_type"] = "sliding mode"
    elif "feedback" in lower:
        summary["control_type"] = "state feedback"
    elif "impulsive" in lower:
        summary["control_type"] = "impulsive control"
    elif "open-loop" in lower:
        summary["control_type"] = "open-loop"

    # Delays
    if "time-varying delay" in lower:
        summary["delays"] = "time-varying"
    elif "delay" in lower:
        summary["delays"] = "state delay"

    # Result type
    if "finite-time" in lower:
        summary["result_type"] = "finite-time"
    elif "mittag-leffler" in lower:
        summary["result_type"] = "Mittag-Leffler"
    elif "exact controllability" in lower or "controllab" in lower:
        summary["result_type"] = "exact controllability"
    elif "quasi-projective" in lower:
        summary["result_type"] = "quasi-projective"
    elif "projective synchronization" in lower:
        summary["result_type"] = "projective synchronization"
    elif "asymptotic" in lower:
        summary["result_type"] = "asymptotic"

    # Method
    if "lyapunov-krasovskii" in lower:
        summary["method"] = "Lyapunov-Krasovskii"
    elif "lyapunov" in lower:
        summary["method"] = "Lyapunov"
    elif "lmi" in lower or "linear matrix inequality" in lower:
        summary["method"] = "LMI"
    elif "banach fixed point" in lower or "fixed point" in lower or "krasnoselskii" in lower:
        summary["method"] = "fixed point"
    elif "gramian" in lower:
        summary["method"] = "Gramian"
    elif "sliding mode" in lower:
        summary["method"] = "sliding mode"

    # If client is available and active, refine with LLM
    if client and client.is_online():
        try:
            tpl_path = "prompts/theorem_summary.txt"
            if os.path.exists(tpl_path):
                with open(tpl_path, "r", encoding="utf-8") as f:
                    tpl = f.read()
                prompt = tpl.replace("{statement_text}", statement[:1500])
                resp = client.chat([{"role": "user", "content": prompt}], json_mode=True, max_tokens=300)
                llm_json = extract_json_block(resp)
                if llm_json and isinstance(llm_json, dict):
                    for k, v in llm_json.items():
                        if v is not None and (not summary.get(k) or summary.get(k) is None):
                            summary[k] = v
        except Exception as e:
            logger.debug("Theorem LLM summary fallback to rule-based: %s", e)

    return summary
