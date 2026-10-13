from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Pattern to detect section headings
SECTION_HEADING_REGEX = re.compile(
    r"^(?:\d+\.?\s+)?(?:Abstract|Introduction|Preliminaries|Main Results|Model Formulation|System Formulation|Controllability|Synchronization|Stability|Numerical Example|Simulation|Conclusion|References)[\w\s,–—]*$",
    re.IGNORECASE,
)

# Pattern to detect mathematical environment beginnings
BLOCK_START_REGEX = re.compile(
    r"^(Theorem|Lemma|Proposition|Corollary|Definition|Assumption|Remark)\s*(\d+(?:\.\d+)?)?[\.\:]?",
    re.IGNORECASE,
)


def approximate_token_count(text: str) -> int:
    """Approximate token count via whitespace splitting + punctuation weighting."""
    return max(1, int(len(text.split()) * 1.25))


class Block:
    """Atomic block of text within a page, such as a paragraph, theorem, or equation."""

    def __init__(self, text: str, page_num: int, is_atomic: bool = False, section: str = ""):
        self.text = text.strip()
        self.page_num = page_num
        self.is_atomic = is_atomic  # Theorem/Lemma/Equation: do NOT split across chunk boundary
        self.section = section
        self.tokens = approximate_token_count(self.text)


def segment_page_into_blocks(page_num: int, page_text: str, current_section: str) -> Tuple[List[Block], str]:
    """Segment a single page into semantic blocks while tracking current section."""
    blocks: List[Block] = []
    lines = page_text.splitlines()

    current_lines: List[str] = []
    in_math_block = False

    def flush_block():
        nonlocal current_lines, in_math_block
        if not current_lines:
            return
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append(
                Block(
                    text=block_text,
                    page_num=page_num,
                    is_atomic=in_math_block,
                    section=current_section,
                )
            )
        current_lines = []
        in_math_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not in_math_block:
                flush_block()
            continue

        # Check for section heading
        if len(stripped) < 80 and SECTION_HEADING_REGEX.match(stripped):
            flush_block()
            current_section = stripped
            continue

        # Check for math theorem / definition start
        match = BLOCK_START_REGEX.match(stripped)
        if match:
            flush_block()
            in_math_block = True
            current_lines.append(stripped)
            continue

        # If inside a math block, check if it terminates
        if in_math_block:
            # If line is another heading or new numbered item, flush
            if BLOCK_START_REGEX.match(stripped):
                flush_block()
                in_math_block = True
                current_lines.append(stripped)
                continue
            current_lines.append(stripped)
        else:
            current_lines.append(stripped)

    flush_block()
    return blocks, current_section


def chunk_document(
    paper_id: str,
    pages: List[Tuple[int, str]],
    target_tokens: int = 650,
    overlap_tokens: int = 100,
) -> List[Dict[str, Any]]:
    """Chunk document text by section (~500-800 tokens), never splitting inside atomic math blocks."""
    all_blocks: List[Block] = []
    current_section = "Introduction"

    for page_num, page_text in pages:
        page_blocks, current_section = segment_page_into_blocks(page_num, page_text, current_section)
        all_blocks.extend(page_blocks)

    if not all_blocks:
        return []

    chunks: List[Dict[str, Any]] = []
    current_chunk_blocks: List[Block] = []
    current_tokens = 0
    chunk_index = 0

    def create_chunk_dict(blocks: List[Block], idx: int) -> Dict[str, Any]:
        text = "\n\n".join(b.text for b in blocks)
        start_page = min(b.page_num for b in blocks)
        end_page = max(b.page_num for b in blocks)
        # Most frequent section or first section
        section = blocks[0].section if blocks else "General"
        return {
            "id": f"{paper_id}_chunk_{idx}",
            "paper_id": paper_id,
            "start_page": start_page,
            "end_page": end_page,
            "section_title": section,
            "chunk_index": idx,
            "text": text,
            "token_count": approximate_token_count(text),
        }

    i = 0
    current_section_name = all_blocks[0].section if all_blocks else ""

    while i < len(all_blocks):
        block = all_blocks[i]
        block_tokens = block.tokens
        section_changed = (block.section != current_section_name) and (current_tokens >= 80)

        # If single block is starting a chunk
        if not current_chunk_blocks:
            current_chunk_blocks.append(block)
            current_tokens += block_tokens
            current_section_name = block.section
            i += 1
            continue

        # Check if section changed or adding block exceeds target
        if section_changed or (current_tokens + block_tokens > target_tokens + 100):
            # Emit current chunk
            chunks.append(create_chunk_dict(current_chunk_blocks, chunk_index))
            chunk_index += 1

            # If section changed, don't carry over overlap across different major sections
            if section_changed:
                current_chunk_blocks = [block]
                current_tokens = block_tokens
                current_section_name = block.section
                i += 1
                continue

            # Retain overlap blocks
            overlap_accum = 0
            overlap_blocks: List[Block] = []
            for b in reversed(current_chunk_blocks):
                if overlap_accum + b.tokens <= overlap_tokens and not b.is_atomic:
                    overlap_blocks.insert(0, b)
                    overlap_accum += b.tokens
                else:
                    break

            current_chunk_blocks = overlap_blocks + [block]
            current_tokens = sum(b.tokens for b in current_chunk_blocks)
            current_section_name = block.section
            i += 1
        else:
            current_chunk_blocks.append(block)
            current_tokens += block_tokens
            i += 1

    if current_chunk_blocks:
        chunks.append(create_chunk_dict(current_chunk_blocks, chunk_index))

    return chunks
