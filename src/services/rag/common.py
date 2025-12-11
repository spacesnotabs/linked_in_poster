from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(slots=True)
class Document:
    doc_id: str  # stable id (e.g., hash of source path/url)
    source: str  # file path or URL
    type: str  # "text", "markdown", "html", "pdf", "code:python", etc.
    title: Optional[str]
    text: str  # cleaned text (for code: the code as text)
    metadata: Dict[str, object]


@dataclass(slots=True)
class DocChunk:
    id: str
    text: str
    path: str
    start_line: int
    end_line: int
    language: str


_WORD_RE = re.compile(r"\w+")


def est_tokens(value: str) -> int:
    """Crude token estimator (~1 token per 0.75 words)."""
    return max(1, int(len(_WORD_RE.findall(value)) / 0.75))


def make_chunk_id(doc_id: str, start: int, end: int) -> str:
    """Create a deterministic chunk id."""
    digest = hashlib.sha1(f"{doc_id}:{start}:{end}".encode("utf-8"))
    return digest.hexdigest()


__all__ = ["Document", "DocChunk", "est_tokens", "make_chunk_id"]

