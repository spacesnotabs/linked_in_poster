"""
General-purpose document chunker for text-like sources (Markdown, TXT, PDF).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from src.utils.utils import pdf_to_text

SUPPORTED_TEXT_EXTS = {".md", ".markdown", ".txt", ".rst"}
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = SUPPORTED_TEXT_EXTS | SUPPORTED_PDF_EXTS

DEFAULT_MAX_LINES = 120
DEFAULT_MAX_CHARS = 2000
DEFAULT_OVERLAP_LINES = 15

HEADING_RE = re.compile(r"^\s{0,3}(#+)\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True)
class Paragraph:
    start_line: int
    end_line: int
    text: str


def is_document(path: Path) -> bool:
    """True if the path points to a supported text document."""
    return path.suffix.lower() in SUPPORTED_EXTS


def read_text(path: Path) -> str:
    """Read a document into text, delegating PDF handling when available."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return pdf_to_text(path)
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def doc_chunks(
    path: Path,
    *,
    max_lines: int = DEFAULT_MAX_LINES,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> Iterable[dict]:
    """
    Yield text chunks that stay within the provided line/character budget.

    Each chunk is decorated with a short header so downstream models keep
    source context (file name / first heading).
    """
    text = read_text(path)
    if not text.strip():
        return

    lines = text.splitlines()
    header = build_header(path, lines)
    doc_id = path_to_id(path)

    for idx, para_block in enumerate(
        build_chunks(lines, max_lines=max_lines, max_chars=max_chars, overlap_lines=overlap_lines),
        start=1,
    ):
        body = para_block.text.strip("\n")
        if not body.strip():
            continue
        merged = "\n\n".join(part for part in (header, body) if part).strip()
        yield {
            "symbol": f"{doc_id}::chunk_{idx}",
            "start_line": para_block.start_line,
            "end_line": para_block.end_line,
            "text": merged,
        }

    yield {
        "symbol": f"{doc_id}::__file__",
        "start_line": 1,
        "end_line": len(lines),
        "text": text,
    }


def build_chunks(
    lines: Sequence[str],
    *,
    max_lines: int,
    max_chars: int,
    overlap_lines: int,
) -> Iterator[Paragraph]:
    """Slide a window over lines, respecting both line and character constraints."""
    total = len(lines)
    start = 0
    while start < total:
        end = start
        char_budget = 0
        while end < total:
            length = len(lines[end]) + 1  # include newline separator
            next_line_span = end - start + 1
            if next_line_span > max_lines or (char_budget + length) > max_chars:
                break
            char_budget += length
            end += 1

        if end == start:
            end += 1

        chunk_lines = lines[start:end]
        content = "\n".join(chunk_lines)
        if content.strip():
            trimmed_start, trimmed_end = trim_line_span(chunk_lines)
            yield Paragraph(
                start_line=start + trimmed_start + 1,
                end_line=start + trimmed_end,
                text=content,
            )

        if end >= total:
            break
        start = max(end - overlap_lines, start + 1)


def trim_line_span(lines: Sequence[str]) -> tuple[int, int]:
    """Locate the first/last non-empty lines within the provided slice."""
    first = 0
    last = len(lines)

    while first < last and not lines[first].strip():
        first += 1
    while last > first and not lines[last - 1].strip():
        last -= 1

    return first, max(last, first + 1)


def build_header(path: Path, lines: Sequence[str]) -> str:
    """Construct a small header with file name and first heading (if present)."""
    title = extract_title(lines)
    parts = []
    stem = path.name
    if stem:
        parts.append(stem)
    if title and title.lower() != stem.lower():
        parts.append(title)
    if not parts:
        return ""
    return " - ".join(parts)


def extract_title(lines: Sequence[str], search_window: int = 40) -> str | None:
    """Grab the first meaningful title/heading within the top section."""
    window = lines[: min(len(lines), search_window)]
    for idx, raw in enumerate(window):
        stripped = raw.strip()
        if not stripped:
            continue
        heading = HEADING_RE.match(raw)
        if heading:
            return heading.group("title").strip()

        # Setext-style underlines (#=) require peeking at next line.
        if idx + 1 < len(window):
            underline = window[idx + 1].strip()
            if underline and set(underline) <= {"=", "-"} and len(set(underline)) == 1:
                return stripped

        # Fallback: use the first non-empty short line as a title.
        if len(stripped) <= 80:
            return stripped

    return None


def path_to_id(path: Path) -> str:
    """Convert the file path into a dotted identifier without the extension."""
    return ".".join(path.with_suffix("").parts)


__all__ = [
    "Paragraph",
    "is_document",
    "doc_chunks",
    "build_header",
    "path_to_id",
]

