"""
Python code chunker using AST parsing.
"""
from __future__ import annotations
import ast, re, textwrap
from pathlib import Path
from typing import Iterable

def is_python(path: Path) -> bool:
    return path.suffix == ".py"

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

# ---------- Header capture (module docstring + early imports) ----------

def _extract_module_docstring(src: str) -> str | None:
    """Return the module docstring (if any), dedented."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    doc = ast.get_docstring(tree, clean=False)
    if not doc:
        return None
    return textwrap.dedent(doc).strip()

def _extract_top_imports(src: str, window_lines: int = 80) -> list[str]:
    """
    Grab import lines from the first N lines so chunks have context
    like 'from app.db import get_user'.
    """
    lines = src.splitlines()
    head = lines[: min(len(lines), window_lines)]
    imports = []
    for L in head:
        # keep simple one-liners; skip comments
        if L.lstrip().startswith(("import ", "from ")) and not L.lstrip().startswith("#"):
            imports.append(L.rstrip())
    return imports

def capture_header(src: str, start_line: int, import_window: int = 80) -> str:
    """
    Build a small header to prepend to a chunk:
    - module docstring (if present)
    - top-of-file import lines (first ~80 lines by default)
    We ignore start_line for now, but keep it in the signature in case you later
    want to trim header differently for very-early chunks.
    """
    pieces: list[str] = []
    doc = _extract_module_docstring(src)
    if doc:
        pieces.append('"""' + doc + '"""')
    imps = _extract_top_imports(src, window_lines=import_window)
    if imps:
        pieces.extend(imps)
    return "\n".join(pieces).strip()

# ---------- Chunking ----------
def py_chunks(path: Path) -> Iterable[dict]:
    """Yield function/class/module chunks with symbol path + lines."""
    src = read_text(path)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        yield from fallback_chunks(src, path)
        return

    module = path_to_module(path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = getattr(node, "name", "unknown")
            # Compute line span
            start = node.lineno
            end = max(getattr(node, "end_lineno", start), start)
            header = capture_header(src, start)
            text = src.splitlines()[start-1:end]
            symbol = f"{module}.{name}"
            yield {
                "symbol": symbol,
                "start_line": start,
                "end_line": end,
                "text": "\n".join(text_with_context(src, start, end, include_imports=True, docstring=True))
            }
    # Also push a file-level chunk for top-of-file context
    yield {
        "symbol": f"{module}.__file__",
        "start_line": 1,
        "end_line": len(src.splitlines()),
        "text": src
    }

def fallback_chunks(src: str, path: Path, max_lines: int = 200) -> Iterable[dict]:
    """Split long files by paragraphs as a last resort."""
    lines = src.splitlines()
    module = path_to_module(path)
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunk = "\n".join(lines[start:end])
        yield {
            "symbol": f"{module}.__seg_{start}",
            "start_line": start + 1,
            "end_line": end,
            "text": chunk
        }
        start = end

def text_with_context(src: str, start: int, end: int, include_imports=True, docstring=True):
    """Extract lines from start to end (1-based, inclusive) with optional context."""
    lines = src.splitlines()
    prefix = []
    if include_imports:
        for i, L in enumerate(lines[:min(len(lines), 50)], start=1):
            if L.startswith(("import ", "from ")): prefix.append(L)
    body = lines[start-1:end]
    # Keep docstring if present
    return prefix + body

def path_to_module(path: Path) -> str:
    """Convert a file path to a Python module path."""
    parts = []
    for p in path.with_suffix("").parts:
        parts.append(p)
    return ".".join(parts)
