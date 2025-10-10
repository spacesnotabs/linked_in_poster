from rag.common import Document
from pathlib import Path
from typing import Sequence, Iterator, Iterable
import fnmatch
import hashlib

# ---------- Helper: extension -> type ----------
EXT_TO_TYPE = {
    # Plain text
    ".txt": "text",
    ".md": "markdown",
    ".rst": "text",
    ".adoc": "text",
    ".html": "html",
    ".htm": "html",
    # Code (expand as you like)
    ".py": "code:python",
    ".ipynb": "code:python",   # you may want a special loader later
    ".c": "code:c",
    ".h": "code:c",
    ".cpp": "code:cpp",
    ".hpp": "code:cpp",
    ".cc": "code:cpp",
    ".js": "code:javascript",
    ".ts": "code:typescript",
    ".tsx": "code:typescript",
    ".jsx": "code:javascript",
    ".java": "code:java",
    ".cs": "code:csharp",
    ".go": "code:go",
    ".rs": "code:rust",
    ".php": "code:php",
    ".swift": "code:swift",
    ".kt": "code:kotlin",
    ".m": "code:objectivec",   # ambiguous with MATLAB; adjust if needed
    ".sh": "code:bash",
    ".bat": "code:bash",
    ".ps1": "code:powershell",
    ".sql": "code:sql",
    ".yaml": "text",
    ".yml": "text",
    ".toml": "text",
    ".ini": "text",
    ".cfg": "text",
    ".xml": "text",
    ".json": "text",
}

# Obvious binaries we’ll skip (handle PDFs later with a PDF loader)
BINARY_EXTS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
    ".pptx", ".docx", ".xlsx", ".zip", ".gz", ".tar", ".7z",
    ".mp3", ".mp4", ".mov", ".avi", ".ogg", ".webm",
}

def guess_doc_type(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in EXT_TO_TYPE:
        return EXT_TO_TYPE[ext]
    if ext in BINARY_EXTS:
        return None
    # safe default: treat unknown, non-binary as text
    return "text"

class FilesystemLoader:
    """
    Walks a directory, selects files by include/exclude globs, and yields Documents.
    Keep it simple; specialize later (PDFs, Jupyter, etc.) as needed.
    """
    def __init__(
        self,
        root: str | Path,
        include: Sequence[str] = ("**/*",),            # globs relative to root
        exclude: Sequence[str] = ("**/.git/**",),      # globs to skip
        max_bytes: int = 2_000_000,                    # skip very large files
        follow_symlinks: bool = False,
        skip_hidden: bool = True,                      # skip dotfiles/dirs
    ):
        self.root = Path(root).resolve()
        self.include = tuple(include)
        self.exclude = tuple(exclude)
        self.max_bytes = max_bytes
        self.follow_symlinks = follow_symlinks
        self.skip_hidden = skip_hidden


    def _iter_paths(self) -> Iterator[Path]:
        # Gather candidates matching any include glob
        included: set[Path] = set()
        for pattern in self.include:
            included.update(self.root.glob(pattern))

        # Filter: files only
        candidates = (p for p in included if p.is_file())

        # Exclude by globs
        def is_excluded(p: Path) -> bool:
            rel = p.relative_to(self.root).as_posix()
            return any(fnmatch.fnmatch(rel, pat) for pat in self.exclude)

        # Skip hidden if requested
        def is_hidden(p: Path) -> bool:
            return any(part.startswith(".") for part in p.relative_to(self.root).parts)

        for p in candidates:
            if not self.follow_symlinks and p.is_symlink():
                continue
            if self.skip_hidden and is_hidden(p):
                continue
            if is_excluded(p):
                continue
            yield p

    @staticmethod
    def _stable_doc_id(path: Path) -> str:
        # Stable id from absolute path (you can add size/mtime if you prefer)
        h = hashlib.sha1(str(path).encode("utf-8"))
        return h.hexdigest()

    def _read_text(self, path: Path) -> str | None:
        # Keep it dead simple: UTF-8 with replacement to avoid crashes
        try:
            if path.stat().st_size > self.max_bytes:
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    def load(self) -> Iterable[Document]:
        for path in self._iter_paths():
            doc_type = guess_doc_type(path)
            if doc_type is None:
                continue  # binary or unsupported type

            text = self._read_text(path)
            if not text:
                continue

            yield Document(
                doc_id=self._stable_doc_id(path),
                source=str(path),
                type=doc_type,
                title=path.stem,
                text=text,
                metadata={
                    "relpath": str(path.relative_to(self.root)),
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                    "ext": path.suffix.lower(),
                },
            )