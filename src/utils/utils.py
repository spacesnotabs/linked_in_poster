from typing import Optional
import pathlib
from typing import Any, Iterable

try:
    import fitz as _fitz  # PyMuPDF
    _fitz_import_error: Optional[Exception] = None
except Exception as exc:  # Capture import issues (e.g. missing static assets)
    _fitz = None
    _fitz_import_error = exc

try:
    from pypdf import PdfReader as _PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader as _PdfReader
    except Exception:
        _PdfReader = None

CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".cpp", ".c", ".hpp", ".h", ".java", ".go", ".rs", ".cs"}
TEXT_EXTS = {".md", ".txt", ".rst", ".pdf"}
DEFAULT_EXTS = CODE_EXTS | TEXT_EXTS

def pdf_to_text(path: pathlib.Path) -> str:
    """Extract text from a PDF file using PyMuPDF or PyPDF2."""
    if _fitz is not None:
        text = ""
        with _fitz.open(str(path)) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    if _PdfReader is not None:
        reader = _PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if _fitz_import_error is not None:
        raise RuntimeError(
            "Failed to import PyMuPDF. Install it with 'pip install pymupdf' or provide a compatible setup."
        ) from _fitz_import_error

    raise RuntimeError(
        "PDF support requires PyMuPDF or PyPDF2. Install one of them (e.g. 'pip install pymupdf' or 'pip install pypdf')."
    )

def iter_files(root: str, exts: Iterable[str] = DEFAULT_EXTS) -> Iterable[pathlib.Path]:
    """Recursively iterate over files in a directory with specified extensions."""
    for p in pathlib.Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def detect_lang(path: pathlib.Path) -> str:
    """Detect programming language or text type based on file extension."""
    m = {
        ".py":"python",".js":"javascript",".ts":"typescript",".tsx":"typescript",
        ".cpp":"cpp",".c":"c",".hpp":"cpp",".h":"c",".java":"java",
        ".go":"go",".rs":"rust",".cs":"csharp",".md":"markdown",".txt":"text",".rst":"rst"
    }
    return m.get(path.suffix.lower(), "text")

def read_text(path: pathlib.Path) -> str:
    """Read text content from a file, handling PDFs appropriately."""
    try:
        if path.suffix.lower() == ".pdf":
            return pdf_to_text(path)
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
