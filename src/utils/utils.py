from typing import Optional

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


def read_file(filename: str) -> str:
    contents: str = ""

    if filename.lower().endswith(".pdf"):
        contents = pdf_to_text(filename)
    else:
        with open(file=filename, mode="r", encoding="utf-8") as file:
            contents = file.read()
    return contents


def pdf_to_text(filepath: str) -> str:
    if _fitz is not None:
        text = ""
        with _fitz.open(filepath) as pdf:
            for page in pdf:
                text += page.get_text()
        return text

    if _PdfReader is not None:
        reader = _PdfReader(filepath)
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    if _fitz_import_error is not None:
        raise RuntimeError(
            "Failed to import PyMuPDF. Install it with 'pip install pymupdf' or provide a compatible setup."
        ) from _fitz_import_error

    raise RuntimeError(
        "PDF support requires PyMuPDF or PyPDF2. Install one of them (e.g. 'pip install pymupdf' or 'pip install pypdf')."
    )
