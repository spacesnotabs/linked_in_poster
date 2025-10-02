from dataclasses import dataclass
import hashlib
import re

# The normalized unit from any loader
@dataclass
class Document:
    doc_id: str              # stable id (e.g., hash of source path/url)
    source: str              # file path or URL
    type: str                # "text", "markdown", "html", "pdf", "code:python", etc.
    title: str | None
    text: str                # cleaned text (for code: the code as text)
    metadata: dict           # anything useful (repo, commit, url anchor, section)

# Retrieval unit stored in the index
@dataclass
class Chunk:
    chunk_id: str            # stable id (doc_id + offsets)
    doc_id: str
    text: str
    start: int               # char (or token) offset in original text
    end: int
    metadata: dict           # section heading, function name, language, url, etc.

# --- token length estimator (works offline, decent for chunk sizing) ---
_word_re = re.compile(r"\w+")
def est_tokens(s: str) -> int:
    # crude ~1 token ≈ 0.75 words; adjust multiplier if you like
    return max(1, int(len(_word_re.findall(s)) / 0.75))

def make_chunk_id(doc_id: str, start: int, end: int) -> str:
    h = hashlib.sha1(f"{doc_id}:{start}:{end}".encode("utf-8")).hexdigest()
    return h