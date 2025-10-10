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

@dataclass
class DocChunk:
    id: str
    text: str
    path: str
    start_line: int
    end_line: int
    language: str

# --- token length estimator (works offline, decent for chunk sizing) ---
_word_re = re.compile(r"\w+")
def est_tokens(s: str) -> int:
    # crude ~1 token ≈ 0.75 words; adjust multiplier if you like
    return max(1, int(len(_word_re.findall(s)) / 0.75))

def make_chunk_id(doc_id: str, start: int, end: int) -> str:
    h = hashlib.sha1(f"{doc_id}:{start}:{end}".encode("utf-8")).hexdigest()
    return h