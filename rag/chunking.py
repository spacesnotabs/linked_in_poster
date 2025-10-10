import tiktoken
from typing import List, Tuple
from .common import DocChunk
from src.utils.utils import iter_files, read_text, detect_lang

def code_aware_split(text: str, language: str,
                     target_tokens: int = 300, overlap_tokens: int = 60) -> List[Tuple[int,int,str]]:
    """
    Heuristic splitter:
      - Prefer blank-line boundaries.
      - Keep chunks ~target_tokens using tokenizer for stability across languages.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    def tok_len(s: str) -> int:
        # Allow all special tokens when counting, so tiktoken won't raise.
        return len(enc.encode(s, disallowed_special=()))

    lines = text.splitlines()
    out, cur, cur_tok = [], [], 0

    # flush current chunk to output
    def flush():
        if not cur: return
        start = cur[0][0]; end = cur[-1][0]
        chunk_text = "\n".join(l for _, l in cur)
        out.append((start, end, chunk_text))

    for i, line in enumerate(lines, start=1):
        line_tokens = tok_len(line) or 1 # avoid zero-length lines, at least one token per line
        boundary = (len(line.strip()) == 0)  # blank line is a good boundary
        # if adding this line would exceed target, flush current chunk
        if cur_tok + line_tokens > target_tokens and boundary:
            flush()
            # start next chunk with an overlap
            if overlap_tokens > 0 and out:
                # take tail tokens from previous
                tail = []
                tok_count = 0
                for j in range(len(cur)-1, -1, -1):
                    tok_count += tok_len(cur[j][1]) or 1
                    tail.append(cur[j])
                    if tok_count >= overlap_tokens:
                        break
                tail.reverse()
                cur = tail[:]  # carry overlap forward
                cur_tok = sum(tok_len(l) or 1 for _, l in cur)
            else:
                cur, cur_tok = [], 0
        cur.append((i, line))
        cur_tok += line_tokens

    flush()
    return out

def build_chunks(repo_dir: str) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for path in iter_files(repo_dir):
        txt = read_text(path)
        if not txt.strip(): continue
        lang = detect_lang(path)
        spans = code_aware_split(txt, lang)
        for (start, end, chunk_text) in spans:
            chunk_id = f"{path}:{start}-{end}"
            chunks.append(DocChunk(
                id=chunk_id, text=chunk_text, path=str(path),
                start_line=start, end_line=end, language=lang
            ))
    return chunks

if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    all_chunks = build_chunks(repo)
    print(f"Extracted {len(all_chunks)} chunks from {repo}")
    for c in all_chunks[:5]:
        print(f"--- {c.id} ({c.language}) ---")
        print(c.text)
        print()