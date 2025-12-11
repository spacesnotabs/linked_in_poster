from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("faiss must be installed to query the index.") from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("sentence-transformers must be installed to query the index.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the FAISS index produced by build_faiss_index.py."
    )
    parser.add_argument(
        "--index",
        "-i",
        type=Path,
        default=Path("data") / "chunks.faiss",
        help="Path to the FAISS index file.",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=Path,
        default=Path("data") / "chunks_metadata.jsonl",
        help="Path to the JSONL metadata file aligned with the index.",
    )
    parser.add_argument(
        "--model-name",
        "-M",
        help="SentenceTransformers model for queries. Defaults to the value recorded in the manifest.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results to return.",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Search query text. If omitted, the tool enters interactive mode.",
    )
    return parser.parse_args()


def iter_metadata(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    return list(iter_metadata(path))


def get_model_name(index_path: Path, override: str | None) -> str:
    if override:
        return override
    manifest_path = index_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = manifest.get("model_name")
        if name:
            return str(name)
    return "sentence-transformers/all-MiniLM-L6-v2"


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    faiss.normalize_L2(vectors)
    return vectors


def encode_queries(model: SentenceTransformer, queries: Iterable[str]) -> np.ndarray:
    embeddings = model.encode(
        list(queries),
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    vectors = np.asarray(embeddings, dtype="float32")
    return normalize_vectors(vectors)


def print_hits(
    query: str,
    distances: np.ndarray,
    indices: np.ndarray,
    metadata: list[dict[str, Any]],
) -> None:
    print(f"\nQuery: {query}")
    print("-" * (7 + len(query)))
    for rank, (score, idx) in enumerate(zip(distances, indices), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        record = metadata[idx]
        snippet = record.get("text", "").replace("\n", " ").strip()
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
        file_path = record.get("file_path", "<unknown>")
        start_line = record.get("start_line")
        end_line = record.get("end_line")
        print(f"{rank}. score={score:.4f} | {file_path}:{start_line}-{end_line}")
        print(f"    {snippet}")
    print()


def main() -> None:
    args = parse_args()
    if not args.index.exists():
        raise SystemExit(f"Index file not found: {args.index}")
    if not args.metadata.exists():
        raise SystemExit(f"Metadata file not found: {args.metadata}")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than zero.")

    metadata = load_metadata(args.metadata)
    model_name = get_model_name(args.index, args.model_name)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    index = faiss.read_index(str(args.index))

    def run_query(query_text: str) -> None:
        query_vector = encode_queries(model, [query_text])
        distances, indices = index.search(query_vector, args.top_k)
        print_hits(query_text, distances[0], indices[0], metadata)

    if args.query:
        run_query(args.query)
        return

    print("Interactive search. Press Ctrl+C to exit.")
    while True:
        try:
            query_text = input("query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query_text:
            continue
        run_query(query_text)


if __name__ == "__main__":
    main()
