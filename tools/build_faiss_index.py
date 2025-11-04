from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, TypeVar

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "faiss is required. Install it with `pip install faiss-cpu`."
    ) from exc

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "sentence-transformers is required. Install it with `pip install sentence-transformers`."
    ) from exc


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: int
    text: str
    file_path: str
    chunker: str
    symbol: str
    start_line: int | None
    end_line: int | None


T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS index from chunked repository data."
    )
    parser.add_argument(
        "--database",
        "-d",
        type=Path,
        default=Path("chunks.sqlite3"),
        help="Path to the SQLite database produced by the chunker.",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model to embed chunks with.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Number of chunks to encode per batch.",
    )
    parser.add_argument(
        "--output-index",
        "-o",
        type=Path,
        default=Path("data") / "chunks.faiss",
        help="Destination for the FAISS index file.",
    )
    parser.add_argument(
        "--metadata",
        "-M",
        type=Path,
        default=Path("data") / "chunks_metadata.jsonl",
        help="Destination for JSONL metadata records aligned with the index.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing index and metadata files.",
    )
    return parser.parse_args()


def log_step(message: str) -> None:
    print(f"[build-index] {message}")


def batched(iterable: Iterator[T], size: int) -> Iterator[List[T]]:
    batch: List[T] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_chunks(connection: sqlite3.Connection) -> Iterator[ChunkRecord]:
    cursor = connection.execute(
        """
        SELECT id, text, file_path, chunker, symbol, start_line, end_line
        FROM chunks
        ORDER BY id ASC
        """
    )
    for row in cursor:
        yield ChunkRecord(
            chunk_id=row[0],
            text=row[1],
            file_path=row[2],
            chunker=row[3],
            symbol=row[4],
            start_line=row[5],
            end_line=row[6],
        )


def ensure_output_targets(index_path: Path, metadata_path: Path, *, force: bool) -> None:
    """Ensure output paths are safe to write."""
    if not force:
        conflicts = [path for path in (index_path, metadata_path) if path.exists()]
        if conflicts:
            names = ", ".join(str(path) for path in conflicts)
            raise SystemExit(f"Refusing to overwrite existing file(s): {names}. Use --force.")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)


def write_manifest(
    manifest_path: Path,
    *,
    model_name: str,
    index_path: Path,
    metadata_path: Path,
    chunk_count: int,
    dimension: int,
) -> None:
    """Write a manifest JSON file describing the built index."""
    manifest = {
        "model_name": model_name,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "chunk_count": chunk_count,
        "vector_dimension": dimension,
        "index_type": "IndexFlatIP",
        "metric": "inner_product",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_index() -> None:
    """Main entry point to build the FAISS index from chunked data."""
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero.")

    if not args.database.exists():
        raise SystemExit(f"Database not found: {args.database}")

    ensure_output_targets(args.output_index, args.metadata, force=args.force)
    manifest_path = args.output_index.with_suffix(".manifest.json")

    log_step(f"Loading SentenceTransformer model: {args.model_name}")
    model = SentenceTransformer(args.model_name, trust_remote_code=True)
    log_step("Model loaded.")

    log_step(f"Opening database: {args.database}")
    connection = sqlite3.connect(args.database)
    try:
        total_chunks = 0
        index = None
        dimension = None

        log_step(f"Streaming chunks and writing metadata to {args.metadata}")
        with args.metadata.open("w", encoding="utf-8") as metadata_stream:
            for batch in batched(load_chunks(connection), args.batch_size):
                texts = [chunk.text for chunk in batch]
                if not texts:
                    continue
                log_step(f"Embedding batch of {len(texts)} chunks (total so far: {total_chunks})")
                embeddings = model.encode(
                    texts,
                    batch_size=args.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                vectors = np.asarray(embeddings, dtype="float32")
                faiss.normalize_L2(vectors)  # ensures cosine similarity with inner product

                if index is None:
                    dimension = int(vectors.shape[1])
                    index = faiss.IndexFlatIP(dimension)
                    log_step(f"Initialized FAISS IndexFlatIP with dimension {dimension}")

                index.add(vectors)

                # We stream metadata so downstream tools can align vector ids.
                for i, chunk in enumerate(batch):
                    record = {
                        "vector_id": total_chunks + i,
                        "chunk_id": chunk.chunk_id,
                        "file_path": chunk.file_path,
                        "chunker": chunk.chunker,
                        "symbol": chunk.symbol,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "text": chunk.text,
                    }
                    metadata_stream.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_chunks += len(batch)
                log_step(f"Processed {total_chunks} chunks so far.")

        if index is None or dimension is None:
            raise SystemExit("No chunks were found in the database; index not created.")

        log_step(f"Writing FAISS index to {args.output_index}")
        faiss.write_index(index, str(args.output_index))
        log_step("FAISS index write complete.")

        log_step(f"Writing manifest to {manifest_path}")
        write_manifest(
            manifest_path,
            model_name=args.model_name,
            index_path=args.output_index,
            metadata_path=args.metadata,
            chunk_count=total_chunks,
            dimension=dimension,
        )
        log_step(f"Indexed {total_chunks} chunks total.")
        log_step(f"Metadata written to {args.metadata}")
        log_step(f"Manifest written to {manifest_path}")
    finally:
        connection.close()
        log_step("Database connection closed.")


if __name__ == "__main__":
    build_index()
