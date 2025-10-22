from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from rag.chunkers import doc_chunker, py_chunker  # type: ignore
except ModuleNotFoundError:
    # Fallback if the project is configured with an explicit src prefix.
    from src.rag.chunkers import doc_chunker, py_chunker  # type: ignore


@dataclass(frozen=True)
class Chunker:
    name: str
    predicate: Callable[[Path], bool]
    chunk_fn: Callable[[Path], Iterable[dict]]


CHUNKERS: tuple[Chunker, ...] = (
    Chunker("python", py_chunker.is_python, py_chunker.py_chunks),
    Chunker("document", doc_chunker.is_document, doc_chunker.doc_chunks),
)

EXCLUDED_DIRS = {".git", "__pycache__", "venv", "build"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk a repository and store results in SQLite.")
    parser.add_argument(
        "repo",
        nargs="?",
        default=".",
        help="Root of the repository to walk (default: current directory).",
    )
    parser.add_argument(
        "--database",
        "-d",
        default="chunks.sqlite3",
        help="SQLite database file that will receive chunk records.",
    )
    return parser.parse_args()


def init_db(connection: sqlite3.Connection) -> None:
    """Create the chunks table/index if they are missing."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            chunker TEXT NOT NULL,
            symbol TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            text TEXT NOT NULL,
            UNIQUE(file_path, chunker, symbol)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_file_chunker
        ON chunks (file_path, chunker)
        """
    )


def should_skip_dir(dirname: str) -> bool:
    return dirname in EXCLUDED_DIRS or dirname.startswith(".")


def should_skip_file(filename: str) -> bool:
    return filename.startswith(".")


def iter_files(repo_root: Path) -> Iterator[Path]:
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if not should_skip_dir(d)]
        for name in files:
            if should_skip_file(name):
                continue
            yield Path(root) / name


def select_chunks(path: Path) -> Iterator[tuple[str, dict]]:
    """
    Yield (chunker_name, chunk) pairs for the file.

    We stop after the first matching chunker so only relevant processors run.
    """
    for chunker in CHUNKERS:
        try:
            if not chunker.predicate(path):
                continue
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[warn] predicate failed for {path}: {exc}", file=sys.stderr)
            continue

        try:
            for chunk in chunker.chunk_fn(path):
                if not chunk:
                    continue
                text = chunk.get("text")
                if text is None or not str(text).strip():
                    continue
                yield chunker.name, chunk
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[warn] chunker {chunker.name} failed for {path}: {exc}", file=sys.stderr)
        finally:
            # Only run one chunker per file.
            break


def store_chunks_for_file(
    connection: sqlite3.Connection,
    repo_root: Path,
    file_path: Path,
) -> int:
    relative_path = str(file_path.resolve().relative_to(repo_root.resolve()))
    inserted = 0
    for chunker_name, chunk in select_chunks(file_path):
        connection.execute(
            """
            INSERT OR REPLACE INTO chunks (
                file_path, chunker, symbol, start_line, end_line, text
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                relative_path,
                chunker_name,
                chunk.get("symbol", ""),
                chunk.get("start_line"),
                chunk.get("end_line"),
                chunk.get("text", ""),
            ),
        )
        inserted += 1
    return inserted


def process_repository(repo_root: Path, database_path: Path) -> None:
    repo_root = repo_root.resolve()
    database_path = database_path.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_root}")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    try:
        init_db(connection)
        total_files = 0
        total_chunks = 0
        for file_path in iter_files(repo_root):
            before = total_chunks
            total_chunks += store_chunks_for_file(connection, repo_root, file_path)
            if total_chunks > before:
                total_files += 1
        connection.commit()
    finally:
        connection.close()

    print(f"Stored {total_chunks} chunks across {total_files} files in {database_path}")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo)
    database_path = Path(args.database)
    process_repository(repo_root, database_path)


if __name__ == "__main__":
    main()
