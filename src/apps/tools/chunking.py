from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.services.rag.chunking_service import CHUNKERS, ChunkProgress, ChunkingService, resolve_chunkers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk files and store the output in SQLite.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Paths to directories or files to chunk (defaults to current directory).",
    )
    parser.add_argument(
        "--database",
        "-d",
        default=str(Path("data") / "chunks.sqlite3"),
        help="SQLite database file that will receive chunk records.",
    )
    parser.add_argument(
        "--include",
        "-i",
        nargs="+",
        metavar="TAG",
        help="Only run chunkers whose name or tags match these values (default: all).",
    )
    parser.add_argument(
        "--exclude",
        "-x",
        nargs="+",
        metavar="TAG",
        help="Skip chunkers whose name or tags match these values.",
    )
    parser.add_argument(
        "--list-chunkers",
        action="store_true",
        help="List available chunkers and exit.",
    )
    return parser.parse_args()


def list_chunkers() -> None:
    print("Available chunkers:")
    for chunker in CHUNKERS:
        tags = ", ".join(sorted(chunker.labels | {chunker.name}))
        description = f" - {chunker.description}" if chunker.description else ""
        print(f"  * {chunker.name} ({tags}){description}")


def run_cli(paths: Sequence[str], args: argparse.Namespace) -> None:
    service = ChunkingService(database_path=Path(args.database))

    def emit_progress(update: ChunkProgress) -> None:
        percent = (update.processed_files / update.total_files * 100) if update.total_files else 0
        print(
            f"[{update.processed_files}/{update.total_files} | {percent:5.1f}%] "
            f"{update.message or ''}"
        )

    summary = service.run(
        paths,
        include=args.include,
        exclude=args.exclude,
        progress=emit_progress,
    )
    print(
        f"\nStored {summary.chunks_written} chunks "
        f"from {summary.files_with_chunks}/{summary.file_count} files "
        f"in {summary.database_path}"
    )


def main() -> None:
    args = parse_args()
    if args.list_chunkers:
        list_chunkers()
        return

    chunkers = resolve_chunkers(args.include, args.exclude)
    print(f"Using chunkers: {', '.join(chunker.name for chunker in chunkers)}")
    run_cli(args.paths, args)


if __name__ == "__main__":
    main()
