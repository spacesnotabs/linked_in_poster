from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import List

from src.infrastructure.config.settings import get_settings
from src.services.llm import LLMService
from src.services.rag import ChunkProgress, ChunkingService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI utilities for the LinkedIn poster project.",
    )
    subparsers = parser.add_subparsers(dest="command")

    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session.")
    chat_parser.set_defaults(command="chat")

    chunk_parser = subparsers.add_parser("chunk", help="Run the chunking workflow.")
    chunk_parser.add_argument(
        "paths",
        nargs="*",
        help="Directories/files/globs to chunk (leave empty to enter them interactively).",
    )
    chunk_parser.add_argument(
        "--include",
        "-i",
        nargs="+",
        metavar="TAG",
        help="Only run chunkers whose name or tags match these values.",
    )
    chunk_parser.add_argument(
        "--exclude",
        "-x",
        nargs="+",
        metavar="TAG",
        help="Skip chunkers whose name or tags match these values.",
    )
    chunk_parser.add_argument(
        "--database",
        "-d",
        default=str(Path("data") / "chunks.sqlite3"),
        help="SQLite database path for storing chunks (default: data/chunks.sqlite3).",
    )
    chunk_parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Skip chunking and open the database inspector.",
    )
    chunk_parser.add_argument(
        "--skip-inspect",
        action="store_true",
        help="Do not open the inspector automatically after chunking completes.",
    )

    return parser


def select_model_menu(models: List[str]) -> str:
    """Display a menu for model selection and return the chosen name."""
    if not models:
        raise RuntimeError("No models are configured.")

    print("Select model to use:")
    for idx, name in enumerate(models, 1):
        print(f"{idx}. {name}")

    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
        print("Invalid selection. Please try again.")


def chat_loop(service: LLMService) -> None:
    """Interactive chat loop using the selected model."""
    print("Type 'exit' to quit.")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() == "exit":
            break
        try:
            chat_response = service.send_prompt(prompt=user_input)
        except Exception as exc:
            print(f"Error sending prompt: {exc}")
            continue
        if chat_response:
            print(f"Chat: {chat_response}")
        else:
            print("No response returned.")


def prompt_for_paths() -> list[str]:
    raw = input("Enter directories/files to chunk (comma separated, blank for current directory): ").strip()
    if not raw:
        return ["."]
    return [part.strip() for part in raw.split(",") if part.strip()]


def chunk_progress_printer(update: ChunkProgress) -> None:
    prefix = f"[{update.processed_files}/{update.total_files}]"
    stats = f"{update.files_with_chunks} files with chunks | {update.chunk_count} total chunks"
    if update.message:
        print(f"{prefix} {stats} -> {update.message}")
    else:
        print(f"{prefix} {stats}")


def format_file_label(label: str) -> str:
    if "::" in label:
        root, rel = label.split("::", 1)
        return f"{rel} (root: {root})"
    return label


def open_chunk_inspector(service: ChunkingService) -> None:
    print("\nChunk database inspector. Press Ctrl+C or choose option 4 to exit.")
    db = service.database
    last_files: list[str] = []
    while True:
        print(
            "\n1) Show summary\n"
            "2) List chunked files\n"
            "3) View chunks for a file\n"
            "4) Exit inspector"
        )
        choice = input("Select option [1-4]: ").strip() or "1"
        if choice == "1":
            summary = db.summarize()
            print(
                f"\nDatabase: {summary['database']}\n"
                f"Total files: {summary['total_files']}\n"
                f"Total chunks: {summary['total_chunks']}"
            )
            if summary["chunkers"]:
                print("By chunker:")
                for row in summary["chunkers"]:
                    print(f"  - {row['chunker']}: {row['count']} chunks")
        elif choice == "2":
            search = input("Filter file path (substring, blank for all): ").strip() or None
            page = db.list_files(search=search, limit=15)
            last_files = [record["file_path"] for record in page["items"]]
            if not page["items"]:
                print("No files found.")
                continue
            print(f"\nShowing {page['count']} of {page['total']} files:")
            for idx, record in enumerate(page["items"], 1):
                label = format_file_label(record["file_path"])
                print(
                    f"{idx:2}. {label}\n"
                    f"    label={record['file_path']}\n"
                    f"    chunker={record['chunker']} "
                    f"chunks={record['chunk_count']} "
                    f"lines={record['first_line']}..{record['last_line']}"
                )
        elif choice == "3":
            file_hint = input(
                "Enter file number from last list or the exact stored label: "
            ).strip()
            if not file_hint:
                continue
            if file_hint.isdigit():
                selection = int(file_hint)
                if 1 <= selection <= len(last_files):
                    file_path = last_files[selection - 1]
                else:
                    print("Number out of range.")
                    continue
            else:
                file_path = file_hint
            limit_raw = input("How many chunks to display? [default 5]: ").strip()
            limit = int(limit_raw) if limit_raw.isdigit() else 5
            page = db.list_chunks(file_path=file_path, limit=limit)
            if not page["items"]:
                print("No chunks found for that file.")
                continue
            for record in page["items"]:
                snippet = textwrap.shorten(record["text"].replace("\n", " "), width=110, placeholder=" …")
                print(
                    f"\n[{record['id']}] {record['symbol']} "
                    f"({record['start_line']}..{record['end_line']})\n"
                    f"{snippet}"
                )
            view_id = input("Enter chunk ID to show full text (blank to skip): ").strip()
            if view_id.isdigit():
                chunk = db.get_chunk(int(view_id))
                if chunk:
                    print("\n" + "-" * 60)
                    print(chunk["text"])
                    print("-" * 60)
                else:
                    print("Chunk not found.")
        elif choice in {"4", "q", "quit", "exit"}:
            break
        else:
            print("Invalid selection.")


def run_chat() -> None:
    settings = get_settings()
    service = LLMService(settings=settings)
    available_models = service.available_models
    model_name = select_model_menu(available_models)
    try:
        service.initialize_model(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize model '{model_name}': {exc}") from exc
    chat_loop(service)


def run_chunk_workflow(args: argparse.Namespace) -> None:
    settings = get_settings()
    service = ChunkingService(settings=settings, database_path=Path(args.database))
    if args.inspect_only:
        open_chunk_inspector(service)
        return

    target_paths = args.paths or prompt_for_paths()
    print(f"Chunking targets: {', '.join(target_paths)}")
    try:
        summary = service.run(
            target_paths,
            include=args.include,
            exclude=args.exclude,
            progress=chunk_progress_printer,
        )
    except Exception as exc:
        print(f"Chunking failed: {exc}")
        return

    print(
        f"\nStored {summary.chunks_written} chunks "
        f"from {summary.files_with_chunks}/{summary.file_count} files.\n"
        f"Database: {summary.database_path}"
    )
    if not args.skip_inspect:
        open_chunk_inspector(service)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "chunk":
        run_chunk_workflow(args)
        return
    run_chat()


if __name__ == "__main__":
    main()
