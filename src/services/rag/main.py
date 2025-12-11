from __future__ import annotations

import os
from pathlib import Path

from .loader import FilesystemLoader


def run_loader(root_path: str, verbose: bool = False) -> None:
    loader = FilesystemLoader(
        root=root_path,
        include=("**/*.md", "**/*.txt", "**/*.py", "**/*.html"),
        exclude=("**/.git/**", "**/node_modules/**", "**/__pycache__/**"),
    )

    for index, doc in enumerate(loader.load()):
        print(f"[{index:04}] {doc.type:<14} {doc.metadata['relpath']}  ({len(doc.text)} chars)")
        if verbose:
            print(doc.text)
            print("-" * 40)


if __name__ == "__main__":
    if len(os.sys.argv) < 2:
        print("Usage: python -m src.services.rag.main <root_path> [-v|--verbose]")
        os.sys.exit(1)

    verbose_flag = False
    if len(os.sys.argv) > 2 and os.sys.argv[2] in ("-v", "--verbose"):
        verbose_flag = True

    run_loader(os.sys.argv[1], verbose_flag)

