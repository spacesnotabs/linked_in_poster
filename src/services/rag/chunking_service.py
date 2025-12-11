from __future__ import annotations

import glob
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Sequence

from src.infrastructure.config.settings import AppSettings, get_settings
from src.services.rag.chunkers import doc_chunker, py_chunker

EXCLUDED_DIRS = {".git", "__pycache__", "venv", "build"}


@dataclass(frozen=True)
class Chunker:
    name: str
    predicate: Callable[[Path], bool]
    chunk_fn: Callable[[Path], Iterable[dict]]
    labels: frozenset[str] = frozenset()
    description: str = ""

    @property
    def keywords(self) -> frozenset[str]:
        return frozenset({self.name.lower(), *{label.lower() for label in self.labels}})


CHUNKERS: tuple[Chunker, ...] = (
    Chunker(
        "python",
        py_chunker.is_python,
        py_chunker.py_chunks,
        labels=frozenset({"code", "py"}),
        description="Python source files chunked via AST.",
    ),
    Chunker(
        "document",
        doc_chunker.is_document,
        doc_chunker.doc_chunks,
        labels=frozenset({"docs", "text", "markdown", "md", "pdf"}),
        description="Markdown/TXT/PDF documents chunked by paragraphs/headings.",
    ),
)


def normalize_keywords(values: Iterable[str]) -> frozenset[str]:
    return frozenset(value.lower() for value in values)


def resolve_chunkers(include: Sequence[str] | None, exclude: Sequence[str] | None) -> tuple[Chunker, ...]:
    include_set = normalize_keywords(include) if include else None
    exclude_set = normalize_keywords(exclude) if exclude else frozenset()

    active: list[Chunker] = []
    for chunker in CHUNKERS:
        keywords = chunker.keywords
        if include_set is not None and not keywords & include_set:
            continue
        if keywords & exclude_set:
            continue
        active.append(chunker)

    if not active:
        available = ", ".join(sorted({kw for chunker in CHUNKERS for kw in chunker.keywords}))
        raise ValueError(f"No chunkers selected. Available tags: {available}")
    return tuple(active)


def should_skip_dir(dirname: str) -> bool:
    return dirname in EXCLUDED_DIRS or dirname.startswith(".")


def should_skip_file(filename: str) -> bool:
    return filename.startswith(".")


@dataclass(frozen=True)
class ChunkTarget:
    root: Path
    file_path: Path

    @property
    def display(self) -> str:
        try:
            resolved_root = self.root.resolve()
            rel = self.file_path.resolve().relative_to(resolved_root)
            return f"{resolved_root.as_posix()}::{rel.as_posix()}"
        except ValueError:
            return str(self.file_path.resolve())


@dataclass(slots=True)
class ChunkProgress:
    total_files: int
    processed_files: int
    chunk_count: int
    files_with_chunks: int
    current_path: str | None = None
    message: str | None = None


@dataclass(slots=True)
class ChunkRunSummary:
    database_path: Path
    requested_paths: tuple[str, ...]
    chunkers: tuple[str, ...]
    file_count: int
    files_with_chunks: int
    chunks_written: int


class ChunkDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def init(self, connection: sqlite3.Connection) -> None:
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

    def summarize(self) -> dict:
        with self.connect() as connection:
            self.init(connection)
            total_chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_files = connection.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()[0]
            per_chunker = [
                {"chunker": row[0], "count": row[1]}
                for row in connection.execute(
                    "SELECT chunker, COUNT(*) FROM chunks GROUP BY chunker ORDER BY chunker"
                )
            ]
        return {
            "database": str(self.path),
            "total_chunks": total_chunks,
            "total_files": total_files,
            "chunkers": per_chunker,
        }

    def list_files(
        self,
        *,
        search: str | None = None,
        chunker: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        like_value = f"%{search}%" if search else None
        with self.connect() as connection:
            self.init(connection)
            rows = connection.execute(
                """
                SELECT file_path, chunker, COUNT(*) as chunk_count,
                       MIN(start_line) as first_line, MAX(end_line) as last_line
                FROM chunks
                WHERE (:chunker IS NULL OR chunker = :chunker)
                  AND (:search IS NULL OR file_path LIKE :search)
                GROUP BY file_path, chunker
                ORDER BY file_path
                LIMIT :limit OFFSET :offset
                """,
                {
                    "chunker": chunker,
                    "search": like_value,
                    "limit": limit,
                    "offset": offset,
                },
            ).fetchall()
            total = connection.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT 1
                    FROM chunks
                    WHERE (:chunker IS NULL OR chunker = :chunker)
                      AND (:search IS NULL OR file_path LIKE :search)
                    GROUP BY file_path, chunker
                )
                """,
                {
                    "chunker": chunker,
                    "search": like_value,
                },
            ).fetchone()[0]
        records = [
            {
                "file_path": row["file_path"],
                "chunker": row["chunker"],
                "chunk_count": row["chunk_count"],
                "first_line": row["first_line"],
                "last_line": row["last_line"],
            }
            for row in rows
        ]
        return {"items": records, "count": len(records), "total": total}

    def list_chunks(
        self,
        *,
        file_path: str | None = None,
        chunker: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        with self.connect() as connection:
            self.init(connection)
            rows = connection.execute(
                """
                SELECT id, file_path, chunker, symbol, start_line, end_line, text
                FROM chunks
                WHERE (:file_path IS NULL OR file_path = :file_path)
                  AND (:chunker IS NULL OR chunker = :chunker)
                ORDER BY id
                LIMIT :limit OFFSET :offset
                """,
                {
                    "file_path": file_path,
                    "chunker": chunker,
                    "limit": limit,
                    "offset": offset,
                },
            ).fetchall()
            total = connection.execute(
                """
                SELECT COUNT(*)
                FROM chunks
                WHERE (:file_path IS NULL OR file_path = :file_path)
                  AND (:chunker IS NULL OR chunker = :chunker)
                """,
                {
                    "file_path": file_path,
                    "chunker": chunker,
                },
            ).fetchone()[0]
        records = [
            {
                "id": row["id"],
                "file_path": row["file_path"],
                "chunker": row["chunker"],
                "symbol": row["symbol"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "text": row["text"],
            }
            for row in rows
        ]
        return {"items": records, "count": len(records), "total": total}

    def get_chunk(self, chunk_id: int) -> dict | None:
        with self.connect() as connection:
            self.init(connection)
            row = connection.execute(
                """
                SELECT id, file_path, chunker, symbol, start_line, end_line, text
                FROM chunks
                WHERE id = :chunk_id
                """,
                {"chunk_id": chunk_id},
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "file_path": row["file_path"],
            "chunker": row["chunker"],
            "symbol": row["symbol"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "text": row["text"],
        }


class ChunkingService:
    def __init__(self, settings: AppSettings | None = None, database_path: Path | None = None) -> None:
        self.settings = settings or get_settings()
        self.database_path = database_path or (self.settings.data_dir / "chunks.sqlite3")
        self.database = ChunkDatabase(self.database_path)

    def available_chunkers(self) -> list[dict]:
        return [
            {
                "name": chunker.name,
                "labels": sorted(chunker.labels),
                "description": chunker.description,
            }
            for chunker in CHUNKERS
        ]

    def run(
        self,
        paths: Sequence[str | Path],
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        progress: Callable[[ChunkProgress], None] | None = None,
    ) -> ChunkRunSummary:
        if not paths:
            raise ValueError("Provide at least one path to chunk.")

        chunkers = resolve_chunkers(include, exclude)
        targets = self._collect_targets(paths, chunkers)
        if not targets:
            raise ValueError("No files matched the requested chunkers.")

        total_files = len(targets)
        processed = 0
        files_with_chunks = 0
        chunk_count = 0

        connection = self.database.connect()
        try:
            self.database.init(connection)
            for target in targets:
                inserted, error_message = self._store_chunks_for_file(connection, target, chunkers)
                processed += 1
                if inserted:
                    files_with_chunks += 1
                    chunk_count += inserted
                    message = f"{inserted} chunks stored for {target.display}"
                else:
                    message = f"No chunks produced for {target.display}"
                if error_message:
                    message = error_message
                if progress:
                    progress(
                        ChunkProgress(
                            total_files=total_files,
                            processed_files=processed,
                            chunk_count=chunk_count,
                            files_with_chunks=files_with_chunks,
                            current_path=target.display,
                            message=message,
                        )
                    )
            connection.commit()
        finally:
            connection.close()

        return ChunkRunSummary(
            database_path=self.database_path,
            requested_paths=tuple(str(Path(p)) for p in paths),
            chunkers=tuple(chunker.name for chunker in chunkers),
            file_count=total_files,
            files_with_chunks=files_with_chunks,
            chunks_written=chunk_count,
        )

    def _collect_targets(self, paths: Sequence[str | Path], chunkers: Sequence[Chunker]) -> list[ChunkTarget]:
        resolved: set[Path] = set()
        targets: list[ChunkTarget] = []
        for raw in paths:
            for expanded in self._expand_path(raw):
                if expanded.is_dir():
                    targets.extend(self._collect_from_dir(expanded, chunkers, resolved))
                elif expanded.is_file():
                    self._append_if_chunkable(expanded, expanded.parent or expanded, chunkers, resolved, targets)
                else:
                    raise FileNotFoundError(f"Path does not exist: {expanded}")
        targets.sort(key=lambda target: target.display)
        return targets

    def _append_if_chunkable(
        self,
        file_path: Path,
        root: Path,
        chunkers: Sequence[Chunker],
        seen: set[Path],
        output: list[ChunkTarget],
    ) -> None:
        resolved = file_path.resolve()
        if resolved in seen:
            return
        if not self._is_chunkable(resolved, chunkers):
            return
        seen.add(resolved)
        output.append(ChunkTarget(root=root.resolve(), file_path=resolved))

    def _collect_from_dir(
        self,
        directory: Path,
        chunkers: Sequence[Chunker],
        seen: set[Path],
    ) -> list[ChunkTarget]:
        output: list[ChunkTarget] = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            for filename in files:
                if should_skip_file(filename):
                    continue
                candidate = Path(root) / filename
                self._append_if_chunkable(candidate, directory, chunkers, seen, output)
        return output

    @staticmethod
    def _has_glob(raw: str | Path) -> bool:
        value = str(raw)
        return any(char in value for char in "*?[]")

    def _expand_path(self, raw: str | Path) -> list[Path]:
        value = Path(raw).expanduser()
        if self._has_glob(value):
            matches = [Path(match).resolve() for match in glob.glob(str(value), recursive=True)]
            if not matches:
                raise FileNotFoundError(f"No files matched pattern: {raw}")
            return matches
        if not value.exists():
            raise FileNotFoundError(f"Path does not exist: {value}")
        return [value.resolve()]

    @staticmethod
    def _is_chunkable(path: Path, chunkers: Sequence[Chunker]) -> bool:
        for chunker in chunkers:
            try:
                if chunker.predicate(path):
                    return True
            except Exception:
                continue
        return False

    def _store_chunks_for_file(
        self,
        connection: sqlite3.Connection,
        target: ChunkTarget,
        chunkers: Sequence[Chunker],
    ) -> tuple[int, str | None]:
        inserted = 0
        for chunker in chunkers:
            try:
                if not chunker.predicate(target.file_path):
                    continue
            except Exception as exc:  # pragma: no cover - defensive
                return 0, f"[warn] predicate failed for {target.display}: {exc}"
            try:
                for chunk in chunker.chunk_fn(target.file_path):
                    text = chunk.get("text")
                    if not text or not str(text).strip():
                        continue
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO chunks (
                            file_path, chunker, symbol, start_line, end_line, text
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            target.display,
                            chunker.name,
                            chunk.get("symbol", ""),
                            chunk.get("start_line"),
                            chunk.get("end_line"),
                            text,
                        ),
                    )
                    inserted += 1
            except Exception as exc:  # pragma: no cover - defensive
                return inserted, f"[warn] chunker {chunker.name} failed for {target.display}: {exc}"
            break
        return inserted, None


@dataclass(slots=True)
class ChunkJob:
    job_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = "pending"
    paths: tuple[str, ...] = ()
    chunkers: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_files: int = 0
    processed_files: int = 0
    files_with_chunks: int = 0
    chunk_count: int = 0
    last_path: str | None = None
    error: str | None = None
    logs: list[dict] = field(default_factory=list)
    database_path: str | None = None

    def snapshot(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "paths": list(self.paths),
            "chunkers": list(self.chunkers),
            "created_at": self.created_at.isoformat() + "Z",
            "started_at": self.started_at.isoformat() + "Z" if self.started_at else None,
            "completed_at": self.completed_at.isoformat() + "Z" if self.completed_at else None,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "files_with_chunks": self.files_with_chunks,
            "chunk_count": self.chunk_count,
            "last_path": self.last_path,
            "error": self.error,
            "logs": list(self.logs),
            "database_path": self.database_path,
        }


class ChunkJobManager:
    def __init__(self, service: ChunkingService) -> None:
        self.service = service
        self._jobs: dict[str, ChunkJob] = {}
        self._lock = threading.Lock()

    def start_job(
        self,
        paths: Sequence[str | Path],
        *,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
    ) -> ChunkJob:
        job_id = uuid.uuid4().hex
        job = ChunkJob(
            job_id=job_id,
            paths=tuple(str(Path(p)) for p in paths),
        )
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job, include, exclude),
            daemon=True,
        )
        thread.start()
        return job

    def _run_job(
        self,
        job: ChunkJob,
        include: Sequence[str] | None,
        exclude: Sequence[str] | None,
    ) -> None:
        job.started_at = datetime.utcnow()
        job.status = "running"

        def progress(update: ChunkProgress) -> None:
            with self._lock:
                job.total_files = update.total_files
                job.processed_files = update.processed_files
                job.files_with_chunks = update.files_with_chunks
                job.chunk_count = update.chunk_count
                job.last_path = update.current_path
                if update.message:
                    self._append_log(job, update.message)

        try:
            summary = self.service.run(job.paths, include=include, exclude=exclude, progress=progress)
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = datetime.utcnow()
            return

        with self._lock:
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.total_files = summary.file_count
            job.processed_files = summary.file_count
            job.files_with_chunks = summary.files_with_chunks
            job.chunk_count = summary.chunks_written
            job.database_path = str(summary.database_path)
            job.chunkers = summary.chunkers
            self._append_log(job, f"Job finished with {summary.chunks_written} chunks.")

    def _append_log(self, job: ChunkJob, message: str) -> None:
        entry = {"timestamp": datetime.utcnow().isoformat() + "Z", "message": message}
        job.logs.append(entry)
        if len(job.logs) > 200:
            del job.logs[0 : len(job.logs) - 200]

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return job.snapshot()

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [job.snapshot() for job in self._jobs.values()]


__all__ = [
    "Chunker",
    "CHUNKERS",
    "ChunkingService",
    "ChunkDatabase",
    "ChunkJobManager",
    "ChunkJob",
    "ChunkProgress",
    "ChunkRunSummary",
    "resolve_chunkers",
]
