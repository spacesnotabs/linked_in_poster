# RAG Services

This package hosts retrieval-augmented generation components such as loaders, chunkers, and retrieval pipelines. The `chunking_service` module now provides the shared workflow that all apps use when slicing repositories into searchable chunks.

## Chunking workflow

- **ChunkingService** wraps the chunker registry (currently Markdown/PDF + Python AST chunkers), walks the supplied paths/directories, and writes chunk records into a SQLite database (`data/chunks.sqlite3` by default). Progress updates are exposed through a callback so CLIs and background jobs can surface live status.
- **ChunkDatabase** abstracts the SQLite schema (table creation, summaries, per-file queries, chunk detail lookups). All API/CLI/database inspectors use these helpers to avoid duplicating SQL.
- **ChunkJobManager** runs `ChunkingService.run` inside background threads, tracks job metadata/logs, and exposes snapshots for the web API. Jobs are lightweight and operate purely on the filesystem, so no external workers are required.

The FastAPI web app, CLI, and the legacy `src/apps/tools/chunking.py` script all consume `ChunkingService`, ensuring a single source of truth for chunk selection, storage, and introspection.
