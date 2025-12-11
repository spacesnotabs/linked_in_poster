# LinkedIn Poster — Modular Monolith Architecture

This document captures the target structure for the ongoing refactor from an ad‑hoc script layout into a modular, testable monolith. The goals are:

- isolate user interfaces (web, CLI, batch tooling) from the core services they consume
- define service boundaries for model lifecycle, retrieval‑augmented generation, and persistence
- centralize configuration and runtime wiring so environments can be swapped without code edits
- keep domain concepts (prompts, chats, chunks, posts) independent from infrastructure choices

## Package Layout

```
src/
  apps/
    web/            # FastAPI app: routes, request models, dependency wiring
    cli/            # Console command(s) that compose services
    tools/          # Batch/utility entry points (chunking, index build/query)
  domain/
    models/         # Pydantic/dataclass definitions shared across services
  services/
    llm/            # Model registry, session manager, prompt catalogue, runtime adapter
    rag/            # Chunking, embedding, retrieval orchestration
  infrastructure/
    config/         # Settings, environment management, configuration loaders
    db/             # SQLAlchemy engine/session factories and ORM models
    vector/         # Vector index persistence (FAISS abstractions, manifests)
  interfaces/
    container.py    # Lightweight dependency container / factory functions
  support/
    logging.py      # Cross-cutting helpers (logging, tracing, utils)
```

Each subdirectory will receive its own README as we migrate code into place.

## Migration Principles

1. **No global state in apps** – interfaces resolve dependencies via the container.
2. **Services own lifecycle** – LLM sessions, prompts, and RAG workflows expose explicit APIs.
3. **Configurable runtime paths** – chat logs, prompt catalogues, and indexes live under a configurable data directory.
4. **Tests guard boundaries** – unit tests will target service APIs; integration tests exercise adapters.
5. **Container-friendly** – no hard-coded local paths; environment variables drive configuration.

## Execution Plan

1. Introduce the new package scaffold and migrate configuration to `infrastructure`.
2. Split the LLM controller into dedicated services and update the web/CLI apps to consume them.
3. Relocate RAG tooling under `services/rag`, expose reusable components, and update batch tools.
4. Add documentation and unit tests for each module boundary; ensure existing functionality is covered.

This document will evolve as the refactor proceeds.

