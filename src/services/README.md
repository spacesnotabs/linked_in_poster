# Services Layer

Service modules encapsulate business workflows and lifecycle management for the application. Planned subpackages:

- `llm/` – model registry, session handling, prompt catalogue, and runtime adapters around `LlmModel`.
- `rag/` – document ingestion, chunking, embedding, retrieval, and agent-oriented orchestration.

Services may depend on `domain` for shared types and on `infrastructure` for persistence/configuration adapters. They should not rely on specific UI frameworks or scripts.
