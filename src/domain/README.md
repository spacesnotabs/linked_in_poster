# Domain Layer

The domain package gathers shared data structures, value objects, and business rules that do not rely on infrastructure concerns. Typical contents include:

- Pydantic models describing prompts, chat transcripts, usage metrics, and chunk metadata.
- Dataclasses or enums used by multiple services.
- Validation helpers that operate purely on in-memory data.

Domain modules must not import from infrastructure or application layers.
