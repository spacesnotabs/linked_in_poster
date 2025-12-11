# Infrastructure Layer

Infrastructure modules provide implementations for external concerns:

- Configuration loading and environment management.
- Database engines, session factories, and ORM models.
- Vector index persistence, caches, and file-system adapters.

These modules should expose factories or interfaces that higher layers consume indirectly through the dependency container.
