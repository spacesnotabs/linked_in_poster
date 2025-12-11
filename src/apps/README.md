# Apps Layer

This package hosts user-facing entry points that compose services without embedding business logic.

- `web/` will contain the FastAPI application (routes, request/response schemas, dependency wiring).
- `cli/` will contain console commands that orchestrate services for interactive use.
- `tools/` will expose batch utilities (chunking, index build/query) as thin wrappers around services.

Each subpackage should depend only on `domain`, `services`, and the shared dependency container.
