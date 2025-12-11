# CLI Application

Console entry points that drive the LinkedIn Poster services belong here. Commands should assemble dependencies (via the service container once available) and delegate all logic to the `LLMService` and related components.

`src/apps/cli/main.py` now exposes two subcommands:

- `python main.py` (or `python main.py chat`) launches the interactive chat loop backed by `LLMService`.
- `python main.py chunk [...]` wires in `ChunkingService`, letting you select directories/files, monitor chunking progress, and step through the SQLite-backed chunk database from the terminal (`--inspect-only` opens the inspector without starting a job).
