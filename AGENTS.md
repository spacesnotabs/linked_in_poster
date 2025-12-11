## Layout
- `src/apps/` – entry points (FastAPI web UI, CLI, batch tools) that compose services.
- `src/services/` – business logic for LLM workflows (`llm/`) and RAG utilities (`rag/`).
- `src/infrastructure/` – configuration, database, and other external adapters.
- `src/domain/` – shared data models used across services.
- `tests/` – unit tests; currently focused on the LLM service façade.

## Testing
Run the suite with:

```sh
python -m unittest discover -s tests
```

The suite uses dummy backends where possible so it runs without native LLM runtimes. Add new tests alongside services as you extend functionality.

## Practices
- Maintain documentation (package READMEs, architecture notes) to keep the modular boundaries clear.
- Prefer explicit type hints and dataclasses/Pydantic models; they improve readability and help static tooling.
- Keep apps free of global state—wire dependencies via services so components remain testable.
- Update or create tests when changing behaviour and ensure scripts rely on shared services rather than duplicating logic.
- Make a plan before you implement any changes.
- Provide comments in the code where appropriate.
