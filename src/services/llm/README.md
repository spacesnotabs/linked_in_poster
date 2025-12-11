# LLM Service

This package contains the components that manage lifecycle and interaction with large language models:

- `service.py` – façade consumed by apps; coordinates registry, sessions, and prompts.
- `registry.py` – loads model definitions, applies runtime overrides, and instantiates runtimes.
- `runtime.py` – thin wrapper around `llama_cpp` providing chat-completion helpers.
- `session_manager.py` – tracks chat history, usage metrics, and log files per model.
- `prompts.py` – loads and renders structured prompt definitions from JSON.

The service exposes a controller-like API so existing interfaces can migrate incrementally while gaining clearer boundaries.
