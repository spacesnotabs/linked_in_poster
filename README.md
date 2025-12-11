# Linked In Poster #
This tool is meant to help someone create LinkedIn posts in their own style using either local or online LLMs.

The codebase now follows a modular monolith layout (see `ARCHITECTURE.md`) with separate packages for apps, services, and infrastructure.

## Model Configuration

Create a `config/model_config.json` file in the project directory with the following structure:

```json
{
  "system_prompt": "You are a helpful assistant.",
  "model_configs": {
    "Mistral 7B Instruct": {
      "model_filename": "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
      "api_key": null
    }
  }
  // Add more models as needed
}
```

- `system_prompt`: (string) The default system prompt for all models.
- `model_configs`: A dictionary containing configurations for different models.
- Each model section (e.g., `"Mistral 7B Instruct"`) contains:
  - `model_filename`: (string) Path to the model file. Use "DUMMY" for a test model that doesn't require a real model file.
  - `api_key`: (string or null) API key if required by the model.

## Running the Application

### Web Interface (Recommended)

Run the FastAPI app with:

```sh
uvicorn src.apps.web.main:app --reload
```

On Windows, you can also use the helper script:

```bat
run_web.bat --reload
```

Open your browser at `http://localhost:8000` to:
- Load and switch between configured models.
- Chat with the selected model and review token usage dashboards.
- Attach additional context files (text or PDF) via the `+` button in the chat box.
- Apply structured prompts stored in `config/prompts.json`.
- Orchestrate repository chunking directly from the UI: enter directories/files, monitor live progress, and inspect chunked files/chunks inside the "Chunking Workspace".

### Command-Line Interface

To launch the CLI chat experience, run:

```sh
python main.py
```

Select a model from the menu and interact via the console. Type `exit` to quit.

The CLI now also exposes the chunking workflow:

```sh
python main.py chunk path/to/project another/path --database data/chunks.sqlite3
```

You can provide multiple directories/files (or glob patterns), pick chunkers, watch progress, and drop into the interactive database inspector (`python main.py chunk --inspect-only`) to review stored chunks at any time.

### Batch Tools

Utility scripts such as chunking and FAISS index building live under `src/apps/tools`. Example:

```sh
python -m src.apps.tools.chunking /path/to/repo --database data/chunks.sqlite3
python -m src.apps.tools.build_faiss_index --database data/chunks.sqlite3
python -m src.apps.tools.query_faiss_index --query "search text"
```

### Tests

Run the unit tests with:

```sh
python -m unittest discover -s tests
```

_AI Disclaimer: I LOVE using AI to help me write code.  I will be the first to admit that.  It feels like magic and can be extremely helpful!  However, for this project, I wanted to learn how to work with both local LLMs and LLM APIs and the best way for me to learn is to write the code myself.  I am using AI for pieces of the code which are boilerplate or I'm already very familiar with writing.  But, for the model implementation I'm doing the work on my own to ensure I grasp and retain the concepts._
