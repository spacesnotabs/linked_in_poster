import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import MODELS_CONFIG, MODEL_NAMES, SYSTEM_PROMPT
from src.core.llm_controller import LLMController
from src.core.llm_model import LlmModel, UsageMetrics
from src.utils.utils import read_text
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ModelSettingsPayload,
    ModelSwitchRequest,
    PromptApplyRequest,
)

# --- FastAPI App Initialization ---
app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="src/api/templates")

# --- Controller state ---
llm_controller = LLMController(system_prompt=SYSTEM_PROMPT)
selected_model_name: Optional[str] = None

TEXT_EXTENSIONS = {
    '.txt',
    '.md',
    '.markdown',
    '.csv',
    '.tsv',
    '.json',
    '.yaml',
    '.yml',
    '.log',
    '.ini',
    '.cfg',
    '.py',
    '.js',
    '.ts',
    '.html',
    '.css',
    '.xml',
    '.tex',
    '.pdf',
}
ADDITIONAL_TEXT_MIME_TYPES = {'application/json'}
PDF_EXTENSION = '.pdf'
PDF_MIME_TYPES = {'application/pdf'}
TEXT_MIME_PREFIX = 'text/'


def initialize_model(model_name: str) -> LlmModel:
    """Initializes the LLM model via the controller."""
    global selected_model_name
    model_instance = llm_controller.initialize_model(model_name)
    selected_model_name = model_instance.model_name
    return model_instance


def _get_active_state() -> Tuple[Optional[str], Dict[str, Any]]:
    state = llm_controller.get_state()
    active_id = state.get("active_model")
    model_state = state.get("models", {}).get(active_id or "", {}) if active_id else {}
    return active_id, model_state


def _get_usage_snapshots() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    empty_usage = UsageMetrics().to_dict()
    _, model_state = _get_active_state()
    usage_block = model_state.get("usage", {}) if model_state else {}
    last_usage = usage_block.get("last_interaction", empty_usage)
    chat_usage = usage_block.get("chat", empty_usage)
    return last_usage, chat_usage


# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/settings", response_class=HTMLResponse)
async def render_settings_page(request: Request):
    return templates.TemplateResponse("model_settings.html", {"request": request})


@app.get("/api/models", response_class=JSONResponse)
async def get_models():
    """Return available model names and the currently selected model."""
    active_model_id, _ = _get_active_state()
    current_model_config = None
    if active_model_id:
        current_model_config = llm_controller.get_model(active_model_id).get_model_config()
    controller_state = llm_controller.get_state()
    return {
        "models": MODEL_NAMES,
        "models_config": MODELS_CONFIG,
        "selected_model": selected_model_name,
        "current_model_config": current_model_config,
        "is_model_loaded": bool(controller_state.get("loaded_models")),
        "controller_state": controller_state,
    }


@app.get("/api/model_settings", response_class=JSONResponse)
async def get_model_settings(model_name: Optional[str] = None):
    """Return effective runtime parameters for the requested model."""

    resolved_name = (model_name or "").strip()
    if not resolved_name:
        resolved_name = selected_model_name or (MODEL_NAMES[0] if MODEL_NAMES else "")

    if not resolved_name:
        return JSONResponse({"error": "No models configured."}, status_code=404)
    if resolved_name not in MODEL_NAMES:
        return JSONResponse({"error": "Model not found."}, status_code=404)

    settings = llm_controller.get_effective_runtime_config(resolved_name)
    is_loaded = resolved_name in llm_controller.loaded_models
    is_active = llm_controller.active_model_id == resolved_name

    current_config = None
    if is_loaded:
        current_config = llm_controller.get_model(resolved_name).get_model_config()

    return {
        "model_name": resolved_name,
        "settings": settings,
        "is_loaded": is_loaded,
        "is_active": is_active,
        "current_config": current_config,
        "requires_reload": is_loaded,
    }


@app.post("/api/model_settings", response_class=JSONResponse)
async def update_model_settings(payload: ModelSettingsPayload):
    """Persist runtime override values for subsequent model loads."""

    data = payload.dict()
    model_name = data.pop("model_name")

    if model_name not in MODEL_NAMES:
        return JSONResponse({"error": "Model not found."}, status_code=404)

    sanitized = llm_controller.set_runtime_overrides(model_name, data)

    return {
        "model_name": model_name,
        "settings": sanitized,
        "requires_reload": model_name in llm_controller.loaded_models,
        "message": (
            "Settings saved. Reload the model to apply changes."
            if model_name in llm_controller.loaded_models
            else "Settings saved. They will be used on next load."
        ),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    if not llm_controller.loaded_models:
        return JSONResponse({"error": "No model loaded."}, status_code=400)

    response_text = llm_controller.send_prompt(chat_request.prompt)
    usage, chat_usage = _get_usage_snapshots()
    return ChatResponse(
        response=response_text or "",
        usage=usage,
        chat_usage=chat_usage,
    )


@app.post("/api/clear")
async def clear_history():
    if not llm_controller.loaded_models:
        return JSONResponse({"error": "No model loaded."}, status_code=400)

    llm_controller.clear_chat_history()
    return JSONResponse({"message": "Chat history cleared"})


def _load_model(model_name: str) -> JSONResponse:
    if model_name not in MODEL_NAMES:
        return JSONResponse({"error": "Model not found"}, status_code=404)
    try:
        model_instance = initialize_model(model_name)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to load model: {exc}"}, status_code=500)

    _, model_state = _get_active_state()
    return JSONResponse({
        "message": f"Loaded model: {model_instance.model_name}",
        "model_config": model_instance.get_model_config(),
        "session_cleared": True,
        "controller_state": llm_controller.get_state(),
        "usage": model_state.get("usage") if model_state else None,
    })


@app.post("/api/load_model")
async def load_model(switch_request: ModelSwitchRequest):
    return _load_model(switch_request.model_name)


@app.post("/api/switch_model")
async def switch_model(switch_request: ModelSwitchRequest):
    return _load_model(switch_request.model_name)


@app.get("/api/prompts", response_class=JSONResponse)
async def get_prompts():
    """Return available prompt metadata and the current defaults."""

    prompt_entries = []
    for name in llm_controller.list_prompts():
        try:
            prompt_definition = llm_controller.get_prompt(name)
        except KeyError:
            continue
        prompt_entries.append({
            "name": name,
            "task": prompt_definition.get("task"),
        })

    controller_state = llm_controller.get_state()
    default_prompt_text = controller_state.get("default_system_prompt", SYSTEM_PROMPT)
    active_prompt_text = llm_controller.get_active_system_prompt() if llm_controller.loaded_models else None

    return {
        "prompts": prompt_entries,
        "default_prompt": default_prompt_text,
        "active_prompt": active_prompt_text,
        "is_model_loaded": bool(llm_controller.loaded_models),
    }


@app.get("/api/prompts/{prompt_name}", response_class=JSONResponse)
async def get_prompt_preview(prompt_name: str):
    """Return the rendered prompt text for preview purposes."""

    if prompt_name == "__default__":
        controller_state = llm_controller.get_state()
        return {
            "name": "__default__",
            "prompt_text": controller_state.get("default_system_prompt", SYSTEM_PROMPT),
            "definition": None,
        }

    try:
        prompt_definition = llm_controller.get_prompt(prompt_name)
        prompt_text = llm_controller.render_prompt(prompt_name)
    except KeyError:
        return JSONResponse({"error": "Prompt not found."}, status_code=404)

    return {
        "name": prompt_name,
        "prompt_text": prompt_text,
        "definition": prompt_definition,
    }


@app.post("/api/prompts/apply", response_class=JSONResponse)
async def apply_prompt(request: PromptApplyRequest):
    """Set the system prompt on the active model session."""

    if not llm_controller.loaded_models:
        return JSONResponse({"error": "No model loaded."}, status_code=400)

    requested_name = (request.prompt_name or "").strip() or "__default__"

    if requested_name == "__default__":
        controller_state = llm_controller.get_state()
        prompt_text = controller_state.get("default_system_prompt", SYSTEM_PROMPT)
        llm_controller.set_system_prompt(prompt_text)
    else:
        try:
            prompt_text = llm_controller.apply_prompt(requested_name)
        except KeyError:
            return JSONResponse({"error": "Prompt not found."}, status_code=404)

    controller_state = llm_controller.get_state()
    message_name = "Default prompt" if requested_name == "__default__" else requested_name
    return {
        "message": f"Applied prompt: {message_name}",
        "prompt_name": requested_name,
        "prompt_text": prompt_text,
        "controller_state": controller_state,
    }


@app.post("/api/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if not llm_controller.loaded_models:
        return JSONResponse({"error": "No model loaded."}, status_code=400)

    filename = file.filename or ""
    content_type = (file.content_type or "").lower()
    suffix = Path(filename).suffix.lower()

    is_pdf = suffix == PDF_EXTENSION or content_type in PDF_MIME_TYPES
    is_text = (
        content_type.startswith(TEXT_MIME_PREFIX)
        or content_type in ADDITIONAL_TEXT_MIME_TYPES
        or suffix in TEXT_EXTENSIONS
    )

    if not filename or not (is_pdf or is_text):
        await file.close()
        return JSONResponse(
            {"error": "Only text or PDF files can be uploaded."},
            status_code=400,
        )

    temp_path = None
    try:
        data = await file.read()
        if not data:
            raise ValueError("Uploaded file is empty.")

        fd, temp_path = tempfile.mkstemp(suffix=suffix if suffix else "")
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(data)

        file_content = read_text(temp_path)
    except (UnicodeDecodeError, ValueError) as exc:
        return JSONResponse(
            {"error": f"Could not read file: {exc}"},
            status_code=400,
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"There was an error uploading the file: {exc}"},
            status_code=500,
        )
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    llm_controller.set_current_context(context=file_content)
    _, model_state = _get_active_state()
    estimated_tokens = model_state.get("pending_context_tokens", 0) if model_state else 0

    confirmation_message = (
        f"File '{file.filename}' uploaded successfully and added as context."
    )

    return JSONResponse(
        {
            "filename": file.filename,
            "content": file_content,
            "message": confirmation_message,
            "estimated_tokens": estimated_tokens,
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
