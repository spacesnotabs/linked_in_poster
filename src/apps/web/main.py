from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.domain.models.llm import UsageMetrics
from src.infrastructure.config.settings import get_settings
from src.services.llm import LLMService
from src.services.rag import ChunkJobManager, ChunkingService
from src.utils.utils import read_text

from .schemas import (
    ChatRequest,
    ChatResponse,
    ChunkJobRequest,
    ModelSettingsPayload,
    ModelSwitchRequest,
    PromptApplyRequest,
)

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".ini",
    ".cfg",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".xml",
    ".tex",
    ".pdf",
}
ADDITIONAL_TEXT_MIME_TYPES = {"application/json"}
PDF_EXTENSION = ".pdf"
PDF_MIME_TYPES = {"application/pdf"}
TEXT_MIME_PREFIX = "text/"


settings = get_settings()
llm_service = LLMService(settings=settings)
chunking_service = ChunkingService(settings=settings)
chunk_job_manager = ChunkJobManager(chunking_service)
selected_model_name: Optional[str] = None
logger = logging.getLogger(__name__)


def _template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def _initialize_model(model_name: str):
    global selected_model_name
    runtime = llm_service.initialize_model(model_name)
    selected_model_name = runtime.model_name
    return runtime


def _get_active_state() -> Tuple[Optional[str], Dict[str, Any]]:
    state = llm_service.get_state()
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


def create_app() -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(_static_dir())), name="static")
    templates = Jinja2Templates(directory=str(_template_dir()))

    @app.get("/", response_class=HTMLResponse)
    async def read_root(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/settings", response_class=HTMLResponse)
    async def render_settings_page(request: Request):
        return templates.TemplateResponse("model_settings.html", {"request": request})

    @app.get("/chunking", response_class=HTMLResponse)
    async def render_chunking_page(request: Request):
        return templates.TemplateResponse("chunking.html", {"request": request})

    @app.get("/api/models", response_class=JSONResponse)
    async def get_models():
        active_model_id, _ = _get_active_state()
        current_model_config = (
            llm_service.describe_model_config(active_model_id) if active_model_id else None
        )
        controller_state = llm_service.get_state()
        return {
            "models": llm_service.available_models,
            "models_config": llm_service.describe_all_models(),
            "selected_model": selected_model_name,
            "current_model_config": current_model_config,
            "is_model_loaded": bool(controller_state.get("loaded_models")),
            "controller_state": controller_state,
        }

    @app.get("/api/model_settings", response_class=JSONResponse)
    async def get_model_settings(model_name: Optional[str] = None):
        resolved_name = (model_name or "").strip()
        if not resolved_name:
            models = llm_service.available_models
            resolved_name = selected_model_name or (models[0] if models else "")

        if not resolved_name:
            return JSONResponse({"error": "No models configured."}, status_code=404)
        if resolved_name not in llm_service.available_models:
            return JSONResponse({"error": "Model not found."}, status_code=404)

        settings_payload = llm_service.get_effective_runtime_config(resolved_name)
        is_loaded = resolved_name in llm_service.loaded_models
        is_active = llm_service.active_model_id == resolved_name

        current_config = None
        if is_loaded:
            current_config = llm_service.get_model(resolved_name).get_config()

        return {
            "model_name": resolved_name,
            "settings": settings_payload,
            "is_loaded": is_loaded,
            "is_active": is_active,
            "current_config": current_config,
            "requires_reload": is_loaded,
        }

    @app.post("/api/model_settings", response_class=JSONResponse)
    async def update_model_settings(payload: ModelSettingsPayload):
        model_name = payload.model_name
        overrides = {
            "n_context_size": payload.n_context_size,
            "context_size": payload.n_context_size,
            "n_threads": payload.n_threads,
            "n_threads_batch": payload.n_threads_batch,
            "temperature": payload.temperature,
            "verbose": payload.verbose,
            "use_mmap": payload.use_mmap,
            "logits_all": payload.logits_all,
        }
        sanitized = llm_service.set_runtime_overrides(model_name, overrides)
        return {
            "model_name": model_name,
            "overrides": sanitized,
            "requires_reload": model_name in llm_service.loaded_models,
        }

    @app.post("/api/models/switch", response_class=JSONResponse)
    async def switch_model(payload: ModelSwitchRequest):
        model_name = (payload.model_name or "").strip()
        if not model_name:
            logger.warning("Received model switch request without a model name.")
            return JSONResponse({"error": "Model name is required."}, status_code=400)
        if model_name not in llm_service.available_models:
            logger.warning("Model switch requested for unknown model '%s'.", model_name)
            return JSONResponse({"error": f"Model '{model_name}' not configured."}, status_code=404)

        logger.info("Attempting to load model '%s'.", model_name)
        try:
            runtime = _initialize_model(model_name)
        except Exception as exc:
            error_message = str(exc).strip() or exc.__class__.__name__
            registry_error = llm_service.get_last_error(model_name)
            error_payload = {"error": error_message}
            if registry_error and registry_error != error_message:
                error_payload["details"] = registry_error
            logger.exception("Failed to load model '%s': %s", model_name, error_message)
            return JSONResponse(error_payload, status_code=500)

        runtime_config = runtime.get_config() or {}
        logger.info(
            "Model '%s' loaded successfully with context_size=%s.",
            runtime.model_name,
            runtime_config.get("context_size") or runtime_config.get("n_context_size"),
        )

        controller_state = llm_service.get_state()
        return {
            "message": f"Model '{model_name}' loaded.",
            "model_name": runtime.model_name,
            "controller_state": controller_state,
            "model_config": llm_service.describe_model_config(runtime.model_name),
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(chat_request: ChatRequest):
        if not llm_service.loaded_models:
            return JSONResponse({"error": "No model loaded."}, status_code=400)

        try:
            response_text = llm_service.send_prompt(chat_request.prompt)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

        last_usage, chat_usage = _get_usage_snapshots()
        return ChatResponse(
            response=response_text or "",
            usage=last_usage,
            chat_usage=chat_usage,
        )

    @app.post("/api/chat/clear", response_class=JSONResponse)
    async def clear_chat():
        if not llm_service.loaded_models:
            return JSONResponse({"error": "No model loaded."}, status_code=400)

        llm_service.clear_chat_history()
        controller_state = llm_service.get_state()
        return {
            "message": "Chat history cleared.",
            "controller_state": controller_state,
        }

    @app.get("/api/prompts", response_class=JSONResponse)
    async def list_prompts():
        prompts = []
        for name in llm_service.list_prompts():
            try:
                prompt_definition = llm_service.get_prompt(name)
            except KeyError:
                continue
            prompts.append({"name": name, "definition": prompt_definition})
        controller_state = llm_service.get_state()
        default_prompt = controller_state.get("default_system_prompt", "")
        active_prompt = ""
        active_model_id = controller_state.get("active_model")
        if active_model_id:
            model_state = controller_state.get("models", {}).get(active_model_id, {})
            active_prompt = model_state.get("system_prompt", "") or ""
        return {
            "prompts": prompts,
            "default_prompt": default_prompt,
            "active_prompt": active_prompt,
        }

    @app.get("/api/prompts/{prompt_name}", response_class=JSONResponse)
    async def get_prompt_preview(prompt_name: str):
        if prompt_name == "__default__":
            controller_state = llm_service.get_state()
            return {
                "name": "__default__",
                "prompt_text": controller_state.get("default_system_prompt", ""),
                "definition": None,
            }

        try:
            prompt_definition = llm_service.get_prompt(prompt_name)
            prompt_text = llm_service.render_prompt(prompt_name)
        except KeyError:
            return JSONResponse({"error": "Prompt not found."}, status_code=404)

        return {
            "name": prompt_name,
            "prompt_text": prompt_text,
            "definition": prompt_definition,
        }

    @app.post("/api/prompts/apply", response_class=JSONResponse)
    async def apply_prompt(request: PromptApplyRequest):
        if not llm_service.loaded_models:
            return JSONResponse({"error": "No model loaded."}, status_code=400)

        requested_name = (request.prompt_name or "").strip() or "__default__"

        if requested_name == "__default__":
            controller_state = llm_service.get_state()
            prompt_text = controller_state.get("default_system_prompt", "")
            llm_service.set_system_prompt(prompt_text)
        else:
            try:
                prompt_text = llm_service.apply_prompt(requested_name)
            except KeyError:
                return JSONResponse({"error": "Prompt not found."}, status_code=404)

        controller_state = llm_service.get_state()
        message_name = "Default prompt" if requested_name == "__default__" else requested_name
        return {
            "message": f"Applied prompt: {message_name}",
            "prompt_name": requested_name,
            "prompt_text": prompt_text,
            "controller_state": controller_state,
        }

    @app.post("/api/uploadfile/")
    async def create_upload_file(file: UploadFile = File(...)):
        if not llm_service.loaded_models:
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
            return JSONResponse({"error": f"Could not read file: {exc}"}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"There was an error uploading the file: {exc}"}, status_code=500)
        finally:
            await file.close()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        active_model_id, _ = _get_active_state()
        estimated_tokens: Optional[int] = None
        if active_model_id:
            try:
                runtime = llm_service.get_model(active_model_id)
                estimated_tokens = runtime.estimate_tokens(file_content)
            except Exception:  # pragma: no cover - defensive for UI estimates
                estimated_tokens = None

        return JSONResponse(
            {
                "filename": file.filename,
                "content": file_content,
                "message": f"File '{file.filename}' attached successfully.",
                "estimated_tokens": estimated_tokens,
            }
        )

    @app.get("/api/chunking/chunkers", response_class=JSONResponse)
    async def list_chunkers():
        return {"chunkers": chunking_service.available_chunkers()}

    @app.get("/api/chunking/jobs", response_class=JSONResponse)
    async def list_chunk_jobs():
        return {"jobs": chunk_job_manager.list_jobs()}

    @app.get("/api/chunking/jobs/{job_id}", response_class=JSONResponse)
    async def get_chunk_job(job_id: str):
        snapshot = chunk_job_manager.get(job_id)
        if not snapshot:
            return JSONResponse({"error": "Job not found."}, status_code=404)
        return {"job": snapshot}

    @app.post("/api/chunking/jobs", response_class=JSONResponse)
    async def create_chunk_job(request: ChunkJobRequest):
        normalized_paths = [path.strip() for path in request.paths if path.strip()]
        if not normalized_paths:
            return JSONResponse({"error": "Provide at least one path to chunk."}, status_code=400)
        try:
            job = chunk_job_manager.start_job(
                normalized_paths,
                include=request.include,
                exclude=request.exclude,
            )
        except (ValueError, FileNotFoundError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to start chunking job: %s", exc)
            return JSONResponse({"error": "Failed to start chunking job."}, status_code=500)
        return {"job": job.snapshot()}

    @app.get("/api/chunking/database/summary", response_class=JSONResponse)
    async def get_chunk_database_summary():
        return chunking_service.database.summarize()

    @app.get("/api/chunking/database/files", response_class=JSONResponse)
    async def get_chunked_files(
        search: Optional[str] = None,
        chunker: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        return chunking_service.database.list_files(
            search=search or None,
            chunker=chunker or None,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/chunking/database/chunks", response_class=JSONResponse)
    async def get_chunked_chunks(
        file_path: Optional[str] = None,
        chunker: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ):
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        return chunking_service.database.list_chunks(
            file_path=file_path or None,
            chunker=chunker or None,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/chunking/database/chunks/{chunk_id}", response_class=JSONResponse)
    async def get_chunk_detail(chunk_id: int):
        chunk = chunking_service.database.get_chunk(chunk_id)
        if not chunk:
            return JSONResponse({"error": "Chunk not found."}, status_code=404)
        return chunk

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
