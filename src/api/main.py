import uvicorn
import os
import sys
from typing import Optional
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.llm_model import LlmModel
from src.config import SYSTEM_PROMPT, MODELS_CONFIG, MODEL_NAMES

# --- FastAPI App Initialization ---
app = FastAPI()
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory="src/api/templates")

# --- Global variable for the model ---
llm_model: Optional[LlmModel] = None
selected_model_name: Optional[str] = None

def initialize_model(model_name: str) -> LlmModel:
    """Initializes the LLM model based on the selected name."""
    global llm_model, selected_model_name
    chosen_model = MODELS_CONFIG.get(model_name, {})
    model_filename = chosen_model.get("model_filename")
    chat_format = chosen_model.get("chat_format")
    if not model_filename:
        raise ValueError("Model filename not found in config.")

    model_instance = LlmModel(
        model_path=model_filename,
        model_name=model_name,
        chat_format=chat_format,
        system_prompt=SYSTEM_PROMPT
    )

    llm_model = model_instance
    selected_model_name = model_instance.model_name
    return model_instance


# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models", response_class=JSONResponse)
async def get_models():
    """Return available model names and the currently selected model."""
    current_model_config = llm_model.get_model_config() if llm_model else None
    return {
        "models": MODEL_NAMES,
        "models_config": MODELS_CONFIG,
        "selected_model": selected_model_name,
        "current_model_config": current_model_config,
        "is_model_loaded": llm_model is not None
    }

from src.api.schemas import ChatRequest, ChatResponse, ModelSwitchRequest

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    if llm_model is None:
        return JSONResponse({"error": "No model loaded."}, status_code=400)

    response = llm_model.send_prompt(chat_request.prompt)
    usage = llm_model.get_last_interaction_usage()
    chat_usage = llm_model.get_chat_usage()
    return ChatResponse(
        response=response or "",
        usage=usage,
        chat_usage=chat_usage
    )

@app.post("/api/clear")
async def clear_history():
    if llm_model is None:
        return JSONResponse({"error": "No model loaded."}, status_code=400)

    llm_model.clear_chat_history()
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

    return JSONResponse({
        "message": f"Loaded model: {model_instance.model_name}",
        "model_config": model_instance.get_model_config(),
        "session_cleared": True
    })


@app.post("/api/load_model")
async def load_model(switch_request: ModelSwitchRequest):
    return _load_model(switch_request.model_name)


@app.post("/api/switch_model")
async def switch_model(switch_request: ModelSwitchRequest):
    return _load_model(switch_request.model_name)

@app.post("/api/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if llm_model is None:
        return JSONResponse({"error": "No model loaded."}, status_code=400)

    try:
        contents = await file.read()
        file_content = contents.decode("utf-8")
    except Exception as e:
        return JSONResponse(content={"error": f"There was an error uploading the file: {e}"}, status_code=500)
    finally:
        await file.close()

    llm_model.set_current_context(context=file_content)
    estimated_tokens = llm_model.get_pending_context_tokens()

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
