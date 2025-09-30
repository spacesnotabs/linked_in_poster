import uvicorn
import os
import sys
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.llm_model import LlmModel
from src.config import SYSTEM_PROMPT, MODELS_CONFIG, MODEL_NAMES

# --- FastAPI App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory="src/api/templates")

# --- Global variable for the model ---
llm_model: LlmModel = None

def initialize_model(model_name: str):
    """Initializes the LLM model based on the selected name."""
    global llm_model
    chosen_model = MODELS_CONFIG.get(model_name, {})
    model_filename = chosen_model.get("model_filename")
    chat_format = chosen_model.get("chat_format")
    if not model_filename:
        raise ValueError("Model filename not found in config.")

    llm_model = LlmModel(
        model_path=model_filename,
        model_name=model_name,
        chat_format=chat_format,
        system_prompt=SYSTEM_PROMPT
    )

# Initialize with the first model in the list
if MODEL_NAMES:
    initialize_model(MODEL_NAMES[0])


# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models", response_class=JSONResponse)
async def get_models():
    return {"models": MODELS_CONFIG, "selected_model": llm_model.model_name}

from src.api.schemas import ChatRequest, ChatResponse, ModelSwitchRequest

@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    response = llm_model.send_prompt(chat_request.prompt)
    return ChatResponse(response=response)

@app.post("/api/clear")
async def clear_history():
    llm_model.clear_chat_history()
    return JSONResponse({"message": "Chat history cleared"})

@app.post("/api/switch_model")
async def switch_model(switch_request: ModelSwitchRequest):
    model_name = switch_request.model_name
    if model_name not in MODEL_NAMES:
        return JSONResponse({"error": "Model not found"}, status_code=404)

    initialize_model(model_name)
    return JSONResponse({"message": f"Switched to model: {model_name}"})

@app.post("/api/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_content = contents.decode("utf-8")
    except Exception as e:
        return JSONResponse(content={"error": f"There was an error uploading the file: {e}"}, status_code=500)
    finally:
        await file.close()

    llm_model.set_current_context(context=file_content)

    confirmation_message = (
        f"File '{file.filename}' uploaded successfully and added as context."
    )

    return JSONResponse(
        {
            "filename": file.filename,
            "content": file_content,
            "message": confirmation_message,
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
