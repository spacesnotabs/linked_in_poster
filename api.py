import json
import uvicorn
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from llm_model import LlmModel

def read_model_config(config_path="model_config.json") -> dict:
    """Reads the model configuration from a JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- FastAPI App Initialization ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Model Configuration ---
config = read_model_config()
system_prompt = config.get("system_prompt", "You are a helpful assistant.")
models_config = config.get("model_configs")
model_names = list(models_config.keys())

# --- Global variable for the model ---
llm_model: LlmModel = None

def initialize_model(model_name: str):
    """Initializes the LLM model based on the selected name."""
    global llm_model
    chosen_model = models_config.get(model_name, {})
    model_filename = chosen_model.get("model_filename")
    if not model_filename:
        raise ValueError("Model filename not found in config.")

    llm_model = LlmModel(
        model_path=model_filename,
        model_name=model_name,
        system_prompt=system_prompt
    )

# Initialize with the first model in the list
initialize_model(model_names[0])


# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models", response_class=JSONResponse)
async def get_models():
    return {"models": model_names, "selected_model": llm_model.model_name}

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse({"error": "Prompt not provided"}, status_code=400)

    response = llm_model.send_prompt(prompt)
    return JSONResponse({"response": response})

@app.post("/api/clear")
async def clear_history():
    llm_model.clear_chat_history()
    return JSONResponse({"message": "Chat history cleared"})

@app.post("/api/switch_model")
async def switch_model(request: Request):
    data = await request.json()
    model_name = data.get("model_name")
    if model_name not in model_names:
        return JSONResponse({"error": "Model not found"}, status_code=404)

    initialize_model(model_name)
    return JSONResponse({"message": f"Switched to model: {model_name}"})

@app.post("/api/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # This is a placeholder for now.
    return {"filename": file.filename, "content_type": file.content_type}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)