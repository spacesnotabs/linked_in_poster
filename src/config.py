import json
import os

DEFAULT_CONFIG = {
    "system_prompt": "You are a helpful assistant.",
    "model_configs": {
        "Dummy Model": {
            "model_filename": "DUMMY",
            "api_key": None
        }
    }
}

def get_config(config_path="model_config.json") -> dict:
    """
    Reads the model configuration from a JSON file.
    If the file doesn't exist, it creates a default one.
    """
    if not os.path.exists(config_path):
        print(f"Configuration file not found at {config_path}. Creating a default one.")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load configuration at module import
APP_CONFIG = get_config()
SYSTEM_PROMPT = APP_CONFIG.get("system_prompt", "You are a helpful assistant.")
MODELS_CONFIG = APP_CONFIG.get("model_configs", {})
MODEL_NAMES = list(MODELS_CONFIG.keys())