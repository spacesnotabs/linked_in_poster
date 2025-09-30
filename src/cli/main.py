import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.llm_model import LlmModel
from src.config import SYSTEM_PROMPT, MODELS_CONFIG, MODEL_NAMES

def select_model_menu() -> str:
    """Displays a menu for model selection and returns the selected model's name."""
    print("Select model to use:")
    for idx, name in enumerate(MODEL_NAMES, 1):
        print(f"{idx}. {name}")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(MODEL_NAMES):
                return MODEL_NAMES[idx]
        print("Invalid selection. Please try again.")

def chat_loop(model) -> None:
    """Handles the chat interaction loop."""
    while True:
        user_input = input("User: ").strip()
        if user_input == 'exit':
            break
        chat_response = model.send_prompt(prompt=user_input)
        if chat_response is not None:
            print(f"Chat: {chat_response}")
        else:
            print("I refuse to answer!")

def main():
    # ask user to choose model and set up the model object
    model_name = select_model_menu()
    chosen_model = MODELS_CONFIG.get(model_name, {})
    model_filename = chosen_model.get("model_filename")
    if not model_filename:
        raise ValueError("Model filename not found in config.")

    model = LlmModel(
        model_path=model_filename,
        model_name=model_name,
        system_prompt=SYSTEM_PROMPT
    )

    # begin the chat loop
    chat_loop(model)

if __name__ == "__main__":
    main()