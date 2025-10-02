import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config import MODELS_CONFIG, MODEL_NAMES, SYSTEM_PROMPT
from src.core.controller import LLMController


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


def chat_loop(controller: LLMController) -> None:
    """Handles the chat interaction loop."""
    while True:
        user_input = input("User: ").strip()
        if user_input == 'exit':
            break
        try:
            chat_response = controller.send_prompt(prompt=user_input)
        except Exception as exc:
            print(f"Error sending prompt: {exc}")
            continue
        if chat_response:
            print(f"Chat: {chat_response}")
        else:
            print("No response returned.")


def main():
    controller = LLMController(models_config=MODELS_CONFIG, system_prompt=SYSTEM_PROMPT)

    model_name = select_model_menu()
    try:
        controller.initialize_model(model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize model '{model_name}': {exc}") from exc

    chat_loop(controller)


if __name__ == "__main__":
    main()
