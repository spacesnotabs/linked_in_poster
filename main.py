import json
from llm_model import LlmModel

def read_model_config(config_path="model_config.json") -> dict:
    """Reads the model configuration from a JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def select_model_menu(config) -> str:
    """Displays a menu for model selection and returns the selected model's config."""
    model_names = list(config.keys())
    print("Select model to use:")
    for idx, name in enumerate(model_names, 1):
        print(f"{idx}. {name}")
    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(model_names):
                selected_name = model_names[idx]
                return selected_name
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
    # get model configuration
    config = read_model_config()
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    models = config.get("model_configs")

    # ask user to choose model and set up the model object
    model_name = select_model_menu(models)
    chosen_model = models.get(model_name, {})
    model_filename = chosen_model.get("model_filename")
    if not model_filename:
        raise ValueError("Model filename not found in config.")

    if model_filename == "DUMMY":
        model_path = "DUMMY"
    else:
        model_path = "models/" + model_filename.split("models/")[-1] if "models/" not in model_filename else model_filename
    model = LlmModel(model_path=model_path, model_name=model_name, system_prompt=system_prompt)

    # begin the chat loop
    chat_loop(model)

if __name__ == "__main__":
    main()