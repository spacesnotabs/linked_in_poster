from llm_model import LlmModel

if __name__ == "__main__":
    print("loading model")
    model = LlmModel(model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

    while True:
        user_input = input("User: ").strip()
        if user_input == 'exit':
            break

        print(model.send_prompt(prompt=user_input))