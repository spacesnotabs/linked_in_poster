# Linked In Poster #
This tool is meant to help someone create LinkedIn posts in their own style using either local or online LLMs.

This project has just begun so as of this writing, all it does is fire up a specific model on your machine and give you a command-line chat interface!  Over time, I'll use that framework to build what this tool is intended to be.

## Model Configuration

Create a `config/model_config.json` file in the project directory with the following structure:

```json
{
  "system_prompt": "You are a helpful assistant.",
  "Mistral 7B Instruct": {
    "model_filename": "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "api_key": null
  }
  // Add more models as needed
}
```

- `system_prompt`: (string) The default system prompt for all models.
- Each model section (e.g., `"Mistral 7B Instruct"`) contains:
  - `model_filename`: (string) Path to the model file.
  - `api_key`: (string or null) API key if required by the model.

## Running the Application

To start the application, run:

```sh
python main.py
```

You will be prompted to select a model from the console menu and then you can chat away!

_AI Disclaimer: I LOVE using AI to help me write code.  I will be the first to admit that.  It feels like magic and can be extremely helpful!  However, for this project, I wanted to learn how to work with both local LLMs and LLM APIs and the best way for me to learn is to write the code myself.  I am using AI for pieces of the code which are boilerplate or I'm already very familiar with writing.  But, for the model implementation I'm doing the work on my own to ensure I grasp and retain the concepts._
