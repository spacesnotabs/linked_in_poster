from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.infrastructure.config.settings import AppSettings
from src.services.llm import LLMService


class LLMServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = TemporaryDirectory()
        base_path = Path(self._temp_dir.name)
        config_dir = base_path / "config"
        data_dir = base_path / "data"
        chat_logs_dir = data_dir / "chats"
        config_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = "Mini Dummy"
        self.config_path = config_dir / "model_config.json"
        self.prompts_path = config_dir / "prompts.json"

        model_config = {
            "system_prompt": "You are a delightful assistant.",
            "model_configs": {
                self.model_name: {
                    "model_filename": "DUMMY",
                    "chat_format": "dummy",
                    "context_window": 128,
                }
            },
        }
        self.config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

        prompts_payload = [
            {
                "name": "test_prompt",
                "task": "Respond enthusiastically.",
                "prompt": "Always reply with 'HELLO!'",
            }
        ]
        self.prompts_path.write_text(json.dumps(prompts_payload, indent=2), encoding="utf-8")

        self.settings = AppSettings(
            model_config_path=self.config_path,
            prompts_path=self.prompts_path,
            data_dir=data_dir,
            chat_logs_dir=chat_logs_dir,
        )
        self.service = LLMService(settings=self.settings)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def test_initialize_and_send_prompt_with_dummy_model(self) -> None:
        runtime = self.service.initialize_model(self.model_name)
        self.assertEqual(runtime.model_name, self.model_name)
        self.assertIn(self.model_name, self.service.loaded_models)

        response = self.service.send_prompt("Hello there!")
        self.assertIsNotNone(response)
        self.assertIn(self.model_name, response)

        state = self.service.get_state()
        self.assertEqual(state["active_model"], self.model_name)
        self.assertGreaterEqual(len(state["chat"]), 2)

        chat_logs_dir = self.settings.chat_logs_dir
        logs = list(chat_logs_dir.glob("*.txt"))
        self.assertTrue(logs, "Expected chat logs to be created.")

    def test_apply_prompt_updates_system_prompt(self) -> None:
        self.service.initialize_model(self.model_name)
        rendered = self.service.apply_prompt("test_prompt", model_id=self.model_name)
        self.assertIn("Respond enthusiastically.", rendered)

        active_prompt = self.service.get_active_system_prompt()
        self.assertIn("Respond enthusiastically.", active_prompt)


if __name__ == "__main__":
    unittest.main()

