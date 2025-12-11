from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    from pydantic import BaseModel, BaseSettings, Field
except ImportError:  # pragma: no cover - lightweight fallback
    class BaseModel:  # type: ignore
        def dict(self, *_, **__) -> Dict[str, Any]:
            return self.__dict__.copy()

    class BaseSettings:  # type: ignore
        def __init__(self, **kwargs: Any) -> None:
            for name, value in self.__class__.__dict__.items():
                if name.startswith("_") or callable(value):
                    continue
                setattr(self, name, kwargs.get(name, value))

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore
        return default

DEFAULT_CONFIG: Dict[str, Any] = {
    "system_prompt": "You are a helpful assistant.",
    "model_configs": {
        "Dummy Model": {
            "model_filename": "DUMMY",
            "chat_format": "dummy",
            "context_window": 2048,
            "api_key": None,
        }
    },
}


class ModelConfig(BaseModel):
    model_filename: str
    chat_format: str | None = None
    context_window: int | None = Field(default=32768, alias="context_size")
    api_key: str | None = None
    n_gpu_layers: int | None = -1


class AppSettings(BaseSettings):
    """Application-wide configuration sourced from environment variables."""

    model_config_path: Path = Field(
        default=Path("config/model_config.json"),
        env="MODEL_CONFIG_PATH",
    )
    prompts_path: Path = Field(
        default=Path("config/prompts.json"),
        env="PROMPTS_PATH",
    )
    data_dir: Path = Field(
        default=Path("data"),
        env="DATA_DIR",
    )
    chat_logs_dir: Path = Field(
        default=Path("data/chats"),
        env="CHAT_LOGS_DIR",
    )
    database_url: str = Field(
        default="sqlite:///data/app.db",
        env="DATABASE_URL",
    )

    class Config:
        env_file = ".env"
        env_prefix = "LINKEDIN_POSTER_"
        case_sensitive = False

    def ensure_runtime_dirs(self) -> None:
        """Create runtime directories required by the application."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chat_logs_dir.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///"):
            sqlite_path = Path(self.database_url.replace("sqlite:///", ""))
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def load_model_catalog(path: Path) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Load the JSON model configuration file, creating a default sample if it is missing.

    Returns a tuple of (system_prompt, model_configs).
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=4), encoding="utf-8")
        return (
            DEFAULT_CONFIG["system_prompt"],
            DEFAULT_CONFIG["model_configs"],
        )

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    system_prompt = payload.get("system_prompt") or DEFAULT_CONFIG["system_prompt"]
    model_configs = payload.get("model_configs")
    if not isinstance(model_configs, dict) or not model_configs:
        raise ValueError("Model configuration file must contain a non-empty 'model_configs' object.")

    return system_prompt, model_configs


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached settings instance."""
    settings = AppSettings()
    settings.ensure_runtime_dirs()
    return settings
