from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.domain.models.llm import UsageMetrics


@dataclass
class ModelSession:
    model_id: str
    system_prompt: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    pending_context: Optional[str] = None
    pending_context_tokens: int = 0
    last_response: Optional[str] = None
    last_usage: UsageMetrics = field(default_factory=UsageMetrics)
    chat_usage: UsageMetrics = field(default_factory=UsageMetrics)
    usage_history: List[UsageMetrics] = field(default_factory=list)
    chat_log_path: Optional[Path] = None


class SessionManager:
    """Maintain chat session state and logging for loaded models."""

    def __init__(self, chat_logs_dir: Path) -> None:
        self._chat_logs_dir = chat_logs_dir
        self._chat_logs_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, ModelSession] = {}

    def ensure_session(self, model_id: str, system_prompt: str) -> ModelSession:
        session = self._sessions.get(model_id)
        if session is None:
            session = ModelSession(model_id=model_id, system_prompt=system_prompt)
            self._sessions[model_id] = session
            self._start_chat_log(session)
            if system_prompt:
                message = self._create_message("system", system_prompt)
                session.chat_history.append(message)
                self._log_chat_message(session, "system", message["content"])
        return session

    def reset_session(self, model_id: str, system_prompt: str) -> ModelSession:
        session = ModelSession(model_id=model_id, system_prompt=system_prompt)
        self._sessions[model_id] = session
        self._start_chat_log(session)
        if system_prompt:
            message = self._create_message("system", system_prompt)
            session.chat_history.append(message)
            self._log_chat_message(session, "system", message["content"])
        return session

    def get_session(self, model_id: str) -> ModelSession:
        session = self._sessions.get(model_id)
        if session is None:
            raise KeyError(f"No session for model '{model_id}'")
        return session

    def append_user_prompt(self, session: ModelSession, prompt: str) -> Dict[str, str]:
        context_tokens = session.pending_context_tokens if session.pending_context else 0
        message = self._create_message("user", prompt, context=session.pending_context)
        if context_tokens:
            message["context_tokens"] = context_tokens
        session.chat_history.append(message)
        session.pending_context = None
        session.pending_context_tokens = 0
        self._log_chat_message(session, "user", message["content"])
        return message

    def append_assistant_response(self, session: ModelSession, response: str) -> Dict[str, str]:
        message = self._create_message("assistant", response)
        session.chat_history.append(message)
        session.last_response = response
        self._log_chat_message(session, "assistant", message["content"])
        return message

    def track_usage(self, session: ModelSession, usage: UsageMetrics) -> None:
        session.last_usage = usage
        session.usage_history.append(usage.clone())
        aggregate = session.chat_usage
        aggregate.prompt_tokens += usage.prompt_tokens
        aggregate.completion_tokens += usage.completion_tokens
        aggregate.total_tokens += usage.total_tokens
        aggregate.context_tokens += usage.context_tokens
        if usage.response_seconds is not None:
            aggregate.response_seconds = (aggregate.response_seconds or 0.0) + usage.response_seconds

    def clear_history(self, model_id: str, *, keep_system_prompt: bool = True) -> ModelSession:
        session = self.get_session(model_id)
        retained_prompt = session.system_prompt if keep_system_prompt else ""
        session = self.reset_session(model_id, retained_prompt)
        return session

    def set_pending_context(self, model_id: str, context: Optional[str], token_estimator) -> ModelSession:
        session = self.get_session(model_id)
        cleaned = (context or "").strip()
        if not cleaned:
            session.pending_context = None
            session.pending_context_tokens = 0
        else:
            session.pending_context = cleaned
            session.pending_context_tokens = token_estimator(cleaned)
        return session

    def _create_message(self, role: str, content: str, *, context: Optional[str] = None) -> Dict[str, str]:
        content_value = content.strip()
        if context:
            content_value = f"<context>{context}</context>\n\n{content_value}"
        return {"role": role, "content": content_value}

    def _start_chat_log(self, session: ModelSession) -> None:
        created_at = datetime.now()
        timestamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
        safe_model = self._sanitize_for_filename(session.model_id)
        filename = f"{safe_model}_{timestamp}.txt"
        session.chat_log_path = self._chat_logs_dir / filename
        header_lines = [
            f"Chat Session Started: {created_at.isoformat(timespec='seconds')}",
            f"Model: {session.model_id}",
            "",
        ]
        session.chat_log_path.write_text("\n".join(header_lines), encoding="utf-8")

    def _log_chat_message(self, session: ModelSession, role: str, content: str) -> None:
        if not content or session.chat_log_path is None:
            return
        timestamp = datetime.now().isoformat(timespec="seconds")
        heading_map = {
            "system": "System Prompt",
            "user": "User Prompt",
            "assistant": "Model Response",
        }
        label = heading_map.get(role, role.title() if role else "Message")
        header = f"[{timestamp}] {label}"
        separator = "-" * len(header)
        normalized = content.replace("\r\n", "\n").strip()
        if not normalized:
            return
        with session.chat_log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{header}\n{separator}\n{normalized}\n\n")

    @staticmethod
    def _sanitize_for_filename(value: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        sanitized = "".join("_" if char in invalid_chars else char for char in value)
        sanitized = sanitized.replace(" ", "_").strip("_")
        return sanitized or "chat"
