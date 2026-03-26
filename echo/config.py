from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(slots=True)
class Settings:
    profile: str = "local"
    strict_profile: bool = False
    model: str = "qwen2.5-coder:7b-oh"
    backend: str = "ollama"
    local_model: str = "qwen2.5-coder:7b-oh"
    local_backend: str = "ollama"
    balanced_model: str = "qwen2.5-coder:7b"
    balanced_backend: str = "ollama"
    deep_model: str = "gpt-4.1"
    deep_backend: str = "openai-compatible"
    fallback_model: str = "gpt-4.1-mini"
    fallback_backend: str = "openai-compatible"
    ollama_url: str = "http://localhost:11434"
    openai_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    backend_timeout: int = 120
    ollama_keep_alive: str = "10m"
    backend_preflight_timeout: int = 8
    backend_slow_threshold_ms: int = 45000
    backend_failure_threshold: int = 2
    max_steps: int = 8
    shell_timeout: int = 20
    context_message_limit: int = 18
    context_char_limit: int = 18000
    allow_write: bool = False
    allow_shell: bool = False
    theme: str = "dark"
    auto_verify: bool = True
    context_file_limit: int = 6
    snippet_line_limit: int = 80

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            profile=os.getenv("ECHO_PROFILE", "local"),
            strict_profile=os.getenv("ECHO_STRICT_PROFILE", "0").lower() in {"1", "true", "yes", "on"},
            model=os.getenv("ECHO_MODEL", os.getenv("MINI_AGENT_MODEL", "qwen2.5-coder:7b-oh")),
            backend=os.getenv("ECHO_BACKEND", "ollama"),
            local_model=os.getenv("ECHO_LOCAL_MODEL", "qwen2.5-coder:7b-oh"),
            local_backend=os.getenv("ECHO_LOCAL_BACKEND", "ollama"),
            balanced_model=os.getenv("ECHO_BALANCED_MODEL", os.getenv("ECHO_MODEL", "qwen2.5-coder:7b")),
            balanced_backend=os.getenv("ECHO_BALANCED_BACKEND", os.getenv("ECHO_BACKEND", "ollama")),
            deep_model=os.getenv("ECHO_DEEP_MODEL", "gpt-4.1"),
            deep_backend=os.getenv("ECHO_DEEP_BACKEND", "openai-compatible"),
            fallback_model=os.getenv("ECHO_FALLBACK_MODEL", os.getenv("ECHO_DEEP_MODEL", "gpt-4.1-mini")),
            fallback_backend=os.getenv("ECHO_FALLBACK_BACKEND", os.getenv("ECHO_DEEP_BACKEND", "openai-compatible")),
            ollama_url=os.getenv("ECHO_OLLAMA_URL", os.getenv("MINI_AGENT_OLLAMA_URL", "http://localhost:11434")),
            openai_url=os.getenv("ECHO_OPENAI_URL", "https://api.openai.com/v1"),
            openai_api_key=os.getenv("ECHO_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            backend_timeout=int(os.getenv("ECHO_BACKEND_TIMEOUT", "120")),
            ollama_keep_alive=os.getenv("ECHO_OLLAMA_KEEP_ALIVE", "10m"),
            backend_preflight_timeout=int(os.getenv("ECHO_BACKEND_PREFLIGHT_TIMEOUT", "8")),
            backend_slow_threshold_ms=int(os.getenv("ECHO_BACKEND_SLOW_THRESHOLD_MS", "45000")),
            backend_failure_threshold=int(os.getenv("ECHO_BACKEND_FAILURE_THRESHOLD", "2")),
            max_steps=int(os.getenv("ECHO_MAX_STEPS", os.getenv("MINI_AGENT_MAX_STEPS", "8"))),
            shell_timeout=int(os.getenv("ECHO_SHELL_TIMEOUT", os.getenv("MINI_AGENT_SHELL_TIMEOUT", "20"))),
            context_message_limit=int(os.getenv("ECHO_CONTEXT_MESSAGE_LIMIT", "18")),
            context_char_limit=int(os.getenv("ECHO_CONTEXT_CHAR_LIMIT", "18000")),
            allow_write=os.getenv("ECHO_ALLOW_WRITE", "0").lower() in {"1", "true", "yes", "on"},
            allow_shell=os.getenv("ECHO_ALLOW_SHELL", "0").lower() in {"1", "true", "yes", "on"},
            theme=os.getenv("ECHO_THEME", "dark"),
            auto_verify=os.getenv("ECHO_AUTO_VERIFY", "1").lower() in {"1", "true", "yes", "on"},
            context_file_limit=int(os.getenv("ECHO_CONTEXT_FILE_LIMIT", "6")),
            snippet_line_limit=int(os.getenv("ECHO_SNIPPET_LINE_LIMIT", "80")),
        )

    def profile_spec(self, profile: str) -> tuple[str, str]:
        profile = profile.lower()
        if profile == "local":
            return self.local_backend, self.local_model
        if profile == "balanced":
            return self.balanced_backend, self.balanced_model
        if profile == "deep":
            return self.deep_backend, self.deep_model
        return self.backend, self.model

    def fallback_spec(self) -> tuple[str, str]:
        return self.fallback_backend, self.fallback_model


def echo_dir(project_root: Path) -> Path:
    return project_root / ".echo"
