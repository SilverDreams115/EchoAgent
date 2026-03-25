from .availability import BackendAvailabilityPolicy, BackendCheckResult, BackendRouteDecision, quick_health_probe, run_backend_check
from .health import backend_log_state, effective_backend_health, normalize_backend_health
from echo.config import Settings

from .openai_compatible import OpenAICompatibleBackend
from .ollama_backend import OllamaBackend


def build_backend(settings: Settings, backend_name: str | None = None, model: str | None = None):
    selected_backend = backend_name or settings.backend
    selected_model = model or settings.model
    if selected_backend == "ollama":
        return OllamaBackend(
            settings.ollama_url,
            selected_model,
            timeout=settings.backend_timeout,
            keep_alive=settings.ollama_keep_alive,
        )
    if selected_backend in {"openai", "openai-compatible"}:
        return OpenAICompatibleBackend(
            settings.openai_url,
            settings.openai_api_key,
            selected_model,
            timeout=settings.backend_timeout,
        )
    raise ValueError(f"Unsupported backend: {selected_backend}")


__all__ = [
    "BackendAvailabilityPolicy",
    "BackendCheckResult",
    "BackendRouteDecision",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "build_backend",
    "quick_health_probe",
    "run_backend_check",
    "normalize_backend_health",
    "effective_backend_health",
    "backend_log_state",
]
