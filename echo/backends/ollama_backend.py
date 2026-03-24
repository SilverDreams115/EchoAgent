from __future__ import annotations

from typing import Any

import requests

from .errors import (
    BackendMalformedResponseError,
    BackendModelMissingError,
    BackendTimeoutError,
    BackendUnreachableError,
)


class OllamaBackend:
    def __init__(self, base_url: str, model: str, timeout: int = 120, keep_alive: str = "10m") -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.backend_name = "ollama"
        self.supports_tools = True
        self.supports_native_tools = False

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            self._raise_for_status(response)
            data = response.json()
        except requests.ReadTimeout as exc:
            raise BackendTimeoutError(
                f"Ollama no respondió dentro de {self.timeout}s al listar modelos.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        except requests.ConnectionError as exc:
            raise BackendUnreachableError(
                f"No se pudo conectar con Ollama en {self.base_url}.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        except ValueError as exc:
            raise BackendMalformedResponseError(
                "Ollama devolvió una respuesta inválida al listar modelos.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        models = data.get("models", []) or []
        return [item.get("name", "") for item in models if isinstance(item, dict) and item.get("name")]

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        if tools:
            payload["tools"] = tools
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            self._raise_for_status(response)
            data = response.json()
        except requests.ReadTimeout as exc:
            raise BackendTimeoutError(
                f"Ollama agotó el tiempo de espera de {self.timeout}s para /api/chat.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        except requests.ConnectionError as exc:
            raise BackendUnreachableError(
                f"Ollama no es alcanzable en {self.base_url}.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        except ValueError as exc:
            raise BackendMalformedResponseError(
                "Ollama devolvió JSON inválido en /api/chat.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
            raise BackendMalformedResponseError(
                "Ollama devolvió una respuesta sin campo message válido.",
                backend=self.backend_name,
                model=self.model,
            )
        return data

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if response.status_code == 404 and self.model in detail:
                raise BackendModelMissingError(
                    f"El modelo {self.model} no está disponible en Ollama.",
                    backend=self.backend_name,
                    model=self.model,
                    detail=detail,
                ) from exc
            if detail:
                raise BackendUnreachableError(
                    f"Error HTTP de Ollama: {response.status_code}.",
                    backend=self.backend_name,
                    model=self.model,
                    detail=detail,
                ) from exc
            raise BackendUnreachableError(
                f"Error HTTP de Ollama: {response.status_code}.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
