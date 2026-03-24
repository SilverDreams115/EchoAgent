from __future__ import annotations

from typing import Any

import requests

from .errors import (
    BackendMalformedResponseError,
    BackendModelMissingError,
    BackendTimeoutError,
    BackendUnreachableError,
)


class OpenAICompatibleBackend:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.backend_name = "openai-compatible"
        self.supports_tools = True
        self.supports_native_tools = True

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout)
            self._raise_for_status(response)
            data = response.json()
        except requests.ReadTimeout as exc:
            raise BackendTimeoutError(
                f"El backend OpenAI-compatible no respondió dentro de {self.timeout}s al listar modelos.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        except requests.ConnectionError as exc:
            raise BackendUnreachableError(
                f"No se pudo conectar con {self.base_url}.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        except ValueError as exc:
            raise BackendMalformedResponseError(
                "El backend OpenAI-compatible devolvió JSON inválido al listar modelos.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        items = data.get("data", []) or []
        return [item.get("id", "") for item in items if isinstance(item, dict) and item.get("id")]

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            self._raise_for_status(response)
            data = response.json()
        except requests.ReadTimeout as exc:
            raise BackendTimeoutError(
                f"El backend OpenAI-compatible agotó el tiempo de espera de {self.timeout}s para /chat/completions.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        except requests.ConnectionError as exc:
            raise BackendUnreachableError(
                f"No se pudo conectar con {self.base_url}.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        except ValueError as exc:
            raise BackendMalformedResponseError(
                "El backend OpenAI-compatible devolvió JSON inválido en /chat/completions.",
                backend=self.backend_name,
                model=self.model,
            ) from exc
        choices = data.get("choices", []) or []
        if not choices:
            raise BackendMalformedResponseError(
                "El backend OpenAI-compatible respondió sin choices.",
                backend=self.backend_name,
                model=self.model,
            )
        message = choices[0].get("message", {}) or {}
        if not isinstance(message, dict):
            raise BackendMalformedResponseError(
                "El backend OpenAI-compatible devolvió un message inválido.",
                backend=self.backend_name,
                model=self.model,
            )
        return {"message": message}

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if response.status_code == 404 and self.model in detail:
                raise BackendModelMissingError(
                    f"El modelo {self.model} no está disponible en el backend OpenAI-compatible.",
                    backend=self.backend_name,
                    model=self.model,
                    detail=detail,
                ) from exc
            if response.status_code == 400 and "model" in detail.lower():
                raise BackendModelMissingError(
                    f"El modelo {self.model} fue rechazado por el backend OpenAI-compatible.",
                    backend=self.backend_name,
                    model=self.model,
                    detail=detail,
                ) from exc
            raise BackendUnreachableError(
                f"Error HTTP del backend OpenAI-compatible: {response.status_code}.",
                backend=self.backend_name,
                model=self.model,
                detail=detail,
            ) from exc
