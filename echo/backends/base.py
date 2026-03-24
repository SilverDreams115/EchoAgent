from __future__ import annotations

from typing import Any, Protocol


class Backend(Protocol):
    model: str
    backend_name: str
    supports_tools: bool
    supports_native_tools: bool

    def list_models(self) -> list[str]: ...

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]: ...
