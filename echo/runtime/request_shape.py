from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RequestShape:
    message_count: int
    total_chars: int
    system_messages: int
    user_messages: int
    assistant_messages: int
    tool_messages: int
    includes_repo_map: bool
    includes_focus_snippets: bool
    compressed_context: bool
    resumed_context: bool


def describe_request_shape(messages: list[dict[str, Any]]) -> RequestShape:
    total_chars = 0
    system_messages = 0
    user_messages = 0
    assistant_messages = 0
    tool_messages = 0
    includes_repo_map = False
    includes_focus_snippets = False
    compressed_context = False
    resumed_context = False

    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", "") or "")
        total_chars += len(content)
        if role == "system":
            system_messages += 1
        elif role == "user":
            user_messages += 1
        elif role == "assistant":
            assistant_messages += 1
        elif role == "tool":
            tool_messages += 1
        if "Repo map:\n" in content:
            includes_repo_map = True
        if "Focus snippets:\n\n" in content:
            includes_focus_snippets = True
        if content.startswith("Operational summary\n"):
            compressed_context = True
        if content.startswith("Resumed session from "):
            resumed_context = True

    return RequestShape(
        message_count=len(messages),
        total_chars=total_chars,
        system_messages=system_messages,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        tool_messages=tool_messages,
        includes_repo_map=includes_repo_map,
        includes_focus_snippets=includes_focus_snippets,
        compressed_context=compressed_context,
        resumed_context=resumed_context,
    )
