from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
TAGGED_BLOCK = re.compile(r"<tool_call>\s*(\{.*?\}|\[.*?\])\s*</tool_call>", re.DOTALL)


def parse_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    payloads: list[Any] = []
    for pattern in (TAGGED_BLOCK, JSON_BLOCK):
        for match in pattern.finditer(text or ""):
            try:
                payloads.append(json.loads(match.group(1)))
            except Exception:
                continue
    if not payloads:
        stripped = (text or "").strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                payloads.append(json.loads(stripped))
            except Exception:
                payloads = []
    calls: list[dict[str, Any]] = []
    for payload in payloads:
        calls.extend(_normalize_payload(payload))
    return calls


def _normalize_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        calls: list[dict[str, Any]] = []
        for item in payload["tool_calls"]:
            normalized = _normalize_call(item)
            if normalized:
                calls.append(normalized)
        return calls
    if isinstance(payload, list):
        calls: list[dict[str, Any]] = []
        for item in payload:
            normalized = _normalize_call(item)
            if normalized:
                calls.append(normalized)
        return calls
    if isinstance(payload, dict):
        normalized = _normalize_call(payload)
        return [normalized] if normalized else []
    return []


def _normalize_call(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if "function" in item and isinstance(item["function"], dict):
        name = str(item["function"].get("name", "")).strip()
        arguments = item["function"].get("arguments", {}) or {}
        return {"function": {"name": name, "arguments": arguments}} if name else None
    name = str(item.get("name") or item.get("tool") or "").strip()
    arguments = item.get("arguments", {}) or {}
    if not name:
        return None
    return {"function": {"name": name, "arguments": arguments}}
