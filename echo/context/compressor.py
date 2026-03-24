from __future__ import annotations

from typing import Any

from echo.runtime import ActivityBus


def compress_messages_if_needed(
    messages: list[dict[str, Any]],
    activity: ActivityBus,
    message_limit: int,
    char_limit: int,
    *,
    objective: str = "",
    restrictions: list[str] | None = None,
    decisions: list[str] | None = None,
    focus_files: list[str] | None = None,
    changed_files: list[str] | None = None,
    errors: list[str] | None = None,
    pending: list[str] | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    if not force and len(messages) <= message_limit and sum(len(str(m.get("content", ""))) for m in messages) <= char_limit:
        return messages, None

    activity.emit("Memory", "running", "Memory compressing context", f"messages={len(messages)}")
    keep_count = 4 if force else 8
    kept = messages[-keep_count:]
    previous = messages[:-keep_count]
    goals: list[str] = [objective] if objective else []
    prior_decisions: list[str] = list(decisions or [])
    changes: list[str] = list(changed_files or [])
    prior_errors: list[str] = list(errors or [])
    for item in previous[-16:]:
        role = item.get("role", "unknown")
        content = str(item.get("content", "")).strip().replace("\n", " ")
        snippet = content[:220]
        if role == "user" and snippet:
            goals.append(snippet)
        elif role == "tool" and ("error" in content.lower() or "returncode" in content):
            prior_errors.append(snippet)
        elif role == "tool":
            changes.append(snippet)
        elif role == "assistant" and snippet:
            prior_decisions.append(snippet)
    summary = "\n".join(
        [
            "Operational summary",
            f"- Current objective: {goals[-1] if goals else 'Keep executing the active task.'}",
            f"- Restrictions: {', '.join((restrictions or [])[:4]) if restrictions else 'Ground the answer in repo evidence.'}",
            f"- Decisions taken: {prior_decisions[-1] if prior_decisions else 'Use inspected repository evidence and minimal edits.'}",
            f"- Focus files: {', '.join((focus_files or [])[-6:]) if focus_files else 'No focused file set yet.'}",
            f"- Changes applied: {changes[-1] if changes else 'No material change registered yet.'}",
            f"- Errors found: {prior_errors[-1] if prior_errors else 'No blocking error captured in compressed context.'}",
            f"- Open pending items: {(pending or ['continue from the latest kept messages and preserve prior constraints.'])[-1]}",
        ]
    )
    compressed = [{"role": "system", "content": summary}] + kept
    activity.emit("Memory", "done", "Summary updated", f"kept={len(compressed)}")
    return compressed, summary
