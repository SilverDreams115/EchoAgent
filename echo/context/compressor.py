from __future__ import annotations

from typing import Any

def build_operational_snapshot(
    *,
    objective: str = "",
    restrictions: list[str] | None = None,
    decisions: list[str] | None = None,
    current_stage_id: str = "",
    focus_files: list[str] | None = None,
    changed_files: list[str] | None = None,
    errors: list[str] | None = None,
    pending: list[str] | None = None,
    validation_commands: list[str] | None = None,
    confirmed_facts: list[str] | None = None,
) -> str:
    lines = [
        "Operational summary",
        f"- Current objective: {objective or 'Keep executing the active task.'}",
        f"- Current stage: {current_stage_id or 'none'}",
        f"- Restrictions: {', '.join((restrictions or [])[:6]) if restrictions else 'Ground the answer in repo evidence.'}",
        f"- Decisions taken: {'; '.join((decisions or [])[-3:]) if decisions else 'Use inspected repository evidence and minimal edits.'}",
        f"- Active files: {', '.join((focus_files or [])[-8:]) if focus_files else 'No focused file set yet.'}",
        f"- Changes applied: {', '.join((changed_files or [])[-6:]) if changed_files else 'No material change registered yet.'}",
        f"- Errors found: {'; '.join((errors or [])[-4:]) if errors else 'No blocking error captured.'}",
        f"- Validation: {'; '.join((validation_commands or [])[-4:]) if validation_commands else 'No validation executed yet.'}",
        f"- Confirmed facts: {'; '.join((confirmed_facts or [])[-4:]) if confirmed_facts else 'No consolidated facts yet.'}",
        f"- Open pending items: {'; '.join((pending or [])[-4:]) if pending else 'continue from the latest kept messages and preserve prior constraints.'}",
    ]
    return "\n".join(lines)


def compress_messages_if_needed(
    messages: list[dict[str, Any]],
    activity: Any,
    message_limit: int,
    char_limit: int,
    *,
    objective: str = "",
    restrictions: list[str] | None = None,
    decisions: list[str] | None = None,
    current_stage_id: str = "",
    focus_files: list[str] | None = None,
    changed_files: list[str] | None = None,
    errors: list[str] | None = None,
    pending: list[str] | None = None,
    validation_commands: list[str] | None = None,
    confirmed_facts: list[str] | None = None,
    force: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    if not force and len(messages) <= message_limit and sum(len(str(m.get("content", ""))) for m in messages) <= char_limit:
        return messages, None

    activity.emit("Memory", "running", "Memory compressing context", f"messages={len(messages)}")
    keep_count = 4 if force else 8
    kept = messages[-keep_count:]
    summary = build_operational_snapshot(
        objective=objective,
        restrictions=restrictions,
        decisions=decisions,
        current_stage_id=current_stage_id,
        focus_files=focus_files,
        changed_files=changed_files,
        errors=errors,
        pending=pending,
        validation_commands=validation_commands,
        confirmed_facts=confirmed_facts,
    )
    compressed = [{"role": "system", "content": summary}] + kept
    activity.emit("Memory", "done", "Summary updated", f"kept={len(compressed)}")
    return compressed, summary
