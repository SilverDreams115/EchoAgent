from __future__ import annotations

from echo.types import SessionState


def build_session_summary(session: SessionState, final_answer: str) -> str:
    lines = [
        f"# Echo session {session.id}",
        "",
        f"Repo: {session.repo_root}",
        f"Modo: {session.mode}",
        f"Modelo: {session.model}",
        f"Objetivo: {session.objective or session.user_prompt}",
        "",
        "## Estado operativo",
        session.operational_summary or "Sin resumen operativo todavía.",
        "",
        "## Restricciones",
    ]
    lines.extend(f"- {item}" for item in (session.restrictions[-12:] or ["none"]))
    lines += ["", "## Working set"]
    lines.extend(f"- {item}" for item in (session.working_set[-12:] or ["none"]))
    lines += [
        "",
        "## Prompt",
        session.user_prompt,
        "",
        "## Archivos en foco",
    ]
    lines.extend(f"- {path}" for path in (session.focus_files[-12:] or ["none"]))
    lines += ["", "## Tool calls"]
    if not session.tool_calls:
        lines.append("- none")
    else:
        for call in session.tool_calls[-12:]:
            lines.append(f"- {call.tool}: {call.result_preview[:160]}")
    lines += ["", "## Decisions"]
    lines.extend(f"- {item}" for item in (session.decisions[-12:] or ["none"]))
    lines += ["", "## Findings"]
    lines.extend(f"- {item}" for item in (session.findings[-12:] or ["none"]))
    lines += ["", "## Pending"]
    lines.extend(f"- {item}" for item in (session.pending[-12:] or ["none"]))
    lines += ["", "## Errors"]
    lines.extend(f"- {item}" for item in (session.errors[-12:] or ["none"]))
    lines += ["", "## Grounding"]
    if not session.grounding_report:
        lines.append("- none")
    else:
        for key, value in session.grounding_report.items():
            lines.append(f"- {key}: {value}")
    lines += ["", "## Validación"]
    lines.extend(f"- {item}" for item in (session.validation[-8:] or ["none"]))
    lines += ["", "## Final answer", final_answer]
    return "\n".join(lines)
