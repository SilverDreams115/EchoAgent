from __future__ import annotations

from echo.types import SessionState


def build_session_summary(session: SessionState, final_answer: str) -> str:
    working = session.working_memory
    episodic = session.episodic_memory
    operational = session.operational_memory
    def render_stage(stage) -> str:
        return f"{stage.stage_id}: {stage.status}: {stage.result or 'none'}"

    lines = [
        f"# Echo session {session.id}",
        "",
        f"Repo: {session.repo_root}",
        f"Modo: {session.mode}",
        f"Modelo: {session.model}",
        f"Objetivo: {working.objective or session.objective or session.user_prompt}",
        f"Etapa actual: {working.current_stage_id or session.current_stage_id or 'none'}",
        "",
        "## Working Memory",
        f"- active_files: {', '.join(working.active_files) or 'none'}",
        f"- recent_tools: {'; '.join(working.recent_tools) or 'none'}",
        f"- recent_evidence: {'; '.join(working.recent_evidence) or 'none'}",
        f"- validation_strategy: {working.validation_strategy}",
        "",
        "## Episodic Memory",
        f"- decisions: {'; '.join(episodic.decisions[-6:]) or 'none'}",
        f"- errors: {'; '.join(episodic.errors[-6:]) or 'none'}",
        f"- retries: {'; '.join(episodic.retries[-4:]) or 'none'}",
        f"- replans: {'; '.join(episodic.replans[-4:]) or 'none'}",
        f"- validations: {'; '.join(episodic.validations[-4:]) or 'none'}",
        f"- changes: {'; '.join(episodic.changes[-6:]) or 'none'}",
        "",
        "## Operational Memory",
        operational.summary or session.operational_summary or "Sin resumen operativo todavía.",
        "",
        "## Plan Stages",
    ]
    lines.extend(
        f"- {render_stage(stage)}"
        for stage in (session.plan_stages[-8:] or [])
    )
    if not session.plan_stages:
        lines.append("- none")
    lines += [
        "",
        "## Stage Progress",
    ]
    lines.extend(f"- {item}" for item in (operational.stage_progress[-8:] or ["none"]))
    lines += ["", "## Confirmed Facts"]
    lines.extend(f"- {item}" for item in (operational.confirmed_facts[-8:] or ["none"]))
    lines += ["", "## Pending"]
    lines.extend(f"- {item}" for item in (operational.pending[-8:] or ["none"]))
    lines += ["", "## Final answer", final_answer]
    return "\n".join(lines)
