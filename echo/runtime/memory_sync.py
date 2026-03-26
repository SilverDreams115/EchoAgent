from __future__ import annotations

from dataclasses import asdict

from echo.context import build_operational_snapshot
from echo.types import ColdMemory, EpisodicMemory, OperationalMemory, RunState, SessionState, WorkingMemory


def confirmed_facts(run_state: RunState, session: SessionState) -> list[str]:
    facts = list(run_state.findings or [])
    facts.extend(f"stage:{stage.stage_id}:{stage.status}" for stage in run_state.plan_stages if stage.status in {"completed", "failed", "replanned"})
    facts.extend(session.findings[-6:])
    return list(dict.fromkeys(facts))[-12:]


def sync_memory_layers(session: SessionState, run_state: RunState) -> None:
    current_stage = run_state.current_stage_id or session.current_stage_id
    recent_tools = [call.tool for call in session.tool_calls[-6:]]
    recent_evidence: list[str] = []
    for call in session.tool_calls[-6:]:
        payload = call.result_preview[:120]
        if call.arguments.get("path"):
            recent_evidence.append(str(call.arguments["path"]))
        elif payload:
            recent_evidence.append(payload)
    stage_progress = [f"{stage.stage_id}:{stage.status}:{stage.result or stage.objective}" for stage in run_state.plan_stages[-8:]]
    retry_notes = [f"retry:{stage.stage_id}:{stage.result}" for stage in run_state.plan_stages if stage.status in {"failed", "replanned"} and stage.result][-8:]
    replan_notes = [f"replan:{stage.replanned_from}->{stage.stage_id}" for stage in run_state.plan_stages if stage.replanned_from][-8:]
    facts = confirmed_facts(run_state, session)
    session.working_memory = WorkingMemory(
        objective=session.objective or run_state.objective,
        current_stage_id=current_stage,
        active_files=list(dict.fromkeys((session.working_set or session.focus_files or run_state.focus_files)[-8:])),
        recent_tools=recent_tools,
        recent_evidence=list(dict.fromkeys(recent_evidence))[-8:],
        validation_strategy=session.working_memory.validation_strategy or "unknown",
    )
    session.episodic_memory = EpisodicMemory(
        decisions=list(dict.fromkeys(session.decisions + run_state.decisions))[-12:],
        errors=list(dict.fromkeys(session.errors + run_state.errors + run_state.open_issues))[-12:],
        retries=retry_notes,
        replans=replan_notes,
        validations=list(dict.fromkeys(session.validation + run_state.validation_commands))[-8:],
        changes=list(dict.fromkeys(session.changed_files + run_state.changed_files))[-12:],
    )
    session.operational_memory = OperationalMemory(
        summary=build_operational_snapshot(
            objective=session.objective or run_state.objective,
            restrictions=session.restrictions or run_state.constraints,
            decisions=session.decisions or run_state.decisions,
            current_stage_id=current_stage,
            focus_files=session.working_set or session.focus_files,
            changed_files=session.changed_files or run_state.changed_files,
            errors=session.errors or run_state.errors,
            pending=session.pending or run_state.pending,
            validation_commands=session.validation or run_state.validation_commands,
            confirmed_facts=facts,
        ),
        confirmed_facts=facts,
        restrictions=list(dict.fromkeys(session.restrictions or run_state.constraints))[-8:],
        stage_progress=stage_progress,
        pending=list(dict.fromkeys(session.pending + run_state.pending + run_state.open_issues))[-12:],
    )
    session.cold_memory = ColdMemory(
        summary_path=session.cold_summary_path,
        session_refs=[item for item in [session.parent_session_id, session.id] if item],
    )
    run_state.working_memory = WorkingMemory(**asdict(session.working_memory))
    run_state.episodic_memory = EpisodicMemory(**asdict(session.episodic_memory))
    run_state.operational_memory = OperationalMemory(**asdict(session.operational_memory))
