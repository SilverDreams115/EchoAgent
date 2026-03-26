from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from echo.types import PlanStage, RunState, SessionState


def _grounding_report_payload(run_state: RunState) -> dict[str, object]:
    return {
        "grounded_file_count": run_state.grounding_report.grounded_file_count,
        "grounded_symbol_count": run_state.grounding_report.grounded_symbol_count,
        "evidence_usage": run_state.grounding_report.evidence_usage,
        "genericity_score": run_state.grounding_report.genericity_score,
        "useful": run_state.grounding_report.useful,
        "claim_types": run_state.grounding_report.claim_types,
        "unsupported_files": run_state.grounding_report.unsupported_files,
        "unsupported_symbols": run_state.grounding_report.unsupported_symbols,
        "unsupported_commands": run_state.grounding_report.unsupported_commands,
        "unsupported_changes": run_state.grounding_report.unsupported_changes,
        "contradiction_flags": run_state.grounding_report.contradiction_flags,
        "validation_strategy_match": run_state.grounding_report.validation_strategy_match,
        "validation_strategy": run_state.grounding_report.validation_strategy,
        "speculation_flags": run_state.grounding_report.speculation_flags,
        "valid": run_state.grounding_report.valid,
        "reason": run_state.grounding_report.reason,
    }


def finalize_session(
    session: SessionState,
    run_state: RunState,
    *,
    final_answer: str,
    store,
    activity,
    update_stage,
    sync_memory_layers,
    build_session_summary,
    write_runtime_artifact,
    runtime_trace_payload,
) -> tuple[str, Path, SessionState]:
    session.activity = activity.recent(120)
    session.objective = session.objective or run_state.objective
    session.restrictions = list(dict.fromkeys(session.restrictions or run_state.constraints))
    session.focus_files = list(dict.fromkeys(session.focus_files + run_state.changed_files))
    session.working_set = list(dict.fromkeys(session.working_set + session.focus_files + run_state.changed_files))
    session.changed_files = list(dict.fromkeys(session.changed_files + run_state.changed_files))
    session.decisions = list(dict.fromkeys(session.decisions + run_state.decisions))
    session.findings = list(dict.fromkeys(session.findings + run_state.findings + run_state.inspected_files))
    session.pending = list(dict.fromkeys(session.pending + run_state.pending + run_state.open_issues))
    session.errors = list(dict.fromkeys(session.errors + run_state.errors + run_state.open_issues))

    close_stage = next((stage for stage in run_state.plan_stages if stage.stage_id == "close"), None)
    if close_stage is not None:
        close_status = "failed" if session.degraded_reason else "completed"
        close_result = session.degraded_reason or "Final answer and stage summary recorded."
        update_stage(session, run_state, "close", status=close_status, result=close_result)

    sync_memory_layers(session, run_state)
    if not session.grounding_report and run_state.grounding_report.reason != "ok":
        session.grounding_report = _grounding_report_payload(run_state)

    session.routing = asdict(run_state.routing)
    session.plan_stages = [PlanStage(**asdict(item)) for item in run_state.plan_stages]
    session.current_stage_id = run_state.current_stage_id
    session.health = {
        "effective": asdict(run_state.backend_health),
        "fresh": asdict(run_state.fresh_backend_health),
        "cached": asdict(store.read_backend_health()),
    }
    session.retry_count = run_state.retry_count
    session.compression_count = run_state.compression_count
    session.artifacts = [asdict(item) for item in run_state.artifacts]
    session.runtime_trace = runtime_trace_payload(run_state)
    session.summary = build_session_summary(session, final_answer)
    session.operational_summary = session.operational_memory.summary or session.operational_summary or session.summary

    write_runtime_artifact(
        run_state,
        "runtime",
        {
            "session_id": session.id,
            "mode": session.mode,
            "routing": session.routing,
            "health": session.health,
            "working_set": session.working_set,
            "current_stage_id": session.current_stage_id,
            "plan_stages": [asdict(stage) for stage in session.plan_stages],
            "working_memory": asdict(session.working_memory),
            "episodic_memory": asdict(session.episodic_memory),
            "operational_memory": asdict(session.operational_memory),
            "cold_memory": asdict(session.cold_memory),
            "tool_calls": [asdict(call) for call in session.tool_calls[-20:]],
            "retry_count": session.retry_count,
            "compression_count": session.compression_count,
            "degraded_reason": session.degraded_reason,
            "grounding_report": session.grounding_report,
            "runtime_trace": session.runtime_trace,
        },
        "runtime audit",
    )
    session.artifacts = [asdict(item) for item in run_state.artifacts]
    path = store.save_session(session)
    store.write_summary(session)
    store.write_active_memory(session)
    store.write_cold_summary(session)
    store.append_session_log(
        {
            "session_id": session.id,
            "mode": session.mode,
            "grounded_answer": session.grounded_answer,
            "retry_count": session.retry_count,
            "compression_count": session.compression_count,
            "degraded_reason": session.degraded_reason,
            "routing": session.routing,
            "health_state": session.health.get("rolling", {}).get("backend_state", "unknown"),
            "runtime_trace": session.runtime_trace,
        }
    )
    return final_answer, path, session
