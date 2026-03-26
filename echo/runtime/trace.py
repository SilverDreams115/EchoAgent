from __future__ import annotations

from dataclasses import asdict

from echo.runtime.budget import monotonic_ms
from echo.types import BackendRequestTrace, RunState, RuntimePhaseTrace, SessionState


def record_runtime_phase(run_state: RunState, phase: str, *, status: str, duration_ms: int, detail: str = "") -> None:
    trace = run_state.runtime_trace
    for item in trace.phases:
        if item.phase == phase:
            item.status = status
            item.duration_ms = duration_ms
            item.detail = detail
            return
    trace.phases.append(RuntimePhaseTrace(phase=phase, status=status, duration_ms=duration_ms, detail=detail))


def record_backend_request(
    run_state: RunState,
    *,
    message_count: int,
    timeout_seconds: int,
    tools_enabled: bool,
    duration_ms: int,
    outcome: str,
    detail: str = "",
) -> None:
    trace = run_state.runtime_trace
    trace.backend_requests.append(
        BackendRequestTrace(
            message_count=message_count,
            timeout_seconds=timeout_seconds,
            tools_enabled=tools_enabled,
            duration_ms=duration_ms,
            outcome=outcome,
            detail=detail,
        )
    )
    if len(trace.backend_requests) > 6:
        trace.backend_requests = trace.backend_requests[-6:]


def update_runtime_outcome(run_state: RunState, session: SessionState) -> None:
    trace = run_state.runtime_trace
    trace.retry_count = run_state.retry_count
    trace.degraded = bool(session.degraded_reason)
    trace.grounded = bool(session.grounded_answer)
    trace.budget_remaining_ms = max(0, run_state.runtime_deadline_ms - monotonic_ms())
    trace.outcome_reason = (
        session.degraded_reason
        or run_state.grounding_report.reason
        or ("grounded" if session.grounded_answer else "completed")
    )


def runtime_trace_payload(run_state: RunState) -> dict[str, object]:
    return asdict(run_state.runtime_trace)
