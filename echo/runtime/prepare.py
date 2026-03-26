from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from echo.backends import BackendAvailabilityPolicy
from echo.types import BackendHealth, ColdMemory, EpisodicMemory, OperationalMemory, RunState, SessionState, WorkingMemory


@dataclass(slots=True)
class PreflightDecision:
    final_answer: str | None = None
    reduce_initial_context: bool = False
    detail: str = ""


@dataclass(slots=True)
class IntakeShape:
    snippet_line_limit: int
    include_repo_map: bool = True
    include_tools_guide: bool = True
    include_stage_guidance: bool = True
    detail: str = "standard"


def choose_intake_shape(
    *,
    mode: str,
    prompt: str,
    focus_files: list[str],
    snippet_line_limit: int,
    backend_native_tools_enabled: bool,
) -> IntakeShape:
    shape = IntakeShape(
        snippet_line_limit=snippet_line_limit,
        include_repo_map=True,
        include_tools_guide=not backend_native_tools_enabled,
        include_stage_guidance=True,
        detail="standard",
    )
    low = (prompt or "").lower()
    explicit_file_prompt = any(token in low for token in [".py", ".md", ".toml", ".json", ".yml", ".yaml"])
    mutating_or_broad = any(
        token in low
        for token in [
            "cambi",
            "modific",
            "edit",
            "patch",
            "fix",
            "refactor",
            "implement",
            "plan",
            "ejecut",
            "run ",
            "shell",
            "comando",
        ]
    )
    if mode == "ask" and explicit_file_prompt and len(focus_files) <= 2 and not mutating_or_broad:
        shape.snippet_line_limit = min(snippet_line_limit, 24)
        shape.include_repo_map = False
        shape.include_tools_guide = False
        shape.include_stage_guidance = False
        shape.detail = "slim-explicit-ask"
    return shape


def resume_seed(
    *,
    loaded: SessionState,
    project_root: Path,
    mode: str,
    model: str,
    user_prompt: str,
    system_prompt: str,
    activity,
) -> SessionState:
    session = SessionState.create(
        repo_root=str(project_root),
        mode=mode,
        model=model,
        user_prompt=user_prompt,
    )
    session.parent_session_id = loaded.id
    session.objective = loaded.objective or loaded.user_prompt
    session.restrictions = list(loaded.restrictions)
    session.operational_summary = loaded.operational_memory.summary or loaded.operational_summary or loaded.summary
    session.focus_files = list(loaded.focus_files)
    session.working_set = list(loaded.working_memory.active_files or loaded.working_set or loaded.focus_files)
    session.decisions = list(loaded.episodic_memory.decisions or loaded.decisions)
    session.findings = list(loaded.operational_memory.confirmed_facts or loaded.findings)
    session.pending = list(loaded.operational_memory.pending or loaded.pending)
    session.changed_files = list(loaded.episodic_memory.changes or loaded.changed_files)
    session.errors = list(loaded.episodic_memory.errors or loaded.errors)
    session.validation = list(loaded.episodic_memory.validations or loaded.validation)
    session.plan_stages = list(loaded.plan_stages)
    session.current_stage_id = loaded.working_memory.current_stage_id or loaded.current_stage_id
    session.working_memory = WorkingMemory(**asdict(loaded.working_memory))
    session.episodic_memory = EpisodicMemory(**asdict(loaded.episodic_memory))
    session.operational_memory = OperationalMemory(**asdict(loaded.operational_memory))
    session.cold_memory = ColdMemory(**asdict(loaded.cold_memory))
    session.messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": (
                f"Resumed session from {loaded.id}\n"
                f"Objective: {session.objective}\n"
                f"Current stage: {session.current_stage_id or 'none'}\n"
                f"Working set: {', '.join(session.working_memory.active_files[-8:] or session.working_set[-8:]) or 'none'}\n"
                f"Recent tools: {'; '.join(session.working_memory.recent_tools[-4:]) or 'none'}\n"
                f"Recent decisions: {'; '.join(session.episodic_memory.decisions[-4:] or session.decisions[-4:]) or 'none'}\n"
                f"Confirmed facts: {'; '.join(session.operational_memory.confirmed_facts[-4:] or session.findings[-4:]) or 'none'}\n"
                f"Pending: {'; '.join(session.operational_memory.pending[-4:] or session.pending[-4:]) or 'none'}\n"
                f"{session.operational_memory.summary or session.operational_summary or 'No prior operational summary.'}"
            ),
        },
    ]
    activity.emit("Resume", "done", "Session loaded", loaded.id)
    activity.emit("Resume", "done", "Objective restored", session.objective)
    activity.emit("Resume", "done", "Working set restored", ", ".join(session.working_set[-6:]) or "none")
    activity.emit("Resume", "done", "Memory summary restored", loaded.id)
    return session


def seed_inspection(
    *,
    project_root: Path,
    prompt: str,
    repo_limit: int,
    file_limit: int,
    snippet_line_limit: int,
    mode: str,
    backend_native_tools_enabled: bool,
    build_repo_map,
    select_relevant_files,
    tools_execute,
    record_tool_call,
    mark_phase,
    session: SessionState,
    run_state: RunState,
) -> tuple[list[str], list[str]]:
    repo_map = build_repo_map(project_root, max_entries=repo_limit)
    mark_phase(run_state, "RepoMap", "done", f"RepoMap scanning entries={len(repo_map)}")
    list_result = tools_execute("list_files", {"path": "", "max_depth": 2})
    record_tool_call(session, run_state, "list_files", {"path": "", "max_depth": 2}, list_result)
    focus_files = select_relevant_files(project_root, prompt, limit=file_limit)
    shape = choose_intake_shape(
        mode=mode,
        prompt=prompt,
        focus_files=focus_files,
        snippet_line_limit=snippet_line_limit,
        backend_native_tools_enabled=backend_native_tools_enabled,
    )
    snippets: list[str] = []
    for rel in focus_files:
        read_result = tools_execute("read_file", {"path": rel})
        record_tool_call(session, run_state, "read_file", {"path": rel}, read_result)
        content = str(read_result.get("content", ""))
        snippet = "\n".join(content.splitlines()[: shape.snippet_line_limit])
        snippets.append(f"FILE: {rel}\n{snippet}")
    run_state.focus_files = list(dict.fromkeys(run_state.focus_files + focus_files))
    session.focus_files = list(dict.fromkeys(session.focus_files + focus_files))
    session.working_set = list(dict.fromkeys(session.working_set + focus_files))
    mark_phase(run_state, "Inspector", "done", f"Selected focus files={len(focus_files)}")
    return repo_map, snippets, shape


def build_intake_messages(
    *,
    system_prompt: str,
    constraints: list[str],
    repo_map: list[str],
    focus_snippets: list[str],
    compatibility_guide: str,
    stage_guidance: dict[str, str] | None,
    shape: IntakeShape,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "Constraints:\n- " + "\n- ".join(constraints)},
    ]
    if shape.include_repo_map:
        messages.append({"role": "system", "content": "Repo map:\n" + "\n".join(repo_map)})
    if focus_snippets:
        messages.append({"role": "system", "content": "Focus snippets:\n\n" + "\n\n".join(focus_snippets)})
    if shape.include_tools_guide:
        messages.append({"role": "system", "content": compatibility_guide})
    if shape.include_stage_guidance and stage_guidance is not None:
        messages.append(stage_guidance)
    return messages


def evaluate_preflight(
    *,
    mode: str,
    rolling: BackendHealth,
    effective: BackendHealth,
    heuristic_plan,
    degraded_answer,
    activity,
    session: SessionState,
    run_state: RunState,
) -> PreflightDecision:
    if mode == "plan" and (
        effective.backend_state in {"ready", "slow"}
        or (effective.backend_state == "reachable" and rolling.recent_failures == 0 and rolling.backend_state in {"unknown", "reachable"})
    ):
        return PreflightDecision(detail=f"preflight ok: {effective.backend_state}")
    if effective.backend_chat_ready:
        if effective.backend_chat_slow:
            activity.emit("Backend", "warning", "Backend unstable", "Chat responde lento")
            activity.emit("Backend", "retrying", "Retry with reduced context", "Slow backend; shrinking context before first request")
            return PreflightDecision(reduce_initial_context=True, detail="backend slow; reducing initial context")
        return PreflightDecision(detail=f"preflight ok: {effective.backend_state}")
    if effective.backend_state == "reachable" and mode == "ask":
        activity.emit("Backend", "warning", "Backend chat unverified", "Reachable backend; first request will verify chat readiness")
        return PreflightDecision(detail="backend reachable; chat unverified")
    reason = effective.last_error or f"backend_state={effective.backend_state}"
    policy = BackendAvailabilityPolicy.classify_mode(mode, effective)
    if policy == "heuristic_plan":
        return PreflightDecision(final_answer=heuristic_plan(session, run_state, reason), detail=f"heuristic plan: {reason}")
    return PreflightDecision(final_answer=degraded_answer(session, run_state, reason, mode), detail=f"degraded: {reason}")
