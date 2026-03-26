from __future__ import annotations

import time
from dataclasses import asdict

from echo.cognition import build_execution_plan
from echo.types import PlanStage, RunState, SessionState


def find_stage(run_state: RunState, stage_id: str) -> PlanStage | None:
    for stage in run_state.plan_stages:
        if stage.stage_id == stage_id:
            return stage
    return None


def set_current_stage(session: SessionState, run_state: RunState, stage_id: str) -> None:
    run_state.current_stage_id = stage_id
    session.current_stage_id = stage_id


def update_stage(
    session: SessionState,
    run_state: RunState,
    activity,
    stage_id: str,
    *,
    status: str,
    result: str = "",
    evidence: list[str] | None = None,
) -> None:
    stage = find_stage(run_state, stage_id)
    if stage is None:
        return
    stage.status = status
    if result:
        stage.result = result
    if evidence:
        stage.evidence = list(dict.fromkeys(stage.evidence + evidence))
    stage.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    session.plan_stages = [PlanStage(**asdict(item)) for item in run_state.plan_stages]
    if status == "running":
        set_current_stage(session, run_state, stage_id)
    activity.emit("Stage", status, stage_id, stage.result or stage.objective)


def replan_stage(session: SessionState, run_state: RunState, activity, stage_id: str, reason: str) -> str:
    stage = find_stage(run_state, stage_id)
    if stage is None:
        return stage_id
    replanned_id = f"{stage.stage_id}-replan-{stage.attempts + 1}"
    stage.status = "replanned"
    stage.result = reason
    stage.attempts += 1
    replanned = PlanStage(
        stage_id=replanned_id,
        objective=stage.objective,
        hypothesis=f"{stage.hypothesis} Replan: {reason}",
        target_files=list(stage.target_files),
        intended_actions=list(stage.intended_actions),
        validation_goal=stage.validation_goal,
        completion_criteria=stage.completion_criteria,
        failure_policy=stage.failure_policy,
        status="pending",
        pending=[reason],
        replanned_from=stage.stage_id,
    )
    run_state.plan_stages.append(replanned)
    session.plan_stages = [PlanStage(**asdict(item)) for item in run_state.plan_stages]
    activity.emit("Stage", "replanned", stage.stage_id, replanned_id)
    return replanned_id


def initialize_plan(
    session: SessionState,
    run_state: RunState,
    prompt: str,
    *,
    validation_strategy: str,
) -> None:
    if run_state.plan_stages:
        return
    stages = build_execution_plan(
        prompt,
        mode=session.mode,
        focus_files=session.focus_files or session.working_set or run_state.focus_files,
        validation_strategy=validation_strategy,
    )
    run_state.plan_stages = stages
    session.plan_stages = [PlanStage(**asdict(item)) for item in stages]
    if stages:
        set_current_stage(session, run_state, stages[0].stage_id)


def plan_guidance_message(run_state: RunState) -> dict[str, str] | None:
    stage = find_stage(run_state, run_state.current_stage_id)
    if stage is None:
        return None
    return {
        "role": "system",
        "content": (
            f"Current stage: {stage.stage_id}. "
            f"Objective: {stage.objective}. "
            f"Hypothesis: {stage.hypothesis}. "
            f"Target files: {', '.join(stage.target_files) or 'none'}. "
            f"Intended actions: {'; '.join(stage.intended_actions) or 'none'}. "
            f"Validation goal: {stage.validation_goal}. "
            f"Completion criteria: {stage.completion_criteria}. "
            f"Failure policy: {stage.failure_policy}."
        ),
    }
