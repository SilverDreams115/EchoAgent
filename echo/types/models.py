from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ActivityEvent:
    stage: str
    status: str
    message: str
    detail: str = ""
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ToolCallRecord:
    tool: str
    arguments: dict[str, Any]
    result_preview: str
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class GroundingReport:
    grounded_file_count: int = 0
    grounded_symbol_count: int = 0
    evidence_usage: int = 0
    genericity_score: int = 0
    useful: bool = True
    claim_types: list[str] = field(default_factory=list)
    unsupported_files: list[str] = field(default_factory=list)
    unsupported_symbols: list[str] = field(default_factory=list)
    unsupported_commands: list[str] = field(default_factory=list)
    unsupported_changes: list[str] = field(default_factory=list)
    contradiction_flags: list[str] = field(default_factory=list)
    validation_strategy_match: bool = True
    validation_strategy: str = "unknown"
    speculation_flags: list[str] = field(default_factory=list)
    valid: bool = True
    reason: str = "ok"


@dataclass(slots=True)
class BackendHealth:
    backend_reachable: bool = False
    backend_chat_ready: bool = False
    backend_chat_slow: bool = False
    backend_state: str = "unknown"
    last_success_ms: int = 0
    last_timeout_ms: int = 0
    average_chat_ms: int = 0
    recent_failures: int = 0
    success_rate: float = 0.0
    chat_probe_count: int = 0
    tags_latency_ms: int = 0
    warm_state: str = "unknown"
    last_error: str = ""
    checked_at: str = field(default_factory=utc_now)
    source: str = "rolling"
    backend_name: str = ""
    model: str = ""
    detail: str = ""


@dataclass(slots=True)
class RoutingDecision:
    primary_backend: str = ""
    primary_model: str = ""
    selected_backend: str = ""
    selected_model: str = ""
    fallback_backend: str = ""
    fallback_model: str = ""
    fallback_available: bool = False
    fallback_selected: bool = False
    policy: str = "primary-only"
    task_complexity: str = "low"
    reason: str = ""


@dataclass(slots=True)
class RuntimeArtifact:
    kind: str
    path: str
    detail: str = ""


@dataclass(slots=True)
class WorkingMemory:
    objective: str = ""
    current_stage_id: str = ""
    active_files: list[str] = field(default_factory=list)
    recent_tools: list[str] = field(default_factory=list)
    recent_evidence: list[str] = field(default_factory=list)
    validation_strategy: str = "unknown"


@dataclass(slots=True)
class EpisodicMemory:
    decisions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    retries: list[str] = field(default_factory=list)
    replans: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OperationalMemory:
    summary: str = ""
    confirmed_facts: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    stage_progress: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ColdMemory:
    summary_path: str = ""
    session_refs: list[str] = field(default_factory=list)
    archived_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class PlanStage:
    stage_id: str
    objective: str
    hypothesis: str
    target_files: list[str] = field(default_factory=list)
    intended_actions: list[str] = field(default_factory=list)
    validation_goal: str = ""
    completion_criteria: str = ""
    failure_policy: str = ""
    status: str = "pending"
    result: str = ""
    evidence: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    attempts: int = 0
    replanned_from: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class SessionState:
    id: str
    repo_root: str
    mode: str
    model: str
    user_prompt: str
    objective: str = ""
    restrictions: list[str] = field(default_factory=list)
    summary: str = ""
    operational_summary: str = ""
    cold_summary_path: str = ""
    focus_files: list[str] = field(default_factory=list)
    working_set: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    plan_stages: list[PlanStage] = field(default_factory=list)
    current_stage_id: str = ""
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    episodic_memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    operational_memory: OperationalMemory = field(default_factory=OperationalMemory)
    cold_memory: ColdMemory = field(default_factory=ColdMemory)
    errors: list[str] = field(default_factory=list)
    parent_session_id: str = ""
    grounded_answer: bool = False
    grounding_report: dict[str, Any] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    health: dict[str, Any] = field(default_factory=dict)
    runtime_trace: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    compression_count: int = 0
    degraded_reason: str = ""
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    activity: list[ActivityEvent] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def create(cls, repo_root: str, mode: str, model: str, user_prompt: str) -> "SessionState":
        return cls(
            id=f"session-{uuid.uuid4().hex[:12]}",
            repo_root=repo_root,
            mode=mode,
            model=model,
            user_prompt=user_prompt,
            objective=user_prompt,
        )


@dataclass(slots=True)
class PhaseRecord:
    phase: str
    status: str
    detail: str = ""
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class RuntimePhaseTrace:
    phase: str
    status: str = "pending"
    duration_ms: int = 0
    detail: str = ""


@dataclass(slots=True)
class BackendRequestTrace:
    message_count: int = 0
    timeout_seconds: int = 0
    tools_enabled: bool = False
    duration_ms: int = 0
    outcome: str = "unknown"
    detail: str = ""


@dataclass(slots=True)
class RuntimeTrace:
    phases: list[RuntimePhaseTrace] = field(default_factory=list)
    backend_requests: list[BackendRequestTrace] = field(default_factory=list)
    retry_count: int = 0
    degraded: bool = False
    grounded: bool = False
    budget_remaining_ms: int = 0
    outcome_reason: str = ""


@dataclass(slots=True)
class RunState:
    session_id: str
    mode: str
    profile: str
    objective: str
    repo_root: str
    backend: str
    model: str
    constraints: list[str] = field(default_factory=list)
    focus_files: list[str] = field(default_factory=list)
    inspected_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    plan_stages: list[PlanStage] = field(default_factory=list)
    current_stage_id: str = ""
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    episodic_memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    operational_memory: OperationalMemory = field(default_factory=OperationalMemory)
    validation_commands: list[str] = field(default_factory=list)
    open_issues: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    phases: list[PhaseRecord] = field(default_factory=list)
    context_free_ratio: float = 1.0
    backend_duration_ms: list[int] = field(default_factory=list)
    backend_health: BackendHealth = field(default_factory=BackendHealth)
    fresh_backend_health: BackendHealth = field(default_factory=lambda: BackendHealth(source="fresh"))
    routing: RoutingDecision = field(default_factory=RoutingDecision)
    fallback_used: bool = False
    fallback_reason: str = ""
    grounding_report: GroundingReport = field(default_factory=GroundingReport)
    retry_count: int = 0
    compression_count: int = 0
    runtime_budget_ms: int = 0
    runtime_deadline_ms: int = 0
    runtime_reserve_ms: int = 0
    runtime_trace: RuntimeTrace = field(default_factory=RuntimeTrace)
    artifacts: list[RuntimeArtifact] = field(default_factory=list)
