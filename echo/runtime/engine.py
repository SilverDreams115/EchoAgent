from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from echo.backends import BackendAvailabilityPolicy, backend_log_state, quick_health_probe
from echo.backends.errors import (
    BackendError,
    BackendMalformedResponseError,
    BackendModelMissingError,
    BackendTimeoutError,
    BackendUnreachableError,
)
from echo.cognition import (
    build_execution_plan,
    build_plan_prompt,
    render_execution_plan,
    build_session_summary,
    detect_validation_strategy,
    validate_final_answer,
)
from echo.config import Settings
from echo.context import build_operational_snapshot, build_repo_map, compress_messages_if_needed, select_relevant_files
from echo.memory import EchoStore
from echo.policies import default_constraints, profile_intake_limits, profile_step_limit, should_auto_verify
from echo.runtime.activity import ActivityBus
from echo.runtime.tool_calling import parse_tool_calls_from_text
from echo.tools import ToolRegistry
from echo.types import BackendHealth, ColdMemory, EpisodicMemory, OperationalMemory, PhaseRecord, PlanStage, RunState, RuntimeArtifact, SessionState, ToolCallRecord, WorkingMemory


SYSTEM_PROMPT = """
You are Echo, a local coding agent.

Rules:
- You are Echo, not Claude, not Anthropic, not ChatGPT.
- Inspect real project files before answering.
- Do not emit fake tool calls or JSON plans to the user.
- Prefer minimal patches over file rewrites.
- Mention concrete file paths and command results.
- If evidence is insufficient, inspect more files before concluding.
""".strip()


def profile_system_prompt(profile: str) -> str:
    if profile == "deep":
        return SYSTEM_PROMPT + "\n- Prioritize stronger reasoning, explicit tradeoffs, and architectural rigor."
    if profile == "balanced":
        return SYSTEM_PROMPT + "\n- Balance speed with grounded engineering judgment."
    return SYSTEM_PROMPT + "\n- Optimize for concise, practical local execution."


class AgentRuntime:
    def __init__(
        self,
        project_root: Path,
        settings: Settings,
        backend: Any,
        store: EchoStore,
        tools: ToolRegistry,
        activity: ActivityBus,
    ) -> None:
        self.project_root = project_root
        self.settings = settings
        self.backend = backend
        self.store = store
        self.tools = tools
        self.activity = activity
        self.last_run_state: RunState | None = None
        self.profile = getattr(settings, "profile", "local")
        self.route_decision = None

    def _backend_tools_enabled(self) -> bool:
        return bool(getattr(self.backend, "supports_tools", False))

    def _backend_native_tools_enabled(self) -> bool:
        return bool(getattr(self.backend, "supports_native_tools", False))

    def _intake_limits(self) -> tuple[int, int, int]:
        return profile_intake_limits(
            self.profile,
            self._backend_tools_enabled(),
            self.settings.context_file_limit,
            self.settings.snippet_line_limit,
        )

    def _step_limit(self) -> int:
        return profile_step_limit(self.profile, self._backend_tools_enabled(), self.settings.max_steps)

    def _context_ratio(self, messages: list[dict[str, Any]]) -> float:
        char_usage = sum(len(str(item.get("content", ""))) for item in messages)
        char_ratio = char_usage / max(self.settings.context_char_limit, 1)
        message_ratio = len(messages) / max(self.settings.context_message_limit, 1)
        used = max(char_ratio, message_ratio)
        return max(0.0, min(1.0, 1.0 - used))

    def _mark_phase(self, run_state: RunState, phase: str, status: str, detail: str) -> None:
        run_state.phases.append(PhaseRecord(phase=phase, status=status, detail=detail))
        self.activity.emit(phase, status, detail, detail)

    def _find_stage(self, run_state: RunState, stage_id: str) -> PlanStage | None:
        for stage in run_state.plan_stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def _set_current_stage(self, session: SessionState, run_state: RunState, stage_id: str) -> None:
        run_state.current_stage_id = stage_id
        session.current_stage_id = stage_id

    def _update_stage(self, session: SessionState, run_state: RunState, stage_id: str, *, status: str, result: str = "", evidence: list[str] | None = None) -> None:
        stage = self._find_stage(run_state, stage_id)
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
            self._set_current_stage(session, run_state, stage_id)
        self.activity.emit("Stage", status, stage_id, stage.result or stage.objective)

    def _replan_stage(self, session: SessionState, run_state: RunState, stage_id: str, reason: str) -> str:
        stage = self._find_stage(run_state, stage_id)
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
        self.activity.emit("Stage", "replanned", stage.stage_id, replanned_id)
        return replanned_id

    def _initialize_plan(self, session: SessionState, run_state: RunState, prompt: str) -> None:
        if run_state.plan_stages:
            return
        stages = build_execution_plan(
            prompt,
            mode=session.mode,
            focus_files=session.focus_files or session.working_set or run_state.focus_files,
            validation_strategy=self._validation_strategy(session),
        )
        run_state.plan_stages = stages
        session.plan_stages = [PlanStage(**asdict(item)) for item in stages]
        if stages:
            self._set_current_stage(session, run_state, stages[0].stage_id)

    def _plan_guidance_message(self, run_state: RunState) -> dict[str, str] | None:
        stage = self._find_stage(run_state, run_state.current_stage_id)
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

    def _confirmed_facts(self, run_state: RunState, session: SessionState) -> list[str]:
        facts = list(run_state.findings or [])
        facts.extend(f"stage:{stage.stage_id}:{stage.status}" for stage in run_state.plan_stages if stage.status in {"completed", "failed", "replanned"})
        facts.extend(session.findings[-6:])
        return list(dict.fromkeys(facts))[-12:]

    def _sync_memory_layers(self, session: SessionState, run_state: RunState) -> None:
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
        session.working_memory = WorkingMemory(
            objective=session.objective or run_state.objective,
            current_stage_id=current_stage,
            active_files=list(dict.fromkeys((session.working_set or session.focus_files or run_state.focus_files)[-8:])),
            recent_tools=recent_tools,
            recent_evidence=list(dict.fromkeys(recent_evidence))[-8:],
            validation_strategy=self._validation_strategy(session),
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
                confirmed_facts=self._confirmed_facts(run_state, session),
            ),
            confirmed_facts=self._confirmed_facts(run_state, session),
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

    def _load_resume_session(self, session_id: str | None) -> SessionState | None:
        target_id = session_id or self.store.latest_session_id()
        if not target_id:
            return None
        loaded = self.store.load_session(target_id)
        self.activity.emit("Resume", "done", "Session loaded", loaded.id)
        return loaded

    def _build_run_state(self, session: SessionState, mode: str, objective: str) -> RunState:
        run_state = RunState(
            session_id=session.id,
            mode=mode,
            profile=self.profile,
            objective=objective,
            repo_root=str(self.project_root),
            backend=self.settings.backend,
            model=self.settings.model,
            constraints=default_constraints(self.profile),
            pending=["Finish with a grounded answer after inspection and validation."],
        )
        run_state.backend_health = self.store.read_backend_health()
        run_state.backend_health.backend_name = run_state.backend_health.backend_name or self.settings.backend
        run_state.backend_health.model = run_state.backend_health.model or self.settings.model
        if self.route_decision is not None:
            run_state.routing.primary_backend = self.route_decision.primary_backend
            run_state.routing.primary_model = self.route_decision.primary_model
            run_state.routing.selected_backend = self.route_decision.selected_backend
            run_state.routing.selected_model = self.route_decision.selected_model
            run_state.routing.fallback_backend = self.route_decision.fallback_backend
            run_state.routing.fallback_model = self.route_decision.fallback_model
            run_state.routing.fallback_available = self.route_decision.fallback_available
            run_state.routing.fallback_selected = self.route_decision.fallback_selected
            run_state.routing.policy = self.route_decision.policy
            run_state.routing.task_complexity = self.route_decision.task_complexity
            run_state.routing.reason = self.route_decision.reason
        return run_state

    def _write_runtime_artifact(self, run_state: RunState, name: str, payload: dict[str, Any], detail: str = "") -> None:
        path = self.store.write_artifact(f"{run_state.session_id}-{name}.json", payload)
        run_state.artifacts.append(RuntimeArtifact(kind="runtime", path=str(path), detail=detail))

    def _record_tool_call(
        self,
        session: SessionState,
        run_state: RunState,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        tracked_path = arguments.get("path")
        if isinstance(tracked_path, str) and tracked_path.strip():
            if tracked_path not in session.focus_files:
                session.focus_files.append(tracked_path)
            if tracked_path not in session.working_set:
                session.working_set.append(tracked_path)
            if name in {"read_file", "read_file_range", "search_symbol", "find_symbol"} and tracked_path not in run_state.inspected_files:
                run_state.inspected_files.append(tracked_path)
            if name in {"write_file", "apply_patch", "insert_before", "insert_after", "replace_range"} and tracked_path not in run_state.changed_files:
                run_state.changed_files.append(tracked_path)
        if name == "validate_project" and "validation_command" in result:
            command = str(result["validation_command"])
            session.validation.append(command)
            run_state.validation_commands.append(command)
        preview = json.dumps(result, ensure_ascii=False)[:220]
        session.tool_calls.append(ToolCallRecord(tool=name, arguments=arguments, result_preview=preview))
        if name in {"search_symbol", "find_symbol"} and isinstance(result.get("matches"), list):
            for match in result["matches"][:8]:
                symbol = match.get("symbol")
                path = match.get("path")
                if symbol and path:
                    finding = f"{path}:{symbol}"
                    if finding not in run_state.findings:
                        run_state.findings.append(finding)
        if result.get("error"):
            issue = f"{name}: {result['error']}"
            run_state.errors.append(issue)
            if issue not in run_state.open_issues:
                run_state.open_issues.append(issue)
        current_stage = self._find_stage(run_state, run_state.current_stage_id)
        if current_stage is not None:
            evidence = []
            if tracked_path:
                evidence.append(str(tracked_path))
            if name == "validate_project" and result.get("validation_command"):
                evidence.append(str(result.get("validation_command")))
            if evidence:
                current_stage.evidence = list(dict.fromkeys(current_stage.evidence + evidence))
                current_stage.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _seed_inspection(self, session: SessionState, run_state: RunState, prompt: str) -> tuple[list[str], list[str]]:
        repo_limit, file_limit, snippet_line_limit = self._intake_limits()
        repo_map = build_repo_map(self.project_root, max_entries=repo_limit)
        self._mark_phase(run_state, "RepoMap", "done", f"RepoMap scanning entries={len(repo_map)}")
        list_result = self.tools.execute("list_files", {"path": "", "max_depth": 2})
        self._record_tool_call(session, run_state, "list_files", {"path": "", "max_depth": 2}, list_result)
        focus_files = select_relevant_files(self.project_root, prompt, limit=file_limit)
        snippets: list[str] = []
        for rel in focus_files:
            read_result = self.tools.execute("read_file", {"path": rel})
            self._record_tool_call(session, run_state, "read_file", {"path": rel}, read_result)
            content = str(read_result.get("content", ""))
            snippet = "\n".join(content.splitlines()[:snippet_line_limit])
            snippets.append(f"FILE: {rel}\n{snippet}")
        run_state.focus_files = list(dict.fromkeys(run_state.focus_files + focus_files))
        session.focus_files = list(dict.fromkeys(session.focus_files + focus_files))
        session.working_set = list(dict.fromkeys(session.working_set + focus_files))
        self._mark_phase(run_state, "Inspector", "done", f"Selected focus files={len(focus_files)}")
        return repo_map, snippets

    def _intake(self, session: SessionState, run_state: RunState, prompt: str) -> list[dict[str, Any]]:
        self._mark_phase(run_state, "Intake", "running", "Analizando objetivo y restricciones")
        repo_map, focus_snippets = self._seed_inspection(session, run_state, prompt)
        self._initialize_plan(session, run_state, prompt)
        inspect_stage = self._find_stage(run_state, "inspect")
        if inspect_stage is not None:
            self._update_stage(
                session,
                run_state,
                "inspect",
                status="completed",
                result=f"Inspected {len(session.focus_files)} focus files.",
                evidence=session.focus_files[-8:],
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": profile_system_prompt(self.profile)},
            {"role": "system", "content": "Constraints:\n- " + "\n- ".join(run_state.constraints)},
            {"role": "system", "content": "Repo map:\n" + "\n".join(repo_map)},
        ]
        if focus_snippets:
            messages.append({"role": "system", "content": "Focus snippets:\n\n" + "\n\n".join(focus_snippets)})
        if not self._backend_native_tools_enabled():
            messages.append({"role": "system", "content": self.tools.compatibility_guide()})
        stage_guidance = self._plan_guidance_message(run_state)
        if stage_guidance is not None:
            messages.append(stage_guidance)
        self._sync_memory_layers(session, run_state)
        session.messages.extend(messages)
        self._mark_phase(run_state, "Intake", "done", "Intake completed")
        return messages

    def _resume_seed(self, loaded: SessionState, prompt: str, mode: str) -> SessionState:
        session = SessionState.create(
            repo_root=str(self.project_root),
            mode=mode,
            model=self.settings.model,
            user_prompt=prompt,
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
            {"role": "system", "content": profile_system_prompt(self.profile)},
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
        self.activity.emit("Resume", "done", "Session loaded", loaded.id)
        self.activity.emit("Resume", "done", "Objective restored", session.objective)
        self.activity.emit("Resume", "done", "Working set restored", ", ".join(session.working_set[-6:]) or "none")
        self.activity.emit("Resume", "done", "Memory summary restored", loaded.id)
        return session

    def _extract_tool_calls(self, message: dict[str, Any], content: str) -> list[dict[str, Any]]:
        native_calls = message.get("tool_calls") or []
        if native_calls:
            return native_calls
        return parse_tool_calls_from_text(content)

    def _collect_tool_previews(self, session: SessionState) -> list[str]:
        return [call.result_preview for call in session.tool_calls[-12:]]

    def _validation_strategy(self, session: SessionState) -> str:
        if session.validation:
            return detect_validation_strategy(project_files=session.focus_files + session.working_set, validation_commands=session.validation)
        return detect_validation_strategy(project_root=self.project_root)

    def _append_backend_log(self, payload: dict[str, Any]) -> None:
        payload.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.store.append_backend_log(payload)

    def _call_backend(
        self,
        run_state: RunState,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        self.activity.emit("Backend", "running", "Backend request started", f"model={self.backend.model}")
        self.activity.emit("Backend", "waiting", "Backend waiting", f"messages={len(messages)}")
        try:
            response = self.backend.chat(messages=messages, tools=tools)
        except BackendTimeoutError as exc:
            duration_ms = int((time.perf_counter() - start) * 1000)
            run_state.backend_duration_ms.append(duration_ms)
            run_state.backend_health.backend_reachable = True
            run_state.backend_health.backend_chat_ready = False
            run_state.backend_health.backend_chat_slow = True
            run_state.backend_health.last_timeout_ms = duration_ms
            run_state.backend_health.recent_failures += 1
            run_state.backend_health.backend_state = "timeout"
            run_state.backend_health.last_error = str(exc)
            self._append_backend_log(
                {
                    "event": "timeout",
                    "backend": self.backend.backend_name,
                    "model": self.backend.model,
                    "duration_ms": duration_ms,
                    "message_count": len(messages),
                    **backend_log_state(
                        backend_reachable=True,
                        backend_chat_ready=False,
                        backend_chat_slow=True,
                        backend_state="timeout",
                    ),
                }
            )
            self.activity.emit("Backend", "failed", "Backend timeout", f"duration_ms={duration_ms}")
            raise
        except BackendError as exc:
            duration_ms = int((time.perf_counter() - start) * 1000)
            run_state.backend_duration_ms.append(duration_ms)
            run_state.backend_health.recent_failures += 1
            run_state.backend_health.last_error = str(exc)
            run_state.backend_health.backend_reachable = not isinstance(exc, BackendUnreachableError)
            run_state.backend_health.backend_chat_ready = False
            run_state.backend_health.backend_state = "unreachable" if isinstance(exc, BackendUnreachableError) else "error"
            self._append_backend_log(
                {
                    "event": "error",
                    "backend": self.backend.backend_name,
                    "model": self.backend.model,
                    "duration_ms": duration_ms,
                    "detail": exc.detail or str(exc),
                    **backend_log_state(
                        backend_reachable=not isinstance(exc, BackendUnreachableError),
                        backend_chat_ready=False,
                        backend_state="unreachable" if isinstance(exc, BackendUnreachableError) else "error",
                    ),
                }
            )
            self.activity.emit("Backend", "failed", "Backend request failed", str(exc))
            raise
        duration_ms = int((time.perf_counter() - start) * 1000)
        run_state.backend_duration_ms.append(duration_ms)
        run_state.backend_health.backend_reachable = True
        run_state.backend_health.backend_chat_ready = True
        run_state.backend_health.backend_chat_slow = duration_ms >= self.settings.backend_slow_threshold_ms
        run_state.backend_health.last_success_ms = duration_ms
        run_state.backend_health.recent_failures = 0
        run_state.backend_health.backend_state = "slow" if run_state.backend_health.backend_chat_slow else "ready"
        self._append_backend_log(
            {
                "event": "response",
                "backend": self.backend.backend_name,
                "model": self.backend.model,
                "duration_ms": duration_ms,
                "message_count": len(messages),
                **backend_log_state(
                    backend_reachable=True,
                    backend_chat_ready=True,
                    backend_chat_slow=run_state.backend_health.backend_chat_slow,
                ),
            }
        )
        self.activity.emit("Backend", "done", "Backend response received", f"duration_ms={duration_ms}")
        return response

    def _reduce_context(self, session: SessionState, run_state: RunState, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        reduced, summary = compress_messages_if_needed(
            messages,
            self.activity,
            self.settings.context_message_limit,
            self.settings.context_char_limit,
            objective=session.objective or run_state.objective,
            restrictions=session.restrictions or run_state.constraints,
            decisions=session.decisions or run_state.decisions,
            current_stage_id=run_state.current_stage_id or session.current_stage_id,
            focus_files=session.working_set or session.focus_files,
            changed_files=session.changed_files or run_state.changed_files,
            errors=session.errors or run_state.errors,
            pending=session.pending or run_state.pending,
            validation_commands=session.validation or run_state.validation_commands,
            confirmed_facts=self._confirmed_facts(run_state, session),
            force=True,
        )
        if summary:
            session.operational_summary = summary
            session.operational_memory.summary = summary
            run_state.compression_count += 1
            session.compression_count = run_state.compression_count
            self.activity.emit("Memory", "done", "Context compressed", f"compression_count={run_state.compression_count}")
        return reduced

    def _with_backend_timeout(self, timeout: int, fn):
        previous = getattr(self.backend, "timeout", None)
        if previous is not None:
            self.backend.timeout = timeout
        try:
            return fn()
        finally:
            if previous is not None:
                self.backend.timeout = previous

    def _backend_health_check(self, run_state: RunState) -> BackendHealth:
        rolling = run_state.backend_health
        rolling.source = "rolling"
        self.activity.emit("Backend", "running", "Backend health check", rolling.backend_state or "unknown")
        fresh = quick_health_probe(self.backend, self.settings, include_chat=False)
        run_state.fresh_backend_health = fresh
        effective = BackendAvailabilityPolicy.effective_backend_health(rolling, fresh)
        if effective.backend_chat_ready:
            self.activity.emit("Backend", "done", "Backend health check", f"{effective.backend_state} duration_ms={effective.last_success_ms}")
        else:
            self.activity.emit("Backend", "failed", "Backend health check", effective.last_error or effective.backend_state)
        return effective

    def _heuristic_plan(self, session: SessionState, run_state: RunState, reason: str) -> str:
        run_state.fallback_used = True
        run_state.fallback_reason = reason
        stage_id = run_state.current_stage_id or "inspect"
        self._update_stage(session, run_state, stage_id, status="failed", result=reason)
        self.activity.emit("Planner", "degraded", "Fallback plan mode", reason)
        files = session.working_set[-6:] or session.focus_files[-6:] or ["README.md"]
        risks = [
            "El backend de chat está inestable, así que el plan se basa en inspección local y memoria operativa.",
            "Puede faltar validación semántica profunda hasta que Ollama vuelva a responder de forma consistente.",
        ]
        next_steps = [
            "Revisar el loop de runtime y el verificador grounded.",
            "Confirmar backend health y repetir smoke cuando /api/chat deje de oscilar.",
        ]
        return "\n".join(
            [
                "Objetivo",
                f"- Mantener Echo operativo aun con backend inestable y reforzar grounding del answer final.",
                "Archivos a revisar",
                *(f"- {item}" for item in files),
                "Riesgos",
                *(f"- {item}" for item in risks),
                "Siguientes pasos",
                *(f"- {item}" for item in next_steps),
            ]
        )

    def _degraded_answer(self, session: SessionState, run_state: RunState, reason: str, mode: str) -> str:
        run_state.fallback_used = True
        run_state.fallback_reason = reason
        session.degraded_reason = reason
        current_stage = run_state.current_stage_id or ("close" if mode == "plan" else "execute")
        self._update_stage(session, run_state, current_stage, status="failed", result=reason)
        if mode == "resume":
            self.activity.emit("Resume", "degraded", "Resume state restored without backend completion", reason)
            return "\n".join(
                [
                    "Echo restauró el estado de la sesión, pero no pudo completar con el backend.",
                    f"Objetivo: {session.objective or session.user_prompt}",
                    f"Working set: {', '.join(session.working_set[-8:]) or 'none'}",
                f"Pendientes: {'; '.join(session.pending[-6:]) or 'none'}",
                f"Etapa detenida: {current_stage}",
                f"Límite actual: {reason}",
            ]
        )
        self.activity.emit("Planner" if mode == "plan" else "Verifier", "degraded", "Fallback answer mode", reason)
        evidence = session.working_set[-6:] or session.focus_files[-6:]
        return "\n".join(
            [
                "Echo reunió inspección local, pero no pudo cerrar con el backend de forma confiable.",
                f"Archivos inspeccionados: {', '.join(evidence) or 'none'}",
                f"Hallazgos recientes: {'; '.join(session.findings[-6:] or run_state.inspected_files[-6:] or ['inspección local completada'])}",
                f"Etapa detenida: {current_stage}",
                f"Backend efectivo: {run_state.fresh_backend_health.backend_state or run_state.backend_health.backend_state}",
                f"Límite actual: {reason}",
            ]
        )

    def _grounding_retry_message(self, session: SessionState, run_state: RunState, reason: str) -> dict[str, str]:
        report = run_state.grounding_report
        return {
            "role": "system",
            "content": (
                "Last answer was not grounded enough. "
                f"Reason: {reason}. "
                f"Working set: {', '.join(session.working_set[-8:]) or 'none'}. "
                f"Inspected files: {', '.join(run_state.inspected_files[-8:] or session.focus_files[-8:]) or 'none'}. "
                f"Recent tool evidence: {' | '.join(self._collect_tool_previews(session)[-4:]) or 'none'}. "
                f"Validation strategy: {self._validation_strategy(session)}. "
                f"Unsupported files: {', '.join(report.unsupported_files[:4]) or 'none'}. "
                f"Unsupported symbols: {', '.join(report.unsupported_symbols[:4]) or 'none'}. "
                f"Unsupported commands: {', '.join(report.unsupported_commands[:4]) or 'none'}. "
                f"Contradictions: {', '.join(report.contradiction_flags[:4]) or 'none'}. "
                "Respond with explicit file references and only claims supported by inspected evidence. "
                "Mention at least one concrete symbol if the evidence contains one. "
                "Do not claim edits, commands, or validation unless the tool evidence proves they happened. "
                "If the evidence is insufficient, say that directly instead of inventing."
            ),
        }

    def _run_model_loop(
        self,
        session: SessionState,
        run_state: RunState,
        messages: list[dict[str, Any]],
        prompt: str,
        mode: str,
    ) -> str:
        if session.mode != "plan":
            execute_stage = self._find_stage(run_state, "execute")
            if execute_stage is not None and execute_stage.status == "pending":
                self._update_stage(session, run_state, "execute", status="running", result="Execution stage started.")
        user_prompt = build_plan_prompt(prompt, profile=self.profile) if mode == "plan" else prompt
        messages.append({"role": "user", "content": user_prompt})
        session.messages.append(messages[-1])
        final_answer = ""
        timeout_retried = False
        grounding_retried = False
        malformed_retried = False
        for step in range(1, self._step_limit() + 1):
            messages, compressed = compress_messages_if_needed(
                messages,
                self.activity,
                self.settings.context_message_limit,
                self.settings.context_char_limit,
                objective=session.objective or run_state.objective,
                restrictions=session.restrictions or run_state.constraints,
                decisions=session.decisions or run_state.decisions,
                current_stage_id=run_state.current_stage_id or session.current_stage_id,
                focus_files=session.working_set or session.focus_files,
                changed_files=session.changed_files or run_state.changed_files,
                errors=session.errors or run_state.errors,
                pending=session.pending or run_state.pending,
                validation_commands=session.validation or run_state.validation_commands,
                confirmed_facts=self._confirmed_facts(run_state, session),
            )
            if compressed:
                session.operational_summary = compressed
                session.operational_memory.summary = compressed
                run_state.compression_count += 1
                session.compression_count = run_state.compression_count
            run_state.context_free_ratio = self._context_ratio(messages)
            self._mark_phase(run_state, "Planner", "running", f"Planner running step={step}")
            tools = self.tools.schema() if self._backend_native_tools_enabled() else None
            try:
                response = self._call_backend(run_state, messages, tools=tools)
            except BackendTimeoutError as exc:
                issue = str(exc)
                run_state.errors.append(issue)
                run_state.open_issues.append(issue)
                session.errors.append(issue)
                if not timeout_retried and len(messages) > 4:
                    timeout_retried = True
                    run_state.retry_count += 1
                    session.retry_count = run_state.retry_count
                    failed_stage = run_state.current_stage_id or "execute"
                    self._update_stage(session, run_state, failed_stage, status="failed", result=issue)
                    replanned_id = self._replan_stage(session, run_state, failed_stage, issue)
                    self._update_stage(session, run_state, replanned_id, status="running", result="Retry after timeout.")
                    self.activity.emit("Backend", "retrying", "Retry with reduced context", "Reducing context after timeout")
                    messages = self._reduce_context(session, run_state, messages)
                    session.working_set = session.working_set[-2:] or session.focus_files[-2:]
                    retry_message = {
                        "role": "system",
                        "content": "Previous request timed out. Use only the minimum working set, cite files explicitly, and respond briefly.",
                    }
                    messages.append(retry_message)
                    session.messages.append(retry_message)
                    continue
                return self._degraded_answer(session, run_state, issue, mode)
            except (BackendModelMissingError, BackendUnreachableError, BackendMalformedResponseError) as exc:
                issue = str(exc)
                run_state.errors.append(issue)
                session.errors.append(issue)
                if isinstance(exc, BackendMalformedResponseError) and not malformed_retried:
                    malformed_retried = True
                    run_state.retry_count += 1
                    session.retry_count = run_state.retry_count
                    failed_stage = run_state.current_stage_id or "execute"
                    self._update_stage(session, run_state, failed_stage, status="failed", result=issue)
                    replanned_id = self._replan_stage(session, run_state, failed_stage, issue)
                    self._update_stage(session, run_state, replanned_id, status="running", result="Retry after malformed response.")
                    self.activity.emit("Backend", "retrying", "Retry with reduced context", "Malformed response; reducing context")
                    messages = self._reduce_context(session, run_state, messages)
                    messages.append(
                        {
                            "role": "system",
                            "content": "Previous response was malformed. Return plain text or valid tool JSON only.",
                        }
                    )
                    continue
                return self._degraded_answer(session, run_state, issue, mode)
            self._mark_phase(run_state, "Planner", "done", f"Planner completed step={step}")
            message = response.get("message", {})
            content = (message.get("content") or "").strip()
            assistant = {"role": "assistant", "content": content}
            tool_calls = self._extract_tool_calls(message, content)
            if self._backend_native_tools_enabled() and tool_calls:
                assistant["tool_calls"] = tool_calls
            messages.append(assistant)
            session.messages.append(assistant)
            if not tool_calls:
                if mode == "ask" and len(session.tool_calls) == 0:
                    run_state.open_issues.append("ask completed without real tool execution")
                    return self._degraded_answer(session, run_state, "El pipeline no ejecutó herramientas reales antes de responder.", mode)
                valid, reason, report = validate_final_answer(
                    content,
                    profile=self.profile,
                    mode=mode,
                    inspected_files=run_state.inspected_files,
                    changed_files=run_state.changed_files,
                    tool_calls=session.tool_calls,
                    tool_result_previews=self._collect_tool_previews(session),
                    working_set=session.working_set,
                    validation_strategy=self._validation_strategy(session),
                )
                run_state.grounding_report.grounded_file_count = int(report.get("grounded_file_count", 0))
                run_state.grounding_report.grounded_symbol_count = int(report.get("grounded_symbol_count", 0))
                run_state.grounding_report.evidence_usage = int(report.get("evidence_usage", 0))
                run_state.grounding_report.genericity_score = int(report.get("genericity_score", 0))
                run_state.grounding_report.useful = bool(report.get("useful", True))
                run_state.grounding_report.claim_types = list(report.get("claim_types", []))
                run_state.grounding_report.unsupported_files = list(report.get("unsupported_files", []))
                run_state.grounding_report.unsupported_symbols = list(report.get("unsupported_symbols", []))
                run_state.grounding_report.unsupported_commands = list(report.get("unsupported_commands", []))
                run_state.grounding_report.unsupported_changes = list(report.get("unsupported_changes", []))
                run_state.grounding_report.contradiction_flags = list(report.get("contradiction_flags", []))
                run_state.grounding_report.validation_strategy_match = bool(report.get("validation_strategy_match", True))
                run_state.grounding_report.validation_strategy = str(report.get("validation_strategy", "unknown"))
                run_state.grounding_report.speculation_flags = list(report.get("speculation_flags", []))
                run_state.grounding_report.valid = bool(report.get("valid", valid))
                run_state.grounding_report.reason = str(report.get("reason", reason))
                if not valid:
                    run_state.open_issues.append(reason)
                    if not grounding_retried:
                        grounding_retried = True
                        run_state.retry_count += 1
                        session.retry_count = run_state.retry_count
                        failed_stage = run_state.current_stage_id or "execute"
                        self._update_stage(session, run_state, failed_stage, status="failed", result=reason)
                        replanned_id = self._replan_stage(session, run_state, failed_stage, reason)
                        self._update_stage(session, run_state, replanned_id, status="running", result="Retry after grounding failure.")
                        self._mark_phase(run_state, "Verifier", "retrying", reason)
                        retry_message = self._grounding_retry_message(session, run_state, reason)
                        messages.append(retry_message)
                        session.messages.append(retry_message)
                        continue
                    return self._degraded_answer(
                        session,
                        run_state,
                        f"Respuesta final no grounded tras retry: {reason}",
                        mode,
                    )
                final_answer = content
                session.grounded_answer = True
                session.grounding_report = report
                current_stage = run_state.current_stage_id or "execute"
                self._update_stage(session, run_state, current_stage, status="completed", result="Grounded answer accepted.")
                break
            for call in tool_calls:
                fn = call.get("function", {})
                name = str(fn.get("name", "")).strip()
                arguments = fn.get("arguments", {}) or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}
                result = self.tools.execute(name, arguments)
                self._record_tool_call(session, run_state, name, arguments, result)
                tool_message = {
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
                messages.append(tool_message)
                session.messages.append(tool_message)
            run_state.pending = ["Consume the latest tool results before finalizing."]
        else:
            final_answer = self._degraded_answer(session, run_state, "Echo alcanzó el máximo de pasos.", mode)
        return final_answer

    def _auto_verify(self, session: SessionState, run_state: RunState) -> None:
        if not self.settings.auto_verify or not should_auto_verify(run_state.mode, run_state.changed_files, profile=self.profile):
            verify_stage = self._find_stage(run_state, "verify")
            if verify_stage is not None and verify_stage.status == "pending":
                self._update_stage(session, run_state, "verify", status="skipped", result="Verification skipped by policy or because there are no changed files.")
            self._sync_memory_layers(session, run_state)
            return
        self._update_stage(session, run_state, "verify", status="running", result="Validation stage started.")
        self._mark_phase(run_state, "Verifier", "running", "Verifier running validation")
        result = self.tools.execute("validate_project", {})
        self._record_tool_call(session, run_state, "validate_project", {}, result)
        command = str(result.get("validation_command", ""))
        strategy = str(result.get("validation_strategy", "unknown"))
        if strategy == "unknown":
            issue = f"Validation unavailable: {result.get('validation_reason', 'unknown strategy')}"
            run_state.open_issues.append(issue)
            run_state.errors.append(issue)
            session.errors.append(issue)
            self._mark_phase(run_state, "Verifier", "failed", issue)
            self._update_stage(session, run_state, "verify", status="failed", result=issue)
        elif result.get("returncode", 1) != 0:
            issue = f"Validation failed: {command}"
            run_state.open_issues.append(issue)
            run_state.errors.append(issue)
            session.errors.append(issue)
            self._mark_phase(run_state, "Verifier", "failed", issue)
            self._update_stage(session, run_state, "verify", status="failed", result=issue)
        else:
            self._mark_phase(run_state, "Verifier", "done", f"Validation passed: {strategy} {command}")
            self._update_stage(session, run_state, "verify", status="completed", result=f"Validation passed: {strategy} {command}", evidence=[command])
        self._sync_memory_layers(session, run_state)

    def _preflight_degrade_if_needed(self, session: SessionState, run_state: RunState, mode: str) -> str | None:
        rolling = BackendHealth(**asdict(run_state.backend_health))
        health = self._backend_health_check(run_state)
        effective = BackendAvailabilityPolicy.effective_backend_health(health, run_state.fresh_backend_health)
        run_state.backend_health = effective
        if mode == "plan" and (
            effective.backend_state in {"ready", "slow"}
            or (effective.backend_state == "reachable" and rolling.recent_failures == 0 and rolling.backend_state in {"unknown", "reachable"})
        ):
            return None
        if effective.backend_chat_ready:
            if effective.backend_chat_slow:
                self.activity.emit("Backend", "warning", "Backend unstable", "Chat responde lento")
                self.activity.emit("Backend", "retrying", "Retry with reduced context", "Slow backend; shrinking context before first request")
            return None
        if effective.backend_state == "reachable" and mode == "ask":
            self.activity.emit("Backend", "warning", "Backend chat unverified", "Reachable backend; first request will verify chat readiness")
            return None
        reason = effective.last_error or f"backend_state={effective.backend_state}"
        policy = BackendAvailabilityPolicy.classify_mode(mode, effective)
        if policy == "heuristic_plan":
            return self._heuristic_plan(session, run_state, reason)
        if policy == "resume_restore_only":
            return self._degraded_answer(session, run_state, reason, mode)
        return self._degraded_answer(session, run_state, reason, mode)

    def _finalize(self, session: SessionState, run_state: RunState, final_answer: str) -> tuple[str, Path, SessionState]:
        session.activity = self.activity.recent(120)
        session.objective = session.objective or run_state.objective
        session.restrictions = list(dict.fromkeys(session.restrictions or run_state.constraints))
        session.focus_files = list(dict.fromkeys(session.focus_files + run_state.changed_files))
        session.working_set = list(dict.fromkeys(session.working_set + session.focus_files + run_state.changed_files))
        session.changed_files = list(dict.fromkeys(session.changed_files + run_state.changed_files))
        session.decisions = list(dict.fromkeys(session.decisions + run_state.decisions))
        session.findings = list(dict.fromkeys(session.findings + run_state.findings + run_state.inspected_files))
        session.pending = list(dict.fromkeys(session.pending + run_state.pending + run_state.open_issues))
        session.errors = list(dict.fromkeys(session.errors + run_state.errors + run_state.open_issues))
        close_stage = self._find_stage(run_state, "close")
        if close_stage is not None:
            close_status = "failed" if session.degraded_reason else "completed"
            close_result = session.degraded_reason or "Final answer and stage summary recorded."
            self._update_stage(session, run_state, "close", status=close_status, result=close_result)
        self._sync_memory_layers(session, run_state)
        if not session.grounding_report and run_state.grounding_report.reason != "ok":
            session.grounding_report = {
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
        session.routing = asdict(run_state.routing)
        session.plan_stages = [PlanStage(**asdict(item)) for item in run_state.plan_stages]
        session.current_stage_id = run_state.current_stage_id
        session.health = {
            "effective": asdict(run_state.backend_health),
            "fresh": asdict(run_state.fresh_backend_health),
            "cached": asdict(self.store.read_backend_health()),
        }
        session.retry_count = run_state.retry_count
        session.compression_count = run_state.compression_count
        session.artifacts = [asdict(item) for item in run_state.artifacts]
        session.summary = build_session_summary(session, final_answer)
        session.operational_summary = session.operational_memory.summary or session.operational_summary or session.summary
        self._write_runtime_artifact(
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
            },
            "runtime audit",
        )
        session.artifacts = [asdict(item) for item in run_state.artifacts]
        path = self.store.save_session(session)
        self.store.write_summary(session)
        self.store.write_active_memory(session)
        self.store.write_cold_summary(session)
        self.store.append_session_log(
            {
                "session_id": session.id,
                "mode": session.mode,
                "grounded_answer": session.grounded_answer,
                "retry_count": session.retry_count,
                "compression_count": session.compression_count,
                "degraded_reason": session.degraded_reason,
                "routing": session.routing,
                "health_state": session.health.get("rolling", {}).get("backend_state", "unknown"),
            }
        )
        self.last_run_state = run_state
        self._mark_phase(run_state, "Summarizer", "done", "Summary updated")
        self._mark_phase(run_state, "Memory", "done", "Cold memory updated")
        return final_answer, path, session

    def run(self, prompt: str, mode: str = "ask", resume_session_id: str | None = None) -> tuple[str, Path, SessionState, RunState]:
        loaded = self._load_resume_session(resume_session_id) if resume_session_id or mode == "resume" else None
        if loaded is not None:
            session = self._resume_seed(loaded, prompt, "resume" if mode == "resume" else mode)
        else:
            session = SessionState.create(
                repo_root=str(self.project_root),
                mode=mode,
                model=self.settings.model,
                user_prompt=prompt,
            )
        run_state = self._build_run_state(session, session.mode, session.objective or prompt)
        if session.plan_stages:
            run_state.plan_stages = [PlanStage(**asdict(item)) for item in session.plan_stages]
            run_state.current_stage_id = session.current_stage_id
        run_state.decisions.append("Use repo evidence and minimal patches.")
        if not run_state.routing.selected_backend:
            run_state.routing.selected_backend = self.backend.backend_name
            run_state.routing.selected_model = self.backend.model
        messages = self._intake(session, run_state, prompt) if loaded is None else list(session.messages)
        if loaded is not None and not run_state.plan_stages:
            self._initialize_plan(session, run_state, prompt)
        degraded = self._preflight_degrade_if_needed(session, run_state, session.mode)
        if degraded is not None:
            final_answer = degraded
        else:
            if session.mode == "plan":
                for stage_id, result in [
                    ("hypothesis", "Hypotheses derived from inspected files and prompt."),
                    ("stage-plan", "Executable stage plan created."),
                    ("verify-plan", "Stage criteria and failure policies verified."),
                    ("close", "Plan rendered for the user."),
                ]:
                    if self._find_stage(run_state, stage_id) is not None:
                        self._update_stage(session, run_state, stage_id, status="completed", result=result)
                final_answer = render_execution_plan(run_state.plan_stages)
            elif run_state.backend_health.backend_chat_slow:
                messages = self._reduce_context(session, run_state, messages)
                session.working_set = session.working_set[-3:] or session.focus_files[-3:]
                final_answer = self._run_model_loop(session, run_state, messages, prompt, session.mode)
            else:
                final_answer = self._run_model_loop(session, run_state, messages, prompt, session.mode)
        self._auto_verify(session, run_state)
        return (*self._finalize(session, run_state, final_answer), run_state)
