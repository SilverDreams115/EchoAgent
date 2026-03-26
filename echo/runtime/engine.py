from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from echo.backends import BackendAvailabilityPolicy
from echo.cognition import (
    build_plan_prompt,
    render_execution_plan,
    build_session_summary,
    detect_validation_strategy,
    validate_final_answer,
)
from echo.config import Settings
from echo.context import build_repo_map, compress_messages_if_needed, select_relevant_files
from echo.memory import EchoStore
from echo.policies import default_constraints, profile_intake_limits, profile_step_limit, should_auto_verify
from echo.runtime.activity import ActivityBus
from echo.runtime.backend_runtime import backend_health_check, perform_backend_request
from echo.runtime.budget import RuntimeBudget, build_runtime_budget
from echo.runtime.finalize import finalize_session
from echo.runtime.memory_sync import confirmed_facts, sync_memory_layers
from echo.runtime.model_loop import run_model_loop
from echo.runtime.outcomes import build_degraded_answer, build_heuristic_plan, build_resume_local_answer, is_resume_summary_only
from echo.runtime.prepare import build_intake_messages, evaluate_preflight, resume_seed, seed_inspection
from echo.runtime.stages import find_stage, initialize_plan, plan_guidance_message, replan_stage, set_current_stage, update_stage
from echo.runtime.trace import record_backend_request, record_runtime_phase, runtime_trace_payload, update_runtime_outcome
from echo.runtime.tool_tracking import record_tool_call
from echo.runtime.tool_calling import parse_tool_calls_from_text
from echo.runtime.verify_flow import run_auto_verify
from echo.tools import ToolRegistry
from echo.types import BackendHealth, PhaseRecord, PlanStage, RunState, RuntimeArtifact, SessionState, ToolCallRecord, WorkingMemory


SYSTEM_PROMPT = """
You are Echo, a local coding agent.

Rules:
- You are Echo, not Claude, not Anthropic, not ChatGPT.
- Inspect real project files before answering.
- Do not emit fake tool calls or JSON plans to the user.
- Prefer minimal patches over file rewrites.
- Mention concrete file paths and command results.
- When inspected evidence contains code symbols, mention at least one concrete symbol from the inspected files.
- Do not reconstruct file contents that were not actually inspected.
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
        return find_stage(run_state, stage_id)

    def _set_current_stage(self, session: SessionState, run_state: RunState, stage_id: str) -> None:
        set_current_stage(session, run_state, stage_id)

    def _update_stage(self, session: SessionState, run_state: RunState, stage_id: str, *, status: str, result: str = "", evidence: list[str] | None = None) -> None:
        update_stage(session, run_state, self.activity, stage_id, status=status, result=result, evidence=evidence)

    def _replan_stage(self, session: SessionState, run_state: RunState, stage_id: str, reason: str) -> str:
        return replan_stage(session, run_state, self.activity, stage_id, reason)

    def _initialize_plan(self, session: SessionState, run_state: RunState, prompt: str) -> None:
        initialize_plan(session, run_state, prompt, validation_strategy=self._validation_strategy(session))

    def _plan_guidance_message(self, run_state: RunState) -> dict[str, str] | None:
        return plan_guidance_message(run_state)

    def _confirmed_facts(self, run_state: RunState, session: SessionState) -> list[str]:
        return confirmed_facts(run_state, session)

    def _sync_memory_layers(self, session: SessionState, run_state: RunState) -> None:
        sync_memory_layers(session, run_state)
        session.working_memory.validation_strategy = self._validation_strategy(session)
        run_state.working_memory = WorkingMemory(**asdict(session.working_memory))

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
        budget = build_runtime_budget(self.settings)
        run_state.runtime_budget_ms = budget.total_ms
        run_state.runtime_deadline_ms = budget.deadline_ms
        run_state.runtime_reserve_ms = budget.reserve_ms
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
        record_tool_call(session, run_state, name, arguments, result)

    def _seed_inspection(self, session: SessionState, run_state: RunState, prompt: str) -> tuple[list[str], list[str], Any]:
        repo_limit, file_limit, snippet_line_limit = self._intake_limits()
        return seed_inspection(
            project_root=self.project_root,
            prompt=prompt,
            repo_limit=repo_limit,
            file_limit=file_limit,
            snippet_line_limit=snippet_line_limit,
            mode=session.mode,
            backend_native_tools_enabled=self._backend_native_tools_enabled(),
            build_repo_map=build_repo_map,
            select_relevant_files=select_relevant_files,
            tools_execute=self.tools.execute,
            record_tool_call=self._record_tool_call,
            mark_phase=self._mark_phase,
            session=session,
            run_state=run_state,
        )

    def _intake(self, session: SessionState, run_state: RunState, prompt: str) -> list[dict[str, Any]]:
        self._mark_phase(run_state, "Intake", "running", "Analizando objetivo y restricciones")
        repo_map, focus_snippets, intake_shape = self._seed_inspection(session, run_state, prompt)
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
        messages = build_intake_messages(
            system_prompt=profile_system_prompt(self.profile),
            constraints=run_state.constraints,
            repo_map=repo_map,
            focus_snippets=focus_snippets,
            compatibility_guide=self.tools.compatibility_guide(),
            stage_guidance=self._plan_guidance_message(run_state),
            shape=intake_shape,
        )
        self._sync_memory_layers(session, run_state)
        session.messages.extend(messages)
        self._mark_phase(run_state, "Intake", "done", f"Intake completed mode={intake_shape.detail}")
        return messages

    def _resume_seed(self, loaded: SessionState, prompt: str, mode: str) -> SessionState:
        return resume_seed(
            loaded=loaded,
            project_root=self.project_root,
            mode=mode,
            model=self.settings.model,
            user_prompt=prompt,
            system_prompt=profile_system_prompt(self.profile),
            activity=self.activity,
        )

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

    def _call_backend(
        self,
        run_state: RunState,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        return perform_backend_request(
            backend=self.backend,
            settings=self.settings,
            store=self.store,
            activity=self.activity,
            run_state=run_state,
            messages=messages,
            tools=tools,
            timeout_seconds=timeout_seconds,
            record_backend_request=record_backend_request,
        )

    def _call_backend_with_timeout(
        self,
        run_state: RunState,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        return self._call_backend(run_state, messages, tools=tools, timeout_seconds=timeout_seconds)

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

    def _backend_health_check(self, run_state: RunState) -> BackendHealth:
        return backend_health_check(
            backend=self.backend,
            settings=self.settings,
            activity=self.activity,
            run_state=run_state,
        )

    def _time_phase(self, run_state: RunState, phase: str, fn):
        started = time.perf_counter()
        status = "done"
        detail = ""
        try:
            result = fn()
            if (
                isinstance(result, tuple)
                and len(result) == 2
                and isinstance(result[0], tuple)
                and len(result[0]) == 2
                and isinstance(result[0][0], str)
            ):
                status = result[0][0]
                detail = result[0][1] if isinstance(result[0][1], str) else ""
                return result[1]
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
                status = result[0]
                if isinstance(result[1], str):
                    detail = result[1]
            elif isinstance(result, str):
                detail = result
            return result
        except Exception as exc:
            status = "failed"
            detail = str(exc)
            raise
        finally:
            record_runtime_phase(
                run_state,
                phase,
                status=status,
                duration_ms=int((time.perf_counter() - started) * 1000),
                detail=detail or phase,
            )

    def _heuristic_plan(self, session: SessionState, run_state: RunState, reason: str) -> str:
        return build_heuristic_plan(
            session,
            run_state,
            reason=reason,
            update_stage=self._update_stage,
            activity=self.activity,
        )

    def _degraded_answer(self, session: SessionState, run_state: RunState, reason: str, mode: str) -> str:
        return build_degraded_answer(
            session,
            run_state,
            reason=reason,
            mode=mode,
            update_stage=self._update_stage,
            activity=self.activity,
        )

    def _resume_summary_only(self, prompt: str) -> bool:
        return is_resume_summary_only(prompt)

    def _resume_local_answer(self, session: SessionState, run_state: RunState) -> str:
        return build_resume_local_answer(session, run_state, activity=self.activity)

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
        return run_model_loop(
            session=session,
            run_state=run_state,
            messages=messages,
            prompt=prompt,
            mode=mode,
            profile=self.profile,
            settings=self.settings,
            budget=RuntimeBudget(
                total_ms=run_state.runtime_budget_ms,
                deadline_ms=run_state.runtime_deadline_ms,
                reserve_ms=run_state.runtime_reserve_ms,
            ),
            backend_native_tools_enabled=self._backend_native_tools_enabled,
            tools_schema=self.tools.schema,
            step_limit=self._step_limit(),
            compress_messages=lambda current: compress_messages_if_needed(
                current,
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
            ),
            context_ratio=self._context_ratio,
            mark_phase=self._mark_phase,
            call_backend=self._call_backend_with_timeout,
            extract_tool_calls=self._extract_tool_calls,
            degraded_answer=self._degraded_answer,
            update_stage=self._update_stage,
            replan_stage=self._replan_stage,
            reduce_context=self._reduce_context,
            grounding_retry_message=self._grounding_retry_message,
            record_tool_call=self._record_tool_call,
            execute_tool=self.tools.execute,
            collect_tool_previews=self._collect_tool_previews,
            validation_strategy=self._validation_strategy,
            find_stage=self._find_stage,
        )

    def _auto_verify(self, session: SessionState, run_state: RunState) -> tuple[str, str]:
        return run_auto_verify(
            session=session,
            run_state=run_state,
            settings=self.settings,
            profile=self.profile,
            should_auto_verify=should_auto_verify,
            find_stage=self._find_stage,
            update_stage=self._update_stage,
            mark_phase=self._mark_phase,
            execute_tool=self.tools.execute,
            record_tool_call=self._record_tool_call,
            sync_memory_layers=self._sync_memory_layers,
        )

    def _preflight_degrade_if_needed(self, session: SessionState, run_state: RunState, mode: str) -> tuple[str | None, bool, str]:
        rolling = BackendHealth(**asdict(run_state.backend_health))
        health = self._backend_health_check(run_state)
        effective = BackendAvailabilityPolicy.effective_backend_health(health, run_state.fresh_backend_health)
        run_state.backend_health = effective
        decision = evaluate_preflight(
            mode=mode,
            rolling=rolling,
            effective=effective,
            heuristic_plan=self._heuristic_plan,
            degraded_answer=self._degraded_answer,
            activity=self.activity,
            session=session,
            run_state=run_state,
        )
        return decision.final_answer, decision.reduce_initial_context, decision.detail

    def _finalize(self, session: SessionState, run_state: RunState, final_answer: str) -> tuple[str, Path, SessionState]:
        update_runtime_outcome(run_state, session)
        final_answer, path, session = finalize_session(
            session,
            run_state,
            final_answer=final_answer,
            store=self.store,
            activity=self.activity,
            update_stage=self._update_stage,
            sync_memory_layers=self._sync_memory_layers,
            build_session_summary=build_session_summary,
            write_runtime_artifact=self._write_runtime_artifact,
            runtime_trace_payload=runtime_trace_payload,
        )
        self.last_run_state = run_state
        self._mark_phase(run_state, "Summarizer", "done", "Summary updated")
        self._mark_phase(run_state, "Memory", "done", "Cold memory updated")
        return final_answer, path, session

    def _persist_runtime_trace(self, session: SessionState, run_state: RunState) -> None:
        session.runtime_trace = runtime_trace_payload(run_state)
        self.store.save_session(session)
        artifact_path = self.store.artifacts / f"{run_state.session_id}-runtime.json"
        if artifact_path.exists():
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            payload["runtime_trace"] = session.runtime_trace
            artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
        if loaded is None:
            messages = self._time_phase(
                run_state,
                "prepare",
                lambda: (("done", "fresh session prepared"), self._intake(session, run_state, prompt)),
            )
        else:
            def prepare_resumed():
                if not run_state.plan_stages:
                    self._initialize_plan(session, run_state, prompt)
                return (("done", f"resumed session parent={session.parent_session_id or 'none'}"), list(session.messages))
            messages = self._time_phase(run_state, "prepare", prepare_resumed)
        if loaded is not None and session.mode == "resume" and self._resume_summary_only(prompt):
            final_answer = self._resume_local_answer(session, run_state)
            record_runtime_phase(run_state, "preflight", status="done", duration_ms=0, detail="resume local summary")
            record_runtime_phase(run_state, "execute", status="done", duration_ms=0, detail="resume local summary")
        else:
            degraded, reduce_initial_context, preflight_detail = self._time_phase(
                run_state,
                "preflight",
                lambda: (("done", "preflight evaluated"), self._preflight_degrade_if_needed(session, run_state, session.mode)),
            )
            if preflight_detail:
                record_runtime_phase(run_state, "preflight", status=run_state.runtime_trace.phases[-1].status, duration_ms=run_state.runtime_trace.phases[-1].duration_ms, detail=preflight_detail)
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
                    final_answer = self._time_phase(run_state, "execute", lambda: (("done", "plan rendered"), render_execution_plan(run_state.plan_stages)))
                elif reduce_initial_context:
                    def execute_reduced():
                        reduced = self._reduce_context(session, run_state, messages)
                        session.working_set = session.working_set[-3:] or session.focus_files[-3:]
                        return (("done", f"retries={run_state.retry_count} degraded={bool(session.degraded_reason)}"), self._run_model_loop(session, run_state, reduced, prompt, session.mode))
                    final_answer = self._time_phase(run_state, "execute", execute_reduced)
                else:
                    final_answer = self._time_phase(
                        run_state,
                        "execute",
                        lambda: (("done", f"retries={run_state.retry_count} degraded={bool(session.degraded_reason)}"), self._run_model_loop(session, run_state, messages, prompt, session.mode)),
                    )
        self._time_phase(run_state, "verify", lambda: self._auto_verify(session, run_state))
        final_answer, path, session = self._time_phase(run_state, "finalize", lambda: (("done", "session finalized"), self._finalize(session, run_state, final_answer)))
        update_runtime_outcome(run_state, session)
        self._persist_runtime_trace(session, run_state)
        return final_answer, path, session, run_state
