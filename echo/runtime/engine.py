from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from echo.backends import BackendAvailabilityPolicy
from echo.backends.errors import (
    BackendError,
    BackendMalformedResponseError,
    BackendModelMissingError,
    BackendTimeoutError,
    BackendUnreachableError,
)
from echo.cognition import (
    build_plan_prompt,
    build_session_summary,
    detect_validation_strategy,
    validate_final_answer,
)
from echo.config import Settings
from echo.context import build_repo_map, compress_messages_if_needed, select_relevant_files
from echo.memory import EchoStore
from echo.policies import default_constraints, profile_intake_limits, profile_step_limit, should_auto_verify
from echo.runtime.activity import ActivityBus
from echo.runtime.tool_calling import parse_tool_calls_from_text
from echo.tools import ToolRegistry
from echo.types import BackendHealth, PhaseRecord, RunState, RuntimeArtifact, SessionState, ToolCallRecord


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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": profile_system_prompt(self.profile)},
            {"role": "system", "content": "Constraints:\n- " + "\n- ".join(run_state.constraints)},
            {"role": "system", "content": "Repo map:\n" + "\n".join(repo_map)},
        ]
        if focus_snippets:
            messages.append({"role": "system", "content": "Focus snippets:\n\n" + "\n\n".join(focus_snippets)})
        if not self._backend_native_tools_enabled():
            messages.append({"role": "system", "content": self.tools.compatibility_guide()})
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
        session.operational_summary = loaded.operational_summary or loaded.summary
        session.focus_files = list(loaded.focus_files)
        session.working_set = list(loaded.working_set or loaded.focus_files)
        session.decisions = list(loaded.decisions)
        session.findings = list(loaded.findings)
        session.pending = list(loaded.pending)
        session.changed_files = list(loaded.changed_files)
        session.errors = list(loaded.errors)
        session.validation = list(loaded.validation)
        session.messages = [
            {"role": "system", "content": profile_system_prompt(self.profile)},
            {
                "role": "system",
                "content": (
                    f"Resumed session from {loaded.id}\n"
                    f"Objective: {session.objective}\n"
                    f"Working set: {', '.join(session.working_set[-8:]) or 'none'}\n"
                    f"Recent decisions: {'; '.join(session.decisions[-4:]) or 'none'}\n"
                    f"Recent findings: {'; '.join(session.findings[-4:]) or 'none'}\n"
                    f"Pending: {'; '.join(session.pending[-4:]) or 'none'}\n"
                    f"{session.operational_summary or 'No prior operational summary.'}"
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
        return detect_validation_strategy(session.focus_files + session.working_set, session.validation)

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
            focus_files=session.working_set or session.focus_files,
            changed_files=session.changed_files or run_state.changed_files,
            errors=session.errors or run_state.errors,
            pending=session.pending or run_state.pending,
            force=True,
        )
        if summary:
            session.operational_summary = summary
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
        health = run_state.backend_health
        health.source = "rolling"
        self.activity.emit("Backend", "running", "Backend health check", health.backend_state or "unknown")
        if health.recent_failures == 0 and health.backend_state == "unknown":
            health.backend_chat_ready = True
            run_state.fresh_backend_health = BackendHealth(
                source="fresh",
                backend_reachable=True,
                backend_chat_ready=True,
                backend_state="assumed_ready",
                backend_name=self.backend.backend_name,
                model=self.backend.model,
                detail="no recent failures; using rolling state",
            )
            return health
        if health.recent_failures < self.settings.backend_failure_threshold and health.backend_state == "ready":
            run_state.fresh_backend_health = BackendHealth(
                source="fresh",
                backend_reachable=True,
                backend_chat_ready=True,
                backend_state="ready",
                backend_name=self.backend.backend_name,
                model=self.backend.model,
                detail="rolling state recent and healthy",
            )
            return health
        fresh = BackendHealth(source="fresh", backend_name=self.backend.backend_name, model=self.backend.model)
        try:
            def _probe():
                return self.backend.chat(
                    messages=[{"role": "user", "content": "Responde solo ok."}],
                    tools=None,
                )
            started = time.perf_counter()
            self._with_backend_timeout(self.settings.backend_preflight_timeout, _probe)
            elapsed = int((time.perf_counter() - started) * 1000)
            health.backend_reachable = True
            health.backend_chat_ready = True
            health.backend_chat_slow = elapsed >= self.settings.backend_slow_threshold_ms
            health.last_success_ms = elapsed
            health.recent_failures = 0
            health.backend_state = "slow" if health.backend_chat_slow else "ready"
            fresh.backend_reachable = True
            fresh.backend_chat_ready = True
            fresh.backend_chat_slow = health.backend_chat_slow
            fresh.last_success_ms = elapsed
            fresh.average_chat_ms = elapsed
            fresh.backend_state = health.backend_state
            fresh.detail = "preflight success"
            self.activity.emit("Backend", "done", "Backend health check", f"ready duration_ms={elapsed}")
        except BackendTimeoutError as exc:
            health.backend_reachable = True
            health.backend_chat_ready = False
            health.backend_chat_slow = True
            health.last_timeout_ms = self.settings.backend_preflight_timeout * 1000
            health.recent_failures += 1
            health.backend_state = "unstable"
            health.last_error = str(exc)
            fresh.backend_reachable = True
            fresh.backend_chat_ready = False
            fresh.backend_chat_slow = True
            fresh.last_timeout_ms = self.settings.backend_preflight_timeout * 1000
            fresh.backend_state = "timeout"
            fresh.last_error = str(exc)
            fresh.detail = "preflight timeout"
            self.activity.emit("Backend", "failed", "Backend unstable", str(exc))
        except BackendError as exc:
            health.backend_reachable = not isinstance(exc, BackendUnreachableError)
            health.backend_chat_ready = False
            health.recent_failures += 1
            health.backend_state = "unstable" if health.backend_reachable else "unreachable"
            health.last_error = str(exc)
            fresh.backend_reachable = health.backend_reachable
            fresh.backend_chat_ready = False
            fresh.backend_state = "unreachable" if isinstance(exc, BackendUnreachableError) else "unstable"
            fresh.last_error = str(exc)
            fresh.detail = "preflight backend error"
            self.activity.emit("Backend", "failed", "Backend unstable", str(exc))
        run_state.fresh_backend_health = fresh
        return health

    def _heuristic_plan(self, session: SessionState, run_state: RunState, reason: str) -> str:
        run_state.fallback_used = True
        run_state.fallback_reason = reason
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
        if mode == "resume":
            self.activity.emit("Resume", "degraded", "Resume state restored without backend completion", reason)
            return "\n".join(
                [
                    "Echo restauró el estado de la sesión, pero no pudo completar con el backend.",
                    f"Objetivo: {session.objective or session.user_prompt}",
                    f"Working set: {', '.join(session.working_set[-8:]) or 'none'}",
                    f"Pendientes: {'; '.join(session.pending[-6:]) or 'none'}",
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
                f"Backend efectivo: {run_state.fresh_backend_health.backend_state or run_state.backend_health.backend_state}",
                f"Límite actual: {reason}",
            ]
        )

    def _grounding_retry_message(self, session: SessionState, run_state: RunState, reason: str) -> dict[str, str]:
        return {
            "role": "system",
            "content": (
                "Last answer was not grounded enough. "
                f"Reason: {reason}. "
                f"Working set: {', '.join(session.working_set[-8:]) or 'none'}. "
                f"Inspected files: {', '.join(run_state.inspected_files[-8:] or session.focus_files[-8:]) or 'none'}. "
                f"Recent tool evidence: {' | '.join(self._collect_tool_previews(session)[-4:]) or 'none'}. "
                f"Validation strategy: {self._validation_strategy(session)}. "
                "Respond with explicit file references and only claims supported by inspected evidence. "
                "Mention at least one concrete symbol if the evidence contains one. "
                "If the backend/model is too weak, say the limit directly instead of inventing."
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
                focus_files=session.working_set or session.focus_files,
                changed_files=session.changed_files or run_state.changed_files,
                errors=session.errors or run_state.errors,
                pending=session.pending or run_state.pending,
            )
            if compressed:
                session.operational_summary = compressed
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
                    tool_result_previews=self._collect_tool_previews(session),
                    working_set=session.working_set,
                    validation_strategy=self._validation_strategy(session),
                )
                run_state.grounding_report.grounded_file_count = int(report.get("grounded_file_count", 0))
                run_state.grounding_report.grounded_symbol_count = int(report.get("grounded_symbol_count", 0))
                run_state.grounding_report.evidence_usage = int(report.get("evidence_usage", 0))
                run_state.grounding_report.genericity_score = int(report.get("genericity_score", 0))
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
            return
        self._mark_phase(run_state, "Verifier", "running", "Verifier running validation")
        result = self.tools.execute("validate_project", {})
        self._record_tool_call(session, run_state, "validate_project", {}, result)
        command = str(result.get("validation_command", ""))
        if result.get("returncode", 1) != 0:
            issue = f"Validation failed: {command}"
            run_state.open_issues.append(issue)
            run_state.errors.append(issue)
            session.errors.append(issue)
            self._mark_phase(run_state, "Verifier", "failed", issue)
        else:
            self._mark_phase(run_state, "Verifier", "done", f"Validation passed: {command}")

    def _preflight_degrade_if_needed(self, session: SessionState, run_state: RunState, mode: str) -> str | None:
        health = self._backend_health_check(run_state)
        effective = BackendAvailabilityPolicy.effective_backend_health(health, run_state.fresh_backend_health)
        run_state.backend_health = effective
        if effective.backend_chat_ready:
            if effective.backend_chat_slow:
                self.activity.emit("Backend", "warning", "Backend unstable", "Chat responde lento")
                self.activity.emit("Backend", "retrying", "Retry with reduced context", "Slow backend; shrinking context before first request")
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
        if not session.grounding_report and run_state.grounding_report.reason != "ok":
            session.grounding_report = {
                "grounded_file_count": run_state.grounding_report.grounded_file_count,
                "grounded_symbol_count": run_state.grounding_report.grounded_symbol_count,
                "evidence_usage": run_state.grounding_report.evidence_usage,
                "genericity_score": run_state.grounding_report.genericity_score,
                "validation_strategy_match": run_state.grounding_report.validation_strategy_match,
                "validation_strategy": run_state.grounding_report.validation_strategy,
                "speculation_flags": run_state.grounding_report.speculation_flags,
                "valid": run_state.grounding_report.valid,
                "reason": run_state.grounding_report.reason,
            }
        session.routing = asdict(run_state.routing)
        session.health = {
            "effective": asdict(run_state.backend_health),
            "fresh": asdict(run_state.fresh_backend_health),
            "cached": asdict(self.store.read_backend_health()),
        }
        session.retry_count = run_state.retry_count
        session.compression_count = run_state.compression_count
        session.artifacts = [asdict(item) for item in run_state.artifacts]
        session.summary = build_session_summary(session, final_answer)
        if not session.operational_summary:
            session.operational_summary = session.summary
        self._write_runtime_artifact(
            run_state,
            "runtime",
            {
                "session_id": session.id,
                "mode": session.mode,
                "routing": session.routing,
                "health": session.health,
                "working_set": session.working_set,
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
        run_state.decisions.append("Use repo evidence and minimal patches.")
        if not run_state.routing.selected_backend:
            run_state.routing.selected_backend = self.backend.backend_name
            run_state.routing.selected_model = self.backend.model
        messages = self._intake(session, run_state, prompt) if loaded is None else list(session.messages)
        degraded = self._preflight_degrade_if_needed(session, run_state, session.mode)
        if degraded is not None:
            final_answer = degraded
        else:
            if run_state.backend_health.backend_chat_slow:
                messages = self._reduce_context(session, run_state, messages)
                session.working_set = session.working_set[-3:] or session.focus_files[-3:]
            final_answer = self._run_model_loop(session, run_state, messages, prompt, session.mode)
        self._auto_verify(session, run_state)
        return (*self._finalize(session, run_state, final_answer), run_state)
