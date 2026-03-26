from __future__ import annotations

from pathlib import Path
from typing import Any

from echo.backends import BackendAvailabilityPolicy, build_backend, quick_health_probe, run_backend_check
from echo.config import Settings
from echo.memory import EchoStore
from echo.runtime import ActivityBus, AgentRuntime
from echo.tools import ToolRegistry
from echo.types import RunState, SessionState


class EchoAgent:
    def __init__(self, project_root: Path, settings: Settings) -> None:
        self.project_root = project_root.resolve()
        self.settings = settings
        self.activity = ActivityBus()
        self.store = EchoStore(self.project_root)
        self.activity.watch(
            lambda event: self.store.append_activity_log(
                {
                    "stage": event.stage,
                    "status": event.status,
                    "message": event.message,
                    "detail": event.detail,
                    "created_at": event.created_at,
                }
            )
        )
        self.tools = ToolRegistry(self.project_root, settings, self.activity)
        self.profile = settings.profile
        self.backend, self.resolved_profile, self.resolved_backend_name, self.resolved_model, self.profile_note = self._resolve_backend(self.profile)
        self.runtime = AgentRuntime(self.project_root, settings, self.backend, self.store, self.tools, self.activity)
        self.last_session_id: str | None = self.store.latest_session_id()
        self.last_context_ratio: float = 1.0
        self.last_run_state: RunState | None = None
        self.routing_policy: str = "primary-only"
        self.primary_backend_name = self.resolved_backend_name
        self.primary_model = self.resolved_model
        self.fallback_backend_name, self.fallback_model = self.settings.fallback_spec()
        self.selected_backend_name = self.resolved_backend_name
        self.selected_model = self.resolved_model
        self.routing_reason = "using primary backend"

    def _resolve_backend(self, requested_profile: str, mode: str = "ask") -> tuple[Any, str, str, str, str]:
        profile = requested_profile.lower()
        note = "exact"
        if profile == "auto":
            profile = "local" if mode in {"ask", "shell"} else "balanced"
            note = "auto-selected profile"
        if profile not in {"local", "balanced", "deep"}:
            profile = self.settings.profile
            note = f"unknown profile requested, using {profile}"
        backend_name, model = self.settings.profile_spec(profile)
        if backend_name in {"openai", "openai-compatible"} and not self.settings.openai_api_key:
            missing = f"profile {profile} requires ECHO_OPENAI_API_KEY for backend {backend_name}"
            if self.settings.strict_profile:
                raise RuntimeError(missing)
            if profile == "deep":
                profile = "balanced"
            elif profile == "balanced":
                profile = "local"
            backend_name, model = self.settings.profile_spec(profile)
            note = f"fallback: {missing}"
        backend = build_backend(self.settings, backend_name=backend_name, model=model)
        return backend, profile, backend_name, model, note

    def _bind_runtime(self, requested_profile: str, mode: str, prompt: str = "") -> None:
        backend, resolved_profile, backend_name, model, note = self._resolve_backend(requested_profile, mode=mode)
        rolling_health = self.store.read_backend_health()
        route = BackendAvailabilityPolicy.route_backend(
            self.settings,
            profile=resolved_profile,
            mode=mode,
            prompt=prompt,
            primary_backend=backend_name,
            primary_model=model,
            primary_health=rolling_health,
        )
        if route.fallback_selected:
            backend = build_backend(self.settings, backend_name=route.selected_backend, model=route.selected_model)
            note = f"{note}; fallback selected: {route.reason}"
            self.activity.emit("Routing", "done", "Fallback backend selected", f"{route.selected_backend}:{route.selected_model}")
        else:
            self.activity.emit("Routing", "done", "Selected backend", f"{route.selected_backend}:{route.selected_model}")
        self.activity.emit("Routing", "done", "Reason for routing", route.reason)
        self.backend = backend
        self.profile = requested_profile
        self.settings.profile = resolved_profile
        self.settings.backend = route.selected_backend
        self.settings.model = route.selected_model
        self.resolved_profile = resolved_profile
        self.resolved_backend_name = route.selected_backend
        self.resolved_model = route.selected_model
        self.profile_note = note
        self.primary_backend_name = route.primary_backend
        self.primary_model = route.primary_model
        self.fallback_backend_name = route.fallback_backend
        self.fallback_model = route.fallback_model
        self.selected_backend_name = route.selected_backend
        self.selected_model = route.selected_model
        self.routing_policy = route.policy
        self.routing_reason = route.reason
        self.store.append_routing_log(
            {
                "primary_backend": route.primary_backend,
                "primary_model": route.primary_model,
                "selected_backend": route.selected_backend,
                "selected_model": route.selected_model,
                "fallback_backend": route.fallback_backend,
                "fallback_model": route.fallback_model,
                "fallback_selected": route.fallback_selected,
                "fallback_available": route.fallback_available,
                "policy": route.policy,
                "task_complexity": route.task_complexity,
                "reason": route.reason,
                "health_source": route.health_source,
            }
        )
        self.runtime = AgentRuntime(self.project_root, self.settings, self.backend, self.store, self.tools, self.activity)
        self.runtime.route_decision = route

    def doctor(self) -> dict[str, Any]:
        rolling = self.store.read_backend_health()
        fresh = quick_health_probe(self.backend, self.settings, include_chat=True)
        effective = BackendAvailabilityPolicy.effective_backend_health(rolling, fresh)
        try:
            models = self.backend.list_models()
            backend_ok = True
            backend_error = ""
        except Exception as exc:
            models = []
            backend_ok = False
            backend_error = str(exc)
            effective = BackendAvailabilityPolicy.effective_backend_health(rolling, fresh)
        recommended_model = self._recommended_model(models)
        return {
            "project_root": str(self.project_root),
            "profile": self.profile,
            "resolved_profile": self.resolved_profile,
            "deep_ready": bool(self.settings.openai_api_key),
            "strict_profile": self.settings.strict_profile,
            "profile_note": self.profile_note,
            "backend": self.settings.backend,
            "backend_primary": self.primary_backend_name,
            "backend_fallback": self.fallback_backend_name if self.fallback_backend_name else "none",
            "backend_policy": self.routing_policy,
            "routing_reason": self.routing_reason,
            "backend_label": getattr(self.backend, "backend_name", self.settings.backend),
            "backend_tools": getattr(self.backend, "supports_tools", False),
            "backend_native_tools": getattr(self.backend, "supports_native_tools", False),
            "ollama_url": self.settings.ollama_url,
            "openai_url": self.settings.openai_url,
            "model": self.resolved_model,
            "backend_reachable": effective.backend_reachable,
            "backend_chat_ready": effective.backend_chat_ready,
            "backend_chat_slow": effective.backend_chat_slow,
            "backend_state": effective.backend_state,
            "backend_tags_reachable": backend_ok,
            "backend_last_success_ms": effective.last_success_ms,
            "backend_last_timeout_ms": effective.last_timeout_ms,
            "backend_recent_failures": effective.recent_failures,
            "backend_average_chat_ms": effective.average_chat_ms,
            "backend_success_rate": effective.success_rate,
            "backend_tags_latency_ms": effective.tags_latency_ms,
            "backend_warm_state": effective.warm_state,
            "backend_health_cached_state": rolling.backend_state,
            "backend_health_cached_ready": rolling.backend_chat_ready,
            "backend_health_cached_source": rolling.source,
            "backend_health_fresh_state": fresh.backend_state,
            "backend_health_fresh_reachable": fresh.backend_reachable,
            "backend_health_fresh_source": fresh.source,
            "backend_health_effective_state": effective.backend_state,
            "backend_health_effective_source": effective.source,
            "backend_timeout_seconds": self.settings.backend_timeout,
            "model_present": self.resolved_model in models if models else False,
            "available_models": models,
            "recommended_model": recommended_model,
            "current_session": self.last_session_id or "none",
            "backend_error": backend_error or fresh.last_error or rolling.last_error or "none",
        }

    def backend_check(self, chat_samples: int = 2) -> dict[str, Any]:
        result = run_backend_check(self.backend, self.settings, self.store, chat_samples=chat_samples)
        effective_health = self.store.read_backend_health()
        route = BackendAvailabilityPolicy.route_backend(
            self.settings,
            profile=self.resolved_profile,
            mode="ask",
            prompt="backend-check routing decision",
            primary_backend=self.primary_backend_name,
            primary_model=self.primary_model,
            primary_health=effective_health,
        )
        return {
            "backend_label": result.backend_name,
            "model": result.model,
            "backend_reachable": result.backend_reachable,
            "backend_chat_ready": result.backend_chat_ready,
            "backend_chat_slow": result.backend_chat_slow,
            "backend_state": result.backend_state,
            "tags_latency_ms": result.tags_latency_ms,
            "last_success_ms": result.last_success_ms,
            "average_chat_ms": result.average_chat_ms,
            "last_timeout_ms": result.last_timeout_ms,
            "recent_failures": result.recent_failures,
            "success_rate": result.success_rate,
            "chat_probe_count": result.chat_probe_count,
            "timeout_active": result.timeout_active,
            "warm_state": result.warm_state,
            "cold_chat_ms": result.cold_chat_ms,
            "warm_chat_ms": result.warm_chat_ms,
            "artifact_path": result.artifact_path,
            "failure_reasons": result.failure_reasons,
            "rolling_health": result.rolling_health,
            "fresh_health": result.fresh_health,
            "effective_decision": result.effective_decision,
            "routing_decision": {
                "selected_backend": route.selected_backend,
                "selected_model": route.selected_model,
                "fallback_selected": route.fallback_selected,
                "policy": route.policy,
                "reason": route.reason,
            },
        }

    def _recommended_model(self, models: list[str]) -> str:
        preferred = [
            "qwen2.5-coder:7b-oh",
            "qwen2.5-coder:7b",
            "qwen3:latest",
        ]
        for name in preferred:
            if name in models:
                return name
        return models[0] if models else self.settings.model

    def current_status(self) -> dict[str, str]:
        run_state = self.last_run_state
        active = self.store.active_branch()
        return {
            "session": self.last_session_id or "none",
            "branch": active.branch_name if active else "none",
            "profile": self.resolved_profile,
            "profile_note": self.profile_note,
            "backend": self.selected_backend_name,
            "model": self.selected_model,
            "context_free": f"{int(self.last_context_ratio * 100)}%",
            "phase": run_state.phases[-1].phase if run_state and run_state.phases else "idle",
            "focus_files": str(len(run_state.focus_files)) if run_state else "0",
            "backend_state": run_state.backend_health.backend_state if run_state else self.store.read_backend_health().backend_state,
            "routing_policy": self.routing_policy,
        }

    def run(
        self,
        prompt: str,
        mode: str = "ask",
        resume_session_id: str | None = None,
        profile: str | None = None,
        branch_name: str | None = None,
    ) -> tuple[str, Path, SessionState]:
        if profile:
            self.settings.profile = profile
        self._bind_runtime(profile or self.settings.profile, mode, prompt)

        # Explicit resume_session_id wins; otherwise resume from branch head.
        effective_resume_id = resume_session_id
        if branch_name and effective_resume_id is None and mode in {"ask", "plan", "resume"}:
            head = self.store.branch_head_session_id(branch_name)
            if head:
                effective_resume_id = head

        answer, session_path, session, run_state = self.runtime.run(
            prompt, mode=mode, resume_session_id=effective_resume_id
        )

        # Attach branch provenance and advance the branch head.
        if branch_name:
            session.branch_name = branch_name
            self.store.save_session(session)
            self.store.update_branch_head(branch_name, session.id)

        self.last_session_id = session.id
        self.last_run_state = run_state
        self.last_context_ratio = run_state.context_free_ratio
        return answer, session_path, session
