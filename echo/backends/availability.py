from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from pathlib import Path
from typing import Any

from echo.backends.errors import BackendError, BackendTimeoutError, BackendUnreachableError
from echo.backends.health import (
    backend_log_state,
    effective_backend_health as merged_backend_health,
    normalize_backend_health,
)
from typing import TYPE_CHECKING

from echo.config import Settings
from echo.types import BackendHealth

if TYPE_CHECKING:
    from echo.memory import EchoStore


@dataclass(slots=True)
class BackendCheckResult:
    backend_name: str
    model: str
    backend_reachable: bool
    backend_chat_ready: bool
    backend_chat_slow: bool
    backend_state: str
    tags_latency_ms: int = 0
    last_success_ms: int = 0
    last_timeout_ms: int = 0
    average_chat_ms: int = 0
    success_rate: float = 0.0
    recent_failures: int = 0
    timeout_active: int = 0
    warm_state: str = "unknown"
    cold_chat_ms: int = 0
    warm_chat_ms: int = 0
    chat_probe_count: int = 0
    failure_reasons: list[str] = field(default_factory=list)
    artifact_path: str = ""
    rolling_health: dict[str, Any] = field(default_factory=dict)
    fresh_health: dict[str, Any] = field(default_factory=dict)
    effective_decision: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendRouteDecision:
    primary_backend: str
    primary_model: str
    selected_backend: str
    selected_model: str
    fallback_backend: str = ""
    fallback_model: str = ""
    fallback_available: bool = False
    fallback_selected: bool = False
    policy: str = "primary-only"
    task_complexity: str = "low"
    reason: str = ""
    health_source: str = "rolling"


class BackendAvailabilityPolicy:
    @staticmethod
    def classify_mode(mode: str, health: BackendHealth) -> str:
        if health.backend_chat_ready:
            return "normal"
        if mode == "plan":
            return "heuristic_plan"
        if mode == "resume":
            return "resume_restore_only"
        return "local_inspection_only"

    @staticmethod
    def route_backend(
        settings: Settings,
        *,
        profile: str,
        mode: str,
        prompt: str,
        primary_backend: str,
        primary_model: str,
        primary_health: BackendHealth,
    ) -> BackendRouteDecision:
        fallback_backend, fallback_model = settings.fallback_spec()
        fallback_available = fallback_backend in {"openai", "openai-compatible"} and bool(settings.openai_api_key)
        complexity = infer_task_complexity(mode, prompt, profile)
        selected_backend = primary_backend
        selected_model = primary_model
        fallback_selected = False
        reason = "using primary backend"
        policy = "primary-only"

        if mode == "ask":
            critical_ask = profile in {"balanced", "deep"} or complexity in {"medium", "high"}
            if critical_ask and fallback_available and (not primary_health.backend_chat_ready or primary_health.backend_chat_slow):
                selected_backend = fallback_backend
                selected_model = fallback_model
                fallback_selected = True
                policy = "hybrid-fallback"
                if not primary_health.backend_chat_ready:
                    reason = "primary backend chat not ready for critical ask"
                else:
                    reason = "primary backend chat is slow for critical ask"
            elif critical_ask:
                policy = "primary-only-critical"
                if not primary_health.backend_chat_ready:
                    reason = "fallback unavailable; keep primary and let runtime degrade"
                elif primary_health.backend_chat_slow:
                    reason = "fallback unavailable; keep primary despite slow chat"
        elif mode in {"plan", "resume"}:
            policy = "local-first-degrade"
            reason = "plan/resume prefer local execution and controlled degradation"

        return BackendRouteDecision(
            primary_backend=primary_backend,
            primary_model=primary_model,
            selected_backend=selected_backend,
            selected_model=selected_model,
            fallback_backend=fallback_backend,
            fallback_model=fallback_model,
            fallback_available=fallback_available,
            fallback_selected=fallback_selected,
            policy=policy,
            task_complexity=complexity,
            reason=reason,
            health_source=primary_health.source or "rolling",
        )

    @staticmethod
    def effective_backend_health(rolling: BackendHealth, fresh: BackendHealth | None = None) -> BackendHealth:
        return merged_backend_health(rolling, fresh)


def quick_health_probe(backend: Any, settings: Settings, *, include_chat: bool = True) -> BackendHealth:
    health = BackendHealth(
        source="fresh",
        backend_name=getattr(backend, "backend_name", settings.backend),
        model=getattr(backend, "model", settings.model),
    )
    started = time.perf_counter()
    try:
        backend.list_models()
        health.backend_reachable = True
        health.tags_latency_ms = int((time.perf_counter() - started) * 1000)
    except BackendTimeoutError as exc:
        health.backend_reachable = True
        health.backend_state = "timeout"
        health.last_timeout_ms = int((time.perf_counter() - started) * 1000)
        health.last_error = str(exc)
        health.detail = "fresh tags probe timeout"
        return normalize_backend_health(health, source="fresh")
    except BackendUnreachableError as exc:
        health.backend_reachable = False
        health.backend_state = "unreachable"
        health.last_error = str(exc)
        health.detail = "fresh tags probe unreachable"
        return normalize_backend_health(health, source="fresh")
    except BackendError as exc:
        health.backend_reachable = False
        health.backend_state = "error"
        health.last_error = str(exc)
        health.detail = "fresh tags probe error"
        return normalize_backend_health(health, source="fresh")

    if not include_chat:
        health.backend_state = "reachable"
        health.detail = "fresh tags probe only"
        return normalize_backend_health(health, source="fresh")

    started = time.perf_counter()
    try:
        backend.chat(messages=[{"role": "user", "content": "Responde solo ok."}], tools=None)
        elapsed = int((time.perf_counter() - started) * 1000)
        health.backend_chat_ready = True
        health.last_success_ms = elapsed
        health.average_chat_ms = elapsed
        health.backend_chat_slow = elapsed >= settings.backend_slow_threshold_ms
        health.backend_state = "slow" if health.backend_chat_slow else "ready"
        health.detail = "fresh chat probe success"
    except BackendTimeoutError as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        health.backend_reachable = True
        health.backend_chat_ready = False
        health.backend_chat_slow = True
        health.last_timeout_ms = elapsed
        health.backend_state = "timeout"
        health.last_error = str(exc)
        health.detail = "fresh chat probe timeout"
    except BackendUnreachableError as exc:
        health.backend_reachable = True
        health.backend_chat_ready = False
        health.backend_state = "unstable"
        health.last_error = str(exc)
        health.detail = "fresh chat probe unreachable"
    except BackendError as exc:
        health.backend_reachable = True
        health.backend_chat_ready = False
        health.backend_state = "unstable"
        health.last_error = str(exc)
        health.detail = "fresh chat probe error"
    return normalize_backend_health(health, source="fresh")


def infer_task_complexity(mode: str, prompt: str, profile: str) -> str:
    if mode != "ask":
        return "low"
    tokens = len((prompt or "").split())
    low = (prompt or "").lower()
    signals = [
        "arquitect",
        "refactor",
        "runtime",
        "backend",
        "core",
        "persist",
        "diagn",
        "inspect",
        "shell",
        "agent",
    ]
    score = sum(1 for item in signals if item in low)
    if profile == "deep" or score >= 3 or tokens >= 20:
        return "high"
    if profile == "balanced" or score >= 1 or tokens >= 8:
        return "medium"
    return "low"


def run_backend_check(
    backend: Any,
    settings: Settings,
    store: EchoStore,
    *,
    chat_samples: int = 2,
) -> BackendCheckResult:
    rolling_health = store.read_backend_health()
    tags_latency_ms = 0
    backend_reachable = False
    failure_reasons: list[str] = []
    models: list[str] = []

    started = time.perf_counter()
    try:
        models = backend.list_models()
        backend_reachable = True
    except BackendError as exc:
        failure_reasons.append(str(exc))
    tags_latency_ms = int((time.perf_counter() - started) * 1000)
    store.append_backend_log(
        {
            "event": "tags_check",
            "backend": getattr(backend, "backend_name", settings.backend),
            "model": getattr(backend, "model", settings.model),
            "duration_ms": tags_latency_ms,
            "ok": backend_reachable,
            "detail": failure_reasons[-1] if failure_reasons else "",
            **backend_log_state(
                backend_reachable=backend_reachable,
                backend_chat_ready=False,
                backend_state="reachable" if backend_reachable else "unreachable",
            ),
        }
    )

    chat_successes: list[int] = []
    last_timeout_ms = 0
    for index in range(chat_samples):
        started = time.perf_counter()
        try:
            backend.chat(
                messages=[{"role": "user", "content": "Responde solo ok."}],
                tools=None,
            )
            elapsed = int((time.perf_counter() - started) * 1000)
            chat_successes.append(elapsed)
            store.append_backend_log(
                {
                    "event": "chat_probe",
                    "backend": getattr(backend, "backend_name", settings.backend),
                    "model": getattr(backend, "model", settings.model),
                    "duration_ms": elapsed,
                    "probe_index": index,
                    "ok": True,
                    **backend_log_state(
                        backend_reachable=True,
                        backend_chat_ready=True,
                        backend_chat_slow=elapsed >= settings.backend_slow_threshold_ms,
                    ),
                }
            )
        except BackendTimeoutError as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            last_timeout_ms = elapsed
            failure_reasons.append(str(exc))
            store.append_backend_log(
                {
                    "event": "chat_probe_timeout",
                    "backend": getattr(backend, "backend_name", settings.backend),
                    "model": getattr(backend, "model", settings.model),
                    "duration_ms": elapsed,
                    "probe_index": index,
                    "ok": False,
                    "detail": str(exc),
                    **backend_log_state(
                        backend_reachable=True,
                        backend_chat_ready=False,
                        backend_chat_slow=True,
                        backend_state="timeout",
                    ),
                }
            )
        except BackendError as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            failure_reasons.append(str(exc))
            store.append_backend_log(
                {
                    "event": "chat_probe_error",
                    "backend": getattr(backend, "backend_name", settings.backend),
                    "model": getattr(backend, "model", settings.model),
                    "duration_ms": elapsed,
                    "probe_index": index,
                    "ok": False,
                    "detail": str(exc),
                    **backend_log_state(
                        backend_reachable=not isinstance(exc, BackendUnreachableError),
                        backend_chat_ready=False,
                        backend_state="unreachable" if isinstance(exc, BackendUnreachableError) else "unstable",
                    ),
                }
            )

    chat_probe_count = chat_samples
    success_rate = len(chat_successes) / max(chat_probe_count, 1)
    average_chat_ms = int(sum(chat_successes) / max(len(chat_successes), 1)) if chat_successes else 0
    cold_chat_ms = chat_successes[0] if chat_successes else 0
    warm_chat_ms = chat_successes[1] if len(chat_successes) > 1 else 0
    warm_state = "unknown"
    if len(chat_successes) >= 2:
        if warm_chat_ms and cold_chat_ms >= warm_chat_ms + 1000:
            warm_state = "cold_then_warm"
        else:
            warm_state = "warm_stable"
    elif len(chat_successes) == 1:
        warm_state = "warm_or_single_success"

    backend_chat_ready = bool(chat_successes)
    backend_chat_slow = average_chat_ms >= settings.backend_slow_threshold_ms if chat_successes else False
    backend_state = "ready"
    if not backend_reachable:
        backend_state = "unreachable"
    elif not backend_chat_ready:
        backend_state = "unstable"
    elif backend_chat_slow:
        backend_state = "slow"

    result = BackendCheckResult(
        backend_name=getattr(backend, "backend_name", settings.backend),
        model=getattr(backend, "model", settings.model),
        backend_reachable=backend_reachable,
        backend_chat_ready=backend_chat_ready,
        backend_chat_slow=backend_chat_slow,
        backend_state=backend_state,
        tags_latency_ms=tags_latency_ms,
        last_success_ms=chat_successes[-1] if chat_successes else 0,
        last_timeout_ms=last_timeout_ms,
        average_chat_ms=average_chat_ms,
        success_rate=success_rate,
        recent_failures=chat_probe_count - len(chat_successes),
        timeout_active=settings.backend_timeout,
        warm_state=warm_state,
        cold_chat_ms=cold_chat_ms,
        warm_chat_ms=warm_chat_ms,
        chat_probe_count=chat_probe_count,
        failure_reasons=failure_reasons[-6:],
    )
    result.rolling_health = asdict(rolling_health)
    result.fresh_health = {
        "source": "fresh",
        "backend_reachable": result.backend_reachable,
        "backend_chat_ready": result.backend_chat_ready,
        "backend_chat_slow": result.backend_chat_slow,
        "backend_state": result.backend_state,
        "last_success_ms": result.last_success_ms,
        "last_timeout_ms": result.last_timeout_ms,
        "average_chat_ms": result.average_chat_ms,
        "recent_failures": result.recent_failures,
        "success_rate": result.success_rate,
        "chat_probe_count": result.chat_probe_count,
        "tags_latency_ms": result.tags_latency_ms,
        "warm_state": result.warm_state,
        "backend_name": result.backend_name,
        "model": result.model,
        "detail": result.failure_reasons[-1] if result.failure_reasons else result.backend_state,
    }
    effective = BackendAvailabilityPolicy.effective_backend_health(
        rolling_health,
        BackendHealth(**result.fresh_health),
    )
    result.effective_decision = {
        "effective_backend_state": effective.backend_state,
        "effective_backend_reachable": effective.backend_reachable,
        "effective_backend_chat_ready": effective.backend_chat_ready,
        "effective_backend_chat_slow": effective.backend_chat_slow,
        "effective_source": effective.source,
    }

    artifact_path = store.artifacts / f"backend-check-{int(time.time())}.json"
    result.artifact_path = str(artifact_path)
    write_backend_check_artifact(artifact_path, result, models)
    store.append_backend_log(
        {
            "event": "backend_check_summary",
            "backend": result.backend_name,
            "model": result.model,
            "duration_ms": result.average_chat_ms,
            "backend_reachable": result.backend_reachable,
            "backend_chat_ready": result.backend_chat_ready,
            "backend_chat_slow": result.backend_chat_slow,
            "backend_state": result.backend_state,
            "average_chat_ms": result.average_chat_ms,
            "last_success_ms": result.last_success_ms,
            "last_timeout_ms": result.last_timeout_ms,
            "recent_failures": result.recent_failures,
            "success_rate": result.success_rate,
            "warm_state": result.warm_state,
            "detail": result.failure_reasons[-1] if result.failure_reasons else result.backend_state,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **backend_log_state(
                backend_reachable=result.backend_reachable,
                backend_chat_ready=result.backend_chat_ready,
                backend_chat_slow=result.backend_chat_slow,
                backend_state=result.backend_state,
            ),
        }
    )
    return result


def write_backend_check_artifact(path: Path, result: BackendCheckResult, models: list[str]) -> Path:
    payload = asdict(result)
    payload["available_models"] = models
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
