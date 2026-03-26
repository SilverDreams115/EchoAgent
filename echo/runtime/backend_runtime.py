from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Callable

from echo.backends import BackendAvailabilityPolicy, backend_log_state, quick_health_probe
from echo.backends.errors import BackendError, BackendTimeoutError, BackendUnreachableError
from echo.types import BackendHealth, RunState


def _append_backend_log(store, payload: dict[str, Any]) -> None:
    payload.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    store.append_backend_log(payload)


def _emit_backend_activity(activity, status: str, message: str, detail: str) -> None:
    activity.emit("Backend", status, message, detail)


def backend_health_check(*, backend, settings, activity, run_state: RunState) -> BackendHealth:
    rolling = run_state.backend_health
    rolling.source = "rolling"
    _emit_backend_activity(activity, "running", "Backend health check", rolling.backend_state or "unknown")
    fresh = quick_health_probe(backend, settings, include_chat=False)
    run_state.fresh_backend_health = fresh
    effective = BackendAvailabilityPolicy.effective_backend_health(rolling, fresh)
    if effective.backend_chat_ready:
        _emit_backend_activity(activity, "done", "Backend health check", f"{effective.backend_state} duration_ms={effective.last_success_ms}")
    else:
        _emit_backend_activity(activity, "failed", "Backend health check", effective.last_error or effective.backend_state)
    return effective


def perform_backend_request(
    *,
    backend,
    settings,
    store,
    activity,
    run_state: RunState,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout_seconds: int,
    record_backend_request: Callable[..., None],
) -> dict[str, Any]:
    previous = getattr(backend, "timeout", None)
    if previous is not None:
        backend.timeout = timeout_seconds
    start = time.perf_counter()
    _emit_backend_activity(activity, "running", "Backend request started", f"model={backend.model} timeout_s={timeout_seconds}")
    _emit_backend_activity(activity, "waiting", "Backend waiting", f"messages={len(messages)} tools={bool(tools)}")
    try:
        response = backend.chat(messages=messages, tools=tools)
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
        _append_backend_log(
            store,
            {
                "event": "timeout",
                "backend": backend.backend_name,
                "model": backend.model,
                "duration_ms": duration_ms,
                "message_count": len(messages),
                "timeout_seconds": timeout_seconds,
                **backend_log_state(
                    backend_reachable=True,
                    backend_chat_ready=False,
                    backend_chat_slow=True,
                    backend_state="timeout",
                ),
            },
        )
        record_backend_request(
            run_state,
            message_count=len(messages),
            timeout_seconds=timeout_seconds,
            tools_enabled=bool(tools),
            duration_ms=duration_ms,
            outcome="timeout",
            detail=str(exc),
        )
        _emit_backend_activity(activity, "failed", "Backend timeout", f"duration_ms={duration_ms}")
        raise
    except BackendError as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        run_state.backend_duration_ms.append(duration_ms)
        run_state.backend_health.recent_failures += 1
        run_state.backend_health.last_error = str(exc)
        run_state.backend_health.backend_reachable = not isinstance(exc, BackendUnreachableError)
        run_state.backend_health.backend_chat_ready = False
        run_state.backend_health.backend_state = "unreachable" if isinstance(exc, BackendUnreachableError) else "error"
        _append_backend_log(
            store,
            {
                "event": "error",
                "backend": backend.backend_name,
                "model": backend.model,
                "duration_ms": duration_ms,
                "message_count": len(messages),
                "timeout_seconds": timeout_seconds,
                "detail": exc.detail or str(exc),
                **backend_log_state(
                    backend_reachable=not isinstance(exc, BackendUnreachableError),
                    backend_chat_ready=False,
                    backend_state="unreachable" if isinstance(exc, BackendUnreachableError) else "error",
                ),
            },
        )
        record_backend_request(
            run_state,
            message_count=len(messages),
            timeout_seconds=timeout_seconds,
            tools_enabled=bool(tools),
            duration_ms=duration_ms,
            outcome="unreachable" if isinstance(exc, BackendUnreachableError) else "error",
            detail=exc.detail or str(exc),
        )
        _emit_backend_activity(activity, "failed", "Backend request failed", str(exc))
        raise
    finally:
        if previous is not None:
            backend.timeout = previous
    duration_ms = int((time.perf_counter() - start) * 1000)
    run_state.backend_duration_ms.append(duration_ms)
    run_state.backend_health.backend_reachable = True
    run_state.backend_health.backend_chat_ready = True
    run_state.backend_health.backend_chat_slow = duration_ms >= settings.backend_slow_threshold_ms
    run_state.backend_health.last_success_ms = duration_ms
    run_state.backend_health.recent_failures = 0
    run_state.backend_health.backend_state = "slow" if run_state.backend_health.backend_chat_slow else "ready"
    _append_backend_log(
        store,
        {
            "event": "response",
            "backend": backend.backend_name,
            "model": backend.model,
            "duration_ms": duration_ms,
            "message_count": len(messages),
            "timeout_seconds": timeout_seconds,
            **backend_log_state(
                backend_reachable=True,
                backend_chat_ready=True,
                backend_chat_slow=run_state.backend_health.backend_chat_slow,
            ),
        },
    )
    record_backend_request(
        run_state,
        message_count=len(messages),
        timeout_seconds=timeout_seconds,
        tools_enabled=bool(tools),
        duration_ms=duration_ms,
        outcome="slow" if run_state.backend_health.backend_chat_slow else "response",
        detail=run_state.backend_health.backend_state,
    )
    _emit_backend_activity(activity, "done", "Backend response received", f"duration_ms={duration_ms}")
    return response
