from __future__ import annotations

from dataclasses import replace

from echo.types import BackendHealth


POSITIVE_CHAT_STATES = {"ready", "slow"}
TERMINAL_FRESH_STATES = {"timeout", "unstable", "unreachable", "error", "malformed"}


def normalize_backend_health(health: BackendHealth, *, source: str | None = None) -> BackendHealth:
    normalized = replace(health)
    if source is not None:
        normalized.source = source

    state = (normalized.backend_state or "").strip().lower()
    if normalized.backend_chat_ready:
        normalized.backend_reachable = True
        normalized.backend_state = "slow" if normalized.backend_chat_slow else "ready"
        return normalized

    if state in POSITIVE_CHAT_STATES:
        normalized.backend_reachable = True
        normalized.backend_chat_ready = True
        normalized.backend_chat_slow = state == "slow"
        normalized.backend_state = state
        return normalized
    if state == "reachable":
        normalized.backend_reachable = True
        normalized.backend_chat_ready = False
        normalized.backend_chat_slow = False
        normalized.backend_state = "reachable"
        return normalized
    if state == "timeout":
        normalized.backend_reachable = True
        normalized.backend_chat_ready = False
        normalized.backend_chat_slow = True
        normalized.backend_state = "timeout"
        return normalized
    if state == "unstable":
        normalized.backend_reachable = True
        normalized.backend_chat_ready = False
        normalized.backend_state = "unstable"
        return normalized
    if state == "unreachable":
        normalized.backend_reachable = False
        normalized.backend_chat_ready = False
        normalized.backend_chat_slow = False
        normalized.backend_state = "unreachable"
        return normalized
    if state in {"error", "malformed"}:
        normalized.backend_chat_ready = False
        normalized.backend_state = state
        return normalized
    if normalized.backend_reachable:
        normalized.backend_chat_ready = False
        normalized.backend_chat_slow = False
        normalized.backend_state = "reachable"
        return normalized
    normalized.backend_chat_ready = False
    normalized.backend_chat_slow = False
    normalized.backend_state = "unknown"
    return normalized


def effective_backend_health(rolling: BackendHealth, fresh: BackendHealth | None = None) -> BackendHealth:
    normalized_rolling = normalize_backend_health(rolling, source=rolling.source or "rolling")
    if fresh is None:
        return normalized_rolling
    normalized_fresh = normalize_backend_health(fresh, source=fresh.source or "fresh")
    if normalized_fresh.backend_state in POSITIVE_CHAT_STATES | TERMINAL_FRESH_STATES:
        return normalized_fresh
    if normalized_fresh.backend_state == "reachable":
        if normalized_rolling.backend_state in POSITIVE_CHAT_STATES:
            merged = replace(normalized_rolling)
            if normalized_fresh.tags_latency_ms:
                merged.tags_latency_ms = normalized_fresh.tags_latency_ms
            return normalize_backend_health(merged, source=normalized_rolling.source or "rolling")
        return normalized_fresh
    return normalized_rolling


def backend_log_state(
    *,
    backend_reachable: bool,
    backend_chat_ready: bool,
    backend_chat_slow: bool = False,
    backend_state: str | None = None,
    **kwargs,
) -> dict[str, object]:
    health = normalize_backend_health(
        BackendHealth(
            backend_reachable=backend_reachable,
            backend_chat_ready=backend_chat_ready,
            backend_chat_slow=backend_chat_slow,
            backend_state=backend_state or "",
        )
    )
    payload: dict[str, object] = {
        "backend_reachable": health.backend_reachable,
        "backend_chat_ready": health.backend_chat_ready,
        "backend_chat_slow": health.backend_chat_slow,
        "backend_state": health.backend_state,
    }
    payload.update(kwargs)
    return payload
