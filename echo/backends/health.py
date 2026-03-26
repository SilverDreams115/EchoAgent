from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

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


def rolling_backend_health_from_log(backend_log: Path) -> BackendHealth:
    health = BackendHealth(source="rolling")
    if not backend_log.exists():
        return normalize_backend_health(health, source="rolling")
    lines = [line for line in backend_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    recent = lines[-12:]
    failures = 0
    for raw in recent:
        try:
            item = json.loads(raw)
        except Exception:
            continue
        health.checked_at = str(item.get("created_at", health.checked_at))
        event = str(item.get("event", ""))
        duration_ms = int(item.get("duration_ms", 0) or 0)
        health.backend_name = str(item.get("backend", health.backend_name))
        health.model = str(item.get("model", health.model))
        explicit_state = item.get("backend_state")
        explicit_reachable = item.get("backend_reachable")
        explicit_chat_ready = item.get("backend_chat_ready")
        explicit_chat_slow = item.get("backend_chat_slow")
        if explicit_reachable is not None:
            health.backend_reachable = bool(explicit_reachable)
        if explicit_chat_ready is not None:
            health.backend_chat_ready = bool(explicit_chat_ready)
        if explicit_chat_slow is not None:
            health.backend_chat_slow = bool(explicit_chat_slow)
        if explicit_state is not None:
            health.backend_state = str(explicit_state)
        if event == "response":
            health.last_success_ms = duration_ms
            failures = 0
        elif event == "backend_check_summary":
            health.last_success_ms = int(item.get("last_success_ms", health.last_success_ms) or 0)
            health.last_timeout_ms = int(item.get("last_timeout_ms", health.last_timeout_ms) or 0)
            health.average_chat_ms = int(item.get("average_chat_ms", health.average_chat_ms) or 0)
            health.success_rate = float(item.get("success_rate", health.success_rate) or 0.0)
            health.warm_state = str(item.get("warm_state", health.warm_state))
            health.chat_probe_count = int(item.get("chat_probe_count", health.chat_probe_count) or 0)
            health.recent_failures = int(item.get("recent_failures", health.recent_failures) or 0)
        elif event == "timeout":
            failures += 1
            if explicit_state is None:
                health.backend_reachable = True
                health.backend_chat_ready = False
                health.backend_chat_slow = True
                health.backend_state = "timeout"
            health.last_timeout_ms = duration_ms
            health.last_error = "timeout"
            health.detail = "recent timeout"
        elif event == "tags_check":
            health.tags_latency_ms = duration_ms
            if explicit_state is None:
                health.backend_reachable = bool(item.get("ok", False))
                health.backend_state = "reachable" if health.backend_reachable else "unreachable"
            if not health.backend_reachable:
                health.last_error = str(item.get("detail", health.last_error))
        elif event == "chat_probe":
            if explicit_state is None:
                health.backend_reachable = True
                health.backend_chat_ready = True
                health.backend_state = "ready"
            health.last_success_ms = duration_ms
        elif event == "chat_probe_timeout":
            failures += 1
            if explicit_state is None:
                health.backend_reachable = True
                health.backend_chat_ready = False
                health.backend_chat_slow = True
                health.backend_state = "timeout"
            health.last_timeout_ms = duration_ms
            health.last_error = str(item.get("detail", "timeout"))
            health.detail = "chat probe timeout"
        elif event == "chat_probe_error":
            failures += 1
            if explicit_state is None:
                unreachable = "no es alcanzable" in str(item.get("detail", "")).lower()
                health.backend_reachable = not unreachable
                health.backend_chat_ready = False
                health.backend_state = "unreachable" if unreachable else "unstable"
            health.last_error = str(item.get("detail", health.last_error))
            health.detail = "chat probe error"
        elif event == "error":
            failures += 1
            detail = str(item.get("detail", ""))
            if explicit_state is None:
                unreachable = "no es alcanzable" in detail.lower()
                health.backend_reachable = not unreachable
                health.backend_chat_ready = False
                health.backend_state = "unreachable" if unreachable else "unstable"
            health.last_error = detail
            health.detail = detail
    health.recent_failures = failures
    return normalize_backend_health(health, source="rolling")
