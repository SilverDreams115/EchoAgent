from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from echo.types import ActivityEvent, BackendHealth, SessionState, ToolCallRecord


class EchoStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.base = project_root / ".echo"
        self.sessions = self.base / "sessions"
        self.memory = self.base / "memory"
        self.artifacts = self.base / "artifacts"
        self.logs = self.base / "logs"
        self.cache = self.base / "cache"
        self.state = self.base / "state"
        for directory in [self.sessions, self.memory, self.artifacts, self.logs, self.cache, self.state]:
            directory.mkdir(parents=True, exist_ok=True)
        self.current_session_file = self.state / "current_session.txt"
        self.command_log = self.logs / "commands.jsonl"
        self.backend_log = self.logs / "backend.jsonl"
        self.activity_log = self.logs / "activity.jsonl"
        self.routing_log = self.logs / "routing.jsonl"
        self.session_log = self.logs / "sessions.jsonl"

    def save_session(self, session: SessionState) -> Path:
        session.updated_at = session.updated_at or session.created_at
        path = self.sessions / f"{session.id}.json"
        path.write_text(json.dumps(asdict(session), ensure_ascii=False, indent=2), encoding="utf-8")
        self.current_session_file.write_text(session.id, encoding="utf-8")
        return path

    def write_summary(self, session: SessionState) -> Path:
        path = self.memory / f"{session.id}.summary.md"
        path.write_text(session.summary, encoding="utf-8")
        return path

    def write_active_memory(self, session: SessionState) -> Path:
        path = self.memory / f"{session.id}.active.md"
        lines = [
            f"# Active memory {session.id}",
            "",
            f"Objective: {session.objective or session.user_prompt}",
            "",
            "## Restrictions",
        ]
        lines.extend(f"- {item}" for item in (session.restrictions[-12:] or ["Keep the workflow grounded in repo evidence."]))
        lines += ["", "## Working set"]
        lines.extend(f"- {item}" for item in (session.working_set[-12:] or ["none"]))
        lines += ["", "## Decisions"]
        lines.extend(f"- {item}" for item in (session.decisions[-12:] or ["none"]))
        lines += ["", "## Findings"]
        lines.extend(f"- {item}" for item in (session.findings[-12:] or ["none"]))
        lines += ["", "## Pending"]
        lines.extend(f"- {item}" for item in (session.pending[-12:] or ["none"]))
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def write_cold_summary(self, session: SessionState) -> Path:
        path = self.memory / f"{session.id}.cold.md"
        lines = [
            f"# Cold memory {session.id}",
            "",
            f"Objective: {session.objective or session.user_prompt}",
            "",
            "## Operational summary",
            session.operational_summary or session.summary,
            "",
            "## Changed files",
        ]
        lines.extend(f"- {item}" for item in (session.changed_files[-20:] or ["none"]))
        lines += ["", "## Errors"]
        lines.extend(f"- {item}" for item in (session.errors[-20:] or ["none"]))
        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")
        session.cold_summary_path = str(path)
        return path

    def load_session(self, session_id: str) -> SessionState:
        path = self.sessions / f"{session_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["tool_calls"] = [ToolCallRecord(**item) for item in data.get("tool_calls", [])]
        data["activity"] = [ActivityEvent(**item) for item in data.get("activity", [])]
        return SessionState(**data)

    def latest_session_id(self) -> str | None:
        if self.current_session_file.exists():
            value = self.current_session_file.read_text(encoding="utf-8").strip()
            return value or None
        sessions = sorted(self.sessions.glob("session-*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not sessions:
            return None
        return sessions[0].stem

    def append_command_log(self, payload: dict[str, object]) -> None:
        with self.command_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_backend_log(self, payload: dict[str, object]) -> None:
        with self.backend_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_activity_log(self, payload: dict[str, object]) -> None:
        with self.activity_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_routing_log(self, payload: dict[str, object]) -> None:
        with self.routing_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append_session_log(self, payload: dict[str, object]) -> None:
        with self.session_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_artifact(self, name: str, payload: dict[str, object]) -> Path:
        path = self.artifacts / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_backend_health(self) -> BackendHealth:
        health = BackendHealth(source="rolling")
        if not self.backend_log.exists():
            return health
        lines = [line for line in self.backend_log.read_text(encoding="utf-8").splitlines() if line.strip()]
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
            if event == "response":
                health.backend_reachable = True
                health.backend_chat_ready = True
                health.last_success_ms = duration_ms
                health.backend_chat_slow = duration_ms >= 45000
                failures = 0
            elif event == "backend_check_summary":
                health.backend_reachable = bool(item.get("backend_reachable", health.backend_reachable))
                health.backend_chat_ready = bool(item.get("backend_chat_ready", health.backend_chat_ready))
                health.backend_chat_slow = bool(item.get("backend_chat_slow", health.backend_chat_slow))
                health.backend_state = str(item.get("backend_state", health.backend_state))
                health.last_success_ms = int(item.get("last_success_ms", health.last_success_ms) or 0)
                health.last_timeout_ms = int(item.get("last_timeout_ms", health.last_timeout_ms) or 0)
                health.average_chat_ms = int(item.get("average_chat_ms", health.average_chat_ms) or 0)
                health.success_rate = float(item.get("success_rate", health.success_rate) or 0.0)
                health.warm_state = str(item.get("warm_state", health.warm_state))
                health.chat_probe_count = int(item.get("chat_probe_count", health.chat_probe_count) or 0)
                health.recent_failures = int(item.get("recent_failures", health.recent_failures) or 0)
            elif event == "timeout":
                failures += 1
                health.backend_reachable = True
                health.backend_chat_ready = False
                health.backend_chat_slow = True
                health.last_timeout_ms = duration_ms
                health.last_error = "timeout"
                health.detail = "recent timeout"
            elif event == "tags_check":
                health.tags_latency_ms = duration_ms
                health.backend_reachable = bool(item.get("ok", False))
                if not health.backend_reachable:
                    health.last_error = str(item.get("detail", health.last_error))
            elif event == "chat_probe":
                health.backend_chat_ready = True
                health.backend_reachable = True
                health.last_success_ms = duration_ms
            elif event == "chat_probe_timeout":
                failures += 1
                health.backend_reachable = True
                health.backend_chat_ready = False
                health.backend_chat_slow = True
                health.last_timeout_ms = duration_ms
                health.last_error = str(item.get("detail", "timeout"))
                health.detail = "chat probe timeout"
            elif event == "chat_probe_error":
                failures += 1
                health.backend_chat_ready = False
                health.backend_reachable = False if "no es alcanzable" in str(item.get("detail", "")).lower() else True
                health.last_error = str(item.get("detail", health.last_error))
                health.detail = "chat probe error"
            elif event == "error":
                failures += 1
                detail = str(item.get("detail", ""))
                health.last_error = detail
                health.detail = detail
                if "no es alcanzable" in detail.lower():
                    health.backend_reachable = False
                    health.backend_chat_ready = False
                else:
                    health.backend_reachable = True
                    health.backend_chat_ready = False
        health.recent_failures = failures
        if health.backend_chat_ready:
            health.backend_state = "slow" if health.backend_chat_slow else "ready"
        elif health.recent_failures:
            health.backend_state = "unstable" if health.backend_reachable else "unreachable"
        return health
