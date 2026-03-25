from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from echo.backends.health import normalize_backend_health
from echo.types import ActivityEvent, BackendHealth, ColdMemory, EpisodicMemory, OperationalMemory, PlanStage, SessionState, ToolCallRecord, WorkingMemory


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
        memory = session.working_memory
        lines = [
            f"# Active memory {session.id}",
            "",
            f"Objective: {memory.objective or session.objective or session.user_prompt}",
            "",
            f"Current stage: {memory.current_stage_id or session.current_stage_id or 'none'}",
            "",
            "## Active files",
        ]
        lines.extend(f"- {item}" for item in (memory.active_files[-12:] or ["none"]))
        lines += ["", "## Recent tools"]
        lines.extend(f"- {item}" for item in (memory.recent_tools[-8:] or ["none"]))
        lines += ["", "## Recent evidence"]
        lines.extend(f"- {item}" for item in (memory.recent_evidence[-8:] or ["none"]))
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def write_cold_summary(self, session: SessionState) -> Path:
        path = self.memory / f"{session.id}.cold.md"
        operational = session.operational_memory
        episodic = session.episodic_memory
        lines = [
            f"# Cold memory {session.id}",
            "",
            f"Objective: {session.objective or session.user_prompt}",
            "",
            "## Operational summary",
            operational.summary or session.operational_summary or session.summary,
            "",
            "## Confirmed facts",
        ]
        lines.extend(f"- {item}" for item in (operational.confirmed_facts[-20:] or ["none"]))
        lines += ["", "## Stage progress"]
        lines.extend(f"- {item}" for item in (operational.stage_progress[-20:] or ["none"]))
        lines += ["", "## Changes"]
        lines.extend(f"- {item}" for item in (episodic.changes[-20:] or ["none"]))
        lines += ["", "## Errors"]
        lines.extend(f"- {item}" for item in (episodic.errors[-20:] or ["none"]))
        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")
        session.cold_summary_path = str(path)
        session.cold_memory.summary_path = str(path)
        return path

    def load_session(self, session_id: str) -> SessionState:
        path = self.sessions / f"{session_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["tool_calls"] = [ToolCallRecord(**item) for item in data.get("tool_calls", [])]
        data["activity"] = [ActivityEvent(**item) for item in data.get("activity", [])]
        data["plan_stages"] = [PlanStage(**item) for item in data.get("plan_stages", [])]
        data["working_memory"] = WorkingMemory(**data.get("working_memory", {}))
        data["episodic_memory"] = EpisodicMemory(**data.get("episodic_memory", {}))
        data["operational_memory"] = OperationalMemory(**data.get("operational_memory", {}))
        data["cold_memory"] = ColdMemory(**data.get("cold_memory", {}))
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
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
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
            return normalize_backend_health(health, source="rolling")
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
