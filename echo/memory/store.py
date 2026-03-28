from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

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
        # Lazy import to avoid the echo.backends → echo.memory → echo.backends.health
        # circular dependency that would occur at module load time.
        from echo.backends.health import rolling_backend_health_from_log
        return rolling_backend_health_from_log(self.backend_log)
