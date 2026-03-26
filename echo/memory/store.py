from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from echo.backends.health import rolling_backend_health_from_log
from echo.types import ActivityEvent, BackendHealth, BranchMeta, BranchState, ColdMemory, EpisodicMemory, OperationalMemory, PlanStage, SessionState, ToolCallRecord, WorkingMemory


def _store_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self.branches = self.state / "branches"
        self.branches.mkdir(parents=True, exist_ok=True)
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
        return rolling_backend_health_from_log(self.backend_log)

    # ------------------------------------------------------------------ #
    # Branch management                                                     #
    # ------------------------------------------------------------------ #

    def _active_branch_file(self) -> Path:
        return self.state / "active_branch.json"

    def active_branch(self) -> BranchState | None:
        """Return the active branch pointer, or None if not yet initialized."""
        path = self._active_branch_file()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return BranchState(**data)
        except Exception:
            return None

    def _write_active_branch(self, state: BranchState) -> None:
        state.updated_at = _store_utc_now()
        self._active_branch_file().write_text(
            json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def set_active_branch(self, branch_name: str) -> None:
        """Persist a new active branch pointer."""
        self._write_active_branch(BranchState(branch_name=branch_name))

    def load_branch(self, name: str) -> BranchMeta | None:
        """Load branch metadata by name, or None if not found / corrupt."""
        path = self.branches / f"{_safe_branch_filename(name)}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return BranchMeta(**data)
        except Exception:
            return None

    def save_branch(self, branch: BranchMeta) -> None:
        """Persist branch metadata (creates or overwrites)."""
        branch.updated_at = _store_utc_now()
        path = self.branches / f"{_safe_branch_filename(branch.name)}.json"
        path.write_text(json.dumps(asdict(branch), ensure_ascii=False, indent=2), encoding="utf-8")

    def update_branch_head(self, branch_name: str, session_id: str) -> None:
        """Advance a branch's head pointer to a new session. Creates the branch if missing."""
        branch = self.load_branch(branch_name)
        if branch is None:
            branch = BranchMeta(name=branch_name, head_session_id=session_id)
        else:
            branch.head_session_id = session_id
        self.save_branch(branch)

    def branch_head_session_id(self, branch_name: str) -> str | None:
        """Return the head session ID for a branch, or None if missing / empty."""
        branch = self.load_branch(branch_name)
        if branch is None:
            return None
        return branch.head_session_id or None

    def list_branches(self) -> list[BranchMeta]:
        """Return all branches sorted by name."""
        results = []
        for path in sorted(self.branches.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append(BranchMeta(**data))
            except Exception:
                pass
        return results

    def delete_branch(self, name: str) -> None:
        """Remove a branch.  Raises ValueError for 'main'."""
        if name == "main":
            raise ValueError("Cannot delete the main branch.")
        path = self.branches / f"{_safe_branch_filename(name)}.json"
        if path.exists():
            path.unlink()

    def ensure_branch_model(self) -> BranchState:
        """Idempotent migration / initialization.

        * If the branch model already exists (active_branch.json present and
          valid), return it unchanged.
        * If corrupted or missing, rebuild a 'main' branch that points to the
          most recent session (or empty if none exists) and activate it.
        """
        active = self.active_branch()
        if active is not None:
            # Verify the referenced branch file also exists; recreate if not.
            if self.load_branch(active.branch_name) is not None:
                return active
        # Migration / first-time init: create main branch from latest session.
        latest = self.latest_session_id()
        main_branch = self.load_branch("main")
        if main_branch is None:
            main_branch = BranchMeta(name="main", head_session_id=latest or "")
            self.save_branch(main_branch)
        elif latest and not main_branch.head_session_id:
            main_branch.head_session_id = latest
            self.save_branch(main_branch)
        state = BranchState(branch_name="main")
        self._write_active_branch(state)
        return state


def _safe_branch_filename(name: str) -> str:
    """Sanitize a branch name to a safe filesystem component."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)[:64]
