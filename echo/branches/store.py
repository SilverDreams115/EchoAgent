from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from echo.branches.models import BranchMergeRecord, BranchState

_DEFAULT_BRANCH = "main"


class BranchStore:
    """Persist branches and merge records under .echo/branches/."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.base = project_root / ".echo" / "branches"
        self.base.mkdir(parents=True, exist_ok=True)
        self._current_branch_file = self.base / "current_branch.txt"

    # ------------------------------------------------------------------
    # Active branch
    # ------------------------------------------------------------------

    def active_branch_name(self) -> str:
        if self._current_branch_file.exists():
            val = self._current_branch_file.read_text(encoding="utf-8").strip()
            if val:
                return val
        if not self.branch_exists(_DEFAULT_BRANCH):
            self.save_branch(BranchState(name=_DEFAULT_BRANCH))
        self._current_branch_file.write_text(_DEFAULT_BRANCH, encoding="utf-8")
        return _DEFAULT_BRANCH

    def set_active_branch(self, name: str) -> None:
        self._current_branch_file.write_text(name, encoding="utf-8")

    # ------------------------------------------------------------------
    # Branch CRUD
    # ------------------------------------------------------------------

    def branch_dir(self, name: str) -> Path:
        return self.base / name

    def branch_exists(self, name: str) -> bool:
        return (self.branch_dir(name) / "branch.json").exists()

    def load_branch(self, name: str) -> BranchState:
        path = self.branch_dir(name) / "branch.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return BranchState(**data)

    def save_branch(self, branch: BranchState) -> None:
        d = self.branch_dir(branch.name)
        d.mkdir(parents=True, exist_ok=True)
        (d / "branch.json").write_text(
            json.dumps(asdict(branch), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def list_branches(self) -> list[str]:
        names = []
        for child in sorted(self.base.iterdir()):
            if child.is_dir() and (child / "branch.json").exists():
                names.append(child.name)
        return names

    def create_branch(
        self,
        name: str,
        parent_branch: str = "",
        parent_session_id: str = "",
        description: str = "",
    ) -> BranchState:
        if self.branch_exists(name):
            return self.load_branch(name)
        branch = BranchState(
            name=name,
            parent_branch=parent_branch,
            parent_session_id=parent_session_id,
            description=description,
        )
        self.save_branch(branch)
        return branch

    # ------------------------------------------------------------------
    # Session tracking per branch
    # ------------------------------------------------------------------

    def add_session_to_branch(self, branch_name: str, session_id: str) -> None:
        if not self.branch_exists(branch_name):
            self.create_branch(branch_name)
        branch = self.load_branch(branch_name)
        if session_id not in branch.session_ids:
            branch.session_ids.append(session_id)
        branch.active_session_id = session_id
        self.save_branch(branch)

    def active_session_for_branch(self, branch_name: str) -> str | None:
        if not self.branch_exists(branch_name):
            return None
        branch = self.load_branch(branch_name)
        return branch.active_session_id or None

    # ------------------------------------------------------------------
    # Merge records
    # ------------------------------------------------------------------

    def save_merge_record(self, record: BranchMergeRecord) -> None:
        merges_dir = self.branch_dir(record.destination_branch) / "merges"
        merges_dir.mkdir(parents=True, exist_ok=True)
        path = merges_dir / f"{record.merge_id}.json"
        path.write_text(
            json.dumps(asdict(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_merge_records(self, branch_name: str) -> list[BranchMergeRecord]:
        merges_dir = self.branch_dir(branch_name) / "merges"
        if not merges_dir.exists():
            return []
        records = []
        for f in sorted(merges_dir.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8"))
            records.append(BranchMergeRecord(**data))
        return records
