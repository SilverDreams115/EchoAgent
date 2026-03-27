from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BranchState:
    """Persistent metadata for a conversational branch."""

    name: str
    created_at: str = field(default_factory=_utc_now)
    parent_branch: str = ""
    parent_session_id: str = ""
    session_ids: list[str] = field(default_factory=list)
    active_session_id: str = ""
    description: str = ""


@dataclass
class BranchMergeRecord:
    """Audit trail for a merge or cherry-pick operation between branches."""

    merge_id: str
    source_branch: str
    destination_branch: str
    strategy: str
    artefact_types: list[str]
    items_merged: dict[str, list[str]]
    conflicts: dict[str, list[str]]
    conflict_resolution: str
    source_session_id: str
    destination_session_id: str
    merged_at: str = field(default_factory=_utc_now)
    operation: str = "merge"  # "merge" or "cherry-pick"
