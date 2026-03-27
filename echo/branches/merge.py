from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from echo.branches.models import BranchMergeRecord
from echo.branches.store import BranchStore
from echo.types.models import EpisodicMemory, OperationalMemory, SessionState

if TYPE_CHECKING:
    from echo.memory.store import EchoStore

ARTEFACT_TYPES: set[str] = {"decisions", "findings", "pending", "facts", "summary", "errors", "changes"}

_ALL_ARTEFACTS = sorted(ARTEFACT_TYPES)


def _normalize(s: str) -> str:
    return s.strip().lower()


def _merge_lists(
    dest: list[str],
    source: list[str],
    strategy: str,
) -> tuple[list[str], list[str]]:
    """Merge two lists. Returns (merged_list, items_added_from_source)."""
    if strategy == "prefer-source":
        src_norm = {_normalize(x) for x in source}
        extras = [x for x in dest if _normalize(x) not in src_norm]
        return source + extras, list(source)
    # union-deduplicate (default) and prefer-destination both deduplicate
    dest_norm = {_normalize(x) for x in dest}
    added = []
    for item in source:
        if _normalize(item) not in dest_norm:
            added.append(item)
            dest_norm.add(_normalize(item))
    return dest + added, added


def _merge_summary(
    dest_summary: str,
    source_summary: str,
    source_branch: str,
    strategy: str,
) -> tuple[str, list[str], list[str]]:
    """Merge two summaries. Returns (merged, items_added, conflicts)."""
    if not source_summary:
        return dest_summary, [], []
    if not dest_summary:
        return source_summary, [f"summary from {source_branch}"], []

    if strategy == "prefer-source":
        return (
            source_summary,
            [f"summary replaced from {source_branch}"],
            ["summary conflict: destination summary overwritten by source"],
        )
    if strategy == "prefer-destination":
        return (
            dest_summary,
            [],
            [f"summary conflict: source summary from {source_branch} discarded"],
        )
    # union-deduplicate: append source as addendum
    merged = f"{dest_summary}\n\n---\nFrom branch {source_branch}:\n{source_summary}"
    return merged, [f"summary appended from {source_branch}"], ["summary conflict: merged both summaries"]


def _extract_artefacts(session: SessionState) -> dict[str, object]:
    """Pull all mergeable artefacts from a session."""
    decisions: list[str] = list(session.decisions)
    for d in session.episodic_memory.decisions:
        if d not in decisions:
            decisions.append(d)

    pending: list[str] = list(session.pending)
    for p in session.operational_memory.pending:
        if p not in pending:
            pending.append(p)

    errors: list[str] = list(session.errors)
    for e in session.episodic_memory.errors:
        if e not in errors:
            errors.append(e)

    changes: list[str] = list(session.changed_files)
    for c in session.episodic_memory.changes:
        if c not in changes:
            changes.append(c)

    return {
        "decisions": decisions,
        "findings": list(session.findings),
        "pending": pending,
        "facts": list(session.operational_memory.confirmed_facts),
        "summary": (
            session.operational_memory.summary
            or session.operational_summary
            or session.summary
            or ""
        ),
        "errors": errors,
        "changes": changes,
    }


def merge_branches(
    source_branch: str,
    destination_branch: str,
    branch_store: BranchStore,
    echo_store: EchoStore,
    strategy: str = "union-deduplicate",
    artefact_types: list[str] | None = None,
) -> tuple[BranchMergeRecord, SessionState]:
    """
    Merge structured artefacts from source_branch into destination_branch.

    Does NOT merge raw transcripts or tool_calls — only structured state:
    decisions, findings, pending, facts, summary, errors, changes.

    Returns (merge_record, new_destination_session).
    The new session is saved and registered in the destination branch.
    """
    if artefact_types is None:
        artefact_types = _ALL_ARTEFACTS

    unknown = set(artefact_types) - ARTEFACT_TYPES
    if unknown:
        raise ValueError(f"Unknown artefact types: {unknown}. Valid: {ARTEFACT_TYPES}")

    # Load source
    source_session_id = branch_store.active_session_for_branch(source_branch)
    if not source_session_id:
        raise ValueError(f"Branch '{source_branch}' has no sessions to merge from.")
    source_session = echo_store.load_session(source_session_id)

    # Load destination (may not exist yet)
    dest_session_id = branch_store.active_session_for_branch(destination_branch)
    dest_session: SessionState | None = None
    if dest_session_id:
        dest_session = echo_store.load_session(dest_session_id)

    src_data = _extract_artefacts(source_session)
    dst_data: dict[str, object] = (
        _extract_artefacts(dest_session)
        if dest_session
        else {k: ([] if k != "summary" else "") for k in ARTEFACT_TYPES}
    )

    merged: dict[str, object] = {}
    items_merged: dict[str, list[str]] = {}
    conflicts: dict[str, list[str]] = {}

    for atype in _ALL_ARTEFACTS:
        if atype not in artefact_types:
            merged[atype] = dst_data[atype]
            continue

        if atype == "summary":
            m_summary, added, conf = _merge_summary(
                str(dst_data["summary"]),
                str(src_data["summary"]),
                source_branch,
                strategy,
            )
            merged["summary"] = m_summary
            if added:
                items_merged["summary"] = added
            if conf:
                conflicts["summary"] = conf
        else:
            m_list, added = _merge_lists(
                list(dst_data.get(atype, [])),  # type: ignore[arg-type]
                list(src_data.get(atype, [])),  # type: ignore[arg-type]
                strategy,
            )
            merged[atype] = m_list
            if added:
                items_merged[atype] = added

    # Build new destination session (a "merge commit" session)
    repo_root = (dest_session or source_session).repo_root
    model = (dest_session or source_session).model

    new_session = SessionState.create(
        repo_root=repo_root,
        mode="resume",
        model=model,
        user_prompt=f"[merge:{source_branch}→{destination_branch}] strategy={strategy}",
    )
    new_session.parent_session_id = dest_session_id or ""
    new_session.decisions = list(merged.get("decisions", []))  # type: ignore[arg-type]
    new_session.findings = list(merged.get("findings", []))  # type: ignore[arg-type]
    new_session.pending = list(merged.get("pending", []))  # type: ignore[arg-type]
    new_session.errors = list(merged.get("errors", []))  # type: ignore[arg-type]
    new_session.changed_files = list(merged.get("changes", []))  # type: ignore[arg-type]
    new_session.operational_summary = str(merged.get("summary", ""))

    new_session.operational_memory = OperationalMemory(
        summary=str(merged.get("summary", "")),
        confirmed_facts=list(merged.get("facts", [])),  # type: ignore[arg-type]
        restrictions=list(dest_session.operational_memory.restrictions) if dest_session else [],
        stage_progress=list(dest_session.operational_memory.stage_progress) if dest_session else [],
        pending=list(merged.get("pending", [])),  # type: ignore[arg-type]
    )
    new_session.episodic_memory = EpisodicMemory(
        decisions=list(merged.get("decisions", [])),  # type: ignore[arg-type]
    )

    if dest_session:
        new_session.working_set = list(dest_session.working_set)
        new_session.focus_files = list(dest_session.focus_files)
        new_session.objective = dest_session.objective or dest_session.user_prompt
    else:
        new_session.working_set = list(source_session.working_set)
        new_session.focus_files = list(source_session.focus_files)
        new_session.objective = source_session.objective or source_session.user_prompt

    echo_store.save_session(new_session)
    branch_store.add_session_to_branch(destination_branch, new_session.id)

    merge_id = f"merge-{uuid.uuid4().hex[:8]}"
    record = BranchMergeRecord(
        merge_id=merge_id,
        source_branch=source_branch,
        destination_branch=destination_branch,
        strategy=strategy,
        artefact_types=list(artefact_types),
        items_merged=items_merged,
        conflicts=conflicts,
        conflict_resolution=strategy,
        source_session_id=source_session_id,
        destination_session_id=new_session.id,
        operation="merge",
    )
    branch_store.save_merge_record(record)

    return record, new_session
