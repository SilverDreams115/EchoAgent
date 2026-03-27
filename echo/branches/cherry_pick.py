from __future__ import annotations

from typing import TYPE_CHECKING

from echo.branches.merge import ARTEFACT_TYPES, merge_branches
from echo.branches.models import BranchMergeRecord
from echo.branches.store import BranchStore
from echo.types.models import SessionState

if TYPE_CHECKING:
    from echo.memory.store import EchoStore


def cherry_pick(
    source_branch: str,
    destination_branch: str,
    branch_store: BranchStore,
    echo_store: EchoStore,
    artefact_types: list[str],
) -> tuple[BranchMergeRecord, SessionState]:
    """
    Cherry-pick specific artefact types from source_branch into destination_branch.

    Only the requested types are brought in (union-deduplicate).
    All other artefact slots remain as-is in the destination branch.

    Valid types: decisions, findings, pending, facts, summary, errors, changes
    """
    if not artefact_types:
        raise ValueError("Must specify at least one artefact type for cherry-pick.")

    unknown = set(artefact_types) - ARTEFACT_TYPES
    if unknown:
        raise ValueError(f"Unknown artefact types: {unknown}. Valid: {ARTEFACT_TYPES}")

    record, session = merge_branches(
        source_branch=source_branch,
        destination_branch=destination_branch,
        branch_store=branch_store,
        echo_store=echo_store,
        strategy="union-deduplicate",
        artefact_types=artefact_types,
    )
    # Mark as cherry-pick in the record and re-save
    record.operation = "cherry-pick"
    record.strategy = f"cherry-pick:{','.join(sorted(artefact_types))}"
    branch_store.save_merge_record(record)

    return record, session
