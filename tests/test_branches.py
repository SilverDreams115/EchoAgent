"""
Tests for the branch system: BranchStore, merge, cherry-pick, intent router.

No backend required — pure filesystem + dataclass logic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# echo.backends must be imported before echo.memory to avoid the circular import
# that exists in the existing codebase (backends.availability → memory → backends.health).
from echo.backends.errors import BackendError  # noqa: F401  (initializes echo.backends)
from echo.branches.cherry_pick import cherry_pick
from echo.branches.merge import ARTEFACT_TYPES, merge_branches
from echo.branches.models import BranchMergeRecord, BranchState
from echo.branches.store import BranchStore
from echo.memory import EchoStore
from echo.types.models import (
    EpisodicMemory,
    OperationalMemory,
    SessionState,
)
from echo.ui.intent_router import route


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> tuple[BranchStore, EchoStore]:
    branch_store = BranchStore(tmp_path)
    echo_store = EchoStore(tmp_path)
    return branch_store, echo_store


def _save_session(
    echo_store: EchoStore,
    branch_store: BranchStore,
    branch_name: str,
    *,
    decisions: list[str] | None = None,
    findings: list[str] | None = None,
    pending: list[str] | None = None,
    facts: list[str] | None = None,
    summary: str = "",
    errors: list[str] | None = None,
) -> SessionState:
    session = SessionState.create(
        repo_root=str(echo_store.project_root),
        mode="ask",
        model="test-model",
        user_prompt="test prompt",
    )
    session.decisions = decisions or []
    session.findings = findings or []
    session.pending = pending or []
    session.errors = errors or []
    session.operational_summary = summary
    session.operational_memory = OperationalMemory(
        summary=summary,
        confirmed_facts=facts or [],
        pending=pending or [],
    )
    session.episodic_memory = EpisodicMemory(decisions=decisions or [])
    echo_store.save_session(session)
    branch_store.add_session_to_branch(branch_name, session.id)
    return session


# ---------------------------------------------------------------------------
# BranchStore: basic CRUD
# ---------------------------------------------------------------------------


def test_active_branch_defaults_to_main(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    assert store.active_branch_name() == "main"


def test_create_branch(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    store.create_branch("feature-x", parent_branch="main")
    assert store.branch_exists("feature-x")
    b = store.load_branch("feature-x")
    assert b.name == "feature-x"
    assert b.parent_branch == "main"


def test_create_branch_idempotent(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    b1 = store.create_branch("feat", parent_branch="main")
    b2 = store.create_branch("feat", parent_branch="main")
    assert b1.name == b2.name


def test_list_branches(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    store.create_branch("main")
    store.create_branch("feature-a")
    store.create_branch("feature-b")
    branches = store.list_branches()
    assert "main" in branches
    assert "feature-a" in branches
    assert "feature-b" in branches


def test_set_active_branch(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    store.create_branch("main")
    store.create_branch("experiment")
    store.set_active_branch("experiment")
    assert store.active_branch_name() == "experiment"


def test_add_session_and_active_session(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    store.create_branch("main")
    store.add_session_to_branch("main", "session-abc")
    store.add_session_to_branch("main", "session-xyz")
    assert store.active_session_for_branch("main") == "session-xyz"
    b = store.load_branch("main")
    assert "session-abc" in b.session_ids
    assert "session-xyz" in b.session_ids


def test_active_session_for_nonexistent_branch_returns_none(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    assert store.active_session_for_branch("ghost") is None


def test_branch_persists_after_reload(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    store.create_branch("feat", parent_branch="main", description="test desc")
    store2 = BranchStore(tmp_path)
    b = store2.load_branch("feat")
    assert b.description == "test desc"
    assert b.parent_branch == "main"


def test_session_isolation_between_branches(tmp_path: Path) -> None:
    store = BranchStore(tmp_path)
    store.create_branch("main")
    store.create_branch("experiment")
    store.add_session_to_branch("main", "session-main")
    store.add_session_to_branch("experiment", "session-exp")
    assert store.active_session_for_branch("main") == "session-main"
    assert store.active_session_for_branch("experiment") == "session-exp"


# ---------------------------------------------------------------------------
# Merge: union-deduplicate (default)
# ---------------------------------------------------------------------------


def test_merge_union_deduplicate_decisions(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", decisions=["decision A", "decision B"])
    _save_session(echo_store, branch_store, "experiment", decisions=["decision B", "decision C"])

    record, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    assert "decision A" in new_session.decisions
    assert "decision B" in new_session.decisions
    assert "decision C" in new_session.decisions
    # No duplicates
    assert new_session.decisions.count("decision B") == 1


def test_merge_union_deduplicate_findings(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", findings=["finding 1"])
    _save_session(echo_store, branch_store, "experiment", findings=["finding 1", "finding 2"])

    _, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    assert "finding 1" in new_session.findings
    assert "finding 2" in new_session.findings
    assert new_session.findings.count("finding 1") == 1


def test_merge_facts_deduplicated(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", facts=["fact A"])
    _save_session(echo_store, branch_store, "experiment", facts=["fact A", "fact B"])

    _, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    confirmed = new_session.operational_memory.confirmed_facts
    assert "fact A" in confirmed
    assert "fact B" in confirmed
    assert confirmed.count("fact A") == 1


def test_merge_summary_union(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", summary="Summary from main.")
    _save_session(echo_store, branch_store, "experiment", summary="Summary from experiment.")

    _, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    assert "Summary from main." in new_session.operational_summary
    assert "Summary from experiment." in new_session.operational_summary


# ---------------------------------------------------------------------------
# Merge: prefer-source strategy
# ---------------------------------------------------------------------------


def test_merge_prefer_source(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", decisions=["dest-only"])
    _save_session(echo_store, branch_store, "experiment", decisions=["src-only"])

    _, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
        strategy="prefer-source",
    )

    # prefer-source: source items come first
    assert new_session.decisions[0] == "src-only"
    assert "dest-only" in new_session.decisions


# ---------------------------------------------------------------------------
# Merge: prefer-destination strategy
# ---------------------------------------------------------------------------


def test_merge_prefer_destination(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", decisions=["dest-only"])
    _save_session(echo_store, branch_store, "experiment", decisions=["src-only"])

    _, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
        strategy="prefer-destination",
    )

    # dest items come first; src items added if not duplicate
    assert new_session.decisions[0] == "dest-only"
    assert "src-only" in new_session.decisions


# ---------------------------------------------------------------------------
# Merge: session lineage
# ---------------------------------------------------------------------------


def test_merge_creates_new_session_with_parent_link(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    dest_sess = _save_session(echo_store, branch_store, "main", decisions=["d1"])
    _save_session(echo_store, branch_store, "experiment", decisions=["d2"])

    _, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    assert new_session.parent_session_id == dest_sess.id
    assert new_session.id != dest_sess.id
    # New session is now active in main
    assert branch_store.active_session_for_branch("main") == new_session.id


def test_merge_new_session_is_persisted(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main")
    _save_session(echo_store, branch_store, "experiment", decisions=["d1"])

    _, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    # Must be loadable from disk
    loaded = echo_store.load_session(new_session.id)
    assert "d1" in loaded.decisions


# ---------------------------------------------------------------------------
# Merge: audit trail (MergeRecord)
# ---------------------------------------------------------------------------


def test_merge_record_persisted(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main")
    _save_session(echo_store, branch_store, "experiment", decisions=["d1"])

    record, _ = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    records = branch_store.load_merge_records("main")
    assert len(records) == 1
    loaded = records[0]
    assert loaded.merge_id == record.merge_id
    assert loaded.source_branch == "experiment"
    assert loaded.destination_branch == "main"
    assert loaded.operation == "merge"


def test_merge_record_items_merged_populated(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", decisions=["d-main"])
    _save_session(echo_store, branch_store, "experiment", decisions=["d-main", "d-new"])

    record, _ = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    # "d-new" was added; "d-main" was a duplicate (not counted as added)
    assert "d-new" in record.items_merged.get("decisions", [])
    assert "d-main" not in record.items_merged.get("decisions", [])


# ---------------------------------------------------------------------------
# Merge: error handling
# ---------------------------------------------------------------------------


def test_merge_source_branch_no_sessions_raises(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main")
    branch_store.create_branch("empty-branch")

    with pytest.raises(ValueError, match="has no sessions"):
        merge_branches(
            source_branch="empty-branch",
            destination_branch="main",
            branch_store=branch_store,
            echo_store=echo_store,
        )


def test_merge_invalid_artefact_type_raises(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main")
    _save_session(echo_store, branch_store, "experiment")

    with pytest.raises(ValueError, match="Unknown artefact types"):
        merge_branches(
            source_branch="experiment",
            destination_branch="main",
            branch_store=branch_store,
            echo_store=echo_store,
            artefact_types=["does-not-exist"],
        )


# ---------------------------------------------------------------------------
# Merge: fresh destination branch (no sessions yet)
# ---------------------------------------------------------------------------


def test_merge_into_empty_destination(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    branch_store.create_branch("main")  # no sessions
    _save_session(echo_store, branch_store, "experiment", decisions=["d1"], findings=["f1"])

    _, new_session = merge_branches(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
    )

    assert "d1" in new_session.decisions
    assert "f1" in new_session.findings
    assert new_session.parent_session_id == ""


# ---------------------------------------------------------------------------
# Cherry-pick
# ---------------------------------------------------------------------------


def test_cherry_pick_only_decisions(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", decisions=["d-main"], findings=["f-main"])
    _save_session(
        echo_store,
        branch_store,
        "experiment",
        decisions=["d-exp"],
        findings=["f-exp"],
        pending=["p-exp"],
    )

    record, new_session = cherry_pick(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
        artefact_types=["decisions"],
    )

    # decisions: merged
    assert "d-main" in new_session.decisions
    assert "d-exp" in new_session.decisions
    # findings: NOT from experiment (only "f-main" from destination)
    assert "f-exp" not in new_session.findings
    assert "f-main" in new_session.findings
    # pending: NOT from experiment
    assert "p-exp" not in new_session.pending

    assert record.operation == "cherry-pick"
    assert "decisions" in record.strategy


def test_cherry_pick_decisions_and_findings(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main")
    _save_session(echo_store, branch_store, "experiment", decisions=["d1"], findings=["f1"])

    _, new_session = cherry_pick(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
        artefact_types=["decisions", "findings"],
    )

    assert "d1" in new_session.decisions
    assert "f1" in new_session.findings


def test_cherry_pick_empty_types_raises(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main")
    _save_session(echo_store, branch_store, "experiment")

    with pytest.raises(ValueError, match="at least one"):
        cherry_pick(
            source_branch="experiment",
            destination_branch="main",
            branch_store=branch_store,
            echo_store=echo_store,
            artefact_types=[],
        )


def test_cherry_pick_record_persisted(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main")
    _save_session(echo_store, branch_store, "experiment", decisions=["d1"])

    record, _ = cherry_pick(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
        artefact_types=["decisions"],
    )

    records = branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].operation == "cherry-pick"
    assert records[0].merge_id == record.merge_id


def test_cherry_pick_preserves_branch_isolation(tmp_path: Path) -> None:
    """Cherry-pick must NOT mutate the source branch."""
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main", decisions=["d-main"])
    src_session = _save_session(
        echo_store, branch_store, "experiment", decisions=["d-exp"]
    )

    cherry_pick(
        source_branch="experiment",
        destination_branch="main",
        branch_store=branch_store,
        echo_store=echo_store,
        artefact_types=["decisions"],
    )

    # Source branch still points to its original session
    exp_active = branch_store.active_session_for_branch("experiment")
    assert exp_active == src_session.id
    # Reloading the source session shows no contamination
    loaded_src = echo_store.load_session(src_session.id)
    assert "d-main" not in loaded_src.decisions


# ---------------------------------------------------------------------------
# Intent router
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_intent",
    [
        ("hola, ¿cómo estás?", "conversation"),
        ("revisa el runtime y dime qué falla", "conversation"),
        ("crea una rama experimento-shell", "branch_new"),
        ("nueva rama feature-x", "branch_new"),
        ("new branch my-feature", "branch_new"),
        ("branch new test-123", "branch_new"),
        ("vuelve a main", "branch_switch"),
        ("switch to main", "branch_switch"),
        ("cambia a feature-x", "branch_switch"),
        ("ir a la rama staging", "branch_switch"),
        ("lista de ramas", "branch_list"),
        ("list branches", "branch_list"),
        ("ver ramas", "branch_list"),
        ("branch status", "branch_status"),
        ("estado de la rama", "branch_status"),
        ("merge experimento-shell", "branch_merge"),
        ("fusionar feature-x", "branch_merge"),
        ("trae todo de experiment", "branch_merge"),
        ("cherry-pick feature-x", "branch_cherry_pick"),
        ("trae solo las decisiones de feature-x", "branch_cherry_pick"),
        ("status", "session_status"),
        ("estado", "session_status"),
        ("exit", "exit"),
        ("quit", "exit"),
        ("salir", "exit"),
        ("help", "help"),
        ("ayuda", "help"),
    ],
)
def test_intent_router(text: str, expected_intent: str) -> None:
    intent, _ = route(text)
    assert intent == expected_intent, f"'{text}' → got '{intent}', expected '{expected_intent}'"


def test_intent_router_extracts_branch_name_new() -> None:
    intent, values = route("crea una rama mi-rama-123")
    assert intent == "branch_new"
    assert values["name"] == "mi-rama-123"


def test_intent_router_extracts_branch_name_switch() -> None:
    intent, values = route("vuelve a main")
    assert intent == "branch_switch"
    assert values["name"] == "main"


def test_intent_router_extracts_merge_source() -> None:
    intent, values = route("merge experimento-shell")
    assert intent == "branch_merge"
    assert values["source"] == "experimento-shell"


def test_intent_router_extracts_cherry_pick_source() -> None:
    intent, values = route("trae solo las decisiones de feature-x")
    assert intent == "branch_cherry_pick"
    assert values["source"] == "feature-x"


def test_intent_router_slash_prefix_is_conversation() -> None:
    # Slash commands are handled upstream by the REPL, not by the router
    # The router should not be called for slash commands
    # But if called, they fall through to conversation (no rule matches)
    intent, _ = route("/branch new foo")
    assert intent == "conversation"


# ---------------------------------------------------------------------------
# ARTEFACT_TYPES constant
# ---------------------------------------------------------------------------


def test_artefact_types_constant() -> None:
    assert "decisions" in ARTEFACT_TYPES
    assert "findings" in ARTEFACT_TYPES
    assert "pending" in ARTEFACT_TYPES
    assert "facts" in ARTEFACT_TYPES
    assert "summary" in ARTEFACT_TYPES
    assert "errors" in ARTEFACT_TYPES
    assert "changes" in ARTEFACT_TYPES


# ---------------------------------------------------------------------------
# Persistence resilience
# ---------------------------------------------------------------------------


def test_reload_branch_store_recovers_active_branch(tmp_path: Path) -> None:
    store1 = BranchStore(tmp_path)
    store1.create_branch("main")
    store1.create_branch("experiment")
    store1.set_active_branch("experiment")

    store2 = BranchStore(tmp_path)
    assert store2.active_branch_name() == "experiment"


def test_reload_branch_sessions_survive(tmp_path: Path) -> None:
    branch_store1, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store1, "main", decisions=["d1"])

    branch_store2 = BranchStore(tmp_path)
    sid = branch_store2.active_session_for_branch("main")
    assert sid is not None
    loaded = echo_store.load_session(sid)
    assert "d1" in loaded.decisions


def test_merge_multiple_times_accumulates_records(tmp_path: Path) -> None:
    branch_store, echo_store = _make_store(tmp_path)
    _save_session(echo_store, branch_store, "main")
    _save_session(echo_store, branch_store, "experiment", decisions=["d1"])

    merge_branches("experiment", "main", branch_store, echo_store)
    # Update experiment branch
    _save_session(echo_store, branch_store, "experiment", decisions=["d1", "d2"])
    merge_branches("experiment", "main", branch_store, echo_store)

    records = branch_store.load_merge_records("main")
    assert len(records) == 2
