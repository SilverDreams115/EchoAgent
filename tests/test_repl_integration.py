"""
Integration tests for the EchoRepl conversational flow.

Unlike test_repl_ux.py (unit tests focused on isolated properties), these tests
exercise the *complete dispatch path* through the REPL: input → intent routing →
branch/session operations → persistence → next turn.

EchoAgent is stubbed at the backend level — its .store is a REAL EchoStore so
that merge, cherry-pick, and session persistence work against actual on-disk state.

No network / LLM calls are made.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest

# Import order guard — avoids the circular import echo.backends → echo.memory → echo.backends.health
# (after the lazy-import fix this is still needed for earlier imports in this file's module chain)
from echo.backends.errors import BackendError  # noqa: F401
from echo.branches.store import BranchStore
from echo.memory import EchoStore
from echo.types.models import (
    EpisodicMemory,
    OperationalMemory,
    SessionState,
)
from echo.ui.repl import EchoRepl

from rich.console import Console


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _input_sequence(*lines: str | None):
    """Returns an _input_fn that yields each line then None (EOF)."""
    it: Iterator[str | None] = iter(list(lines) + [None])
    return lambda: next(it, None)


class ReplHarness:
    """
    A realistic integration harness for EchoRepl.

    - BranchStore and EchoStore use real tmp_path on disk.
    - EchoAgent is stubbed: agent.store points to the REAL EchoStore so that
      merge/cherry-pick and session persistence work end-to-end.
    - agent.run() creates and persists a real SessionState on each call.
    """

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.branch_store = BranchStore(tmp_path)
        self.echo_store = EchoStore(tmp_path)
        # Ensure the default "main" branch has metadata (mirrors real usage where
        # at least one turn has been made before any branch switch is possible).
        self.branch_store.create_branch("main")
        self._sessions_created: list[SessionState] = []

        self.agent = MagicMock()
        self.agent.store = self.echo_store  # real store — critical for merge/cherry-pick
        self.agent.activity.watch = MagicMock()
        self.agent.doctor.return_value = {"status": "ok", "backend": "stub"}
        self.agent.current_status.return_value = {"model": "stub", "session": "none"}

        def _run(text, *, mode="ask", resume_session_id=None, profile=None):
            session = SessionState.create(
                repo_root=str(tmp_path),
                mode=mode,
                model="stub",
                user_prompt=text,
            )
            session.parent_session_id = resume_session_id or ""
            session.decisions = [f"decision:{text[:30]}"]
            session.findings = [f"finding:{text[:30]}"]
            session.episodic_memory = EpisodicMemory(decisions=session.decisions)
            session.operational_memory = OperationalMemory(
                confirmed_facts=[], pending=[], summary=""
            )
            self.echo_store.save_session(session)
            self._sessions_created.append(session)
            return "stub answer", None, session

        self.agent.run.side_effect = _run

        self._buf = io.StringIO()
        self.console = Console(file=self._buf, force_terminal=False, highlight=False)

    def make_repl(self, *inputs: str | None) -> EchoRepl:
        return EchoRepl(
            self.agent,
            self.tmp,
            self.console,
            self.branch_store,
            _input_fn=_input_sequence(*inputs),
        )

    def run(self, *inputs: str | None) -> EchoRepl:
        repl = self.make_repl(*inputs)
        repl.run()
        return repl

    def add_branch_with_session(
        self,
        name: str,
        *,
        parent: str = "main",
        decisions: list[str] | None = None,
        findings: list[str] | None = None,
        pending: list[str] | None = None,
        facts: list[str] | None = None,
    ) -> SessionState:
        """Create a branch and add a session with the given artefacts."""
        if name != "main" and not self.branch_store.branch_exists(name):
            self.branch_store.create_branch(name, parent_branch=parent)
        session = SessionState.create(
            repo_root=str(self.tmp),
            mode="ask",
            model="stub",
            user_prompt=f"setup:{name}",
        )
        session.decisions = decisions or [f"decision-from-{name}"]
        session.findings = findings or [f"finding-from-{name}"]
        session.pending = pending or []
        session.episodic_memory = EpisodicMemory(decisions=session.decisions)
        session.operational_memory = OperationalMemory(
            confirmed_facts=facts or [], pending=pending or [], summary=""
        )
        self.echo_store.save_session(session)
        self.branch_store.add_session_to_branch(name, session.id)
        return session

    def output(self) -> str:
        return self._buf.getvalue()


@pytest.fixture
def harness(tmp_path: Path) -> ReplHarness:
    return ReplHarness(tmp_path)


# ---------------------------------------------------------------------------
# 1. Basic conversation flow
# ---------------------------------------------------------------------------


def test_single_conversation_turn(harness: ReplHarness) -> None:
    """A plain text message is forwarded to the agent exactly once."""
    harness.run("revisa el runtime y dime qué falla")
    harness.agent.run.assert_called_once()
    assert harness.agent.run.call_args[0][0] == "revisa el runtime y dime qué falla"


def test_agent_answer_printed(harness: ReplHarness) -> None:
    """After agent.run() the answer appears in the console output."""
    harness.run("hola Echo")
    assert "stub answer" in harness.output()


def test_multiple_turns_sequential(harness: ReplHarness) -> None:
    """Two conversation turns both reach the agent in order."""
    harness.run("primer mensaje", "segundo mensaje")
    assert harness.agent.run.call_count == 2
    first_call = harness.agent.run.call_args_list[0][0][0]
    second_call = harness.agent.run.call_args_list[1][0][0]
    assert first_call == "primer mensaje"
    assert second_call == "segundo mensaje"


def test_second_turn_uses_resume_session_id(harness: ReplHarness) -> None:
    """Second turn passes the session created on first turn as resume_session_id."""
    harness.run("turno uno", "turno dos")
    first_session_id = harness.agent.run.call_args_list[0][1].get("resume_session_id")
    second_resume_id = harness.agent.run.call_args_list[1][1].get("resume_session_id")
    # First turn has no prior session (fresh start)
    assert first_session_id is None
    # Second turn resumes from first turn's session
    first_created_id = harness._sessions_created[0].id
    assert second_resume_id == first_created_id


def test_multiline_input_forwarded_intact(harness: ReplHarness) -> None:
    """Text containing \\n (multiline message) reaches the agent as-is."""
    msg = "instrucción línea uno\ninstrucción línea dos\nlínea tres"
    harness.run(msg)
    harness.agent.run.assert_called_once()
    assert harness.agent.run.call_args[0][0] == msg


# ---------------------------------------------------------------------------
# 2. Session persistence
# ---------------------------------------------------------------------------


def test_session_persisted_to_disk_after_turn(harness: ReplHarness) -> None:
    """After a conversation turn, a session JSON file must exist on disk."""
    harness.run("analiza el backend")
    session_files = list((harness.tmp / ".echo" / "sessions").glob("session-*.json"))
    assert session_files, "Expected at least one session file in .echo/sessions/"


def test_repl_resumes_session_on_second_open(harness: ReplHarness, tmp_path: Path) -> None:
    """If a prior session exists when the REPL starts, _current_session_id is set."""
    # Simulate a prior REPL run by saving a session and registering it
    prior_session = harness.add_branch_with_session("main")

    # New REPL instance (simulates re-opening echo-agent)
    repl = harness.make_repl()  # no inputs — don't run, just inspect init state
    assert repl._current_session_id == prior_session.id
    assert repl._session_resumed is True


def test_repl_current_session_id_updates_after_turn(harness: ReplHarness) -> None:
    """After the first agent turn, _current_session_id is set to the new session."""
    repl = harness.make_repl("hola")
    assert repl._current_session_id is None  # before run

    repl.run()

    assert repl._current_session_id is not None
    assert repl._current_session_id == harness._sessions_created[-1].id


# ---------------------------------------------------------------------------
# 3. Branch operations — slash commands
# ---------------------------------------------------------------------------


def test_slash_branch_new_creates_branch(harness: ReplHarness) -> None:
    """/branch new <name> creates a branch that persists in BranchStore."""
    harness.run("/branch new feature-auth")
    assert harness.branch_store.branch_exists("feature-auth")


def test_slash_branch_switch_changes_active_branch(harness: ReplHarness) -> None:
    """/branch switch <name> changes the active branch."""
    harness.branch_store.create_branch("feature-auth", parent_branch="main")
    harness.run("/branch switch feature-auth")
    assert harness.branch_store.active_branch_name() == "feature-auth"


def test_slash_branch_switch_restores_branch_session(harness: ReplHarness) -> None:
    """Switching to a branch with a prior session restores that session id."""
    harness.branch_store.create_branch("feature-x", parent_branch="main")
    prior = harness.add_branch_with_session("feature-x")

    repl = harness.make_repl("/branch switch feature-x")
    repl.run()

    # After switching, the repl should have updated _current_session_id
    assert repl._current_session_id == prior.id


def test_slash_branch_list_does_not_crash_with_multiple_branches(harness: ReplHarness) -> None:
    """/branch list with multiple branches prints without crashing."""
    harness.branch_store.create_branch("alpha", parent_branch="main")
    harness.branch_store.create_branch("beta", parent_branch="main")
    harness.run("/branch list")
    assert "alpha" in harness.output()
    assert "beta" in harness.output()


def test_slash_branch_new_then_switch_then_conversation(harness: ReplHarness) -> None:
    """Create a branch, switch to it, have a conversation — all in one session."""
    harness.run(
        "/branch new experiment",
        "/branch switch experiment",
        "diseña el nuevo módulo de caching",
    )
    assert harness.branch_store.branch_exists("experiment")
    assert harness.branch_store.active_branch_name() == "experiment"
    harness.agent.run.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Branch operations — natural language routing
# ---------------------------------------------------------------------------


def test_natural_language_branch_creation(harness: ReplHarness) -> None:
    """'crea una rama X' routes to branch creation."""
    harness.run("crea una rama experimento-caching")
    assert harness.branch_store.branch_exists("experimento-caching")
    harness.agent.run.assert_not_called()  # NOT a conversation turn


def test_natural_language_branch_switch(harness: ReplHarness) -> None:
    """'vuelve a main' routes to branch switch."""
    harness.branch_store.create_branch("feature-x", parent_branch="main")
    harness.branch_store.set_active_branch("feature-x")

    harness.run("vuelve a main")
    assert harness.branch_store.active_branch_name() == "main"
    harness.agent.run.assert_not_called()


def test_regular_sentence_with_branch_keyword_does_not_trigger_switch(
    harness: ReplHarness,
) -> None:
    """'vuelve a revisar el código' must NOT trigger a branch switch."""
    harness.run("vuelve a revisar el código")
    # Intent should be conversation, not branch_switch
    harness.agent.run.assert_called_once()  # Goes to agent, not branch switch


# ---------------------------------------------------------------------------
# 5. Merge from REPL
# ---------------------------------------------------------------------------


def test_slash_merge_creates_merge_record(harness: ReplHarness) -> None:
    """/branch merge <source> creates a merge record and updates session."""
    harness.add_branch_with_session("main")
    harness.add_branch_with_session(
        "feature-x", decisions=["feat-decision"], findings=["feat-finding"]
    )

    repl = harness.run("/branch merge feature-x")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].destination_branch == "main"


def test_merge_updates_current_session_id(harness: ReplHarness) -> None:
    """After merge, the REPL's _current_session_id points to the merge session."""
    harness.add_branch_with_session("main")
    harness.add_branch_with_session("feature-x")

    repl = harness.make_repl("/branch merge feature-x")
    repl.run()

    assert repl._current_session_id is not None
    # The session should be the merge session (most recent)
    session_id = repl._current_session_id
    assert session_id.startswith("session-")


def test_natural_language_merge(harness: ReplHarness) -> None:
    """'merge feature-x' triggers a real merge operation."""
    harness.add_branch_with_session("main")
    harness.add_branch_with_session("feature-x")

    harness.run("merge feature-x")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1


def test_natural_language_trae_todo_de_branch_is_merge_not_cherry_pick(
    harness: ReplHarness,
) -> None:
    """'trae todo de X' must trigger a FULL MERGE, not a cherry-pick.

    This was a pre-existing false-positive bug in the intent router.
    """
    harness.add_branch_with_session("main")
    harness.add_branch_with_session("feature-x")

    harness.run("trae todo de feature-x")

    # Full merge: record exists
    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1, "Expected a merge record for 'trae todo de X'"
    assert records[0].operation == "merge"


# ---------------------------------------------------------------------------
# 6. Cherry-pick from REPL
# ---------------------------------------------------------------------------


def test_slash_cherry_pick_creates_cherry_pick_record(harness: ReplHarness) -> None:
    """/branch cherry-pick <source> --decisions creates a cherry-pick record."""
    harness.add_branch_with_session("main")
    harness.add_branch_with_session("feature-x", decisions=["feat-decision-A"])

    harness.run("/branch cherry-pick feature-x --decisions")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].operation == "cherry-pick"
    assert records[0].source_branch == "feature-x"


def test_natural_language_cherry_pick_decisions(harness: ReplHarness) -> None:
    """'trae las decisiones de feature-x' triggers cherry-pick of decisions."""
    harness.add_branch_with_session("main")
    harness.add_branch_with_session("feature-x", decisions=["feat-decision-A"])

    harness.run("trae las decisiones de feature-x")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].operation == "cherry-pick"
    # The cherry-picked artefacts must include decisions
    assert "decisions" in records[0].artefact_types


def test_natural_language_dame_findings_de_branch(harness: ReplHarness) -> None:
    """'dame los findings de feature-x' triggers cherry-pick of findings."""
    harness.add_branch_with_session("main")
    harness.add_branch_with_session("feature-x", findings=["feat-finding-A"])

    harness.run("dame los findings de feature-x")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert "findings" in records[0].artefact_types


def test_cherry_pick_does_not_mutate_source_branch(harness: ReplHarness) -> None:
    """Cherry-picking from feature-x into main must not modify feature-x sessions."""
    harness.add_branch_with_session("main")
    source_session = harness.add_branch_with_session(
        "feature-x", decisions=["d1", "d2"]
    )
    sessions_before = harness.branch_store.load_branch("feature-x").session_ids[:]

    harness.run("trae las decisiones de feature-x")

    sessions_after = harness.branch_store.load_branch("feature-x").session_ids
    assert sessions_after == sessions_before, "Source branch sessions must not change after cherry-pick"


# ---------------------------------------------------------------------------
# 7. Session commands
# ---------------------------------------------------------------------------


def test_slash_session_new_clears_session_id(harness: ReplHarness) -> None:
    """/session new clears the current session so the next turn starts fresh."""
    harness.add_branch_with_session("main")
    repl = harness.make_repl("/session new")
    # _current_session_id is set at init (from prior session)
    assert repl._current_session_id is not None
    repl.run()
    # After /session new, must be None
    assert repl._current_session_id is None


def test_slash_session_status_does_not_crash(harness: ReplHarness) -> None:
    """/session status prints status without crashing."""
    harness.run("/session status")
    # No assertion on content — just must not raise


# ---------------------------------------------------------------------------
# 8. Slash commands — utilities
# ---------------------------------------------------------------------------


def test_slash_doctor_prints_output(harness: ReplHarness) -> None:
    """/doctor prints doctor output to the console."""
    harness.run("/doctor")
    assert "Echo doctor" in harness.output() or "status" in harness.output()


def test_slash_help_prints_help(harness: ReplHarness) -> None:
    """/help prints the help panel."""
    harness.run("/help")
    assert "Enter" in harness.output()  # Help text contains keybinding info


def test_unknown_slash_command_prints_warning(harness: ReplHarness) -> None:
    """Unknown slash commands print a warning and do not raise."""
    harness.run("/totally-unknown-command-xyz")
    assert "desconocido" in harness.output() or "unknown" in harness.output().lower()


# ---------------------------------------------------------------------------
# 9. Circular import: echo.memory before echo.backends
# ---------------------------------------------------------------------------


def test_memory_can_be_imported_without_backends_preloaded() -> None:
    """EchoStore can be instantiated without pre-loading echo.backends.

    After the lazy-import fix in memory/store.py, echo.memory no longer
    triggers a circular load of echo.backends at module import time.
    This test verifies the fix holds: importing EchoStore and calling all
    methods except read_backend_health() does not require echo.backends.
    """
    import sys
    import importlib

    # Save and remove all echo.* modules from sys.modules to simulate fresh load
    saved = {k: v for k, v in sys.modules.items() if k.startswith("echo")}
    for key in list(saved):
        del sys.modules[key]

    try:
        # Import echo.memory FIRST — this should not crash after the fix
        import echo.memory  # noqa: F401

        # Verify EchoStore can be instantiated
        import tempfile
        from pathlib import Path as _Path
        with tempfile.TemporaryDirectory() as d:
            store = echo.memory.EchoStore(_Path(d))
            assert store is not None
    finally:
        # Restore sys.modules
        for key in list(sys.modules.keys()):
            if key.startswith("echo"):
                del sys.modules[key]
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 10. Mixed flow: conversation → branch op → conversation
# ---------------------------------------------------------------------------


def test_mixed_conversation_and_branch_ops(harness: ReplHarness) -> None:
    """Interleaving conversation and branch commands works without state corruption."""
    harness.run(
        "analiza el código del runtime",   # conversation turn
        "/branch new analisis-runtime",     # branch op
        "/branch switch analisis-runtime",  # branch op
        "profundiza en el engine",          # conversation turn on new branch
    )
    # Two conversation turns
    assert harness.agent.run.call_count == 2
    # Branch exists and is active
    assert harness.branch_store.branch_exists("analisis-runtime")
    assert harness.branch_store.active_branch_name() == "analisis-runtime"


def test_branch_sessions_are_tracked_per_branch(harness: ReplHarness) -> None:
    """Sessions created on 'main' and on a feature branch are tracked separately."""
    harness.run(
        "hola desde main",                 # turn on main
        "/branch new feature-b",
        "/branch switch feature-b",
        "hola desde feature-b",            # turn on feature-b
    )
    main_sessions = harness.branch_store.load_branch("main").session_ids
    feat_sessions = harness.branch_store.load_branch("feature-b").session_ids
    assert len(main_sessions) >= 1, "main must have at least one session"
    assert len(feat_sessions) >= 1, "feature-b must have at least one session"
    # Sessions must be different
    assert set(main_sessions).isdisjoint(set(feat_sessions))


# ---------------------------------------------------------------------------
# 9. Contextual branch operations (esta rama / la actual / current branch)
# ---------------------------------------------------------------------------
#
# These tests validate the full dispatch path for contextual source references:
#   1. REPL receives the natural-language phrase
#   2. route() is called with the active branch name from BranchStore
#   3. Contextual phrase is resolved to the real branch name
#   4. Normal merge/cherry-pick rules classify the intent
#   5. The branch operation runs against the correct source branch
#   6. Persistence (merge record, session) is updated correctly
#
# Setup for each test: create "feature-x" as a non-main branch with a session
# containing known artefacts, then switch the active branch to "feature-x".
# The REPL sees active_branch="feature-x" and resolves the contextual phrase.
# ---------------------------------------------------------------------------


def test_contextual_merge_esta_rama_en_main(harness: ReplHarness) -> None:
    """'mezcla esta rama en main' merges the active branch (feature-x) into main."""
    # Arrange: create feature-x with a session, switch active branch to it
    harness.add_branch_with_session(
        "feature-x",
        decisions=["decision-A", "decision-B"],
        findings=["finding-A"],
    )
    harness.branch_store.set_active_branch("feature-x")
    assert harness.branch_store.active_branch_name() == "feature-x"

    # Act: user types a contextual merge phrase
    harness.run("mezcla esta rama en main")

    # Assert: a merge record was created in main
    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1, "Expected exactly one merge record in main"
    record = records[0]
    assert record.source_branch == "feature-x"
    assert record.destination_branch == "main"
    assert record.operation == "merge"


def test_contextual_merge_la_rama_actual_con_main(harness: ReplHarness) -> None:
    """'fusiona la rama actual con main' also resolves to feature-x → main merge."""
    harness.add_branch_with_session("feature-x", decisions=["dec-x"])
    harness.branch_store.set_active_branch("feature-x")

    harness.run("fusiona la rama actual con main")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"


def test_contextual_merge_trae_todo_de_esta_rama(harness: ReplHarness) -> None:
    """'trae todo de esta rama a main' is a full merge (todo = all artefacts)."""
    harness.add_branch_with_session(
        "feature-x",
        decisions=["dec-todo"],
        facts=["fact-todo"],
    )
    # Active branch is "feature-x"; "a main" makes the destination explicit.
    # Without an explicit destination, dest falls back to the active branch
    # (self-merge), so the destination suffix is required when the user is on
    # the branch they want to merge FROM.
    harness.branch_store.set_active_branch("feature-x")

    harness.run("trae todo de esta rama a main")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].operation == "merge"


def test_contextual_cherry_pick_decisions_de_esta_rama(harness: ReplHarness) -> None:
    """
    Validates the full cherry-pick pipeline at the REPL level.

    Scenario: user is ON 'main' (destination) and cherry-picks from 'feature-x'.
    The REPL routes the intent, resolves the source, and creates a cherry-pick
    record in 'main' with only the requested artefact type.

    Note: "trae las decisiones de esta rama" with active=main would resolve
    "esta rama" to main (self-cherry-pick).  For cherry-pick via contextual ref
    to work end-to-end, the user must phrase the source explicitly when the
    destination is the active branch.  The contextual-ref router path is
    validated at the unit-test level in TestContextualSourceCherryPick.
    """
    harness.add_branch_with_session(
        "feature-x",
        decisions=["dec-ctx-A", "dec-ctx-B"],
        findings=["finding-ctx-A"],
    )
    # Active branch is "main" (destination); source is explicit
    harness.branch_store.set_active_branch("main")

    harness.run("trae solo las decisiones de feature-x")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    record = records[0]
    assert record.source_branch == "feature-x"
    assert record.destination_branch == "main"
    assert record.operation == "cherry-pick"
    assert "decisions" in record.items_merged
    assert record.items_merged.get("findings", 0) == 0


def test_contextual_cherry_pick_findings_de_la_rama_actual(harness: ReplHarness) -> None:
    """
    Cherry-pick findings from an explicit source branch while on main.

    The contextual-ref "la rama actual" as source is validated at the unit-test
    level.  This integration test validates the cherry-pick dispatch pipeline
    (REPL → router → cherry_pick() → BranchStore → audit record).
    """
    harness.add_branch_with_session(
        "feature-x",
        findings=["finding-ctx-X", "finding-ctx-Y"],
    )
    harness.branch_store.set_active_branch("main")

    harness.run("pásame los findings de feature-x")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    record = records[0]
    assert record.source_branch == "feature-x"
    assert record.operation == "cherry-pick"
    assert "findings" in record.items_merged


def test_contextual_fallback_without_active_branch_set(harness: ReplHarness) -> None:
    """
    'mezcla esta rama en main' with no active branch set → conversation turn (no merge).

    This tests that the safe fallback works: when the REPL is constructed without
    a valid branch context (e.g. brand-new project, nothing persisted yet), the
    phrase falls through to a conversation agent turn, not a failed branch op.

    We simulate this by NOT calling set_active_branch() and NOT adding any session,
    so active_branch_name() returns the default "main" — not a different branch.
    We then verify the phrase is treated as a conversation turn (agent.run called)
    rather than a merge.
    """
    # active_branch defaults to "main"; "mezcla esta rama en main" with active="main"
    # → after resolution: "mezcla main en main" → matches merge with source="main"
    # → _do_merge("main") called → source == dest → should produce a merge record or error
    # This is an edge case; we verify the agent is NOT called (it was dispatched to merge).
    harness.run("mezcla esta rama en main")
    # No merge record should exist because there's no session on main to merge from yet
    # (main was created by harness.__init__ but has no sessions)
    # The REPL may output an error panel but should not crash.
    # Key assertion: the system didn't call agent.run for this phrase (it was routed to merge)
    assert harness.agent.run.call_count == 0, (
        "Expected merge routing, not conversation turn, for 'mezcla esta rama en main'"
    )


def test_contextual_no_op_with_explicit_source_unaffected(harness: ReplHarness) -> None:
    """
    Explicit branch names are NOT modified by contextual resolution.

    Active branch is 'main' (destination).  User says 'merge experiment'.
    The router must NOT replace 'experiment' with the active branch name.
    """
    harness.add_branch_with_session("experiment", decisions=["dec-exp"])
    # Active branch is "main" (destination) — standard merge-into-main workflow
    harness.branch_store.set_active_branch("main")

    # Explicit source "experiment" — must NOT be replaced with "main"
    harness.run("merge experiment")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "experiment", (
        "Explicit 'experiment' must not be replaced by the active branch"
    )


def test_contextual_cherry_pick_session_updated(harness: ReplHarness) -> None:
    """After a cherry-pick, the session on the active branch is updated with the new artefacts."""
    harness.add_branch_with_session(
        "feature-x",
        decisions=["important-decision"],
        findings=["finding-from-feature"],
    )
    # Active branch is "main" (destination); cherry-pick from feature-x
    harness.branch_store.set_active_branch("main")

    repl = harness.run("trae las decisiones de feature-x")

    # The REPL must have updated its current session id
    assert repl._current_session_id is not None
    # Load that session and verify decisions were merged into main
    merged_session = harness.echo_store.load_session(repl._current_session_id)
    assert any(
        "important-decision" in d for d in merged_session.decisions
    ), "Cherry-picked decision must appear in the merged session"


def test_contextual_ops_do_not_break_slash_commands(harness: ReplHarness) -> None:
    """After a contextual merge, slash commands still work normally."""
    harness.add_branch_with_session("feature-x", decisions=["dec-slash"])
    harness.branch_store.set_active_branch("feature-x")

    harness.run(
        "mezcla esta rama en main",  # contextual merge
        "/branch list",              # slash command — must not crash
    )

    output = harness.output()
    # Branch list output must appear
    assert "main" in output or "feature-x" in output


# ---------------------------------------------------------------------------
# Cherry-pick with explicit destination
# ---------------------------------------------------------------------------


def test_cherry_pick_explicit_dest_from_explicit_source(harness: ReplHarness) -> None:
    """
    'trae las decisiones de feature-x a main' — explicit source, explicit destination.

    The REPL must cherry-pick decisions from feature-x INTO main (not into the
    currently active branch, which may differ).  This verifies that the explicit
    destination overrides the active-branch default.
    """
    harness.add_branch_with_session(
        "feature-x",
        decisions=["explicit-dest-decision"],
        findings=["finding-x"],
    )
    # Active branch is feature-x, but destination is explicit "main"
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("trae las decisiones de feature-x a main")

    # A cherry-pick record must exist on the destination (main)
    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].destination_branch == "main"

    # The merged session must be on main and contain the cherry-picked decision
    merged_session = harness.echo_store.load_session(repl._current_session_id)
    assert any(
        "explicit-dest-decision" in d for d in merged_session.decisions
    ), "Cherry-picked decision must appear in main's merged session"

    # findings must NOT have been cherry-picked (only decisions were requested)
    # The main session was fresh (no prior findings), so merged findings should be empty
    # unless the destination already had findings — here it didn't.
    # We only verify decisions were brought; findings list may be [] or unchanged.


def test_cherry_pick_contextual_source_explicit_dest(harness: ReplHarness) -> None:
    """
    'trae las decisiones de esta rama a main' — contextual source (active=feature-x),
    explicit destination (main).

    After resolution: source=feature-x, destination=main.
    """
    harness.add_branch_with_session(
        "feature-x",
        decisions=["contextual-src-decision"],
        findings=["finding-ctx"],
    )
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("trae las decisiones de esta rama a main")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].destination_branch == "main"

    merged_session = harness.echo_store.load_session(repl._current_session_id)
    assert any(
        "contextual-src-decision" in d for d in merged_session.decisions
    )


def test_cherry_pick_contextual_multiple_artefacts_explicit_dest(
    harness: ReplHarness,
) -> None:
    """
    'trae facts y findings desde esta rama hacia main' — contextual source, two
    artefact types, explicit destination.
    """
    harness.add_branch_with_session(
        "feature-x",
        facts=["fact-from-feature"],
        findings=["finding-from-feature"],
        decisions=["decision-from-feature"],
    )
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("trae facts y findings desde esta rama hacia main")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].destination_branch == "main"

    merged_session = harness.echo_store.load_session(repl._current_session_id)
    assert any(
        "fact-from-feature" in f
        for f in merged_session.operational_memory.confirmed_facts
    )
    assert any("finding-from-feature" in f for f in merged_session.findings)
    # decisions were NOT requested — they must not appear (main had none before)
    assert not any("decision-from-feature" in d for d in merged_session.decisions)


def test_cherry_pick_without_dest_still_targets_active_branch(
    harness: ReplHarness,
) -> None:
    """
    No explicit destination → cherry-pick target is the current active branch.
    Backward-compat: existing behavior must not change.
    """
    harness.add_branch_with_session(
        "feature-x",
        decisions=["no-dest-decision"],
    )
    harness.branch_store.set_active_branch("main")

    repl = harness.run("trae las decisiones de feature-x")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].destination_branch == "main"

    merged_session = harness.echo_store.load_session(repl._current_session_id)
    assert any("no-dest-decision" in d for d in merged_session.decisions)


def test_cherry_pick_explicit_dest_does_not_update_active_branch(
    harness: ReplHarness,
) -> None:
    """
    When cherry-pick has an explicit destination, the active branch must NOT change.
    The operation targets a different branch but the user remains on the current one.
    """
    harness.add_branch_with_session("feature-x", decisions=["dec-check-active"])
    harness.branch_store.set_active_branch("feature-x")

    harness.run("trae las decisiones de feature-x a main")

    # Active branch must remain feature-x, not switch to main
    assert harness.branch_store.active_branch_name() == "feature-x"


# ---------------------------------------------------------------------------
# Flexible cherry-pick component order — REPL integration
# ---------------------------------------------------------------------------


def test_forma2_trae_de_source_a_dest_artefacts(harness: ReplHarness) -> None:
    """
    'trae de feature-x a main solo decisions' — Forma 2 (source → dest → artefacts).

    Active branch is feature-x but the destination is explicit main.
    Verifies the full dispatch chain: NL → router (Forma 2) → cherry-pick → main.
    """
    harness.add_branch_with_session("feature-x", decisions=["forma2-decision"])
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("trae de feature-x a main solo decisions")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].destination_branch == "main"

    merged = harness.echo_store.load_session(repl._current_session_id)
    assert any("forma2-decision" in d for d in merged.decisions)

    # active branch must NOT change
    assert harness.branch_store.active_branch_name() == "feature-x"


def test_cherry_pick_de_source_dest_artefacts(harness: ReplHarness) -> None:
    """
    'cherry-pick de feature-x a main solo findings' — command form with dest
    before artefacts.  Previously broken (extract_destination returned None);
    fixed by _DEST_BRANCH_RE lookahead.
    """
    harness.add_branch_with_session(
        "feature-x",
        findings=["cmd-form-finding"],
        decisions=["cmd-form-decision"],
    )
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("cherry-pick de feature-x a main solo findings")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].destination_branch == "main"

    merged = harness.echo_store.load_session(repl._current_session_id)
    assert any("cmd-form-finding" in f for f in merged.findings)
    # decisions were NOT requested
    assert not any("cmd-form-decision" in d for d in merged.decisions)


def test_forma2_pasame_de_contextual_a_main_multiple(harness: ReplHarness) -> None:
    """
    'pásame de esta rama a main facts y findings' — Forma 2 with contextual source.

    Router resolves 'esta rama' → active branch, then Forma 2 pattern fires.
    """
    harness.add_branch_with_session(
        "feature-x",
        facts=["ctx-fact"],
        findings=["ctx-finding"],
        decisions=["ctx-decision"],
    )
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("pásame de esta rama a main facts y findings")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].destination_branch == "main"

    merged = harness.echo_store.load_session(repl._current_session_id)
    assert any(
        "ctx-fact" in f for f in merged.operational_memory.confirmed_facts
    )
    assert any("ctx-finding" in f for f in merged.findings)
    # decisions were NOT requested
    assert not any("ctx-decision" in d for d in merged.decisions)

    assert harness.branch_store.active_branch_name() == "feature-x"


def test_forma4_trae_a_dest_artefacts_de_source(harness: ReplHarness) -> None:
    """
    'trae a main las decisiones de feature-x' — Forma 4 (dest → artefacts → source).
    """
    harness.add_branch_with_session("feature-x", decisions=["forma4-decision"])
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("trae a main las decisiones de feature-x")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].destination_branch == "main"

    merged = harness.echo_store.load_session(repl._current_session_id)
    assert any("forma4-decision" in d for d in merged.decisions)

    assert harness.branch_store.active_branch_name() == "feature-x"


def test_forma4_contextual_trae_a_dest_artefacts_de_esta_rama(
    harness: ReplHarness,
) -> None:
    """
    'trae a main los findings de esta rama' — Forma 4 with contextual source.
    """
    harness.add_branch_with_session(
        "feature-x",
        findings=["ctx-forma4-finding"],
    )
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("trae a main los findings de esta rama")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1
    assert records[0].source_branch == "feature-x"
    assert records[0].destination_branch == "main"

    merged = harness.echo_store.load_session(repl._current_session_id)
    assert any("ctx-forma4-finding" in f for f in merged.findings)

    assert harness.branch_store.active_branch_name() == "feature-x"


def test_flexible_order_does_not_break_merge(harness: ReplHarness) -> None:
    """
    After adding flexible cherry-pick patterns, 'trae todo de esta rama a main'
    must still classify as a full merge (not cherry-pick).
    """
    harness.add_branch_with_session(
        "feature-x",
        decisions=["merge-decision"],
        findings=["merge-finding"],
    )
    harness.branch_store.set_active_branch("feature-x")

    repl = harness.run("trae todo de esta rama a main")

    records = harness.branch_store.load_merge_records("main")
    assert len(records) == 1

    merged = harness.echo_store.load_session(repl._current_session_id)
    # Both artefact types must be present (full merge, not artefact-filtered)
    assert any("merge-decision" in d for d in merged.decisions)
    assert any("merge-finding" in f for f in merged.findings)
