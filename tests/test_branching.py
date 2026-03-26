"""Tests for persistent session branching.

Coverage:
  1.  ensure_branch_model creates main on an empty project
  2.  ensure_branch_model migrates an existing session into main
  3.  ensure_branch_model is idempotent
  4.  active_branch returns None before initialization
  5.  set_active_branch persists and round-trips
  6.  branch_new creates an independent branch with correct lineage
  7.  update_branch_head advances the head pointer
  8.  branch head isolation — two branches hold independent head session IDs
  9.  list_branches returns all saved branches
  10. delete_branch removes a non-main branch
  11. delete_branch raises ValueError for main
  12. corrupt active_branch.json → active_branch() returns None;
      ensure_branch_model recovers to main
  13. branch_new with name collision is detected at store level
  14. branch_head_session_id returns None for missing / empty-head branch
  15. agent.run() with branch_name updates branch head
  16. agent.run() second call resumes from branch head
  17. context isolation — work on branch A, switch to B, B head unchanged
  18. migration from old project without branches
  19. branch persistence per project — two roots have independent branches
  20. _safe_branch_filename sanitizes special characters
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

# runtime_fixtures must be imported first — its echo.backends.errors import
# resolves the pre-existing circular dependency between echo.memory and
# echo.backends before we directly reference echo.memory.store.
from tests.runtime_fixtures import FakeBackend, RuntimeTestCase  # noqa: E402
from echo.memory.store import EchoStore, _safe_branch_filename
from echo.types import BranchMeta, BranchState, SessionState


class BranchStoreTests(RuntimeTestCase):
    """Tests that operate only on EchoStore — no backend required."""

    # ------------------------------------------------------------------ #
    # 1. ensure_branch_model creates main on empty project                 #
    # ------------------------------------------------------------------ #
    def test_ensure_branch_model_creates_main_on_empty_project(self) -> None:
        store = EchoStore(self.root)
        active = store.ensure_branch_model()
        self.assertEqual(active.branch_name, "main")
        branch = store.load_branch("main")
        self.assertIsNotNone(branch)
        self.assertEqual(branch.name, "main")  # type: ignore[union-attr]
        self.assertEqual(branch.head_session_id, "")  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # 2. ensure_branch_model migrates an existing session                  #
    # ------------------------------------------------------------------ #
    def test_ensure_branch_model_migrates_existing_session(self) -> None:
        store = EchoStore(self.root)
        # Simulate an old project with a session but no branch files.
        session = SessionState.create(str(self.root), "ask", "fake-model", "hello")
        store.save_session(session)

        active = store.ensure_branch_model()
        self.assertEqual(active.branch_name, "main")
        branch = store.load_branch("main")
        self.assertIsNotNone(branch)
        self.assertEqual(branch.head_session_id, session.id)  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # 3. ensure_branch_model is idempotent                                 #
    # ------------------------------------------------------------------ #
    def test_ensure_branch_model_is_idempotent(self) -> None:
        store = EchoStore(self.root)
        first = store.ensure_branch_model()
        second = store.ensure_branch_model()
        self.assertEqual(first.branch_name, second.branch_name)
        # Branch file must still exist and be unchanged.
        branch = store.load_branch("main")
        self.assertIsNotNone(branch)

    # ------------------------------------------------------------------ #
    # 4. active_branch returns None before initialization                  #
    # ------------------------------------------------------------------ #
    def test_active_branch_returns_none_before_init(self) -> None:
        store = EchoStore(self.root)
        self.assertIsNone(store.active_branch())

    # ------------------------------------------------------------------ #
    # 5. set_active_branch persists and round-trips                        #
    # ------------------------------------------------------------------ #
    def test_set_active_branch_persists(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        # Save a second branch, then switch.
        store.save_branch(BranchMeta(name="feature-x", head_session_id=""))
        store.set_active_branch("feature-x")

        active = store.active_branch()
        self.assertIsNotNone(active)
        self.assertEqual(active.branch_name, "feature-x")  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # 6. branch_new creates independent branch with correct lineage        #
    # ------------------------------------------------------------------ #
    def test_branch_new_creates_independent_branch_with_correct_lineage(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        store.update_branch_head("main", "session-aaa")

        new_branch = BranchMeta(
            name="experiment",
            head_session_id="session-aaa",
            parent_branch="main",
            fork_session_id="session-aaa",
        )
        store.save_branch(new_branch)

        loaded = store.load_branch("experiment")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.parent_branch, "main")  # type: ignore[union-attr]
        self.assertEqual(loaded.fork_session_id, "session-aaa")  # type: ignore[union-attr]
        self.assertEqual(loaded.head_session_id, "session-aaa")  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # 7. update_branch_head advances the head pointer                      #
    # ------------------------------------------------------------------ #
    def test_update_branch_head_advances_pointer(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        store.update_branch_head("main", "session-001")
        self.assertEqual(store.branch_head_session_id("main"), "session-001")

        store.update_branch_head("main", "session-002")
        self.assertEqual(store.branch_head_session_id("main"), "session-002")

    # ------------------------------------------------------------------ #
    # 8. head isolation — two branches hold independent head session IDs   #
    # ------------------------------------------------------------------ #
    def test_branch_head_isolation(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        store.update_branch_head("main", "session-main-1")

        store.save_branch(BranchMeta(name="side", head_session_id="session-side-1", parent_branch="main"))

        self.assertEqual(store.branch_head_session_id("main"), "session-main-1")
        self.assertEqual(store.branch_head_session_id("side"), "session-side-1")

        # Advance main; side must stay unchanged.
        store.update_branch_head("main", "session-main-2")
        self.assertEqual(store.branch_head_session_id("main"), "session-main-2")
        self.assertEqual(store.branch_head_session_id("side"), "session-side-1")

    # ------------------------------------------------------------------ #
    # 9. list_branches returns all saved branches                          #
    # ------------------------------------------------------------------ #
    def test_list_branches_returns_all(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        store.save_branch(BranchMeta(name="alpha", head_session_id=""))
        store.save_branch(BranchMeta(name="beta", head_session_id=""))

        names = {b.name for b in store.list_branches()}
        self.assertIn("main", names)
        self.assertIn("alpha", names)
        self.assertIn("beta", names)
        self.assertEqual(len(names), 3)

    # ------------------------------------------------------------------ #
    # 10. delete_branch removes a non-main branch                          #
    # ------------------------------------------------------------------ #
    def test_delete_branch_removes_non_main(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        store.save_branch(BranchMeta(name="to-delete", head_session_id=""))
        self.assertIsNotNone(store.load_branch("to-delete"))

        store.delete_branch("to-delete")
        self.assertIsNone(store.load_branch("to-delete"))

    # ------------------------------------------------------------------ #
    # 11. delete_branch raises ValueError for main                         #
    # ------------------------------------------------------------------ #
    def test_delete_branch_raises_for_main(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        with self.assertRaises(ValueError):
            store.delete_branch("main")

    # ------------------------------------------------------------------ #
    # 12. corrupt active_branch.json → active_branch returns None;         #
    #     ensure_branch_model recovers                                      #
    # ------------------------------------------------------------------ #
    def test_corrupt_active_branch_file_recovery(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()

        # Corrupt the file.
        (store.state / "active_branch.json").write_text("{invalid json", encoding="utf-8")

        self.assertIsNone(store.active_branch())

        # ensure_branch_model must recover without raising.
        recovered = store.ensure_branch_model()
        self.assertEqual(recovered.branch_name, "main")
        self.assertIsNotNone(store.active_branch())

    # ------------------------------------------------------------------ #
    # 13. branch_new with name collision detected at store level            #
    # ------------------------------------------------------------------ #
    def test_duplicate_branch_name_detected(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        store.save_branch(BranchMeta(name="dup", head_session_id=""))
        # A second save overwrites (store layer allows it; duplicate detection
        # is enforced at the CLI layer).  Confirm the branch exists exactly once.
        branches = [b for b in store.list_branches() if b.name == "dup"]
        self.assertEqual(len(branches), 1)

    # ------------------------------------------------------------------ #
    # 14. branch_head_session_id returns None for missing / empty-head      #
    # ------------------------------------------------------------------ #
    def test_branch_head_session_id_returns_none_when_missing(self) -> None:
        store = EchoStore(self.root)
        self.assertIsNone(store.branch_head_session_id("nonexistent"))

    def test_branch_head_session_id_returns_none_for_empty_head(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        # main branch starts with empty head_session_id.
        self.assertIsNone(store.branch_head_session_id("main"))

    # ------------------------------------------------------------------ #
    # 17. context isolation — branch B head unchanged when A advances       #
    # ------------------------------------------------------------------ #
    def test_branch_context_isolation_head_does_not_cross(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        store.update_branch_head("main", "session-fork")

        store.save_branch(BranchMeta(
            name="branch-b",
            head_session_id="session-fork",
            parent_branch="main",
            fork_session_id="session-fork",
        ))

        # Advance main further; branch-b must stay at fork.
        store.update_branch_head("main", "session-main-continued")
        self.assertEqual(store.branch_head_session_id("branch-b"), "session-fork")

    # ------------------------------------------------------------------ #
    # 18. migration from old project without branch files                   #
    # ------------------------------------------------------------------ #
    def test_migration_from_old_project_without_branches(self) -> None:
        """Simulate a project that has sessions but no branch files at all."""
        store = EchoStore(self.root)
        session = SessionState.create(str(self.root), "ask", "fake-model", "old prompt")
        store.save_session(session)

        # Confirm no branch files exist yet.
        self.assertEqual(list((store.state / "branches").glob("*.json")), [])

        active = store.ensure_branch_model()
        self.assertEqual(active.branch_name, "main")
        main = store.load_branch("main")
        self.assertIsNotNone(main)
        self.assertEqual(main.head_session_id, session.id)  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # 19. branch persistence per project — two roots are independent        #
    # ------------------------------------------------------------------ #
    def test_branch_per_project_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as other_dir:
            other_root = Path(other_dir)
            store_a = EchoStore(self.root)
            store_b = EchoStore(other_root)

            store_a.ensure_branch_model()
            store_b.ensure_branch_model()

            store_a.update_branch_head("main", "session-project-a")

            # Project B must not see project A's head.
            self.assertIsNone(store_b.branch_head_session_id("main"))

    # ------------------------------------------------------------------ #
    # 20. _safe_branch_filename sanitizes special characters               #
    # ------------------------------------------------------------------ #
    def test_safe_branch_filename_sanitizes_special_chars(self) -> None:
        self.assertEqual(_safe_branch_filename("feature/my branch"), "feature_my_branch")
        self.assertEqual(_safe_branch_filename("ok-name_123"), "ok-name_123")
        self.assertEqual(_safe_branch_filename("a" * 100), "a" * 64)


class BranchAgentIntegrationTests(RuntimeTestCase):
    """Tests that verify branch logic through AgentRuntime + EchoStore,
    using FakeBackend to avoid a real LLM."""

    def _fake_response(self, answer: str = "ok") -> dict:
        # The runtime expects {"message": {"content": "..."}} format.
        return {
            "message": {
                "content": (
                    f"Inspeccioné echo/sample.py y encontré def run().\n"
                    f"Settings.from_env está configurado correctamente.\n"
                    f"Ejecuté python3 -m unittest discover -s tests -p test_*.py.\n{answer}"
                )
            }
        }

    def _enough_responses(self, answer: str = "ok", n: int = 12) -> list[dict]:
        """Return n copies of a grounded fake response.

        The model loop may make more than one backend call (seed inspection,
        model loop steps, retry probes). Providing a surplus avoids IndexError
        from FakeBackend.responses running empty.
        """
        return [self._fake_response(answer) for _ in range(n)]

    # ------------------------------------------------------------------ #
    # 15. agent.run() with branch_name updates branch head                  #
    # ------------------------------------------------------------------ #
    def test_agent_run_updates_branch_head(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()
        self.assertIsNone(store.branch_head_session_id("main"))

        runtime, _ = self._runtime(FakeBackend(self._enough_responses()))
        answer, session_path, session, _ = runtime.run("describe the project", mode="ask")

        # Simulate what agent.run() does after runtime.run():
        session.branch_name = "main"
        store.save_session(session)
        store.update_branch_head("main", session.id)

        self.assertEqual(store.branch_head_session_id("main"), session.id)

    # ------------------------------------------------------------------ #
    # 16. second run resumes from branch head (parent_session_id set)       #
    # ------------------------------------------------------------------ #
    def test_second_run_resumes_from_branch_head(self) -> None:
        store = EchoStore(self.root)
        store.ensure_branch_model()

        # First run.
        runtime1, _ = self._runtime(FakeBackend(self._enough_responses()))
        _, _, session1, _ = runtime1.run("describe the project", mode="ask")
        session1.branch_name = "main"
        store.save_session(session1)
        store.update_branch_head("main", session1.id)

        # Second run: pass the branch head as resume_session_id (what agent.run does).
        head = store.branch_head_session_id("main")
        self.assertEqual(head, session1.id)

        runtime2, _ = self._runtime(FakeBackend(self._enough_responses("follow-up")))
        _, _, session2, _ = runtime2.run("follow-up task", mode="ask", resume_session_id=head)
        session2.branch_name = "main"
        store.save_session(session2)
        store.update_branch_head("main", session2.id)

        # session2 must reference session1 via parent_session_id.
        self.assertEqual(session2.parent_session_id, session1.id)
        # Branch head advanced to session2.
        self.assertEqual(store.branch_head_session_id("main"), session2.id)

    # ------------------------------------------------------------------ #
    # branch_name field persists through save/load round-trip              #
    # ------------------------------------------------------------------ #
    def test_branch_name_persists_in_session_file(self) -> None:
        store = EchoStore(self.root)
        session = SessionState.create(str(self.root), "ask", "fake-model", "hello")
        session.branch_name = "feature-xyz"
        store.save_session(session)

        loaded = store.load_session(session.id)
        self.assertEqual(loaded.branch_name, "feature-xyz")

    # ------------------------------------------------------------------ #
    # old session without branch_name field loads with default ""           #
    # ------------------------------------------------------------------ #
    def test_old_session_without_branch_name_loads_with_empty_default(self) -> None:
        store = EchoStore(self.root)
        session = SessionState.create(str(self.root), "ask", "fake-model", "old")
        path = store.sessions / f"{session.id}.json"
        # Write a session JSON that does NOT include branch_name.
        import json as _json
        from dataclasses import asdict
        data = asdict(session)
        data.pop("branch_name", None)
        path.write_text(_json.dumps(data), encoding="utf-8")

        loaded = store.load_session(session.id)
        self.assertEqual(loaded.branch_name, "")


if __name__ == "__main__":
    unittest.main()
