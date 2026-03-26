from __future__ import annotations

from unittest.mock import patch

from echo.config import Settings
from echo.tools.shell_policy import validate_shell_command
from tests.runtime_fixtures import FakeBackend, RuntimeTestCase


class ShellPolicyTests(RuntimeTestCase):
    def test_settings_default_to_safe_shell_and_write_disabled(self) -> None:
        settings = Settings()
        self.assertFalse(settings.allow_shell)
        self.assertFalse(settings.allow_write)

    def test_shell_policy_allows_only_explicit_safe_prefixes(self) -> None:
        self.assertTrue(validate_shell_command("python3 -m unittest discover -s tests -p test_*.py").allowed)
        self.assertFalse(validate_shell_command("python3 script.py").allowed)
        self.assertFalse(validate_shell_command("python3 -m pip install typer").allowed)

    def test_run_shell_rejects_shell_metacharacters(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("run_shell", {"command": "git status --short && pwd"})
        self.assertEqual(result["policy_decision"], "blocked")
        self.assertIn("metacharacters", result["error"])

    def test_run_shell_rejects_unlisted_executable(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("run_shell", {"command": "echo hello"})
        self.assertEqual(result["policy_decision"], "blocked")
        self.assertIn("not allowed by policy", result["error"])

    def test_run_shell_rejects_destructive_executable(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("run_shell", {"command": "rm -rf ."})
        self.assertEqual(result["policy_decision"], "blocked")
        self.assertIn("blocked by policy", result["error"])

    def test_run_shell_executes_allowed_argv_without_shell(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("run_shell", {"command": "python3 -m unittest discover -s tests -p test_*.py"})
        self.assertEqual(result["policy_decision"], "allowed")
        self.assertEqual(result["argv"][:3], ["python3", "-m", "unittest"])
        self.assertEqual(result["returncode"], 0)

    def test_git_status_runs_through_same_shell_policy(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        (self.root / ".git").mkdir()
        result = runtime.tools.execute("git_status", {})
        self.assertEqual(result["policy_decision"], "allowed")
        self.assertEqual(result["argv"][:2], ["git", "status"])

    def test_search_text_falls_back_to_python_when_rg_missing(self) -> None:
        runtime, _ = self._runtime(FakeBackend([]))
        (self.root / "echo" / "extra.py").write_text("VALUE = 'needle'\n", encoding="utf-8")
        with patch("echo.tools.registry.shutil.which", return_value=None):
            result = runtime.tools.execute("search_text", {"query": "needle"})
        self.assertEqual(result["engine"], "python")
        self.assertTrue(any("echo/extra.py:1:" in line for line in result["matches"]))

    def test_search_symbol_is_explicitly_python_only(self) -> None:
        runtime, _ = self._runtime(FakeBackend([]))
        result = runtime.tools.execute("search_symbol", {"symbol": "run"})
        self.assertEqual(result["scope"], "python-only")
