from __future__ import annotations

import json

from echo.cognition.validation import detect_validation_plan, detect_validation_strategy
from tests.runtime_fixtures import FakeBackend, RuntimeTestCase


class ValidationDetectionTests(RuntimeTestCase):
    def test_detect_validation_plan_prefers_pytest_when_configured(self) -> None:
        (self.root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "pytest")
        self.assertEqual(plan.command, "python3 -m pytest")

    def test_detect_validation_plan_ignores_incidental_pytest_text_in_pyproject(self) -> None:
        (self.root / "pyproject.toml").write_text('[project]\nname = "demo"\ndependencies = ["pytest"]\n', encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "unittest")

    def test_detect_validation_plan_uses_unittest_layout(self) -> None:
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "unittest")
        self.assertEqual(plan.command, "python3 -m unittest discover -s tests -p test_*.py")

    def test_detect_validation_plan_falls_back_to_compileall_for_python_without_tests(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "compileall")
        self.assertIn("python3 -m compileall", plan.command)

    def test_detect_validation_plan_uses_package_json_test_script(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        for item in (self.root / "echo").glob("*"):
            item.unlink()
        (self.root / "echo").rmdir()
        (self.root / "package.json").write_text(json.dumps({"scripts": {"test": "vitest"}}), encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "npm-test")
        self.assertEqual(plan.command, "npm test")

    def test_detect_validation_plan_returns_unknown_for_package_without_safe_script(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        for item in (self.root / "echo").glob("*"):
            item.unlink()
        (self.root / "echo").rmdir()
        (self.root / "package.json").write_text(json.dumps({"scripts": {"dev": "vite"}}), encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "unknown")
        self.assertEqual(plan.command, "")

    def test_detect_validation_plan_uses_typecheck_as_last_safe_js_fallback(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        for item in (self.root / "echo").glob("*"):
            item.unlink()
        (self.root / "echo").rmdir()
        (self.root / "yarn.lock").write_text("", encoding="utf-8")
        (self.root / "package.json").write_text(json.dumps({"scripts": {"typecheck": "tsc --noEmit"}}), encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "yarn-typecheck")
        self.assertEqual(plan.command, "yarn typecheck")

    def test_detect_validation_strategy_from_evidence_uses_commands_not_python_file_presence(self) -> None:
        strategy = detect_validation_strategy(project_files=["echo/sample.py"], validation_commands=["python3 -m unittest discover -s tests -p test_*.py"])
        self.assertEqual(strategy, "unittest")

    def test_validate_project_returns_unknown_without_safe_strategy(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        for item in (self.root / "echo").glob("*"):
            item.unlink()
        (self.root / "echo").rmdir()
        (self.root / "package.json").write_text(json.dumps({"scripts": {"dev": "vite"}}), encoding="utf-8")
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("validate_project", {})
        self.assertEqual(result["validation_strategy"], "unknown")
        self.assertIn("No safe validation strategy", result["error"])

    def test_detect_validation_plan_ignores_virtualenv_noise(self) -> None:
        noise_tests = self.root / ".venv" / "lib" / "python3.12" / "site-packages" / "tests"
        noise_tests.mkdir(parents=True)
        (noise_tests / "test_noise.py").write_text("def test_noise():\n    assert False\n", encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "unittest")
