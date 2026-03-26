from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re

from echo.fsignore import iter_project_files

VALIDATION_STRATEGIES = {
    "pytest",
    "unittest",
    "npm-test",
    "pnpm-test",
    "yarn-test",
    "npm-run-lint",
    "pnpm-run-lint",
    "yarn-lint",
    "npm-run-typecheck",
    "pnpm-run-typecheck",
    "yarn-typecheck",
    "compileall",
    "unknown",
}


@dataclass(slots=True)
class ValidationPlan:
    strategy: str
    command: str = ""
    reason: str = ""


def _read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _has_pytest_configuration(project_root: Path) -> tuple[bool, str]:
    if (project_root / "pytest.ini").exists():
        return True, "pytest.ini present"
    if (project_root / "conftest.py").exists():
        return True, "conftest.py present"
    if re.search(r"(?m)^\s*\[pytest\]\s*$", _read_text(project_root / "tox.ini")):
        return True, "tox.ini pytest section present"
    if re.search(r"(?m)^\s*\[tool:pytest\]\s*$", _read_text(project_root / "setup.cfg")):
        return True, "setup.cfg tool:pytest section present"
    if re.search(r"(?m)^\s*\[tool\.pytest(?:\.ini_options)?\]\s*$", _read_text(project_root / "pyproject.toml")):
        return True, "pyproject pytest section present"
    return False, ""


def _python_test_layout(project_root: Path) -> tuple[bool, str]:
    tests_dir = project_root / "tests"
    if not tests_dir.exists() or not tests_dir.is_dir():
        return False, ""
    test_files = [path.relative_to(project_root) for path in iter_project_files(tests_dir, "*.py")]
    if any(path.name.startswith("test_") for path in test_files):
        return True, "tests/test_*.py layout detected"
    if any(path.name.endswith("_test.py") for path in test_files):
        return True, "tests/*_test.py layout detected"
    return False, ""


def infer_validation_strategy_from_evidence(*, project_files: list[str] | None = None, validation_commands: list[str] | None = None) -> str:
    commands = " ".join(validation_commands or []).lower()
    files = [item.lower() for item in (project_files or [])]
    if "python -m pytest" in commands or "python3 -m pytest" in commands or re.search(r"(^|\s)pytest(\s|$)", commands):
        return "pytest"
    if "python -m unittest" in commands or "python3 -m unittest" in commands or "unittest" in commands:
        return "unittest"
    if "npm test" in commands:
        return "npm-test"
    if "pnpm test" in commands:
        return "pnpm-test"
    if "yarn test" in commands:
        return "yarn-test"
    if "npm run lint" in commands:
        return "npm-run-lint"
    if "pnpm run lint" in commands:
        return "pnpm-run-lint"
    if "yarn lint" in commands:
        return "yarn-lint"
    if "npm run typecheck" in commands:
        return "npm-run-typecheck"
    if "pnpm run typecheck" in commands:
        return "pnpm-run-typecheck"
    if "yarn typecheck" in commands:
        return "yarn-typecheck"
    if "compileall" in commands:
        return "compileall"
    if any(item.endswith(".py") for item in files):
        return "unknown"
    return "unknown"


def detect_validation_plan(project_root: Path) -> ValidationPlan:
    package_json = project_root / "package.json"
    has_python = any(iter_project_files(project_root, "*.py"))

    has_pytest_config, pytest_reason = _has_pytest_configuration(project_root)
    if has_pytest_config:
        return ValidationPlan(strategy="pytest", command="python3 -m pytest", reason=pytest_reason)

    has_unittest_tests, unittest_reason = _python_test_layout(project_root)
    if has_unittest_tests:
        return ValidationPlan(
            strategy="unittest",
            command="python3 -m unittest discover -s tests -p test_*.py",
            reason=unittest_reason,
        )

    if package_json.exists():
        package = _read_json(package_json)
        scripts = package.get("scripts", {})
        if isinstance(scripts, dict):
            if "test" in scripts:
                manager = "npm"
                if (project_root / "pnpm-lock.yaml").exists():
                    manager = "pnpm"
                elif (project_root / "yarn.lock").exists():
                    manager = "yarn"
                command = {"npm": "npm test", "pnpm": "pnpm test", "yarn": "yarn test"}[manager]
                return ValidationPlan(strategy=f"{manager}-test", command=command, reason="package.json test script detected")
            if "lint" in scripts:
                manager = "npm"
                if (project_root / "pnpm-lock.yaml").exists():
                    manager = "pnpm"
                elif (project_root / "yarn.lock").exists():
                    manager = "yarn"
                command = {"npm": "npm run lint", "pnpm": "pnpm run lint", "yarn": "yarn lint"}[manager]
                return ValidationPlan(
                    strategy=f"{manager}-run-lint" if manager != "yarn" else "yarn-lint",
                    command=command,
                    reason="package.json lint script detected and no safer test runner was detected",
                )
            if "typecheck" in scripts:
                manager = "npm"
                if (project_root / "pnpm-lock.yaml").exists():
                    manager = "pnpm"
                elif (project_root / "yarn.lock").exists():
                    manager = "yarn"
                command = {"npm": "npm run typecheck", "pnpm": "pnpm run typecheck", "yarn": "yarn typecheck"}[manager]
                return ValidationPlan(
                    strategy=f"{manager}-run-typecheck" if manager != "yarn" else "yarn-typecheck",
                    command=command,
                    reason="package.json typecheck script detected and no safer test runner was detected",
                )
        return ValidationPlan(strategy="unknown", command="", reason="package.json detected but no safe validation script found")

    if has_python:
        target = "echo" if (project_root / "echo").exists() else "."
        return ValidationPlan(strategy="compileall", command=f"python3 -m compileall {target}", reason="python files detected without test runner config")

    return ValidationPlan(strategy="unknown", command="", reason="no supported validation strategy detected")


def detect_validation_strategy(
    project_root: Path | None = None,
    project_files: list[str] | None = None,
    validation_commands: list[str] | None = None,
) -> str:
    if project_root is not None:
        return detect_validation_plan(project_root).strategy
    return infer_validation_strategy_from_evidence(project_files=project_files, validation_commands=validation_commands)
