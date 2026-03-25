from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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


def detect_validation_plan(project_root: Path) -> ValidationPlan:
    pyproject = project_root / "pyproject.toml"
    pytest_ini = project_root / "pytest.ini"
    conftest = project_root / "conftest.py"
    tests_dir = project_root / "tests"
    package_json = project_root / "package.json"

    pyproject_text = pyproject.read_text(encoding="utf-8") if pyproject.exists() else ""
    has_pytest_config = pytest_ini.exists() or conftest.exists() or "[tool.pytest" in pyproject_text or "pytest" in pyproject_text
    has_unittest_tests = tests_dir.exists() and any(tests_dir.rglob("test_*.py"))
    has_python = any(project_root.rglob("*.py"))

    if has_pytest_config:
        return ValidationPlan(strategy="pytest", command="python3 -m pytest", reason="pytest config or layout detected")
    if has_unittest_tests:
        return ValidationPlan(
            strategy="unittest",
            command="python3 -m unittest discover -s tests -p test_*.py",
            reason="unittest-style tests detected",
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
                return ValidationPlan(strategy=f"{manager}-run-lint" if manager != "yarn" else "yarn-lint", command=command, reason="package.json lint script detected")
            if "typecheck" in scripts:
                manager = "npm"
                if (project_root / "pnpm-lock.yaml").exists():
                    manager = "pnpm"
                command = {"npm": "npm run typecheck", "pnpm": "pnpm run typecheck"}[manager]
                return ValidationPlan(strategy=f"{manager}-run-typecheck", command=command, reason="package.json typecheck script detected")
        return ValidationPlan(strategy="unknown", command="", reason="package.json detected but no safe validation script found")

    if has_python:
        target = "echo" if (project_root / "echo").exists() else "."
        return ValidationPlan(strategy="compileall", command=f"python3 -m compileall {target}", reason="python files detected without test runner config")

    return ValidationPlan(strategy="unknown", command="", reason="no supported validation strategy detected")


def detect_validation_strategy(project_root: Path) -> str:
    return detect_validation_plan(project_root).strategy
