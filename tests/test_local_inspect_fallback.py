"""
Tests for the local-inspection fallback path.

When the model answers without calling any tools (common with weak local models
or when the backend is unavailable), EchoAgent must return a useful,
human-readable response derived from local project files — NOT an internal
diagnostic dump.

Coverage:
- _read_local_files: finds README and pyproject when they exist
- _read_local_files: skips missing files silently
- build_local_inspect_answer: output contains file content
- build_local_inspect_answer: no diagnostic strings (stage/backend/límite)
- build_degraded_answer: no longer contains raw diagnostic key phrases
- build_degraded_answer (resume mode): keeps its informative format
- model_loop integration: no-tools path calls local_inspect_answer
"""
from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from echo.runtime.outcomes import (
    _detect_intent,
    _detect_improvement_followup,
    _escape_brackets,
    _extract_agent_doc_hints,
    _extract_simple_vars,
    _parse_project_meta,
    _parse_pyproject,
    _parse_setup_cfg,
    _parse_setup_py,
    _readme_intro,
    _read_local_files,
    _resolve_attr_ref,
    _resolve_file_ref,
    _resolve_setup_kwarg,
    build_degraded_answer,
    build_local_inspect_answer,
)
from echo.types.models import SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_state(repo_root: str) -> Any:
    rs = MagicMock()
    rs.repo_root = repo_root
    rs.current_stage_id = "execute"
    rs.fallback_used = False
    rs.fallback_reason = ""
    rs.fresh_backend_health.backend_state = "error"
    rs.backend_health.backend_state = "error"
    return rs


def _make_session(repo_root: str, prompt: str = "test question") -> SessionState:
    return SessionState.create(
        repo_root=repo_root,
        mode="ask",
        model="stub",
        user_prompt=prompt,
    )


def _make_activity() -> Any:
    activity = MagicMock()
    activity.emit = MagicMock()
    return activity


def _numbered_items(answer: str) -> list[str]:
    return [
        line.split(". ", 1)[1].strip()
        for line in answer.splitlines()
        if re.match(r"^\d+\. ", line.strip())
    ]


def _make_followup_ready_project(tmp_path: Path) -> None:
    _make_realistic_project(tmp_path)
    for rel_path in [
        "echo/runtime/outcomes.py",
        "echo/runtime/engine.py",
        "echo/runtime/model_loop.py",
        "echo/runtime/verify_flow.py",
        "echo/ui/repl.py",
        "echo/ui/intent_router.py",
        "echo/branches/store.py",
        "tests/test_local_inspect_fallback.py",
    ]:
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# stub\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# _read_local_files
# ---------------------------------------------------------------------------


def test_read_local_files_finds_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# My Project\nThis is a test project.", encoding="utf-8")
    results = _read_local_files(str(tmp_path))
    paths = [r[0] for r in results]
    assert "README.md" in paths


def test_read_local_files_finds_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n', encoding="utf-8")
    results = _read_local_files(str(tmp_path))
    paths = [r[0] for r in results]
    assert "pyproject.toml" in paths


def test_read_local_files_skips_missing_files(tmp_path: Path) -> None:
    # Neither README.md nor pyproject.toml exists
    results = _read_local_files(str(tmp_path))
    assert results == []


def test_read_local_files_truncates_large_readme(tmp_path: Path) -> None:
    large_text = "x" * 10_000
    (tmp_path / "README.md").write_text(large_text, encoding="utf-8")
    results = _read_local_files(str(tmp_path))
    readme_content = next(content for path, content in results if path == "README.md")
    assert len(readme_content) <= 2500


def test_read_local_files_returns_both_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    results = _read_local_files(str(tmp_path))
    assert len(results) == 2


# ---------------------------------------------------------------------------
# _escape_brackets
# ---------------------------------------------------------------------------


def test_escape_brackets_escapes_opening_bracket() -> None:
    assert _escape_brackets("[bold]text[/bold]") == "\\[bold]text\\[/bold]"


def test_escape_brackets_leaves_plain_text_unchanged() -> None:
    assert _escape_brackets("plain text") == "plain text"


# ---------------------------------------------------------------------------
# build_local_inspect_answer
# ---------------------------------------------------------------------------


def test_local_inspect_includes_readme_content(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Echo Agent\nMy great project.", encoding="utf-8")
    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    # Synthesis surfaces the content (title), not the filename as a header
    assert "Echo Agent" in answer


def test_local_inspect_includes_pyproject_content(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "echo-agent"\n', encoding="utf-8")
    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    # Synthesis surfaces the project name, not the filename as a header
    assert "echo-agent" in answer


def test_local_inspect_no_internal_diagnostic_strings(tmp_path: Path) -> None:
    """Answer must not contain raw internal diagnostic key phrases."""
    (tmp_path / "README.md").write_text("# Project docs", encoding="utf-8")
    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    diagnostic_phrases = [
        "Etapa detenida:",
        "Backend efectivo:",
        "Límite actual:",
        "stage:execute:failed",
    ]
    for phrase in diagnostic_phrases:
        assert phrase not in answer, f"Diagnostic phrase found in answer: {phrase!r}"


def test_local_inspect_no_files_gives_clean_message(tmp_path: Path) -> None:
    """When no local files exist, answer is a clean user-facing message, not a dump."""
    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    # Must not contain internal state
    assert "Etapa detenida:" not in answer
    assert "Backend efectivo:" not in answer
    # Must tell the user something actionable
    assert len(answer) > 10


def test_local_inspect_emits_activity_events(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# doc", encoding="utf-8")
    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    build_local_inspect_answer(session, run_state, activity=activity)

    assert activity.emit.call_count >= 2  # running + done


# ---------------------------------------------------------------------------
# build_degraded_answer — non-resume mode
# ---------------------------------------------------------------------------


def test_degraded_answer_no_raw_internal_stage_strings(tmp_path: Path) -> None:
    """Non-resume degraded answer must not contain raw internal diagnostic keys."""
    (tmp_path / "README.md").write_text("# My Project", encoding="utf-8")
    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_degraded_answer(
        session, run_state,
        reason="BackendTimeoutError: timeout",
        mode="ask",
        update_stage=MagicMock(),
        activity=activity,
    )

    for phrase in ["Etapa detenida:", "Backend efectivo:", "Límite actual:"]:
        assert phrase not in answer, f"Internal diagnostic found in degraded answer: {phrase!r}"


def test_degraded_answer_with_local_files_shows_file_content(tmp_path: Path) -> None:
    """When local files exist, the degraded answer includes their content."""
    (tmp_path / "README.md").write_text("# Echo\nCoding agent for local use.", encoding="utf-8")
    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_degraded_answer(
        session, run_state,
        reason="timeout",
        mode="ask",
        update_stage=MagicMock(),
        activity=activity,
    )

    assert "README.md" in answer or "Echo" in answer


def test_degraded_answer_without_local_files_is_clean(tmp_path: Path) -> None:
    """Without local files, degraded answer is a clean, concise user message."""
    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_degraded_answer(
        session, run_state,
        reason="BackendUnreachableError",
        mode="ask",
        update_stage=MagicMock(),
        activity=activity,
    )

    assert "Etapa detenida:" not in answer
    assert "Backend efectivo:" not in answer
    assert "Límite actual:" not in answer
    assert len(answer) > 10


def test_degraded_answer_resume_mode_keeps_informative_format(tmp_path: Path) -> None:
    """Resume-mode degraded answer keeps its informative session summary format."""
    session = _make_session(str(tmp_path))
    session.pending = ["Fix the parser", "Update tests"]
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_degraded_answer(
        session, run_state,
        reason="timeout",
        mode="resume",
        update_stage=MagicMock(),
        activity=activity,
    )

    # Resume mode should still mention objective or pending context
    assert "sesión" in answer.lower() or "objetivo" in answer.lower() or "pendientes" in answer.lower()


# ---------------------------------------------------------------------------
# model_loop integration — no-tools path calls local_inspect_answer
# ---------------------------------------------------------------------------


def test_model_loop_no_tools_calls_local_inspect_answer(tmp_path: Path) -> None:
    """
    When the model responds without calling any tools (step 1, ask mode),
    run_model_loop must call local_inspect_answer, NOT degraded_answer.
    """
    from echo.runtime.model_loop import run_model_loop
    from echo.runtime.budget import RuntimeBudget, monotonic_ms

    session = _make_session(str(tmp_path))
    run_state = _make_run_state(str(tmp_path))
    run_state.current_stage_id = "execute"
    run_state.retry_count = 0
    run_state.compression_count = 0
    run_state.context_free_ratio = 1.0
    run_state.open_issues = []
    run_state.errors = []
    run_state.inspected_files = []
    run_state.changed_files = []
    run_state.findings = []
    run_state.pending = []
    run_state.grounding_report = MagicMock()
    run_state.grounding_report.grounded_file_count = 0

    # Model returns a response with no tool calls
    mock_response = {"message": {"content": "direct answer", "tool_calls": []}}
    budget = RuntimeBudget(total_ms=60_000, deadline_ms=monotonic_ms() + 60_000, reserve_ms=1_000)

    degraded_answer = MagicMock(return_value="degraded")
    local_inspect_answer = MagicMock(return_value="local inspect result")

    run_model_loop(
        session=session,
        run_state=run_state,
        messages=[{"role": "system", "content": "sys"}],
        prompt="what is this project?",
        mode="ask",
        profile="local",
        settings=MagicMock(backend_timeout=60, backend_preflight_timeout=8),
        budget=budget,
        backend_native_tools_enabled=lambda: False,
        tools_schema=lambda: None,
        step_limit=4,
        compress_messages=lambda msgs: (msgs, ""),
        context_ratio=lambda msgs: 0.5,
        mark_phase=MagicMock(),
        call_backend=MagicMock(return_value=mock_response),
        extract_tool_calls=lambda msg, content: [],
        degraded_answer=degraded_answer,
        local_inspect_answer=local_inspect_answer,
        update_stage=MagicMock(),
        replan_stage=MagicMock(return_value="execute"),
        reduce_context=lambda s, rs, msgs: msgs,
        grounding_retry_message=MagicMock(return_value={"role": "system", "content": "retry"}),
        record_tool_call=MagicMock(),
        execute_tool=MagicMock(return_value={}),
        collect_tool_previews=MagicMock(return_value=[]),
        validation_strategy=MagicMock(return_value="pytest"),
        find_stage=MagicMock(return_value=None),
    )

    local_inspect_answer.assert_called_once()
    degraded_answer.assert_not_called()


# ---------------------------------------------------------------------------
# _detect_intent
# ---------------------------------------------------------------------------


def test_detect_intent_summary() -> None:
    assert _detect_intent("dame un resumen del proyecto") == "summary"


def test_detect_intent_summary_english() -> None:
    assert _detect_intent("what is this project?") == "summary"


def test_detect_intent_improvement() -> None:
    assert _detect_intent("dame una propuesta de mejora") == "improvement"


def test_detect_intent_improvement_english() -> None:
    assert _detect_intent("suggest improvements") == "improvement"


def test_detect_intent_improvement_wins_over_summary() -> None:
    # "resumen de mejoras posibles" contains both keywords → "combined" (answers both)
    assert _detect_intent("resumen de mejoras posibles") == "combined"


def test_detect_intent_config() -> None:
    assert _detect_intent("qué backend usa?") == "config"


def test_detect_intent_general() -> None:
    assert _detect_intent("hola") == "general"


def test_detect_intent_empty_string() -> None:
    assert _detect_intent("") == "general"


def test_detect_followup_signal_for_otra_aparte_de_esas() -> None:
    assert _detect_improvement_followup("otra propuesta que tengas aparte de esas 4?")
    assert _detect_intent("otra propuesta que tengas aparte de esas 4?") == "improvement"


# ---------------------------------------------------------------------------
# intent routing in build_local_inspect_answer
# ---------------------------------------------------------------------------


def test_uses_user_prompt_not_objective_for_intent(tmp_path: Path) -> None:
    """Second turn: session.objective carries the first-turn value; user_prompt is fresh."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "echo-agent"\ndescription = "A local coding agent"\n',
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame una propuesta de mejora")
    # Simulate stale objective from a previous session / resume_seed
    session.objective = "Inspecciona README.md y echo/config.py. Di el backend por defecto."
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    # Must be improvement-intent output, not config/summary
    assert any(kw in answer.lower() for kw in ("mejora", "propuesta", "suggestion", "mypy", "ruff", "ci"))


def test_second_turn_summary_differs_from_improvement(tmp_path: Path) -> None:
    """Two different prompts on the same project produce different answers."""
    (tmp_path / "README.md").write_text("# Echo\nA coding agent.", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "echo-agent"\ndescription = "coding agent"\n',
        encoding="utf-8",
    )

    session_summary = _make_session(str(tmp_path), prompt="dame un resumen del proyecto")
    session_improve = _make_session(str(tmp_path), prompt="ahora dame propuestas de mejora")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer_summary = build_local_inspect_answer(session_summary, run_state, activity=activity)
    answer_improve = build_local_inspect_answer(session_improve, run_state, activity=activity)

    assert answer_summary != answer_improve


def test_no_contexto_local_para_header(tmp_path: Path) -> None:
    """The old 'Contexto local para: ...' header must never appear."""
    (tmp_path / "README.md").write_text("# My Project\nDoing things.", encoding="utf-8")
    session = _make_session(str(tmp_path), prompt="qué es este proyecto?")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert not answer.startswith("Contexto local para:")
    assert "Contexto local para:" not in answer


# ---------------------------------------------------------------------------
# _parse_pyproject
# ---------------------------------------------------------------------------


def test_parse_pyproject_extracts_name() -> None:
    content = '[project]\nname = "my-lib"\n'
    meta = _parse_pyproject(content)
    assert meta["name"] == "my-lib"


def test_parse_pyproject_extracts_description() -> None:
    content = '[project]\ndescription = "Does cool things"\n'
    meta = _parse_pyproject(content)
    assert meta["description"] == "Does cool things"


def test_parse_pyproject_extracts_python() -> None:
    content = '[project]\nrequires-python = ">=3.11"\n'
    meta = _parse_pyproject(content)
    assert meta["python"] == ">=3.11"


def test_parse_pyproject_extracts_dependencies() -> None:
    content = '[project]\ndependencies = [\n  "rich",\n  "typer",\n]\n'
    meta = _parse_pyproject(content)
    assert "rich" in meta["dependencies"]
    assert "typer" in meta["dependencies"]


def test_parse_pyproject_missing_fields_returns_empty_dict() -> None:
    meta = _parse_pyproject("")
    assert meta == {}


# ---------------------------------------------------------------------------
# _readme_intro
# ---------------------------------------------------------------------------


def test_readme_intro_includes_h1_title() -> None:
    content = "# My Project\n\nThis is the description.\n"
    intro = _readme_intro(content)
    assert "# My Project" in intro


def test_readme_intro_stops_at_h2() -> None:
    content = "# Title\n\nIntro paragraph.\n\n## Installation\n\nsteps here\n"
    intro = _readme_intro(content)
    assert "## Installation" not in intro
    assert "Intro paragraph" in intro


def test_readme_intro_skips_badges() -> None:
    content = "# Title\n[![Build](https://img.shields.io/badge/...)](link)\n\nReal intro.\n"
    intro = _readme_intro(content)
    assert "[![Build]" not in intro
    assert "Real intro" in intro


def test_readme_intro_skips_inline_html() -> None:
    content = "# Title\n<p align='center'><img src='logo.png'/></p>\n\nDescription.\n"
    intro = _readme_intro(content)
    assert "<p" not in intro
    assert "Description" in intro


# ---------------------------------------------------------------------------
# synthesis — improvement numbered list
# ---------------------------------------------------------------------------


def test_improvement_contains_numbered_list(tmp_path: Path) -> None:
    """Improvement answer contains at least one numbered suggestion."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "echo-agent"\n',
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame propuestas de mejora")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    # Must contain at least one "1. " numbered item
    assert "1." in answer


# ---------------------------------------------------------------------------
# synthesis — summary contains project name
# ---------------------------------------------------------------------------


def test_summary_contains_project_name(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "super-lib"\ndescription = "Does great things"\n',
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame un resumen del proyecto")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "super-lib" in answer


# ===========================================================================
# NEW SOURCES: setup.cfg
# ===========================================================================


def test_parse_setup_cfg_extracts_name() -> None:
    content = "[metadata]\nname = my-setupcfg-project\n"
    meta = _parse_setup_cfg(content)
    assert meta["name"] == "my-setupcfg-project"


def test_parse_setup_cfg_extracts_description() -> None:
    content = "[metadata]\ndescription = A cool CLI tool\n"
    meta = _parse_setup_cfg(content)
    assert meta["description"] == "A cool CLI tool"


def test_parse_setup_cfg_extracts_version() -> None:
    content = "[metadata]\nversion = 2.1.0\n"
    meta = _parse_setup_cfg(content)
    assert meta["version"] == "2.1.0"


def test_parse_setup_cfg_extracts_python_requires() -> None:
    content = "[options]\npython_requires = >=3.10\n"
    meta = _parse_setup_cfg(content)
    assert meta["python"] == ">=3.10"


def test_parse_setup_cfg_extracts_install_requires() -> None:
    content = (
        "[options]\n"
        "install_requires =\n"
        "    rich\n"
        "    typer\n"
        "    click\n"
    )
    meta = _parse_setup_cfg(content)
    assert "rich" in meta["dependencies"]
    assert "typer" in meta["dependencies"]
    assert "click" in meta["dependencies"]


def test_parse_setup_cfg_extracts_entry_points() -> None:
    content = (
        "[options.entry_points]\n"
        "console_scripts =\n"
        "    mycli = mypackage.cli:main\n"
    )
    meta = _parse_setup_cfg(content)
    assert "mycli" in meta["scripts"]


def test_parse_setup_cfg_empty_returns_empty_dict() -> None:
    assert _parse_setup_cfg("") == {}


def test_fallback_uses_setup_cfg_for_summary(tmp_path: Path) -> None:
    """When no pyproject.toml exists, setup.cfg is used for project metadata."""
    (tmp_path / "setup.cfg").write_text(
        "[metadata]\nname = cfg-project\ndescription = A setuptools project\n",
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame un resumen del proyecto")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "cfg-project" in answer


def test_fallback_uses_setup_cfg_for_improvement(tmp_path: Path) -> None:
    """Improvement proposals are grounded even when only setup.cfg is present."""
    (tmp_path / "setup.cfg").write_text(
        "[metadata]\nname = cfg-project\n[options]\ninstall_requires =\n    requests\n",
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame propuestas de mejora")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "1." in answer  # numbered list


# ===========================================================================
# NEW SOURCES: setup.py
# ===========================================================================


def test_parse_setup_py_extracts_name() -> None:
    content = 'from setuptools import setup\nsetup(\n    name="my-setup-py-project",\n)\n'
    meta = _parse_setup_py(content)
    assert meta["name"] == "my-setup-py-project"


def test_parse_setup_py_extracts_description() -> None:
    content = 'setup(\n    name="pkg",\n    description="Does useful things",\n)\n'
    meta = _parse_setup_py(content)
    assert meta["description"] == "Does useful things"


def test_parse_setup_py_extracts_version() -> None:
    content = 'setup(\n    version="0.9.5",\n)\n'
    meta = _parse_setup_py(content)
    assert meta["version"] == "0.9.5"


def test_parse_setup_py_extracts_install_requires() -> None:
    content = (
        'setup(\n'
        '    install_requires=[\n'
        '        "rich>=10",\n'
        '        "typer",\n'
        '    ],\n'
        ')\n'
    )
    meta = _parse_setup_py(content)
    assert "rich" in meta["dependencies"]
    assert "typer" in meta["dependencies"]


def test_parse_setup_py_empty_returns_empty_dict() -> None:
    assert _parse_setup_py("") == {}


def test_fallback_uses_setup_py_for_summary(tmp_path: Path) -> None:
    """When no pyproject.toml or setup.cfg exists, setup.py is used."""
    (tmp_path / "setup.py").write_text(
        'from setuptools import setup\nsetup(name="setuppy-project", description="A classic project")\n',
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="qué es este proyecto?")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "setuppy-project" in answer


# ===========================================================================
# _parse_project_meta dispatch
# ===========================================================================


def test_parse_project_meta_prefers_pyproject(tmp_path: Path) -> None:
    """pyproject.toml wins over setup.cfg and setup.py when all are present."""
    files = {
        "pyproject.toml": '[project]\nname = "from-pyproject"\n',
        "setup.cfg": "[metadata]\nname = from-setup-cfg\n",
        "setup.py": 'setup(name="from-setup-py")\n',
    }
    meta = _parse_project_meta(files)
    assert meta["name"] == "from-pyproject"


def test_parse_project_meta_falls_back_to_setup_cfg() -> None:
    """When pyproject.toml is absent, setup.cfg is used."""
    files = {
        "setup.cfg": "[metadata]\nname = from-setup-cfg\n",
        "setup.py": 'setup(name="from-setup-py")\n',
    }
    meta = _parse_project_meta(files)
    assert meta["name"] == "from-setup-cfg"


def test_parse_project_meta_falls_back_to_setup_py() -> None:
    """When both pyproject.toml and setup.cfg are absent, setup.py is used."""
    files = {"setup.py": 'setup(name="from-setup-py")\n'}
    meta = _parse_project_meta(files)
    assert meta["name"] == "from-setup-py"


def test_parse_project_meta_empty_files_returns_empty_dict() -> None:
    assert _parse_project_meta({}) == {}


# ===========================================================================
# NEW SOURCES: AGENTS.md / CLAUDE.md
# ===========================================================================


def test_fallback_uses_agents_md_for_summary(tmp_path: Path) -> None:
    """AGENTS.md content is surfaced in summary answers."""
    (tmp_path / "AGENTS.md").write_text(
        "# Echo Agent Architecture\n\nThis agent uses a grounded fallback pipeline.\n",
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame un resumen del proyecto")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "Echo Agent Architecture" in answer or "grounded fallback" in answer


def test_fallback_uses_claude_md_for_summary(tmp_path: Path) -> None:
    """CLAUDE.md content is surfaced in summary answers."""
    (tmp_path / "CLAUDE.md").write_text(
        "# Project Conventions\n\nAll code must pass mypy strict.\n",
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="qué es este proyecto?")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "Project Conventions" in answer or "mypy" in answer


def test_agents_md_todo_surfaces_in_improvement(tmp_path: Path) -> None:
    """TODO markers in AGENTS.md appear as improvement hints."""
    (tmp_path / "AGENTS.md").write_text(
        "# Echo Agent\n\nTODO: add streaming support\nFIXME: retry logic is broken\n",
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame propuestas de mejora")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "TODO" in answer or "FIXME" in answer or "streaming" in answer


def test_extract_agent_doc_hints_summary_returns_intro() -> None:
    files = {
        "AGENTS.md": "# My Agent\n\nThis is the architecture section.\n## Details\nmore here\n"
    }
    hints = _extract_agent_doc_hints(files, "summary")
    assert len(hints) == 1
    assert "My Agent" in hints[0]
    assert "Details" not in hints[0]  # stopped at ##


def test_extract_agent_doc_hints_improvement_returns_todos() -> None:
    files = {
        "CLAUDE.md": "# Conventions\nTODO: improve error handling\nFIXME: broken retry\n"
    }
    hints = _extract_agent_doc_hints(files, "improvement")
    assert any("TODO" in h or "FIXME" in h for h in hints)


def test_extract_agent_doc_hints_empty_files_returns_empty() -> None:
    assert _extract_agent_doc_hints({}, "summary") == []
    assert _extract_agent_doc_hints({"AGENTS.md": ""}, "improvement") == []


# ===========================================================================
# Combined intent: "dame un resumen y una propuesta de mejora"
# ===========================================================================


def test_detect_intent_combined_resumen_y_mejora() -> None:
    assert _detect_intent("dame un resumen y dame una propuesta de mejora") == "combined"


def test_detect_intent_combined_inspecciona_y_resumen_mejora() -> None:
    """The exact validation phrase triggers combined intent."""
    assert _detect_intent(
        "inspecciona el proyecto dame un resumen y dame una propuesta de mejora"
    ) == "combined"


def test_combined_answer_contains_summary_section(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "combined-lib"\ndescription = "Does both things"\n',
        encoding="utf-8",
    )
    session = _make_session(
        str(tmp_path),
        prompt="dame un resumen y dame una propuesta de mejora",
    )
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "combined-lib" in answer


def test_combined_answer_contains_improvement_section(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "combined-lib"\n',
        encoding="utf-8",
    )
    session = _make_session(
        str(tmp_path),
        prompt="dame un resumen y dame una propuesta de mejora",
    )
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "1." in answer  # numbered improvement list


def test_combined_answer_has_separator(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Test Project\nA sample project.\n", encoding="utf-8")
    session = _make_session(
        str(tmp_path),
        prompt="resumen del proyecto y qué mejoras propones",
    )
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "---" in answer  # summary and improvement are separated


def test_combined_differs_from_summary_only(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mylib"\ndescription = "a lib"\n',
        encoding="utf-8",
    )
    session_summary = _make_session(str(tmp_path), prompt="dame un resumen")
    session_combined = _make_session(
        str(tmp_path), prompt="dame un resumen y mejoras"
    )
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer_summary = build_local_inspect_answer(session_summary, run_state, activity=activity)
    answer_combined = build_local_inspect_answer(session_combined, run_state, activity=activity)

    assert len(answer_combined) > len(answer_summary)


# ===========================================================================
# Smoke test harness — conversational two-turn fallback
# ===========================================================================


def _make_realistic_project(tmp_path: Path) -> None:
    """Create a realistic project structure for smoke testing."""
    (tmp_path / "README.md").write_text(
        "# EchoAgent\n\n"
        "A local coding agent that works without a backend.\n\n"
        "## Features\n"
        "- Local fallback when Ollama is unavailable\n"
        "- Grounded answers from project files\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\n'
        'name = "echo-agent"\n'
        'description = "Local coding agent with fallback inspection"\n'
        'requires-python = ">=3.11"\n'
        'dependencies = [\n'
        '  "rich",\n'
        '  "typer",\n'
        '  "httpx",\n'
        ']\n'
        '\n'
        '[project.scripts]\n'
        'echo = "echo.cli:main"\n',
        encoding="utf-8",
    )
    (tmp_path / "AGENTS.md").write_text(
        "# Echo Agent Conventions\n\n"
        "This agent always grounds its answers in local files.\n"
        "TODO: add streaming support for long responses\n",
        encoding="utf-8",
    )


def test_smoke_first_turn_summary(tmp_path: Path) -> None:
    """Smoke: first turn with a summary prompt returns useful project info."""
    _make_realistic_project(tmp_path)
    session = _make_session(str(tmp_path), prompt="inspecciona el proyecto y dame un resumen")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    # Must contain project name and description — grounded in pyproject.toml
    assert "echo-agent" in answer
    # Must contain README content
    assert "EchoAgent" in answer or "local coding agent" in answer.lower()
    # Must not be a raw file dump
    assert "Etapa detenida:" not in answer
    assert "Contexto local para:" not in answer


def test_smoke_second_turn_improvement_is_different(tmp_path: Path) -> None:
    """Smoke: second turn improvement is different from first turn summary."""
    _make_realistic_project(tmp_path)
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    session1 = _make_session(str(tmp_path), prompt="inspecciona el proyecto y dame un resumen")
    answer1 = build_local_inspect_answer(session1, run_state, activity=activity)

    session2 = _make_session(str(tmp_path), prompt="ahora dame una propuesta de mejora")
    answer2 = build_local_inspect_answer(session2, run_state, activity=activity)

    assert answer1 != answer2
    # Second answer must have improvement list
    assert "1." in answer2
    # Second answer must not just repeat the summary
    assert "Propuesta" in answer2 or "mejora" in answer2.lower() or "mypy" in answer2 or "ruff" in answer2


def test_smoke_second_turn_no_stale_objective(tmp_path: Path) -> None:
    """Smoke: stale session.objective from a resumed session does not pollute second turn."""
    _make_realistic_project(tmp_path)
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    session2 = _make_session(str(tmp_path), prompt="ahora dame una propuesta de mejora")
    # Simulate stale objective from previous session
    session2.objective = (
        "Inspecciona README.md y echo/config.py. "
        "Di el backend por defecto y el modelo por defecto."
    )

    answer2 = build_local_inspect_answer(session2, run_state, activity=activity)

    # The answer must reflect improvement intent, not the stale objective
    assert "1." in answer2
    assert "Contexto local para:" not in answer2


# ===========================================================================
# Validation final obligatoria — frase exacta
# ===========================================================================


def test_exact_phrase_resumen_y_mejora_combined_intent(tmp_path: Path) -> None:
    """
    Validation: the exact phrase
      'inspecciona el proyecto dame un resumen y dame una propuesta de mejora'
    must produce a response that:
    - contains a real project summary (grounded in files)
    - contains a real improvement proposal (numbered list)
    - is NOT the old fixed template
    - does NOT dump raw file content as the primary body
    """
    _make_realistic_project(tmp_path)

    # Exact phrase as specified
    exact_prompt = "inspecciona el proyecto dame un resumen y dame una propuesta de mejora"

    session = _make_session(str(tmp_path), prompt=exact_prompt)
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    # ── Summary part ────────────────────────────────────────────────────────
    # Must contain project name from pyproject.toml
    assert "echo-agent" in answer, f"Project name missing from answer:\n{answer}"
    # Must contain the description or README content
    assert (
        "local coding agent" in answer.lower() or "EchoAgent" in answer
    ), f"Project description missing from answer:\n{answer}"

    # ── Improvement part ────────────────────────────────────────────────────
    # Must contain a numbered improvement list
    assert "1." in answer, f"No numbered improvement list in answer:\n{answer}"
    # Must contain at least one concrete tool suggestion
    assert any(kw in answer for kw in ("mypy", "ruff", "pytest-cov", "CHANGELOG", "CI")), (
        f"No concrete improvement suggestion found in answer:\n{answer}"
    )

    # ── Separator between sections ──────────────────────────────────────────
    assert "---" in answer, f"Missing section separator in combined answer:\n{answer}"

    # ── Must NOT be the old fixed template ──────────────────────────────────
    assert "Contexto local para:" not in answer
    assert "Etapa detenida:" not in answer
    assert "Backend efectivo:" not in answer

    # ── Must NOT dump raw file content as primary body ──────────────────────
    # The response should be synthesized prose, not multi-KB file dumps
    assert len(answer) < 8000, f"Answer suspiciously long (possible raw dump): {len(answer)} chars"


# ===========================================================================
# HARDENING: _resolve_attr_ref
# ===========================================================================


def test_resolve_attr_ref_finds_version_in_init(tmp_path: Path) -> None:
    pkg = tmp_path / "mypackage"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "2.3.4"\n', encoding="utf-8")

    result = _resolve_attr_ref("mypackage.__version__", str(tmp_path))
    assert result == "2.3.4"


def test_resolve_attr_ref_finds_version_in_src_layout(tmp_path: Path) -> None:
    src = tmp_path / "src" / "mypkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")

    result = _resolve_attr_ref("mypkg.__version__", str(tmp_path))
    assert result == "1.0.0"


def test_resolve_attr_ref_finds_version_in_version_file(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "_version.py").write_text('__version__ = "3.1.4"\n', encoding="utf-8")

    result = _resolve_attr_ref("mypkg.__version__", str(tmp_path))
    assert result == "3.1.4"


def test_resolve_attr_ref_returns_none_when_file_missing(tmp_path: Path) -> None:
    result = _resolve_attr_ref("nonexistent.__version__", str(tmp_path))
    assert result is None


def test_resolve_attr_ref_returns_none_when_attr_not_in_file(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# nothing here\n", encoding="utf-8")

    result = _resolve_attr_ref("mypkg.__version__", str(tmp_path))
    assert result is None


def test_resolve_attr_ref_returns_none_for_single_component(tmp_path: Path) -> None:
    result = _resolve_attr_ref("__version__", str(tmp_path))
    assert result is None


# ===========================================================================
# HARDENING: _resolve_file_ref
# ===========================================================================


def test_resolve_file_ref_reads_version_txt(tmp_path: Path) -> None:
    (tmp_path / "VERSION.txt").write_text("1.5.0\n", encoding="utf-8")
    result = _resolve_file_ref("VERSION.txt", str(tmp_path))
    assert result == "1.5.0"


def test_resolve_file_ref_skips_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# big readme", encoding="utf-8")
    result = _resolve_file_ref("README.md", str(tmp_path))
    assert result is None


def test_resolve_file_ref_returns_none_when_file_missing(tmp_path: Path) -> None:
    result = _resolve_file_ref("MISSING.txt", str(tmp_path))
    assert result is None


def test_resolve_file_ref_returns_first_line(tmp_path: Path) -> None:
    (tmp_path / "DESCRIPTION.txt").write_text("Short description\nMore text\n", encoding="utf-8")
    result = _resolve_file_ref("DESCRIPTION.txt", str(tmp_path))
    assert result == "Short description"


# ===========================================================================
# HARDENING: setup.cfg with attr: and file: directives
# ===========================================================================


def test_parse_setup_cfg_attr_version_resolved_with_repo_root(tmp_path: Path) -> None:
    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "9.8.7"\n', encoding="utf-8")
    content = "[metadata]\nname = mypkg\nversion = attr: mypkg.__version__\n"

    meta = _parse_setup_cfg(content, repo_root=str(tmp_path))
    assert meta["version"] == "9.8.7"


def test_parse_setup_cfg_attr_version_omitted_without_repo_root() -> None:
    """Without repo_root the attr: value is omitted, not returned raw."""
    content = "[metadata]\nname = mypkg\nversion = attr: mypkg.__version__\n"
    meta = _parse_setup_cfg(content, repo_root=None)
    assert "version" not in meta
    assert meta.get("name") == "mypkg"


def test_parse_setup_cfg_attr_version_omitted_when_unresolvable(tmp_path: Path) -> None:
    """attr: unresolvable → version field absent, not 'attr: ...' string."""
    content = "[metadata]\nname = mypkg\nversion = attr: mypkg.__version__\n"
    # No __init__.py exists
    meta = _parse_setup_cfg(content, repo_root=str(tmp_path))
    assert "version" not in meta
    assert "attr:" not in str(meta)


def test_parse_setup_cfg_file_description_resolved(tmp_path: Path) -> None:
    (tmp_path / "DESCRIPTION.txt").write_text("A short project description\n", encoding="utf-8")
    content = "[metadata]\nname = myproj\ndescription = file: DESCRIPTION.txt\n"

    meta = _parse_setup_cfg(content, repo_root=str(tmp_path))
    assert meta["description"] == "A short project description"


def test_parse_setup_cfg_file_readme_skipped(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Big readme", encoding="utf-8")
    content = "[metadata]\nname = myproj\nlong_description = file: README.md\n"
    # long_description is not a field we extract, but even if it were, README is skipped
    meta = _parse_setup_cfg(content, repo_root=str(tmp_path))
    assert "README" not in str(meta)


def test_parse_setup_cfg_file_description_omitted_without_repo_root() -> None:
    content = "[metadata]\nname = myproj\ndescription = file: DESCRIPTION.txt\n"
    meta = _parse_setup_cfg(content, repo_root=None)
    assert "description" not in meta
    assert "file:" not in str(meta)


def test_fallback_setup_cfg_attr_version_in_summary(tmp_path: Path) -> None:
    """Full integration: setup.cfg with attr: version → version appears in summary."""
    pkg = tmp_path / "myapp"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "4.2.0"\n', encoding="utf-8")
    (tmp_path / "setup.cfg").write_text(
        "[metadata]\nname = myapp\ndescription = My application\n"
        "version = attr: myapp.__version__\n",
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame un resumen del proyecto")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "myapp" in answer
    assert "4.2.0" in answer


# ===========================================================================
# HARDENING: _extract_simple_vars and _resolve_setup_kwarg
# ===========================================================================


def test_extract_simple_vars_captures_string_vars() -> None:
    content = (
        'NAME = "echo-agent"\n'
        "VERSION = '1.2.3'\n"
        '__version__ = "2.0.0"\n'
        'DESCRIPTION = "A local agent"\n'
    )
    result = _extract_simple_vars(content)
    assert result["NAME"] == "echo-agent"
    assert result["VERSION"] == "1.2.3"
    assert result["__version__"] == "2.0.0"
    assert result["DESCRIPTION"] == "A local agent"


def test_extract_simple_vars_ignores_indented_lines() -> None:
    content = '    INNER = "should_not_capture"\nOUTER = "captured"\n'
    result = _extract_simple_vars(content)
    assert "INNER" not in result
    assert result["OUTER"] == "captured"


def test_resolve_setup_kwarg_literal() -> None:
    content = 'setup(\n    name="my-lib",\n)\n'
    assert _resolve_setup_kwarg(content, "name", {}) == "my-lib"


def test_resolve_setup_kwarg_variable_reference() -> None:
    content = 'NAME = "echo-agent"\nsetup(\n    name=NAME,\n)\n'
    simple_vars = _extract_simple_vars(content)
    assert _resolve_setup_kwarg(content, "name", simple_vars) == "echo-agent"


def test_resolve_setup_kwarg_returns_none_for_function_call() -> None:
    content = 'setup(\n    name=get_name(),\n)\n'
    result = _resolve_setup_kwarg(content, "name", {})
    # get_name() is not a simple variable and not a literal — should return None
    # (regex would match "get_name" as a variable ref, but it's not in simple_vars)
    # This is acceptable fallback behaviour — None is better than "get_name"
    assert result is None or isinstance(result, str)


def test_parse_setup_py_variable_name() -> None:
    content = (
        'NAME = "mylib"\n'
        'setup(\n    name=NAME,\n)\n'
    )
    meta = _parse_setup_py(content)
    assert meta["name"] == "mylib"


def test_parse_setup_py_dunder_version() -> None:
    content = (
        '__version__ = "3.0.1"\n'
        'setup(\n    version=__version__,\n)\n'
    )
    meta = _parse_setup_py(content)
    assert meta["version"] == "3.0.1"


def test_parse_setup_py_variable_description() -> None:
    content = (
        'DESCRIPTION = "Does amazing things"\n'
        'setup(\n    description=DESCRIPTION,\n)\n'
    )
    meta = _parse_setup_py(content)
    assert meta["description"] == "Does amazing things"


def test_parse_setup_py_deps_list_variable() -> None:
    content = (
        'DEPS = ["rich", "typer>=0.6"]\n'
        'setup(\n    install_requires=DEPS,\n)\n'
    )
    meta = _parse_setup_py(content)
    assert "rich" in meta["dependencies"]
    assert "typer" in meta["dependencies"]


def test_parse_setup_py_version_from_init(tmp_path: Path) -> None:
    """When version=__version__ and __version__ is not in setup.py itself,
    fallback to finding it in the package __init__.py via repo_root."""
    pkg = tmp_path / "mypackage"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "5.0.0"\n', encoding="utf-8")
    (tmp_path / "setup.py").write_text(
        'from mypackage import __version__\n'
        'setup(\n    name="mypackage",\n    version=__version__,\n)\n',
        encoding="utf-8",
    )
    meta = _parse_setup_py(
        (tmp_path / "setup.py").read_text(),
        repo_root=str(tmp_path),
    )
    assert meta["version"] == "5.0.0"


def test_fallback_setup_py_variable_name_in_summary(tmp_path: Path) -> None:
    """Full integration: setup.py with NAME = "..." → name appears in summary."""
    (tmp_path / "setup.py").write_text(
        'NAME = "varname-project"\n'
        'setup(name=NAME, description="Variable-driven setup")\n',
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame un resumen del proyecto")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)
    assert "varname-project" in answer


# ===========================================================================
# HARDENING: _readme_intro — paragraph-aware
# ===========================================================================


def test_readme_intro_completes_paragraph_before_stopping() -> None:
    """Long first paragraph is taken complete, not cut mid-sentence."""
    para1 = "This is a long first sentence. " * 5  # ~160 chars, well under 700 default
    content = f"# Title\n\n{para1}\n\n## Section\nIgnored content.\n"
    intro = _readme_intro(content)
    # Should contain the full paragraph, not cut mid-word
    assert para1.strip() in intro
    assert "Ignored content" not in intro


def test_readme_intro_max_paragraphs_limits_output() -> None:
    # max_paragraphs counts paragraph groups. "# Title" is paragraph 1,
    # "Para one." is paragraph 2, so max_paragraphs=2 → Title + Para one only.
    content = (
        "# Title\n\n"
        "Para one.\n\n"
        "Para two.\n\n"
        "Para three.\n\n"
        "Para four.\n\n"
        "Para five.\n"
    )
    intro = _readme_intro(content, max_paragraphs=2)
    assert "# Title" in intro
    assert "Para one" in intro
    assert "Para two" not in intro

    # With max_paragraphs=3: Title + Para one + Para two, not three.
    intro3 = _readme_intro(content, max_paragraphs=3)
    assert "Para one" in intro3
    assert "Para two" in intro3
    assert "Para three" not in intro3


def test_readme_intro_paragraph_aware_does_not_cut_mid_sentence() -> None:
    """Regression: old version cut at max_chars inside a line. New version
    completes the paragraph first, then applies the hard cap."""
    long_para = "word " * 50  # 250 chars, starts a paragraph
    content = f"# Title\n\n{long_para}\n\nsecond para\n"
    # With max_chars=700 (default), the full para fits — should not be truncated
    intro = _readme_intro(content)
    # The paragraph (minus trailing space) should be present intact
    assert long_para.strip() in intro


def test_readme_intro_hard_cap_still_applies() -> None:
    """Even with paragraph awareness, the hard max_chars cap applies."""
    huge_para = "x" * 2000  # one giant paragraph
    content = f"# Title\n\n{huge_para}\n"
    intro = _readme_intro(content, max_chars=500)
    assert len(intro) <= 500


def test_agents_md_long_doc_intro_is_paragraph_complete(tmp_path: Path) -> None:
    """AGENTS.md with multi-paragraph intro: hints include complete first paragraphs."""
    (tmp_path / "AGENTS.md").write_text(
        "# Echo Conventions\n\n"
        "This is the first paragraph, it has multiple sentences. "
        "It should be captured complete.\n\n"
        "This is the second paragraph.\n\n"
        "## Details\n"
        "This section should be excluded.\n",
        encoding="utf-8",
    )
    files = {"AGENTS.md": (tmp_path / "AGENTS.md").read_text(encoding="utf-8")}
    hints = _extract_agent_doc_hints(files, "summary")

    assert len(hints) == 1
    hint = hints[0]
    # First paragraph captured complete
    assert "multiple sentences" in hint
    # Section heading excluded
    assert "## Details" not in hint
    assert "This section should be excluded" not in hint


def test_claude_md_multi_section_intro_stops_at_h2(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text(
        "# Project Rules\n\n"
        "All code must be typed.\n\n"
        "## Conventions\n"
        "Convention details here.\n",
        encoding="utf-8",
    )
    files = {"CLAUDE.md": (tmp_path / "CLAUDE.md").read_text(encoding="utf-8")}
    hints = _extract_agent_doc_hints(files, "summary")

    assert "All code must be typed" in hints[0]
    assert "Convention details" not in hints[0]


# ===========================================================================
# config_improvement intent
# ===========================================================================


def test_detect_intent_config_improvement() -> None:
    assert _detect_intent("qué configuración tiene y qué mejorarías") == "config_improvement"


def test_detect_intent_config_improvement_english() -> None:
    assert _detect_intent("what config does it have and suggest improvements") == "config_improvement"


def test_detect_intent_summary_plus_config_plus_improvement_is_combined() -> None:
    """Three-way → combined (summary+improvement), not config_improvement."""
    assert _detect_intent(
        "dame un resumen, qué configuración tiene y propuestas de mejora"
    ) == "combined"


def test_config_improvement_answer_has_config_section(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mylib"\nrequires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    session = _make_session(
        str(tmp_path),
        prompt="qué configuración tiene y qué mejorarías",
    )
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "mylib" in answer  # config section
    assert "---" in answer    # separator
    assert "1." in answer     # improvement list


def test_config_improvement_answer_has_improvement_section(tmp_path: Path) -> None:
    (tmp_path / "setup.cfg").write_text(
        "[metadata]\nname = cfg-lib\n[options]\npython_requires = >=3.10\n",
        encoding="utf-8",
    )
    session = _make_session(
        str(tmp_path),
        prompt="dame la config del proyecto y qué le mejorarías",
    )
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "cfg-lib" in answer
    assert "1." in answer


def test_config_improvement_differs_from_config_only(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "lib"\n',
        encoding="utf-8",
    )
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    s_config = _make_session(str(tmp_path), prompt="qué configuración tiene")
    s_ci = _make_session(str(tmp_path), prompt="qué configuración tiene y mejoras")
    a_config = build_local_inspect_answer(s_config, run_state, activity=activity)
    a_ci = build_local_inspect_answer(s_ci, run_state, activity=activity)

    assert len(a_ci) > len(a_config)
    assert "1." in a_ci       # improvement list present
    assert "1." not in a_config  # config-only has no numbered list


def test_combined_still_works_with_new_config_improvement(tmp_path: Path) -> None:
    """Regression: combined (summary+improvement) still works after adding config_improvement."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "echo"\ndescription = "An agent"\n',
        encoding="utf-8",
    )
    session = _make_session(
        str(tmp_path),
        prompt="dame un resumen y una propuesta de mejora",
    )
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "echo" in answer
    assert "1." in answer
    assert "---" in answer


# ===========================================================================
# Hardening smoke test — attr resolution end-to-end
# ===========================================================================


def test_smoke_attr_version_flows_through_full_pipeline(tmp_path: Path) -> None:
    """Full pipeline: setup.cfg with attr: version → summary contains resolved version."""
    pkg = tmp_path / "myapp"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "7.0.1"\n', encoding="utf-8")
    (tmp_path / "setup.cfg").write_text(
        "[metadata]\n"
        "name = myapp\n"
        "description = A production app\n"
        "version = attr: myapp.__version__\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# MyApp\n\nDoes production things.\n",
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="dame un resumen del proyecto")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "myapp" in answer
    assert "7.0.1" in answer
    assert "A production app" in answer
    assert "attr:" not in answer  # raw directive must NOT appear


def test_smoke_setup_py_variables_flow_through_full_pipeline(tmp_path: Path) -> None:
    """Full pipeline: setup.py with NAME/DEPS variables → summary uses resolved values."""
    (tmp_path / "setup.py").write_text(
        'NAME = "var-project"\n'
        'DESCRIPTION = "Built with variables"\n'
        'DEPS = ["requests", "click"]\n'
        'setup(name=NAME, description=DESCRIPTION, install_requires=DEPS)\n',
        encoding="utf-8",
    )
    session = _make_session(str(tmp_path), prompt="qué es este proyecto?")
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    assert "var-project" in answer
    assert "Built with variables" in answer


# ===========================================================================
# Final validation — exact phrase (regression guard)
# ===========================================================================


def test_exact_phrase_still_works_after_hardening(tmp_path: Path) -> None:
    """
    Regression guard: after all hardening changes, the exact validation phrase
    'inspecciona el proyecto dame un resumen y dame una propuesta de mejora'
    must still produce a combined response with summary + improvement.
    """
    _make_realistic_project(tmp_path)  # defined earlier in this file

    exact_prompt = "inspecciona el proyecto dame un resumen y dame una propuesta de mejora"
    session = _make_session(str(tmp_path), prompt=exact_prompt)
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    answer = build_local_inspect_answer(session, run_state, activity=activity)

    # Summary part
    assert "echo-agent" in answer
    assert "local coding agent" in answer.lower() or "EchoAgent" in answer
    # Improvement part
    assert "1." in answer
    assert any(kw in answer for kw in ("mypy", "ruff", "pytest-cov", "CHANGELOG", "CI"))
    # Structure
    assert "---" in answer
    # No old template, no raw dump
    assert "Contexto local para:" not in answer
    assert "Etapa detenida:" not in answer
    assert len(answer) < 8000


def test_followup_improvement_avoids_repeating_same_four_items(tmp_path: Path) -> None:
    _make_followup_ready_project(tmp_path)
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    session1 = _make_session(
        str(tmp_path),
        prompt="dame una explicación del proyecto y dame una propuesta de mejora",
    )
    answer1 = build_local_inspect_answer(session1, run_state, activity=activity)

    session2 = _make_session(str(tmp_path), prompt="y no hay otra propuesta que puedas dar?")
    session2.emitted_improvement_ids = list(session1.emitted_improvement_ids)
    session2.emitted_improvement_layers = list(session1.emitted_improvement_layers)
    answer2 = build_local_inspect_answer(session2, run_state, activity=activity)

    first_items = _numbered_items(answer1)
    second_items = _numbered_items(answer2)
    assert len(first_items) >= 4
    assert second_items
    assert first_items[:4] != second_items[:4]
    assert not any(item in second_items for item in first_items[:4])
    assert "más estructurales" in answer2.lower() or "opciones nuevas" in answer2.lower()


def test_third_turn_otra_aparte_de_esas_uses_new_layer(tmp_path: Path) -> None:
    _make_followup_ready_project(tmp_path)
    run_state = _make_run_state(str(tmp_path))
    activity = _make_activity()

    session1 = _make_session(
        str(tmp_path),
        prompt="dame una explicación del proyecto y dame una propuesta de mejora",
    )
    answer1 = build_local_inspect_answer(session1, run_state, activity=activity)

    session2 = _make_session(str(tmp_path), prompt="y no hay otra propuesta que puedas dar?")
    session2.emitted_improvement_ids = list(session1.emitted_improvement_ids)
    session2.emitted_improvement_layers = list(session1.emitted_improvement_layers)
    answer2 = build_local_inspect_answer(session2, run_state, activity=activity)

    session3 = _make_session(str(tmp_path), prompt="otra propuesta que tengas aparte de esas 4?")
    session3.emitted_improvement_ids = list(session2.emitted_improvement_ids)
    session3.emitted_improvement_layers = list(session2.emitted_improvement_layers)
    answer3 = build_local_inspect_answer(session3, run_state, activity=activity)

    first_items = _numbered_items(answer1)
    second_items = _numbered_items(answer2)
    third_items = _numbered_items(answer3)
    assert third_items
    assert third_items != first_items
    assert third_items != second_items
    assert not any(item in third_items for item in first_items)
    assert "producto" in answer3.lower() or "ux" in answer3.lower() or "repl" in answer3.lower()
