"""
Tests for resolve_project_root() and _looks_like_project_root().

Covers:
- Execution from the repo root itself.
- Execution from a subdirectory inside the repo.
- Execution from an ambiguous parent that holds multiple sibling repos.
- No mixing of sibling trees (e.g. head-check/ and out/).
- Safe fallback when no valid root is found.
- Explicit --project-dir always wins.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from echo.cli.app import _looks_like_project_root, resolve_project_root


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(base: Path, name: str = "proj") -> Path:
    """Create a minimal EchoAgent-like project layout under base/name."""
    root = base / name
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (root / "echo").mkdir()
    (root / "echo" / "__init__.py").write_text("")
    return root


def _make_git_repo(path: Path) -> None:
    """Initialise a bare-minimum git repo so git rev-parse works."""
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)


# ---------------------------------------------------------------------------
# _looks_like_project_root
# ---------------------------------------------------------------------------


def test_looks_like_project_root_true(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    assert _looks_like_project_root(root) is True


def test_looks_like_project_root_false_no_pyproject(tmp_path: Path) -> None:
    (tmp_path / "echo").mkdir()
    assert _looks_like_project_root(tmp_path) is False


def test_looks_like_project_root_false_no_echo_dir(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    assert _looks_like_project_root(tmp_path) is False


def test_looks_like_project_root_false_empty_dir(tmp_path: Path) -> None:
    assert _looks_like_project_root(tmp_path) is False


# ---------------------------------------------------------------------------
# resolve_project_root — git-based resolution
# ---------------------------------------------------------------------------


def test_resolve_from_repo_root_via_git(tmp_path: Path) -> None:
    """Running from the repo root returns the repo root via git."""
    root = _make_project(tmp_path)
    _make_git_repo(root)
    result = resolve_project_root(root)
    assert result == root


def test_resolve_from_subdirectory_via_git(tmp_path: Path) -> None:
    """Running from a subdirectory resolves to the git repo root."""
    root = _make_project(tmp_path)
    _make_git_repo(root)
    subdir = root / "echo" / "cli"
    subdir.mkdir(parents=True)
    result = resolve_project_root(subdir)
    assert result == root


def test_resolve_does_not_walk_downward_into_sibling_repos(tmp_path: Path) -> None:
    """
    From an ambiguous parent that contains two sibling repos, git lookup fails
    (the parent is not itself a repo), and the sentinel walk-up also fails
    (the parent has no pyproject.toml).  The result falls back to the parent cwd.
    The important invariant is that neither sibling repo is picked blindly.
    """
    sibling_a = _make_project(tmp_path, "repo-a")
    sibling_b = _make_project(tmp_path, "repo-b")
    _make_git_repo(sibling_a)
    _make_git_repo(sibling_b)
    # tmp_path itself is NOT a git repo and has no pyproject.toml
    result = resolve_project_root(tmp_path)
    # Must not silently pick either sibling
    assert result != sibling_a
    assert result != sibling_b
    # Falls back to cwd (tmp_path)
    assert result == tmp_path


def test_resolve_from_sibling_a_stays_in_sibling_a(tmp_path: Path) -> None:
    """Running from inside repo-a must not cross into repo-b."""
    sibling_a = _make_project(tmp_path, "repo-a")
    sibling_b = _make_project(tmp_path, "repo-b")  # noqa: F841
    _make_git_repo(sibling_a)
    result = resolve_project_root(sibling_a)
    assert result == sibling_a


# ---------------------------------------------------------------------------
# resolve_project_root — sentinel walk-up fallback (no git)
# ---------------------------------------------------------------------------


def test_resolve_sentinel_walkup_from_project_root(tmp_path: Path) -> None:
    """Sentinel walk-up finds pyproject.toml + echo/ when git is absent."""
    root = _make_project(tmp_path)
    # No git init — forces sentinel path
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = resolve_project_root(root)
    assert result == root


def test_resolve_sentinel_walkup_from_subdirectory(tmp_path: Path) -> None:
    """Sentinel walk-up climbs from a subdirectory to the project root."""
    root = _make_project(tmp_path)
    subdir = root / "echo" / "cli"
    subdir.mkdir(parents=True)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = resolve_project_root(subdir)
    assert result == root


def test_resolve_falls_back_to_cwd_when_no_sentinels(tmp_path: Path) -> None:
    """When there is no git and no sentinels, the cwd is returned unchanged."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = resolve_project_root(tmp_path)
    assert result == tmp_path


# ---------------------------------------------------------------------------
# resolve_project_root — git OSError / timeout resilience
# ---------------------------------------------------------------------------


def test_resolve_handles_git_oserror_gracefully(tmp_path: Path) -> None:
    """If git is not installed, fall through to sentinel walk-up."""
    root = _make_project(tmp_path)
    with patch("subprocess.run", side_effect=OSError("git not found")):
        result = resolve_project_root(root)
    assert result == root


def test_resolve_handles_git_timeout_gracefully(tmp_path: Path) -> None:
    """If git times out, fall through to sentinel walk-up."""
    import subprocess as _subprocess

    root = _make_project(tmp_path)
    with patch("subprocess.run", side_effect=_subprocess.TimeoutExpired("git", 5)):
        result = resolve_project_root(root)
    assert result == root


# ---------------------------------------------------------------------------
# build_agent — explicit --project-dir always wins
# ---------------------------------------------------------------------------


def test_build_agent_explicit_project_dir_bypasses_resolution(tmp_path: Path) -> None:
    """An explicit --project-dir is used directly without any resolution."""
    root = _make_project(tmp_path)
    from echo.cli.app import build_agent

    agent, resolved_root, _ = build_agent(project_dir=str(root))
    assert resolved_root == root
