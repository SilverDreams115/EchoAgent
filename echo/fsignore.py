from __future__ import annotations

from fnmatch import fnmatch
import os
from pathlib import Path
from typing import Iterator


IGNORED_NAMES = {
    ".git",
    ".echo",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".cache",
}


def is_ignored_name(name: str) -> bool:
    return name in IGNORED_NAMES


def is_ignored_path(path: Path, project_root: Path | None = None) -> bool:
    parts = path.parts
    if project_root is not None:
        try:
            parts = path.relative_to(project_root).parts
        except ValueError:
            parts = path.parts
    return any(part in IGNORED_NAMES for part in parts)


def iter_project_files(project_root: Path, pattern: str = "*") -> Iterator[Path]:
    for current_root, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [name for name in dirnames if not is_ignored_name(name)]
        root_path = Path(current_root)
        for filename in filenames:
            if is_ignored_name(filename):
                continue
            if fnmatch(filename, pattern):
                yield root_path / filename
