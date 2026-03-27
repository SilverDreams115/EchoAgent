from __future__ import annotations

from pathlib import Path


IGNORED = {".git", ".venv", "node_modules", "__pycache__", ".echo"}


def build_repo_map(project_root: Path, max_depth: int = 2, max_entries: int = 200) -> list[str]:
    lines: list[str] = []
    count = 0

    def walk(current: Path, depth: int) -> None:
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return
        try:
            children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except Exception:
            return
        for child in children:
            if child.name in IGNORED:
                continue
            rel = child.relative_to(project_root)
            prefix = "  " * depth
            lines.append(f"{prefix}{rel}/" if child.is_dir() else f"{prefix}{rel}")
            count += 1
            if child.is_dir():
                walk(child, depth + 1)
            if count >= max_entries:
                break

    walk(project_root, 0)
    return lines
