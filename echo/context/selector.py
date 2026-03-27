from __future__ import annotations

from pathlib import Path
import re


TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
    ".html",
    ".sh",
}

IGNORED = {".git", ".venv", "node_modules", "__pycache__", ".echo"}
PATH_PATTERN = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
ROOT_FILE_PATTERN = re.compile(r"\b[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+\b")


def _normalize_token(token: str) -> str:
    return token.strip("`'\"()[]{}<>.,:;!?").lower()


def _tokens(text: str) -> list[str]:
    return [token for item in re.findall(r"[a-zA-Z0-9_./-]+", text or "") if len((token := _normalize_token(item))) > 2]


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def _extract_explicit_files(project_root: Path, prompt: str) -> list[str]:
    explicit: list[str] = []
    seen: set[str] = set()
    candidates = PATH_PATTERN.findall(prompt or "") + ROOT_FILE_PATTERN.findall(prompt or "")
    for match in candidates:
        candidate = match.strip("`'\"()[]{}<>.,:;!?")
        path = project_root / candidate
        if not path.is_file() or not _is_text_file(path):
            continue
        rel = str(path.relative_to(project_root))
        if rel not in seen:
            seen.add(rel)
            explicit.append(rel)
    return explicit


def select_relevant_files(project_root: Path, prompt: str, limit: int = 6) -> list[str]:
    explicit = _extract_explicit_files(project_root, prompt)
    if explicit:
        return explicit[:limit]
    tokens = set(_tokens(prompt))
    scored: list[tuple[int, str]] = []
    for path in project_root.rglob("*"):
        if any(part in IGNORED for part in path.parts):
            continue
        if not path.is_file() or not _is_text_file(path):
            continue
        rel = str(path.relative_to(project_root))
        if rel in explicit:
            continue
        name_score = sum(6 for token in tokens if token in rel.lower())
        content_score = 0
        try:
            sample = path.read_text(encoding="utf-8")[:5000].lower()
        except Exception:
            continue
        for token in tokens:
            if token in sample:
                content_score += 2
        score = name_score + content_score
        if score > 0:
            scored.append((score, rel))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [rel for _, rel in scored[:limit]]


def build_focus_snippets(project_root: Path, files: list[str], line_limit: int = 80) -> list[str]:
    snippets: list[str] = []
    for rel in files:
        path = project_root / rel
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        content = "\n".join(lines[:line_limit])
        snippets.append(f"FILE: {rel}\n{content}")
    return snippets
