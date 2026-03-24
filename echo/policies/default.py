from __future__ import annotations


def default_constraints(profile: str = "local") -> list[str]:
    base = [
        "Inspect the repository before changing code.",
        "Prefer minimal patches over full rewrites.",
        "Avoid destructive commands by default.",
        "Ground every answer in real files or command output.",
        "Run validation after relevant edits.",
    ]
    if profile == "balanced":
        return base + [
            "Prefer concrete implementation guidance over broad brainstorming.",
            "Keep scope controlled and list affected files when they matter.",
        ]
    if profile == "deep":
        return base + [
            "Bias toward deeper inspection before final conclusions.",
            "State risks, tradeoffs, and unresolved assumptions explicitly.",
            "Use tools when available before asserting architectural claims.",
        ]
    return base + [
        "Keep answers concise and practical for local execution.",
    ]


def should_auto_verify(mode: str, changed_files: list[str], profile: str = "local") -> bool:
    if mode == "plan" or not changed_files:
        return False
    return profile in {"balanced", "deep"} or bool(changed_files)


def profile_intake_limits(profile: str, tools_enabled: bool, file_limit: int, snippet_line_limit: int) -> tuple[int, int, int]:
    if tools_enabled:
        if profile == "deep":
            return 220, max(file_limit, 8), max(snippet_line_limit, 120)
        if profile == "balanced":
            return 160, max(file_limit, 6), max(snippet_line_limit, 80)
        return 120, min(file_limit, 4), min(snippet_line_limit, 60)
    if profile == "deep":
        return 80, min(file_limit, 4), min(snippet_line_limit, 60)
    if profile == "balanced":
        return 60, min(file_limit, 4), min(snippet_line_limit, 50)
    return 40, min(file_limit, 3), min(snippet_line_limit, 40)


def profile_step_limit(profile: str, tools_enabled: bool, default_steps: int) -> int:
    if not tools_enabled:
        return 1
    if profile == "deep":
        return max(default_steps, 10)
    if profile == "balanced":
        return max(6, default_steps)
    return min(default_steps, 4)
