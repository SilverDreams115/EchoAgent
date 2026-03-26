from __future__ import annotations

from dataclasses import dataclass
import re
import shlex


SHELL_META_PATTERN = re.compile(r"[|&;<>`]|(?:\$\()")
BLOCKED_EXECUTABLES = {"rm", "mkfs", "shutdown", "reboot", "poweroff", "sudo", "su", "dd"}
SAFE_COMMAND_POLICIES: dict[str, set[tuple[str, ...]]] = {
    "python": {
        ("-m", "compileall"),
        ("-m", "unittest"),
        ("-m", "pytest"),
    },
    "python3": {
        ("-m", "compileall"),
        ("-m", "unittest"),
        ("-m", "pytest"),
    },
    "pytest": {()},
    "git": {
        ("status",),
        ("diff",),
    },
    "npm": {
        ("test",),
        ("run", "test"),
        ("run", "lint"),
        ("run", "build"),
        ("run", "check"),
        ("run", "typecheck"),
    },
    "pnpm": {
        ("test",),
        ("run", "test"),
        ("run", "lint"),
        ("run", "build"),
        ("run", "check"),
        ("run", "typecheck"),
    },
    "yarn": {
        ("test",),
        ("lint",),
        ("build",),
        ("check",),
        ("typecheck",),
    },
}


@dataclass(slots=True)
class ShellCommandDecision:
    allowed: bool
    argv: list[str]
    reason: str = ""


def validate_shell_command(command: str) -> ShellCommandDecision:
    stripped = command.strip()
    if not stripped:
        return ShellCommandDecision(False, [], "Shell command is empty.")
    if SHELL_META_PATTERN.search(stripped):
        return ShellCommandDecision(False, [], "Shell metacharacters are not allowed.")
    try:
        argv = shlex.split(stripped, posix=True)
    except ValueError as exc:
        return ShellCommandDecision(False, [], f"Shell command could not be parsed safely: {exc}")
    if not argv:
        return ShellCommandDecision(False, [], "Shell command is empty.")

    executable = argv[0]
    if executable in BLOCKED_EXECUTABLES:
        return ShellCommandDecision(False, argv, f"Executable blocked by policy: {executable}")
    if executable not in SAFE_COMMAND_POLICIES:
        return ShellCommandDecision(False, argv, f"Executable not allowed by policy: {executable}")

    if executable == "git":
        if len(argv) >= 3 and argv[1] == "reset" and "--hard" in argv[2:]:
            return ShellCommandDecision(False, argv, "Git reset --hard is blocked by policy.")
        if len(argv) >= 2 and argv[1] == "clean":
            return ShellCommandDecision(False, argv, "Git clean is blocked by policy.")

    allowed_prefixes = SAFE_COMMAND_POLICIES[executable]
    tail = tuple(argv[1:])
    if not any(tail[: len(prefix)] == prefix for prefix in allowed_prefixes):
        return ShellCommandDecision(False, argv, f"Arguments not allowed by policy for executable: {executable}")
    return ShellCommandDecision(True, argv)
