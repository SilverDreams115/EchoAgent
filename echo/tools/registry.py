from __future__ import annotations

from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from echo.cognition.validation import detect_validation_plan
from echo.config import Settings
from echo.memory import EchoStore
from echo.runtime.activity import ActivityBus
from echo.tools.shell_policy import validate_shell_command


TEXT_PATTERNS = [
    re.compile(r"^\s*def\s+([a-zA-Z0-9_]+)\s*\(", re.MULTILINE),
    re.compile(r"^\s*class\s+([a-zA-Z0-9_]+)\s*[:(]", re.MULTILINE),
    re.compile(r"^\s*async\s+def\s+([a-zA-Z0-9_]+)\s*\(", re.MULTILINE),
]
class ToolRegistry:
    def __init__(self, project_root: Path, settings: Settings, activity: ActivityBus) -> None:
        self.project_root = project_root
        self.settings = settings
        self.activity = activity
        self.store = EchoStore(project_root)

    def schema(self) -> list[dict[str, Any]]:
        return [
            self._tool("list_files", "List files in the current project.", {"path": {"type": "string"}, "max_depth": {"type": "integer"}}),
            self._tool("read_file", "Read a text file from the current project.", {"path": {"type": "string"}}, ["path"]),
            self._tool(
                "read_file_range",
                "Read a line range from a text file in the current project.",
                {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                ["path", "start_line", "end_line"],
            ),
            self._tool("search_text", "Search text in the repository.", {"query": {"type": "string"}}, ["query"]),
            self._tool("grep_code", "Search code text in the repository.", {"query": {"type": "string"}}, ["query"]),
            self._tool("search_symbol", "Search code symbols in a specific file or the whole repo.", {"symbol": {"type": "string"}, "path": {"type": "string"}}, ["symbol"]),
            self._tool("find_symbol", "Find code symbols in the repository.", {"symbol": {"type": "string"}, "path": {"type": "string"}}, ["symbol"]),
            self._tool("write_file", "Write a text file inside the current project.", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
            self._tool(
                "apply_patch",
                "Apply a minimal text patch to one file by replacing a known snippet.",
                {"path": {"type": "string"}, "find": {"type": "string"}, "replace": {"type": "string"}, "replace_all": {"type": "boolean"}},
                ["path", "find", "replace"],
            ),
            self._tool("insert_before", "Insert text before a known snippet in one file.", {"path": {"type": "string"}, "anchor": {"type": "string"}, "content": {"type": "string"}}, ["path", "anchor", "content"]),
            self._tool("insert_after", "Insert text after a known snippet in one file.", {"path": {"type": "string"}, "anchor": {"type": "string"}, "content": {"type": "string"}}, ["path", "anchor", "content"]),
            self._tool(
                "replace_range",
                "Replace a line range in a text file.",
                {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}, "content": {"type": "string"}},
                ["path", "start_line", "end_line", "content"],
            ),
            self._tool("run_shell", "Run a safe shell command inside the project.", {"command": {"type": "string"}}, ["command"]),
            self._tool("validate_project", "Run a reasonable validation command for the current project.", {}),
            self._tool("git_status", "Run git status --short inside the project.", {}),
            self._tool("git_diff", "Run git diff --stat inside the project.", {}),
        ]

    def _tool(self, name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties},
            },
        }
        if required:
            payload["function"]["parameters"]["required"] = required
        return payload

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = handler(arguments)
        except Exception as exc:
            result = {"error": str(exc)}
        self._log_command({"kind": "tool", "tool": name, "arguments": arguments, "result": result})
        return result

    def compatibility_guide(self) -> str:
        lines = [
            "Available tools and required arguments:",
        ]
        for item in self.schema():
            fn = item["function"]
            required = ", ".join(fn["parameters"].get("required", [])) or "none"
            lines.append(f"- {fn['name']}({required})")
        lines += [
            "",
            'When a tool is needed, respond with JSON only using this shape:',
            '{"tool_calls":[{"name":"read_file","arguments":{"path":"echo/runtime/engine.py"}}]}',
            "Do not include explanations in the same message as tool JSON.",
        ]
        return "\n".join(lines)

    def _resolve_path(self, relative: str) -> Path:
        target = (self.project_root / relative).resolve()
        if self.project_root not in target.parents and target != self.project_root:
            raise ValueError("Path escapes project root")
        return target

    def _log_command(self, payload: dict[str, object]) -> None:
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.store.append_command_log(payload)

    def _reject_command(self, command: str, reason: str, argv: list[str] | None = None) -> dict[str, Any]:
        result = {
            "command": command,
            "argv": argv or [],
            "policy_decision": "blocked",
            "error": reason,
        }
        self._log_command({"kind": "tool-policy", "tool": "run_shell", "command": command, "argv": argv or [], "decision": "blocked", "reason": reason})
        return result

    def _in_git_repo(self) -> bool:
        return (self.project_root / ".git").exists()

    def _read_text(self, rel: str) -> tuple[Path, str]:
        path = self._resolve_path(rel)
        return path, path.read_text(encoding="utf-8")

    def _write_text(self, rel: str, updated: str, original: str) -> dict[str, Any]:
        path = self._resolve_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
        diff = "\n".join(
            difflib.unified_diff(
                original.splitlines(),
                updated.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        return {"path": rel, "diff": diff[:12000]}

    def tool_list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        rel = arguments.get("path", "") or ""
        depth = int(arguments.get("max_depth", 2) or 2)
        root = self._resolve_path(rel)
        results: list[str] = []

        def walk(current: Path, level: int) -> None:
            if level > depth:
                return
            children = sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            for child in children:
                if child.name in {".git", ".venv", "node_modules", "__pycache__", ".echo"}:
                    continue
                rel_path = child.relative_to(self.project_root)
                results.append(str(rel_path) + ("/" if child.is_dir() else ""))
                if child.is_dir():
                    walk(child, level + 1)

        self.activity.emit("Inspector", "running", "Inspector listing files", rel or ".")
        walk(root, 0)
        self.activity.emit("Inspector", "done", "Files listed", f"count={len(results)}")
        return {"items": results[:300]}

    def tool_read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        rel = arguments["path"]
        self.activity.emit("Inspector", "running", "Inspector reading file", rel)
        _, content = self._read_text(rel)
        self.activity.emit("Inspector", "done", "File read", rel)
        return {"path": rel, "content": content[:24000]}

    def tool_read_file_range(self, arguments: dict[str, Any]) -> dict[str, Any]:
        rel = arguments["path"]
        start_line = max(1, int(arguments["start_line"]))
        end_line = max(start_line, int(arguments["end_line"]))
        self.activity.emit("Inspector", "running", "Inspector reading file range", f"{rel}:{start_line}-{end_line}")
        _, content = self._read_text(rel)
        lines = content.splitlines()
        selected = lines[start_line - 1:end_line]
        numbered = "\n".join(f"{index}: {line}" for index, line in enumerate(selected, start=start_line))
        self.activity.emit("Inspector", "done", "File range read", rel)
        return {"path": rel, "start_line": start_line, "end_line": end_line, "content": numbered}

    def tool_search_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments["query"])
        self.activity.emit("Inspector", "running", "Inspector searching text", query)
        command = ["rg", "-n", "--hidden", "--glob", "!.echo/**", "--glob", "!.venv/**", query, str(self.project_root)]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=self.settings.shell_timeout)
        matches = completed.stdout.splitlines()[:120]
        self.activity.emit("Inspector", "done", "Text search completed", f"matches={len(matches)}")
        return {"query": query, "matches": matches, "returncode": completed.returncode}

    def tool_grep_code(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.tool_search_text(arguments)

    def tool_search_symbol(self, arguments: dict[str, Any]) -> dict[str, Any]:
        symbol = str(arguments["symbol"])
        rel = str(arguments.get("path", "") or "")
        targets = [self._resolve_path(rel)] if rel else [path for path in self.project_root.rglob("*.py") if ".echo" not in path.parts and ".venv" not in path.parts]
        self.activity.emit("Inspector", "running", "Inspector searching symbol", symbol)
        results: list[dict[str, Any]] = []
        for path in targets:
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for pattern in TEXT_PATTERNS:
                for match in pattern.finditer(text):
                    if match.group(1) == symbol:
                        line = text[:match.start()].count("\n") + 1
                        results.append({"path": str(path.relative_to(self.project_root)), "line": line, "symbol": symbol})
        self.activity.emit("Inspector", "done", "Symbol search completed", f"matches={len(results)}")
        return {"symbol": symbol, "matches": results[:80]}

    def tool_find_symbol(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.tool_search_symbol(arguments)

    def tool_write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.allow_write:
            return {"error": "Writing files is disabled by policy."}
        rel = arguments["path"]
        content = arguments["content"]
        path = self._resolve_path(rel)
        original = path.read_text(encoding="utf-8") if path.exists() else ""
        self.activity.emit("Executor", "running", "Executor writing file", rel)
        result = self._write_text(rel, content, original)
        self.activity.emit("Executor", "done", "File written", rel)
        result["written"] = len(content)
        return result

    def tool_apply_patch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.allow_write:
            return {"error": "Writing files is disabled by policy."}
        rel = arguments["path"]
        find = arguments["find"]
        replace = arguments["replace"]
        replace_all = bool(arguments.get("replace_all", False))
        _, original = self._read_text(rel)
        if find not in original:
            return {"error": f"Snippet not found in {rel}"}
        self.activity.emit("Executor", "running", "Executor applying patch", rel)
        updated = original.replace(find, replace) if replace_all else original.replace(find, replace, 1)
        result = self._write_text(rel, updated, original)
        self.activity.emit("Executor", "done", "Patch applied", rel)
        return result

    def tool_insert_before(self, arguments: dict[str, Any]) -> dict[str, Any]:
        rel = arguments["path"]
        anchor = arguments["anchor"]
        content = arguments["content"]
        _, original = self._read_text(rel)
        if anchor not in original:
            return {"error": f"Anchor not found in {rel}"}
        updated = original.replace(anchor, content + anchor, 1)
        self.activity.emit("Executor", "running", "Executor inserting before anchor", rel)
        result = self._write_text(rel, updated, original)
        self.activity.emit("Executor", "done", "Insert before completed", rel)
        return result

    def tool_insert_after(self, arguments: dict[str, Any]) -> dict[str, Any]:
        rel = arguments["path"]
        anchor = arguments["anchor"]
        content = arguments["content"]
        _, original = self._read_text(rel)
        if anchor not in original:
            return {"error": f"Anchor not found in {rel}"}
        updated = original.replace(anchor, anchor + content, 1)
        self.activity.emit("Executor", "running", "Executor inserting after anchor", rel)
        result = self._write_text(rel, updated, original)
        self.activity.emit("Executor", "done", "Insert after completed", rel)
        return result

    def tool_replace_range(self, arguments: dict[str, Any]) -> dict[str, Any]:
        rel = arguments["path"]
        start_line = max(1, int(arguments["start_line"]))
        end_line = max(start_line, int(arguments["end_line"]))
        content = str(arguments["content"])
        _, original = self._read_text(rel)
        lines = original.splitlines()
        replacement = content.splitlines()
        updated_lines = lines[:start_line - 1] + replacement + lines[end_line:]
        updated = "\n".join(updated_lines) + ("\n" if original.endswith("\n") else "")
        self.activity.emit("Executor", "running", "Executor replacing line range", f"{rel}:{start_line}-{end_line}")
        result = self._write_text(rel, updated, original)
        self.activity.emit("Executor", "done", "Line range replaced", rel)
        return result

    def tool_run_shell(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.allow_shell:
            return {"error": "Shell execution is disabled by policy."}
        command = str(arguments["command"])
        decision = validate_shell_command(command)
        if not decision.allowed:
            return self._reject_command(command, decision.reason, decision.argv)
        self.activity.emit("Executor", "running", "Executor running shell", command)
        completed = subprocess.run(
            decision.argv,
            shell=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=self.settings.shell_timeout,
        )
        status = "done" if completed.returncode == 0 else "failed"
        self.activity.emit("Verifier", status, "Shell command finished", f"rc={completed.returncode}")
        return {
            "command": command,
            "argv": decision.argv,
            "policy_decision": "allowed",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
        }

    def tool_validate_project(self, arguments: dict[str, Any]) -> dict[str, Any]:
        plan = detect_validation_plan(self.project_root)
        if plan.strategy == "unknown" or not plan.command:
            self.activity.emit("Verifier", "warning", "Verifier validation unavailable", plan.reason)
            return {
                "validation_strategy": plan.strategy,
                "validation_reason": plan.reason,
                "validation_command": "",
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "error": "No safe validation strategy available for this repository.",
            }
        self.activity.emit("Verifier", "running", "Verifier running tests", plan.command)
        result = self.tool_run_shell({"command": plan.command})
        return {
            "validation_strategy": plan.strategy,
            "validation_reason": plan.reason,
            "validation_command": plan.command,
            **result,
        }

    def tool_git_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._in_git_repo():
            return {"error": "No .git directory found in project root."}
        return self.tool_run_shell({"command": "git status --short"})

    def tool_git_diff(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._in_git_repo():
            return {"error": "No .git directory found in project root."}
        return self.tool_run_shell({"command": "git diff --stat"})
