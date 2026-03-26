from __future__ import annotations

import json
import time

from echo.types import ToolCallRecord


def record_tool_call(session, run_state, name: str, arguments: dict[str, object], result: dict[str, object]) -> None:
    tracked_path = arguments.get("path")
    if isinstance(tracked_path, str) and tracked_path.strip():
        if tracked_path not in session.focus_files:
            session.focus_files.append(tracked_path)
        if tracked_path not in session.working_set:
            session.working_set.append(tracked_path)
        if name in {"read_file", "read_file_range", "search_symbol", "find_symbol"} and tracked_path not in run_state.inspected_files:
            run_state.inspected_files.append(tracked_path)
        if name in {"write_file", "apply_patch", "insert_before", "insert_after", "replace_range"} and tracked_path not in run_state.changed_files:
            run_state.changed_files.append(tracked_path)
    if name == "validate_project" and "validation_command" in result:
        command = str(result["validation_command"])
        session.validation.append(command)
        run_state.validation_commands.append(command)
    if name in {"read_file", "read_file_range"}:
        preview_payload = {
            "path": result.get("path", ""),
            "content": str(result.get("content", ""))[:2200],
        }
        preview = json.dumps(preview_payload, ensure_ascii=False)
    elif name == "list_files":
        preview = json.dumps({"items": list(result.get("items", []))[:80]}, ensure_ascii=False)
    elif name in {"search_symbol", "find_symbol"}:
        preview_payload = {
            "path": result.get("path", ""),
            "matches": list(result.get("matches", []))[:8],
        }
        preview = json.dumps(preview_payload, ensure_ascii=False)
    elif name == "validate_project":
        preview_payload = {
            "validation_strategy": result.get("validation_strategy", ""),
            "validation_command": result.get("validation_command", ""),
            "returncode": result.get("returncode", -1),
            "error": result.get("error", ""),
            "stderr": str(result.get("stderr", ""))[:400],
        }
        preview = json.dumps(preview_payload, ensure_ascii=False)
    else:
        preview = json.dumps(result, ensure_ascii=False)[:400]
    session.tool_calls.append(ToolCallRecord(tool=name, arguments=arguments, result_preview=preview))
    if name in {"search_symbol", "find_symbol"} and isinstance(result.get("matches"), list):
        for match in result["matches"][:8]:
            symbol = match.get("symbol")
            path = match.get("path")
            if symbol and path:
                finding = f"{path}:{symbol}"
                if finding not in run_state.findings:
                    run_state.findings.append(finding)
    if result.get("error"):
        issue = f"{name}: {result['error']}"
        run_state.errors.append(issue)
        if issue not in run_state.open_issues:
            run_state.open_issues.append(issue)
    current_stage = next((stage for stage in run_state.plan_stages if stage.stage_id == run_state.current_stage_id), None)
    if current_stage is not None:
        evidence = []
        if tracked_path:
            evidence.append(str(tracked_path))
        if name == "validate_project" and result.get("validation_command"):
            evidence.append(str(result.get("validation_command")))
        if evidence:
            current_stage.evidence = list(dict.fromkeys(current_stage.evidence + evidence))
            current_stage.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
