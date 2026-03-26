from __future__ import annotations


def run_auto_verify(
    *,
    session,
    run_state,
    settings,
    profile: str,
    should_auto_verify,
    find_stage,
    update_stage,
    mark_phase,
    execute_tool,
    record_tool_call,
    sync_memory_layers,
) -> tuple[str, str]:
    if not settings.auto_verify or not should_auto_verify(run_state.mode, run_state.changed_files, profile=profile):
        verify_stage = find_stage(run_state, "verify")
        if verify_stage is not None and verify_stage.status == "pending":
            update_stage(session, run_state, "verify", status="skipped", result="Verification skipped by policy or because there are no changed files.")
        sync_memory_layers(session, run_state)
        return "done", "verification skipped"
    update_stage(session, run_state, "verify", status="running", result="Validation stage started.")
    mark_phase(run_state, "Verifier", "running", "Verifier running validation")
    result = execute_tool("validate_project", {})
    record_tool_call(session, run_state, "validate_project", {}, result)
    command = str(result.get("validation_command", ""))
    strategy = str(result.get("validation_strategy", "unknown"))
    if strategy == "unknown":
        issue = f"Validation unavailable: {result.get('validation_reason', 'unknown strategy')}"
        run_state.open_issues.append(issue)
        run_state.errors.append(issue)
        session.errors.append(issue)
        mark_phase(run_state, "Verifier", "failed", issue)
        update_stage(session, run_state, "verify", status="failed", result=issue)
        sync_memory_layers(session, run_state)
        return "done", issue
    if result.get("returncode", 1) != 0:
        issue = f"Validation failed: {command}"
        run_state.open_issues.append(issue)
        run_state.errors.append(issue)
        session.errors.append(issue)
        mark_phase(run_state, "Verifier", "failed", issue)
        update_stage(session, run_state, "verify", status="failed", result=issue)
        sync_memory_layers(session, run_state)
        return "done", issue
    mark_phase(run_state, "Verifier", "done", f"Validation passed: {strategy} {command}")
    update_stage(session, run_state, "verify", status="completed", result=f"Validation passed: {strategy} {command}", evidence=[command])
    sync_memory_layers(session, run_state)
    return "done", f"validation passed: {strategy}"
