from __future__ import annotations

import json

from echo.backends.errors import BackendMalformedResponseError, BackendModelMissingError, BackendTimeoutError, BackendUnreachableError
from echo.cognition import build_plan_prompt, validate_final_answer
from echo.runtime.budget import RuntimeBudget


def _sync_grounding_report(run_state, valid: bool, reason: str, report: dict[str, object]) -> None:
    run_state.grounding_report.grounded_file_count = int(report.get("grounded_file_count", 0))
    run_state.grounding_report.grounded_symbol_count = int(report.get("grounded_symbol_count", 0))
    run_state.grounding_report.evidence_usage = int(report.get("evidence_usage", 0))
    run_state.grounding_report.genericity_score = int(report.get("genericity_score", 0))
    run_state.grounding_report.useful = bool(report.get("useful", True))
    run_state.grounding_report.claim_types = list(report.get("claim_types", []))
    run_state.grounding_report.unsupported_files = list(report.get("unsupported_files", []))
    run_state.grounding_report.unsupported_symbols = list(report.get("unsupported_symbols", []))
    run_state.grounding_report.unsupported_commands = list(report.get("unsupported_commands", []))
    run_state.grounding_report.unsupported_changes = list(report.get("unsupported_changes", []))
    run_state.grounding_report.contradiction_flags = list(report.get("contradiction_flags", []))
    run_state.grounding_report.validation_strategy_match = bool(report.get("validation_strategy_match", True))
    run_state.grounding_report.validation_strategy = str(report.get("validation_strategy", "unknown"))
    run_state.grounding_report.speculation_flags = list(report.get("speculation_flags", []))
    run_state.grounding_report.valid = bool(report.get("valid", valid))
    run_state.grounding_report.reason = str(report.get("reason", reason))


def run_model_loop(
    *,
    session,
    run_state,
    messages: list[dict[str, object]],
    prompt: str,
    mode: str,
    profile: str,
    settings,
    budget: RuntimeBudget,
    backend_native_tools_enabled,
    tools_schema,
    step_limit: int,
    compress_messages,
    context_ratio,
    mark_phase,
    call_backend,
    extract_tool_calls,
    degraded_answer,
    local_inspect_answer,
    update_stage,
    replan_stage,
    reduce_context,
    grounding_retry_message,
    record_tool_call,
    execute_tool,
    collect_tool_previews,
    validation_strategy,
    find_stage,
) -> str:
    if session.mode != "plan":
        execute_stage = find_stage(run_state, "execute")
        if execute_stage is not None and execute_stage.status == "pending":
            update_stage(session, run_state, "execute", status="running", result="Execution stage started.")

    user_prompt = build_plan_prompt(prompt, profile=profile) if mode == "plan" else prompt
    messages.append({"role": "user", "content": user_prompt})
    session.messages.append(messages[-1])
    final_answer = ""
    timeout_retried = False
    grounding_retried = False
    malformed_retried = False
    min_retry_window_ms = max(1000, settings.backend_preflight_timeout * 1000)

    for step in range(1, step_limit + 1):
        if budget.expired():
            return degraded_answer(session, run_state, "Se agotó el presupuesto total de tiempo del runtime.", mode)

        messages, compressed = compress_messages(messages)
        if compressed:
            session.operational_summary = compressed
            session.operational_memory.summary = compressed
            run_state.compression_count += 1
            session.compression_count = run_state.compression_count
        run_state.context_free_ratio = context_ratio(messages)
        mark_phase(run_state, "Planner", "running", f"Planner running step={step}")
        tools = tools_schema() if backend_native_tools_enabled() else None
        request_timeout_seconds = budget.request_timeout_seconds(settings.backend_timeout)

        try:
            response = call_backend(run_state, messages, tools=tools, timeout_seconds=request_timeout_seconds)
        except BackendTimeoutError as exc:
            issue = str(exc)
            run_state.errors.append(issue)
            run_state.open_issues.append(issue)
            session.errors.append(issue)
            if not timeout_retried and len(messages) > 4 and budget.allows_retry(min_retry_window_ms=min_retry_window_ms):
                timeout_retried = True
                run_state.retry_count += 1
                session.retry_count = run_state.retry_count
                failed_stage = run_state.current_stage_id or "execute"
                update_stage(session, run_state, failed_stage, status="failed", result=issue)
                replanned_id = replan_stage(session, run_state, failed_stage, issue)
                update_stage(session, run_state, replanned_id, status="running", result="Retry after timeout.")
                mark_phase(run_state, "Backend", "retrying", "Retry with reduced context")
                messages = reduce_context(session, run_state, messages)
                session.working_set = session.working_set[-2:] or session.focus_files[-2:]
                retry_message = {
                    "role": "system",
                    "content": "Previous request timed out. Use only the minimum working set, cite files explicitly, and respond briefly.",
                }
                messages.append(retry_message)
                session.messages.append(retry_message)
                continue
            return degraded_answer(session, run_state, issue, mode)
        except (BackendModelMissingError, BackendUnreachableError, BackendMalformedResponseError) as exc:
            issue = str(exc)
            run_state.errors.append(issue)
            session.errors.append(issue)
            if isinstance(exc, BackendMalformedResponseError) and not malformed_retried and budget.allows_retry(min_retry_window_ms=min_retry_window_ms):
                malformed_retried = True
                run_state.retry_count += 1
                session.retry_count = run_state.retry_count
                failed_stage = run_state.current_stage_id or "execute"
                update_stage(session, run_state, failed_stage, status="failed", result=issue)
                replanned_id = replan_stage(session, run_state, failed_stage, issue)
                update_stage(session, run_state, replanned_id, status="running", result="Retry after malformed response.")
                mark_phase(run_state, "Backend", "retrying", "Retry with reduced context")
                messages = reduce_context(session, run_state, messages)
                messages.append(
                    {
                        "role": "system",
                        "content": "Previous response was malformed. Return plain text or valid tool JSON only.",
                    }
                )
                continue
            return degraded_answer(session, run_state, issue, mode)

        mark_phase(run_state, "Planner", "done", f"Planner completed step={step}")
        message = response.get("message", {})
        content = (message.get("content") or "").strip()
        assistant = {"role": "assistant", "content": content}
        tool_calls = extract_tool_calls(message, content)
        if backend_native_tools_enabled() and tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        session.messages.append(assistant)

        if not tool_calls:
            if mode == "ask" and len(session.tool_calls) == 0:
                run_state.open_issues.append("ask completed without real tool execution")
                return local_inspect_answer(session, run_state)
            valid, reason, report = validate_final_answer(
                content,
                profile=profile,
                mode=mode,
                inspected_files=run_state.inspected_files,
                changed_files=run_state.changed_files,
                tool_calls=session.tool_calls,
                tool_result_previews=collect_tool_previews(session),
                working_set=session.working_set,
                validation_strategy=validation_strategy(session),
            )
            _sync_grounding_report(run_state, valid, reason, report)
            if not valid:
                run_state.open_issues.append(reason)
                if not grounding_retried and budget.allows_retry(min_retry_window_ms=min_retry_window_ms):
                    grounding_retried = True
                    run_state.retry_count += 1
                    session.retry_count = run_state.retry_count
                    failed_stage = run_state.current_stage_id or "execute"
                    update_stage(session, run_state, failed_stage, status="failed", result=reason)
                    replanned_id = replan_stage(session, run_state, failed_stage, reason)
                    update_stage(session, run_state, replanned_id, status="running", result="Retry after grounding failure.")
                    mark_phase(run_state, "Verifier", "retrying", reason)
                    retry_message = grounding_retry_message(session, run_state, reason)
                    messages.append(retry_message)
                    session.messages.append(retry_message)
                    continue
                return degraded_answer(session, run_state, f"Respuesta final no grounded tras retry: {reason}", mode)
            final_answer = content
            session.grounded_answer = True
            session.grounding_report = report
            current_stage = run_state.current_stage_id or "execute"
            update_stage(session, run_state, current_stage, status="completed", result="Grounded answer accepted.")
            break

        for call in tool_calls:
            fn = call.get("function", {})
            name = str(fn.get("name", "")).strip()
            arguments = fn.get("arguments", {}) or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}
            result = execute_tool(name, arguments)
            record_tool_call(session, run_state, name, arguments, result)
            tool_message = {
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False),
            }
            messages.append(tool_message)
            session.messages.append(tool_message)
        run_state.pending = ["Consume the latest tool results before finalizing."]
    else:
        final_answer = degraded_answer(session, run_state, "Echo alcanzó el máximo de pasos.", mode)
    return final_answer
