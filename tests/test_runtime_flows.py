from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from echo.backends.errors import BackendMalformedResponseError, BackendTimeoutError, BackendUnreachableError
from echo.backends.availability import BackendAvailabilityPolicy, run_backend_check
from echo.backends.health import effective_backend_health, normalize_backend_health
from echo.cognition.planner import build_execution_plan
from echo.cognition.validation import detect_validation_plan
from echo.cognition.verifier import evaluate_final_answer
from echo.context.compressor import compress_messages_if_needed
from echo.config import Settings
from echo.core import EchoAgent
from echo.memory import EchoStore
from echo.runtime import ActivityBus, AgentRuntime
from echo.tools import ToolRegistry
from echo.types import ToolCallRecord

try:
    from echo.cli.app import app as cli_app
    from typer.testing import CliRunner
except ModuleNotFoundError:
    cli_app = None
    CliRunner = None


class FakeBackend:
    backend_name = "fake"
    model = "fake-model"
    supports_tools = True
    supports_native_tools = False

    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.timeout = 120

    def list_models(self) -> list[str]:
        return [self.model]

    def chat(self, messages, tools=None):
        self.calls += 1
        return self.responses.pop(0)


class TimeoutOnceBackend(FakeBackend):
    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            raise BackendTimeoutError("timeout", backend="fake", model="fake-model")
        return self.responses.pop(0)


class UnreachableBackend(FakeBackend):
    def chat(self, messages, tools=None):
        self.calls += 1
        raise BackendUnreachableError("backend caído", backend="fake", model="fake-model")


class MalformedOnceBackend(FakeBackend):
    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            raise BackendMalformedResponseError("respuesta inválida", backend="fake", model="fake-model")
        return self.responses.pop(0)


class RuntimeFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.cli_runner = CliRunner() if CliRunner is not None else None
        (self.root / "echo").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "echo" / "sample.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
        (self.root / "README.md").write_text("# Demo\n", encoding="utf-8")
        (self.root / "tests" / "test_sample.py").write_text(
            "import unittest\n\n\nclass SampleTests(unittest.TestCase):\n    def test_placeholder(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        self.settings = Settings.from_env()
        self.settings.allow_shell = False
        self.settings.auto_verify = False
        self.settings.context_char_limit = 600
        self.settings.context_message_limit = 8
        self.settings.backend_failure_threshold = 1
        self.settings.backend_preflight_timeout = 1

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _runtime(self, backend) -> tuple[AgentRuntime, EchoStore]:
        activity = ActivityBus()
        store = EchoStore(self.root)
        activity.watch(
            lambda event: store.append_activity_log(
                {
                    "stage": event.stage,
                    "status": event.status,
                    "message": event.message,
                    "detail": event.detail,
                    "created_at": event.created_at,
                }
            )
        )
        tools = ToolRegistry(self.root, self.settings, activity)
        return AgentRuntime(self.root, self.settings, backend, store, tools, activity), store

    def _runtime_with_shell(self, backend) -> tuple[AgentRuntime, EchoStore]:
        self.settings.allow_shell = True
        return self._runtime(backend)

    def _mark_backend_unstable(self, store: EchoStore, event: str = "error") -> None:
        payload = {
            "event": event,
            "backend": "fake",
            "model": "fake-model",
            "duration_ms": 1000,
            "detail": "backend caído",
            "created_at": "2026-03-24T00:00:00Z",
        }
        if event == "timeout":
            payload["detail"] = "timeout"
        store.append_backend_log(payload)

    def _tool_call(self, tool: str, arguments: dict[str, object], result: dict[str, object]) -> ToolCallRecord:
        return ToolCallRecord(tool=tool, arguments=arguments, result_preview=json.dumps(result, ensure_ascii=False))

    def test_answer_validator_rejects_generic_output_without_grounding(self) -> None:
        report = evaluate_final_answer(
            "Aqui tienes una inspección general del proyecto. Puedes usar pytest para validarlo.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path": "echo/sample.py", "content": "def run()"}'],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertEqual(report.grounded_file_count, 0)
        self.assertFalse(report.validation_strategy_match)

    def test_answer_validator_rejects_wrong_validation_strategy(self) -> None:
        report = evaluate_final_answer(
            "Revisa echo/sample.py y luego ejecuta pytest tests/test_sample.py",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path": "echo/sample.py", "content": "def run()"}'],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertIn("pytest", "Revisa echo/sample.py y luego ejecuta pytest tests/test_sample.py")
        self.assertFalse(report.validation_strategy_match)

    def test_detect_validation_plan_prefers_pytest_when_configured(self) -> None:
        (self.root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "pytest")
        self.assertEqual(plan.command, "python3 -m pytest")

    def test_execution_plan_has_required_stage_fields(self) -> None:
        stages = build_execution_plan("analiza echo/sample.py", mode="ask", focus_files=["echo/sample.py"], validation_strategy="unittest")
        self.assertGreaterEqual(len(stages), 3)
        for stage in stages:
            self.assertTrue(stage.stage_id)
            self.assertTrue(stage.objective)
            self.assertTrue(stage.hypothesis)
            self.assertIsInstance(stage.target_files, list)
            self.assertTrue(stage.intended_actions)
            self.assertTrue(stage.validation_goal)
            self.assertTrue(stage.completion_criteria)
            self.assertTrue(stage.failure_policy)

    def test_detect_validation_plan_uses_unittest_layout(self) -> None:
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "unittest")
        self.assertEqual(plan.command, "python3 -m unittest discover -s tests -p test_*.py")

    def test_detect_validation_plan_falls_back_to_compileall_for_python_without_tests(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "compileall")
        self.assertIn("python3 -m compileall", plan.command)

    def test_detect_validation_plan_uses_package_json_test_script(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        for item in (self.root / "echo").glob("*"):
            item.unlink()
        (self.root / "echo").rmdir()
        (self.root / "package.json").write_text(json.dumps({"scripts": {"test": "vitest"}}), encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "npm-test")
        self.assertEqual(plan.command, "npm test")

    def test_detect_validation_plan_returns_unknown_for_package_without_safe_script(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        for item in (self.root / "echo").glob("*"):
            item.unlink()
        (self.root / "echo").rmdir()
        (self.root / "package.json").write_text(json.dumps({"scripts": {"dev": "vite"}}), encoding="utf-8")
        plan = detect_validation_plan(self.root)
        self.assertEqual(plan.strategy, "unknown")
        self.assertEqual(plan.command, "")

    def test_validate_project_returns_unknown_without_safe_strategy(self) -> None:
        for item in (self.root / "tests").glob("*"):
            item.unlink()
        (self.root / "tests").rmdir()
        for item in (self.root / "echo").glob("*"):
            item.unlink()
        (self.root / "echo").rmdir()
        (self.root / "package.json").write_text(json.dumps({"scripts": {"dev": "vite"}}), encoding="utf-8")
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("validate_project", {})
        self.assertEqual(result["validation_strategy"], "unknown")
        self.assertEqual(result["returncode"], -1)
        self.assertIn("No safe validation strategy", result["error"])

    def test_answer_validator_requires_symbols_when_evidence_contains_them(self) -> None:
        report = evaluate_final_answer(
            "Revisé echo/sample.py y confirmé que ahí vive la lógica principal.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run():\\n    return \\"ok\\"\\nclass Runner:"}'],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertEqual(report.grounded_file_count, 1)
        self.assertEqual(report.grounded_symbol_count, 0)

    def test_answer_validator_rejects_unread_file_claim(self) -> None:
        report = evaluate_final_answer(
            "Revisé echo/other.py y confirmé el cambio.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run()"}'],
            tool_calls=[self._tool_call("read_file", {"path": "echo/sample.py"}, {"path": "echo/sample.py", "content": "def run()"})],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertIn("echo/other.py", report.unsupported_files)

    def test_answer_validator_rejects_symbol_without_evidence(self) -> None:
        report = evaluate_final_answer(
            "Revisé echo/sample.py y la función missing_symbol.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run():\\n    return \\"ok\\""}'],
            tool_calls=[self._tool_call("read_file", {"path": "echo/sample.py"}, {"path": "echo/sample.py", "content": "def run():\n    return \"ok\""})],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertIn("missing_symbol", report.unsupported_symbols)

    def test_answer_validator_rejects_validation_claim_without_executed_command(self) -> None:
        report = evaluate_final_answer(
            "Revisé echo/sample.py y validé con pytest.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run()"}'],
            tool_calls=[self._tool_call("read_file", {"path": "echo/sample.py"}, {"path": "echo/sample.py", "content": "def run()"})],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertIn("validation-not-executed", report.unsupported_commands)

    def test_answer_validator_rejects_unknown_validation_claim_as_executed(self) -> None:
        report = evaluate_final_answer(
            "Revisé echo/sample.py y ejecuté npm test sin errores.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run()"}', '{"validation_strategy":"unknown","validation_command":"","returncode":-1,"error":"No safe validation strategy available for this repository."}'],
            tool_calls=[
                self._tool_call("read_file", {"path": "echo/sample.py"}, {"path": "echo/sample.py", "content": "def run()"}),
                self._tool_call("validate_project", {}, {"validation_strategy": "unknown", "validation_reason": "no safe validation strategy", "validation_command": "", "returncode": -1, "error": "No safe validation strategy available for this repository."}),
            ],
            working_set=["echo/sample.py"],
            validation_strategy="unknown",
        )
        self.assertFalse(report.valid)
        self.assertIn("validation-not-executed", report.unsupported_commands)

    def test_answer_validator_rejects_contradiction_with_failed_validation(self) -> None:
        report = evaluate_final_answer(
            "Revisé echo/sample.py, la función run, y todo pasó sin errores tras ejecutar python3 -m unittest.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run()"}', '{"validation_command":"python3 -m unittest","returncode":1,"stderr":"boom"}'],
            tool_calls=[
                self._tool_call("read_file", {"path": "echo/sample.py"}, {"path": "echo/sample.py", "content": "def run()"}),
                self._tool_call("validate_project", {}, {"validation_command": "python3 -m unittest", "returncode": 1, "stderr": "boom"}),
            ],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertIn("validation-claimed-success-but-failed", report.contradiction_flags)

    def test_answer_validator_rejects_generic_answer_even_with_file_reference(self) -> None:
        report = evaluate_final_answer(
            "Revisé echo/sample.py. Contiene la lógica principal y esto te permitirá seguir.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run():\\n    return \\"ok\\"\\nclass Runner:"}'],
            tool_calls=[self._tool_call("read_file", {"path": "echo/sample.py"}, {"path": "echo/sample.py", "content": "def run():\n    return \"ok\"\nclass Runner:"})],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertGreaterEqual(report.genericity_score, 5)

    def test_timeout_retries_with_reduced_context(self) -> None:
        backend = TimeoutOnceBackend(
            [
                {
                    "message": {
                        "content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."
                    }
                },
                {
                    "message": {
                        "content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."
                    }
                },
            ]
        )
        runtime, store = self._runtime(backend)
        answer, _, session, run_state = runtime.run("inspecciona echo/sample.py y responde", mode="ask")
        self.assertIn("echo/sample.py", answer)
        self.assertGreaterEqual(backend.calls, 2)
        backend_log = store.backend_log.read_text(encoding="utf-8")
        self.assertIn('"event": "timeout"', backend_log)
        activity_log = store.activity_log.read_text(encoding="utf-8")
        self.assertIn("Retry with reduced context", activity_log)
        self.assertTrue(session.grounded_answer)
        self.assertTrue(run_state.grounding_report.valid)
        self.assertGreaterEqual(session.retry_count, 1)

    def test_resume_returns_restored_state_without_backend(self) -> None:
        backend = FakeBackend([{"message": {"content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."}}])
        runtime, store = self._runtime(backend)
        _, _, session, _ = runtime.run("analiza echo/sample.py", mode="ask")
        session.decisions.append("Revisar el runtime antes de editar.")
        session.findings.append("echo/sample.py es un punto de entrada.")
        session.pending.append("Validar cambios.")
        session.working_set.append("echo/sample.py")
        store.save_session(session)
        self._mark_backend_unstable(store, event="error")
        resumed_backend = UnreachableBackend([])
        resumed_runtime, _ = self._runtime(resumed_backend)
        answer, _, resumed_session, run_state = resumed_runtime.run("continúa", mode="resume", resume_session_id=session.id)
        self.assertIn("Echo restauró el estado", answer)
        self.assertIn("echo/sample.py", resumed_session.working_set)
        self.assertEqual(resumed_session.working_memory.active_files[-1], "echo/sample.py")
        self.assertTrue(resumed_session.working_memory.current_stage_id)
        self.assertTrue(run_state.fallback_used)
        self.assertTrue(resumed_session.degraded_reason)

    def test_compression_preserves_operational_facts_and_stage(self) -> None:
        activity = ActivityBus()
        messages = [{"role": "user", "content": f"mensaje {i}"} for i in range(12)]
        compressed, summary = compress_messages_if_needed(
            messages,
            activity,
            message_limit=4,
            char_limit=40,
            objective="resolver bug",
            restrictions=["no romper tests"],
            decisions=["inspeccionar primero"],
            current_stage_id="execute",
            focus_files=["echo/sample.py"],
            changed_files=["echo/sample.py"],
            errors=["timeout"],
            pending=["validar cambios"],
            validation_commands=["python3 -m unittest"],
            confirmed_facts=["echo/sample.py:run"],
        )
        self.assertIsNotNone(summary)
        self.assertIn("Current stage: execute", summary)
        self.assertIn("Validation: python3 -m unittest", summary)
        self.assertIn("Confirmed facts: echo/sample.py:run", summary)
        self.assertEqual(compressed[0]["role"], "system")

    def test_resume_rehydrates_selective_context_not_full_replay(self) -> None:
        backend = FakeBackend([{"message": {"content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."}}])
        runtime, store = self._runtime(backend)
        _, _, session, _ = runtime.run("analiza echo/sample.py", mode="ask")
        session.messages.extend({"role": "assistant", "content": f"ruido {i}"} for i in range(20))
        store.save_session(session)
        resumed_backend = UnreachableBackend([])
        resumed_runtime, _ = self._runtime(resumed_backend)
        _, _, resumed_session, _ = resumed_runtime.run("continúa", mode="resume", resume_session_id=session.id)
        self.assertLessEqual(len(resumed_session.messages), 3)
        self.assertIn("Current stage:", resumed_session.messages[1]["content"])
        self.assertIn("Working set:", resumed_session.messages[1]["content"])

    def test_memory_layers_do_not_diverge_on_active_files(self) -> None:
        backend = FakeBackend([{"message": {"content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."}}])
        runtime, _ = self._runtime(backend)
        _, _, session, _ = runtime.run("analiza echo/sample.py", mode="ask")
        self.assertEqual(session.working_memory.active_files, session.working_set[-len(session.working_memory.active_files):])

    def test_summary_is_coherent_with_plan_stages_tools_and_episodic_memory(self) -> None:
        backend = FakeBackend([{"message": {"content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."}}])
        runtime, _ = self._runtime(backend)
        _, _, session, _ = runtime.run("analiza echo/sample.py", mode="ask")
        self.assertIn(session.working_memory.current_stage_id or "close", session.summary)
        self.assertIn("read_file", session.summary)
        if session.episodic_memory.validations:
            self.assertIn(session.episodic_memory.validations[-1], session.summary)

    def test_long_session_can_continue_from_memory_layers(self) -> None:
        backend = FakeBackend([{"message": {"content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."}}])
        runtime, store = self._runtime(backend)
        _, _, session, _ = runtime.run("analiza echo/sample.py", mode="ask")
        session.operational_memory.summary = "Operational summary\n- Current objective: analiza echo/sample.py\n- Current stage: verify\n- Active files: echo/sample.py\n- Open pending items: validar cambios"
        session.working_memory.current_stage_id = "verify"
        session.working_memory.active_files = ["echo/sample.py"]
        store.save_session(session)
        resumed_backend = UnreachableBackend([])
        resumed_runtime, _ = self._runtime(resumed_backend)
        answer, _, resumed_session, _ = resumed_runtime.run("continúa", mode="resume", resume_session_id=session.id)
        self.assertIn("echo/sample.py", answer)
        self.assertEqual(resumed_session.working_memory.current_stage_id, "verify")

    def test_plan_falls_back_when_chat_unhealthy(self) -> None:
        backend = UnreachableBackend([])
        runtime, store = self._runtime(backend)
        self._mark_backend_unstable(store, event="error")
        answer, _, session, run_state = runtime.run("crea un plan", mode="plan")
        self.assertIn("Objetivo", answer)
        self.assertIn("Archivos a revisar", answer)
        self.assertGreater(len(session.tool_calls), 0)
        self.assertTrue(run_state.fallback_used)

    def test_runtime_registers_stage_progress_for_local_plan_mode(self) -> None:
        backend = FakeBackend([])
        runtime, _ = self._runtime(backend)
        answer, _, session, run_state = runtime.run("planifica cambios en echo/sample.py", mode="plan")
        self.assertIn("Etapas", answer)
        statuses = {stage.stage_id: stage.status for stage in run_state.plan_stages}
        self.assertEqual(statuses["inspect"], "completed")
        self.assertEqual(statuses["stage-plan"], "completed")
        self.assertEqual(statuses["verify-plan"], "completed")
        self.assertEqual(statuses["close"], "completed")
        self.assertTrue(session.plan_stages)

    def test_failed_stage_is_not_marked_completed_after_degraded_retry(self) -> None:
        backend = FakeBackend(
            [
                {"message": {"content": "Cambié echo/sample.py y validé con python3 -m unittest."}},
                {"message": {"content": "Actualicé echo/sample.py y todo pasó sin errores."}},
            ]
        )
        runtime, _ = self._runtime(backend)
        _, _, session, run_state = runtime.run("inspecciona echo/sample.py", mode="ask")
        execute_like = [stage for stage in run_state.plan_stages if stage.stage_id.startswith("execute")]
        self.assertTrue(any(stage.status == "failed" for stage in execute_like))
        self.assertFalse(any(stage.stage_id == "execute" and stage.status == "completed" for stage in execute_like))
        self.assertTrue(session.degraded_reason)

    def test_retry_replan_is_traced_in_plan_stages(self) -> None:
        backend = TimeoutOnceBackend(
            [
                {"message": {"content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."}},
                {"message": {"content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."}},
            ]
        )
        runtime, _ = self._runtime(backend)
        _, _, _, run_state = runtime.run("inspecciona echo/sample.py y responde", mode="ask")
        self.assertTrue(any(stage.replanned_from == "execute" for stage in run_state.plan_stages))
        self.assertGreaterEqual(run_state.retry_count, 1)

    def test_final_summary_includes_stage_state(self) -> None:
        backend = FakeBackend([{"message": {"content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."}}])
        runtime, _ = self._runtime(backend)
        _, _, session, _ = runtime.run("analiza echo/sample.py", mode="ask")
        self.assertIn("## Plan Stages", session.summary)
        self.assertIn("inspect: completed", session.summary)

    def test_degraded_answer_reflects_stage_where_runtime_stopped(self) -> None:
        backend = FakeBackend(
            [
                {"message": {"content": "Cambié echo/sample.py y validé con python3 -m unittest."}},
                {"message": {"content": "Actualicé echo/sample.py y todo pasó sin errores."}},
            ]
        )
        runtime, _ = self._runtime(backend)
        answer, _, _, _ = runtime.run("inspecciona echo/sample.py", mode="ask")
        self.assertIn("Etapa detenida:", answer)

    def test_generic_model_answer_degrades_honestly_after_retry(self) -> None:
        backend = FakeBackend(
            [
                {"message": {"content": "Aquí tienes una inspección general. Podría convenir pytest."}},
                {"message": {"content": "Parece que el proyecto es simple y probablemente baste revisar el código."}},
            ]
        )
        runtime, _ = self._runtime(backend)
        answer, _, session, run_state = runtime.run("inspecciona el repo", mode="ask")
        self.assertIn("Echo reunió inspección local", answer)
        self.assertFalse(session.grounded_answer)
        self.assertFalse(run_state.grounding_report.valid)
        self.assertTrue(session.degraded_reason)

    def test_retry_degrades_honestly_after_unsupported_change_claim(self) -> None:
        backend = FakeBackend(
            [
                {"message": {"content": "Cambié echo/sample.py y validé con python3 -m unittest."}},
                {"message": {"content": "Actualicé echo/sample.py y todo pasó sin errores."}},
            ]
        )
        runtime, _ = self._runtime(backend)
        answer, _, session, run_state = runtime.run("inspecciona echo/sample.py", mode="ask")
        self.assertIn("Echo reunió inspección local", answer)
        self.assertFalse(session.grounded_answer)
        self.assertFalse(run_state.grounding_report.valid)
        self.assertGreaterEqual(run_state.retry_count, 1)

    def test_malformed_response_retries_and_records_artifact(self) -> None:
        backend = MalformedOnceBackend(
            [
                {
                    "message": {
                        "content": "Revisé echo/sample.py, la función run y tests/test_sample.py. La validación correcta es unittest."
                    }
                }
            ]
        )
        runtime, _ = self._runtime(backend)
        answer, session_path, session, run_state = runtime.run("inspecciona echo/sample.py y responde", mode="ask")
        self.assertIn("echo/sample.py", answer)
        self.assertGreaterEqual(run_state.retry_count, 1)
        self.assertTrue(any(item["path"].endswith("-runtime.json") for item in session.artifacts))
        self.assertTrue(Path(session_path).exists())

    def test_backend_check_reports_probe_metrics(self) -> None:
        backend = FakeBackend([{"message": {"content": "ok"}}, {"message": {"content": "ok"}}])
        _, store = self._runtime(backend)
        result = run_backend_check(backend, self.settings, store, chat_samples=2)
        self.assertTrue(result.backend_reachable)
        self.assertTrue(result.backend_chat_ready)
        self.assertEqual(result.chat_probe_count, 2)
        self.assertGreaterEqual(result.success_rate, 1.0)
        self.assertTrue(Path(result.artifact_path).exists())
        self.assertIsInstance(result.rolling_health, dict)
        self.assertIsInstance(result.fresh_health, dict)
        self.assertIsInstance(result.effective_decision, dict)

    def test_backend_check_exposes_routing_decision(self) -> None:
        settings = Settings.from_env()
        settings.openai_api_key = "test-key"
        settings.fallback_backend = "openai-compatible"
        settings.fallback_model = "gpt-4.1-mini"
        agent = EchoAgent(self.root, settings)
        agent.backend = FakeBackend([{"message": {"content": "ok"}}, {"message": {"content": "ok"}}])
        agent.primary_backend_name = "ollama"
        agent.primary_model = "qwen2.5-coder:7b-oh"
        agent.store.append_backend_log(
            {
                "event": "timeout",
                "backend": "ollama",
                "model": "qwen2.5-coder:7b-oh",
                "duration_ms": 2000,
                "created_at": "2026-03-24T00:00:00Z",
            }
        )
        data = agent.backend_check(chat_samples=1)
        self.assertIn("routing_decision", data)
        self.assertIn("selected_backend", data["routing_decision"])

    def test_backend_policy_classifies_fallback_modes(self) -> None:
        from echo.types import BackendHealth

        health = BackendHealth(backend_reachable=False, backend_chat_ready=False, backend_state="unreachable")
        self.assertEqual(BackendAvailabilityPolicy.classify_mode("plan", health), "heuristic_plan")
        self.assertEqual(BackendAvailabilityPolicy.classify_mode("resume", health), "resume_restore_only")
        self.assertEqual(BackendAvailabilityPolicy.classify_mode("ask", health), "local_inspection_only")

    def test_hybrid_routing_selects_fallback_for_critical_ask(self) -> None:
        self.settings.openai_api_key = "test-key"
        self.settings.fallback_backend = "openai-compatible"
        self.settings.fallback_model = "gpt-4.1-mini"
        health = EchoStore(self.root).read_backend_health()
        health.backend_chat_ready = False
        health.backend_chat_slow = True
        route = BackendAvailabilityPolicy.route_backend(
            self.settings,
            profile="balanced",
            mode="ask",
            prompt="haz análisis profundo del runtime y backend",
            primary_backend="ollama",
            primary_model="qwen2.5-coder:7b-oh",
            primary_health=health,
        )
        self.assertTrue(route.fallback_selected)
        self.assertEqual(route.selected_backend, "openai-compatible")
        self.assertEqual(route.policy, "hybrid-fallback")

    def test_hybrid_routing_stays_local_without_fallback_credentials(self) -> None:
        self.settings.openai_api_key = ""
        health = EchoStore(self.root).read_backend_health()
        health.backend_chat_ready = False
        health.backend_chat_slow = True
        route = BackendAvailabilityPolicy.route_backend(
            self.settings,
            profile="balanced",
            mode="ask",
            prompt="haz análisis profundo del runtime y backend",
            primary_backend="ollama",
            primary_model="qwen2.5-coder:7b-oh",
            primary_health=health,
        )
        self.assertFalse(route.fallback_selected)
        self.assertEqual(route.selected_backend, "ollama")
        self.assertEqual(route.policy, "primary-only-critical")

    def test_doctor_labels_cached_and_fresh_health(self) -> None:
        settings = Settings.from_env()
        settings.allow_shell = False
        settings.auto_verify = False
        agent = EchoAgent(self.root, settings)
        agent.backend = FakeBackend([{"message": {"content": "ok"}}])
        agent.resolved_model = "fake-model"
        agent.selected_model = "fake-model"
        agent.selected_backend_name = "fake"
        agent.store.append_backend_log(
            {
                "event": "timeout",
                "backend": "fake",
                "model": "fake-model",
                "duration_ms": 1200,
                "created_at": "2026-03-24T00:00:00Z",
            }
        )
        data = agent.doctor()
        self.assertIn("backend_health_cached_state", data)
        self.assertIn("backend_health_fresh_state", data)
        self.assertIn("backend_health_effective_state", data)
        self.assertEqual(data["backend_health_cached_source"], "rolling")
        self.assertEqual(data["backend_health_fresh_source"], "fresh")
        self.assertTrue(data["backend_chat_ready"])
        self.assertEqual(data["backend_state"], "ready")

    def test_health_normalization_makes_impossible_state_impossible(self) -> None:
        from echo.types import BackendHealth

        normalized = normalize_backend_health(
            BackendHealth(backend_reachable=False, backend_chat_ready=True, backend_state="unreachable")
        )
        self.assertTrue(normalized.backend_reachable)
        self.assertTrue(normalized.backend_chat_ready)
        self.assertEqual(normalized.backend_state, "ready")

    def test_effective_health_prefers_fresh_unreachable_over_stale_ready(self) -> None:
        from echo.types import BackendHealth

        rolling = BackendHealth(backend_reachable=True, backend_chat_ready=True, backend_state="ready", source="rolling")
        fresh = BackendHealth(backend_reachable=False, backend_chat_ready=False, backend_state="unreachable", source="fresh")
        effective = effective_backend_health(rolling, fresh)
        self.assertFalse(effective.backend_reachable)
        self.assertFalse(effective.backend_chat_ready)
        self.assertEqual(effective.backend_state, "unreachable")
        self.assertEqual(effective.source, "fresh")

    def test_doctor_does_not_report_chat_ready_from_stale_rolling_state(self) -> None:
        settings = Settings.from_env()
        settings.allow_shell = False
        settings.auto_verify = False
        agent = EchoAgent(self.root, settings)
        agent.store.append_backend_log(
            {
                "event": "response",
                "backend": "fake",
                "model": "fake-model",
                "duration_ms": 1000,
                "backend_reachable": True,
                "backend_chat_ready": True,
                "backend_chat_slow": False,
                "backend_state": "ready",
            }
        )
        agent.backend = UnreachableBackend([])
        agent.resolved_model = "fake-model"
        agent.selected_model = "fake-model"
        agent.selected_backend_name = "fake"
        data = agent.doctor()
        self.assertTrue(data["backend_reachable"])
        self.assertFalse(data["backend_chat_ready"])
        self.assertEqual(data["backend_state"], "unstable")
        self.assertEqual(data["backend_health_effective_source"], "fresh")

    def test_preflight_degrades_from_fresh_unreachable_even_without_prior_failures(self) -> None:
        backend = UnreachableBackend([])
        runtime, _ = self._runtime(backend)
        answer, _, _, run_state = runtime.run("inspecciona echo/sample.py", mode="ask")
        self.assertIn("Echo reunió inspección local", answer)
        self.assertFalse(run_state.backend_health.backend_chat_ready)
        self.assertEqual(run_state.backend_health.backend_state, "unreachable")
        self.assertTrue(run_state.fallback_used)

    def test_backend_check_and_doctor_agree_on_unreachable_backend(self) -> None:
        settings = Settings.from_env()
        settings.allow_shell = False
        settings.auto_verify = False
        agent = EchoAgent(self.root, settings)
        agent.backend = UnreachableBackend([])
        agent.resolved_model = "fake-model"
        agent.selected_model = "fake-model"
        agent.selected_backend_name = "fake"
        doctor = agent.doctor()
        backend_check = agent.backend_check(chat_samples=1)
        self.assertFalse(doctor["backend_chat_ready"])
        self.assertEqual(doctor["backend_state"], "unstable")
        self.assertFalse(backend_check["backend_chat_ready"])
        self.assertEqual(backend_check["backend_state"], "unstable")

    def test_smoke_aborts_before_agent_run_when_doctor_reports_unhealthy_chat(self) -> None:
        if self.cli_runner is None or cli_app is None:
            self.skipTest("typer no está instalado en este entorno de validación")

        class FakeSmokeAgent:
            def doctor(self):
                return {
                    "profile": "local",
                    "resolved_profile": "local",
                    "deep_ready": False,
                    "backend_primary": "fake",
                    "backend_fallback": "none",
                    "backend_policy": "primary-only",
                    "routing_reason": "test",
                    "backend_label": "fake",
                    "backend_tools": True,
                    "model": "fake-model",
                    "backend_reachable": True,
                    "backend_chat_ready": False,
                    "backend_chat_slow": False,
                    "backend_state": "reachable",
                    "backend_health_cached_state": "ready",
                    "backend_health_fresh_state": "reachable",
                    "backend_health_effective_state": "reachable",
                }

        with patch("echo.cli.app.safe_build_agent", return_value=(FakeSmokeAgent(), self.root, Settings.from_env())), patch(
            "echo.cli.app.safe_agent_run"
        ) as mocked_run:
            result = self.cli_runner.invoke(cli_app, ["smoke", "--project-dir", str(self.root)])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("chat no está sano", result.stdout)
        mocked_run.assert_not_called()

    def test_run_shell_rejects_shell_metacharacters(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("run_shell", {"command": "git status --short && pwd"})
        self.assertEqual(result["policy_decision"], "blocked")
        self.assertIn("metacharacters", result["error"])

    def test_run_shell_rejects_unlisted_executable(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("run_shell", {"command": "echo hello"})
        self.assertEqual(result["policy_decision"], "blocked")
        self.assertIn("not allowed by policy", result["error"])

    def test_run_shell_rejects_destructive_executable(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("run_shell", {"command": "rm -rf ."})
        self.assertEqual(result["policy_decision"], "blocked")
        self.assertIn("blocked by policy", result["error"])

    def test_run_shell_executes_allowed_argv_without_shell(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        result = runtime.tools.execute("run_shell", {"command": "python3 -m unittest discover -s tests -p test_*.py"})
        self.assertEqual(result["policy_decision"], "allowed")
        self.assertEqual(result["argv"][:3], ["python3", "-m", "unittest"])
        self.assertEqual(result["returncode"], 0)

    def test_git_status_runs_through_same_shell_policy(self) -> None:
        runtime, _ = self._runtime_with_shell(FakeBackend([]))
        (self.root / ".git").mkdir()
        result = runtime.tools.execute("git_status", {})
        self.assertEqual(result["policy_decision"], "allowed")
        self.assertEqual(result["argv"][:2], ["git", "status"])

    def test_strict_deep_profile_still_requires_frontier_backend(self) -> None:
        settings = Settings.from_env()
        settings.profile = "deep"
        settings.strict_profile = True
        settings.openai_api_key = ""
        with self.assertRaises(RuntimeError):
            EchoAgent(self.root, settings)._resolve_backend("deep", mode="ask")


if __name__ == "__main__":
    unittest.main()
