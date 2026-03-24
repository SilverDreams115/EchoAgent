from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from echo.backends.errors import BackendMalformedResponseError, BackendTimeoutError, BackendUnreachableError
from echo.backends.availability import BackendAvailabilityPolicy, run_backend_check
from echo.cognition.verifier import evaluate_final_answer
from echo.config import Settings
from echo.core import EchoAgent
from echo.memory import EchoStore
from echo.runtime import ActivityBus, AgentRuntime
from echo.tools import ToolRegistry


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
        (self.root / "echo").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "echo" / "sample.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
        (self.root / "README.md").write_text("# Demo\n", encoding="utf-8")
        (self.root / "tests" / "test_sample.py").write_text("import unittest\n", encoding="utf-8")
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
        self.assertTrue(run_state.fallback_used)
        self.assertTrue(resumed_session.degraded_reason)

    def test_plan_falls_back_when_chat_unhealthy(self) -> None:
        backend = UnreachableBackend([])
        runtime, store = self._runtime(backend)
        self._mark_backend_unstable(store, event="error")
        answer, _, session, run_state = runtime.run("crea un plan", mode="plan")
        self.assertIn("Objetivo", answer)
        self.assertIn("Archivos a revisar", answer)
        self.assertGreater(len(session.tool_calls), 0)
        self.assertTrue(run_state.fallback_used)

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

    def test_strict_deep_profile_still_requires_frontier_backend(self) -> None:
        settings = Settings.from_env()
        settings.profile = "deep"
        settings.strict_profile = True
        settings.openai_api_key = ""
        with self.assertRaises(RuntimeError):
            EchoAgent(self.root, settings)._resolve_backend("deep", mode="ask")


if __name__ == "__main__":
    unittest.main()
