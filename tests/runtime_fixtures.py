from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from echo.backends.errors import BackendMalformedResponseError, BackendTimeoutError, BackendUnreachableError
from echo.config import Settings
from echo.memory import EchoStore
from echo.runtime import ActivityBus, AgentRuntime
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


class SlowTimeoutBackend(FakeBackend):
    def __init__(self, delay_seconds: float = 1.1) -> None:
        super().__init__([])
        self.delay_seconds = delay_seconds

    def chat(self, messages, tools=None):
        self.calls += 1
        time.sleep(self.delay_seconds)
        raise BackendTimeoutError("timeout", backend="fake", model="fake-model")


class RuntimeTestCase(unittest.TestCase):
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
        from echo.tools import ToolRegistry

        registry = ToolRegistry(self.root, self.settings, activity)
        return AgentRuntime(self.root, self.settings, backend, store, registry, activity), store

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
