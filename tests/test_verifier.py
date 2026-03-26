from __future__ import annotations

from echo.cognition.verifier import evaluate_final_answer
from echo.context.selector import select_relevant_files
from tests.runtime_fixtures import RuntimeTestCase


class VerifierTests(RuntimeTestCase):
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
        self.assertFalse(report.validation_strategy_match)

    def test_answer_validator_requires_symbols_when_evidence_contains_them(self) -> None:
        report = evaluate_final_answer(
            "Inspeccioné echo/config.py y confirmé que todo está bien.",
            inspected_files=["echo/config.py"],
            tool_result_previews=['{"path":"echo/config.py","content":"class Settings:\\n    backend: str = \\"ollama\\"\\n    model: str = \\"qwen\\""}'],
            working_set=["echo/config.py"],
            validation_strategy="unknown",
        )
        self.assertFalse(report.valid)
        self.assertEqual(report.grounded_symbol_count, 0)

    def test_answer_validator_reports_missing_inspected_file_reason_stably(self) -> None:
        report = evaluate_final_answer(
            "El proyecto está bien organizado y contiene varios módulos.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path": "echo/sample.py", "content": "def run()"}'],
            working_set=["echo/sample.py"],
        )
        self.assertFalse(report.valid)
        self.assertIn("archivos inspeccionados", report.reason.lower())

    def test_answer_validator_rejects_unread_file_claim(self) -> None:
        report = evaluate_final_answer(
            "Leí echo/other.py y confirmé su comportamiento.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path": "echo/sample.py", "content": "def run()"}'],
            working_set=["echo/sample.py"],
        )
        self.assertFalse(report.valid)
        self.assertIn("echo/other.py", report.unsupported_files)

    def test_answer_validator_rejects_symbol_without_evidence(self) -> None:
        report = evaluate_final_answer(
            "Inspeccioné echo/sample.py y confirmé que existe la función bootstrap.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path": "echo/sample.py", "content": "def run()"}'],
            working_set=["echo/sample.py"],
        )
        self.assertFalse(report.valid)
        self.assertIn("bootstrap", report.unsupported_symbols)

    def test_answer_validator_does_not_treat_plain_backend_value_as_symbol(self) -> None:
        report = evaluate_final_answer(
            "Inspeccioné echo/config.py y confirmé que el backend por defecto es ollama.",
            inspected_files=["echo/config.py"],
            tool_result_previews=['{"path":"echo/config.py","content":"backend: str = \\"ollama\\""}'],
            working_set=["echo/config.py"],
        )
        self.assertTrue(report.valid)
        self.assertNotIn("ollama", report.unsupported_symbols)

    def test_answer_validator_accepts_env_symbols_when_present_in_evidence(self) -> None:
        report = evaluate_final_answer(
            "Inspeccioné echo/config.py y confirmé que from_env usa ECHO_BACKEND y ECHO_MODEL.",
            inspected_files=["echo/config.py"],
            tool_result_previews=['{"path":"echo/config.py","content":"def from_env():\\n    os.getenv(\\"ECHO_BACKEND\\")\\n    os.getenv(\\"ECHO_MODEL\\")"}'],
            working_set=["echo/config.py"],
        )
        self.assertTrue(report.valid)

    def test_answer_validator_accepts_field_identifier_when_present_in_evidence(self) -> None:
        report = evaluate_final_answer(
            "Inspeccioné echo/config.py y confirmé que Settings define backend y model.",
            inspected_files=["echo/config.py"],
            tool_result_previews=['{"path":"echo/config.py","content":"class Settings:\\n    backend: str = \\"ollama\\"\\n    model: str = \\"qwen\\""}'],
            working_set=["echo/config.py"],
        )
        self.assertTrue(report.valid)

    def test_select_relevant_files_prioritizes_explicit_path_with_punctuation(self) -> None:
        result = select_relevant_files(self.root, "Revisa (echo/sample.py), por favor.", limit=3)
        self.assertEqual(result[0], "echo/sample.py")

    def test_answer_validator_rejects_validation_claim_without_executed_command(self) -> None:
        report = evaluate_final_answer(
            "Inspeccioné echo/sample.py y ejecuté unittest sin errores.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run()"}'],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)

    def test_answer_validator_rejects_unknown_validation_claim_as_executed(self) -> None:
        report = evaluate_final_answer(
            "Validé el proyecto y todo pasó correctamente.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run()"}'],
            working_set=["echo/sample.py"],
            validation_strategy="unknown",
        )
        self.assertFalse(report.valid)

    def test_answer_validator_rejects_contradiction_with_failed_validation(self) -> None:
        report = evaluate_final_answer(
            "Ejecuté unittest y todo pasó sin errores.",
            inspected_files=["echo/sample.py"],
            tool_calls=[self._tool_call("validate_project", {}, {"validation_command": "python3 -m unittest", "returncode": 1, "stderr": "boom"})],
            tool_result_previews=['{"validation_command":"python3 -m unittest","returncode":1,"stderr":"boom"}'],
            working_set=["echo/sample.py"],
            validation_strategy="unittest",
        )
        self.assertFalse(report.valid)
        self.assertIn("validation-claimed-success-but-failed", report.contradiction_flags)

    def test_answer_validator_rejects_generic_answer_even_with_file_reference(self) -> None:
        report = evaluate_final_answer(
            "Inspeccioné echo/sample.py. A grandes rasgos, el proyecto está bien estructurado.",
            inspected_files=["echo/sample.py"],
            tool_result_previews=['{"path":"echo/sample.py","content":"def run()"}'],
            working_set=["echo/sample.py"],
        )
        self.assertFalse(report.valid)
