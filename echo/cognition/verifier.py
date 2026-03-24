from __future__ import annotations

from dataclasses import asdict
import re

from echo.types import GroundingReport


GENERIC_PATTERNS = [
    "aquí tienes",
    "aqui tienes",
    "archivos adicionales relevantes",
    "esto te permitirá",
    "puedes usar el siguiente comando",
    "incluye métodos para",
    "contiene la lógica principal",
    "implementa métodos adicionales",
    "simula diferentes escenarios",
    "maneja la persistencia",
    "en general",
    "a grandes rasgos",
]
SPECULATION_PATTERNS = ["probablemente", "parece", "podría", "quizá", "tal vez"]
FILE_PATTERN = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
SYMBOL_PATTERN = re.compile(r"\b(?:class|def|async def)\s+([A-Za-z_][A-Za-z0-9_]*)|\b([A-Z][A-Za-z0-9_]{2,}|[a-z_][A-Za-z0-9_]{2,})\s*\(")


def _collect_symbols(tool_result_previews: list[str]) -> set[str]:
    symbols: set[str] = set()
    for preview in tool_result_previews:
        for match in SYMBOL_PATTERN.finditer(preview or ""):
            symbol = match.group(1) or match.group(2)
            if symbol and len(symbol) >= 3:
                symbols.add(symbol)
    return symbols


def _normalize(text: str) -> str:
    return (text or "").lower()


def detect_validation_strategy(project_files: list[str], validation_commands: list[str]) -> str:
    commands = " ".join(validation_commands).lower()
    files = " ".join(project_files).lower()
    if "unittest" in commands or "tests/test_" in files:
        return "unittest"
    if "pytest" in commands or "pytest" in files:
        return "pytest"
    return "unknown"


def evaluate_final_answer(
    text: str,
    *,
    profile: str = "local",
    mode: str = "ask",
    inspected_files: list[str] | None = None,
    tool_result_previews: list[str] | None = None,
    working_set: list[str] | None = None,
    validation_strategy: str = "unknown",
) -> GroundingReport:
    inspected_files = inspected_files or []
    tool_result_previews = tool_result_previews or []
    working_set = working_set or []
    low = _normalize(text)
    validation_strategy = validation_strategy or "unknown"
    report = GroundingReport(validation_strategy=validation_strategy)

    banned = [
        "soy claude",
        "anthropic",
        "no tengo acceso a archivos",
        "no tengo la capacidad de inspeccionar",
    ]
    for item in banned:
        if item in low:
            report.valid = False
            report.reason = f"Respuesta inválida por patrón: {item}"
            return report
    if "tool_call" in low or '"function"' in low:
        report.valid = False
        report.reason = "Respuesta inválida: expone pseudo tool calls."
        return report

    available_files = set(inspected_files + working_set)
    mentioned_files = {
        path for path in set(inspected_files + working_set)
        if path and path.lower() in low
    }
    if not mentioned_files:
        extracted = set(FILE_PATTERN.findall(text or ""))
        mentioned_files = {item for item in extracted if item in available_files}
    report.grounded_file_count = len(mentioned_files)

    evidence_usage = 0
    for item in tool_result_previews[-10:]:
        snippet = (item or "")[:180].lower()
        tokens = [token for token in re.findall(r"[a-z0-9_./:-]{4,}", snippet) if token not in {"path", "content", "matches", "symbol"}]
        if snippet and any(token in low for token in tokens[:10]):
            evidence_usage += 1
    report.evidence_usage = evidence_usage
    symbols = _collect_symbols(tool_result_previews)
    mentioned_symbols = {symbol for symbol in symbols if symbol.lower() in low}
    report.grounded_symbol_count = len(mentioned_symbols)

    genericity_score = 0
    if len((text or "").strip()) < 60:
        genericity_score += 2
    for pattern in GENERIC_PATTERNS:
        if pattern in low:
            genericity_score += 2
    if report.grounded_file_count == 0:
        genericity_score += 3
    if symbols and report.grounded_symbol_count == 0:
        genericity_score += 3
    if report.evidence_usage == 0 and tool_result_previews:
        genericity_score += 3
    if report.grounded_file_count and report.grounded_symbol_count == 0 and len(symbols) >= 2:
        genericity_score += 2
    report.genericity_score = genericity_score

    speculation = [item for item in SPECULATION_PATTERNS if item in low]
    report.speculation_flags = speculation

    strategy_match = True
    if validation_strategy == "unittest" and "pytest" in low:
        strategy_match = False
    if validation_strategy == "pytest" and "unittest" in low:
        strategy_match = False
    report.validation_strategy_match = strategy_match

    if mode == "plan":
        required = ["objetivo", "archivos a revisar", "riesgos", "siguientes pasos"]
        missing = [item for item in required if item not in low]
        if missing:
            report.valid = False
            report.reason = f"Plan inválido: faltan secciones {', '.join(missing)}."
            return report

    if tool_result_previews and report.grounded_file_count == 0:
        report.valid = False
        report.reason = "Respuesta no grounded: no menciona archivos inspeccionados."
    elif symbols and report.grounded_symbol_count == 0 and mode == "ask":
        report.valid = False
        report.reason = "Respuesta no grounded: no menciona símbolos concretos vistos en la inspección."
    elif tool_result_previews and report.evidence_usage == 0:
        report.valid = False
        report.reason = "Respuesta no grounded: no reutiliza evidencia de herramientas."
    elif not report.validation_strategy_match:
        report.valid = False
        report.reason = f"Estrategia de validación inconsistente con el repo: {validation_strategy}."
    elif speculation and tool_result_previews:
        report.valid = False
        report.reason = f"Respuesta especulativa sin apoyo suficiente: {', '.join(speculation)}."
    elif genericity_score >= 5:
        report.valid = False
        report.reason = "Respuesta demasiado genérica para el contexto inspeccionado."
    elif profile in {"balanced", "deep"} and len((text or "").strip()) < 30:
        report.valid = False
        report.reason = "Respuesta demasiado breve para el perfil actual."
    return report


def validate_final_answer(
    text: str,
    *,
    profile: str = "local",
    mode: str = "ask",
    inspected_files: list[str] | None = None,
    tool_result_previews: list[str] | None = None,
    working_set: list[str] | None = None,
    validation_strategy: str = "unknown",
) -> tuple[bool, str, dict[str, object]]:
    report = evaluate_final_answer(
        text,
        profile=profile,
        mode=mode,
        inspected_files=inspected_files,
        tool_result_previews=tool_result_previews,
        working_set=working_set,
        validation_strategy=validation_strategy,
    )
    return report.valid, report.reason, asdict(report)
