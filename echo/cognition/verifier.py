from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re

from echo.types import GroundingReport, ToolCallRecord
from .validation import detect_validation_plan


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
SYMBOL_CLAIM_PATTERN = re.compile(r"\b(?:función|funcion|clase|método|metodo|símbolo|simbolo)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
BACKTICK_SYMBOL_PATTERN = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
CHANGE_VERB_PATTERN = re.compile(r"\b(cambi[ée]|modifiqu[ée]|actualic[ée]|edit[ée]|ajust[ée]|a[ñn]ad[íi])\b", re.IGNORECASE)
SUCCESS_PATTERNS = ["sin errores", "todo correcto", "todo pasó", "validado correctamente", "tests pasaron", "pasó la validación"]
VALIDATION_PATTERNS = ["pytest", "unittest", "npm test", "pnpm test", "yarn test", "python -m unittest", "python3 -m unittest"]
COMMAND_CLAIM_PATTERN = re.compile(r"`([^`\n]+)`")
VALIDATION_EXECUTION_PATTERN = re.compile(
    r"\b(?:ejecut[ée]|ejecute|corr[íi]|corri|valid[ée]|valide|pasó|pasaron)\b.{0,50}\b(?:pytest|unittest|npm test|pnpm test|yarn test)\b",
    re.IGNORECASE,
)


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


def _safe_json_loads(text: str) -> dict[str, object]:
    try:
        loaded = json.loads(text)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _command_prefix(command: str) -> str:
    parts = command.strip().split()
    if not parts:
        return ""
    return " ".join(parts[:3]).lower()


def _tool_evidence(tool_calls: list[ToolCallRecord] | None, tool_result_previews: list[str]) -> dict[str, object]:
    evidence = {
        "read_files": set(),
        "changed_files": set(),
        "executed_commands": set(),
        "validation_commands": set(),
        "successful_validation_commands": set(),
        "failed_validation_commands": set(),
        "tool_errors": [],
    }
    for call in tool_calls or []:
        path = str(call.arguments.get("path", "") or "")
        if call.tool in {"read_file", "read_file_range", "search_symbol", "find_symbol"} and path:
            evidence["read_files"].add(path)
        if call.tool in {"write_file", "apply_patch", "insert_before", "insert_after", "replace_range"} and path:
            evidence["changed_files"].add(path)
        payload = _safe_json_loads(call.result_preview)
        command = str(payload.get("command", "") or payload.get("validation_command", "") or "")
        if command:
            evidence["executed_commands"].add(command)
        if call.tool == "validate_project":
            validation_command = str(payload.get("validation_command", "") or command)
            if validation_command:
                evidence["validation_commands"].add(validation_command)
                if int(payload.get("returncode", 1) or 1) == 0:
                    evidence["successful_validation_commands"].add(validation_command)
                else:
                    evidence["failed_validation_commands"].add(validation_command)
        if payload.get("error"):
            evidence["tool_errors"].append(str(payload["error"]))
    evidence["symbols"] = _collect_symbols(tool_result_previews)
    return evidence


def _extract_claimed_files(text: str) -> set[str]:
    return set(FILE_PATTERN.findall(text or ""))


def _extract_claimed_symbols(text: str) -> set[str]:
    symbols = {match.group(1) for match in SYMBOL_CLAIM_PATTERN.finditer(text or "")}
    symbols.update(match.group(1) for match in BACKTICK_SYMBOL_PATTERN.finditer(text or "") if len(match.group(1)) >= 3)
    return {item for item in symbols if item}


def _extract_command_claims(text: str) -> set[str]:
    commands = {match.group(1).strip() for match in COMMAND_CLAIM_PATTERN.finditer(text or "") if " " in match.group(1).strip()}
    return {item for item in commands if item}


def detect_validation_strategy(project_root: Path | None = None, project_files: list[str] | None = None, validation_commands: list[str] | None = None) -> str:
    if project_root is not None:
        return detect_validation_plan(project_root).strategy
    commands = " ".join(validation_commands or []).lower()
    files = " ".join(project_files or []).lower()
    if "python -m pytest" in commands or "python3 -m pytest" in commands or "pytest" in commands:
        return "pytest"
    if "python -m unittest" in commands or "python3 -m unittest" in commands or "unittest" in commands:
        return "unittest"
    if "npm test" in commands:
        return "npm-test"
    if "pnpm test" in commands:
        return "pnpm-test"
    if "yarn test" in commands:
        return "yarn-test"
    if "npm run lint" in commands:
        return "npm-run-lint"
    if "pnpm run lint" in commands:
        return "pnpm-run-lint"
    if "yarn lint" in commands:
        return "yarn-lint"
    if "npm run typecheck" in commands:
        return "npm-run-typecheck"
    if "pnpm run typecheck" in commands:
        return "pnpm-run-typecheck"
    if "compileall" in commands or any(item.endswith(".py") for item in files.split()):
        return "compileall" if commands else "unknown"
    return "unknown"


def evaluate_final_answer(
    text: str,
    *,
    profile: str = "local",
    mode: str = "ask",
    inspected_files: list[str] | None = None,
    changed_files: list[str] | None = None,
    tool_calls: list[ToolCallRecord] | None = None,
    tool_result_previews: list[str] | None = None,
    working_set: list[str] | None = None,
    validation_strategy: str = "unknown",
) -> GroundingReport:
    inspected_files = inspected_files or []
    changed_files = changed_files or []
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

    evidence = _tool_evidence(tool_calls, tool_result_previews)
    evidence_files = set(inspected_files) | set(changed_files) | set(evidence["read_files"]) | set(evidence["changed_files"])
    mentioned_files = {path for path in evidence_files if path and path.lower() in low}
    if not mentioned_files:
        extracted = _extract_claimed_files(text)
        mentioned_files = {item for item in extracted if item in evidence_files}
        report.unsupported_files = sorted(item for item in extracted if item not in evidence_files)
    report.grounded_file_count = len(mentioned_files)

    evidence_usage = 0
    for item in tool_result_previews[-10:]:
        snippet = (item or "")[:180].lower()
        tokens = [token for token in re.findall(r"[a-z0-9_./:-]{4,}", snippet) if token not in {"path", "content", "matches", "symbol"}]
        if snippet and any(token in low for token in tokens[:10]):
            evidence_usage += 1
    report.evidence_usage = evidence_usage

    evidence_symbols = set(evidence["symbols"])
    claimed_symbols = _extract_claimed_symbols(text)
    mentioned_symbols = {symbol for symbol in evidence_symbols if symbol.lower() in low} | {symbol for symbol in claimed_symbols if symbol in evidence_symbols}
    report.grounded_symbol_count = len(mentioned_symbols)
    report.unsupported_symbols = sorted(symbol for symbol in claimed_symbols if symbol not in evidence_symbols)

    executed_command_prefixes = {_command_prefix(item) for item in evidence["executed_commands"] if item}
    claimed_commands = _extract_command_claims(text)
    unsupported_commands: set[str] = set()
    for command in claimed_commands:
        prefix = _command_prefix(command)
        if prefix and prefix not in executed_command_prefixes and command.lower() not in low:
            continue
        if prefix and prefix not in executed_command_prefixes:
            unsupported_commands.add(command)
    report.unsupported_commands = sorted(unsupported_commands)

    change_claimed = bool(CHANGE_VERB_PATTERN.search(text or ""))
    if change_claimed:
        report.claim_types.append("change")
        if report.unsupported_files:
            report.unsupported_changes = list(report.unsupported_files)
        elif not (changed_files or evidence["changed_files"]):
            report.unsupported_changes = ["no_changed_files"]

    if mentioned_files:
        report.claim_types.append("file")
    if claimed_symbols:
        report.claim_types.append("symbol")
    if claimed_commands:
        report.claim_types.append("command")
    validation_claimed = bool(VALIDATION_EXECUTION_PATTERN.search(text or "")) or any(
        command.lower() in low and any(token in low for token in ["ejecut", "corr", "valid", "pasó", "pasaron"])
        for command in claimed_commands
    )
    if validation_claimed:
        report.claim_types.append("validation")
        if not evidence["validation_commands"]:
            report.unsupported_commands = sorted(set(report.unsupported_commands + ["validation-not-executed"]))
        elif evidence["failed_validation_commands"] and any(token in low for token in ["validado", "pasó", "sin errores", "todo correcto"]):
            report.contradiction_flags.append("validation-claimed-success-but-failed")

    if any(token in low for token in SUCCESS_PATTERNS) and evidence["tool_errors"]:
        report.contradiction_flags.append("success-claimed-with-tool-errors")
    if report.unsupported_files:
        report.claim_types.append("file")
    if report.unsupported_symbols:
        report.claim_types.append("symbol")

    genericity_score = 0
    if len((text or "").strip()) < 60:
        genericity_score += 2
    for pattern in GENERIC_PATTERNS:
        if pattern in low:
            genericity_score += 2
    if report.grounded_file_count == 0:
        genericity_score += 3
    if evidence_symbols and report.grounded_symbol_count == 0:
        genericity_score += 3
    if report.evidence_usage == 0 and tool_result_previews:
        genericity_score += 3
    if report.grounded_file_count and report.grounded_symbol_count == 0 and len(evidence_symbols) >= 2:
        genericity_score += 2
    if mode == "ask" and report.claim_types == ["file"]:
        genericity_score += 2
    report.genericity_score = genericity_score

    speculation = [item for item in SPECULATION_PATTERNS if item in low]
    report.speculation_flags = speculation

    strategy_match = True
    strategy_markers = {
        "pytest": ["pytest"],
        "unittest": ["unittest"],
        "compileall": ["compileall"],
        "npm-test": ["npm test"],
        "pnpm-test": ["pnpm test"],
        "yarn-test": ["yarn test"],
        "npm-run-lint": ["npm run lint"],
        "pnpm-run-lint": ["pnpm run lint"],
        "yarn-lint": ["yarn lint"],
        "npm-run-typecheck": ["npm run typecheck"],
        "pnpm-run-typecheck": ["pnpm run typecheck"],
    }
    mentioned_strategies = {name for name, markers in strategy_markers.items() if any(marker in low for marker in markers)}
    if validation_strategy != "unknown" and mentioned_strategies and validation_strategy not in mentioned_strategies:
        strategy_match = False
    if validation_strategy == "unknown" and validation_claimed and not evidence["validation_commands"]:
        strategy_match = False
    report.validation_strategy_match = strategy_match

    report.useful = bool(
        report.grounded_symbol_count
        or evidence_usage
        or (validation_claimed and evidence["validation_commands"])
        or (change_claimed and not report.unsupported_changes)
    )

    if mode == "plan":
        required = ["objetivo", "archivos a revisar", "riesgos", "siguientes pasos"]
        missing = [item for item in required if item not in low]
        if missing:
            report.valid = False
            report.reason = f"Plan inválido: faltan secciones {', '.join(missing)}."
            return report

    if report.unsupported_files:
        report.valid = False
        report.reason = f"Respuesta no grounded: cita archivos no inspeccionados: {', '.join(report.unsupported_files[:4])}."
    elif report.unsupported_symbols:
        report.valid = False
        report.reason = f"Respuesta no grounded: cita símbolos sin evidencia: {', '.join(report.unsupported_symbols[:4])}."
    elif report.unsupported_commands:
        report.valid = False
        report.reason = f"Respuesta no grounded: afirma comandos o validaciones no ejecutados: {', '.join(report.unsupported_commands[:4])}."
    elif report.unsupported_changes:
        report.valid = False
        report.reason = "Respuesta no grounded: afirma cambios que no ocurrieron."
    elif report.contradiction_flags:
        report.valid = False
        report.reason = f"Respuesta contradictoria con evidencia real: {', '.join(report.contradiction_flags[:4])}."
    elif tool_result_previews and report.grounded_file_count == 0:
        report.valid = False
        report.reason = "Respuesta no grounded: no menciona archivos inspeccionados."
    elif evidence_symbols and report.grounded_symbol_count == 0 and mode == "ask":
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
    elif not report.useful and mode == "ask":
        report.valid = False
        report.reason = "Respuesta insuficiente: no concluye nada verificable ni accionable."
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
    changed_files: list[str] | None = None,
    tool_calls: list[ToolCallRecord] | None = None,
    tool_result_previews: list[str] | None = None,
    working_set: list[str] | None = None,
    validation_strategy: str = "unknown",
) -> tuple[bool, str, dict[str, object]]:
    report = evaluate_final_answer(
        text,
        profile=profile,
        mode=mode,
        inspected_files=inspected_files,
        changed_files=changed_files,
        tool_calls=tool_calls,
        tool_result_previews=tool_result_previews,
        working_set=working_set,
        validation_strategy=validation_strategy,
    )
    return report.valid, report.reason, asdict(report)
