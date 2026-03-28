from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

from echo.types import GroundingReport, ToolCallRecord


SYMBOL_PATTERN = re.compile(r"\b(?:class|def|async def)\s+([A-Za-z_][A-Za-z0-9_]*)|\b([A-Z][A-Za-z0-9_]{2,}|[a-z_][A-Za-z0-9_]{2,})\s*\(")
ENV_SYMBOL_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
FIELD_SYMBOL_PATTERN = re.compile(r"\b([a-z_][A-Za-z0-9_]*)\s*:")
FILE_PATTERN = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
SYMBOL_CLAIM_PATTERN = re.compile(r"\b(?:función|funcion|clase|método|metodo|símbolo|simbolo)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
BACKTICK_SYMBOL_PATTERN = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
COMMAND_CLAIM_PATTERN = re.compile(r"`([^`\n]+)`")

# Bug A: words that follow "función/método/clase X" in natural Spanish prose
# but are NOT function/class names.  Minimum-length guard (≥4) plus this set.
NON_SYMBOL_WORDS = {
    "concreto", "real", "principal", "correcto", "breve",
    # conjunctions / prepositions frequently captured as "función que …"
    "que", "del", "para", "con", "una", "los", "las", "por",
    # Spanish verbs/adjectives that describe a function without naming it
    "devuelve", "retorna", "recibe", "acepta", "permite", "usa",
    "llamada", "llamado", "definida", "definido", "declarada",
}

# Bug B: Python built-ins and keywords that appear legitimately in backticks
# but are NOT project-specific symbols and will never be in evidence.symbols.
_PYTHON_BUILTINS: frozenset[str] = frozenset({
    "True", "False", "None",
    "self", "cls", "args", "kwargs",
    "classmethod", "staticmethod", "property",
    "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "type", "object",
})


@dataclass(slots=True)
class VerifierEvidence:
    read_files: set[str] = field(default_factory=set)
    changed_files: set[str] = field(default_factory=set)
    executed_commands: set[str] = field(default_factory=set)
    validation_commands: set[str] = field(default_factory=set)
    successful_validation_commands: set[str] = field(default_factory=set)
    failed_validation_commands: set[str] = field(default_factory=set)
    tool_errors: list[str] = field(default_factory=list)
    symbols: set[str] = field(default_factory=set)


@dataclass(slots=True)
class VerifierClaims:
    files: set[str] = field(default_factory=set)
    symbols: set[str] = field(default_factory=set)
    commands: set[str] = field(default_factory=set)


@dataclass(slots=True)
class VerifierDecision:
    valid: bool
    reason: str


def normalize_text(text: str) -> str:
    return (text or "").lower()


def safe_json_loads(text: str) -> dict[str, object]:
    try:
        loaded = json.loads(text)
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def command_prefix(command: str) -> str:
    parts = command.strip().split()
    if not parts:
        return ""
    return " ".join(parts[:3]).lower()


def collect_symbols(tool_result_previews: list[str]) -> set[str]:
    symbols: set[str] = set()
    for preview in tool_result_previews:
        payload = safe_json_loads(preview or "")
        text_sources = [preview or ""]
        content = payload.get("content")
        if isinstance(content, str):
            text_sources.append(content)
        matches = payload.get("matches")
        if isinstance(matches, list):
            for match in matches[:8]:
                if isinstance(match, dict):
                    text_sources.extend(str(match.get(key, "")) for key in ("symbol", "content", "line"))
        for source in text_sources:
            for match in SYMBOL_PATTERN.finditer(source):
                symbol = match.group(1) or match.group(2)
                if symbol and len(symbol) >= 3:
                    symbols.add(symbol)
            for symbol in ENV_SYMBOL_PATTERN.findall(source):
                if len(symbol) >= 3:
                    symbols.add(symbol)
            for symbol in FIELD_SYMBOL_PATTERN.findall(source):
                if len(symbol) >= 3:
                    symbols.add(symbol)
    return symbols


def collect_tool_evidence(tool_calls: list[ToolCallRecord] | None, tool_result_previews: list[str]) -> VerifierEvidence:
    evidence = VerifierEvidence(symbols=collect_symbols(tool_result_previews))
    for call in tool_calls or []:
        path = str(call.arguments.get("path", "") or "")
        if call.tool in {"read_file", "read_file_range", "search_symbol", "find_symbol"} and path:
            evidence.read_files.add(path)
        if call.tool in {"write_file", "apply_patch", "insert_before", "insert_after", "replace_range"} and path:
            evidence.changed_files.add(path)
        payload = safe_json_loads(call.result_preview)
        command = str(payload.get("command", "") or payload.get("validation_command", "") or "")
        if command:
            evidence.executed_commands.add(command)
        if call.tool == "validate_project":
            validation_command = str(payload.get("validation_command", "") or command)
            if validation_command:
                evidence.validation_commands.add(validation_command)
                if int(payload.get("returncode", 1) or 1) == 0:
                    evidence.successful_validation_commands.add(validation_command)
                else:
                    evidence.failed_validation_commands.add(validation_command)
        if payload.get("error"):
            evidence.tool_errors.append(str(payload["error"]))
    return evidence


def extract_claims(text: str) -> VerifierClaims:
    files = set(FILE_PATTERN.findall(text or ""))

    # Bug A fix: require min length ≥4 so short Spanish words ("que", "es",
    # "bas") from "función X" pattern are dropped before the NON_SYMBOL_WORDS
    # filter even runs.
    symbols = {
        match.group(1)
        for match in SYMBOL_CLAIM_PATTERN.finditer(text or "")
        if len(match.group(1)) >= 4
    }

    # Bug B fix: exclude Python built-ins/keywords from backtick symbols; they
    # will never appear in evidence.symbols and cause false grounding failures.
    symbols.update(
        match.group(1)
        for match in BACKTICK_SYMBOL_PATTERN.finditer(text or "")
        if len(match.group(1)) >= 3
        and match.group(1) not in _PYTHON_BUILTINS
        and ("_" in match.group(1) or any(ch.isupper() for ch in match.group(1)))
    )
    commands = {match.group(1).strip() for match in COMMAND_CLAIM_PATTERN.finditer(text or "") if " " in match.group(1).strip()}
    return VerifierClaims(
        files=files,
        symbols={item for item in symbols if item and item.lower() not in NON_SYMBOL_WORDS},
        commands={item for item in commands if item},
    )


def compute_evidence_usage(low: str, tool_result_previews: list[str]) -> int:
    evidence_usage = 0
    for item in tool_result_previews[-10:]:
        snippet = (item or "")[:180].lower()
        tokens = [token for token in re.findall(r"[a-z0-9_./:-]{4,}", snippet) if token not in {"path", "content", "matches", "symbol"}]
        if snippet and any(token in low for token in tokens[:10]):
            evidence_usage += 1
    return evidence_usage


def compute_genericity_score(*, text: str, low: str, grounded_file_count: int, grounded_symbol_count: int, evidence_symbols: set[str], generic_patterns: list[str]) -> int:
    genericity_score = 0
    if len((text or "").strip()) < 60:
        genericity_score += 2
    for pattern in generic_patterns:
        if pattern in low:
            genericity_score += 2
    if grounded_file_count == 0:
        genericity_score += 3
    if evidence_symbols and grounded_symbol_count == 0:
        genericity_score += 3
    return genericity_score


def synthesize_verifier_decision(
    report: GroundingReport,
    *,
    profile: str,
    mode: str,
    text: str,
    tool_result_previews: list[str],
    evidence: VerifierEvidence,
) -> VerifierDecision:
    if mode == "plan":
        low = normalize_text(text)
        required = ["objetivo", "archivos a revisar", "riesgos", "siguientes pasos"]
        missing = [item for item in required if item not in low]
        if missing:
            return VerifierDecision(False, f"Plan inválido: faltan secciones {', '.join(missing)}.")

    if report.unsupported_files:
        return VerifierDecision(False, f"Respuesta no grounded: cita archivos no inspeccionados: {', '.join(report.unsupported_files[:4])}.")
    if report.unsupported_symbols:
        return VerifierDecision(False, f"Respuesta no grounded: cita símbolos sin evidencia: {', '.join(report.unsupported_symbols[:4])}.")
    if report.unsupported_commands:
        return VerifierDecision(False, f"Respuesta no grounded: afirma comandos o validaciones no ejecutados: {', '.join(report.unsupported_commands[:4])}.")
    if report.unsupported_changes:
        return VerifierDecision(False, "Respuesta no grounded: afirma cambios que no ocurrieron.")
    if report.contradiction_flags:
        return VerifierDecision(False, f"Respuesta contradictoria con evidencia real: {', '.join(report.contradiction_flags[:4])}.")
    if tool_result_previews and report.grounded_file_count == 0:
        return VerifierDecision(False, "Respuesta no grounded: no menciona archivos inspeccionados.")
    if evidence.symbols and report.grounded_symbol_count == 0 and mode == "ask":
        return VerifierDecision(False, "Respuesta no grounded: no menciona símbolos concretos vistos en la inspección.")
    if tool_result_previews and report.evidence_usage == 0:
        return VerifierDecision(False, "Respuesta no grounded: no reutiliza evidencia de herramientas.")
    if not report.validation_strategy_match:
        return VerifierDecision(False, f"Estrategia de validación inconsistente con el repo: {report.validation_strategy}.")
    if report.speculation_flags and tool_result_previews:
        return VerifierDecision(False, f"Respuesta especulativa sin apoyo suficiente: {', '.join(report.speculation_flags)}.")
    if not report.useful and mode == "ask":
        return VerifierDecision(False, "Respuesta insuficiente: no concluye nada verificable ni accionable.")
    if report.genericity_score >= 5:
        return VerifierDecision(False, "Respuesta demasiado genérica para el contexto inspeccionado.")
    if profile in {"balanced", "deep"} and len((text or "").strip()) < 30:
        return VerifierDecision(False, "Respuesta demasiado breve para el perfil actual.")
    return VerifierDecision(True, report.reason or "ok")
