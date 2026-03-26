from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import re

from echo.types import GroundingReport, ToolCallRecord
from .validation import detect_validation_plan, infer_validation_strategy_from_evidence
from .verifier_support import (
    VerifierClaims,
    VerifierDecision,
    VerifierEvidence,
    collect_tool_evidence,
    command_prefix,
    compute_evidence_usage,
    compute_genericity_score,
    extract_claims,
    normalize_text,
    synthesize_verifier_decision,
)


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
CHANGE_VERB_PATTERN = re.compile(r"\b(cambi[ée]|modifiqu[ée]|actualic[ée]|edit[ée]|ajust[ée]|a[ñn]ad[íi])\b", re.IGNORECASE)
SUCCESS_PATTERNS = ["sin errores", "todo correcto", "todo pasó", "validado correctamente", "tests pasaron", "pasó la validación"]
VALIDATION_EXECUTION_PATTERN = re.compile(
    r"\b(?:ejecut[ée]|ejecute|corr[íi]|corri|valid[ée]|valide|pasó|pasaron)\b.{0,50}\b(?:pytest|unittest|npm test|pnpm test|yarn test)\b",
    re.IGNORECASE,
)

def detect_validation_strategy(project_root: Path | None = None, project_files: list[str] | None = None, validation_commands: list[str] | None = None) -> str:
    if project_root is not None:
        return detect_validation_plan(project_root).strategy
    return infer_validation_strategy_from_evidence(project_files=project_files, validation_commands=validation_commands)


def _banned_pattern_reason(low: str) -> str | None:
    banned = [
        "soy claude",
        "anthropic",
        "no tengo acceso a archivos",
        "no tengo la capacidad de inspeccionar",
    ]
    for item in banned:
        if item in low:
            return f"Respuesta inválida por patrón: {item}"
    if "tool_call" in low or '"function"' in low:
        return "Respuesta inválida: expone pseudo tool calls."
    return None


def _apply_file_checks(report: GroundingReport, *, low: str, claims: VerifierClaims, evidence: VerifierEvidence, inspected_files: list[str], changed_files: list[str]) -> None:
    evidence_files = set(inspected_files) | set(changed_files) | evidence.read_files | evidence.changed_files
    mentioned_files = {path for path in evidence_files if path and path.lower() in low}
    if not mentioned_files:
        mentioned_files = {item for item in claims.files if item in evidence_files}
        report.unsupported_files = sorted(item for item in claims.files if item not in evidence_files)
    report.grounded_file_count = len(mentioned_files)


def _apply_symbol_checks(report: GroundingReport, *, low: str, claims: VerifierClaims, evidence: VerifierEvidence) -> None:
    mentioned_symbols = {symbol for symbol in evidence.symbols if symbol.lower() in low} | {symbol for symbol in claims.symbols if symbol in evidence.symbols}
    report.grounded_symbol_count = len(mentioned_symbols)
    report.unsupported_symbols = sorted(symbol for symbol in claims.symbols if symbol not in evidence.symbols)


def _apply_command_checks(report: GroundingReport, *, claims: VerifierClaims, evidence: VerifierEvidence) -> None:
    executed_command_prefixes = {command_prefix(item) for item in evidence.executed_commands if item}
    unsupported_commands: set[str] = set()
    for command in claims.commands:
        prefix = command_prefix(command)
        if prefix and prefix not in executed_command_prefixes:
            unsupported_commands.add(command)
    report.unsupported_commands = sorted(unsupported_commands)


def _apply_claim_type_flags(report: GroundingReport, *, claims: VerifierClaims, change_claimed: bool, validation_claimed: bool) -> None:
    if change_claimed:
        report.claim_types.append("change")
    if report.grounded_file_count or report.unsupported_files:
        report.claim_types.append("file")
    if claims.symbols or report.unsupported_symbols:
        report.claim_types.append("symbol")
    if claims.commands:
        report.claim_types.append("command")
    if validation_claimed:
        report.claim_types.append("validation")


def _apply_change_checks(report: GroundingReport, *, change_claimed: bool, changed_files: list[str], evidence: VerifierEvidence) -> None:
    if not change_claimed:
        return
    if report.unsupported_files:
        report.unsupported_changes = list(report.unsupported_files)
    elif not (changed_files or evidence.changed_files):
        report.unsupported_changes = ["no_changed_files"]


def _apply_validation_checks(report: GroundingReport, *, low: str, validation_claimed: bool, validation_strategy: str, evidence: VerifierEvidence) -> None:
    if validation_claimed:
        if not evidence.validation_commands:
            report.unsupported_commands = sorted(set(report.unsupported_commands + ["validation-not-executed"]))
        elif evidence.failed_validation_commands and any(token in low for token in ["validado", "pasó", "sin errores", "todo correcto"]):
            report.contradiction_flags.append("validation-claimed-success-but-failed")

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
        "yarn-typecheck": ["yarn typecheck"],
    }
    mentioned_strategies = {name for name, markers in strategy_markers.items() if any(marker in low for marker in markers)}
    strategy_match = True
    if validation_strategy != "unknown" and mentioned_strategies and validation_strategy not in mentioned_strategies:
        strategy_match = False
    if validation_strategy == "unknown" and validation_claimed and not evidence.validation_commands:
        strategy_match = False
    report.validation_strategy_match = strategy_match


def _apply_genericity_and_usefulness(
    report: GroundingReport,
    *,
    text: str,
    low: str,
    mode: str,
    tool_result_previews: list[str],
    evidence: VerifierEvidence,
    change_claimed: bool,
    validation_claimed: bool,
) -> None:
    report.evidence_usage = compute_evidence_usage(low, tool_result_previews)
    report.genericity_score = compute_genericity_score(
        text=text,
        low=low,
        grounded_file_count=report.grounded_file_count,
        grounded_symbol_count=report.grounded_symbol_count,
        evidence_symbols=evidence.symbols,
        generic_patterns=GENERIC_PATTERNS,
    )
    if report.evidence_usage == 0 and tool_result_previews:
        report.genericity_score += 3
    if report.grounded_file_count and report.grounded_symbol_count == 0 and len(evidence.symbols) >= 2:
        report.genericity_score += 2
    if mode == "ask" and report.claim_types == ["file"]:
        report.genericity_score += 2
    report.speculation_flags = [item for item in SPECULATION_PATTERNS if item in low]
    report.useful = bool(
        report.grounded_symbol_count
        or report.evidence_usage
        or (validation_claimed and evidence.validation_commands)
        or (change_claimed and not report.unsupported_changes)
    )


def _apply_verifier_decision(report: GroundingReport, decision: VerifierDecision) -> GroundingReport:
    report.valid = decision.valid
    report.reason = decision.reason
    return report


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
    low = normalize_text(text)
    validation_strategy = validation_strategy or "unknown"
    report = GroundingReport(validation_strategy=validation_strategy)

    banned_reason = _banned_pattern_reason(low)
    if banned_reason is not None:
        report.valid = False
        report.reason = banned_reason
        return report

    evidence = collect_tool_evidence(tool_calls, tool_result_previews)
    claims = extract_claims(text)
    _apply_file_checks(report, low=low, claims=claims, evidence=evidence, inspected_files=inspected_files, changed_files=changed_files)
    _apply_symbol_checks(report, low=low, claims=claims, evidence=evidence)
    _apply_command_checks(report, claims=claims, evidence=evidence)

    change_claimed = bool(CHANGE_VERB_PATTERN.search(text or ""))
    validation_claimed = bool(VALIDATION_EXECUTION_PATTERN.search(text or "")) or any(
        command.lower() in low and any(token in low for token in ["ejecut", "corr", "valid", "pasó", "pasaron"])
        for command in claims.commands
    )
    _apply_claim_type_flags(report, claims=claims, change_claimed=change_claimed, validation_claimed=validation_claimed)
    _apply_change_checks(report, change_claimed=change_claimed, changed_files=changed_files, evidence=evidence)
    _apply_validation_checks(report, low=low, validation_claimed=validation_claimed, validation_strategy=validation_strategy, evidence=evidence)

    if any(token in low for token in SUCCESS_PATTERNS) and evidence.tool_errors:
        report.contradiction_flags.append("success-claimed-with-tool-errors")
    _apply_genericity_and_usefulness(
        report,
        text=text,
        low=low,
        mode=mode,
        tool_result_previews=tool_result_previews,
        evidence=evidence,
        change_claimed=change_claimed,
        validation_claimed=validation_claimed,
    )
    return _apply_verifier_decision(
        report,
        synthesize_verifier_decision(
            report,
            profile=profile,
            mode=mode,
            text=text,
            tool_result_previews=tool_result_previews,
            evidence=evidence,
        ),
    )


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
