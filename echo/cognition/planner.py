from __future__ import annotations

from dataclasses import asdict

from echo.types import PlanStage


def build_execution_plan(user_prompt: str, *, mode: str, focus_files: list[str] | None = None, validation_strategy: str = "unknown") -> list[PlanStage]:
    focus_files = list(dict.fromkeys(focus_files or []))
    validation_goal = {
        "unknown": "Explicar explícitamente si no hay validación segura disponible.",
        "compileall": "Confirmar que compileall puede ejecutarse sin errores si hay cambios.",
    }.get(validation_strategy, f"Usar la estrategia de validación {validation_strategy} cuando aplique.")

    if mode == "plan":
        return [
            PlanStage(
                stage_id="inspect",
                objective="Inspeccionar el repo y delimitar el problema real.",
                hypothesis="La inspección inicial identifica archivos relevantes y restricciones reales.",
                target_files=focus_files,
                intended_actions=["listar archivos relevantes", "leer archivos de foco", "extraer contexto del repo"],
                validation_goal="No aplica; etapa de inspección.",
                completion_criteria="Archivos relevantes identificados y evidencia local registrada.",
                failure_policy="Si falta evidencia, ampliar inspección antes de proponer etapas.",
            ),
            PlanStage(
                stage_id="hypothesis",
                objective="Formular hipótesis operativas concretas.",
                hypothesis="El objetivo puede descomponerse en etapas verificables con archivos y riesgos concretos.",
                target_files=focus_files,
                intended_actions=["definir hipótesis", "relacionar hipótesis con archivos", "anotar riesgos"],
                validation_goal="Hipótesis alineadas con la evidencia inspeccionada.",
                completion_criteria="Existe al menos una hipótesis defendible con evidencia del repo.",
                failure_policy="Si la hipótesis no se sostiene, replantear con la evidencia nueva.",
            ),
            PlanStage(
                stage_id="stage-plan",
                objective="Construir un plan por etapas ejecutable por el runtime.",
                hypothesis="Las etapas pueden ordenarse con objetivos, criterios y políticas de fallo explícitos.",
                target_files=focus_files,
                intended_actions=["definir etapas", "asignar criterios de éxito", "asignar políticas de fallo"],
                validation_goal="Cada etapa debe tener completion_criteria y failure_policy.",
                completion_criteria="Todas las etapas tienen campos obligatorios completos.",
                failure_policy="Si una etapa no puede definirse con claridad, reducir alcance o degradar honestamente.",
            ),
            PlanStage(
                stage_id="verify-plan",
                objective="Verificar consistencia entre objetivo, hipótesis y etapas.",
                hypothesis="El plan no contradice la evidencia ni salta verificaciones críticas.",
                target_files=focus_files,
                intended_actions=["comprobar consistencia", "revisar riesgos", "anotar dependencias"],
                validation_goal="El plan final debe ser ejecutable y trazable.",
                completion_criteria="No hay etapas sin criterio de éxito ni contradicciones internas.",
                failure_policy="Si hay contradicciones, replanear antes de cerrar.",
            ),
            PlanStage(
                stage_id="close",
                objective="Cerrar con un plan legible y operativo para el usuario.",
                hypothesis="El plan final refleja el estado real de inspección y los siguientes pasos.",
                target_files=focus_files,
                intended_actions=["resumir etapas", "resumir riesgos", "señalar próximos pasos"],
                validation_goal="El cierre debe reflejar el estado real por etapa.",
                completion_criteria="La salida final resume etapas, riesgos y pendientes sin contradicciones.",
                failure_policy="Si no hay evidencia suficiente, cerrar con límite explícito.",
            ),
        ]

    return [
        PlanStage(
            stage_id="inspect",
            objective="Inspeccionar archivos y contexto necesarios antes de responder.",
            hypothesis="La evidencia local alcanza para orientar la ejecución sin derivar.",
            target_files=focus_files,
            intended_actions=["listar archivos", "leer archivos de foco", "delimitar working set"],
            validation_goal="No aplica; etapa de inspección.",
            completion_criteria="Existe working set y evidencia real suficiente para continuar.",
            failure_policy="Si la evidencia es insuficiente, inspeccionar más antes de continuar.",
        ),
        PlanStage(
            stage_id="execute",
            objective="Resolver la tarea principal con cambios o conclusiones concretas.",
            hypothesis="La tarea puede resolverse con acciones trazables y claims soportados.",
            target_files=focus_files,
            intended_actions=["usar tools reales", "producir cambios mínimos o conclusiones grounded", "registrar evidencia"],
            validation_goal=validation_goal,
            completion_criteria="Se completa la acción principal sin claims no soportados.",
            failure_policy="Si falla la etapa, replanear una vez y luego degradar honestamente.",
        ),
        PlanStage(
            stage_id="verify",
            objective="Verificar el resultado de la etapa principal.",
            hypothesis="El resultado puede validarse o declararse como no validable con motivo explícito.",
            target_files=focus_files,
            intended_actions=["ejecutar validación cuando aplique", "revisar grounding", "registrar issues"],
            validation_goal=validation_goal,
            completion_criteria="El resultado queda validado o la limitación queda documentada con precisión.",
            failure_policy="Si la verificación falla, registrar fallo y no marcar la etapa como completada.",
        ),
        PlanStage(
            stage_id="close",
            objective="Cerrar con estado real, riesgos y pendientes.",
            hypothesis="El cierre final puede resumir el estado real por etapa sin contradicciones.",
            target_files=focus_files,
            intended_actions=["resumir estado por etapa", "anotar riesgos", "dejar siguiente paso óptimo"],
            validation_goal="El cierre debe reflejar el estado real por etapa.",
            completion_criteria="La respuesta final coincide con el estado real de ejecución.",
            failure_policy="Si faltan evidencias, degradar honestamente indicando la etapa detenida.",
        ),
    ]


def render_execution_plan(stages: list[PlanStage]) -> str:
    inspect_stage = next((stage for stage in stages if stage.stage_id == "inspect"), stages[0] if stages else None)
    objective = stages[0].objective if stages else "Sin etapas definidas."
    target_files = inspect_stage.target_files if inspect_stage else []
    hypotheses = [stage.hypothesis for stage in stages if stage.hypothesis]
    risks = [stage.failure_policy for stage in stages if stage.failure_policy]
    next_steps = [f"{stage.stage_id}: {stage.objective}" for stage in stages]
    lines = [
        "Objetivo",
        f"- {objective}",
        "Archivos a revisar",
        *(f"- {item}" for item in (target_files or ["README.md"])),
        "Hipótesis",
        *(f"- {item}" for item in (hypotheses or ["No hay hipótesis registradas."])),
        "Riesgos",
        *(f"- {item}" for item in (risks or ["No hay política de fallo registrada."])),
        "Siguientes pasos",
        *(f"- {item}" for item in (next_steps or ["Sin etapas registradas."])),
        "Etapas",
    ]
    for stage in stages:
        lines.append(f"- {stage.stage_id}: {stage.objective}")
        lines.append(f"  hypothesis={stage.hypothesis}")
        lines.append(f"  target_files={', '.join(stage.target_files) or 'none'}")
        lines.append(f"  intended_actions={'; '.join(stage.intended_actions) or 'none'}")
        lines.append(f"  validation_goal={stage.validation_goal}")
        lines.append(f"  completion_criteria={stage.completion_criteria}")
        lines.append(f"  failure_policy={stage.failure_policy}")
    return "\n".join(lines)


def build_plan_prompt(user_prompt: str, profile: str = "local") -> str:
    style = {
        "local": "Keep it short and implementation-first.",
        "balanced": "Balance speed with solid technical reasoning.",
        "deep": "Go deeper on risks, sequencing, and architectural impact.",
    }.get(profile, "Keep it concise and grounded.")
    return (
        "Build a disciplined implementation plan in Spanish. "
        "Use this exact structure with plain prose bullets: "
        "Objetivo, Archivos a revisar, Hipótesis, Riesgos, Siguientes pasos. "
        "Keep it consistent with an executable stage model: stage_id, objective, hypothesis, target_files, intended_actions, validation_goal, completion_criteria, failure_policy. "
        "Do not output fake tool calls or JSON. "
        "If inspection is incomplete, say what was inspected and what still needs inspection. "
        f"{style} Task: {user_prompt}"
    )
