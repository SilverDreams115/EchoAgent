from __future__ import annotations

from echo.types import RunState, SessionState


def build_heuristic_plan(
    session: SessionState,
    run_state: RunState,
    *,
    reason: str,
    update_stage,
    activity,
) -> str:
    run_state.fallback_used = True
    run_state.fallback_reason = reason
    stage_id = run_state.current_stage_id or "inspect"
    update_stage(session, run_state, stage_id, status="failed", result=reason)
    activity.emit("Planner", "degraded", "Fallback plan mode", reason)
    files = session.working_set[-6:] or session.focus_files[-6:] or ["README.md"]
    risks = [
        "El backend de chat está inestable, así que el plan se basa en inspección local y memoria operativa.",
        "Puede faltar validación semántica profunda hasta que Ollama vuelva a responder de forma consistente.",
    ]
    next_steps = [
        "Revisar el loop de runtime y el verificador grounded.",
        "Confirmar backend health y repetir smoke cuando /api/chat deje de oscilar.",
    ]
    return "\n".join(
        [
            "Objetivo",
            f"- Mantener Echo operativo aun con backend inestable y reforzar grounding del answer final.",
            "Archivos a revisar",
            *(f"- {item}" for item in files),
            "Riesgos",
            *(f"- {item}" for item in risks),
            "Siguientes pasos",
            *(f"- {item}" for item in next_steps),
        ]
    )


def build_degraded_answer(
    session: SessionState,
    run_state: RunState,
    *,
    reason: str,
    mode: str,
    update_stage,
    activity,
) -> str:
    run_state.fallback_used = True
    run_state.fallback_reason = reason
    session.degraded_reason = reason
    current_stage = run_state.current_stage_id or ("close" if mode == "plan" else "execute")
    update_stage(session, run_state, current_stage, status="failed", result=reason)
    if mode == "resume":
        activity.emit("Resume", "degraded", "Resume state restored without backend completion", reason)
        return "\n".join(
            [
                "Echo restauró el estado de la sesión, pero no pudo completar con el backend.",
                f"Objetivo: {session.objective or session.user_prompt}",
                f"Working set: {', '.join(session.working_set[-8:]) or 'none'}",
                f"Pendientes: {'; '.join(session.pending[-6:]) or 'none'}",
                f"Etapa detenida: {current_stage}",
                f"Límite actual: {reason}",
            ]
        )
    activity.emit("Planner" if mode == "plan" else "Verifier", "degraded", "Fallback answer mode", reason)
    evidence = session.working_set[-6:] or session.focus_files[-6:]
    return "\n".join(
        [
            "Echo reunió inspección local, pero no pudo cerrar con el backend de forma confiable.",
            f"Archivos inspeccionados: {', '.join(evidence) or 'none'}",
            f"Hallazgos recientes: {'; '.join(session.findings[-6:] or run_state.inspected_files[-6:] or ['inspección local completada'])}",
            f"Etapa detenida: {current_stage}",
            f"Backend efectivo: {run_state.fresh_backend_health.backend_state or run_state.backend_health.backend_state}",
            f"Límite actual: {reason}",
        ]
    )


def is_resume_summary_only(prompt: str) -> bool:
    low = (prompt or "").lower()
    resume_markers = ["resume", "resum", "working set", "pendient", "objetivo"]
    return sum(1 for marker in resume_markers if marker in low) >= 3


def build_resume_local_answer(session: SessionState, run_state: RunState, *, activity) -> str:
    activity.emit("Resume", "done", "Resume local summary", session.id)
    return "\n".join(
        [
            "Echo restauró la sesión desde memoria local.",
            f"Objetivo: {session.objective or run_state.objective}",
            f"Working set: {', '.join(session.working_set[-8:] or session.focus_files[-8:]) or 'none'}",
            f"Pendientes: {'; '.join(session.pending[-6:] or run_state.pending[-6:]) or 'none'}",
        ]
    )
