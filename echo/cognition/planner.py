from __future__ import annotations


def build_plan_prompt(user_prompt: str, profile: str = "local") -> str:
    style = {
        "local": "Keep it short and implementation-first.",
        "balanced": "Balance speed with solid technical reasoning.",
        "deep": "Go deeper on risks, sequencing, and architectural impact.",
    }.get(profile, "Keep it concise and grounded.")
    return (
        "Build a disciplined implementation plan in Spanish. "
        "Use this exact structure with plain prose bullets: "
        "Objetivo, Archivos a revisar, Riesgos, Siguientes pasos. "
        "Use project evidence when available and mention concrete file paths. "
        "Do not output fake tool calls or JSON. "
        "If inspection is incomplete, say what was inspected and what still needs inspection. "
        f"{style} Task: {user_prompt}"
    )
