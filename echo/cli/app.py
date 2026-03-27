from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from echo.backends.errors import BackendError
from echo.config import Settings
from echo.core import EchoAgent
from echo.ui import EchoShell

app = typer.Typer(help="Echo local coding agent")
console = Console()


def profile_option() -> str | None:
    return typer.Option(None, "--profile", help="Execution profile: local, balanced, deep, auto")


def strict_profile_option() -> bool:
    return typer.Option(False, "--strict-profile", help="Fail instead of falling back when the requested profile is not ready")

def build_agent(
    project_dir: str | None = None,
    profile: str | None = None,
    strict_profile: bool = False,
) -> tuple[EchoAgent, Path, Settings]:
    root = Path(project_dir).resolve() if project_dir else Path.cwd().resolve()
    settings = Settings.from_env()
    if profile:
        settings.profile = profile
    settings.strict_profile = strict_profile
    return EchoAgent(root, settings), root, settings


def safe_build_agent(
    project_dir: str | None = None,
    profile: str | None = None,
    strict_profile: bool = False,
) -> tuple[EchoAgent, Path, Settings]:
    try:
        return build_agent(project_dir, profile, strict_profile)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2)


def safe_agent_run(agent: EchoAgent, prompt: str, *, mode: str, resume_session_id: str | None = None, profile: str | None = None):
    try:
        return agent.run(prompt, mode=mode, resume_session_id=resume_session_id, profile=profile)
    except BackendError as exc:
        console.print(f"[red]Error de backend:[/red] {exc}")
        raise typer.Exit(code=2)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=2)


@app.command()
def doctor(
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
) -> None:
    agent, _, _ = safe_build_agent(project_dir, profile, strict_profile)
    data = agent.doctor()
    table = Table(title="Echo doctor")
    table.add_column("Check", style="cyan")
    table.add_column("Value", style="white")
    for key, value in data.items():
        table.add_row(str(key), ", ".join(value) if isinstance(value, list) else str(value))
    console.print(table)


@app.command("backend-check")
def backend_check(
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
    chat_samples: int = typer.Option(2, "--chat-samples", min=1, max=4, help="How many minimal chat probes to run"),
) -> None:
    agent, _, _ = safe_build_agent(project_dir, profile, strict_profile)
    data = agent.backend_check(chat_samples=chat_samples)
    table = Table(title="Echo backend-check")
    table.add_column("Check", style="cyan")
    table.add_column("Value", style="white")
    for key, value in data.items():
        table.add_row(str(key), ", ".join(value) if isinstance(value, list) else str(value))
    console.print(table)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Task to execute"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
) -> None:
    agent, _, _ = safe_build_agent(project_dir, profile, strict_profile)
    answer, session_path, session = safe_agent_run(agent, prompt, mode="ask", profile=profile)
    console.print(Panel(answer, title="Echo", expand=True))
    console.print(f"Sesión guardada en: {session_path}")
    console.print(f"Herramientas usadas: {len(session.tool_calls)}")


@app.command()
def plan(
    prompt: str = typer.Argument(..., help="Task to plan"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
) -> None:
    agent, _, _ = safe_build_agent(project_dir, profile, strict_profile)
    answer, session_path, session = safe_agent_run(agent, prompt, mode="plan", profile=profile)
    console.print(Panel(answer, title="Plan", expand=True))
    console.print(f"Sesión guardada en: {session_path}")
    console.print(f"Herramientas usadas: {len(session.tool_calls)}")


@app.command()
def shell(
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
) -> None:
    agent, root, _ = safe_build_agent(project_dir, profile, strict_profile)
    EchoShell(agent, root, console).run()


@app.command()
def resume(
    prompt: str = typer.Argument("Continúa desde la última sesión y sigue trabajando.", help="Prompt to continue the latest session"),
    session_id: str | None = typer.Option(None, "--session-id", help="Specific session to resume"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
) -> None:
    agent, _, _ = safe_build_agent(project_dir, profile, strict_profile)
    answer, session_path, session = safe_agent_run(agent, prompt, mode="resume", resume_session_id=session_id, profile=profile)
    console.print(Panel(answer, title="Resume", expand=True))
    console.print(f"Sesión guardada en: {session_path}")
    console.print(f"Herramientas usadas: {len(session.tool_calls)}")


@app.command()
def smoke(
    prompt: str = typer.Argument("Responde con un diagnóstico breve del backend actual.", help="Prompt used for smoke validation"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
) -> None:
    agent, _, _ = safe_build_agent(project_dir, profile, strict_profile)
    ask_prompt = (
        "Inspecciona README.md y echo/config.py. Responde de forma breve y grounded al contexto. "
        "Debes citar ambos archivos y al menos un identificador real de echo/config.py, por ejemplo Settings o from_env. "
        "Evita copiar fragmentos largos; responde en 2-4 frases claras sobre qué dicen esos archivos para este contexto. "
        f"Contexto: {prompt}"
    )
    doctor_data = agent.doctor()
    table = Table(title="Echo smoke")
    table.add_column("Check", style="cyan")
    table.add_column("Value", style="white")
    for key in [
        "profile",
        "resolved_profile",
        "deep_ready",
        "backend_primary",
        "backend_fallback",
        "backend_policy",
        "routing_reason",
        "backend_label",
        "backend_tools",
        "model",
        "backend_reachable",
        "backend_chat_ready",
        "backend_chat_slow",
        "backend_state",
        "backend_health_cached_state",
        "backend_health_fresh_state",
        "backend_health_effective_state",
    ]:
        if key in doctor_data:
            value = doctor_data[key]
            table.add_row(str(key), str(value))
    console.print(table)
    if not doctor_data.get("backend_reachable"):
        console.print("[red]Smoke failed:[/red] fallo del backend: no es alcanzable ni siquiera para comprobación básica.")
        raise typer.Exit(code=2)
    if not doctor_data.get("backend_chat_ready"):
        if doctor_data.get("backend_policy") == "hybrid-fallback":
            console.print("[yellow]Smoke:[/yellow] backend primario no está sano, pero hay fallback híbrido configurado.")
        else:
            console.print("[red]Smoke failed:[/red] fallo del backend: chat no está sano para ejecutar flujo completo.")
            raise typer.Exit(code=2)
    if doctor_data.get("backend_chat_slow") and doctor_data.get("backend_policy") != "hybrid-fallback":
        console.print("[yellow]Smoke:[/yellow] backend primario lento; el flujo puede degradar o tardar.")
    plan_answer, plan_path, plan_session = safe_agent_run(
        agent,
        f"Inspecciona el repo y crea un plan sólido. Contexto: {prompt}",
        mode="plan",
        profile=profile,
    )
    if any(section not in plan_answer.lower() for section in ["objetivo", "archivos a revisar", "riesgos", "siguientes pasos"]):
        console.print("[red]Smoke failed:[/red] fallo del agente: plan no devolvió la estructura obligatoria.")
        raise typer.Exit(code=3)
    if len(plan_session.tool_calls) == 0:
        console.print("[red]Smoke failed:[/red] fallo del agente: plan no usó herramientas reales.")
        raise typer.Exit(code=3)
    if not plan_answer.lower().count("echo/") and not plan_answer.lower().count("readme"):
        console.print("[red]Smoke failed:[/red] fallo del agente: plan no quedó grounded en archivos del repo.")
        raise typer.Exit(code=3)
    answer, session_path, session = safe_agent_run(agent, ask_prompt, mode="ask", profile=profile)
    if len(session.tool_calls) == 0:
        console.print("[red]Smoke failed:[/red] fallo del agente: ask no usó herramientas reales.")
        raise typer.Exit(code=3)
    if not session.grounded_answer:
        console.print("[red]Smoke failed:[/red] fallo de grounding: answer final insuficientemente grounded.")
        raise typer.Exit(code=3)
    if session.grounding_report.get("grounded_symbol_count", 0) == 0:
        console.print("[red]Smoke failed:[/red] fallo de grounding: answer final no citó símbolos concretos.")
        raise typer.Exit(code=3)
    resume_answer, resume_path, resume_session = safe_agent_run(
        agent,
        "Continúa desde la última sesión y resume objetivo, working set y pendientes.",
        mode="resume",
        resume_session_id=session.id,
        profile=profile,
    )
    if not (resume_session.working_set or resume_session.operational_summary):
        console.print("[red]Smoke failed:[/red] fallo del agente: resume no restauró estado útil.")
        raise typer.Exit(code=3)
    console.print(Panel(plan_answer, title="Smoke Plan", expand=True))
    console.print(f"Plan session: {plan_path}")
    console.print(Panel(answer, title="Smoke Ask", expand=True))
    console.print(f"Ask session: {session_path}")
    console.print(Panel(resume_answer, title="Smoke Resume", expand=True))
    console.print(f"Resume session: {resume_path}")
    console.print(f"Herramientas usadas en ask: {len(session.tool_calls)}")


def echo_agent_main() -> None:
    """
    Entry point for `echo-agent`.

    Opens the conversational REPL directly — no subcommand needed.
    Branch-aware, session-persistent, with merge/cherry-pick support.

    Usage:
        echo-agent [--project-dir DIR] [--profile PROFILE]
    """
    import argparse

    from echo.branches.store import BranchStore
    from echo.ui.repl import EchoRepl

    parser = argparse.ArgumentParser(
        prog="echo-agent",
        description="Echo conversational CLI — habla con Echo directamente.",
    )
    parser.add_argument("--project-dir", default=None, metavar="DIR", help="Raíz del proyecto (default: cwd)")
    parser.add_argument("--profile", default=None, metavar="PROFILE", help="Perfil de ejecución: local, balanced, deep")
    args = parser.parse_args()

    try:
        agent, root, _ = build_agent(args.project_dir, args.profile)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(2) from exc

    branch_store = BranchStore(root)
    EchoRepl(agent, root, console, branch_store).run()


if __name__ == "__main__":
    app()
