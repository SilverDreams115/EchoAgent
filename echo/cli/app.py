from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from echo.backends.errors import BackendError
from echo.config import Settings
from echo.core import EchoAgent
from echo.types import BranchMeta
from echo.ui import EchoShell

app = typer.Typer(help="Echo local coding agent")
branch_app = typer.Typer(help="Manage conversation branches within a project session")
app.add_typer(branch_app, name="branch")
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


def safe_agent_run(
    agent: EchoAgent,
    prompt: str,
    *,
    mode: str,
    resume_session_id: str | None = None,
    profile: str | None = None,
    branch_name: str | None = None,
):
    try:
        return agent.run(
            prompt,
            mode=mode,
            resume_session_id=resume_session_id,
            profile=profile,
            branch_name=branch_name,
        )
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
    active = agent.store.ensure_branch_model()
    branch_name = active.branch_name
    answer, session_path, session = safe_agent_run(
        agent, prompt, mode="ask", profile=profile, branch_name=branch_name
    )
    console.print(Panel(answer, title=f"Echo [{branch_name}]", expand=True))
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
    active = agent.store.ensure_branch_model()
    branch_name = active.branch_name
    answer, session_path, session = safe_agent_run(
        agent, prompt, mode="plan", profile=profile, branch_name=branch_name
    )
    console.print(Panel(answer, title=f"Plan [{branch_name}]", expand=True))
    console.print(f"Sesión guardada en: {session_path}")
    console.print(f"Herramientas usadas: {len(session.tool_calls)}")


@app.command()
def shell(
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
) -> None:
    agent, root, _ = safe_build_agent(project_dir, profile, strict_profile)
    agent.store.ensure_branch_model()
    EchoShell(agent, root, console).run()


@app.command()
def resume(
    prompt: str = typer.Argument("Continúa desde la última sesión y sigue trabajando.", help="Prompt to continue the latest session"),
    session_id: str | None = typer.Option(None, "--session-id", help="Specific session to resume (overrides branch head)"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root to inspect"),
    profile: str | None = profile_option(),
    strict_profile: bool = strict_profile_option(),
) -> None:
    agent, _, _ = safe_build_agent(project_dir, profile, strict_profile)
    active = agent.store.ensure_branch_model()
    branch_name = active.branch_name
    answer, session_path, session = safe_agent_run(
        agent, prompt, mode="resume",
        resume_session_id=session_id,  # explicit --session-id wins; None → branch head
        profile=profile,
        branch_name=branch_name,
    )
    console.print(Panel(answer, title=f"Resume [{branch_name}]", expand=True))
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


# ------------------------------------------------------------------ #
# Branch commands                                                       #
# ------------------------------------------------------------------ #

@branch_app.command("status")
def branch_status(
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root"),
) -> None:
    """Show the active branch and its current head session."""
    agent, _, _ = safe_build_agent(project_dir)
    active = agent.store.ensure_branch_model()
    branch = agent.store.load_branch(active.branch_name)

    table = Table(title="Branch status")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("active_branch", active.branch_name)
    if branch:
        table.add_row("head_session", branch.head_session_id or "none")
        table.add_row("parent_branch", branch.parent_branch or "—")
        table.add_row("fork_session", branch.fork_session_id or "—")
        table.add_row("description", branch.description or "—")
        table.add_row("created_at", branch.created_at)
        table.add_row("updated_at", branch.updated_at)
    else:
        table.add_row("head_session", "none")
    console.print(table)


@branch_app.command("list")
def branch_list(
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root"),
) -> None:
    """List all branches for this project."""
    agent, _, _ = safe_build_agent(project_dir)
    active = agent.store.ensure_branch_model()
    branches = agent.store.list_branches()

    table = Table(title="Branches")
    table.add_column("Name", style="cyan")
    table.add_column("Active", style="green")
    table.add_column("Head session", style="white")
    table.add_column("Parent", style="white")
    table.add_column("Updated", style="white")
    for branch in branches:
        marker = "✓" if branch.name == active.branch_name else ""
        table.add_row(
            branch.name,
            marker,
            branch.head_session_id or "none",
            branch.parent_branch or "—",
            branch.updated_at[:19],
        )
    if not branches:
        console.print("No branches found. Run any ask/plan/resume command to initialize.")
    else:
        console.print(table)


@branch_app.command("new")
def branch_new(
    name: str = typer.Argument(..., help="Name for the new branch"),
    from_branch: str | None = typer.Option(None, "--from", help="Source branch to fork from (default: active branch)"),
    description: str = typer.Option("", "--description", help="Optional description"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root"),
) -> None:
    """Create a new branch forked from the active (or specified) branch and activate it."""
    agent, _, _ = safe_build_agent(project_dir)
    active = agent.store.ensure_branch_model()

    if agent.store.load_branch(name):
        console.print(f"[red]Error:[/red] branch '{name}' already exists.")
        raise typer.Exit(code=1)

    source_name = from_branch or active.branch_name
    source = agent.store.load_branch(source_name)
    if source is None and from_branch:
        console.print(f"[red]Error:[/red] source branch '{from_branch}' not found.")
        raise typer.Exit(code=1)

    fork_head = source.head_session_id if source else ""
    new_branch = BranchMeta(
        name=name,
        head_session_id=fork_head,
        parent_branch=source_name,
        fork_session_id=fork_head,
        description=description,
    )
    agent.store.save_branch(new_branch)
    agent.store.set_active_branch(name)
    console.print(f"Branch [cyan]'{name}'[/cyan] created from [white]'{source_name}'[/white] and activated.")
    if fork_head:
        console.print(f"Fork point: session [white]{fork_head}[/white]")
    else:
        console.print("Fork point: empty (no sessions on source branch yet)")


@branch_app.command("switch")
def branch_switch(
    name: str = typer.Argument(..., help="Branch name to activate"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root"),
) -> None:
    """Switch the active branch."""
    agent, _, _ = safe_build_agent(project_dir)
    agent.store.ensure_branch_model()

    branch = agent.store.load_branch(name)
    if branch is None:
        console.print(f"[red]Error:[/red] branch '{name}' not found.")
        branches = agent.store.list_branches()
        if branches:
            console.print("Available branches: " + ", ".join(b.name for b in branches))
        raise typer.Exit(code=1)

    agent.store.set_active_branch(name)
    console.print(f"Switched to branch [cyan]'{name}'[/cyan].")
    if branch.head_session_id:
        console.print(f"Head session: [white]{branch.head_session_id}[/white]")
    else:
        console.print("Head session: none (branch has no sessions yet)")


@branch_app.command("show")
def branch_show(
    name: str = typer.Argument(..., help="Branch name to inspect"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root"),
) -> None:
    """Show detailed metadata for a branch."""
    agent, _, _ = safe_build_agent(project_dir)
    agent.store.ensure_branch_model()

    branch = agent.store.load_branch(name)
    if branch is None:
        console.print(f"[red]Error:[/red] branch '{name}' not found.")
        raise typer.Exit(code=1)

    table = Table(title=f"Branch: {name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("name", branch.name)
    table.add_row("head_session", branch.head_session_id or "none")
    table.add_row("parent_branch", branch.parent_branch or "—")
    table.add_row("fork_session", branch.fork_session_id or "—")
    table.add_row("description", branch.description or "—")
    table.add_row("created_at", branch.created_at)
    table.add_row("updated_at", branch.updated_at)

    if branch.head_session_id:
        try:
            session = agent.store.load_session(branch.head_session_id)
            table.add_row("head_objective", (session.objective or session.user_prompt)[:80])
            table.add_row("head_mode", session.mode)
            table.add_row("head_grounded", str(session.grounded_answer))
            table.add_row("head_tool_calls", str(len(session.tool_calls)))
            table.add_row("head_changed_files", str(len(session.changed_files)))
        except Exception:
            table.add_row("head_session_detail", "unavailable")
    console.print(table)


@branch_app.command("delete")
def branch_delete(
    name: str = typer.Argument(..., help="Branch name to delete"),
    project_dir: str | None = typer.Option(None, "--project-dir", help="Project root"),
) -> None:
    """Delete a branch (cannot delete 'main' or the active branch)."""
    agent, _, _ = safe_build_agent(project_dir)
    active = agent.store.ensure_branch_model()

    if name == "main":
        console.print("[red]Error:[/red] cannot delete the main branch.")
        raise typer.Exit(code=1)
    if active.branch_name == name:
        console.print(f"[red]Error:[/red] cannot delete the active branch '{name}'. Switch to another branch first.")
        raise typer.Exit(code=1)
    if agent.store.load_branch(name) is None:
        console.print(f"[red]Error:[/red] branch '{name}' not found.")
        raise typer.Exit(code=1)

    agent.store.delete_branch(name)
    console.print(f"Branch [cyan]'{name}'[/cyan] deleted.")


if __name__ == "__main__":
    app()
