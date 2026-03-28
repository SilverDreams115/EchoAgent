"""
EchoRepl — conversational REPL, primary interface for echo-agent.

Input model
-----------
- Enter          → envía el mensaje  (comportamiento principal de chat)
- Alt+Enter      → nueva línea dentro del mensaje  (Esc+Enter en cualquier terminal)
- Ctrl + D       → salir del REPL

The REPL accepts a _input_fn parameter for testing: when provided, it is
called instead of the interactive prompt_toolkit session.  Pass a callable
that returns the next line (str) or None to signal EOF / exit.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from echo.branches.cherry_pick import cherry_pick
from echo.branches.merge import ARTEFACT_TYPES, merge_branches
from echo.branches.models import BranchMergeRecord, BranchState
from echo.branches.store import BranchStore
from echo.core import EchoAgent
from echo.types import ActivityEvent
from echo.ui.intent_router import Intent, extract_artefacts, route

_HELP_TEXT = """\
[bold cyan]Echo conversational CLI[/bold cyan]

Escribe naturalmente — sin prefijos obligatorios.

[bold]Input:[/bold]
  [cyan]Enter[/cyan]          → enviar el mensaje
  [cyan]Alt + Enter[/cyan]    → nueva línea dentro del mensaje  (Esc+Enter en cualquier terminal)
  [cyan]Ctrl + D[/cyan]       → salir de Echo

[bold]Slash commands:[/bold]
  [cyan]/help[/cyan]
  [cyan]/exit[/cyan]
  [cyan]/session status[/cyan]
  [cyan]/session new[/cyan]
  [cyan]/branch status[/cyan]
  [cyan]/branch list[/cyan]
  [cyan]/branch new <nombre>[/cyan]
  [cyan]/branch switch <nombre>[/cyan]
  [cyan]/branch show <nombre>[/cyan]
  [cyan]/branch merge <fuente> [--strategy union-deduplicate|prefer-source|prefer-destination][/cyan]
  [cyan]/branch cherry-pick <fuente> [--decisions] [--findings] [--pending] [--facts] [--summary][/cyan]
  [cyan]/doctor[/cyan]

[bold]Estrategias de merge:[/bold]
  union-deduplicate (default) · prefer-source · prefer-destination

[bold]Lenguaje natural (ejemplos):[/bold]
  crea una rama experimento-shell
  vuelve a main · cambia a feature-x
  merge experimento-shell
  trae las decisiones de experimento-shell
  pásame los findings de feature-x
  cherry-pick feature-x --summary --facts
"""

_ARTEFACT_FLAG_MAP: dict[str, str] = {
    "--decisions": "decisions",
    "--findings": "findings",
    "--pending": "pending",
    "--facts": "facts",
    "--summary": "summary",
    "--errors": "errors",
    "--changes": "changes",
}


def _parse_artefact_flags(args: list[str]) -> list[str]:
    return [_ARTEFACT_FLAG_MAP[a.lower()] for a in args if a.lower() in _ARTEFACT_FLAG_MAP]


def _build_chat_kb() -> KeyBindings:
    """Key bindings for the chat prompt.

    Enter submits the current message (primary chat action).
    Alt+Enter (Esc+Enter in any terminal) inserts a newline for multiline messages.
    Ctrl+D signals exit.
    """
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        """Enter submits the buffer — default chat behavior."""
        event.app.exit(result=event.app.current_buffer.text)

    @kb.add("escape", "enter")
    def _newline(event):
        """Alt+Enter (Esc+Enter) inserts a newline for multiline messages."""
        event.app.current_buffer.insert_text("\n")

    @kb.add("c-d")
    def _eof(event):
        """Ctrl+D signals exit."""
        event.app.exit(result=None)

    return kb


class EchoRepl:
    """
    Conversational REPL — the primary interface for echo-agent.

    The user types natural language directly (no /ask, /plan, /resume required).
    Branch operations, merge, and cherry-pick are available via natural phrasing
    or explicit slash commands.

    Parameters
    ----------
    _input_fn
        Optional callable for testing.  When provided, replaces the interactive
        prompt_toolkit input.  Should return str for each turn or None to signal
        EOF / exit.  Ignored in normal (non-test) usage.
    """

    def __init__(
        self,
        agent: EchoAgent,
        project_root: Path,
        console: Console,
        branch_store: BranchStore,
        *,
        _input_fn: Callable[[], str | None] | None = None,
    ) -> None:
        self.agent = agent
        self.project_root = project_root
        self.console = console
        self.branch_store = branch_store
        self._input_fn = _input_fn
        self._queue: Queue[ActivityEvent] = Queue()
        self.agent.activity.watch(self._queue.put)
        # Restore current session from active branch (or fall back to store's latest)
        active_branch = branch_store.active_branch_name()
        self._current_session_id: str | None = (
            branch_store.active_session_for_branch(active_branch)
            or agent.store.latest_session_id()
        )
        # Track whether a session was found at startup (used by _print_header)
        self._session_resumed: bool = self._current_session_id is not None

    # ------------------------------------------------------------------
    # Header + prompt
    # ------------------------------------------------------------------

    def _print_header(self) -> None:
        branch = self.branch_store.active_branch_name()
        if self._session_resumed and self._current_session_id:
            session_label = f"resumida · {self._current_session_id[:8]}"
        else:
            session_label = "nueva"
        t = Text()
        t.append("Echo", style="bold bright_cyan")
        t.append("  ·  repo: ", style="dim white")
        t.append(self.project_root.name, style="bold white")
        t.append("  ·  branch: ", style="dim white")
        t.append(branch, style="bold green")
        t.append("  ·  session: ", style="dim white")
        t.append(session_label, style="white")
        self.console.print(Panel(t, border_style="bright_black", padding=(0, 1)))
        self.console.print(
            "[dim]Enter=enviar  ·  Alt+Enter=nueva línea  ·  /help=comandos  ·  Ctrl+D=salir[/dim]\n"
        )

    def _prompt_line(self) -> str | None:
        if self._input_fn is not None:
            return self._input_fn()

        history_file = self.project_root / ".echo" / "prompt_history.txt"
        history = FileHistory(str(history_file))
        branch = self.branch_store.active_branch_name()
        kb = _build_chat_kb()
        try:
            result = pt_prompt(
                f"[{branch}] > ",
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="  ↳ ",
            )
            return result
        except (EOFError, KeyboardInterrupt):
            return None

    # ------------------------------------------------------------------
    # Agent turn
    # ------------------------------------------------------------------

    def _run_agent_turn(self, user_text: str) -> str:
        holder: dict = {"done": False, "answer": "", "session": None, "error": None}

        def worker() -> None:
            try:
                answer, _path, session = self.agent.run(
                    user_text,
                    mode="ask",
                    resume_session_id=self._current_session_id,
                )
                holder["answer"] = answer
                holder["session"] = session
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc
            finally:
                holder["done"] = True

        thread = Thread(target=worker, daemon=True)
        thread.start()

        with Live(
            Spinner("dots", text="[dim]Echo está pensando…[/dim]"),
            console=self.console,
            transient=True,
            refresh_per_second=8,
        ):
            while not holder["done"]:
                try:
                    while True:
                        self._queue.get_nowait()
                except Empty:
                    pass
                time.sleep(0.05)

        if holder["error"]:
            return f"[red]Error:[/red] {holder['error']}"

        session = holder["session"]
        if session:
            self._current_session_id = session.id
            branch = self.branch_store.active_branch_name()
            self.branch_store.add_session_to_branch(branch, session.id)

        return str(holder.get("answer") or "")

    # ------------------------------------------------------------------
    # Slash command dispatch
    # ------------------------------------------------------------------

    def _handle_slash(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower()
        rest = parts[1:]

        if cmd in {"/exit", "/quit"}:
            self.console.print("Saliendo. Hasta luego.")
            sys.exit(0)

        if cmd == "/help":
            self.console.print(Panel(_HELP_TEXT, title="Ayuda", border_style="cyan"))
            return

        if cmd == "/doctor":
            self._cmd_doctor()
            return

        if cmd == "/session":
            self._handle_session_cmd(rest)
            return

        if cmd == "/branch":
            self._handle_branch_cmd(rest)
            return

        self.console.print(f"[yellow]Comando desconocido: {cmd}. Usa /help.[/yellow]")

    def _cmd_doctor(self) -> None:
        from rich.table import Table

        data = self.agent.doctor()
        table = Table(title="Echo doctor")
        table.add_column("Check", style="cyan")
        table.add_column("Value", style="white")
        for key, value in data.items():
            table.add_row(str(key), ", ".join(value) if isinstance(value, list) else str(value))
        self.console.print(table)

    # ------------------------------------------------------------------
    # /session commands
    # ------------------------------------------------------------------

    def _handle_session_cmd(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "status"

        if sub == "status":
            status = self.agent.current_status()
            branch = self.branch_store.active_branch_name()
            lines = [f"branch: [bold green]{branch}[/bold green]"]
            lines += [f"{k}: {v}" for k, v in status.items()]
            self.console.print(
                Panel("\n".join(lines), title="Session Status", border_style="cyan", expand=False)
            )
            return

        if sub == "new":
            self._current_session_id = None
            branch = self.branch_store.active_branch_name()
            self.console.print(f"[green]Nueva sesión limpia en branch '{branch}'.[/green]")
            return

        self.console.print(f"[yellow]Subcomando de session desconocido: {sub}. Usa /help.[/yellow]")

    # ------------------------------------------------------------------
    # /branch commands
    # ------------------------------------------------------------------

    def _handle_branch_cmd(self, args: list[str]) -> None:
        sub = args[0].lower() if args else "status"
        rest = args[1:]

        if sub == "status":
            name = self.branch_store.active_branch_name()
            if self.branch_store.branch_exists(name):
                self._print_branch_info(self.branch_store.load_branch(name), active=True)
            else:
                self.console.print(
                    f"[yellow]Branch '{name}' no tiene metadata aún (se crea al primer uso).[/yellow]"
                )
            return

        if sub == "list":
            self._cmd_branch_list()
            return

        if sub == "new" and rest:
            self._create_and_switch_branch(rest[0])
            return

        if sub == "switch" and rest:
            self._switch_branch(rest[0])
            return

        if sub == "show" and rest:
            name = rest[0]
            if not self.branch_store.branch_exists(name):
                self.console.print(f"[red]Branch '{name}' no existe.[/red]")
            else:
                active = self.branch_store.active_branch_name()
                self._print_branch_info(
                    self.branch_store.load_branch(name), active=(name == active)
                )
            return

        if sub == "merge" and rest:
            source = rest[0]
            strategy = "union-deduplicate"
            for i, a in enumerate(rest):
                if a == "--strategy" and i + 1 < len(rest):
                    strategy = rest[i + 1]
            self._do_merge(source, strategy=strategy)
            return

        if sub in {"cherry-pick", "cherry_pick"} and rest:
            source = rest[0]
            # Try explicit flags first; fall back to text extraction
            atypes = _parse_artefact_flags(rest[1:])
            if not atypes:
                atypes = extract_artefacts(" ".join(rest[1:]))
            if not atypes:
                atypes = ["decisions", "findings"]
            self._do_cherry_pick(source, atypes)
            return

        self.console.print(
            f"[yellow]Subcomando de branch desconocido: {sub}. Usa /help.[/yellow]"
        )

    def _cmd_branch_list(self) -> None:
        branches = self.branch_store.list_branches()
        active = self.branch_store.active_branch_name()
        if not branches:
            self.console.print("[dim]No hay ramas todavía. Se crea 'main' automáticamente.[/dim]")
            return
        for name in branches:
            if name == active:
                self.console.print(f"  [bold green]* {name}[/bold green] [dim](active)[/dim]")
            else:
                self.console.print(f"    {name}")

    def _print_branch_info(self, b: BranchState, active: bool = False) -> None:
        active_marker = " [bold green](active)[/bold green]" if active else ""
        records = self.branch_store.load_merge_records(b.name)
        text = (
            f"[bold]{b.name}[/bold]{active_marker}\n"
            f"created:        {b.created_at}\n"
            f"parent:         {b.parent_branch or '(root)'}\n"
            f"sessions:       {len(b.session_ids)}\n"
            f"active session: {b.active_session_id or 'none'}\n"
            f"merges:         {len(records)}\n"
        )
        if b.description:
            text += f"description:    {b.description}\n"
        self.console.print(
            Panel(text, title=f"Branch: {b.name}", border_style="green", expand=False)
        )

    # ------------------------------------------------------------------
    # Branch operations
    # ------------------------------------------------------------------

    def _create_and_switch_branch(self, name: str) -> None:
        current = self.branch_store.active_branch_name()
        if self.branch_store.branch_exists(name):
            self.console.print(f"[yellow]Branch '{name}' ya existe.[/yellow]")
        else:
            self.branch_store.create_branch(
                name=name,
                parent_branch=current,
                parent_session_id=self._current_session_id or "",
            )
            self.console.print(f"[green]Branch '{name}' creado desde '{current}'.[/green]")
        self.branch_store.set_active_branch(name)
        self._current_session_id = self.branch_store.active_session_for_branch(name)
        session_label = self._current_session_id or "nueva"
        self.console.print(
            f"[green]→ Ahora en branch '[bold]{name}[/bold]'.  Sesión: {session_label}[/green]"
        )

    def _switch_branch(self, name: str) -> None:
        if not self.branch_store.branch_exists(name):
            self.console.print(
                f"[red]Branch '{name}' no existe.[/red] "
                f"Usa [cyan]/branch new {name}[/cyan] para crearlo."
            )
            return
        self.branch_store.set_active_branch(name)
        self._current_session_id = self.branch_store.active_session_for_branch(name)
        session_label = self._current_session_id or "nueva"
        self.console.print(
            f"[green]→ Cambiado a branch '[bold]{name}[/bold]'.  Sesión: {session_label}[/green]"
        )

    def _do_merge(
        self,
        source: str,
        strategy: str = "union-deduplicate",
        destination: str | None = None,
    ) -> None:
        # Honor an explicit destination from the NL phrase ("mezcla X en main").
        # Fall back to the active branch when the phrase has no destination suffix.
        dest = destination or self.branch_store.active_branch_name()
        if source == dest:
            self.console.print("[yellow]No puedes hacer merge de una rama consigo misma.[/yellow]")
            return
        if not self.branch_store.branch_exists(source):
            self.console.print(f"[red]Branch fuente '{source}' no existe.[/red]")
            return
        valid = {"union-deduplicate", "prefer-source", "prefer-destination"}
        if strategy not in valid:
            self.console.print(
                f"[red]Estrategia '{strategy}' inválida. Válidas: {', '.join(sorted(valid))}[/red]"
            )
            return
        self.console.print(f"[cyan]Mergeando '{source}' → '{dest}' [{strategy}]…[/cyan]")
        try:
            record, new_session = merge_branches(
                source_branch=source,
                destination_branch=dest,
                branch_store=self.branch_store,
                echo_store=self.agent.store,
                strategy=strategy,
            )
            self._current_session_id = new_session.id
            self._print_op_result(record)
        except ValueError as exc:
            self.console.print(f"[red]Error en merge:[/red] {exc}")

    def _do_cherry_pick(
        self,
        source: str,
        artefact_types: list[str],
        destination: str | None = None,
    ) -> None:
        dest = destination or self.branch_store.active_branch_name()
        if source == dest:
            self.console.print(
                "[yellow]No puedes cherry-pick de una rama consigo misma.[/yellow]"
            )
            return
        if not self.branch_store.branch_exists(source):
            self.console.print(f"[red]Branch fuente '{source}' no existe.[/red]")
            return
        self.console.print(
            f"[cyan]Cherry-pick '{source}' → '{dest}': {', '.join(artefact_types)}…[/cyan]"
        )
        try:
            record, new_session = cherry_pick(
                source_branch=source,
                destination_branch=dest,
                branch_store=self.branch_store,
                echo_store=self.agent.store,
                artefact_types=artefact_types,
            )
            self._current_session_id = new_session.id
            self._print_op_result(record, is_cherry_pick=True)
        except ValueError as exc:
            self.console.print(f"[red]Error en cherry-pick:[/red] {exc}")

    def _print_op_result(
        self, record: BranchMergeRecord, is_cherry_pick: bool = False
    ) -> None:
        op = "Cherry-pick" if is_cherry_pick else "Merge"
        total_in = sum(len(v) for v in record.items_merged.values())
        total_conf = sum(len(v) for v in record.conflicts.values())
        lines = [
            f"[green]{op} completado.[/green]  ID: [dim]{record.merge_id}[/dim]",
            f"[dim]{record.source_branch} → {record.destination_branch}  strategy={record.strategy}[/dim]",
            f"Items incorporados: {total_in}",
        ]
        for atype, items in record.items_merged.items():
            lines.append(f"  [cyan]{atype}[/cyan]: {len(items)}")
        if total_conf:
            lines.append(f"[yellow]Conflictos: {total_conf}[/yellow]")
        lines.append(f"Nueva sesión: [dim]{record.destination_session_id}[/dim]")
        self.console.print(
            Panel("\n".join(lines), title=f"{op}", border_style="green", expand=False)
        )

    # ------------------------------------------------------------------
    # Natural language intent dispatch
    # ------------------------------------------------------------------

    def _handle_natural_intent(self, intent: str, values: dict) -> bool:
        """Return True if the intent was handled (not a conversation turn)."""

        if intent == "branch_new":
            name = values.get("name", "")
            if name:
                self._create_and_switch_branch(name)
                return True

        elif intent == "branch_switch":
            name = values.get("name", "")
            if name:
                self._switch_branch(name)
                return True

        elif intent == "branch_list":
            self._cmd_branch_list()
            return True

        elif intent == "branch_status":
            self._handle_branch_cmd(["status"])
            return True

        elif intent == "branch_show":
            name = values.get("name", "")
            if name:
                self._handle_branch_cmd(["show", name])
                return True

        elif intent == "branch_merge":
            source = values.get("source", "")
            destination = values.get("destination") or None
            if source:
                self._do_merge(source, destination=destination)
                return True

        elif intent == "branch_cherry_pick":
            source = values.get("source", "")
            artefacts: list[str] = values.get("artefacts") or []
            destination = values.get("destination") or None
            if not artefacts:
                artefacts = ["decisions", "findings"]
            if source:
                self._do_cherry_pick(source, artefacts, destination=destination)
                return True

        elif intent == "session_status":
            self._handle_session_cmd(["status"])
            return True

        elif intent == "session_new":
            self._handle_session_cmd(["new"])
            return True

        elif intent == "exit":
            self.console.print("Saliendo. Hasta luego.")
            sys.exit(0)

        elif intent == "help":
            self.console.print(Panel(_HELP_TEXT, title="Ayuda", border_style="cyan"))
            return True

        return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._print_header()
        while True:
            text = self._prompt_line()
            if text is None:
                self.console.print("\nSaliendo. Hasta luego.")
                break
            text = text.strip()
            if not text:
                continue

            # Slash commands take priority
            if text.startswith("/"):
                self._handle_slash(text)
                continue

            # Natural language routing — pass active branch so contextual
            # phrases like "mezcla esta rama en main" can be resolved.
            active_branch = self.branch_store.active_branch_name()
            intent, values = route(text, active_branch=active_branch)
            if intent != "conversation":
                if self._handle_natural_intent(intent, values):
                    continue

            # Default path: conversation turn with the agent
            answer = self._run_agent_turn(text)
            if answer:
                self.console.print()
                self.console.print(f"[bold bright_cyan]Echo:[/bold bright_cyan] {answer}")
                self.console.print()
