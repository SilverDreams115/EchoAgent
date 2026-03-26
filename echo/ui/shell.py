from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
import time
from pathlib import Path

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from echo.core import EchoAgent
from echo.types import ActivityEvent


class EchoShell:
    def __init__(self, agent: EchoAgent, project_root: Path, console: Console) -> None:
        self.agent = agent
        self.project_root = project_root
        self.console = console
        self._queue: Queue[ActivityEvent] = Queue()
        self._events: list[ActivityEvent] = []
        self._last_prompt = ""
        self._last_answer = ""
        self.agent.activity.watch(self._queue.put)

    def _header(self) -> Panel:
        status = self.agent.current_status()
        text = Text()
        text.append("Echo Control Plane\n", style="bold bright_cyan")
        text.append(f"repo {self.project_root.name}  ", style="white")
        text.append(f"branch {status.get('branch', 'none')}  ", style="green")
        text.append(f"profile {status['profile']}  ", style="white")
        text.append(f"backend {status['backend']}  ", style="bold white")
        text.append(f"model {status['model']}  ", style="white")
        text.append(f"policy {status['routing_policy']}\n", style="white")
        text.append(f"session {status['session']}  ", style="white")
        text.append(f"context free {status['context_free']}  ", style="green")
        text.append(f"backend state {status['backend_state']}", style="yellow")
        return Panel(text, title="Session", border_style="cyan")

    def _activity_table(self) -> Table:
        table = Table(expand=True)
        table.add_column("Proceso", style="cyan", no_wrap=True)
        table.add_column("Estado", style="magenta", no_wrap=True)
        table.add_column("Detalle", style="white")
        for event in self._events[-12:]:
            detail = (event.message + " | " + event.detail).strip(" |")
            table.add_row(event.stage, event.status, detail[:96])
        if not self._events:
            table.add_row("Echo", "idle", "Sin actividad todavía")
        return table

    def _conversation_panel(self, prompt_text: str, final_text: str) -> Panel:
        body = Text()
        body.append("User\n", style="bold white")
        body.append((prompt_text.strip() or "Sin prompt todavía") + "\n\n", style="white")
        body.append("Echo\n", style="bold bright_cyan")
        body.append(final_text.strip() or "Esperando instrucción...", style="white")
        return Panel(body, title="Conversación", border_style="bright_black")

    def _status_panel(self) -> Panel:
        status = self.agent.current_status()
        text = Text()
        text.append("/doctor  ", style="cyan")
        text.append("/ask  ", style="cyan")
        text.append("/plan  ", style="cyan")
        text.append("/resume  ", style="cyan")
        text.append("/memory  ", style="cyan")
        text.append("/branch  ", style="cyan")
        text.append("/status  ", style="cyan")
        text.append("/exit", style="cyan")
        text.append("\n")
        text.append(f"branch {status.get('branch', 'none')}  ", style="green")
        text.append(f"session {status['session']}  ", style="white")
        text.append(f"context free {status['context_free']}", style="green")
        return Panel(text, title="Estado", border_style="bright_black")

    def _focus_panel(self) -> Panel:
        run_state = self.agent.last_run_state
        body = Text()
        if not run_state or not run_state.focus_files:
            body.append("Sin archivos en foco todavía", style="white")
        else:
            for path in run_state.focus_files[-8:]:
                body.append(path + "\n", style="white")
        return Panel(body, title="Archivos en Foco", border_style="bright_black")

    def _validation_panel(self) -> Panel:
        run_state = self.agent.last_run_state
        body = Text()
        if not run_state or not run_state.validation_commands:
            body.append("Sin validación ejecutada todavía", style="white")
        else:
            for command in run_state.validation_commands[-4:]:
                body.append(command + "\n", style="white")
        if run_state:
            body.append(f"\nRetries: {run_state.retry_count}\n", style="yellow")
            body.append(f"Compression: {run_state.compression_count}\n", style="yellow")
            if run_state.fallback_used:
                body.append(f"Degraded: {run_state.fallback_reason}\n", style="red")
        if run_state and run_state.open_issues:
            body.append("\nIssues\n", style="bold red")
            for item in run_state.open_issues[-4:]:
                body.append(item + "\n", style="red")
        return Panel(body, title="Validation / Runtime", border_style="bright_black")

    def _backend_panel(self) -> Panel:
        run_state = self.agent.last_run_state
        body = Text()
        body.append(f"selected {self.agent.selected_backend_name}:{self.agent.selected_model}\n", style="bold white")
        if self.agent.fallback_backend_name:
            body.append(f"fallback {self.agent.fallback_backend_name}:{self.agent.fallback_model}\n", style="white")
        body.append(f"reason {self.agent.routing_reason}\n", style="white")
        if run_state:
            body.append(f"effective {run_state.backend_health.backend_state}\n", style="yellow")
            body.append(f"fresh {run_state.fresh_backend_health.backend_state}\n", style="yellow")
            body.append(f"cached {run_state.backend_health.source}\n", style="white")
        return Panel(body, title="Backend / Routing", border_style="bright_black")

    def _render_layout(self, prompt_text: str, final_text: str) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), size=5),
            Layout(name="body", ratio=1),
            Layout(Panel("Composer multilinea  Esc+Enter enviar  Enter nueva linea", title="Composer", border_style="bright_black"), size=3),
            Layout(Panel("Echo shell  /doctor /ask /plan /resume /memory /status /exit", border_style="bright_black"), size=3),
        )
        layout["body"].split_row(
            Layout(self._conversation_panel(prompt_text, final_text), ratio=7),
            Layout(name="side", ratio=3),
        )
        layout["body"]["side"].split_column(
            Layout(Panel(self._activity_table(), title="Activity Timeline", border_style="bright_black"), ratio=4),
            Layout(self._backend_panel(), ratio=2),
            Layout(self._focus_panel(), ratio=2),
            Layout(self._validation_panel(), ratio=2),
            Layout(self._status_panel(), ratio=2),
        )
        return layout

    def _multiline_input(self) -> str | None:
        history = FileHistory(str(self.project_root / ".echo" / "prompt_history.txt"))
        kb = KeyBindings()

        @kb.add("escape", "enter")
        def _(event):
            event.app.exit(result=event.app.current_buffer.text)

        @kb.add("c-d")
        def _(event):
            event.app.exit(result=None)

        return prompt(
            "echo> ",
            multiline=True,
            history=history,
            key_bindings=kb,
            prompt_continuation="... ",
        )

    def _run_with_live(self, prompt_text: str, mode: str) -> None:
        holder: dict[str, object] = {"done": False, "answer": "", "session_path": None, "error": None}

        def worker() -> None:
            try:
                active = self.agent.store.active_branch()
                branch_name = active.branch_name if active else None
                answer, session_path, session = self.agent.run(
                    prompt_text, mode=mode, branch_name=branch_name
                )
                holder["answer"] = answer
                holder["session_path"] = session_path
                holder["tool_calls"] = len(session.tool_calls)
            except Exception as exc:
                holder["error"] = exc
            finally:
                holder["done"] = True

        thread = Thread(target=worker, daemon=True)
        thread.start()

        with Live(self._render_layout(prompt_text, ""), console=self.console, refresh_per_second=8, transient=False) as live:
            while True:
                try:
                    while True:
                        event = self._queue.get_nowait()
                        self._events.append(event)
                except Empty:
                    pass
                final_text = str(holder.get("answer") or "") if holder.get("done") else "Trabajando..."
                if holder.get("error"):
                    final_text = f"Error: {holder['error']}"
                live.update(self._render_layout(prompt_text, final_text))
                if holder.get("done"):
                    break
                time.sleep(0.08)
        self._last_prompt = prompt_text
        self._last_answer = str(holder.get("answer") or "")

        if holder.get("session_path"):
            self.console.print(f"Sesión guardada en: {holder['session_path']}")
            self.console.print(f"Herramientas usadas: {holder.get('tool_calls', 0)}")

    def _handle_branch_command(self, text: str) -> None:
        """Handle /branch sub-commands inside the shell."""
        parts = text.split(maxsplit=2)
        sub = parts[1] if len(parts) > 1 else "status"
        arg = parts[2] if len(parts) > 2 else ""

        store = self.agent.store
        active = store.ensure_branch_model()

        if sub == "status":
            branch = store.load_branch(active.branch_name)
            lines = [f"active: {active.branch_name}"]
            if branch:
                lines.append(f"head_session: {branch.head_session_id or 'none'}")
                lines.append(f"parent: {branch.parent_branch or '—'}")
            self.console.print(Panel("\n".join(lines), title="Branch status", expand=False))

        elif sub == "list":
            branches = store.list_branches()
            lines = []
            for b in branches:
                marker = "* " if b.name == active.branch_name else "  "
                lines.append(f"{marker}{b.name}  head={b.head_session_id or 'none'}  parent={b.parent_branch or '—'}")
            self.console.print(Panel("\n".join(lines) or "No branches.", title="Branches", expand=False))

        elif sub == "new" and arg:
            name = arg.strip()
            if store.load_branch(name):
                self.console.print(f"[red]Branch '{name}' already exists.[/red]")
                return
            source = store.load_branch(active.branch_name)
            fork_head = source.head_session_id if source else ""
            from echo.types import BranchMeta
            new_branch = BranchMeta(
                name=name,
                head_session_id=fork_head,
                parent_branch=active.branch_name,
                fork_session_id=fork_head,
            )
            store.save_branch(new_branch)
            store.set_active_branch(name)
            self.console.print(f"[green]Branch '{name}' created and activated.[/green]")

        elif sub == "switch" and arg:
            name = arg.strip()
            if store.load_branch(name) is None:
                self.console.print(f"[red]Branch '{name}' not found.[/red]")
                return
            store.set_active_branch(name)
            self.console.print(f"[green]Switched to branch '{name}'.[/green]")

        else:
            self.console.print(
                "/branch status | /branch list | /branch new <name> | /branch switch <name>"
            )

    def run(self) -> None:
        self.console.print(self._header())
        self.console.print("/doctor, /ask <texto>, /plan <texto>, /resume [texto], /memory, /branch, /status, /exit")
        while True:
            text = self._multiline_input()
            if text is None:
                self.console.print("Saliendo.")
                break
            text = text.strip()
            if not text:
                continue
            if text in {"/exit", "exit", "quit"}:
                self.console.print("Saliendo.")
                break
            if text.startswith("/branch"):
                self._handle_branch_command(text)
                continue
            if text == "/doctor":
                data = self.agent.doctor()
                table = Table(title="Echo doctor")
                table.add_column("Check", style="cyan")
                table.add_column("Value", style="white")
                for key, value in data.items():
                    table.add_row(str(key), ", ".join(value) if isinstance(value, list) else str(value))
                self.console.print(table)
                continue
            if text == "/memory":
                latest = self.agent.store.latest_session_id()
                if latest:
                    session = self.agent.store.load_session(latest)
                    self.console.print(Panel(session.operational_summary or session.summary, title=f"Memoria {latest}", expand=True))
                else:
                    self.console.print("Sin sesiones todavía.")
                continue
            if text == "/status":
                status = self.agent.current_status()
                self.console.print(Panel("\n".join(f"{k}: {v}" for k, v in status.items()), title="Status", expand=False))
                continue
            if text.startswith("/plan "):
                self._events.clear()
                self._run_with_live(text[6:].strip(), mode="plan")
                continue
            if text.startswith("/ask "):
                self._events.clear()
                self._run_with_live(text[5:].strip(), mode="ask")
                continue
            if text == "/resume":
                self._events.clear()
                self._run_with_live("Continúa desde la última sesión y resume el estado actual.", mode="resume")
                continue
            if text.startswith("/resume "):
                self._events.clear()
                self._run_with_live(text[8:].strip(), mode="resume")
                continue
            self._events.clear()
            self._run_with_live(text, mode="ask")
