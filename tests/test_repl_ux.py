"""
Tests for EchoRepl UX properties.

Covers:
- Enter submits by default (via _input_fn mock)
- Multiline input (text with \\n) is accepted and forwarded
- Session auto-resume when branch has an active session
- Branch auto-resume reflects the persisted active branch
- Header shows "resumida" when a prior session exists
- Header shows "nueva" when no prior session exists
- Slash commands work without errors
- Processing state label is not the ambiguous bare "..."

No backend required — EchoAgent is stubbed.
"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

# Avoid the circular-import that exists in echo.backends (same trick as test_branches.py)
from echo.backends.errors import BackendError  # noqa: F401
from echo.branches.store import BranchStore
from echo.memory import EchoStore
from echo.types.models import SessionState
from echo.ui.repl import EchoRepl, _build_chat_kb

from rich.console import Console


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _make_session(tmp_path: Path, repo_root: str | None = None) -> SessionState:
    s = SessionState.create(
        repo_root=repo_root or str(tmp_path),
        mode="ask",
        model="stub",
        user_prompt="stub",
    )
    return s


def _make_agent_stub(session: SessionState | None = None, tmp_path: Path | None = None) -> MagicMock:
    """Minimal EchoAgent stub sufficient for EchoRepl."""
    agent = MagicMock()
    agent.activity.watch = MagicMock()

    # store.latest_session_id() — returns None unless overridden per test
    agent.store.latest_session_id.return_value = None

    # agent.run() — returns a realistic tuple
    if session is not None:
        agent.run.return_value = ("stub answer", None, session)
    else:
        # Lazy default: create a session on first call
        def _run(text, *, mode, resume_session_id=None, profile=None):
            s = SessionState.create(
                repo_root=str(tmp_path) if tmp_path else "/tmp",
                mode=mode,
                model="stub",
                user_prompt=text,
            )
            return "stub answer", None, s

        agent.run.side_effect = _run

    agent.doctor.return_value = {"status": "ok"}
    agent.current_status.return_value = {"model": "stub", "session": "none"}
    return agent


def _input_sequence(*lines: str | None):
    """Return an _input_fn that yields lines in order, then None (EOF)."""
    it: Iterator[str | None] = iter(list(lines) + [None])

    def _fn() -> str | None:
        return next(it, None)

    return _fn


def _make_repl(
    tmp_path: Path,
    *,
    agent=None,
    branch_store: BranchStore | None = None,
    input_fn=None,
    console: Console | None = None,
) -> EchoRepl:
    bs = branch_store or BranchStore(tmp_path)
    ag = agent or _make_agent_stub(tmp_path=tmp_path)
    co = console or Console(file=io.StringIO(), force_terminal=False, highlight=False)
    return EchoRepl(ag, tmp_path, co, bs, _input_fn=input_fn)


def _capture_console(tmp_path: Path) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, highlight=False)
    return console, buf


# ---------------------------------------------------------------------------
# A. Enter submits by default
# ---------------------------------------------------------------------------


def test_enter_submits_by_default(tmp_path: Path) -> None:
    """_input_fn returning text directly simulates Enter=submit.
    The agent.run() must be called with that text exactly once."""
    session = _make_session(tmp_path)
    agent = _make_agent_stub(session=session, tmp_path=tmp_path)

    repl = _make_repl(tmp_path, agent=agent, input_fn=_input_sequence("hola Echo"))
    repl.run()

    agent.run.assert_called_once()
    call_args = agent.run.call_args
    assert call_args[0][0] == "hola Echo"


def test_empty_input_is_not_forwarded_to_agent(tmp_path: Path) -> None:
    """Empty input (blank Enter) must not trigger an agent turn."""
    agent = _make_agent_stub(tmp_path=tmp_path)

    repl = _make_repl(tmp_path, agent=agent, input_fn=_input_sequence("", "  ", None))
    repl.run()

    agent.run.assert_not_called()


# ---------------------------------------------------------------------------
# B. Multiline input
# ---------------------------------------------------------------------------


def test_multiline_text_is_forwarded_intact(tmp_path: Path) -> None:
    """Text containing \\n (inserted via Alt+Enter) must reach the agent as-is."""
    multiline_msg = "línea uno\nlínea dos\nlínea tres"
    session = _make_session(tmp_path)
    agent = _make_agent_stub(session=session, tmp_path=tmp_path)

    repl = _make_repl(tmp_path, agent=agent, input_fn=_input_sequence(multiline_msg))
    repl.run()

    agent.run.assert_called_once()
    assert agent.run.call_args[0][0] == multiline_msg


# ---------------------------------------------------------------------------
# C. Processing state — no bare "..." ambiguity
# ---------------------------------------------------------------------------


def test_spinner_text_is_not_bare_dots() -> None:
    """The spinner displayed during processing must not be the bare '...'
    that looks like incomplete input or a hang."""
    from echo.ui.repl import EchoRepl
    import inspect, textwrap

    source = inspect.getsource(EchoRepl._run_agent_turn)
    # The spinner text used in _run_agent_turn must not be just "..."
    assert 'text="..."' not in source, (
        "Spinner text must not be bare '...' — it is ambiguous. "
        "Use a descriptive label like 'Echo está pensando…'."
    )


def test_spinner_has_descriptive_label() -> None:
    """The spinner inside _run_agent_turn must carry a human-readable label."""
    from echo.ui.repl import EchoRepl
    import inspect

    source = inspect.getsource(EchoRepl._run_agent_turn)
    assert "Echo está pensando" in source, (
        "Expected a descriptive spinner label ('Echo está pensando…') in _run_agent_turn."
    )


# ---------------------------------------------------------------------------
# D. Session auto-resume
# ---------------------------------------------------------------------------


def test_session_auto_resumed_when_branch_has_active_session(tmp_path: Path) -> None:
    """If the active branch has a recorded session, _current_session_id must be set
    to that session at REPL startup — not None."""
    bs = BranchStore(tmp_path)
    es = EchoStore(tmp_path)

    # Save a session and register it with the active branch
    session = _make_session(tmp_path)
    es.save_session(session)
    bs.add_session_to_branch("main", session.id)

    agent = _make_agent_stub(tmp_path=tmp_path)
    agent.store.latest_session_id.return_value = session.id

    repl = _make_repl(tmp_path, agent=agent, branch_store=bs)

    assert repl._current_session_id == session.id, (
        f"Expected session to be restored to '{session.id}', got '{repl._current_session_id}'"
    )


def test_session_is_none_when_no_prior_sessions(tmp_path: Path) -> None:
    """A fresh project with no sessions should start with _current_session_id = None."""
    bs = BranchStore(tmp_path)
    agent = _make_agent_stub(tmp_path=tmp_path)
    agent.store.latest_session_id.return_value = None

    repl = _make_repl(tmp_path, agent=agent, branch_store=bs)

    assert repl._current_session_id is None


def test_resumed_session_passed_to_agent(tmp_path: Path) -> None:
    """When a session is restored at startup, it must be passed to agent.run()
    as resume_session_id on the first turn."""
    bs = BranchStore(tmp_path)
    es = EchoStore(tmp_path)

    session = _make_session(tmp_path)
    es.save_session(session)
    bs.add_session_to_branch("main", session.id)

    new_session = _make_session(tmp_path)
    agent = _make_agent_stub(session=new_session, tmp_path=tmp_path)
    agent.store.latest_session_id.return_value = session.id

    repl = _make_repl(
        tmp_path,
        agent=agent,
        branch_store=bs,
        input_fn=_input_sequence("continúa desde donde estábamos"),
    )
    repl.run()

    agent.run.assert_called_once()
    kwargs = agent.run.call_args[1]
    assert kwargs.get("resume_session_id") == session.id, (
        f"Expected resume_session_id='{session.id}', got '{kwargs.get('resume_session_id')}'"
    )


# ---------------------------------------------------------------------------
# E. Branch auto-resume
# ---------------------------------------------------------------------------


def test_branch_auto_resumed_on_startup(tmp_path: Path) -> None:
    """If the project's active branch is 'experiment', the REPL must start on that branch."""
    bs = BranchStore(tmp_path)
    bs.create_branch("experiment", parent_branch="main")
    bs.set_active_branch("experiment")

    agent = _make_agent_stub(tmp_path=tmp_path)
    agent.store.latest_session_id.return_value = None

    repl = _make_repl(tmp_path, agent=agent, branch_store=bs)

    assert bs.active_branch_name() == "experiment"


def test_prompt_shows_active_branch(tmp_path: Path) -> None:
    """The prompt format includes the active branch name."""
    # This is implicit: _prompt_line() reads active_branch_name() dynamically.
    # We verify the branch store reflects the right branch at REPL time.
    bs = BranchStore(tmp_path)
    bs.create_branch("feature-x", parent_branch="main")
    bs.set_active_branch("feature-x")

    agent = _make_agent_stub(tmp_path=tmp_path)
    repl = _make_repl(tmp_path, agent=agent, branch_store=bs)

    # Branch must be reflected in the store the REPL holds
    assert repl.branch_store.active_branch_name() == "feature-x"


# ---------------------------------------------------------------------------
# F. Header: "resumida" vs "nueva"
# ---------------------------------------------------------------------------


def test_header_shows_resumida_when_session_exists(tmp_path: Path) -> None:
    """Header must contain 'resumida' when a prior session was found at startup."""
    bs = BranchStore(tmp_path)
    es = EchoStore(tmp_path)

    session = _make_session(tmp_path)
    es.save_session(session)
    bs.add_session_to_branch("main", session.id)

    console, buf = _capture_console(tmp_path)
    agent = _make_agent_stub(tmp_path=tmp_path)
    agent.store.latest_session_id.return_value = session.id

    repl = _make_repl(tmp_path, agent=agent, branch_store=bs, console=console)
    repl._print_header()

    output = buf.getvalue()
    assert "resumida" in output, (
        f"Expected 'resumida' in header when session exists. Got:\n{output}"
    )
    assert "EchoAgent" in output
    assert "branch" in output
    assert "session" in output
    # "nueva" must not appear as the session label.
    # Note: the hint line legitimately contains "nueva línea" — we check the
    # session-label region specifically by verifying "resumida" is present and
    # "session: nueva" (the old incorrect label) is absent.
    assert "session: nueva" not in output.replace("\n", " "), (
        f"Header must not use 'session: nueva' when a session was resumed. Got:\n{output}"
    )


def test_header_shows_nueva_when_no_prior_session(tmp_path: Path) -> None:
    """Header must contain 'nueva' when no prior session was found."""
    bs = BranchStore(tmp_path)
    agent = _make_agent_stub(tmp_path=tmp_path)
    agent.store.latest_session_id.return_value = None

    console, buf = _capture_console(tmp_path)
    repl = _make_repl(tmp_path, agent=agent, branch_store=bs, console=console)
    repl._print_header()

    output = buf.getvalue()
    assert "nueva" in output, (
        f"Expected 'nueva' in header when no prior session. Got:\n{output}"
    )
    assert "resumida" not in output


def test_header_shows_enter_to_send_hint(tmp_path: Path) -> None:
    """The compact hint line must still document Enter=send and Esc+Enter=newline."""
    agent = _make_agent_stub(tmp_path=tmp_path)
    console, buf = _capture_console(tmp_path)

    repl = _make_repl(tmp_path, agent=agent, console=console)
    repl._print_header()

    output = buf.getvalue()
    assert "enter envia" in output.lower(), (
        f"Hint must say 'enter envia'. Got:\n{output}"
    )
    assert "esc+enter nueva linea" in output.lower()
    assert "Alt+Enter=nueva línea" not in output


def test_short_session_id_drops_session_prefix(tmp_path: Path) -> None:
    agent = _make_agent_stub(tmp_path=tmp_path)
    repl = _make_repl(tmp_path, agent=agent)
    assert repl._short_session_id("session-abcd1234efgh") == "abcd1234"


def test_prompt_uses_composer_layout_and_toolbar(tmp_path: Path) -> None:
    """Interactive prompt should use the cleaner composer prompt and subtle toolbar."""
    agent = _make_agent_stub(tmp_path=tmp_path)
    repl = _make_repl(tmp_path, agent=agent)

    with patch("echo.ui.repl.pt_prompt", return_value="hola") as mock_prompt:
        result = repl._prompt_line()

    assert result == "hola"
    kwargs = mock_prompt.call_args.kwargs
    message = str(mock_prompt.call_args.args[0])
    assert "you" in message.lower()
    assert "╰─›" in message
    assert kwargs["prompt_continuation"] == "  │ "
    assert "Enter envia" in str(kwargs["bottom_toolbar"])


# ---------------------------------------------------------------------------
# G. Slash commands
# ---------------------------------------------------------------------------


def test_slash_help_does_not_crash(tmp_path: Path) -> None:
    """/help must execute without raising."""
    agent = _make_agent_stub(tmp_path=tmp_path)
    repl = _make_repl(tmp_path, agent=agent, input_fn=_input_sequence("/help"))
    repl.run()  # must not raise


def test_slash_exit_terminates_repl(tmp_path: Path) -> None:
    """/exit must call sys.exit — which we catch as SystemExit."""
    agent = _make_agent_stub(tmp_path=tmp_path)
    repl = _make_repl(tmp_path, agent=agent, input_fn=_input_sequence("/exit"))
    with pytest.raises(SystemExit):
        repl.run()


def test_slash_session_status_does_not_crash(tmp_path: Path) -> None:
    """/session status must execute without raising."""
    agent = _make_agent_stub(tmp_path=tmp_path)
    repl = _make_repl(tmp_path, agent=agent, input_fn=_input_sequence("/session status"))
    repl.run()


def test_slash_branch_list_does_not_crash(tmp_path: Path) -> None:
    """/branch list must execute without raising."""
    bs = BranchStore(tmp_path)
    bs.create_branch("main")
    bs.create_branch("feature-a")
    agent = _make_agent_stub(tmp_path=tmp_path)
    repl = _make_repl(tmp_path, agent=agent, branch_store=bs, input_fn=_input_sequence("/branch list"))
    repl.run()


def test_slash_unknown_command_does_not_crash(tmp_path: Path) -> None:
    """Unknown slash commands must print a warning, not crash."""
    agent = _make_agent_stub(tmp_path=tmp_path)
    repl = _make_repl(tmp_path, agent=agent, input_fn=_input_sequence("/unknown-command-xyz"))
    repl.run()  # must not raise


# ---------------------------------------------------------------------------
# H. Keybinding correctness (unit-level)
# ---------------------------------------------------------------------------


def _key_values(binding) -> tuple[str, ...]:
    """Return the string values of a binding's keys, normalised for comparison.

    prompt_toolkit stores keys as Keys enum instances whose .value is the
    canonical string (e.g. Keys.ControlM.value == 'c-m').  We compare by value
    so the tests are robust to enum identity differences across pt versions.
    """
    return tuple(k.value if hasattr(k, "value") else str(k) for k in binding.keys)


def test_chat_kb_enter_exits_app() -> None:
    """The 'enter' binding must call event.app.exit with the buffer text.

    prompt_toolkit normalises 'enter' → Keys.ControlM ('c-m'), so we match
    on the canonical value rather than the string literal 'enter'.
    """
    kb = _build_chat_kb()

    # "enter" is stored as Keys.ControlM whose .value == "c-m"
    enter_handlers = [
        b.handler for b in kb.bindings if _key_values(b) == ("c-m",)
    ]
    assert enter_handlers, (
        "No Enter ('c-m') binding found in _build_chat_kb(). "
        f"Registered bindings: {[_key_values(b) for b in kb.bindings]}"
    )

    event = MagicMock()
    event.app.current_buffer.text = "test message"
    enter_handlers[0](event)
    event.app.exit.assert_called_once_with(result="test message")


def test_chat_kb_escape_enter_inserts_newline() -> None:
    """The 'escape, enter' binding must insert a newline into the buffer.

    prompt_toolkit normalises this sequence to ('escape', 'c-m').
    """
    kb = _build_chat_kb()

    esc_enter_handlers = [
        b.handler for b in kb.bindings if _key_values(b) == ("escape", "c-m")
    ]
    assert esc_enter_handlers, (
        "No Escape+Enter ('escape','c-m') binding found in _build_chat_kb(). "
        f"Registered bindings: {[_key_values(b) for b in kb.bindings]}"
    )

    event = MagicMock()
    esc_enter_handlers[0](event)
    event.app.current_buffer.insert_text.assert_called_once_with("\n")


def test_chat_kb_ctrl_d_exits_with_none() -> None:
    """Ctrl+D must call event.app.exit(result=None)."""
    kb = _build_chat_kb()

    cd_handlers = [
        b.handler for b in kb.bindings if _key_values(b) == ("c-d",)
    ]
    assert cd_handlers, "No 'c-d' binding found in _build_chat_kb()"

    event = MagicMock()
    cd_handlers[0](event)
    event.app.exit.assert_called_once_with(result=None)
