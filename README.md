# EchoAgent

[![CI](https://github.com/SilverDreams115/EchoAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/SilverDreams115/EchoAgent/actions/workflows/ci.yml)

Echo is a local coding agent CLI focused on grounded repo inspection, real tool execution, resumable sessions, backend-aware routing, and honest degradation when the model or backend is weak.

The primary interface is a **conversational REPL** — type naturally, no subcommands required.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

echo-agent          # opens the conversational REPL directly
```

Inside the REPL:

```
[main] > revisa el runtime y dime qué falla
Echo: <respuesta del agente>

[main] > crea una rama experimento-shell
→ Branch 'experimento-shell' creado desde 'main'. Sesión: nueva

[experimento-shell] > en esta rama intenta rediseñar el shell
Echo: <respuesta del agente>

[experimento-shell] > vuelve a main
→ Cambiado a branch 'main'. Sesión: session-abc123

[main] > trae las decisiones de experimento-shell
Cherry-pick 'experimento-shell' → 'main': decisions, findings…
```

## What It Does

- inspects real project files before answering
- executes real tools for file reads, patches, shell commands, validation, and git inspection
- plans work in explicit stages with tracked progress, retries, replans, and stage-aware summaries
- keeps per-repo operational state under `.echo/`
- persists layered memory for working context, episodic events, operational summary, and cold session history
- supports Ollama and OpenAI-compatible backends
- uses `local`, `balanced`, and `deep` execution profiles
- preserves resumable working state with compression and memory summaries
- degrades cleanly when the backend is unavailable or unstable
- avoids repeating the same improvement list in local fallback follow-ups by remembering proposals already emitted per session/branch
- escalates local fallback suggestions progressively from base hygiene to structural and product-oriented improvements
- verifies final answers against observed repo evidence, tool output, and executed validation
- maintains **conversational branches** with merge and cherry-pick of structured context

## Conversational REPL

`echo-agent` opens a REPL directly — no subcommands, no menus.

```
╭──────────────────────────── EchoAgent · my-project ───────────────────────────╮
│ branch main   session resumida · a1b2c3d4                                     │
│ enter envia  ·  esc+enter nueva linea  ·  /help  ·  ctrl+d salir              │
╰───────────────────────────────────────────────────────────────────────────────╯

╭─ you main
╰─› 
```

The session and active branch are loaded automatically from `.echo/`. Each turn is persisted and linked to the previous one via `parent_session_id`.

When the backend falls back to local inspection, Echo also remembers which improvement proposals were already shown in the current session/branch. Follow-ups such as `otra propuesta`, `otra aparte de esas 4` or `algo más avanzado` produce new grounded suggestions instead of replaying the same base list.

### Input model

| Key | Action |
|---|---|
| `Enter` | **Envía el mensaje** — comportamiento principal de chat |
| `Alt+Enter` (o `Esc+Enter`) | Inserta una nueva línea dentro del mensaje — útil para prompts largos y estructurados |
| `Ctrl+D` | Sale del REPL |

Multiline example:
```
[main] > analiza el módulo de runtime:
  ↳ - enfócate en los stages y el retry logic
  ↳ - dime si hay deuda técnica obvia
```
*(Alt+Enter agrega cada `↳`; Enter al final envía todo como un solo mensaje)*

The `session:` field in the header shows:
- `nueva` — no prior session found for this branch (fresh start)
- `resumida · <short-id>` — an existing session was restored automatically from `.echo/`

### Slash Commands

```
/help
/exit
/session status
/session new
/branch status
/branch list
/branch new <name>
/branch switch <name>
/branch show <name>
/branch merge <source> [--strategy union-deduplicate|prefer-source|prefer-destination]
/branch cherry-pick <source> [--decisions] [--findings] [--pending] [--facts] [--summary]
/doctor
```

### Natural Language Routing

Branch operations can be expressed naturally:

| Phrase | Action |
|---|---|
| `crea una rama experimento-shell` | `/branch new experimento-shell` |
| `nueva rama feature-x` | `/branch new feature-x` |
| `vuelve a main` | `/branch switch main` |
| `cambia a feature-x` | `/branch switch feature-x` |
| `merge experimento-shell` | `/branch merge experimento-shell` |
| `trae las decisiones de experimento-shell` | `/branch cherry-pick experimento-shell --decisions` |
| `dame los findings de feature-x` | `/branch cherry-pick feature-x --findings` |
| `quiero los errores de feature-x` | `/branch cherry-pick feature-x --errors` |
| `trae todo de feature-x` | `/branch merge feature-x` (full merge) |
| `pásame todo de feature-x` | `/branch merge feature-x` (full merge) |
| `extrae las decisiones y findings de feature-x` | `/branch cherry-pick feature-x --decisions --findings` |
| `lista de ramas` | `/branch list` |
| `status` | `/session status` |

### Contextual references to the active branch

The router understands phrases that refer to the current branch without naming it explicitly:

| Phrase | Resolves to | Action |
|---|---|---|
| `mezcla esta rama en main` | active branch → main | full merge |
| `fusiona la rama actual con main` | active branch → main | full merge |
| `haz merge de la actual a main` | active branch → main | full merge |
| `incorpora todo de esta rama en main` | active branch → main | full merge |
| `combina esta con main` | active branch → main | full merge |
| `mezcla esta en main` | active branch → main | full merge |
| `trae todo de esta rama a main` | active branch → main | full merge |
| `trae las decisiones de esta rama` | active branch | cherry-pick decisions → active branch |
| `pásame los findings de la rama actual` | active branch | cherry-pick findings → active branch |
| `incorpora únicamente el summary de la actual` | active branch | cherry-pick summary → active branch |
| `trae las decisiones de esta rama a main` | active branch → main | cherry-pick decisions → main |
| `pásame los findings de la actual a main` | active branch → main | cherry-pick findings → main |
| `incorpora únicamente el summary de esta rama en main` | active branch → main | cherry-pick summary → main |
| `trae facts y findings desde esta rama hacia main` | active branch → main | cherry-pick facts + findings → main |

**Supported contextual phrases:** `esta rama`, `esta rama activa`, `la rama actual`, `la rama activa`, `la actual`, `current branch`, `active branch`, `this branch`, `current one`, `esta`.

**Merge with contextual source requires an explicit destination** ("en main", "a main", etc.) when the user is on the source branch — otherwise the destination defaults to the active branch (which would be a self-merge).

**Cherry-pick with contextual source** supports an optional explicit destination ("a main", "hacia main", "en main"). Without one, the destination defaults to the current active branch (standard cherry-pick model). The active branch does NOT change after cherry-pick — only the target branch's session is updated.

### Cherry-pick — component order flexibility

The router accepts multiple natural orderings of source, destination, and artefacts. All of the following are equivalent:

| Order | Example |
|---|---|
| artefacts → source → dest | `trae las decisiones de feature-x a main` |
| source → dest → artefacts | `trae de feature-x a main solo decisions` |
| dest → artefacts → source | `trae a main las decisiones de feature-x` |
| command + dest + artefacts | `cherry-pick de feature-x a main solo decisions` |

The same flexibility applies with contextual source references:

```
trae de esta rama a main solo findings
pásame de la actual a main facts y findings
trae a main las decisiones de esta rama
dame a main solo findings de la actual
```

Everything else goes directly to the agent as a conversation turn.

## Branches

Each branch is independent. Sessions are tracked per branch under `.echo/branches/`.

```bash
echo-agent

[main] > /branch new experiment
→ Branch 'experiment' creado desde 'main'. Sesión: nueva

[experiment] > /branch list
  * experiment  (active)
    main

[experiment] > /branch switch main
→ Cambiado a branch 'main'.

[main] > /branch show experiment
╭─ Branch: experiment ──────────────────────╮
│ created: 2026-03-27T...                   │
│ parent:  main                             │
│ sessions: 3                               │
│ merges:  1                                │
╰───────────────────────────────────────────╯
```

## Merge Between Branches

Merge operates on **structured artefacts**, not raw transcripts. The mergeable types are:

`decisions` · `findings` · `pending` · `facts` · `summary` · `errors` · `changes`

```bash
# While on main:
/branch merge experiment

# Or with an explicit strategy:
/branch merge experiment --strategy prefer-source
```

**Strategies:**

| Strategy | Behavior |
|---|---|
| `union-deduplicate` (default) | Union of both branches, deduplicated (case-insensitive) |
| `prefer-source` | Source items first; destination adds non-duplicates |
| `prefer-destination` | Destination intact; only new items from source are added |

Merge creates a new session in the destination branch (`parent_session_id` links to the previous one). The agent can resume from this session with the full merged context. The merge is recorded in `.echo/branches/<dest>/merges/` for auditability.

## Cherry-Pick Between Branches

Bring only specific artefact types from another branch:

```bash
/branch cherry-pick experiment --decisions --findings
/branch cherry-pick experiment --pending --facts
```

Or naturally:

```
trae las decisiones de experimento-shell
```

Cherry-pick does **not** mutate the source branch. Only the requested types are incorporated; all other artefact slots in the destination remain unchanged.

## Security Policy for Mutating Actions

The agent follows a conservative policy for operations that modify the repo:

- `write_file` and `apply_patch` require `ECHO_ALLOW_WRITE=true` (default: off in strict mode)
- `run_shell` is constrained to an explicit safe list of executables — no shell metacharacters, no `shell=True`, no destructive commands
- Grounding validation rejects answers that claim file changes without proof in the tool log
- Branch operations (merge, cherry-pick) only modify `.echo/` state — they never touch project files

## Profiles

- `local`: local-first, practical, grounded, best when Ollama is healthy
- `balanced`: stronger defaults for non-trivial asks, with hybrid routing when fallback exists
- `deep`: prefers a frontier backend and can enforce strict backend availability

```bash
echo-agent --profile balanced
echo-agent --profile deep
echocode ask --profile balanced "analiza runtime y routing"
echocode ask --profile deep --strict-profile "haz análisis profundo"
```

## Backend Setup

### Ollama

```bash
ECHO_BACKEND=ollama \
ECHO_MODEL=qwen2.5-coder:7b-oh \
echo-agent
```

### OpenAI-Compatible

```bash
ECHO_BACKEND=openai-compatible \
ECHO_MODEL=gpt-4.1-mini \
ECHO_OPENAI_API_KEY=YOUR_KEY \
echo-agent
```

### Strict Deep Profile

```bash
ECHO_OPENAI_API_KEY=YOUR_KEY \
echocode smoke --profile deep --strict-profile "haz una verificación breve del backend"
```

## Legacy Subcommands

`echocode` subcommands still work for scripting and CI:

```bash
echocode doctor
echocode backend-check --chat-samples 2
echocode ask "inspecciona echo/runtime/engine.py y explica el flujo"
echocode plan "refuerza grounding y runtime"
echocode resume "continúa desde la última sesión"
echocode shell       # TUI multi-panel (alternative to echo-agent)
echocode smoke "valida doctor, ask y resume"
```

## Validation

Echo detects a validation strategy from the real repo layout instead of assuming `unittest`.

- `pytest` when the repo has `pytest.ini`, `conftest.py`, or pytest config in `pyproject.toml`
- `unittest` when the repo follows a standard `tests/test_*.py` layout without pytest-specific config
- `npm/pnpm/yarn test|lint|typecheck` when `package.json` exposes a safe script
- `compileall` as a Python fallback when there are Python sources but no runnable test layout
- `unknown` when no safe validation strategy can be proven from repo evidence

```bash
python3 -m pytest tests/
echocode smoke "diagnóstico breve del backend actual"
```

## Execution Model

Echo runs around a small staged runtime instead of a single free-form turn.

- `inspect`: establish objective, current stage, active files, and evidence
- `execute`: read, patch, and reason with tool-backed evidence
- `verify`: run validation for the detected stack or report why no safe validation exists
- `close`: summarize the real stage outcomes and persist layered memory for `resume`

The planner and runtime share the same stage model, so a failed stage is recorded as `failed` or `replanned` instead of silently disappearing.

## Memory Model

Echo persists compact operational memory instead of replaying the whole transcript.

- `working memory`: objective, current stage, active files, recent tools, recent evidence
- `episodic memory`: decisions, errors, retries, replans, validations, changes
- `operational summary`: confirmed facts, restrictions, stage progress, pending items
- `cold memory`: long-session persistence without re-injecting irrelevant history

Branch merge and cherry-pick operate directly on these memory layers — the merged session carries the consolidated artefacts forward as context for the next agent turn.

## Operational Notes

- Echo stores runtime state, logs, memory, sessions, and branch metadata under `.echo/`.
- `.gitignore` should exclude `.echo/` along with virtual environments and local secrets.
- `doctor` shows cached, fresh, and effective backend state so routing decisions are auditable.
- Session artifacts persist a `runtime_trace` with phase timings, backend request outcomes, retry count, grounding outcome, and remaining time budget.
- Simple `ask` requests that target one or two explicit files are slimmed before backend dispatch.
- Shell execution is constrained to a safe policy: no shell metacharacters, no `shell=True`, no destructive executables.

## Limitations

- Natural language routing uses regex rules — complex or ambiguous sentences may fall through to conversation (safe default).
- Cherry-pick from natural language defaults to `decisions + findings` when no artefact keywords are found; use the slash command for precise control.
- Forma 4 ("trae a DEST artefacts de SOURCE") requires the source to be a valid ASCII branch name at the end of the phrase; branches with accented characters won't match and fall safely to conversation.
- Branch merge/cherry-pick do not merge raw transcripts, tool call logs, or message history — only structured artefacts.
- Contextual phrases like "esta rama" require `active_branch` context from the REPL; without it they fall through safely to conversation.
- "esto" (without "rama") is not a contextual ref — it can match as a branch name if a branch literally named "esto" exists.
