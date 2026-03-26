# EchoAgent

[![CI](https://github.com/SilverDreams115/EchoAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/SilverDreams115/EchoAgent/actions/workflows/ci.yml)

Echo is a local coding agent CLI focused on grounded repo inspection, real tool execution, resumable sessions, backend-aware routing, and honest degradation when the model or backend is weak.

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
- verifies final answers against observed repo evidence, tool output, and executed validation

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

```bash
echocode doctor
echocode backend-check --chat-samples 1
echocode shell
echocode ask "inspecciona este proyecto"
echocode plan "propón un cambio seguro"
echocode resume
```

## Profiles

- `local`: local-first, practical, grounded, best when Ollama is healthy.
- `balanced`: stronger defaults for non-trivial asks, with hybrid routing when fallback exists.
- `deep`: prefers a frontier backend and can enforce strict backend availability.

```bash
echocode doctor --profile local
echocode ask --profile balanced "analiza runtime y routing"
echocode ask --profile deep "haz análisis profundo"
echocode ask --profile deep --strict-profile "haz análisis profundo"
```

## Backend Setup

### Ollama

```bash
ECHO_BACKEND=ollama \
ECHO_MODEL=qwen2.5-coder:7b-oh \
echocode doctor
```

### OpenAI-Compatible

```bash
ECHO_BACKEND=openai-compatible \
ECHO_MODEL=gpt-4.1-mini \
ECHO_OPENAI_API_KEY=YOUR_KEY \
echocode doctor
```

### Strict Deep Profile

```bash
ECHO_OPENAI_API_KEY=YOUR_KEY \
echocode smoke --profile deep --strict-profile "haz una verificación breve del backend"
```

## Validation

Echo detects a validation strategy from the real repo layout instead of assuming `unittest`.

- `pytest` when the repo has `pytest.ini`, `conftest.py`, or pytest config in `pyproject.toml`
- `unittest` when the repo follows a standard `tests/test_*.py` layout without pytest-specific config
- `npm/pnpm/yarn test|lint|typecheck` when `package.json` exposes a safe script
- `compileall` as a Python fallback when there are Python sources but no runnable test layout
- `unknown` when no safe validation strategy can be proven from repo evidence

```bash
PYTHONPATH=. python3 -m unittest tests.test_runtime_flows
python -m unittest discover -s tests -p 'test_*.py'
echocode smoke "diagnóstico breve del backend actual"
```

## Operational Notes

- Echo stores runtime state, logs, memory, and session artifacts under `.echo/`.
- `.gitignore` excludes `.echo/`, virtual environments, caches, and local secrets.
- For critical `ask` workflows, use a healthy OpenAI-compatible fallback if Ollama chat is unstable.
- `doctor` shows cached, fresh, and effective backend state so routing decisions are auditable.
- `doctor`, `backend-check`, `ask`, `plan`, `resume`, and `smoke` share the same normalized backend health model.
- Final answer grounding rejects unsupported file, symbol, command, change, and validation claims.
- Shell execution is constrained to a safe policy: no shell metacharacters, no `shell=True`, and no destructive executables.
- Session artifacts persist a `runtime_trace` with phase timings, backend request outcomes, retry count, grounding outcome, and remaining time budget.
- Simple `ask` requests that target one or two explicit files are slimmed before backend dispatch: fewer snippet lines, no repo map, and no extra tool/stage guidance when that context would only add latency.

## Execution Model

Echo runs around a small staged runtime instead of a single free-form turn.

- `inspect`: establish objective, current stage, active files, and evidence
- `execute`: read, patch, and reason with tool-backed evidence
- `verify`: run validation for the detected stack or report why no safe validation exists
- `close`: summarize the real stage outcomes and persist layered memory for `resume`

The planner and runtime share the same stage model, so a failed stage is recorded as `failed` or `replanned` instead of silently disappearing.

The runtime is intentionally split into small owners instead of a single giant loop.

- `prepare.py`: resume seeding, inspection seeding, intake message construction, preflight decision
- `model_loop.py`: request loop, retry gating, grounding retry, honest degradation after budget exhaustion
- `backend_runtime.py`: backend request dispatch, timeout wrapping, backend activity, backend log mutation, request outcome normalization
- `verify_flow.py`: auto-verify handoff from runtime into detected project validation
- `finalize.py`: session closing, summary/materialization, artifact persistence
- `trace.py`: runtime phase timing and compact request trace persistence

## Memory Model

Echo persists compact operational memory instead of replaying the whole transcript.

- `working memory`: objective, current stage, active files, recent tools, recent evidence
- `episodic memory`: decisions, errors, retries, replans, validations, changes
- `operational summary`: confirmed facts, restrictions, stage progress, pending items
- `cold memory`: long-session persistence without re-injecting irrelevant history

## Runtime Trace

Each persisted session keeps a compact runtime trace so slow or degraded runs stay auditable without replaying the full transcript.

- high-level phases: `prepare`, `preflight`, `execute`, `verify`, `finalize`
- backend request trace: message count, role counts, effective timeout, tools enabled, request duration, outcome
- request-shape fields: total chars, repo-map included or omitted, focus snippets present, compressed context, resumed context
- final runtime outcome: retry count, degraded vs grounded, remaining budget, terminal reason

This trace is written both into the session JSON and the runtime artifact under `.echo/artifacts/`.

## Useful Commands

```bash
echocode doctor
echocode backend-check --chat-samples 2
echocode ask "inspecciona echo/runtime/engine.py y explica el flujo"
echocode plan "refuerza grounding y runtime"
echocode resume "continúa desde la última sesión"
echocode smoke "valida doctor, ask y resume"
```
