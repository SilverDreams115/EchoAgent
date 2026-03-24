# EchoAgent

[![CI](https://github.com/SilverDreams115/EchoAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/SilverDreams115/EchoAgent/actions/workflows/ci.yml)

Echo is a local coding agent CLI focused on grounded repo inspection, real tool execution, resumable sessions, backend-aware routing, and honest degradation when the model or backend is weak.

## What It Does

- inspects real project files before answering
- executes real tools for file reads, patches, shell commands, validation, and git inspection
- keeps per-repo operational state under `.echo/`
- supports Ollama and OpenAI-compatible backends
- uses `local`, `balanced`, and `deep` execution profiles
- preserves resumable working state with compression and memory summaries
- degrades cleanly when the backend is unavailable or unstable

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

Echo already uses `unittest` when the repo matches that layout.

```bash
python -m unittest discover -s tests -p 'test_*.py'
echocode smoke "diagnóstico breve del backend actual"
```

## Operational Notes

- Echo stores runtime state, logs, memory, and session artifacts under `.echo/`.
- `.gitignore` excludes `.echo/`, virtual environments, caches, and local secrets.
- For critical `ask` workflows, use a healthy OpenAI-compatible fallback if Ollama chat is unstable.
- `doctor` shows cached, fresh, and effective backend state so routing decisions are auditable.

## Useful Commands

```bash
echocode doctor
echocode backend-check --chat-samples 2
echocode ask "inspecciona echo/runtime/engine.py y explica el flujo"
echocode plan "refuerza grounding y runtime"
echocode resume "continúa desde la última sesión"
echocode smoke "valida doctor, ask y resume"
```
