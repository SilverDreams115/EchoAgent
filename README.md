# Echo

Echo is a local coding agent CLI with:
- multiline composer
- live activity panel
- clean `.echo/` state per repo
- backend layer for Ollama and OpenAI-compatible APIs
- native and compatibility tool execution
- execution profiles: local, balanced, deep
- file tools and safe shell execution
- resumable sessions
- operational memory with context compression
- explicit runtime phases: intake, planner, inspector, executor, verifier, summarizer

## Run

```bash
pip install -e .
echocode doctor
echocode shell
echocode ask "inspecciona este proyecto"
echocode plan "propón cambios mínimos"
echocode resume
ECHO_BACKEND_TIMEOUT=240 ECHO_OLLAMA_KEEP_ALIVE=15m echocode ask "haz una inspección profunda"
```

## Backends

```bash
ECHO_BACKEND=ollama ECHO_MODEL=qwen2.5-coder:7b-oh echocode doctor
ECHO_BACKEND=openai-compatible ECHO_OPENAI_API_KEY=... ECHO_MODEL=gpt-4.1-mini echocode doctor
```

## Profiles

```bash
echocode doctor --profile local
echocode ask --profile local "inspecciona este repo"
echocode plan --profile balanced "propón un cambio seguro"
echocode ask --profile deep "haz análisis profundo"
echocode ask --profile deep --strict-profile "haz análisis profundo"
echocode smoke --profile deep --strict-profile "haz una verificación breve del backend"
```

- `local`: Ollama, rápido, grounded, con tool execution en modo compatibilidad.
- `balanced`: prioriza un modelo práctico; usa fallback a local si falta backend frontier.
- `deep`: prioriza backend frontier y degrada a `balanced` si falta configuración.

Si quieres exigir `deep` real y fallar si no está configurado:

```bash
ECHO_OPENAI_API_KEY=TU_API_KEY echocode doctor --profile deep --strict-profile
ECHO_OPENAI_API_KEY=TU_API_KEY echocode ask --profile deep --strict-profile "haz análisis profundo"
ECHO_OPENAI_API_KEY=TU_API_KEY echocode smoke --profile deep --strict-profile "haz una verificación breve del backend"
```
