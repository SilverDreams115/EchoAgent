# Changelog

All notable changes to EchoAgent are documented in this file.

---

## [0.1.0] — 2026-03-25

First releaseable baseline. EchoAgent is a local coding agent CLI that operates
against a local LLM backend (Ollama or any OpenAI-compatible endpoint) and
produces grounded, resumable agentic sessions over a local project tree.

### Added

- **npm launcher** (`package.json` + `bin/echo-agent.js`): installs via
  `npm install -g .`; manages a per-user Python venv transparently; works on
  Linux, macOS, and Windows.
- **`echo-agent` global command**: primary CLI entry point, safe across all
  platforms. Avoids registering `echo` as a global command because `echo` is a
  shell builtin on most Unix-like systems and cannot be reliably overridden from
  `PATH` alone.
- **`echocode` compatibility command**: maps to the same launcher for users
  already scripted against the earlier name.
- **`echo/fsignore.py`**: single source of truth for ignored directory names
  (`.git`, `.venv`, `node_modules`, `__pycache__`, etc.); used by file listing,
  search, and validation detection to exclude environmental noise.
- **`echo/runtime/tool_tracking.py`**: extracted tool-call tracking logic out
  of `engine.py` into its own module; `record_tool_call` is now independently
  testable and the engine no longer holds that responsibility inline.
- **`tests/runtime_fixtures.py`**: shared `FakeBackend`, `RuntimeTestCase`, and
  helper variants (`TimeoutOnceBackend`, `UnreachableBackend`,
  `MalformedOnceBackend`, `SlowTimeoutBackend`) used across all test modules.
- **`tests/test_shell_policy.py`**: dedicated test module for shell policy,
  default settings, `rg` fallback, and `search_symbol` scope.
- **`tests/test_validation_detection.py`**: dedicated test module for
  validation strategy detection, virtualenv noise filtering, and edge cases.
- **`tests/test_verifier.py`**: dedicated test module for the final-answer
  verifier (grounding, strategy matching, symbol evidence).

### Changed

- **Defaults hardened**: `allow_write` and `allow_shell` now default to `False`.
  Both environment variables (`ECHO_ALLOW_WRITE`, `ECHO_ALLOW_SHELL`) default
  to `"0"`. Write and shell capabilities must be explicitly opted into.
- **Shell policy hardened**: `python -m pip` and `python3 -m pip` removed from
  the list of allowed shell commands. Package management is not a safe
  automated operation within agentic sessions.
- **`search_text` / `grep_code`**: fall back to a pure-Python line scan when
  `rg` is not available on the host; result includes `"engine": "rg"` or
  `"engine": "python"` for transparency.
- **`search_symbol` / `find_symbol`**: tool schema and return value now
  explicitly document Python-only scope. Descriptions changed from "code
  symbols" to "Python symbols". Return value includes `"scope": "python-only"`.
  Non-`.py` files are skipped unconditionally.
- **`echo/cognition/validation.py`**: uses `iter_project_files` instead of
  `rglob` to exclude virtualenv noise when detecting test layout and checking
  for Python sources.
- **CI (`.github/workflows/ci.yml`)**: adds explicit focused-suite step
  (`test_runtime_flows`, `test_validation_detection`, `test_verifier`,
  `test_shell_policy`) before the full discover, and a npm-launcher surface
  check step.

### Known limitations

- EchoAgent is **not a self-contained binary**. The npm launcher installs the
  Node.js wrapper; the actual agent requires Python 3.10+ and the package
  dependencies to be present or installable.
- **`echo` is not registered as a global command**. Use `echo-agent` or
  `echocode`.
- **Symbol search is Python-only**. `search_symbol` and `find_symbol` operate
  only on `.py` files. Non-Python symbol search is not yet implemented.
- **Backend depends on local machine setup**. `echo-agent doctor` will report
  `deep_ready: False` unless a compatible LLM backend is running and configured
  in the environment.
- **Windows support via npm launcher is implemented but not CI-tested** at this
  baseline. The launcher includes Windows path logic; runtime compatibility
  is not yet validated in CI.
