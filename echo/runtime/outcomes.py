from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from echo.types import RunState, SessionState


# ── Markup safety ─────────────────────────────────────────────────────────────

def _escape_brackets(text: str) -> str:
    """Escape Rich markup brackets so file content renders as plain text."""
    return text.replace("[", "\\[")


# ── Intent detection ──────────────────────────────────────────────────────────

_IMPROVEMENT_KWS: tuple[str, ...] = (
    "mejora", "mejorar", "mejoras", "propuesta", "propón",
    "sugerencia", "sugerencias", "optimiza", "refactor",
    "improve", "improvement", "suggestion", "suggestions",
    "qué falta", "que falta", "cómo mejorar", "como mejorar",
    "qué cambiarías", "que cambiarías",
)
_SUMMARY_KWS: tuple[str, ...] = (
    "resumen", "summary", "qué es", "que es", "describe",
    "overview", "explica", "de qué trata", "de que trata",
    "what is", "tell me about", "dame un resumen",
    "qué hace", "que hace", "cuéntame", "cuentame",
    "dame info", "dame información", "información del proyecto",
    "info del proyecto",
)
_CONFIG_KWS: tuple[str, ...] = (
    "config", "configuración", "configuracion",
    "backend", "modelo", "model", "settings",
    "timeout", "ollama", "openai", "perfil", "profile",
)
_NOVELTY_KWS: tuple[str, ...] = (
    "otra propuesta", "otra mejora", "otra aparte", "otra distinta",
    "otra que no sea", "alguna adicional", "más propuestas", "mas propuestas",
    "dame otras", "otras mejoras", "algo más avanzado", "algo mas avanzado",
    "más profundo", "mas profundo", "más profunda", "mas profunda",
    "otra más", "otra mas", "otra opción", "otra opcion",
    "aparte de esas", "aparte de esos", "otra alternativa",
)
_ADVANCED_IMPROVEMENT_KWS: tuple[str, ...] = (
    "avanzad", "profund", "arquitect", "runtime", "estructura",
    "producto", "ux", "repl", "tui", "flujo",
)
_IMPROVEMENT_LAYER_ORDER: tuple[str, ...] = ("base", "structural", "product")


@dataclass(frozen=True, slots=True)
class ImprovementProposal:
    proposal_id: str
    layer: str
    text: str


def _detect_improvement_followup(prompt: str) -> bool:
    low = (prompt or "").lower()
    return any(kw in low for kw in _NOVELTY_KWS)


def _prefers_advanced_improvements(prompt: str) -> bool:
    low = (prompt or "").lower()
    return any(kw in low for kw in _ADVANCED_IMPROVEMENT_KWS)


def _intent_includes_improvement(intent: str) -> bool:
    return intent in {"improvement", "combined", "config_improvement"}


def _detect_intent(prompt: str) -> str:
    """
    Classify the current user prompt into a local-inspection strategy.

    Returns one of:
        "combined"           — summary + improvement keywords present
        "config_improvement" — config + improvement keywords (no summary)
        "improvement"        — improvement keywords only
        "summary"            — summary keywords only
        "config"             — config keywords only
        "general"            — none of the above

    "combined" covers the three-way case (summary + config + improvement) because
    summary+improvement is the most informative response available locally.

    "config_improvement" is NOT implemented for summary+config because those two
    outputs overlap heavily (both surface project name, deps, stack) — combining
    them would produce redundant output.

    This function must always be called with session.user_prompt — the raw current
    message set in SessionState.create(). session.objective must never be used
    here because in resumed sessions it carries the previous session's objective.
    """
    low = (prompt or "").lower()
    has_improvement = any(kw in low for kw in _IMPROVEMENT_KWS)
    has_summary = any(kw in low for kw in _SUMMARY_KWS)
    has_config = any(kw in low for kw in _CONFIG_KWS)
    has_novelty = _detect_improvement_followup(prompt)

    # Three-way and two-way with summary → "combined" (summary+improvement)
    if has_improvement and has_summary:
        return "combined"
    # Config + improvement (no summary) → "config_improvement"
    if has_improvement and has_config:
        return "config_improvement"
    if has_improvement or has_novelty:
        return "improvement"
    if has_summary:
        return "summary"
    if has_config:
        return "config"
    return "general"


# ── File reading ──────────────────────────────────────────────────────────────

# Files to read per intent.
# setup.cfg and setup.py act as metadata fallbacks for pyproject.toml.
# AGENTS.md / CLAUDE.md supply architecture context; they are secondary sources.
_FILE_SPECS: dict[str, list[tuple[str, int]]] = {
    "summary": [
        ("README.md", 3000), ("pyproject.toml", 800),
        ("setup.cfg", 800), ("setup.py", 600),
        ("AGENTS.md", 1000), ("CLAUDE.md", 800),
    ],
    "improvement": [
        ("README.md", 2000), ("pyproject.toml", 1200),
        ("setup.cfg", 1000), ("setup.py", 800),
        ("AGENTS.md", 800), ("CLAUDE.md", 600),
    ],
    "config": [
        ("pyproject.toml", 1200), ("setup.cfg", 1200),
        ("setup.py", 800), ("README.md", 1000),
    ],
    "config_improvement": [
        ("pyproject.toml", 1200), ("setup.cfg", 1200),
        ("setup.py", 800), ("README.md", 1000),
        ("AGENTS.md", 600), ("CLAUDE.md", 500),
    ],
    "general": [
        ("README.md", 2000), ("pyproject.toml", 600),
        ("setup.cfg", 600), ("setup.py", 400),
        ("AGENTS.md", 600), ("CLAUDE.md", 500),
    ],
    "combined": [
        ("README.md", 3000), ("pyproject.toml", 1200),
        ("setup.cfg", 1000), ("setup.py", 800),
        ("AGENTS.md", 1000), ("CLAUDE.md", 800),
    ],
}


def _read_files_for_intent(repo_root: str, intent: str) -> dict[str, str]:
    """Read project files most relevant to the detected intent.
    Returns {rel_path: content} for files that exist and are readable."""
    root = Path(repo_root)
    specs = _FILE_SPECS.get(intent, _FILE_SPECS["general"])
    result: dict[str, str] = {}
    for rel_path, max_chars in specs:
        p = root / rel_path
        if p.exists():
            try:
                result[rel_path] = p.read_text(encoding="utf-8", errors="replace")[:max_chars]
            except OSError:
                pass
    return result


def _read_local_files(repo_root: str) -> list[tuple[str, str]]:
    """Return (rel_path, content) pairs for key project files.
    Kept for backwards-compatible callers; prefer _read_files_for_intent."""
    return list(_read_files_for_intent(repo_root, "general").items())


# ── Safe attr/file reference resolution ──────────────────────────────────────

def _resolve_attr_ref(attr_spec: str, repo_root: str) -> str | None:
    """Safely resolve a setup.cfg `attr:` reference without executing any code.

    attr_spec: e.g. "mypackage.__version__" or "mypackage.submod.__version__"

    Algorithm:
    1. Parse the top-level package name and the attribute name from the dotted path.
    2. Search for ``attr_name = "literal"`` in standard package locations:
       {pkg}/__init__.py, src/{pkg}/__init__.py, {pkg}/_version.py, src/{pkg}/_version.py
    3. Return the literal value if found; None if unresolvable.

    Never imports modules, never executes code.
    """
    parts = attr_spec.strip().split('.')
    if len(parts) < 2:
        return None

    attr_name = parts[-1]      # e.g. "__version__"
    pkg_root = parts[0]        # e.g. "mypackage" (top-level component only)

    root = Path(repo_root)
    candidates = [
        root / pkg_root / "__init__.py",
        root / "src" / pkg_root / "__init__.py",
        root / pkg_root / "_version.py",
        root / "src" / pkg_root / "_version.py",
    ]

    pattern = re.compile(
        rf'^{re.escape(attr_name)}\s*=\s*["\']([^"\']+)["\']',
        re.MULTILINE,
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")[:4000]
                m = pattern.search(text)
                if m:
                    return m.group(1)
            except OSError:
                pass
    return None


def _resolve_file_ref(file_spec: str, repo_root: str) -> str | None:
    """Safely resolve a setup.cfg `file:` reference.

    Returns the first non-empty line of the file (up to 200 chars), or None if:
    - the file is a README (too long for a description field)
    - the file does not exist or cannot be read
    """
    path_str = file_spec.strip().split(',')[0].strip()  # take first path if comma-separated
    if re.search(r'readme', path_str, re.IGNORECASE):
        return None  # README files are too long for description use
    p = Path(repo_root) / path_str
    if not p.exists():
        return None
    try:
        content = p.read_text(encoding="utf-8", errors="replace")[:200].strip()
        first_line = content.split('\n')[0].strip()
        return first_line if first_line else None
    except OSError:
        return None


# ── Structured extraction ─────────────────────────────────────────────────────

def _parse_pyproject(content: str) -> dict[str, str]:
    """Extract key fields from pyproject.toml with lightweight regex.
    Returns a dict with a subset of: name, description, python, dependencies, scripts."""
    meta: dict[str, str] = {}

    m = re.search(r'^\s*name\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if m:
        meta["name"] = m.group(1)

    m = re.search(r'^\s*description\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if m:
        meta["description"] = m.group(1)

    m = re.search(r'^\s*requires-python\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if m:
        meta["python"] = m.group(1)

    deps_block = re.search(
        r'^\s*dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL | re.MULTILINE
    )
    if deps_block:
        pkgs = re.findall(r'"([^"]+)"', deps_block.group(1))
        names = [re.split(r'[>=<!\s]', p)[0] for p in pkgs]
        if names:
            shown = names[:7]
            meta["dependencies"] = ", ".join(shown)
            if len(names) > 7:
                meta["dependencies"] += f" (+{len(names) - 7})"

    scripts_block = re.search(
        r'\[project\.scripts\](.*?)(?=\[|$)', content, re.DOTALL
    )
    if scripts_block:
        script_names = re.findall(
            r'^\s*(\S+)\s*=', scripts_block.group(1), re.MULTILINE
        )
        if script_names:
            meta["scripts"] = ", ".join(script_names)

    return meta


def _parse_setup_cfg(content: str, repo_root: str | None = None) -> dict[str, str]:
    """Extract project metadata from setup.cfg [metadata] and [options] sections.

    Handles the standard setuptools INI format, including `attr:` and `file:`
    directives when repo_root is provided:

      [metadata]
      name = myproject
      version = attr: mypackage.__version__   ← resolved via _resolve_attr_ref
      description = file: DESCRIPTION.txt      ← resolved via _resolve_file_ref

      [options]
      python_requires = >=3.9
      install_requires =
          rich
          typer

      [options.entry_points]
      console_scripts =
          myapp = mypackage.cli:main

    If attr:/file: cannot be resolved (repo_root absent or file not found) the
    field is omitted rather than surfacing the raw directive string to the user.
    """
    meta: dict[str, str] = {}

    def _resolve_scalar(raw: str) -> str | None:
        val = raw.strip()
        if val.startswith('attr:'):
            attr_spec = val[5:].strip()
            return _resolve_attr_ref(attr_spec, repo_root) if repo_root else None
        if val.startswith('file:'):
            file_spec = val[5:].strip()
            return _resolve_file_ref(file_spec, repo_root) if repo_root else None
        if val and not val.startswith('['):
            return val
        return None

    m = re.search(r'^\s*name\s*=\s*(.+)', content, re.MULTILINE)
    if m:
        resolved = _resolve_scalar(m.group(1))
        if resolved:
            meta["name"] = resolved

    m = re.search(r'^\s*description\s*=\s*(.+)', content, re.MULTILINE)
    if m:
        resolved = _resolve_scalar(m.group(1))
        if resolved:
            meta["description"] = resolved

    m = re.search(r'^\s*version\s*=\s*(.+)', content, re.MULTILINE)
    if m:
        resolved = _resolve_scalar(m.group(1))
        if resolved:
            meta["version"] = resolved

    m = re.search(r'python_requires\s*=\s*(.+)', content, re.MULTILINE)
    if m:
        val = m.group(1).strip()
        # python_requires is rarely an attr: reference; take literal only
        if val and not val.startswith('attr:') and not val.startswith('file:'):
            meta["python"] = val

    # install_requires — multi-line continuation block
    block = re.search(r'install_requires\s*=\s*\n((?:[ \t]+\S[^\n]*\n?)*)', content)
    if block:
        lines = block.group(1).splitlines()
        names: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                pkg_name = re.split(r'[>=<!\s\[]', stripped)[0]
                if pkg_name:
                    names.append(pkg_name)
        if names:
            shown = names[:7]
            meta["dependencies"] = ", ".join(shown)
            if len(names) > 7:
                meta["dependencies"] += f" (+{len(names) - 7})"

    ep_block = re.search(r'\[options\.entry_points\](.*?)(?=\[|$)', content, re.DOTALL)
    if ep_block:
        script_names = re.findall(r'^\s*(\S+)\s*=', ep_block.group(1), re.MULTILINE)
        if script_names:
            meta["scripts"] = ", ".join(script_names)

    return meta


def _extract_simple_vars(content: str) -> dict[str, str]:
    """Extract module-level bare string assignments from Python source.

    Returns {VARNAME: "value"} for patterns like:
        NAME = "echo-agent"
        VERSION = '1.2.3'
        __version__ = "1.0.0"
        DESCRIPTION = "A cool tool"

    Safe: regex-only, does not execute code.  Only captures simple string
    literals; ignores expressions, function calls, and multi-line assignments.
    """
    result: dict[str, str] = {}
    for m in re.finditer(
        r'^([A-Za-z_]\w*)\s*=\s*["\']([^"\']*)["\']',
        content, re.MULTILINE,
    ):
        result[m.group(1)] = m.group(2)
    return result


def _resolve_setup_kwarg(
    content: str,
    key: str,
    simple_vars: dict[str, str],
) -> str | None:
    """Find keyword=value in a setup() call.

    Checks in order:
    1. key="literal"  or  key='literal'
    2. key=VARNAME   where VARNAME is in simple_vars

    Returns the resolved string value or None.
    """
    # Direct string literal
    m = re.search(rf'\b{re.escape(key)}\s*=\s*["\']([^"\']+)["\']', content)
    if m:
        return m.group(1)
    # Variable reference
    m = re.search(rf'\b{re.escape(key)}\s*=\s*([A-Za-z_]\w*)\b', content)
    if m:
        varname = m.group(1)
        return simple_vars.get(varname)
    return None


def _parse_setup_py(content: str, repo_root: str | None = None) -> dict[str, str]:
    """Extract project metadata from setup.py without executing the file.

    Handles common real-world patterns beyond simple string literals:
      NAME = "echo-agent"          ← module-level variable
      __version__ = "1.2.3"        ← version variable
      setup(
          name=NAME,               ← variable reference resolved
          version=__version__,     ← resolved from simple_vars or pkg __init__
          install_requires=DEPS,   ← list variable resolved
      )

    Falls back gracefully: if a pattern cannot be resolved without executing
    code, the field is omitted rather than returning a raw variable name.
    """
    meta: dict[str, str] = {}
    simple_vars = _extract_simple_vars(content)

    name = _resolve_setup_kwarg(content, "name", simple_vars)
    if name:
        meta["name"] = name

    desc = _resolve_setup_kwarg(content, "description", simple_vars)
    if desc:
        meta["description"] = desc

    version = _resolve_setup_kwarg(content, "version", simple_vars)
    if not version and repo_root:
        # Last resort: look for __version__ in the top-level package if name known
        pkg = meta.get("name") or name
        if pkg:
            version = _resolve_attr_ref(f"{pkg}.__version__", repo_root)
    if version:
        meta["version"] = version

    python_req = _resolve_setup_kwarg(content, "python_requires", simple_vars)
    if python_req:
        meta["python"] = python_req

    # install_requires: try inline list literal first, then variable reference
    names: list[str] = []
    m = re.search(r'install_requires\s*=\s*\[([^\]]*)\]', content, re.DOTALL)
    if m:
        pkgs = re.findall(r'["\']([^"\']+)["\']', m.group(1))
        names = [re.split(r'[>=<!\s]', p)[0] for p in pkgs if p.strip()]
        names = [n for n in names if n]
    if not names:
        # Variable reference: install_requires=VARNAME
        m2 = re.search(r'install_requires\s*=\s*([A-Za-z_]\w*)\b', content)
        if m2:
            varname = m2.group(1)
            # Find VARNAME = ["pkg1", "pkg2", ...]
            list_m = re.search(
                rf'^{re.escape(varname)}\s*=\s*\[([^\]]*)\]',
                content, re.MULTILINE | re.DOTALL,
            )
            if list_m:
                pkgs = re.findall(r'["\']([^"\']+)["\']', list_m.group(1))
                names = [re.split(r'[>=<!\s]', p)[0] for p in pkgs if p.strip()]
                names = [n for n in names if n]
    if names:
        shown = names[:7]
        meta["dependencies"] = ", ".join(shown)
        if len(names) > 7:
            meta["dependencies"] += f" (+{len(names) - 7})"

    # console_scripts inside entry_points
    m = re.search(r'console_scripts[^=\[]*[=:]\s*\[?\s*["\']([^"\'=]+)\s*=', content, re.DOTALL)
    if m:
        meta["scripts"] = m.group(1).strip()

    return meta


def _parse_project_meta(
    files: dict[str, str],
    *,
    repo_root: str | None = None,
) -> dict[str, str]:
    """Unified entry point: extract project metadata from the first available
    project definition file (pyproject.toml > setup.cfg > setup.py).

    repo_root is forwarded to setup.cfg and setup.py parsers to enable safe
    resolution of attr: directives and module-level variable references.
    """
    if "pyproject.toml" in files:
        return _parse_pyproject(files["pyproject.toml"])
    if "setup.cfg" in files:
        return _parse_setup_cfg(files["setup.cfg"], repo_root=repo_root)
    if "setup.py" in files:
        return _parse_setup_py(files["setup.py"], repo_root=repo_root)
    return {}


def _readme_intro(content: str, max_chars: int = 700, max_paragraphs: int = 4) -> str:
    """Extract the opening section of a README or doc file.

    Paragraph-aware extraction:
    - Keeps the H1 title so the project name is visible.
    - Skips badge lines ([![...) and inline HTML.
    - Collects complete paragraphs — never cuts mid-sentence.
    - Stops at the first level-2 heading after real content is found.
    - max_paragraphs caps paragraph count; max_chars is a hard safety limit.
    """
    lines = content.split('\n')
    output_paras: list[list[str]] = [[]]   # list of paragraphs, each = list of lines
    seen_content = False
    para_count = 0

    for line in lines:
        stripped = line.strip()

        # Stop at ## after content has been collected
        if stripped.startswith('## ') and seen_content:
            break

        # Skip badges and inline HTML
        if stripped.startswith('[![') or (stripped.startswith('<') and '>' in stripped):
            continue

        if stripped:
            seen_content = True
            output_paras[-1].append(line)
        else:
            # Blank line = paragraph boundary
            if output_paras[-1]:                 # current paragraph has content
                para_count += 1
                if para_count >= max_paragraphs:
                    break
                output_paras.append([])

    # Keep the in-progress paragraph at the end (no trailing blank needed)
    output_paras = [p for p in output_paras if p]

    result = '\n\n'.join('\n'.join(para) for para in output_paras).strip()

    # Hard safety cap — applied after collecting complete paragraphs
    if len(result) > max_chars:
        result = result[:max_chars].rstrip()

    return result


def _extract_agent_doc_hints(files: dict[str, str], intent: str) -> list[str]:
    """Extract focused content from AGENTS.md and CLAUDE.md for the given intent.

    These files are treated as supplementary sources — never as the primary
    answer body.  For summary/general intents the opening section is surfaced
    (paragraph-aware, stops at ##).  For improvement intents, TODO/FIXME/debt
    markers are extracted as additional grounded suggestions.
    """
    hints: list[str] = []
    for doc_name in ("AGENTS.md", "CLAUDE.md"):
        content = files.get(doc_name, "")
        if not content.strip():
            continue
        if intent in ("summary", "general", "combined"):
            intro = _readme_intro(content, max_chars=500, max_paragraphs=3)
            if intro:
                hints.append(f"[{doc_name}]\n{intro}")
        if intent in ("improvement", "combined", "config_improvement"):
            todo_lines = [
                line.strip()
                for line in content.split('\n')
                if any(
                    marker in line.upper()
                    for marker in ("TODO", "FIXME", "DEBT", "HACK", "KNOWN ISSUE", "IMPROVE")
                )
            ]
            if todo_lines:
                hints.append(
                    f"[{doc_name} — trabajo pendiente]\n" +
                    "\n".join(todo_lines[:5])
                )
    return hints


# ── Synthesis by intent ───────────────────────────────────────────────────────

def _synthesize_summary(
    files: dict[str, str],
    repo_root: str | None = None,
) -> str:
    """Compose a natural project summary from inspected files.
    Works with pyproject.toml, setup.cfg, or setup.py as the metadata source."""
    parts: list[str] = []

    meta = _parse_project_meta(files, repo_root=repo_root)
    readme = files.get("README.md", "")

    if meta:
        name = meta.get("name", "")
        desc = meta.get("description", "")
        python_req = meta.get("python", "")
        version = meta.get("version", "")
        deps = meta.get("dependencies", "")
        scripts = meta.get("scripts", "")

        if name or desc:
            header = f"**{name}**" if name else ""
            if version:
                header = f"{header} v{version}" if header else f"v{version}"
            if desc:
                header = f"{header} — {desc}" if header else desc
            parts.append(header)

        tech: list[str] = []
        if python_req:
            tech.append(f"Python {python_req}")
        if deps:
            tech.append(f"dependencias: {deps}")
        if scripts:
            tech.append(f"comandos: {scripts}")
        if tech:
            parts.append("Stack: " + " · ".join(tech))

    if readme:
        intro = _readme_intro(readme)
        if intro:
            parts.append(intro)

    # Supplement with agent/convention docs when present
    agent_hints = _extract_agent_doc_hints(files, "summary")
    parts.extend(agent_hints)

    if not parts:
        return (
            "No se encontró suficiente información para generar un resumen.\n"
            "Verifica que existe README.md, pyproject.toml, setup.cfg o setup.py "
            "en la raíz del proyecto."
        )

    return "\n\n".join(parts)


def _append_proposal(
    proposals: list[ImprovementProposal],
    seen: set[str],
    proposal_id: str,
    layer: str,
    text: str,
) -> None:
    if proposal_id in seen:
        return
    proposals.append(ImprovementProposal(proposal_id=proposal_id, layer=layer, text=text))
    seen.add(proposal_id)


def _collect_improvement_proposals(repo_root: str, files: dict[str, str]) -> list[ImprovementProposal]:
    """Build grounded improvement proposals grouped by progression layer."""
    proposals: list[ImprovementProposal] = []
    seen: set[str] = set()
    root = Path(repo_root)

    project_def = (
        files.get("pyproject.toml", "") +
        files.get("setup.cfg", "") +
        files.get("setup.py", "")
    ).lower()
    readme = files.get("README.md", "")

    if project_def:
        if "mypy" not in project_def:
            _append_proposal(
                proposals, seen, "base:mypy", "base",
                "Añadir `mypy` a dev-deps — el proyecto usa anotaciones de tipo "
                "pero no tiene verificación estática declarada"
            )
        if "ruff" not in project_def and "flake8" not in project_def and "pylint" not in project_def:
            _append_proposal(
                proposals, seen, "base:ruff", "base",
                "Configurar `ruff` para linting y formateo automático — "
                "no se detectó linter en la configuración del proyecto"
            )
        if "coverage" not in project_def and "pytest-cov" not in project_def:
            _append_proposal(
                proposals, seen, "base:coverage", "base",
                "Añadir `pytest-cov` para métricas de cobertura de tests"
            )

    if not (root / "CHANGELOG.md").exists() and not (root / "CHANGELOG").exists():
        _append_proposal(
            proposals, seen, "base:changelog", "base",
            "Crear `CHANGELOG.md` — no existe en el proyecto"
        )

    if not (root / ".github" / "workflows").exists():
        _append_proposal(
            proposals, seen, "base:ci", "base",
            "Configurar GitHub Actions CI — no se detectó `.github/workflows/`"
        )

    if readme and len(readme.strip()) < 300:
        _append_proposal(
            proposals, seen, "base:readme-depth", "base",
            "Ampliar README — está muy corto, faltan ejemplos de uso e instalación"
        )

    meta = _parse_project_meta(files, repo_root=repo_root)
    name = meta.get("name")
    if name:
        _append_proposal(
            proposals, seen, "base:distribution", "base",
            f"Documentar instalación de `{name}` desde fuente en README "
            f"o publicar a PyPI"
        )

    if (root / "echo" / "runtime" / "outcomes.py").exists() and (root / "echo" / "runtime" / "engine.py").exists():
        _append_proposal(
            proposals, seen, "structural:fallback-split", "structural",
            "Separar la lógica de fallback local y catálogo de propuestas fuera de "
            "`echo/runtime/outcomes.py` para que intent routing, memoria y síntesis "
            "queden más mantenibles y testeables"
        )
    if (root / "echo" / "runtime" / "model_loop.py").exists() and (root / "echo" / "runtime" / "verify_flow.py").exists():
        _append_proposal(
            proposals, seen, "structural:integration-hardening", "structural",
            "Añadir smoke tests multi-turno del runtime/fallback para cubrir "
            "seguimientos conversacionales y evitar regresiones entre `model_loop.py` "
            "y la síntesis local"
        )
    if (root / "echo" / "ui" / "intent_router.py").exists():
        _append_proposal(
            proposals, seen, "structural:router-observability", "structural",
            "Extender el router/intérprete conversacional con señales auditables de "
            "follow-up y novedad para distinguir mejor entre resumen inicial y "
            "peticiones de profundización"
        )
    if (root / "tests" / "test_local_inspect_fallback.py").exists():
        _append_proposal(
            proposals, seen, "structural:anti-repeat-tests", "structural",
            "Endurecer las pruebas de fallback con aserciones de no repetición "
            "sustancial entre turnos, no solo detección de intent"
        )
    if (root / "echo" / "ui" / "repl.py").exists():
        _append_proposal(
            proposals, seen, "product:repl-polish", "product",
            "Pulir el REPL con un header más compacto, composer más claro y separación "
            "más nítida entre chat principal y señales auxiliares"
        )
    if (root / "echo" / "branches").exists() and (root / ".echo").exists() is False:
        _append_proposal(
            proposals, seen, "product:branch-ux", "product",
            "Añadir ayudas visuales ligeras para sesión/rama activa y acciones recientes "
            "sin mezclar telemetría técnica con la conversación"
        )

    agent_hints = _extract_agent_doc_hints(files, "improvement")
    for idx, hint in enumerate(agent_hints, 1):
        _append_proposal(
            proposals,
            seen,
            f"base:docs-hint-{idx}",
            "base",
            hint,
        )
    return proposals


def _choose_improvement_layer(
    prompt: str,
    emitted_layers: list[str],
    available_by_layer: dict[str, list[ImprovementProposal]],
) -> str:
    if _prefers_advanced_improvements(prompt):
        for layer in ("structural", "product", "base"):
            if available_by_layer.get(layer):
                return layer
    if _detect_improvement_followup(prompt):
        for layer in _IMPROVEMENT_LAYER_ORDER:
            if layer not in emitted_layers and available_by_layer.get(layer):
                return layer
        for layer in _IMPROVEMENT_LAYER_ORDER:
            if available_by_layer.get(layer):
                return layer
    for layer in _IMPROVEMENT_LAYER_ORDER:
        if available_by_layer.get(layer):
            return layer
    return "base"


def _render_improvement_response(
    proposals: list[ImprovementProposal],
    *,
    layer: str,
    followup: bool,
) -> str:
    heading_map = {
        "base": "Propuestas de mejora inmediatas",
        "structural": "Otras mejoras, más estructurales",
        "product": "Otras mejoras, más de producto/UX",
    }
    intro_map = {
        "base": "Basadas en la inspección local del proyecto:",
        "structural": "Para no repetir las anteriores, aquí van opciones nuevas apoyadas en la estructura real del repo:",
        "product": "Como siguiente capa, estas mejoras apuntan al uso diario del agente y del REPL:",
    }
    lines = [heading_map.get(layer, "Propuestas de mejora")]
    if followup:
        lines.append(intro_map.get(layer, intro_map["base"]))
    else:
        lines.append("Basadas en la inspección local del proyecto:")
    lines.append("")
    for index, proposal in enumerate(proposals, 1):
        lines.append(f"{index}. {proposal.text}")
    return "\n".join(lines)


def _synthesize_improvement(
    repo_root: str,
    files: dict[str, str],
    *,
    prompt: str,
    session: SessionState | None = None,
) -> str:
    """Build grounded, progressive improvement proposals and remember emitted items."""
    all_proposals = _collect_improvement_proposals(repo_root, files)
    if not all_proposals:
        return (
            "El proyecto parece bien configurado a nivel local.\n"
            "Para sugerencias más profundas el backend de análisis necesita estar activo."
        )

    emitted_ids = set(session.emitted_improvement_ids if session is not None else [])
    emitted_layers = list(session.emitted_improvement_layers if session is not None else [])
    available = [proposal for proposal in all_proposals if proposal.proposal_id not in emitted_ids]
    available_by_layer = {
        layer: [proposal for proposal in available if proposal.layer == layer]
        for layer in _IMPROVEMENT_LAYER_ORDER
    }
    followup = _detect_improvement_followup(prompt)
    chosen_layer = _choose_improvement_layer(prompt, emitted_layers, available_by_layer)
    selected = available_by_layer.get(chosen_layer, [])[:4]

    if not selected and followup:
        unseen = available[:3]
        if unseen:
            selected = unseen
            chosen_layer = unseen[0].layer

    if not selected:
        selected = all_proposals[:4]
        chosen_layer = selected[0].layer

    if session is not None:
        for proposal in selected:
            if proposal.proposal_id not in session.emitted_improvement_ids:
                session.emitted_improvement_ids.append(proposal.proposal_id)
            if proposal.layer not in session.emitted_improvement_layers:
                session.emitted_improvement_layers.append(proposal.layer)

    return _render_improvement_response(selected, layer=chosen_layer, followup=followup)


def _synthesize_config(
    files: dict[str, str],
    repo_root: str | None = None,
) -> str:
    """Describe project configuration extracted from local files."""
    meta = _parse_project_meta(files, repo_root=repo_root)
    readme = files.get("README.md", "")
    parts: list[str] = []

    if meta:
        if meta.get("name"):
            parts.append(f"Proyecto: **{meta['name']}**")
        if meta.get("version"):
            parts.append(f"Versión: {meta['version']}")
        if meta.get("python"):
            parts.append(f"Python requerido: {meta['python']}")
        if meta.get("dependencies"):
            parts.append(f"Dependencias: {meta['dependencies']}")
        if meta.get("scripts"):
            parts.append(f"Comandos disponibles: `{meta['scripts']}`")
        all_content = " ".join(files.values())
        if "ECHO_" in all_content:
            parts.append(
                "Variables de entorno: el proyecto acepta variables ECHO_* "
                "(ver README para la lista completa)"
            )

    if not parts:
        if readme:
            intro = _readme_intro(readme, max_chars=400)
            if intro:
                return intro
        return (
            "No se encontró información de configuración en los archivos locales.\n"
            "Consulta el README o usa /doctor para ver el estado del backend."
        )

    return "\n".join(parts)


def _synthesize_general(
    files: dict[str, str],
    prompt: str,  # noqa: ARG001
    repo_root: str | None = None,
) -> str:
    """General fallback — surface a project summary as the most useful default."""
    result = _synthesize_summary(files, repo_root=repo_root)
    if result and "No se encontró" not in result:
        return result
    readme = files.get("README.md", "")
    if readme:
        intro = _readme_intro(readme, max_chars=400)
        if intro:
            return intro
    return (
        "Los archivos locales no contienen suficiente contexto para esta pregunta.\n"
        "El backend de análisis necesita estar activo para respuestas detalladas."
    )


def _synthesize_for_prompt(
    repo_root: str,
    files: dict[str, str],
    prompt: str,
    *,
    session: SessionState | None = None,
) -> str:
    """Route synthesis to the function matching the detected intent.

    "combined":           summary + "---" + improvement (single response)
    "config_improvement": config + "---" + improvement
    All other intents:    single section.
    """
    intent = _detect_intent(prompt)
    if intent == "combined":
        summary = _synthesize_summary(files, repo_root=repo_root)
        improvement = _synthesize_improvement(repo_root, files, prompt=prompt, session=session)
        return f"{summary}\n\n---\n\n{improvement}"
    if intent == "config_improvement":
        config = _synthesize_config(files, repo_root=repo_root)
        improvement = _synthesize_improvement(repo_root, files, prompt=prompt, session=session)
        return f"{config}\n\n---\n\n{improvement}"
    if intent == "summary":
        return _synthesize_summary(files, repo_root=repo_root)
    if intent == "improvement":
        return _synthesize_improvement(repo_root, files, prompt=prompt, session=session)
    if intent == "config":
        return _synthesize_config(files, repo_root=repo_root)
    return _synthesize_general(files, prompt, repo_root=repo_root)


# ── Main public functions ─────────────────────────────────────────────────────

def build_local_inspect_answer(
    session: SessionState,
    run_state: RunState,
    *,
    activity,
) -> str:
    """
    Fallback when the model answered without executing any tools.

    Intent is derived from session.user_prompt — the raw current message set in
    SessionState.create().  session.objective is intentionally NOT used here
    because it may carry the previous session's objective (set by resume_seed in
    prepare.py) or an internal plan directive.

    Files are selected based on the detected intent, and a natural-language
    response is synthesized from their contents.  Raw file contents are never
    returned as the response body.

    Supports: README.md, pyproject.toml, setup.cfg, setup.py, AGENTS.md, CLAUDE.md
    Combined intents:
      summary + improvement   → "combined"
      config + improvement    → "config_improvement"
    """
    prompt = session.user_prompt or ""

    activity.emit(
        "Fallback", "running", "Local inspection",
        f"intent from user_prompt: {prompt[:60]!r}",
    )

    intent = _detect_intent(prompt)
    files = _read_files_for_intent(run_state.repo_root, intent)

    if not files:
        activity.emit("Fallback", "done", "Local inspection: no files", run_state.repo_root)
        return (
            "No se encontraron archivos de proyecto para inspeccionar.\n"
            f"Directorio: {run_state.repo_root}\n"
            "Ejecuta desde la raíz del proyecto o usa --project-dir."
        )

    answer = _synthesize_for_prompt(run_state.repo_root, files, prompt, session=session)
    activity.emit("Fallback", "done", f"Local inspection: {intent}", f"{len(files)} files")
    return answer


def build_heuristic_plan(
    session: SessionState,
    run_state: RunState,
    *,
    reason: str,
    update_stage,
    activity,
) -> str:
    run_state.fallback_used = True
    run_state.fallback_reason = reason
    stage_id = run_state.current_stage_id or "inspect"
    update_stage(session, run_state, stage_id, status="failed", result=reason)
    activity.emit("Planner", "degraded", "Fallback plan mode", reason)
    files = session.working_set[-6:] or session.focus_files[-6:] or ["README.md"]
    risks = [
        "El backend de chat está inestable, así que el plan se basa en inspección local y memoria operativa.",
        "Puede faltar validación semántica profunda hasta que Ollama vuelva a responder de forma consistente.",
    ]
    next_steps = [
        "Revisar el loop de runtime y el verificador grounded.",
        "Confirmar backend health y repetir smoke cuando /api/chat deje de oscilar.",
    ]
    return "\n".join([
        "Objetivo",
        "- Mantener Echo operativo aun con backend inestable y reforzar grounding del answer final.",
        "Archivos a revisar",
        *(f"- {item}" for item in files),
        "Riesgos",
        *(f"- {item}" for item in risks),
        "Siguientes pasos",
        *(f"- {item}" for item in next_steps),
    ])


def build_degraded_answer(
    session: SessionState,
    run_state: RunState,
    *,
    reason: str,
    mode: str,
    update_stage,
    activity,
) -> str:
    run_state.fallback_used = True
    run_state.fallback_reason = reason
    session.degraded_reason = reason
    current_stage = run_state.current_stage_id or ("close" if mode == "plan" else "execute")
    update_stage(session, run_state, current_stage, status="failed", result=reason)

    if mode == "resume":
        activity.emit(
            "Resume", "degraded",
            "Resume state restored without backend completion", reason,
        )
        return "\n".join([
            "Echo restauró el estado de la sesión, pero no pudo completar con el backend.",
            f"Objetivo: {session.objective or session.user_prompt}",
            f"Working set: {', '.join(session.working_set[-8:]) or 'none'}",
            f"Pendientes: {'; '.join(session.pending[-6:]) or 'none'}",
            f"Etapa detenida: {current_stage}",
            f"Límite actual: {reason}",
        ])

    activity.emit(
        "Planner" if mode == "plan" else "Verifier",
        "degraded", "Fallback answer mode", reason,
    )

    prompt = session.user_prompt or ""
    intent = _detect_intent(prompt)
    files = _read_files_for_intent(run_state.repo_root, intent)

    if files:
        content = _synthesize_for_prompt(run_state.repo_root, files, prompt, session=session)
        return (
            "El backend no está disponible. "
            "Respuesta local basada en archivos del proyecto:\n\n"
            f"{content}\n\n"
            "Usa /doctor para verificar la conexión al backend."
        )

    evidence = session.working_set[-6:] or session.focus_files[-6:]
    parts: list[str] = ["El backend no respondió de forma confiable."]
    if evidence:
        parts.append(f"Archivos inspeccionados: {', '.join(evidence)}")
    parts.append("Usa /doctor para verificar el estado o intenta de nuevo.")
    return "\n".join(parts)


def is_resume_summary_only(prompt: str) -> bool:
    low = (prompt or "").lower()
    resume_markers = ["resume", "resum", "working set", "pendient", "objetivo"]
    return sum(1 for marker in resume_markers if marker in low) >= 3


def build_resume_local_answer(session: SessionState, run_state: RunState, *, activity) -> str:
    activity.emit("Resume", "done", "Resume local summary", session.id)
    return "\n".join([
        "Echo restauró la sesión desde memoria local.",
        f"Objetivo: {session.objective or run_state.objective}",
        f"Working set: {', '.join(session.working_set[-8:] or session.focus_files[-8:]) or 'none'}",
        f"Pendientes: {'; '.join(session.pending[-6:] or run_state.pending[-6:]) or 'none'}",
    ])
