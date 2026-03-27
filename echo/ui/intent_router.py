from __future__ import annotations

import re
from typing import Literal

Intent = Literal[
    "conversation",
    "branch_new",
    "branch_switch",
    "branch_list",
    "branch_status",
    "branch_show",
    "branch_merge",
    "branch_cherry_pick",
    "session_status",
    "session_new",
    "exit",
    "help",
]

_NAME = r"([a-zA-Z0-9_\-]+)"

# (compiled_pattern, intent, extractor_fn)
_RULES: list[tuple[re.Pattern[str], str, object]] = []


def _r(pattern: str, intent: str):
    def _decorator(fn):
        _RULES.append((re.compile(pattern, re.IGNORECASE), intent, fn))
        return fn
    return _decorator


@_r(r"^(exit|quit|salir|chau|bye)\s*$", "exit")
def _exit(m, _): return {}

@_r(r"^(help|ayuda|\?)\s*$", "help")
def _help(m, _): return {}

@_r(r"^(status|estado)\s*$", "session_status")
def _session_status(m, _): return {}


# --- Branch new ---
@_r(
    r"(?:crea(?:r)?\s+(?:una\s+)?rama|nueva\s+rama|new\s+branch|branch\s+new)\s+" + _NAME,
    "branch_new",
)
def _branch_new(m, _): return {"name": m.group(1)}

# Also: "abre una rama X", "inicia rama X"
@_r(
    r"(?:abre?\s+(?:una\s+)?rama|inicia\s+(?:una\s+)?rama|start\s+branch)\s+" + _NAME,
    "branch_new",
)
def _branch_new2(m, _): return {"name": m.group(1)}


# --- Branch switch ---
@_r(
    r"(?:vuelve?\s+a|go\s+(?:back\s+)?to|switch\s+(?:to\s+)?|regresa\s+a)\s*" + _NAME,
    "branch_switch",
)
def _branch_switch(m, _): return {"name": m.group(1)}

@_r(
    r"cambia(?:r)?\s+a\s+(?:la\s+(?:rama\s+)?)?" + _NAME,
    "branch_switch",
)
def _branch_switch2(m, _): return {"name": m.group(1)}

@_r(
    r"cambia(?:r)?\s+de\s+(?:la\s+)?rama\s+(?:a\s+)?" + _NAME,
    "branch_switch",
)
def _branch_switch2b(m, _): return {"name": m.group(1)}

@_r(
    r"(?:ir(?:me)?\s+a(?:\s+la)?\s+rama)\s+" + _NAME,
    "branch_switch",
)
def _branch_switch3(m, _): return {"name": m.group(1)}


# --- Branch list ---
@_r(
    r"(?:lista(?:r)?(?:\s+(?:de\s+)?(?:las?\s+)?)?ramas?|list\s+branches?|ver\s+ramas?|show\s+branches?|branches?\s+list)",
    "branch_list",
)
def _branch_list(m, _): return {}


# --- Branch status ---
@_r(r"(?:branch\s+status|estado\s+(?:de\s+(?:la\s+)?)?rama)", "branch_status")
def _branch_status(m, _): return {}


# --- Branch show ---
@_r(r"(?:muestra(?:r)?\s+(?:la\s+)?rama|show\s+branch)\s+" + _NAME, "branch_show")
def _branch_show(m, _): return {"name": m.group(1)}


# --- Cherry-pick (must come before merge to avoid false positives) ---
@_r(
    r"(?:cherry.?pick)\s+" + _NAME,
    "branch_cherry_pick",
)
def _cherry1(m, _): return {"source": m.group(1)}

@_r(
    r"(?:trae\s+(?:solo\s+)?(?:las?\s+)?(?:decisiones?|findings?|pendings?|hechos?)\s+(?:de|from))\s+" + _NAME,
    "branch_cherry_pick",
)
def _cherry2(m, _): return {"source": m.group(1)}

@_r(
    r"(?:importa\s+(?:solo\s+)?(?:decisiones?|findings?))\s+(?:de|from)\s+" + _NAME,
    "branch_cherry_pick",
)
def _cherry3(m, _): return {"source": m.group(1)}


# --- Branch merge ---
@_r(
    r"(?:merge|fusiona(?:r)?|incorpora(?:r)?|combina(?:r)?)\s+" + _NAME,
    "branch_merge",
)
def _merge1(m, _): return {"source": m.group(1)}

@_r(
    r"(?:trae\s+(?:todo\s+(?:lo\s+)?)?(?:de|desde|lo de))\s+" + _NAME,
    "branch_merge",
)
def _merge2(m, _): return {"source": m.group(1)}

@_r(
    r"(?:trae\s+(?:a\s+main\s+)?(?:solo\s+)?(?:las?\s+)?(?:decisiones?\s+y\s+findings?|findings?\s+de))\s+" + _NAME,
    "branch_cherry_pick",
)
def _cherry4(m, _): return {"source": m.group(1)}


def route(text: str) -> tuple[Intent, dict]:
    """
    Classify natural language into an Intent.

    Slash commands are handled upstream by the REPL — this function only routes
    plain natural-language text. If somehow called with a '/' prefix, it returns
    "conversation" so the REPL falls back to the correct handler.

    Returns (intent, extracted_values_dict).
    """
    stripped = text.strip()
    if stripped.startswith("/"):
        return "conversation", {}
    for pattern, intent, extractor in _RULES:
        m = pattern.search(stripped)
        if m:
            return intent, extractor(m, stripped)  # type: ignore[operator]
    return "conversation", {}
