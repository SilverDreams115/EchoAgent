"""
Intent router for the EchoRepl conversational shell.

Design principles:
- All patterns are regex-based and deterministic — no ML, fully auditable.
- Slash commands are handled upstream; the router returns "conversation" for them.
- Patterns are ordered: more specific before more general.
- Branch-name tokens require ≥ 2 chars (avoids single-letter false positives).
- Command-style patterns use ^ / $ anchors to avoid matching fragments embedded
  inside longer sentences.
- Artefact extraction is separate from intent classification; call extract_artefacts()
  when you need the specific types the user mentioned.

False-positive reduction strategy:
- Merge / cherry-pick patterns use `^` so they only fire when the user starts
  the line with the keyword, not when the word appears mid-sentence.
- Branch-switch patterns use `$` so the branch name must be the last token —
  "vuelve a revisar el código" won't match because "el" follows "revisar".
- _BRANCH_NAME requires ≥ 2 chars, reducing accidental matches on prepositions.

Merge vs cherry-pick signal model:
- "todo" is the primary merge signal — if the user says "trae todo de X" or
  "mezcla todo lo de X" they want all artefacts, i.e. a full merge.
- Merge rules also fire for bare verb-only forms: "merge X", "fusiona X",
  "mezcla X", "integra X", "absorbe X", "incorpora X".
- Optional destination suffixes ("a main", "en main", "con main", etc.) are
  tolerated by merge rules but not captured — the destination is the current
  branch as far as the command executor is concerned.
- Cherry-pick rules require an explicit artefact keyword (decisions, findings,
  pending, facts, summary, errors, changes / their Spanish synonyms) between
  the verb and the "de/desde/from <branch>" clause.  Without an artefact word
  the regex cannot match, so the text falls through to conversation.
- "incorpora [únicamente] artefact de X" is cherry-pick, not merge, because
  an artefact word is present.  "incorpora X" or "incorpora todo de X" are
  merge because no artefact word constrains the selection.
- Cherry-pick patterns accept the same optional destination suffix as merge
  patterns ("a main", "en main", etc.).  When a destination is present,
  route() injects it as values["destination"] so the REPL can override the
  default active-branch target.  Artefact extraction runs on the full text
  and is not affected by the destination because artefact keywords never
  overlap with branch names or prepositions.

Component order flexibility in cherry-pick:
- The router supports multiple natural orderings of source, destination, and
  artefacts without any loss of determinism.
- Forma 1/3: artefacts before source, optional destination at end.
    "trae las decisiones de X [a main]"
    "incorpora findings de X [en main]"
- Forma 2 (new): source before destination, artefacts at end (or absent).
    "trae de X a main solo findings"
    "pásame de X a main facts y findings"
    "cherry-pick de X a main solo decisions"  ← destination extractor fix
- Forma 4 (new): destination immediately after verb, artefacts, then source.
    "trae a main las decisiones de X"
    "dame a main solo findings de X"
- The destination is extracted by extract_destination(), which uses a
  lookahead (not end-of-string anchor) so it finds "a main" regardless of
  whether artefact tokens follow it.

Contextual source resolution:
- route() accepts an optional active_branch parameter.
- When the user refers to "esta rama", "la actual", "current branch", etc.
  (a contextual reference to the active branch), route() substitutes the
  actual branch name before running pattern matching.
- If active_branch is None or empty and a contextual reference is detected,
  the text falls through to "conversation" (safe fallback).
- The substitution is transparent: "mezcla esta rama en main" becomes
  "mezcla feature-x en main", which is then classified by the normal rules.
- is_contextual_source_ref() and resolve_contextual_source() are public so
  callers can inspect or test the resolution step independently.
"""

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

# Branch names: alphanumeric start, then alphanumeric / dash / underscore, ≥ 2 chars total.
_BRANCH_NAME = r"([a-zA-Z0-9][a-zA-Z0-9_\-]{1,})"

# Non-capturing branch reference (used inside _DEST_SUFFIX).
_BRANCH_REF = r"[a-zA-Z0-9][a-zA-Z0-9_\-]{1,}"

# Optional destination suffix: "a/hacia/en/con/dentro de <branch>"
# Placed at the end of merge patterns so "merge foo a main" is still a merge.
_DEST_SUFFIX = r"(?:\s+(?:a|hacia|en|con|dentro\s+de)\s+" + _BRANCH_REF + r")?"

# Capturing version of the destination suffix — used by extract_destination().
#
# Uses a lookahead (?=\s|$) instead of \s*$ so it finds "a main" even when
# artefact tokens follow the destination (e.g. "cherry-pick de X a main solo
# decisions").  The first match in the string is returned, which is the most
# natural interpretation when the user writes the destination before the
# artefacts.
_DEST_BRANCH_RE = re.compile(
    r"\s+(?:a|hacia|en|con|dentro\s+de)\s+(" + _BRANCH_REF + r")(?=\s|$)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Contextual source reference resolution
#
# These phrases all mean "the branch I'm currently on".  They are matched
# before the normal rule loop; when found, route() substitutes the actual
# active branch name so existing rules can classify the intent normally.
#
# Ordering: longest alternatives first so "esta rama activa" matches before
# "esta rama" and "esta", preventing partial matches.
# ---------------------------------------------------------------------------

_CONTEXTUAL_SOURCE_RE = re.compile(
    r"\b(?:"
    r"esta\s+rama\s+activa"
    r"|esta\s+rama"
    r"|la\s+rama\s+activa"
    r"|la\s+rama\s+actual"
    r"|la\s+actual"
    r"|current\s+branch"
    r"|active\s+branch"
    r"|this\s+branch"
    r"|current\s+one"
    r"|esta"  # short bare form — comes last to reduce accidental matches
    r")\b",
    re.IGNORECASE,
)


def is_contextual_source_ref(text: str) -> bool:
    """
    Return True if *text* contains a phrase that contextually refers to the
    active branch (e.g. "esta rama", "la actual", "current branch", "esta").

    Used by route() to decide whether active_branch substitution is needed.
    Exposed publicly so callers and tests can inspect the detection step.
    """
    return bool(_CONTEXTUAL_SOURCE_RE.search(text))


def resolve_contextual_source(text: str, active_branch: str) -> str:
    """
    Replace the *first* contextual branch reference in *text* with
    *active_branch*, enabling existing routing rules to classify the intent.

    Examples (active_branch="feature-x"):
        "mezcla esta rama en main"            → "mezcla feature-x en main"
        "trae las decisiones de la actual"    → "trae las decisiones de feature-x"
        "fusiona la rama actual con main"     → "fusiona feature-x con main"
        "cherry-pick de esta a main --facts"  → "cherry-pick de feature-x a main --facts"
    """
    return _CONTEXTUAL_SOURCE_RE.sub(active_branch, text, count=1)


# ---------------------------------------------------------------------------
# Artefact keyword map (used by extract_artefacts)
# ---------------------------------------------------------------------------

_ARTEFACT_KEYWORD_MAP: dict[str, str] = {
    # decisions
    "decision": "decisions",
    "decisions": "decisions",
    "decisiones": "decisions",
    "decisión": "decisions",
    # findings
    "finding": "findings",
    "findings": "findings",
    "hallazgo": "findings",
    "hallazgos": "findings",
    "resultado": "findings",
    "resultados": "findings",
    # pending / tasks
    "pending": "pending",
    "pendings": "pending",
    "pendiente": "pending",
    "pendientes": "pending",
    "tarea": "pending",
    "tareas": "pending",
    "accion": "pending",
    "acciones": "pending",
    # facts
    "fact": "facts",
    "facts": "facts",
    "hecho": "facts",
    "hechos": "facts",
    "dato": "facts",
    "datos": "facts",
    # summary
    "summary": "summary",
    "resumen": "summary",
    "sumario": "summary",
    # errors
    "error": "errors",
    "errors": "errors",
    "errores": "errors",
    "bug": "errors",
    "bugs": "errors",
    "fallo": "errors",
    "fallos": "errors",
    "falla": "errors",
    "fallas": "errors",
    # changes
    "change": "changes",
    "changes": "changes",
    "cambio": "changes",
    "cambios": "changes",
    "modificacion": "changes",
    "modificaciones": "changes",
    "modificación": "changes",
}


def extract_artefacts(text: str) -> list[str]:
    """
    Extract mentioned artefact types from natural language text.

    Returns a list of canonical artefact type names in mention order,
    deduplicated. Returns an empty list if none are found.

    Examples:
        "trae las decisiones y findings de foo" → ["decisions", "findings"]
        "cherry-pick foo --summary --facts"     → ["summary", "facts"]
        "merge foo"                             → []
    """
    words = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+", text.lower())
    found: list[str] = []
    seen: set[str] = set()
    for word in words:
        atype = _ARTEFACT_KEYWORD_MAP.get(word)
        if atype and atype not in seen:
            found.append(atype)
            seen.add(atype)
    return found


def extract_destination(text: str) -> str | None:
    """
    Extract the explicit destination branch from a merge phrase, if present.
    Returns None when no destination suffix is found.

    Examples:
        "mezcla feature-x en main"          → "main"
        "trae todo de feature-x a main"     → "main"
        "merge feature-x"                   → None
        "mezcla esta rama en main"          → "main"  (before/after resolution)
    """
    m = _DEST_BRANCH_RE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

# Each entry: (compiled_pattern, intent, extractor_fn(match, text) → dict)
_RULES: list[tuple[re.Pattern[str], str, object]] = []


def _r(pattern: str, intent: str):
    def _decorator(fn):
        _RULES.append((re.compile(pattern, re.IGNORECASE), intent, fn))
        return fn
    return _decorator


# ---------------------------------------------------------------------------
# Terminal / utility intents  (anchored — full-line only)
# ---------------------------------------------------------------------------

@_r(r"^(exit|quit|salir|chau|bye)\s*$", "exit")
def _exit(m, _): return {}

@_r(r"^(help|ayuda|\?)\s*$", "help")
def _help(m, _): return {}

@_r(r"^(status|estado)\s*$", "session_status")
def _session_status(m, _): return {}

@_r(r"^nueva\s+sesión\s*$|^new\s+session\s*$", "session_new")
def _session_new(m, _): return {}


# ---------------------------------------------------------------------------
# Branch creation  (keyword-anchored — require branch/rama keyword)
# ---------------------------------------------------------------------------

@_r(
    r"(?:crea(?:r)?\s+(?:una\s+)?rama|nueva\s+rama|new\s+branch|branch\s+new)\s+" + _BRANCH_NAME,
    "branch_new",
)
def _branch_new(m, _): return {"name": m.group(1)}

@_r(
    r"(?:abre?\s+(?:una\s+)?rama|inicia\s+(?:una\s+)?rama|start\s+branch)\s+" + _BRANCH_NAME,
    "branch_new",
)
def _branch_new2(m, _): return {"name": m.group(1)}


# ---------------------------------------------------------------------------
# Branch switch  (end-anchored — branch name must be last token)
# ---------------------------------------------------------------------------

@_r(
    r"^(?:vuelve?\s+a|regresa\s+a|go\s+(?:back\s+)?to)\s+" + _BRANCH_NAME + r"\s*$",
    "branch_switch",
)
def _branch_switch(m, _): return {"name": m.group(1)}

@_r(
    r"^switch\s+(?:to\s+)?" + _BRANCH_NAME + r"\s*$",
    "branch_switch",
)
def _branch_switch_en(m, _): return {"name": m.group(1)}

@_r(
    r"^cambia(?:r)?\s+a\s+(?:la\s+(?:rama\s+)?)?" + _BRANCH_NAME + r"\s*$",
    "branch_switch",
)
def _branch_switch2(m, _): return {"name": m.group(1)}

@_r(
    r"^cambia(?:r)?\s+de\s+(?:la\s+)?rama\s+(?:a\s+)?" + _BRANCH_NAME + r"\s*$",
    "branch_switch",
)
def _branch_switch2b(m, _): return {"name": m.group(1)}

@_r(
    r"^ir(?:me)?\s+a(?:\s+la)?\s+rama\s+" + _BRANCH_NAME + r"\s*$",
    "branch_switch",
)
def _branch_switch3(m, _): return {"name": m.group(1)}


# ---------------------------------------------------------------------------
# Branch list / status / show
# ---------------------------------------------------------------------------

@_r(
    r"(?:lista(?:r)?(?:\s+(?:de\s+)?(?:las?\s+)?)?ramas?|list\s+branches?|ver\s+ramas?|show\s+branches?|branches?\s+list)",
    "branch_list",
)
def _branch_list(m, _): return {}

@_r(r"(?:^branch\s+status$|estado\s+(?:de\s+(?:la\s+)?)?rama)", "branch_status")
def _branch_status(m, _): return {}

@_r(r"^(?:muestra(?:r)?\s+(?:la\s+)?rama|show\s+branch)\s+" + _BRANCH_NAME + r"\s*$", "branch_show")
def _branch_show(m, _): return {"name": m.group(1)}


# ---------------------------------------------------------------------------
# Merge  (start-anchored — keyword must lead the sentence)
#
# These MUST come before the cherry-pick "trae" patterns because some merge
# phrases ("trae todo de X") would otherwise be captured by the generic
# cherry-pick "trae WORD de branch" rule.
# ---------------------------------------------------------------------------

@_r(
    r"^(?:merge|fusiona(?:r)?|combina(?:r)?|mezcla(?:r)?|integra(?:r)?)\s+"
    + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_merge",
)
def _merge1(m, _): return {"source": m.group(1)}

# "incorpora <branch>" — kept but anchored, with optional destination
@_r(
    r"^incorpora(?:r)?\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_merge",
)
def _merge2(m, _): return {"source": m.group(1)}

# "trae todo (lo) [útil/importante/relevante] de/desde <branch> [a dest]"
# Must be before cherry-pick "trae" rules so it takes priority.
@_r(
    r"^trae\s+todo\s+(?:lo\s+)?(?:(?:útil|importante|relevante|el\s+contenido)\s+)?"
    r"(?:de|desde|from)\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_merge",
)
def _merge3(m, _): return {"source": m.group(1)}

# "pásame / dame / pasa / tráeme todo [lo] de/desde <branch> [a dest]"
@_r(
    r"^(?:pás[ae]me?|dame?|trae(?:me)?|tráeme?|pasa(?:r)?)\s+todo\s+(?:lo\s+)?"
    r"(?:(?:útil|importante|relevante)\s+)?(?:de|desde|from)\s+"
    + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_merge",
)
def _merge_dame_todo(m, _): return {"source": m.group(1)}

# "mezcla/combina/fusiona/integra todo (lo) [útil] de/desde <branch> [en/a dest]"
@_r(
    r"^(?:mezcla(?:r)?|combina(?:r)?|fusiona(?:r)?|integra(?:r)?)\s+todo\s+(?:lo\s+)?"
    r"(?:(?:útil|importante|relevante)\s+)?(?:de|desde|en|from)\s+"
    + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_merge",
)
def _merge_verb_todo(m, _): return {"source": m.group(1)}

# "incorpora todo [lo] [útil] de/desde <branch> [a dest]"
@_r(
    r"^incorpora(?:r)?\s+todo\s+(?:lo\s+)?(?:(?:útil|importante|relevante)\s+)?"
    r"(?:de|desde|from)\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_merge",
)
def _merge_incorpora_todo(m, _): return {"source": m.group(1)}

# "haz merge de <branch> / haz un merge de <branch>"
@_r(
    r"^haz\s+(?:un\s+)?merge\s+(?:de\s+)?" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_merge",
)
def _merge4(m, _): return {"source": m.group(1)}

# "absorbe <branch>" — informal merge
@_r(
    r"^absorbe(?:r)?\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_merge",
)
def _merge5(m, _): return {"source": m.group(1)}


# ---------------------------------------------------------------------------
# Cherry-pick  (more specific — after merge rules, before conversation)
#
# Patterns capture both the source branch and any mentioned artefact types.
# The 'artefacts' key in the returned dict contains the extracted types
# (may be empty if none are mentioned explicitly).
#
# False-positive guard: patterns that match "WORD de BRANCH" explicitly
# exclude "todo" (which signals a full merge, handled above).
# ---------------------------------------------------------------------------

# "cherry-pick de <branch>" — MUST come before the bare form because the bare
# form's _BRANCH_NAME would otherwise match "de" (a 2-char token) and capture
# it as the source branch, leaving the real branch name in the trailing group.
# _DEST_SUFFIX allows "cherry-pick de feature-x a main".
@_r(
    r"^cherry.?pick\s+de\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_de_bare(m, _): return {"source": m.group(1), "artefacts": []}

@_r(
    r"^cherry.?pick\s+de\s+" + _BRANCH_NAME + r"(.+)$",
    "branch_cherry_pick",
)
def _cherry_de_flags(m, text):
    return {"source": m.group(1), "artefacts": extract_artefacts(m.group(2))}

# "cherry-pick <branch>" — bare form without "de" prefix
# _DEST_SUFFIX allows "cherry-pick feature-x a main".
@_r(
    r"^cherry.?pick\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_bare(m, _): return {"source": m.group(1), "artefacts": []}

@_r(
    r"^cherry.?pick\s+" + _BRANCH_NAME + r"(.+)$",
    "branch_cherry_pick",
)
def _cherry_with_flags(m, text):
    return {"source": m.group(1), "artefacts": extract_artefacts(m.group(2))}

# ---------------------------------------------------------------------------
# Cherry-pick — flexible component order (Forma 2 and Forma 4)
#
# These rules cover natural orderings where source / destination / artefacts
# appear in a sequence that the existing Forma 1/3 patterns cannot handle.
#
# Forma 2: "VERB de SOURCE a/hacia/en/con DEST [artefacts]"
#   "trae de feature-x a main solo findings"
#   "pásame de la actual a main facts y findings"
#   "incorpora de feature-x a main las decisiones"
#
# Guard: (?!\s+todo\b) prevents "trae de X a Y todo" being misclassified as
# cherry-pick (that phrase has no recognised artefact and no merge signal
# either, so falling to conversation is the right outcome).
# ---------------------------------------------------------------------------

@_r(
    r"^(?:pás[ae]me?|dame?|trae(?:me)?|tráeme?|incorpora(?:r)?)\s+"
    r"(?:de|desde|from)\s+" + _BRANCH_NAME  # group(1) = source
    + r"\s+(?:a|hacia|en|con|dentro\s+de)\s+" + _BRANCH_NAME  # group(2) = dest (used by extract_destination)
    + r"(?!\s+todo\b)(?:\s+.+)?$",
    "branch_cherry_pick",
)
def _cherry_source_dest_artefacts(m, text):
    # destination is resolved by route() via extract_destination(stripped)
    return {"source": m.group(1), "artefacts": extract_artefacts(text)}


# Forma 4: "VERB a/hacia/en/con DEST [solo] artefacts de/desde SOURCE"
#   "trae a main las decisiones de feature-x"
#   "dame a main solo findings de feature-x"
#   "pásame a main facts y findings de la actual"
#
# Guard: (?!todo\b) prevents matching "trae a main todo de X".
# Risk note: the artefact section is free-form; an ASCII branch name at the
# end is required.  Sentences like "trae a main el resumen del proyecto de
# feature-x" classify as cherry-pick (summary artefact, source=feature-x).
# Branches named with accents won't match _BRANCH_NAME so those fall safely
# to conversation.
@_r(
    r"^(?:pás[ae]me?|dame?|trae(?:me)?|tráeme?)\s+"
    r"(?:a|hacia|en|con|dentro\s+de)\s+" + _BRANCH_NAME  # group(1) = dest (extract_destination also finds this)
    + r"\s+(?:solo\s+|únicamente\s+|solamente\s+)?(?:(?:el|los?|las?)\s+)?(?!todo\b)"
    r"(?:[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:\s+y\s+[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)*)"
    r"\s+(?:de|desde|from)\s+" + _BRANCH_NAME  # group(2) = source
    + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_dest_artefacts_source(m, text):
    # destination is extracted from the full text by route() — it finds
    # "a DEST" via the non-anchored _DEST_BRANCH_RE lookahead.
    return {"source": m.group(2), "artefacts": extract_artefacts(text)}


# "trae solo/las decisiones/findings/... de/desde <branch>"
# Guard: (?!todo\b) prevents matching "trae todo de X" (that's a full merge).
@_r(
    r"^trae\s+(?:solo\s+)?(?:(?:el|los?|las?)\s+)?(?!todo\b)"
    r"(?:[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:\s+y\s+[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)*)"
    r"\s+(?:de|desde|from)\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_trae(m, text):
    return {"source": m.group(1), "artefacts": extract_artefacts(text)}

# "pásame / dame / tráeme los findings de/desde <branch>"
# Guard: (?!todo\b) prevents matching "dame todo de X" (full merge, handled above).
@_r(
    r"^(?:pás[ae]me?|dame?|trae(?:me)?)\s+(?:(?:el|los?|las?)\s+)?(?!todo\b)"
    r"(?:[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:\s+y\s+[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)*)"
    r"\s+(?:de|desde|from)\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_dame(m, text):
    return {"source": m.group(1), "artefacts": extract_artefacts(text)}

# "quiero [solo] los decisions/findings de/desde <branch>"
@_r(
    r"^quiero\s+(?:solo\s+)?(?:(?:el|los?|las?)\s+)?"
    r"(?:[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:\s+y\s+[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)*)"
    r"\s+(?:de|desde|from)\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_quiero(m, text):
    return {"source": m.group(1), "artefacts": extract_artefacts(text)}

# "importa las decisiones de/desde <branch>"
@_r(
    r"^importa(?:r)?\s+(?:solo\s+)?(?:(?:el|los?|las?)\s+)?[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+"
    r"\s+(?:de|desde|from)\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_importa(m, text):
    return {"source": m.group(1), "artefacts": extract_artefacts(text)}

# "extrae los findings de/desde <branch>" / "extrae solo las decisions de/desde <branch>"
@_r(
    r"^extrae(?:r)?\s+(?:solo\s+)?(?:(?:el|los?|las?)\s+)?"
    r"(?:[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:\s+y\s+[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)*)"
    r"\s+(?:de|desde|from)\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_extrae(m, text):
    return {"source": m.group(1), "artefacts": extract_artefacts(text)}

# "incorpora [solo/únicamente/solamente] los findings/decisions de/desde <branch>"
# Guard: (?!todo\b) prevents matching "incorpora todo de X" (full merge).
# This rule must come AFTER the merge "incorpora <branch>" rules.
@_r(
    r"^incorpora(?:r)?\s+(?:solo\s+|únicamente\s+|solamente\s+)?"
    r"(?:(?:el|los?|las?)\s+)?(?!todo\b)"
    r"(?:[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+(?:\s+y\s+[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+)*)"
    r"\s+(?:de|desde|from)\s+" + _BRANCH_NAME + _DEST_SUFFIX + r"\s*$",
    "branch_cherry_pick",
)
def _cherry_incorpora_artefacts(m, text):
    return {"source": m.group(1), "artefacts": extract_artefacts(text)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def route(text: str, active_branch: str | None = None) -> tuple[Intent, dict]:
    """
    Classify natural language text into an Intent.

    Slash commands must be handled upstream — if this function receives text
    starting with '/', it returns ("conversation", {}) so the REPL falls back
    to its slash handler.

    active_branch, when provided, enables contextual source resolution: phrases
    like "esta rama", "la actual", "current branch", "esta" are substituted
    with the real branch name before pattern matching, so normal merge/cherry-pick
    rules can classify them correctly.  If active_branch is None or empty and a
    contextual reference is detected, the text falls through to "conversation".

    Returns (intent, values_dict).

    The values_dict contains:
    - branch operations:      {"name": str}  or  {"source": str, "artefacts": list[str]}
    - session / util intents: {}
    - conversation:           {}
    """
    stripped = text.strip()
    if stripped.startswith("/"):
        return "conversation", {}

    # Contextual source resolution: "mezcla esta rama en main" becomes
    # "mezcla feature-x en main" so the normal merge rules fire correctly.
    # Without active_branch, contextual phrases safely fall to conversation.
    if is_contextual_source_ref(stripped):
        if not active_branch:
            return "conversation", {}
        stripped = resolve_contextual_source(stripped, active_branch)

    for pattern, intent, extractor in _RULES:
        m = pattern.search(stripped)
        if m:
            result: dict = extractor(m, stripped)  # type: ignore[operator]
            # For merge intents, extract the destination branch if explicitly
            # present in the phrase ("mezcla X en main" → destination="main").
            # The REPL uses this to override the default active-branch target.
            if intent in ("branch_merge", "branch_cherry_pick"):
                result["destination"] = extract_destination(stripped)
            return intent, result
    return "conversation", {}
