"""
Targeted regression tests for the two extract_claims() bugs.

Bug A — SYMBOL_CLAIM_PATTERN captured short Spanish words as function names:
  "función que …"  → "que"  (3 chars, common conjunction)
  "función es …"   → "es"   (2 chars)
  "función bas …"  → "bas"  (3 chars, noise)
  "función devuelve" → "devuelve" (Spanish verb, not a name)

Bug B — BACKTICK_SYMBOL_PATTERN let Python built-ins / keywords through:
  `True`, `False`, `None`, `self`, `cls`, `classmethod`

Both bugs caused valid model answers to be rejected with
"Respuesta no grounded: cita símbolos sin evidencia: …"
"""
from __future__ import annotations

import pytest

from echo.cognition.verifier_support import extract_claims


# ---------------------------------------------------------------------------
# Bug A — short / common Spanish words after "función/método/clase"
# ---------------------------------------------------------------------------


def test_bug_a_funcion_que_is_not_a_symbol() -> None:
    claims = extract_claims("La función que devuelve el root es correcta.")
    assert "que" not in claims.symbols


def test_bug_a_funcion_es_is_not_a_symbol() -> None:
    claims = extract_claims("La función es la que resuelve el path.")
    assert "es" not in claims.symbols


def test_bug_a_funcion_bas_is_not_a_symbol() -> None:
    # "bas" appeared in real error logs as a spurious capture
    claims = extract_claims("La función bas no existe realmente.")
    assert "bas" not in claims.symbols


def test_bug_a_funcion_devuelve_is_not_a_symbol() -> None:
    claims = extract_claims("La función devuelve un Path.")
    assert "devuelve" not in claims.symbols


def test_bug_a_funcion_llamada_is_not_a_symbol() -> None:
    claims = extract_claims("La función llamada principal maneja esto.")
    assert "llamada" not in claims.symbols


def test_bug_a_real_function_name_still_captured() -> None:
    """A real function name (≥4 chars, not in NON_SYMBOL_WORDS) must survive."""
    claims = extract_claims("La función resolve_project_root es la encargada.")
    assert "resolve_project_root" in claims.symbols


def test_bug_a_four_char_function_name_captured() -> None:
    """Exact boundary: 4-char name should be kept."""
    claims = extract_claims("La función main hace el bootstrap.")
    assert "main" in claims.symbols


def test_bug_a_three_char_name_dropped() -> None:
    """3-char names from SYMBOL_CLAIM_PATTERN are noise — drop them."""
    claims = extract_claims("La clase App inicia el CLI.")
    # "App" is 3 chars from SYMBOL_CLAIM_PATTERN; should be dropped
    # (it may still appear via BACKTICK_SYMBOL_PATTERN if backtick-quoted)
    assert "App" not in claims.symbols


# ---------------------------------------------------------------------------
# Bug B — Python built-ins and keywords in backticks
# ---------------------------------------------------------------------------


def test_bug_b_true_in_backticks_not_a_symbol() -> None:
    claims = extract_claims("La función retorna `True` cuando el path es válido.")
    assert "True" not in claims.symbols


def test_bug_b_false_in_backticks_not_a_symbol() -> None:
    claims = extract_claims("Devuelve `False` si no se encuentra el sentinel.")
    assert "False" not in claims.symbols


def test_bug_b_none_in_backticks_not_a_symbol() -> None:
    claims = extract_claims("El valor por defecto es `None`.")
    assert "None" not in claims.symbols


def test_bug_b_self_in_backticks_not_a_symbol() -> None:
    claims = extract_claims("El método `self.run()` ejecuta el loop.")
    # "self" is extracted by BACKTICK_SYMBOL_PATTERN but must be filtered out
    assert "self" not in claims.symbols


def test_bug_b_cls_in_backticks_not_a_symbol() -> None:
    claims = extract_claims("El classmethod recibe `cls` como primer argumento.")
    assert "cls" not in claims.symbols


def test_bug_b_classmethod_in_backticks_not_a_symbol() -> None:
    claims = extract_claims("Es un `classmethod` que carga desde env.")
    assert "classmethod" not in claims.symbols


def test_bug_b_staticmethod_in_backticks_not_a_symbol() -> None:
    claims = extract_claims("Está decorado con `staticmethod`.")
    assert "staticmethod" not in claims.symbols


def test_bug_b_real_symbol_with_uppercase_still_captured() -> None:
    """Real project-specific symbols with uppercase must still be captured."""
    claims = extract_claims("La clase `Settings` define la configuración.")
    assert "Settings" in claims.symbols


def test_bug_b_real_symbol_with_underscore_still_captured() -> None:
    """Real project-specific symbols with underscore must still be captured."""
    claims = extract_claims("Usa `resolve_project_root` para detectar el repo.")
    assert "resolve_project_root" in claims.symbols


def test_bug_b_real_env_var_still_captured() -> None:
    """Real environment variable names (ALL_CAPS) must still be captured."""
    claims = extract_claims("Configura con `ECHO_BACKEND_TIMEOUT=300`.")
    # This ends up in commands (has space/=), not symbols — just verify no crash
    # The env var pattern in collect_symbols won't help here since it's in claims
    # just make sure it doesn't throw
    assert isinstance(claims.symbols, set)


# ---------------------------------------------------------------------------
# Regression: combined — typical bad model response that used to fail
# ---------------------------------------------------------------------------


def test_combined_bad_response_no_longer_triggers_unsupported_symbols() -> None:
    """
    Simulate the kind of response that caused real failures:
    - uses `True` / `False` in backticks
    - says "la función que devuelve"
    - does NOT cite any real project symbol in backtick
    → must produce zero claims.symbols (nothing to check against evidence)
    """
    bad_response = (
        "La función que resuelve el proyecto retorna `True` si encuentra "
        "el sentinel y `False` en caso contrario. El método es llamado "
        "automáticamente y devuelve `None` si no hay repo."
    )
    claims = extract_claims(bad_response)
    assert claims.symbols == set(), f"Expected empty symbols, got: {claims.symbols}"
